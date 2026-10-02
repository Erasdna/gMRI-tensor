from abc import ABC
from abc import abstractmethod
from multiprocessing import get_context
from typing import Any
from typing import Literal

import numpy as np
import tensorly as tl
import torch
from gMRItensor import run_CP_decomposition_repeated
from gMRItensor import run_PARAFAC2_decomposition_repeated
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from tlviz.factor_tools import factor_match_score
from tqdm import tqdm


#: One fit's factors. For PARAFAC2 the middle entry is a ragged list of
#: per-subject evolving states; for CP it is a regular matrix.
Factors = list[Any]


def _comparable_modes(factors: Factors) -> list[torch.Tensor]:
    """The modes that mean the same thing across two disjoint-subject fits.

    CP's `[A, B, C]` are all regular and shared, so it passes through
    unchanged. PARAFAC2's evolving mode is indexed by *that subject's* own
    timepoints, so it is dropped; `A` is kept only so `skip_mode=0` still
    names mode 0.
    """
    subject_mode, evolving, label_mode = factors
    if isinstance(evolving, torch.Tensor):
        return [subject_mode, evolving, label_mode]
    return [subject_mode, label_mode]


def _factor_to_cpu(factor: Any) -> Any:
    """Move one factor to CPU, ragged evolving-state lists included."""
    if isinstance(factor, torch.Tensor):
        return factor.cpu()
    return [f.cpu() for f in factor]


def _align_pair(
    factors_i: Factors,
    rows_i: list[int],
    factors_j: Factors,
    rows_j: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Restrict two fits to the same subjects, as regular matrices.

    Mode 0 is row-selected. PARAFAC2's evolving mode is a ragged list, which
    `factor_match_score` rejects outright -- `tlviz` raises `TypeError` on a
    list sitting beside tensors -- so the selected `B_i` are concatenated
    into one `(sum_i J_i, rank)` matrix.

    That stacking is exact rather than a convenience: the PARAFAC2 constraint
    fixes `||B_i[:, r]||` independent of `i`, so the stacked column cosine
    equals the mean per-subject cosine, and a subject with 2 timepoints
    weighs the same as one with 40.

    Both fits are built here together so the per-subject block heights can be
    cross-checked. A subject-index misalignment is otherwise silent whenever
    the totals happen to agree.
    """
    subject_i, evolving_i, label_i = factors_i
    subject_j, evolving_j, label_j = factors_j
    selected_i = [subject_i[rows_i], evolving_i, label_i]
    selected_j = [subject_j[rows_j], evolving_j, label_j]

    if isinstance(evolving_i, torch.Tensor):  # CP, or any shared regular mode
        return selected_i, selected_j

    blocks_i = [evolving_i[k] for k in rows_i]
    blocks_j = [evolving_j[k] for k in rows_j]
    heights_i = [b.shape[0] for b in blocks_i]
    heights_j = [b.shape[0] for b in blocks_j]
    if heights_i != heights_j:
        raise ValueError(
            "The two fits disagree on how many timepoints the shared subjects "
            f"have ({heights_i} vs {heights_j}). The same subject is "
            "decomposed from the same slice in both fits, so this means the "
            "subject-index mapping is wrong, not the data.",
        )

    selected_i[1] = torch.cat(blocks_i, dim=0)
    selected_j[1] = torch.cat(blocks_j, dim=0)
    return selected_i, selected_j


class ReplicabilityEngine(ABC):
    """Base class for replicability analysis engines."""

    def __init__(
        self,
        seed: int = 0,
    ) -> None:
        """Initialize the replicability engine."""
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(seed)

    @abstractmethod
    def generate_tasks(
        self,
        n_tot: int,
        stratification: torch.Tensor | None = None,
    ):
        """Generate list of tasks for computing CP decompositions.

        Args:
            n_tot: Total number of samples
            stratification: Optional stratification labels for splitting

        Returns:
            List of (task_id, indices) tuples

        Notes
        -----
        `inds`/`stratification` are index-bookkeeping arrays for
        scikit-learn's splitters, which need numpy-convertible CPU data, so
        they are always kept on CPU -- including a `stratification` passed in
        on another device. The decomposed tensors' device is handled
        separately, in `evaluate_replicability_multiproc`.
        """
        inds = torch.arange(n_tot)
        if stratification is None:
            stratification = torch.ones(n_tot)
        else:
            stratification = stratification.cpu()

        return inds, stratification

    @abstractmethod
    def compute_fms(
        self,
        decomposition_results: dict[Any, tuple[list[int], Any, list[torch.Tensor]]],
    ):
        """Compute pairwise FMS scores from each task's factors.

        `decomposition_results` maps task_id to (indices, weights, factors).
        The returned tuple format depends on the engine.
        """
        pass


class HalfHalfEngine(ReplicabilityEngine):
    """Replicability engine using half-half splits."""

    def __init__(
        self,
        repeats: int,
        seed: int = 0,
    ) -> None:
        """Initialize half-half split engine, with `repeats` random splits."""
        super().__init__(seed)
        self.repeats = repeats
        self.rskf = StratifiedShuffleSplit(
            n_splits=repeats,
            test_size=0.5,
            random_state=self.seed,
        )

    def generate_tasks(
        self,
        n_tot: int,
        stratification: torch.Tensor | None = None,
    ) -> list[tuple[tuple[int, int], list[int]]]:
        """Generate ((split_index, half_index), indices) tasks."""
        inds, stratification = super().generate_tasks(n_tot, stratification)
        tasks = []
        for i, (train_idx, test_idx) in enumerate(
            self.rskf.split(inds, y=stratification),
        ):
            # Task ID: (split_index, half_index [0 or 1])
            tasks.append(((i, 0), train_idx.tolist()))
            tasks.append(((i, 1), test_idx.tolist()))

        return tasks

    def compute_fms(
        self,
        decomposition_results: dict[
            tuple[int, int],
            tuple[list[int], Any, list[torch.Tensor]],
        ],
    ) -> list[tuple[int, float]]:
        """Compute FMS between paired halves, as (split_index, fms) tuples.

        The two halves are disjoint subject sets, so mode 0 has no
        correspondence -- hence `skip_mode=0`. For PARAFAC2 the evolving mode
        is per-subject and so has none either, leaving the label mode as the
        only comparable one. It is dropped rather than summarised: the
        subject-independent part of the evolving mode is `B_i.T @ B_i`, which
        measures component geometry rather than shape and scores a perfect
        1.0 for two fits with completely unrelated time courses.
        """
        fms_results = []
        for s in range(self.repeats):
            _, weights_0, factors_0 = decomposition_results[(s, 0)]
            _, weights_1, factors_1 = decomposition_results[(s, 1)]

            score = factor_match_score(
                (weights_0, _comparable_modes(factors_0)),
                (weights_1, _comparable_modes(factors_1)),
                skip_mode=0,
                consider_weights=False,
            )
            fms_results.append((s, score))
        return fms_results


class CrossValidationEngine(ReplicabilityEngine):
    """Replicability engine using cross-validation folds."""

    def __init__(
        self,
        splits: int,
        repeats: int,
        seed: int = 0,
    ) -> None:
        """Initialize a `repeats` x `splits`-fold cross-validation engine."""
        super().__init__(seed)
        self.splits = splits
        self.repeats = repeats

        self.rskf = RepeatedStratifiedKFold(
            n_splits=self.splits,
            n_repeats=self.repeats,
            random_state=seed,
        )
        self.nb_folds = self.splits * self.repeats

    def generate_tasks(
        self,
        n_tot: int,
        stratification: torch.Tensor | None = None,
    ) -> list[tuple[int, list[int]]]:
        """Generate (fold_index, train_indices) tasks."""
        inds, stratification = super().generate_tasks(n_tot, stratification)
        tasks = []
        for fold_idx, (train_idx, _) in enumerate(
            self.rskf.split(inds, y=stratification),
        ):
            tasks.append((fold_idx, train_idx.tolist()))
        return tasks

    def compute_fms(
        self,
        decomposition_results: dict[int, tuple[list[int], Any, list[torch.Tensor]]],
    ) -> list[tuple[np.ndarray, int, int, float]]:
        """Compute pairwise FMS between folds within each repeat.

        Returns (common_subjects, fold_i, fold_j, fms_score) tuples.
        """
        fms_results = []
        for repeat in range(self.repeats):
            for split in range(self.splits):
                i = repeat * self.splits + split
                ids_i, weights_i, fac_i = decomposition_results[i]
                fold_limit = (repeat + 1) * self.splits
                for j in range(i + 1, fold_limit):
                    ids_j, weights_j, fac_j = decomposition_results[j]

                    common_subjects = np.intersect1d(ids_i, ids_j)
                    if len(common_subjects) == 0:
                        continue

                    id_map_i = {sub_id: idx for idx, sub_id in enumerate(ids_i)}
                    id_map_j = {sub_id: idx for idx, sub_id in enumerate(ids_j)}
                    fac_i_aligned, fac_j_aligned = _align_pair(
                        fac_i,
                        [id_map_i[s] for s in common_subjects],
                        fac_j,
                        [id_map_j[s] for s in common_subjects],
                    )

                    score = factor_match_score(
                        (weights_i, fac_i_aligned),
                        (weights_j, fac_j_aligned),
                        consider_weights=False,
                    )
                    fms_results.append((common_subjects, i, j, score))

        return fms_results


def _n_samples(tensor: torch.Tensor | list[torch.Tensor]) -> int:
    """Number of samples (subjects) along mode 0.

    Works for both a regular tensor and a ragged list of per-subject slices.
    """
    return len(tensor)


def _get_device(tensor: torch.Tensor | list[torch.Tensor]) -> torch.device:
    """Device the input lives on.

    Works for both a regular tensor and a ragged list of per-subject slices
    (which have no `.device` of their own -- the first slice's device is used).
    """
    return tensor.device if isinstance(tensor, torch.Tensor) else tensor[0].device


def _init_worker_backend() -> None:
    """Pool initializer: set up TensorLy's backend in each worker process.

    A "spawn"ed worker is a fresh process that never ran `setup_backend`, so
    without this TensorLy falls back to numpy and cannot interpret the torch
    tensors handed to it.
    """
    tl.set_backend("pytorch")


def _decomposition_worker(
    task_args: tuple[
        Any,
        list[int],
        torch.Tensor | list[torch.Tensor],
        int,
        Literal["CP", "PARAFAC2"],
        dict[str, Any],
    ],
) -> tuple[Any, list[int], Any, list[torch.Tensor]]:
    """Worker function for parallel CP/PARAFAC2 decomposition.

    Args:
        task_args: (task_id, indices, full_tensor, rank, method, kwargs)

    Returns:
        (task_id, indices, weights, factors), tensors on CPU. For PARAFAC2
        `factors` is `[subject_mode, evolving_states, label_mode]`, whose
        middle entry is a ragged per-subject list -- see `_align_pair` and
        `_comparable_modes` for how the engines handle that.
    """
    task_id, indices, full_tensor, rank, method, kwargs = task_args

    try:
        if method == "CP":
            assert isinstance(full_tensor, torch.Tensor)
            sub_tensor = full_tensor[indices]
            weights, factors, _ = run_CP_decomposition_repeated(
                sub_tensor, rank=rank, device=full_tensor.device, **kwargs
            )
        elif method == "PARAFAC2":
            sub_slices = [full_tensor[i] for i in indices]
            model, _ = run_PARAFAC2_decomposition_repeated(
                sub_slices, rank=rank, device=_get_device(sub_slices), **kwargs
            )
            weights = model.weights
            factors = [
                model.subject_mode,
                model.evolving_states,
                model.label_mode,
            ]
        else:
            raise ValueError(f"Unknown decomposition method: {method!r}")

        # Move results to CPU to avoid device memory issues in multiprocessing
        factors = [_factor_to_cpu(f) for f in factors]
        weights = weights.cpu() if isinstance(weights, torch.Tensor) else weights

        return task_id, indices, weights, factors
    except Exception as e:
        raise RuntimeError(f"Decomposition failed for task {task_id}: {e}") from e


def evaluate_replicability_multiproc(
    replicability_engine: ReplicabilityEngine,
    tensor: torch.Tensor | list[torch.Tensor],
    rank: int,
    method: Literal["CP", "PARAFAC2"] = "CP",
    stratification: torch.Tensor | None = None,
    n_procs: int = 1,
    **CP_kwargs: Any,
) -> list[tuple[Any, ...]]:
    """Evaluate replicability using repeated CP or PARAFAC2 decompositions.

    Args:
        replicability_engine: Engine defining the replicability strategy
        tensor: Input to decompose. A regular tensor (samples × ...) for
            method="CP", or a list of per-subject slices (mode 1 may be
            ragged) for method="PARAFAC2".
        rank: Number of components for the decomposition
        method: "CP" or "PARAFAC2"
        stratification: Optional stratification labels for splitting
        n_procs: Number of parallel processes. Must be 1 on CUDA.
        **CP_kwargs: Forwarded to run_CP_decomposition_repeated /
            run_PARAFAC2_decomposition_repeated. This is how PARAFAC2-only
            options reach the solver -- `solver="matcouply"`, `nn_modes`,
            `aoadmm_options`, `aoadmm_loss_tolerance` -- which is why this
            module needs no solver-specific code. They `TypeError` with
            `method="CP"`. Do not pass `return_diagnostics`: the worker
            unpacks a fixed 4-tuple.

    Returns:
        List of FMS score tuples (format depends on engine type)

    Raises:
        ValueError: If `n_procs >= 2` and `tensor` is on CUDA. Worker
            processes against a CUDA context from an already-initialized
            parent are unreliable across driver setups, so this is rejected
            rather than silently falling back to one process.
    """
    is_cuda = _get_device(tensor).type == "cuda"
    if is_cuda and n_procs >= 2:
        # Checked before any work starts, to fail fast on misconfiguration.
        raise ValueError(
            f"n_procs={n_procs} requests multiprocessing, but `tensor` is on "
            "CUDA. Running multiple worker processes against a CUDA context "
            "is unsafe/unreliable across GPU driver setups -- pass "
            "n_procs=1 to run sequentially on the GPU, or move `tensor` to "
            "CPU first to use multiple processes.",
        )

    tasks = replicability_engine.generate_tasks(
        _n_samples(tensor),
        stratification,
    )

    task_args = [
        (task_id, indices, tensor, rank, method, CP_kwargs)
        for task_id, indices in tasks
    ]

    results_dict: dict[Any, tuple[list[int], Any, list[torch.Tensor]]] = {}

    # Use sequential processing for CUDA (multiprocessing doesn't work well with
    # CUDA -- see the ValueError above) or when n_procs < 2
    if n_procs < 2 or is_cuda:
        for task in tqdm(
            task_args,
            desc="Computing decompositions (sequential)",
        ):
            task_id, indices, weights, factors = _decomposition_worker(task)
            results_dict[task_id] = (indices, weights, factors)
    else:
        # "spawn" rather than the platform-default "fork"
        with get_context("spawn").Pool(
            n_procs,
            initializer=_init_worker_backend,
        ) as pool:
            for task_id, indices, weights, factors in tqdm(
                pool.imap_unordered(_decomposition_worker, task_args),
                total=len(task_args),
                desc=f"Computing decompositions (parallel, {n_procs} procs)",
            ):
                results_dict[task_id] = (indices, weights, factors)

    return replicability_engine.compute_fms(results_dict)
