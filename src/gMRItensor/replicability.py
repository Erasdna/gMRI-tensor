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
        """Compute FMS between paired halves, as (split_index, fms) tuples."""
        fms_results = []
        for s in range(self.repeats):
            _, weights_0, factors_0 = decomposition_results[(s, 0)]
            _, weights_1, factors_1 = decomposition_results[(s, 1)]

            score = factor_match_score(
                (weights_0, factors_0),
                (weights_1, factors_1),
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

                    # Align Mode 0 indices
                    id_map_i = {sub_id: idx for idx, sub_id in enumerate(ids_i)}
                    id_map_j = {sub_id: idx for idx, sub_id in enumerate(ids_j)}

                    # Take only overlapping subjects when comparing factors
                    fac_i_aligned = [
                        fac_i[0][[id_map_i[s] for s in common_subjects]],
                    ] + fac_i[1:]
                    fac_j_aligned = [
                        fac_j[0][[id_map_j[s] for s in common_subjects]],
                    ] + fac_j[1:]

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
        (task_id, indices, weights, factors), tensors on CPU. PARAFAC2's
        projections are dropped: they only matter for reconstructing
        per-subject time patterns, and its `factors = [A, B, C]` are regular
        like CP's, so `compute_fms` needs no method-specific handling.
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
            weights, factors, _projections, _ = run_PARAFAC2_decomposition_repeated(
                sub_slices, rank=rank, device=_get_device(sub_slices), **kwargs
            )
        else:
            raise ValueError(f"Unknown decomposition method: {method!r}")

        # Move results to CPU to avoid device memory issues in multiprocessing
        factors = [f.cpu() for f in factors]
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
