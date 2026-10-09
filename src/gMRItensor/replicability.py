from abc import ABC
from abc import abstractmethod
from collections.abc import Hashable
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from tlviz.factor_tools import factor_match_score

from .decomposition import _resolve_options
from .decomposition import CMFModel
from .decomposition import Method
from .decomposition import PARAFAC2Model
from .jobs import collect
from .jobs import CPModel
from .jobs import GroupSummary
from .jobs import InMemoryStore
from .jobs import plan_replicability
from .jobs import run_tasks


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


def _weights_and_factors(
    model: PARAFAC2Model | CMFModel | CPModel | None,
) -> tuple[torch.Tensor, Factors]:
    """A fit as `(weights, factors)`; PARAFAC2's and CMF's mode 1 is the
    ragged list.

    CMF's subject mode is derived from its time courses (see `CMFModel`), so
    it is replaced by ones, which always match: scoring it would count the
    time courses twice.
    """
    if isinstance(model, CMFModel):
        return model.weights, [
            torch.ones_like(model.subject_mode),
            model.evolving_states,
            model.label_mode,
        ]
    if isinstance(model, PARAFAC2Model):
        return model.weights, [
            model.subject_mode,
            model.evolving_states,
            model.label_mode,
        ]
    assert model is not None
    weights, factors = model
    return weights, list(factors)


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

    For PARAFAC2 that stacking is exact rather than a convenience: the
    constraint fixes `||B_i[:, r]||` independent of `i`, so the stacked
    column cosine equals the mean per-subject cosine, and a subject with 2
    timepoints weighs the same as one with 40. CMF's `B_i` carry each
    subject's amplitude instead, so there the stacked cosine weighs subjects
    by their amplitude -- the comparison the model itself makes.

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
        """Initialize the replicability engine.

        The splits depend on `seed` alone, through the scikit-learn
        splitter's own `random_state` -- no global RNG is touched -- so every
        job of a distributed run regenerates identical splits.
        """
        self.seed = seed

    @abstractmethod
    def generate_tasks(
        self,
        n_tot: int,
        stratification: ArrayLike | None = None,
    ) -> Any:
        """Generate list of tasks for computing CP decompositions.

        Args:
            n_tot: Total number of samples
            stratification: Optional stratification labels for splitting;
                any labels numpy accepts (strings included), or a tensor.

        Returns:
            List of (task_id, indices) tuples

        Notes
        -----
        `inds`/`stratification` are index-bookkeeping arrays for
        scikit-learn's splitters, which need numpy-convertible CPU data, so
        they are always kept on CPU -- including a `stratification` passed in
        on another device. The decomposed tensors' device is handled
        separately, in `jobs.run_tasks`.
        """
        inds = np.arange(n_tot)
        if stratification is None:
            labels = np.ones(n_tot)
        elif isinstance(stratification, torch.Tensor):
            labels = stratification.cpu().numpy()
        else:
            labels = np.asarray(stratification)
        return inds, labels

    @abstractmethod
    def compute_fms(
        self,
        summaries: Mapping[Hashable, GroupSummary],
    ) -> Any:
        """Compute pairwise FMS scores from each group's best fit.

        `summaries` is `jobs.collect`'s output. A comparison involving a
        group with no successful fit scores NaN. The returned tuple format
        depends on the engine.
        """


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
        stratification: ArrayLike | None = None,
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
        summaries: Mapping[Hashable, GroupSummary],
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
        fms_results: list[tuple[int, float]] = []
        for s in range(self.repeats):
            best_0 = summaries[(s, 0)].best
            best_1 = summaries[(s, 1)].best
            if best_0 is None or best_1 is None:
                fms_results.append((s, float("nan")))
                continue
            weights_0, factors_0 = _weights_and_factors(best_0.model)
            weights_1, factors_1 = _weights_and_factors(best_1.model)

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
        stratification: ArrayLike | None = None,
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
        summaries: Mapping[Hashable, GroupSummary],
    ) -> list[tuple[np.ndarray, int, int, float]]:
        """Compute pairwise FMS between folds within each repeat.

        Returns (common_subjects, fold_i, fold_j, fms_score) tuples.
        """
        fms_results: list[tuple[np.ndarray, int, int, float]] = []
        for repeat in range(self.repeats):
            for split in range(self.splits):
                i = repeat * self.splits + split
                ids_i, best_i = summaries[i].indices, summaries[i].best
                fold_limit = (repeat + 1) * self.splits
                for j in range(i + 1, fold_limit):
                    ids_j, best_j = summaries[j].indices, summaries[j].best

                    common_subjects = np.intersect1d(ids_i, ids_j)
                    if len(common_subjects) == 0:
                        continue
                    if best_i is None or best_j is None:
                        fms_results.append((common_subjects, i, j, float("nan")))
                        continue
                    weights_i, fac_i = _weights_and_factors(best_i.model)
                    weights_j, fac_j = _weights_and_factors(best_j.model)

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


def evaluate_replicability_multiproc(
    replicability_engine: ReplicabilityEngine,
    tensor: torch.Tensor | list[torch.Tensor],
    rank: int,
    method: Method = "CP",
    stratification: ArrayLike | None = None,
    n_procs: int = 1,
    **CP_kwargs: Any,
) -> list[tuple[Any, ...]]:
    """Evaluate replicability using repeated CP, PARAFAC2 or CMF fits.

    The centralised form of `plan_replicability -> run_tasks -> collect ->
    compute_fms`; a distributed run of the same steps gives identical scores.

    Args:
        replicability_engine: Engine defining the replicability strategy
        tensor: Input to decompose. A regular tensor (samples × ...) for
            method="CP", or a list of per-subject slices (mode 1 may be
            ragged) for method="PARAFAC2" or "CMF".
        rank: Number of components for the decomposition
        method: "CP", "PARAFAC2" or "CMF"
        stratification: Optional stratification labels for splitting
        n_procs: Number of parallel processes, spread over every
            (group, seed) fit. Must be 1 on CUDA.
        **CP_kwargs: run_CP_decomposition_repeated /
            run_PARAFAC2_decomposition_repeated options, with their defaults.
            `init_repeats` is the number of restarts per group. This is how
            PARAFAC2-only options reach the solver -- `solver="matcouply"`,
            `nn_modes`, `aoadmm_options`, `aoadmm_loss_tolerance` -- which is
            why this module needs no solver-specific code. They `TypeError`
            with `method="CP"`. `restart_procs` is ignored.

    Returns:
        List of FMS score tuples (format depends on engine type). A split or
        fold pair whose group had no converged restart scores NaN, with a
        warning, rather than aborting the whole evaluation.

    Raises:
        ValueError: If `n_procs >= 2` and the fit runs on CUDA. Worker
            processes against a CUDA context from an already-initialized
            parent are unreliable across driver setups, so this is rejected
            rather than silently falling back to one process.
    """
    n_restarts = _resolve_options(method, CP_kwargs)["init_repeats"]
    plan = plan_replicability(
        replicability_engine,
        len(tensor),
        n_restarts,
        stratification,
    )
    store = InMemoryStore()
    run_tasks(plan, tensor, rank, method, store, n_procs=n_procs, **CP_kwargs)
    summaries = collect(plan, store, warn=True, method=method, **CP_kwargs)
    return replicability_engine.compute_fms(summaries)
