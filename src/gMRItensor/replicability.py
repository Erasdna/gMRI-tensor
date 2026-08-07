from abc import ABC
from abc import abstractmethod
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch
from gMRItensor import run_CP_decomposition_repeated
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from tlviz.factor_tools import factor_match_score
from tqdm import tqdm


class ReplicabilityEngine(ABC):
    """Base class for replicability analysis engines."""

    def __init__(
        self,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        """Initialize the replicability engine.

        Args:
            device: Device to use for tensor operations ('cpu' or 'cuda')
            seed: Random seed for reproducibility
        """
        self.device = device
        self.seed = seed
        # Set seed
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
        """
        inds = torch.arange(n_tot).to(self.device)
        if stratification is None:
            stratification = torch.ones(n_tot).to(self.device)

        return inds, stratification

    @abstractmethod
    def compute_fms(
        self,
        decomposition_results: dict[Any, tuple[list[int], Any, list[torch.Tensor]]],
    ):
        """Processes factor outputs and calculates pairwise FMS scores.

        Args:
            decomposition_results: Dictionary mapping task_id to (indices, weights, factors)

        Returns:
            List of FMS score tuples (format depends on engine type)
        """
        pass


class HalfHalfEngine(ReplicabilityEngine):
    """Replicability engine using half-half splits."""

    def __init__(
        self,
        repeats: int,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        """Initialize half-half split engine.

        Args:
            repeats: Number of random half-half splits to perform
            device: Device to use for tensor operations
            seed: Random seed for reproducibility
        """
        super().__init__(device, seed)
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
        """Generate half-half split tasks.

        Args:
            n_tot: Total number of samples
            stratification: Optional stratification labels

        Returns:
            List of ((split_index, half_index), indices) tuples
        """
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
        """Compute FMS between paired halves.

        Args:
            decomposition_results: Dictionary mapping (split, half) to decomposition results

        Returns:
            List of (split_index, fms_score) tuples
        """
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
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        """Initialize cross-validation engine.

        Args:
            splits: Number of folds per repeat
            repeats: Number of times to repeat the cross-validation
            device: Device to use for tensor operations
            seed: Random seed for reproducibility
        """
        super().__init__(device, seed)
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
        """Generate cross-validation fold tasks.

        Args:
            n_tot: Total number of samples
            stratification: Optional stratification labels

        Returns:
            List of (fold_index, train_indices) tuples
        """
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

        Args:
            decomposition_results: Dictionary mapping fold_index to decomposition results

        Returns:
            List of (common_subjects, fold_i, fold_j, fms_score) tuples
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


def _decomposition_worker(
    task_args: tuple[Any, list[int], torch.Tensor, int, dict[str, Any]],
) -> tuple[Any, list[int], Any, list[torch.Tensor]]:
    """Worker function for parallel CP decomposition.

    Args:
        task_args: Tuple of (task_id, indices, full_tensor, rank, CP_kwargs)

    Returns:
        Tuple of (task_id, indices, weights, factors) with tensors on CPU
    """
    task_id, indices, full_tensor, rank, CP_kwargs = task_args

    try:
        # Slice sub-tensor (already on correct device)
        sub_tensor = full_tensor[indices]
        weights, factors, _ = run_CP_decomposition_repeated(
            sub_tensor, rank=rank, device=full_tensor.device, **CP_kwargs
        )

        # Move results to CPU to avoid device memory issues in multiprocessing
        factors = [f.cpu() for f in factors]
        weights = weights.cpu() if isinstance(weights, torch.Tensor) else weights

        return task_id, indices, weights, factors
    except Exception as e:
        raise RuntimeError(f"Decomposition failed for task {task_id}: {e}") from e


def evaluate_replicability_multiproc(
    replicability_engine: ReplicabilityEngine,
    tensor: torch.Tensor,
    rank: int,
    stratification: torch.Tensor | None = None,
    n_procs: int = 1,
    **CP_kwargs: Any,
) -> list[tuple[Any, ...]]:
    """Evaluate replicability using repeated CP decompositions.

    Args:
        replicability_engine: Engine defining the replicability strategy
        tensor: Input tensor to decompose (samples × features × ...)
        rank: Number of components for CP decomposition
        stratification: Optional stratification labels for splitting
        n_procs: Number of parallel processes (ignored if using CUDA)
        **CP_kwargs: Additional arguments passed to run_CP_decomposition_repeated

    Returns:
        List of FMS score tuples (format depends on engine type)
    """
    tasks = replicability_engine.generate_tasks(
        tensor.shape[0],
        stratification,
    )

    task_args = [
        (task_id, indices, tensor, rank, CP_kwargs) for task_id, indices in tasks
    ]

    results_dict: dict[Any, tuple[list[int], Any, list[torch.Tensor]]] = {}

    # Use sequential processing for CUDA (multiprocessing doesn't work well with CUDA)
    # or when n_procs < 2
    if n_procs < 2 or tensor.device.type == "cuda":
        for task in tqdm(
            task_args,
            desc="Computing decompositions (sequential)",
        ):
            task_id, indices, weights, factors = _decomposition_worker(task)
            results_dict[task_id] = (indices, weights, factors)
    else:
        with Pool(n_procs) as pool:
            for task_id, indices, weights, factors in tqdm(
                pool.imap_unordered(_decomposition_worker, task_args),
                total=len(task_args),
                desc=f"Computing decompositions (parallel, {n_procs} procs)",
            ):
                results_dict[task_id] = (indices, weights, factors)

    return replicability_engine.compute_fms(results_dict)
