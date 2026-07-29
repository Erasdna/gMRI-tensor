from abc import ABC
from abc import abstractmethod
from multiprocessing import Pool

import numpy as np
import torch
from gMRItensor import run_CP_decomposition_repeated
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from tlviz.factor_tools import factor_match_score
from tqdm import tqdm


class ReplicabilityEngine(ABC):
    def __init__(
        self,
        device="cpu",
        seed=0,
    ):
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
        """Generate list for computing CP with multiprocessing"""
        inds = torch.arange(n_tot).to(self.device)
        if stratification is None:
            stratification = torch.ones(n_tot).to(self.device)

        return inds, stratification

    @abstractmethod
    def compute_fms(self, decomposition_results: dict) -> list:
        """Processes factor outputs and calculates pairwise FMS scores."""
        pass


class HalfHalfEngine(ReplicabilityEngine):
    def __init__(
        self,
        repeats: int,
        device: torch.device = "cpu",
        seed: int = 0,
    ):
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
    ):
        inds, stratification = super().generate_tasks(n_tot, stratification)
        tasks = []
        for i, el in enumerate(self.rskf.split(inds, y=stratification)):
            # Task ID: (split_index, half_index [0 or 1])
            tasks.append(((i, 0), el[0]))
            tasks.append(((i, 1), el[1]))

        return tasks

    def compute_fms(self, decomposition_results: dict) -> list:
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
    def __init__(
        self,
        splits: int,
        repeats: int,
        device: torch.device = "cpu",
        seed: int = 0,
    ):
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
        n_tot,
        stratification: torch.Tensor | None = None,
    ):
        inds, stratification = super().generate_tasks(n_tot, stratification)
        tasks = []
        for fold_idx, (train_idx, _) in enumerate(
            self.rskf.split(inds, y=stratification),
        ):
            tasks.append((fold_idx, train_idx.tolist()))
        return tasks

    def compute_fms(self, decomposition_results: dict) -> list:
        fms_results = []
        for i in range(self.nb_folds):  # Total nb of folds
            print(decomposition_results[i])
            ids_i, weights_i, fac_i = decomposition_results[i]

            for j in range(
                i + 1,
                self.nb_folds,
            ):  # Compare only with not yet seen combinations
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


def _decomposition_worker(task_args):
    task_id, indices, full_tensor, rank, CP_kwargs = task_args

    # Slice sub-tensor and move to target computation device
    sub_tensor = full_tensor[indices]
    weights, factors, _ = run_CP_decomposition_repeated(
        sub_tensor, rank=rank, device=full_tensor.device, **CP_kwargs
    )

    # Return CPU factor tensors
    factors = [f.cpu() for f in factors]
    weights = weights.cpu() if isinstance(weights, torch.Tensor) else weights

    return task_id, indices, weights, factors


def evaluate_replicability_multiproc(
    replicability_engine: ReplicabilityEngine,
    tensor: torch.Tensor,
    rank: int,
    stratification: torch.Tensor | None = None,
    n_procs: int = 1,
    **CP_kwargs,
):

    input = [
        (task_id, indices, tensor, rank, CP_kwargs)
        for task_id, indices in replicability_engine.generate_tasks(
            tensor.shape[0],
            stratification,
        )
    ]

    results_dict = {}
    if n_procs < 2 or tensor.device.type == "cuda":
        for task in tqdm(
            input,
            desc="Computing Decompositions sequential",
        ):
            task_id, indices, weights, factors = _decomposition_worker(task)
            results_dict[task_id] = (indices, weights, factors)
    else:
        with Pool(n_procs) as pool:
            for task_id, indices, weights, factors in tqdm(
                pool.imap_unordered(_decomposition_worker, input),
                total=len(input),
                desc="Computing Decompositions in parallel",
            ):
                results_dict[task_id] = (indices, weights, factors)

    return replicability_engine.compute_fms(results_dict)
