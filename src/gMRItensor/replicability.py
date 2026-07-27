import numpy as np
import torch
from gMRItensor import run_CP_decomposition_repeated
from sklearn.model_selection import RepeatedKFold
from tlviz.factor_tools import factor_match_score


def half_half_split_replicability(
    tensor: torch.Tensor, rank: int, splits: int, seed=0, **kwargs
):  # Split tensor along first dimension
    torch.manual_seed(seed)
    for n in range(splits):
        n_total = tensor.shape[0]
        shuffled_indices = torch.randperm(n_total).tolist()
        half_size = len(shuffled_indices) // 2

        half_half_factors = []
        for iids in [shuffled_indices[:half_size], shuffled_indices[half_size:]]:
            use_tensor = tensor[iids]
            weights, factors, best_error = run_CP_decomposition_repeated(
                use_tensor, rank, device=tensor.device, **kwargs
            )
            half_half_factors.append((weights, factors))

        fms = factor_match_score(
            half_half_factors[0],
            half_half_factors[1],
            skip_mode=0,
            consider_weights=False,
        )

        yield shuffled_indices, fms


def repeated_CV_replicability(
    tensor: torch.Tensor, rank: int, CV_repeats: int, CV_splits: int, seed=0, **kwargs
):
    rskf = RepeatedKFold(n_splits=CV_splits, n_repeats=CV_repeats, random_state=0)

    CV_dict: dict = {"repeat": [], "fold": [], "ids": [], "factors": []}
    for i, el in enumerate(rskf.split(torch.arange(tensor.shape[0]))):
        CV_dict["repeat"].append(i // CV_splits)
        CV_dict["fold"].append(i % CV_splits)
        CV_dict["ids"].append(el[0])

        use_tensor = tensor[el[0]]

        weights, factors, best_error = run_CP_decomposition_repeated(
            use_tensor,
            rank,
            device=tensor.device,
            **kwargs,
        )
        CV_dict["factors"].append(factors.copy())

    nb = CV_repeats * CV_splits
    assert nb == len(CV_dict["repeat"])
    for i in range(nb):
        for j in range(i + 1, nb):
            subject_intersection = list(set(CV_dict["ids"][i]) & set(CV_dict["ids"][j]))
            weights = torch.ones(rank)
            comps = []
            for ind in [i, j]:
                subject_iids = np.isin(CV_dict["ids"][ind], subject_intersection)

                fac = CV_dict["factors"][ind]
                factors = [fac[0][subject_iids]] + [
                    fac[i] for i in range(1, len(tensor.shape))
                ]
                comps.append((weights, factors))

            fms = factor_match_score(comps[0], comps[1], consider_weights=False)
            yield subject_intersection, i, j, fms
