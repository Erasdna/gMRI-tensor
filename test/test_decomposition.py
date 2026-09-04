import os

import numpy as np
import pytest
import torch
from gMRItensor import compute_CP_decomposition
from gMRItensor import run_CP_decomposition_repeated
from gMRItensor import run_PARAFAC2_decomposition_repeated
from gMRItensor import setup_backend
from gMRItensor.decomposition import ConvergenceError


def test_backend():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    assert device.type == "cpu"
    os.environ["GMRITENSOR_USE_GPU"] = "TRUE"
    device = setup_backend()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def run_CP(use_gpu):
    os.environ["GMRITENSOR_USE_GPU"] = use_gpu
    device = setup_backend()

    tensor_1 = np.outer(np.array([0, 0, 1, 0]), np.array([0, 1, 0, 0])).astype(float)
    tensor = torch.from_numpy(tensor_1).to(device)
    decomp, errors = compute_CP_decomposition(tensor, 1, 1000, 1)
    print(decomp)
    weights, factors = decomp
    print(weights)

    assert torch.allclose(weights, torch.ones(1, dtype=weights.dtype, device=device))


def test_CP_cpu():
    run_CP("FALSE")


def test_CP_gpu():
    run_CP("TRUE")


def make_low_rank_tensor(seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    a = rng.random((8, 2))
    b = rng.random((6, 2))
    c = rng.random((5, 2))
    tensor = np.einsum("ir,jr,kr->ijk", a, b, c) - shift
    return torch.from_numpy(tensor).float()


def test_CP_non_negative_default():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    tensor = make_low_rank_tensor()

    _, factors, _ = run_CP_decomposition_repeated(
        tensor,
        rank=2,
        CP_max_iter=500,
        CP_init_repeats=5,
        device=device,
        progress_bar=False,
    )
    assert all((f >= -1e-6).all() for f in factors)


def test_CP_plain_allows_negative_factors():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    # Shift the data below zero so an unconstrained fit genuinely needs
    # negative entries to fit well -- a non-negative fit could not reproduce
    # this without much higher error.
    tensor = make_low_rank_tensor(shift=0.3)

    _, factors, _ = run_CP_decomposition_repeated(
        tensor,
        rank=2,
        CP_max_iter=500,
        CP_init_repeats=5,
        device=device,
        progress_bar=False,
        non_negative=False,
    )
    assert any((f < 0).any() for f in factors)


def run_PARAFAC2(use_gpu):
    os.environ["GMRITENSOR_USE_GPU"] = use_gpu
    device = setup_backend()

    rng = np.random.default_rng(0)
    # Ragged: 3 subjects with 4/5/6 time points, sharing 5 labels/regions.
    slices = [
        torch.from_numpy(rng.random((n_timepoints, 5))).to(device)
        for n_timepoints in (4, 5, 6)
    ]
    weights, factors, projections, error = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        PARAFAC2_max_iter=200,
        PARAFAC2_init_repeats=3,
        device=device,
        progress_bar=False,
    )

    assert weights.shape == (2,)
    # factors = [A (subjects x rank), B (rank x rank), C (labels x rank)]
    assert [f.shape for f in factors] == [(3, 2), (2, 2), (5, 2)]
    assert [p.shape for p in projections] == [(4, 2), (5, 2), (6, 2)]
    assert error.numel() == 1


def test_PARAFAC2_cpu():
    run_PARAFAC2("FALSE")


def test_PARAFAC2_gpu():
    run_PARAFAC2("TRUE")


def test_PARAFAC2_no_convergence_raises():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()

    rng = np.random.default_rng(0)
    slices = [
        torch.from_numpy(rng.random((n_timepoints, 5))).to(device)
        for n_timepoints in (4, 5, 6)
    ]
    with pytest.raises(ConvergenceError):
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            PARAFAC2_max_iter=1,
            PARAFAC2_init_repeats=2,
            device=device,
            progress_bar=False,
        )


def make_low_rank_tensor_with_nan(seed=0):
    tensor = make_low_rank_tensor(seed=seed).clone()
    tensor[0, 0, 0] = torch.nan
    tensor[3, 2, 1] = torch.nan
    return tensor


def test_CP_nan_without_imputation_raises():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    setup_backend()
    tensor = make_low_rank_tensor_with_nan()

    with pytest.raises(ValueError):
        compute_CP_decomposition(tensor, rank=2, CP_max_iter=100, random_state=0)


def test_CP_nan_with_imputation_succeeds():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    tensor = make_low_rank_tensor_with_nan()

    weights, factors, error = run_CP_decomposition_repeated(
        tensor,
        rank=2,
        CP_max_iter=500,
        CP_init_repeats=5,
        device=device,
        progress_bar=False,
        allow_nan_imputation=True,
    )
    assert [f.shape for f in factors] == [(8, 2), (6, 2), (5, 2)]
    assert torch.isfinite(error)


def test_PARAFAC2_rejects_nan():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    tensor = make_low_rank_tensor_with_nan().to(device)
    slices = [tensor[i] for i in range(tensor.shape[0])]

    with pytest.raises(ValueError):
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            PARAFAC2_max_iter=50,
            PARAFAC2_init_repeats=2,
            device=device,
            progress_bar=False,
        )
