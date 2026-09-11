import os

import numpy as np
import pytest
import torch
from gMRItensor import compute_CP_decomposition
from gMRItensor import run_CP_decomposition_repeated
from gMRItensor import run_PARAFAC2_decomposition_repeated
from gMRItensor import setup_backend
from gMRItensor.decomposition import _in_worker_process
from gMRItensor.decomposition import _init_restart_worker_backend
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
        max_iter=500,
        init_repeats=5,
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
        max_iter=500,
        init_repeats=5,
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
        max_iter=200,
        init_repeats=3,
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


def test_PARAFAC2_normalize_factors():
    # `normalize` wasn't previously exposed by run_PARAFAC2_decomposition_repeated
    # even though compute_PARAFAC2_decomposition already supported it -- with
    # normalize=True, every factor matrix's columns should come out unit-norm.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()

    rng = np.random.default_rng(0)
    slices = [
        torch.from_numpy(rng.random((n_timepoints, 5))).to(device)
        for n_timepoints in (4, 5, 6)
    ]
    _, factors, _, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=200,
        init_repeats=3,
        device=device,
        progress_bar=False,
        normalize=True,
    )
    for factor in factors:
        norms = torch.linalg.norm(factor, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


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
            max_iter=1,
            init_repeats=2,
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
        max_iter=500,
        init_repeats=5,
        device=device,
        progress_bar=False,
        allow_nan_imputation=True,
    )
    assert [f.shape for f in factors] == [(8, 2), (6, 2), (5, 2)]
    assert torch.isfinite(error)


def test_in_worker_process_false_in_main_process():
    assert not _in_worker_process()


def test_init_restart_worker_backend_caps_threads():
    # Regression test: torch.set_num_threads doesn't survive into a spawned
    # worker process, so left unset each worker fell back to its own
    # (often much larger) default thread pool -- restart_procs processes
    # each ALSO fanning out into a full thread pool oversubscribed the CPU
    # rather than actually parallelizing across restart_procs total threads.
    original = torch.get_num_threads()
    try:
        _init_restart_worker_backend(2)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(original)


def test_CP_restart_procs_parallel_converges_as_well_as_sequential():
    # Splitting restarts across worker processes should reach comparably low
    # reconstruction error to running them sequentially in-process. Not
    # asserted bit-identical: separate processes can pick up different BLAS
    # thread counts, so the exact floating-point trajectory (and possibly
    # which restart ends up best) isn't guaranteed to match, only the
    # resulting fit quality.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    tensor = make_low_rank_tensor()

    _, _, error_seq = run_CP_decomposition_repeated(
        tensor,
        rank=2,
        max_iter=500,
        init_repeats=6,
        device=device,
        progress_bar=False,
        restart_procs=1,
    )
    _, factors_par, error_par = run_CP_decomposition_repeated(
        tensor,
        rank=2,
        max_iter=500,
        init_repeats=6,
        device=device,
        progress_bar=False,
        restart_procs=2,
    )
    assert torch.isfinite(error_par)
    assert error_par < 10 * error_seq
    assert all((f >= -1e-6).all() for f in factors_par)  # non_negative default


def test_PARAFAC2_restart_procs_parallel_succeeds():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    rng = np.random.default_rng(0)
    slices = [
        torch.from_numpy(rng.random((n_timepoints, 5))).to(device)
        for n_timepoints in (4, 5, 6)
    ]

    weights, factors, projections, error = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=200,
        init_repeats=4,
        device=device,
        progress_bar=False,
        restart_procs=2,
    )
    assert weights.shape == (2,)
    assert torch.isfinite(error)


def test_restart_procs_rejected_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    os.environ["GMRITENSOR_USE_GPU"] = "TRUE"
    device = setup_backend()
    tensor = make_low_rank_tensor().to(device)

    with pytest.raises(ValueError, match="CUDA"):
        run_CP_decomposition_repeated(
            tensor,
            rank=2,
            max_iter=10,
            init_repeats=2,
            device=device,
            progress_bar=False,
            restart_procs=2,
        )


def test_restart_procs_falls_back_inside_worker_process(monkeypatch, capsys):
    # Guards against nesting process pools: e.g. inside one of
    # evaluate_replicability_multiproc's own workers, restart_procs should
    # silently drop to 1 (sequential) instead of spawning a second layer of
    # processes.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    tensor = make_low_rank_tensor()

    monkeypatch.setattr(
        "gMRItensor.decomposition._in_worker_process",
        lambda: True,
    )
    _, _, error = run_CP_decomposition_repeated(
        tensor,
        rank=2,
        max_iter=500,
        init_repeats=3,
        device=device,
        progress_bar=False,
        restart_procs=4,
        verbose_level=1,
    )
    assert torch.isfinite(error)
    assert "falling back to restart_procs=1" in capsys.readouterr().out


def test_PARAFAC2_rejects_nan():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    tensor = make_low_rank_tensor_with_nan().to(device)
    slices = [tensor[i] for i in range(tensor.shape[0])]

    with pytest.raises(ValueError):
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            max_iter=50,
            init_repeats=2,
            device=device,
            progress_bar=False,
        )
