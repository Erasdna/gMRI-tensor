import dataclasses
import os
import warnings

import numpy as np
import pytest
import torch
from gMRItensor import compute_CP_decomposition
from gMRItensor import compute_PARAFAC2_decomposition
from gMRItensor import PARAFAC2Diagnostics
from gMRItensor import run_CP_decomposition_repeated
from gMRItensor import run_PARAFAC2_decomposition_repeated
from gMRItensor import setup_backend
from gMRItensor.decomposition import _in_worker_process
from gMRItensor.decomposition import _init_restart_worker_backend
from gMRItensor.decomposition import _nn_violations
from gMRItensor.decomposition import _resolve_nn_modes
from gMRItensor.decomposition import _suggest_max_iter
from gMRItensor.decomposition import ConvergenceError
from gMRItensor.plotting.evolving_mode import reconstruct_evolving_factors
from gMRItensor.plotting.utils import scale_mode
from tensorly.parafac2_tensor import Parafac2Tensor

SOLVERS = ["tensorly", "matcouply"]


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


def make_parafac2_slices(device, seed=0, sizes=(4, 5, 6), n_labels=5):
    """Ragged per-subject slices: `sizes` time points each, shared labels.

    Shared by the TensorLy and matcouply tests so both solvers are always
    exercised on byte-identical input.
    """
    rng = np.random.default_rng(seed)
    return [
        torch.from_numpy(rng.random((n_timepoints, n_labels))).to(device)
        for n_timepoints in sizes
    ]


def run_PARAFAC2(use_gpu):
    os.environ["GMRITENSOR_USE_GPU"] = use_gpu
    device = setup_backend()

    slices = make_parafac2_slices(device)
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

    slices = make_parafac2_slices(device)
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

    slices = make_parafac2_slices(device)
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
    slices = make_parafac2_slices(device)

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


# ---------------------------------------------------------------------------
# matcouply (AO-ADMM) solver
# ---------------------------------------------------------------------------


def test_PARAFAC2_matcouply_non_negative_mode1():
    # The whole reason the matcouply solver exists here: TensorLy's ALS
    # cannot enforce non-negativity on mode 1 (the evolving/time mode), and
    # AO-ADMM can.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    _, factors, projections, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=3,
        device=device,
        progress_bar=False,
        solver="matcouply",
        nn_modes=(0, 1, 2),
    )
    assert (factors[0] >= -1e-6).all()
    assert (factors[2] >= -1e-6).all()
    # factors[1] is AO-ADMM's *coordinate matrix*, not a per-subject factor,
    # and is expected to carry negative entries even under a fully
    # non-negative fit. The constraint holds on projections[i] @ factors[1],
    # which is the quantity with a physical meaning.
    for projection in projections:
        assert (projection @ factors[1] >= -1e-4).all()


def test_PARAFAC2_tensorly_rejects_nn_mode1():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    with pytest.raises(ValueError, match="matcouply"):
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            max_iter=50,
            init_repeats=1,
            device=device,
            progress_bar=False,
            nn_modes=(0, 1, 2),
        )


def test_PARAFAC2_default_nn_modes_is_solver_dependent():
    # Switching solver should switch the default along with it: TensorLy
    # cannot constrain mode 1, matcouply can and does by default.
    assert _resolve_nn_modes("auto", "tensorly") == (0, 2)
    assert _resolve_nn_modes("auto", "matcouply") == (0, 1, 2)
    # An explicit value, including None for "unconstrained", wins over both.
    assert _resolve_nn_modes(None, "matcouply") is None
    assert _resolve_nn_modes((0,), "matcouply") == (0,)


def test_PARAFAC2_matcouply_restores_torch_defaults():
    # matcouply only runs in float64, and only allocates on torch's default
    # device, so the fit temporarily changes both globals. Regression guard
    # on the context manager's `finally`: they must be restored on the happy
    # path AND when the fit is rejected.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)
    dtype_before = torch.get_default_dtype()
    device_before = torch.get_default_device()

    run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
    )
    assert torch.get_default_dtype() is dtype_before
    assert torch.get_default_device() == device_before

    with pytest.raises(ConvergenceError):
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            max_iter=1,
            init_repeats=2,
            device=device,
            progress_bar=False,
            solver="matcouply",
        )
    assert torch.get_default_dtype() is dtype_before
    assert torch.get_default_device() == device_before


def test_PARAFAC2_matcouply_restart_procs_parallel_succeeds():
    # Proves the float64/device setup reaches spawned workers: it travels
    # with the fit via _restart_worker rather than being installed by the
    # pool initializer.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    weights, _, _, error = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=4,
        device=device,
        progress_bar=False,
        solver="matcouply",
        restart_procs=2,
    )
    assert weights.shape == (2,)
    assert torch.isfinite(error)


def test_PARAFAC2_matcouply_normalize_factors():
    # matcouply has no normalize option of its own, so this is synthesised
    # from the orthonormality of the projections -- mirrors
    # test_PARAFAC2_normalize_factors for the TensorLy path.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    weights, factors, _, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=3,
        device=device,
        progress_bar=False,
        solver="matcouply",
        normalize=True,
    )
    for factor in factors:
        norms = torch.linalg.norm(factor, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert not torch.allclose(weights, torch.ones_like(weights))


def test_PARAFAC2_matcouply_projections_orthonormal():
    # Guards the undocumented matcouply layout this wrapper reads the
    # basis/coordinate matrices out of (auxes[1][0]): if a future matcouply
    # reorders its penalties, the "projections" would stop being orthonormal
    # here rather than failing loudly elsewhere.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    _, _, projections, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
    )
    identity = torch.eye(2)
    for projection in projections:
        assert torch.allclose(projection.T @ projection, identity, atol=1e-5)


@pytest.mark.parametrize(
    "bad_options",
    [{"not_a_real_option": 1}, {"n_iter_max": 10}, {"tol": 1e-3}],
)
def test_PARAFAC2_matcouply_rejects_bad_aoadmm_options(bad_options):
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    with pytest.raises(ValueError):
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            max_iter=50,
            init_repeats=1,
            device=device,
            progress_bar=False,
            solver="matcouply",
            aoadmm_options=bad_options,
        )


def test_PARAFAC2_rejects_unknown_solver():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    with pytest.raises(ValueError, match="Unknown PARAFAC2 solver"):
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            max_iter=50,
            init_repeats=1,
            device=device,
            progress_bar=False,
            solver="aoadmm",
        )


@pytest.mark.parametrize(
    "aoadmm_only_kwargs",
    [{"aoadmm_options": {"l1_penalty": 0.1}}, {"aoadmm_loss_tolerance": 1e-8}],
)
def test_PARAFAC2_rejects_aoadmm_only_options_on_tensorly(aoadmm_only_kwargs):
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    with pytest.raises(ValueError, match="matcouply"):
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            max_iter=50,
            init_repeats=1,
            device=device,
            progress_bar=False,
            **aoadmm_only_kwargs,
        )


def test_PARAFAC2_matcouply_rejects_nan():
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
            solver="matcouply",
        )


def test_PARAFAC2_matcouply_gpu():
    # Mirrors the existing _gpu pattern (falls through to CPU when CUDA is
    # absent). matcouply allocates its factors on torch's *default* device
    # rather than the input's, so without the scoped device override this
    # fails with a cpu/cuda mismatch.
    os.environ["GMRITENSOR_USE_GPU"] = "TRUE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    weights, factors, projections, error = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
    )
    assert weights.shape == (2,)
    assert [f.shape for f in factors] == [(3, 2), (2, 2), (5, 2)]
    assert [p.shape for p in projections] == [(4, 2), (5, 2), (6, 2)]
    assert torch.isfinite(error)


# ---------------------------------------------------------------------------
# Convergence reporting: two thresholds, and advice when they bite
# ---------------------------------------------------------------------------


def make_parafac2_shifted_gaussians(device, n_subjects=12, n_labels=20, rank=3):
    """Non-negative, approximately-PARAFAC2 data that needs real iterations.

    `make_parafac2_slices` is pure noise, which AO-ADMM settles on almost
    immediately -- useless for exercising the iteration-budget behaviour. The
    shifted-Gaussian time profiles here are the standard PARAFAC2 simulation
    and take hundreds of iterations to converge, which is what the
    reconstruction gate and the restart advisory are about.
    """
    rng = np.random.default_rng(0)
    subject = np.abs(rng.normal(size=(n_subjects, rank))) + 0.5
    labels = np.abs(rng.normal(size=(n_labels, rank))) + 0.5

    def evolving(n_timepoints, shift):
        time_grid = np.linspace(0, 10, n_timepoints)[:, None]
        centres = np.linspace(2.0, 8.0, rank)[None, :] + shift
        return np.exp(-((time_grid - centres) ** 2) / 2.0)

    return [
        torch.tensor(
            evolving(8 + i % 4, 0.1 * i) @ np.diag(subject[i]) @ labels.T,
            dtype=torch.float32,
        ).to(device)
        for i in range(n_subjects)
    ]


def test_PARAFAC2_matcouply_rejects_underconverged():
    # The acceptance gate: AO-ADMM stopping on its own penalized objective is
    # not enough -- if the reconstruction error is still moving, the fit is
    # rejected rather than silently returned.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_shifted_gaussians(device)

    with pytest.raises(ConvergenceError) as excinfo:
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=3,
            max_iter=150,
            init_repeats=2,
            device=device,
            progress_bar=False,
            solver="matcouply",
        )
    # The numbers saying what to change live on the chained cause.
    cause = excinfo.value.__cause__
    assert isinstance(cause, ConvergenceError)
    assert cause.reason == "reconstruction"
    assert "PARAFAC2_max_iter" in str(cause)
    assert "aoadmm_loss_tolerance" in str(cause)


def test_PARAFAC2_matcouply_accepts_at_iteration_limit():
    # The complement of the test above: running to the iteration limit is NOT
    # by itself a failure. matcouply's own stopping condition is on the
    # penalized objective and stays False here, yet the reconstruction error
    # has converged, so the fit is accepted -- and said so out loud.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_shifted_gaussians(device)

    with pytest.warns(UserWarning, match="accepted at the iteration limit"):
        *_, diagnostics = run_PARAFAC2_decomposition_repeated(
            slices,
            rank=3,
            max_iter=2000,
            init_repeats=1,
            device=device,
            progress_bar=False,
            solver="matcouply",
            return_diagnostics=True,
        )
    assert diagnostics.reached_max_iter
    assert diagnostics.loss_converged is False
    assert diagnostics.feasible is True
    # Accepted precisely because the comparable gate was met.
    assert (
        diagnostics.reconstruction_error_change < diagnostics.reconstruction_tolerance
    )


def test_PARAFAC2_matcouply_advises_raising_max_iter():
    # A run where restarts systematically run out of iterations should say so
    # once, with a concrete suggestion, rather than failing opaquely.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_shifted_gaussians(device)

    with pytest.raises(ConvergenceError) as excinfo:
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=3,
            max_iter=150,
            init_repeats=2,
            device=device,
            progress_bar=False,
            solver="matcouply",
        )
    assert "rejected" in str(excinfo.value)
    assert excinfo.value.__cause__.suggested_max_iter > 150


def test_PARAFAC2_no_advisory_when_healthy():
    # Guards against the nudge turning into noise: a run that converges
    # comfortably must say nothing at all.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            max_iter=2000,
            init_repeats=3,
            device=device,
            progress_bar=False,
            solver="matcouply",
        )
    advisories = [w for w in caught if "PARAFAC2(solver=" in str(w.message)]
    assert advisories == []


def make_decaying_rec_errors(decay=0.95, n_iter=400, step=0.1):
    """Error history whose per-iteration change decays geometrically."""
    rec_errors = [1.0]
    for i in range(n_iter):
        rec_errors.append(rec_errors[-1] - step * (decay**i))
    return rec_errors


@pytest.mark.parametrize("truncate_at", [100, 150])
def test_suggest_max_iter_extrapolation(truncate_at):
    # Geometrically decaying deltas with a known crossing point: extrapolating
    # from partway through must land at or past it (the 2x safety factor
    # makes the suggestion an over-estimate, never an under-estimate -- an
    # under-estimate would send the caller back for another failed run).
    tolerance = 1e-5
    rec_errors = make_decaying_rec_errors()
    deltas = [abs(b - a) for a, b in zip(rec_errors, rec_errors[1:])]
    true_crossing = next(i for i, d in enumerate(deltas) if d < tolerance)

    suggested = _suggest_max_iter(rec_errors[:truncate_at], tolerance)
    assert suggested is not None
    assert suggested >= true_crossing


def test_suggest_max_iter_returns_none_when_already_converged():
    # Nothing to suggest once the tolerance is already met.
    rec_errors = make_decaying_rec_errors()
    assert _suggest_max_iter(rec_errors, 1e-5) is None


def test_suggest_max_iter_returns_none_when_not_decaying():
    # Flat or rising error means more iterations are not the answer, so there
    # is nothing honest to suggest. The flat case is the subtle one: its
    # fitted slope is negative only through floating-point noise, which
    # without the extrapolation cap produced suggestions around 5e18.
    assert _suggest_max_iter([1.0, 0.5], 1e-5) is None
    assert _suggest_max_iter([1.0 + 0.1 * i for i in range(60)], 1e-5) is None
    assert _suggest_max_iter([1.1**i for i in range(60)], 1e-5) is None


def test_repeat_with_restarts_chains_last_error():
    # When every restart fails, the generic "nothing converged" message must
    # not swallow the per-restart detail -- that is where the actionable
    # numbers are. Uses the CP path, which shares the restart skeleton.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    tensor = make_low_rank_tensor()

    with pytest.raises(ConvergenceError) as excinfo:
        run_CP_decomposition_repeated(
            tensor,
            rank=2,
            max_iter=1,
            init_repeats=2,
            device=device,
            progress_bar=False,
        )
    assert isinstance(excinfo.value.__cause__, ConvergenceError)
    assert "rejected" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Solver interchangeability: the two solvers' outputs must be plug-and-play
# ---------------------------------------------------------------------------
#
# These assert *structure*, never numerical agreement. The two solvers
# optimise different objectives and will not land on the same factors; what
# must hold is that anything consuming one solver's output accepts the
# other's unchanged.


def fit_both_solvers(device, **overrides):
    """Fit the same slices with each solver, returning the full 5-tuples."""
    slices = make_parafac2_slices(device)
    kwargs = dict(
        rank=2,
        max_iter=500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        return_diagnostics=True,
    )
    kwargs.update(overrides)
    return slices, {
        solver: run_PARAFAC2_decomposition_repeated(slices, solver=solver, **kwargs)
        for solver in SOLVERS
    }


@pytest.mark.parametrize("solver", SOLVERS)
def test_PARAFAC2_return_contract(solver):
    # Single source of truth for the return contract, run against both
    # solvers so it cannot drift between them.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)
    common = dict(
        rank=2,
        max_iter=500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver=solver,
    )

    result = run_PARAFAC2_decomposition_repeated(slices, **common)
    assert len(result) == 4
    weights, factors, projections, error = result

    with_diagnostics = run_PARAFAC2_decomposition_repeated(
        slices,
        return_diagnostics=True,
        **common,
    )
    assert len(with_diagnostics) == 5
    assert isinstance(with_diagnostics[4], PARAFAC2Diagnostics)

    assert isinstance(weights, torch.Tensor)
    assert isinstance(factors, list) and len(factors) == 3
    assert isinstance(projections, list) and len(projections) == len(slices)
    assert isinstance(error, torch.Tensor)

    assert tuple(weights.shape) == (2,)
    assert [tuple(f.shape) for f in factors] == [(3, 2), (2, 2), (5, 2)]
    assert [tuple(p.shape) for p in projections] == [(4, 2), (5, 2), (6, 2)]

    # matcouply runs in float64 internally; it must still hand back float32
    # CPU tensors like the TensorLy path does.
    for tensor in [weights, error, *factors, *projections]:
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"

    assert error.numel() == 1
    assert torch.isfinite(error)
    assert error >= 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_compute_PARAFAC2_returns_parafac2_tensor(solver):
    # Anything typed against TensorLy's container keeps working for both
    # solvers -- the matcouply result is adapted, not a parallel type.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    result, errors, diagnostics = compute_PARAFAC2_decomposition(
        slices,
        rank=2,
        PARAFAC2_max_iter=500,
        solver=solver,
    )
    assert isinstance(result, Parafac2Tensor)
    assert isinstance(errors, list) and errors
    assert isinstance(diagnostics, PARAFAC2Diagnostics)
    assert diagnostics.solver == solver


def test_PARAFAC2_solvers_are_structurally_interchangeable():
    # Compares the two results to each other, which the parametrized contract
    # above cannot express: adding a field or changing a dtype on one path
    # without the other fails here.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    _, fits = fit_both_solvers(device)
    tensorly_fit, matcouply_fit = fits["tensorly"], fits["matcouply"]

    assert len(tensorly_fit) == len(matcouply_fit)
    for left, right in zip(tensorly_fit, matcouply_fit):
        assert type(left) is type(right)
        if isinstance(left, torch.Tensor):
            assert left.shape == right.shape
            assert left.dtype == right.dtype
            assert left.device == right.device
        elif isinstance(left, list):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                assert left_item.shape == right_item.shape
                assert left_item.dtype == right_item.dtype
                assert left_item.device == right_item.device


def test_PARAFAC2_diagnostics_fields_line_up_across_solvers():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    _, fits = fit_both_solvers(device)
    tensorly_diag = fits["tensorly"][4]
    matcouply_diag = fits["matcouply"][4]

    field_names = [f.name for f in dataclasses.fields(PARAFAC2Diagnostics)]
    assert [f.name for f in dataclasses.fields(tensorly_diag)] == field_names
    assert [f.name for f in dataclasses.fields(matcouply_diag)] == field_names

    # Comparable fields are populated for both solvers.
    comparable = [
        "relative_reconstruction_error",
        "reconstruction_error_change",
        "reconstruction_tolerance",
        "n_iter",
        "max_iter",
        "reached_max_iter",
    ]
    for field in comparable:
        assert getattr(tensorly_diag, field) is not None
        assert getattr(matcouply_diag, field) is not None

    # Solver-specific fields are present but empty on the TensorLy side,
    # which has no penalized objective and no feasibility notion.
    for field in [
        "loss_converged",
        "loss_tolerance",
        "feasible",
        "max_feasibility_gap",
    ]:
        assert getattr(tensorly_diag, field) is None
        assert getattr(matcouply_diag, field) is not None

    # The returned error is the quantity the diagnostics report.
    for solver in SOLVERS:
        error, diagnostics = fits[solver][3], fits[solver][4]
        assert float(error) == pytest.approx(
            diagnostics.relative_reconstruction_error,
            rel=1e-5,
        )


def test_PARAFAC2_one_kwargs_dict_routes_to_either_solver():
    # The property the docstrings claim: callers should not have to branch on
    # solver, including for nn_modes, whose default differs between them.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)
    shared = dict(
        rank=2,
        max_iter=500,
        init_repeats=2,
        tolerance=1e-4,
        device=device,
        progress_bar=False,
    )

    for solver in SOLVERS:
        weights, _, _, error = run_PARAFAC2_decomposition_repeated(
            slices,
            solver=solver,
            **shared,
        )
        assert tuple(weights.shape) == (2,)
        assert torch.isfinite(error)


# ---------------------------------------------------------------------------
# Non-negativity actually holds in the factors we hand back
# ---------------------------------------------------------------------------


def make_parafac2_unexpressed_component(device, n_subjects=16, n_labels=24, rank=2):
    """Data where some regions carry no signal, so a loading should be 0.

    This is the shape that exposes the bug: AO-ADMM enforces non-negativity
    exactly on its *auxiliary* variables and only to within the feasibility
    gap on the primal. A loading that should be exactly 0 lands at about
    -1e-07 in the primal, which is enough to flip the sign of everything it
    multiplies.
    """
    rng = np.random.default_rng(1)
    subject = np.abs(rng.normal(size=(n_subjects, rank))) + 0.5
    labels = np.abs(rng.normal(size=(n_labels, rank))) + 0.5
    labels[:5] = 0.0

    def evolving(n_timepoints, shift):
        time_grid = np.linspace(0, 10, n_timepoints)[:, None]
        centres = np.array([3.0, 6.0])[None, :] + shift
        return np.exp(-((time_grid - centres) ** 2) / 2.0)

    return [
        torch.tensor(
            evolving(12, 0.1 * i) @ np.diag(subject[i]) @ labels.T
            + 0.05 * rng.normal(size=(12, n_labels)),
            dtype=torch.float32,
        ).to(device)
        for i in range(n_subjects)
    ]


def test_PARAFAC2_matcouply_subject_mode_exactly_non_negative():
    # Regression test. The subject and region factors used to come from
    # matcouply's primal variables, which satisfy non-negativity only to
    # within the feasibility gap -- measured at -7.7e-07. They now come from
    # the auxiliaries, which satisfy it exactly, so `>= 0.0` holds strictly
    # rather than only `>= -1e-6`.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_unexpressed_component(device)

    _, factors, _, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=3,
        max_iter=1500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
    )
    assert factors[0].min() >= 0.0
    assert factors[2].min() >= 0.0


def test_PARAFAC2_matcouply_amplitude_scaled_profiles_keep_sign():
    # The reported symptom, end to end: scaling each subject's evolving
    # factor by its own loading and then normalising per subject. A negative
    # loading mirrors the whole curve, and scale_mode renormalises the
    # 1e-07-magnitude result back to full amplitude.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_unexpressed_component(device)

    _, factors, projections, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=3,
        max_iter=1500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
    )
    subject_mode, time_mode, roi_mode = factors
    evolving = reconstruct_evolving_factors(
        np.ones(time_mode.shape[-1]),
        (subject_mode, time_mode, roi_mode),
        projections,
    )
    amplitude_scaled = np.stack(evolving) * subject_mode.numpy()[:, None, :]
    scaled = np.stack([scale_mode(s) for s in amplitude_scaled])

    assert not np.isnan(scaled).any()
    for subject in range(scaled.shape[0]):
        for component in range(scaled.shape[2]):
            curve = scaled[subject, :, component]
            # A wholly-negative, full-amplitude curve is the bug's signature.
            assert not ((curve <= 1e-12).all() and np.abs(curve).max() > 0.1)


def test_PARAFAC2_unconstrained_mode_returns_primal():
    # Only constrained modes have a non-negativity auxiliary to read, so an
    # unconstrained mode must fall back to the primal rather than indexing
    # into an empty aux list.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    _, factors, _, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
        nn_modes=(0,),
    )
    assert factors[0].min() >= 0.0
    assert [tuple(f.shape) for f in factors] == [(3, 2), (2, 2), (5, 2)]


@pytest.mark.parametrize(
    ("solver", "expected"),
    [("tensorly", (0, 2)), ("matcouply", (0, 1, 2))],
)
def test_PARAFAC2_diagnostics_records_resolved_nn_modes(solver, expected):
    # A fit should be self-describing: without this, "which nn_modes did this
    # actually use?" cannot be answered after the fact.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    *_, diagnostics = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver=solver,
        return_diagnostics=True,
    )
    assert diagnostics.nn_modes == expected
    assert set(diagnostics.max_nn_violation) == set(expected)
    assert all(v < 1e-3 for v in diagnostics.max_nn_violation.values())


def test_nn_violation_is_relative_per_component():
    # Scoring relative to each component's own scale is what catches the bug:
    # a -1e-07 loading is invisible against a total norm of order 1, yet it
    # flips every curve it multiplies.
    weak = torch.tensor([[1.0, -1e-7], [2.0, 1e-7]])
    strong = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    projections = [torch.eye(2)]

    violations = _nn_violations([weak, strong, strong], projections, (0,))
    assert violations[0] == pytest.approx(1.0, rel=1e-6)

    clean = _nn_violations([strong, strong, strong], projections, (0,))
    assert clean[0] == 0.0
