import dataclasses
import inspect
import os
import warnings

import numpy as np
import pytest
import tensorly as tl
import torch
from gMRItensor import CMFModel
from gMRItensor import compute_CMF_decomposition
from gMRItensor import compute_CP_decomposition
from gMRItensor import compute_PARAFAC2_decomposition
from gMRItensor import PARAFAC2Diagnostics
from gMRItensor import PARAFAC2Model
from gMRItensor import run_CMF_decomposition_repeated
from gMRItensor import run_CP_decomposition_repeated
from gMRItensor import run_PARAFAC2_decomposition_repeated
from gMRItensor import setup_backend
from gMRItensor.decomposition import _in_worker_process
from gMRItensor.decomposition import _init_restart_worker_backend
from gMRItensor.decomposition import _nn_violations
from gMRItensor.decomposition import _resolve_nn_modes
from gMRItensor.decomposition import _suggest_max_iter
from gMRItensor.decomposition import _to_solver_slices
from gMRItensor.decomposition import _zero_negligible_loadings
from gMRItensor.decomposition import ConvergenceError
from gMRItensor.plotting.evolving_mode import evolving_factors_to_numpy
from gMRItensor.plotting.utils import scale_mode
from matcouply.decomposition import cmf_aoadmm
from matcouply.decomposition import parafac2_aoadmm

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
    model, error = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=200,
        init_repeats=3,
        device=device,
        progress_bar=False,
    )

    assert model.weights.shape == (2,)
    assert model.subject_mode.shape == (3, 2)
    assert model.label_mode.shape == (5, 2)
    # One (n_timepoints_i, rank) time course per subject.
    assert [b.shape for b in model.evolving_states] == [(4, 2), (5, 2), (6, 2)]
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
    model, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=200,
        init_repeats=3,
        device=device,
        progress_bar=False,
        normalize=True,
    )
    for factor in (model.subject_mode, model.label_mode):
        norms = torch.linalg.norm(factor, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    # The evolving mode shares one scalar per component across subjects, so
    # the unit norm lives on the cross-product, not each B_i's columns.
    gram = sum(b.T @ b for b in model.evolving_states) / len(model.evolving_states)
    assert torch.allclose(torch.diagonal(gram), torch.ones(2), atol=1e-3)


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

    model, error = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=200,
        init_repeats=4,
        device=device,
        progress_bar=False,
        restart_procs=2,
    )
    assert model.weights.shape == (2,)
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

    model, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=3,
        device=device,
        progress_bar=False,
        solver="matcouply",
        nn_modes=(0, 1, 2),
    )
    assert (model.subject_mode >= -1e-6).all()
    assert (model.label_mode >= -1e-6).all()
    # The evolving states are matcouply's primal, constrained directly --
    # not rebuilt from a projection, which is what used to flip signs.
    for evolving in model.evolving_states:
        assert (evolving >= -1e-4).all()


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
    # with the fit via _fit_one rather than being installed by the
    # pool initializer.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    model, error = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=4,
        device=device,
        progress_bar=False,
        solver="matcouply",
        restart_procs=2,
    )
    assert model.weights.shape == (2,)
    assert torch.isfinite(error)


def test_PARAFAC2_matcouply_normalize_factors():
    # matcouply has no normalize option of its own, so this is synthesised
    # from the orthonormality of the projections -- mirrors
    # test_PARAFAC2_normalize_factors for the TensorLy path.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    model, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=3,
        device=device,
        progress_bar=False,
        solver="matcouply",
        normalize=True,
    )
    for factor in (model.subject_mode, model.label_mode):
        norms = torch.linalg.norm(factor, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert not torch.allclose(model.weights, torch.ones_like(model.weights))


@pytest.mark.parametrize("solver", SOLVERS)
def test_PARAFAC2_return_projections_opt_in(solver):
    # The Kiers form is available on request but never by default: rebuilding
    # B_i = P_i @ Delta is what introduced per-subject sign flips, so the
    # default path must not depend on it.
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

    default_model, _ = run_PARAFAC2_decomposition_repeated(slices, **common)
    assert default_model.kiers is None

    model, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        return_projections=True,
        **common,
    )
    coordinate_matrix, projections = model.kiers
    assert tuple(coordinate_matrix.shape) == (2, 2)
    assert [tuple(p.shape) for p in projections] == [(4, 2), (5, 2), (6, 2)]
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

    model, error = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
    )
    assert model.weights.shape == (2,)
    assert model.subject_mode.shape == (3, 2)
    assert [b.shape for b in model.evolving_states] == [(4, 2), (5, 2), (6, 2)]
    assert torch.isfinite(error)


# ---------------------------------------------------------------------------
# compute_device: fit CPU-resident input elsewhere without a host-side copy
# ---------------------------------------------------------------------------


def test_to_solver_slices_casts_on_target_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slices = [s.float() for s in make_parafac2_slices("cpu")]
    originals = [s.clone() for s in slices]

    converted = _to_solver_slices(slices, device)

    for s, original, c in zip(slices, originals, converted):
        assert c.dtype == torch.float64
        assert c.device.type == device.type
        # The caller's input is left as it was.
        assert s.dtype == torch.float32
        assert s.device.type == "cpu"
        assert torch.equal(s, original)
        torch.testing.assert_close(c.cpu(), original.double())


@pytest.mark.parametrize("solver", SOLVERS)
def test_PARAFAC2_compute_device_none_preserves_behaviour(solver):
    setup_backend()
    slices = make_parafac2_slices("cpu")
    kwargs = dict(rank=2, PARAFAC2_max_iter=500, random_state=0, solver=solver)

    default_model, default_errors, _ = compute_PARAFAC2_decomposition(slices, **kwargs)
    explicit_model, explicit_errors, _ = compute_PARAFAC2_decomposition(
        slices, compute_device=torch.device("cpu"), **kwargs
    )

    torch.testing.assert_close(default_model.subject_mode, explicit_model.subject_mode)
    torch.testing.assert_close(default_model.label_mode, explicit_model.label_mode)
    assert [float(e) for e in default_errors] == [float(e) for e in explicit_errors]


def test_PARAFAC2_matcouply_cpu_input_cuda_compute():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    setup_backend()
    slices = [s.float() for s in make_parafac2_slices("cpu")]

    model, _, _ = compute_PARAFAC2_decomposition(
        slices,
        rank=2,
        PARAFAC2_max_iter=500,
        solver="matcouply",
        compute_device=torch.device("cuda"),
    )

    assert model.label_mode.device.type == "cuda"
    assert all(s.device.type == "cpu" and s.dtype == torch.float32 for s in slices)


def test_restart_procs_rejected_on_cuda_compute_device():
    # The guard runs before any work, so this needs no GPU.
    setup_backend()
    slices = make_parafac2_slices("cpu")

    with pytest.raises(ValueError, match="CUDA"):
        run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            max_iter=10,
            init_repeats=2,
            progress_bar=False,
            restart_procs=2,
            solver="matcouply",
            compute_device=torch.device("cuda"),
        )


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


def _library_tol(function):
    return inspect.signature(function).parameters["tol"].default


def test_tolerances_default_to_the_solvers_own(monkeypatch):
    # Every method defaults to its library's own `tol` instead of a value
    # hard-coded here; None means "the solver's default".
    for function in (
        run_CP_decomposition_repeated,
        run_PARAFAC2_decomposition_repeated,
    ):
        assert inspect.signature(function).parameters["tolerance"].default is None
    parameters = inspect.signature(run_PARAFAC2_decomposition_repeated).parameters
    assert parameters["aoadmm_loss_tolerance"].default is None

    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)
    *_, tensorly_diagnostics = compute_PARAFAC2_decomposition(
        slices,
        2,
        PARAFAC2_max_iter=5000,
    )
    assert tensorly_diagnostics.reconstruction_tolerance == _library_tol(
        tl.decomposition.parafac2,
    )
    *_, matcouply_diagnostics = compute_PARAFAC2_decomposition(
        slices,
        2,
        PARAFAC2_max_iter=5000,
        solver="matcouply",
    )
    assert matcouply_diagnostics.loss_tolerance == _library_tol(parafac2_aoadmm)
    # No extra reconstruction gate unless a tolerance is asked for.
    assert matcouply_diagnostics.reconstruction_tolerance is None
    assert matcouply_diagnostics.loss_converged is True

    seen = {}

    def spy(tensor, **kwargs):
        seen[kwargs["normalize_factors"]] = kwargs["tol"]
        raise ConvergenceError("stop")

    from gMRItensor import decomposition

    tensor = make_low_rank_tensor()
    for non_negative, library in [
        (True, tl.decomposition.non_negative_parafac),
        (False, tl.decomposition.parafac),
    ]:
        name = "non_negative_parafac_compiled" if non_negative else "parafac_compiled"
        monkeypatch.setattr(decomposition, name, spy)
        with pytest.raises(ConvergenceError):
            compute_CP_decomposition(tensor, 2, non_negative=non_negative)
        assert seen.pop(False) == _library_tol(library)


def test_PARAFAC2_matcouply_keeps_iteration_limit_fits_without_tolerance():
    # Like matcouply itself, a fit that runs out of iterations before its own
    # criterion is met is returned rather than rejected -- and said so.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_shifted_gaussians(device)

    with pytest.warns(UserWarning, match="stopped at the iteration limit"):
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
    assert diagnostics.reached_max_iter and diagnostics.loss_converged is False
    assert diagnostics.reconstruction_tolerance is None


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
            tolerance=1e-5,
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
            tolerance=1e-5,
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
            tolerance=1e-5,
        )
    assert "rejected" in str(excinfo.value)
    assert excinfo.value.__cause__.suggested_max_iter > 150


def test_PARAFAC2_healthy_run_is_silent():
    # Guards against the warnings turning into noise. A comfortable run must
    # emit neither the restart advisory nor the non-negativity warning -- the
    # latter only stays quiet because negligible residue is zeroed in every
    # constrained mode, so there is nothing left to report.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, _ = run_PARAFAC2_decomposition_repeated(
            slices,
            rank=2,
            max_iter=2000,
            init_repeats=3,
            device=device,
            progress_bar=False,
            solver="matcouply",
        )
    ours = [w for w in caught if "PARAFAC2(solver=" in str(w.message)]
    assert [str(w.message) for w in ours] == []

    # The reason it is silent: no negative residue survives anywhere.
    assert model.subject_mode.min() >= 0.0
    assert model.label_mode.min() >= 0.0
    assert min(float(b.min()) for b in model.evolving_states) >= 0.0


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
        tolerance=1e-5,  # the same gate for both, so the fields compare
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
    assert len(result) == 2
    model, error = result

    with_diagnostics = run_PARAFAC2_decomposition_repeated(
        slices,
        return_diagnostics=True,
        **common,
    )
    assert len(with_diagnostics) == 3
    assert isinstance(with_diagnostics[2], PARAFAC2Diagnostics)

    assert isinstance(model, PARAFAC2Model)
    assert isinstance(model.weights, torch.Tensor)
    assert isinstance(model.evolving_states, list)
    assert len(model.evolving_states) == len(slices)
    assert isinstance(error, torch.Tensor)

    assert tuple(model.weights.shape) == (2,)
    assert tuple(model.subject_mode.shape) == (3, 2)
    assert tuple(model.label_mode.shape) == (5, 2)
    assert [tuple(b.shape) for b in model.evolving_states] == [
        (4, 2),
        (5, 2),
        (6, 2),
    ]
    # Each subject's evolving state has that subject's own row count.
    assert [b.shape[0] for b in model.evolving_states] == [s.shape[0] for s in slices]

    # matcouply runs in float64 internally; it must still hand back float32
    # CPU tensors like the TensorLy path does.
    for tensor in [
        model.weights,
        model.subject_mode,
        model.label_mode,
        error,
        *model.evolving_states,
    ]:
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"

    assert error.numel() == 1
    assert torch.isfinite(error)
    assert error >= 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_compute_PARAFAC2_returns_model(solver):
    # Both solvers return the same coupled-matrix type, so callers never
    # branch on solver.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    result, errors, diagnostics = compute_PARAFAC2_decomposition(
        slices,
        rank=2,
        PARAFAC2_max_iter=500,
        solver=solver,
    )
    assert isinstance(result, PARAFAC2Model)
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
    left_model, right_model = tensorly_fit[0], matcouply_fit[0]
    for field in ("weights", "subject_mode", "label_mode"):
        left = getattr(left_model, field)
        right = getattr(right_model, field)
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        assert left.device == right.device
    assert len(left_model.evolving_states) == len(right_model.evolving_states)
    for left, right in zip(
        left_model.evolving_states,
        right_model.evolving_states,
    ):
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        assert left.device == right.device
    assert (left_model.kiers is None) == (right_model.kiers is None)


def test_PARAFAC2_diagnostics_fields_line_up_across_solvers():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    _, fits = fit_both_solvers(device)
    tensorly_diag = fits["tensorly"][2]
    matcouply_diag = fits["matcouply"][2]

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
        error, diagnostics = fits[solver][1], fits[solver][2]
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
        model, error = run_PARAFAC2_decomposition_repeated(
            slices,
            solver=solver,
            **shared,
        )
        assert tuple(model.weights.shape) == (2,)
        assert torch.isfinite(error)


# ---------------------------------------------------------------------------
# Non-negativity actually holds in the factors we hand back
# ---------------------------------------------------------------------------


def make_parafac2_unexpressed_component(device, n_subjects=16, n_labels=24, rank=2):
    """Data where some regions carry no signal, so a loading should be 0.

    A subject that barely expresses a component is where the old Kiers
    retrofit went wrong: the Procrustes fit for that direction was weakly
    determined and could land reflected, mirroring that subject's curve.
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

    model, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=3,
        max_iter=1500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
        tolerance=1e-5,
    )
    assert model.subject_mode.min() >= 0.0
    assert model.label_mode.min() >= 0.0


def test_PARAFAC2_matcouply_amplitude_scaled_profiles_keep_sign():
    # The reported symptom, end to end: scaling each subject's evolving
    # factor by its own loading and then normalising per subject. A negative
    # loading mirrors the whole curve, and scale_mode renormalises the
    # 1e-07-magnitude result back to full amplitude.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_unexpressed_component(device)

    model, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=3,
        max_iter=1500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
        tolerance=1e-5,
    )
    evolving = evolving_factors_to_numpy(model.evolving_states)
    amplitude_scaled = np.stack(evolving) * model.subject_mode.numpy()[:, None, :]
    scaled = np.stack([scale_mode(s) for s in amplitude_scaled])

    assert not np.isnan(scaled).any()
    for subject in range(scaled.shape[0]):
        for component in range(scaled.shape[2]):
            curve = scaled[subject, :, component]
            # A wholly-negative, full-amplitude curve is the bug's signature.
            assert not ((curve <= 1e-12).all() and np.abs(curve).max() > 0.1)


def test_PARAFAC2_partial_nn_modes():
    # Only mode 0 constrained: the other modes are free, and nothing in the
    # contract changes.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_slices(device)

    model, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=2,
        max_iter=500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
        tolerance=1e-5,
        nn_modes=(0,),
    )
    assert model.subject_mode.min() >= -1e-6
    assert tuple(model.subject_mode.shape) == (3, 2)
    assert [tuple(b.shape) for b in model.evolving_states] == [
        (4, 2),
        (5, 2),
        (6, 2),
    ]


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

    def model(subject_mode):
        return PARAFAC2Model(
            weights=torch.ones(2),
            subject_mode=subject_mode,
            evolving_states=[strong],
            label_mode=strong,
        )

    violations = _nn_violations(model(weak), (0,))
    assert violations[0] == pytest.approx(1.0, rel=1e-6)

    clean = _nn_violations(model(strong), (0,))
    assert clean[0] == 0.0


def test_zero_negligible_loadings_both_signs():
    # A loading negligible relative to its component means the subject does
    # not express it. Sign is irrelevant to that judgement -- and a small
    # POSITIVE loading is just as dangerous as a negative one, since
    # normalising the amplitude-scaled curve restores either to full
    # amplitude.
    factor = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, -3e-8],  # negligible, negative
            [4.0, 5e-8],  # negligible, positive
            [0.5, 0.25],  # small but real -- must survive
        ],
    )
    zeroed = _zero_negligible_loadings(factor, rtol=1e-6)

    assert zeroed[1, 1] == 0.0
    assert zeroed[2, 1] == 0.0
    assert zeroed[3, 1] == pytest.approx(0.25)
    # The other component is untouched: the tolerance is per component.
    assert torch.equal(zeroed[:, 0], factor[:, 0])
    # rtol=0 disables it entirely.
    assert torch.equal(_zero_negligible_loadings(factor, rtol=0.0), factor)


def test_PARAFAC2_negligible_loadings_become_exactly_zero():
    # End to end: an unexpressed subject-component should read as 0, not as
    # solver residue whose sign flips the whole curve.
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_parafac2_unexpressed_component(device)

    model, _ = run_PARAFAC2_decomposition_repeated(
        slices,
        rank=3,
        max_iter=1500,
        init_repeats=2,
        device=device,
        progress_bar=False,
        solver="matcouply",
        tolerance=1e-5,
    )
    # No residue of either sign survives in a constrained mode.
    assert model.subject_mode.min() >= 0.0
    assert model.label_mode.min() >= 0.0
    nonzero = model.subject_mode[model.subject_mode != 0]
    if nonzero.numel():
        scale = model.subject_mode.abs().max()
        assert (nonzero.abs() > 1e-6 * scale).all()


# ---------------------------------------------------------------------------
# Non-negative coupled matrix factorization (CMF)
# ---------------------------------------------------------------------------


def make_cmf_slices(device, n_subjects=6, n_labels=8, rank=2, seed=0):
    """A planted non-negative CMF: subject-specific time courses `B_i`, with
    different shapes per subject, and one shared label mode `C`."""
    rng = np.random.default_rng(seed)
    labels = rng.random((n_labels, rank))
    return [
        torch.from_numpy(rng.random((4 + i % 3, rank)) @ labels.T).float().to(device)
        for i in range(n_subjects)
    ]


def test_CMF_shape_contract():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_cmf_slices(device)

    model, errors, diagnostics = compute_CMF_decomposition(
        slices,
        2,
        CMF_max_iter=5000,
    )

    assert isinstance(model, CMFModel)
    assert torch.equal(model.weights, torch.ones(2, dtype=model.weights.dtype))
    assert tuple(model.label_mode.shape) == (8, 2)
    assert [tuple(B.shape) for B in model.evolving_states] == [
        (s.shape[0], 2) for s in slices
    ]
    # The subject mode is each subject's amplitude, derived from B_i.
    expected = torch.stack([B.pow(2).mean(dim=0).sqrt() for B in model.evolving_states])
    torch.testing.assert_close(model.subject_mode, expected)
    for factor in [model.label_mode, *model.evolving_states]:
        assert (factor >= 0).all()
    assert float(errors[-1]) < 0.05  # recovers the planted model
    # ... with the factors returned: A really stayed at ones.
    for X, B in zip(slices, model.evolving_states):
        residual = X.double() - B @ model.label_mode.T
        assert float(torch.linalg.norm(residual) / torch.linalg.norm(X)) < 0.05
    assert diagnostics.solver == "matcouply"
    assert diagnostics.nn_modes == (1, 2)
    assert diagnostics.loss_tolerance == _library_tol(cmf_aoadmm)


def test_CMF_normalize_moves_the_scale_into_the_time_courses():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_cmf_slices(device)

    plain, *_ = compute_CMF_decomposition(slices, 2, CMF_max_iter=5000)
    normalized, *_ = compute_CMF_decomposition(
        slices,
        2,
        CMF_max_iter=5000,
        normalize_factors=True,
    )

    torch.testing.assert_close(
        torch.linalg.norm(normalized.label_mode, dim=0),
        torch.ones(2, dtype=normalized.label_mode.dtype),
    )
    for B_plain, B_normalized in zip(plain.evolving_states, normalized.evolving_states):
        torch.testing.assert_close(
            B_normalized @ normalized.label_mode.T,
            B_plain @ plain.label_mode.T,
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.parametrize("key", ["parafac2", "update_A", "tol", "non_negative"])
def test_CMF_rejects_managed_options(key):
    slices = make_cmf_slices(torch.device("cpu"))
    with pytest.raises(ValueError, match="managed by compute_CMF_decomposition"):
        compute_CMF_decomposition(slices, 2, aoadmm_options={key: True})


def test_CMF_rejects_nan_and_unconverged_fits():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_cmf_slices(device)

    with pytest.raises(ValueError, match="NaN"):
        compute_CMF_decomposition(
            [slices[0] * float("nan"), *slices[1:]],
            2,
        )
    with pytest.raises(ConvergenceError) as excinfo:
        compute_CMF_decomposition(slices, 2, CMF_max_iter=2, CMF_tolerance=1e-12)
    assert excinfo.value.reason in ("reconstruction", "feasibility")


def test_CMF_defaults_accept_a_slowly_converging_fit():
    # Noisy data where matcouply's own 1e-8 criterion needs far more than the
    # default 2000 iterations: the default run still returns a model, with a
    # warning rather than "no decomposition converged".
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    rng = np.random.default_rng(1)
    slices = [
        s + 0.1 * float(s.std()) * torch.from_numpy(np.abs(rng.normal(size=s.shape)))
        for s in make_parafac2_shifted_gaussians(device)
    ]

    with pytest.warns(UserWarning, match="stopped at the iteration limit"):
        model, error = run_CMF_decomposition_repeated(
            [s.float() for s in slices],
            rank=3,
            init_repeats=1,
            device=device,
            progress_bar=False,
        )
    assert isinstance(model, CMFModel) and float(error) < 0.2


def test_run_CMF_restarts_in_parallel_match_sequential():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    slices = make_cmf_slices(device)
    shared = dict(
        rank=2,
        max_iter=5000,
        init_repeats=2,
        device=device,
        progress_bar=False,
    )

    sequential, sequential_error = run_CMF_decomposition_repeated(slices, **shared)
    parallel, parallel_error = run_CMF_decomposition_repeated(
        slices,
        restart_procs=2,
        **shared,
    )

    assert isinstance(sequential, CMFModel)
    torch.testing.assert_close(sequential.label_mode, parallel.label_mode)
    torch.testing.assert_close(sequential_error, parallel_error)
