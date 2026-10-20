import math
import os
from collections.abc import Iterator

import numpy as np
import pytest
import tensorly as tl
import torch
from gMRItensor import CMFModel
from gMRItensor import setup_backend
from gMRItensor.replicability import _weights_and_factors
from gMRItensor.replicability import compute_FMS
from gMRItensor.replicability import CrossValidationEngine
from gMRItensor.replicability import evaluate_replicability_multiproc
from gMRItensor.replicability import HalfHalfEngine
from scipy.special import comb


@pytest.fixture
def numpy_backend() -> Iterator[None]:
    """TensorLy's default backend, as in a fresh `collect` process that
    never calls `setup_backend`; the previous backend is restored after."""
    previous = tl.get_backend()
    tl.set_backend("numpy")
    yield
    tl.set_backend(previous)


def test_compute_FMS_accepts_torch_fits_on_numpy_backend(numpy_backend: None) -> None:
    # Regression test: `gmri replicability collect` loads torch fits in a
    # fresh process, and tlviz rejected their torch weights there.
    generator = torch.Generator().manual_seed(0)
    factors = [torch.randn(n, 2, generator=generator) for n in (5, 4, 3)]
    weights = torch.ones(2)
    # The same components, reordered, rescaled and with two signs flipped.
    swapped = [
        -factors[0][:, [1, 0]],
        -factors[1][:, [1, 0]],
        3 * factors[2][:, [1, 0]],
    ]

    score = compute_FMS((weights, factors), (weights, swapped))

    assert isinstance(score, float)
    assert score == pytest.approx(1.0)


def test_half_half_engine_input():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    setup_backend()
    repeats = 100
    engine = HalfHalfEngine(repeats=repeats)
    n_tot = 30
    tasks = engine.generate_tasks(n_tot)
    assert len(tasks) == repeats * 2
    for task in tasks:
        assert len(task[1]) == n_tot // 2


def test_CV_engine_input():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    setup_backend()

    splits = 10
    repeats = 10
    engine = CrossValidationEngine(splits=splits, repeats=repeats)
    tasks = engine.generate_tasks(30)

    assert len(tasks) == engine.nb_folds
    assert engine.nb_folds == splits * repeats


def run_replicability(procs):
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    # This only exercises the multiproc/engine plumbing (task counts, result
    # shapes) -- not decomposition accuracy -- so keep the tensor, restart
    # count and iteration budget small.
    tensor = torch.randn(10, 5, 6).to(device)

    CV_splits = 3
    CV_repeats = 1
    CV_engine = CrossValidationEngine(splits=CV_splits, repeats=CV_repeats)

    half_repeats = 2
    half_engine = HalfHalfEngine(repeats=half_repeats)

    half_fms = evaluate_replicability_multiproc(
        half_engine,
        tensor,
        2,
        n_procs=procs,
        init_repeats=2,
        max_iter=100,
        verbose_level=0,
        tolerance=1e-4,
        progress_bar=False,
    )
    assert len(half_fms) == half_repeats

    CV_fms = evaluate_replicability_multiproc(
        CV_engine,
        tensor,
        2,
        n_procs=procs,
        init_repeats=2,
        max_iter=100,
        verbose_level=0,
        tolerance=1e-4,
        progress_bar=False,
    )
    assert len(CV_fms) == CV_repeats * comb(CV_splits, 2, exact=True)


def test_replicability_serial():
    run_replicability(1)


def test_replicability_parallel():
    run_replicability(4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_replicability_multiproc_rejects_cuda():
    # Running worker processes against an already-CUDA-initialized parent is
    # unsafe/unreliable across GPU driver setups, so n_procs >= 2 with a CUDA
    # tensor must raise rather than silently fall back to sequential.
    os.environ["GMRITENSOR_USE_GPU"] = "TRUE"
    device = setup_backend()
    tensor = torch.randn(6, 4, 5).abs().to(device)

    engine = HalfHalfEngine(repeats=1)
    with pytest.raises(ValueError, match="CUDA"):
        evaluate_replicability_multiproc(
            engine,
            tensor,
            2,
            n_procs=2,
            max_iter=10,
            init_repeats=1,
            progress_bar=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_replicability_cuda_sequential_default_stratification():
    # Regression test: ReplicabilityEngine.generate_tasks used to build its
    # default stratification array on the engine's device, which broke
    # scikit-learn's splitter for a CUDA engine even with n_procs=1 (no
    # multiprocessing involved at all) -- it needs plain CPU/numpy-
    # convertible data, not the tensor being decomposed.
    os.environ["GMRITENSOR_USE_GPU"] = "TRUE"
    device = setup_backend()
    tensor = torch.randn(6, 4, 5).abs().to(device)

    engine = HalfHalfEngine(repeats=1)
    fms = evaluate_replicability_multiproc(
        engine,
        tensor,
        2,
        n_procs=1,
        max_iter=100,
        init_repeats=2,
        progress_bar=False,
    )
    assert len(fms) == 1


def run_replicability_parafac2(procs, solver="tensorly", method="PARAFAC2"):
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    # Ragged: 12 subjects with 4-6 time points each, sharing 20 labels/regions.
    # .abs() so the non-negativity-constrained default fit is well-posed.
    tensor = [torch.randn(4 + (i % 3), 20).to(device).abs() for i in range(12)]

    CV_splits = 3
    CV_repeats = 1
    CV_engine = CrossValidationEngine(splits=CV_splits, repeats=CV_repeats)

    half_repeats = 3
    half_engine = HalfHalfEngine(repeats=half_repeats)

    # `solver` reaches run_PARAFAC2_decomposition_repeated through
    # evaluate_replicability_multiproc's **CP_kwargs, which is why the
    # replicability path needed no changes of its own.
    common = dict(
        method=method,
        n_procs=procs,
        init_repeats=3,
        max_iter=500,
        verbose_level=0,
        tolerance=1e-4,
        progress_bar=False,
    )
    if method == "PARAFAC2":
        common["solver"] = solver

    half_fms = evaluate_replicability_multiproc(
        half_engine,
        tensor,
        2,
        **common,
    )
    assert len(half_fms) == half_repeats

    CV_fms = evaluate_replicability_multiproc(
        CV_engine,
        tensor,
        2,
        **common,
    )
    assert len(CV_fms) == CV_repeats * comb(CV_splits, 2, exact=True)
    return half_fms, CV_fms


def test_replicability_parafac2_serial():
    run_replicability_parafac2(1)


@pytest.mark.parametrize("solver", ["tensorly", "matcouply"])
def test_replicability_parafac2_accepts_either_solver(solver):
    # The plug-and-play claim at the top level: FMS scoring consumes
    # (weights, factors) from either solver without branching.
    half_fms, CV_fms = run_replicability_parafac2(1, solver=solver)
    for scores in (half_fms, CV_fms):
        for entry in scores:
            fms = entry[-1]
            assert math.isfinite(float(fms))


def test_replicability_cmf_scores_are_finite():
    half_fms, CV_fms = run_replicability_parafac2(1, method="CMF")
    for scores in (half_fms, CV_fms):
        for entry in scores:
            assert 0.0 <= float(entry[-1]) <= 1.0 + 1e-6


def test_cmf_subject_mode_is_not_scored():
    # CMF's subject mode is derived from B_i, so scoring it would count the
    # time courses twice; it is replaced by ones, which always match.
    model = CMFModel(
        weights=torch.ones(2),
        subject_mode=torch.rand(3, 2),
        evolving_states=[torch.rand(4, 2) for _ in range(3)],
        label_mode=torch.rand(5, 2),
    )
    _, (subject_mode, evolving, label_mode) = _weights_and_factors(model)
    assert torch.equal(subject_mode, torch.ones(3, 2))
    assert evolving is model.evolving_states
    assert label_mode is model.label_mode


def _cp_fit(seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    return np.ones(3), [rng.random((n, 3)) for n in (6, 5, 4)]


def test_compute_fms_is_one_for_identical_fits():
    fit = _cp_fit()
    assert compute_FMS(fit, fit) == pytest.approx(1.0)


def test_compute_fms_ignores_permutation_and_scaling():
    weights, factors = _cp_fit()
    order = [2, 0, 1]
    scaled = [factor[:, order] * 3.0 for factor in factors]
    assert compute_FMS((weights, factors), (weights * 5, scaled)) == pytest.approx(
        1.0,
    )


def test_compute_fms_is_invariant_to_how_components_are_scaled() -> None:
    """Normalising a fit (one scalar per component and mode, moved into the
    weights) leaves its FMS unchanged: tlviz normalises the columns itself,
    and `consider_weights=False` ignores the weights. The stacked evolving
    mode of PARAFAC2/CMF is rescaled the same way, one scalar per component."""
    weights, factors = _cp_fit()
    rng = np.random.default_rng(1)
    scales = [rng.uniform(0.1, 10.0, 3) for _ in factors]
    rescaled = [factor * scale for factor, scale in zip(factors, scales)]
    new_weights = weights / np.prod(scales, axis=0)
    assert compute_FMS((weights, factors), (new_weights, rescaled)) == pytest.approx(
        1.0,
    )
    assert compute_FMS((weights, factors), (weights * 1e6, factors)) == pytest.approx(
        1.0,
    )


def test_compute_fms_skip_mode_ignores_that_mode():
    weights, factors = _cp_fit(0)
    _, other = _cp_fit(1)
    mixed = [other[0], factors[1], factors[2]]
    assert compute_FMS((weights, factors), (weights, mixed)) < 0.99
    assert compute_FMS(
        (weights, factors),
        (weights, mixed),
        skip_mode=0,
    ) == pytest.approx(1.0)
