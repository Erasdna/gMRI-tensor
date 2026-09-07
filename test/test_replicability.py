import os

import pytest
import torch
from gMRItensor import setup_backend
from gMRItensor.replicability import CrossValidationEngine
from gMRItensor.replicability import evaluate_replicability_multiproc
from gMRItensor.replicability import HalfHalfEngine
from scipy.special import comb


def test_half_half_engine_input():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    repeats = 100
    engine = HalfHalfEngine(
        repeats=repeats,
        device=device,
    )
    n_tot = 30
    tasks = engine.generate_tasks(n_tot)
    assert len(tasks) == repeats * 2
    for task in tasks:
        assert len(task[1]) == n_tot // 2


def test_CV_engine_input():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()

    splits = 10
    repeats = 10
    engine = CrossValidationEngine(
        splits=splits,
        repeats=repeats,
        device=device,
    )
    tasks = engine.generate_tasks(30)

    assert len(tasks) == engine.nb_folds
    assert engine.nb_folds == splits * repeats


def run_replicability(procs):
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    # This only exercises the multiproc/engine plumbing (task counts, result
    # shapes) -- not decomposition accuracy -- so keep the tensor, restart
    # count and iteration budget small; a much bigger fit here previously
    # made this test take several minutes for no added coverage.
    tensor = torch.randn(10, 5, 6).to(device)

    CV_splits = 3
    CV_repeats = 1
    CV_engine = CrossValidationEngine(
        splits=CV_splits,
        repeats=CV_repeats,
        device=device,
    )

    half_repeats = 2
    half_engine = HalfHalfEngine(
        repeats=half_repeats,
        device=device,
    )

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

    engine = HalfHalfEngine(repeats=1, device=device)
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

    engine = HalfHalfEngine(repeats=1, device=device)
    fms = evaluate_replicability_multiproc(
        engine,
        tensor,
        2,
        n_procs=1,
        max_iter=10,
        init_repeats=1,
        progress_bar=False,
    )
    assert len(fms) == 1


def run_replicability_parafac2(procs):
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    # Ragged: 12 subjects with 4-6 time points each, sharing 20 labels/regions.
    tensor = [torch.randn(4 + (i % 3), 20).to(device).abs() for i in range(12)]

    CV_splits = 3
    CV_repeats = 1
    CV_engine = CrossValidationEngine(
        splits=CV_splits,
        repeats=CV_repeats,
        device=device,
    )

    half_repeats = 3
    half_engine = HalfHalfEngine(
        repeats=half_repeats,
        device=device,
    )

    half_fms = evaluate_replicability_multiproc(
        half_engine,
        tensor,
        2,
        method="PARAFAC2",
        n_procs=procs,
        init_repeats=3,
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
        method="PARAFAC2",
        n_procs=procs,
        init_repeats=3,
        max_iter=100,
        verbose_level=0,
        tolerance=1e-4,
        progress_bar=False,
    )
    assert len(CV_fms) == CV_repeats * comb(CV_splits, 2, exact=True)


def test_replicability_parafac2_serial():
    run_replicability_parafac2(1)


if __name__ == "__main__":
    print("--- Debugging Test ---")
    test_replicability_parallel()
    print("--- Test Completed Successfully ---")
