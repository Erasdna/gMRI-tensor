import os

import numpy as np
import torch
from gMRItensor import setup_backend
from gMRItensor.replicability import CrossValidationEngine
from gMRItensor.replicability import evaluate_replicability_multiproc
from gMRItensor.replicability import HalfHalfEngine


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
    tensor = torch.randn(30, 4, 10000).to(device)

    CV_splits = 5
    CV_repeats = 2
    CV_engine = CrossValidationEngine(
        splits=CV_splits,
        repeats=CV_repeats,
        device=device,
    )

    half_repeats = 10
    half_engine = HalfHalfEngine(
        repeats=half_repeats,
        device=device,
    )

    half_fms = evaluate_replicability_multiproc(
        half_engine,
        tensor,
        3,
        n_procs=procs,
        CP_init_repeats=10,
        CP_max_iter=5000,
        CP_verbose_level=0,
        CP_tolerance=1e-7,
        progress_bar=False,
    )
    print(half_fms)
    assert len(half_fms) == half_repeats

    CV_fms = evaluate_replicability_multiproc(
        CV_engine,
        tensor,
        3,
        n_procs=procs,
        CP_init_repeats=10,
        CP_max_iter=5000,
        CP_verbose_level=0,
        CP_tolerance=1e-7,
        progress_bar=False,
    )
    assert len(CV_fms) == np.cumsum(np.arange(CV_splits * CV_repeats))[-1]


def test_replicability_serial():
    run_replicability(1)


def test_replicability_parallel():
    run_replicability(10)


if __name__ == "__main__":
    print("--- Debugging Test ---")
    test_replicability_parallel()
    print("--- Test Completed Successfully ---")
