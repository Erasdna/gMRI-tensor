import math
import os

import numpy as np
import pytest
import torch
from gMRItensor import collect
from gMRItensor import ConvergenceError
from gMRItensor import DirectoryStore
from gMRItensor import fit_task
from gMRItensor import FitTask
from gMRItensor import InMemoryStore
from gMRItensor import job_slice
from gMRItensor import n_jobs
from gMRItensor import PARAFAC2Diagnostics
from gMRItensor import PARAFAC2Model
from gMRItensor import plan_replicability
from gMRItensor import plan_restarts
from gMRItensor import RestartResult
from gMRItensor import run_CP_decomposition_repeated
from gMRItensor import run_PARAFAC2_decomposition_repeated
from gMRItensor import run_tasks
from gMRItensor import setup_backend
from gMRItensor.replicability import CrossValidationEngine
from gMRItensor.replicability import evaluate_replicability_multiproc
from gMRItensor.replicability import HalfHalfEngine


@pytest.fixture(autouse=True)
def cpu_backend() -> None:
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    setup_backend()


def make_slices(n_subjects: int = 10, seed: int = 0) -> list[torch.Tensor]:
    """Ragged, non-negative per-subject slices with a shared label mode."""
    rng = np.random.default_rng(seed)
    return [
        torch.from_numpy(rng.random((4 + i % 3, 7))).float() for i in range(n_subjects)
    ]


def make_tensor(n_subjects: int = 10, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    a, b, c = rng.random((n_subjects, 2)), rng.random((6, 2)), rng.random((5, 2))
    return torch.from_numpy(np.einsum("ir,jr,kr->ijk", a, b, c)).float()


FIT_OPTIONS = dict(max_iter=300, tolerance=1e-4, progress_bar=False)


def fake_model(scale: float) -> PARAFAC2Model:
    return PARAFAC2Model(
        weights=torch.ones(2) * scale,
        subject_mode=torch.ones(3, 2),
        evolving_states=[torch.ones(4, 2), torch.ones(5, 2), torch.ones(4, 2)],
        label_mode=torch.ones(6, 2),
    )


def fake_diagnostics() -> PARAFAC2Diagnostics:
    return PARAFAC2Diagnostics(
        solver="tensorly",
        n_iter=10,
        max_iter=100,
        reached_max_iter=False,
        relative_reconstruction_error=0.1,
        reconstruction_error_change=1e-6,
        reconstruction_tolerance=1e-5,
        nn_modes=(0, 2),
        max_nn_violation={0: 0.0, 2: 0.0},
    )


# --- Planning -------------------------------------------------------------


def test_plan_restarts_is_ordered_by_seed():
    plan = plan_restarts(4, 3)
    assert plan == [FitTask("full", (0, 1, 2, 3), seed) for seed in range(3)]


def test_plan_replicability_is_deterministic_across_engines():
    # Every distributed job builds its own engine; they must agree exactly,
    # which only holds because no global RNG feeds the splits.
    first = plan_replicability(HalfHalfEngine(3, seed=7), 20, 2)
    torch.manual_seed(123)
    np.random.seed(123)
    second = plan_replicability(HalfHalfEngine(3, seed=7), 20, 2)
    assert first == second
    assert len(first) == 3 * 2 * 2
    # Grouped, then seeded.
    assert [t.group for t in first[:4]] == [(0, 0), (0, 0), (0, 1), (0, 1)]
    assert [t.seed for t in first[:4]] == [0, 1, 0, 1]


def test_stratified_halves_are_disjoint_and_cover_everyone():
    labels = ["HC"] * 10 + ["PD"] * 10
    plan = plan_replicability(HalfHalfEngine(4, seed=0), 20, 1, labels)
    halves = {task.group: set(task.indices) for task in plan}
    for split in range(4):
        half_0, half_1 = halves[(split, 0)], halves[(split, 1)]
        assert not half_0 & half_1
        assert half_0 | half_1 == set(range(20))
        # Stratified: each half gets half of each label.
        assert sum(i < 10 for i in half_0) == 5


@pytest.mark.parametrize("tasks_per_job", [1, 3, 5, 100])
def test_job_slices_cover_plan_exactly_once(tasks_per_job):
    plan = plan_replicability(CrossValidationEngine(3, 2), 12, 2)
    slices = [
        job_slice(plan, j, tasks_per_job) for j in range(n_jobs(plan, tasks_per_job))
    ]
    assert [task for block in slices for task in block] == plan


def test_job_slice_rejects_out_of_range():
    plan = plan_restarts(5, 4)
    with pytest.raises(IndexError):
        job_slice(plan, 2, 2)
    with pytest.raises(IndexError):
        job_slice(plan, -1, 2)
    with pytest.raises(ValueError):
        n_jobs(plan, 0)


# --- Fitting --------------------------------------------------------------


def test_fit_task_returns_failure_as_value():
    task = plan_restarts(10, 1)[0]
    result = fit_task(task, make_slices(), 2, "PARAFAC2", max_iter=1)
    assert not result.ok
    assert result.model is None and result.error is None
    assert result.failure_message
    failure = result.failure()
    assert isinstance(failure, ConvergenceError)
    assert failure.reason == result.failure_reason


def test_fit_task_rejects_unknown_option():
    task = plan_restarts(10, 1)[0]
    with pytest.raises(TypeError):
        fit_task(task, make_slices(), 2, "PARAFAC2", not_an_option=1)


def test_fit_task_applies_transform_to_subset():
    seen = []

    def transform(subset):
        seen.append(len(subset))
        return [s / s.std() for s in subset]

    task = FitTask("half", (0, 2, 4), 0)
    fit_task(task, make_slices(), 2, "PARAFAC2", transform=transform, **FIT_OPTIONS)
    assert seen == [3]


def test_parafac2_restarts_match_run_repeated():
    slices = make_slices()
    model, error = run_PARAFAC2_decomposition_repeated(
        slices,
        2,
        init_repeats=3,
        **FIT_OPTIONS,
    )
    plan = plan_restarts(len(slices), 3)
    store = InMemoryStore()
    run_tasks(plan, slices, 2, "PARAFAC2", store, **FIT_OPTIONS)
    best = collect(plan, store)["full"].best
    assert best is not None and isinstance(best.model, PARAFAC2Model)
    assert best.error == pytest.approx(float(error), abs=0)
    assert torch.equal(best.model.subject_mode, model.subject_mode)


def test_cp_restarts_match_run_repeated():
    tensor = make_tensor()
    weights, factors, error = run_CP_decomposition_repeated(
        tensor,
        2,
        init_repeats=3,
        **FIT_OPTIONS,
    )
    plan = plan_restarts(len(tensor), 3)
    store = InMemoryStore()
    run_tasks(plan, tensor, 2, "CP", store, **FIT_OPTIONS)
    best = collect(plan, store)["full"].best
    assert best is not None
    best_weights, best_factors = best.model
    assert torch.equal(best_weights, weights)
    assert all(torch.equal(a, b) for a, b in zip(best_factors, factors))


def test_run_tasks_skips_stored_tasks():
    plan = plan_restarts(10, 2)
    store = InMemoryStore()
    sentinel = RestartResult(plan[0], fake_model(1.0), 0.5, fake_diagnostics())
    store.save(sentinel)
    run_tasks(plan, make_slices(), 2, "PARAFAC2", store, **FIT_OPTIONS)
    assert store.load(plan[0]) is sentinel
    assert store.load(plan[1]) is not None


# --- Gathering ------------------------------------------------------------


def test_collect_picks_lowest_error_and_reports_missing():
    plan = plan_restarts(3, 4)
    store = InMemoryStore()
    store.save(RestartResult(plan[0], fake_model(1.0), 0.3, fake_diagnostics()))
    store.save(RestartResult(plan[1], fake_model(2.0), 0.1, fake_diagnostics()))
    store.save(
        RestartResult(
            plan[2],
            failure_reason="reconstruction",
            failure_message="not converged",
            failure_level=1e-3,
            failure_suggested_max_iter=900,
        ),
    )
    summary = collect(plan, store)["full"]
    assert summary.best is not None and summary.best.task.seed == 1
    assert summary.missing == [3]
    assert summary.indices == (0, 1, 2)
    tally = summary.tally
    assert (tally.attempted, tally.succeeded, tally.failed) == (3, 2, 1)
    assert tally.reasons["reconstruction"] == 1
    assert tally.median_level("reconstruction") == 1e-3
    assert tally.suggested_max_iter() == 900


def test_collect_ties_go_to_lowest_seed():
    plan = plan_restarts(3, 2)
    store = InMemoryStore()
    for task in reversed(plan):
        store.save(RestartResult(task, fake_model(1.0), 0.2, fake_diagnostics()))
    best = collect(plan, store)["full"].best
    assert best is not None and best.task.seed == 0


def test_collect_warns_when_a_group_never_converged():
    plan = plan_restarts(3, 1)
    store = InMemoryStore()
    store.save(RestartResult(plan[0], failure_reason="degenerate", failure_message="x"))
    with pytest.warns(UserWarning, match="none of 1 restarts converged"):
        summary = collect(plan, store)["full"]
    assert summary.best is None


# --- DirectoryStore -------------------------------------------------------


def test_directory_store_round_trip(tmp_path):
    store = DirectoryStore(tmp_path)
    ok_task = FitTask((2, 1), (0, 3, 5), 0)
    failed_task = FitTask((2, 1), (0, 3, 5), 1)
    cp_task = FitTask("full", (0, 1), 0)
    ok = RestartResult(ok_task, fake_model(3.0), 0.25, fake_diagnostics())
    failed = RestartResult(
        failed_task,
        failure_reason="feasibility",
        failure_message="gap",
        failure_level=0.5,
    )
    cp = RestartResult(cp_task, (torch.ones(2), [torch.ones(2, 2)] * 3), 0.1)
    for result in (ok, failed, cp):
        store.save(result)

    assert (tmp_path / "2_1" / "seed_0000.pt").exists()
    loaded = store.load(ok_task)
    assert loaded is not None and isinstance(loaded.model, PARAFAC2Model)
    assert loaded.error == 0.25 and loaded.diagnostics == ok.diagnostics
    assert torch.equal(loaded.model.weights, ok.model.weights)
    assert len(loaded.model.evolving_states) == 3
    assert store.load(failed_task) == failed
    loaded_cp = store.load(cp_task)
    assert loaded_cp is not None and loaded_cp.error == 0.1
    assert store.load(FitTask((2, 1), (0, 3, 5), 2)) is None
    assert not list(tmp_path.rglob("*.tmp"))


def test_directory_store_rejects_another_plans_result(tmp_path):
    store = DirectoryStore(tmp_path)
    store.save(RestartResult(FitTask("full", (0, 1), 0), fake_model(1.0), 0.1))
    with pytest.raises(ValueError, match="different plan"):
        store.load(FitTask("full", (0, 2), 0))


def test_directory_store_rejects_unnameable_group(tmp_path):
    store = DirectoryStore(tmp_path)
    with pytest.raises(ValueError):
        store.path(FitTask("a/b", (0,), 0))
    with pytest.raises(TypeError):
        store.path(FitTask(1.5, (0,), 0))


# --- Replicability --------------------------------------------------------


def test_compute_fms_is_nan_for_a_failed_half():
    engine = HalfHalfEngine(1, seed=0)
    plan = plan_replicability(engine, 6, 1)
    store = InMemoryStore()
    store.save(RestartResult(plan[0], fake_model(1.0), 0.1, fake_diagnostics()))
    store.save(RestartResult(plan[1], failure_reason="degenerate", failure_message=""))
    with pytest.warns(UserWarning):
        summaries = collect(plan, store)
    [(split, score)] = engine.compute_fms(summaries)
    assert split == 0 and math.isnan(score)


@pytest.mark.parametrize(
    "method, data",
    [("PARAFAC2", make_slices(12)), ("CP", make_tensor(12))],
)
@pytest.mark.parametrize(
    "make_engine",
    [lambda: HalfHalfEngine(2, seed=3), lambda: CrossValidationEngine(3, 1, seed=3)],
)
def test_distributed_matches_centralised(tmp_path, method, data, make_engine):
    # The core contract: running the plan job by job through files, then
    # gathering, gives bit-identical scores to the centralised call.
    labels = ["a", "b"] * 6
    central = evaluate_replicability_multiproc(
        make_engine(),
        data,
        2,
        method=method,
        stratification=labels,
        init_repeats=2,
        **FIT_OPTIONS,
    )

    engine = make_engine()
    plan = plan_replicability(engine, len(data), 2, labels)
    store = DirectoryStore(tmp_path)
    for job in range(n_jobs(plan, 3)):
        run_tasks(job_slice(plan, job, 3), data, 2, method, store, **FIT_OPTIONS)
    distributed = engine.compute_fms(collect(plan, store))

    assert len(central) == len(distributed)
    for (*keys_a, score_a), (*keys_b, score_b) in zip(central, distributed):
        for key_a, key_b in zip(keys_a, keys_b):
            np.testing.assert_array_equal(key_a, key_b)
        assert math.isfinite(float(score_a))
        assert float(score_a) == float(score_b)
