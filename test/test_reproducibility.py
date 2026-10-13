import numpy as np
import pandas as pd
import pytest
import torch
from gMRItensor import CMFModel
from gMRItensor import FitTask
from gMRItensor import PARAFAC2Model
from gMRItensor import RestartResult
from gMRItensor.reproducibility import read_reproducibility
from gMRItensor.reproducibility import restart_fms_matrix
from gMRItensor.reproducibility import write_reproducibility


def cp_model(seed: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return torch.ones(2), [torch.rand(n, 2, generator=generator) for n in (5, 4, 3)]


def restart(
    seed: int,
    error: float | None,
    model: object | None = None,
) -> RestartResult:
    task = FitTask("full", tuple(range(5)), seed)
    if error is None:
        return RestartResult(task, failure_reason="max_iter", failure_message="no")
    return RestartResult(task, model=model, error=error)  # type: ignore[arg-type]


def test_fms_matrix_is_sorted_by_error_and_symmetric():
    results = [
        restart(0, 0.3, cp_model(0)),
        restart(1, 0.1, cp_model(1)),
        restart(2, 0.2, cp_model(2)),
    ]
    fms, errors = restart_fms_matrix(results)
    assert list(fms.index) == [1, 2, 0]
    assert list(fms.columns) == [1, 2, 0]
    assert list(errors) == [0.1, 0.2, 0.3]
    np.testing.assert_allclose(fms.to_numpy(), fms.to_numpy().T)
    np.testing.assert_allclose(np.diag(fms.to_numpy()), 1.0)
    assert ((fms.to_numpy() >= 0) & (fms.to_numpy() <= 1 + 1e-6)).all()


def test_fms_matrix_skips_failed_restarts_and_breaks_ties_by_seed():
    results = [
        restart(0, None),
        restart(1, 0.1, cp_model(1)),
        restart(2, 0.1, cp_model(2)),
    ]
    fms, errors = restart_fms_matrix(results)
    assert list(fms.index) == [1, 2]
    assert list(errors.index) == [1, 2]


def test_same_minimum_scores_one():
    weights, factors = cp_model(0)
    permuted = [factor[:, [1, 0]] * 2.0 for factor in factors]
    results = [
        restart(0, 0.1, (weights, factors)),
        restart(1, 0.1, (weights, permuted)),
    ]
    fms, _ = restart_fms_matrix(results)
    assert fms.loc[0, 1] == pytest.approx(1.0)


@pytest.mark.parametrize("model_type", [PARAFAC2Model, CMFModel])
def test_ragged_models_are_scored(model_type):
    generator = torch.Generator().manual_seed(0)

    def model(seed_offset: float) -> object:
        return model_type(
            weights=torch.ones(2),
            subject_mode=torch.rand(3, 2, generator=generator),
            evolving_states=[
                torch.rand(n, 2, generator=generator) + seed_offset for n in (4, 5, 4)
            ],
            label_mode=torch.rand(6, 2, generator=generator),
        )

    first = model(0.0)
    results = [
        restart(0, 0.1, first),
        restart(1, 0.2, model(0.0)),
        restart(2, 0.3, first),
    ]
    fms, _ = restart_fms_matrix(results)
    assert fms.shape == (3, 3)
    assert fms.loc[0, 2] == pytest.approx(1.0)
    assert fms.loc[0, 1] < 1.0


def test_csv_round_trip(tmp_path):
    results = [restart(seed, 0.1 * (3 - seed), cp_model(seed)) for seed in range(3)]
    fms, errors = restart_fms_matrix(results)
    path = write_reproducibility(results, tmp_path / "rank")
    assert path == tmp_path / "rank" / "reproducibility.csv"
    assert list(pd.read_csv(path).columns[:2]) == ["seed", "error"]

    read_fms, read_errors = read_reproducibility(path)
    pd.testing.assert_frame_equal(read_fms, fms, check_names=False)
    pd.testing.assert_series_equal(read_errors, errors, check_names=False)
