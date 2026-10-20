import dataclasses
import warnings
from typing import Any
from typing import Literal

import numpy as np
import pytest
from gMRItensor.model_io import SavedDecomposition
from gMRItensor.outliers import _robust_residual_cutoff
from gMRItensor.outliers import data_slices
from gMRItensor.outliers import leverage
from gMRItensor.outliers import outlier_table
from gMRItensor.outliers import reconstruct_slices
from gMRItensor.outliers import subject_residuals
from tlviz.outliers import compute_leverage

N_SUBJECTS, RANK, N_TIMEPOINTS, N_LABELS = 20, 2, 5, 12
Method = Literal["cp", "parafac2", "cmf"]
HIGH_LEVERAGE, BADLY_FITTED = 3, 11


def make_model(method: Method, rank: int = RANK, seed: int = 0) -> SavedDecomposition:
    """A model with one subject far out in the subject mode (`HIGH_LEVERAGE`);
    PARAFAC2/CMF slices have different lengths."""
    rng = np.random.default_rng(seed)
    subject_mode = rng.uniform(0.5, 1.5, (N_SUBJECTS, rank))
    subject_mode[HIGH_LEVERAGE] = [6.0] + [0.2] * (rank - 1)
    common: dict[str, Any] = {
        "rank": rank,
        "error": 0.1,
        "weights": rng.uniform(0.5, 2.0, rank),
        "subjects": np.array([f"sub-{i:02d}" for i in range(N_SUBJECTS)]),
        "labels": np.arange(N_LABELS),
        "label_index": np.zeros(N_LABELS, dtype=int),
        "scale_mean": None,
        "scale_std": None,
        "centered": False,
        "label_mode": rng.uniform(size=(N_LABELS, rank)),
    }
    if method == "cp":
        return SavedDecomposition(
            method="cp",
            subject_mode=subject_mode,
            time_mode=rng.uniform(size=(N_TIMEPOINTS, rank)),
            timepoints=np.arange(N_TIMEPOINTS),
            **common,
        )
    lengths = rng.integers(3, N_TIMEPOINTS + 1, N_SUBJECTS)
    states = [rng.uniform(size=(t, rank)) for t in lengths]
    if method == "cmf":
        subject_mode = np.stack([np.sqrt((B**2).mean(axis=0)) for B in states])
    return SavedDecomposition(
        method=method,
        subject_mode=subject_mode,
        evolving_states=states,
        timepoints=[np.arange(t) for t in lengths],
        **common,
    )


def noisy_data(saved: SavedDecomposition, seed: int = 1) -> list[np.ndarray]:
    """The model plus small noise, with `BADLY_FITTED` given 2x the noise SD
    (4x the residual)."""
    rng = np.random.default_rng(seed)
    slices = [x + rng.normal(0, 0.05, x.shape) for x in reconstruct_slices(saved)]
    slices[BADLY_FITTED] = reconstruct_slices(saved)[BADLY_FITTED] + rng.normal(
        0,
        0.1,
        slices[BADLY_FITTED].shape,
    )
    return slices


def test_reconstruct_cp_matches_the_full_tensor() -> None:
    saved = make_model("cp")
    assert saved.time_mode is not None
    expected = np.einsum(
        "r,ir,jr,kr->ijk",
        saved.weights,
        saved.subject_mode,
        saved.time_mode,
        saved.label_mode,
    )

    np.testing.assert_allclose(np.stack(reconstruct_slices(saved)), expected)


def test_reconstruct_parafac2_scales_each_slice_by_its_subject_weights() -> None:
    saved = make_model("parafac2")
    assert saved.evolving_states is not None
    i = 4
    B, a = saved.evolving_states[i], saved.subject_mode[i]
    expected = B @ np.diag(saved.weights * a) @ saved.label_mode.T

    slices = reconstruct_slices(saved)

    assert [s.shape[0] for s in slices] == [len(t) for t in saved.timepoints]
    np.testing.assert_allclose(slices[i], expected)


def test_reconstruct_cmf_ignores_the_derived_subject_mode() -> None:
    saved = make_model("cmf")
    assert saved.evolving_states is not None
    i = 4
    expected = saved.evolving_states[i] @ np.diag(saved.weights) @ saved.label_mode.T

    np.testing.assert_allclose(reconstruct_slices(saved)[i], expected)


def test_data_slices_splits_a_tensor_and_passes_slices_through() -> None:
    import torch

    tensor = torch.arange(24.0).reshape(2, 3, 4)
    slices = [torch.ones(2, 4), torch.zeros(3, 4)]

    assert [s.shape for s in data_slices(tensor)] == [(3, 4), (3, 4)]
    np.testing.assert_array_equal(data_slices(tensor)[1], tensor[1].numpy())
    assert [s.shape for s in data_slices(slices)] == [(2, 4), (3, 4)]


def test_residual_is_a_per_entry_mean_over_observed_entries() -> None:
    estimated = [np.zeros((2, 3)), np.zeros((4, 3)), np.zeros((4, 3))]
    observed = [np.ones((2, 3)), np.ones((4, 3)), np.ones((4, 3))]
    observed[2][1:3] = np.nan  # two missing sessions: still error 1 per entry

    residual = subject_residuals(observed, estimated)

    np.testing.assert_allclose(residual, [1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize("method", ["cp", "parafac2", "cmf"])
def test_planted_outliers_are_flagged(method: Method) -> None:
    saved = make_model(method)

    table, cutoffs = outlier_table(saved, noisy_data(saved))

    assert list(table["subject"]) == list(saved.subjects)
    np.testing.assert_allclose(table["leverage"], compute_leverage(saved.subject_mode))
    assert table["leverage"].sum() == pytest.approx(RANK)
    assert table["residual"].sum() == pytest.approx(1.0)
    assert set(table.index[table["high_residual_robust"]]) == {BADLY_FITTED}
    if method != "cmf":  # CMF's subject mode is derived, not planted
        assert set(table.index[table["high_leverage"]]) == {HIGH_LEVERAGE}
    assert cutoffs.leverage is not None and cutoffs.residual is not None


def test_the_tlviz_residual_cutoff_is_masked_by_a_strong_outlier() -> None:
    """Why the robust cut-off exists: one subject fitted much worse than the
    rest inflates the spread it is judged against."""
    rng = np.random.default_rng(0)
    residual = rng.chisquare(60, N_SUBJECTS) / 60
    residual[BADLY_FITTED] *= 8
    residual /= residual.sum()
    saved = make_model("cp")
    estimated = reconstruct_slices(saved)
    # Data whose per-subject mean squared residual is `residual`.
    observed = [x + np.sqrt(r) for x, r in zip(estimated, residual)]

    table, cutoffs = outlier_table(saved, observed)

    np.testing.assert_allclose(table["residual"], residual)
    assert not table["high_residual"].any()
    assert list(table.index[table["high_residual_robust"]]) == [BADLY_FITTED]
    assert cutoffs.robust_residual is not None
    assert cutoffs.residual is not None
    assert cutoffs.robust_residual < cutoffs.residual


def test_no_false_alarms_without_outliers() -> None:
    rng = np.random.default_rng(3)
    residual = rng.chisquare(60, 200) / 60

    cutoff = _robust_residual_cutoff(residual / residual.sum(), 0.01 / 200)

    assert cutoff is not None and (residual / residual.sum() < cutoff).all()


def test_rank_one_uses_the_hotelling_leverage_cutoff() -> None:
    saved = make_model("cp", rank=1)

    table, cutoffs = outlier_table(saved, noisy_data(saved))

    assert cutoffs.leverage is not None
    assert table.loc[HIGH_LEVERAGE, "high_leverage"]


def test_residual_cutoffs_are_none_without_spread() -> None:
    saved = make_model("cp")
    exact = reconstruct_slices(saved)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table, cutoffs = outlier_table(saved, exact)

    assert cutoffs.residual is None and cutoffs.robust_residual is None
    assert not table[["high_residual", "high_residual_robust"]].any().any()


def test_leverage_cutoff_is_none_with_too_few_subjects() -> None:
    saved = make_model("cp")
    two = dataclasses.replace(
        saved,
        subject_mode=saved.subject_mode[:2],
        subjects=saved.subjects[:2],
    )

    table, cutoffs = outlier_table(two, noisy_data(saved)[:2])

    assert cutoffs.leverage is None
    assert not table["high_leverage"].any()


def test_data_must_match_the_model_subjects() -> None:
    saved = make_model("cp")

    with pytest.raises(ValueError, match="19 data slices for a model of 20"):
        outlier_table(saved, reconstruct_slices(saved)[:-1])


def test_leverage_matches_tlviz_at_full_rank() -> None:
    saved = make_model("cp")

    np.testing.assert_allclose(
        leverage(saved.subject_mode),
        compute_leverage(saved.subject_mode),
    )


@pytest.mark.parametrize("deficiency", ["dead component", "more components"])
def test_rank_deficient_subject_mode_gets_leverage_but_no_cutoff(
    deficiency: str,
) -> None:
    """A component no subject expresses, or more components than subjects,
    makes `AᵀA` singular: leverage stays defined (`diag(A A⁺)`, summing to
    the numerical rank) and only its cut-off is dropped."""
    if deficiency == "dead component":
        saved = make_model("cp")
        subject_mode = saved.subject_mode.copy()
        subject_mode[:, 1] = 0.0
        expected_rank = 1
    else:
        saved = make_model("cp", rank=3)
        subject_mode = saved.subject_mode[:2]
        expected_rank = 2
    dead = dataclasses.replace(
        saved,
        subject_mode=subject_mode,
        subjects=saved.subjects[: len(subject_mode)],
    )

    table, cutoffs = outlier_table(dead, reconstruct_slices(dead))

    assert cutoffs.leverage is None
    assert not table["high_leverage"].any()
    assert table["leverage"].sum() == pytest.approx(expected_rank)


def test_residuals_refuse_slices_of_another_shape() -> None:
    with pytest.raises(ValueError, match="subject 1: data of shape"):
        subject_residuals(
            [np.zeros((3, 4)), np.zeros((4, 4))],
            [np.zeros((3, 4)), np.zeros((3, 4))],
        )
