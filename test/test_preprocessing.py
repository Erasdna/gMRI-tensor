import numpy as np
import pandas as pd
import pytest
from gMRItensor.preprocessing import prepare_parafac2_slices
from gMRItensor.preprocessing import prepare_tensor


def make_long_df(timepoints_per_subject, labels=(10, 20, 30), missing=None, seed=0):
    """Build a synthetic long-format tracer DataFrame.

    `timepoints_per_subject` maps subject -> list of observed time points.
    `missing` is an optional set of (subject, time_point, label) triples whose
    value should be NaN, simulating one region not being observed.
    """
    rng = np.random.default_rng(seed)
    missing = missing or set()
    rows = []
    for subject, time_points in timepoints_per_subject.items():
        for t in time_points:
            for label in labels:
                value = rng.random()
                if (subject, t, label) in missing:
                    value = np.nan
                rows.append(
                    {
                        "subject": subject,
                        "time_point": t,
                        "labels": label,
                        "values": value,
                    },
                )
    return pd.DataFrame(rows)


def test_prepare_tensor_crashes_on_ragged_timepoints():
    # Regression/documentation test: prepare_tensor assumes every subject has
    # a row for every time point -- one subject missing a whole time point
    # (not just a NaN value, but a structurally absent row) breaks its
    # rectangular reshape. This is exactly the case prepare_parafac2_slices
    # is designed to handle instead (see test below).
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})
    with pytest.raises(ValueError):
        prepare_tensor(df)


def test_prepare_parafac2_slices_handles_ragged_timepoints():
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    slices, subjects, timepoints_per_subject, labels = prepare_parafac2_slices(df)

    assert list(subjects) == ["s1", "s2", "s3"]
    assert list(labels) == [10, 20, 30]
    assert [s.shape for s in slices] == [(3, 3), (3, 3), (2, 3)]
    assert list(timepoints_per_subject[0]) == [0, 1, 2]
    assert list(timepoints_per_subject[2]) == [0, 1]


def test_prepare_parafac2_slices_drops_label_missing_anywhere():
    # A single missing value for one subject/time point still drops that
    # label everywhere -- the region/label mode must stay regular for
    # PARAFAC2 just as it does for prepare_tensor.
    df = make_long_df(
        {"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1, 2]},
        missing={("s3", 1, 20)},
    )

    tensor, _, _, tensor_labels = prepare_tensor(df)
    slices, _, _, parafac2_labels = prepare_parafac2_slices(df)

    assert list(tensor_labels) == [10, 30]
    assert list(parafac2_labels) == [10, 30]
    assert all(s.shape[1] == 2 for s in slices)


def test_prepare_parafac2_slices_min_timepoints_filtering():
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    slices, subjects, _, _ = prepare_parafac2_slices(df, min_timepoints=3)

    assert list(subjects) == ["s1", "s2"]
    assert len(slices) == 2
