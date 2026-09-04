import numpy as np
import pandas as pd
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


def test_prepare_tensor_regular_pads_missing_timepoints_with_nan():
    # Regression test: prepare_tensor used to assume every subject has a row
    # for every time point and crash its rectangular reshape otherwise. It
    # should now NaN-pad the missing (subject, time_point) combination
    # instead, without dropping any label just because one subject has less
    # follow-up than the rest.
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    tensor, subjects, time_points, labels = prepare_tensor(df)

    assert list(subjects) == ["s1", "s2", "s3"]
    assert list(time_points) == [0, 1, 2]
    assert list(labels) == [10, 20, 30]
    assert tensor.shape == (3, 3, 3)
    # s3's time point 2 was never observed -> NaN, not dropped or crashed.
    assert np.isnan(tensor[2, 2]).all()
    assert not np.isnan(tensor[2, 0]).any()
    assert not np.isnan(tensor[0]).any()


def test_prepare_tensor_ragged_matches_regular_values():
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    tensor, subjects, time_points, labels = prepare_tensor(df, require_regular=True)
    slices, subjects2, timepoints_per_subject, labels2 = prepare_tensor(
        df,
        require_regular=False,
    )

    assert list(subjects) == list(subjects2)
    assert list(labels) == list(labels2)
    assert [s.shape for s in slices] == [(3, 3), (3, 3), (2, 3)]
    assert list(timepoints_per_subject[2]) == [0, 1]
    # The ragged slices contain exactly the non-NaN part of the regular array.
    np.testing.assert_allclose(slices[2], tensor[2, :2])


def test_prepare_tensor_drops_label_missing_anywhere():
    # A single missing value for one subject/time point still drops that
    # label everywhere -- the region/label mode must stay regular regardless
    # of require_regular, since a genuinely missing value can't be recovered.
    df = make_long_df(
        {"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1, 2]},
        missing={("s3", 1, 20)},
    )

    tensor, _, _, regular_labels = prepare_tensor(df, require_regular=True)
    slices, _, _, ragged_labels = prepare_tensor(df, require_regular=False)

    assert list(regular_labels) == [10, 30]
    assert list(ragged_labels) == [10, 30]
    assert tensor.shape[2] == 2
    assert all(s.shape[1] == 2 for s in slices)


def test_prepare_tensor_min_timepoints_filtering():
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    slices, subjects, _, _ = prepare_tensor(
        df,
        require_regular=False,
        min_timepoints=3,
    )

    assert list(subjects) == ["s1", "s2"]
    assert len(slices) == 2
