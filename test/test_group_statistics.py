import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from gMRItensor.group_statistics import Aggregate
from gMRItensor.group_statistics import aggregate_subjects
from gMRItensor.group_statistics import compare_groups_over_time
from gMRItensor.group_statistics import flag_roi_values
from gMRItensor.group_statistics import resolve_subject_groups
from gMRItensor.group_statistics import significance_summary
from gMRItensor.group_statistics import subjects_outside_sd
from gMRItensor.group_statistics import subjects_with_groups
from gMRItensor.group_statistics import summarize_cohort_over_time
from gMRItensor.group_statistics import summarize_groups_over_time
from gMRItensor.plotting.evolving_mode import plot_evolving_mode
from gMRItensor.plotting.utils import scale_mode
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def make_shifted_long_df(
    n_per_group: int = 8,
    timepoints: tuple[int, ...] = (0, 1, 2, 3),
    shifted: dict[str, tuple[int, ...]] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Long frame with columns `roi, subject, group, timepoint, median`.

    Group "B" is shifted by +10 at `shifted[roi]` time points, so a
    difference is detectable there and nowhere else.
    """
    shifted = shifted if shifted is not None else {"ventricles": (2, 3), "wm": ()}
    rng = np.random.default_rng(seed)
    rows = []
    for roi, shifted_timepoints in shifted.items():
        for i in range(2 * n_per_group):
            group = "A" if i < n_per_group else "B"
            for t in timepoints:
                value = rng.normal()
                if group == "B" and t in shifted_timepoints:
                    value += 10.0
                rows.append(
                    {
                        "roi": roi,
                        "subject": f"s{i}",
                        "group": group,
                        "timepoint": t,
                        "median": value,
                    },
                )
    return pd.DataFrame(rows)


def test_summarize_groups_over_time_uses_facet_and_value_columns():
    long_df = pd.DataFrame(
        {
            "roi": ["r", "r", "r"],
            "group": ["A", "A", "B"],
            "timepoint": [0, 0, 0],
            "median": [1.0, 3.0, 10.0],
        },
    )

    summary = summarize_groups_over_time(long_df, facet="roi", value="median")

    assert list(summary.columns) == ["roi", "group", "timepoint", "mean", "sem", "n"]
    row_a = summary[summary["group"] == "A"].iloc[0]
    assert row_a["mean"] == pytest.approx(2.0)
    assert row_a["n"] == 2
    # Single-subject sem is filled to 0, not NaN, so ribbons do not gap.
    assert summary[summary["group"] == "B"].iloc[0]["sem"] == 0.0


def test_group_differences_are_significant_only_where_shifted():
    long_df = make_shifted_long_df()

    significance = compare_groups_over_time(
        long_df,
        ["A", "B"],
        facet="roi",
        value="median",
    )

    assert list(significance.columns) == ["roi", "timepoint", "p_value", "p_adj"]
    ventricles = significance[significance["roi"] == "ventricles"]
    assert (ventricles.loc[ventricles["timepoint"] >= 2, "p_adj"] < 0.05).all()
    assert not (ventricles.loc[ventricles["timepoint"] < 2, "p_adj"] < 0.05).any()
    assert not (significance.loc[significance["roi"] == "wm", "p_adj"] < 0.05).any()


def test_group_differences_fdr_family_is_per_facet():
    # Each facet's time points are one hypothesis family: the adjusted
    # p-values for "wm" must not depend on how many tests "ventricles" had.
    long_df = make_shifted_long_df()
    wm_only = long_df[long_df["roi"] == "wm"]

    both = compare_groups_over_time(long_df, ["A", "B"], "roi", "median")
    alone = compare_groups_over_time(wm_only, ["A", "B"], "roi", "median")

    np.testing.assert_allclose(
        both.loc[both["roi"] == "wm", "p_adj"].to_numpy(),
        alone["p_adj"].to_numpy(),
    )


def test_group_differences_skip_timepoints_below_min_group_n():
    long_df = make_shifted_long_df(n_per_group=1)

    significance = compare_groups_over_time(
        long_df,
        ["A", "B"],
        facet="roi",
        value="median",
    )

    assert significance.empty
    assert list(significance.columns) == ["roi", "timepoint", "p_value", "p_adj"]


def test_resolve_subject_groups_rejects_missing_subject():
    subject_info = pd.DataFrame({"subjects": ["s0"], "group": ["A"]})

    assert resolve_subject_groups(["s0"], subject_info, "group") == ["A"]
    with pytest.raises(ValueError, match="not found"):
        resolve_subject_groups(["s0", "s1"], subject_info, "group")
    with pytest.raises(ValueError, match="missing"):
        resolve_subject_groups(["s0"], subject_info, "diagnosis")


def test_subjects_with_groups_leaves_out_blank_and_unknown_subjects():
    subject_info = pd.DataFrame(
        {
            "subjects": ["s0", "s1", "s2", "s3", "s4"],
            "group": ["A", "A", None, "B", "B"],
        },
    )

    positions, dropped = subjects_with_groups(
        ["s0", "s1", "s2", "s3", "s4", "s5"],
        subject_info,
        "group",
    )

    assert positions == [0, 1, 3, 4]
    assert dropped == ["s2", "s5"]  # blank group; no subject-info row


def test_subjects_with_groups_needs_two_per_group():
    subject_info = pd.DataFrame(
        {"subjects": ["s0", "s1", "s2", "s3"], "group": ["A", None, "B", "B"]},
    )

    with pytest.raises(ValueError, match="'A'.*1 subject"):
        subjects_with_groups(["s0", "s1", "s2", "s3"], subject_info, "group")
    with pytest.raises(ValueError, match="at least 2 groups"):
        subjects_with_groups(["s2", "s3"], subject_info, "group")
    with pytest.raises(ValueError, match="missing"):
        subjects_with_groups(["s0"], subject_info, "diagnosis")


def test_plot_evolving_mode_significance_matches_direct_scipy():
    # Regression test for moving the statistics out of evolving_mode: the
    # returned significance must equal Mann-Whitney + per-component BH
    # computed directly here, independent of the moved helpers.
    rng = np.random.default_rng(1)
    n_per_group, timepoints = 6, np.arange(4)
    subjects = [f"s{i}" for i in range(2 * n_per_group)]
    groups = ["A"] * n_per_group + ["B"] * n_per_group
    subject_info = pd.DataFrame({"subjects": subjects, "group": groups})
    factors = [
        rng.normal(size=(len(timepoints), 2)) + (3.0 if g == "B" else 0.0)
        for g in groups
    ]

    fig, _, significance = plot_evolving_mode(
        factors,
        [timepoints] * len(subjects),
        subjects,
        subject_info,
        group_variable="group",
    )
    plt.close(fig)

    scaled = [scale_mode(factor) for factor in factors]
    for component in range(2):
        expected_p = [
            mannwhitneyu(
                [scaled[i][t, component] for i in range(n_per_group)],
                [scaled[i][t, component] for i in range(n_per_group, len(subjects))],
                alternative="two-sided",
            ).pvalue
            for t in timepoints
        ]
        expected_adj = multipletests(expected_p, method="fdr_bh")[1]
        got = significance[significance["component"] == component].sort_values(
            "timepoint",
        )
        np.testing.assert_allclose(got["p_value"], expected_p)
        np.testing.assert_allclose(got["p_adj"], expected_adj)
        np.testing.assert_array_equal(got["significant"], expected_adj < 0.05)


def test_significance_summary_names_significant_rois_and_higher_group():
    long_df = make_shifted_long_df()
    tables = {}
    for statistic in ("median",):
        summary = summarize_groups_over_time(long_df, facet="roi", value=statistic)
        significance = compare_groups_over_time(long_df, ["A", "B"], "roi", statistic)
        significance["significant"] = significance["p_adj"] < 0.05
        tables[statistic] = (summary, significance)

    result = significance_summary(tables)

    assert list(result.columns) == [
        "statistic",
        "roi",
        "n_tested",
        "significant_timepoints",
        "min_p_adj",
        "higher_group",
    ]
    rows = result.set_index("roi")
    assert rows.loc["ventricles", "significant_timepoints"] == "2, 3"
    assert rows.loc["ventricles", "higher_group"] == "B"
    assert rows.loc["ventricles", "n_tested"] == 4
    assert rows.loc["wm", "significant_timepoints"] == ""


def test_aggregate_subjects_trimmed_drops_the_extremes_and_nans() -> None:
    column = [100.0, 1.0, 2.0, 3.0, np.nan, 4.0, 5.0, -50.0]
    values = np.array([column, [1.0] * 8]).T  # (8 subjects, 2 columns)

    aggregate, n = aggregate_subjects(values, "trimmed", trim=2)

    # 7 finite values; dropping two at each end leaves 2, 3, 4.
    np.testing.assert_allclose(aggregate, [3.0, 1.0])
    np.testing.assert_array_equal(n, [7, 8])


def test_aggregate_subjects_trimmed_falls_back_to_the_median() -> None:
    values = np.array([[1.0], [2.0], [10.0], [np.nan]])

    aggregate, n = aggregate_subjects(values, "trimmed", trim=2)

    np.testing.assert_allclose(aggregate, [2.0])
    np.testing.assert_array_equal(n, [3])


def test_aggregate_subjects_trim_zero_is_the_mean() -> None:
    values = np.array([[1.0, 4.0], [2.0, np.nan], [6.0, 8.0]])

    trimmed, _ = aggregate_subjects(values, "trimmed", trim=0)
    mean, _ = aggregate_subjects(values, "mean")

    np.testing.assert_allclose(trimmed, [3.0, 6.0])
    np.testing.assert_allclose(mean, [3.0, 6.0])


@pytest.mark.parametrize("method", ["trimmed", "mean", "median"])
def test_aggregate_subjects_matches_numpy_and_leaves_empty_columns_nan(
    method: Aggregate,
) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(9, 4))
    values[rng.random(values.shape) < 0.2] = np.nan
    values[:, -1] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aggregate, n = aggregate_subjects(values, method, trim=2)

    for column in range(3):
        finite = np.sort(values[np.isfinite(values[:, column]), column])
        expected = {
            "trimmed": finite[2:-2].mean() if len(finite) >= 5 else np.median(finite),
            "mean": finite.mean(),
            "median": np.median(finite),
        }[method]
        assert aggregate[column] == pytest.approx(expected)
    assert np.isnan(aggregate[-1]) and n[-1] == 0


def test_aggregate_subjects_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="method"):
        aggregate_subjects(np.ones((3, 1)), "mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trim"):
        aggregate_subjects(np.ones((3, 1)), "trimmed", trim=-1)


def test_summarize_cohort_over_time_pools_groups():
    long_df = pd.DataFrame(
        {
            "roi": ["r"] * 5,
            "group": ["A", "A", "B", "B", "A"],
            "timepoint": [0, 0, 0, 0, 6],
            "median": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )

    stats = summarize_cohort_over_time(long_df, facet="roi", value="median")

    assert list(stats.columns) == ["roi", "timepoint", "mean", "std", "n"]
    first = stats[stats["timepoint"] == 0].iloc[0]
    assert first["mean"] == pytest.approx(2.5)
    assert first["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert first["n"] == 4
    single = stats[stats["timepoint"] == 6].iloc[0]
    assert single["std"] == 0.0 and single["n"] == 1


def test_subjects_outside_sd_names_each_at_its_most_extreme_point():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        for timepoint in (0, 6, 24):
            value = rng.normal()
            if i == 7:
                value += {0: 0.0, 6: 8.0, 24: 6.0}[timepoint]
            rows.append(
                {
                    "roi": "r",
                    "subject": f"s{i}",
                    "group": "A" if i < 15 else "B",
                    "timepoint": timepoint,
                    "median": value,
                },
            )
    long_df = pd.DataFrame(rows)
    cohort = summarize_cohort_over_time(long_df, facet="roi", value="median")

    outside = subjects_outside_sd(long_df, cohort, facet="roi", value="median")

    assert list(outside.columns) == [
        "roi",
        "subject",
        "group",
        "timepoint",
        "median",
        "z",
    ]
    assert outside[["subject", "group", "timepoint"]].values.tolist() == [
        ["s7", "A", 6],
    ]
    assert outside["z"].iloc[0] > 3


def test_subjects_outside_sd_finds_nothing_in_a_small_cohort():
    """With n <= 10 no point can lie beyond 3 sample SDs."""
    long_df = pd.DataFrame(
        {
            "roi": "r",
            "subject": [f"s{i}" for i in range(10)],
            "group": "A",
            "timepoint": 0,
            "median": [0.0] * 9 + [100.0],
        },
    )
    cohort = summarize_cohort_over_time(long_df, facet="roi", value="median")

    assert subjects_outside_sd(long_df, cohort, facet="roi", value="median").empty


def test_a_time_point_where_every_value_ties_is_not_tested() -> None:
    """Regression: one all-tied time point (a t=0 scan used as its own
    baseline) gave p = NaN, and BH then made every p_adj of that ROI NaN."""
    rows = []
    for i in range(5):
        for group, shift in (("A", 0.0), ("B", 3.0)):
            rows.append({"roi": "r", "timepoint": 0, "group": group, "value": 0.0})
            rows.append(
                {"roi": "r", "timepoint": 6, "group": group, "value": shift + i},
            )

    table = compare_groups_over_time(
        pd.DataFrame(rows),
        ["A", "B"],
        facet="roi",
        value="value",
    )

    assert list(table["timepoint"]) == [6]
    assert np.isfinite(table["p_adj"].astype(float)).all()


def _label_values(n: int = 12) -> pd.DataFrame:
    """`n` subjects x labels 1, 2 x time points 4, 24 of `median`, spread
    evenly over [1, 1 + 0.01 (n - 1)] per cell: |robust z| < 1.5 for all."""
    rows = [
        {
            "subject": f"s{i:02d}",
            "group": "AB"[i % 2],
            "timepoint": t,
            "label": label,
            "median": 1.0 + 0.01 * ((5 * i + label + t) % n),
        }
        for i in range(n)
        for t in (4, 24)
        for label in (1, 2)
    ]
    return pd.DataFrame(rows)


def _at(frame: pd.DataFrame, subject: str, t: int, label: int) -> pd.Series:
    return (
        (frame["subject"] == subject)
        & (frame["timepoint"] == t)
        & (frame["label"] == label)
    )


def test_flag_roi_values_flags_planted_values_per_label_and_session() -> None:
    frame = _label_values()
    frame.loc[_at(frame, "s03", 24, 2), "median"] = 5.0  # extreme in one ROI
    frame.loc[_at(frame, "s07", 4, 1), "median"] = -1.0  # failed scan

    flags = flag_roi_values(frame, "median")

    found = set(zip(flags["subject"], flags["timepoint"], flags["label"]))
    assert found == {("s03", 24, 2), ("s07", 4, 1)}
    assert list(flags.columns) == [*frame.columns, "robust_z"]
    s03 = flags[flags["subject"] == "s03"].iloc[0]
    others = frame[(frame["timepoint"] == 24) & (frame["label"] == 2)]["median"]
    mad = (others - others.median()).abs().median()
    assert s03["robust_z"] == pytest.approx(
        (5.0 - others.median()) / (1.4826 * mad),
    )
    assert (flags.loc[flags["subject"] == "s07", "robust_z"] < 0).all()


def test_flag_roi_values_threshold_is_a_parameter() -> None:
    frame = _label_values()
    frame.loc[_at(frame, "s03", 24, 2), "median"] = 5.0

    assert len(flag_roi_values(frame, "median", z_cut=1e6)) == 0
    assert len(flag_roi_values(frame, "median", z_cut=0.5)) > 1


def test_flag_roi_values_skips_small_samples_tied_values_and_nan() -> None:
    few = _label_values(n=4)
    few.loc[_at(few, "s01", 4, 1), "median"] = 100.0
    tied = _label_values()
    tied["median"] = 0.0
    tied.loc[_at(tied, "s01", 4, 1), "median"] = 1.0  # MAD = 0
    missing = _label_values()
    missing.loc[_at(missing, "s01", 4, 1), "median"] = np.nan

    assert flag_roi_values(few, "median").empty
    assert flag_roi_values(tied, "median").empty
    assert flag_roi_values(missing, "median").empty
    assert len(flag_roi_values(few, "median", min_n=4)) == 1
