import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from gMRItensor.group_statistics import compare_groups_over_time
from gMRItensor.group_statistics import compare_roi_groups
from gMRItensor.group_statistics import load_roi_statistics
from gMRItensor.group_statistics import resolve_subject_groups
from gMRItensor.group_statistics import summarize_groups_over_time
from gMRItensor.group_statistics import summarize_roi_statistics
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


def _write_roi_statistics(long_df: pd.DataFrame, path) -> tuple:
    """`make_shifted_long_df` as a `write_preprocessed_data` ROI statistics
    file, plus the matching `subject_info`."""
    stats = long_df.rename(columns={"timepoint": "time_point"})
    subject_info = stats[["subject", "group"]].drop_duplicates()
    subject_info = subject_info.rename(columns={"subject": "subjects"})
    stats = stats.drop(columns="group").assign(roi_type="group")
    stats["mean"] = stats["median"] + 1.0
    stats.to_parquet(path)
    return path, subject_info.rename(columns={"group": "diagnosis"})


def test_load_roi_statistics_attaches_groups_and_filters_rois(tmp_path):
    long_df = make_shifted_long_df()
    path, subject_info = _write_roi_statistics(long_df, tmp_path / "stats.parquet")

    stats = load_roi_statistics(path, subject_info, "diagnosis", rois=["wm"])

    assert set(stats["roi"]) == {"wm"}
    assert "timepoint" in stats.columns and "time_point" not in stats.columns
    expected = long_df[long_df["roi"] == "wm"].set_index(["subject", "timepoint"])
    got = stats.set_index(["subject", "timepoint"]).loc[expected.index]
    np.testing.assert_array_equal(got["group"], expected["group"])
    assert len(load_roi_statistics(path, subject_info, "diagnosis")) == len(long_df)


def test_load_roi_statistics_rejects_unknown_roi_and_subject(tmp_path):
    path, subject_info = _write_roi_statistics(
        make_shifted_long_df(),
        tmp_path / "stats.parquet",
    )

    with pytest.raises(ValueError, match="not_a_roi"):
        load_roi_statistics(path, subject_info, "diagnosis", rois=["not_a_roi"])
    with pytest.raises(ValueError, match="not found"):
        load_roi_statistics(path, subject_info.iloc[1:], "diagnosis")


def test_summarize_roi_statistics_ignores_missing_values():
    long_df = make_shifted_long_df()
    long_df.loc[0, "median"] = np.nan

    summary = summarize_roi_statistics(long_df, "median")

    expected = summarize_groups_over_time(
        long_df.dropna(subset=["median"]),
        facet="roi",
        value="median",
    )
    pd.testing.assert_frame_equal(summary, expected)
    first = long_df.loc[0]
    row = summary[
        (summary["roi"] == first["roi"])
        & (summary["group"] == first["group"])
        & (summary["timepoint"] == first["timepoint"])
    ].iloc[0]
    assert row["n"] == 7  # 8 subjects per group, one missing


def test_compare_roi_groups_flags_shifted_timepoints():
    long_df = make_shifted_long_df()
    long_df.loc[0, "median"] = np.nan  # must not turn a p-value into NaN

    significance = compare_roi_groups(long_df, "median", alpha=0.05)

    assert list(significance.columns) == [
        "roi",
        "timepoint",
        "p_value",
        "p_adj",
        "significant",
    ]
    assert significance["p_value"].notna().all()
    flagged = set(
        significance.loc[significance["significant"], ["roi", "timepoint"]].apply(
            tuple,
            axis=1,
        ),
    )
    assert flagged == {("ventricles", 2), ("ventricles", 3)}
    assert not compare_roi_groups(long_df, "median", alpha=0.0)["significant"].any()


def test_compare_roi_groups_min_group_n_and_bad_statistic():
    long_df = make_shifted_long_df(n_per_group=3)

    significance = compare_roi_groups(long_df, "median", min_group_n=4)

    assert significance.empty
    assert "significant" in significance.columns
    with pytest.raises(ValueError, match="not_a_statistic"):
        compare_roi_groups(long_df, "not_a_statistic")
    with pytest.raises(ValueError, match="not_a_statistic"):
        summarize_roi_statistics(long_df, "not_a_statistic")
