"""Group comparisons of long-format values over time (no plotting).

Shared by the decomposition's evolving mode (faceted by component) and the
ROI-level tracer analysis (faceted by ROI).
"""
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def resolve_subject_groups(
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
) -> list[str]:
    """Look up each subject's `group_variable` value, in `subjects` order.

    Raises `ValueError` if a required column or any subject is missing.
    """
    if "subjects" not in subject_info.columns:
        raise ValueError("subject_info must contain a 'subjects' column")
    if group_variable not in subject_info.columns:
        raise ValueError(f"'{group_variable}' column missing from subject_info")

    subject_to_group = subject_info.set_index("subjects")[group_variable]
    missing = [s for s in subjects if s not in subject_to_group.index]
    if missing:
        raise ValueError(f"Subject(s) not found in subject_info: {missing}")
    return [subject_to_group.loc[s] for s in subjects]


def summarize_groups_over_time(
    long_df: pd.DataFrame,
    facet: str,
    value: str = "value",
) -> pd.DataFrame:
    """Per-group mean +/- SEM of `value` at each time point, per `facet`.

    `long_df` needs `facet`, `group`, `timepoint` and `value` columns, one
    row per subject observation. Returns columns `facet`, `group`,
    `timepoint`, `mean`, `sem`, `n`. `sem` is filled from NaN to 0.0 at
    `n == 1`, so a ribbon does not gap at single-subject time points.
    """
    stats = (
        long_df.groupby([facet, "group", "timepoint"])[value]
        .agg(mean="mean", sem="sem", n="count")
        .reset_index()
    )
    stats["sem"] = stats["sem"].fillna(0.0)
    return stats


def compare_groups_over_time(
    long_df: pd.DataFrame,
    categories: list[str],
    facet: str,
    value: str = "value",
    min_group_n: int = 2,
) -> pd.DataFrame:
    """Test for a group difference at each time point, per `facet`.

    A time point is tested only if every group in `categories` has at least
    `min_group_n` observations at that exact time point value -- the
    smallest `n` at which these rank-based tests are non-degenerate. Two
    groups use a two-sided Mann-Whitney U (as
    `subject_mode.make_subject_boxplot` does); more use Kruskal-Wallis;
    fewer are not tested at all.

    P-values are Benjamini-Hochberg FDR corrected *per facet value*, so each
    facet's time points form one hypothesis family rather than the whole
    figure.

    Returns columns `facet`, `timepoint`, `p_value`, `p_adj`, empty if
    nothing was testable.
    """
    columns = [facet, "timepoint", "p_value", "p_adj"]
    if len(categories) < 2:
        return pd.DataFrame(columns=columns)

    records = []
    for facet_value, facet_df in long_df.groupby(facet):
        facet_records = []
        for timepoint, timepoint_df in facet_df.groupby("timepoint"):
            group_values = [
                timepoint_df.loc[timepoint_df["group"] == category, value].to_numpy()
                for category in categories
            ]
            if any(len(values) < min_group_n for values in group_values):
                continue
            if len(categories) == 2:
                _, p_value = mannwhitneyu(
                    group_values[0],
                    group_values[1],
                    alternative="two-sided",
                )
            else:
                _, p_value = kruskal(*group_values)
            facet_records.append(
                {facet: facet_value, "timepoint": timepoint, "p_value": p_value},
            )

        if facet_records:
            _, p_adj, _, _ = multipletests(
                [record["p_value"] for record in facet_records],
                method="fdr_bh",
            )
            for record, adjusted in zip(facet_records, p_adj):
                record["p_adj"] = adjusted
            records.extend(facet_records)

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records)[columns]


def load_roi_statistics(
    path: Path | str,
    subject_info: pd.DataFrame,
    group_variable: str,
    rois: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a `write_preprocessed_data` ROI statistics file for group analysis.

    Returns the long frame (one row per subject, time point and ROI) with
    `time_point` renamed to `timepoint` and each subject's `group_variable`
    value attached as `group`, optionally restricted to `rois`. Raises
    `ValueError` for a requested ROI missing from the file or a subject
    missing from `subject_info`.
    """
    stats = pd.read_parquet(path)
    if rois is not None:
        missing = sorted(set(rois) - set(stats["roi"]))
        if missing:
            raise ValueError(f"ROI(s) not in {path}: {missing}")
        stats = stats[stats["roi"].isin(rois)]

    subjects = sorted(stats["subject"].unique())
    groups = resolve_subject_groups(subjects, subject_info, group_variable)
    stats = stats.rename(columns={"time_point": "timepoint"})
    stats["group"] = stats["subject"].map(dict(zip(subjects, groups)))
    return stats.reset_index(drop=True)


def _observed(stats_df: pd.DataFrame, statistic: str) -> pd.DataFrame:
    """Rows of `stats_df` with a value for `statistic` (e.g. no NaN median)."""
    if statistic not in stats_df.columns:
        raise ValueError(f"Statistic '{statistic}' is not a column of the frame")
    return stats_df.dropna(subset=[statistic])


def summarize_roi_statistics(stats_df: pd.DataFrame, statistic: str) -> pd.DataFrame:
    """Per-ROI, per-group mean +/- SEM of `statistic` at each time point.

    `stats_df` is `load_roi_statistics` output. Missing values are dropped
    first, so `n` counts the subjects that contribute. Returns columns
    `roi`, `group`, `timepoint`, `mean`, `sem`, `n` (ribbon plot input).
    """
    return summarize_groups_over_time(
        _observed(stats_df, statistic),
        facet="roi",
        value=statistic,
    )


def compare_roi_groups(
    stats_df: pd.DataFrame,
    statistic: str,
    min_group_n: int = 2,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test for a group difference in `statistic` at each time point, per ROI.

    `stats_df` is `load_roi_statistics` output; every group in it is
    compared, after dropping missing values. Tests and the per-ROI BH-FDR
    family are as in `compare_groups_over_time`. Returns columns `roi`,
    `timepoint`, `p_value`, `p_adj`, `significant` (`p_adj < alpha`).
    """
    observed = _observed(stats_df, statistic)
    significance = compare_groups_over_time(
        observed,
        sorted(stats_df["group"].unique()),
        facet="roi",
        value=statistic,
        min_group_n=min_group_n,
    )
    significance["significant"] = significance["p_adj"] < alpha
    return significance
