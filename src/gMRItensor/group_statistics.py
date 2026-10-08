"""Group comparisons of long-format values over time (no plotting).

Shared by the decomposition's evolving mode (faceted by component) and the
ROI-level tracer analysis (faceted by ROI).
"""
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
