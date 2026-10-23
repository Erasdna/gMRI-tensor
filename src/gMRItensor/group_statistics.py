"""Group comparisons of long-format values over time (no plotting).

Shared by the decomposition's evolving mode (faceted by component) and the
ROI-level tracer analysis (faceted by ROI); `aggregate_subjects` builds the
group-average maps.
"""
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def require_columns(
    available: Sequence[str],
    requested: Mapping[str, str],
    source: str,
) -> None:
    """Raise unless every `requested` column (name -> what asked for it, e.g.
    a flag) is in `available`; the message lists the available columns."""
    # str.format: add-trailing-comma breaks "(" inside an f-string.
    missing = [
        "'{}' ({})".format(column, role)
        for column, role in requested.items()
        if column not in available
    ]
    if missing:
        noun = "column" if len(missing) == 1 else "columns"
        raise ValueError(
            f"{source} has no {noun} {', '.join(missing)}; available columns: "
            + ", ".join(str(column) for column in available),
        )


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


def subjects_with_groups(
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
    min_per_group: int = 2,
) -> tuple[list[int], list[str]]:
    """Which `subjects` have a `group_variable` value in `subject_info`.

    Returns `(positions, dropped)`: the positions in `subjects` of those
    that do, and the ids of those that do not (no `subject_info` row, or a
    blank cell). Raises `ValueError` if a required column is missing, fewer
    than 2 groups remain, or a group keeps fewer than `min_per_group`
    subjects.
    """
    if "subjects" not in subject_info.columns:
        raise ValueError("subject_info must contain a 'subjects' column")
    if group_variable not in subject_info.columns:
        raise ValueError(f"'{group_variable}' column missing from subject_info")

    subject_to_group = subject_info.set_index("subjects")[group_variable]
    positions = [
        i
        for i, s in enumerate(subjects)
        if s in subject_to_group.index and pd.notna(subject_to_group.loc[s])
    ]
    kept = set(positions)
    dropped = [s for i, s in enumerate(subjects) if i not in kept]

    counts = pd.Series([subject_to_group.loc[subjects[i]] for i in positions])
    counts = counts.value_counts()
    if len(counts) < 2:
        raise ValueError(
            f"'{group_variable}' needs at least 2 groups with data, found "
            f"{len(counts)}",
        )
    small = counts[counts < min_per_group]
    if not small.empty:
        listed = ", ".join(f"{g!r} has {n} subject(s)" for g, n in small.items())
        raise ValueError(
            f"every '{group_variable}' group needs at least {min_per_group} "
            f"subjects with data: {listed}",
        )
    return positions, dropped


Aggregate = Literal["trimmed", "mean", "median"]
AGGREGATES: tuple[str, ...] = ("trimmed", "mean", "median")


def aggregate_subjects(
    values: np.ndarray,
    method: Aggregate = "trimmed",
    trim: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate `(n_subjects, n_columns)` values over subjects, per column.

    Non-finite values are missing and ignored. `trimmed` is the mean after
    dropping the `trim` highest and lowest finite values of a column; a
    column with fewer than `2 * trim + 1` finite values gets its median
    instead (the limit of trimming). A column without finite values is NaN.

    Returns `(aggregate, n)`, `n` the finite values per column.
    """
    if method not in AGGREGATES:
        raise ValueError(f"method must be one of {AGGREGATES}, got {method!r}")
    if trim < 0:
        raise ValueError(f"trim must be >= 0, got {trim}")
    values = np.where(np.isfinite(values), values, np.nan).astype(float)
    finite = np.isfinite(values)
    n = finite.sum(axis=0)
    has_values = n > 0
    aggregate = np.full(values.shape[1], np.nan)
    if not has_values.any():
        return aggregate, n
    present = values[:, has_values]
    n_present = n[has_values]
    if method == "mean":
        aggregate[has_values] = np.nansum(present, axis=0) / n_present
    elif method == "median":
        aggregate[has_values] = np.nanmedian(present, axis=0)
    else:
        ordered = np.sort(present, axis=0)  # NaN sorts last
        rank = np.arange(len(ordered))[:, None]
        kept = (rank >= trim) & (rank < n_present - trim)
        n_kept = n_present - 2 * trim
        trimmed = np.where(kept, ordered, 0.0).sum(axis=0) / np.maximum(n_kept, 1)
        aggregate[has_values] = np.where(
            n_kept >= 1,
            trimmed,
            np.nanmedian(present, axis=0),
        )
    return aggregate, n


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


def summarize_cohort_over_time(
    long_df: pd.DataFrame,
    facet: str,
    value: str = "value",
) -> pd.DataFrame:
    """Mean and SD of `value` over all subjects, regardless of group, at each
    time point, per `facet`.

    The reference against which a single subject's curve can be judged an
    outlier (`plot_roi_evolution_rows`'s SD bands). Returns columns
    `facet`, `timepoint`, `mean`, `std`, `n`; `std` is 0.0 at `n == 1`.
    """
    stats = (
        long_df.groupby([facet, "timepoint"])[value]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def subjects_outside_sd(
    long_df: pd.DataFrame,
    cohort: pd.DataFrame,
    facet: str,
    value: str = "value",
    n_sd: float = 3.0,
) -> pd.DataFrame:
    """Subjects with a point beyond the cohort mean +/- `n_sd` SD, per `facet`.

    `cohort` is `summarize_cohort_over_time` output for the same `long_df`
    (needs `subject`, `group`, `facet`, `timepoint`, `value`). One row per
    such subject and facet value, at its most extreme point: columns
    `facet`, `subject`, `group`, `timepoint`, `value`, `z`. With the sample
    SD no point can exceed `(n - 1) / sqrt(n)` SD, so with 10 or fewer
    subjects nothing is found at `n_sd = 3`.
    """
    columns = [facet, "subject", "group", "timepoint", value, "z"]
    # Renamed: `long_df` may hold a `mean` statistic of its own.
    reference = cohort[[facet, "timepoint", "mean", "std"]].rename(
        columns={"mean": "_cohort_mean", "std": "_cohort_std"},
    )
    merged = long_df.dropna(subset=[value]).merge(reference, on=[facet, "timepoint"])
    std = merged["_cohort_std"]
    merged["z"] = (merged[value] - merged["_cohort_mean"]) / std.where(std > 0)
    merged = merged.dropna(subset=["z"])
    if merged.empty:
        return pd.DataFrame(columns=columns)
    worst = merged.loc[
        merged["z"].abs().groupby([merged[facet], merged["subject"]]).idxmax()
    ]
    return worst.loc[worst["z"].abs() > n_sd, columns].reset_index(drop=True)


# 1.4826 * MAD estimates the SD of normally distributed data.
_MAD_TO_SD = 1.4826


def flag_roi_values(
    long_df: pd.DataFrame,
    value: str,
    z_cut: float = 3.5,
    min_n: int = 5,
    by: Sequence[str] = ("timepoint", "label"),
) -> pd.DataFrame:
    """Values far from the rest of the cohort, judged per `by` (ROI and
    session): the rows of `long_df` with `|robust_z| > z_cut`, plus a
    `robust_z` column.

    `robust_z = (x - median) / (1.4826 * MAD)` over all subjects with a
    finite `value` in that `by` cell, regardless of group, MAD being the
    median absolute deviation from the median: the modified z-score of
    Iglewicz and Hoaglin (1993), whose recommended cut-off is 3.5. Median
    and MAD keep the outliers sought from inflating the scale they are
    judged against. Cells with fewer than `min_n` values, or MAD = 0, are
    not judged. A screening heuristic with no multiple-comparison
    correction: a flag means "look at this scan", not "exclude".

    Iglewicz, B. and Hoaglin, D. C. (1993). How to Detect and Handle
    Outliers. ASQC Basic References in Quality Control: Statistical
    Techniques, vol. 16. Milwaukee, WI: ASQC Quality Press.
    """
    observed = long_df[np.isfinite(long_df[value].astype(float))]
    cells = observed.groupby(list(by))[value]
    median = cells.transform("median")
    deviation = (observed[value] - median).abs()
    scale = _MAD_TO_SD * deviation.groupby([observed[c] for c in by]).transform(
        "median",
    )
    judged = (cells.transform("count") >= min_n) & (scale > 0)
    z = (observed[value] - median) / scale.where(judged)
    flagged = z.abs() > z_cut
    return observed[flagged].assign(robust_z=z[flagged]).reset_index(drop=True)


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
    fewer are not tested at all, nor are time points where every value ties.

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
            if not np.isfinite(p_value):
                # Every value tied (e.g. a t=0 scan used as its own
                # baseline): untestable, and a NaN would make BH return
                # NaN for every other time point of this facet.
                continue
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


def significance_summary(
    tables: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """One row per statistic and ROI: where the groups differ over time.

    `tables` maps each statistic to its `(summary, significance)` pair:
    `summarize_groups_over_time` / `compare_groups_over_time` output faceted
    by `roi`, the latter with a `significant` column. Returns `statistic`,
    `roi`, `n_tested` (time points tested), `significant_timepoints`
    (comma-separated, empty if none), `min_p_adj` and `higher_group` (the
    group with the highest mean at the time point of `min_p_adj`).
    """
    columns = [
        "statistic",
        "roi",
        "n_tested",
        "significant_timepoints",
        "min_p_adj",
        "higher_group",
    ]
    rows = []
    for statistic, (summary, significance) in tables.items():
        for roi in summary["roi"].drop_duplicates():
            tested = significance[significance["roi"] == roi]
            row: dict[str, Any] = {
                "statistic": statistic,
                "roi": roi,
                "n_tested": len(tested),
                "significant_timepoints": ", ".join(
                    str(t)
                    for t in sorted(tested.loc[tested["significant"], "timepoint"])
                ),
                "min_p_adj": np.nan,
                "higher_group": "",
            }
            if not tested.empty:
                best = tested.loc[tested["p_adj"].astype(float).idxmin()]
                row["min_p_adj"] = float(best["p_adj"])
                at_best = summary[
                    (summary["roi"] == roi)
                    & (summary["timepoint"] == best["timepoint"])
                ]
                row["higher_group"] = str(
                    at_best.loc[at_best["mean"].idxmax(), "group"],
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)
