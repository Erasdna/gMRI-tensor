"""Plotting for PARAFAC2's evolving (subject-specific time) mode."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import torch
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import scale_mode
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

plt.style.use(["science", "no-latex"])
matplotlib.use("Agg")


def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def evolving_factors_to_numpy(
    evolving_states: list[torch.Tensor] | list[np.ndarray],
) -> list[np.ndarray]:
    """Convert a model's `evolving_states` to numpy for plotting.

    `run_PARAFAC2_decomposition_repeated` now returns each subject's time
    course directly, so no reconstruction is needed -- this only bridges
    torch to the numpy the rest of this module works in.
    """
    return [_to_numpy(factor) for factor in evolving_states]


def _build_long_evolving_dataframe(
    scaled_factors: list[np.ndarray],
    timepoints_per_subject: list[np.ndarray],
    subjects: list[str],
    groups: list[str],
    n_components: int,
) -> pd.DataFrame:
    """Stack per-subject evolving-mode factors into one long-form frame.

    One row per `(subject, timepoint, component)`, with columns `subject`,
    `group`, `timepoint`, `component`, `value`. Timepoints keep their raw
    values rather than being binned, so a later `groupby("timepoint")`
    aligns ragged per-subject timepoints by exact value.
    """
    component_columns = [str(c) for c in range(n_components)]
    frames = []
    for factor, timepoints, subject, group in zip(
        scaled_factors,
        timepoints_per_subject,
        subjects,
        groups,
    ):
        frame = pd.DataFrame(factor[:, :n_components], columns=component_columns)
        frame["subject"] = subject
        frame["group"] = group
        frame["timepoint"] = np.asarray(timepoints)
        frames.append(frame)

    long_df = pd.concat(frames, ignore_index=True).melt(
        id_vars=["subject", "group", "timepoint"],
        value_vars=component_columns,
        var_name="component",
        value_name="value",
    )
    long_df["component"] = long_df["component"].astype(int)
    return long_df


def _compute_group_ribbon_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate `_build_long_evolving_dataframe` output into ribbon stats.

    Returns columns `component`, `group`, `timepoint`, `mean`, `sem`, `n`.
    `sem` is filled from NaN to 0.0 at `n == 1`, so the ribbon does not gap
    at single-subject timepoints.
    """
    stats = (
        long_df.groupby(["component", "group", "timepoint"])["value"]
        .agg(mean="mean", sem="sem", n="count")
        .reset_index()
    )
    stats["sem"] = stats["sem"].fillna(0.0)
    return stats


def _test_group_differences_over_time(
    long_df: pd.DataFrame,
    categories: list[str],
    min_group_n: int = 2,
) -> pd.DataFrame:
    """Test for a group difference at each timepoint, per component.

    A timepoint is tested only if every group in `categories` has at least
    `min_group_n` subjects at that exact timepoint value -- the smallest `n`
    at which these rank-based tests are non-degenerate. Two groups use a
    two-sided Mann-Whitney U (as `subject_mode.make_subject_boxplot` does);
    more use Kruskal-Wallis; fewer are not tested at all.

    P-values are Benjamini-Hochberg FDR corrected *per component*, so each
    component's timepoints form one hypothesis family rather than the whole
    figure.

    Returns columns `component`, `timepoint`, `p_value`, `p_adj`, empty if
    nothing was testable.
    """
    columns = ["component", "timepoint", "p_value", "p_adj"]
    if len(categories) < 2:
        return pd.DataFrame(columns=columns)

    records = []
    for component, component_df in long_df.groupby("component"):
        component_records = []
        for timepoint, timepoint_df in component_df.groupby("timepoint"):
            group_values = [
                timepoint_df.loc[
                    timepoint_df["group"] == category,
                    "value",
                ].to_numpy()
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
            component_records.append(
                {"component": component, "timepoint": timepoint, "p_value": p_value},
            )

        if component_records:
            _, p_adj, _, _ = multipletests(
                [record["p_value"] for record in component_records],
                method="fdr_bh",
            )
            for record, adjusted in zip(component_records, p_adj):
                record["p_adj"] = adjusted
            records.extend(component_records)

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records)[columns]


def _plot_ribbon_column(
    ax: matplotlib.axes.Axes,
    ribbon_stats: pd.DataFrame,
    categories: list[str],
    color_by_group: dict[str, str],
) -> tuple[float, float]:
    """Draw a mean +/- SEM ribbon per group, for one component.

    `ribbon_stats` is `_compute_group_ribbon_stats` output pre-filtered to
    that component. Returns the `(ymin, ymax)` actually drawn, for row-wise
    y-limit finalization.
    """
    ymin, ymax = np.inf, -np.inf
    for category in categories:
        group_stats = ribbon_stats.loc[
            ribbon_stats["group"] == category,
        ].sort_values("timepoint")
        if group_stats.empty:
            continue
        timepoints = group_stats["timepoint"].to_numpy()
        mean = group_stats["mean"].to_numpy()
        sem = group_stats["sem"].to_numpy()
        color = color_by_group[category]
        ax.plot(timepoints, mean, color=color, marker="o", markersize=3, label=category)
        ax.fill_between(timepoints, mean - sem, mean + sem, color=color, alpha=0.25)
        ymin = min(ymin, np.min(mean - sem))
        ymax = max(ymax, np.max(mean + sem))

    return ymin, ymax


def _plot_group_subject_column(
    ax: matplotlib.axes.Axes,
    factors: list[np.ndarray],
    timepoints_per_subject: list[np.ndarray],
    component: int,
    color: str,
) -> tuple[float, float]:
    """Draw one group's individual subject curves for one component.

    `factors` are the scaled evolving-mode factors for this group's subjects
    only. Returns the `(ymin, ymax)` actually drawn, for row-wise y-limit
    finalization.
    """
    ymin, ymax = np.inf, -np.inf
    for factor, timepoints in zip(factors, timepoints_per_subject):
        values = factor[:, component]
        ax.plot(timepoints, values, color=color, alpha=0.6, marker="o", markersize=3)
        ymin = min(ymin, np.min(values))
        ymax = max(ymax, np.max(values))
    return ymin, ymax


def _finalize_evolving_mode_axes(
    axs: np.ndarray,
    row_ylims: list[tuple[float, float]],
) -> None:
    """Apply a shared, padded y-limit across each row's axes.

    Same two-pass pattern as `subject_mode._finalize_boxplot_axes`: draw
    first so the data range is known, then equalize each row so the ribbon
    and per-group columns stay visually comparable. `row_ylims` holds one
    `(ymin, ymax)` per component.
    """
    for component, (ymin, ymax) in enumerate(row_ylims):
        if not (np.isfinite(ymin) and np.isfinite(ymax)):
            continue
        span = ymax - ymin if ymax > ymin else 1.0
        for ax in axs[component, :]:
            ax.set_ylim(ymin - 0.05 * span, ymax + 0.05 * span)


def plot_evolving_mode(
    evolving_factors: list[np.ndarray],
    timepoints_per_subject: list[np.ndarray],
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
    page_width: float = 7.0,
    width_to_height_ratio: float = 1.618,
    min_group_n: int = 2,
    significance_alpha: float = 0.05,
) -> tuple[matplotlib.figure.Figure, np.ndarray, pd.DataFrame]:
    """Plot each subject's own PARAFAC2 evolving-mode (time) pattern.

    One row per component, `1 + n_groups` columns: the first column is a
    ribbon plot (per-group mean +/- SEM band, colored by `group_variable`),
    followed by one column per group showing that group's individual
    subjects' own reconstructed time curves. Per-timepoint group-difference
    significance is computed but not drawn -- it's returned as a
    `pd.DataFrame` instead, see `Returns`.

    Parameters
    ----------
    evolving_factors : list[np.ndarray]
        Per-subject `(n_timepoints_i, rank)` factors, e.g. from
        `reconstruct_evolving_factors`.
    timepoints_per_subject : list[np.ndarray]
        Time point arrays matching `evolving_factors[i]`'s rows.
    subjects : list[str]
        Subject identifiers, in `evolving_factors` order.
    subject_info : pd.DataFrame
        Subject metadata, with a `subjects` column and a `group_variable`
        column.
    group_variable : str
        Column in `subject_info` to color-code by.
    page_width, width_to_height_ratio : float, optional
        Figure sizing, see `compute_figsize`.
    min_group_n : int, optional
        Minimum subjects per group at a timepoint for it to be tested.
    significance_alpha : float, optional
        FDR-corrected threshold for the `significant` column.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray, pd.DataFrame]
        `(fig, axs, significance)`. `axs` is `(n_components, 1 + n_groups)`:
        column 0 is the ribbon, the rest one per group in
        `sorted(set(groups))` order. `significance` has columns
        `component`, `timepoint`, `p_value`, `p_adj`, `significant`, and is
        empty if nothing was testable.

    Raises
    ------
    ValueError
        If input validation fails.

    Notes
    -----
    Each subject's slice is scaled independently, unlike `scale_mode`'s
    usual whole-matrix use: subjects can have different numbers of time
    points, so there is no shared axis to normalize across.

    Ribbon aggregation aligns subjects by *exact* timepoint value, with no
    interpolation or binning.

    Significance is returned rather than drawn: a marker repeated at every
    tested point along a continuous axis reads poorly, and
    `statannotations.Annotator` (used in `subject_mode`) only does bracket
    annotations between categorical x-positions.
    """
    if not (len(evolving_factors) == len(timepoints_per_subject) == len(subjects)):
        raise ValueError(
            "evolving_factors, timepoints_per_subject and subjects must have the "
            f"same length, got {len(evolving_factors)}, "
            f"{len(timepoints_per_subject)}, {len(subjects)}",
        )
    if "subjects" not in subject_info.columns:
        raise ValueError("subject_info must contain a 'subjects' column")
    if group_variable not in subject_info.columns:
        raise ValueError(f"'{group_variable}' column missing from subject_info")

    subject_to_group = subject_info.set_index("subjects")[group_variable]
    missing = [s for s in subjects if s not in subject_to_group.index]
    if missing:
        raise ValueError(f"Subject(s) not found in subject_info: {missing}")
    groups = [subject_to_group.loc[s] for s in subjects]

    categories = sorted(set(groups))
    colors = get_color_palette(len(categories))
    color_by_group = dict(zip(categories, colors))

    n_components = evolving_factors[0].shape[1]
    scaled_factors = [scale_mode(factor) for factor in evolving_factors]

    long_df = _build_long_evolving_dataframe(
        scaled_factors,
        timepoints_per_subject,
        subjects,
        groups,
        n_components,
    )
    ribbon_stats = _compute_group_ribbon_stats(long_df)
    significance = _test_group_differences_over_time(
        long_df,
        categories,
        min_group_n=min_group_n,
    )
    significance["significant"] = significance["p_adj"] < significance_alpha

    n_columns = 1 + len(categories)
    figsize = compute_figsize(
        n_components,
        n_columns,
        page_width=page_width,
        width_to_height_ratio=width_to_height_ratio,
    )
    fig, axs = plt.subplots(
        n_components,
        n_columns,
        figsize=figsize,
        layout="compressed",
        squeeze=False,
    )

    group_indices = {
        category: [i for i, group in enumerate(groups) if group == category]
        for category in categories
    }

    row_ylims = []
    for component in range(n_components):
        ribbon_stats_component = ribbon_stats.loc[
            ribbon_stats["component"] == component,
        ]
        row_min, row_max = _plot_ribbon_column(
            axs[component, 0],
            ribbon_stats_component,
            categories,
            color_by_group,
        )

        for col, category in enumerate(categories, start=1):
            indices = group_indices[category]
            g_min, g_max = _plot_group_subject_column(
                axs[component, col],
                [scaled_factors[i] for i in indices],
                [timepoints_per_subject[i] for i in indices],
                component,
                color_by_group[category],
            )
            row_min, row_max = min(row_min, g_min), max(row_max, g_max)
        row_ylims.append((row_min, row_max))

        row_timepoints = sorted(
            long_df.loc[long_df["component"] == component, "timepoint"].unique(),
        )
        for ax in axs[component, :]:
            ax.set_xticks(row_timepoints)

        axs[component, 0].set_ylabel(rf"Component {component+1}")
        if component == 0:
            axs[component, 0].set_title(r"Mean $\pm$ SEM")
            for col, category in enumerate(categories, start=1):
                axs[component, col].set_title(str(category))
            axs[component, 0].legend(frameon=True, framealpha=0.9)
        if component == n_components - 1:
            for ax in axs[component, :]:
                ax.set_xlabel("Time after injection")

    _finalize_evolving_mode_axes(axs, row_ylims)
    fig.align_titles()

    return fig, axs, significance
