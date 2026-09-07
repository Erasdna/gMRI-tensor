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
from tensorly.parafac2_tensor import apply_parafac2_projections

plt.style.use(["science", "no-latex"])
matplotlib.use("Agg")


def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def reconstruct_evolving_factors(
    weights: torch.Tensor | np.ndarray,
    factors: list[torch.Tensor] | list[np.ndarray],
    projections: list[torch.Tensor] | list[np.ndarray],
) -> list[np.ndarray]:
    """Reconstruct each subject's own PARAFAC2 evolving-mode (time) factor.

    PARAFAC2's shared evolving-mode basis `factors[1]` (`B`, shape
    `(rank, rank)`) isn't directly interpretable per subject; each subject's
    own time pattern is `projections[i] @ B`. Thin wrapper around
    `tensorly.parafac2_tensor.apply_parafac2_projections`, converting the
    result to plain numpy arrays, since the rest of this plotting module
    operates on numpy (not torch) throughout.

    Parameters
    ----------
    weights, factors, projections
        As returned by `gMRItensor.run_PARAFAC2_decomposition_repeated`
        (`factors = [A, B, C]`; `projections[i]` has shape
        `(n_timepoints_i, rank)`).

    Returns
    -------
    list[np.ndarray]
        One `(n_timepoints_i, rank)` array per subject.
    """
    _, (_, evolving_factors, _) = apply_parafac2_projections(
        (weights, factors, projections),
    )
    return [_to_numpy(factor) for factor in evolving_factors]


def _build_long_evolving_dataframe(
    scaled_factors: list[np.ndarray],
    timepoints_per_subject: list[np.ndarray],
    subjects: list[str],
    groups: list[str],
    n_components: int,
) -> pd.DataFrame:
    """Stack per-subject evolving-mode factors into one long-form frame.

    One row per `(subject, timepoint, component)`. Timepoints are kept as
    their raw values (not binned or interpolated), so a later
    `groupby("timepoint")` is exactly the "exact-value match" alignment
    used to aggregate ragged per-subject timepoints across a group.

    Parameters
    ----------
    scaled_factors : list[np.ndarray]
        Per-subject, per-subject-scaled evolving-mode factors.
    timepoints_per_subject : list[np.ndarray]
        Per-subject time point arrays matching `scaled_factors[i]`'s rows.
    subjects, groups : list[str]
        Subject identifiers and their group labels, same order as
        `scaled_factors`.
    n_components : int
        Number of components (columns) in each subject's factor.

    Returns
    -------
    pd.DataFrame
        Columns `subject`, `group`, `timepoint`, `component`, `value`.
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
    """Aggregate a long-form evolving-mode frame into per-group ribbon stats.

    Parameters
    ----------
    long_df : pd.DataFrame
        As returned by `_build_long_evolving_dataframe`.

    Returns
    -------
    pd.DataFrame
        Columns `component`, `group`, `timepoint`, `mean`, `sem`, `n`.
        `sem` (standard error of the mean) is `NaN` from pandas at `n == 1`;
        it is filled to `0.0` here so a ribbon band doesn't gap at
        single-subject timepoints.
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

    A timepoint is only tested for a component if every group in
    `categories` has at least `min_group_n` subjects with data at that
    exact timepoint value -- the smallest `n` at which the rank-based tests
    below are non-degenerate. With exactly two groups, a two-sided
    Mann-Whitney U test is used (matching the test already used for
    categorical group comparisons in
    `gMRItensor.plotting.subject_mode.make_subject_boxplot`); with more
    than two groups, its multi-group generalization, Kruskal-Wallis, is
    used instead. With fewer than two groups no comparison is possible and
    nothing is tested.

    P-values are corrected for multiple comparisons with Benjamini-Hochberg
    FDR, scoped *per component* -- i.e. all timepoints tested for a given
    component form one hypothesis-testing family, not the whole figure.

    Parameters
    ----------
    long_df : pd.DataFrame
        As returned by `_build_long_evolving_dataframe`.
    categories : list[str]
        Group labels to compare, e.g. `sorted(set(groups))`.
    min_group_n : int, optional
        Minimum number of subjects each group must have at a timepoint for
        it to be tested. By default 2.

    Returns
    -------
    pd.DataFrame
        Columns `component`, `timepoint`, `p_value`, `p_adj`. Empty (but
        correctly columned) if fewer than two groups are given or no
        timepoint meets `min_group_n` for every group.
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
    significance: pd.DataFrame,
    categories: list[str],
    color_by_group: dict[str, str],
    significance_alpha: float,
) -> tuple[float, float]:
    """Draw a mean +/- SEM ribbon per group, with significance markers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    ribbon_stats : pd.DataFrame
        `_compute_group_ribbon_stats` output, pre-filtered to one
        component.
    significance : pd.DataFrame
        `_test_group_differences_over_time` output, pre-filtered to the
        same component. May be empty.
    categories : list[str]
        Group labels, in the order they should be plotted/legended.
    color_by_group : dict[str, str]
        Color for each group.
    significance_alpha : float
        Threshold on `p_adj` below which a timepoint is marked significant.

    Returns
    -------
    tuple[float, float]
        `(ymin, ymax)` actually drawn (ribbon bands and significance
        markers), for row-wise y-limit finalization.
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

    if not significance.empty:
        significant_timepoints = significance.loc[
            significance["p_adj"] < significance_alpha,
            "timepoint",
        ]
        if len(significant_timepoints) and np.isfinite(ymax):
            ax.plot(
                significant_timepoints,
                np.full(len(significant_timepoints), ymax),
                linestyle="none",
                marker="*",
                markersize=8,
                color="black",
                label="_nolegend_",
            )

    return ymin, ymax


def _plot_group_subject_column(
    ax: matplotlib.axes.Axes,
    factors: list[np.ndarray],
    timepoints_per_subject: list[np.ndarray],
    component: int,
    color: str,
) -> tuple[float, float]:
    """Draw one group's individual subject curves for one component.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    factors : list[np.ndarray]
        Scaled evolving-mode factors, restricted to this group's subjects.
    timepoints_per_subject : list[np.ndarray]
        Matching per-subject time point arrays.
    component : int
        Component index (column of each factor) to plot.
    color : str
        Line color for this group.

    Returns
    -------
    tuple[float, float]
        `(ymin, ymax)` actually drawn, for row-wise y-limit finalization.
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

    Direct analogue of `subject_mode._finalize_boxplot_axes`'s two-pass
    pattern: rendering happens first so the actual data range is known,
    then every axes in a row is given the same y-limits so the ribbon
    column and per-group columns stay visually comparable. Extra headroom
    is added on top so significance star markers don't collide with data.

    Parameters
    ----------
    axs : np.ndarray
        2D axes array, shape `(n_components, n_columns)`.
    row_ylims : list[tuple[float, float]]
        One `(ymin, ymax)` per row (component), as returned by
        `_plot_ribbon_column`/`_plot_group_subject_column`.
    """
    for component, (ymin, ymax) in enumerate(row_ylims):
        if not (np.isfinite(ymin) and np.isfinite(ymax)):
            continue
        span = ymax - ymin if ymax > ymin else 1.0
        for ax in axs[component, :]:
            ax.set_ylim(ymin - 0.05 * span, ymax + 0.2 * span)


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
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot each subject's own PARAFAC2 evolving-mode (time) pattern.

    One row per component, `1 + n_groups` columns: the first column is a
    ribbon plot (per-group mean +/- SEM band, colored by `group_variable`,
    with star markers above timepoints where the groups significantly
    differ), followed by one column per group showing that group's
    individual subjects' own reconstructed time curves.

    Parameters
    ----------
    evolving_factors : list[np.ndarray]
        Per-subject evolving-mode factors -- `evolving_factors[i]` has shape
        `(n_timepoints_i, rank)`, e.g. from `reconstruct_evolving_factors`.
    timepoints_per_subject : list[np.ndarray]
        Per-subject time point arrays matching `evolving_factors[i]`'s rows.
    subjects : list[str]
        Subject identifiers, in the same order as `evolving_factors`.
    subject_info : pd.DataFrame
        DataFrame containing subject metadata, with a `subjects` column and
        a `group_variable` column.
    group_variable : str
        Column name in `subject_info` to color-code subjects by.
    page_width : float, optional
        Target page width in inches. By default 7.0.
    width_to_height_ratio : float, optional
        Desired width-to-height ratio for each subplot. By default 1.618
        (golden ratio).
    min_group_n : int, optional
        Minimum number of subjects a group must have at a timepoint for a
        group-difference test to be run there. By default 2, the smallest
        `n` at which the underlying rank-based test is non-degenerate.
    significance_alpha : float, optional
        FDR-corrected p-value threshold below which a timepoint is marked
        significant on the ribbon plot. By default 0.05.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and 2D axes array with shape `(n_components, 1 + n_groups)`
        -- note this is a wider grid than this function used to return
        (previously `(n_components, 1)`): column 0 is the ribbon plot,
        columns 1..n_groups are one per group (in `sorted(set(groups))`
        order).

    Raises
    ------
    ValueError
        If input validation fails.

    Notes
    -----
    Unlike `scale_mode`'s usual whole-matrix usage elsewhere in this
    package, each subject's evolving-mode slice is scaled independently
    here: there is no single shared axis-0 to normalize across subjects at
    once, since subjects can have different numbers of time points.

    Ribbon aggregation aligns subjects by *exact* timepoint value (no
    interpolation or binning): a group's mean/SEM at a given timepoint is
    computed from whichever of its subjects have data at that exact value.
    A group's SEM is filled to `0.0` (instead of pandas' `NaN`) at
    timepoints with only one subject, so the ribbon band doesn't gap.

    Group-difference testing (see `_test_group_differences_over_time`) uses
    a two-sided Mann-Whitney U test for two groups, or Kruskal-Wallis for
    more than two, run independently at each timepoint that has at least
    `min_group_n` subjects per group, with Benjamini-Hochberg FDR
    correction applied per component. Significant timepoints are marked
    with a plain star above the ribbon rather than `statannotations`
    brackets: `statannotations.Annotator` annotates pairwise comparisons
    between fixed categorical x-positions (as used for boxplots in
    `gMRItensor.plotting.subject_mode.make_subject_boxplot`), which doesn't
    fit a comparison repeated at many points along a continuous time axis.
    With a single group, no comparison is possible and no tests are run.
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
        significance_component = significance.loc[
            significance["component"] == component,
        ]
        row_min, row_max = _plot_ribbon_column(
            axs[component, 0],
            ribbon_stats_component,
            significance_component,
            categories,
            color_by_group,
            significance_alpha,
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
                ax.set_xlabel("Time")

    _finalize_evolving_mode_axes(axs, row_ylims)
    fig.align_titles()

    return fig, axs
