"""Tracer evolution per ROI and subject group, drawn from precomputed tables.

Inputs are `group_statistics` outputs (`summarize_roi_statistics`,
`compare_roi_groups`, `load_roi_statistics`); nothing here computes
statistics.
"""
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from gMRItensor.plotting.utils import apply_row_ylims
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import plot_group_ribbons
from gMRItensor.plotting.utils import plot_subject_curves
from gMRItensor.plotting.utils import resolve_page_width

plt.style.use(["science", "no-latex"])
matplotlib.use("Agg")

TIME_LABEL = "Time after injection [h]"

_STATISTIC_LABELS = {
    "median": "Median signal (a.u.)",
    "mean": "Mean signal (a.u.)",
    "median_concentration": "Median concentration (mM)",
    "mean_concentration": "Mean concentration (mM)",
    "total_amount": "Total amount (mmol)",
}

# Gap [pt] between the top of the ribbon and a significance marker.
_STAR_OFFSET_PT = 1.0
# Extra figure height [in] for the figure legend and shared time label,
# which take fixed space regardless of the number of rows.
_FIGURE_LABEL_HEIGHT = 0.5


def statistic_label(statistic: str) -> str:
    """Y-axis label with unit for a `compute_roi_statistics` column."""
    return _STATISTIC_LABELS.get(statistic, statistic)


def _check_rois(summary: pd.DataFrame, rois: Sequence[str]) -> None:
    missing = [roi for roi in rois if roi not in set(summary["roi"])]
    if missing:
        raise ValueError(f"ROI(s) not in the summary table: {missing}")


def _group_colors(summary: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Groups in sorted order and their colors, as in `plot_evolving_mode`."""
    categories = sorted(summary["group"].unique())
    return categories, dict(zip(categories, get_color_palette(len(categories))))


def _significant_timepoints(significance: pd.DataFrame, roi: str) -> list[float]:
    rows = significance[(significance["roi"] == roi) & significance["significant"]]
    return sorted(rows["timepoint"])


def _figsize(
    n_rows: int,
    n_cols: int,
    page_width: str | float,
    width_to_height_ratio: float,
) -> tuple[float, float]:
    width, height = compute_figsize(
        n_rows,
        n_cols,
        page_width=resolve_page_width(page_width),
        width_to_height_ratio=width_to_height_ratio,
    )
    return width, height + _FIGURE_LABEL_HEIGHT


def _add_figure_labels(
    fig: matplotlib.figure.Figure,
    legend_ax: matplotlib.axes.Axes,
    n_groups: int,
    statistic: str,
) -> None:
    """One legend above all panels plus shared time and statistic labels.

    Figure-level labels instead of per-axes ones, so narrow (single column)
    figures do not collide.
    """
    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=n_groups,
        frameon=False,
    )
    fontsize = plt.rcParams["axes.labelsize"]
    fig.supxlabel(TIME_LABEL, fontsize=fontsize)
    fig.supylabel(statistic_label(statistic), fontsize=fontsize)


def _mark_significance(
    ribbon_ax: matplotlib.axes.Axes,
    row_axes: Sequence[matplotlib.axes.Axes],
    roi_summary: pd.DataFrame,
    timepoints: list[float],
) -> None:
    """Draw `*` just above the highest ribbon edge at each significant time.

    The star is offset in points and the y-limit of every axis in
    `row_axes` is raised until it fits inside the axes, so it clears both
    the markers and the titles at any figure size (call after the layout is
    drawn). A row keeps one shared y-range.
    """
    if not timepoints:
        return
    ribbon_top = (
        (roi_summary["mean"] + roi_summary["sem"])
        .groupby(roi_summary["timepoint"])
        .max()
    )
    star_top = max(ribbon_top[t] for t in timepoints)
    axes_height_pt = ribbon_ax.get_window_extent().height * 72 / ribbon_ax.figure.dpi
    star_fraction = min(
        (_STAR_OFFSET_PT + 1.2 * plt.rcParams["font.size"]) / axes_height_pt,
        0.5,
    )
    ymin, ymax = ribbon_ax.get_ylim()
    new_ymax = max(ymax, ymin + (star_top - ymin) / (1 - star_fraction))
    for ax in row_axes:
        ax.set_ylim(ymin, new_ymax)
    for timepoint in timepoints:
        ribbon_ax.annotate(
            "*",
            xy=(timepoint, ribbon_top[timepoint]),
            xytext=(0, _STAR_OFFSET_PT),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def plot_roi_evolution_rows(
    summary: pd.DataFrame,
    subject_values: pd.DataFrame,
    significance: pd.DataFrame,
    rois: Sequence[str],
    statistic: str,
    page_width: str | float = "double",
    width_to_height_ratio: float = 1.618,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """One row per ROI: group ribbons, then each group's subject curves.

    Same layout as `plot_evolving_mode`, with ROIs as rows. `summary` is
    `summarize_roi_statistics` output, `subject_values` the
    `load_roi_statistics` frame it came from and `significance`
    `compare_roi_groups` output for the same `statistic`. Significant time
    points get a `*` above the ribbon. `page_width` is a `JOURNAL_WIDTHS`
    name or inches.

    Returns `(fig, axs)`, `axs` of shape `(len(rois), 1 + n_groups)`.
    """
    _check_rois(summary, rois)
    categories, color_by_group = _group_colors(summary)
    n_columns = 1 + len(categories)
    fig, axs = plt.subplots(
        len(rois),
        n_columns,
        figsize=_figsize(len(rois), n_columns, page_width, width_to_height_ratio),
        layout="compressed",
        squeeze=False,
    )

    observed = subject_values.dropna(subset=[statistic])
    row_ylims = []
    for row, roi in enumerate(rois):
        roi_summary = summary[summary["roi"] == roi]
        row_min, row_max = plot_group_ribbons(
            axs[row, 0],
            roi_summary,
            categories,
            color_by_group,
        )
        roi_values = observed[observed["roi"] == roi]
        for col, category in enumerate(categories, start=1):
            group_values = roi_values[roi_values["group"] == category]
            curves = [
                (
                    subject_df["timepoint"].to_numpy(),
                    subject_df[statistic].to_numpy(),
                )
                for _, subject_df in group_values.sort_values("timepoint").groupby(
                    "subject",
                )
            ]
            g_min, g_max = plot_subject_curves(
                axs[row, col],
                curves,
                color_by_group[category],
            )
            row_min, row_max = min(row_min, g_min), max(row_max, g_max)
        row_ylims.append((row_min, row_max))

        for ax in axs[row, :]:
            ax.set_xticks(sorted(roi_summary["timepoint"].unique()))
        axs[row, 0].set_ylabel(roi)

    axs[0, 0].set_title(r"Mean $\pm$ SEM")
    for col, category in enumerate(categories, start=1):
        axs[0, col].set_title(str(category))
    _add_figure_labels(fig, axs[0, 0], len(categories), statistic)

    apply_row_ylims(axs, row_ylims)
    fig.draw_without_rendering()  # fix axes sizes for `_mark_significance`
    for row, roi in enumerate(rois):
        _mark_significance(
            axs[row, 0],
            axs[row, :],
            summary[summary["roi"] == roi],
            _significant_timepoints(significance, roi),
        )
    fig.align_titles()
    return fig, axs


def plot_roi_evolution_panels(
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    rois: Sequence[str],
    statistic: str,
    n_rows: int,
    n_cols: int,
    page_width: str | float = "double",
    sharey: bool = False,
    width_to_height_ratio: float = 1.618,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """A grid of group-ribbon panels, one per ROI, filled row by row.

    Inputs as in `plot_roi_evolution_rows`. One figure legend for all
    panels; axes beyond `len(rois)` are hidden. With `sharey` every panel
    gets the same y-range, otherwise each is scaled to its own data.

    Returns `(fig, axs)`, `axs` of shape `(n_rows, n_cols)`.
    """
    _check_rois(summary, rois)
    if len(rois) > n_rows * n_cols:
        raise ValueError(
            f"{len(rois)} ROIs do not fit a {n_rows} x {n_cols} grid",
        )
    categories, color_by_group = _group_colors(summary)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=_figsize(n_rows, n_cols, page_width, width_to_height_ratio),
        layout="compressed",
        squeeze=False,
    )
    n_panels = len(rois)
    panels = list(axs.flat[:n_panels])
    for ax in axs.flat[n_panels:]:
        ax.set_visible(False)

    ylims = []
    for ax, roi in zip(panels, rois):
        roi_summary = summary[summary["roi"] == roi]
        ylims.append(plot_group_ribbons(ax, roi_summary, categories, color_by_group))
        ax.set_xticks(sorted(roi_summary["timepoint"].unique()))
        ax.set_title(roi)
    if sharey:
        shared = (min(lo for lo, _ in ylims), max(hi for _, hi in ylims))
        ylims = [shared] * len(ylims)
    apply_row_ylims(np.array(panels).reshape(-1, 1), ylims)
    _add_figure_labels(fig, panels[0], len(categories), statistic)
    fig.draw_without_rendering()  # fix axes sizes for `_mark_significance`
    for ax, roi in zip(panels, rois):
        _mark_significance(
            ax,
            [ax],
            summary[summary["roi"] == roi],
            _significant_timepoints(significance, roi),
        )
    if sharey:
        top = max(ax.get_ylim()[1] for ax in panels)
        for ax in panels:
            ax.set_ylim(ax.get_ylim()[0], top)
    return fig, axs


def figure_path(
    output_dir: Path | str,
    grid_name: str | None,
    roi: str | None,
    statistic: str,
    layout: str,
    page: int | None = None,
) -> Path:
    """Path stem (no extension, see `save_figure`) of an ROI figure.

    Individual figures (`grid_name=None`) are
    `<output_dir>/figures/roi/single/<statistic>/<roi>__<layout>`; grids are
    `<output_dir>/figures/roi/<grid_name>/<grid_name>__<statistic>__<layout>`
    with `__p<page>` appended for paginated grids.
    """
    roi_dir = Path(output_dir) / "figures" / "roi"
    if grid_name is None:
        if roi is None:
            raise ValueError("An individual figure needs a roi")
        return roi_dir / "single" / statistic / f"{roi}__{layout}"
    name = f"{grid_name}__{statistic}__{layout}"
    if page is not None:
        name += f"__p{page}"
    return roi_dir / grid_name / name
