"""Tracer evolution per ROI and subject group, drawn from precomputed tables.

Inputs are the tables `gmri plot statistics` writes (per-scan ROI rows,
`summarize_groups_over_time` and `compare_groups_over_time` output faceted
by `roi`); nothing here computes statistics.
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
from gMRItensor.plotting.utils import plot_sd_bands
from gMRItensor.plotting.utils import plot_subject_curves
from gMRItensor.plotting.utils import resolve_page_width
from gMRItensor.plotting.utils import SD_BAND_ALPHAS
from gMRItensor.plotting.utils import SD_BAND_COLOR
from gMRItensor.roi_groups import roi_display_name
from matplotlib.patches import Patch

plt.style.use(["science", "no-latex"])
matplotlib.use("Agg")

TIME_LABEL = "Time after injection [h]"

_STATISTIC_LABELS = {
    "median_concentration": "Median concentration (mM)",
    "mean_concentration": "Mean concentration (mM)",
    "total_amount": "Total amount (mmol)",
}

# Gap [pt] between the top of the ribbon and a significance marker.
_STAR_OFFSET_PT = 1.0
# Extra figure height [in] for the figure legend and shared time label,
# which take fixed space regardless of the number of rows.
_FIGURE_LABEL_HEIGHT = 0.5
SD_BAND_LABEL = r"Cohort mean $\pm$1/2/3 SD"


def statistic_label(statistic: str, signal: str = "delta_R1") -> str:
    """Y-axis label with unit for an ROI statistic.

    `signal` is the preprocessed signal (`read_signal_metadata`): `delta_R1`
    (in 1/s) or `ratio` (T1w post/baseline).
    """
    if statistic in ("median", "mean"):
        name = statistic.capitalize()
        return f"{name} signal ratio" if signal == "ratio" else f"{name} ΔR1 (1/s)"
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
    statistic: str,
    signal: str,
    sd_bands: bool = False,
) -> None:
    """One legend above all panels plus shared time and statistic labels.

    Figure-level labels instead of per-axes ones, so narrow (single column)
    figures do not collide. With `sd_bands` the legend also names the grey
    cohort bands.
    """
    handles, labels = legend_ax.get_legend_handles_labels()
    if sd_bands:
        handles.append(
            Patch(
                color=SD_BAND_COLOR,
                alpha=sum(SD_BAND_ALPHAS),
                linewidth=0,
            ),
        )
        labels.append(SD_BAND_LABEL)
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=len(handles),
        frameon=False,
    )
    fontsize = plt.rcParams["axes.labelsize"]
    fig.supxlabel(TIME_LABEL, fontsize=fontsize)
    fig.supylabel(statistic_label(statistic, signal), fontsize=fontsize)


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


def _name_points(
    group_axes: Sequence[matplotlib.axes.Axes],
    categories: list[str],
    points: pd.DataFrame,
    statistic: str,
) -> None:
    """Name each subject next to its point in its group's column: above a
    high point, below a low one."""
    # As text: a CSV round trip turns numeric group codes into strings.
    names = [str(category) for category in categories]
    for _, point in points.iterrows():
        if str(point["group"]) not in names:
            continue
        above = point["z"] > 0
        group_axes[names.index(str(point["group"]))].annotate(
            str(point["subject"]),
            xy=(point["timepoint"], point[statistic]),
            xytext=(3, 3 if above else -3),
            textcoords="offset points",
            ha="left",
            va="bottom" if above else "top",
            fontsize="x-small",
        )


def plot_roi_evolution_rows(
    summary: pd.DataFrame,
    subject_values: pd.DataFrame,
    significance: pd.DataFrame,
    rois: Sequence[str],
    statistic: str,
    page_width: str | float = "double",
    width_to_height_ratio: float = 1.618,
    signal: str = "delta_R1",
    cohort: pd.DataFrame | None = None,
    outside: pd.DataFrame | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """One row per ROI: group ribbons, then each group's subject curves.

    Same layout as `plot_evolving_mode`, with ROIs as rows; each row shares
    one y axis, labelled in the ribbon column only. `summary` is
    `summarize_groups_over_time(..., facet="roi")` output, `subject_values`
    the per-scan ROI frame it came from (with `group` and `timepoint`) and
    `significance` the matching `compare_groups_over_time` output with a
    `significant` column, for the same `statistic`. Significant time
    points get a `*` above the ribbon. `page_width` is a `JOURNAL_WIDTHS`
    name or inches. With `cohort` (`summarize_cohort_over_time(...,
    facet="roi")` output) every subject column also gets the cohort mean
    and grey +/-1, 2 and 3 SD bands behind the curves, to spot outlying
    subjects; the y-range then covers +/-3 SD. Subjects in `outside`
    (`subjects_outside_sd(..., facet="roi")` output) are named at that
    point, as the outlier figure names subjects.

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
        sharey="row",
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
            if cohort is not None:
                b_min, b_max = plot_sd_bands(
                    axs[row, col],
                    cohort[cohort["roi"] == roi],
                )
                row_min, row_max = min(row_min, b_min), max(row_max, b_max)
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
        axs[row, 0].set_ylabel(roi_display_name(roi))
        if outside is not None:
            _name_points(
                axs[row, 1:],
                categories,
                outside[outside["roi"] == roi],
                statistic,
            )

    axs[0, 0].set_title(r"Mean $\pm$ SEM")
    for col, category in enumerate(categories, start=1):
        axs[0, col].set_title(str(category))
    _add_figure_labels(
        fig,
        axs[0, 0],
        statistic,
        signal,
        sd_bands=cohort is not None,
    )

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
    signal: str = "delta_R1",
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
        ax.set_title(roi_display_name(roi))
    if sharey:
        shared = (min(lo for lo, _ in ylims), max(hi for _, hi in ylims))
        ylims = [shared] * len(ylims)
    apply_row_ylims(np.array(panels).reshape(-1, 1), ylims)
    _add_figure_labels(fig, panels[0], statistic, signal)
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
    roi: str,
    statistic: str,
    layout: str,
) -> Path:
    """Path stem (no extension, see `save_figure`) of one ROI's figure:
    `<output_dir>/figures/roi/single/<statistic>/<roi>__<layout>`."""
    return (
        Path(output_dir) / "figures" / "roi" / "single" / statistic / f"{roi}__{layout}"
    )
