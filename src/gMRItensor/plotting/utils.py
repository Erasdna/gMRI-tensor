"""Shared utility functions for plotting."""
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use(["science", "no-latex"])

# Spatial modes are unit-norm scaled by `scale_mode`, so they have no
# physical unit.
SPATIAL_COLORBAR_LABEL = "Loading (a.u.)"

# Journal figure widths in inches, for `page_width` arguments.
JOURNAL_WIDTHS = {"single": 3.5, "onehalf": 5.5, "double": 7.0}
#: x-axis label of every time-course figure.
TIME_LABEL = "Time after injection [h]"


def scale_mode(arr: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Scale a mode's columns to unit L2 norm; zero out negligible columns.

    A column whose norm is negligible carries no shape, only solver residue,
    so normalizing it to unit norm would manufacture a full-amplitude curve
    out of numerical noise. Such columns are returned as zeros instead, as
    is an all-zero column (never NaN).

    `rtol` is relative to the largest column norm in the same array, not
    absolute. An absolute floor fails both ways: low enough to be safe and
    it misses ~1e-07 residue; high enough to catch that and it flattens a
    real component in an array that is uniformly small-scale. Relative means
    a column is zeroed only when it is vanishingly small next to its own
    siblings, and never when every column shares a small scale.
    """
    norms = np.linalg.norm(arr, axis=0)
    largest = norms.max(initial=0.0)
    negligible = norms <= rtol * largest
    safe_norms = np.where(negligible, 1.0, norms)
    return np.where(negligible[None, :], 0.0, arr / safe_norms[None, :])


def compute_figsize(
    n_components: int,
    n_columns: int,
    page_width: float,
    width_to_height_ratio: float = 1.618,
    add_title_margin: bool = True,
) -> tuple[float, float]:
    """Compute a `(width, height)` figure size in inches for a subplot grid.

    `page_width` is the target journal width: typically 3.5 (single column),
    5.5 (1.5 column) or 7.0 (double column). `width_to_height_ratio` applies
    per subplot and defaults to the golden ratio.
    """
    width = page_width
    height_per_row = (width / n_columns) / width_to_height_ratio
    # 10% extra per row for titles and labels.
    height = height_per_row * n_components * (1.1 if add_title_margin else 1.0)

    return (width, height)


def get_color_palette(n_colors: int) -> list:
    """Pick `n_colors` from matplotlib's tab10 palette, cycling past 10."""
    return [f"C{3*i % 10}" for i in range(n_colors)]


def create_colorbar_with_offset(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    mappable: matplotlib.cm.ScalarMappable,
    cax: matplotlib.axes.Axes,
    label: str | None = None,
    format_string: str | None = None,
    precision: int = 1,
    horizontal_alignment: Literal["left", "center", "right"] = "center",
) -> float:
    """Create a colorbar in scientific notation and return its text offset.

    `format_string` (e.g. `':.2f'`, `':.1e'`) overrides `precision` when
    given. Returns half the exponent text's width in axes coordinates, which
    callers use to shift the colorbar so the exponent does not overlap
    neighbouring elements.
    """
    formatter: plt.matplotlib.ticker.Formatter
    if format_string is None:

        class PrecisionScalarFormatter(plt.matplotlib.ticker.ScalarFormatter):
            def _set_format(self) -> None:
                self.format = f"%.{precision}f"

        precision_formatter = PrecisionScalarFormatter(useMathText=True)
        precision_formatter.set_scientific(True)
        precision_formatter.set_powerlimits((0, 0))
        precision_formatter.set_useOffset(False)
        formatter = precision_formatter
    else:
        formatter = plt.matplotlib.ticker.StrMethodFormatter(format_string)

    cbar = fig.colorbar(
        mappable,
        cax=cax,
        ax=ax,
        use_gridspec=True,
        format=formatter,
    )
    cbar.ax.yaxis.offsetText.set_visible(True)
    cbar.ax.yaxis.offsetText.set_horizontalalignment(horizontal_alignment)
    if label is not None:
        cbar.set_label(label=label)

    # Draw first, or the exponent text has no measurable extent yet.
    fig.canvas.draw()
    offset_text_bbox = cbar.ax.yaxis.offsetText.get_window_extent()
    offset_text_bbox_ax = offset_text_bbox.transformed(ax.transAxes.inverted())

    return offset_text_bbox_ax.width / 2.0


def plot_group_ribbons(
    ax: matplotlib.axes.Axes,
    summary: pd.DataFrame,
    categories: list[str],
    color_by_group: dict[str, str],
) -> tuple[float, float]:
    """Draw a mean +/- SEM ribbon per group, for one facet.

    `summary` is `group_statistics.summarize_groups_over_time` output
    pre-filtered to that facet. Returns the `(ymin, ymax)` actually drawn,
    for row-wise y-limit finalization.
    """
    ymin, ymax = np.inf, -np.inf
    for category in categories:
        group_stats = summary.loc[summary["group"] == category].sort_values(
            "timepoint",
        )
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


#: Opacity of the nested cohort SD bands, innermost (±1 SD) first; they
#: stack, so the ±1 SD region is the darkest.
SD_BAND_ALPHAS = (0.18, 0.12, 0.08)
SD_BAND_COLOR = "0.45"


def plot_sd_bands(
    ax: matplotlib.axes.Axes,
    cohort: pd.DataFrame,
    n_sd: Sequence[int] = (1, 2, 3),
) -> tuple[float, float]:
    """Draw the cohort mean and nested mean +/- k SD bands in grey, below
    any curves, for one facet.

    `cohort` is `group_statistics.summarize_cohort_over_time` output
    pre-filtered to that facet. Returns the `(ymin, ymax)` drawn, for
    row-wise y-limit finalization.
    """
    stats = cohort.sort_values("timepoint")
    timepoints = stats["timepoint"].to_numpy()
    mean = stats["mean"].to_numpy()
    std = stats["std"].to_numpy()
    for k, alpha in zip(n_sd, SD_BAND_ALPHAS):
        ax.fill_between(
            timepoints,
            mean - k * std,
            mean + k * std,
            color=SD_BAND_COLOR,
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )
    ax.plot(timepoints, mean, color=SD_BAND_COLOR, linewidth=0.8, zorder=1)
    widest = max(n_sd)
    return float(np.min(mean - widest * std)), float(np.max(mean + widest * std))


def plot_subject_curves(
    ax: matplotlib.axes.Axes,
    curves: list[tuple[np.ndarray, np.ndarray]],
    color: str,
) -> tuple[float, float]:
    """Draw one `(timepoints, values)` curve per subject, all in `color`.

    Returns the `(ymin, ymax)` actually drawn, for row-wise y-limit
    finalization.
    """
    ymin, ymax = np.inf, -np.inf
    for timepoints, values in curves:
        ax.plot(timepoints, values, color=color, alpha=0.6, marker="o", markersize=3)
        ymin = min(ymin, np.min(values))
        ymax = max(ymax, np.max(values))
    return ymin, ymax


def apply_row_ylims(
    axs: np.ndarray,
    row_ylims: list[tuple[float, float]],
) -> None:
    """Apply a shared, padded y-limit across each row's axes.

    Same two-pass pattern as `subject_mode._finalize_boxplot_axes`: draw
    first so the data range is known, then equalize each row so the ribbon
    and per-group columns stay visually comparable. `row_ylims` holds one
    `(ymin, ymax)` per row; non-finite rows are left untouched.
    """
    for row, (ymin, ymax) in enumerate(row_ylims):
        if not (np.isfinite(ymin) and np.isfinite(ymax)):
            continue
        span = ymax - ymin if ymax > ymin else 1.0
        for ax in axs[row, :]:
            ax.set_ylim(ymin - 0.05 * span, ymax + 0.05 * span)


def resolve_page_width(page_width: str | float) -> float:
    """A `JOURNAL_WIDTHS` name or a width in inches, as inches."""
    if isinstance(page_width, str):
        if page_width not in JOURNAL_WIDTHS:
            raise ValueError(
                f"page_width must be one of {sorted(JOURNAL_WIDTHS)} or a number "
                f"of inches, got {page_width!r}",
            )
        return JOURNAL_WIDTHS[page_width]
    return float(page_width)


def save_figure(
    fig: matplotlib.figure.Figure,
    path_stem: Path | str,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save `fig` as `<path_stem>.<format>` per format, then close it.

    Parent directories are created. Returns the written paths, in `formats`
    order.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [path_stem.with_name(f"{path_stem.name}.{fmt}") for fmt in formats]
    try:
        for path in paths:
            fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
    return paths
