"""Restart reproducibility: pairwise FMS between restarts, sorted by error."""
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from gMRItensor.plotting.utils import resolve_page_width

plt.style.use(["science", "no-latex"])
matplotlib.use("Agg")

#: Sequential, one hue: FMS is a magnitude from 0 (unrelated) to 1 (same).
FMS_CMAP = "Blues"
#: Tick labels are thinned to at most this many per axis.
MAX_TICK_LABELS = 20


def _order_ticks(ax: matplotlib.axes.Axes, n: int, axis: str) -> None:
    """Tick the restarts by their place in the error ordering, 1 = lowest."""
    step = max(1, math.ceil(n / MAX_TICK_LABELS))
    positions = list(range(0, n, step))
    labels = [str(i + 1) for i in positions]
    if axis == "x":
        ax.set_xticks(positions, labels)
    else:
        ax.set_yticks(positions, labels)
    ax.tick_params(which="minor", length=0)


def _colour_range(
    fms: np.ndarray,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    """`(vmin, vmax)`, defaulting to the off-diagonal (restart-pair) range.

    The diagonal is always 1 and carries no information, so it is left out;
    it simply saturates when the pairs stay below 1.
    """
    pairs = fms[~np.eye(len(fms), dtype=bool)]
    pairs = pairs[np.isfinite(pairs)]
    low = float(pairs.min()) if vmin is None and pairs.size else vmin
    high = float(pairs.max()) if vmax is None and pairs.size else vmax
    low = 0.0 if low is None else low
    high = 1.0 if high is None else high
    if low >= high:  # all pairs equal: centre them in a small range
        low, high = low - 1e-6, high + 1e-6
    return low, high


def plot_reproducibility(
    fms: pd.DataFrame,
    errors: pd.Series,
    page_width: str | float = "single",
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Draw the restart-vs-restart FMS matrix with each restart's error above.

    `fms` and `errors` are `reproducibility.restart_fms_matrix`'s output (or
    `read_reproducibility`'s): restarts already sorted by error. The axes
    show only that order (1 = lowest error); the seeds stay in the CSV.
    Low-error restarts that reached the same minimum show as a dark
    block in the top-left corner.

    The colour range spans the FMS between distinct restarts, so small
    differences stay visible when every score is close to 1. Pass `vmin`
    and `vmax` (e.g. 0 and 1) for a fixed range that is comparable across
    figures.

    Returns `(fig, axs)`, `axs = [error_ax, fms_ax, colorbar_ax]`.
    """
    n = len(fms)
    width = resolve_page_width(page_width)
    fig = plt.figure(figsize=(width, width * 1.15), layout="compressed")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[20, 1])
    fms_ax = fig.add_subplot(grid[1, 0])
    error_ax = fig.add_subplot(grid[0, 0], sharex=fms_ax)
    colorbar_ax = fig.add_subplot(grid[1, 1])

    values = fms.to_numpy(dtype=float)
    low, high = _colour_range(values, vmin, vmax)
    image = fms_ax.imshow(
        values,
        cmap=FMS_CMAP,
        vmin=low,
        vmax=high,
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(image, cax=colorbar_ax, label="FMS")
    _order_ticks(fms_ax, n, "x")
    _order_ticks(fms_ax, n, "y")
    fms_ax.set_xlabel("Restart, by error")
    fms_ax.set_ylabel("Restart, by error")

    error_ax.plot(
        np.arange(n),
        errors.to_numpy(dtype=float),
        color="0.25",
        linewidth=1.0,
        marker="o",
        markersize=2.5,
    )
    error_ax.set_ylabel("Error")
    error_ax.tick_params(axis="x", which="both", labelbottom=False)
    error_ax.set_xlim(-0.5, n - 0.5)
    return fig, np.array([error_ax, fms_ax, colorbar_ax], dtype=object)
