"""Shared CP time mode: one component per row, drawn over the time points."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from gMRItensor.plotting.mode_grid import _plot_time_column_cp
from gMRItensor.plotting.roi_evolution import TIME_LABEL
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import resolve_page_width
from gMRItensor.plotting.utils import scale_mode

plt.style.use(["science", "no-latex"])
matplotlib.use("Agg")


def plot_time_mode(
    time_mode: np.ndarray,
    time_points: list | np.ndarray,
    page_width: str | float = "double",
    width_to_height_ratio: float = 3.0,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot each component of a CP time mode `(n_timepoints, rank)`.

    Columns are unit-norm scaled (`scale_mode`), as in `plot_mode_grid`'s
    time column. Returns `(fig, axs)`, `axs` of shape `(rank,)`.
    """
    scaled = scale_mode(np.asarray(time_mode))
    rank = scaled.shape[1]
    fig, axs = plt.subplots(
        rank,
        1,
        figsize=compute_figsize(
            rank,
            1,
            page_width=resolve_page_width(page_width),
            width_to_height_ratio=width_to_height_ratio,
        ),
        layout="compressed",
        squeeze=False,
    )
    for component, ax in enumerate(axs[:, 0]):
        _plot_time_column_cp(ax, scaled, list(time_points), component)
        ax.set_ylabel(f"Component {component + 1}")
    axs[-1, 0].set_xlabel(TIME_LABEL)
    return fig, axs[:, 0]
