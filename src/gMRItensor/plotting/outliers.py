"""Leverage vs. residual per subject, with the outlier cut-offs.

Draws `gMRItensor.outliers.outlier_table` output; nothing here computes
diagnostics. Layout follows Fida et al. (2022, Fig. 8).
"""
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Prevent GUI display errors in containers/terminals
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: F401,E402
from gMRItensor.outliers import FLAG_COLUMNS  # noqa: E402
from gMRItensor.outliers import OutlierCutoffs  # noqa: E402
from gMRItensor.plotting.utils import compute_figsize  # noqa: E402
from gMRItensor.plotting.utils import get_color_palette  # noqa: E402
from gMRItensor.plotting.utils import resolve_page_width  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.style.use(["science", "no-latex"])

#: Colour of subjects without a group and of the cut-off lines.
_NEUTRAL = "0.5"
#: A cut-off beyond this multiple of the largest value is noted, not drawn,
#: so it does not squash the subjects into a corner.
_MAX_AXIS_STRETCH = 1.5


def _labelled(table: pd.DataFrame) -> pd.Index:
    """Flagged subjects plus the highest on each axis."""
    flagged = table.index[table[list(FLAG_COLUMNS)].any(axis=1)]
    return flagged.union([table["leverage"].idxmax(), table["residual"].idxmax()])


def _visible(cutoff: float | None, largest: float) -> bool:
    return cutoff is not None and cutoff <= _MAX_AXIS_STRETCH * largest


def plot_outliers(
    table: pd.DataFrame,
    cutoffs: OutlierCutoffs,
    groups: Sequence[Any],
    group_variable: str,
    page_width: str | float = "single",
    width_to_height_ratio: float = 1.3,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Scatter each subject's leverage against its normalised residual.

    `table` and `cutoffs` are `outlier_table` output; `groups` holds each
    row's group value (None: no group), coloured as in the other figures
    (groups sorted by value, `get_color_palette`), subjects without one
    grey. Dashed lines
    mark the leverage and tlviz residual cut-offs, a dotted line the robust
    residual cut-off; one far beyond all subjects is only listed, with its
    value, in the legend above the axes. Flagged subjects and the highest on each axis are
    named.
    """
    fig, ax = plt.subplots(
        figsize=compute_figsize(
            1,
            1,
            page_width=resolve_page_width(page_width),
            width_to_height_ratio=width_to_height_ratio,
        ),
        layout="compressed",
    )
    categories = sorted({g for g in groups if g is not None})
    colors = get_color_palette(len(categories))
    series = [
        (str(category), color, [g == category for g in groups])
        for category, color in zip(categories, colors)
    ]
    series.append((f"No {group_variable}", _NEUTRAL, [g is None for g in groups]))
    for label, color, rows in series:
        if not any(rows):
            continue
        ax.scatter(
            table.loc[rows, "leverage"],
            table.loc[rows, "residual"],
            s=12,
            color=color,
            label=label,
            zorder=3,
        )

    max_leverage = float(table["leverage"].max())
    max_residual = float(table["residual"].max())
    lines = (
        (ax.axvline, cutoffs.leverage, max_leverage, "--", "Leverage cut-off"),
        (ax.axhline, cutoffs.residual, max_residual, "--", "Residual cut-off"),
        (
            ax.axhline,
            cutoffs.robust_residual,
            max_residual,
            ":",
            "Robust residual cut-off",
        ),
    )
    x_top, y_top = max_leverage, max_residual
    off_scale: list[Line2D] = []
    for draw, cutoff, largest, style, label in lines:
        if cutoff is None:
            continue
        line_style = {"color": _NEUTRAL, "linestyle": style, "linewidth": 0.8}
        if not _visible(cutoff, largest):
            # A legend entry only, with its value.
            off_scale.append(
                Line2D(
                    [],
                    [],
                    label=f"{label}: {cutoff:.3g} (off scale)",
                    **line_style,
                ),
            )
            continue
        draw(cutoff, label=label, **line_style)
        if draw == ax.axvline:
            x_top = max(x_top, cutoff)
        else:
            y_top = max(y_top, cutoff)
    ax.set_xlim(0, 1.08 * x_top)
    ax.set_ylim(0, 1.08 * y_top)

    x_max = ax.get_xlim()[1]
    for row in _labelled(table):
        x = table.at[row, "leverage"]
        # Near the right edge, name the point on its left so the name stays
        # inside the axes.
        right_edge = x > 0.8 * x_max
        ax.annotate(
            table.at[row, "subject"],
            xy=(x, table.at[row, "residual"]),
            xytext=(-3 if right_edge else 3, 3),
            textcoords="offset points",
            ha="right" if right_edge else "left",
            fontsize="x-small",
        )

    ax.set_xlabel("Leverage")
    ax.set_ylabel("Normalised residual")
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(off_scale)
    labels.extend(line.get_label() for line in off_scale)
    # Above the axes, as in the other figures, so it never hides a subject.
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=2,
        frameon=False,
        fontsize="x-small",
    )
    return fig, ax
