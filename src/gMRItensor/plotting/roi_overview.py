"""Every subject's value in every ROI, one row per session, outliers flagged.

An orientation figure for spotting problematic subjects and ROIs, not a
publication figure: large and vector, to be zoomed. Inputs are per-label
rows (`subject`, `group`, `timepoint`, `label` and the statistic) and
`flag_roi_values` output; nothing here computes statistics.
"""
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # Prevent GUI display errors in containers/terminals
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: F401,E402
from gMRItensor.plotting.utils import get_color_palette  # noqa: E402
from gMRItensor.roi_groups import Block  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.style.use(["science", "no-latex"])

# Width [in] per label of the all-ROI and the region figures, height [in]
# per session row, and width [in] of the column of row notes.
_LABEL_WIDTH = 0.11
_REGION_LABEL_WIDTH = 0.13
_ROW_HEIGHT = 1.7
_NOTE_WIDTH = 1.3
# Extra size [in] for the legend, axis labels and block names.
_MARGIN_WIDTH = 0.7
_MARGIN_HEIGHT = 1.0
# Narrower regions are drawn this many labels wide.
_MIN_REGION_LABELS = 3
# Blocks narrower than this many labels get their name rotated.
_NARROW_BLOCK = 6
# A subject is named in the row note when its flagged values make up at
# least this share (and 2) of the figure's ROIs.
_NOTE_SHARE = 0.1
_NAME_FONTSIZE = 2.2


def _session_label(timepoint: float) -> str:
    return f"{timepoint:g} h"


class _Groups:
    """Group colours (sorted, `get_color_palette`) and their side-by-side
    offsets within a label."""

    def __init__(self, groups: pd.Series) -> None:
        self.categories = sorted(groups.unique())
        self.colors = dict(
            zip(self.categories, get_color_palette(len(self.categories))),
        )
        n = len(self.categories)
        self.dodge = {
            g: (i - (n - 1) / 2) * 0.32 for i, g in enumerate(self.categories)
        }


def _strip(
    ax: matplotlib.axes.Axes,
    session: pd.DataFrame,
    order: Sequence[int],
    value: str,
    groups: _Groups,
    flags: pd.DataFrame,
    size: float,
    rng: np.random.Generator,
) -> None:
    """One dot per subject and label at `order`'s positions, groups side by
    side with a little jitter; each flagged value ringed and named."""
    data = session[session["label"].isin(order)].dropna(subset=[value])
    position = {label: i for i, label in enumerate(order)}
    x = (
        data["label"].map(position).to_numpy(dtype=float)
        + data["group"].map(groups.dodge).to_numpy(dtype=float)
        + rng.uniform(-0.08, 0.08, len(data))
    )
    data = data.assign(_x=x)
    for group in groups.categories:
        members = data[data["group"] == group]
        ax.scatter(
            members["_x"],
            members[value],
            s=size,
            color=groups.colors[group],
            alpha=0.75,
            linewidths=0,
        )
    flagged = data.merge(flags[["subject", "label"]], on=["subject", "label"])
    if not flagged.empty:
        ax.scatter(
            flagged["_x"],
            flagged[value],
            s=size + 5,
            facecolors="none",
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
            gid="flags",
        )
        for _, point in flagged.iterrows():
            ax.annotate(
                str(point["subject"]),
                (point["_x"], point[value]),
                xytext=(1.6, 0),
                textcoords="offset points",
                fontsize=_NAME_FONTSIZE,
                va="center",
                color="0.15",
            )
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.axhline(0, color="0.6", lw=0.4, zorder=0)


def _label_ticks(ax: matplotlib.axes.Axes, order: Sequence[int], size: float) -> None:
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(label) for label in order], rotation=90, fontsize=size)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", length=1.5, top=False)
    ax.tick_params(axis="y", labelsize=5.5)


def _row_note(
    ax: matplotlib.axes.Axes,
    flags: pd.DataFrame,
    labels: Sequence[int],
) -> None:
    """The number of the row's flagged values among `labels`, and the
    subjects flagged in at least `_NOTE_SHARE` of them (a scan-wide
    problem)."""
    flagged = flags[flags["label"].isin(labels)]
    n_rois = len(labels)
    counts = flagged.groupby("subject").size().sort_values(ascending=False)
    counts = counts[counts >= max(2, _NOTE_SHARE * n_rois)]
    lines = [f"{len(flagged)} flagged values"]
    lines += [f"{subject}: {n}/{n_rois} ROIs" for subject, n in counts.items()]
    ax.axis("off")
    ax.text(0, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=5.5)


def _legend(
    fig: matplotlib.figure.Figure,
    groups: _Groups,
    z_cut: float,
    title: str,
) -> None:
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=groups.colors[g],
            markersize=4,
            label=str(g),
        )
        for g in groups.categories
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor="k",
            markersize=4,
            label=f"Flagged: |robust z| > {z_cut:g} among all subjects for that "
            "ROI and session (subject named; zoom in)",
        ),
    )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncols=len(handles),
        frameon=False,
        fontsize=7,
        title=title or None,
        title_fontsize=8,
    )


def _timepoints(
    values: pd.DataFrame,
    timepoints: Sequence[float] | None,
) -> list[float]:
    return sorted(values["timepoint"].unique()) if timepoints is None else [*timepoints]


def plot_roi_overview(
    values: pd.DataFrame,
    blocks: Sequence[Block],
    value: str,
    flags: pd.DataFrame,
    ylabel: str,
    title: str = "",
    z_cut: float = 3.5,
    timepoints: Sequence[float] | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Every label of `blocks` along x (in block order), one row per session.

    `values` has one row per subject, session and label (`subject`,
    `group`, `timepoint`, `label`, `value`); `flags` is `flag_roi_values`
    output for it, made with `z_cut`. Each row has its own y-scale; each
    flagged value is ringed and named, and the note beside the row counts
    them. `timepoints` (default: those in `values`) are the rows.

    Returns `(fig, axs)`, `axs` of shape `(n_sessions, 1)`.
    """
    order = [label for _, ids in blocks for label in ids]
    rows = _timepoints(values, timepoints)
    groups = _Groups(values["group"])
    plot_width = _LABEL_WIDTH * len(order)
    fig, grid = plt.subplots(
        len(rows),
        2,
        figsize=(
            plot_width + _NOTE_WIDTH + _MARGIN_WIDTH,
            _ROW_HEIGHT * len(rows) + _MARGIN_HEIGHT,
        ),
        layout="constrained",
        gridspec_kw={"width_ratios": [plot_width, _NOTE_WIDTH]},
        squeeze=False,
    )
    axs = grid[:, :1]
    rng = np.random.default_rng(0)
    for row, timepoint in enumerate(rows):
        ax = axs[row, 0]
        session_flags = flags[flags["timepoint"] == timepoint]
        _strip(
            ax,
            values[values["timepoint"] == timepoint],
            order,
            value,
            groups,
            session_flags,
            5,
            rng,
        )
        _row_note(grid[row, 1], session_flags, order)
        _label_ticks(ax, order, 4.5)
        start = 0
        for name, ids in blocks:
            if start:
                ax.axvline(start - 0.5, color="0.75", lw=0.5, zorder=0)
            if row == 0:
                narrow = len(ids) < _NARROW_BLOCK
                ax.text(
                    start + (len(ids) - 1) / 2,
                    1.02,
                    name,
                    transform=ax.get_xaxis_transform(),
                    ha="left" if narrow else "center",
                    va="bottom",
                    fontsize=6,
                    rotation=45 if narrow else 0,
                    rotation_mode="anchor",
                )
            start += len(ids)
        ax.text(
            0.003,
            0.95,
            _session_label(timepoint),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontweight="bold",
        )
        ax.set_ylabel(ylabel, fontsize=7)
        if row < len(rows) - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axs[-1, 0].set_xlabel("FreeSurfer label", fontsize=7)
    _legend(fig, groups, z_cut, title)
    return fig, axs


def plot_region_columns(
    values: pd.DataFrame,
    regions: Sequence[Block],
    value: str,
    flags: pd.DataFrame,
    ylabel: str,
    title: str = "",
    z_cut: float = 3.5,
    timepoints: Sequence[float] | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """One column per region (its labels along x, width per label), one row
    per session; every panel has its own y-scale.

    Inputs as in `plot_roi_overview`. The note beside each row counts the
    flagged values over all the regions' labels.

    Returns `(fig, axs)`, `axs` of shape `(n_sessions, len(regions))`.
    """
    rows = _timepoints(values, timepoints)
    groups = _Groups(values["group"])
    widths = [max(len(ids), _MIN_REGION_LABELS) for _, ids in regions]
    note_width = _NOTE_WIDTH / _REGION_LABEL_WIDTH
    fig, grid = plt.subplots(
        len(rows),
        len(regions) + 1,
        figsize=(
            _REGION_LABEL_WIDTH * sum(widths) + _NOTE_WIDTH + _MARGIN_WIDTH * 2,
            _ROW_HEIGHT * len(rows) + _MARGIN_HEIGHT,
        ),
        layout="constrained",
        gridspec_kw={"width_ratios": [*widths, note_width]},
        squeeze=False,
    )
    axs = grid[:, :-1]
    labels = [label for _, ids in regions for label in ids]
    rng = np.random.default_rng(0)
    for row, timepoint in enumerate(rows):
        session = values[values["timepoint"] == timepoint]
        session_flags = flags[flags["timepoint"] == timepoint]
        for col, (name, ids) in enumerate(regions):
            ax = axs[row, col]
            _strip(ax, session, ids, value, groups, session_flags, 6, rng)
            _label_ticks(ax, ids, 5)
            if row == 0:
                ax.set_title(name, fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{_session_label(timepoint)}\n{ylabel}", fontsize=6.5)
            if row < len(rows) - 1:
                ax.tick_params(axis="x", labelbottom=False)
        _row_note(grid[row, -1], session_flags, labels)
    _legend(fig, groups, z_cut, title)
    return fig, axs
