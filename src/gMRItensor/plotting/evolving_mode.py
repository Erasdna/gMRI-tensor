"""Plotting for PARAFAC2's and CMF's evolving (subject-specific time) mode."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import torch
from gMRItensor.group_statistics import compare_groups_over_time
from gMRItensor.group_statistics import resolve_subject_groups
from gMRItensor.group_statistics import summarize_groups_over_time
from gMRItensor.model_io import NEGLIGIBLE_SUBJECT_WEIGHT
from gMRItensor.model_io import negligible_subjects
from gMRItensor.plotting.utils import apply_row_ylims
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import plot_group_ribbons
from gMRItensor.plotting.utils import plot_subject_curves
from gMRItensor.plotting.utils import scale_mode
from gMRItensor.plotting.utils import TIME_LABEL
from matplotlib.lines import Line2D

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

    The states are each subject's time course as fitted, so this only
    bridges torch to the numpy the rest of this module works in.
    """
    return [_to_numpy(factor) for factor in evolving_states]


def build_long_evolving_dataframe(
    scaled_factors: list[np.ndarray],
    timepoints_per_subject: list[np.ndarray],
    subjects: list[str],
    groups: list[str],
    n_components: int,
    negligible: np.ndarray | None = None,
) -> pd.DataFrame:
    """Stack per-subject evolving-mode factors into one long-form frame.

    One row per `(subject, timepoint, component)`, with columns `subject`,
    `group`, `timepoint`, `component`, `value`. Timepoints keep their raw
    values rather than being binned, so a later `groupby("timepoint")`
    aligns ragged per-subject timepoints by exact value. `negligible`, a
    `(n_subjects, n_components)` mask (`negligible_mask`), leaves out those
    subjects' rows for those components.
    """
    component_columns = [str(c) for c in range(n_components)]
    frames = []
    for position, (factor, timepoints, subject, group) in enumerate(
        zip(scaled_factors, timepoints_per_subject, subjects, groups),
    ):
        frame = pd.DataFrame(factor[:, :n_components], columns=component_columns)
        frame["subject"] = subject
        frame["group"] = group
        frame["timepoint"] = np.asarray(timepoints)
        frame["_position"] = position
        frames.append(frame)

    long_df = pd.concat(frames, ignore_index=True).melt(
        id_vars=["subject", "group", "timepoint", "_position"],
        value_vars=component_columns,
        var_name="component",
        value_name="value",
    )
    long_df["component"] = long_df["component"].astype(int)
    if negligible is not None:
        dropped = negligible[long_df["_position"], long_df["component"]]
        long_df = long_df[~dropped].reset_index(drop=True)
    return long_df.drop(columns="_position")


def negligible_mask(
    subject_mode: np.ndarray | None,
    n_subjects: int,
    n_components: int,
    atol: float,
) -> np.ndarray | None:
    """`(n_subjects, n_components)` mask of the components each subject does
    not express (`negligible_subjects`), or None without `subject_mode`.
    Raises `ValueError` unless `subject_mode` has a row per subject and a
    column per component."""
    if subject_mode is None:
        return None
    subject_mode = np.asarray(subject_mode)
    if subject_mode.ndim != 2 or subject_mode.shape[0] != n_subjects:
        raise ValueError(
            f"subject_mode must have one row per subject ({n_subjects}), got "
            f"shape {subject_mode.shape}",
        )
    if subject_mode.shape[1] < n_components:
        raise ValueError(
            f"subject_mode has {subject_mode.shape[1]} components, the evolving "
            f"mode {n_components}",
        )
    return negligible_subjects(subject_mode[:, :n_components], atol)


def with_zeroed_curves(
    negligible: np.ndarray | None,
    scaled_factors: list[np.ndarray],
) -> np.ndarray | None:
    """`negligible`, plus the curves `scale_mode` set to zero (negligible
    next to the subject's other columns): drawn as zero, they must not enter
    the ribbon or the tests as zeros either. None stays None."""
    if negligible is None:
        return None
    n_components = negligible.shape[1]
    zeroed = np.array(
        [~np.any(factor[:, :n_components] != 0, axis=0) for factor in scaled_factors],
    )
    return negligible | zeroed


def all_timepoints(timepoints_per_subject: list[np.ndarray]) -> list[float]:
    """Every subject's time points, sorted: the x-ticks of each row, so a
    time point scanned only by a dropped subject keeps its tick."""
    return sorted(set(np.concatenate([np.asarray(t) for t in timepoints_per_subject])))


def group_legend(
    fig: matplotlib.figure.Figure,
    categories: list[str],
    color_by_group: dict[str, str],
) -> None:
    """One group legend above all panels, as in `roi_evolution`. Built from
    the colours rather than the drawn ribbons, so it is there even when no
    subject expresses the first component."""
    handles = [
        Line2D([], [], color=color_by_group[c], marker="o", markersize=3)
        for c in categories
    ]
    fig.legend(
        handles,
        [str(c) for c in categories],
        loc="outside upper center",
        ncols=len(categories),
        frameon=False,
    )


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
    subject_mode: np.ndarray | None = None,
    negligible_atol: float = NEGLIGIBLE_SUBJECT_WEIGHT,
) -> tuple[matplotlib.figure.Figure, np.ndarray, pd.DataFrame]:
    """Plot each subject's own PARAFAC2/CMF evolving-mode (time) pattern.

    One row per component, `1 + n_groups` columns: the first column is a
    ribbon plot (per-group mean +/- SEM band, colored by `group_variable`),
    followed by one column per group showing that group's individual
    subjects' own reconstructed time curves. Per-timepoint group-difference
    significance is computed but not drawn -- it's returned as a
    `pd.DataFrame` instead, see `Returns`.

    Parameters
    ----------
    evolving_factors : list[np.ndarray]
        Per-subject `(n_timepoints_i, rank)` evolving states `B_i`, e.g.
        from `evolving_factors_to_numpy`. Each subject's columns are scaled
        to unit norm: the curves show shape, the subject weights (see
        `subject_mode`) only decide which are drawn.
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
    subject_mode : np.ndarray, optional
        `(n_subjects, rank)` subject weights. A subject whose weight is
        below `negligible_atol` in absolute value does not express that
        component: its curve is drawn as zero, and it is left out of the
        ribbon and the significance tests.
    negligible_atol : float, optional
        See `subject_mode`.

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
    groups = resolve_subject_groups(subjects, subject_info, group_variable)

    categories = sorted(set(groups))
    colors = get_color_palette(len(categories))
    color_by_group = dict(zip(categories, colors))

    n_components = evolving_factors[0].shape[1]
    negligible = negligible_mask(
        subject_mode,
        len(subjects),
        n_components,
        negligible_atol,
    )
    scaled_factors = [scale_mode(factor) for factor in evolving_factors]
    negligible = with_zeroed_curves(negligible, scaled_factors)

    long_df = build_long_evolving_dataframe(
        scaled_factors,
        timepoints_per_subject,
        subjects,
        groups,
        n_components,
        negligible,
    )
    ribbon_stats = summarize_groups_over_time(long_df, facet="component")
    significance = compare_groups_over_time(
        long_df,
        categories,
        facet="component",
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

    # Dropped subjects are still drawn, as zero, in their group's column.
    drawn_factors = scaled_factors
    if negligible is not None:
        drawn_factors = [
            np.where(flags, 0.0, factor)
            for factor, flags in zip(scaled_factors, negligible)
        ]
    row_timepoints = all_timepoints(timepoints_per_subject)

    row_ylims = []
    for component in range(n_components):
        ribbon_stats_component = ribbon_stats.loc[
            ribbon_stats["component"] == component,
        ]
        row_min, row_max = plot_group_ribbons(
            axs[component, 0],
            ribbon_stats_component,
            categories,
            color_by_group,
        )

        for col, category in enumerate(categories, start=1):
            g_min, g_max = plot_subject_curves(
                axs[component, col],
                [
                    (timepoints_per_subject[i], drawn_factors[i][:, component])
                    for i in group_indices[category]
                ],
                color_by_group[category],
            )
            row_min, row_max = min(row_min, g_min), max(row_max, g_max)
        row_ylims.append((row_min, row_max))

        for ax in axs[component, :]:
            ax.set_xticks(row_timepoints)

        axs[component, 0].set_ylabel(rf"Component {component+1}")
        if component == 0:
            axs[component, 0].set_title(r"Mean $\pm$ SEM")
            for col, category in enumerate(categories, start=1):
                axs[component, col].set_title(str(category))
        if component == n_components - 1:
            for ax in axs[component, :]:
                ax.set_xlabel(TIME_LABEL)

    apply_row_ylims(axs, row_ylims)
    group_legend(fig, categories, color_by_group)
    fig.align_titles()

    return fig, axs, significance
