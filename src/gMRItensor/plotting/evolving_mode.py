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
    negligible: np.ndarray | None = None,
) -> pd.DataFrame:
    """Stack per-subject evolving-mode factors into one long-form frame.

    One row per `(subject, timepoint, component)`, with columns `subject`,
    `group`, `timepoint`, `component`, `value`. Timepoints keep their raw
    values rather than being binned, so a later `groupby("timepoint")`
    aligns ragged per-subject timepoints by exact value. `negligible`, a
    `(n_subjects, n_components)` mask, leaves out those subjects' rows for
    those components.
    """
    component_columns = [str(c) for c in range(n_components)]
    frames = []
    for i, (factor, timepoints, subject, group) in enumerate(
        zip(scaled_factors, timepoints_per_subject, subjects, groups),
    ):
        values = np.asarray(factor[:, :n_components], dtype=float)
        if negligible is not None:
            # NaN marks the rows to leave out after the melt.
            values = np.where(negligible[i, :n_components], np.nan, values)
        frame = pd.DataFrame(values, columns=component_columns)
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
    if negligible is not None:
        long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)
    return long_df


def _mask_negligible(
    scaled_factors: list[np.ndarray],
    subject_mode: np.ndarray | None,
    atol: float,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Zero the components a subject does not express (`negligible_subjects`
    of `subject_mode`); returns the factors and the mask, None without
    `subject_mode`."""
    if subject_mode is None:
        return scaled_factors, None
    negligible = negligible_subjects(subject_mode, atol)
    masked = [
        np.where(flags[None, : factor.shape[1]], 0.0, factor)
        for factor, flags in zip(scaled_factors, negligible)
    ]
    return masked, negligible


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
        Per-subject `(n_timepoints_i, rank)` time courses, e.g. from
        `model_io.subject_time_courses` (PARAFAC2's scaled by the subject
        weights).
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
    scaled_factors, negligible = _mask_negligible(
        [scale_mode(factor) for factor in evolving_factors],
        subject_mode,
        negligible_atol,
    )

    long_df = _build_long_evolving_dataframe(
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
                    (timepoints_per_subject[i], scaled_factors[i][:, component])
                    for i in group_indices[category]
                ],
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
                ax.set_xlabel("Time after injection [h]")

    apply_row_ylims(axs, row_ylims)
    fig.align_titles()

    return fig, axs, significance
