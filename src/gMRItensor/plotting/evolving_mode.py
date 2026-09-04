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


def plot_evolving_mode(
    evolving_factors: list[np.ndarray],
    timepoints_per_subject: list[np.ndarray],
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
    page_width: float = 7.0,
    width_to_height_ratio: float = 1.618,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot each subject's own PARAFAC2 evolving-mode (time) pattern.

    One row per component; every subject's own reconstructed time curve for
    that component is plotted against their own time points, colored by
    `group_variable`.

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

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and 2D axes array with shape `(n_components, 1)`.

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

    figsize = compute_figsize(
        n_components,
        1,
        page_width=page_width,
        width_to_height_ratio=width_to_height_ratio,
    )
    fig, axs = plt.subplots(
        n_components,
        1,
        figsize=figsize,
        layout="compressed",
        squeeze=False,
    )

    for component in range(n_components):
        ax = axs[component, 0]
        seen_groups: set = set()
        for factor, timepoints, group in zip(
            scaled_factors,
            timepoints_per_subject,
            groups,
        ):
            ax.plot(
                timepoints,
                factor[:, component],
                color=color_by_group[group],
                alpha=0.6,
                marker="o",
                markersize=3,
                label=group if group not in seen_groups else None,
            )
            seen_groups.add(group)

        ax.set_ylabel(rf"Component {component+1}")
        if component == 0:
            ax.set_title(f"Evolving mode v {group_variable}")
            ax.legend(frameon=True, framealpha=0.9)
        if component == n_components - 1:
            ax.set_xlabel("Time")

    return fig, axs
