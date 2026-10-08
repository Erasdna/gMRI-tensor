import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from gMRItensor.group_statistics import resolve_subject_groups
from gMRItensor.group_statistics import summarize_groups_over_time
from gMRItensor.plotting.evolving_mode import _build_long_evolving_dataframe
from gMRItensor.plotting.spatial_mode import _percentile_vlim
from gMRItensor.plotting.spatial_mode import plot_enhancement_with_background
from gMRItensor.plotting.subject_mode import _prepare_plotting_dataframe
from gMRItensor.plotting.subject_mode import make_subject_boxplot
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import create_colorbar_with_offset
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import plot_group_ribbons
from gMRItensor.plotting.utils import scale_mode
from gMRItensor.plotting.utils import scatter_to_volume
from gMRItensor.plotting.utils import SPATIAL_COLORBAR_LABEL
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])


def _plot_time_column_cp(
    ax: matplotlib.axes.Axes,
    scaled_time_mode: np.ndarray,
    time_points: list,
    component: int,
) -> None:
    """Draw one component of a shared (CP) time mode as a single line."""
    ax.plot(
        time_points,
        scaled_time_mode[:, component],
        color=f"C{component}",
        marker="o",
    )
    ax.set_xticks(time_points)
    ax.set_ylim(-0.1, 1.1 * np.max(scaled_time_mode))


def _plot_time_column_evolving(
    ax: matplotlib.axes.Axes,
    ribbon_stats_component: pd.DataFrame,
    categories: list[str],
    color_by_group: dict[str, str],
) -> None:
    """Draw one component of a PARAFAC2 evolving mode as per-group ribbons.

    Same mean +/- SEM ribbon as the left column of `plot_evolving_mode`,
    with the same 5% y-padding as `apply_row_ylims`.
    """
    ymin, ymax = plot_group_ribbons(
        ax,
        ribbon_stats_component,
        categories,
        color_by_group,
    )
    if np.isfinite(ymin) and np.isfinite(ymax):
        span = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin - 0.05 * span, ymax + 0.05 * span)
    ax.set_xticks(sorted(ribbon_stats_component["timepoint"].unique()))


def plot_mode_grid(
    spatial_mode: np.ndarray,
    time_mode: np.ndarray | list[np.ndarray],
    subject_mode: np.ndarray,
    index_list: np.ndarray,
    csf_index_mask: np.ndarray,
    parenchyma_index_mask: np.ndarray,
    background: np.ndarray,
    sagittal_slice: int,
    time_points: list,
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
    page_width: float = 7.0,
    share_colorbar_scaling: bool = False,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot a component-per-row grid: time, subject boxplot, and spatial mode.

    The time column accepts either decomposition's time mode:

    - CP: `time_mode` is one shared `(n_timepoints, rank)` array and
      `time_points` its timepoints; drawn as a single line per component.
    - PARAFAC2: `time_mode` is a list of per-subject `(n_timepoints_i, rank)`
      evolving factors (e.g. from `evolving_factors_to_numpy`) and
      `time_points` the matching list of per-subject timepoint arrays, both
      in `subjects` order. Drawn as a per-group mean +/- SEM ribbon, as in
      the left column of `plot_evolving_mode`, with group colors matching
      the subject boxplot.

    The Parenchyma/CSF columns use the same masking and percentile scaling
    as `plot_spatial_mode`. `share_colorbar_scaling` likewise mirrors that
    function's parameter: True pools both regions into one `vmin`/`vmax`
    per component, so color intensity is comparable between the two spatial
    columns; False (default) scales each column from its own values.
    """
    # TODO: Verify the all inpute modes have same nb of components
    n_components = spatial_mode.shape[1]

    scaled_spatial_mode = scale_mode(spatial_mode)
    scaled_subject_mode = scale_mode(subject_mode)

    is_evolving = isinstance(time_mode, list)
    if isinstance(time_mode, list):
        if not (len(time_mode) == len(time_points) == len(subjects)):
            raise ValueError(
                "For a PARAFAC2 evolving mode, time_mode, time_points and "
                f"subjects must have the same length, got {len(time_mode)}, "
                f"{len(time_points)}, {len(subjects)}",
            )
        groups = resolve_subject_groups(subjects, subject_info, group_variable)
        categories = sorted(set(groups))
        color_by_group = dict(zip(categories, get_color_palette(len(categories))))
        # Per-subject scaling, as in `plot_evolving_mode`.
        long_df = _build_long_evolving_dataframe(
            [scale_mode(np.asarray(factor)) for factor in time_mode],
            [np.asarray(timepoints) for timepoints in time_points],
            subjects,
            groups,
            n_components,
        )
        ribbon_stats = summarize_groups_over_time(long_df, facet="component")
    else:
        scaled_time_mode = scale_mode(time_mode)

    combined_index_mask = parenchyma_index_mask | csf_index_mask

    # Width ratios follow the real image aspect ratio.
    sagittal_shape = background[sagittal_slice].shape  # (height, width)
    image_aspect_ratio = sagittal_shape[0] / sagittal_shape[1]

    # Throwaway figure, only to measure the colorbar's width.
    temp_fig, temp_ax = plt.subplots(1, 1, figsize=(5, 5))
    # Deterministic dummy data: drawing from np.random here would consume
    # the caller's global RNG state.
    temp_data = np.linspace(0.0, 1.0, 100).reshape(10, 10)
    temp_im = temp_ax.imshow(temp_data)
    temp_cax = inset_axes(
        temp_ax,
        width="5%",
        height="80%",
        loc="center right",
        bbox_to_anchor=(0.15, 0.0, 1, 1),
        bbox_transform=temp_ax.transAxes,
        borderpad=0,
    )
    text_width_offset = create_colorbar_with_offset(
        temp_fig,
        temp_ax,
        temp_im,
        temp_cax,
        None,
    )

    # 5% width + text offset + the 0.25 bbox_to_anchor offset.
    colorbar_width_fraction = 0.05 + text_width_offset + 0.25
    plt.close(temp_fig)

    # Time/subject plots are ~square; spatial ones need colorbar room.
    image_with_colorbar_ratio = image_aspect_ratio * (1 + colorbar_width_fraction)

    width_ratios = [1, 1, image_with_colorbar_ratio, image_with_colorbar_ratio]

    figsize = compute_figsize(
        n_components=n_components,
        n_columns=4,
        page_width=page_width,
        width_ratios=width_ratios,
        width_to_height_ratio=image_with_colorbar_ratio
        / (sum(width_ratios) / len(width_ratios)),
    )
    fig, axs = plt.subplots(
        n_components,
        4,
        gridspec_kw={"width_ratios": width_ratios},
        figsize=figsize,
        layout="constrained",
        squeeze=False,
    )
    fig.set_layout_engine(
        "constrained",
        w_pad=0.02,  # Width padding between axes
        h_pad=0.02,  # Height padding between axes
    )

    df = _prepare_plotting_dataframe(
        scaled_subject_mode,
        subjects,
        subject_info,
        group_variable,
    )
    # Match boxplot colors to the ribbon's, in the boxplot's category order.
    boxplot_colors = (
        [color_by_group[c] for c in df[group_variable].unique()]
        if is_evolving
        else None
    )

    for component in range(n_components):

        time_ax = axs[component, 0]
        if is_evolving:
            _plot_time_column_evolving(
                time_ax,
                ribbon_stats.loc[ribbon_stats["component"] == component],
                categories,
                color_by_group,
            )
            if component == 0:
                time_ax.legend(frameon=True, framealpha=0.9)
        else:
            _plot_time_column_cp(time_ax, scaled_time_mode, time_points, component)
        time_ax.set_ylabel(f"Component {component+1}")
        if component == n_components - 1:
            time_ax.set_xlabel("Time after injection [h]")

        subject_ax = axs[component, 1]

        make_subject_boxplot(
            subject_ax,
            df,
            x_column=group_variable,
            y_column=f"comp_{component}",
            legend=False,
            colors=boxplot_colors,
        )

        subject_ax.set_ylabel("")
        if component == n_components - 1:
            subject_ax.set_xlabel(group_variable)
        else:
            subject_ax.set_xlabel("")

        parenchyma_ax = axs[component, 2]
        csf_ax = axs[component, 3]

        if share_colorbar_scaling:
            shared_vmin, shared_vmax = _percentile_vlim(
                scaled_spatial_mode[combined_index_mask, component],
            )

        for ax, ids_mask in zip(
            [parenchyma_ax, csf_ax],
            [parenchyma_index_mask, csf_index_mask],
        ):
            spatial_component, voxel_mask = scatter_to_volume(
                scaled_spatial_mode[:, component],
                index_list,
                background.shape,
                mask=ids_mask,
            )
            if share_colorbar_scaling:
                vmin, vmax = shared_vmin, shared_vmax
            else:
                vmin, vmax = _percentile_vlim(spatial_component)

            im = plot_enhancement_with_background(
                ax,
                np.flip(np.rot90(background[sagittal_slice], 1), 1),
                np.flip(np.rot90(spatial_component[sagittal_slice], 1), 1),
                "plasma",
                vmin=vmin,
                vmax=vmax,
                mask=np.flip(np.rot90(voxel_mask[sagittal_slice], 1), 1),
            )
            # Measure the exponent text, then reposition around it.
            temp_cax = inset_axes(
                ax,
                width="5%",
                height="80%",
                loc="center right",
                bbox_to_anchor=(0.15, 0.0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            text_width_offset = create_colorbar_with_offset(fig, ax, im, temp_cax, None)

            temp_cax.remove()
            cax_divider = inset_axes(
                ax,
                width="5%",
                height="80%",
                loc="center right",
                bbox_to_anchor=(text_width_offset, 0.0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            create_colorbar_with_offset(
                fig,
                ax,
                im,
                cax_divider,
                SPATIAL_COLORBAR_LABEL,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        if component == 0:
            for ax_obj, title in zip(
                axs[component],
                [
                    "Evolving mode" if is_evolving else "Time mode",
                    "Subject mode",
                    "Spatial mode (Parenchyma)",
                    "Spatial mode (CSF)",
                ],
            ):
                ax_obj.set_title(title)
            fig.align_titles()

    return fig, axs
