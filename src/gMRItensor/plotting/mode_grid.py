import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from gMRItensor.plotting.spatial_mode import _percentile_vlim
from gMRItensor.plotting.spatial_mode import plot_enhancement_with_background
from gMRItensor.plotting.subject_mode import _prepare_plotting_dataframe
from gMRItensor.plotting.subject_mode import make_subject_boxplot
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import create_colorbar_with_offset
from gMRItensor.plotting.utils import scale_mode
from gMRItensor.plotting.utils import scatter_to_volume
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])


def plot_mode_grid(
    spatial_mode: np.ndarray,
    time_mode: np.ndarray,
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

    The Parenchyma/CSF columns use the same masking and percentile scaling
    as `plot_spatial_mode`. `share_colorbar_scaling` likewise mirrors that
    function's parameter: True pools both regions into one `vmin`/`vmax`
    per component, so color intensity is comparable between the two spatial
    columns; False (default) scales each column from its own values.
    """
    # TODO: Verify the all inpute modes have same nb of components
    n_components = spatial_mode.shape[1]

    scaled_spatial_mode = scale_mode(spatial_mode)
    scaled_time_mode = scale_mode(time_mode)
    scaled_subject_mode = scale_mode(subject_mode)

    combined_index_mask = parenchyma_index_mask | csf_index_mask

    # Width ratios follow the real image aspect ratio.
    sagittal_shape = background[sagittal_slice].shape  # (height, width)
    image_aspect_ratio = sagittal_shape[0] / sagittal_shape[1]

    # Throwaway figure, only to measure the colorbar's width.
    temp_fig, temp_ax = plt.subplots(1, 1, figsize=(5, 5))
    temp_data = np.random.rand(10, 10)
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

    for component in range(n_components):

        time_ax = axs[component, 0]
        time_ax.plot(
            time_points,
            scaled_time_mode[:, component],
            color=f"C{component}",
            marker="o",
        )
        time_ax.set_ylabel(f"Component {component+1}")
        if component == n_components - 1:
            time_ax.set_xlabel("Time after injection [h]")
        time_ax.set_xticks(time_points)
        time_ax.set_ylim(-0.1, 1.1 * np.max(scaled_time_mode))

        subject_ax = axs[component, 1]

        make_subject_boxplot(
            subject_ax,
            df,
            x_column=group_variable,
            y_column=f"comp_{component}",
            legend=False,
        )

        subject_ax.set_ylabel("")
        if component == n_components - 1:
            subject_ax.set_xlabel("Patient group")
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
            create_colorbar_with_offset(fig, ax, im, cax_divider, None)
            ax.set_xticks([])
            ax.set_yticks([])

        if component == 0:
            for ax_obj, title in zip(
                axs[component],
                [
                    "Time mode",
                    "Subject mode",
                    "Spatial mode (Parenchyma)",
                    "Spatial mode (CSF)",
                ],
            ):
                ax_obj.set_title(title)
            fig.align_titles()

    return fig, axs
