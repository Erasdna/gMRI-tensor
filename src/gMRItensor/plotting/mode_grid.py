import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from gMRItensor.plotting.spatial_mode import plot_enhancement_with_background
from gMRItensor.plotting.subject_mode import _prepare_plotting_dataframe
from gMRItensor.plotting.subject_mode import make_subject_boxplot
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import create_colorbar_with_offset
from gMRItensor.plotting.utils import scale_mode
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
) -> tuple[matplotlib.figure.Figure, np.ndarray]:

    # TODO: Verify the all inpute modes have same nb of components
    n_components = spatial_mode.shape[1]

    scaled_spatial_mode = scale_mode(spatial_mode)
    scaled_time_mode = scale_mode(time_mode)
    scaled_subject_mode = scale_mode(subject_mode)

    # Calculate width ratios based on actual image aspect ratios
    # Get the shape of the sagittal slice
    sagittal_shape = background[sagittal_slice].shape  # (height, width)
    image_aspect_ratio = sagittal_shape[0] / sagittal_shape[1]

    # Create temporary figure to measure colorbar width
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

    # Calculate colorbar width fraction in axes coordinates
    # Colorbar is 5% width + text_width_offset + bbox_to_anchor offset (0.15)
    colorbar_width_fraction = 0.05 + text_width_offset + 0.15
    plt.close(temp_fig)

    # Time and subject plots are roughly square (aspect ratio ~1)
    # Spatial images need extra width for colorbar
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

        # Plot time mode
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

        # Plot subject mode
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

        # Plot spatial mode
        parenchyma_ax = axs[component, 2]
        csf_ax = axs[component, 3]

        for ax, ids_mask in zip(
            [parenchyma_ax, csf_ax],
            [parenchyma_index_mask, csf_index_mask],
        ):
            spatial_component = np.zeros(background.shape)
            # Create mask of assigned voxels
            voxel_mask = np.zeros(background.shape, dtype=bool)
            voxel_mask[*index_list[ids_mask].T] = True

            spatial_component[*index_list[ids_mask].T] = scaled_spatial_mode[
                ids_mask,
                component,
            ]

            im = plot_enhancement_with_background(
                ax,
                np.flip(np.rot90(background[sagittal_slice], 1), 1),
                np.flip(np.rot90(spatial_component[sagittal_slice], 1), 1),
                "plasma",
                vmin=np.percentile(spatial_component[spatial_component > 0], 5),
                vmax=np.percentile(spatial_component[spatial_component > 0], 95),
                mask=np.flip(np.rot90(voxel_mask[sagittal_slice], 1), 1),
            )
            # Create temporary colorbar to measure exponent text width
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

            # Remove temporary colorbar
            temp_cax.remove()
            # Create final colorbar with adjusted position
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
