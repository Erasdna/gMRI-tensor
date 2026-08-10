import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from gMRItensor.plotting.spatial_mode import make_colorbar
from gMRItensor.plotting.spatial_mode import plot_enhancement_with_background
from gMRItensor.plotting.subject_mode import _prepare_plotting_dataframe
from gMRItensor.plotting.subject_mode import make_subject_boxplot
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import scale_mode
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    base_width_per_col: float = 2.0,
    base_height_per_row: float = 2.0,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:

    # TODO: Verify the all inpute modes have same nb of components
    n_components = spatial_mode.shape[1]

    scaled_spatial_mode = scale_mode(spatial_mode)
    scaled_time_mode = scale_mode(time_mode)
    scaled_subject_mode = scale_mode(subject_mode)

    width_ratios = [1, 1, 1.05, 1.05]
    figsize = compute_figsize(
        n_components=n_components,
        n_columns=4,
        base_width_per_col=base_width_per_col,
        base_height_per_row=base_height_per_row,
        width_ratios=width_ratios,
    )
    fig, axs = plt.subplots(
        n_components,
        4,
        gridspec_kw={"width_ratios": width_ratios},
        figsize=figsize,
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
            divider = make_axes_locatable(ax)
            cax_divider = divider.append_axes("right", size="5%", pad=0.1)
            make_colorbar(fig, ax[-1], im, cax_divider, None, None)
    return fig, axs
