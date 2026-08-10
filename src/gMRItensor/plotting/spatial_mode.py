import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import scale_mode
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])


def plot_enhancement_with_background(
    ax,
    background,
    signal,
    cmap,
    vmin,
    vmax,
    mask=None,
):
    ax.imshow(
        background,
        cmap="gray",
        interpolation="nearest",
        rasterized=True,
    )
    # If mask is provided, use it; otherwise mask zeros
    if mask is not None:
        masked_signal = np.ma.masked_where(~mask, signal)
    else:
        masked_signal = np.ma.masked_values(signal, 0)

    cax = ax.pcolormesh(
        masked_signal,
        cmap=cmap,
        alpha=0.995,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    return cax


def make_colorbar(fig, ax, cax, cax_divider, label):
    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    cbar = fig.colorbar(
        cax,
        cax=cax_divider,
        ax=ax,
        use_gridspec=True,
        format=formatter,
    )
    # Center-align the exponent text above the colorbar
    cbar.ax.yaxis.offsetText.set_visible(True)
    cbar.ax.yaxis.offsetText.set_horizontalalignment("center")
    cbar.set_label(label=label)

    # Force draw to get accurate text dimensions
    fig.canvas.draw()

    # Get the exponent text bounding box in display coordinates
    offset_text_bbox = cbar.ax.yaxis.offsetText.get_window_extent()

    # Transform to axes coordinates of the parent ax
    offset_text_bbox_ax = offset_text_bbox.transformed(ax.transAxes.inverted())

    # Calculate required left shift (half the text width in axes coordinates)
    text_width_ax = offset_text_bbox_ax.width

    return text_width_ax / 2.0


def plot_brain(
    fig,
    ax,
    signal,
    background,
    cmap,
    slices,
    vmin=0,
    vmax=2,
    label="Coefficient",
    mask=None,
):
    # Extract mask slices if mask is provided
    mask_0 = np.flip(np.rot90(mask[slices[0]], 1), 1) if mask is not None else None
    mask_1 = np.rot90(mask[:, slices[1]]) if mask is not None else None
    mask_2 = np.rot90(mask[..., slices[2]], 1) if mask is not None else None

    plot_enhancement_with_background(
        ax[0],
        np.flip(np.rot90(background[slices[0]], 1), 1),
        np.flip(np.rot90(signal[slices[0]], 1), 1),
        cmap,
        vmin,
        vmax,
        mask=mask_0,
    )

    plot_enhancement_with_background(
        ax[1],
        np.rot90(background[:, slices[1]]),
        np.rot90(signal[:, slices[1]]),
        cmap,
        vmin,
        vmax,
        mask=mask_1,
    )

    im = plot_enhancement_with_background(
        ax[2],
        np.rot90(background[..., slices[2]], 1),
        np.rot90(signal[..., slices[2]], 1),
        cmap,
        vmin,
        vmax,
        mask=mask_2,
    )

    ax[-1].set_axis_off()
    cax = inset_axes(ax[-1], width="100%", height="90%", loc="center")
    make_colorbar(fig, ax[-1], im, cax, label, None)

    for jj in range(len(ax)):
        ax[jj].set_xticks([])
        ax[jj].set_yticks([])


def plot_spatial_mode(
    spatial_mode: np.ndarray,
    index_list: np.ndarray,
    csf_index_mask: np.ndarray,
    parenchyma_index_mask: np.ndarray,
    background: np.ndarray,
    slices: list,
    figsize: tuple[float, float] | None = None,
    base_width_per_col: float = 2.0,
    base_height_per_row: float = 2.0,
):
    """Plot spatial mode components mapped onto brain slices.

    Parameters
    ----------
    spatial_mode : np.ndarray
        Spatial mode matrix
    index_list : np.ndarray
        Index mapping for spatial locations
    csf_index_mask : np.ndarray
        Mask for CSF regions
    parenchyma_index_mask : np.ndarray
        Mask for parenchyma regions
    background : np.ndarray
        Background image
    slices : list
        List of slice indices to plot
    figsize : tuple[float, float] | None, optional
        Figure size in inches. If None, automatically computed based on
        image dimensions and number of components.
    base_width_per_col : float, optional
        Base width per column in inches for automatic sizing, by default 2.0
    base_height_per_row : float, optional
        Base height per row in inches for automatic sizing, by default 2.0

    Yields
    ------
    tuple
        (figure, axes, name) for each tissue type (CSF, Parenchyma)
    """
    assert spatial_mode.shape[0] == index_list.shape[0]

    scaled_spatial_mode = scale_mode(spatial_mode)

    background_shape = background.shape
    width_ratios = [
        1,
        background_shape[0] / background_shape[1],
        background_shape[0] / background_shape[2],
        0.05,
    ]
    n_components = spatial_mode.shape[1]

    # Compute figsize if not provided
    if figsize is None:
        figsize = compute_figsize(
            n_components=n_components,
            n_columns=4,
            base_width_per_col=base_width_per_col,
            base_height_per_row=base_height_per_row,
            width_ratios=width_ratios,
        )
    for i, (name, ids_mask) in enumerate(
        zip(["CSF", "Parenchyma"], [csf_index_mask, parenchyma_index_mask]),
    ):
        big_fig, big_ax = plt.subplots(
            n_components,
            4,
            figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
            layout="tight",
        )

        for component in range(n_components):
            spatial_component = np.zeros(background_shape)
            # Create mask of assigned voxels
            voxel_mask = np.zeros(background_shape, dtype=bool)
            voxel_mask[*index_list[ids_mask].T] = True

            spatial_component[*index_list[ids_mask].T] = scaled_spatial_mode[
                ids_mask,
                component,
            ]
            plot_brain(
                big_fig,
                big_ax[component],
                spatial_component,
                background,
                "plasma",
                slices,
                vmin=np.percentile(spatial_component[spatial_component > 0], 5),
                vmax=np.percentile(spatial_component[spatial_component > 0], 95),
                label="Coefficient",
                mask=voxel_mask,
            )
        yield big_fig, big_ax, name
