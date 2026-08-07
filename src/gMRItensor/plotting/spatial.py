import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import scale_mode
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])


def plot_enhancement_with_background(ax, background, signal, cmap, vmin, vmax):
    ax.imshow(
        background,
        cmap="gray",
        interpolation="nearest",
        rasterized=True,
    )
    cax = ax.pcolormesh(
        np.ma.masked_values(signal, 0),
        cmap=cmap,
        alpha=0.995,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    return cax


def make_colorbar(fig, ax, cax, cax_divider, label, shrink):
    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    cbar = fig.colorbar(
        cax,
        cax=cax_divider,
        ax=ax,
        use_gridspec=True,
        shrink=shrink,
        format=formatter,
    )
    cbar.ax.yaxis.offsetText.set_visible(True)
    cbar.ax.yaxis.offsetText.set_horizontalalignment("right")
    cbar.set_label(label=label)


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
):
    print(vmin, vmax)

    plot_enhancement_with_background(
        ax[0],
        np.flip(np.rot90(background[slices[0]], 1), 1),
        np.flip(np.rot90(signal[slices[0]], 1), 1),
        cmap,
        vmin,
        vmax,
    )

    plot_enhancement_with_background(
        ax[1],
        np.rot90(background[:, slices[1]]),
        np.rot90(signal[:, slices[1]]),
        cmap,
        vmin,
        vmax,
    )

    im = plot_enhancement_with_background(
        ax[2],
        np.rot90(background[..., slices[2]], 1),
        np.rot90(signal[..., slices[2]], 1),
        cmap,
        vmin,
        vmax,
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
        0.1,
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
                vmin=np.percentile(scaled_spatial_mode[ids_mask, component], 5),
                vmax=np.percentile(scaled_spatial_mode[ids_mask, component], 95),
                label="Coefficient",
            )
        yield big_fig, big_ax, name
