import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import create_colorbar_with_offset
from gMRItensor.plotting.utils import scale_mode
from gMRItensor.plotting.utils import scatter_to_volume
from gMRItensor.plotting.utils import SPATIAL_COLORBAR_LABEL
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

    cax = inset_axes(
        ax[-1],
        width="100%",
        height="70%",
        loc="center right",
        bbox_to_anchor=(0.05, 0.0, 1, 1),
        bbox_transform=ax[-1].transAxes,
        borderpad=0,
    )
    create_colorbar_with_offset(
        fig,
        ax[-1],
        im,
        cax,
        label,
        horizontal_alignment="left",
    )

    for jj in range(len(ax)):
        ax[jj].set_xticks([])
        ax[jj].set_yticks([])


def _percentile_vlim(
    values: np.ndarray,
    low: float = 5,
    high: float = 95,
) -> tuple[float, float]:
    """Percentile range of `values`' nonzero entries.

    Zero is excluded rather than clipped to: `scatter_to_volume` fills
    voxels outside the mask with exact 0, which is not a real value and
    would otherwise dominate the percentile. Genuine negative entries, e.g.
    from an unconstrained spatial mode, are kept.
    """
    nonzero = values[values != 0]
    return np.percentile(nonzero, low), np.percentile(nonzero, high)


def plot_spatial_mode(
    spatial_mode: np.ndarray,
    index_list: np.ndarray,
    region_masks: dict[str, np.ndarray],
    background: np.ndarray,
    slices: list,
    page_width: float = 7.0,
    width_to_height_ratio: float = 1.618,
    share_colorbar_scaling: bool = False,
):
    """Plot spatial mode components mapped onto brain slices.

    Works the same way whether `spatial_mode` was computed directly on
    voxels or on agglomerated ROIs and broadcast out to voxels via
    `gMRItensor.plotting.utils.expand_roi_mode_to_voxels`.

    Parameters
    ----------
    spatial_mode : np.ndarray
        One row per entry in `index_list`.
    index_list : np.ndarray
        `(n_voxels, ndim)` voxel coordinates, one row per `spatial_mode` row.
    region_masks : dict[str, np.ndarray]
        Named `(n_voxels,)` boolean masks splitting `index_list`'s rows into
        regions to plot separately, as built by
        `region_masks_from_segmentations`.
    background : np.ndarray
        Background image.
    slices : list
        Slice indices to plot.
    page_width, width_to_height_ratio : float, optional
        Figure sizing, see `compute_figsize`.
    share_colorbar_scaling : bool, optional
        If True, each component's colorbar range is computed once over the
        union of all `region_masks` and reused, making color intensity
        comparable across regions. If False (default), each region is scaled
        from its own values only -- not comparable, but uses each region's
        color range fully.

    Yields
    ------
    tuple
        `(figure, axes, name)` per region in `region_masks`.
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

    shared_vlims: list[tuple[float, float]] | None = None
    if share_colorbar_scaling:
        combined_mask = np.zeros(index_list.shape[0], dtype=bool)
        for ids_mask in region_masks.values():
            combined_mask |= ids_mask
        shared_vlims = [
            _percentile_vlim(scaled_spatial_mode[combined_mask, component])
            for component in range(n_components)
        ]

    figsize = compute_figsize(
        n_components=n_components,
        n_columns=sum(width_ratios),
        page_width=page_width,
        width_ratios=width_ratios[:-1],
        width_to_height_ratio=width_to_height_ratio,
    )
    for name, ids_mask in region_masks.items():
        big_fig, big_ax = plt.subplots(
            n_components,
            4,
            figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
            layout="compressed",
        )
        if n_components == 1:
            big_ax = [big_ax]

        for component in range(n_components):
            spatial_component, voxel_mask = scatter_to_volume(
                scaled_spatial_mode[:, component],
                index_list,
                background_shape,
                mask=ids_mask,
            )
            if shared_vlims is not None:
                vmin, vmax = shared_vlims[component]
            else:
                vmin, vmax = _percentile_vlim(spatial_component)
            plot_brain(
                big_fig,
                big_ax[component],
                spatial_component,
                background,
                "plasma",
                slices,
                vmin=vmin,
                vmax=vmax,
                label=SPATIAL_COLORBAR_LABEL,
                mask=voxel_mask,
            )
            big_ax[component][0].set_ylabel(f"Component {component+1}")

        yield big_fig, big_ax, name
