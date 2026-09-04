"""Shared utility functions for plotting."""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["science", "no-latex"])


def scale_mode(arr: np.ndarray) -> np.ndarray:
    """Scale mode by L2 norm along first axis.

    Parameters
    ----------
    arr : np.ndarray
        Mode array to scale

    Returns
    -------
    np.ndarray
        Scaled mode array
    """

    return arr / np.linalg.norm(arr, axis=0)[None, :]


def compute_figsize(
    n_components: int,
    n_columns: int,
    page_width: float,
    width_ratios: list[float] | None = None,
    width_to_height_ratio: float = 1.618,
    add_title_margin=True,
) -> tuple[float, float]:
    """Compute appropriate figure size based on page width and aspect ratio.

    This function calculates a figure size that accounts for:
    - Target page width for journal submissions
    - Number of subplot rows (components) and columns
    - Custom width ratios for columns (e.g., for images with different aspect ratios)
    - Desired width-to-height ratio for individual subplots

    Parameters
    ----------
    n_components : int
        Number of components (rows in the subplot grid)
    n_columns : int
        Number of columns in the subplot grid
    page_width : float
        Target page width in inches (e.g., 3.5 for single column, 7.0 for double column)
    width_ratios : list[float] | None, optional
        Relative width ratios for each column. If provided, these ratios
        are used to scale the width of each column. For example, [1, 0.5, 0.5, 0.1]
        means the first column is full width, second and third are half width,
        and fourth (colorbar) is 10% width. By default None.
    width_to_height_ratio : float, optional
        Desired width-to-height ratio for each subplot (width/height).
        Default is 1.618 (golden ratio). By default 1.618.

    Returns
    -------
    tuple[float, float]
        Figure size as (width, height) in inches

    Notes
    -----
    Common journal page widths:
    - Single column: 3.5 inches
    - 1.5 column: 5.5 inches
    - Double column: 7.0 inches
    """
    # Use target page width directly
    width = page_width

    # Height per row should maintain the specified aspect ratio
    height_per_row = (width / n_columns) / width_to_height_ratio

    # Add extra space for titles, labels (10% per row)
    height = height_per_row * n_components * (1.1 if add_title_margin else 1.0)

    return (width, height)


def get_color_palette(n_colors: int) -> list:
    """Get color palette for plotting.

    Parameters
    ----------
    n_colors : int
        Number of colors needed

    Returns
    -------
    list
        List of colors from matplotlib's tab10 palette, cycling if
        n_colors exceeds 10.
    """
    return [f"C{i % 10}" for i in range(n_colors)]


def scatter_to_volume(
    values: np.ndarray,
    index_list: np.ndarray,
    shape: tuple[int, ...],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter per-voxel values onto a dense volume, at their (i, j, k) coordinates.

    Shared "place per-voxel values onto a dense 3D array" logic used by both
    `plot_spatial_mode` and `plot_mode_grid` to turn a flat, index-listed
    spatial mode into a brain-shaped volume for plotting.

    Parameters
    ----------
    values : np.ndarray
        Per-voxel values, shape `(len(index_list),)` (or `(len(index_list),
        mask.sum())` worth of rows if `mask` is given -- see below).
    index_list : np.ndarray
        `(n_voxels, ndim)` array of voxel coordinates, one row per entry in
        `values`.
    shape : tuple[int, ...]
        Shape of the dense volume to scatter onto (e.g. the background
        image's shape).
    mask : np.ndarray | None, optional
        Boolean array of shape `(len(index_list),)` selecting a subset of
        voxels to place (both `values` and `index_list` are indexed by it
        first). If None, every voxel in `index_list` is placed. By default
        None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(volume, voxel_mask)`: `volume` has `values` scattered at their
        coordinates (zero elsewhere), and `voxel_mask` is a same-shaped
        boolean array marking which voxels were actually placed.
    """
    if mask is not None:
        index_list = index_list[mask]
        values = values[mask]

    volume = np.zeros(shape)
    voxel_mask = np.zeros(shape, dtype=bool)
    volume[*index_list.T] = values
    voxel_mask[*index_list.T] = True
    return volume, voxel_mask


def merge_segmentations(
    segmentations: dict[str, np.ndarray],
    label_overrides: dict[int, str] | None = None,
    offsets: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Merge several named, integer-labeled segmentations into one.

    Combines segmentations such as a CSF and a parenchyma/tissue atlas into
    a single integer-labeled volume suitable for `expand_roi_mode_to_voxels`,
    offsetting each segmentation's labels so they never collide with another
    segmentation's labels in the merged output.

    Optionally reassigns specific label ids to a different named
    segmentation *before* merging -- e.g. moving ventricle labels out of a
    parenchyma segmentation into a CSF segmentation:
    `label_overrides={label: "CSF" for label in ventricle_ids}`.

    Where segmentations overlap at the same voxel (both nonzero), the
    later-listed segmentation in `segmentations` wins -- pass the
    lowest-priority segmentation first (e.g. `{"Parenchyma": ..., "CSF":
    ...}` if CSF should take precedence, matching a typical brainmask where
    CSF is checked last and overrides tissue).

    Parameters
    ----------
    segmentations : dict[str, np.ndarray]
        Named integer-labeled segmentation volumes, all the same shape
        (region id per voxel, 0 = background/not included). Overlapping,
        later entries take precedence -- see above.
    label_overrides : dict[int, str] | None, optional
        Maps a label id to the name of the segmentation it should belong to
        instead. Wherever that label id is found (nonzero) in any *other*
        named segmentation, those voxels are moved to the named target
        segmentation (zeroed out at their original location) before
        merging. By default None (no overrides).
    offsets : dict[str, int] | None, optional
        Per-segmentation label offset added to that segmentation's (possibly
        overridden) labels before merging, so labels originating from
        different named segmentations stay distinguishable in the merged
        output. If None, each segmentation is assigned a distinct offset of
        `10000 * i` in `segmentations`' order (so the first segmentation's
        labels are left unshifted). By default None.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]
        `(merged, segmentations_after_override, offsets)`:
        - `merged`: single integer-labeled volume combining all inputs,
          with `offsets` applied -- pass to `expand_roi_mode_to_voxels`.
        - `segmentations_after_override`: the per-name volumes *before*
          offsetting but *after* applying `label_overrides` -- pass to
          `region_masks_from_segmentations` for masks that reflect the
          override (e.g. so a moved ventricle voxel counts as "CSF").
        - `offsets`: the offset actually used for each name (useful to
          recover/verify the numbering used elsewhere, e.g. when the
          decomposition's region ids were built with the same convention).
    """
    segmentations = {name: seg.copy() for name, seg in segmentations.items()}

    if label_overrides:
        for label_id, target_name in label_overrides.items():
            if target_name not in segmentations.keys():
                raise ValueError(
                    f"Override target {target_name!r} is not in segmentations",
                )
            target_seg = segmentations[target_name]
            moved = np.zeros(target_seg.shape, dtype=bool)
            for name, seg in segmentations.items():
                if name == target_name:
                    continue
                source_mask = seg == label_id
                if source_mask.any():
                    seg[source_mask] = 0
                    moved |= source_mask
            target_seg[moved] = label_id

    if offsets is None:
        offsets = {name: 10000 * i for i, name in enumerate(segmentations)}

    shape = next(iter(segmentations.values())).shape
    merged = np.zeros(shape, dtype=int)
    for name, seg in segmentations.items():
        offset_seg = np.where(seg > 0, seg + offsets[name], 0)
        merged = np.where(offset_seg > 0, offset_seg, merged)

    return merged, segmentations, offsets


def expand_roi_mode_to_voxels(
    roi_mode: np.ndarray,
    roi_ids: np.ndarray,
    segmentation: np.ndarray,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast an ROI-level spatial mode out to every segmented voxel.

    For a decomposition computed on agglomerated ROIs (e.g. via
    `gMRItensor.preprocessing.prepare_tensor`, whose returned `labels` is
    exactly this function's `roi_ids`): every voxel in `segmentation` gets
    the decomposition row belonging to its region id, so the result can be
    plotted with `plot_spatial_mode` the same way a genuinely per-voxel
    decomposition would be. Vectorized via a dense id-to-row lookup table,
    rather than searching `roi_ids` once per voxel.

    Parameters
    ----------
    roi_mode : np.ndarray
        ROI-level spatial mode, shape `(len(roi_ids), n_components)`.
    roi_ids : np.ndarray
        Region id for each row of `roi_mode`.
    segmentation : np.ndarray
        Integer-labeled segmentation volume (region id per voxel, 0 =
        background/not included). Any combining of separate segmentations
        (e.g. merging tissue types into one label space) is expected to
        already be done by the caller.
    fill_value : float, optional
        Value assigned to voxels whose region id in `segmentation` isn't
        present in `roi_ids`. By default 0.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(voxel_mode, index_list)`: `voxel_mode` has shape `(n_voxels,
        n_components)` and `index_list` (shape `(n_voxels, ndim)`) are the
        coordinates of every voxel with `segmentation > 0`, ready to pass
        straight to `plot_spatial_mode`.
    """
    index_list = np.argwhere(segmentation > 0)
    voxel_region_ids = segmentation[*index_list.T].astype(int)
    roi_ids = roi_ids.astype(int)

    max_id = max(int(roi_ids.max()), int(voxel_region_ids.max()))
    lookup = np.full(max_id + 1, -1, dtype=int)
    lookup[roi_ids] = np.arange(len(roi_ids))
    row_indices = lookup[voxel_region_ids]

    voxel_mode = np.full((len(index_list), roi_mode.shape[1]), fill_value)
    found = row_indices >= 0
    voxel_mode[found] = roi_mode[row_indices[found]]

    return voxel_mode, index_list


def region_masks_from_segmentations(
    index_list: np.ndarray,
    **segmentations: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build named per-voxel boolean masks from named segmentation volumes.

    Replaces manually writing `segmentation[*index_list.T] > 0` once per
    named region, e.g.
    `region_masks_from_segmentations(index_list, CSF=csf_seg, Parenchyma=tissue_seg)`.

    Parameters
    ----------
    index_list : np.ndarray
        `(n_voxels, ndim)` array of voxel coordinates, as returned by
        `expand_roi_mode_to_voxels` or built directly (e.g.
        `np.argwhere(mask)`).
    **segmentations : np.ndarray
        Named segmentation volumes (same shape as the full image); a voxel
        belongs to a region if its value there is greater than zero.

    Returns
    -------
    dict[str, np.ndarray]
        One `(n_voxels,)` boolean mask per named segmentation, in the same
        order they were passed, ready for `plot_spatial_mode`'s
        `region_masks` argument.
    """
    return {
        name: segmentation[*index_list.T] > 0
        for name, segmentation in segmentations.items()
    }


def create_colorbar_with_offset(
    fig,
    ax,
    mappable,
    cax,
    label: str | None = None,
    format_string: str | None = None,
    precision: int = 1,
    horizontal_alignment="center",
) -> float:
    """Create a colorbar with scientific notation and return required offset.

    This function creates a colorbar with scientific notation formatting and
    calculates the horizontal offset needed to prevent the exponent text from
    overlapping with other plot elements.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the colorbar
    ax : matplotlib.axes.Axes
        Axes associated with the colorbar
    mappable : matplotlib.cm.ScalarMappable
        Mappable object (e.g., from imshow or pcolormesh)
    cax : matplotlib.axes.Axes
        Axes for the colorbar
    label : str | None, optional
        Label for the colorbar. By default None.
    format_string : str | None, optional
        Format string for colorbar tick labels (e.g., ':.2f', ':.1e').
        If provided, this overrides the precision parameter.
        By default None.
    precision : int, optional
        Number of decimal places to show in scientific notation.
        Only used when format_string is None. By default 2.

    Returns
    -------
    float
        Width offset in axes coordinates needed to accommodate the exponent text.
        This value represents half the text width and can be used to adjust
        the position of the colorbar to prevent overlap.
    """
    formatter: plt.matplotlib.ticker.Formatter
    if format_string is None:
        # Subclass ScalarFormatter to control precision
        class PrecisionScalarFormatter(plt.matplotlib.ticker.ScalarFormatter):
            def _set_format(self):
                # Override to set custom precision
                self.format = f"%.{precision}f"

        precision_formatter = PrecisionScalarFormatter(useMathText=True)
        precision_formatter.set_scientific(True)
        precision_formatter.set_powerlimits((0, 0))
        precision_formatter.set_useOffset(False)
        formatter = precision_formatter
    else:
        formatter = plt.matplotlib.ticker.StrMethodFormatter(format_string)

    cbar = fig.colorbar(
        mappable,
        cax=cax,
        ax=ax,
        use_gridspec=True,
        format=formatter,
    )
    # Center-align the exponent text above the colorbar
    cbar.ax.yaxis.offsetText.set_visible(True)
    cbar.ax.yaxis.offsetText.set_horizontalalignment(horizontal_alignment)
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
