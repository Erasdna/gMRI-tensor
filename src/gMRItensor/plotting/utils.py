"""Shared utility functions for plotting."""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["science", "no-latex"])


def scale_mode(arr: np.ndarray) -> np.ndarray:
    """Scale a mode's columns to unit L2 norm."""
    return arr / np.linalg.norm(arr, axis=0)[None, :]


def compute_figsize(
    n_components: int,
    n_columns: int,
    page_width: float,
    width_ratios: list[float] | None = None,
    width_to_height_ratio: float = 1.618,
    add_title_margin=True,
) -> tuple[float, float]:
    """Compute a `(width, height)` figure size in inches for a subplot grid.

    `page_width` is the target journal width: typically 3.5 (single column),
    5.5 (1.5 column) or 7.0 (double column). `width_to_height_ratio` applies
    per subplot and defaults to the golden ratio.
    """
    width = page_width
    height_per_row = (width / n_columns) / width_to_height_ratio
    # 10% extra per row for titles and labels.
    height = height_per_row * n_components * (1.1 if add_title_margin else 1.0)

    return (width, height)


def get_color_palette(n_colors: int) -> list:
    """Pick `n_colors` from matplotlib's tab10 palette, cycling past 10."""
    return [f"C{3*i % 10}" for i in range(n_colors)]


def scatter_to_volume(
    values: np.ndarray,
    index_list: np.ndarray,
    shape: tuple[int, ...],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter per-voxel `values` onto a dense volume of the given `shape`.

    Turns a flat, index-listed spatial mode into a brain-shaped volume for
    `plot_spatial_mode` and `plot_mode_grid`. `index_list` is `(n_voxels,
    ndim)` coordinates, one row per entry in `values`; `mask` optionally
    selects a subset of them.

    Returns `(volume, voxel_mask)`, the latter marking which voxels were
    placed (`volume` is zero elsewhere, which is indistinguishable from a
    genuine zero value).
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
    """Merge named, integer-labeled segmentations into one labeled volume.

    Combines e.g. a CSF and a parenchyma atlas into a single volume for
    `expand_roi_mode_to_voxels`, offsetting each segmentation's labels so
    they cannot collide. Where two segmentations overlap at a voxel, the
    later-listed one wins, so pass the lowest-priority one first.

    `label_overrides` reassigns specific label ids to a different named
    segmentation before merging -- e.g. moving ventricle labels out of a
    parenchyma atlas with `{label: "CSF" for label in ventricle_ids}`. Moved
    voxels keep their bare label id rather than taking the target's offset,
    since that is the id the decomposition's region ids were computed with.

    `offsets` defaults to `10000 * i` in `segmentations` order, leaving the
    first unshifted.

    Returns `(merged, segmentations_after_override, offsets)`. The second is
    pre-offset but post-override, ready for
    `region_masks_from_segmentations` so masks reflect the override.
    """
    segmentations = {name: seg.copy() for name, seg in segmentations.items()}
    shape = next(iter(segmentations.values())).shape
    overridden = np.zeros(shape, dtype=bool)

    if label_overrides:
        for label_id, target_name in label_overrides.items():
            if target_name not in segmentations:
                raise ValueError(
                    f"Override target {target_name!r} is not in segmentations",
                )
            target_seg = segmentations[target_name]
            moved = np.zeros(shape, dtype=bool)
            for name, seg in segmentations.items():
                if name == target_name:
                    continue
                source_mask = seg == label_id
                if source_mask.any():
                    seg[source_mask] = 0
                    moved |= source_mask
            target_seg[moved] = label_id
            overridden |= moved

    if offsets is None:
        offsets = {name: 10000 * i for i, name in enumerate(segmentations)}

    merged = np.zeros(shape, dtype=int)
    for name, seg in segmentations.items():
        # Overridden voxels keep their bare label id -- only a
        # segmentation's own native labels get its offset.
        seg_offset = np.where(overridden, 0, offsets[name])
        offset_seg = np.where(seg > 0, seg + seg_offset, 0)
        merged = np.where(offset_seg > 0, offset_seg, merged)

    return merged, segmentations, offsets


def expand_roi_mode_to_voxels(
    roi_mode: np.ndarray,
    roi_ids: np.ndarray,
    segmentation: np.ndarray,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast an ROI-level spatial mode out to every segmented voxel.

    For a decomposition computed on agglomerated ROIs (`roi_ids` is
    `prepare_tensor`'s returned `labels`), gives every voxel the row
    belonging to its region id, so the result plots like a genuinely
    per-voxel decomposition. Voxels whose region id is absent from `roi_ids`
    get `fill_value`.

    `segmentation` must already be a single label space; use
    `merge_segmentations` first if combining atlases.

    Returns `(voxel_mode, index_list)` for every voxel with
    `segmentation > 0`, ready for `plot_spatial_mode`.
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
    segmentations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build named per-voxel boolean masks from named segmentation volumes.

    A voxel belongs to a region where that segmentation is greater than
    zero. Takes the same dict shape `merge_segmentations` returns, so its
    `segmentations_after_override` can be passed straight through. Returns
    one `(n_voxels,)` mask per name, for `plot_spatial_mode`'s
    `region_masks`.
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
    """Create a colorbar in scientific notation and return its text offset.

    `format_string` (e.g. `':.2f'`, `':.1e'`) overrides `precision` when
    given. Returns half the exponent text's width in axes coordinates, which
    callers use to shift the colorbar so the exponent does not overlap
    neighbouring elements.
    """
    formatter: plt.matplotlib.ticker.Formatter
    if format_string is None:

        class PrecisionScalarFormatter(plt.matplotlib.ticker.ScalarFormatter):
            def _set_format(self):
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
    cbar.ax.yaxis.offsetText.set_visible(True)
    cbar.ax.yaxis.offsetText.set_horizontalalignment(horizontal_alignment)
    cbar.set_label(label=label)

    # Draw first, or the exponent text has no measurable extent yet.
    fig.canvas.draw()
    offset_text_bbox = cbar.ax.yaxis.offsetText.get_window_extent()
    offset_text_bbox_ax = offset_text_bbox.transformed(ax.transAxes.inverted())

    return offset_text_bbox_ax.width / 2.0
