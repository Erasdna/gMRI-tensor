import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from gMRItensor.plotting.evolving_mode import plot_evolving_mode
from gMRItensor.plotting.spatial_mode import plot_spatial_mode
from gMRItensor.plotting.subject_mode import _prepare_plotting_dataframe
from gMRItensor.plotting.subject_mode import make_subject_boxplot
from gMRItensor.plotting.subject_mode import make_variable_correlation
from gMRItensor.plotting.subject_mode import plot_subject_mode
from gMRItensor.plotting.subject_mode import plot_subject_mode_correlation
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import create_colorbar_with_offset
from gMRItensor.plotting.utils import expand_roi_mode_to_voxels
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import merge_segmentations
from gMRItensor.plotting.utils import region_masks_from_segmentations
from gMRItensor.plotting.utils import scale_mode
from gMRItensor.plotting.utils import scatter_to_volume

matplotlib.use("Agg")


# TODO: gMRItensor.plotting.mode_grid still needs coverage; it requires a
# bigger synthetic-volumes-plus-subject-metadata fixture than fits here.


def make_subject_info(n_per_group: int = 6):
    n = 2 * n_per_group
    rng = np.random.default_rng(0)
    subjects = [f"sub-{i:02d}" for i in range(n)]
    return (
        pd.DataFrame(
            {
                "subjects": subjects,
                "group": ["A"] * n_per_group + ["B"] * n_per_group,
                "age": rng.normal(50, 10, n),
            },
        ),
        subjects,
    )


# --- gMRItensor.plotting.utils ---


def test_scale_mode():
    arr = np.array([[3.0, 0.0], [4.0, 5.0]])
    scaled = scale_mode(arr)
    norms = np.linalg.norm(scaled, axis=0)
    assert np.allclose(norms, 1.0)


def test_compute_figsize_basic():
    width, height = compute_figsize(
        n_components=2,
        n_columns=1,
        page_width=7.0,
        width_to_height_ratio=1.618,
        add_title_margin=False,
    )
    assert width == 7.0
    expected_height = (7.0 / 1) / 1.618 * 2
    assert height == pytest.approx(expected_height)


def test_compute_figsize_title_margin():
    _, height_no_margin = compute_figsize(
        n_components=1,
        n_columns=1,
        page_width=7.0,
        add_title_margin=False,
    )
    _, height_with_margin = compute_figsize(
        n_components=1,
        n_columns=1,
        page_width=7.0,
        add_title_margin=True,
    )
    assert height_with_margin == pytest.approx(height_no_margin * 1.1)


def test_get_color_palette_length():
    for n in [1, 2, 5]:
        colors = get_color_palette(n)
        assert len(colors) == n


def test_create_colorbar_with_offset():
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.rand(4, 4))
    cax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    offset = create_colorbar_with_offset(fig, ax, im, cax, label="value")
    assert isinstance(offset, float)
    plt.close(fig)


# --- gMRItensor.plotting.subject_mode: _prepare_plotting_dataframe ---


def test_prepare_plotting_dataframe_valid():
    subject_info, subjects = make_subject_info()
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects), 3))
    df = _prepare_plotting_dataframe(subject_mode, subjects, subject_info, "group")
    assert len(df) == len(subjects)
    assert {"comp_0", "comp_1", "comp_2", "group"}.issubset(df.columns)


def test_prepare_plotting_dataframe_row_mismatch():
    subject_info, subjects = make_subject_info()
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects) - 1, 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(subject_mode, subjects, subject_info, "group")


def test_prepare_plotting_dataframe_missing_group_column():
    subject_info, subjects = make_subject_info()
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects), 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(
            subject_mode,
            subjects,
            subject_info,
            "does_not_exist",
        )


def test_prepare_plotting_dataframe_missing_subjects_column():
    subject_info, subjects = make_subject_info()
    subject_info = subject_info.rename(columns={"subjects": "id"})
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects), 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(subject_mode, subjects, subject_info, "group")


def test_prepare_plotting_dataframe_empty_merge():
    subject_info, _ = make_subject_info()
    other_subjects = ["not-a-subject"] * len(subject_info)
    subject_mode = np.random.default_rng(0).normal(size=(len(other_subjects), 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(subject_mode, other_subjects, subject_info, "group")


def test_prepare_plotting_dataframe_single_group():
    subject_info, subjects = make_subject_info()
    subject_info = subject_info.copy()
    subject_info["group"] = "A"
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects), 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(subject_mode, subjects, subject_info, "group")


# --- gMRItensor.plotting.subject_mode: make_subject_boxplot / make_variable_correlation ---


def make_boxplot_df(n_per_group: int = 8):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "group": ["A"] * n_per_group + ["B"] * n_per_group,
            "value": np.concatenate(
                [rng.normal(0, 1, n_per_group), rng.normal(1, 1, n_per_group)],
            ),
        },
    )


def test_make_subject_boxplot_boxplot():
    df = make_boxplot_df()
    fig, ax = plt.subplots()
    ylim, required_xlim = make_subject_boxplot(
        ax,
        df,
        x_column="group",
        y_column="value",
        legend=True,
    )
    assert len(ylim) == 2
    assert required_xlim is not None
    plt.close(fig)


def test_make_subject_boxplot_no_legend():
    df = make_boxplot_df()
    fig, ax = plt.subplots()
    _, required_xlim = make_subject_boxplot(
        ax,
        df,
        x_column="group",
        y_column="value",
        legend=False,
    )
    assert required_xlim is None
    plt.close(fig)


def test_make_variable_correlation():
    rng = np.random.default_rng(2)
    n = 20
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": x * 2 + rng.normal(scale=0.1, size=n),
            "group": ["A"] * (n // 2) + ["B"] * (n // 2),
        },
    )
    fig, ax = plt.subplots()
    make_variable_correlation(ax, df, x_column="x", y_column="y", category="group")
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    plt.close(fig)


# --- gMRItensor.plotting.subject_mode: full figure smoke tests ---


def test_plot_subject_mode():
    subject_info, subjects = make_subject_info()
    n_components = 3
    subject_mode = np.random.default_rng(3).normal(size=(len(subjects), n_components))

    fig, axs = plot_subject_mode(
        subject_mode,
        subjects,
        subject_info,
        group_variable="group",
        plotting_variables=["age"],
    )
    assert axs.shape == (n_components, 2)
    plt.close(fig)


def test_plot_subject_mode_correlation():
    subject_info, subjects = make_subject_info()
    n_components = 2
    subject_mode = np.random.default_rng(4).normal(size=(len(subjects), n_components))

    fig, axs = plot_subject_mode_correlation(
        subject_mode,
        subjects,
        subject_info,
        group_variable="group",
    )
    assert axs.shape == (n_components, n_components)
    plt.close(fig)


def test_plot_subject_mode_single_component():
    # Regression test: matplotlib squeezes plt.subplots' axes array when any
    # grid dimension is 1, which used to break indexing like ax[0]/ax[i].
    subject_info, subjects = make_subject_info()
    n_components = 1
    subject_mode = np.random.default_rng(5).normal(size=(len(subjects), n_components))

    fig, axs = plot_subject_mode(
        subject_mode,
        subjects,
        subject_info,
        group_variable="group",
        plotting_variables=["age"],
    )
    assert axs.shape == (n_components, 2)
    plt.close(fig)


def test_plot_subject_mode_correlation_single_component():
    # Regression test for the same axes-squeeze hazard as above.
    subject_info, subjects = make_subject_info()
    n_components = 1
    subject_mode = np.random.default_rng(6).normal(size=(len(subjects), n_components))

    fig, axs = plot_subject_mode_correlation(
        subject_mode,
        subjects,
        subject_info,
        group_variable="group",
    )
    assert axs.shape == (n_components, n_components)
    plt.close(fig)


# --- gMRItensor.plotting.evolving_mode ---


def make_evolving_factors(n_subjects_per_group=3, n_components=2, seed=0):
    rng = np.random.default_rng(seed)
    subjects = [f"sub-{i:02d}" for i in range(2 * n_subjects_per_group)]
    subject_info = pd.DataFrame(
        {
            "subjects": subjects,
            "group": ["A"] * n_subjects_per_group + ["B"] * n_subjects_per_group,
        },
    )
    # Ragged: each subject gets a different number of time points.
    evolving_factors = []
    timepoints_per_subject = []
    for i in range(len(subjects)):
        n_timepoints = 4 + (i % 3)
        evolving_factors.append(rng.normal(size=(n_timepoints, n_components)))
        timepoints_per_subject.append(np.arange(n_timepoints))
    return evolving_factors, timepoints_per_subject, subjects, subject_info


def test_plot_evolving_mode():
    evolving_factors, timepoints, subjects, subject_info = make_evolving_factors()
    fig, axs = plot_evolving_mode(
        evolving_factors,
        timepoints,
        subjects,
        subject_info,
        group_variable="group",
    )
    assert axs.shape == (2, 1)
    plt.close(fig)


def test_plot_evolving_mode_single_component():
    # Regression test for the same axes-squeeze hazard exercised elsewhere in
    # this file (n_components == 1).
    evolving_factors, timepoints, subjects, subject_info = make_evolving_factors(
        n_components=1,
    )
    fig, axs = plot_evolving_mode(
        evolving_factors,
        timepoints,
        subjects,
        subject_info,
        group_variable="group",
    )
    assert axs.shape == (1, 1)
    plt.close(fig)


def test_plot_evolving_mode_length_mismatch():
    evolving_factors, timepoints, subjects, subject_info = make_evolving_factors()
    with pytest.raises(ValueError):
        plot_evolving_mode(
            evolving_factors,
            timepoints,
            subjects[:-1],
            subject_info,
            group_variable="group",
        )


def test_plot_evolving_mode_missing_subject():
    evolving_factors, timepoints, subjects, subject_info = make_evolving_factors()
    subject_info = subject_info[subject_info["subjects"] != subjects[0]]
    with pytest.raises(ValueError):
        plot_evolving_mode(
            evolving_factors,
            timepoints,
            subjects,
            subject_info,
            group_variable="group",
        )


# --- gMRItensor.plotting.utils: spatial helpers ---


def test_scatter_to_volume():
    index_list = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    values = np.array([1.0, 2.0, 3.0])
    volume, voxel_mask = scatter_to_volume(values, index_list, shape=(4, 4, 4))

    assert volume[0, 0, 0] == 1.0
    assert volume[1, 1, 1] == 2.0
    assert volume[2, 2, 2] == 3.0
    assert volume[3, 3, 3] == 0.0
    assert voxel_mask.sum() == 3
    assert voxel_mask[0, 0, 0] and not voxel_mask[3, 3, 3]


def test_scatter_to_volume_with_mask():
    index_list = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    values = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False, True])
    volume, voxel_mask = scatter_to_volume(
        values,
        index_list,
        shape=(4, 4, 4),
        mask=mask,
    )

    assert volume[0, 0, 0] == 1.0
    assert volume[1, 1, 1] == 0.0
    assert volume[2, 2, 2] == 3.0
    assert voxel_mask.sum() == 2
    assert not voxel_mask[1, 1, 1]


def make_segmentation(shape=(6, 6, 6)):
    """3 parenchyma ROIs (10, 20, 30) plus a CSF ROI (1000); ROI 30 is
    deliberately left out of the decomposition (`rois`) below, to exercise
    the fill_value path.
    """
    tissue_seg = np.zeros(shape, dtype=int)
    tissue_seg[0:2] = 10
    tissue_seg[2:4] = 20
    tissue_seg[4:5] = 30
    csf_seg = np.zeros(shape, dtype=int)
    csf_seg[5:6] = 1000
    combined_seg = np.where(csf_seg > 0, csf_seg, tissue_seg)
    return tissue_seg, csf_seg, combined_seg


def test_merge_segmentations_default_offsets():
    tissue_seg, csf_seg, _ = make_segmentation()

    merged, segs_after, offsets = merge_segmentations(
        {"Parenchyma": tissue_seg, "CSF": csf_seg},
    )

    assert offsets == {"Parenchyma": 0, "CSF": 10000}
    # Unshifted first segmentation, offset second segmentation.
    assert np.array_equal(
        merged[tissue_seg == 10],
        np.full((tissue_seg == 10).sum(), 10),
    )
    assert np.array_equal(
        merged[csf_seg == 1000],
        np.full((csf_seg == 1000).sum(), 11000),
    )
    # No overrides -> segmentations pass through unchanged.
    assert np.array_equal(segs_after["Parenchyma"], tissue_seg)
    assert np.array_equal(segs_after["CSF"], csf_seg)


def test_merge_segmentations_overlap_precedence():
    # Later-listed segmentation wins where both are nonzero at the same voxel.
    a = np.array([[5]])
    b = np.array([[100]])

    merged, _, offsets = merge_segmentations({"A": a, "B": b})
    assert merged[0, 0] == 100 + offsets["B"]

    merged2, _, offsets2 = merge_segmentations({"B": b, "A": a})
    assert merged2[0, 0] == 5 + offsets2["A"]


def test_merge_segmentations_custom_offsets():
    tissue_seg, csf_seg, _ = make_segmentation()

    merged, _, offsets = merge_segmentations(
        {"Parenchyma": tissue_seg, "CSF": csf_seg},
        offsets={"Parenchyma": 0, "CSF": 5000},
    )
    assert offsets == {"Parenchyma": 0, "CSF": 5000}
    assert np.array_equal(
        merged[csf_seg == 1000],
        np.full((csf_seg == 1000).sum(), 6000),
    )


def test_merge_segmentations_label_overrides():
    # Move label 20 (a "ventricle") out of Parenchyma and into CSF before merging.
    tissue_seg, csf_seg, _ = make_segmentation()
    original_csf_count = (csf_seg > 0).sum()
    original_ventricle_count = (tissue_seg == 20).sum()

    merged, segs_after, offsets = merge_segmentations(
        {"Parenchyma": tissue_seg, "CSF": csf_seg},
        label_overrides={20: "CSF"},
    )

    # Moved out of Parenchyma...
    assert not np.any(segs_after["Parenchyma"] == 20)
    # ...and into CSF, keeping its original label id.
    assert np.array_equal(segs_after["CSF"] == 20, tissue_seg == 20)
    assert (
        segs_after["CSF"] > 0
    ).sum() == original_csf_count + original_ventricle_count
    # Merged output keeps the overridden voxels' bare label id -- they do
    # NOT get CSF's offset, since the decomposition's region ids were
    # computed with the original (unshifted) label.
    assert np.array_equal(
        merged[tissue_seg == 20],
        np.full(original_ventricle_count, 20),
    )
    assert offsets["CSF"] != 0  # sanity check the offset itself is nonzero


def test_merge_segmentations_invalid_override_target():
    tissue_seg, csf_seg, _ = make_segmentation()
    with pytest.raises(ValueError):
        merge_segmentations(
            {"Parenchyma": tissue_seg, "CSF": csf_seg},
            label_overrides={20: "DoesNotExist"},
        )


def test_merge_segmentations_matches_manual_ventricle_relabeling():
    # Regression test against the original hand-written pattern this utility
    # replaces -- including the manual "roi -= 10000" undo step the original
    # script needed downstream, since overridden voxels must NOT be offset.
    tissue_seg, csf_seg, _ = make_segmentation()
    ventricles = [20, 30]

    csf_manual = np.where(np.isin(tissue_seg, ventricles), tissue_seg, csf_seg)
    tissue_manual = np.where(np.isin(tissue_seg, ventricles), 0, tissue_seg)
    combined_manual = np.where(csf_manual > 0, csf_manual + 10000, tissue_manual)
    # Undo the offset for voxels that were moved in from Parenchyma.
    is_ventricle = np.isin(tissue_seg, ventricles)
    combined_manual = np.where(is_ventricle, tissue_seg, combined_manual)

    merged, segs_after, _ = merge_segmentations(
        {"Parenchyma": tissue_seg, "CSF": csf_seg},
        label_overrides={v: "CSF" for v in ventricles},
    )

    assert np.array_equal(merged, combined_manual)
    assert np.array_equal(segs_after["CSF"], csf_manual)
    assert np.array_equal(segs_after["Parenchyma"], tissue_manual)


def test_expand_roi_mode_to_voxels():
    tissue_seg, csf_seg, combined_seg = make_segmentation()
    rois = np.array([10, 20, 1000])
    rng = np.random.default_rng(0)
    roi_mode = rng.random((len(rois), 2))

    voxel_mode, index_list = expand_roi_mode_to_voxels(roi_mode, rois, combined_seg)

    assert len(index_list) == int((combined_seg > 0).sum())
    np.testing.assert_array_equal(index_list, np.argwhere(combined_seg > 0))

    voxel_region_ids = combined_seg[*index_list.T]
    # Voxels in ROI 10/20/1000 get their matching decomposition row.
    assert np.allclose(voxel_mode[voxel_region_ids == 10], roi_mode[0])
    assert np.allclose(voxel_mode[voxel_region_ids == 1000], roi_mode[2])
    # ROI 30 was never in `rois` -> fill_value (default 0.0).
    assert np.all(voxel_mode[voxel_region_ids == 30] == 0.0)


def test_expand_roi_mode_to_voxels_custom_fill_value():
    _, _, combined_seg = make_segmentation()
    rois = np.array([10, 20, 1000])
    roi_mode = np.ones((len(rois), 2))

    voxel_mode, index_list = expand_roi_mode_to_voxels(
        roi_mode,
        rois,
        combined_seg,
        fill_value=-1.0,
    )
    voxel_region_ids = combined_seg[*index_list.T]
    assert np.all(voxel_mode[voxel_region_ids == 30] == -1.0)


def test_region_masks_from_segmentations():
    tissue_seg, csf_seg, combined_seg = make_segmentation()
    index_list = np.argwhere(combined_seg > 0)

    masks = region_masks_from_segmentations(
        index_list,
        {"CSF": csf_seg, "Parenchyma": tissue_seg},
    )

    assert list(masks.keys()) == ["CSF", "Parenchyma"]
    assert masks["CSF"].sum() == int((csf_seg[*index_list.T] > 0).sum())
    assert masks["Parenchyma"].sum() == int((tissue_seg[*index_list.T] > 0).sum())
    # Every voxel belongs to exactly one of the two masks here.
    assert np.array_equal(
        masks["CSF"] | masks["Parenchyma"],
        np.ones(len(index_list), dtype=bool),
    )


def test_plot_spatial_mode_roi_broadcast():
    # End-to-end: mirrors the real ROI-decomposition -> voxel plotting
    # workflow (expand_roi_mode_to_voxels -> region_masks_from_segmentations
    # -> plot_spatial_mode) that used to require a manual per-voxel loop.
    tissue_seg, csf_seg, combined_seg = make_segmentation()
    rois = np.array([10, 20, 1000])
    rng = np.random.default_rng(0)
    roi_mode = rng.random((len(rois), 2))
    background = rng.random(combined_seg.shape)

    voxel_mode, index_list = expand_roi_mode_to_voxels(roi_mode, rois, combined_seg)
    region_masks = region_masks_from_segmentations(
        index_list,
        {"CSF": csf_seg, "Parenchyma": tissue_seg},
    )

    results = list(
        plot_spatial_mode(
            voxel_mode,
            index_list,
            region_masks,
            background,
            slices=[2, 2, 2],
            page_width=5.0,
            width_to_height_ratio=1.0,
        ),
    )

    assert [name for _, _, name in results] == ["CSF", "Parenchyma"]
    for fig, axs, _ in results:
        assert axs.shape == (2, 4)
        plt.close(fig)
