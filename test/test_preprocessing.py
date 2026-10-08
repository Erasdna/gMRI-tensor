import warnings
from collections import Counter
from operator import itemgetter

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from gMRItensor import preprocessing
from gMRItensor.preprocessing import _atomic_outputs
from gMRItensor.preprocessing import _iter_parallel
from gMRItensor.preprocessing import _load_labeled_tracer_voxels
from gMRItensor.preprocessing import compute_roi_scaling
from gMRItensor.preprocessing import compute_roi_statistics
from gMRItensor.preprocessing import compute_tracer_from_image
from gMRItensor.preprocessing import compute_tracer_parallel
from gMRItensor.preprocessing import iter_tracer_results
from gMRItensor.preprocessing import load_tensor_from_parquet
from gMRItensor.preprocessing import prepare_tensor
from gMRItensor.preprocessing import scale_tensor
from gMRItensor.preprocessing import tracer_to_concentration
from gMRItensor.preprocessing import write_preprocessed_data
from gMRItensor.preprocessing import write_tracer_parquet
from nibabel.orientations import axcodes2ornt
from nibabel.orientations import io_orientation
from nibabel.orientations import ornt_transform


def make_long_df(timepoints_per_subject, labels=(10, 20, 30), missing=None, seed=0):
    """Build a synthetic long-format tracer DataFrame.

    `timepoints_per_subject` maps subject -> list of observed time points.
    `missing` is an optional set of (subject, time_point, label) triples whose
    value should be NaN, simulating one region not being observed.
    `label_index` (as `compute_tracer_from_image` would produce it) is
    derived from each label's position in `labels`, consistently across
    every subject/time_point row.
    """
    rng = np.random.default_rng(seed)
    missing = missing or set()
    label_index_of = {label: i for i, label in enumerate(labels)}
    rows = []
    for subject, time_points in timepoints_per_subject.items():
        for t in time_points:
            for label in labels:
                value = rng.random()
                if (subject, t, label) in missing:
                    value = np.nan
                rows.append(
                    {
                        "subject": subject,
                        "time_point": t,
                        "labels": label,
                        "label_index": label_index_of[label],
                        "values": value,
                    },
                )
    return pd.DataFrame(rows)


def test_prepare_tensor_regular_pads_missing_timepoints_with_nan():
    # Regression test: prepare_tensor used to assume every subject has a row
    # for every time point and crash its rectangular reshape otherwise. It
    # should now NaN-pad the missing (subject, time_point) combination
    # instead, without dropping any label just because one subject has less
    # follow-up than the rest.
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    tensor, subjects, time_points, labels, label_index = prepare_tensor(df)

    assert list(subjects) == ["s1", "s2", "s3"]
    assert list(time_points) == [0, 1, 2]
    assert list(labels) == [10, 20, 30]
    assert list(label_index) == [0, 1, 2]
    assert tensor.shape == (3, 3, 3)
    # s3's time point 2 was never observed -> NaN, not dropped or crashed.
    assert np.isnan(tensor[2, 2]).all()
    assert not np.isnan(tensor[2, 0]).any()
    assert not np.isnan(tensor[0]).any()


def test_prepare_tensor_ragged_matches_regular_values():
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    tensor, subjects, time_points, labels, label_index = prepare_tensor(
        df,
        require_regular=True,
    )
    slices, subjects2, timepoints_per_subject, labels2, label_index2 = prepare_tensor(
        df,
        require_regular=False,
    )

    assert list(subjects) == list(subjects2)
    assert list(labels) == list(labels2)
    assert list(label_index) == list(label_index2)
    assert [s.shape for s in slices] == [(3, 3), (3, 3), (2, 3)]
    assert list(timepoints_per_subject[2]) == [0, 1]
    # The ragged slices contain exactly the non-NaN part of the regular array.
    np.testing.assert_allclose(slices[2], tensor[2, :2])


def test_prepare_tensor_drops_label_missing_anywhere():
    # A single missing value for one subject/time point still drops that
    # label everywhere -- the region/label mode must stay regular regardless
    # of require_regular, since a genuinely missing value can't be recovered.
    df = make_long_df(
        {"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1, 2]},
        missing={("s3", 1, 20)},
    )

    tensor, _, _, regular_labels, regular_label_index = prepare_tensor(
        df,
        require_regular=True,
    )
    slices, _, _, ragged_labels, ragged_label_index = prepare_tensor(
        df,
        require_regular=False,
    )

    assert list(regular_labels) == [10, 30]
    assert list(ragged_labels) == [10, 30]
    # label 20 (label_index 1) was dropped -- surviving label_index values
    # are not renumbered/compacted, they keep the original 0/2.
    assert list(regular_label_index) == [0, 2]
    assert list(ragged_label_index) == [0, 2]
    assert tensor.shape[2] == 2
    assert all(s.shape[1] == 2 for s in slices)


def test_prepare_tensor_min_timepoints_filtering():
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    slices, subjects, _, _, _ = prepare_tensor(
        df,
        require_regular=False,
        min_timepoints=3,
    )

    assert list(subjects) == ["s1", "s2"]
    assert len(slices) == 2


def test_scale_tensor_regular_and_ragged_agree():
    # Same underlying data, regular vs. ragged prepare_tensor output -- the
    # per-ROI mean/std pooled over subjects and time points should match, and
    # scaling should preserve the ragged shape/NaN-padding relationship the
    # same way prepare_tensor itself does.
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    tensor, _, _, _, _ = prepare_tensor(df, require_regular=True)
    slices, _, _, _, _ = prepare_tensor(df, require_regular=False)

    scaled_tensor, mean_t, std_t = scale_tensor(tensor, center=True)
    scaled_slices, mean_s, std_s = scale_tensor(slices, center=True)

    np.testing.assert_allclose(mean_t, mean_s)
    np.testing.assert_allclose(std_t, std_s)
    assert isinstance(scaled_slices, list)
    np.testing.assert_allclose(scaled_slices[2], scaled_tensor[2, :2])
    # Centered and scaled by its own pooled stats -> zero mean, unit std.
    pooled = scaled_tensor.reshape(-1, scaled_tensor.shape[-1])
    np.testing.assert_allclose(np.nanmean(pooled, axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(np.nanstd(pooled, axis=0), 1, atol=1e-10)


def test_scale_tensor_without_center_only_divides_by_std():
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2]})
    tensor, _, _, _, _ = prepare_tensor(df)

    scaled, mean, std = scale_tensor(tensor, center=False)

    np.testing.assert_allclose(mean, np.nanmean(tensor.reshape(-1, 3), axis=0))
    np.testing.assert_allclose(scaled, tensor / std)


def test_scale_tensor_applies_precomputed_mean_and_std():
    # Fitting scaling on one set of data and applying it to another (e.g.
    # held-out subjects) should use the passed-in mean/std, not recompute
    # them, and should still report back exactly what was passed in.
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2]})
    tensor, _, _, _, _ = prepare_tensor(df)
    fit_mean, fit_std = compute_roi_scaling(tensor)

    other = tensor + 5
    scaled, mean, std = scale_tensor(other, center=True, mean=fit_mean, std=fit_std)

    np.testing.assert_array_equal(mean, fit_mean)
    np.testing.assert_array_equal(std, fit_std)
    np.testing.assert_allclose(scaled, (other - fit_mean) / fit_std)


def test_scale_tensor_handles_zero_variance_roi():
    # A constant ROI has std 0; dividing by it should not produce NaN/inf --
    # the value is already equal to its own mean, so it should come out 0
    # after centering rather than blowing up.
    df = make_long_df({"s1": [0, 1], "s2": [0, 1]}, labels=(10,))
    df.loc[df["labels"] == 10, "values"] = 3.0

    tensor, _, _, _, _ = prepare_tensor(df)
    scaled, mean, std = scale_tensor(tensor, center=True)

    assert std[0] == 0
    assert np.all(np.isfinite(scaled))
    np.testing.assert_allclose(scaled, 0)


def _as_las(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient `img` to LAS storage, preserving the physical grid it encodes.

    Used to simulate an image written by different software with an
    equivalent-but-different axis-order/flip convention, without changing
    which physical location each voxel represents.
    """
    transform = ornt_transform(
        io_orientation(img.affine),
        axcodes2ornt(("L", "A", "S")),
    )
    return img.as_reoriented(transform)


def test_compute_tracer_from_image_allows_equivalent_orientation(tmp_path):
    # Regression test: baseline/post-injection/mask on the same physical
    # grid, but post-injection stored in a different (LAS) axis convention
    # than baseline's RAS+. This used to raise a false-positive "not
    # aligned" ValueError; canonicalizing before comparing affines should
    # now recover the same values as if both had been stored identically.
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    baseline_data = np.arange(27, dtype=float).reshape(3, 3, 3)
    post_injection_data = np.arange(27, 100, dtype=float)[:27].reshape(3, 3, 3)
    mask_data = np.ones((3, 3, 3))
    segmentation_data = np.ones((3, 3, 3))  # single ROI covering every voxel

    baseline_img = nib.Nifti1Image(baseline_data, affine)
    post_injection_img = _as_las(nib.Nifti1Image(post_injection_data, affine))
    mask_img = nib.Nifti1Image(mask_data, affine)
    segmentation_img = nib.Nifti1Image(segmentation_data, affine)

    baseline_path = tmp_path / "baseline.nii"
    post_injection_path = tmp_path / "post_injection.nii"
    mask_path = tmp_path / "mask.nii"
    segmentation_path = tmp_path / "segmentation.nii"
    nib.save(baseline_img, baseline_path)
    nib.save(post_injection_img, post_injection_path)
    nib.save(mask_img, mask_path)
    nib.save(segmentation_img, segmentation_path)

    _, tracer, _, _ = compute_tracer_from_image(
        baseline_path,
        post_injection_path,
        "R1map",
        mask_path,
        segmentation_path,
        func=None,
    )

    expected = (post_injection_data - baseline_data).ravel()
    np.testing.assert_allclose(tracer, expected)


def test_compute_tracer_from_image_rejects_different_grid(tmp_path):
    # Images that are genuinely on different grids (different voxel size)
    # must still be rejected after canonicalization -- reorientation only
    # normalizes axis order/flips, it doesn't resample.
    baseline_data = np.ones((3, 3, 3))
    post_injection_data = np.ones((3, 3, 3))
    mask_data = np.ones((3, 3, 3))
    segmentation_data = np.ones((3, 3, 3))

    baseline_img = nib.Nifti1Image(baseline_data, np.diag([2.0, 2.0, 2.0, 1.0]))
    post_injection_img = nib.Nifti1Image(
        post_injection_data,
        np.diag([3.0, 3.0, 3.0, 1.0]),
    )
    mask_img = nib.Nifti1Image(mask_data, np.diag([2.0, 2.0, 2.0, 1.0]))
    segmentation_img = nib.Nifti1Image(segmentation_data, np.diag([2.0, 2.0, 2.0, 1.0]))

    baseline_path = tmp_path / "baseline.nii"
    post_injection_path = tmp_path / "post_injection.nii"
    mask_path = tmp_path / "mask.nii"
    segmentation_path = tmp_path / "segmentation.nii"
    nib.save(baseline_img, baseline_path)
    nib.save(post_injection_img, post_injection_path)
    nib.save(mask_img, mask_path)
    nib.save(segmentation_img, segmentation_path)

    with pytest.raises(ValueError, match="not aligned"):
        compute_tracer_from_image(
            baseline_path,
            post_injection_path,
            "R1map",
            mask_path,
            segmentation_path,
        )


def _make_two_roi_images(tmp_path):
    """Build baseline/post-injection/mask/segmentation fixtures shared by the
    ROI-aggregate and per-voxel `compute_tracer_from_image` tests below.

    `(2, 2, 2)` volume, `baseline=0` everywhere so R1map tracer ==
    `post_injection` directly (trivial expected values); two real ROIs (`1`,
    `2`) plus background (`0`) voxels deliberately holding out-of-range
    values (`999`) so a filtering bug would be obvious.
    """
    affine = np.eye(4)
    shape = (2, 2, 2)
    baseline_data = np.zeros(shape)
    post_injection_flat = np.array([10, 20, 100, 200, 300, 999, 999, 30], dtype=float)
    post_injection_data = post_injection_flat.reshape(shape)
    mask_data = np.ones(shape)
    segmentation_flat = np.array([1, 1, 2, 2, 2, 0, 0, 1], dtype=float)
    segmentation_data = segmentation_flat.reshape(shape)

    baseline_path = tmp_path / "baseline.nii"
    post_injection_path = tmp_path / "post_injection.nii"
    mask_path = tmp_path / "mask.nii"
    segmentation_path = tmp_path / "segmentation.nii"
    nib.save(nib.Nifti1Image(baseline_data, affine), baseline_path)
    nib.save(nib.Nifti1Image(post_injection_data, affine), post_injection_path)
    nib.save(nib.Nifti1Image(mask_data, affine), mask_path)
    nib.save(nib.Nifti1Image(segmentation_data, affine), segmentation_path)

    return {
        "baseline_path": baseline_path,
        "post_injection_path": post_injection_path,
        "mask_path": mask_path,
        "segmentation_path": segmentation_path,
        "post_injection_flat": post_injection_flat,
        "segmentation_flat": segmentation_flat,
    }


def test_compute_tracer_from_image_aggregates_per_roi_and_excludes_background(tmp_path):
    # Two real ROIs (labels 1 and 2) plus unlabeled background (label 0)
    # voxels that must be excluded from aggregation entirely -- if they
    # leaked in, their deliberately out-of-range values would corrupt the
    # per-ROI median.
    paths = _make_two_roi_images(tmp_path)

    labels, values, label_index, index_list = compute_tracer_from_image(
        paths["baseline_path"],
        paths["post_injection_path"],
        "R1map",
        paths["mask_path"],
        paths["segmentation_path"],
    )

    np.testing.assert_array_equal(labels, [1, 2])
    np.testing.assert_allclose(values, [20.0, 200.0])
    # One row per ROI already -- label_index is always 0 here, so pivoting
    # on (labels, label_index) later is equivalent to pivoting on labels
    # alone, preserving tolerance for a subject missing an ROI.
    np.testing.assert_array_equal(label_index, [0, 0])
    assert len(index_list) == 2
    for i, label in enumerate(labels):
        expected_coords = np.argwhere(
            paths["segmentation_flat"].reshape(2, 2, 2) == label,
        )
        np.testing.assert_array_equal(
            sorted(index_list[i].tolist()),
            sorted(expected_coords.tolist()),
        )


def test_compute_tracer_from_image_per_voxel_matches_own_label_and_value(tmp_path):
    # func=None: one row per voxel instead of one per ROI, but background
    # voxels (label 0) must still be excluded, and each returned label must
    # correspond to that same voxel's own value, in order.
    paths = _make_two_roi_images(tmp_path)

    labels, values, label_index, index_list = compute_tracer_from_image(
        paths["baseline_path"],
        paths["post_injection_path"],
        "R1map",
        paths["mask_path"],
        paths["segmentation_path"],
        func=None,
    )

    assert labels.shape == values.shape == label_index.shape
    assert len(index_list) == len(labels)
    assert len(labels) == 6  # background voxels (label 0) excluded

    expected_mask = paths["segmentation_flat"] > 1e-6
    expected_labels = paths["segmentation_flat"][expected_mask]
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_allclose(values, paths["post_injection_flat"][expected_mask])

    # label_index is each voxel's rank among prior voxels of the same ROI
    # (0, 1, 2, ... resetting per ROI) -- computed independently here via a
    # running per-label counter, not by mirroring the implementation.
    counts: dict[float, int] = {}
    expected_label_index = []
    for label in expected_labels:
        expected_label_index.append(counts.get(label, 0))
        counts[label] = counts.get(label, 0) + 1
    np.testing.assert_array_equal(label_index, expected_label_index)

    # Regression test: per-voxel coordinates used to be a list of one (1, 3)
    # array per voxel (~6x the memory of a single array, pickled back from
    # every worker); now one (n_voxels, 3) array, as scatter_to_volume takes.
    assert isinstance(index_list, np.ndarray)
    assert index_list.shape == (6, 3)
    np.testing.assert_array_equal(
        index_list,
        np.argwhere(paths["segmentation_flat"].reshape(2, 2, 2) > 1e-6),
    )


def test_compute_tracer_parallel_returns_shared_index_list(tmp_path):
    paths = _make_two_roi_images(tmp_path)
    args = {
        "baseline_path": paths["baseline_path"],
        "post_injection_path": paths["post_injection_path"],
        "signal_type": "R1map",
        "mask_path": paths["mask_path"],
        "segmentation_path": paths["segmentation_path"],
        "func": np.nanmedian,
    }
    # Two "time points" of one subject, both reading the same images.
    args_list = [
        {**args, "subject": "s1", "time_point": 0},
        {**args, "subject": "s1", "time_point": 1},
    ]

    df, index_list = compute_tracer_parallel(args_list, n_procs=1)

    assert "label_index" in df.columns
    assert len(df) == 2 * 2  # 2 time points x 2 ROIs

    direct_labels, _, _, direct_index_list = compute_tracer_from_image(**args)
    assert len(index_list) == len(direct_labels)
    for got, expected in zip(index_list, direct_index_list):
        np.testing.assert_array_equal(got, expected)


def test_compute_tracer_parallel_tolerates_roi_missing_in_some_images(tmp_path):
    # Real per-subject/native-space segmentations can legitimately differ in
    # which ROIs they capture -- this must not be treated as an error, and
    # prepare_tensor's existing dropna step should tolerate it exactly as it
    # always did for aggregate-mode input (label_index is 0 for every
    # ROI-aggregate row, so pivoting on (labels, label_index) is equivalent
    # to pivoting on labels alone).
    paths = _make_two_roi_images(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    # A second segmentation missing ROI 2 entirely.
    other_segmentation = np.where(
        paths["segmentation_flat"].reshape(2, 2, 2) == 2,
        0,
        paths["segmentation_flat"].reshape(2, 2, 2),
    )
    other_segmentation_path = other_dir / "segmentation.nii"
    nib.save(nib.Nifti1Image(other_segmentation, np.eye(4)), other_segmentation_path)

    args = {
        "baseline_path": paths["baseline_path"],
        "post_injection_path": paths["post_injection_path"],
        "signal_type": "R1map",
        "mask_path": paths["mask_path"],
        "func": np.nanmedian,
    }
    args_list = [
        {
            **args,
            "segmentation_path": paths["segmentation_path"],
            "subject": "s1",
            "time_point": 0,
        },
        {
            **args,
            "segmentation_path": other_segmentation_path,
            "subject": "s1",
            "time_point": 1,
        },
    ]

    df, _ = compute_tracer_parallel(args_list, n_procs=1)
    tensor, _, _, labels, label_index = prepare_tensor(df)

    # ROI 2 is missing at time point 1 -> dropped everywhere; only ROI 1
    # (present at both time points) survives.
    assert list(labels) == [1]
    assert list(label_index) == [0]
    assert tensor.shape[-1] == 1


def test_compute_tracer_parallel_keeps_label_index_zero_when_segmentation_differs(
    tmp_path,
):
    # Invariant: every ROI-aggregate row gets the same label_index (0),
    # regardless of which ROI it is -- label_index only needs to
    # distinguish rows *within* one (subject, time_point, labels) group
    # (needed for per-voxel mode's repeated ROI ids), not to distinguish
    # one ROI from another. That's independent of whether segmentation_path
    # is shared across images or not (compute_tracer_parallel no longer
    # special-cases that); this test just uses two different segmentation
    # files (same ROI content, different paths) as one concrete case of it.
    paths = _make_two_roi_images(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_segmentation_path = other_dir / "segmentation.nii"
    nib.save(
        nib.Nifti1Image(paths["segmentation_flat"].reshape(2, 2, 2), np.eye(4)),
        other_segmentation_path,
    )

    args = {
        "baseline_path": paths["baseline_path"],
        "post_injection_path": paths["post_injection_path"],
        "signal_type": "R1map",
        "mask_path": paths["mask_path"],
        "func": np.nanmedian,
    }
    args_list = [
        {
            **args,
            "segmentation_path": paths["segmentation_path"],
            "subject": "s1",
            "time_point": 0,
        },
        {
            **args,
            "segmentation_path": other_segmentation_path,
            "subject": "s1",
            "time_point": 1,
        },
    ]

    df, _ = compute_tracer_parallel(args_list, n_procs=1)

    assert set(df["labels"]) == {1, 2}
    np.testing.assert_array_equal(df["label_index"].to_numpy(), 0)


def test_prepare_tensor_per_voxel_style_labels_are_not_collapsed():
    # Regression test for the bug this whole redesign fixes: two distinct
    # label_index values (0 and 1) sharing the same labels ROI id (as
    # func=None per-voxel output would produce for two voxels of one ROI)
    # must survive as two separate tensor columns, not be silently
    # collapsed into one by pivot_table's aggfunc="first".
    rows = []
    for subject in ("s1", "s2"):
        for t in (0, 1):
            rows.append(
                {
                    "subject": subject,
                    "time_point": t,
                    "labels": 7,
                    "label_index": 0,
                    "values": 1.0,
                },
            )
            rows.append(
                {
                    "subject": subject,
                    "time_point": t,
                    "labels": 7,
                    "label_index": 1,
                    "values": 2.0,
                },
            )
    df = pd.DataFrame(rows)

    tensor, _, _, labels, label_index = prepare_tensor(df)

    assert tensor.shape[-1] == 2
    np.testing.assert_array_equal(labels, [7, 7])
    np.testing.assert_array_equal(label_index, [0, 1])
    np.testing.assert_allclose(tensor[..., 0], 1.0)
    np.testing.assert_allclose(tensor[..., 1], 2.0)


def _make_streaming_args_list(tmp_path, func, n_images=4):
    """`args_list` over the two-ROI fixture, one distinct post-injection image
    per entry (scaled by `t + 1`), so out-of-order or mis-tagged results
    would show up as wrong values. Two subjects, alternating."""
    paths = _make_two_roi_images(tmp_path)
    args_list = []
    for t in range(n_images):
        post_injection_path = tmp_path / f"post_injection_{t}.nii"
        nib.save(
            nib.Nifti1Image(
                paths["post_injection_flat"].reshape(2, 2, 2) * (t + 1),
                np.eye(4),
            ),
            post_injection_path,
        )
        args_list.append(
            {
                "baseline_path": paths["baseline_path"],
                "post_injection_path": post_injection_path,
                "signal_type": "R1map",
                "mask_path": paths["mask_path"],
                "segmentation_path": paths["segmentation_path"],
                "func": func,
                "subject": f"s{t % 2}",
                "time_point": t // 2,
            },
        )
    return args_list


def _direct_kwargs(args):
    return {k: v for k, v in args.items() if k not in ("subject", "time_point")}


@pytest.mark.parametrize("n_procs", [1, 2])
@pytest.mark.parametrize("func", [np.nanmedian, None])
def test_iter_tracer_results_yields_in_order_matching_direct_calls(
    tmp_path,
    n_procs,
    func,
):
    # 5 images with n_procs=2 means a 4-wide in-flight window that has to be
    # refilled, exercising the bounded-submission path.
    args_list = _make_streaming_args_list(tmp_path, func, n_images=5)

    results = list(iter_tracer_results(args_list, n_procs=n_procs))

    assert len(results) == len(args_list)
    for result, args in zip(results, args_list):
        assert result.subject == args["subject"]
        assert result.time_point == args["time_point"]
        labels, values, label_index, index_list = compute_tracer_from_image(
            **_direct_kwargs(args),
        )
        np.testing.assert_array_equal(result.labels, labels)
        np.testing.assert_allclose(result.values, values)
        np.testing.assert_array_equal(result.label_index, label_index)
        if func is None:
            np.testing.assert_array_equal(result.index_list, index_list)


def test_iter_tracer_results_is_lazy(tmp_path):
    # A missing file in the second entry must only fail once that entry is
    # actually consumed -- proof that results are computed on demand.
    args_list = _make_streaming_args_list(tmp_path, None, n_images=2)
    args_list[1]["post_injection_path"] = tmp_path / "does_not_exist.nii"

    results = iter_tracer_results(args_list, n_procs=1)
    first = next(results)
    assert first.subject == "s0"
    with pytest.raises(FileNotFoundError):
        next(results)


def test_compute_tracer_parallel_per_voxel_returns_coordinate_array(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, None, n_images=2)

    df, index_list = compute_tracer_parallel(args_list, n_procs=1)

    assert isinstance(index_list, np.ndarray)
    assert index_list.shape == (6, 3)
    assert len(df) == 2 * 6


@pytest.mark.parametrize("func", [np.nanmedian, None])
def test_write_tracer_parquet_round_trips_to_same_tensor(tmp_path, func):
    args_list = _make_streaming_args_list(tmp_path, func)
    expected_df, expected_index_list = compute_tracer_parallel(args_list, n_procs=1)

    tracer_path, coords_path = write_tracer_parquet(
        args_list,
        tmp_path / "tracer.parquet",
        n_procs=1,
    )

    # One row group per image, so the file can later be read image by image.
    assert pq.ParquetFile(tracer_path).num_row_groups == len(args_list)

    df = pd.read_parquet(tracer_path)
    columns = ["subject", "time_point", "labels", "label_index", "values"]
    pd.testing.assert_frame_equal(
        df[columns],
        expected_df[columns],
        check_dtype=False,  # labels are int64 on disk, float from np.rint
    )

    expected = prepare_tensor(expected_df)
    got = prepare_tensor(df)
    for got_part, expected_part in zip(got, expected):
        np.testing.assert_array_equal(got_part, expected_part)

    if func is not None:
        # ROI mode: coordinates from one image need not describe another's
        # ROIs (native-space segmentations), so no sidecar is written.
        assert coords_path is None
        assert not (tmp_path / "tracer.coords.parquet").exists()
        return

    assert coords_path == tmp_path / "tracer.coords.parquet"
    coords = pd.read_parquet(coords_path)
    np.testing.assert_array_equal(
        coords[["i", "j", "k"]].to_numpy(),
        expected_index_list,
    )
    # Each coordinate row keeps the (labels, label_index) key of its tensor
    # column, so spatial modes can be mapped back onto voxels.
    segmentation = nib.load(args_list[0]["segmentation_path"]).get_fdata()
    np.testing.assert_array_equal(
        coords["labels"],
        segmentation[*expected_index_list.T],
    )
    np.testing.assert_array_equal(
        coords["label_index"],
        expected_df.query("subject == 's0' and time_point == 0")["label_index"],
    )


def test_write_tracer_parquet_leaves_no_tmp_files(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, None, n_images=2)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    write_tracer_parquet(args_list, out_dir / "tracer.parquet", n_procs=1)

    assert sorted(p.name for p in out_dir.iterdir()) == [
        "tracer.coords.parquet",
        "tracer.parquet",
    ]


def test_write_tracer_parquet_cleans_up_on_failure(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, None, n_images=2)
    args_list[1]["post_injection_path"] = tmp_path / "does_not_exist.nii"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        write_tracer_parquet(args_list, out_dir / "tracer.parquet", n_procs=1)

    assert list(out_dir.iterdir()) == []


def test_write_tracer_parquet_rejects_empty_args_list(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_tracer_parquet([], tmp_path / "tracer.parquet")


def _assert_matches_prepare_tensor(got, df, model, **kwargs):
    expected = prepare_tensor(df, require_regular=model == "cp", **kwargs)
    for got_part, expected_part in zip(got, expected):
        if isinstance(expected_part, list):
            assert len(got_part) == len(expected_part)
            for g, e in zip(got_part, expected_part):
                np.testing.assert_allclose(g, e, rtol=1e-6)
        elif expected_part.dtype.kind == "f":
            np.testing.assert_allclose(got_part, expected_part, rtol=1e-6)
        else:
            np.testing.assert_array_equal(got_part, expected_part)


def _write_long_df(df, path, row_group_size=None):
    """Write `df` with the given row-group layout (None: one group)."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


@pytest.mark.parametrize("model", ["cp", "parafac2"])
@pytest.mark.parametrize("func", [np.nanmedian, None])
def test_load_tensor_from_parquet_matches_prepare_tensor(tmp_path, model, func):
    args_list = _make_streaming_args_list(tmp_path, func)
    tracer_path, _ = write_tracer_parquet(
        args_list,
        tmp_path / "tracer.parquet",
        n_procs=1,
    )

    got = load_tensor_from_parquet(tracer_path, model)

    data = got[0]
    assert all(
        part.dtype == np.float32 for part in (data if model == "parafac2" else [data])
    )
    _assert_matches_prepare_tensor(got, pd.read_parquet(tracer_path), model)


@pytest.mark.parametrize("model", ["cp", "parafac2"])
@pytest.mark.parametrize("row_group_size", [None, 2, 5])
def test_load_tensor_from_parquet_is_row_group_layout_independent(
    tmp_path,
    model,
    row_group_size,
):
    # Regression test: images split across, or sharing, row groups must give
    # the same tensor as one row group per image. Shuffled so chunks of one
    # image arrive out of order and interleaved with other images.
    df = make_long_df({"a": [0, 1, 2], "b": [0, 2], "c": [1]})
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    path = _write_long_df(df, tmp_path / "tracer.parquet", row_group_size)

    got = load_tensor_from_parquet(path, model, min_timepoints=1, dtype=np.float64)

    _assert_matches_prepare_tensor(got, df, model)


@pytest.mark.parametrize("model", ["cp", "parafac2"])
def test_load_tensor_from_parquet_missing_timepoints_and_labels(tmp_path, model):
    df = make_long_df(
        {"a": [0, 1, 2], "b": [0, 2]},
        missing={("b", 2, 20)},
    )
    # Label 30 absent (not just NaN) from one observed image.
    df = df[~((df["subject"] == "a") & (df["time_point"] == 1) & (df["labels"] == 30))]
    path = _write_long_df(df, tmp_path / "tracer.parquet", row_group_size=3)

    got = load_tensor_from_parquet(path, model, min_timepoints=1)

    np.testing.assert_array_equal(got[3], [10])
    np.testing.assert_array_equal(got[4], [0])
    if model == "cp":
        assert got[0].shape == (2, 3, 1)
        assert np.isnan(got[0][1, 1]).all()
    else:
        assert [s.shape for s in got[0]] == [(3, 1), (2, 1)]
    _assert_matches_prepare_tensor(got, df, model)


@pytest.mark.parametrize("model", ["cp", "parafac2"])
def test_load_tensor_from_parquet_min_timepoints(tmp_path, model):
    df = make_long_df({"a": [0, 1, 2], "b": [0]})
    path = _write_long_df(df, tmp_path / "tracer.parquet", row_group_size=3)

    got = load_tensor_from_parquet(path, model, min_timepoints=2)

    np.testing.assert_array_equal(got[1], ["a"])
    _assert_matches_prepare_tensor(got, df, model, min_timepoints=2)


@pytest.mark.parametrize("model", ["cp", "parafac2"])
def test_load_tensor_from_parquet_group_filtering(tmp_path, model):
    df = make_long_df({"a": [0, 1], "b": [0, 1], "c": [0, 1]})
    df["group"] = df["subject"].map({"a": "x", "b": "y", "c": "x"})
    path = _write_long_df(df, tmp_path / "tracer.parquet", row_group_size=4)

    got = load_tensor_from_parquet(path, model, group_filtering=("group", "x"))

    np.testing.assert_array_equal(got[1], ["a", "c"])
    _assert_matches_prepare_tensor(got, df, model, group_filtering=("group", "x"))


def test_load_tensor_from_parquet_rejects_bad_arguments(tmp_path):
    path = _write_long_df(make_long_df({"a": [0]}), tmp_path / "tracer.parquet")

    with pytest.raises(ValueError, match="model"):
        load_tensor_from_parquet(path, "tucker")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not in the file"):
        load_tensor_from_parquet(path, "cp", group_filtering=("group", "x"))
    with pytest.raises(ValueError, match="No tracer rows"):
        load_tensor_from_parquet(path, "cp", group_filtering=("subject", "zzz"))


@pytest.mark.parametrize("model", ["cp", "parafac2"])
def test_load_tensor_from_parquet_drops_all_nan_session(tmp_path, model):
    # Regression test: one all-NaN session used to veto every column, giving
    # a tensor with 0 labels.
    labels = (10, 20, 30)
    df = make_long_df(
        {"a": [0, 1, 2], "b": [0, 1, 2], "c": [0, 1, 2]},
        labels=labels,
        missing={("b", 1, label) for label in labels},
    )
    path = _write_long_df(df, tmp_path / "tracer.parquet", row_group_size=3)

    got = load_tensor_from_parquet(path, model)

    # Default min_timepoints needs every time point, so "b" goes entirely.
    np.testing.assert_array_equal(got[1], ["a", "c"])
    np.testing.assert_array_equal(got[3], labels)
    _assert_matches_prepare_tensor(got, df[df["subject"] != "b"], model)


def test_load_tensor_from_parquet_cp_keeps_incomplete_subject_as_nan(tmp_path):
    labels = (10, 20, 30)
    df = make_long_df(
        {"a": [0, 1, 2], "b": [0, 1, 2]},
        labels=labels,
        missing={("b", 1, label) for label in labels},
    )
    path = _write_long_df(df, tmp_path / "tracer.parquet")

    tensor, subjects, _, got_labels, _ = load_tensor_from_parquet(
        path,
        "cp",
        min_timepoints=1,
    )

    np.testing.assert_array_equal(subjects, ["a", "b"])
    np.testing.assert_array_equal(got_labels, labels)
    assert np.isnan(tensor[1, 1]).all()
    assert np.isfinite(np.delete(tensor.reshape(-1, 3), 4, axis=0)).all()


@pytest.mark.parametrize("row_group_size", [None, 7])
def test_load_tensor_from_parquet_max_invalid_fraction(tmp_path, row_group_size):
    # Session ("a", 0) is 95% NaN: invalid at the 0.9 default, valid (and so
    # vetoing its NaN columns) at 1.0. Small row groups split sessions, so
    # the fraction must be accumulated across chunks.
    labels = tuple(range(1, 21))
    df = make_long_df(
        {"a": [0, 1], "b": [0, 1]},
        labels=labels,
        missing={("a", 0, label) for label in labels[:19]},
    )
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    path = _write_long_df(df, tmp_path / "tracer.parquet", row_group_size)

    _, subjects, _, got_labels, _ = load_tensor_from_parquet(path, "parafac2")
    np.testing.assert_array_equal(subjects, ["b"])
    np.testing.assert_array_equal(got_labels, labels)

    got = load_tensor_from_parquet(path, "parafac2", max_invalid_fraction=1.0)
    np.testing.assert_array_equal(got[1], ["a", "b"])
    np.testing.assert_array_equal(got[3], [20])
    _assert_matches_prepare_tensor(got, df, "parafac2")


def test_load_tensor_from_parquet_raises_without_surviving_data(tmp_path):
    # Each session is half NaN, on disjoint labels: all sessions valid, but no
    # column finite in all of them.
    df = make_long_df(
        {"a": [0, 1]},
        labels=(10, 20),
        missing={("a", 0, 10), ("a", 1, 20)},
    )
    path = _write_long_df(df, tmp_path / "tracer.parquet")
    with pytest.raises(ValueError, match="No column"):
        load_tensor_from_parquet(path, "cp")

    with pytest.raises(ValueError, match="No subject"):
        load_tensor_from_parquet(path, "cp", max_invalid_fraction=0.4)


def test_load_labeled_tracer_voxels_matches_per_voxel_tracer_and_voxel_volume(
    tmp_path,
):
    # The shared loader must give exactly what per-voxel
    # compute_tracer_from_image returns, plus the voxel volume from the
    # affine (2 x 3 x 4 mm voxels -> 24 mm^3), which ROI totals need.
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    paths = _make_two_roi_images(tmp_path)
    for key in ("baseline_path", "post_injection_path", "mask_path"):
        data = nib.load(paths[key]).get_fdata()
        nib.save(nib.Nifti1Image(data, affine), paths[key])
    nib.save(
        nib.Nifti1Image(paths["segmentation_flat"].reshape(2, 2, 2), affine),
        paths["segmentation_path"],
    )
    image_kwargs = {
        "baseline_path": paths["baseline_path"],
        "post_injection_path": paths["post_injection_path"],
        "signal_type": "R1map",
        "mask_path": paths["mask_path"],
        "segmentation_path": paths["segmentation_path"],
    }

    tracer, segmentation, voxel_coords, voxel_volume_mm3 = _load_labeled_tracer_voxels(
        **image_kwargs,
    )
    labels, values, _, index_list = compute_tracer_from_image(
        **image_kwargs,
        func=None,
    )

    np.testing.assert_allclose(tracer, values)
    np.testing.assert_array_equal(np.rint(segmentation), labels)
    np.testing.assert_array_equal(voxel_coords, index_list)
    assert voxel_volume_mm3 == pytest.approx(24.0)


@pytest.mark.parametrize("n_procs", [1, 2])
def test_iter_parallel_yields_worker_results_in_order(n_procs):
    # 7 entries with n_procs=2 exceeds the 4-wide in-flight window, so the
    # refill path runs. itemgetter is picklable, unlike a test-local lambda.
    args_list = [{"x": i} for i in range(7)]

    results = list(_iter_parallel(args_list, itemgetter("x"), n_procs, desc="test"))

    assert results == [(args, args["x"]) for args in args_list]


def test_atomic_outputs_moves_written_files_into_place(tmp_path):
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"

    with _atomic_outputs(first, second) as (tmp_first, tmp_second):
        tmp_first.write_text("a")
        # tmp_second deliberately never written (e.g. no coords sidecar).

    assert first.read_text() == "a"
    assert not second.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.parquet"]


def test_atomic_outputs_removes_tmp_files_on_failure(tmp_path):
    first = tmp_path / "a.parquet"

    with pytest.raises(RuntimeError):
        with _atomic_outputs(first) as (tmp_first,):
            tmp_first.write_text("partial")
            raise RuntimeError("boom")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("signal_type", ["R1map", "T1map"])
def test_tracer_to_concentration_converts_delta_r1_to_mm(signal_type):
    # 0.0032 1/ms = 3.2 1/s, which at r1 = 3.2 1/(mM s) is exactly 1 mM.
    tracer = np.array([0.0032, 0.0064, np.nan])

    concentration = tracer_to_concentration(tracer, signal_type)

    np.testing.assert_allclose(concentration, [1.0, 2.0, np.nan])
    np.testing.assert_allclose(
        tracer_to_concentration(tracer * 1000, signal_type, time_unit="s"),
        concentration,
    )
    np.testing.assert_allclose(
        tracer_to_concentration(tracer, signal_type, relaxivity=1.6),
        [2.0, 4.0, np.nan],
    )


def test_tracer_to_concentration_rejects_bad_arguments():
    with pytest.raises(ValueError, match="T1w"):
        tracer_to_concentration(np.ones(3), "T1w")
    with pytest.raises(ValueError, match="time_unit"):
        tracer_to_concentration(np.ones(3), "R1map", time_unit="min")


def _roi_rows(stats: pd.DataFrame) -> dict[str, pd.Series]:
    return {row["roi"]: row for _, row in stats.iterrows()}


def test_compute_roi_statistics_per_label_and_pooled_group():
    # Raw (unrounded) segmentation ids, as `_load_labeled_tracer_voxels`
    # returns them; label 4 has no finite voxel at all.
    tracer = np.array([1.0, 2.0, 3.0, 10.0, 20.0, np.nan, 5.0, np.nan])
    segmentation = np.array([1, 1, 1.0000001, 2, 2, 2, 3, 4])
    groups = {"g12": np.array([1, 2]), "absent": np.array([99])}

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no empty-slice warnings for label 4
        stats = compute_roi_statistics(tracer, segmentation, 2.0, groups)

    assert list(stats["roi"]) == ["1", "2", "3", "4", "g12"]
    assert list(stats["roi_type"]) == ["label"] * 4 + ["group"]
    rows = _roi_rows(stats)
    assert (rows["1"]["n_voxels"], rows["1"]["n_valid"]) == (3, 3)
    assert rows["1"]["median"] == pytest.approx(2.0)
    assert rows["1"]["volume_mm3"] == pytest.approx(6.0)
    assert (rows["2"]["n_voxels"], rows["2"]["n_valid"]) == (3, 2)
    assert rows["2"]["median"] == pytest.approx(15.0)
    assert rows["2"]["mean"] == pytest.approx(15.0)
    assert rows["4"]["n_valid"] == 0
    assert np.isnan(rows["4"]["median"]) and np.isnan(rows["4"]["mean"])
    # The group median pools voxels, it is not a median of label medians.
    assert (rows["g12"]["n_voxels"], rows["g12"]["n_valid"]) == (6, 5)
    assert rows["g12"]["median"] == pytest.approx(3.0)
    assert rows["g12"]["mean"] == pytest.approx(7.2)
    assert rows["g12"]["volume_mm3"] == pytest.approx(12.0)
    concentration_columns = ["median_concentration", "mean_concentration"]
    assert stats[concentration_columns + ["total_amount"]].isna().all().all()


def test_compute_roi_statistics_concentration_and_total_amount():
    tracer = np.array([1.0, 2.0, 3.0, 4.0])
    segmentation = np.array([1, 1, 1, 2])
    concentration = np.array([1.0, 2.0, np.nan, 4.0])  # mM

    stats = compute_roi_statistics(
        tracer,
        segmentation,
        24.0,
        {"both": np.array([1, 2])},
        concentration=concentration,
        include_labels=False,
    )

    assert list(stats["roi"]) == ["both"]
    row = stats.iloc[0]
    assert row["median_concentration"] == pytest.approx(2.0)
    assert row["mean_concentration"] == pytest.approx(7.0 / 3.0)
    # mM * mm^3 = 1e-6 mmol, summed over the finite voxels only.
    assert row["total_amount"] == pytest.approx(7.0 * 24.0 * 1e-6)


def _set_affine(args_list, affine):
    """Re-save every image of `args_list` on the same data with `affine`."""
    keys = ("baseline_path", "post_injection_path", "mask_path", "segmentation_path")
    for path in {args[key] for args in args_list for key in keys}:
        # Copy first: `get_fdata` may memory-map the file being overwritten.
        data = np.array(nib.load(path).get_fdata())
        nib.save(nib.Nifti1Image(data, affine), path)


@pytest.mark.parametrize("n_procs", [1, 2])
@pytest.mark.parametrize("func", [np.nanmedian, None])
def test_write_preprocessed_data_tracer_output_matches_write_tracer_parquet(
    tmp_path,
    func,
    n_procs,
):
    # Regression: the single-pass writer must give exactly the tracer files
    # `write_tracer_parquet` gives, so `load_tensor_from_parquet` is unchanged.
    args_list = _make_streaming_args_list(tmp_path, func)
    reference = tmp_path / "reference"
    reference.mkdir()
    expected_tracer, expected_coords = write_tracer_parquet(
        args_list,
        reference / "tracer.parquet",
        n_procs=1,
    )

    paths = write_preprocessed_data(args_list, tmp_path / "out", n_procs=n_procs)

    assert paths.tracer == tmp_path / "out" / "data" / "tracer.parquet"
    assert paths.roi_statistics == tmp_path / "out" / "data" / "roi_statistics.parquet"
    pd.testing.assert_frame_equal(
        pd.read_parquet(paths.tracer),
        pd.read_parquet(expected_tracer),
    )
    assert (
        pq.ParquetFile(paths.tracer).num_row_groups
        == pq.ParquetFile(expected_tracer).num_row_groups
    )
    if func is None:
        pd.testing.assert_frame_equal(
            pd.read_parquet(paths.coords),
            pd.read_parquet(expected_coords),
        )
    else:
        assert paths.coords is None and expected_coords is None
        assert not (tmp_path / "out" / "data" / "tracer.coords.parquet").exists()


def test_write_preprocessed_data_roi_statistics_round_trip(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, np.nanmedian)
    _set_affine(args_list, np.diag([2.0, 3.0, 4.0, 1.0]))  # 24 mm^3 voxels
    groups = {"both": np.array([1, 2])}

    paths = write_preprocessed_data(args_list, tmp_path, roi_groups=groups, n_procs=1)

    stats = pd.read_parquet(paths.roi_statistics)
    assert list(stats.columns[:2]) == ["subject", "time_point"]
    for args in args_list:
        got = stats.query(
            f"subject == '{args['subject']}' and time_point == {args['time_point']}",
        ).drop(columns=["subject", "time_point"])
        tracer, segmentation, _, voxel_volume_mm3 = _load_labeled_tracer_voxels(
            **{k: v for k, v in _direct_kwargs(args).items() if k != "func"},
        )
        expected = compute_roi_statistics(
            tracer,
            segmentation,
            voxel_volume_mm3,
            groups,
            concentration=tracer_to_concentration(tracer, "R1map"),
        )
        pd.testing.assert_frame_equal(got.reset_index(drop=True), expected)

    # First image: label 1 holds 10, 20, 30 [1/ms] -> 1e4 / 3.2 mM per unit.
    first = stats.query("subject == 's0' and time_point == 0 and roi == '1'").iloc[0]
    assert first["volume_mm3"] == pytest.approx(3 * 24.0)
    assert first["total_amount"] == pytest.approx(60.0 * 1000 / 3.2 * 24.0 * 1e-6)


@pytest.mark.parametrize(
    "kwargs, signal_type",
    [({}, "T1w"), ({"relaxivity": None}, "R1map")],
)
def test_write_preprocessed_data_without_concentration(tmp_path, kwargs, signal_type):
    args_list = _make_streaming_args_list(tmp_path, np.nanmedian, n_images=2)
    # Nonzero baseline, so the T1w ratio is finite.
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)),
        args_list[0]["baseline_path"],
    )
    for args in args_list:
        args["signal_type"] = signal_type

    paths = write_preprocessed_data(args_list, tmp_path, n_procs=1, **kwargs)

    stats = pd.read_parquet(paths.roi_statistics)
    assert stats["median"].notna().all()
    columns = ["median_concentration", "mean_concentration", "total_amount"]
    assert stats[columns].isna().all().all()


def test_write_preprocessed_data_loads_each_image_once(tmp_path, monkeypatch):
    args_list = _make_streaming_args_list(tmp_path, None)
    calls: Counter = Counter()
    original_loader = preprocessing._load_labeled_tracer_voxels

    def counting_loader(*args, **kwargs):
        calls[args[1]] += 1  # keyed by post_injection_path
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(preprocessing, "_load_labeled_tracer_voxels", counting_loader)

    write_preprocessed_data(args_list, tmp_path, n_procs=1)

    assert calls == Counter(args["post_injection_path"] for args in args_list)
    assert set(calls.values()) == {1}


def test_write_preprocessed_data_cleans_up_on_failure(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, None, n_images=2)
    args_list[1]["post_injection_path"] = tmp_path / "does_not_exist.nii"

    with pytest.raises(FileNotFoundError):
        write_preprocessed_data(args_list, tmp_path / "out", n_procs=1)

    assert list((tmp_path / "out" / "data").iterdir()) == []


def test_write_preprocessed_data_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_preprocessed_data([], tmp_path)
    args_list = _make_streaming_args_list(tmp_path, None, n_images=1)
    with pytest.raises(ValueError, match="time_unit"):
        write_preprocessed_data(args_list, tmp_path, time_unit="min")
