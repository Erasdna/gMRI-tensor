import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from gMRItensor.preprocessing import compute_roi_scaling
from gMRItensor.preprocessing import compute_tracer_from_image
from gMRItensor.preprocessing import prepare_tensor
from gMRItensor.preprocessing import scale_tensor
from nibabel.orientations import axcodes2ornt
from nibabel.orientations import io_orientation
from nibabel.orientations import ornt_transform


def make_long_df(timepoints_per_subject, labels=(10, 20, 30), missing=None, seed=0):
    """Build a synthetic long-format tracer DataFrame.

    `timepoints_per_subject` maps subject -> list of observed time points.
    `missing` is an optional set of (subject, time_point, label) triples whose
    value should be NaN, simulating one region not being observed.
    """
    rng = np.random.default_rng(seed)
    missing = missing or set()
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

    tensor, subjects, time_points, labels = prepare_tensor(df)

    assert list(subjects) == ["s1", "s2", "s3"]
    assert list(time_points) == [0, 1, 2]
    assert list(labels) == [10, 20, 30]
    assert tensor.shape == (3, 3, 3)
    # s3's time point 2 was never observed -> NaN, not dropped or crashed.
    assert np.isnan(tensor[2, 2]).all()
    assert not np.isnan(tensor[2, 0]).any()
    assert not np.isnan(tensor[0]).any()


def test_prepare_tensor_ragged_matches_regular_values():
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    tensor, subjects, time_points, labels = prepare_tensor(df, require_regular=True)
    slices, subjects2, timepoints_per_subject, labels2 = prepare_tensor(
        df,
        require_regular=False,
    )

    assert list(subjects) == list(subjects2)
    assert list(labels) == list(labels2)
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

    tensor, _, _, regular_labels = prepare_tensor(df, require_regular=True)
    slices, _, _, ragged_labels = prepare_tensor(df, require_regular=False)

    assert list(regular_labels) == [10, 30]
    assert list(ragged_labels) == [10, 30]
    assert tensor.shape[2] == 2
    assert all(s.shape[1] == 2 for s in slices)


def test_prepare_tensor_min_timepoints_filtering():
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2], "s3": [0, 1]})

    slices, subjects, _, _ = prepare_tensor(
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

    tensor, _, _, _ = prepare_tensor(df, require_regular=True)
    slices, _, _, _ = prepare_tensor(df, require_regular=False)

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
    tensor, _, _, _ = prepare_tensor(df)

    scaled, mean, std = scale_tensor(tensor, center=False)

    np.testing.assert_allclose(mean, np.nanmean(tensor.reshape(-1, 3), axis=0))
    np.testing.assert_allclose(scaled, tensor / std)


def test_scale_tensor_applies_precomputed_mean_and_std():
    # Fitting scaling on one set of data and applying it to another (e.g.
    # held-out subjects) should use the passed-in mean/std, not recompute
    # them, and should still report back exactly what was passed in.
    df = make_long_df({"s1": [0, 1, 2], "s2": [0, 1, 2]})
    tensor, _, _, _ = prepare_tensor(df)
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

    tensor, _, _, _ = prepare_tensor(df)
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

    baseline_img = nib.Nifti1Image(baseline_data, affine)
    post_injection_img = _as_las(nib.Nifti1Image(post_injection_data, affine))
    mask_img = nib.Nifti1Image(mask_data, affine)

    baseline_path = tmp_path / "baseline.nii"
    post_injection_path = tmp_path / "post_injection.nii"
    mask_path = tmp_path / "mask.nii"
    nib.save(baseline_img, baseline_path)
    nib.save(post_injection_img, post_injection_path)
    nib.save(mask_img, mask_path)

    _, tracer = compute_tracer_from_image(
        baseline_path,
        post_injection_path,
        "R1map",
        mask_path,
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

    baseline_img = nib.Nifti1Image(baseline_data, np.diag([2.0, 2.0, 2.0, 1.0]))
    post_injection_img = nib.Nifti1Image(
        post_injection_data,
        np.diag([3.0, 3.0, 3.0, 1.0]),
    )
    mask_img = nib.Nifti1Image(mask_data, np.diag([2.0, 2.0, 2.0, 1.0]))

    baseline_path = tmp_path / "baseline.nii"
    post_injection_path = tmp_path / "post_injection.nii"
    mask_path = tmp_path / "mask.nii"
    nib.save(baseline_img, baseline_path)
    nib.save(post_injection_img, post_injection_path)
    nib.save(mask_img, mask_path)

    with pytest.raises(ValueError, match="not aligned"):
        compute_tracer_from_image(
            baseline_path,
            post_injection_path,
            "R1map",
            mask_path,
        )
