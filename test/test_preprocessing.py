import json
import warnings
from collections import Counter
from collections.abc import Callable
from collections.abc import Iterator
from operator import itemgetter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from gMRItensor import preprocessing
from gMRItensor.preprocessing import _atomic_outputs
from gMRItensor.preprocessing import _coords_table
from gMRItensor.preprocessing import _iter_parallel
from gMRItensor.preprocessing import _load_labeled_tracer_voxels
from gMRItensor.preprocessing import _shell
from gMRItensor.preprocessing import _tracer_rows
from gMRItensor.preprocessing import _TRACER_SCHEMA
from gMRItensor.preprocessing import _tracer_table
from gMRItensor.preprocessing import compute_roi_scaling
from gMRItensor.preprocessing import compute_roi_signal
from gMRItensor.preprocessing import compute_tracer
from gMRItensor.preprocessing import delta_r1_per_second
from gMRItensor.preprocessing import invalidate_rois
from gMRItensor.preprocessing import load_tensor_from_parquet
from gMRItensor.preprocessing import load_tensor_with_exclusions
from gMRItensor.preprocessing import read_signal_metadata
from gMRItensor.preprocessing import scale_tensor
from gMRItensor.preprocessing import TracerResult
from gMRItensor.preprocessing import write_preprocessed_data
from nibabel.orientations import axcodes2ornt
from nibabel.orientations import io_orientation
from nibabel.orientations import ornt_transform


# --------------------------------------------------------------------------
# Reference implementations
#
# The straightforward in-memory pipeline that `load_tensor_from_parquet`
# streams: rows per image from the production loader (`tracer_rows`), a
# long frame (`tracer_frame`), a pivot (`prepare_tensor`). The streaming
# loader is checked against it.


def tracer_rows(
    baseline_path: Path,
    post_injection_path: Path,
    signal_type: str,
    mask_path: Path,
    segmentation_path: Path,
    func: Callable | None = np.nanmedian,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | np.ndarray]:
    """`(labels, values, label_index, index_list)` of one image: per ROI
    (aggregated by `func`) or, with `func=None`, per voxel."""
    tracer, segmentation, voxel_coords, _ = _load_labeled_tracer_voxels(
        baseline_path,
        post_injection_path,
        signal_type,
        mask_path,
        segmentation_path,
    )
    return _tracer_rows(tracer, segmentation, voxel_coords, func)


def _tracer_results(args_list: list[dict[str, Any]]) -> Iterator[TracerResult]:
    for args in args_list:
        image = {k: v for k, v in args.items() if k not in ("subject", "time_point")}
        yield TracerResult(args["subject"], args["time_point"], *tracer_rows(**image))


def tracer_frame(
    args_list: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[np.ndarray] | np.ndarray | None]:
    """Every image's `tracer_rows` in one long frame (`subject`,
    `time_point`, `labels`, `label_index`, `values`), plus the first
    image's `index_list`."""
    frames = []
    index_list = None
    for result in _tracer_results(args_list):
        if index_list is None:
            index_list = result.index_list
        frames.append(
            pd.DataFrame(
                {
                    "labels": result.labels,
                    "label_index": result.label_index,
                    "values": result.values,
                    "subject": result.subject,
                    "time_point": result.time_point,
                },
            ),
        )
    return pd.concat(frames, ignore_index=True), index_list


def write_tracer_rows(
    args_list: list[dict[str, Any]],
    output_path: Path,
) -> tuple[Path, Path | None]:
    """`tracer_frame` as the parquet `load_tensor_from_parquet` reads, one
    row group per image, plus the per-voxel `.coords.parquet` sidecar."""
    per_voxel = args_list[0]["func"] is None
    tracer_path = Path(output_path)
    coords_path = tracer_path.with_suffix(".coords.parquet")
    with pq.ParquetWriter(tracer_path, _TRACER_SCHEMA) as writer:
        for i, result in enumerate(_tracer_results(args_list)):
            if i == 0 and per_voxel:
                pq.write_table(_coords_table(result), coords_path)
            table = _tracer_table(result)
            writer.write_table(table, row_group_size=max(table.num_rows, 1))
    return tracer_path, coords_path if per_voxel else None


def _pivot_tracer_df(
    df: pd.DataFrame,
    group_filtering: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Filter by group and pivot the long-format tracer DataFrame.

    First step of `prepare_tensor`. Produces a frame indexed by (subject,
    time_point) with one column per `(labels, label_index)` pair; only
    observed (subject, time_point) combinations get a row.

    Pivots on the pair rather than `labels` alone because in per-voxel mode
    one ROI id spans many rows, where `aggfunc="first"` would silently keep
    one arbitrary voxel. `label_index` is always 0 in ROI-aggregate mode, so
    this stays equivalent to pivoting on `labels` there.
    """
    if group_filtering is not None:
        df = df.query(f"{group_filtering[0]}=='{group_filtering[1]}'")

    return df.pivot_table(
        index=["subject", "time_point"],
        columns=["labels", "label_index"],
        values="values",
        aggfunc="first",  # Safe: (labels, label_index) is unique per row.
    )


def prepare_tensor(
    df: pd.DataFrame,
    group_filtering: tuple[str, str] | None = None,
    require_regular: bool = True,
    min_timepoints: int = 1,
) -> tuple[Any, ...]:
    """Prepare subject x time x label tensor data from a long-format tracer DataFrame.

    Subjects need not share the same time points: only observed (subject,
    time_point) combinations decide which labels to keep, so one subject's
    missing scan does not drop a region for everyone. The label set must
    still be complete across all observed combinations, since a value
    missing at an observed time point cannot be recovered here.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format tracer data from `tracer_frame`, with
        `subject`, `time_point`, `labels`, `label_index`, `values` columns.
    group_filtering : tuple[str, str] | None, optional
        `(column, value)` to filter to a single group before pivoting.
    require_regular : bool, optional
        If True (default), returns a regular `(subjects, time_points,
        labels)` array where unobserved combinations become NaN rows. That
        array is **not** directly decomposable by `compute_CP_decomposition`
        -- impute or mask the NaNs first.

        If False, returns a ragged list of `(n_timepoints_i, n_labels)`
        slices using only each subject's observed time points. This is what
        `run_PARAFAC2_decomposition_repeated` expects.
    min_timepoints : int, optional
        Minimum observed time points to keep a subject; others are dropped
        and reported. Consider raising to 2 for PARAFAC2, so every subject's
        evolving factor has enough points to be meaningful.

    Returns
    -------
    If `require_regular`:
        `(tensor, subjects, time_points, labels, label_index)`, with
        `tensor` of shape `(subjects, time_points, labels)`, possibly
        containing NaN. `(labels[i], label_index[i])` identifies which
        `tracer_rows` row -- and so which voxel coordinates in
        `tracer_frame`'s `index_list` -- column `i` came from.
    Otherwise:
        `(slices, subjects, timepoints_per_subject, labels, label_index)`,
        with `slices[i]` ordered by `timepoints_per_subject[i]`.
    """
    pivot_df = _pivot_tracer_df(df, group_filtering)

    # Drop labels with a NaN among the *observed* rows only. A structurally
    # missing time point is not a row here, so it cannot force an otherwise
    # well-observed label to be dropped for everyone.
    valid_pivot = pivot_df.dropna(axis=1, how="any")
    # Columns are (labels, label_index) pairs -- see _pivot_tracer_df.
    columns = valid_pivot.columns.tolist()
    labels = np.array([c[0] for c in columns]).astype(int)
    label_index = np.array([c[1] for c in columns]).astype(int)

    subjects = []
    timepoints_per_subject = []
    slices = []
    dropped_subjects = []
    for subject, subject_df in valid_pivot.groupby(level="subject"):
        time_points = subject_df.index.get_level_values("time_point").to_numpy()
        order = np.argsort(time_points)
        if len(order) < min_timepoints:
            dropped_subjects.append((subject, len(order)))
            continue
        subjects.append(subject)
        timepoints_per_subject.append(time_points[order].astype(int))
        slices.append(subject_df.to_numpy()[order])

    if dropped_subjects:
        print(
            f"prepare_tensor: dropped {len(dropped_subjects)} subject(s) with "
            f"fewer than {min_timepoints} observed time points: {dropped_subjects}",
        )

    subjects_arr = np.array(subjects).astype(str)

    if not require_regular:
        return slices, subjects_arr, timepoints_per_subject, labels, label_index

    # Unobserved subject/time_point combinations become NaN rather than
    # being assumed to exist.
    all_timepoints = sorted({t for tps in timepoints_per_subject for t in tps})
    timepoint_index = {t: i for i, t in enumerate(all_timepoints)}
    tensor = np.full((len(subjects), len(all_timepoints), len(labels)), np.nan)
    for i, (subject_slice, time_points) in enumerate(
        zip(slices, timepoints_per_subject),
    ):
        for row, t in zip(subject_slice, time_points):
            tensor[i, timepoint_index[t]] = row

    return (
        tensor,
        subjects_arr,
        np.array(all_timepoints).astype(int),
        labels,
        label_index,
    )


def make_long_df(timepoints_per_subject, labels=(10, 20, 30), missing=None, seed=0):
    """Build a synthetic long-format tracer DataFrame.

    `timepoints_per_subject` maps subject -> list of observed time points.
    `missing` is an optional set of (subject, time_point, label) triples whose
    value should be NaN, simulating one region not being observed.
    `label_index` (as `tracer_rows` would produce it) is
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


def test_tracer_rows_allows_equivalent_orientation(tmp_path):
    # Regression test: baseline/post-injection/mask on the same physical
    # grid, but post-injection stored in a different (LAS) axis convention
    # than baseline's RAS+. This used to raise a false-positive "not
    # aligned" ValueError; canonicalizing before comparing affines should
    # now recover the same values as if both had been stored identically.
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    baseline_data = np.arange(1, 28, dtype=float).reshape(3, 3, 3)
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

    _, tracer, _, _ = tracer_rows(
        baseline_path,
        post_injection_path,
        "R1map",
        mask_path,
        segmentation_path,
        func=None,
    )

    expected = (post_injection_data - baseline_data).ravel()
    np.testing.assert_allclose(tracer, expected)


def test_tracer_rows_rejects_different_grid(tmp_path):
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
        tracer_rows(
            baseline_path,
            post_injection_path,
            "R1map",
            mask_path,
            segmentation_path,
        )


def _make_two_roi_images(tmp_path):
    """Build baseline/post-injection/mask/segmentation fixtures shared by the
    ROI-aggregate and per-voxel `tracer_rows` tests below.

    `(2, 2, 2)` volume, `baseline=1` everywhere and `post_injection` one
    more than `post_injection_flat`, so the R1map tracer is
    `post_injection_flat` (trivial expected values; R1 = 0 would mark an
    invalid voxel); two real ROIs (`1`,
    `2`) plus background (`0`) voxels deliberately holding out-of-range
    values (`999`) so a filtering bug would be obvious.
    """
    affine = np.eye(4)
    shape = (2, 2, 2)
    baseline_data = np.ones(shape)
    post_injection_flat = np.array([10, 20, 100, 200, 300, 999, 999, 30], dtype=float)
    post_injection_data = (post_injection_flat + 1).reshape(shape)
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


def test_tracer_rows_aggregates_per_roi_and_excludes_background(tmp_path):
    # Two real ROIs (labels 1 and 2) plus unlabeled background (label 0)
    # voxels that must be excluded from aggregation entirely -- if they
    # leaked in, their deliberately out-of-range values would corrupt the
    # per-ROI median.
    paths = _make_two_roi_images(tmp_path)

    labels, values, label_index, index_list = tracer_rows(
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


def test_tracer_rows_per_voxel_matches_own_label_and_value(tmp_path):
    # func=None: one row per voxel instead of one per ROI, but background
    # voxels (label 0) must still be excluded, and each returned label must
    # correspond to that same voxel's own value, in order.
    paths = _make_two_roi_images(tmp_path)

    labels, values, label_index, index_list = tracer_rows(
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


def test_tracer_frame_returns_shared_index_list(tmp_path):
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

    df, index_list = tracer_frame(args_list)

    assert "label_index" in df.columns
    assert len(df) == 2 * 2  # 2 time points x 2 ROIs

    direct_labels, _, _, direct_index_list = tracer_rows(**args)
    assert len(index_list) == len(direct_labels)
    for got, expected in zip(index_list, direct_index_list):
        np.testing.assert_array_equal(got, expected)


def test_tracer_frame_tolerates_roi_missing_in_some_images(tmp_path):
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

    df, _ = tracer_frame(args_list)
    tensor, _, _, labels, label_index = prepare_tensor(df)

    # ROI 2 is missing at time point 1 -> dropped everywhere; only ROI 1
    # (present at both time points) survives.
    assert list(labels) == [1]
    assert list(label_index) == [0]
    assert tensor.shape[-1] == 1


def test_tracer_frame_keeps_label_index_zero_when_segmentation_differs(
    tmp_path,
):
    # Invariant: every ROI-aggregate row gets the same label_index (0),
    # regardless of which ROI it is -- label_index only needs to
    # distinguish rows *within* one (subject, time_point, labels) group
    # (needed for per-voxel mode's repeated ROI ids), not to distinguish
    # one ROI from another, whether or not segmentation_path is shared
    # across images; this test uses two different segmentation files (same
    # ROI content, different paths) as one concrete case of it.
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

    df, _ = tracer_frame(args_list)

    assert set(df["labels"]) == {1, 2}
    np.testing.assert_array_equal(df["label_index"].to_numpy(), 0)


def test_prepare_tensor_per_voxel_style_labels_are_not_collapsed():
    # Two distinct label_index values (0 and 1) sharing the same labels ROI
    # id (as func=None per-voxel output would produce for two voxels of one
    # ROI) must survive as two separate tensor columns, not be silently
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


def test_tracer_frame_per_voxel_returns_coordinate_array(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, None, n_images=2)

    df, index_list = tracer_frame(args_list)

    assert isinstance(index_list, np.ndarray)
    assert index_list.shape == (6, 3)
    assert len(df) == 2 * 6


@pytest.mark.parametrize("func", [np.nanmedian, None])
def test_load_tensor_from_parquet_round_trips_the_reference_frame(tmp_path, func):
    args_list = _make_streaming_args_list(tmp_path, func)
    expected_df, expected_index_list = tracer_frame(args_list)

    tracer_path, coords_path = write_tracer_rows(
        args_list,
        tmp_path / "tracer.parquet",
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
    tracer_path, _ = write_tracer_rows(
        args_list,
        tmp_path / "tracer.parquet",
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
    # The shared loader must give exactly what per-voxel `tracer_rows`
    # returns, plus the voxel volume from the
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
    labels, values, _, index_list = tracer_rows(
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

    with pytest.raises(RuntimeError), _atomic_outputs(first) as (tmp_first,):
        tmp_first.write_text("partial")
        raise RuntimeError("boom")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("signal_type", ["T1map", "R1map"])
def test_delta_r1_per_second(signal_type):
    # A ΔR1 of 1e-4 1/ms is 0.1 1/s, whatever unit the maps were in.
    tracer = np.array([1e-4, np.nan])

    np.testing.assert_allclose(
        delta_r1_per_second(tracer, signal_type, "ms"),
        [0.1, np.nan],
    )
    np.testing.assert_allclose(
        delta_r1_per_second(tracer * 1000, signal_type, "s"),
        [0.1, np.nan],
    )


def test_delta_r1_per_second_leaves_t1w_ratio_and_rejects_bad_arguments():
    ratio = np.array([1.2, 0.9])

    np.testing.assert_array_equal(delta_r1_per_second(ratio, "T1w", "ms"), ratio)
    with pytest.raises(ValueError, match="time_unit"):
        delta_r1_per_second(ratio, "T1map", "min")
    with pytest.raises(ValueError, match="T2map"):
        delta_r1_per_second(ratio, "T2map", "ms")


def test_compute_roi_signal_per_label():
    # Raw (unrounded) segmentation ids; label 4 has no finite voxel.
    signal = np.array([1.0, 2.0, 3.0, 10.0, 20.0, np.nan, np.nan])
    segmentation = np.array([1, 1, 1.0000001, 2, 2, 2, 4])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no empty-slice warnings for label 4
        roi = compute_roi_signal(signal, segmentation, 2.0)

    assert list(roi.columns) == [
        "label",
        "median",
        "mean",
        "n_voxels",
        "n_valid",
        "n_rejected",
        "voxel_volume_mm3",
    ]
    assert (roi["n_rejected"] == 0).all()
    assert list(roi["label"]) == [1, 2, 4]
    rows = roi.set_index("label")
    assert (rows.loc[1, "median"], rows.loc[1, "mean"]) == (2.0, 2.0)
    assert (rows.loc[2, "n_voxels"], rows.loc[2, "n_valid"]) == (3, 2)
    assert rows.loc[2, "median"] == pytest.approx(15.0)
    assert rows.loc[4, "n_valid"] == 0 and np.isnan(rows.loc[4, "median"])
    assert (rows["voxel_volume_mm3"] == 2.0).all()


def _set_affine(args_list, affine):
    """Re-save every image of `args_list` on the same data with `affine`."""
    keys = ("baseline_path", "post_injection_path", "mask_path", "segmentation_path")
    for path in {args[key] for args in args_list for key in keys}:
        # Copy first: `get_fdata` may memory-map the file being overwritten.
        data = np.array(nib.load(path).get_fdata())
        nib.save(nib.Nifti1Image(data, affine), path)


def _scan_kwargs(args):
    keys = ("baseline_path", "post_injection_path", "signal_type")
    keys += ("mask_path", "segmentation_path")
    return {key: args[key] for key in keys}


def test_write_preprocessed_data_roi_signal(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, None)
    _set_affine(args_list, np.diag([2.0, 3.0, 4.0, 1.0]))  # 24 mm^3 voxels

    paths = write_preprocessed_data(args_list, tmp_path / "out", n_procs=1)

    assert paths.roi_signal == tmp_path / "out" / "data" / "roi_signal.parquet"
    assert paths.voxels is None and paths.coords is None
    roi = pd.read_parquet(paths.roi_signal)
    # The same table as CSV, for viewing.
    assert paths.roi_signal_csv == tmp_path / "out" / "data" / "roi_signal.csv"
    pd.testing.assert_frame_equal(
        pd.read_csv(paths.roi_signal_csv, dtype={"subject": str}),
        roi,
    )
    assert list(roi.columns[:2]) == ["subject", "time_point"]
    for args in args_list:
        got = roi.query(
            f"subject == '{args['subject']}' and time_point == {args['time_point']}",
        ).drop(columns=["subject", "time_point"])
        tracer, segmentation, _, volume = _load_labeled_tracer_voxels(
            **_scan_kwargs(args),
        )
        expected = compute_roi_signal(
            delta_r1_per_second(tracer, "R1map", "ms"),
            segmentation,
            volume,
        )
        pd.testing.assert_frame_equal(got.reset_index(drop=True), expected)
    assert read_signal_metadata(paths.roi_signal) == {
        "kind": "roi_signal",
        "input_type": "R1map",
        "signal": "delta_R1",
        "unit": "1/s",
    }


def test_write_preprocessed_data_t1w_keeps_the_ratio(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, None, n_images=2)
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)),
        args_list[0]["baseline_path"],
    )
    for args in args_list:
        args["signal_type"] = "T1w"

    paths = write_preprocessed_data(args_list, tmp_path, n_procs=1)

    roi = pd.read_parquet(paths.roi_signal)
    first = roi.query("subject == 's0' and time_point == 0").set_index("label")
    assert first.loc[1, "median"] == pytest.approx(20.0)  # post/baseline, unscaled
    assert read_signal_metadata(paths.roi_signal)["signal"] == "ratio"
    assert read_signal_metadata(paths.roi_signal)["unit"] == "1"


@pytest.mark.parametrize("n_procs", [1, 2])
def test_write_preprocessed_data_store_voxels(tmp_path, n_procs):
    args_list = _make_streaming_args_list(tmp_path, None)
    reference = tmp_path / "reference"
    reference.mkdir()
    expected_tracer, expected_coords = write_tracer_rows(
        args_list,
        reference / "tracer.parquet",
    )

    paths = write_preprocessed_data(
        args_list,
        tmp_path / "out",
        store_voxels=True,
        n_procs=n_procs,
    )

    # Same per-voxel rows as before, with ΔR1 converted from 1/ms to 1/s.
    expected = pd.read_parquet(expected_tracer)
    expected["values"] *= 1000.0
    pd.testing.assert_frame_equal(pd.read_parquet(paths.voxels), expected)
    pd.testing.assert_frame_equal(
        pd.read_parquet(paths.coords),
        pd.read_parquet(expected_coords),
    )
    assert read_signal_metadata(paths.voxels)["kind"] == "voxels"
    # Only the ROI table gets a CSV copy.
    assert [p.name for p in paths.voxels.parent.glob("*.csv")] == ["roi_signal.csv"]
    coords_metadata = read_signal_metadata(paths.coords)
    assert json.loads(coords_metadata["shape"]) == [2, 2, 2]
    np.testing.assert_array_equal(json.loads(coords_metadata["affine"]), np.eye(4))
    # Which file the template came from, for the record.
    assert coords_metadata["template"] == str(
        Path(args_list[0]["segmentation_path"]).resolve(),
    )
    tensor, *_ = load_tensor_from_parquet(paths.voxels, "cp")
    assert tensor.shape == (2, 2, 6)


def test_store_voxels_rejects_scans_on_different_templates(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, None)
    other = tmp_path / "other_segmentation.nii"
    segmentation = np.array([1, 1, 2, 2, 2, 0, 1, 1], dtype=float).reshape(2, 2, 2)
    nib.save(nib.Nifti1Image(segmentation, np.eye(4)), other)
    args_list[2]["segmentation_path"] = other

    with pytest.raises(ValueError, match="s0.*1.*template"):
        write_preprocessed_data(
            args_list,
            tmp_path / "out",
            store_voxels=True,
            n_procs=1,
        )
    assert list((tmp_path / "out" / "data").iterdir()) == []


def test_write_preprocessed_data_loads_each_image_once(tmp_path, monkeypatch):
    args_list = _make_streaming_args_list(tmp_path, None)
    calls: Counter = Counter()
    original_loader = preprocessing._load_labeled_tracer_voxels

    def counting_loader(*args, **kwargs):
        calls[args[1]] += 1  # keyed by post_injection_path
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(preprocessing, "_load_labeled_tracer_voxels", counting_loader)

    write_preprocessed_data(args_list, tmp_path, store_voxels=True, n_procs=1)

    assert calls == Counter(args["post_injection_path"] for args in args_list)
    assert set(calls.values()) == {1}


def test_write_preprocessed_data_cleans_up_on_failure(tmp_path):
    args_list = _make_streaming_args_list(tmp_path, None, n_images=2)
    args_list[1]["post_injection_path"] = tmp_path / "does_not_exist.nii"

    with pytest.raises(FileNotFoundError):
        write_preprocessed_data(
            args_list,
            tmp_path / "out",
            store_voxels=True,
            n_procs=1,
        )

    assert list((tmp_path / "out" / "data").iterdir()) == []


def test_write_preprocessed_data_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_preprocessed_data([], tmp_path)
    args_list = _make_streaming_args_list(tmp_path, None, n_images=2)
    with pytest.raises(ValueError, match="time_unit"):
        write_preprocessed_data(args_list, tmp_path, time_unit="min")
    args_list[1]["signal_type"] = "T1map"
    with pytest.raises(ValueError, match="signal_type"):
        write_preprocessed_data(args_list, tmp_path)


def test_load_tensor_from_parquet_impute_missing_keeps_nan_entries(tmp_path):
    # "b" misses time point 2 entirely; label 20 is NaN in one of "a"'s
    # scans, which would otherwise drop it for everyone; label 30 is never
    # finite, so it still goes.
    sessions = {"a": [0, 1, 2], "b": [0, 1]}
    never_finite = {(s, t, 30) for s, tps in sessions.items() for t in tps}
    df = make_long_df(sessions, missing={("a", 1, 20)} | never_finite)
    path = _write_long_df(df, tmp_path / "tracer.parquet")

    with pytest.warns(UserWarning, match=r"1 missing session.*\('b', 2\).*1 scattered"):
        tensor, subjects, timepoints, labels, _ = load_tensor_from_parquet(
            path,
            "cp",
            min_timepoints=2,
            impute_missing=True,
        )

    np.testing.assert_array_equal(subjects, ["a", "b"])
    np.testing.assert_array_equal(timepoints, [0, 1, 2])
    np.testing.assert_array_equal(labels, [10, 20])
    assert np.isnan(tensor[1, 2]).all()
    assert np.isnan(tensor[0, 1, 1])
    assert np.count_nonzero(np.isnan(tensor)) == 3


def test_load_tensor_from_parquet_impute_missing_is_cp_only(tmp_path):
    path = _write_long_df(make_long_df({"a": [0]}), tmp_path / "tracer.parquet")

    with pytest.raises(ValueError, match="impute_missing"):
        load_tensor_from_parquet(path, "parafac2", impute_missing=True)


def test_load_tensor_from_parquet_impute_missing_without_nan_is_silent(tmp_path):
    df = make_long_df({"a": [0, 1], "b": [0, 1]})
    path = _write_long_df(df, tmp_path / "tracer.parquet")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = load_tensor_from_parquet(path, "cp", impute_missing=True)

    _assert_matches_prepare_tensor(got, df, "cp")


def test_zero_in_an_r1_map_is_not_a_valid_voxel() -> None:
    """A 0 marks a failed fit in R1 maps as in T1 maps; it must not count as
    ΔR1 = -R1_baseline (or +R1_post)."""
    baseline = np.array([0.0, 0.5, 0.5])
    post = np.array([0.7, 0.0, 0.7])

    tracer = compute_tracer(baseline, post, "R1map")

    assert np.isnan(tracer[:2]).all()
    assert tracer[2] == pytest.approx(0.2)


@pytest.mark.parametrize("model", ["cp", "parafac2"])
def test_excluded_subjects_are_left_out_as_if_absent(tmp_path, model, capsys):
    # "c" alone has time point 3 and alone has label 30 NaN at 1: without it,
    # every time point is 0-2 and every label is kept.
    df = make_long_df(
        {"a": [0, 1, 2], "b": [0, 1, 2], "c": [0, 1, 2, 3]},
        missing={("c", 1, 30)},
    )
    path = _write_long_df(df, tmp_path / "tracer.parquet", row_group_size=4)

    got = load_tensor_from_parquet(path, model, exclude_subjects=("c",))

    assert len(got) == 5
    np.testing.assert_array_equal(got[1], ["a", "b"])
    np.testing.assert_array_equal(got[3], [10, 20, 30])
    _assert_matches_prepare_tensor(got, df[df["subject"] != "c"], model)
    assert "Rejected at command line: 1 subject(s): c" in capsys.readouterr().err


def test_excluded_subjects_reasons(tmp_path):
    labels = tuple(range(1, 21))
    df = make_long_df(
        {"a": [0, 1, 2], "b": [0, 2], "c": [0, 1, 2], "d": [0, 1]},
        labels=labels,
        # c's session 1 is 95% NaN: invalid at the 0.9 default.
        missing={("c", 1, label) for label in labels[:19]},
    )
    path = _write_long_df(df, tmp_path / "tracer.parquet")

    *tensor, exclusions = load_tensor_with_exclusions(
        path,
        "parafac2",
        exclude_subjects=("d",),
    )
    excluded = exclusions.subjects

    np.testing.assert_array_equal(tensor[1], ["a"])
    assert list(excluded.columns) == ["subject", "reason"]
    reasons = dict(zip(excluded["subject"], excluded["reason"]))
    assert list(reasons) == ["b", "c", "d"]
    assert reasons["b"] == (
        "too few valid time points (2 of 3; needs 3): time point 1 missing"
    )
    assert reasons["c"] == (
        "too few valid time points (2 of 3; needs 3): time point 1 has 95% "
        "invalid values (more than --max-invalid-fraction 0.9)"
    )
    # Short of sessions too, but only the command-line reason is given.
    assert reasons["d"] == "rejected at command line"


def test_excluded_subjects_empty_without_exclusions(tmp_path):
    path = _write_long_df(
        make_long_df({"a": [0, 1], "b": [0, 1]}),
        tmp_path / "tracer.parquet",
    )

    *_, exclusions = load_tensor_with_exclusions(path, "cp")

    assert exclusions.subjects.empty
    assert list(exclusions.subjects.columns) == ["subject", "reason"]
    assert exclusions.columns.empty
    assert list(exclusions.columns.columns) == [
        "label",
        "n_columns",
        "n_dropped",
        "scans",
    ]


def test_exclude_subjects_rejects_unknown_ids_and_excluding_everyone(tmp_path):
    path = _write_long_df(
        make_long_df({"a": [0, 1], "b": [0, 1]}),
        tmp_path / "tracer.parquet",
    )

    with pytest.raises(ValueError, match=r"not in the data: \['x', 'zz'\]"):
        load_tensor_from_parquet(path, "cp", exclude_subjects=("a", "zz", "x"))
    with pytest.raises(ValueError, match="every subject is excluded"):
        load_tensor_from_parquet(path, "cp", exclude_subjects=("a", "b"))


def test_compute_roi_signal_counts_rejected_voxels() -> None:
    segmentation = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    signal = np.array([1.0, np.nan, np.nan, 2.0, 1.0, np.nan, 3.0, 5.0])
    rejected = np.array([False, True, False, False, False, True, False, False])

    roi = compute_roi_signal(signal, segmentation, 1.0, rejected).set_index("label")

    assert roi["n_rejected"].tolist() == [1, 1]
    assert roi["n_valid"].tolist() == [2, 3]


def test_invalidate_rois_above_the_fraction_only() -> None:
    roi = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "median": [1.0, 2.0, 3.0],
            "mean": [1.0, 2.0, 3.0],
            "n_voxels": [4, 4, 4],
            "n_valid": [2, 1, 4],  # half invalid; three quarters; none
            "n_rejected": [1, 2, 0],
            "voxel_volume_mm3": [1.0, 1.0, 1.0],
        },
    )

    out, invalid = invalidate_rois(roi, 0.5)

    assert invalid.to_dict("records") == [
        {
            "label": 2,
            "n_voxels": 4,
            "n_nonfinite": 1,
            "n_rejected": 2,
            "invalid_share": 0.75,
        },
    ]
    assert np.isnan(out.loc[1, ["median", "mean"]].astype(float)).all()
    assert out["n_valid"].tolist() == [2, 0, 4]  # exactly 50% is kept
    assert invalidate_rois(roi, 1.0)[1].empty


def test_shell_is_measured_in_mm_outside_the_mask() -> None:
    mask = np.zeros((9, 9, 9))
    mask[4, 4, 4] = 1
    affine = np.diag([2.0, 1.0, 1.0, 1.0])  # voxels 2 mm along x

    shell = _shell(mask, affine, 2.0)

    assert not shell[4, 4, 4]  # inside the mask
    assert shell[5, 4, 4] and not shell[6, 4, 4]  # 2 mm, 4 mm along x
    assert shell[4, 6, 4] and not shell[4, 7, 4]  # 2 mm, 3 mm along y


def test_excluded_columns_name_the_scans_behind_each_dropped_label(tmp_path):
    # Label 20 is NaN in ("b", 2); label 30 is absent from ("a", 1).
    df = make_long_df({"a": [0, 1, 2], "b": [0, 1, 2]}, missing={("b", 2, 20)})
    df = df[~((df["subject"] == "a") & (df["time_point"] == 1) & (df["labels"] == 30))]
    path = _write_long_df(df, tmp_path / "tracer.parquet", row_group_size=3)

    *tensor, exclusions = load_tensor_with_exclusions(path, "parafac2")

    np.testing.assert_array_equal(tensor[3], [10])
    assert exclusions.columns.values.tolist() == [
        [20, 1, 1, "b t=2"],
        [30, 1, 1, "a t=1"],
    ]
    # CP keeps the columns finite somewhere and imputes the rest.
    *_, imputed = load_tensor_with_exclusions(path, "cp", impute_missing=True)
    assert imputed.columns.empty


def test_excluded_columns_count_voxels_and_shorten_long_scan_lists(tmp_path):
    subjects = {f"s{i:02d}": [0] for i in range(12)}
    # Two "voxels" of label 10 (label_index 0, 1); one is NaN in every scan.
    rows = [
        {"subject": s, "time_point": 0, "labels": 10, "label_index": i, "values": v}
        for s in subjects
        for i, v in ((0, 1.0), (1, np.nan))
    ]
    path = _write_long_df(pd.DataFrame(rows), tmp_path / "voxels.parquet")

    *_, exclusions = load_tensor_with_exclusions(path, "parafac2")

    row = exclusions.columns.iloc[0]
    assert (row["label"], row["n_columns"], row["n_dropped"]) == (10, 2, 1)
    assert row["scans"].startswith("s00 t=0; s01 t=0;")
    assert row["scans"].endswith("(+2 more)")
