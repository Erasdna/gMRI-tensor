from multiprocessing import Pool
from pathlib import Path
from typing import Callable
from typing import cast

import nibabel as nib
import numexpr as ne
import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from scipy.ndimage import labeled_comprehension
from tqdm.contrib import tenumerate


def compute_tracer(baseline: np.ndarray, post_injection: np.ndarray, signal_type: str):

    if signal_type == "T1map":
        expr = "where((abs(post_injection) < 1e-6) | (abs(baseline) < 1e-6), nan, (1 / post_injection) - (1 / baseline))"  # noqa: E501
    elif signal_type == "R1map":
        expr = "post_injection - baseline"  # noqa: E501
    elif signal_type == "T1w":
        expr = "where(abs(baseline) < 1e-6, nan, post_injection / baseline)"
    else:
        raise ValueError("Invalid argument value for 'signal_type'")

    return ne.evaluate(
        expr,
        local_dict={
            "baseline": baseline,
            "post_injection": post_injection,
            "nan": np.nan,
        },
    )


def compute_tracer_from_image(
    baseline_path: Path,
    post_injection_path: Path,
    signal_type: str,
    mask_path: Path,
    segmentation_path: Path,
    func: Callable | None = np.nanmedian,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Compute tracer signal per voxel or per ROI from a set of aligned NIfTI images.

    Loads `baseline_path`, `post_injection_path`, `mask_path`, and
    `segmentation_path`, reorienting each to the closest canonical (RAS+)
    axis convention so images stored with an equivalent-but-different axis
    order/flip aren't falsely rejected as misaligned (this does not
    resample, so genuinely different grids -- different voxel size, origin,
    or oblique rotation -- are still rejected). All four images must share
    the same affine.

    The tracer signal (see `compute_tracer`) is computed for voxels inside
    `mask` (`mask > 0`), then further restricted to voxels tagged with a
    real ROI in `segmentation` (segmentation value > 1e-6); untagged/
    background voxels (e.g. label 0) are always excluded, in both modes.

    Parameters
    ----------
    baseline_path, post_injection_path : Path
        Paths to the pre- and post-contrast-injection images.
    signal_type : str
        Passed to `compute_tracer` -- one of "T1map", "R1map", "T1w".
    mask_path : Path
        Path to a binary(-ish) mask image; voxels with mask > 0 are kept.
    segmentation_path : Path
        Path to an integer-valued ROI/label image, on the same grid as the
        other images. Every voxel kept in the output is tagged with its own
        ROI id from this image.
    func : Callable | None, optional
        Reduction function passed to `scipy.ndimage.labeled_comprehension`
        (e.g. `np.nanmedian`, the default, or `np.nanmean`) to aggregate the
        tracer signal to one value per ROI. If None, aggregation is skipped
        and one row per voxel is returned instead, each still tagged with
        its own ROI id.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]
        `(labels, values, label_index, index_list)`, all length-matched:

        - `labels[i]`: that row's ROI id (rounded to the nearest integer).
          If `func` is not None: the sorted unique ROI ids present. If
          `func` is None: the ROI id of the i-th kept voxel (repeats across
          voxels of the same ROI).
        - `values[i]`: `func` applied to that ROI's voxels, or (if `func`
          is None) that single voxel's own unaggregated tracer signal.
        - `label_index[i]`: `i` itself (`np.arange(len(labels))`) -- a
          plain dense row enumeration, always unique. Exists because
          `labels` values aren't necessarily contiguous/unique-per-row (in
          per-voxel mode many rows share a `labels` value), so this is the
          stable per-row key downstream code (`prepare_tensor`) pivots on
          instead of `labels`.
        - `index_list[i]`: `(n_i, ndim)` array of the voxel coordinate(s)
          behind that row -- every voxel in that ROI (`func` not None) or
          just that one voxel (`func` is None, `n_i == 1`).

    Raises
    ------
    ValueError
        If any two of baseline/post-injection/mask/segmentation are not on
        the same affine grid (after canonicalization).
    """
    baseline_nifti = cast(
        Nifti1Image,
        nib.as_closest_canonical(nib.load(baseline_path)),
    )
    post_injection_nifti = cast(
        Nifti1Image,
        nib.as_closest_canonical(nib.load(post_injection_path)),
    )
    mask_nifti = cast(Nifti1Image, nib.as_closest_canonical(nib.load(mask_path)))

    # Reorient every image to the closest canonical (RAS+) axis convention
    # first, so that two images which are actually on the same physical grid
    # but stored with a different (yet equivalent) axis order/flip don't fail
    # this check. It does not resample, so genuinely different grids
    # (different voxel size, origin, or oblique rotation) still fail here.
    if not np.allclose(baseline_nifti.affine, post_injection_nifti.affine):
        raise ValueError("Baseline and post-injection images are not aligned")
    if not np.allclose(baseline_nifti.affine, mask_nifti.affine):
        raise ValueError("Baseline and mask images are not aligned")

    segmentation_nifti = cast(
        Nifti1Image,
        nib.as_closest_canonical(nib.load(segmentation_path)),
    )
    if not np.allclose(baseline_nifti.affine, segmentation_nifti.affine):
        raise ValueError("Baseline and segmentation images are not aligned")

    mask = mask_nifti.get_fdata()
    tracer = compute_tracer(
        baseline_nifti.get_fdata()[mask > 0],
        post_injection_nifti.get_fdata()[mask > 0],
        signal_type,
    )
    segmentation = segmentation_nifti.get_fdata()[mask > 0]
    # Voxel (i, j, k) coordinates, aligned 1:1 with tracer/segmentation --
    # boolean indexing (`[mask > 0]`) and np.argwhere traverse in the same
    # (row-major) order.
    voxel_coords = np.argwhere(mask > 0)

    # Background/unlabeled voxels (segmentation id ~0) are never real ROIs
    # -- excluded up front, shared by both branches below.
    labeled_voxels = segmentation > 1e-6
    segmentation = segmentation[labeled_voxels]
    tracer = tracer[labeled_voxels]
    voxel_coords = voxel_coords[labeled_voxels]

    if func is None:
        labels = np.rint(segmentation)
        index_list = [coord[None, :] for coord in voxel_coords]
        return labels, tracer, np.arange(len(labels)), index_list

    unique_labels = np.unique(segmentation)
    values = labeled_comprehension(
        tracer,
        segmentation,
        unique_labels,
        func,
        default=np.nan,
        out_dtype=float,
    )
    index_list = [voxel_coords[segmentation == label] for label in unique_labels]
    return np.rint(unique_labels), values, np.arange(len(unique_labels)), index_list


def _compute_tracer_worker(args):
    return compute_tracer_from_image(
        args["baseline_path"],
        args["post_injection_path"],
        args["signal_type"],
        args["mask_path"],
        args["segmentation_path"],
        func=args["func"],
    )


def compute_tracer_parallel(args_list, n_procs: int = 5):
    """Run `compute_tracer_from_image` over `args_list`, sequentially or in parallel.

    Each `args_list` entry is a dict of `compute_tracer_from_image` keyword
    arguments plus `"subject"`/`"time_point"`. Returns `(df, index_list)`:
    `df` is the long-format DataFrame (`subject`, `time_point`, `labels`,
    `label_index`, `values`) that `prepare_tensor` consumes, and
    `index_list` is the shared voxel-coordinate metadata from
    `compute_tracer_from_image` (see its docstring), taken from the first
    image processed.

    `label_index` (and thus `index_list`) is only meaningful if every image
    shares the exact same mask/segmentation grid -- every image's
    `(labels, label_index)` is checked against the first, and a `ValueError`
    is raised on any disagreement (e.g. one image missing an ROI another
    has), rather than silently mixing up which tensor column data belongs
    to.
    """
    results_dict = []
    reference: tuple[np.ndarray, np.ndarray] | None = None
    index_list: list[np.ndarray] | None = None

    def collect(task_id, labels, values, label_index, this_index_list):
        nonlocal reference, index_list
        if reference is None:
            reference = (labels, label_index)
            index_list = this_index_list
        elif not (
            np.array_equal(labels, reference[0])
            and np.array_equal(label_index, reference[1])
        ):
            raise ValueError(
                f"compute_tracer_from_image returned labels/label_index for "
                f"task {task_id} that disagree with earlier images -- "
                "compute_tracer_parallel assumes every image shares the same "
                "mask/segmentation grid so label_index consistently "
                "identifies the same ROI/voxel across subjects and time "
                "points.",
            )
        tmp_dict = {
            "labels": labels,
            "label_index": label_index,
            "values": values,
            "subject": args_list[task_id]["subject"],
            "time_point": args_list[task_id]["time_point"],
        }
        results_dict.append(pd.DataFrame(tmp_dict))

    if n_procs == 1:
        for i, args in tenumerate(
            args_list,
            desc="Computing tracer signal sequential",
        ):
            labels, values, label_index, this_index_list = _compute_tracer_worker(args)
            collect(i, labels, values, label_index, this_index_list)
    else:
        ne.set_num_threads(1)
        with Pool(n_procs) as pool:
            for i, (labels, values, label_index, this_index_list) in tenumerate(
                pool.imap(_compute_tracer_worker, args_list),
                total=len(args_list),
                desc="Computing tracer signal in parallel",
            ):
                collect(i, labels, values, label_index, this_index_list)

    return pd.concat(results_dict, ignore_index=True), index_list


def compute_roi_scaling(
    data: np.ndarray | list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-ROI mean and standard deviation from `prepare_tensor` output.

    Pools over every subject and time point -- i.e. every axis except the
    last (label/ROI) one -- ignoring NaNs, so it works whether `data` is the
    regular `(subjects, time_points, labels)` array or the ragged
    `list[np.ndarray]` of per-subject `(n_timepoints_i, labels)` slices that
    `prepare_tensor(..., require_regular=False)` returns.

    Parameters
    ----------
    data : np.ndarray | list[np.ndarray]
        Output tensor or list of slices from `prepare_tensor`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(mean, std)`, each of shape `(labels,)`.
    """
    pooled = (
        data.reshape(-1, data.shape[-1])
        if isinstance(data, np.ndarray)
        else np.concatenate(data, axis=0)
    )
    mean = np.nanmean(pooled, axis=0)
    std = np.nanstd(pooled, axis=0)
    return mean, std


def scale_tensor(
    data: np.ndarray | list[np.ndarray],
    center: bool = False,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray | list[np.ndarray], np.ndarray, np.ndarray]:
    """Scale `prepare_tensor` output per ROI, over all subjects and time points.

    Handles both possible outputs of `prepare_tensor`: the regular
    `(subjects, time_points, labels)` array (`require_regular=True`) and the
    ragged `list[np.ndarray]` of per-subject slices (`require_regular=False`).
    In both cases, scaling is per-label (last axis), pooling over every other
    axis, matching the (subjects, time_points) pooling `compute_roi_scaling`
    does.

    Parameters
    ----------
    data : np.ndarray | list[np.ndarray]
        Output tensor or list of slices from `prepare_tensor`.
    center : bool, optional
        If True, subtract the per-ROI mean before dividing by the per-ROI
        standard deviation. By default False.
    mean, std : np.ndarray | None, optional
        Precomputed per-ROI `(labels,)` mean/std to apply instead of
        computing them from `data` -- e.g. to apply scaling fit on training
        subjects to held-out ones. By default None, in which case both are
        computed from `data` via `compute_roi_scaling`.

    Returns
    -------
    tuple[np.ndarray | list[np.ndarray], np.ndarray, np.ndarray]
        `(scaled, mean, std)`. `scaled` has the same type/shape as `data`.
        `mean` and `std` are the values used, so the same scaling can be
        re-applied later (e.g. to held-out data) by passing them back in.
    """
    if mean is None or std is None:
        computed_mean, computed_std = compute_roi_scaling(data)
        mean = computed_mean if mean is None else mean
        std = computed_std if std is None else std

    # A zero-variance ROI is already constant (equal to its own mean), so
    # guard against a 0/0 NaN by leaving it undivided rather than raising.
    safe_std = np.where(std == 0, 1.0, std)

    def _scale(arr: np.ndarray) -> np.ndarray:
        return ((arr - mean) if center else arr) / safe_std

    scaled = _scale(data) if isinstance(data, np.ndarray) else [_scale(s) for s in data]
    return scaled, mean, std


def _pivot_tracer_df(
    df: pd.DataFrame,
    group_filtering: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Filter by group and pivot the long-format tracer DataFrame.

    First step of `prepare_tensor`: optionally filters `df` to one group,
    then pivots to a DataFrame indexed by (subject, time_point) with one
    column per `label_index`. Only (subject, time_point) combinations
    actually present in `df` get a row -- no NaN-filled rows are introduced
    for combinations that were never observed.

    Pivots on `label_index` rather than `labels`: `labels` (the ROI id) can
    repeat across many rows sharing one `(subject, time_point)` when `df`
    was built from per-voxel `compute_tracer_from_image` output (many
    voxels, one ROI id each) -- pivoting on that directly would let
    `aggfunc="first"` silently keep one arbitrary voxel and drop the rest.
    `label_index` is unique per row by construction (see
    `compute_tracer_from_image`), so this is safe for both ROI-aggregate
    and per-voxel input.
    """
    if group_filtering is not None:
        df = df.query(f"{group_filtering[0]}=='{group_filtering[1]}'")

    return df.pivot_table(
        index=["subject", "time_point"],
        columns="label_index",
        values="values",
        aggfunc="first",  # Safe: label_index is unique per row.
    )


def prepare_tensor(
    df: pd.DataFrame,
    group_filtering: tuple[str, str] | None = None,
    require_regular: bool = True,
    min_timepoints: int = 1,
):
    """Prepare subject x time x label tensor data from a long-format tracer DataFrame.

    Subjects are not required to share the same set of time points: only
    (subject, time_point) combinations that are actually observed are used
    to decide which labels to keep, so one subject's missing scan no longer
    forces a spatial region to be dropped for every subject. The label
    (region) set still needs to be complete and shared across all
    subjects/time points -- that constraint is unavoidable, since a value
    genuinely missing at an observed time point can't be recovered here.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format tracer DataFrame (as produced by `compute_tracer_parallel`),
        with `subject`, `time_point`, `labels`, `label_index`, `values`
        columns.
    group_filtering : tuple[str, str] | None, optional
        `(column, value)` to filter `df` to a single group before pivoting.
        By default None.
    require_regular : bool, optional
        If True (default), returns a single regular `(subjects, time_points,
        labels)` array: any (subject, time_point) combination that was never
        observed becomes a NaN row rather than being silently dropped or
        crashing the reshape. That NaN-padded array is **not** directly
        decomposable by `compute_CP_decomposition`, which doesn't support
        missing values -- impute or mask those NaNs first.

        If False, returns a ragged `list[np.ndarray]` -- one
        `(n_timepoints_i, n_labels)` slice per subject, using only that
        subject's own observed time points, no NaN padding. This is the
        shape `run_PARAFAC2_decomposition_repeated` expects, since PARAFAC2's
        evolving mode can have a different size per subject.
    min_timepoints : int, optional
        Minimum number of observed time points required to keep a subject;
        subjects with fewer are dropped and reported. By default 1 (drop
        only subjects with no data at all). If you plan to use
        `require_regular=False` for PARAFAC2, consider raising this to 2, so
        every subject's evolving factor has enough points to be meaningful.

    Returns
    -------
    If `require_regular`:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            `(tensor, subjects, time_points, labels, label_index)`. `tensor`
            has shape `(len(subjects), len(time_points), len(labels))` and
            may contain NaN for subject/time_point combinations that were
            never observed. `label_index[i]` is column `i`'s `label_index`
            value from `df` (see `compute_tracer_from_image`), unchanged --
            not renumbered after dropping columns -- so it can still be used
            to look up that column's voxel coordinates in
            `compute_tracer_parallel`'s `index_list`.
    Otherwise:
        tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]
            `(slices, subjects, timepoints_per_subject, labels, label_index)`.
            `slices[i]` has shape `(len(timepoints_per_subject[i]),
            len(labels))`, ordered by `timepoints_per_subject[i]`.
    """
    pivot_df = _pivot_tracer_df(df, group_filtering)

    # Drop any label with a NaN among the *actually observed* (subject,
    # time_point) rows. A subject's structurally missing time point is not a
    # row here at all, so it can't force an otherwise well-observed label to
    # be dropped for everyone else.
    valid_pivot = pivot_df.dropna(axis=1, how="any")
    label_index = np.array(valid_pivot.columns.tolist()).astype(int)

    # Recover each surviving column's ROI id. label_index -> labels is a
    # stable mapping (compute_tracer_parallel already verified every image
    # agrees on it), so any one row per label_index gives the right answer.
    label_map = df.drop_duplicates("label_index").set_index("label_index")["labels"]
    labels = label_map.loc[label_index].to_numpy().astype(int)

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

    # Build a regular (subjects x time_points x labels) array, filling any
    # subject/time_point combination that was never observed with NaN,
    # instead of assuming (and crashing if not) that one already exists.
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
