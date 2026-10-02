from collections import deque
from itertools import islice
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterator
from typing import NamedTuple

import nibabel as nib
import numexpr as ne
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nibabel.nifti1 import Nifti1Image
from scipy.ndimage import labeled_comprehension
from tqdm import tqdm


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


def _within_group_rank(values: np.ndarray) -> np.ndarray:
    """0-based rank of each element among prior equal elements, in order.

    E.g. `[1, 1, 2, 2, 2, 1] -> [0, 1, 0, 1, 2, 2]`. All-zeros whenever
    `values` has no repeats (e.g. already one row per ROI).
    """
    return pd.Series(values).groupby(values).cumcount().to_numpy()


def compute_tracer_from_image(
    baseline_path: Path,
    post_injection_path: Path,
    signal_type: str,
    mask_path: Path,
    segmentation_path: Path,
    func: Callable | None = np.nanmedian,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | np.ndarray]:
    """Compute tracer signal per voxel or per ROI from aligned NIfTI images.

    All four images are reoriented to the closest canonical (RAS+)
    convention, so an equivalent-but-differently-stored axis order is not
    falsely rejected. No resampling is done, so genuinely different grids
    still raise.

    The tracer signal (`compute_tracer`) is computed inside `mask > 0` and
    restricted to voxels tagged with a real ROI (segmentation > 1e-6);
    background voxels are always excluded.

    `func` aggregates each ROI's voxels via
    `scipy.ndimage.labeled_comprehension`. Pass None to skip aggregation and
    get one row per voxel instead, each still tagged with its ROI id.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | np.ndarray]
        `(labels, values, label_index, index_list)`, all length-matched:

        - `labels[i]`: the row's ROI id. Sorted unique ids when aggregating,
          otherwise the i-th voxel's id (so it repeats).
        - `values[i]`: the aggregated or single-voxel tracer signal.
        - `label_index[i]`: 0-based rank among prior rows sharing that ROI
          id, so `(labels[i], label_index[i])` is unique even in per-voxel
          mode. Always 0 when aggregating. `prepare_tensor` pivots on the
          pair, which avoids requiring contiguous ROI ids or the same ROIs
          in every image.
        - `index_list[i]`: `(n_i, ndim)` coordinates behind that row when
          aggregating. Per-voxel, a single `(n_voxels, ndim)` array whose
          row `i` is that voxel's coordinate.

    Raises
    ------
    ValueError
        If the images are not on the same affine grid.
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
    voxel_coords = np.argwhere(mask > 0)

    # Background/unlabeled voxels (segmentation id ~0) are never real ROIs
    # -- excluded up front, shared by both branches below.
    labeled_voxels = segmentation > 1e-6
    segmentation = segmentation[labeled_voxels]
    tracer = tracer[labeled_voxels]
    voxel_coords = voxel_coords[labeled_voxels]

    if func is None:
        labels = np.rint(segmentation)
        return labels, tracer, _within_group_rank(labels), voxel_coords

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
    return (
        np.rint(unique_labels),
        values,
        _within_group_rank(unique_labels),
        index_list,
    )


class TracerResult(NamedTuple):
    """One image's `compute_tracer_from_image` output plus its subject/time point."""

    subject: Any
    time_point: Any
    labels: np.ndarray
    values: np.ndarray
    label_index: np.ndarray
    index_list: list[np.ndarray] | np.ndarray


def _compute_tracer_worker(
    args: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | np.ndarray]:
    return compute_tracer_from_image(
        args["baseline_path"],
        args["post_injection_path"],
        args["signal_type"],
        args["mask_path"],
        args["segmentation_path"],
        func=args["func"],
    )


def iter_tracer_results(
    args_list: list[dict[str, Any]],
    n_procs: int = 5,
) -> Iterator[TracerResult]:
    """Lazily yield one `TracerResult` per `args_list` entry, in order.

    Each entry holds `compute_tracer_from_image` keyword arguments plus
    `"subject"`/`"time_point"`. With `n_procs > 1` at most `2 * n_procs`
    images are in flight, so memory stays bounded however slowly results are
    consumed (`Pool.imap` would instead keep buffering finished results).
    """

    def tag(args: dict[str, Any], result: tuple) -> TracerResult:
        return TracerResult(args["subject"], args["time_point"], *result)

    if n_procs == 1:
        for args in tqdm(args_list, desc="Computing tracer signal sequential"):
            yield tag(args, _compute_tracer_worker(args))
        return

    ne.set_num_threads(1)
    remaining = iter(args_list)
    with (
        Pool(n_procs) as pool,
        tqdm(
            total=len(args_list),
            desc="Computing tracer signal in parallel",
        ) as progress,
    ):
        pending: deque[tuple[dict[str, Any], AsyncResult]] = deque()

        def submit(args: dict[str, Any]) -> None:
            pending.append((args, pool.apply_async(_compute_tracer_worker, (args,))))

        for args in islice(remaining, 2 * n_procs):
            submit(args)
        while pending:
            args, async_result = pending.popleft()
            # Refill before yielding, so workers stay busy meanwhile.
            next_args = next(remaining, None)
            if next_args is not None:
                submit(next_args)
            result = async_result.get()
            progress.update()
            yield tag(args, result)


def compute_tracer_parallel(
    args_list: list[dict[str, Any]],
    n_procs: int = 5,
) -> tuple[pd.DataFrame, list[np.ndarray] | np.ndarray | None]:
    """Run `compute_tracer_from_image` over `args_list`, sequentially or in parallel."""
    frames = []
    index_list: list[np.ndarray] | np.ndarray | None = None
    for result in iter_tracer_results(args_list, n_procs):
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


_TRACER_SCHEMA = pa.schema(
    [
        ("subject", pa.string()),
        ("time_point", pa.int64()),
        ("labels", pa.int64()),
        ("label_index", pa.int64()),
        ("values", pa.float64()),
    ],
)


def _tracer_table(result: TracerResult) -> pa.Table:
    n = len(result.values)
    return pa.table(
        {
            "subject": pa.repeat(pa.scalar(str(result.subject)), n),
            "time_point": pa.repeat(pa.scalar(result.time_point, pa.int64()), n),
            "labels": np.asarray(result.labels).astype(np.int64),
            "label_index": np.asarray(result.label_index).astype(np.int64),
            "values": np.asarray(result.values, dtype=np.float64),
        },
        schema=_TRACER_SCHEMA,
    )


def _coords_table(result: TracerResult) -> pa.Table:
    """One `(labels, label_index, i, j, k)` row per voxel of a per-voxel result."""
    coords = cast(np.ndarray, result.index_list)
    return pa.table(
        {
            "labels": np.asarray(result.labels).astype(np.int64),
            "label_index": np.asarray(result.label_index).astype(np.int64),
            "i": coords[:, 0],
            "j": coords[:, 1],
            "k": coords[:, 2],
        },
    )


def write_tracer_parquet(
    args_list: list[dict[str, Any]],
    output_path: Path | str,
    n_procs: int = 5,
) -> tuple[Path, Path | None]:
    """Stream `iter_tracer_results` to parquet, one row group per image.

    Writes the long-format frame `prepare_tensor` consumes (read it back with
    `pd.read_parquet`) while holding at most a few images in memory. In
    per-voxel mode (`func=None`) voxel coordinates also go to a
    `.coords.parquet` sidecar, one `(labels, label_index, i, j, k)` row per
    voxel, taken from the first image -- so all images are assumed to share
    one template. ROI mode writes no sidecar, since one image's ROI voxels
    need not describe any other's. Files are only moved into place on
    success.

    Returns `(tracer_path, coords_path)`, `coords_path` None in ROI mode.
    """
    if not args_list:
        raise ValueError("args_list is empty; nothing to write")

    per_voxel = args_list[0]["func"] is None
    tracer_path = Path(output_path)
    coords_path = tracer_path.with_suffix(".coords.parquet")
    tmp_tracer = tracer_path.with_name(tracer_path.name + ".tmp")
    tmp_coords = coords_path.with_name(coords_path.name + ".tmp")

    try:
        with pq.ParquetWriter(tmp_tracer, _TRACER_SCHEMA) as writer:
            for i, result in enumerate(iter_tracer_results(args_list, n_procs)):
                if i == 0 and per_voxel:
                    pq.write_table(_coords_table(result), tmp_coords)
                table = _tracer_table(result)
                writer.write_table(table, row_group_size=max(table.num_rows, 1))
    except BaseException:
        tmp_tracer.unlink(missing_ok=True)
        tmp_coords.unlink(missing_ok=True)
        raise

    tmp_tracer.replace(tracer_path)
    if not per_voxel:
        return tracer_path, None
    tmp_coords.replace(coords_path)
    return tracer_path, coords_path


def compute_roi_scaling(
    data: np.ndarray | list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-ROI `(mean, std)`, each shape `(labels,)`.

    Pools over every axis but the last, ignoring NaNs, so it accepts either
    `prepare_tensor` output: the regular `(subjects, time_points, labels)`
    array or the ragged list of per-subject slices.
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

    Accepts either `prepare_tensor` output; scaling is per-label (last axis),
    pooling over every other, as `compute_roi_scaling` does. `center`
    subtracts the mean before dividing.

    Pass a precomputed `mean`/`std` to apply a scaling fit elsewhere, e.g.
    from training subjects to held-out ones. Returns `(scaled, mean, std)`
    with `scaled` the same type and shape as `data`, so the values used can
    be fed back in later.
    """
    if mean is None or std is None:
        computed_mean, computed_std = compute_roi_scaling(data)
        mean = computed_mean if mean is None else mean
        std = computed_std if std is None else std

    # A zero-variance ROI is constant, so leave it undivided rather than
    # producing a 0/0 NaN.
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
):
    """Prepare subject x time x label tensor data from a long-format tracer DataFrame.

    Subjects need not share the same time points: only observed (subject,
    time_point) combinations decide which labels to keep, so one subject's
    missing scan does not drop a region for everyone. The label set must
    still be complete across all observed combinations, since a value
    missing at an observed time point cannot be recovered here.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format tracer data from `compute_tracer_parallel`, with
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
        `compute_tracer_from_image` row -- and so which voxel coordinates in
        `compute_tracer_parallel`'s `index_list` -- column `i` came from.
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
