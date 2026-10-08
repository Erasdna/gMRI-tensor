from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from functools import partial
from itertools import islice
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterator
from typing import Literal
from typing import NamedTuple

import nibabel as nib
import numexpr as ne
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
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


def _load_labeled_tracer_voxels(
    baseline_path: Path,
    post_injection_path: Path,
    signal_type: str,
    mask_path: Path,
    segmentation_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the four aligned images and keep the labeled voxels inside the mask.

    The single image-reading step shared by every per-image output, so each
    image is read once however many products are derived from it. See
    `compute_tracer_from_image` for the canonicalization and filtering rules.

    Returns `(tracer, segmentation, voxel_coords, voxel_volume_mm3)`: one
    entry (or `(ndim,)` coordinate row) per labeled voxel, with
    `segmentation` the raw (unrounded) ids, plus the volume of one voxel
    from the affine.
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
    # -- excluded up front, for every downstream product.
    labeled_voxels = segmentation > 1e-6
    voxel_volume_mm3 = float(abs(np.linalg.det(baseline_nifti.affine[:3, :3])))
    return (
        tracer[labeled_voxels],
        segmentation[labeled_voxels],
        voxel_coords[labeled_voxels],
        voxel_volume_mm3,
    )


def _tracer_rows(
    tracer: np.ndarray,
    segmentation: np.ndarray,
    voxel_coords: np.ndarray,
    func: Callable | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | np.ndarray]:
    """Turn `_load_labeled_tracer_voxels` output into `compute_tracer_from_image` rows."""
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
    tracer, segmentation, voxel_coords, _ = _load_labeled_tracer_voxels(
        baseline_path,
        post_injection_path,
        signal_type,
        mask_path,
        segmentation_path,
    )
    return _tracer_rows(tracer, segmentation, voxel_coords, func)


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


def _iter_parallel(
    args_list: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], Any],
    n_procs: int,
    desc: str,
) -> Iterator[tuple[dict[str, Any], Any]]:
    """Lazily yield `(args, worker(args))` per `args_list` entry, in order.

    With `n_procs > 1` at most `2 * n_procs` entries are in flight, so memory
    stays bounded however slowly results are consumed (`Pool.imap` would
    instead keep buffering finished results). `worker` must be picklable.
    """
    if n_procs == 1:
        for args in tqdm(args_list, desc=f"{desc} sequential"):
            yield args, worker(args)
        return

    ne.set_num_threads(1)
    remaining = iter(args_list)
    with (
        Pool(n_procs) as pool,
        tqdm(total=len(args_list), desc=f"{desc} in parallel") as progress,
    ):
        pending: deque[tuple[dict[str, Any], AsyncResult]] = deque()

        def submit(args: dict[str, Any]) -> None:
            pending.append((args, pool.apply_async(worker, (args,))))

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
            yield args, result


def iter_tracer_results(
    args_list: list[dict[str, Any]],
    n_procs: int = 5,
) -> Iterator[TracerResult]:
    """Lazily yield one `TracerResult` per `args_list` entry, in order.

    Each entry holds `compute_tracer_from_image` keyword arguments plus
    `"subject"`/`"time_point"`. Memory stays bounded as in `_iter_parallel`.
    """
    for args, result in _iter_parallel(
        args_list,
        _compute_tracer_worker,
        n_procs,
        desc="Computing tracer signal",
    ):
        yield TracerResult(args["subject"], args["time_point"], *result)


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


@contextmanager
def _atomic_outputs(*paths: Path) -> Iterator[tuple[Path, ...]]:
    """Yield a `.tmp` sibling per path, moved into place only on success.

    Tmp files never written (e.g. an optional sidecar) are simply skipped on
    success; on any exception every tmp file is removed and nothing is moved,
    so a failed run leaves no partial outputs behind.
    """
    tmp_paths = tuple(path.with_name(path.name + ".tmp") for path in paths)
    try:
        yield tmp_paths
    except BaseException:
        for tmp_path in tmp_paths:
            tmp_path.unlink(missing_ok=True)
        raise

    for tmp_path, path in zip(tmp_paths, paths):
        if tmp_path.exists():
            tmp_path.replace(path)


def write_tracer_parquet(
    args_list: list[dict[str, Any]],
    output_path: Path | str,
    n_procs: int = 5,
) -> tuple[Path, Path | None]:
    """Stream `iter_tracer_results` to parquet, one row group per image.

    Writes the long-format frame `prepare_tensor` consumes while holding at
    most a few images in memory. Read it back with `load_tensor_from_parquet`,
    which builds the tensor without loading the whole frame. In
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

    with _atomic_outputs(tracer_path, coords_path) as (tmp_tracer, tmp_coords):
        with pq.ParquetWriter(tmp_tracer, _TRACER_SCHEMA) as writer:
            for i, result in enumerate(iter_tracer_results(args_list, n_procs)):
                if i == 0 and per_voxel:
                    pq.write_table(_coords_table(result), tmp_coords)
                table = _tracer_table(result)
                writer.write_table(table, row_group_size=max(table.num_rows, 1))

    return tracer_path, coords_path if per_voxel else None


def tracer_to_concentration(
    tracer: np.ndarray,
    signal_type: str,
    relaxivity: float = 3.2,
    time_unit: Literal["s", "ms"] = "ms",
) -> np.ndarray:
    """Convert a ΔR1 tracer signal to contrast agent concentration [mM].

    `c = ΔR1 / r1`, with ΔR1 from `compute_tracer` in 1/`time_unit` (1/ms
    for T1 maps in ms) and `relaxivity` r1 in 1/(mM·s). The default 3.2 is
    gadobutrol at 3T in water/CSF; one r1 is applied to every voxel, so
    parenchymal concentrations are approximate (tissue r1 is not measured).

    Raises `ValueError` for `"T1w"`, whose signal ratio is not ΔR1.
    """
    if signal_type not in ("T1map", "R1map"):
        raise ValueError(
            f"Concentration needs a T1map or R1map signal, got {signal_type!r}",
        )
    if time_unit not in ("s", "ms"):
        raise ValueError(f"time_unit must be 's' or 'ms', got {time_unit!r}")
    delta_r1_per_s = tracer * 1000.0 if time_unit == "ms" else tracer
    return delta_r1_per_s / relaxivity


# Statistic columns of `compute_roi_statistics`, for group analysis/plots.
ROI_STATISTIC_COLUMNS = (
    "median",
    "mean",
    "median_concentration",
    "mean_concentration",
    "total_amount",
)

_ROI_STATISTICS_DTYPES = {
    "roi": str,
    "roi_type": str,
    "n_voxels": np.int64,
    "n_valid": np.int64,
    "volume_mm3": np.float64,
    "median": np.float64,
    "mean": np.float64,
    "median_concentration": np.float64,
    "mean_concentration": np.float64,
    "total_amount": np.float64,
}

# mM * mm^3 = (1e-3 mol / 1e-3 m^3) * 1e-9 m^3 = 1e-9 mol = 1e-6 mmol.
_MM_MM3_TO_MMOL = 1e-6


def _finite_median_mean(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.median(finite)), float(np.mean(finite))


def _roi_statistics_row(
    roi: str,
    roi_type: str,
    tracer: np.ndarray,
    concentration: np.ndarray | None,
    voxel_volume_mm3: float,
) -> dict[str, Any]:
    median, mean = _finite_median_mean(tracer)
    median_concentration, mean_concentration, total_amount = np.nan, np.nan, np.nan
    if concentration is not None:
        median_concentration, mean_concentration = _finite_median_mean(concentration)
        finite_concentration = concentration[np.isfinite(concentration)]
        if finite_concentration.size:
            total_amount = (
                float(finite_concentration.sum()) * voxel_volume_mm3 * _MM_MM3_TO_MMOL
            )
    return {
        "roi": roi,
        "roi_type": roi_type,
        "n_voxels": len(tracer),
        "n_valid": int(np.count_nonzero(np.isfinite(tracer))),
        "volume_mm3": len(tracer) * voxel_volume_mm3,
        "median": median,
        "mean": mean,
        "median_concentration": median_concentration,
        "mean_concentration": mean_concentration,
        "total_amount": total_amount,
    }


def compute_roi_statistics(
    tracer: np.ndarray,
    segmentation: np.ndarray,
    voxel_volume_mm3: float,
    roi_groups: Mapping[str, np.ndarray] | None = None,
    concentration: np.ndarray | None = None,
    include_labels: bool = True,
) -> pd.DataFrame:
    """Per-label and per-ROI-group statistics of one image's labeled voxels.

    Inputs are `_load_labeled_tracer_voxels` output (`segmentation` ids are
    rounded here) and optionally `tracer_to_concentration` of `tracer`.
    One `roi_type="label"` row per label id (`roi=str(id)`, ascending) if
    `include_labels`, then one `roi_type="group"` row per `roi_groups` entry
    (e.g. `roi_groups.resolve_roi_groups` output) pooling the voxels of its
    ids -- so a group median is the exact median over those voxels. Groups
    without voxels in this image are omitted.

    Statistics use finite voxels only: `n_voxels`/`volume_mm3` count all of
    the ROI's voxels, `n_valid` the finite ones. `median`/`mean` are of the
    tracer signal; `median_concentration`/`mean_concentration` [mM] and
    `total_amount` [mmol, sum over finite voxels] are NaN without
    `concentration`.
    """
    labels = np.rint(segmentation).astype(np.int64)
    records = []

    if include_labels:
        order = np.argsort(labels, kind="stable")
        unique_labels, starts = np.unique(labels[order], return_index=True)
        for label, voxels in zip(unique_labels, np.split(order, starts[1:])):
            records.append(
                _roi_statistics_row(
                    str(label),
                    "label",
                    tracer[voxels],
                    None if concentration is None else concentration[voxels],
                    voxel_volume_mm3,
                ),
            )

    for name, ids in (roi_groups or {}).items():
        in_group = np.isin(labels, ids)
        if not in_group.any():
            continue
        records.append(
            _roi_statistics_row(
                name,
                "group",
                tracer[in_group],
                None if concentration is None else concentration[in_group],
                voxel_volume_mm3,
            ),
        )

    return pd.DataFrame.from_records(
        records,
        columns=list(_ROI_STATISTICS_DTYPES),
    ).astype(_ROI_STATISTICS_DTYPES)


_ROI_STATISTICS_SCHEMA = pa.schema(
    [
        ("subject", pa.string()),
        ("time_point", pa.int64()),
        ("roi", pa.string()),
        ("roi_type", pa.string()),
        ("n_voxels", pa.int64()),
        ("n_valid", pa.int64()),
        ("volume_mm3", pa.float64()),
        ("median", pa.float64()),
        ("mean", pa.float64()),
        ("median_concentration", pa.float64()),
        ("mean_concentration", pa.float64()),
        ("total_amount", pa.float64()),
    ],
)


def _roi_statistics_table(
    subject: Any,
    time_point: Any,
    roi_statistics: pd.DataFrame,
) -> pa.Table:
    n = len(roi_statistics)
    columns: dict[str, Any] = {
        "subject": pa.repeat(pa.scalar(str(subject)), n),
        "time_point": pa.repeat(pa.scalar(time_point, pa.int64()), n),
    }
    for name in _ROI_STATISTICS_DTYPES:
        columns[name] = pa.array(
            roi_statistics[name].to_numpy(),
            type=_ROI_STATISTICS_SCHEMA.field(name).type,
        )
    return pa.table(columns, schema=_ROI_STATISTICS_SCHEMA)


class PreprocessedPaths(NamedTuple):
    """Files written by `write_preprocessed_data`; `coords` None in ROI mode."""

    tracer: Path
    coords: Path | None
    roi_statistics: Path


def _preprocess_worker(
    args: dict[str, Any],
    roi_groups: Mapping[str, np.ndarray] | None,
    relaxivity: float | None,
    time_unit: Literal["s", "ms"],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | np.ndarray],
    pd.DataFrame,
]:
    """Load one image once; return its tracer rows and ROI statistics."""
    tracer, segmentation, voxel_coords, voxel_volume_mm3 = _load_labeled_tracer_voxels(
        args["baseline_path"],
        args["post_injection_path"],
        args["signal_type"],
        args["mask_path"],
        args["segmentation_path"],
    )
    concentration = (
        None
        if relaxivity is None or args["signal_type"] == "T1w"
        else tracer_to_concentration(
            tracer,
            args["signal_type"],
            relaxivity,
            time_unit,
        )
    )
    roi_statistics = compute_roi_statistics(
        tracer,
        segmentation,
        voxel_volume_mm3,
        roi_groups,
        concentration,
    )
    return (
        _tracer_rows(tracer, segmentation, voxel_coords, args["func"]),
        roi_statistics,
    )


def write_preprocessed_data(
    args_list: list[dict[str, Any]],
    output_dir: Path | str,
    roi_groups: Mapping[str, np.ndarray] | None = None,
    relaxivity: float | None = 3.2,
    time_unit: Literal["s", "ms"] = "ms",
    n_procs: int = 5,
) -> PreprocessedPaths:
    """Read every image once and write all data products to `output_dir/data/`.

    Per `args_list` entry (as in `write_tracer_parquet`) the images are
    loaded once, and both products are derived from that load:

    - `tracer.parquet` (+ `tracer.coords.parquet` per voxel): exactly what
      `write_tracer_parquet` writes, for `load_tensor_from_parquet`.
    - `roi_statistics.parquet`: `compute_roi_statistics` per image, with
      leading `subject`/`time_point` columns. Concentration columns are NaN
      for `T1w` images or `relaxivity=None`; see `tracer_to_concentration`.

    One row group per image in both files; files are only moved into place
    on success.
    """
    if not args_list:
        raise ValueError("args_list is empty; nothing to write")
    if time_unit not in ("s", "ms"):
        raise ValueError(f"time_unit must be 's' or 'ms', got {time_unit!r}")

    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    per_voxel = args_list[0]["func"] is None
    coords_path = data_dir / "tracer.coords.parquet"
    paths = PreprocessedPaths(
        tracer=data_dir / "tracer.parquet",
        coords=coords_path if per_voxel else None,
        roi_statistics=data_dir / "roi_statistics.parquet",
    )
    worker = partial(
        _preprocess_worker,
        roi_groups=roi_groups,
        relaxivity=relaxivity,
        time_unit=time_unit,
    )

    with (
        _atomic_outputs(
            paths.tracer,
            coords_path,
            paths.roi_statistics,
        ) as (tmp_tracer, tmp_coords, tmp_roi_statistics),
        pq.ParquetWriter(tmp_tracer, _TRACER_SCHEMA) as tracer_writer,
        pq.ParquetWriter(
            tmp_roi_statistics,
            _ROI_STATISTICS_SCHEMA,
        ) as roi_statistics_writer,
    ):
        for i, (args, (rows, roi_statistics)) in enumerate(
            _iter_parallel(args_list, worker, n_procs, desc="Preprocessing images"),
        ):
            result = TracerResult(args["subject"], args["time_point"], *rows)
            if i == 0 and per_voxel:
                pq.write_table(_coords_table(result), tmp_coords)
            table = _tracer_table(result)
            tracer_writer.write_table(table, row_group_size=max(table.num_rows, 1))
            roi_statistics_writer.write_table(
                _roi_statistics_table(
                    args["subject"],
                    args["time_point"],
                    roi_statistics,
                ),
            )

    return paths


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


# `(labels, label_index)` packed into one int64, so a column is a scalar
# whose sort order matches `_pivot_tracer_df`'s lexicographic column order
# (both parts non-negative, `label_index < 2**32`).
_KEY_SHIFT = 32
_KEY_MASK = (1 << _KEY_SHIFT) - 1


class _ImageChunk(NamedTuple):
    """Rows of one (subject, time_point) image from a single row group."""

    subject: str
    time_point: int
    keys: np.ndarray
    values: np.ndarray


def _iter_image_chunks(
    parquet_file: pq.ParquetFile,
    group_filtering: tuple[str, str] | None,
    with_keys: bool = True,
) -> Iterator[_ImageChunk]:
    """Yield each row group of a tracer parquet file split per image.

    An image may span several row groups or share one with other images, so
    callers must not assume one chunk per image. With `with_keys=False` the
    `labels`/`label_index` columns are not read and `keys` is empty.
    """
    columns = ["subject", "time_point", "values"]
    if with_keys:
        columns += ["labels", "label_index"]
    if group_filtering is not None:
        if group_filtering[0] not in parquet_file.schema_arrow.names:
            raise ValueError(
                f"group_filtering column {group_filtering[0]!r} is not in the file",
            )
        if group_filtering[0] not in columns:
            columns.append(group_filtering[0])

    for i in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(i, columns=columns)
        if group_filtering is not None:
            column, value = group_filtering
            table = table.filter(pc.equal(pc.cast(table[column], pa.string()), value))
        if table.num_rows == 0:
            continue
        table = table.unify_dictionaries().combine_chunks()

        subject_column = table["subject"].chunk(0)
        if pa.types.is_dictionary(subject_column.type):
            subject_codes = subject_column.indices.to_numpy(zero_copy_only=False)
            subject_names = subject_column.dictionary.to_pylist()
        else:
            subject_names, subject_codes = np.unique(
                subject_column.to_numpy(zero_copy_only=False),
                return_inverse=True,
            )
        time_points = table["time_point"].to_numpy().astype(np.int64)
        keys = (
            (table["labels"].to_numpy().astype(np.int64) << _KEY_SHIFT)
            | table["label_index"].to_numpy().astype(np.int64)
            if with_keys
            else np.empty(0, dtype=np.int64)
        )
        values = table["values"].to_numpy()
        del table

        if (
            subject_codes.min() == subject_codes.max()
            and time_points.min() == time_points.max()
        ):
            yield _ImageChunk(
                str(subject_names[subject_codes[0]]),
                int(time_points[0]),
                keys,
                values,
            )
            continue

        pairs, inverse = np.unique(
            np.column_stack([subject_codes, time_points]),
            axis=0,
            return_inverse=True,
        )
        inverse = inverse.ravel()
        for j, (code, time_point) in enumerate(pairs):
            rows = inverse == j
            yield _ImageChunk(
                str(subject_names[code]),
                int(time_point),
                keys[rows] if with_keys else keys,
                values[rows],
            )


def _invalid_fraction_per_session(
    parquet_file: pq.ParquetFile,
    group_filtering: tuple[str, str] | None,
) -> dict[tuple[str, int], float]:
    """Non-finite fraction of each `(subject, time_point)` session's values.

    With a shared template every session has the same rows, so this is the
    fraction of invalid voxels (or ROIs).
    """
    n_rows: dict[tuple[str, int], int] = {}
    n_invalid: dict[tuple[str, int], int] = {}
    for chunk in _iter_image_chunks(parquet_file, group_filtering, with_keys=False):
        session = (chunk.subject, chunk.time_point)
        n_rows[session] = n_rows.get(session, 0) + len(chunk.values)
        n_invalid[session] = n_invalid.get(session, 0) + int(
            np.count_nonzero(~np.isfinite(chunk.values)),
        )
    return {session: n_invalid[session] / n_rows[session] for session in n_rows}


def _scan_tracer_parquet(
    parquet_file: pq.ParquetFile,
    group_filtering: tuple[str, str] | None,
    sessions: set[tuple[str, int]],
) -> np.ndarray:
    """Sorted keys finite in every one of `sessions`.

    The columns `prepare_tensor`'s `dropna` would keep if `sessions` were the
    only observed rows; chunks of other sessions are ignored.
    """
    keys = np.empty(0, dtype=np.int64)
    finite_count = np.empty(0, dtype=np.int64)
    # Images usually share one voxel template, so reuse the previous chunk's
    # positions instead of re-sorting ~10M keys per image.
    previous_keys: np.ndarray | None = None
    previous_positions = np.empty(0, dtype=np.intp)

    for chunk in _iter_image_chunks(parquet_file, group_filtering):
        if (chunk.subject, chunk.time_point) not in sessions:
            continue
        finite_keys = chunk.keys[np.isfinite(chunk.values)]
        if previous_keys is None or not np.array_equal(finite_keys, previous_keys):
            merged = np.union1d(keys, finite_keys)
            merged_count = np.zeros(len(merged), dtype=np.int64)
            merged_count[np.searchsorted(merged, keys)] = finite_count
            keys, finite_count = merged, merged_count
            previous_keys = finite_keys
            previous_positions = np.searchsorted(keys, finite_keys)
        finite_count[previous_positions] += 1

    return keys[finite_count == len(sessions)]


def load_tensor_from_parquet(
    path: Path | str,
    model: Literal["cp", "parafac2"],
    group_filtering: tuple[str, str] | None = None,
    min_timepoints: int | None = None,
    max_invalid_fraction: float = 0.9,
    dtype: npt.DTypeLike = np.float32,
) -> tuple[
    np.ndarray | list[np.ndarray],
    np.ndarray,
    np.ndarray | list[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Build `prepare_tensor`'s output directly from a `write_tracer_parquet` file.

    Equivalent to `prepare_tensor(pd.read_parquet(path), ...)`, but streams
    the file one row group at a time into a preallocated output, so the
    long-format frame -- many times the tensor's size for per-voxel data --
    is never held in memory. Three passes over the file: session validity,
    columns to keep, then values.

    Sessions are judged before columns: a `(subject, time_point)` session
    whose non-finite fraction exceeds `max_invalid_fraction` is treated as
    unobserved, so one corrupt scan cannot veto every column. Subjects are
    then kept by their number of valid sessions, and only the kept sessions
    decide which columns are finite everywhere. With `max_invalid_fraction=1`
    and `min_timepoints=1` this matches `prepare_tensor` (except that
    dropped subjects no longer veto columns).

    Parameters
    ----------
    path : Path | str
        Long-format tracer parquet file, as written by `write_tracer_parquet`.
        Any row-group layout works; `(subject, time_point, labels,
        label_index)` is assumed unique, which that writer guarantees.
    model : {"cp", "parafac2"}
        `"cp"` returns the regular `(subjects, time_points, labels)` tensor
        (`prepare_tensor(require_regular=True)`), `"parafac2"` the ragged
        per-subject slices (`require_regular=False`).
    group_filtering : tuple[str, str] | None, optional
        As in `prepare_tensor`.
    min_timepoints : int | None, optional
        Minimum valid sessions to keep a subject. None (default) requires a
        valid session at every time point, so `"cp"` has no NaN rows and
        `"parafac2"` only complete subjects. For `"cp"`, a lower value keeps
        incomplete subjects with NaN rows, for
        `run_CP_decomposition_repeated(allow_nan_imputation=True)`.
    max_invalid_fraction : float, optional
        Sessions with a larger non-finite fraction are dropped. Default 0.9.
    dtype : npt.DTypeLike, optional
        Output dtype. float32 by default, halving memory versus float64.

    Returns
    -------
    Same tuples as `prepare_tensor` for the corresponding `require_regular`.
    """
    if model not in ("cp", "parafac2"):
        raise ValueError(f"model must be 'cp' or 'parafac2', got {model!r}")

    parquet_file = pq.ParquetFile(path, read_dictionary=["subject"])
    invalid_fraction = _invalid_fraction_per_session(parquet_file, group_filtering)
    if not invalid_fraction:
        raise ValueError("No tracer rows left to build a tensor from")

    observed: dict[str, set[int]] = {}
    invalid_sessions = []
    for (subject, time_point), fraction in sorted(invalid_fraction.items()):
        valid_time_points = observed.setdefault(subject, set())
        if fraction > max_invalid_fraction:
            invalid_sessions.append((subject, time_point, round(fraction, 3)))
        else:
            valid_time_points.add(time_point)

    if invalid_sessions:
        print(
            f"load_tensor_from_parquet: dropped {len(invalid_sessions)} session(s) "
            f"with more than {max_invalid_fraction:.0%} invalid values "
            f"(subject, time_point, fraction): {invalid_sessions}",
        )

    required = (
        len(set().union(*observed.values()))
        if min_timepoints is None
        else min_timepoints
    )
    subjects = []
    timepoints_per_subject = []
    dropped_subjects = []
    for subject in sorted(observed):
        time_points = np.array(sorted(observed[subject]), dtype=int)
        if len(time_points) < max(required, 1):
            dropped_subjects.append((subject, len(time_points)))
            continue
        subjects.append(subject)
        timepoints_per_subject.append(time_points)

    if dropped_subjects:
        print(
            f"load_tensor_from_parquet: dropped {len(dropped_subjects)} subject(s) "
            f"with fewer than {required} valid time points: {dropped_subjects}",
        )
    if not subjects:
        raise ValueError(
            f"No subject has {required} valid time points; lower min_timepoints "
            "or raise max_invalid_fraction",
        )

    sessions = {
        (subject, int(t))
        for subject, time_points in zip(subjects, timepoints_per_subject)
        for t in time_points
    }
    valid_keys = _scan_tracer_parquet(parquet_file, group_filtering, sessions)
    if len(valid_keys) == 0:
        raise ValueError(
            "No column is finite in every kept session; lower "
            "max_invalid_fraction to drop more sessions",
        )

    all_timepoints = np.array(
        sorted({t for tps in timepoints_per_subject for t in tps}),
        dtype=int,
    )
    subject_index = {subject: i for i, subject in enumerate(subjects)}
    n_labels = len(valid_keys)

    output: np.ndarray | list[np.ndarray]
    if model == "cp":
        tensor = np.full((len(subjects), len(all_timepoints), n_labels), np.nan, dtype)
        timepoint_index = {t: i for i, t in enumerate(all_timepoints)}

        def target_row(subject: str, time_point: int) -> np.ndarray:
            return tensor[subject_index[subject], timepoint_index[time_point]]

        output = tensor
    else:
        # Every kept key is finite in every kept session, so each row is
        # filled completely by the last pass.
        slices = [
            np.empty((len(tps), n_labels), dtype) for tps in timepoints_per_subject
        ]
        row_index = [
            {t: i for i, t in enumerate(tps)} for tps in timepoints_per_subject
        ]

        def target_row(subject: str, time_point: int) -> np.ndarray:
            i = subject_index[subject]
            return slices[i][row_index[i][time_point]]

        output = slices

    previous_keys: np.ndarray | None = None
    positions = np.empty(0, dtype=np.intp)
    kept = np.empty(0, dtype=bool)
    for chunk in _iter_image_chunks(parquet_file, group_filtering):
        if (chunk.subject, chunk.time_point) not in sessions:
            continue
        if previous_keys is None or not np.array_equal(chunk.keys, previous_keys):
            positions = np.searchsorted(valid_keys, chunk.keys)
            kept = positions < n_labels
            kept[kept] = valid_keys[positions[kept]] == chunk.keys[kept]
            positions = positions[kept]
            previous_keys = chunk.keys
        target_row(chunk.subject, chunk.time_point)[positions] = chunk.values[kept]

    labels = (valid_keys >> _KEY_SHIFT).astype(int)
    label_index = (valid_keys & _KEY_MASK).astype(int)
    subjects_arr = np.array(subjects).astype(str)
    if model == "cp":
        return output, subjects_arr, all_timepoints, labels, label_index
    return output, subjects_arr, timepoints_per_subject, labels, label_index
