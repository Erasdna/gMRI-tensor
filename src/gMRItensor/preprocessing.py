"""Images to signal tables: ΔR1 (or the T1w ratio) per voxel, its ROI
statistics, and the tensors the decompositions are fitted to.
"""
import json
import sys
import warnings
from collections import Counter
from collections import deque
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
from contextlib import ExitStack
from functools import partial
from itertools import islice
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Any
from typing import cast
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
from gMRItensor.noise import noise_scale
from gMRItensor.noise import NOISE_SETS
from gMRItensor.noise import reject_negative
from gMRItensor.noise import SCAN_NOISE_COLUMNS
from gMRItensor.roi_groups import is_csf
from nibabel.nifti1 import Nifti1Image
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import labeled_comprehension
from tqdm import tqdm


def compute_tracer(
    baseline: np.ndarray,
    post_injection: np.ndarray,
    signal_type: str,
) -> np.ndarray:
    """The tracer signal per voxel: ΔR1 = 1/T1_post - 1/T1_baseline for
    T1map, R1_post - R1_baseline for R1map (in the maps' own units), and
    post/baseline for T1w. Voxels where a map is 0 (or near 0) are NaN."""

    if signal_type == "T1map":
        expr = (
            "where((abs(post_injection) < 1e-6) | (abs(baseline) < 1e-6), nan, "
            "(1 / post_injection) - (1 / baseline))"
        )
    elif signal_type == "R1map":
        # 0 marks a voxel the mapping failed on, as in T1 maps.
        expr = "where((post_injection == 0) | (baseline == 0), nan, post_injection - baseline)"
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


class _Images(NamedTuple):
    baseline: np.ndarray
    post: np.ndarray
    mask: np.ndarray
    segmentation: np.ndarray
    affine: np.ndarray


def _load_aligned_images(
    baseline_path: Path,
    post_injection_path: Path,
    mask_path: Path,
    segmentation_path: Path,
) -> _Images:
    """The four images of a scan as arrays, reoriented to the closest
    canonical (RAS+) orientation; `ValueError` unless they share a grid."""
    niftis = [
        cast(Nifti1Image, nib.as_closest_canonical(nib.load(path)))
        for path in (baseline_path, post_injection_path, mask_path, segmentation_path)
    ]
    for nifti, name in zip(niftis[1:], ("post-injection", "mask", "segmentation")):
        if not np.allclose(niftis[0].affine, nifti.affine):
            raise ValueError(f"Baseline and {name} images are not aligned")
    baseline, post, mask, segmentation = (nifti.get_fdata() for nifti in niftis)
    return _Images(baseline, post, mask, segmentation, niftis[0].affine)


def _load_labeled_tracer_voxels(
    baseline_path: Path,
    post_injection_path: Path,
    signal_type: str,
    mask_path: Path,
    segmentation_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the four aligned images and keep the labeled voxels inside the mask.

    The single image-reading step shared by every per-image output, so each
    image is read once however many products are derived from it.

    All four images are reoriented to the closest canonical (RAS+)
    orientation, so an equivalent but differently stored axis order is not
    rejected; nothing is resampled, so genuinely different grids raise
    `ValueError`. The signal (`compute_tracer`) is kept inside `mask > 0`
    and only for voxels with a real label (segmentation > 1e-6): background
    voxels are always excluded.

    Returns `(tracer, segmentation, voxel_coords, voxel_volume_mm3)`: one
    entry (or `(ndim,)` coordinate row) per labeled voxel, with
    `segmentation` the raw (unrounded) ids, plus the volume of one voxel
    from the affine.
    """
    images = _load_aligned_images(
        baseline_path,
        post_injection_path,
        mask_path,
        segmentation_path,
    )
    mask = images.mask
    tracer = compute_tracer(
        images.baseline[mask > 0],
        images.post[mask > 0],
        signal_type,
    )
    segmentation = images.segmentation[mask > 0]
    voxel_coords = np.argwhere(mask > 0)

    # Background/unlabeled voxels (segmentation id ~0) are never real ROIs
    # -- excluded up front, for every downstream product.
    labeled_voxels = segmentation > 1e-6
    voxel_volume_mm3 = float(abs(np.linalg.det(images.affine[:3, :3])))
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
    """`_load_labeled_tracer_voxels` output as `(labels, values,
    label_index, index_list)` rows: one per labeled voxel with `func` None,
    else one per label, its voxels aggregated by `func`.

    `(labels[i], label_index[i])` is unique even when a label repeats over
    voxels: `label_index` ranks a row among earlier rows of its label (0
    when aggregating). `index_list` holds each row's voxel coordinates:
    one `(n_voxels, ndim)` array per voxel row, or a list with each label's
    voxels when aggregating.
    """
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
    """One image's `_tracer_rows` output plus its subject and time point."""

    subject: Any
    time_point: Any
    labels: np.ndarray
    values: np.ndarray
    label_index: np.ndarray
    index_list: list[np.ndarray] | np.ndarray


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


SIGNAL_TYPES = ("T1map", "R1map", "T1w")


def delta_r1_per_second(
    tracer: np.ndarray,
    signal_type: str,
    time_unit: str = "ms",
) -> np.ndarray:
    """`compute_tracer` output in the unit preprocessing stores it in.

    T1map (T1 in `time_unit`) and R1map (R1 in 1/`time_unit`) both give ΔR1
    = R1_post - R1_baseline, returned in 1/s. T1w gives the post/baseline
    ratio, which has no unit and is returned unchanged.
    """
    if time_unit not in ("s", "ms"):
        raise ValueError(f"time_unit must be 's' or 'ms', got {time_unit!r}")
    if signal_type not in SIGNAL_TYPES:
        raise ValueError(
            f"signal_type must be one of {SIGNAL_TYPES}, got {signal_type!r}",
        )
    if signal_type == "T1w" or time_unit == "s":
        return tracer
    return tracer * 1000.0


def _signal_metadata(kind: str, signal_type: str) -> dict[str, str]:
    is_ratio = signal_type == "T1w"
    return {
        "kind": kind,
        "input_type": signal_type,
        "signal": "ratio" if is_ratio else "delta_R1",
        "unit": "1" if is_ratio else "1/s",
    }


def read_signal_metadata(path: Path | str) -> dict[str, str]:
    """The metadata `write_preprocessed_data` stores in each of its files.

    `kind` (`roi_signal`, `voxels` or `coords`), `input_type`, `signal`
    (`delta_R1` or `ratio`) and `unit`; coords files add the template
    `shape` and `affine` as JSON, and `template`, the first scan's resolved
    segmentation path. `template` is a record only: the common-template
    check compares labels and voxel coordinates, not files.
    """
    metadata = pq.read_schema(path).metadata or {}
    return {
        key.decode(): value.decode()
        for key, value in metadata.items()
        if not key.startswith(b"pandas") and not key.startswith(b"ARROW")
    }


_ROI_SIGNAL_DTYPES = {
    "label": np.int64,
    "median": np.float64,
    "mean": np.float64,
    "n_voxels": np.int64,
    "n_valid": np.int64,
    "n_rejected": np.int64,
    "voxel_volume_mm3": np.float64,
}


def compute_roi_signal(
    signal: np.ndarray,
    segmentation: np.ndarray,
    voxel_volume_mm3: float,
    rejected: np.ndarray | None = None,
) -> pd.DataFrame:
    """Median, mean and voxel counts of `signal` per segmentation label.

    `segmentation` ids are rounded. `n_voxels` counts every voxel of the
    label, `n_valid` the finite ones the median/mean are taken over (NaN if
    none), and `n_rejected` those marked in `rejected` (already NaN in
    `signal`; see `reject_negative`). One row per label, ascending;
    `voxel_volume_mm3` is repeated so region volumes and totals can be
    computed later.
    """
    labels = np.rint(segmentation).astype(np.int64)
    order = np.argsort(labels, kind="stable")
    unique_labels, starts = np.unique(labels[order], return_index=True)
    records = []
    for label, voxels in zip(unique_labels, np.split(order, starts[1:])):
        values = signal[voxels]
        finite = values[np.isfinite(values)]
        records.append(
            {
                "label": int(label),
                "median": float(np.median(finite)) if finite.size else np.nan,
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "n_voxels": len(values),
                "n_valid": finite.size,
                "n_rejected": (
                    int(np.count_nonzero(rejected[voxels]))
                    if rejected is not None
                    else 0
                ),
                "voxel_volume_mm3": voxel_volume_mm3,
            },
        )
    return pd.DataFrame.from_records(
        records,
        columns=list(_ROI_SIGNAL_DTYPES),
    ).astype(_ROI_SIGNAL_DTYPES)


_ROI_SIGNAL_SCHEMA = pa.schema(
    [
        ("subject", pa.string()),
        ("time_point", pa.int64()),
        ("label", pa.int64()),
        ("median", pa.float64()),
        ("mean", pa.float64()),
        ("n_voxels", pa.int64()),
        ("n_valid", pa.int64()),
        ("n_rejected", pa.int64()),
        ("voxel_volume_mm3", pa.float64()),
    ],
)

#: Columns of `invalid_rois.csv`.
INVALID_ROI_COLUMNS = [
    "subject",
    "time_point",
    "label",
    "compartment",
    "n_voxels",
    "n_nonfinite",
    "n_rejected",
    "invalid_share",
]


def invalidate_rois(
    roi: pd.DataFrame,
    max_invalid_voxel_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`compute_roi_signal` output with every label whose share of invalid
    voxels (non-finite, rejected ones included) exceeds
    `max_invalid_voxel_fraction` set to NaN `median`/`mean` and `n_valid`
    0, so no statistic and no region weight comes from it.

    Returns the frame and the invalidated labels' `label, n_voxels,
    n_nonfinite` (invalid before any rejection), `n_rejected` and
    `invalid_share`.
    """
    share = 1 - roi["n_valid"] / roi["n_voxels"]
    invalid = share > max_invalid_voxel_fraction
    table = roi.assign(
        n_nonfinite=roi["n_voxels"] - roi["n_valid"] - roi["n_rejected"],
        invalid_share=share,
    ).loc[invalid, ["label", "n_voxels", "n_nonfinite", "n_rejected", "invalid_share"]]
    out = roi.copy()
    out.loc[invalid, ["median", "mean"]] = np.nan
    out.loc[invalid, "n_valid"] = 0
    return out, table.reset_index(drop=True)


def _roi_signal_table(subject: Any, time_point: Any, roi: pd.DataFrame) -> pa.Table:
    n = len(roi)
    columns: dict[str, Any] = {
        "subject": pa.repeat(pa.scalar(str(subject)), n),
        "time_point": pa.repeat(pa.scalar(time_point, pa.int64()), n),
    }
    for name in _ROI_SIGNAL_DTYPES:
        columns[name] = pa.array(
            roi[name].to_numpy(),
            type=_ROI_SIGNAL_SCHEMA.field(name).type,
        )
    return pa.table(columns, schema=_ROI_SIGNAL_SCHEMA)


class PreprocessedPaths(NamedTuple):
    """Files written by `write_preprocessed_data`; voxel files None unless
    `store_voxels`, QC files None without a negative cutoff."""

    roi_signal: Path
    voxels: Path | None
    coords: Path | None
    noise: Path | None = None
    invalid_rois: Path | None = None
    #: `roi_signal` as CSV, for viewing (without the parquet metadata).
    roi_signal_csv: Path | None = None


class NegativeCutoff(NamedTuple):
    """The negative-value rule `write_preprocessed_data` applies.

    `cutoffs` per compartment in 1/s (NaN: no cutoff there); `noise` the
    per-scan estimates (`gMRItensor.noise.summarize_noise` table), None
    when the noise SD was fixed; `source` `outside`, `inside` or `fixed`;
    `metadata` is added to the parquet files.
    """

    cutoffs: dict[str, float]
    csf_offset: int
    max_invalid_voxel_fraction: float
    noise: pd.DataFrame | None
    source: str
    metadata: dict[str, str]


class _Rule(NamedTuple):
    """What a pass-2 worker needs of `NegativeCutoff` (cheap to pickle)."""

    cutoffs: dict[str, float]
    csf_offset: int
    max_invalid_voxel_fraction: float


def _compartments(segmentation: np.ndarray, csf_offset: int) -> np.ndarray:
    """True for CSF voxels (`is_csf`) of raw segmentation ids."""
    return is_csf(np.rint(segmentation).astype(np.int64), csf_offset)


def _preprocess_worker(
    args: dict[str, Any],
    time_unit: str,
    store_voxels: bool,
    rule: _Rule | None = None,
) -> tuple[pd.DataFrame, tuple[Any, ...] | None, dict[str, Any] | None]:
    """Load one scan once; return its ROI signal, optionally its voxel rows,
    and with a `rule` what the rule did (`_apply_rule`)."""
    tracer, segmentation, voxel_coords, voxel_volume_mm3 = _load_labeled_tracer_voxels(
        args["baseline_path"],
        args["post_injection_path"],
        args["signal_type"],
        args["mask_path"],
        args["segmentation_path"],
    )
    signal = delta_r1_per_second(tracer, args["signal_type"], time_unit)
    effects = None
    if rule is None:
        roi = compute_roi_signal(signal, segmentation, voxel_volume_mm3)
    else:
        signal, roi, effects = _apply_rule(signal, segmentation, voxel_volume_mm3, rule)
    rows = (
        _tracer_rows(signal, segmentation, voxel_coords, None) if store_voxels else None
    )
    return roi, rows, effects


def _apply_rule(
    signal: np.ndarray,
    segmentation: np.ndarray,
    voxel_volume_mm3: float,
    rule: _Rule,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Reject the values below their compartment's cutoff and invalidate
    the ROIs with too many invalid voxels; every voxel of such an ROI is
    NaN in the returned signal too, so the voxel data agree with the ROIs.

    Returns the signal, the ROI frame and the effects: per compartment the
    labelled voxels, the rejected ones and the invalidated ROIs, and the
    `invalidate_rois` table with a `compartment` column.
    """
    csf = _compartments(segmentation, rule.csf_offset)
    rejected = reject_negative(signal, csf, rule.cutoffs)
    signal = np.where(rejected, np.nan, signal)
    roi, invalid = invalidate_rois(
        compute_roi_signal(signal, segmentation, voxel_volume_mm3, rejected),
        rule.max_invalid_voxel_fraction,
    )
    labels = np.rint(segmentation).astype(np.int64)
    signal = np.where(np.isin(labels, invalid["label"]), np.nan, signal)
    invalid_csf = is_csf(invalid["label"].to_numpy(), rule.csf_offset)
    invalid["compartment"] = np.where(invalid_csf, "csf", "parenchyma")
    counts = {"parenchyma": (~csf, ~invalid_csf), "csf": (csf, invalid_csf)}
    effects = {
        "compartments": {
            name: {
                "n_voxels": int(np.count_nonzero(voxels)),
                "n_rejected": int(np.count_nonzero(rejected & voxels)),
                "n_invalid_rois": int(np.count_nonzero(rois)),
            }
            for name, (voxels, rois) in counts.items()
        },
        "invalid_rois": invalid,
    }
    return signal, roi, effects


def _shell(
    mask: np.ndarray,
    affine: np.ndarray,
    shell_mm: float,
) -> np.ndarray:
    """Voxels outside `mask` within `shell_mm` of it (Euclidean, in mm)."""
    inside = mask > 0
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    distance = distance_transform_edt(~inside, sampling=voxel_sizes)
    return ~inside & (distance <= shell_mm)


# Outside the mask, a baseline value further than this factor from the
# brain's range is air or a fitting failure, not tissue.
_BASELINE_WINDOW = 4.0


def _noise_worker(
    args: dict[str, Any],
    time_unit: str,
    csf_offset: int,
    shell_mm: float,
) -> list[tuple[str, int, float, float]]:
    """Pass 1 for one scan: `(set, n, mode, sigma)` of its ΔR1 [1/s] per
    `NOISE_SETS` set (`noise_scale`).

    `outside`: the voxels outside the mask within `shell_mm` of it whose
    baseline lies within a factor `_BASELINE_WINDOW` of the brain's 1st-99th
    percentile range (other tissue such as fat or muscle stays in; air and
    fitting failures near 0 or far above any tissue value do not);
    `parenchyma` and `csf`: the labelled voxels inside the mask.
    """
    images = _load_aligned_images(
        args["baseline_path"],
        args["post_injection_path"],
        args["mask_path"],
        args["segmentation_path"],
    )
    signal_type = args["signal_type"]

    def delta_r1(where: np.ndarray) -> np.ndarray:
        tracer = compute_tracer(images.baseline[where], images.post[where], signal_type)
        return delta_r1_per_second(tracer, signal_type, time_unit)

    labelled = (images.mask > 0) & (images.segmentation > 1e-6)
    brain_baseline = images.baseline[labelled]
    brain_baseline = brain_baseline[np.isfinite(brain_baseline) & (brain_baseline != 0)]
    outside = _shell(images.mask, images.affine, shell_mm)
    if brain_baseline.size:
        low, high = np.percentile(brain_baseline, [1, 99])
        outside &= (images.baseline >= low / _BASELINE_WINDOW) & (
            images.baseline <= high * _BASELINE_WINDOW
        )
    inside = delta_r1(labelled)
    csf = _compartments(images.segmentation[labelled], csf_offset)
    sets = {
        "outside": delta_r1(outside),
        "parenchyma": inside[~csf],
        "csf": inside[csf],
    }
    rows = []
    for name in NOISE_SETS:
        values = sets[name]
        mode, sigma = noise_scale(values)
        rows.append((name, int(np.count_nonzero(np.isfinite(values))), mode, sigma))
    return rows


def estimate_scan_noise(
    args_list: list[dict[str, Any]],
    time_unit: str,
    csf_offset: int,
    shell_mm: float,
    n_procs: int,
) -> pd.DataFrame:
    """Pass 1: every scan's noise estimate per set (`SCAN_NOISE_COLUMNS`),
    in `args_list` order."""
    worker = partial(
        _noise_worker,
        time_unit=time_unit,
        csf_offset=csf_offset,
        shell_mm=shell_mm,
    )
    rows = [
        (args["subject"], args["time_point"], *row)
        for args, scan_rows in _iter_parallel(
            args_list,
            worker,
            n_procs,
            desc="Estimating noise",
        )
        for row in scan_rows
    ]
    return pd.DataFrame(rows, columns=SCAN_NOISE_COLUMNS)


#: Columns of `noise.csv`.
NOISE_COLUMNS = [
    *SCAN_NOISE_COLUMNS,
    "mode_z",
    "sigma_z",
    "used_in_reference",
    "cutoff",
    "n_rejected",
    "rejected_share",
    "n_invalid_rois",
]


def _noise_table(
    cutoff: NegativeCutoff,
    scans: list[tuple[Any, Any]],
    effects: list[dict[str, Any]],
) -> pd.DataFrame:
    """`noise.csv`: the pass-1 estimates per scan and set (all NaN when the
    noise SD was fixed), the cutoff applied, and what it did."""
    if cutoff.noise is None:
        table = pd.DataFrame(
            [(subject, t, name) for subject, t in scans for name in NOISE_SETS],
            columns=["subject", "time_point", "set"],
        )
        for column in ("n", "mode", "sigma", "mode_z", "sigma_z"):
            table[column] = np.nan
        table["used_in_reference"] = False
    else:
        table = cutoff.noise.copy()
    outside_cutoff = (
        cutoff.cutoffs["parenchyma"] if cutoff.source == "outside" else np.nan
    )
    table["cutoff"] = table["set"].map({**cutoff.cutoffs, "outside": outside_cutoff})
    done = pd.DataFrame(
        [
            {
                "subject": subject,
                "time_point": t,
                "set": name,
                "n_rejected": counts["n_rejected"],
                "rejected_share": counts["n_rejected"] / counts["n_voxels"]
                if counts["n_voxels"]
                else np.nan,
                "n_invalid_rois": counts["n_invalid_rois"],
            }
            for (subject, t), scan in zip(scans, effects)
            for name, counts in scan["compartments"].items()
        ],
    )
    table = table.astype({"subject": str}).merge(
        done.astype({"subject": str}),
        on=["subject", "time_point", "set"],
        how="left",
    )
    return table[NOISE_COLUMNS]


def _template_metadata(segmentation_path: Path) -> dict[str, str]:
    """Shape and affine of the (canonical) template grid, as JSON, and the
    resolved path of the segmentation it was read from (for the record)."""
    image = cast(Nifti1Image, nib.as_closest_canonical(nib.load(segmentation_path)))
    return {
        "shape": json.dumps([int(n) for n in image.shape[:3]]),
        "affine": json.dumps(np.asarray(image.affine).tolist()),
        "template": str(Path(segmentation_path).resolve()),
    }


def _with_metadata(table: pa.Table, metadata: dict[str, str]) -> pa.Table:
    return table.replace_schema_metadata(metadata)


def write_preprocessed_data(
    args_list: list[dict[str, Any]],
    output_dir: Path | str,
    time_unit: str = "ms",
    store_voxels: bool = False,
    n_procs: int = 5,
    variant: str = "",
    cutoff: NegativeCutoff | None = None,
) -> PreprocessedPaths:
    """Read every scan once and write the data products to
    `output_dir/data/<variant>/` (`output_dir/data/` by default).

    Each `args_list` entry holds a scan's `subject`, `time_point`,
    `baseline_path`, `post_injection_path`, `mask_path`, `segmentation_path`
    and `signal_type`; all must share one `signal_type`. The signal is ΔR1 in 1/s for T1map and
    R1map input (`delta_r1_per_second`, `time_unit` being the unit of the
    maps) and the post/baseline ratio for T1w.

    - `roi_signal.parquet`: `compute_roi_signal` per scan, with leading
      `subject`/`time_point` columns; `roi_signal.csv` is the same table
      (without the metadata), for viewing.
    - With `store_voxels`: `voxels.parquet` (one row per labeled voxel, in
      the schema `load_tensor_from_parquet` reads) and `voxels.coords.parquet` (each voxel's
      `(labels, label_index)` key and `(i, j, k)` index, with the template
      `shape`/`affine` in its metadata). Every scan must have the same
      labeled voxels as the first -- one common template -- or `ValueError`.

    - With a `cutoff`: values below their compartment's cutoff become NaN
      and ROIs with too many invalid voxels are invalidated
      (`_apply_rule`), in both the ROI and the voxel data; `noise.csv`
      (`NOISE_COLUMNS`) and `invalid_rois.csv` (`INVALID_ROI_COLUMNS`) say
      what was estimated and done.

    Each file records `read_signal_metadata`. One row group per scan; files
    are only moved into place on success.
    """
    if not args_list:
        raise ValueError("args_list is empty; nothing to write")
    if time_unit not in ("s", "ms"):
        raise ValueError(f"time_unit must be 's' or 'ms', got {time_unit!r}")
    signal_types = {args["signal_type"] for args in args_list}
    if len(signal_types) != 1:
        raise ValueError(
            f"all scans must share one signal_type, got {sorted(signal_types)}",
        )
    (signal_type,) = signal_types

    data_dir = Path(output_dir) / "data" / variant
    data_dir.mkdir(parents=True, exist_ok=True)
    roi_path = data_dir / "roi_signal.parquet"
    roi_csv_path = data_dir / "roi_signal.csv"
    voxels_path = data_dir / "voxels.parquet"
    coords_path = data_dir / "voxels.coords.parquet"
    noise_path = data_dir / "noise.csv"
    invalid_path = data_dir / "invalid_rois.csv"
    extra = cutoff.metadata if cutoff is not None else {}
    roi_metadata = {**_signal_metadata("roi_signal", signal_type), **extra}
    voxels_metadata = {**_signal_metadata("voxels", signal_type), **extra}
    rule = (
        None
        if cutoff is None
        else _Rule(cutoff.cutoffs, cutoff.csf_offset, cutoff.max_invalid_voxel_fraction)
    )
    worker = partial(
        _preprocess_worker,
        time_unit=time_unit,
        store_voxels=store_voxels,
        rule=rule,
    )
    scans: list[tuple[Any, Any]] = []
    effects: list[dict[str, Any]] = []
    roi_tables: list[pa.Table] = []

    with ExitStack() as stack:
        (
            tmp_roi,
            tmp_roi_csv,
            tmp_voxels,
            tmp_coords,
            tmp_noise,
            tmp_invalid,
        ) = stack.enter_context(
            _atomic_outputs(
                roi_path,
                roi_csv_path,
                voxels_path,
                coords_path,
                noise_path,
                invalid_path,
            ),
        )
        roi_writer = stack.enter_context(
            pq.ParquetWriter(tmp_roi, _ROI_SIGNAL_SCHEMA.with_metadata(roi_metadata)),
        )
        voxels_writer = (
            stack.enter_context(
                pq.ParquetWriter(
                    tmp_voxels,
                    _TRACER_SCHEMA.with_metadata(voxels_metadata),
                ),
            )
            if store_voxels
            else None
        )
        first: TracerResult | None = None
        for args, (roi, rows, scan_effects) in _iter_parallel(
            args_list,
            worker,
            n_procs,
            desc="Preprocessing images",
        ):
            roi_table = _roi_signal_table(args["subject"], args["time_point"], roi)
            roi_writer.write_table(_with_metadata(roi_table, roi_metadata))
            roi_tables.append(roi_table)
            if scan_effects is not None:
                scans.append((args["subject"], args["time_point"]))
                effects.append(scan_effects)
            if voxels_writer is None:
                continue
            result = TracerResult(args["subject"], args["time_point"], *rows)
            if first is None:
                first = result
                coords_metadata = {
                    **_signal_metadata("coords", signal_type),
                    **_template_metadata(args["segmentation_path"]),
                }
                pq.write_table(
                    _with_metadata(_coords_table(result), coords_metadata),
                    tmp_coords,
                )
            elif not (
                np.array_equal(result.labels, first.labels)
                and np.array_equal(result.index_list, first.index_list)
            ):
                raise ValueError(
                    f"scan (subject {result.subject}, time point "
                    f"{result.time_point}) has different labeled voxels than "
                    f"the first scan (subject {first.subject}, time point "
                    f"{first.time_point}); store_voxels needs every scan on one "
                    "common template",
                )
            table = _tracer_table(result)
            voxels_writer.write_table(
                _with_metadata(table, voxels_metadata),
                row_group_size=max(table.num_rows, 1),
            )
        pa.concat_tables(roi_tables).to_pandas().to_csv(tmp_roi_csv, index=False)
        if cutoff is not None:
            _noise_table(cutoff, scans, effects).to_csv(tmp_noise, index=False)
            invalid = [
                scan["invalid_rois"].assign(subject=subject, time_point=t)
                for (subject, t), scan in zip(scans, effects)
                if len(scan["invalid_rois"])
            ]
            table = (
                pd.concat(invalid)[INVALID_ROI_COLUMNS]
                if invalid
                else pd.DataFrame(columns=INVALID_ROI_COLUMNS)
            )
            table.to_csv(tmp_invalid, index=False)

    return PreprocessedPaths(
        roi_path,
        voxels_path if store_voxels else None,
        coords_path if store_voxels else None,
        noise_path if cutoff is not None else None,
        invalid_path if cutoff is not None else None,
        roi_csv_path,
    )


def compute_roi_scaling(
    data: np.ndarray | list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-ROI `(mean, std)`, each shape `(labels,)`.

    Pools over every axis but the last, ignoring NaNs, so it accepts either
    tensor form: the regular `(subjects, time_points, labels)` array or the
    ragged list of per-subject slices.
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
    """Scale a tensor per ROI, over all subjects and time points.

    Accepts either tensor form (see `compute_roi_scaling`); scaling is
    per-label (last axis),
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


# `(labels, label_index)` packed into one int64, so a column is a scalar
# whose sort order matches the lexicographic order of the pairs (both parts
# non-negative, `label_index < 2**32`).
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


#: Columns of `Exclusions.columns`.
EXCLUDED_COLUMN_COLUMNS = ["label", "n_columns", "n_dropped", "scans"]
# Scans named per dropped label before the rest are only counted.
_LISTED_SCANS = 10


class Exclusions(NamedTuple):
    """What `load_tensor_with_exclusions` left out of the tensor.

    `subjects`: every left-out subject and why (`EXCLUDED_COLUMNS`).
    `columns`: per label with dropped columns, how many of its columns
    there were and were dropped, and the kept scans in which they were
    non-finite or missing (`EXCLUDED_COLUMN_COLUMNS`).
    """

    subjects: pd.DataFrame
    columns: pd.DataFrame


def _scan_list(scans: list[tuple[str, int]]) -> str:
    listed = "; ".join(f"{subject} t={t}" for subject, t in scans[:_LISTED_SCANS])
    rest = len(scans) - _LISTED_SCANS
    return listed + (f" (+{rest} more)" if rest > 0 else "")


def _scan_tracer_parquet(
    parquet_file: pq.ParquetFile,
    group_filtering: tuple[str, str] | None,
    sessions: set[tuple[str, int]],
    any_finite: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Sorted keys finite in every one of `sessions`, and the dropped ones.

    The columns with a finite value in every session of `sessions`; chunks
    of other sessions are ignored. `any_finite` keeps the keys finite in at
    least one of `sessions` instead. The second value lists the dropped
    columns per label (`EXCLUDED_COLUMN_COLUMNS`), with the sessions in
    which they were non-finite or missing; it is built from per-(session,
    label) counts, so it stays small for voxel input.
    """
    keys = np.empty(0, dtype=np.int64)
    finite_count = np.empty(0, dtype=np.int64)
    seen = np.empty(0, dtype=np.int64)
    entries: Counter[tuple[str, int, int]] = Counter()
    nonfinite: Counter[tuple[str, int, int]] = Counter()
    # Images usually share one voxel template, so reuse the previous chunk's
    # positions instead of re-sorting ~10M keys per image.
    previous_keys: np.ndarray | None = None
    previous_positions = np.empty(0, dtype=np.intp)
    previous_seen: np.ndarray | None = None

    for chunk in _iter_image_chunks(parquet_file, group_filtering):
        session = (chunk.subject, chunk.time_point)
        if session not in sessions:
            continue
        finite = np.isfinite(chunk.values)
        finite_keys = chunk.keys[finite]
        if previous_keys is None or not np.array_equal(finite_keys, previous_keys):
            merged = np.union1d(keys, finite_keys)
            merged_count = np.zeros(len(merged), dtype=np.int64)
            merged_count[np.searchsorted(merged, keys)] = finite_count
            keys, finite_count = merged, merged_count
            previous_keys = finite_keys
            previous_positions = np.searchsorted(keys, finite_keys)
        finite_count[previous_positions] += 1
        if previous_seen is None or not np.array_equal(chunk.keys, previous_seen):
            seen = np.union1d(seen, chunk.keys)
            previous_seen = chunk.keys
        labels = chunk.keys >> _KEY_SHIFT
        for counter, chunk_labels in ((entries, labels), (nonfinite, labels[~finite])):
            for label, n in zip(*np.unique(chunk_labels, return_counts=True)):
                counter[session + (int(label),)] += int(n)

    valid = keys if any_finite else keys[finite_count == len(sessions)]
    seen_labels, n_columns = np.unique(seen >> _KEY_SHIFT, return_counts=True)
    columns_of = dict(zip(seen_labels.tolist(), n_columns.tolist()))
    dropped_labels, n_dropped = np.unique(
        np.setdiff1d(seen, valid) >> _KEY_SHIFT,
        return_counts=True,
    )
    rows = []
    for label, n in zip(dropped_labels.tolist(), n_dropped.tolist()):
        scans = [
            session
            for session in sorted(sessions)
            if nonfinite[session + (label,)]
            or entries[session + (label,)] < columns_of[label]
        ]
        rows.append((label, columns_of[label], n, _scan_list(scans)))
    return valid, pd.DataFrame(rows, columns=EXCLUDED_COLUMN_COLUMNS)


def _warn_imputed_entries(
    tensor: np.ndarray,
    subjects: list[str],
    time_points: np.ndarray,
    sessions: set[tuple[str, int]],
) -> None:
    """Warn how many missing sessions and scattered entries of the CP
    `tensor` are NaN, so left for the fit to impute."""
    missing_sessions = [
        (subject, int(t))
        for subject in subjects
        for t in time_points
        if (subject, int(t)) not in sessions
    ]
    n_scattered = (
        int(np.count_nonzero(np.isnan(tensor)))
        - len(
            missing_sessions,
        )
        * tensor.shape[-1]
    )
    if missing_sessions or n_scattered:
        warnings.warn(
            f"load_tensor_from_parquet: imputing {len(missing_sessions)} missing "
            f"session(s) (subject, time_point) {missing_sessions} and "
            f"{n_scattered} scattered NaN entr(ies) of "
            f"{tensor.size} in the CP fit",
            UserWarning,
            stacklevel=3,
        )


TensorParts = tuple[
    np.ndarray | list[np.ndarray],
    np.ndarray,
    np.ndarray | list[np.ndarray],
    np.ndarray,
    np.ndarray,
]

#: Columns of the `excluded` table of `load_tensor_with_exclusions`.
EXCLUDED_COLUMNS = ["subject", "reason"]
REJECTED_AT_COMMAND_LINE = "rejected at command line"


def _judge_subjects(
    invalid_fraction: dict[tuple[str, int], float],
    max_invalid_fraction: float,
    min_timepoints: int | None,
    exclude_subjects: Sequence[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Which subjects to keep, and why each other one is left out.

    `exclude_subjects` are left out first (`REJECTED_AT_COMMAND_LINE`), so
    they decide neither the study's time points nor the default
    `min_timepoints` (all of them). A session with more than
    `max_invalid_fraction` non-finite values is not valid; a subject with
    fewer than `min_timepoints` valid sessions is left out, its reason
    naming each missing or invalid time point. Notes go to stderr, so the
    stdout of `plan` stays the job count.

    Returns `{subject: sorted valid time points}` of the kept subjects and
    the `EXCLUDED_COLUMNS` table, sorted by subject. Raises for an id in
    `exclude_subjects` that is not in `invalid_fraction`, or when nobody
    is kept.
    """
    present = {subject for subject, _ in invalid_fraction}
    unknown = sorted(set(exclude_subjects) - present)
    if unknown:
        raise ValueError(f"--exclude-subjects: subject(s) not in the data: {unknown}")
    rejected = sorted(set(exclude_subjects))
    if rejected:
        print(
            f"Rejected at command line: {len(rejected)} subject(s): "
            f"{', '.join(rejected)}",
            file=sys.stderr,
        )
    if len(rejected) == len(present):
        raise ValueError("every subject is excluded (--exclude-subjects)")

    sessions: dict[str, dict[int, float]] = {}
    for (subject, time_point), fraction in sorted(invalid_fraction.items()):
        if subject not in rejected:
            sessions.setdefault(subject, {})[time_point] = fraction
    invalid_sessions = [
        (subject, time_point, round(fraction, 3))
        for subject, fractions in sessions.items()
        for time_point, fraction in fractions.items()
        if fraction > max_invalid_fraction
    ]
    if invalid_sessions:
        print(
            f"load_tensor_from_parquet: dropped {len(invalid_sessions)} session(s) "
            f"with more than {max_invalid_fraction:.0%} invalid values "
            f"(subject, time_point, fraction): {invalid_sessions}",
            file=sys.stderr,
        )

    valid = {
        subject: [t for t, f in fractions.items() if f <= max_invalid_fraction]
        for subject, fractions in sessions.items()
    }
    study = sorted({t for fractions in sessions.values() for t in fractions})
    required = (
        len({t for time_points in valid.values() for t in time_points})
        if min_timepoints is None
        else min_timepoints
    )
    kept: dict[str, np.ndarray] = {}
    reasons = {subject: REJECTED_AT_COMMAND_LINE for subject in rejected}
    dropped_subjects = []
    for subject, time_points in valid.items():
        if len(time_points) >= max(required, 1):
            kept[subject] = np.array(sorted(time_points), dtype=int)
            continue
        dropped_subjects.append((subject, len(time_points)))
        problems = [
            f"time point {t} missing"
            if t not in sessions[subject]
            else f"time point {t} has {sessions[subject][t]:.0%} invalid values "
            f"(more than --max-invalid-fraction {max_invalid_fraction:g})"
            for t in study
            if t not in time_points
        ]
        reasons[subject] = (
            f"too few valid time points ({len(time_points)} of {len(study)}; "
            f"needs {required})" + (": " + "; ".join(problems) if problems else "")
        )

    if dropped_subjects:
        print(
            f"load_tensor_from_parquet: dropped {len(dropped_subjects)} subject(s) "
            f"with fewer than {required} valid time points: {dropped_subjects}",
            file=sys.stderr,
        )
    if not kept:
        raise ValueError(
            f"No subject has {required} valid time points; lower min_timepoints "
            "or raise max_invalid_fraction",
        )
    excluded = pd.DataFrame(sorted(reasons.items()), columns=EXCLUDED_COLUMNS)
    return kept, excluded


def load_tensor_from_parquet(
    path: Path | str,
    model: Literal["cp", "parafac2"],
    group_filtering: tuple[str, str] | None = None,
    min_timepoints: int | None = None,
    max_invalid_fraction: float = 0.9,
    dtype: npt.DTypeLike = np.float32,
    impute_missing: bool = False,
    exclude_subjects: Sequence[str] = (),
) -> TensorParts:
    """The subject x time x label tensor of a long-format tracer parquet:
    `load_tensor_with_exclusions` without the table of left-out subjects."""
    *parts, _ = load_tensor_with_exclusions(
        path,
        model,
        group_filtering,
        min_timepoints,
        max_invalid_fraction,
        dtype,
        impute_missing,
        exclude_subjects,
    )
    return cast(TensorParts, tuple(parts))


def load_tensor_with_exclusions(
    path: Path | str,
    model: Literal["cp", "parafac2"],
    group_filtering: tuple[str, str] | None = None,
    min_timepoints: int | None = None,
    max_invalid_fraction: float = 0.9,
    dtype: npt.DTypeLike = np.float32,
    impute_missing: bool = False,
    exclude_subjects: Sequence[str] = (),
) -> tuple[
    np.ndarray | list[np.ndarray],
    np.ndarray,
    np.ndarray | list[np.ndarray],
    np.ndarray,
    np.ndarray,
    Exclusions,
]:
    """The subject x time x label tensor of a long-format tracer parquet,
    and what was left out of it (`Exclusions`): the subjects with the
    reason (`_judge_subjects`) and the dropped columns.

    The file has one row per scan and label (or voxel), as
    `write_preprocessed_data` writes `voxels.parquet`. It is streamed one
    row group at a time into a preallocated output, so the long-format
    frame -- many times the tensor's size for per-voxel data -- is never
    held in memory. Three passes over the file: session validity, columns
    to keep, then values. Columns are the `(labels, label_index)` pairs,
    sorted; subjects sort as text and each subject's time points
    numerically.

    Sessions are judged before columns: a `(subject, time_point)` session
    whose non-finite fraction exceeds `max_invalid_fraction` is treated as
    unobserved, so one corrupt scan cannot veto every column. Subjects are
    then kept by their number of valid sessions, and only the kept sessions
    decide which columns are finite everywhere: a column is dropped only if
    it is non-finite in a kept session, so a dropped subject cannot veto it.

    `impute_missing` (`"cp"` only) relaxes the column rule: a column is kept
    if it is finite in at least one kept session, and its other entries stay
    NaN, as do the kept subjects' missing sessions, for
    `run_CP_decomposition_repeated(allow_nan_imputation=True)` to impute. A
    `UserWarning` gives how many sessions and scattered entries that is.

    Parameters
    ----------
    path : Path | str
        Long-format tracer parquet file (`subject`, `time_point`, `labels`,
        `label_index`, `values`). Any row-group layout works; `(subject,
        time_point, labels, label_index)` is assumed unique, which
        `write_preprocessed_data` guarantees.
    model : {"cp", "parafac2"}
        `"cp"` returns the regular `(subjects, time_points, labels)` tensor,
        `"parafac2"` the ragged per-subject slices.
    group_filtering : tuple[str, str] | None, optional
        `(column, value)`: keep only the rows whose `column` equals `value`.
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
    impute_missing : bool, optional
        Keep the NaN entries of the `"cp"` tensor for imputation rather than
        dropping their columns (see above). Default False.
    exclude_subjects : Sequence[str], optional
        Subjects to leave out, as if their rows were not in the file: they
        decide neither the time points nor the columns. An id not in the
        file raises.

    Returns
    -------
    For `"cp"`: `(tensor, subjects, time_points, labels, label_index,
    exclusions)`, the tensor `(subjects, time_points, labels)`. For
    `"parafac2"`: `(slices, subjects, timepoints_per_subject, labels,
    label_index, exclusions)`, `slices[i]` of shape
    `(len(timepoints_per_subject[i]), labels)`. Column `j` is the pair
    `(labels[j], label_index[j])`. `exclusions` lists every subject in the
    file that is left out, with the reason, and every dropped column.
    """
    if model not in ("cp", "parafac2"):
        raise ValueError(f"model must be 'cp' or 'parafac2', got {model!r}")
    if impute_missing and model != "cp":
        raise ValueError("impute_missing applies to model='cp' only")

    parquet_file = pq.ParquetFile(path, read_dictionary=["subject"])
    invalid_fraction = _invalid_fraction_per_session(parquet_file, group_filtering)
    if not invalid_fraction:
        raise ValueError("No tracer rows left to build a tensor from")

    kept_sessions, excluded = _judge_subjects(
        invalid_fraction,
        max_invalid_fraction,
        min_timepoints,
        exclude_subjects,
    )
    subjects = sorted(kept_sessions)
    timepoints_per_subject = [kept_sessions[subject] for subject in subjects]

    sessions = {
        (subject, int(t))
        for subject, time_points in zip(subjects, timepoints_per_subject)
        for t in time_points
    }
    valid_keys, dropped_columns = _scan_tracer_parquet(
        parquet_file,
        group_filtering,
        sessions,
        any_finite=impute_missing,
    )
    exclusions = Exclusions(excluded, dropped_columns)
    if len(valid_keys) == 0:
        raise ValueError(
            "No column is finite in any kept session"
            if impute_missing
            else "No column is finite in every kept session; lower "
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

    if impute_missing:
        _warn_imputed_entries(tensor, subjects, all_timepoints, sessions)

    labels = (valid_keys >> _KEY_SHIFT).astype(int)
    label_index = (valid_keys & _KEY_MASK).astype(int)
    subjects_arr = np.array(subjects).astype(str)
    if model == "cp":
        return output, subjects_arr, all_timepoints, labels, label_index, exclusions
    return output, subjects_arr, timepoints_per_subject, labels, label_index, exclusions
