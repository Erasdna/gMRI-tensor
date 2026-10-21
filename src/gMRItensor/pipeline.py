"""The `gmri` stages, each taking its `gMRItensor.options` dataclass.

`run_preprocessing` reads images; every other stage reads only files a
previous stage wrote, so each can be run (and rerun) on its own -- from
the `gmri` command line or from a script.
"""
import dataclasses
import hashlib
import json
import shutil
import sys
import warnings
from collections.abc import Hashable
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import cast
from typing import NamedTuple

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from gMRItensor.decomposition import CMFModel
from gMRItensor.decomposition import ConvergenceError
from gMRItensor.decomposition import Method
from gMRItensor.decomposition import PARAFAC2Model
from gMRItensor.decomposition import setup_backend
from gMRItensor.decomposition.parafac2 import _check_parafac2_rank
from gMRItensor.group_statistics import aggregate_subjects
from gMRItensor.group_statistics import compare_groups_over_time
from gMRItensor.group_statistics import resolve_subject_groups
from gMRItensor.group_statistics import significance_summary
from gMRItensor.group_statistics import subjects_outside_sd
from gMRItensor.group_statistics import subjects_with_groups
from gMRItensor.group_statistics import summarize_cohort_over_time
from gMRItensor.group_statistics import summarize_groups_over_time
from gMRItensor.jobs import collect
from gMRItensor.jobs import Data
from gMRItensor.jobs import DirectoryStore
from gMRItensor.jobs import FitTask
from gMRItensor.jobs import GroupSummary
from gMRItensor.jobs import job_slice
from gMRItensor.jobs import n_jobs
from gMRItensor.jobs import plan_replicability
from gMRItensor.jobs import plan_restarts
from gMRItensor.jobs import RestartResult
from gMRItensor.jobs import run_tasks
from gMRItensor.jobs import StoreStatus
from gMRItensor.jobs import TaskCounts
from gMRItensor.model_io import component_weights
from gMRItensor.model_io import load_decomposition
from gMRItensor.model_io import MODEL_FILE
from gMRItensor.model_io import resolve_model
from gMRItensor.model_io import save_decomposition
from gMRItensor.model_io import SavedDecomposition
from gMRItensor.model_io import select_subjects
from gMRItensor.model_io import sort_components
from gMRItensor.options import CONCENTRATION_STATISTICS
from gMRItensor.options import DecompositionOptions
from gMRItensor.options import DecompositionPlotOptions
from gMRItensor.options import FitOptions
from gMRItensor.options import GroupMapPlotOptions
from gMRItensor.options import options_from_json
from gMRItensor.options import options_to_json
from gMRItensor.options import PreprocessingOptions
from gMRItensor.options import ReplicabilityOptions
from gMRItensor.options import RUNNER_METHODS
from gMRItensor.options import StatisticsPlotOptions
from gMRItensor.options import TensorOptions
from gMRItensor.outliers import data_slices
from gMRItensor.outliers import FLAG_COLUMNS
from gMRItensor.outliers import outlier_table
from gMRItensor.plotting.evolving_mode import plot_evolving_mode
from gMRItensor.plotting.mode_grid import plot_mode_grid
from gMRItensor.plotting.outliers import plot_outliers
from gMRItensor.plotting.reproducibility import plot_reproducibility
from gMRItensor.plotting.roi_evolution import figure_path
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_panels
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_rows
from gMRItensor.plotting.spatial_mode import plot_spatial_mode
from gMRItensor.plotting.subject_mode import plot_subject_mode
from gMRItensor.plotting.time_mode import plot_time_mode
from gMRItensor.plotting.utils import resolve_page_width
from gMRItensor.plotting.utils import save_figure
from gMRItensor.preprocessing import compute_roi_scaling
from gMRItensor.preprocessing import load_tensor_from_parquet
from gMRItensor.preprocessing import PreprocessedPaths
from gMRItensor.preprocessing import read_signal_metadata
from gMRItensor.preprocessing import scale_tensor
from gMRItensor.preprocessing import write_preprocessed_data
from gMRItensor.replicability import CrossValidationEngine
from gMRItensor.replicability import evaluate_replicability_multiproc
from gMRItensor.replicability import HalfHalfEngine
from gMRItensor.replicability import ReplicabilityEngine
from gMRItensor.reproducibility import read_reproducibility
from gMRItensor.reproducibility import write_reproducibility
from gMRItensor.roi_groups import add_concentration
from gMRItensor.roi_groups import aggregate_roi_signal
from gMRItensor.roi_groups import is_csf
from gMRItensor.volumes import expand_roi_mode_to_voxels
from nibabel.nifti1 import Nifti1Image

matplotlib.use("Agg")

#: Default r1 [1/(mM s)] for concentrations: gadobutrol at 3T.
DEFAULT_RELAXIVITY = 3.2

MANIFEST_COLUMNS = (
    "subject",
    "time_point",
    "baseline_path",
    "post_injection_path",
    "mask_path",
    "segmentation_path",
)
_PATH_COLUMNS = MANIFEST_COLUMNS[2:]


def _absolute_paths(value: Any) -> Any:
    """`value` with every `Path` in it made absolute, through dicts, lists
    and tuples."""
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key: _absolute_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_absolute_paths(item) for item in value]
    return value


def write_provenance(output_dir: Path, command: str, options: Any) -> Path:
    """Record a stage's settings as `<output_dir>/<command>.json`.

    Paths are recorded absolute, so a record is still valid when read from
    another working directory.
    """
    from gMRItensor import __version__

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{command}.json"
    record = {
        "gMRItensor": __version__,
        "options": _absolute_paths(dataclasses.asdict(options)),
    }
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def _require(path: Path, stage: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run `gmri {stage}` first")


def _read_subject_info(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"subjects": str})


# --------------------------------------------------------------------------
# Preprocessing


def read_manifest(options: PreprocessingOptions) -> list[dict[str, Any]]:
    """Turn the manifest CSV into `write_preprocessed_data` arguments.

    Paths are relative to the manifest. Raises `ValueError` for missing
    columns, empty cells, non-integer time points, duplicate `(subject,
    time_point)` rows or image files that do not exist -- before any image
    is read.
    """
    path = options.manifest
    manifest = pd.read_csv(path, dtype={"subject": str})
    missing_columns = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing_columns:
        raise ValueError(f"{path}: missing column(s) {missing_columns}")

    values = manifest[list(MANIFEST_COLUMNS)]
    blank = values.isna() | values.astype(str).apply(lambda c: c.str.strip() == "")
    if blank.to_numpy().any():
        cells = [
            (int(row), column)
            for column in MANIFEST_COLUMNS
            for row in manifest.index[blank[column]]
        ]
        raise ValueError(f"{path}: empty value(s) at (row, column): {cells[:5]}")

    time_points = pd.to_numeric(manifest["time_point"], errors="coerce")
    bad = manifest[~np.isfinite(time_points) | (time_points != time_points.round())]
    if not bad.empty:
        raise ValueError(
            f"{path}: time_point must be an integer, rows "
            f"{bad.index.tolist()}: {bad['time_point'].tolist()}",
        )
    manifest["time_point"] = time_points.astype(int)

    duplicated = manifest.duplicated(["subject", "time_point"], keep=False)
    if duplicated.any():
        pairs = sorted(
            set(manifest.loc[duplicated, ["subject", "time_point"]].itertuples(False)),
        )
        raise ValueError(f"{path}: duplicate (subject, time_point) rows: {pairs}")

    base = path.parent
    for column in _PATH_COLUMNS:
        manifest[column] = [base / value for value in manifest[column]]
    missing_files = sorted(
        {str(p) for c in _PATH_COLUMNS for p in manifest[c] if not p.exists()},
    )
    if missing_files:
        raise ValueError(
            f"{path}: {len(missing_files)} image file(s) not found, "
            f"e.g. {missing_files[:5]}",
        )

    return [
        {
            **{column: row[column] for column in _PATH_COLUMNS},
            "signal_type": options.input_type,
            "subject": row["subject"],
            "time_point": int(row["time_point"]),
        }
        for _, row in manifest.iterrows()
    ]


def run_preprocessing(options: PreprocessingOptions) -> PreprocessedPaths:
    """`gmri preprocess`: images -> `<output_dir>/data/` ROI (+ voxel) tables."""
    paths = write_preprocessed_data(
        read_manifest(options),
        options.output_dir,
        time_unit=options.time_unit or "ms",  # unused for T1w
        store_voxels=options.store_voxels,
        n_procs=options.n_procs,
    )
    write_provenance(options.output_dir, "preprocess", options)
    return paths


def _metadata(path: Path, kind: str) -> dict[str, str]:
    """`read_signal_metadata` of a `gmri preprocess` file of `kind`."""
    _require(path, "preprocess")
    metadata = read_signal_metadata(path)
    if metadata.get("kind") != kind:
        found = metadata.get("kind")
        raise ValueError(
            f"{path} is not a `gmri preprocess` {kind} file; its kind is {found!r}",
        )
    return metadata


# --------------------------------------------------------------------------
# ROI statistics and their figures


class StatisticsPlotResult(NamedTuple):
    """What `run_statistics_plots` wrote: figures and the summary table."""

    figures: list[Path]
    summary: pd.DataFrame


def format_significance_summary(summary: pd.DataFrame, alpha: float) -> str:
    """The terminal table of `significance_summary` rows with a difference."""
    significant = summary[summary["significant_timepoints"] != ""]
    if significant.empty:
        return f"No significant group differences (adjusted p < {alpha})."
    table = significant.assign(min_p_adj=significant["min_p_adj"].map("{:.3g}".format))
    return f"Significant group differences (adjusted p < {alpha}):\n" + table.to_string(
        index=False,
    )


def _subject_groups(
    subjects: list[str],
    subject_info: Path,
    group_variable: str,
) -> dict[str, Any]:
    """Each subject's `group_variable` value. Subjects with a blank value
    are left out, with a note; one missing from `subject_info` raises."""
    groups = resolve_subject_groups(
        subjects,
        _read_subject_info(subject_info),
        group_variable,
    )
    blank = [s for s, group in zip(subjects, groups) if pd.isna(group)]
    if blank:
        print(
            f"Leaving out {len(blank)} subject(s) without a '{group_variable}' "
            f"value: {', '.join(map(str, blank))}",
        )
    return {s: group for s, group in zip(subjects, groups) if pd.notna(group)}


def run_statistics_plots(options: StatisticsPlotOptions) -> StatisticsPlotResult:
    """`gmri plot statistics`: ROI signal -> group tables, summary, figures.

    Aggregates the requested regions and labels (`aggregate_roi_signal`),
    adds concentrations if a concentration statistic is requested, and
    writes to `<output_dir>/roi_analysis/`: `roi_statistics.parquet`, one
    `summary__<stat>.parquet` and `significance__<stat>.csv` per statistic,
    and `summary.csv` (with `sd_bands`, also `cohort__<stat>.parquet`, the
    cohort mean and SD behind the bands, and `outside_3sd__<stat>.csv`, the
    subjects named for a point beyond +/-3 SD). Then one figure per ROI and
    statistic, drawn from those saved tables, under
    `<output_dir>/figures/roi/`.
    """
    metadata = _metadata(options.roi_signal, "roi_signal")
    signal = metadata.get("signal", "delta_R1")
    wants_concentration = [
        s for s in options.statistics if s in CONCENTRATION_STATISTICS
    ]
    if signal == "ratio":
        if wants_concentration:
            raise ValueError(
                f"{wants_concentration} need ΔR1, but {options.roi_signal} holds "
                "the T1w signal ratio: concentration is not available for T1w",
            )
        if options.relaxivity is not None:
            raise ValueError("relaxivity has no meaning for T1w input")

    roi_signal = pd.read_parquet(options.roi_signal)
    regions = options.region_groups()
    if not options.regions:
        # The default (every preset): skip presets with no label in the data;
        # explicitly requested regions still fail in `aggregate_roi_signal`.
        present = set(roi_signal["label"])
        regions = {name: ids for name, ids in regions.items() if present & set(ids)}
    frame = aggregate_roi_signal(roi_signal, regions, options.label_ids())
    if wants_concentration:
        frame = add_concentration(frame, options.relaxivity or DEFAULT_RELAXIVITY)
    groups = _subject_groups(
        sorted(frame["subject"].unique()),
        options.subject_info,
        options.group_variable,
    )
    frame = frame[frame["subject"].isin(groups)].copy()
    frame["group"] = frame["subject"].map(groups)
    frame = frame.rename(columns={"time_point": "timepoint"})
    categories = sorted(set(groups.values()))

    tables_dir = Path(options.output_dir) / "roi_analysis"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(tables_dir / "roi_statistics.parquet", index=False)
    tables = {}
    for statistic in options.statistics:
        observed = frame.dropna(subset=[statistic])
        summary_path = tables_dir / f"summary__{statistic}.parquet"
        significance_path = tables_dir / f"significance__{statistic}.csv"
        summarize_groups_over_time(observed, facet="roi", value=statistic).to_parquet(
            summary_path,
            index=False,
        )
        significance = compare_groups_over_time(
            observed,
            categories,
            facet="roi",
            value=statistic,
            min_group_n=options.min_group_n,
        )
        significance["significant"] = significance["p_adj"] < options.alpha
        significance.to_csv(significance_path, index=False)
        if options.sd_bands:
            cohort = summarize_cohort_over_time(observed, facet="roi", value=statistic)
            cohort.to_parquet(tables_dir / f"cohort__{statistic}.parquet", index=False)
            subjects_outside_sd(observed, cohort, facet="roi", value=statistic).to_csv(
                tables_dir / f"outside_3sd__{statistic}.csv",
                index=False,
            )
        tables[statistic] = (
            pd.read_parquet(summary_path),
            # Label ROIs are named str(id): keep them strings, not ints.
            pd.read_csv(significance_path, dtype={"roi": str}),
        )
    summary = significance_summary(tables)
    summary.to_csv(tables_dir / "summary.csv", index=False)

    written: list[Path] = []
    for statistic, (summary_table, significance) in tables.items():
        cohort = outside = None
        if options.sd_bands:
            cohort = pd.read_parquet(tables_dir / f"cohort__{statistic}.parquet")
            outside = pd.read_csv(
                tables_dir / f"outside_3sd__{statistic}.csv",
                dtype={"roi": str, "subject": str, "group": str},
            )
        for roi in frame["roi"].drop_duplicates():
            if options.layout == "rows":
                fig, _ = plot_roi_evolution_rows(
                    summary_table,
                    frame,
                    significance,
                    [roi],
                    statistic,
                    page_width=options.page_width,
                    signal=signal,
                    cohort=cohort,
                    outside=outside,
                )
            else:
                fig, _ = plot_roi_evolution_panels(
                    summary_table,
                    significance,
                    [roi],
                    statistic,
                    1,
                    1,
                    page_width=options.page_width,
                    signal=signal,
                )
            written.extend(
                save_figure(
                    fig,
                    figure_path(
                        options.output_dir,
                        roi,
                        statistic,
                        options.layout,
                    ),
                    options.formats,
                    options.dpi,
                ),
            )
    write_provenance(options.output_dir, "plot_statistics", options)
    return StatisticsPlotResult(written, summary)


# --------------------------------------------------------------------------
# Decomposition input


class DecompositionInput(NamedTuple):
    """`load_tensor_from_parquet` output as torch, plus the preprocessing
    applied and, for voxel input, where each column sits on the template."""

    data: torch.Tensor | list[torch.Tensor]
    subjects: np.ndarray
    timepoints: np.ndarray | list[np.ndarray]
    labels: np.ndarray
    label_index: np.ndarray
    scale_mean: np.ndarray | None
    scale_std: np.ndarray | None
    centered: bool
    voxel_coords: np.ndarray | None = None
    template_shape: np.ndarray | None = None
    template_affine: np.ndarray | None = None
    template_path: str | None = None


def _roi_tensor_source(path: Path, statistic: str) -> pa.BufferReader:
    """`roi_signal.parquet` as an in-memory per-label tracer table."""
    roi = pq.read_table(path, columns=["subject", "time_point", "label", statistic])
    table = pa.table(
        {
            "subject": roi["subject"],
            "time_point": roi["time_point"],
            "labels": roi["label"],
            "label_index": pa.array(np.zeros(roi.num_rows, dtype=np.int64)),
            "values": roi[statistic],
        },
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return pa.BufferReader(sink.getvalue())


def _voxel_template(
    path: Path,
    labels: np.ndarray,
    label_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """Coordinates of each tensor column, plus the template shape/affine and
    the recorded template path (None if the file records none)."""
    coords_path = path.with_name(path.name.replace(".parquet", ".coords.parquet"))
    _require(coords_path, "preprocess --store-voxels")
    metadata = read_signal_metadata(coords_path)
    coords = pd.read_parquet(coords_path).set_index(["labels", "label_index"])
    order = pd.MultiIndex.from_arrays([labels, label_index])
    voxel_coords = coords.loc[order, ["i", "j", "k"]].to_numpy(dtype=np.int64)
    return (
        voxel_coords,
        np.asarray(json.loads(metadata["shape"]), dtype=np.int64),
        np.asarray(json.loads(metadata["affine"]), dtype=np.float64),
        metadata.get("template"),
    )


def load_decomposition_input(
    path: Path,
    method: str,
    tensor: TensorOptions,
    statistic: str = "median",
) -> DecompositionInput:
    """Load `gmri preprocess` output as a CP tensor or PARAFAC2 slices.

    `roi_signal.parquet` gives one column per label (its `statistic`);
    `voxels.parquet` one column per voxel, with the voxel coordinates.
    Optionally centered and/or scaled per column over all subjects and time
    points. CP keeps NaN for missing sessions and scattered invalid values
    (see `load_tensor_from_parquet`'s `impute_missing`), and the fit imputes
    them.
    """
    _require(path, "preprocess")
    kind = read_signal_metadata(path).get("kind")
    if kind == "roi_signal":
        source: Any = _roi_tensor_source(path, statistic)
    elif kind == "voxels":
        source = path
    else:
        raise ValueError(
            f"{path} is neither roi_signal.parquet nor voxels.parquet from "
            "`gmri preprocess`",
        )
    data, subjects, timepoints, labels, label_index = load_tensor_from_parquet(
        source,
        "cp" if method == "cp" else "parafac2",
        min_timepoints=tensor.min_timepoints,
        max_invalid_fraction=tensor.max_invalid_fraction,
        impute_missing=method == "cp",
    )
    mean = std = None
    if tensor.scale:
        data, mean, std = scale_tensor(data, center=tensor.center)
    elif tensor.center:
        mean, _ = compute_roi_scaling(data)
        data = data - mean if isinstance(data, np.ndarray) else [s - mean for s in data]
    torch_data = (
        torch.as_tensor(data)
        if isinstance(data, np.ndarray)
        else [torch.as_tensor(s) for s in data]
    )
    template: tuple[Any, Any, Any, Any] = (None, None, None, None)
    if kind == "voxels":
        template = _voxel_template(path, labels, label_index)
    return DecompositionInput(
        torch_data,
        subjects,
        timepoints,
        labels,
        label_index,
        mean,
        std,
        tensor.center,
        *template,
    )


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _method(method: str) -> Method:
    return RUNNER_METHODS[method]


def _fit_options(fit: FitOptions, method: str) -> dict[str, Any]:
    """`run_*_decomposition_repeated` options from `fit` (also what
    `jobs.run_tasks`/`jobs.collect` take; restart-loop ones are ignored there).

    CP's non-negativity is all-or-nothing (`non_negative`), and CP imputes
    the input's NaN entries (`allow_nan_imputation`); PARAFAC2 and
    CMF take the modes (`nn_modes`, `"auto"` = the method's default), and
    PARAFAC2 also its `solver` (CMF only has matcouply's).
    """
    specific: dict[str, Any]
    if method == "cp":
        specific = {
            "non_negative": fit.non_negative_modes is not None,
            "allow_nan_imputation": True,
        }
    else:
        specific = {"nn_modes": fit.non_negative_modes}
        if fit.solver is not None and method == "parafac2":
            specific["solver"] = fit.solver
    return {
        "init_repeats": fit.restarts,
        "max_iter": fit.max_iter,
        "tolerance": fit.tolerance,
        "progress_bar": False,  # quiet batch logs unless extra asks
        **specific,
        **fit.extra,
    }


# One distributed task: the rank it belongs to and its (group, seed) fit.
_RankTask = tuple[int, FitTask]


def _on_device(data: Data, device: torch.device) -> Data:
    """`data` (a tensor, or each slice of a slice list) on `device`."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return [s.to(device) for s in data]


def _job_tasks(
    plan: Sequence[_RankTask],
    job: int,
    tasks_per_job: int,
) -> list[_RankTask]:
    """Job `job`'s tasks (`jobs.job_slice`); an index outside the plan is
    a `ValueError`, the CLI's user error, worded for `--job`."""
    try:
        return job_slice(plan, job, tasks_per_job)
    except IndexError:
        total = n_jobs(plan, tasks_per_job)
        raise ValueError(
            f"job {job} is out of range: the plan has {total} jobs",
        ) from None


#: Fit settings that do not change what one restart fits, so they may
#: differ between `plan`, its jobs and `collect` (the rank has its own
#: store, and the restart count only adds or drops restarts).
_RUNTIME_SETTINGS = ("output_dir", "ranks", "distributed", "n_procs")
_RUNTIME_FIT_SETTINGS = ("restarts", "restart_procs")
_RUNTIME_FIT_EXTRAS = ("progress_bar", "verbose_level")


def settings_fingerprint(options: DecompositionOptions | ReplicabilityOptions) -> str:
    """A short hash of every setting that changes what a restart fits: the
    input, statistic, tensor and fit options (the splits are checked per
    restart by `DirectoryStore.load`)."""
    values = {
        key: value
        for key, value in options_to_json(options)["options"].items()
        if key not in _RUNTIME_SETTINGS
    }
    fit = {
        key: value
        for key, value in values["fit"].items()
        if key not in _RUNTIME_FIT_SETTINGS
    }
    fit["extra"] = {
        key: value
        for key, value in fit["extra"].items()
        if key not in _RUNTIME_FIT_EXTRAS
    }
    values["fit"] = fit
    text = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _store(
    options: DecompositionOptions | ReplicabilityOptions,
    rank: int,
) -> DirectoryStore:
    """Rank `rank`'s restarts, `<store_dir>/<method>_rank_<r>/`, tied to the
    fit settings of `options`: restarts fitted with other settings are
    refused rather than reused."""
    return DirectoryStore(
        options.store_dir / _model_name(options.method, rank),
        settings=settings_fingerprint(options),
    )


def _run_job_tasks(
    tasks: Sequence[_RankTask],
    data: DecompositionInput,
    options: DecompositionOptions | ReplicabilityOptions,
    n_procs: int,
    fit_kwargs: dict[str, Any],
    resume: bool = True,
) -> dict[int, TaskCounts]:
    """Fit `tasks` rank by rank into each rank's store (`_store`); return
    what each rank fitted, reused and refitted.

    `resume=False` refits tasks the store already holds (see `run_tasks`).
    The fits run on `setup_backend()`'s device (`GMRITENSOR_USE_GPU`).
    """
    fit_kwargs = dict(fit_kwargs)
    progress_bar = bool(fit_kwargs.pop("progress_bar", False))
    tensor = _on_device(data.data, setup_backend())
    counts = {}
    for rank, rank_tasks in groupby(tasks, key=lambda item: item[0]):
        counts[rank] = run_tasks(
            [task for _, task in rank_tasks],
            tensor,
            rank,
            _method(options.method),
            _store(options, rank),
            n_procs=n_procs,
            progress_bar=progress_bar,
            resume=resume,
            **fit_kwargs,
        )
    return counts


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _print_job_counts(counts: dict[int, TaskCounts]) -> None:
    """One line per rank of what a `run --job` did with the store."""
    for rank, done in counts.items():
        note = " (fitted with other settings)" if done.refitted else ""
        print(
            f"rank {rank}: {done.fitted} fitted, {done.reused} reused, "
            f"{done.refitted} refitted{note}",
        )


def _store_statuses(
    options: DecompositionOptions | ReplicabilityOptions,
    plan: Sequence[_RankTask],
) -> list[StoreStatus]:
    """What each rank's store holds for every task of `plan`, in order."""
    stores = {rank: _store(options, rank) for rank in dict.fromkeys(r for r, _ in plan)}
    return [stores[rank].status(task) for rank, task in plan]


def _report_store(
    options: DecompositionOptions | ReplicabilityOptions,
    plan: Sequence[_RankTask],
) -> None:
    """Tell `plan`'s user, on stderr, which planned restarts the jobs will
    reuse and which they will refit over results with other settings."""
    statuses = _store_statuses(options, plan)
    current, stale = statuses.count("current"), statuses.count("stale")
    if current:
        print(
            f"{_plural(current, 'restart')} already stored with these settings "
            "will be reused.",
            file=sys.stderr,
        )
    if stale:
        print(
            f"{_plural(stale, 'restart')} in {options.store_dir} "
            f"{'was' if stale == 1 else 'were'} fitted with other settings (or "
            "another plan) and will be refitted and overwritten.",
            file=sys.stderr,
        )


def _remove_planned_stores(
    options: DecompositionOptions | ReplicabilityOptions,
) -> None:
    """`plan --overwrite`: delete the stored restarts of every planned rank,
    whatever they were fitted with, and say so on stderr."""
    for rank in options.ranks:
        root = _store(options, rank).root
        if root.exists():
            n_restarts = sum(1 for _ in root.rglob("seed_*.pt"))
            shutil.rmtree(root)
            print(
                f"Removed {_plural(n_restarts, 'stored restart')} from {root}.",
                file=sys.stderr,
            )


def _require_complete_store(
    options: DecompositionOptions | ReplicabilityOptions,
    plan: Sequence[_RankTask],
) -> None:
    """Refuse to collect while restarts of `plan` are missing or stale
    (`collect` cannot fit), naming the jobs to run (`_job_tasks` order)."""
    statuses = _store_statuses(options, plan)
    per_job = options.distributed.tasks_per_job
    problems: list[str] = []
    jobs: set[int] = set()
    for rank in dict.fromkeys(r for r, _ in plan):
        positions = [i for i, (r, _) in enumerate(plan) if r == rank]
        missing = [i for i in positions if statuses[i] == "missing"]
        stale = [i for i in positions if statuses[i] == "stale"]
        parts = []
        if missing:
            parts.append(f"{_plural(len(missing), 'restart')} missing")
        if stale:
            parts.append(
                f"{_plural(len(stale), 'restart')} fitted with other settings "
                "or another plan",
            )
        if parts:
            problems.append(f"rank {rank}: {' and '.join(parts)}")
            jobs.update(i // per_job for i in missing + stale)
    if problems:
        listed = ", ".join(str(job) for job in sorted(jobs))
        raise ValueError(
            f"{'; '.join(problems)}; run `run --job` for "
            f"{'job' if len(jobs) == 1 else 'jobs'} {listed} first",
        )


def _collect_rank(
    plan: Sequence[FitTask],
    options: DecompositionOptions | ReplicabilityOptions,
    rank: int,
    fit_kwargs: dict[str, Any],
) -> dict[Hashable, GroupSummary]:
    """Gather one rank's restarts; refuse if any restart has no result yet."""
    store = _store(options, rank)
    summaries = collect(plan, store, method=_method(options.method), **fit_kwargs)
    missing = sum(len(summary.missing) for summary in summaries.values())
    if missing:
        raise ValueError(
            f"rank {rank}: {missing} of {len(plan)} restarts missing from "
            f"{store.root}; run every `run --job` first",
        )
    return summaries


_PLAN_FILE = "plan.json"


def _write_plan(options: DecompositionOptions | ReplicabilityOptions) -> Path:
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _PLAN_FILE
    path.write_text(json.dumps(options_to_json(options), indent=2))
    return path


def load_plan(output_dir: Path, expected: type) -> Any:
    """The options `plan` saved in `<output_dir>/plan.json`."""
    path = Path(output_dir) / _PLAN_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `plan` first")
    try:
        options = options_from_json(json.loads(path.read_text()))
    # ValueError: options that fail validation.
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{path} is not a valid plan ({type(error).__name__}: {error}); "
            "run `plan` again",
        ) from None
    if not isinstance(options, expected):
        raise ValueError(
            f"{path} is a {type(options).__name__} plan, not {expected.__name__}",
        )
    # The directory holding plan.json is the output directory, even if it
    # was moved after planning.
    return dataclasses.replace(options, output_dir=Path(output_dir).resolve())


# --------------------------------------------------------------------------
# Decomposition


def _saved_decomposition(
    rank: int,
    model: PARAFAC2Model | CMFModel | tuple[Any, list[Any]],
    error: float,
    data: DecompositionInput,
) -> SavedDecomposition:
    """A fit as a `SavedDecomposition`, components largest first."""
    common: dict[str, Any] = {
        "rank": rank,
        "error": float(error),
        "subjects": data.subjects,
        "timepoints": data.timepoints,
        "labels": data.labels,
        "label_index": data.label_index,
        "scale_mean": data.scale_mean,
        "scale_std": data.scale_std,
        "centered": data.centered,
        "voxel_coords": data.voxel_coords,
        "template_shape": data.template_shape,
        "template_affine": data.template_affine,
        "template_path": data.template_path,
    }
    if isinstance(model, (PARAFAC2Model, CMFModel)):
        return sort_components(
            SavedDecomposition(
                method="cmf" if isinstance(model, CMFModel) else "parafac2",
                weights=_numpy(model.weights),
                subject_mode=_numpy(model.subject_mode),
                label_mode=_numpy(model.label_mode),
                evolving_states=[_numpy(state) for state in model.evolving_states],
                **common,
            ),
        )
    weights, factors = model
    subject_mode, time_mode, label_mode = (_numpy(f) for f in factors)
    return sort_components(
        SavedDecomposition(
            method="cp",
            weights=_numpy(weights),
            subject_mode=subject_mode,
            label_mode=label_mode,
            time_mode=time_mode,
            **common,
        ),
    )


def _model_name(method: str, rank: int) -> str:
    """`<method>_rank_<r>`: the model directory's name, and everything named
    after it (figures, the restart store), so fits never overwrite each
    other."""
    return f"{method}_rank_{rank}"


def _update_fits_table(path: Path, fits: list[SavedDecomposition]) -> None:
    """Add `fits` to the `fits.csv` at `path`, one row per method and rank,
    replacing earlier rows for the same method and rank."""
    rows = pd.DataFrame(
        {
            "method": [s.method for s in fits],
            "rank": [s.rank for s in fits],
            "error": [s.error for s in fits],
            "model": [f"{_model_name(s.method, s.rank)}/{MODEL_FILE}" for s in fits],
        },
    )
    if path.exists():
        earlier = pd.read_csv(path)
        replaced = set(zip(rows["method"], rows["rank"]))
        kept = [
            (method, rank) not in replaced
            for method, rank in zip(earlier["method"], earlier["rank"])
        ]
        rows = pd.concat([earlier[kept], rows], ignore_index=True)
    rows.sort_values(["method", "rank"]).to_csv(path, index=False)


def _write_decompositions(
    options: DecompositionOptions,
    fits: list[SavedDecomposition],
) -> list[Path]:
    """One directory per fit, `<output_dir>/<method>_rank_<r>/`, holding
    `model.h5` and `decompose.json` (the settings of that fit), and
    `fits.csv` in `output_dir`, one row per model there."""
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for saved in fits:
        model_dir = output_dir / _model_name(saved.method, saved.rank)
        model_dir.mkdir(exist_ok=True)
        save_decomposition(model_dir / MODEL_FILE, saved)
        write_provenance(
            model_dir,
            "decompose",
            dataclasses.replace(options, ranks=(saved.rank,)),
        )
        written.append(model_dir / MODEL_FILE)
    _update_fits_table(output_dir / "fits.csv", fits)
    return written


def _decomposition_input(
    options: DecompositionOptions | ReplicabilityOptions,
) -> DecompositionInput:
    return load_decomposition_input(
        Path(options.input),
        options.method,
        options.tensor,
        options.statistic,
    )


def run_decomposition(options: DecompositionOptions) -> list[Path]:
    """`gmri decompose run`: one fit per rank, in this process.

    Every restart is saved to the store, as a distributed run saves them,
    and refitted even if the store already holds it. Writes each
    rank's model directory and `fits.csv` (see `_write_decompositions`) and
    its reproducibility files (see `_finish_rank`); returns the `model.h5`
    paths in `ranks` order.
    """
    data = _decomposition_input(options)
    fit_kwargs = _fit_options(options.fit, options.method)
    plan = plan_restarts(len(data.subjects), options.fit.restarts)
    _run_job_tasks(
        [(rank, task) for rank in options.ranks for task in plan],
        data,
        options,
        options.fit.restart_procs,
        fit_kwargs,
        resume=False,
    )
    return _finish_ranks(options, plan, fit_kwargs, data)


def _finish_ranks(
    options: DecompositionOptions,
    plan: Sequence[FitTask],
    fit_kwargs: dict[str, Any],
    data: DecompositionInput,
) -> list[Path]:
    """Write every rank that converged (`_finish_rank`,
    `_write_decompositions`); then raise `ConvergenceError` naming the
    ranks that did not, so one failed rank does not lose the others."""
    fits, failed = [], []
    for rank in options.ranks:
        try:
            fits.append(_finish_rank(plan, options, rank, fit_kwargs, data))
        except ConvergenceError as error:
            failed.append(error)
    written = _write_decompositions(options, fits) if fits else []
    if failed:
        kept = (
            f"; the other ranks were written to {options.output_dir}" if written else ""
        )
        raise ConvergenceError(
            "; ".join(str(error) for error in failed) + kept,
        ) from failed[-1]
    return written


def _decomposition_plan(
    options: DecompositionOptions,
    n_subjects: int,
) -> list[_RankTask]:
    return [
        (rank, task)
        for rank in options.ranks
        for task in plan_restarts(n_subjects, options.fit.restarts)
    ]


def plan_decomposition(options: DecompositionOptions, overwrite: bool = False) -> int:
    """`gmri decompose plan`: save `plan.json`; return the number of jobs.

    An impossible PARAFAC2 rank is refused here, before any job runs. Says
    on stderr which stored restarts the jobs will reuse or refit, or, with
    `overwrite`, first removes the planned ranks' stored restarts.
    """
    data = _decomposition_input(options)
    if options.method == "parafac2":
        for rank in options.ranks:
            _check_parafac2_rank(data.data, rank)
    plan = _decomposition_plan(options, len(data.subjects))
    if overwrite:
        _remove_planned_stores(options)
    else:
        _report_store(options, plan)
    _write_plan(options)
    return n_jobs(plan, options.distributed.tasks_per_job)


def run_decomposition_job(output_dir: Path, job: int) -> int:
    """`gmri decompose run --job N`: fit job `N`'s restarts into the store.

    Settings come from `plan.json`. Restarts already stored with these
    settings are skipped, so a re-queued job resumes; ones stored with other
    settings are refitted and overwritten. Prints what each rank fitted,
    reused and refitted. Returns the number of tasks.
    """
    options = load_plan(output_dir, DecompositionOptions)
    data = _decomposition_input(options)
    plan = _decomposition_plan(options, len(data.subjects))
    tasks = _job_tasks(plan, job, options.distributed.tasks_per_job)
    counts = _run_job_tasks(
        tasks,
        data,
        options,
        options.fit.restart_procs,
        _fit_options(options.fit, options.method),
    )
    _print_job_counts(counts)
    return len(tasks)


def collect_decomposition(output_dir: Path) -> list[Path]:
    """`gmri decompose collect`: best restart per rank -> the same files as
    `run_decomposition`."""
    options = load_plan(output_dir, DecompositionOptions)
    data = _decomposition_input(options)
    _require_complete_store(options, _decomposition_plan(options, len(data.subjects)))
    fit_kwargs = _fit_options(options.fit, options.method)
    plan = plan_restarts(len(data.subjects), options.fit.restarts)
    return _finish_ranks(options, plan, fit_kwargs, data)


REPRODUCIBILITY_STEM = "reproducibility"
#: How `decompose run` and `collect` save each rank's reproducibility figure.
REPRODUCIBILITY_FIGURE: dict[str, Any] = {"formats": ("pdf", "png"), "dpi": 300}


def _finish_rank(
    plan: Sequence[FitTask],
    options: DecompositionOptions,
    rank: int,
    fit_kwargs: dict[str, Any],
    data: DecompositionInput,
) -> SavedDecomposition:
    """One rank's best restart, after writing its reproducibility files.

    Next to the restarts, in `<store_dir>/<method>_rank_<r>/`:
    `reproducibility.csv` (pairwise FMS between the successful restarts,
    sorted by error) and its figure, `reproducibility.{pdf,png}`.

    Raises
    ------
    ConvergenceError
        If no restart converged; the last failure is chained.
    """
    with warnings.catch_warnings():
        # Raised as a ConvergenceError below instead, on one line.
        warnings.filterwarnings("ignore", message=r"Group .* restarts converged")
        summary = _collect_rank(plan, options, rank, fit_kwargs)["full"]
    best = summary.best
    if best is None or best.model is None or best.error is None:
        raise ConvergenceError(
            f"No decomposition converged within {len(plan)} restarts at rank "
            f"{rank}{summary.tally.failure_summary()}",
        ) from summary.tally.last_failure

    # _collect_rank refuses missing restarts, so every task has a result.
    store = _store(options, rank)
    results = [cast(RestartResult, store.load(task)) for task in plan]
    csv = write_reproducibility(results, store.root)
    fig, _ = plot_reproducibility(*read_reproducibility(csv))
    save_figure(fig, store.root / REPRODUCIBILITY_STEM, **REPRODUCIBILITY_FIGURE)
    return _saved_decomposition(rank, best.model, best.error, data)


def run_reproducibility_plot(
    csv: Path,
    output: Path | None = None,
    page_width: str | float = "single",
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """`gmri plot reproducibility`: redraw a saved `reproducibility.csv`.

    The figure goes to `output` (a path stem), by default next to the CSV.
    """
    csv = Path(csv)
    _require(csv, "decompose run")
    fig, _ = plot_reproducibility(*read_reproducibility(csv), page_width=page_width)
    stem = csv.with_suffix("") if output is None else Path(output)
    return save_figure(fig, stem, formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Replicability


def _stratification(
    options: ReplicabilityOptions,
    subjects: np.ndarray,
) -> torch.Tensor | None:
    """Integer codes of `stratify_by`, in tensor-subject order."""
    if options.stratify_by is None or options.subject_info is None:
        return None
    groups = resolve_subject_groups(
        list(subjects),
        _read_subject_info(options.subject_info),
        options.stratify_by,
    )
    blank = [str(s) for s, group in zip(subjects, groups) if pd.isna(group)]
    if blank:
        raise ValueError(
            f"subject(s) without a '{options.stratify_by}' value to stratify "
            f"by: {', '.join(blank)}",
        )
    codes, _ = pd.factorize(pd.Series(groups))
    return torch.as_tensor(codes)


def _engine(options: ReplicabilityOptions) -> ReplicabilityEngine:
    if options.engine == "cv":
        return CrossValidationEngine(
            splits=int(options.splits or 3),
            repeats=options.repeats,
            seed=options.seed,
        )
    return HalfHalfEngine(repeats=options.repeats, seed=options.seed)


def _score_rows(
    options: ReplicabilityOptions,
    rank: int,
    scores: Sequence[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    rows = []
    for score in scores:
        if options.engine == "cv":
            common, fold_i, fold_j, fms = score
            rows.append(
                {
                    "method": options.method,
                    "rank": rank,
                    "fold_i": fold_i,
                    "fold_j": fold_j,
                    "n_common": len(common),
                    "fms": float(fms),
                },
            )
        else:
            split, fms = score
            rows.append(
                {
                    "method": options.method,
                    "rank": rank,
                    "split": split,
                    "fms": float(fms),
                },
            )
    return rows


def _write_replicability(
    options: ReplicabilityOptions,
    rows: list[dict[str, Any]],
) -> Path:
    columns = (
        ["method", "rank", "fold_i", "fold_j", "n_common", "fms"]
        if options.engine == "cv"
        else ["method", "rank", "split", "fms"]
    )
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "replicability.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    write_provenance(output_dir, "replicability", options)
    return path


def run_replicability(options: ReplicabilityOptions) -> Path:
    """`gmri replicability run`: factor match scores per rank, in this process.

    The (optionally scaled) tensor is split by the engine; each rank gets a
    fresh engine with `seed`, so ranks see the same splits. The fits run on
    `setup_backend()`'s device (`GMRITENSOR_USE_GPU`).
    """
    data = _decomposition_input(options)
    stratification = _stratification(options, data.subjects)
    tensor = _on_device(data.data, setup_backend())
    rows = []
    for rank in options.ranks:
        scores = evaluate_replicability_multiproc(
            _engine(options),
            tensor,
            rank,
            method=_method(options.method),
            stratification=stratification,
            n_procs=options.n_procs,
            **_fit_options(options.fit, options.method),
        )
        rows.extend(_score_rows(options, rank, scores))
    return _write_replicability(options, rows)


def _replicability_plans(
    options: ReplicabilityOptions,
    data: DecompositionInput,
) -> dict[int, list[FitTask]]:
    stratification = _stratification(options, data.subjects)
    return {
        rank: plan_replicability(
            _engine(options),
            len(data.subjects),
            options.fit.restarts,
            stratification,
        )
        for rank in options.ranks
    }


def _flatten(plans: dict[int, list[FitTask]]) -> list[_RankTask]:
    return [(rank, task) for rank, plan in plans.items() for task in plan]


def plan_replicability_jobs(
    options: ReplicabilityOptions,
    overwrite: bool = False,
) -> int:
    """`gmri replicability plan`: save `plan.json`; return the number of
    jobs. Reports or, with `overwrite`, first removes the stored restarts
    as `plan_decomposition` does."""
    data = _decomposition_input(options)
    plan = _flatten(_replicability_plans(options, data))
    if overwrite:
        _remove_planned_stores(options)
    else:
        _report_store(options, plan)
    _write_plan(options)
    return n_jobs(plan, options.distributed.tasks_per_job)


def run_replicability_job(output_dir: Path, job: int) -> int:
    """`gmri replicability run --job N`: fit job `N`'s split/fold restarts,
    with the settings from `plan.json`, resuming and refitting as
    `run_decomposition_job` does. Returns the number of tasks."""
    options = load_plan(output_dir, ReplicabilityOptions)
    data = _decomposition_input(options)
    plan = _flatten(_replicability_plans(options, data))
    tasks = _job_tasks(plan, job, options.distributed.tasks_per_job)
    counts = _run_job_tasks(
        tasks,
        data,
        options,
        options.n_procs,
        _fit_options(options.fit, options.method),
    )
    _print_job_counts(counts)
    return len(tasks)


def collect_replicability(output_dir: Path) -> Path:
    """`gmri replicability collect`: score the gathered fits ->
    `replicability.csv`, as `run_replicability` writes it."""
    options = load_plan(output_dir, ReplicabilityOptions)
    data = _decomposition_input(options)
    plans = _replicability_plans(options, data)
    _require_complete_store(options, _flatten(plans))
    fit_kwargs = _fit_options(options.fit, options.method)
    rows = []
    for rank, plan in plans.items():
        summaries = _collect_rank(plan, options, rank, fit_kwargs)
        rows.extend(_score_rows(options, rank, _engine(options).compute_fms(summaries)))
    return _write_replicability(options, rows)


# --------------------------------------------------------------------------
# Decomposition figures


class _Spatial(NamedTuple):
    """A spatial mode laid out on voxels of the template grid."""

    mode: np.ndarray
    index_list: np.ndarray
    csf: np.ndarray
    background: np.ndarray
    slices: tuple[int, int, int]


def _canonical_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Voxel data and affine of `path`, in canonical (RAS) orientation."""
    image = cast(Nifti1Image, nib.as_closest_canonical(nib.load(path)))
    return np.asarray(image.get_fdata()), np.asarray(image.affine)


def _check_grid(
    path: Path,
    data: np.ndarray,
    affine: np.ndarray,
    shape: tuple[int, ...],
    expected_affine: np.ndarray | None,
    reference: str,
) -> None:
    """Raise unless `path` lies on the `shape`/`expected_affine` grid."""
    same_affine = expected_affine is None or np.allclose(affine, expected_affine)
    if tuple(data.shape[:3]) != tuple(shape) or not same_affine:
        raise ValueError(
            f"{path} is not on the grid of {reference}: shape "
            f"{tuple(data.shape[:3])} vs {tuple(shape)}"
            + ("" if same_affine else ", and a different affine"),
        )


class _LabelSpace(NamedTuple):
    """Where the rows of a label-space array sit: one label id per row and,
    for voxel data, each row's template coordinates and the template grid.
    `owner` names the data in messages (`model`, `input`)."""

    labels: np.ndarray
    voxel_coords: np.ndarray | None
    template_shape: np.ndarray | None
    template_affine: np.ndarray | None
    template_path: str | None
    owner: str


def _model_label_space(saved: SavedDecomposition) -> _LabelSpace:
    return _LabelSpace(
        saved.labels,
        saved.voxel_coords,
        saved.template_shape,
        saved.template_affine,
        saved.template_path,
        "model",
    )


def _spatial_inputs(
    label_mode: np.ndarray,
    space: _LabelSpace,
    segmentation_path: Path | None,
    background_path: Path | None,
    slices: tuple[int, int, int] | None,
    fill_value: float | None = None,
) -> _Spatial | None:
    """Voxel layout of `label_mode` (one row per `space.labels` entry); None
    for ROI data without a segmentation to map onto. Segmented voxels whose
    label is not in `space.labels` get `fill_value`, or with None are left
    out of the layout, so they show only the background.

    Every image given is checked against the grid it is drawn on: voxel
    data's template (its stored shape and affine), or the segmentation of
    ROI data.
    """
    owner = space.owner
    segmentation = segmentation_affine = None
    if segmentation_path is not None:
        segmentation_data, segmentation_affine = _canonical_image(segmentation_path)
        segmentation = np.rint(segmentation_data).astype(np.int64)

    if space.voxel_coords is None:
        if segmentation is None:
            return None
        if not np.isin(space.labels, segmentation).any():
            first = list(space.labels[:5])
            raise ValueError(
                f"{segmentation_path} contains none of the {owner}'s labels "
                f"{first}...; it must be in the {owner}'s label space",
            )
        mode, index_list = expand_roi_mode_to_voxels(
            label_mode,
            space.labels,
            segmentation,
            fill_value=0.0 if fill_value is None else fill_value,
        )
        if fill_value is None:
            modelled = np.isin(segmentation[tuple(index_list.T)], space.labels)
            mode, index_list = mode[modelled], index_list[modelled]
        voxel_labels = segmentation[tuple(index_list.T)]
        shape = tuple(segmentation.shape)
        grid_affine = segmentation_affine
        reference = f"the segmentation {segmentation_path}"
    else:
        mode, index_list = label_mode, space.voxel_coords
        voxel_labels = space.labels
        shape = tuple(int(n) for n in np.asarray(space.template_shape))
        grid_affine = space.template_affine
        reference = f"the {owner}'s template"
        if space.template_path:
            reference += f", {space.template_path},"
        if segmentation is not None:
            _check_grid(
                Path(segmentation_path),  # type: ignore[arg-type]
                segmentation,
                segmentation_affine,  # type: ignore[arg-type]
                shape,
                grid_affine,
                reference,
            )

    if background_path is not None:
        background, background_affine = _canonical_image(background_path)
        _check_grid(
            Path(background_path),
            background,
            background_affine,
            shape,
            grid_affine,
            reference,
        )
    elif segmentation is not None:
        background = (segmentation > 0).astype(float)
    else:
        background = np.zeros(shape)
        background[tuple(index_list.T)] = 1.0

    shown = slices or tuple(int(i) for i in np.rint(index_list.mean(axis=0)))
    if not all(0 <= s < n for s, n in zip(shown, shape)):
        raise ValueError(
            f"slices {tuple(shown)} are outside the grid; each must be in "
            f"0..{tuple(n - 1 for n in shape)}",
        )
    return _Spatial(
        mode,
        index_list,
        is_csf(voxel_labels),
        background,
        shown,  # type: ignore[arg-type]
    )


def _component_weights_table(weights: np.ndarray) -> pd.DataFrame:
    """`component` (1-based), `weight` and `relative_weight` (its share of
    the total; zero for an all-zero fit rather than NaN)."""
    total = weights.sum()
    relative = weights / total if total > 0 else np.zeros_like(weights)
    return pd.DataFrame(
        {
            "component": np.arange(1, len(weights) + 1),
            "weight": weights,
            "relative_weight": relative,
        },
    )


def _same_values(a: Any, b: Any) -> bool:
    """Whether two saved fields hold the same values: None only equals
    None, ragged lists compare element by element, numbers to 1e-5."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple)) and any(np.ndim(item) > 0 for item in a):
        return len(a) == len(b) and all(map(_same_values, a, b))
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "biufc" and b.dtype.kind in "biufc":
        return bool(np.allclose(a, b, rtol=1e-5, equal_nan=True))
    return bool((a.astype(str) == b.astype(str)).all())


def load_model_data(
    model: Path,
    saved: SavedDecomposition,
    input_override: Path | None = None,
) -> list[np.ndarray]:
    """The data `model` was fitted to, one array per subject, in the space
    of the fit (same subjects, scaling and centring).

    `model` is anything `resolve_model` accepts. Settings come from the
    `decompose.json` next to the model file; the input path from there
    unless `input_override` is given.

    Raises
    ------
    FileNotFoundError
        If the record or the input is missing.
    ValueError
        If the record is not this model's (method, rank), or the reloaded
        data differs from what the model was fitted to: its subjects,
        labels, time points, centring or scaling.
    """
    record_path = resolve_model(model)[0].with_name("decompose.json")
    _require(record_path, "decompose")
    record = json.loads(record_path.read_text())["options"]
    if record.get("method") != saved.method or saved.rank not in record.get(
        "ranks",
        (),
    ):
        raise ValueError(
            f"{record_path} records a {record.get('method')} fit at rank(s) "
            f"{record.get('ranks')}, not this {saved.method} rank-{saved.rank} "
            "model",
        )
    path = Path(input_override or record["input"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path}, the input recorded in {record_path}, not found; pass --input",
        )
    data = load_decomposition_input(
        path,
        saved.method,
        TensorOptions(**record["tensor"]),
        record["statistic"],
    )
    fields = {
        "subjects": ([str(s) for s in data.subjects], [str(s) for s in saved.subjects]),
        "labels": (data.labels, saved.labels),
        "label_index": (data.label_index, saved.label_index),
        "timepoints": (data.timepoints, saved.timepoints),
        "centred": (bool(data.centered), bool(saved.centered)),
        "scale_mean": (data.scale_mean, saved.scale_mean),
        "scale_std": (data.scale_std, saved.scale_std),
    }
    differ = [name for name, (a, b) in fields.items() if not _same_values(a, b)]
    if differ:
        raise ValueError(
            f"the data reloaded from {path} differs from what {model} was "
            f"fitted to ({', '.join(differ)}); was the input preprocessed "
            "again after the fit?",
        )
    return data_slices(data.data)


def _optional_groups(
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
) -> list[Any]:
    """Each subject's group value, None if it has none."""
    lookup = subject_info.drop_duplicates("subjects").set_index("subjects")
    lookup = lookup[group_variable]
    return [
        lookup[s] if s in lookup.index and pd.notna(lookup[s]) else None
        for s in subjects
    ]


def run_decomposition_plots(options: DecompositionPlotOptions) -> list[Path]:
    """`gmri plot decomposition`: one saved model -> figures.

    Components are drawn largest first (`load_decomposition` sorts them),
    also for models saved before sorting existed.

    Draws the selected parts (`DecompositionPlotOptions.selected_parts`) to
    `<output_dir>/figures/decomposition/<model>__<part>.<ext>`; the subject
    mode gets `<subject_prefix>_` in front when one is set. Every run also
    writes `<model>__component_weights.csv` there: each component's
    amplitude once its modes are normalised (`component_weights`), the
    scale the figures leave out. Parts that group subjects leave out (and
    name) subjects without a `group_variable` value, as long as every group
    keeps 2 (`subjects_with_groups`). Spatial parts of an ROI model need a
    segmentation: without one they are skipped with a note if no part was
    chosen explicitly, and raise otherwise.

    The outlier part reloads the data the model was fitted to
    (`load_model_data`) and scores every fitted subject, also those without
    a group (`outlier_table`, written as `<model>__outliers.csv`). If that
    data cannot be reloaded as fitted, the part is skipped with a note if
    no part was chosen explicitly, and raises otherwise.

    `<model>` is the model's name: its directory, or the stem of any other
    model file (`resolve_model`).

    Returns the figure files only: the CSVs (component weights, evolving-mode
    significance, outliers) are written next to them but not listed.
    """
    model_file, stem = resolve_model(options.model)
    saved = load_decomposition(model_file)
    # Of the full fit, before any subjects are left out below.
    weights = component_weights(saved)
    subject_info = _read_subject_info(options.subject_info)
    subjects = [str(s) for s in saved.subjects]
    page_width = resolve_page_width(options.page_width)
    out_dir = Path(options.output_dir) / "figures" / "decomposition"
    parts = list(options.selected_parts())

    spatial = None
    if {"mode_grid", "spatial"} & set(parts):
        spatial = _spatial_inputs(
            saved.label_mode,
            _model_label_space(saved),
            options.segmentation,
            options.background,
            options.slices,
        )
        if spatial is None:
            message = (
                "the spatial parts (mode_grid, spatial) of an ROI model need "
                "--segmentation, a label volume in the decomposition's label space"
            )
            if options.parts:
                raise ValueError(message)
            print(f"Skipping {message}.")
            parts = [p for p in parts if p not in ("mode_grid", "spatial")]

    outliers = None
    outlier_groups: list[str | None] = []
    if "outliers" in parts:
        # Before subjects without a group are left out: leverage and
        # residuals are relative to the model as fitted.
        try:
            observed = load_model_data(options.model, saved, options.input)
        except (FileNotFoundError, ValueError) as error:
            if options.parts:
                raise
            print(f"Skipping the outlier part: {error}")
            parts.remove("outliers")
        else:
            outliers = outlier_table(saved, observed, options.outlier_alpha)
            outlier_groups = _optional_groups(
                subjects,
                subject_info,
                options.group_variable,
            )

    grouped = {"mode_grid", "subject_mode"} | (
        set() if saved.method == "cp" else {"time"}
    )
    if grouped & set(parts):
        keep, dropped = subjects_with_groups(
            subjects,
            subject_info,
            options.group_variable,
        )
        if dropped:
            print(
                f"Leaving out {len(dropped)} subject(s) without a "
                f"'{options.group_variable}' value: {', '.join(dropped)}",
            )
            saved = select_subjects(saved, keep)
            subjects = [subjects[i] for i in keep]

    out_dir.mkdir(parents=True, exist_ok=True)
    _component_weights_table(weights).to_csv(
        out_dir / f"{stem}__component_weights.csv",
        index=False,
    )

    written: list[Path] = []

    def save(fig: matplotlib.figure.Figure, part: str, prefix: str = "") -> None:
        name = f"{prefix}_{stem}__{part}" if prefix else f"{stem}__{part}"
        written.extend(
            save_figure(fig, out_dir / name, options.formats, options.dpi),
        )

    is_cp = saved.method == "cp"
    time_mode: Any = saved.time_mode if is_cp else saved.evolving_states
    time_points: Any = list(saved.timepoints) if is_cp else saved.timepoints

    if "mode_grid" in parts and spatial is not None:
        fig, _ = plot_mode_grid(
            spatial.mode,
            time_mode,
            saved.subject_mode,
            spatial.index_list,
            spatial.csf,
            ~spatial.csf,
            spatial.background,
            spatial.slices[0],
            time_points,
            subjects,
            subject_info,
            options.group_variable,
            page_width=page_width,
            negligible_atol=options.min_subject_coefficient,
        )
        save(fig, "mode_grid")
    if "subject_mode" in parts:
        fig, _ = plot_subject_mode(
            saved.subject_mode,
            subjects,
            subject_info,
            options.group_variable,
            numerical_variables=list(options.numerical),
            categorical_variables=list(options.categorical),
            page_width=page_width,
            min_panel_width=options.min_panel_width,
        )
        save(fig, "subject_mode", options.subject_prefix)
    if "time" in parts:
        if is_cp:
            assert saved.time_mode is not None  # CP models always have one
            fig, _ = plot_time_mode(saved.time_mode, time_points, page_width=page_width)
            save(fig, "time_mode")
        else:
            fig, _, significance = plot_evolving_mode(
                list(time_mode),
                list(saved.timepoints),
                subjects,
                subject_info,
                options.group_variable,
                page_width=page_width,
                min_group_n=options.min_group_n,
                significance_alpha=options.alpha,
                subject_mode=saved.subject_mode,
                negligible_atol=options.min_subject_coefficient,
            )
            save(fig, "evolving_mode")
            significance.to_csv(
                out_dir / f"{stem}__evolving_mode_significance.csv",
                index=False,
            )
    if "spatial" in parts and spatial is not None:
        region_masks = {
            name: mask
            for name, mask in (("parenchyma", ~spatial.csf), ("csf", spatial.csf))
            if mask.any()
        }
        for fig, _, name in plot_spatial_mode(
            spatial.mode,
            spatial.index_list,
            region_masks,
            spatial.background,
            list(spatial.slices),
            page_width=page_width,
        ):
            save(fig, f"spatial_{name}")
    if outliers is not None:
        table, cutoffs = outliers
        fig, _ = plot_outliers(
            table,
            cutoffs,
            outlier_groups,
            options.group_variable,
            page_width=min(page_width, resolve_page_width("single")),
        )
        save(fig, "outliers")
        table.to_csv(out_dir / f"{stem}__outliers.csv", index=False)
        flagged = table.loc[table[list(FLAG_COLUMNS)].any(axis=1), "subject"]
        print(
            f"Possible outliers (alpha {options.outlier_alpha}): "
            f"{', '.join(flagged) if len(flagged) else 'none'}.",
        )
    write_provenance(Path(options.output_dir), "plot_decomposition", options)
    return written


# --------------------------------------------------------------------------
# Group-average maps


class GroupAverages(NamedTuple):
    """`group_averages` output: each group's subject aggregate per column
    (a label of `roi_signal`, or a voxel) and time point.

    `values` is `(n_columns, n_timepoints, n_groups)`, NaN where a group has
    no finite value; `n_subjects` `(n_timepoints, n_groups)` counts the
    subjects scanned at that time point. Voxel input also has each column's
    template coordinates and the template grid.
    """

    values: np.ndarray
    n_subjects: np.ndarray
    timepoints: list[int]
    groups: list[str]
    labels: np.ndarray
    label_index: np.ndarray
    signal: str
    voxel_coords: np.ndarray | None = None
    template_shape: np.ndarray | None = None
    template_affine: np.ndarray | None = None
    template_path: str | None = None


_SIGNAL_LABELS = {"delta_R1": "ΔR1 (1/s)", "ratio": "Signal ratio"}


def _selected_timepoints(
    available: Sequence[int],
    requested: Sequence[int],
    path: Path,
) -> list[int]:
    if not requested:
        return sorted(available)
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(
            f"time point(s) {missing} not in {path}; it has {sorted(available)}",
        )
    return sorted(set(requested))


def group_averages(options: GroupMapPlotOptions) -> GroupAverages:
    """Aggregate each group's subjects per column, at each time point.

    Columns are the labels of `roi_signal.parquet` (its `options.statistic`)
    or the voxels of `voxels.parquet`, read one time point at a time.
    Subjects are aggregated with `aggregate_subjects(options.aggregate,
    options.trim)`; groups are the sorted `options.group_variable` values.
    """
    path = Path(options.input)
    _require(path, "preprocess")
    metadata = read_signal_metadata(path)
    kind = metadata.get("kind")
    template: tuple[Any, ...] = (None, None, None, None)
    if kind == "roi_signal":
        roi = pd.read_parquet(
            path,
            columns=["subject", "time_point", "label", options.statistic],
        )
        roi = roi.rename(columns={"label": "labels", options.statistic: "values"})
        labels = np.sort(roi["labels"].unique())
        label_index = np.zeros(len(labels), dtype=np.int64)

        def read_scans(timepoint: int) -> pd.DataFrame:
            scans = roi[roi["time_point"] == timepoint]
            return scans.assign(label_index=0)

    elif kind == "voxels":
        coords_path = path.with_name(path.name.replace(".parquet", ".coords.parquet"))
        _require(coords_path, "preprocess --store-voxels")
        keys = pd.read_parquet(coords_path, columns=["labels", "label_index"])
        labels = keys["labels"].to_numpy(dtype=np.int64)
        label_index = keys["label_index"].to_numpy(dtype=np.int64)
        template = _voxel_template(path, labels, label_index)

        def read_scans(timepoint: int) -> pd.DataFrame:
            return pq.read_table(
                path,
                columns=["subject", "labels", "label_index", "values"],
                filters=[("time_point", "==", timepoint)],
            ).to_pandas()

    else:
        raise ValueError(
            f"{path} is neither roi_signal.parquet nor voxels.parquet from "
            "`gmri preprocess`",
        )

    scans_index = (
        pq.read_table(path, columns=["subject", "time_point"])
        .to_pandas()
        .drop_duplicates()
    )
    subject_groups = _subject_groups(
        sorted(scans_index["subject"].unique()),
        options.subject_info,
        options.group_variable,
    )
    groups = sorted(set(subject_groups.values()))
    timepoints = _selected_timepoints(
        [int(t) for t in scans_index["time_point"].unique()],
        options.timepoints,
        path,
    )

    columns = pd.MultiIndex.from_arrays([labels, label_index])
    values = np.full((len(columns), len(timepoints), len(groups)), np.nan)
    n_subjects = np.zeros((len(timepoints), len(groups)), dtype=np.int64)
    for t, timepoint in enumerate(timepoints):
        scans = read_scans(timepoint)
        scan_groups = scans["subject"].map(subject_groups)
        for g, group in enumerate(groups):
            group_scans = scans[scan_groups == group]
            present = pd.Index(sorted(group_scans["subject"].unique()))
            n_subjects[t, g] = len(present)
            if present.empty:
                continue
            rows = present.get_indexer(group_scans["subject"])
            cols = columns.get_indexer(
                pd.MultiIndex.from_arrays(
                    [group_scans["labels"], group_scans["label_index"]],
                ),
            )
            known = cols >= 0
            matrix = np.full((len(present), len(columns)), np.nan)
            matrix[rows[known], cols[known]] = group_scans["values"].to_numpy()[known]
            values[:, t, g], _ = aggregate_subjects(
                matrix,
                options.aggregate,
                options.trim,
            )

    return GroupAverages(
        values,
        n_subjects,
        timepoints,
        [str(group) for group in groups],
        labels,
        label_index,
        metadata.get("signal", "delta_R1"),
        *template,
    )


def _value_range(values: np.ndarray) -> tuple[float, float]:
    """5th-95th percentile of the finite, nonzero `values` ((0, 1) if none)."""
    shown = values[np.isfinite(values) & (values != 0)]
    if shown.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(shown, [5, 95])
    return float(low), float(high)


def run_group_map_plots(options: GroupMapPlotOptions) -> list[Path]:
    """`gmri plot group-maps`: one figure per time point and region.

    Each figure draws every group's average map (`group_averages`) as one
    row of `plot_spatial_mode`, with parenchyma and CSF in separate figures.
    A region's colour range is shared by all groups and time points, so
    the figures are comparable. Saved to `<output_dir>/figures/group_maps/
    <input>__<aggregate>__t<time point>__<region>.<ext>`.
    """
    averages = group_averages(options)
    n_columns, n_timepoints, n_groups = averages.values.shape
    spatial = _spatial_inputs(
        averages.values.reshape(n_columns, n_timepoints * n_groups),
        _LabelSpace(
            averages.labels,
            averages.voxel_coords,
            averages.template_shape,
            averages.template_affine,
            averages.template_path,
            "input",
        ),
        options.segmentation,
        options.background,
        options.slices,
        fill_value=np.nan,
    )
    if spatial is None:
        raise ValueError(
            f"{options.input} is per-ROI: group maps need --segmentation, a "
            "label volume in its label space",
        )
    maps = spatial.mode.reshape(-1, n_timepoints, n_groups)
    regions = {
        name: mask
        for name, mask in (("parenchyma", ~spatial.csf), ("csf", spatial.csf))
        if mask.any()
    }
    vlims = {name: _value_range(maps[mask]) for name, mask in regions.items()}
    colorbar_label = _SIGNAL_LABELS.get(averages.signal, averages.signal)
    out_dir = Path(options.output_dir) / "figures" / "group_maps"
    stem = f"{Path(options.input).stem}__{options.aggregate}"

    written: list[Path] = []
    for t, timepoint in enumerate(averages.timepoints):
        row_labels = [
            f"{group} (n={n})"
            for group, n in zip(averages.groups, averages.n_subjects[t])
        ]
        for fig, _, name in plot_spatial_mode(
            maps[:, t, :],
            spatial.index_list,
            regions,
            spatial.background,
            list(spatial.slices),
            page_width=resolve_page_width(options.page_width),
            scale=False,
            row_labels=row_labels,
            colorbar_label=colorbar_label,
            vlims=vlims,
        ):
            written.extend(
                save_figure(
                    fig,
                    out_dir / f"{stem}__t{timepoint}__{name}",
                    options.formats,
                    options.dpi,
                ),
            )
    write_provenance(Path(options.output_dir), "plot_group_maps", options)
    return written
