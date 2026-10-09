"""The `gmri` stages, each taking its `gMRItensor.options` dataclass.

`run_preprocessing` reads images; every other stage reads only files a
previous stage wrote, so each can be run (and rerun) on its own -- from
the `gmri` command line or from a script.
"""
import dataclasses
import json
import math
from collections.abc import Hashable
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import cast
from typing import Literal
from typing import NamedTuple

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from gMRItensor.decomposition import PARAFAC2Model
from gMRItensor.decomposition import run_CP_decomposition_repeated
from gMRItensor.decomposition import run_PARAFAC2_decomposition_repeated
from gMRItensor.decomposition import setup_backend
from gMRItensor.group_statistics import compare_groups_over_time
from gMRItensor.group_statistics import resolve_subject_groups
from gMRItensor.group_statistics import significance_summary
from gMRItensor.group_statistics import summarize_groups_over_time
from gMRItensor.jobs import collect
from gMRItensor.jobs import DirectoryStore
from gMRItensor.jobs import FitTask
from gMRItensor.jobs import GroupSummary
from gMRItensor.jobs import plan_replicability
from gMRItensor.jobs import plan_restarts
from gMRItensor.jobs import run_tasks
from gMRItensor.model_io import load_decomposition
from gMRItensor.model_io import save_decomposition
from gMRItensor.model_io import SavedDecomposition
from gMRItensor.options import CONCENTRATION_STATISTICS
from gMRItensor.options import DecompositionOptions
from gMRItensor.options import DecompositionPlotOptions
from gMRItensor.options import FitOptions
from gMRItensor.options import options_from_json
from gMRItensor.options import options_to_json
from gMRItensor.options import PreprocessingOptions
from gMRItensor.options import ReplicabilityOptions
from gMRItensor.options import StatisticsPlotOptions
from gMRItensor.options import TensorOptions
from gMRItensor.plotting.evolving_mode import plot_evolving_mode
from gMRItensor.plotting.mode_grid import plot_mode_grid
from gMRItensor.plotting.roi_evolution import figure_path
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_panels
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_rows
from gMRItensor.plotting.spatial_mode import plot_spatial_mode
from gMRItensor.plotting.subject_mode import plot_subject_mode
from gMRItensor.plotting.time_mode import plot_time_mode
from gMRItensor.plotting.utils import expand_roi_mode_to_voxels
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
from gMRItensor.roi_groups import add_concentration
from gMRItensor.roi_groups import aggregate_roi_signal
from gMRItensor.roi_groups import get_roi_presets
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


def write_provenance(output_dir: Path, command: str, options: Any) -> Path:
    """Record a stage's settings as `<output_dir>/<command>.json`."""
    from gMRItensor import __version__

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{command}.json"
    record = {"gMRItensor": __version__, "options": dataclasses.asdict(options)}
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


def run_statistics_plots(options: StatisticsPlotOptions) -> StatisticsPlotResult:
    """`gmri plot statistics`: ROI signal -> group tables, summary, figures.

    Aggregates the requested regions and labels (`aggregate_roi_signal`),
    adds concentrations if a concentration statistic is requested, and
    writes to `<output_dir>/roi_analysis/`: `roi_statistics.parquet`, one
    `summary__<stat>.parquet` and `significance__<stat>.csv` per statistic,
    and `summary.csv`. Then one figure per ROI and statistic, drawn from
    those saved tables, under `<output_dir>/figures/roi/`.
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
    subjects = sorted(frame["subject"].unique())
    groups = resolve_subject_groups(
        subjects,
        _read_subject_info(options.subject_info),
        options.group_variable,
    )
    frame["group"] = frame["subject"].map(dict(zip(subjects, groups)))
    frame = frame.rename(columns={"time_point": "timepoint"})
    categories = sorted(frame["group"].unique())

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
        tables[statistic] = (
            pd.read_parquet(summary_path),
            # Label ROIs are named str(id): keep them strings, not ints.
            pd.read_csv(significance_path, dtype={"roi": str}),
        )
    summary = significance_summary(tables)
    summary.to_csv(tables_dir / "summary.csv", index=False)

    written: list[Path] = []
    for statistic, (summary_table, significance) in tables.items():
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
                        None,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates of each tensor column, plus the template shape/affine."""
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
    points.
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
    template: tuple[Any, Any, Any] = (None, None, None)
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


def _method(method: str) -> Literal["CP", "PARAFAC2"]:
    return "CP" if method == "cp" else "PARAFAC2"


def _fit_options(fit: FitOptions, method: str) -> dict[str, Any]:
    """`run_*_decomposition_repeated` options from `fit` (also what
    `jobs.run_tasks`/`jobs.collect` take; restart-loop ones are ignored there).

    CP's non-negativity is all-or-nothing (`non_negative`); PARAFAC2 takes
    the modes (`nn_modes`, `"auto"` = the solver's default) and `solver`.
    """
    specific: dict[str, Any]
    if method == "cp":
        specific = {"non_negative": fit.non_negative_modes is not None}
    else:
        specific = {"nn_modes": fit.non_negative_modes}
        if fit.solver is not None:
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


def _n_jobs(plan: Sequence[_RankTask], tasks_per_job: int) -> int:
    return math.ceil(len(plan) / tasks_per_job)


def _job_tasks(
    plan: Sequence[_RankTask],
    job: int,
    tasks_per_job: int,
) -> list[_RankTask]:
    total = _n_jobs(plan, tasks_per_job)
    if not 0 <= job < total:
        raise ValueError(f"job {job} is out of range: the plan has {total} jobs")
    start = job * tasks_per_job
    stop = start + tasks_per_job
    return list(plan[start:stop])


def _run_job_tasks(
    tasks: Sequence[_RankTask],
    data: DecompositionInput,
    method: str,
    store_dir: Path,
    n_procs: int,
    options: dict[str, Any],
) -> None:
    """Fit `tasks` rank by rank into `<store_dir>/rank_<r>/`."""
    options = dict(options)
    progress_bar = bool(options.pop("progress_bar", False))
    setup_backend()
    for rank, rank_tasks in groupby(tasks, key=lambda item: item[0]):
        run_tasks(
            [task for _, task in rank_tasks],
            data.data,
            rank,
            _method(method),
            DirectoryStore(store_dir / f"rank_{rank}"),
            n_procs=n_procs,
            progress_bar=progress_bar,
            **options,
        )


def _collect_rank(
    plan: Sequence[FitTask],
    store_dir: Path,
    rank: int,
    method: str,
    options: dict[str, Any],
) -> dict[Hashable, GroupSummary]:
    """Gather one rank's restarts; refuse if any restart has no result yet."""
    summaries = collect(
        plan,
        DirectoryStore(store_dir / f"rank_{rank}"),
        method=_method(method),
        **options,
    )
    missing = sum(len(summary.missing) for summary in summaries.values())
    if missing:
        raise ValueError(
            f"rank {rank}: {missing} of {len(plan)} restarts missing from "
            f"{store_dir}; run every `run --job` first",
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
    options = options_from_json(json.loads(path.read_text()))
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
    model: PARAFAC2Model | tuple[Any, list[Any]],
    error: float,
    data: DecompositionInput,
) -> SavedDecomposition:
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
    }
    if isinstance(model, PARAFAC2Model):
        return SavedDecomposition(
            method="parafac2",
            weights=_numpy(model.weights),
            subject_mode=_numpy(model.subject_mode),
            label_mode=_numpy(model.label_mode),
            evolving_states=[_numpy(state) for state in model.evolving_states],
            **common,
        )
    weights, factors = model
    subject_mode, time_mode, label_mode = (_numpy(f) for f in factors)
    return SavedDecomposition(
        method="cp",
        weights=_numpy(weights),
        subject_mode=subject_mode,
        label_mode=label_mode,
        time_mode=time_mode,
        **common,
    )


def _write_decompositions(
    options: DecompositionOptions,
    fits: list[SavedDecomposition],
) -> list[Path]:
    """`rank_<r>.h5` per fit, `fits.csv` and the provenance record."""
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for saved in fits:
        path = output_dir / f"rank_{saved.rank}.h5"
        save_decomposition(path, saved)
        written.append(path)
    pd.DataFrame(
        {"rank": [s.rank for s in fits], "error": [s.error for s in fits]},
    ).to_csv(output_dir / "fits.csv", index=False)
    write_provenance(output_dir, "decompose", options)
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

    Writes `rank_<r>.h5`, `fits.csv` and `decompose.json`; returns the model
    paths in `ranks` order.
    """
    data = _decomposition_input(options)
    device = setup_backend()
    fit_kwargs = {
        "restart_procs": options.fit.restart_procs,
        "device": device,
        **_fit_options(options.fit, options.method),
    }
    fits = []
    for rank in options.ranks:
        if options.method == "cp":
            weights, factors, error = run_CP_decomposition_repeated(
                data.data,
                rank,
                **fit_kwargs,
            )
            model: PARAFAC2Model | tuple[Any, list[Any]] = (weights, factors)
        else:
            model, error = run_PARAFAC2_decomposition_repeated(
                data.data,
                rank,
                **fit_kwargs,
            )
        fits.append(_saved_decomposition(rank, model, error, data))
    return _write_decompositions(options, fits)


def _decomposition_plan(
    options: DecompositionOptions,
    n_subjects: int,
) -> list[_RankTask]:
    return [
        (rank, task)
        for rank in options.ranks
        for task in plan_restarts(n_subjects, options.fit.restarts)
    ]


def plan_decomposition(options: DecompositionOptions) -> int:
    """`gmri decompose plan`: save `plan.json`; return the number of jobs."""
    data = _decomposition_input(options)
    plan = _decomposition_plan(options, len(data.subjects))
    _write_plan(options)
    return _n_jobs(plan, options.distributed.tasks_per_job)


def run_decomposition_job(output_dir: Path, job: int) -> int:
    """`gmri decompose run --job N`: fit job `N`'s restarts into the store.

    Settings come from `plan.json`. Restarts already in the store are
    skipped, so a re-queued job resumes. Returns the number of tasks.
    """
    options = load_plan(output_dir, DecompositionOptions)
    data = _decomposition_input(options)
    plan = _decomposition_plan(options, len(data.subjects))
    tasks = _job_tasks(plan, job, options.distributed.tasks_per_job)
    _run_job_tasks(
        tasks,
        data,
        options.method,
        options.store_dir,
        options.fit.restart_procs,
        _fit_options(options.fit, options.method),
    )
    return len(tasks)


def collect_decomposition(output_dir: Path) -> list[Path]:
    """`gmri decompose collect`: best restart per rank -> the same files as
    `run_decomposition`."""
    options = load_plan(output_dir, DecompositionOptions)
    data = _decomposition_input(options)
    fit_kwargs = _fit_options(options.fit, options.method)
    fits = []
    for rank in options.ranks:
        plan = plan_restarts(len(data.subjects), options.fit.restarts)
        summary = _collect_rank(
            plan,
            options.store_dir,
            rank,
            options.method,
            fit_kwargs,
        )
        best = summary["full"].best
        if best is None or best.model is None or best.error is None:
            raise ValueError(f"rank {rank}: no restart converged")
        fits.append(_saved_decomposition(rank, best.model, best.error, data))
    return _write_decompositions(options, fits)


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
                    "rank": rank,
                    "fold_i": fold_i,
                    "fold_j": fold_j,
                    "n_common": len(common),
                    "fms": float(fms),
                },
            )
        else:
            split, fms = score
            rows.append({"rank": rank, "split": split, "fms": float(fms)})
    return rows


def _write_replicability(
    options: ReplicabilityOptions,
    rows: list[dict[str, Any]],
) -> Path:
    columns = (
        ["rank", "fold_i", "fold_j", "n_common", "fms"]
        if options.engine == "cv"
        else ["rank", "split", "fms"]
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
    fresh engine with `seed`, so ranks see the same splits.
    """
    data = _decomposition_input(options)
    stratification = _stratification(options, data.subjects)
    setup_backend()
    rows = []
    for rank in options.ranks:
        scores = evaluate_replicability_multiproc(
            _engine(options),
            data.data,
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


def plan_replicability_jobs(options: ReplicabilityOptions) -> int:
    """`gmri replicability plan`: save `plan.json`; return the number of jobs."""
    data = _decomposition_input(options)
    plan = _flatten(_replicability_plans(options, data))
    _write_plan(options)
    return _n_jobs(plan, options.distributed.tasks_per_job)


def run_replicability_job(output_dir: Path, job: int) -> int:
    """`gmri replicability run --job N`: fit job `N`'s split/fold restarts,
    with the settings from `plan.json`. Returns the number of tasks."""
    options = load_plan(output_dir, ReplicabilityOptions)
    data = _decomposition_input(options)
    plan = _flatten(_replicability_plans(options, data))
    tasks = _job_tasks(plan, job, options.distributed.tasks_per_job)
    _run_job_tasks(
        tasks,
        data,
        options.method,
        options.store_dir,
        options.n_procs,
        _fit_options(options.fit, options.method),
    )
    return len(tasks)


def collect_replicability(output_dir: Path) -> Path:
    """`gmri replicability collect`: score the gathered fits ->
    `replicability.csv`, as `run_replicability` writes it."""
    options = load_plan(output_dir, ReplicabilityOptions)
    data = _decomposition_input(options)
    fit_kwargs = _fit_options(options.fit, options.method)
    rows = []
    for rank, plan in _replicability_plans(options, data).items():
        summaries = _collect_rank(
            plan,
            options.store_dir,
            rank,
            options.method,
            fit_kwargs,
        )
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


def _canonical_data(path: Path) -> np.ndarray:
    image = cast(Nifti1Image, nib.as_closest_canonical(nib.load(path)))
    return np.asarray(image.get_fdata())


def _spatial_inputs(
    saved: SavedDecomposition,
    options: DecompositionPlotOptions,
) -> _Spatial | None:
    """Voxel layout of `saved.label_mode`; None for an ROI model without a
    segmentation to map onto."""
    segmentation = (
        np.rint(_canonical_data(options.segmentation)).astype(np.int64)
        if options.segmentation is not None
        else None
    )
    if saved.voxel_coords is None:
        if segmentation is None:
            return None
        if not np.isin(saved.labels, segmentation).any():
            first = list(saved.labels[:5])
            raise ValueError(
                f"{options.segmentation} contains none of the model's labels "
                f"{first}...; it must be in the decomposition's label space",
            )
        mode, index_list = expand_roi_mode_to_voxels(
            saved.label_mode,
            saved.labels,
            segmentation,
        )
        voxel_labels = segmentation[tuple(index_list.T)]
        shape = segmentation.shape
    else:
        mode, index_list = saved.label_mode, saved.voxel_coords
        voxel_labels = saved.labels
        shape = tuple(int(n) for n in np.asarray(saved.template_shape))

    if options.background is not None:
        background = _canonical_data(options.background)
        if background.shape[:3] != tuple(shape):
            raise ValueError(
                f"background {options.background} has shape {background.shape}, "
                f"but the template is {tuple(shape)}",
            )
    elif segmentation is not None:
        background = (segmentation > 0).astype(float)
    else:
        background = np.zeros(shape)
        background[tuple(index_list.T)] = 1.0

    csf = np.isin(voxel_labels, get_roi_presets()["all_csf"])
    slices = options.slices or tuple(int(i) for i in np.rint(index_list.mean(axis=0)))
    return _Spatial(mode, index_list, csf, background, slices)  # type: ignore[arg-type]


def run_decomposition_plots(options: DecompositionPlotOptions) -> list[Path]:
    """`gmri plot decomposition`: one saved model -> figures.

    Draws the selected parts (`DecompositionPlotOptions.selected_parts`) to
    `<output_dir>/figures/decomposition/<model>__<part>.<ext>`. Spatial
    parts of an ROI model need a segmentation: without one they are skipped
    with a note if no part was chosen explicitly, and raise otherwise.
    """
    saved = load_decomposition(options.model)
    subject_info = _read_subject_info(options.subject_info)
    subjects = [str(s) for s in saved.subjects]
    page_width = resolve_page_width(options.page_width)
    out_dir = Path(options.output_dir) / "figures" / "decomposition"
    stem = Path(options.model).stem
    parts = list(options.selected_parts())

    spatial = None
    if {"mode_grid", "spatial"} & set(parts):
        spatial = _spatial_inputs(saved, options)
        if spatial is None:
            message = (
                "the spatial parts (mode_grid, spatial) of an ROI model need "
                "--segmentation, a label volume in the decomposition's label space"
            )
            if options.parts:
                raise ValueError(message)
            print(f"Skipping {message}.")
            parts = [p for p in parts if p not in ("mode_grid", "spatial")]

    written: list[Path] = []

    def save(fig: matplotlib.figure.Figure, part: str) -> None:
        written.extend(
            save_figure(fig, out_dir / f"{stem}__{part}", options.formats, options.dpi),
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
        )
        save(fig, "mode_grid")
    if "subject_mode" in parts:
        fig, _ = plot_subject_mode(
            saved.subject_mode,
            subjects,
            subject_info,
            options.group_variable,
            list(options.covariates),
            page_width=page_width,
        )
        save(fig, "subject_mode")
    if "time" in parts:
        if is_cp:
            fig, _ = plot_time_mode(saved.time_mode, time_points, page_width=page_width)
            save(fig, "time_mode")
        else:
            fig, _, significance = plot_evolving_mode(
                list(saved.evolving_states or []),
                list(saved.timepoints),
                subjects,
                subject_info,
                options.group_variable,
                page_width=page_width,
                min_group_n=options.min_group_n,
                significance_alpha=options.alpha,
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
    write_provenance(Path(options.output_dir), "plot_decomposition", options)
    return written
