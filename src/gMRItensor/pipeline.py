"""The `gmri` stages, each run from its own config and communicating via files.

`run_preprocessing` reads images; every other stage reads only files a
previous stage wrote, so each can be run (and rerun) on its own.
"""
import math
import shutil
from collections.abc import Hashable
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import Literal
from typing import NamedTuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from gMRItensor.config import ConfigError
from gMRItensor.config import DecompositionConfig
from gMRItensor.config import FitConfig
from gMRItensor.config import PlottingConfig
from gMRItensor.config import PreprocessingConfig
from gMRItensor.config import ReplicabilityConfig
from gMRItensor.config import TensorConfig
from gMRItensor.decomposition import PARAFAC2Model
from gMRItensor.decomposition import run_CP_decomposition_repeated
from gMRItensor.decomposition import run_PARAFAC2_decomposition_repeated
from gMRItensor.decomposition import setup_backend
from gMRItensor.group_statistics import compare_roi_groups
from gMRItensor.group_statistics import load_roi_statistics
from gMRItensor.group_statistics import resolve_subject_groups
from gMRItensor.group_statistics import summarize_roi_statistics
from gMRItensor.jobs import collect
from gMRItensor.jobs import DirectoryStore
from gMRItensor.jobs import FitTask
from gMRItensor.jobs import GroupSummary
from gMRItensor.jobs import plan_replicability
from gMRItensor.jobs import plan_restarts
from gMRItensor.jobs import run_tasks
from gMRItensor.model_io import save_decomposition
from gMRItensor.model_io import SavedDecomposition
from gMRItensor.plotting.roi_evolution import figure_path
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_panels
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_rows
from gMRItensor.plotting.utils import save_figure
from gMRItensor.preprocessing import compute_roi_scaling
from gMRItensor.preprocessing import load_tensor_from_parquet
from gMRItensor.preprocessing import PreprocessedPaths
from gMRItensor.preprocessing import scale_tensor
from gMRItensor.preprocessing import write_preprocessed_data
from gMRItensor.replicability import CrossValidationEngine
from gMRItensor.replicability import evaluate_replicability_multiproc
from gMRItensor.replicability import HalfHalfEngine
from gMRItensor.replicability import ReplicabilityEngine
from gMRItensor.roi_groups import resolve_roi_groups

matplotlib.use("Agg")

MANIFEST_COLUMNS = (
    "subject",
    "time_point",
    "baseline_path",
    "post_injection_path",
    "mask_path",
    "segmentation_path",
)
_PATH_COLUMNS = MANIFEST_COLUMNS[2:]
_AGGREGATIONS = {"median": np.nanmedian, "mean": np.nanmean, "voxel": None}


def copy_config(config_source: Path, output_dir: Path, stage: str) -> Path:
    """Copy a stage's config to `<output_dir>/<stage>.yaml` for provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{stage}.yaml"
    if target.resolve() != config_source.resolve():
        shutil.copyfile(config_source, target)
    return target


def read_manifest(config: PreprocessingConfig) -> list[dict[str, Any]]:
    """Turn the manifest CSV into `write_preprocessed_data` arguments.

    Paths are relative to the manifest. Raises `ConfigError` for missing
    columns, non-integer time points, duplicate `(subject, time_point)` rows
    or image files that do not exist -- before any image is read.
    """
    manifest = pd.read_csv(config.manifest, dtype={"subject": str})
    missing_columns = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing_columns:
        raise ConfigError(f"{config.manifest}: missing column(s) {missing_columns}")

    values = manifest[list(MANIFEST_COLUMNS)]
    blank = values.isna() | values.astype(str).apply(lambda c: c.str.strip() == "")
    if blank.to_numpy().any():
        cells = [
            (int(row), column)
            for column in MANIFEST_COLUMNS
            for row in manifest.index[blank[column]]
        ]
        raise ConfigError(
            f"{config.manifest}: empty value(s) at (row, column): {cells[:5]}",
        )

    time_points = pd.to_numeric(manifest["time_point"], errors="coerce")
    bad = manifest[~np.isfinite(time_points) | (time_points != time_points.round())]
    if not bad.empty:
        raise ConfigError(
            f"{config.manifest}: time_point must be an integer, rows "
            f"{bad.index.tolist()}: {bad['time_point'].tolist()}",
        )
    manifest["time_point"] = time_points.astype(int)

    duplicated = manifest.duplicated(["subject", "time_point"], keep=False)
    if duplicated.any():
        pairs = sorted(
            set(manifest.loc[duplicated, ["subject", "time_point"]].itertuples(False)),
        )
        raise ConfigError(
            f"{config.manifest}: duplicate (subject, time_point) rows: {pairs}",
        )

    base = config.manifest.parent
    for column in _PATH_COLUMNS:
        manifest[column] = [base / value for value in manifest[column]]
    missing_files = sorted(
        {str(path) for c in _PATH_COLUMNS for path in manifest[c] if not path.exists()},
    )
    if missing_files:
        raise ConfigError(
            f"{config.manifest}: {len(missing_files)} image file(s) not found, "
            f"e.g. {missing_files[:5]}",
        )

    return [
        {
            **{column: row[column] for column in _PATH_COLUMNS},
            "signal_type": config.signal_type,
            "func": _AGGREGATIONS[config.aggregation],
            "subject": row["subject"],
            "time_point": int(row["time_point"]),
        }
        for _, row in manifest.iterrows()
    ]


def run_preprocessing(config: PreprocessingConfig) -> PreprocessedPaths:
    """`gmri preprocess`: images -> `<output_dir>/data/` tracer + ROI tables."""
    args_list = read_manifest(config)
    regions = config.regions
    roi_groups = resolve_roi_groups(regions.presets, regions.custom, regions.csf_offset)
    paths = write_preprocessed_data(
        args_list,
        config.output_dir,
        roi_groups=roi_groups,
        relaxivity=config.relaxivity,
        time_unit=config.time_unit,
        n_procs=config.n_procs,
    )
    copy_config(config.source, config.output_dir, "preprocessing")
    return paths


def _require(path: Path, stage: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `gmri {stage}` first")


def grid_pages(
    rois: Sequence[str],
    layout: str,
    n_rows: int,
    n_cols: int,
) -> list[tuple[str, ...]]:
    """Split a grid's ROIs into pages: `n_rows` per rows page, `n_rows * n_cols`
    per panels page."""
    per_page = n_rows if layout == "rows" else n_rows * n_cols
    pages = []
    for start in range(0, len(rois), per_page):
        end = start + per_page
        pages.append(tuple(rois[start:end]))
    return pages


def _draw(
    layout: str,
    rois: Sequence[str],
    statistic: str,
    tables: tuple[pd.DataFrame, pd.DataFrame],
    stats: pd.DataFrame,
    page_width: str | float,
    n_rows: int = 1,
    n_cols: int = 1,
    sharey: bool = False,
) -> matplotlib.figure.Figure:
    summary, significance = tables
    if layout == "rows":
        fig, _ = plot_roi_evolution_rows(
            summary,
            stats,
            significance,
            rois,
            statistic,
            page_width=page_width,
        )
    else:
        fig, _ = plot_roi_evolution_panels(
            summary,
            significance,
            rois,
            statistic,
            n_rows,
            n_cols,
            page_width=page_width,
            sharey=sharey,
        )
    return fig


def run_plotting(config: PlottingConfig) -> list[Path]:
    """`gmri plot`: ROI statistics -> group tables -> figures.

    Summary and significance tables are written to `<output_dir>/roi_analysis/`
    for every statistic a figure or grid uses, then figures are drawn from
    those saved tables. Every configured ROI is checked against the file
    before anything is written. Returns the written figure paths.
    """
    _require(config.roi_statistics, "preprocess")
    subject_info = pd.read_csv(config.subject_info, dtype={"subjects": str})
    used = {roi for figure in config.figures for roi in figure.rois}
    used |= {roi for grid in config.grids for roi in grid.rois}
    rois = sorted(used)
    stats = load_roi_statistics(
        config.roi_statistics,
        subject_info,
        config.group_variable,
        rois=rois,
    )

    tables_dir = config.output_dir / "roi_analysis"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for statistic in config.statistics:
        summary_path = tables_dir / f"summary__{statistic}.parquet"
        significance_path = tables_dir / f"significance__{statistic}.csv"
        summarize_roi_statistics(stats, statistic).to_parquet(summary_path, index=False)
        compare_roi_groups(
            stats,
            statistic,
            min_group_n=config.min_group_n,
            alpha=config.alpha,
        ).to_csv(significance_path, index=False)
        tables[statistic] = (
            pd.read_parquet(summary_path),
            # Label ROIs are named str(id): keep them strings, not ints.
            pd.read_csv(significance_path, dtype={"roi": str}),
        )

    written: list[Path] = []

    def save(fig: matplotlib.figure.Figure, stem: Path) -> None:
        written.extend(save_figure(fig, stem, config.formats, config.dpi))

    for figure in config.figures:
        for statistic in figure.statistics:
            for roi in figure.rois:
                fig = _draw(
                    figure.layout,
                    [roi],
                    statistic,
                    tables[statistic],
                    stats,
                    figure.page_width,
                )
                save(
                    fig,
                    figure_path(config.output_dir, None, roi, statistic, figure.layout),
                )

    for grid in config.grids:
        pages = grid_pages(grid.rois, grid.layout, grid.n_rows, grid.n_cols)
        for statistic in grid.statistics:
            for page_number, page in enumerate(pages, start=1):
                fig = _draw(
                    grid.layout,
                    page,
                    statistic,
                    tables[statistic],
                    stats,
                    grid.page_width,
                    grid.n_rows,
                    grid.n_cols,
                    grid.sharey,
                )
                save(
                    fig,
                    figure_path(
                        config.output_dir,
                        grid.name,
                        None,
                        statistic,
                        grid.layout,
                        page=page_number if len(pages) > 1 else None,
                    ),
                )

    copy_config(config.source, config.output_dir, "plotting")
    return written


class DecompositionInput(NamedTuple):
    """`load_tensor_from_parquet` output as torch, plus the scaling applied."""

    data: torch.Tensor | list[torch.Tensor]
    subjects: np.ndarray
    timepoints: np.ndarray | list[np.ndarray]
    labels: np.ndarray
    label_index: np.ndarray
    scale_mean: np.ndarray | None
    scale_std: np.ndarray | None
    centered: bool


def load_decomposition_input(
    path: Path,
    method: str,
    tensor: TensorConfig,
) -> DecompositionInput:
    """Load the tracer parquet as a CP tensor or PARAFAC2 slices, optionally
    centered and/or scaled per label over all subjects and time points."""
    _require(path, "preprocess")
    data, subjects, timepoints, labels, label_index = load_tensor_from_parquet(
        path,
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
    return DecompositionInput(
        torch_data,
        subjects,
        timepoints,
        labels,
        label_index,
        mean,
        std,
        tensor.center,
    )


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _method(method: str) -> Literal["CP", "PARAFAC2"]:
    return "CP" if method == "cp" else "PARAFAC2"


def _fit_options(fit: FitConfig, method: str) -> dict[str, Any]:
    """`run_*_decomposition_repeated` options from `fit` (also what
    `jobs.run_tasks`/`jobs.collect` take; restart-loop ones are ignored there).

    CP's non-negativity is all-or-nothing (`non_negative`); PARAFAC2 takes
    the modes (`nn_modes`, `"auto"` = the solver's default).
    """
    non_negativity: dict[str, Any] = (
        {"non_negative": fit.non_negative_modes is not None}
        if method == "cp"
        else {"nn_modes": fit.non_negative_modes}
    )
    return {
        "init_repeats": fit.restarts,
        "max_iter": fit.max_iter,
        "tolerance": fit.tolerance,
        "progress_bar": False,  # quiet batch logs unless fit.options asks
        **non_negativity,
        **fit.options,
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


def _saved_decomposition(
    method: str,
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
    config: DecompositionConfig,
    fits: list[SavedDecomposition],
) -> list[Path]:
    """`rank_<r>.h5` per fit, `fits.csv` and the config copy."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for saved in fits:
        path = config.output_dir / f"rank_{saved.rank}.h5"
        save_decomposition(path, saved)
        written.append(path)
    pd.DataFrame(
        {"rank": [saved.rank for saved in fits], "error": [s.error for s in fits]},
    ).to_csv(config.output_dir / "fits.csv", index=False)
    copy_config(config.source, config.output_dir, "decomposition")
    return written


def run_decomposition(config: DecompositionConfig) -> list[Path]:
    """`gmri decompose run`: one fit per rank, in this process.

    Writes `rank_<r>.h5`, `fits.csv` and `decomposition.yaml`; returns the
    model paths in `ranks` order.
    """
    data = load_decomposition_input(config.input, config.method, config.tensor)
    device = setup_backend()
    options = {
        "restart_procs": config.fit.restart_procs,
        "device": device,
        **_fit_options(config.fit, config.method),
    }
    fits = []
    for rank in config.ranks:
        if config.method == "cp":
            weights, factors, error = run_CP_decomposition_repeated(
                data.data,
                rank,
                **options,
            )
            model: PARAFAC2Model | tuple[Any, list[Any]] = (weights, factors)
        else:
            model, error = run_PARAFAC2_decomposition_repeated(
                data.data,
                rank,
                **options,
            )
        fits.append(_saved_decomposition(config.method, rank, model, error, data))
    return _write_decompositions(config, fits)


def _decomposition_plan(
    config: DecompositionConfig,
    n_subjects: int,
) -> list[_RankTask]:
    return [
        (rank, task)
        for rank in config.ranks
        for task in plan_restarts(n_subjects, config.fit.restarts)
    ]


def plan_decomposition(config: DecompositionConfig) -> int:
    """`gmri decompose plan`: number of `run --job` jobs to submit."""
    data = load_decomposition_input(config.input, config.method, config.tensor)
    plan = _decomposition_plan(config, len(data.subjects))
    return _n_jobs(plan, config.distributed.tasks_per_job)


def run_decomposition_job(config: DecompositionConfig, job: int) -> int:
    """`gmri decompose run --job N`: fit job `N`'s restarts into the store.

    Restarts already in the store are skipped, so a re-queued job resumes.
    Returns the number of tasks in the job.
    """
    data = load_decomposition_input(config.input, config.method, config.tensor)
    plan = _decomposition_plan(config, len(data.subjects))
    tasks = _job_tasks(plan, job, config.distributed.tasks_per_job)
    _run_job_tasks(
        tasks,
        data,
        config.method,
        config.store_dir,
        config.fit.restart_procs,
        _fit_options(config.fit, config.method),
    )
    return len(tasks)


def collect_decomposition(config: DecompositionConfig) -> list[Path]:
    """`gmri decompose collect`: best restart per rank -> the same files as
    `run_decomposition`."""
    data = load_decomposition_input(config.input, config.method, config.tensor)
    options = _fit_options(config.fit, config.method)
    fits = []
    for rank in config.ranks:
        plan = plan_restarts(len(data.subjects), config.fit.restarts)
        summary = _collect_rank(plan, config.store_dir, rank, config.method, options)
        best = summary["full"].best
        if best is None or best.model is None or best.error is None:
            raise ValueError(f"rank {rank}: no restart converged")
        fits.append(
            _saved_decomposition(config.method, rank, best.model, best.error, data),
        )
    return _write_decompositions(config, fits)


def _stratification(
    config: ReplicabilityConfig,
    subjects: np.ndarray,
) -> torch.Tensor | None:
    """Integer codes of `stratify_by`, in tensor-subject order."""
    if config.stratify_by is None or config.subject_info is None:
        return None
    subject_info = pd.read_csv(config.subject_info, dtype={"subjects": str})
    groups = resolve_subject_groups(list(subjects), subject_info, config.stratify_by)
    codes, _ = pd.factorize(pd.Series(groups))
    return torch.as_tensor(codes)


def _engine(config: ReplicabilityConfig) -> ReplicabilityEngine:
    if config.engine == "cv":
        return CrossValidationEngine(
            splits=int(config.splits or 3),
            repeats=config.repeats,
            seed=config.seed,
        )
    return HalfHalfEngine(repeats=config.repeats, seed=config.seed)


def _score_rows(
    config: ReplicabilityConfig,
    rank: int,
    scores: Sequence[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    rows = []
    for score in scores:
        if config.engine == "cv":
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
    config: ReplicabilityConfig,
    rows: list[dict[str, Any]],
) -> Path:
    columns = (
        ["rank", "fold_i", "fold_j", "n_common", "fms"]
        if config.engine == "cv"
        else ["rank", "split", "fms"]
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "replicability.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    copy_config(config.source, config.output_dir, "replicability")
    return path


def run_replicability(config: ReplicabilityConfig) -> Path:
    """`gmri replicability run`: factor match scores per rank, in this process.

    The (optionally scaled) tensor is split by the engine; each rank gets a
    fresh engine with `seed`, so ranks see the same splits.
    """
    data = load_decomposition_input(config.input, config.method, config.tensor)
    stratification = _stratification(config, data.subjects)
    setup_backend()
    rows = []
    for rank in config.ranks:
        scores = evaluate_replicability_multiproc(
            _engine(config),
            data.data,
            rank,
            method=_method(config.method),
            stratification=stratification,
            n_procs=config.n_procs,
            **_fit_options(config.fit, config.method),
        )
        rows.extend(_score_rows(config, rank, scores))
    return _write_replicability(config, rows)


def _replicability_plans(
    config: ReplicabilityConfig,
    data: DecompositionInput,
) -> dict[int, list[FitTask]]:
    stratification = _stratification(config, data.subjects)
    return {
        rank: plan_replicability(
            _engine(config),
            len(data.subjects),
            config.fit.restarts,
            stratification,
        )
        for rank in config.ranks
    }


def _flatten(plans: dict[int, list[FitTask]]) -> list[_RankTask]:
    return [(rank, task) for rank, plan in plans.items() for task in plan]


def plan_replicability_jobs(config: ReplicabilityConfig) -> int:
    """`gmri replicability plan`: number of `run --job` jobs to submit."""
    data = load_decomposition_input(config.input, config.method, config.tensor)
    plan = _flatten(_replicability_plans(config, data))
    return _n_jobs(plan, config.distributed.tasks_per_job)


def run_replicability_job(config: ReplicabilityConfig, job: int) -> int:
    """`gmri replicability run --job N`: fit job `N`'s split/fold restarts.

    Returns the number of tasks in the job.
    """
    data = load_decomposition_input(config.input, config.method, config.tensor)
    plan = _flatten(_replicability_plans(config, data))
    tasks = _job_tasks(plan, job, config.distributed.tasks_per_job)
    _run_job_tasks(
        tasks,
        data,
        config.method,
        config.store_dir,
        config.n_procs,
        _fit_options(config.fit, config.method),
    )
    return len(tasks)


def collect_replicability(config: ReplicabilityConfig) -> Path:
    """`gmri replicability collect`: score the gathered fits ->
    `replicability.csv`, as `run_replicability` writes it."""
    data = load_decomposition_input(config.input, config.method, config.tensor)
    options = _fit_options(config.fit, config.method)
    rows = []
    for rank, plan in _replicability_plans(config, data).items():
        summaries = _collect_rank(plan, config.store_dir, rank, config.method, options)
        rows.extend(_score_rows(config, rank, _engine(config).compute_fms(summaries)))
    return _write_replicability(config, rows)
