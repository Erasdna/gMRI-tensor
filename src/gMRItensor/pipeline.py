"""The `gmri` stages, each run from its own config and communicating via files.

`run_preprocessing` reads images; every other stage reads only files a
previous stage wrote, so each can be run (and rerun) on its own.
"""
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from gMRItensor.config import ConfigError
from gMRItensor.config import PlottingConfig
from gMRItensor.config import PreprocessingConfig
from gMRItensor.group_statistics import compare_roi_groups
from gMRItensor.group_statistics import load_roi_statistics
from gMRItensor.group_statistics import summarize_roi_statistics
from gMRItensor.plotting.roi_evolution import figure_path
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_panels
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_rows
from gMRItensor.plotting.utils import save_figure
from gMRItensor.preprocessing import PreprocessedPaths
from gMRItensor.preprocessing import write_preprocessed_data
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

    time_points = pd.to_numeric(manifest["time_point"], errors="coerce")
    bad = manifest[time_points.isna() | (time_points != time_points.round())]
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
            pd.read_csv(significance_path),
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
