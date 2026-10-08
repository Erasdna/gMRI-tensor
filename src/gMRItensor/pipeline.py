"""The `gmri` stages, each run from its own config and communicating via files.

`run_preprocessing` reads images; every other stage reads only files a
previous stage wrote, so each can be run (and rerun) on its own.
"""
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gMRItensor.config import ConfigError
from gMRItensor.config import PreprocessingConfig
from gMRItensor.preprocessing import PreprocessedPaths
from gMRItensor.preprocessing import write_preprocessed_data
from gMRItensor.roi_groups import resolve_roi_groups

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
