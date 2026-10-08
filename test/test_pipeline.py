from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from gMRItensor import preprocessing
from gMRItensor.config import ConfigError
from gMRItensor.config import load_preprocessing_config
from gMRItensor.config import PreprocessingConfig
from gMRItensor.pipeline import read_manifest
from gMRItensor.pipeline import run_preprocessing


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def preprocessing_config(study: Any, **overrides: Any) -> PreprocessingConfig:
    data = {
        "manifest": "scans.csv",
        "output_dir": "results",
        "signal_type": "T1map",
        "n_procs": 1,
        "regions": {"presets": ["ventricles", "thalamus"]},
        **overrides,
    }
    return load_preprocessing_config(
        write_yaml(study.root / "preprocessing.yaml", data),
    )


def test_run_preprocessing_writes_data_and_config(synthetic_study: Any) -> None:
    config = preprocessing_config(synthetic_study)

    paths = run_preprocessing(config)

    results = synthetic_study.root / "results"
    assert paths.tracer == results / "data" / "tracer.parquet"
    assert paths.coords is None
    assert (results / "preprocessing.yaml").read_text() == config.source.read_text()
    stats = pd.read_parquet(paths.roi_statistics)
    groups = stats[stats["roi_type"] == "group"]
    assert set(groups["roi"]) == {"ventricles", "thalamus"}
    assert len(groups) == 2 * 6 * 3  # groups x subjects x time points
    assert (stats["median_concentration"] > 0).all()
    tracer = pd.read_parquet(paths.tracer)
    assert set(tracer["subject"]) == {f"sub-{s:02d}" for s in range(6)}


def test_read_manifest_builds_args_list(synthetic_study: Any) -> None:
    config = preprocessing_config(synthetic_study, aggregation="voxel")

    args_list = read_manifest(config)

    assert len(args_list) == 18
    first = args_list[0]
    assert first["subject"] == "sub-00" and first["time_point"] == 0
    assert first["func"] is None
    assert first["signal_type"] == "T1map"
    assert first["baseline_path"] == synthetic_study.root / "images" / "sub-00_base.nii"


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda df: pd.concat([df, df.iloc[[0]]]), "duplicate"),
        (lambda df: df.assign(time_point=df["time_point"] + 0.5), "time_point"),
        (lambda df: df.drop(columns="mask_path"), "mask_path"),
    ],
)
def test_read_manifest_rejects_bad_rows(
    synthetic_study: Any,
    edit: Any,
    match: str,
) -> None:
    manifest = pd.read_csv(synthetic_study.manifest)
    edit(manifest).to_csv(synthetic_study.manifest, index=False)
    config = preprocessing_config(synthetic_study)

    with pytest.raises(ConfigError, match=match):
        read_manifest(config)


def test_run_preprocessing_reports_missing_images_before_loading(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = pd.read_csv(synthetic_study.manifest)
    manifest.loc[3, "post_injection_path"] = "images/missing.nii"
    manifest.to_csv(synthetic_study.manifest, index=False)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an image was loaded")

    monkeypatch.setattr(preprocessing, "_load_labeled_tracer_voxels", fail)

    with pytest.raises(ConfigError, match="missing.nii"):
        run_preprocessing(preprocessing_config(synthetic_study))
