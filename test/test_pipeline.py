from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from gMRItensor import preprocessing
from gMRItensor.config import ConfigError
from gMRItensor.config import load_plotting_config
from gMRItensor.config import load_preprocessing_config
from gMRItensor.config import PlottingConfig
from gMRItensor.config import PreprocessingConfig
from gMRItensor.pipeline import grid_pages
from gMRItensor.pipeline import read_manifest
from gMRItensor.pipeline import run_plotting
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
        "regions": {"presets": ["ventricles", "thalamus", "white_matter"]},
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
    assert set(groups["roi"]) == {"ventricles", "thalamus", "white_matter"}
    assert len(groups) == 3 * 6 * 3  # groups x subjects x time points
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


def plotting_config(study: Any, **overrides: Any) -> PlottingConfig:
    data = {
        "roi_statistics": "results/data/roi_statistics.parquet",
        "subject_info": "subjects.csv",
        "group_variable": "diagnosis",
        "output_dir": "results",
        "figures": [
            {
                "rois": ["ventricles", "thalamus"],
                "statistics": ["median_concentration"],
                "layout": "rows",
                "page_width": "single",
            },
        ],
        "grids": [
            {
                "name": "overview",
                "rois": ["ventricles", "thalamus", "white_matter"],
                "statistics": ["total_amount"],
                "layout": "panels",
                "n_rows": 1,
                "n_cols": 2,
            },
        ],
        "formats": ["png"],
        "dpi": 50,
        **overrides,
    }
    return load_plotting_config(write_yaml(study.root / "plotting.yaml", data))


def test_run_plotting_writes_tables_and_figures(synthetic_study: Any) -> None:
    run_preprocessing(preprocessing_config(synthetic_study))
    config = plotting_config(synthetic_study)

    written = run_plotting(config)

    results = synthetic_study.root / "results"
    figures = results / "figures" / "roi"
    assert sorted(written) == sorted(
        [
            figures / "single" / "median_concentration" / "ventricles__rows.png",
            figures / "single" / "median_concentration" / "thalamus__rows.png",
            figures / "overview" / "overview__total_amount__panels__p1.png",
            figures / "overview" / "overview__total_amount__panels__p2.png",
        ],
    )
    assert all(path.stat().st_size > 0 for path in written)
    for statistic in ("median_concentration", "total_amount"):
        summary = pd.read_parquet(
            results / "roi_analysis" / f"summary__{statistic}.parquet",
        )
        assert set(summary["roi"]) == {"ventricles", "thalamus", "white_matter"}
        significance = pd.read_csv(
            results / "roi_analysis" / f"significance__{statistic}.csv",
        )
        assert list(significance.columns) == [
            "roi",
            "timepoint",
            "p_value",
            "p_adj",
            "significant",
        ]
    assert (results / "plotting.yaml").exists()


def test_grid_pages() -> None:
    assert grid_pages(["a", "b", "c"], "rows", 2, 5) == [("a", "b"), ("c",)]
    assert grid_pages(list("abcde"), "panels", 2, 2) == [
        ("a", "b", "c", "d"),
        ("e",),
    ]
    assert grid_pages(["a"], "panels", 2, 2) == [("a",)]


def test_run_plotting_unknown_roi_writes_nothing(synthetic_study: Any) -> None:
    run_preprocessing(preprocessing_config(synthetic_study))
    figures = [{"rois": ["not_a_roi"], "statistics": ["median"], "layout": "rows"}]
    config = plotting_config(synthetic_study, figures=figures, grids=[])

    with pytest.raises(ValueError, match="not_a_roi"):
        run_plotting(config)

    results = synthetic_study.root / "results"
    assert not (results / "figures").exists()
    assert not (results / "roi_analysis").exists()


def test_run_plotting_before_preprocessing(synthetic_study: Any) -> None:
    with pytest.raises(FileNotFoundError, match="gmri preprocess"):
        run_plotting(plotting_config(synthetic_study))
