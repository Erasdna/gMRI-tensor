from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from gMRItensor import pipeline
from gMRItensor import preprocessing
from gMRItensor.config import ConfigError
from gMRItensor.config import DecompositionConfig
from gMRItensor.config import load_decomposition_config
from gMRItensor.config import load_plotting_config
from gMRItensor.config import load_preprocessing_config
from gMRItensor.config import load_replicability_config
from gMRItensor.config import PlottingConfig
from gMRItensor.config import PreprocessingConfig
from gMRItensor.config import ReplicabilityConfig
from gMRItensor.model_io import load_decomposition
from gMRItensor.pipeline import grid_pages
from gMRItensor.pipeline import read_manifest
from gMRItensor.pipeline import run_decomposition
from gMRItensor.pipeline import run_plotting
from gMRItensor.pipeline import run_preprocessing
from gMRItensor.pipeline import run_replicability


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
        (lambda df: df.assign(mask_path=[""] + list(df["mask_path"][1:])), "mask_path"),
        (lambda df: df.assign(subject=[""] + list(df["subject"][1:])), "subject"),
        (
            lambda df: df.assign(
                time_point=[float("inf")] + list(df["time_point"][1:]),
            ),
            "time_point",
        ),
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


def decomposition_config(study: Any, **overrides: Any) -> DecompositionConfig:
    data = {
        "input": "results/data/tracer.parquet",
        "output_dir": "results/decompositions/test",
        "method": "parafac2",
        "ranks": [1, 2],
        "fit": {"restarts": 2, "max_iter": 50},
        **overrides,
    }
    return load_decomposition_config(
        write_yaml(study.root / "decomposition.yaml", data),
    )


def test_run_decomposition_parafac2(synthetic_study: Any) -> None:
    run_preprocessing(preprocessing_config(synthetic_study))
    config = decomposition_config(synthetic_study)

    written = run_decomposition(config)

    out = synthetic_study.root / "results" / "decompositions" / "test"
    assert written == [out / "rank_1.h5", out / "rank_2.h5"]
    fits = pd.read_csv(out / "fits.csv")
    assert list(fits["rank"]) == [1, 2]
    assert fits["error"].between(0, 1).all()
    saved = load_decomposition(out / "rank_2.h5")
    assert saved.method == "parafac2"
    assert saved.evolving_states is not None
    assert len(saved.evolving_states) == 6
    assert saved.evolving_states[0].shape == (3, 2)
    assert saved.label_mode.shape == (5, 2)
    assert saved.subject_mode.shape == (6, 2)
    assert saved.scale_std is not None and saved.scale_std.shape == (5,)
    assert list(saved.subjects) == [f"sub-{s:02d}" for s in range(6)]
    assert (out / "decomposition.yaml").exists()


def test_run_decomposition_cp_wiring(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the CP fit: the real one triggers torch.compile, which is slow.
    calls = []

    def fake_cp(tensor: torch.Tensor, rank: int, **kwargs: Any) -> tuple:
        calls.append((tuple(tensor.shape), rank, kwargs))
        factors = [torch.rand(n, rank) for n in tensor.shape]
        return torch.ones(rank), factors, torch.tensor(0.25)

    monkeypatch.setattr(pipeline, "run_CP_decomposition_repeated", fake_cp)
    run_preprocessing(preprocessing_config(synthetic_study))
    config = decomposition_config(
        synthetic_study,
        method="cp",
        ranks=[3],
        tensor={"scale": False},
        fit={"restarts": 4, "max_iter": 10, "options": {"non_negative": False}},
    )

    run_decomposition(config)

    ((shape, rank, kwargs),) = calls
    assert shape == (6, 3, 5) and rank == 3
    assert kwargs["init_repeats"] == 4 and kwargs["max_iter"] == 10
    assert kwargs["non_negative"] is False
    out = synthetic_study.root / "results" / "decompositions" / "test"
    saved = load_decomposition(out / "rank_3.h5")
    assert saved.time_mode is not None and saved.time_mode.shape == (3, 3)
    np.testing.assert_array_equal(saved.timepoints, [0, 6, 24])
    assert saved.error == pytest.approx(0.25)
    assert saved.scale_mean is None


def test_run_decomposition_before_preprocessing(synthetic_study: Any) -> None:
    with pytest.raises(FileNotFoundError, match="gmri preprocess"):
        run_decomposition(decomposition_config(synthetic_study))


def replicability_config(study: Any, **overrides: Any) -> ReplicabilityConfig:
    data = {
        "input": "results/data/tracer.parquet",
        "subject_info": "subjects.csv",
        "output_dir": "results/replicability/test",
        "method": "parafac2",
        "ranks": [1],
        "fit": {"restarts": 2, "max_iter": 50},
        "engine": "halfhalf",
        "repeats": 2,
        "stratify_by": "diagnosis",
        **overrides,
    }
    return load_replicability_config(
        write_yaml(study.root / "replicability.yaml", data),
    )


def test_run_replicability_halfhalf(synthetic_study: Any) -> None:
    run_preprocessing(preprocessing_config(synthetic_study))

    path = run_replicability(replicability_config(synthetic_study))

    out = synthetic_study.root / "results" / "replicability" / "test"
    assert path == out / "replicability.csv"
    scores = pd.read_csv(path)
    assert list(scores.columns) == ["rank", "split", "fms"]
    assert list(scores["split"]) == [0, 1]
    assert scores["fms"].between(0, 1).all()
    assert (out / "replicability.yaml").exists()


def test_run_replicability_cv_columns(synthetic_study: Any) -> None:
    run_preprocessing(preprocessing_config(synthetic_study))
    config = replicability_config(synthetic_study, engine="cv", splits=3, repeats=1)

    scores = pd.read_csv(run_replicability(config))

    assert list(scores.columns) == ["rank", "fold_i", "fold_j", "n_common", "fms"]
    assert len(scores) == 3  # fold pairs within the one repeat
    assert (scores["n_common"] > 0).all()


def test_run_replicability_missing_subject_info_row(synthetic_study: Any) -> None:
    run_preprocessing(preprocessing_config(synthetic_study))
    subject_info = pd.read_csv(synthetic_study.subject_info)
    subject_info.iloc[1:].to_csv(synthetic_study.subject_info, index=False)

    with pytest.raises(ValueError, match="sub-00"):
        run_replicability(replicability_config(synthetic_study))


def test_run_plotting_keeps_numeric_label_rois_as_strings(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Label ROIs are named str(id); read back from CSV they must not turn
    # into ints, or no significance marker ever matches its ROI.
    run_preprocessing(preprocessing_config(synthetic_study))
    seen = []
    original = pipeline.plot_roi_evolution_rows

    def spy(summary: Any, stats: Any, significance: Any, *args: Any, **kw: Any) -> Any:
        seen.append((summary, significance))
        return original(summary, stats, significance, *args, **kw)

    monkeypatch.setattr(pipeline, "plot_roi_evolution_rows", spy)
    figures = [{"rois": ["4"], "statistics": ["median"], "layout": "rows"}]
    run_plotting(plotting_config(synthetic_study, figures=figures, grids=[]))

    ((summary, significance),) = seen
    assert set(summary["roi"]) == {"4"}
    assert set(significance["roi"]) == {"4"}
