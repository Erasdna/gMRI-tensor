from pathlib import Path
from typing import Any

import pytest
import yaml
from gMRItensor.cli import main


def _configs(root: Path) -> dict[str, Path]:
    data: dict[str, dict[str, Any]] = {
        "preprocessing": {
            "manifest": "scans.csv",
            "output_dir": "results",
            "signal_type": "T1map",
            "n_procs": 1,
            "regions": {"presets": ["ventricles", "white_matter"]},
        },
        "plotting": {
            "roi_statistics": "results/data/roi_statistics.parquet",
            "subject_info": "subjects.csv",
            "group_variable": "diagnosis",
            "output_dir": "results",
            "figures": [
                {
                    "rois": ["ventricles"],
                    "statistics": ["median_concentration"],
                    "layout": "rows",
                },
            ],
            "formats": ["png"],
            "dpi": 50,
        },
        "decomposition": {
            "input": "results/data/tracer.parquet",
            "output_dir": "results/decompositions/parafac2",
            "method": "parafac2",
            "ranks": [2],
            "fit": {"restarts": 2, "max_iter": 50},
        },
        "replicability": {
            "input": "results/data/tracer.parquet",
            "subject_info": "subjects.csv",
            "output_dir": "results/replicability/parafac2",
            "method": "parafac2",
            "ranks": [1],
            "fit": {"restarts": 2, "max_iter": 50},
            "engine": "halfhalf",
            "repeats": 2,
            "stratify_by": "diagnosis",
        },
    }
    paths = {}
    for name, config in data.items():
        paths[name] = root / f"{name}.yaml"
        paths[name].write_text(yaml.safe_dump(config))
    return paths


def test_cli_runs_each_stage_alone(synthetic_study: Any) -> None:
    configs = _configs(synthetic_study.root)
    results = synthetic_study.root / "results"

    assert main(["preprocess", str(configs["preprocessing"])]) == 0
    assert (results / "data" / "roi_statistics.parquet").exists()
    assert main(["plot", str(configs["plotting"])]) == 0
    assert (results / "figures" / "roi" / "single" / "median_concentration").is_dir()
    assert main(["decompose", str(configs["decomposition"])]) == 0
    assert (results / "decompositions" / "parafac2" / "rank_2.h5").exists()
    assert main(["replicability", str(configs["replicability"])]) == 0
    assert (results / "replicability" / "parafac2" / "replicability.csv").exists()


def test_cli_reports_config_errors(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = synthetic_study.root / "preprocessing.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "manifest": "scans.csv",
                "output_dir": "r",
                "signal_type": "T1map",
                "x": 1,
            },
        ),
    )

    assert main(["preprocess", str(path)]) == 2
    assert "x: unknown key" in capsys.readouterr().err


def test_cli_plot_before_preprocess(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configs = _configs(synthetic_study.root)

    assert main(["plot", str(configs["plotting"])]) == 2
    assert "gmri preprocess" in capsys.readouterr().err


def test_cli_requires_a_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2
    assert "preprocess" in capsys.readouterr().err
