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
    assert main(["decompose", "run", str(configs["decomposition"])]) == 0
    assert (results / "decompositions" / "parafac2" / "rank_2.h5").exists()
    assert main(["replicability", "run", str(configs["replicability"])]) == 0
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


@pytest.mark.parametrize(
    "command, name, output",
    [
        ("decompose", "decomposition", "rank_2.h5"),
        ("replicability", "replicability", "replicability.csv"),
    ],
)
def test_cli_plan_run_collect(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
    command: str,
    name: str,
    output: str,
) -> None:
    configs = _configs(synthetic_study.root)
    config = str(configs[name])
    assert main(["preprocess", str(configs["preprocessing"])]) == 0
    capsys.readouterr()

    assert main([command, "plan", config]) == 0
    n_jobs = int(capsys.readouterr().out)  # stdout is just the job count
    assert n_jobs > 1
    for job in range(n_jobs):
        assert main([command, "run", config, "--job", str(job)]) == 0
    assert main([command, "collect", config]) == 0

    out_dir = (
        synthetic_study.root
        / "results"
        / name.replace("decomposition", "decompositions")
        / "parafac2"
    )
    assert (out_dir / output).exists()


def test_cli_run_job_out_of_range(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configs = _configs(synthetic_study.root)
    assert main(["preprocess", str(configs["preprocessing"])]) == 0

    code = main(["decompose", "run", str(configs["decomposition"]), "--job", "99"])

    assert code == 2
    assert "out of range" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--help"], ["preprocess", "plot", "decompose", "replicability"]),
        (["decompose", "--help"], ["plan", "run", "collect"]),
        (["replicability", "run", "--help"], ["--job", "config"]),
        (["plot", "--help"], ["config"]),
    ],
)
def test_cli_help(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected: list[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert all(word in out for word in expected)
