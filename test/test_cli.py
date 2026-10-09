from pathlib import Path
from typing import Any

import pytest
from gMRItensor.cli import main


def _run(argv: list[Any]) -> int:
    return main([str(arg) for arg in argv])


def _preprocess(study: Any, *extra: str) -> Path:
    results = study.root / "results"
    code = _run(
        [
            "preprocess",
            "--manifest",
            study.manifest,
            "--output-dir",
            results,
            "--input-type",
            "T1map",
            "--time-unit",
            "ms",
            "--n-procs",
            1,
            *extra,
        ],
    )
    assert code == 0
    return results


def _fit_args(results: Path, name: str) -> list[Any]:
    return [
        "--input",
        results / "data" / "roi_signal.parquet",
        "--output-dir",
        results / name,
        "--method",
        "parafac2",
        "--ranks",
        1,
        "--restarts",
        2,
        "--max-iter",
        50,
    ]


def test_cli_runs_every_command(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _preprocess(synthetic_study, "--store-voxels")
    assert (results / "data" / "voxels.parquet").exists()

    common = [
        "--subject-info",
        synthetic_study.subject_info,
        "--group-variable",
        "diagnosis",
    ]
    assert (
        _run(
            [
                "plot",
                "statistics",
                "--roi-signal",
                results / "data" / "roi_signal.parquet",
                *common,
                "--output-dir",
                results,
                "--region",
                "ventricles",
                "--region",
                "mine=10,49",
                "--rois",
                "4",
                "--statistics",
                "median",
                "median_concentration",
                "--formats",
                "png",
                "--dpi",
                50,
            ],
        )
        == 0
    )
    assert "group differences" in capsys.readouterr().out
    assert (results / "roi_analysis" / "summary.csv").exists()

    assert _run(["decompose", "run", *_fit_args(results, "decomposition")]) == 0
    model = results / "decomposition" / "rank_1.h5"
    assert model.exists()

    replicability = [
        "replicability",
        "run",
        *_fit_args(results, "replicability"),
        "--engine",
        "halfhalf",
        "--repeats",
        2,
        "--subject-info",
        synthetic_study.subject_info,
        "--stratify-by",
        "diagnosis",
    ]
    assert _run(replicability) == 0
    assert (results / "replicability" / "replicability.csv").exists()

    plot = [
        "plot",
        "decomposition",
        "--model",
        model,
        *common,
        "--output-dir",
        results,
        "--segmentation",
        synthetic_study.root / "images" / "seg.nii",
        "--formats",
        "png",
        "--dpi",
        50,
    ]
    assert _run(plot) == 0
    assert (results / "figures" / "decomposition" / "rank_1__mode_grid.png").exists()
    assert _run([*plot, "--time"]) == 0


@pytest.mark.parametrize("command", ["decompose", "replicability"])
def test_cli_plan_run_collect(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    results = _preprocess(synthetic_study)
    extra = (
        ["--engine", "halfhalf", "--repeats", 1] if command == "replicability" else []
    )
    capsys.readouterr()

    assert _run([command, "plan", *_fit_args(results, command), *extra]) == 0
    n_jobs = int(capsys.readouterr().out)  # stdout is just the job count
    assert n_jobs > 1
    for job in range(n_jobs):
        # Array jobs only need the output directory: the rest is in plan.json.
        assert (
            _run([command, "run", "--output-dir", results / command, "--job", job]) == 0
        )
    assert _run([command, "collect", "--output-dir", results / command]) == 0

    output = "rank_1.h5" if command == "decompose" else "replicability.csv"
    assert (results / command / output).exists()


def test_cli_reports_errors_on_one_line(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _preprocess(synthetic_study)
    capsys.readouterr()

    code = _run(
        ["decompose", "run", *_fit_args(results, "d"), "--non-negative-modes", "7"],
    )
    assert code == 2
    assert "non_negative_modes" in capsys.readouterr().err

    assert _run(["decompose", "run", "--output-dir", results / "d", "--job", 0]) == 2
    assert "plan" in capsys.readouterr().err

    assert _run(["decompose", "run", "--output-dir", results / "d"]) == 2
    assert "--input" in capsys.readouterr().err


def test_cli_plot_before_preprocess(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _run(
        [
            "plot",
            "statistics",
            "--roi-signal",
            synthetic_study.root / "results" / "data" / "roi_signal.parquet",
            "--subject-info",
            synthetic_study.subject_info,
            "--group-variable",
            "diagnosis",
            "--output-dir",
            synthetic_study.root / "results",
        ],
    )

    assert code == 2
    assert "gmri preprocess" in capsys.readouterr().err


def test_cli_parses_fit_options_and_modes() -> None:
    from gMRItensor.cli import _fit_option
    from gMRItensor.cli import _non_negative_modes

    assert _fit_option("progress_bar=true") == ("progress_bar", True)
    assert _fit_option('aoadmm_options={"l2_penalty": 0.1}') == (
        "aoadmm_options",
        {"l2_penalty": 0.1},
    )
    assert _fit_option("label=abc") == ("label", "abc")
    assert _non_negative_modes("auto") == "auto"
    assert _non_negative_modes("none") is None
    assert _non_negative_modes("2,0") == (2, 0)


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--help"], ["preprocess", "decompose", "replicability", "plot"]),
        (["decompose", "--help"], ["plan", "run", "collect"]),
        (["decompose", "plan", "--help"], ["--ranks", "--tasks-per-job", "--center"]),
        (["replicability", "run", "--help"], ["--job", "--engine"]),
        (["plot", "--help"], ["statistics", "decomposition"]),
        (["plot", "statistics", "--help"], ["--region", "--rois", "--relaxivity"]),
        (
            ["plot", "decomposition", "--help"],
            ["--segmentation", "--mode-grid", "--slices"],
        ),
        (["preprocess", "--help"], ["--input-type", "--store-voxels", "--time-unit"]),
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
