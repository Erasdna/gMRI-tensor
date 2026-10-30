from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from gMRItensor.cli import main
from gMRItensor.jobs import FitTask
from gMRItensor.jobs import RestartResult
from gMRItensor.model_io import save_decomposition
from gMRItensor.model_io import SavedDecomposition
from gMRItensor.reproducibility import write_reproducibility


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
                "--sd-bands",
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
    assert (results / "roi_analysis" / "cohort__median.parquet").exists()

    overview = [
        "plot",
        "overview",
        "--roi-signal",
        results / "data" / "roi_signal.parquet",
        *common,
        "--output-dir",
        results,
        "--z-cut",
        3,
        "--formats",
        "png",
        "--dpi",
        50,
    ]
    assert _run(overview) == 0
    assert (results / "roi_analysis" / "overview_flags__median.csv").exists()
    assert (results / "figures" / "roi" / "overview" / "median").is_dir()

    assert _run(["decompose", "run", *_fit_args(results, "decomposition")]) == 0
    model = results / "decomposition" / "parafac2_rank_1"
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
    assert (
        results / "figures" / "decomposition" / "parafac2_rank_1__mode_grid.png"
    ).exists()
    assert _run([*plot, "--time"]) == 0
    capsys.readouterr()
    outliers = [
        *plot,
        "--outliers",
        "--outlier-alpha",
        0.05,
        "--input",
        results / "data" / "roi_signal.parquet",
    ]
    assert _run(outliers) == 0
    assert "Possible outliers (alpha 0.05)" in capsys.readouterr().out
    assert (
        results / "figures" / "decomposition" / "parafac2_rank_1__outliers.csv"
    ).exists()
    assert _run([*outliers[:-2], "--outlier-alpha", 1.5]) == 2
    assert "outlier_alpha" in capsys.readouterr().err


def test_cli_plot_subject_mode_variables(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _preprocess(synthetic_study)
    assert _run(["decompose", "run", *_fit_args(results, "decomposition")]) == 0
    plot = [
        "plot",
        "decomposition",
        "--model",
        results / "decomposition" / "parafac2_rank_1",
        "--subject-info",
        synthetic_study.subject_info,
        "--group-variable",
        "diagnosis",
        "--output-dir",
        results,
        "--subject-mode",
        "--formats",
        "png",
        "--dpi",
        50,
    ]

    variables = ["--categorical", "sex", "apoe", "--numerical", "age"]
    assert (
        _run(
            [
                *plot,
                *variables,
                "--min-panel-width",
                2,
            ],
        )
        == 0
    )
    figure = results / "figures" / "decomposition" / "parafac2_rank_1__subject_mode.png"
    assert figure.exists()
    provenance = (results / "plot_decomposition.json").read_text()
    assert '"categorical": [\n      "sex",\n      "apoe"\n    ]' in provenance
    assert '"min_panel_width": 2.0' in provenance

    assert _run([*plot, *variables, "--subject-prefix", "sleep"]) == 0
    assert (figure.parent / "sleep_parafac2_rank_1__subject_mode.png").exists()

    capsys.readouterr()
    assert _run([*plot, "--categorical", "sexx"]) == 2
    error = capsys.readouterr().err
    assert "'sexx' (--categorical)" in error
    assert "available columns: subjects, diagnosis, sex, apoe, age" in error

    assert _run([*plot, "--numerical", "sex"]) == 2
    assert "'sex' is not numeric" in capsys.readouterr().err


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

    output = (
        "parafac2_rank_1/model.h5" if command == "decompose" else "replicability.csv"
    )
    assert (results / command / output).exists()


def test_cli_exclude_subjects_in_plan_run_collect(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _preprocess(synthetic_study)
    out = results / "decompose"
    plan = ["decompose", "plan", *_fit_args(results, "decompose")]
    capsys.readouterr()

    assert _run([*plan, "--exclude-subjects", "sub-99", "sub-01"]) == 2
    assert "not in the data: ['sub-99']" in capsys.readouterr().err
    assert not (out / "plan.json").exists()

    assert _run([*plan, "--exclude-subjects", "sub-01"]) == 0
    captured = capsys.readouterr()
    n_jobs = int(captured.out)  # the notes go to stderr
    assert "Rejected at command line: 1 subject(s): sub-01" in captured.err
    assert (out / "excluded_subjects.csv").exists()
    for job in range(n_jobs):
        assert _run(["decompose", "run", "--output-dir", out, "--job", job]) == 0
    assert _run(["decompose", "collect", "--output-dir", out]) == 0
    excluded = (out / "parafac2_rank_1" / "excluded_subjects.csv").read_text()
    assert "sub-01,rejected at command line" in excluded

    # The jobs take the exclusion from plan.json only.
    run_job = ["decompose", "run", "--output-dir", out, "--job", 0]
    assert _run([*run_job, "--exclude-subjects", "sub-02"]) == 2
    assert "--exclude-subjects cannot be used with --job" in capsys.readouterr().err

    # A failed plan with --overwrite leaves the stored restarts alone.
    stored = sorted((out / "restarts").rglob("*"))
    assert stored
    assert _run([*plan, "--exclude-subjects", "sub-99", "--overwrite"]) == 2
    assert sorted((out / "restarts").rglob("*")) == stored

    replicability = [
        "replicability",
        "run",
        *_fit_args(results, "replicability"),
        "--engine",
        "halfhalf",
        "--repeats",
        1,
        "--exclude-subjects",
        "sub-99",
    ]
    assert _run(replicability) == 2
    assert "sub-99" in capsys.readouterr().err
    assert not (results / "replicability").exists()


def test_cli_preprocess_with_a_negative_cutoff(
    head_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = head_study.root / "results"
    preprocess = [
        "preprocess",
        "--manifest",
        head_study.manifest,
        "--output-dir",
        results,
        "--input-type",
        "T1map",
        "--time-unit",
        "ms",
        "--n-procs",
        1,
    ]

    assert _run([*preprocess, "--negative-cutoff", 4, "--noise-sd", "0.01"]) == 0
    assert "k4_sd0.01" in capsys.readouterr().out
    folder = results / "data" / "k4_sd0.01"
    assert (folder / "noise.csv").exists() and (folder / "invalid_rois.csv").exists()

    assert _run([*preprocess, "--noise-source", "inside"]) == 2
    assert "noise_source needs negative_cutoff" in capsys.readouterr().err
    assert _run([*preprocess, "--negative-cutoff", 0]) == 2
    assert "negative_cutoff must be > 0" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _run([*preprocess, "--negative-cutoff", 4, "--noise-sd", "brain=0.01"])
    assert "parenchyma=SD,csf=SD" in capsys.readouterr().err
    t1w = [*preprocess[:5], "--input-type", "T1w", "--negative-cutoff", 4]
    assert _run(t1w) == 2
    assert "T1w" in capsys.readouterr().err


def test_cli_replots_reproducibility(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        RestartResult(
            FitTask("full", (0, 1, 2), seed),
            (torch.ones(2), [torch.rand(n, 2) for n in (3, 4, 5)]),
            0.1 * (seed + 1),
        )
        for seed in range(3)
    ]
    csv = write_reproducibility(results, tmp_path)

    stem = tmp_path / "replot" / "fms"
    argv = ["plot", "reproducibility", csv, "--output", stem, "--formats", "png"]
    assert _run([*argv, "--dpi", 50]) == 0
    assert (tmp_path / "replot" / "fms.png").exists()
    assert "1 figure file(s)" in capsys.readouterr().out

    assert _run(["plot", "reproducibility", tmp_path / "missing.csv"]) == 2


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

    statistics = [
        "plot",
        "statistics",
        "--roi-signal",
        results / "data" / "roi_signal.parquet",
        "--subject-info",
        synthetic_study.subject_info,
        "--group-variable",
        "diagnosis",
        "--output-dir",
        results,
        "--layout",
        "panels",
        "--sd-bands",
    ]
    assert _run(statistics) == 2
    assert "--sd-bands needs --layout rows" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _run([*statistics[:-3], "--overview"])
    assert "unrecognized arguments: --overview" in capsys.readouterr().err
    overview = ["plot", "overview", *statistics[2:-3], "--z-cut", 0]
    assert _run(overview) == 2
    assert "z_cut must be > 0" in capsys.readouterr().err


def test_cli_cmf(synthetic_study: Any) -> None:
    results = _preprocess(synthetic_study)
    args = _fit_args(results, "cmf")
    args[args.index("parafac2")] = "cmf"
    assert _run(["decompose", "run", *args, "--tolerance", "1e-4"]) == 0

    model = results / "cmf" / "cmf_rank_1"
    plot = [
        "plot",
        "decomposition",
        "--model",
        model,
        "--subject-info",
        synthetic_study.subject_info,
        "--group-variable",
        "diagnosis",
        "--output-dir",
        results,
        "--time",
        "--subject-mode",
        "--formats",
        "png",
        "--dpi",
        50,
    ]
    assert _run(plot) == 0
    assert (
        results / "figures" / "decomposition" / "cmf_rank_1__evolving_mode.png"
    ).exists()


def test_cli_planned_jobs_take_settings_only_from_the_plan(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _preprocess(synthetic_study)
    assert _run(["decompose", "plan", *_fit_args(results, "planned")]) == 0
    capsys.readouterr()

    # Even a value equal to the default is refused: the plan decides.
    extras: list[list[Any]] = [
        ["--ranks", 2],
        ["--restarts=50"],
        ["--fit-option", "x=1"],
    ]
    for extra in extras:
        code = _run(
            [
                "decompose",
                "run",
                "--output-dir",
                results / "planned",
                "--job",
                0,
                *extra,
            ],
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "cannot be used with --job" in err and "plan.json" in err


def test_cli_reports_and_overwrites_the_store(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _preprocess(synthetic_study)
    planned = results / "planned"
    assert _run(["decompose", "plan", *_fit_args(results, "planned")]) == 0
    capsys.readouterr()
    assert _run(["decompose", "run", "--output-dir", planned, "--job", 0]) == 0
    assert "rank 1: 1 fitted, 0 reused, 0 refitted" in capsys.readouterr().out

    args = _fit_args(results, "planned")
    args[args.index("--max-iter") + 1] = 60
    assert _run(["decompose", "plan", *args]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "2"  # only the job count, for N=$(...)
    assert "1 restart in" in captured.err and "will be refitted" in captured.err

    assert _run(["decompose", "plan", *args, "--overwrite"]) == 0
    assert "Removed 1 stored restart" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _run(["decompose", "run", "--output-dir", planned, "--job", 0, "--overwrite"])


def test_cli_refuses_abbreviated_flags(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: argparse expanded `--tol` to `--tolerance`, so `run --job`
    accepted a planned setting it should refuse, and then ignored it."""
    results = _preprocess(synthetic_study)
    assert _run(["decompose", "plan", *_fit_args(results, "planned")]) == 0
    capsys.readouterr()

    with pytest.raises(SystemExit) as raised:
        _run(
            [
                "decompose",
                "run",
                "--output-dir",
                results / "planned",
                "--job",
                0,
                "--tol",
                "1e-3",
            ],
        )
    assert raised.value.code == 2
    assert "unrecognized arguments: --tol" in capsys.readouterr().err


def test_cli_refuses_arguments_that_would_be_ignored(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _preprocess(synthetic_study)
    capsys.readouterr()

    assert (
        _run(["decompose", "run", *_fit_args(results, "x"), "--tasks-per-job", 2]) == 2
    )
    assert "--tasks-per-job sets the job size of `plan`" in capsys.readouterr().err
    replicability = [
        "replicability",
        "plan",
        *_fit_args(results, "y"),
        "--engine",
        "halfhalf",
        "--repeats",
        1,
    ]
    with pytest.raises(SystemExit):
        _run([*replicability, "--restart-procs", 2])
    assert "unrecognized arguments: --restart-procs" in capsys.readouterr().err
    assert _run([*replicability, "--splits", 3]) == 2
    assert "splits applies to engine cv only" in capsys.readouterr().err


def test_cli_convergence_failure_and_bad_plan_are_one_line(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _preprocess(synthetic_study)
    args = _fit_args(results, "unconverged")
    args[args.index("--max-iter") + 1] = 1
    capsys.readouterr()

    assert _run(["decompose", "run", *args]) == 2
    err = capsys.readouterr().err.strip()
    assert "No decomposition converged" in err and "\n" not in err
    assert "Traceback" not in err

    planned = results / "broken"
    planned.mkdir()
    (planned / "plan.json").write_text('{"type": "DecompositionOptions"')
    assert _run(["decompose", "run", "--output-dir", planned, "--job", 0]) == 2
    assert "not a valid plan" in capsys.readouterr().err


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


def _group_map_args(study: Any, data: str, *extra: Any) -> list[Any]:
    return [
        "plot",
        "group-maps",
        "--input",
        study.root / "results" / "data" / data,
        "--subject-info",
        study.subject_info,
        "--group-variable",
        "diagnosis",
        "--output-dir",
        study.root / "results",
        "--formats",
        "png",
        "--dpi",
        50,
        *extra,
    ]


def test_cli_plot_group_maps(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _preprocess(synthetic_study, "--store-voxels")

    code = _run(
        _group_map_args(
            synthetic_study,
            "voxels.parquet",
            "--aggregate",
            "mean",
            "--timepoints",
            6,
            24,
            "--slices",
            2,
            3,
            3,
        ),
    )

    assert code == 0
    figures = results / "figures" / "group_maps"
    assert sorted(p.name for p in figures.iterdir()) == [
        "voxels__mean__t24__csf.png",
        "voxels__mean__t24__parenchyma.png",
        "voxels__mean__t6__csf.png",
        "voxels__mean__t6__parenchyma.png",
    ]
    assert "4 figure file(s)" in capsys.readouterr().out

    code = _run(
        _group_map_args(
            synthetic_study,
            "roi_signal.parquet",
            "--statistic",
            "mean",
            "--trim",
            1,
            "--segmentation",
            synthetic_study.root / "images" / "seg.nii",
        ),
    )
    assert code == 0
    assert len(list(figures.glob("roi_signal__trimmed__t*__*.png"))) == 6


def test_cli_plot_group_maps_errors(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(_group_map_args(synthetic_study, "voxels.parquet")) == 2
    assert "gmri preprocess" in capsys.readouterr().err

    _preprocess(synthetic_study)
    assert _run(_group_map_args(synthetic_study, "roi_signal.parquet")) == 2
    assert "--segmentation" in capsys.readouterr().err
    assert (
        _run(_group_map_args(synthetic_study, "roi_signal.parquet", "--trim", -1)) == 2
    )
    assert "trim" in capsys.readouterr().err


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
        (
            ["decompose", "plan", "--help"],
            ["--ranks", "--tasks-per-job", "--center", "cmf", "--exclude-subjects"],
        ),
        (["decompose", "run", "--help"], ["--exclude-subjects"]),
        (["replicability", "run", "--help"], ["--job", "--engine"]),
        (["plot", "--help"], ["statistics", "overview", "decomposition"]),
        (["plot", "statistics", "--help"], ["--region", "--rois", "--relaxivity"]),
        (["plot", "overview", "--help"], ["--z-cut", "--relaxivity", "Iglewicz"]),
        (
            ["plot", "decomposition", "--help"],
            ["--segmentation", "--mode-grid", "--slices"],
        ),
        (
            ["preprocess", "--help"],
            [
                "--input-type",
                "--store-voxels",
                "--time-unit",
                "--negative-cutoff",
                "--noise-source",
                "--noise-sd",
                "--max-invalid-voxel-fraction",
            ],
        ),
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


def test_cli_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(0)
    n = 16
    subjects = [f"sub-{i:02d}" for i in range(n)]
    diagnosis = np.repeat(["Control", "PD"], n // 2)
    subject_mode = np.column_stack(
        [1 + 2 * (diagnosis == "PD") + rng.normal(0, 0.1, n), rng.uniform(1, 2, n)],
    )
    model_dir = tmp_path / "cp_rank_2"
    model_dir.mkdir()
    save_decomposition(
        model_dir / "model.h5",
        SavedDecomposition(
            method="cp",
            rank=2,
            error=0.1,
            weights=np.ones(2),
            subject_mode=subject_mode,
            label_mode=rng.uniform(size=(4, 2)),
            subjects=np.array(subjects),
            timepoints=np.arange(3),
            labels=np.arange(4),
            label_index=np.zeros(4, dtype=int),
            time_mode=rng.uniform(size=(3, 2)),
        ),
    )
    info = pd.DataFrame(
        {
            "subjects": subjects[1:],  # sub-00 has no row
            "diagnosis": diagnosis[1:],
            "sex": np.tile(["F", "M"], n // 2)[1:],
            "age": rng.normal(65, 8, n - 1).round(1),
            "egfr": rng.normal(90, 10, n - 1).round(0),
            "apoe": rng.choice([2, 3], n - 1),
        },
    )
    subject_info = tmp_path / "subjects.csv"
    info.to_csv(subject_info, index=False)
    results = tmp_path / "results"

    argv = [
        "stats",
        "--model",
        model_dir,
        "--subject-info",
        subject_info,
        "--output-dir",
        results,
        "--primary-categorical",
        "diagnosis",
        "sex",
        "--primary-numerical",
        "age",
        "--clinical-categorical",
        "apoe",
        "--clinical-numerical",
        "egfr",
        "--n-permutations",
        200,
        "--subject-prefix",
        "clinical",
    ]
    assert _run(argv) == 0

    out = capsys.readouterr().out
    assert "Leaving out 1 subject(s) without a row" in out
    assert "Primary: significant effects" in out
    primary = pd.read_csv(
        results / "statistics" / "clinical_cp_rank_2__subject_stats_primary.csv",
    )
    clinical = pd.read_csv(
        results / "statistics" / "clinical_cp_rank_2__subject_stats_clinical.csv",
    )
    assert set(primary["term"]) == {"diagnosis", "sex", "age"}
    assert set(clinical["term"]) == {"apoe", "egfr"}
    assert (primary["n"] == n - 1).all()
    first = primary[(primary["component"] == 1) & (primary["term"] == "diagnosis")]
    assert bool(first["significant"].item())
    assert (results / "stats.json").exists()

    assert _run([*argv[:-2], "--primary-numerical", "sex"]) == 2


def _evolving_model(directory: Path, method: str, n: int, rng: Any) -> Path:
    """A saved PARAFAC2/CMF model of `n` subjects, rank 2, 4 time points;
    subject k's component-2 curve scales with k."""
    directory.mkdir()
    timepoints = [np.array([0, 6, 24, 48])] * n
    shape = np.array([[0.2, 1.0], [1.0, 0.6], [0.6, 0.3], [0.2, 0.1]])
    states = [
        shape * [rng.uniform(0.8, 1.2), 0.5 + k / n] + rng.normal(0, 0.01, (4, 2))
        for k in range(n)
    ]
    save_decomposition(
        directory / "model.h5",
        SavedDecomposition(
            method=method,  # type: ignore[arg-type]
            rank=2,
            error=0.1,
            weights=np.ones(2),
            subject_mode=np.ones((n, 2)),
            label_mode=rng.uniform(size=(4, 2)),
            subjects=np.array([f"sub-{i:02d}" for i in range(n)]),
            timepoints=timepoints,
            labels=np.arange(4),
            label_index=np.zeros(4, dtype=int),
            evolving_states=states,
        ),
    )
    return directory


@pytest.mark.parametrize("method", ["cmf", "parafac2"])
def test_cli_predict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    method: str,
) -> None:
    rng = np.random.default_rng(0)
    n = 30
    model_dir = _evolving_model(tmp_path / f"{method}_rank_2", method, n, rng)
    info = pd.DataFrame(
        {
            "subjects": [f"sub-{i:02d}" for i in range(n)],
            "diagnosis": np.resize(["Control", "PD"], n),
            "age": rng.normal(65, 8, n).round(1),
            "egfr": 60 + 40 * np.arange(n) / n + rng.normal(0, 2, n),
        },
    )
    subject_info = tmp_path / "subjects.csv"
    info.to_csv(subject_info, index=False)
    results = tmp_path / "results"
    argv = [
        "predict",
        "--model",
        model_dir,
        "--subject-info",
        subject_info,
        "--output-dir",
        results,
        "--outcomes",
        "egfr",
        "--primary-categorical",
        "diagnosis",
        "--primary-numerical",
        "age",
        "--n-permutations",
        200,
        "--cv-repeats",
        3,
    ]

    assert _run(argv) == 0

    out = capsys.readouterr().out
    assert "egfr: best by CV R²" in out
    table = pd.read_csv(
        results / "statistics" / f"{method}_rank_2__curve_prediction.csv",
    )
    assert len(table) == 4 * 3  # 4 covariate sets x (baseline + 2 components)
    assert table["significant"].any()  # component 2's curve carries egfr
    assert (results / "predict.json").exists()


def test_cli_predict_refuses_a_cp_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    model_dir = tmp_path / "cp_rank_2"
    model_dir.mkdir()
    save_decomposition(
        model_dir / "model.h5",
        SavedDecomposition(
            method="cp",
            rank=2,
            error=0.1,
            weights=np.ones(2),
            subject_mode=np.ones((4, 2)),
            label_mode=rng.uniform(size=(3, 2)),
            subjects=np.array(["a", "b", "c", "d"]),
            timepoints=np.arange(3),
            labels=np.arange(3),
            label_index=np.zeros(3, dtype=int),
            time_mode=rng.uniform(size=(3, 2)),
        ),
    )
    subject_info = tmp_path / "subjects.csv"
    pd.DataFrame({"subjects": ["a", "b", "c", "d"], "egfr": [1, 2, 3, 4]}).to_csv(
        subject_info,
        index=False,
    )
    argv = [
        "predict",
        "--model",
        model_dir,
        "--subject-info",
        subject_info,
        "--output-dir",
        tmp_path / "results",
        "--outcomes",
        "egfr",
    ]

    assert _run(argv) == 2
