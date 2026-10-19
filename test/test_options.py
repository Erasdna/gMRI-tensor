import dataclasses
from pathlib import Path
from typing import Any

import pytest
from gMRItensor.options import DecompositionOptions
from gMRItensor.options import DecompositionPlotOptions
from gMRItensor.options import FitOptions
from gMRItensor.options import GroupMapPlotOptions
from gMRItensor.options import options_from_json
from gMRItensor.options import options_to_json
from gMRItensor.options import PreprocessingOptions
from gMRItensor.options import ReplicabilityOptions
from gMRItensor.options import StatisticsPlotOptions
from gMRItensor.options import TensorOptions


def _decomposition(**overrides: Any) -> DecompositionOptions:
    values: dict[str, Any] = {
        "input": Path("roi_signal.parquet"),
        "output_dir": Path("out"),
        "method": "parafac2",
        "ranks": (2,),
        **overrides,
    }
    return DecompositionOptions(**values)


def test_preprocessing_options(tmp_path: Path) -> None:
    manifest = tmp_path / "scans.csv"
    manifest.touch()

    options = PreprocessingOptions(manifest, tmp_path / "out", "T1map", "ms")

    assert (options.time_unit, options.store_voxels, options.n_procs) == (
        "ms",
        False,
        5,
    )
    # R1 maps are usually in 1/s and T1 maps in ms: no unit is assumed.
    for input_type in ("T1map", "R1map"):
        with pytest.raises(ValueError, match="time_unit is required"):
            PreprocessingOptions(manifest, tmp_path, input_type)  # type: ignore
    assert PreprocessingOptions(manifest, tmp_path, "T1w").time_unit is None
    with pytest.raises(ValueError, match="input_type"):
        PreprocessingOptions(manifest, tmp_path, "T2map")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        PreprocessingOptions(tmp_path / "missing.csv", tmp_path, "T1map", "ms")


@pytest.mark.parametrize(
    "modes, expected",
    [("auto", "auto"), (None, None), ((2, 0, 2), (0, 2))],
)
def test_non_negative_modes_are_normalised(modes: Any, expected: Any) -> None:
    assert FitOptions(non_negative_modes=modes).non_negative_modes == expected


def test_fit_tolerance_defaults_to_the_solvers() -> None:
    assert FitOptions().tolerance is None


@pytest.mark.parametrize(
    "key, flag",
    [
        ("init_repeats", "--restarts"),
        ("max_iter", "--max-iter"),
        ("tolerance", "--tolerance"),
        ("restart_procs", "--restart-procs"),
        ("rank", "--ranks"),
        ("allow_nan_imputation", "pipeline"),
    ],
)
def test_fit_options_cannot_override_the_flags(key: str, flag: str) -> None:
    # Distributed and local runs would otherwise disagree: `plan` reads the
    # flags, not `extra`.
    with pytest.raises(ValueError, match=flag):
        FitOptions(extra={key: 5})


@pytest.mark.parametrize("method", ["cp", "parafac2", "cmf"])
def test_unknown_fit_option_fails_before_fitting(method: str) -> None:
    with pytest.raises(ValueError, match="bogus"):
        _decomposition(method=method, fit=FitOptions(extra={"bogus": 1}))


@pytest.mark.parametrize("method", ["cp", "parafac2", "cmf"])
def test_fits_cannot_be_normalized(method: str) -> None:
    """Fits keep the scale in the factors (weights are ones) for decompose
    and replicability alike; plotting does the normalising."""
    fit = FitOptions(extra={"normalize": True})
    with pytest.raises(ValueError, match="normalize"):
        _decomposition(method=method, fit=fit)
    with pytest.raises(ValueError, match="normalize"):
        ReplicabilityOptions(
            input=Path("x.parquet"),
            output_dir=Path("out"),
            method=method,  # type: ignore[arg-type]
            ranks=(2,),
            engine="cv",
            repeats=1,
            splits=3,
            fit=fit,
        )


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: FitOptions(non_negative_modes=(3,)), "non_negative_modes"),
        (lambda: FitOptions(restarts=0), "restarts"),
        (lambda: FitOptions(tolerance=0.0), "tolerance"),
        (lambda: FitOptions(tolerance=-1.0), "tolerance"),
        (lambda: FitOptions(extra={"nn_modes": (0,)}), "nn_modes"),
        (lambda: FitOptions(solver="other"), "solver"),  # type: ignore[arg-type]
        (lambda: TensorOptions(max_invalid_fraction=1.5), "max_invalid_fraction"),
        (lambda: _decomposition(ranks=()), "ranks"),
        (lambda: _decomposition(method="tucker"), "method"),
        (lambda: _decomposition(statistic="max"), "statistic"),
        (
            lambda: _decomposition(
                method="cp",
                fit=FitOptions(non_negative_modes=(0,)),
            ),
            "CP",
        ),
        (
            lambda: _decomposition(method="cp", fit=FitOptions(solver="matcouply")),
            "solver",
        ),
        (
            lambda: _decomposition(method="cmf", fit=FitOptions(solver="tensorly")),
            "solver",
        ),
        (lambda: _decomposition(tensor=TensorOptions(center=True)), "center"),
    ],
)
def test_option_errors(make: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def test_center_without_non_negativity() -> None:
    options = _decomposition(
        tensor=TensorOptions(center=True, scale=False),
        fit=FitOptions(non_negative_modes=None),
    )

    assert options.tensor.center and options.fit.non_negative_modes is None
    assert options.store_dir == Path("out") / "restarts"


def test_replicability_rules(tmp_path: Path) -> None:
    common: dict[str, Any] = {
        "input": Path("x.parquet"),
        "output_dir": Path("out"),
        "method": "parafac2",
        "ranks": (2,),
    }
    with pytest.raises(ValueError, match="splits"):
        ReplicabilityOptions(**common, engine="cv", repeats=1, splits=2)
    with pytest.raises(ValueError, match="stratify_by"):
        ReplicabilityOptions(**common, engine="halfhalf", repeats=1, stratify_by="x")


def _subject_info(tmp_path: Path) -> Path:
    path = tmp_path / "subjects.csv"
    path.write_text("subjects,diagnosis,sex,age\nsub-01,PD,F,61\n")
    return path


def test_statistics_plot_options(tmp_path: Path) -> None:
    subject_info = _subject_info(tmp_path)
    common: dict[str, Any] = {
        "roi_signal": tmp_path / "roi_signal.parquet",
        "subject_info": subject_info,
        "group_variable": "diagnosis",
        "output_dir": tmp_path,
    }

    default = StatisticsPlotOptions(**common)
    assert default.statistics == ("median",)
    assert "white_matter" in default.region_groups()  # all presets by default
    only_labels = StatisticsPlotOptions(**common, rois=("4", "10"))
    assert only_labels.region_groups() == {}
    assert only_labels.label_ids() == [4, 10]
    custom = StatisticsPlotOptions(**common, regions=("mine=17,53", "thalamus"))
    assert list(custom.region_groups()) == ["mine", "thalamus"]
    cases: list[tuple[dict[str, Any], str]] = [
        ({"statistics": ("max",)}, "statistics"),
        ({"regions": ("nope",)}, "nope"),
        ({"rois": ("csf",)}, "rois"),
        ({"relaxivity": -1.0}, "relaxivity"),
        ({"layout": "grid"}, "layout"),
    ]
    for bad, match in cases:
        with pytest.raises(ValueError, match=match):
            StatisticsPlotOptions(**common, **bad)


def test_decomposition_plot_options(tmp_path: Path) -> None:
    model = tmp_path / "rank_2.h5"
    model.touch()
    subject_info = _subject_info(tmp_path)
    common: dict[str, Any] = {
        "model": model,
        "subject_info": subject_info,
        "group_variable": "diagnosis",
        "output_dir": tmp_path,
    }

    assert DecompositionPlotOptions(**common).selected_parts() == (
        "mode_grid",
        "subject_mode",
        "time",
        "spatial",
    )
    assert DecompositionPlotOptions(**common, parts=("time",)).selected_parts() == (
        "time",
    )
    with pytest.raises(ValueError, match="parts"):
        DecompositionPlotOptions(**common, parts=("everything",))
    with pytest.raises(ValueError, match="slices"):
        DecompositionPlotOptions(**common, slices=(1, 2))  # type: ignore[arg-type]


def test_options_json_round_trip() -> None:
    options = _decomposition(
        ranks=(2, 3),
        tensor=TensorOptions(min_timepoints=2),
        fit=FitOptions(non_negative_modes=(0, 2), extra={"progress_bar": True}),
    )

    back = options_from_json(options_to_json(options))

    # Paths come back absolute (relative to where `plan` ran).
    assert (back.input, back.output_dir) == (
        Path("roi_signal.parquet").resolve(),
        Path("out").resolve(),
    )
    assert back == dataclasses.replace(
        options,
        input=back.input,
        output_dir=back.output_dir,
    )


def test_decomposition_plot_options_split_subject_variables(tmp_path: Path) -> None:
    model = tmp_path / "pd_rank_2.h5"
    model.touch()
    common: dict[str, Any] = {
        "model": model,
        "subject_info": _subject_info(tmp_path),
        "group_variable": "diagnosis",
        "output_dir": tmp_path,
    }

    options = DecompositionPlotOptions(
        **common,
        categorical=("sex", "diagnosis"),
        numerical=("age",),
    )

    assert options.categorical == ("sex", "diagnosis")
    assert options.numerical == ("age",)
    with pytest.raises(ValueError, match="min_panel_width"):
        DecompositionPlotOptions(**common, min_panel_width=0.0)
    assert DecompositionPlotOptions(**common).min_subject_coefficient == 1e-6
    assert DecompositionPlotOptions(**common, min_subject_coefficient=0.0)
    with pytest.raises(ValueError, match="min_subject_coefficient"):
        DecompositionPlotOptions(**common, min_subject_coefficient=-1e-6)


@pytest.mark.parametrize("prefix", ["a/b", "a\\b", ".hidden"])
def test_decomposition_plot_options_reject_bad_subject_prefix(
    tmp_path: Path,
    prefix: str,
) -> None:
    model = tmp_path / "pd_rank_2.h5"
    model.touch()
    with pytest.raises(ValueError, match="subject_prefix"):
        DecompositionPlotOptions(
            model=model,
            subject_info=_subject_info(tmp_path),
            group_variable="diagnosis",
            output_dir=tmp_path,
            subject_prefix=prefix,
        )


def _misspelled_column_cases(tmp_path: Path) -> list[tuple[Any, str]]:
    subject_info = _subject_info(tmp_path)
    model = tmp_path / "pd_rank_2.h5"
    model.touch()
    plot: dict[str, Any] = {
        "model": model,
        "subject_info": subject_info,
        "group_variable": "diagnosis",
        "output_dir": tmp_path,
    }
    misspelled_group: dict[str, Any] = {**plot, "group_variable": "diagnose"}
    return [
        (
            lambda: DecompositionPlotOptions(**misspelled_group),
            "'diagnose' (--group-variable)",
        ),
        (
            lambda: DecompositionPlotOptions(**plot, categorical=("sexx",)),
            "'sexx' (--categorical)",
        ),
        (
            lambda: DecompositionPlotOptions(**plot, numerical=("agee",)),
            "'agee' (--numerical)",
        ),
        (
            lambda: StatisticsPlotOptions(
                roi_signal=tmp_path / "roi_signal.parquet",
                subject_info=subject_info,
                group_variable="diagnose",
                output_dir=tmp_path,
            ),
            "'diagnose' (--group-variable)",
        ),
        (
            lambda: GroupMapPlotOptions(
                input=tmp_path / "voxels.parquet",
                subject_info=subject_info,
                group_variable="diagnose",
                output_dir=tmp_path,
            ),
            "'diagnose' (--group-variable)",
        ),
        (
            lambda: ReplicabilityOptions(
                input=Path("x.parquet"),
                output_dir=Path("out"),
                method="parafac2",
                ranks=(2,),
                engine="halfhalf",
                repeats=1,
                subject_info=subject_info,
                stratify_by="diagnose",
            ),
            "'diagnose' (--stratify-by)",
        ),
    ]


def test_misspelled_subject_columns_list_the_available_ones(tmp_path: Path) -> None:
    for make, missing in _misspelled_column_cases(tmp_path):
        with pytest.raises(ValueError) as error:
            make()
        message = str(error.value)
        assert "subjects.csv" in message
        assert missing in message
        assert "available columns: subjects, diagnosis, sex, age" in message
