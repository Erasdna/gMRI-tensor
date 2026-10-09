import dataclasses
from pathlib import Path
from typing import Any

import pytest
from gMRItensor.options import DecompositionOptions
from gMRItensor.options import DecompositionPlotOptions
from gMRItensor.options import FitOptions
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


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: FitOptions(non_negative_modes=(3,)), "non_negative_modes"),
        (lambda: FitOptions(restarts=0), "restarts"),
        (lambda: FitOptions(tolerance=0.0), "tolerance"),
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


def test_statistics_plot_options(tmp_path: Path) -> None:
    subject_info = tmp_path / "subjects.csv"
    subject_info.touch()
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
    subject_info = tmp_path / "subjects.csv"
    subject_info.touch()
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
