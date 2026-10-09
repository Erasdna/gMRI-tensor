import shutil
from pathlib import Path
from typing import Callable

import pytest
import yaml
from gMRItensor.config import ConfigError
from gMRItensor.config import load_decomposition_config
from gMRItensor.config import load_plotting_config
from gMRItensor.config import load_preprocessing_config
from gMRItensor.config import load_replicability_config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _copy_example(name: str, directory: Path) -> Path:
    """Copy `examples/<name>.yaml` next to the empty files it references."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "scans.csv").touch()
    (directory / "subjects.csv").touch()
    return Path(shutil.copy(EXAMPLES / f"{name}.yaml", directory))


def _write_config(directory: Path, name: str, data: dict) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _preprocessing(**overrides: object) -> dict:
    return {
        "manifest": "scans.csv",
        "output_dir": "results",
        "signal_type": "T1map",
        **overrides,
    }


def _plotting(**overrides: object) -> dict:
    return {
        "roi_statistics": "results/data/roi_statistics.parquet",
        "subject_info": "subjects.csv",
        "group_variable": "diagnosis",
        "output_dir": "results",
        "figures": [
            {"rois": ["ventricles"], "statistics": ["median"], "layout": "rows"},
        ],
        **overrides,
    }


def _decomposition(**overrides: object) -> dict:
    return {
        "input": "results/data/tracer.parquet",
        "output_dir": "results/decompositions/x",
        "method": "parafac2",
        "ranks": [2],
        **overrides,
    }


def _replicability(**overrides: object) -> dict:
    return {
        "input": "results/data/tracer.parquet",
        "output_dir": "results/replicability/x",
        "method": "parafac2",
        "ranks": [2],
        "engine": "halfhalf",
        "repeats": 2,
        **overrides,
    }


@pytest.mark.parametrize(
    "name, loader",
    [
        ("preprocessing", load_preprocessing_config),
        ("plotting", load_plotting_config),
        ("decomposition", load_decomposition_config),
        ("replicability", load_replicability_config),
    ],
)
def test_examples_load(tmp_path: Path, name: str, loader: Callable) -> None:
    config = loader(_copy_example(name, tmp_path))

    assert config.source == tmp_path / f"{name}.yaml"


def test_example_values(tmp_path: Path) -> None:
    preprocessing = load_preprocessing_config(
        _copy_example("preprocessing", tmp_path),
    )
    decomposition = load_decomposition_config(
        _copy_example("decomposition", tmp_path),
    )

    assert preprocessing.signal_type == "T1map"
    assert preprocessing.aggregation == "median"
    assert preprocessing.relaxivity == 3.2
    assert preprocessing.regions.custom == {"hippocampus_amygdala": (17, 18, 53, 54)}
    assert decomposition.ranks == (2, 3, 4)
    assert decomposition.fit.options == {"solver": "tensorly"}
    assert decomposition.tensor.min_timepoints is None


def test_paths_resolve_relative_to_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "scans.csv").touch()
    path = _write_config(config_dir, "preprocessing", _preprocessing())
    monkeypatch.chdir(tmp_path)

    config = load_preprocessing_config(Path("cfg") / "preprocessing.yaml")

    assert config.manifest == config_dir / "scans.csv"
    assert config.output_dir == config_dir / "results"
    assert config.source == path


def test_float_fields_accept_yaml_exponent_strings(tmp_path: Path) -> None:
    # PyYAML (YAML 1.1) reads `1e-5` as the string "1e-5".
    path = tmp_path / "decomposition.yaml"
    path.write_text(yaml.safe_dump(_decomposition()) + "fit: {tolerance: 1e-5}\n")

    config = load_decomposition_config(path)

    assert config.fit.tolerance == 1e-5


def test_plotting_statistics_union(tmp_path: Path) -> None:
    (tmp_path / "subjects.csv").touch()
    grid = {
        "name": "g",
        "rois": ["ventricles"],
        "statistics": ["total_amount", "median"],
        "layout": "panels",
        "n_rows": 1,
        "n_cols": 1,
    }
    path = _write_config(tmp_path, "plotting", _plotting(grids=[grid]))

    config = load_plotting_config(path)

    assert config.statistics == ("median", "total_amount")
    assert config.grids[0].sharey is False
    assert config.figures[0].page_width == "double"


@pytest.mark.parametrize(
    "loader, data, match",
    [
        (load_preprocessing_config, _preprocessing(foo=1), "foo"),
        (load_preprocessing_config, _preprocessing(aggregation="max"), "aggregation"),
        (
            load_preprocessing_config,
            _preprocessing(regions={"presets": ["nope"]}),
            "nope",
        ),
        (
            load_preprocessing_config,
            _preprocessing(regions={"custom": {"empty": []}}),
            r"regions\.custom\.empty",
        ),
        (load_preprocessing_config, _preprocessing(n_procs=0), "n_procs"),
        (
            load_preprocessing_config,
            _preprocessing(manifest="missing.csv"),
            "manifest",
        ),
        (
            load_plotting_config,
            _plotting(
                figures=[
                    {"rois": ["v"], "statistics": ["median"], "layout": "grid"},
                ],
            ),
            r"figures\[0\]\.layout",
        ),
        (
            load_plotting_config,
            _plotting(
                figures=[{"rois": ["v"], "statistics": ["max"], "layout": "rows"}],
            ),
            r"figures\[0\]\.statistics",
        ),
        (
            load_plotting_config,
            _plotting(
                grids=[
                    {
                        "name": "g",
                        "rois": ["v"],
                        "statistics": ["median"],
                        "layout": "panels",
                        "n_rows": 0,
                    },
                ],
            ),
            r"grids\[0\]\.n_rows",
        ),
        (load_plotting_config, _plotting(alpha=1.5), "alpha"),
        (
            load_plotting_config,
            _plotting(
                figures=[
                    {
                        "rois": ["v"],
                        "statistics": ["median"],
                        "layout": "rows",
                        "page_width": "poster",
                    },
                ],
            ),
            "page_width",
        ),
        (load_decomposition_config, _decomposition(method="tucker"), "method"),
        (load_decomposition_config, _decomposition(ranks=[0]), "ranks"),
        (load_decomposition_config, _decomposition(fit={"bad": 1}), r"fit\.bad"),
        (
            load_replicability_config,
            _replicability(stratify_by="diagnosis"),
            "stratify_by",
        ),
        (load_replicability_config, _replicability(engine="cv"), "splits"),
        (load_replicability_config, _replicability(input=None), "input"),
    ],
)
def test_errors(tmp_path: Path, loader: Callable, data: dict, match: str) -> None:
    (tmp_path / "scans.csv").touch()
    (tmp_path / "subjects.csv").touch()
    path = _write_config(tmp_path, "config", data)

    with pytest.raises(ConfigError, match=match):
        loader(path)


def test_error_message_names_config_file(tmp_path: Path) -> None:
    (tmp_path / "scans.csv").touch()
    path = _write_config(tmp_path, "preprocessing", _preprocessing(foo=1))

    with pytest.raises(ConfigError) as excinfo:
        load_preprocessing_config(path)

    assert str(excinfo.value).startswith(f"{path}: foo:")


def test_relaxivity_default_and_explicit_null(tmp_path: Path) -> None:
    (tmp_path / "scans.csv").touch()
    default = _write_config(tmp_path, "a", _preprocessing())
    disabled = _write_config(tmp_path, "b", _preprocessing(relaxivity=None))

    assert load_preprocessing_config(default).relaxivity == 3.2
    assert load_preprocessing_config(disabled).relaxivity is None


def test_cv_needs_three_splits(tmp_path: Path) -> None:
    # Two folds have disjoint training sets, so no pair could be compared.
    path = _write_config(tmp_path, "r", _replicability(engine="cv", splits=2))

    with pytest.raises(ConfigError, match="splits"):
        load_replicability_config(path)


@pytest.mark.parametrize(
    "make, loader",
    [
        (_decomposition, load_decomposition_config),
        (_replicability, load_replicability_config),
    ],
)
def test_distributed_block(tmp_path: Path, make: Callable, loader: Callable) -> None:
    default = loader(_write_config(tmp_path, "a", make()))
    custom = loader(
        _write_config(
            tmp_path,
            "b",
            make(distributed={"store": "shards", "tasks_per_job": 4}),
        ),
    )

    assert default.store_dir == default.output_dir / "restarts"
    assert default.distributed.tasks_per_job == 1
    assert custom.store_dir == custom.output_dir / "shards"
    assert custom.distributed.tasks_per_job == 4
    with pytest.raises(ConfigError, match=r"distributed\.tasks_per_job"):
        loader(_write_config(tmp_path, "c", make(distributed={"tasks_per_job": 0})))


def test_configuration_docs_cover_every_key() -> None:
    import dataclasses

    from gMRItensor import config as configs

    docs = (
        Path(__file__).resolve().parents[1] / "docs" / "configuration.md"
    ).read_text()
    classes = [
        configs.PreprocessingConfig,
        configs.RegionsConfig,
        configs.PlottingConfig,
        configs.FigureSpec,
        configs.GridSpec,
        configs.DecompositionConfig,
        configs.ReplicabilityConfig,
        configs.TensorConfig,
        configs.FitConfig,
        configs.DistributedConfig,
    ]
    keys = {f.name for cls in classes for f in dataclasses.fields(cls)} - {"source"}

    undocumented = sorted(
        key for key in keys if f"`{key}`" not in docs and f"`regions.{key}`" not in docs
    )
    assert undocumented == []
