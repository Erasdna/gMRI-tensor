"""Self-contained YAML configs, one per pipeline stage.

Each stage (`gmri preprocess|plot|decompose|replicability`) reads only its
own file: inputs and outputs are named there, and paths are relative to the
file. Unknown keys are errors, and every error names the file and field,
e.g. `plotting.yaml: grids[1].n_rows: must be >= 1`.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Literal

import yaml
from gMRItensor.plotting.utils import JOURNAL_WIDTHS
from gMRItensor.preprocessing import ROI_STATISTIC_COLUMNS
from gMRItensor.roi_groups import resolve_roi_groups


class ConfigError(ValueError):
    """An invalid config, with the file and field in the message."""


_REQUIRED = object()


class _Reader:
    """Read one YAML mapping field by field, tracking the field path.

    Each typed getter pops its key, so `finish` can reject leftover
    (unknown) keys.
    """

    def __init__(self, data: Any, source: Path, prefix: str = "") -> None:
        self.source = source
        self.prefix = prefix
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"{source}: {prefix or '<root>'}: expected a mapping")
        self._data = dict(data)

    def field_path(self, key: str) -> str:
        return f"{self.prefix}.{key}" if self.prefix else key

    def error(self, key: str, message: str) -> ConfigError:
        return ConfigError(f"{self.source}: {self.field_path(key)}: {message}")

    def _take(self, key: str, default: Any, nullable: bool = False) -> Any:
        if nullable and key in self._data and self._data[key] is None:
            del self._data[key]
            return None
        value = self._data.pop(key, None)
        if value is None:
            if default is _REQUIRED:
                raise self.error(key, "required")
            return default
        return value

    def string(self, key: str, default: Any = _REQUIRED) -> Any:
        value = self._take(key, default)
        if value is not default and not isinstance(value, str):
            raise self.error(key, f"expected a string, got {value!r}")
        return value

    def choice(
        self,
        key: str,
        choices: tuple[str, ...],
        default: Any = _REQUIRED,
    ) -> Any:
        value = self.string(key, default)
        if value is not default and value not in choices:
            raise self.error(key, f"must be one of {list(choices)}, got {value!r}")
        return value

    def integer(self, key: str, default: Any = _REQUIRED, minimum: int = 1) -> Any:
        value = self._take(key, default)
        if value is default:
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.error(key, f"expected an integer, got {value!r}")
        if value < minimum:
            raise self.error(key, f"must be >= {minimum}, got {value}")
        return value

    def number(
        self,
        key: str,
        default: Any = _REQUIRED,
        low: float | None = None,
        high: float | None = None,
        strict: bool = False,
        nullable: bool = False,
    ) -> Any:
        """A float; numeric strings are accepted, since YAML reads `1e-5` as one."""
        value = self._take(key, default, nullable)
        if value is default or value is None:
            return value
        try:
            if isinstance(value, bool):
                raise ValueError
            number = float(value)
        except (TypeError, ValueError):
            raise self.error(key, f"expected a number, got {value!r}") from None
        if (low is not None and (number <= low if strict else number < low)) or (
            high is not None and (number >= high if strict else number > high)
        ):
            bracket = "()" if strict else "[]"
            raise self.error(
                key,
                f"must be in {bracket[0]}{low}, {high}{bracket[1]}, got {number}",
            )
        return number

    def boolean(self, key: str, default: Any = _REQUIRED) -> Any:
        value = self._take(key, default)
        if value is not default and not isinstance(value, bool):
            raise self.error(key, f"expected true or false, got {value!r}")
        return value

    def path(self, key: str, default: Any = _REQUIRED, must_exist: bool = False) -> Any:
        value = self.string(key, default)
        if value is default:
            return value
        path = self.source.parent / value
        if must_exist and not path.exists():
            raise self.error(key, f"file not found: {path}")
        return path

    def strings(
        self,
        key: str,
        default: Any = _REQUIRED,
        choices: tuple[str, ...] | None = None,
    ) -> Any:
        values = self._take(key, default)
        if values is default:
            return values
        if not isinstance(values, list) or not values:
            raise self.error(key, f"expected a non-empty list, got {values!r}")
        for value in values:
            if not isinstance(value, str):
                raise self.error(key, f"expected strings, got {value!r}")
            if choices is not None and value not in choices:
                raise self.error(key, f"{value!r} is not one of {list(choices)}")
        return tuple(values)

    def integers(self, key: str, default: Any = _REQUIRED, minimum: int = 1) -> Any:
        values = self._take(key, default)
        if values is default:
            return values
        if not isinstance(values, list) or not values:
            raise self.error(key, f"expected a non-empty list, got {values!r}")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise self.error(key, f"expected integers, got {value!r}")
            if value < minimum:
                raise self.error(key, f"values must be >= {minimum}, got {value}")
        return tuple(values)

    def mapping(self, key: str) -> dict[str, Any]:
        value = self._take(key, {})
        if not isinstance(value, Mapping):
            raise self.error(key, f"expected a mapping, got {value!r}")
        return dict(value)

    def child(self, key: str) -> "_Reader":
        return _Reader(self._take(key, {}), self.source, self.field_path(key))

    def children(self, key: str) -> list["_Reader"]:
        values = self._take(key, [])
        if not isinstance(values, list):
            raise self.error(key, f"expected a list, got {values!r}")
        return [
            _Reader(value, self.source, f"{self.field_path(key)}[{i}]")
            for i, value in enumerate(values)
        ]

    def finish(self) -> None:
        for key in self._data:
            raise self.error(key, "unknown key")


def _open(path: Path | str) -> _Reader:
    source = Path(path).absolute()
    with open(source) as file:
        return _Reader(yaml.safe_load(file), source)


@dataclass(frozen=True)
class RegionsConfig:
    """ROI groups computed at preprocessing; see `resolve_roi_groups`."""

    presets: tuple[str, ...] = ()
    custom: dict[str, tuple[int, ...]] = field(default_factory=dict)
    csf_offset: int = 10000


@dataclass(frozen=True)
class PreprocessingConfig:
    """`gmri preprocess`: manifest of scans -> `<output_dir>/data/`."""

    source: Path
    manifest: Path
    output_dir: Path
    signal_type: Literal["T1map", "R1map", "T1w"]
    aggregation: Literal["median", "mean", "voxel"] = "median"
    relaxivity: float | None = 3.2
    time_unit: Literal["s", "ms"] = "ms"
    n_procs: int = 5
    regions: RegionsConfig = field(default_factory=RegionsConfig)


@dataclass(frozen=True)
class FigureSpec:
    """Individual figures: one per ROI and statistic."""

    rois: tuple[str, ...]
    statistics: tuple[str, ...]
    layout: Literal["rows", "panels"]
    page_width: str | float = "double"


@dataclass(frozen=True)
class GridSpec:
    """Several ROIs per figure, paginated when they do not fit."""

    name: str
    rois: tuple[str, ...]
    statistics: tuple[str, ...]
    layout: Literal["rows", "panels"]
    n_rows: int
    n_cols: int = 1
    page_width: str | float = "double"
    sharey: bool = False


@dataclass(frozen=True)
class PlottingConfig:
    """`gmri plot`: ROI statistics -> group tables -> figures."""

    source: Path
    roi_statistics: Path
    subject_info: Path
    group_variable: str
    output_dir: Path
    min_group_n: int = 2
    alpha: float = 0.05
    figures: tuple[FigureSpec, ...] = ()
    grids: tuple[GridSpec, ...] = ()
    formats: tuple[str, ...] = ("pdf", "png")
    dpi: int = 300

    @property
    def statistics(self) -> tuple[str, ...]:
        """Every statistic a figure or grid uses, sorted."""
        used = {s for spec in self.figures for s in spec.statistics}
        used |= {s for spec in self.grids for s in spec.statistics}
        return tuple(sorted(used))


@dataclass(frozen=True)
class TensorConfig:
    """How the tracer parquet becomes a tensor; see `load_tensor_from_parquet`."""

    scale: bool = True
    min_timepoints: int | None = None
    max_invalid_fraction: float = 0.9


@dataclass(frozen=True)
class FitConfig:
    """Restart settings plus `options` forwarded to the decomposition runner."""

    restarts: int = 50
    max_iter: int = 2000
    tolerance: float = 1e-5
    restart_procs: int = 1
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecompositionConfig:
    """`gmri decompose`: tracer parquet -> one fit per rank."""

    source: Path
    input: Path
    output_dir: Path
    method: Literal["cp", "parafac2"]
    ranks: tuple[int, ...]
    tensor: TensorConfig = field(default_factory=TensorConfig)
    fit: FitConfig = field(default_factory=FitConfig)


@dataclass(frozen=True)
class ReplicabilityConfig:
    """`gmri replicability`: tracer parquet -> factor match scores per rank."""

    source: Path
    input: Path
    output_dir: Path
    method: Literal["cp", "parafac2"]
    ranks: tuple[int, ...]
    engine: Literal["halfhalf", "cv"]
    repeats: int
    subject_info: Path | None = None
    splits: int | None = None
    stratify_by: str | None = None
    n_procs: int = 1
    seed: int = 0
    tensor: TensorConfig = field(default_factory=TensorConfig)
    fit: FitConfig = field(default_factory=FitConfig)


def _read_regions(reader: _Reader) -> RegionsConfig:
    presets = reader.strings("presets", ())
    custom_reader = reader.child("custom")
    custom = {
        name: custom_reader.integers(name, minimum=0)
        for name in list(custom_reader._data)
    }
    csf_offset = reader.integer("csf_offset", 10000)
    reader.finish()
    try:
        resolve_roi_groups(presets, custom, csf_offset)
    except ValueError as error:
        raise ConfigError(f"{reader.source}: {reader.prefix}: {error}") from None
    return RegionsConfig(presets, custom, csf_offset)


def load_preprocessing_config(path: Path | str) -> PreprocessingConfig:
    """Read and validate a `gmri preprocess` config."""
    reader = _open(path)
    config = PreprocessingConfig(
        source=reader.source,
        manifest=reader.path("manifest", must_exist=True),
        output_dir=reader.path("output_dir"),
        signal_type=reader.choice("signal_type", ("T1map", "R1map", "T1w")),
        aggregation=reader.choice(
            "aggregation",
            ("median", "mean", "voxel"),
            "median",
        ),
        relaxivity=reader.number(
            "relaxivity",
            3.2,
            low=0.0,
            strict=True,
            nullable=True,
        ),
        time_unit=reader.choice("time_unit", ("s", "ms"), "ms"),
        n_procs=reader.integer("n_procs", 5),
        regions=_read_regions(reader.child("regions")),
    )
    reader.finish()
    return config


def _read_page_width(reader: _Reader) -> str | float:
    value = reader._data.get("page_width", "double")
    if isinstance(value, str) and value in JOURNAL_WIDTHS:
        reader._data.pop("page_width", None)
        return value
    if isinstance(value, str):
        raise reader.error(
            "page_width",
            f"must be one of {sorted(JOURNAL_WIDTHS)} or inches, got {value!r}",
        )
    return reader.number("page_width", low=0.0, strict=True)


def _read_figure(reader: _Reader) -> FigureSpec:
    spec = FigureSpec(
        rois=reader.strings("rois"),
        statistics=reader.strings("statistics", choices=ROI_STATISTIC_COLUMNS),
        layout=reader.choice("layout", ("rows", "panels")),
        page_width=_read_page_width(reader),
    )
    reader.finish()
    return spec


def _read_grid(reader: _Reader) -> GridSpec:
    spec = GridSpec(
        name=reader.string("name"),
        rois=reader.strings("rois"),
        statistics=reader.strings("statistics", choices=ROI_STATISTIC_COLUMNS),
        layout=reader.choice("layout", ("rows", "panels")),
        n_rows=reader.integer("n_rows"),
        n_cols=reader.integer("n_cols", 1),
        page_width=_read_page_width(reader),
        sharey=reader.boolean("sharey", False),
    )
    reader.finish()
    return spec


def load_plotting_config(path: Path | str) -> PlottingConfig:
    """Read and validate a `gmri plot` config."""
    reader = _open(path)
    config = PlottingConfig(
        source=reader.source,
        roi_statistics=reader.path("roi_statistics"),
        subject_info=reader.path("subject_info", must_exist=True),
        group_variable=reader.string("group_variable"),
        output_dir=reader.path("output_dir"),
        min_group_n=reader.integer("min_group_n", 2),
        alpha=reader.number("alpha", 0.05, low=0.0, high=1.0, strict=True),
        figures=tuple(_read_figure(child) for child in reader.children("figures")),
        grids=tuple(_read_grid(child) for child in reader.children("grids")),
        formats=reader.strings("formats", ("pdf", "png")),
        dpi=reader.integer("dpi", 300),
    )
    reader.finish()
    if not config.figures and not config.grids:
        raise reader.error("figures", "no figures or grids configured")
    return config


def _read_tensor(reader: _Reader) -> TensorConfig:
    tensor = TensorConfig(
        scale=reader.boolean("scale", True),
        min_timepoints=reader.integer("min_timepoints", None),
        max_invalid_fraction=reader.number(
            "max_invalid_fraction",
            0.9,
            low=0.0,
            high=1.0,
        ),
    )
    reader.finish()
    return tensor


def _read_fit(reader: _Reader) -> FitConfig:
    fit = FitConfig(
        restarts=reader.integer("restarts", 50),
        max_iter=reader.integer("max_iter", 2000),
        tolerance=reader.number("tolerance", 1e-5, low=0.0, strict=True),
        restart_procs=reader.integer("restart_procs", 1),
        options=reader.mapping("options"),
    )
    reader.finish()
    return fit


def load_decomposition_config(path: Path | str) -> DecompositionConfig:
    """Read and validate a `gmri decompose` config."""
    reader = _open(path)
    config = DecompositionConfig(
        source=reader.source,
        input=reader.path("input"),
        output_dir=reader.path("output_dir"),
        method=reader.choice("method", ("cp", "parafac2")),
        ranks=reader.integers("ranks"),
        tensor=_read_tensor(reader.child("tensor")),
        fit=_read_fit(reader.child("fit")),
    )
    reader.finish()
    return config


def load_replicability_config(path: Path | str) -> ReplicabilityConfig:
    """Read and validate a `gmri replicability` config."""
    reader = _open(path)
    config = ReplicabilityConfig(
        source=reader.source,
        input=reader.path("input"),
        output_dir=reader.path("output_dir"),
        method=reader.choice("method", ("cp", "parafac2")),
        ranks=reader.integers("ranks"),
        engine=reader.choice("engine", ("halfhalf", "cv")),
        repeats=reader.integer("repeats"),
        subject_info=reader.path("subject_info", None, must_exist=True),
        splits=reader.integer("splits", None, minimum=2),
        stratify_by=reader.string("stratify_by", None),
        n_procs=reader.integer("n_procs", 1),
        seed=reader.integer("seed", 0, minimum=0),
        tensor=_read_tensor(reader.child("tensor")),
        fit=_read_fit(reader.child("fit")),
    )
    reader.finish()
    if config.stratify_by is not None and config.subject_info is None:
        raise reader.error("stratify_by", "needs subject_info")
    if config.engine == "cv" and config.splits is None:
        raise reader.error("splits", "required for engine: cv")
    return config
