"""Validated settings for each `gmri` command (built by `gMRItensor.cli`).

Each stage function in `gMRItensor.pipeline` takes one of these. They can
equally be built in a script; invalid values raise `ValueError` on
construction, so a stage never starts with settings it would reject later.
"""
import dataclasses
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import pandas as pd
from gMRItensor.decomposition import Method
from gMRItensor.decomposition.restarts import _resolve_options
from gMRItensor.group_statistics import Aggregate
from gMRItensor.group_statistics import AGGREGATES
from gMRItensor.group_statistics import require_columns
from gMRItensor.model_io import NEGLIGIBLE_SUBJECT_WEIGHT
from gMRItensor.model_io import resolve_model
from gMRItensor.plotting.utils import JOURNAL_WIDTHS
from gMRItensor.preprocessing import SIGNAL_TYPES
from gMRItensor.roi_groups import get_roi_presets
from gMRItensor.roi_groups import parse_region

#: Statistics `gmri plot statistics` can compute per ROI.
STATISTICS = (
    "median",
    "mean",
    "median_concentration",
    "mean_concentration",
    "total_amount",
)
#: The statistics that need ΔR1 input and a relaxivity.
CONCENTRATION_STATISTICS = STATISTICS[2:]
#: The parts `gmri plot decomposition` can draw.
DECOMPOSITION_PARTS = ("mode_grid", "subject_mode", "time", "spatial", "outliers")
#: Runner options that have their own flag or are set by the pipeline, so
#: `--fit-option` cannot override them (`plan` would not see the override).
_MANAGED_FIT_OPTIONS: dict[str, str] = {
    "init_repeats": "--restarts",
    "max_iter": "--max-iter",
    "tolerance": "--tolerance",
    "restart_procs": "--restart-procs",
    "nn_modes": "--non-negative-modes",
    "non_negative": "--non-negative-modes",
    "allow_nan_imputation": "the pipeline (CP always imputes NaN entries)",
    "solver": "--solver",
    "rank": "--ranks",
    "device": "the pipeline (GMRITENSOR_USE_GPU)",
    "compute_device": "the pipeline (GMRITENSOR_USE_GPU)",
    "return_diagnostics": "the pipeline",
    "method": "--method",
    "n_procs": "--n-procs",
    "stratification": "--stratify-by",
}

#: `--method` values -> the restart machinery's method names.
RUNNER_METHODS: dict[str, Method] = {"cp": "CP", "parafac2": "PARAFAC2", "cmf": "CMF"}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_page_width(page_width: str | float) -> None:
    if isinstance(page_width, str):
        _check(
            page_width in JOURNAL_WIDTHS,
            f"page_width must be one of {sorted(JOURNAL_WIDTHS)} or inches, "
            f"got {page_width!r}",
        )
    else:
        _check(page_width > 0, f"page_width must be > 0, got {page_width}")


def _require_file(path: Path, name: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{name}: file not found: {path}")


def _require_optional_files(options: Any, names: tuple[str, ...]) -> None:
    """`_require_file` for each of the path fields `names` that is set."""
    for name in names:
        path = getattr(options, name)
        if path is not None:
            _require_file(path, name)


def _normalise_ranks(options: Any) -> None:
    """Store `options.ranks` as a tuple of ints, all >= 1 (frozen-safe)."""
    object.__setattr__(options, "ranks", tuple(int(r) for r in options.ranks))
    _check(
        bool(options.ranks) and min(options.ranks) >= 1,
        f"ranks must be >= 1, got {options.ranks}",
    )


def _check_statistic(statistic: str) -> None:
    _check(
        statistic in ("median", "mean"),
        f"statistic must be median or mean, got {statistic!r}",
    )


def _check_figure(page_width: str | float, dpi: int) -> None:
    _check_page_width(page_width)
    _check(dpi >= 1, f"dpi must be >= 1, got {dpi}")


def _check_slices(slices: tuple[int, int, int] | None) -> None:
    _check(
        slices is None or len(slices) == 3,
        f"slices must be three indices (i j k), got {slices}",
    )


def _check_subject_columns(path: Path, requested: dict[str, str]) -> None:
    """Raise unless the subject-info CSV has every `requested` column
    (name -> its flag), reading only the header."""
    _require_file(path, "subject_info")
    columns = list(pd.read_csv(path, nrows=0).columns)
    require_columns(columns, requested, str(path))


@dataclass(frozen=True)
class PreprocessingOptions:
    """`gmri preprocess`: manifest of scans -> `<output_dir>/data/`."""

    manifest: Path
    output_dir: Path
    input_type: Literal["T1map", "R1map", "T1w"]
    time_unit: Literal["ms", "s"] | None = None
    store_voxels: bool = False
    n_procs: int = 5

    def __post_init__(self) -> None:
        _check(
            self.input_type in SIGNAL_TYPES,
            f"input_type must be one of {SIGNAL_TYPES}, got {self.input_type!r}",
        )
        # No default: T1 maps are usually in ms but R1 maps in 1/s, and a
        # wrong unit silently scales ΔR1 by 1000.
        _check(
            self.input_type == "T1w" or self.time_unit is not None,
            f"time_unit is required for {self.input_type} input (ms or s)",
        )
        _check(
            self.time_unit in ("ms", "s", None),
            f"time_unit must be ms or s, got {self.time_unit!r}",
        )
        _check(self.n_procs >= 1, f"n_procs must be >= 1, got {self.n_procs}")
        _require_file(self.manifest, "manifest")


@dataclass(frozen=True)
class TensorOptions:
    """How the input becomes a tensor; see `load_tensor_from_parquet`."""

    scale: bool = True
    center: bool = False
    min_timepoints: int | None = None
    max_invalid_fraction: float = 0.9

    def __post_init__(self) -> None:
        _check(
            self.min_timepoints is None or self.min_timepoints >= 1,
            f"min_timepoints must be >= 1, got {self.min_timepoints}",
        )
        _check(
            0.0 <= self.max_invalid_fraction <= 1.0,
            f"max_invalid_fraction must be in [0, 1], got {self.max_invalid_fraction}",
        )


@dataclass(frozen=True)
class FitOptions:
    """Restart settings for `run_*_decomposition_repeated`.

    `non_negative_modes`: `"auto"` (the solver's default), None
    (unconstrained) or mode indices -- 0 subject, 1 time/evolving, 2 label.
    `tolerance` None uses the solver's own default. `solver` is PARAFAC2's
    (None = its default). `extra` holds further
    runner options, passed unchanged.
    """

    restarts: int = 50
    max_iter: int = 2000
    tolerance: float | None = None
    restart_procs: int = 1
    non_negative_modes: Literal["auto"] | tuple[int, ...] | None = "auto"
    solver: Literal["tensorly", "matcouply"] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("restarts", "max_iter", "restart_procs"):
            value = getattr(self, name)
            _check(value >= 1, f"{name} must be >= 1, got {value}")
        _check(
            self.tolerance is None or self.tolerance > 0,
            f"tolerance must be > 0, got {self.tolerance}",
        )
        _check(
            self.solver in (None, "tensorly", "matcouply"),
            f"solver must be tensorly or matcouply, got {self.solver!r}",
        )
        modes = self.non_negative_modes
        if modes not in ("auto", None):
            normalised = tuple(sorted(set(int(m) for m in modes)))  # type: ignore[union-attr]
            _check(
                bool(normalised) and set(normalised) <= {0, 1, 2},
                f"non_negative_modes are 0 (subject), 1 (time), 2 (label), got {modes}",
            )
            object.__setattr__(self, "non_negative_modes", normalised)
        for key, owner in _MANAGED_FIT_OPTIONS.items():
            _check(
                key not in self.extra,
                f"fit option {key} is set by {owner}, not as an extra fit option",
            )


@dataclass(frozen=True)
class DistributedOptions:
    """`plan` / `run --job` / `collect`: where restarts are stored and how
    many fits one job runs."""

    store: str = "restarts"
    tasks_per_job: int = 1

    def __post_init__(self) -> None:
        _check(
            self.tasks_per_job >= 1,
            f"tasks_per_job must be >= 1, got {self.tasks_per_job}",
        )


def _check_fit(method: str, tensor: TensorOptions, fit: FitOptions) -> None:
    _check(
        method in ("cp", "parafac2", "cmf"),
        f"method must be cp, parafac2 or cmf, got {method!r}",
    )
    modes = fit.non_negative_modes
    if method == "cp":
        _check(
            not isinstance(modes, tuple) or modes == (0, 1, 2),
            "CP constrains all modes or none: non_negative_modes must be auto, "
            "none or 0,1,2",
        )
        _check(fit.solver is None, "solver applies to PARAFAC2 only")
    if method == "cmf":
        _check(
            fit.solver in (None, "matcouply"),
            "CMF is fitted with matcouply only; solver must be unset or matcouply",
        )
    _check(
        not tensor.center or modes is None,
        "center makes the data negative; it needs non_negative_modes none",
    )
    # Unknown extra options fail here, before `plan` saves them.
    try:
        _resolve_options(RUNNER_METHODS[method], dict(fit.extra))
    except TypeError as error:
        keys = sorted(fit.extra)
        raise ValueError(
            f"unknown fit option for {method}: {error}; given {keys}",
        ) from None


@dataclass(frozen=True)
class DecompositionOptions:
    """`gmri decompose`: `roi_signal.parquet` (per-label `statistic`) or
    `voxels.parquet` -> one fit per rank."""

    input: Path
    output_dir: Path
    method: Literal["cp", "parafac2", "cmf"]
    ranks: tuple[int, ...]
    statistic: Literal["median", "mean"] = "median"
    tensor: TensorOptions = field(default_factory=TensorOptions)
    fit: FitOptions = field(default_factory=FitOptions)
    distributed: DistributedOptions = field(default_factory=DistributedOptions)

    def __post_init__(self) -> None:
        _normalise_ranks(self)
        _check_statistic(self.statistic)
        _check_fit(self.method, self.tensor, self.fit)

    @property
    def store_dir(self) -> Path:
        """Where distributed jobs keep their per-restart results."""
        return Path(self.output_dir) / self.distributed.store


@dataclass(frozen=True)
class ReplicabilityOptions:
    """`gmri replicability`: fits on split halves or folds -> factor match
    scores per rank."""

    input: Path
    output_dir: Path
    method: Literal["cp", "parafac2", "cmf"]
    ranks: tuple[int, ...]
    engine: Literal["halfhalf", "cv"]
    repeats: int
    splits: int | None = None
    subject_info: Path | None = None
    stratify_by: str | None = None
    n_procs: int = 1
    seed: int = 0
    statistic: Literal["median", "mean"] = "median"
    tensor: TensorOptions = field(default_factory=TensorOptions)
    fit: FitOptions = field(default_factory=FitOptions)
    distributed: DistributedOptions = field(default_factory=DistributedOptions)

    def __post_init__(self) -> None:
        _normalise_ranks(self)
        _check(
            self.engine in ("halfhalf", "cv"),
            f"engine must be halfhalf or cv, got {self.engine!r}",
        )
        _check(self.repeats >= 1, f"repeats must be >= 1, got {self.repeats}")
        if self.engine == "cv":
            # Two folds have disjoint training sets: nothing to compare.
            _check(
                self.splits is not None and self.splits >= 3,
                f"engine cv needs splits >= 3, got {self.splits}",
            )
        else:
            _check(self.splits is None, "splits applies to engine cv only")
        _check(
            self.stratify_by is None or self.subject_info is not None,
            "stratify_by needs subject_info",
        )
        _check(
            self.subject_info is None or self.stratify_by is not None,
            "subject_info is only read with stratify_by",
        )
        _check(
            self.fit.restart_procs == 1,
            "restart_procs applies to decompose; replicability spreads its "
            "fits with n_procs",
        )
        if self.stratify_by is not None:
            _check_subject_columns(
                self.subject_info,  # type: ignore[arg-type]
                {self.stratify_by: "--stratify-by"},
            )
        _check(self.n_procs >= 1, f"n_procs must be >= 1, got {self.n_procs}")
        _check_statistic(self.statistic)
        _check_fit(self.method, self.tensor, self.fit)

    @property
    def store_dir(self) -> Path:
        """Where distributed jobs keep their per-restart results."""
        return Path(self.output_dir) / self.distributed.store


@dataclass(frozen=True)
class StatisticsPlotOptions:
    """`gmri plot statistics`: ROI signal -> group statistics -> figures.

    `regions` are `parse_region` specs (presets or `name=ids`), each plotted
    as one ROI; `rois` are label ids plotted separately, or `"all"` labels.
    With neither, every preset region is used. `relaxivity` (default 3.2
    when a concentration statistic is requested) converts ΔR1 to mM.
    `sd_bands` draws the cohort mean +/- 1, 2 and 3 SD behind the subject
    curves (rows layout only).
    """

    roi_signal: Path
    subject_info: Path
    group_variable: str
    output_dir: Path
    regions: tuple[str, ...] = ()
    rois: Literal["all"] | tuple[str, ...] | None = None
    statistics: tuple[str, ...] = ("median",)
    relaxivity: float | None = None
    alpha: float = 0.05
    min_group_n: int = 2
    layout: Literal["rows", "panels"] = "rows"
    sd_bands: bool = False
    page_width: str | float = "double"
    formats: tuple[str, ...] = ("pdf", "png")
    dpi: int = 300
    csf_offset: int = 10000

    def __post_init__(self) -> None:
        bad = [s for s in self.statistics if s not in STATISTICS]
        _check(
            bool(self.statistics) and not bad,
            f"statistics must be from {STATISTICS}, got {bad or '[]'}",
        )
        for spec in self.regions:
            parse_region(spec, self.csf_offset)
        if self.rois not in (None, "all"):
            _check(
                all(
                    str(roi).lstrip("-").isdigit()
                    for roi in self.rois  # type: ignore[union-attr]
                ),
                f"rois must be 'all' or label ids, got {self.rois}",
            )
        _check(
            self.relaxivity is None or self.relaxivity > 0,
            f"relaxivity must be > 0, got {self.relaxivity}",
        )
        _check(0 < self.alpha < 1, f"alpha must be in (0, 1), got {self.alpha}")
        _check(
            self.min_group_n >= 1,
            f"min_group_n must be >= 1, got {self.min_group_n}",
        )
        _check(
            self.layout in ("rows", "panels"),
            f"layout must be rows or panels, got {self.layout!r}",
        )
        _check(
            not self.sd_bands or self.layout == "rows",
            "--sd-bands needs --layout rows (panels have no subject curves)",
        )
        _check_figure(self.page_width, self.dpi)
        _check_subject_columns(
            self.subject_info,
            {self.group_variable: "--group-variable"},
        )

    def region_groups(self) -> dict[str, np.ndarray]:
        """`name -> label ids` of the regions to plot (every preset by default)."""
        specs = self.regions
        if not specs and self.rois is None:
            specs = tuple(get_roi_presets(self.csf_offset))
        return dict(parse_region(spec, self.csf_offset) for spec in specs)

    def label_ids(self) -> list[int] | Literal["all"] | None:
        """Single labels to plot: `"all"`, ids, or None."""
        if self.rois is None or self.rois == "all":
            return self.rois  # type: ignore[return-value]
        return [int(roi) for roi in self.rois]


@dataclass(frozen=True)
class DecompositionPlotOptions:
    """`gmri plot decomposition`: one saved model -> figures.

    `parts` from `DECOMPOSITION_PARTS` (empty = all). Spatial parts of an
    ROI model need `segmentation`, a label volume in the decomposition's
    label space (CSF ids at +10000); `background` defaults to its mask and
    `slices` (i, j, k) to the centre of mass. The subject mode is drawn as
    boxplots by `group_variable` and each `categorical` column and as
    scatters against each `numerical` column, all from `subject_info`;
    with `min_panel_width` (inches) that figure grows beyond `page_width`
    rather than squeezing its panels. A `subject_prefix` is prepended to
    that figure's name as `<prefix>_`, so runs with different variable sets
    do not overwrite each other. The `outliers` part flags subjects at a
    family-wise `outlier_alpha` (Bonferroni over subjects); it reloads the
    model's input recorded in the `decompose.json` next to it, or `input` if
    given. `model` is a model directory (`<method>_rank_<r>/`), its
    `model.h5`, or any other saved model file.
    """

    model: Path
    subject_info: Path
    group_variable: str
    output_dir: Path
    parts: tuple[str, ...] = ()
    segmentation: Path | None = None
    background: Path | None = None
    slices: tuple[int, int, int] | None = None
    categorical: tuple[str, ...] = ()
    numerical: tuple[str, ...] = ()
    min_panel_width: float | None = None
    subject_prefix: str = ""
    alpha: float = 0.05
    min_group_n: int = 2
    # PARAFAC2/CMF only: CP has no per-subject time curves to drop.
    min_subject_coefficient: float = NEGLIGIBLE_SUBJECT_WEIGHT
    outlier_alpha: float = 0.01
    input: Path | None = None
    page_width: str | float = "double"
    formats: tuple[str, ...] = ("pdf", "png")
    dpi: int = 300

    def __post_init__(self) -> None:
        bad = [p for p in self.parts if p not in DECOMPOSITION_PARTS]
        _check(
            not any(c in self.subject_prefix for c in "/\\")
            and not self.subject_prefix.startswith("."),
            "subject_prefix may not contain '/' or '\\' or start with '.', "
            f"got {self.subject_prefix!r}",
        )
        _check(not bad, f"parts must be from {DECOMPOSITION_PARTS}, got {bad}")
        _check_slices(self.slices)
        _check(0 < self.alpha < 1, f"alpha must be in (0, 1), got {self.alpha}")
        _check(
            self.min_panel_width is None or self.min_panel_width > 0,
            f"min_panel_width must be > 0, got {self.min_panel_width}",
        )
        _check(
            self.min_subject_coefficient >= 0,
            f"min_subject_coefficient must be >= 0, got {self.min_subject_coefficient}",
        )
        _check(
            0 < self.outlier_alpha < 1,
            f"outlier_alpha must be in (0, 1), got {self.outlier_alpha}",
        )
        _check_figure(self.page_width, self.dpi)
        _require_file(resolve_model(self.model)[0], "model")
        _check_subject_columns(
            self.subject_info,
            {
                **{column: "--numerical" for column in self.numerical},
                **{column: "--categorical" for column in self.categorical},
                self.group_variable: "--group-variable",
            },
        )
        _require_optional_files(self, ("segmentation", "background", "input"))

    def selected_parts(self) -> tuple[str, ...]:
        """The parts to draw, in `DECOMPOSITION_PARTS` order."""
        chosen = self.parts or DECOMPOSITION_PARTS
        return tuple(part for part in DECOMPOSITION_PARTS if part in chosen)


@dataclass(frozen=True)
class GroupMapPlotOptions:
    """`gmri plot group-maps`: per time point, each group's average map.

    `input` is `roi_signal.parquet` (its per-label `statistic`, drawn on
    `segmentation`, which is then required) or `voxels.parquet`. Subjects
    are aggregated per ROI/voxel with `aggregate` (`trimmed` drops the
    `trim` highest and lowest values). `timepoints` limits the figures
    (empty = all); `background` and `slices` as in `DecompositionPlotOptions`.
    """

    input: Path
    subject_info: Path
    group_variable: str
    output_dir: Path
    aggregate: Aggregate = "trimmed"
    trim: int = 2
    statistic: Literal["median", "mean"] = "median"
    segmentation: Path | None = None
    background: Path | None = None
    slices: tuple[int, int, int] | None = None
    timepoints: tuple[int, ...] = ()
    page_width: str | float = "double"
    formats: tuple[str, ...] = ("pdf", "png")
    dpi: int = 300

    def __post_init__(self) -> None:
        _check(
            self.aggregate in AGGREGATES,
            f"aggregate must be one of {AGGREGATES}, got {self.aggregate!r}",
        )
        _check(self.trim >= 0, f"trim must be >= 0, got {self.trim}")
        _check_statistic(self.statistic)
        _check_slices(self.slices)
        _check_figure(self.page_width, self.dpi)
        _check_subject_columns(
            self.subject_info,
            {self.group_variable: "--group-variable"},
        )
        _require_optional_files(self, ("segmentation", "background"))


_PLANNED = {
    "DecompositionOptions": DecompositionOptions,
    "ReplicabilityOptions": ReplicabilityOptions,
}
_NESTED = {
    "tensor": TensorOptions,
    "fit": FitOptions,
    "distributed": DistributedOptions,
}
_PATHS = ("input", "output_dir", "subject_info")


def options_to_json(
    options: DecompositionOptions | ReplicabilityOptions,
) -> dict[str, Any]:
    """JSON-serialisable form of planned options (for `plan.json`).

    Paths are made absolute, so jobs started from another working directory
    (e.g. SLURM array tasks) read and write the same files.
    """
    data = dataclasses.asdict(options)
    for key in _PATHS:
        if data.get(key) is not None:
            data[key] = str(Path(data[key]).resolve())
    return {"type": type(options).__name__, "options": data}


def options_from_json(
    data: dict[str, Any],
) -> DecompositionOptions | ReplicabilityOptions:
    """Rebuild `options_to_json` output (paths, tuples and nested options)."""
    cls = _PLANNED[data["type"]]
    values = dict(data["options"])
    for key in _PATHS:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    values["ranks"] = tuple(values["ranks"])
    nested = {}
    for key, nested_cls in _NESTED.items():
        nested_values = dict(values[key])
        modes = nested_values.get("non_negative_modes")
        if isinstance(modes, list):
            nested_values["non_negative_modes"] = tuple(modes)
        nested[key] = nested_cls(**nested_values)
    return cls(**{**values, **nested})  # type: ignore[arg-type]
