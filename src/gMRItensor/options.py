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
DECOMPOSITION_PARTS = ("mode_grid", "subject_mode", "time", "spatial")
#: Runner options set by `FitOptions` fields rather than `extra`.
_MANAGED_FIT_OPTIONS = ("nn_modes", "non_negative", "solver")


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
    `solver` is PARAFAC2's (None = its default). `extra` holds further
    runner options, passed unchanged.
    """

    restarts: int = 50
    max_iter: int = 2000
    tolerance: float = 1e-5
    restart_procs: int = 1
    non_negative_modes: Literal["auto"] | tuple[int, ...] | None = "auto"
    solver: Literal["tensorly", "matcouply"] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("restarts", "max_iter", "restart_procs"):
            value = getattr(self, name)
            _check(value >= 1, f"{name} must be >= 1, got {value}")
        _check(self.tolerance > 0, f"tolerance must be > 0, got {self.tolerance}")
        _check(
            self.solver in (None, "tensorly", "matcouply"),
            f"solver must be tensorly or matcouply, got {self.solver!r}",
        )
        modes = self.non_negative_modes
        if modes not in ("auto", None):
            normalised = tuple(sorted(set(int(m) for m in modes)))  # type: ignore[union-attr]
            _check(
                bool(normalised) and set(normalised) <= {0, 1, 2},
                "non_negative_modes are 0 (subject), 1 (time), 2 (label), got "
                f"{modes}",
            )
            object.__setattr__(self, "non_negative_modes", normalised)
        for key in _MANAGED_FIT_OPTIONS:
            _check(
                key not in self.extra,
                f"set {key} with its own option, not as an extra fit option",
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
        method in ("cp", "parafac2"),
        f"method must be cp or parafac2, got {method!r}",
    )
    modes = fit.non_negative_modes
    if method == "cp":
        _check(
            not isinstance(modes, tuple) or modes == (0, 1, 2),
            "CP constrains all modes or none: non_negative_modes must be auto, "
            "none or 0,1,2",
        )
        _check(fit.solver is None, "solver applies to PARAFAC2 only")
    _check(
        not tensor.center or modes is None,
        "center makes the data negative; it needs non_negative_modes none",
    )


@dataclass(frozen=True)
class DecompositionOptions:
    """`gmri decompose`: `roi_signal.parquet` (per-label `statistic`) or
    `voxels.parquet` -> one fit per rank."""

    input: Path
    output_dir: Path
    method: Literal["cp", "parafac2"]
    ranks: tuple[int, ...]
    statistic: Literal["median", "mean"] = "median"
    tensor: TensorOptions = field(default_factory=TensorOptions)
    fit: FitOptions = field(default_factory=FitOptions)
    distributed: DistributedOptions = field(default_factory=DistributedOptions)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(int(r) for r in self.ranks))
        _check(
            bool(self.ranks) and min(self.ranks) >= 1,
            f"ranks must be >= 1, got {self.ranks}",
        )
        _check(
            self.statistic in ("median", "mean"),
            f"statistic must be median or mean, got {self.statistic!r}",
        )
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
    method: Literal["cp", "parafac2"]
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
        object.__setattr__(self, "ranks", tuple(int(r) for r in self.ranks))
        _check(
            bool(self.ranks) and min(self.ranks) >= 1,
            f"ranks must be >= 1, got {self.ranks}",
        )
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
        _check(
            self.stratify_by is None or self.subject_info is not None,
            "stratify_by needs subject_info",
        )
        _check(self.n_procs >= 1, f"n_procs must be >= 1, got {self.n_procs}")
        _check(
            self.statistic in ("median", "mean"),
            f"statistic must be median or mean, got {self.statistic!r}",
        )
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
        _check_page_width(self.page_width)
        _check(self.dpi >= 1, f"dpi must be >= 1, got {self.dpi}")
        _require_file(self.subject_info, "subject_info")

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
    `slices` (i, j, k) to the centre of mass.
    """

    model: Path
    subject_info: Path
    group_variable: str
    output_dir: Path
    parts: tuple[str, ...] = ()
    segmentation: Path | None = None
    background: Path | None = None
    slices: tuple[int, int, int] | None = None
    covariates: tuple[str, ...] = ()
    alpha: float = 0.05
    min_group_n: int = 2
    page_width: str | float = "double"
    formats: tuple[str, ...] = ("pdf", "png")
    dpi: int = 300

    def __post_init__(self) -> None:
        bad = [p for p in self.parts if p not in DECOMPOSITION_PARTS]
        _check(not bad, f"parts must be from {DECOMPOSITION_PARTS}, got {bad}")
        _check(
            self.slices is None or len(self.slices) == 3,
            f"slices must be three indices (i j k), got {self.slices}",
        )
        _check(0 < self.alpha < 1, f"alpha must be in (0, 1), got {self.alpha}")
        _check_page_width(self.page_width)
        _check(self.dpi >= 1, f"dpi must be >= 1, got {self.dpi}")
        _require_file(self.model, "model")
        _require_file(self.subject_info, "subject_info")
        for name in ("segmentation", "background"):
            path = getattr(self, name)
            if path is not None:
                _require_file(path, name)

    def selected_parts(self) -> tuple[str, ...]:
        """The parts to draw, in `DECOMPOSITION_PARTS` order."""
        chosen = self.parts or DECOMPOSITION_PARTS
        return tuple(part for part in DECOMPOSITION_PARTS if part in chosen)


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
