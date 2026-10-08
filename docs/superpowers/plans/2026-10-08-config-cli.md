# Config + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run preprocessing, ROI plotting, decomposition and replicability each on its own via `gmri <command> <config.yaml>`.

**Architecture:** `config.py` parses one self-contained YAML per stage into frozen dataclasses. `pipeline.py` has one `run_*` function per stage, communicating only through files. `model_io.py` stores fits as HDF5. `cli.py` is a thin argparse entry point.

**Tech Stack:** Python 3.12, PyYAML (new), h5py, pyarrow/pandas, existing gMRItensor modules, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-config-cli-design.md`

## Global Constraints
- Work in worktree `.claude/worktrees/roi-evolution`, branch `feature/cli`. Commit after each task (local only; ask before any push or merge).
- `uv add pyyaml`; run everything with `uv run`.
- Every function fully type-hinted. Formatting via black 22.10 (hook), pre-commit must pass.
- Plotting code keeps `matplotlib.use("Agg")` and the existing palette.
- Config paths resolve relative to the config file; unknown keys are errors; errors are `ConfigError(ValueError)` with message `"<config file>: <field path>: <problem>"`.
- Definition of done per task: its tests pass; at the end `uv run pytest` and `uv run pre-commit run --all-files` both pass.

## Review Focus
- YAML `1e-5` parses as a **string** in PyYAML → float fields must accept numeric strings (`tolerance: 1e-5` works). Test in Task 1.
- Running `gmri` from a different working directory → all paths still resolve against the config file. Test in Task 1.
- Duplicate `(subject, time_point)` rows or non-integer `time_point` in the manifest → `ConfigError` naming the rows, before any image is loaded. Test in Task 3.
- A figure/grid ROI absent from `roi_statistics.parquet` → one error naming the ROI before any figure is written. Test in Task 4.
- A tensor subject missing from `subject_info` with `stratify_by` → error naming the subject. Test in Task 6.

---

### Task 1: Config dataclasses and loaders

**Files:**
- Create: `src/gMRItensor/config.py`, `examples/preprocessing.yaml`, `examples/plotting.yaml`, `examples/decomposition.yaml`, `examples/replicability.yaml`, `test/test_config.py`
- Modify: `pyproject.toml` (pyyaml dependency, via `uv add pyyaml`), `src/gMRItensor/preprocessing.py` (export `ROI_STATISTIC_COLUMNS`)

**Interfaces:**
- Produces (all `@dataclass(frozen=True)`; `source: Path` is the config file itself):
  - `ConfigError(ValueError)`
  - `RegionsConfig(presets: tuple[str, ...] = (), custom: dict[str, tuple[int, ...]] = {}, csf_offset: int = 10000)`
  - `PreprocessingConfig(source, manifest: Path, output_dir: Path, signal_type: Literal["T1map","R1map","T1w"], aggregation: Literal["median","mean","voxel"] = "median", relaxivity: float | None = 3.2, time_unit: Literal["s","ms"] = "ms", n_procs: int = 5, regions: RegionsConfig)`
  - `FigureSpec(rois: tuple[str, ...], statistics: tuple[str, ...], layout: Literal["rows","panels"], page_width: str | float = "double")`
  - `GridSpec(name: str, rois, statistics, layout, n_rows: int, n_cols: int = 1, page_width = "double", sharey: bool = False)`
  - `PlottingConfig(source, roi_statistics: Path, subject_info: Path, group_variable: str, output_dir: Path, min_group_n: int = 2, alpha: float = 0.05, figures: tuple[FigureSpec, ...] = (), grids: tuple[GridSpec, ...] = (), formats: tuple[str, ...] = ("pdf","png"), dpi: int = 300)` with property `statistics -> tuple[str, ...]` (sorted union over figures and grids)
  - `TensorConfig(scale: bool = True, min_timepoints: int | None = None, max_invalid_fraction: float = 0.9)`
  - `FitConfig(restarts: int = 50, max_iter: int = 2000, tolerance: float = 1e-5, restart_procs: int = 1, options: dict[str, Any] = {})`
  - `DecompositionConfig(source, input: Path, output_dir: Path, method: Literal["cp","parafac2"], ranks: tuple[int, ...], tensor: TensorConfig, fit: FitConfig)`
  - `ReplicabilityConfig(source, input, subject_info: Path | None, output_dir, method, ranks, tensor, fit, engine: Literal["halfhalf","cv"], repeats: int, splits: int | None = None, stratify_by: str | None = None, n_procs: int = 1, seed: int = 0)`
  - `load_preprocessing_config(path: Path | str) -> PreprocessingConfig`, and likewise `load_plotting_config`, `load_decomposition_config`, `load_replicability_config`.
  - `preprocessing.ROI_STATISTIC_COLUMNS = ("median", "mean", "median_concentration", "mean_concentration", "total_amount")`
- Validation at load: enums; ints ≥ 1 (`n_procs`, `ranks`, `n_rows`, `n_cols`, `repeats`, `restarts`, `dpi`); `0 < alpha < 1`; `0 <= max_invalid_fraction <= 1`; preset names exist (`roi_groups.get_roi_presets`); statistics in `ROI_STATISTIC_COLUMNS`; `page_width` is a `JOURNAL_WIDTHS` name or positive number; `engine: cv` requires `splits >= 2`; `stratify_by` requires `subject_info`; user-authored files (`manifest`, `subject_info`) must exist. Stage outputs (`roi_statistics`, `input`) are checked at run time, not here.

- [ ] **Step 1: Write failing tests** in `test/test_config.py`:
  - `test_examples_load` — each `examples/*.yaml` loads with its loader (create the referenced `scans.csv`/`subjects.csv` by copying the example into `tmp_path` alongside empty files).
  - `test_paths_resolve_relative_to_config_file` — config in `tmp_path/cfg/`, `manifest: scans.csv`; `monkeypatch.chdir(tmp_path)`; `config.manifest == tmp_path / "cfg" / "scans.csv"`.
  - `test_float_fields_accept_yaml_exponent_strings` — `fit: {tolerance: 1e-5}` → `config.fit.tolerance == 1e-5`.
  - `test_plotting_statistics_union` — figures use `median`, grid uses `total_amount, median` → `config.statistics == ("median", "total_amount")`.
  - `test_errors` parametrized over (yaml snippet, loader, match): unknown key `foo` → `"foo"`; `layout: grid` → `"layout"`; `presets: [nope]` → `"nope"`; `n_rows: 0` → `"grids\[0\].n_rows"`; missing manifest file → `"manifest"`; `stratify_by` without `subject_info` → `"stratify_by"`; `engine: cv` without `splits` → `"splits"`. All raise `ConfigError`.
- [ ] **Step 2:** `uv run pytest test/test_config.py -q` → fails on import.
- [ ] **Step 3:** `uv add pyyaml`; implement `config.py` (one private `_Reader` helper that tracks the field path, pops known keys and raises on leftovers) and the four example YAMLs matching the spec.
- [ ] **Step 4:** `uv run pytest test/test_config.py -q` → all pass.
- [ ] **Step 5:** Commit `Add stage configs and loaders`.

### Task 2: Decomposition HDF5 storage

**Files:**
- Create: `src/gMRItensor/model_io.py`, `test/test_model_io.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) SavedDecomposition(method: Literal["cp","parafac2"], rank: int, error: float, weights: np.ndarray, subject_mode: np.ndarray, label_mode: np.ndarray, subjects: np.ndarray, timepoints: np.ndarray | list[np.ndarray], labels: np.ndarray, label_index: np.ndarray, time_mode: np.ndarray | None = None, evolving_states: list[np.ndarray] | None = None, scale_mean: np.ndarray | None = None, scale_std: np.ndarray | None = None)`
  - `save_decomposition(path: Path, saved: SavedDecomposition) -> None` — HDF5 layout as in the spec (`evolving_states/<i>`, `timepoints/<i>` groups for PARAFAC2; `method`, `rank`, `error` attrs; subjects as UTF-8 strings). Atomic: write `<path>.tmp`, then replace.
  - `load_decomposition(path: Path) -> SavedDecomposition`

- [ ] **Step 1: Failing tests**: `test_round_trip_cp` and `test_round_trip_parafac2` (ragged: subjects with 2 and 3 time points) — every array field `np.testing.assert_array_equal`, `subjects` round-trips as `str`, attrs equal; `test_round_trip_without_scaling` — `scale_mean is None` after load.
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** implement with `h5py`.
- [ ] **Step 4:** run → passes.
- [ ] **Step 5:** Commit `Add HDF5 storage for decompositions`.

### Task 3: Preprocessing stage

**Files:**
- Create: `src/gMRItensor/pipeline.py`, `test/conftest.py`, `test/test_pipeline.py`

**Interfaces:**
- Consumes: `PreprocessingConfig`, `write_preprocessed_data`, `resolve_roi_groups`.
- Produces:
  - `copy_config(config_source: Path, output_dir: Path, stage: str) -> Path` — copies to `<output_dir>/<stage>.yaml`.
  - `read_manifest(config: PreprocessingConfig) -> list[dict[str, Any]]` — `args_list` entries with `func` from aggregation (`median → np.nanmedian`, `mean → np.nanmean`, `voxel → None`), paths resolved relative to the manifest. Raises `ConfigError` for missing columns, duplicate `(subject, time_point)`, non-integer time points, missing image files (lists up to 5).
  - `run_preprocessing(config: PreprocessingConfig) -> PreprocessedPaths`
- `test/conftest.py` fixture `synthetic_study(tmp_path) -> SyntheticStudy` (NamedTuple: `root, manifest, subject_info`): 6 subjects (3 `PD`, 3 `Control`), time points 0/6/24, 6×6×6 images, labels `{4, 10, 49, 2, 41}`, T1map with a PD ventricle enhancement shift at 24 h (baseline T1 1500 ms, ΔR1 ~1e-4/ms), so concentrations are finite and nonzero.

- [ ] **Step 1: Failing tests**: `test_run_preprocessing_writes_data_and_config` (files under `root/data/`, `root/preprocessing.yaml` exists, roi_statistics contains group rows for configured presets); `test_read_manifest_rejects_duplicates_and_bad_timepoints` (two cases, `ConfigError` match `"duplicate"` / `"time_point"`); `test_read_manifest_reports_missing_images` (match the missing filename, no image loaded — monkeypatch `preprocessing._load_labeled_tracer_voxels` to raise if called).
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run → passes.
- [ ] **Step 5:** Commit `Add preprocessing stage`.

### Task 4: Plotting stage

**Files:**
- Modify: `src/gMRItensor/pipeline.py`, `test/test_pipeline.py`

**Interfaces:**
- Consumes: `PlottingConfig`, `load_roi_statistics`, `summarize_roi_statistics`, `compare_roi_groups`, `plot_roi_evolution_rows/panels`, `figure_path`, `save_figure`.
- Produces:
  - `grid_pages(rois: Sequence[str], layout: str, n_rows: int, n_cols: int) -> list[tuple[str, ...]]` — chunks of `n_rows` (rows) or `n_rows * n_cols` (panels).
  - `run_plotting(config: PlottingConfig) -> list[Path]` — missing `roi_statistics` → `FileNotFoundError` mentioning `gmri preprocess`; loads stats once for the union of all ROIs (so an unknown ROI fails before drawing); writes `roi_analysis/summary__<stat>.parquet` and `significance__<stat>.csv`; reads them back and draws; individual figures use `figure_path(output_dir, None, roi, stat, layout)` with a 1-ROI rows figure or 1×1 panels; grids use `__p<k>` (k from 1) only when >1 page; copies config as `plotting.yaml`. Returns written figure paths.

- [ ] **Step 1: Failing tests** (input: `run_preprocessing` on `synthetic_study`): `test_run_plotting_writes_tables_and_figures` (expected path set for one figure entry × 2 ROIs and one 3-ROI panels grid with `n_rows=1, n_cols=2` → `__p1`, `__p2`); `test_grid_pages` (rows and panels chunking, exact tuples); `test_run_plotting_unknown_roi_writes_nothing` (`ValueError` match ROI name; `figures/` absent).
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run → passes.
- [ ] **Step 5:** Commit `Add plotting stage`.

### Task 5: Decomposition stage

**Files:**
- Modify: `src/gMRItensor/pipeline.py`, `test/test_pipeline.py`

**Interfaces:**
- Consumes: `DecompositionConfig`, `load_tensor_from_parquet`, `scale_tensor`, `setup_backend`, `run_CP_decomposition_repeated`, `run_PARAFAC2_decomposition_repeated`, `save_decomposition`.
- Produces:
  - `load_decomposition_input(path: Path, method: str, tensor: TensorConfig) -> DecompositionInput` (NamedTuple: `data: torch.Tensor | list[torch.Tensor]`, `subjects`, `timepoints`, `labels`, `label_index`, `scale_mean: np.ndarray | None`, `scale_std: np.ndarray | None`) — missing file → `FileNotFoundError` mentioning `gmri preprocess`; scaling via `scale_tensor(center=False)` when `tensor.scale`.
  - `run_decomposition(config: DecompositionConfig) -> list[Path]` — per rank: fit with `init_repeats=fit.restarts, max_iter, tolerance, restart_procs, device, **fit.options`; CP factors map `[subject, time, label]`; write `rank_<r>.h5`; then `fits.csv` (`rank, error`) and `decomposition.yaml`.

- [ ] **Step 1: Failing tests**: `test_run_decomposition_parafac2` (real tensorly fit, ranks `[1, 2]`, `restarts: 2`, `max_iter: 50`; `fits.csv` has 2 rows; `load_decomposition("rank_2.h5")` has 6 `evolving_states`, `label_mode.shape[1] == 2`); `test_run_decomposition_cp_wiring` (monkeypatch `pipeline.run_CP_decomposition_repeated` with a stub returning random `(weights, [A, B, C], error)` to avoid `torch.compile`; saved `time_mode.shape == (n_timepoints, rank)`).
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run → passes.
- [ ] **Step 5:** Commit `Add decomposition stage`.

### Task 6: Replicability stage

**Files:**
- Modify: `src/gMRItensor/pipeline.py`, `test/test_pipeline.py`

**Interfaces:**
- Consumes: `ReplicabilityConfig`, `load_decomposition_input` (Task 5), `HalfHalfEngine`, `CrossValidationEngine`, `evaluate_replicability_multiproc`, `resolve_subject_groups`.
- Produces: `run_replicability(config: ReplicabilityConfig) -> Path` — calls `setup_backend()` first; stratification = `torch.as_tensor(pd.factorize(groups)[0])` from `subject_info[stratify_by]` in tensor-subject order; per rank, a fresh engine (`seed`) and `evaluate_replicability_multiproc(engine, data, rank, method="CP"|"PARAFAC2", stratification, n_procs, init_repeats=fit.restarts, max_iter, tolerance, **fit.options)`; rows `rank, split, fms` (halfhalf) or `rank, fold_i, fold_j, n_common, fms` (cv) → `replicability.csv`; copies `replicability.yaml`.

- [ ] **Step 1: Failing tests**: `test_run_replicability_halfhalf` (PARAFAC2, ranks `[1]`, `repeats: 2`, `restarts: 2`, `stratify_by: diagnosis`; csv columns and 2 rows, `0 <= fms <= 1`); `test_run_replicability_cv_columns` (`splits: 2, repeats: 1`); `test_run_replicability_missing_subject_info_row` (drop one subject from `subject_info` → `ValueError` match its name).
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run → passes.
- [ ] **Step 5:** Commit `Add replicability stage`.

### Task 7: CLI, docs

**Files:**
- Create: `src/gMRItensor/cli.py`, `test/test_cli.py`
- Modify: `pyproject.toml` (`[project.scripts] gmri = "gMRItensor.cli:main"`), `README.md` (a "Command line" subsection under Usage pointing to `examples/`)

**Interfaces:**
- Consumes: the four loaders and four `run_*` functions.
- Produces: `main(argv: Sequence[str] | None = None) -> int` — subcommands `preprocess`, `plot`, `decompose`, `replicability`, each taking one positional config path; `ConfigError`/`FileNotFoundError` → message on stderr, return 2; success → 0.

- [ ] **Step 1: Failing tests**: `test_cli_runs_each_stage_alone` — write the four configs into `tmp_path` for `synthetic_study`, run `main(["preprocess", ...])`, `main(["plot", ...])`, `main(["decompose", ...])`, `main(["replicability", ...])`, each returns 0 and its outputs exist; `test_cli_reports_config_errors` (bad key → returns 2, stderr contains field path, via `capsys`); `test_cli_plot_before_preprocess` (returns 2, stderr mentions `gmri preprocess`).
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** implement; `uv sync` so the script is installed; README section.
- [ ] **Step 4:** `uv run pytest test/test_cli.py -q` passes; `uv run gmri --help` lists the four commands.
- [ ] **Step 5:** Full verification: `uv run pytest` and `uv run pre-commit run --all-files` pass.
- [ ] **Step 6:** Commit `Add gmri command line interface`.
