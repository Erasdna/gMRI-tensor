# Config + CLI (Phase 2) — design

## Goal
Run each part of the pipeline on its own from a YAML file: preprocessing, ROI plotting, decomposition, replicability. Each config is self-contained. It names its own inputs and outputs and never reads another config. Stages communicate only through files on disk.

## Configs (`src/gMRItensor/config.py`, `pyyaml`)
- One loader per stage: `load_preprocessing_config`, `load_plotting_config`, `load_decomposition_config`, `load_replicability_config`. Each returns a frozen dataclass.
- Paths are resolved relative to the config file.
- Validation errors name the field, e.g. `grids[1].n_rows: must be >= 1`, and unknown keys are errors.
- Decomposition and replicability share the `TensorConfig` and `FitConfig` dataclasses. The YAML keys are the same, but each file holds its own copy.

`preprocessing.yaml`
```yaml
manifest: scans.csv       # subject,time_point,baseline_path,post_injection_path,mask_path,segmentation_path
output_dir: results       # -> results/data/
signal_type: T1map        # T1map | R1map | T1w
aggregation: median       # median | mean | voxel (per-voxel rows)
relaxivity: 3.2           # null -> concentration columns NaN
time_unit: ms             # ms | s
n_procs: 5
regions:
  presets: [ventricles, white_matter, limbic_system]
  custom: {my_region: [17, 53]}
  csf_offset: 10000
```
- Manifest paths are relative to the CSV. Missing columns or missing files are reported up front, before any image is loaded.

`plotting.yaml`
```yaml
roi_statistics: results/data/roi_statistics.parquet
subject_info: subjects.csv          # `subjects` column + group_variable
group_variable: diagnosis
output_dir: results                 # -> results/roi_analysis/, results/figures/roi/
min_group_n: 2
alpha: 0.05
figures:                            # one figure per ROI x statistic
  - {rois: [ventricles, thalamus], statistics: [median_concentration], layout: rows, page_width: double}
grids:                              # several ROIs per figure, paginated
  - {name: csf, rois: [...], statistics: [total_amount], layout: panels, n_rows: 2, n_cols: 3, page_width: double, sharey: false}
formats: [pdf, png]
dpi: 300
```
- **Tables before figures.** Tables are computed for the union of the statistics used by `figures` and `grids`. They are written to `roi_analysis/summary__<stat>.parquet` and `significance__<stat>.csv`, and the figures are then drawn from the saved tables only.
- **Grid layouts.** `layout: rows` in a grid paginates `n_rows` ROIs per page. `panels` paginates `n_rows * n_cols` per page, and overflow gets `__p<k>`.

`decomposition.yaml`
```yaml
input: results/data/tracer.parquet
output_dir: results/decompositions/parafac2_roi
method: parafac2             # cp | parafac2
ranks: [2, 3, 4]
tensor: {scale: true, min_timepoints: null, max_invalid_fraction: 0.9}
fit: {restarts: 50, max_iter: 2000, tolerance: 1.0e-5, restart_procs: 1, options: {solver: matcouply}}
```
- `fit.options` is forwarded verbatim to `run_{CP,PARAFAC2}_decomposition_repeated`. Unknown options raise the runner's `TypeError`.

`replicability.yaml`
```yaml
input: results/data/tracer.parquet
subject_info: subjects.csv   # only needed with stratify_by
output_dir: results/replicability/parafac2_halfhalf
method: parafac2
ranks: [2, 3, 4]
tensor: {scale: true, min_timepoints: null, max_invalid_fraction: 0.9}
fit: {restarts: 20, max_iter: 2000, tolerance: 1.0e-5, options: {}}
engine: halfhalf             # halfhalf | cv
repeats: 20
splits: 5                    # cv only
stratify_by: diagnosis       # optional subject_info column
n_procs: 4
seed: 0
```
- Scaling is applied to the full tensor before splitting, because `evaluate_replicability_multiproc` splits internally.

## Stages and CLI
- **`src/gMRItensor/pipeline.py`** has `run_preprocessing`, `run_plotting`, `run_decomposition` and `run_replicability`, each taking its config dataclass.
- **`src/gMRItensor/cli.py`** provides the `gmri` console script: `gmri {preprocess|plot|decompose|replicability} <config.yaml>`.
- **Inputs:** a missing input file fails with a message naming the stage that produces it.
- **Provenance:** each stage copies its config to `<output_dir>/<stage>.yaml`.
- **Backend:** decomposition and replicability call `setup_backend()`.

## Outputs
```
results/data/                    tracer.parquet, [tracer.coords.parquet], roi_statistics.parquet
results/preprocessing.yaml
results/roi_analysis/            summary__<stat>.parquet, significance__<stat>.csv
results/figures/roi/             single/<stat>/<roi>__<layout>.<ext>, <grid>/<grid>__<stat>__<layout>[__p<k>].<ext>
results/plotting.yaml
<decomposition output_dir>/      rank_<r>.h5, fits.csv (rank, error), decomposition.yaml
<replicability output_dir>/      replicability.csv, replicability.yaml
```
- **`model_io.py`:** `save_decomposition(path, SavedDecomposition)` and `load_decomposition(path) -> SavedDecomposition`.
- **HDF5 contents:**
  - Datasets: `weights`, `subject_mode`, `label_mode`, and either `time_mode` (CP) or `evolving_states/<i>` (PARAFAC2).
  - Indexing and scaling: `subjects`, `timepoints` (CP) or `timepoints/<i>` (PARAFAC2), `labels`, `label_index`, `scale_mean`, `scale_std`.
  - Attributes: `method`, `rank`, `error`.
- **`replicability.csv` columns:** half-half gives `rank, split, fms`; cross-validation gives `rank, fold_i, fold_j, n_common, fms`.

## Testing
- **`test_config.py`:** every loader on a valid example, plus targeted errors: unknown key, bad enum, missing file, bad grid shape and `stratify_by` without `subject_info`.
- **`test_model_io.py`:** CP and PARAFAC2 round trips.
- **`test_cli.py`:** each command runs alone on synthetic NIfTI data and pre-built inputs, and output trees are checked. Decomposition uses a tiny tensorly PARAFAC2 fit (2 restarts) to stay fast.
- **Example configs:** `examples/*.yaml`, which the tests also parse.

## Out of scope
Mode figures from saved models, SLURM array splitting and spatial maps.
