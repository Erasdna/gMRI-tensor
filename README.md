# gMRI-tensor

Tools for tensor (CP / PARAFAC2) and coupled matrix (CMF) decomposition of gadolinium-enhanced MRI (gMRI) data: tracer-signal preprocessing from NIfTI images, decomposition and replicability analysis, and plotting utilities for the resulting subject/spatial/temporal modes.

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Installation

Python 3.10 or newer, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run gmri --help    # or activate .venv and call gmri directly
```

Release notes: [CHANGELOG.md](CHANGELOG.md).

## Package overview

- `gMRItensor.preprocessing` — compute the tracer signal (ΔR1 in 1/s from T1 or R1 maps, or the T1w ratio) from baseline and post-injection images, write per-label (`roi_signal`) and optionally per-voxel tables (`write_preprocessed_data`), and stream them into decomposition tensors (`load_tensor_from_parquet`).
- `gMRItensor.pipeline` / `gMRItensor.options` — the `gmri` steps as functions taking validated settings, for use from scripts.
- `gMRItensor.decomposition` — CP, PARAFAC2 and non-negative CMF decomposition (`compute_CP_decomposition`, `compute_PARAFAC2_decomposition`, `compute_CMF_decomposition`), with multi-restart runners (`run_CP_decomposition_repeated`, `run_PARAFAC2_decomposition_repeated`, `run_CMF_decomposition_repeated`) and `setup_backend` (CPU threads or the GPU). CPU fits run on TensorLy's numpy backend in float64; GPU fits on PyTorch. One module per method (`cp`, `parafac2`, `cmf`), plus `models` (results and diagnostics) and `restarts` (the runners).
- `gMRItensor.model_io` — one fitted model per HDF5 file (`save_decomposition`, `load_decomposition`), components sorted by weight.
- `gMRItensor.outliers` — subject-mode leverage and per-subject residuals with Bonferroni-corrected cut-offs (`outlier_table`), after Fida et al. (2022).
- `gMRItensor.volumes` — label volumes and voxel layouts (`scatter_to_volume`, `expand_roi_mode_to_voxels`, `merge_segmentations`, `region_masks_from_segmentations`).
- `gMRItensor.replicability` — split-half and cross-validation engines (`HalfHalfEngine`, `CrossValidationEngine`) for assessing decomposition replicability, plus `evaluate_replicability_multiproc` for running them in parallel.
- `gMRItensor.jobs` — scatter/gather restarts (`plan_restarts`, `plan_replicability`, `job_slice`, `run_tasks`, `collect`, `DirectoryStore`) for spreading one fit or replicability analysis over many jobs, e.g. a SLURM array, with the same results as the centralised path.
- `gMRItensor.roi_groups` — FreeSurfer label presets (ventricles, grey/white matter, limbic system, CSF counterparts at `id + 10000`, …), `parse_region` for presets or custom `name=ids` regions, `aggregate_roi_signal` for combining labels into regions and `add_concentration` for ΔR1 → mM.
- `gMRItensor.group_statistics` — group summaries (mean ± SEM) and per-time-point group comparisons (Mann-Whitney / Kruskal-Wallis, BH-FDR per facet) over time, shared by the evolving mode and the ROI analysis.
- `gMRItensor.plotting` — visualization of decomposition modes (subject-mode boxplots and correlations, spatial-mode brain overlays, mode grids, evolving-mode trajectories, outlier diagnostics) and ROI tracer evolution (`plot_roi_evolution_rows`, `plot_roi_evolution_panels`).

## Usage

The steps below follow the pipeline order. Each step reads the output of the previous one from disk, so it can run as a separate job.

### Command line

Every step is a `gmri` command with plain arguments (run them with `uv run gmri ...`, or with `.venv` activated), and each reads only files earlier steps wrote. Every argument is described in [docs/cli.md](docs/cli.md), and `gmri <command> --help` lists them with their defaults.

```bash
gmri preprocess --manifest scans.csv --output-dir results --input-type T1map --time-unit ms [--store-voxels]
gmri plot statistics --roi-signal results/data/roi_signal.parquet --subject-info subjects.csv \
    --group-variable diagnosis --output-dir results --statistics median total_amount [--sd-bands]
gmri plot overview --roi-signal results/data/roi_signal.parquet --subject-info subjects.csv \
    --group-variable diagnosis --output-dir results
gmri decompose run --input results/data/roi_signal.parquet --output-dir results/parafac2 \
    --method parafac2 --ranks 2 3 4
gmri replicability run --input results/data/roi_signal.parquet --output-dir results/replicability \
    --method parafac2 --ranks 2 3 4 --engine halfhalf --repeats 20
gmri plot decomposition --model results/parafac2/parafac2_rank_3 --subject-info subjects.csv \
    --group-variable diagnosis --output-dir results --segmentation template_seg.nii.gz
gmri plot group-maps --input results/data/voxels.parquet --subject-info subjects.csv \
    --group-variable diagnosis --output-dir results
gmri plot reproducibility results/parafac2/restarts/parafac2_rank_3/reproducibility.csv
```

- **`preprocess`:** stores the per-label median, mean and voxel counts (and, with `--store-voxels`, every voxel's value) as ΔR1 in 1/s, or as the ratio for T1w. `--negative-cutoff 4` also makes values below −4 noise SDs invalid, with the noise level estimated from the data, and writes to `data/k4/` with QC tables (`noise.csv`, `invalid_rois.csv`).
- **`plot statistics`:**
  - **Regions:** combines labels into regions, either presets or `--region name=ids`.
  - **Concentration:** converts to concentration (`--relaxivity`, default 3.2).
  - **Output:** prints the significant group differences, writes one table per statistic and draws one figure per ROI. `--sd-bands` adds the cohort mean ± 1, 2 and 3 SD to spot outlying subjects.
- **`plot overview`:** draws every label for every subject and session and flags outlying values (modified z-score, Iglewicz & Hoaglin 1993), to find problematic subjects and ROIs.
- **`decompose` / `replicability`:** fit per-ROI (`roi_signal.parquet`) or per-voxel (`voxels.parquet`) data. Each model gets its own directory, `<method>_rank_<r>/` (`model.h5` and the `decompose.json` it was fitted with). `--exclude-subjects ID ...` leaves subjects out; every left-out subject (also those short of valid time points) is listed with the reason in `excluded_subjects.csv`. They can also spread their restarts over a SLURM array with `plan`, then `run --job $SLURM_ARRAY_TASK_ID`, then `collect`.
- **`plot decomposition`:** draws the mode grid, subject mode, time or evolving mode, spatial maps and outlier diagnostics (`--mode-grid --subject-mode --time --spatial --outliers`; all by default).
- **`plot group-maps`:** each group's average map per time point. **`plot reproducibility`:** redraws a restart-vs-restart FMS matrix.

The sections below show the same steps through the Python API.

### Preprocessing

Each `args_list` entry describes one scan: the baseline, post-injection, mask and segmentation NIfTI paths on one grid, the `signal_type`, and the `subject` and `time_point` it belongs to. `gmri preprocess` builds this list from the manifest CSV.

```python
from gMRItensor.options import TensorOptions
from gMRItensor.pipeline import load_decomposition_input
from gMRItensor.preprocessing import write_preprocessed_data

args_list = [
    {
        "baseline_path": "sub-01/ses-00_T1map.nii.gz",
        "post_injection_path": "sub-01/ses-24_T1map.nii.gz",
        "mask_path": "sub-01/brain_mask.nii.gz",
        "segmentation_path": "sub-01/aparc+aseg.nii.gz",
        "signal_type": "T1map",  # or "R1map", "T1w"
        "subject": "sub-01",
        "time_point": 24,
    },
    # ... one entry per scan
]
paths = write_preprocessed_data(
    args_list, "results", time_unit="ms", store_voxels=False, n_procs=5
)

data = load_decomposition_input(
    paths.roi_signal, "parafac2", TensorOptions(scale=True), statistic="median"
)
slices, subjects, timepoints = (
    data.data,
    data.subjects,
    data.timepoints,
)  # torch slices, one per subject
```

- **Signal:** ΔR1 = R1_post − R1_baseline in 1/s for T1 or R1 maps, or the post/baseline ratio for T1w. Only voxels inside the mask with a nonzero segmentation label are used.
- **Outputs:**
  - Every image is read once.
  - `results/data/roi_signal.parquet` gets each label's median, mean, voxel counts and voxel volume.
  - `store_voxels=True` adds `voxels.parquet` and `voxels.coords.parquet`, which map every voxel back to the common template.
- **Tensor columns:** `load_decomposition_input` turns either file into a tensor. With `roi_signal` input, each column is one label's `statistic`.
- **Loading rules** (`load_tensor_from_parquet`): sessions with more than `max_invalid_fraction` non-finite values are dropped, and only columns finite in every kept session remain.

### Decomposition

```python
from gMRItensor import (
    run_CMF_decomposition_repeated,
    run_CP_decomposition_repeated,
    run_PARAFAC2_decomposition_repeated,
    setup_backend,
)

device = (
    setup_backend()
)  # GPU if GMRITENSOR_USE_GPU=TRUE, else CPU with $CPUS_PER_TASK threads

model, error = run_PARAFAC2_decomposition_repeated(
    slices, rank=3, init_repeats=50, device=device
)
# model.subject_mode: (subjects, rank); model.evolving_states[i]: subject i's (n_timepoints_i, rank) time course;
# model.label_mode: (labels, rank)

cmf, error = run_CMF_decomposition_repeated(
    slices, rank=3, init_repeats=50, device=device
)
# The same fields as PARAFAC2, without the PARAFAC2 constraint: cmf.evolving_states[i] is free per subject.
# The subject weights are fixed at one, so cmf.subject_mode is derived: the RMS of each subject's time course.

tensor = load_decomposition_input(
    paths.roi_signal, "cp", TensorOptions()
).data  # (subjects, time points, labels)
weights, factors, error = run_CP_decomposition_repeated(tensor, rank=3, device=device)
```

- All runners fit from `init_repeats` random restarts and keep the best accepted fit. `restart_procs` spreads the restarts over processes.
- CPU fits run on TensorLy's numpy backend in float64 and return torch tensors; `GMRITENSOR_FIT_BACKEND=torch` runs them on torch instead. GPU fits always use PyTorch.
- Tolerances default to `None`: TensorLy's own `tol`, and `1e-7` on matcouply's penalized objective (`aoadmm_loss_tolerance`). A TensorLy restart that has not converged by its `tol` is rejected; a matcouply one is kept, as matcouply returns it, with a warning.
- PARAFAC2 supports `solver="matcouply"` (default; AO-ADMM, with constraints passed through `aoadmm_options`) or `"tensorly"` (ALS). `nn_modes` chooses which modes are non-negative (all three by default with matcouply).
- CP non-negativity is set with `non_negative`. `allow_nan_imputation=True` fits tensors that contain NaN entries (from `load_tensor_from_parquet(..., impute_missing=True)`): it imputes them, and the error covers the observed entries only. `gmri decompose` always does this for CP.
- CMF uses matcouply's `cmf_aoadmm` with its own defaults (except the loss tolerance above). `nn_modes="auto"` constrains the time and label modes, and `aoadmm_options` passes penalties and solver settings.
- PARAFAC2 and CMF raise a `UserWarning` when most restarts fail for the same reason, naming the setting to change (e.g. `max_iter`).

### Replicability

```python
from gMRItensor.replicability import HalfHalfEngine, evaluate_replicability_multiproc

scores = evaluate_replicability_multiproc(
    HalfHalfEngine(repeats=20),
    slices,
    rank=3,
    method="PARAFAC2",
    n_procs=4,
    init_repeats=20,
)
```

- `HalfHalfEngine` fits independent random halves of the subjects. `CrossValidationEngine(splits, repeats)` fits k-fold subsets.
- The result is a list of factor match scores between fits, one per comparison. The tuple layout depends on the engine.
- `stratification` keeps group proportions equal across splits.
- Keyword arguments beyond the engine's own (here `init_repeats`) are forwarded to the decomposition runner.

### Distributed restarts

`gMRItensor.jobs` splits the same work into `(group, seed)` tasks, so restarts can run as separate jobs (e.g. a SLURM array) that share a `ResultStore`:

```python
from gMRItensor.jobs import (
    DirectoryStore,
    collect,
    job_slice,
    plan_replicability,
    run_tasks,
)

plan = plan_replicability(
    HalfHalfEngine(repeats=20, seed=0), len(slices), n_restarts=10
)
store = DirectoryStore("results/rank3")
run_tasks(job_slice(plan, job_index, tasks_per_job=4), slices, 3, "PARAFAC2", store)
# ...once every job has finished:
fms = HalfHalfEngine(repeats=20, seed=0).compute_fms(collect(plan, store))
```

- `plan_restarts(n_samples, n_restarts)` plans a plain fit. `plan_replicability` plans one group per split or fold of the engine.
- Every job regenerates the same plan, and `job_slice` gives each job its share. Tasks already in the store are skipped, so a re-queued job resumes.
- `collect` picks each group's best restart in seed order. The results are the same as `run_*_decomposition_repeated` and `evaluate_replicability_multiproc`.

### Plotting decomposition modes

```python
from gMRItensor.plotting import (
    evolving_factors_to_numpy,
    plot_evolving_mode,
    plot_subject_mode,
)

fig, axs = plot_subject_mode(
    model.subject_mode.numpy(), subjects, subject_info, "diagnosis", ["age"]
)
fig, axs, significance = plot_evolving_mode(
    evolving_factors_to_numpy(model.evolving_states),
    timepoints,
    subjects,
    subject_info,
    "diagnosis",
)
```

- `subject_info` has a `subjects` column plus the grouping and other subject columns.
- `plot_subject_mode` draws one row per component: a boxplot by group (with a pooled `All` box) and by each of the `categorical_variables` (light grey boxes, subjects as group-coloured dots), with significance annotations, then a scatter against each of the `numerical_variables`.
- `plot_subject_mode_correlation` draws a component-by-component matrix with group comparisons.
- `plot_evolving_mode` draws per-group mean ± SEM ribbons and individual subject curves. Its per-time-point group comparisons are returned as a table, not drawn.
- `plot_spatial_mode` overlays the spatial mode on brain slices, yielding one figure per region from `gMRItensor.volumes.region_masks_from_segmentations`. `gMRItensor.volumes.expand_roi_mode_to_voxels` broadcasts an ROI-level mode out to voxels.
- `plot_outliers` draws `gMRItensor.outliers.outlier_table`: each subject's leverage against its residual, with the cut-offs.
- `plot_mode_grid` combines the spatial, time and subject modes in one figure.
- Figures size to a journal `page_width` in inches. All plotting uses the non-interactive Agg backend.

### ROI tracer evolution

Regions, concentrations and statistics are computed from `roi_signal.parquet` at plotting time. `gmri plot statistics` does all of this; in a script:

```python
import pandas as pd
from gMRItensor.group_statistics import (
    compare_groups_over_time,
    resolve_subject_groups,
    summarize_groups_over_time,
)
from gMRItensor.plotting import plot_roi_evolution_panels, save_figure
from gMRItensor.roi_groups import add_concentration, aggregate_roi_signal, parse_region

regions = dict(
    parse_region(spec) for spec in ["ventricles", "white_matter", "my_region=17,53"]
)
rois = aggregate_roi_signal(pd.read_parquet("results/data/roi_signal.parquet"), regions)
rois = add_concentration(rois, relaxivity=3.2).rename(
    columns={"time_point": "timepoint"}
)
subjects = sorted(rois["subject"].unique())
rois["group"] = rois["subject"].map(
    dict(zip(subjects, resolve_subject_groups(subjects, subject_info, "diagnosis")))
)

summary = summarize_groups_over_time(rois, facet="roi", value="median_concentration")
significance = compare_groups_over_time(
    rois, sorted(rois["group"].unique()), facet="roi", value="median_concentration"
)
significance["significant"] = significance["p_adj"] < 0.05

fig, _ = plot_roi_evolution_panels(
    summary, significance, list(regions), "median_concentration", n_rows=1, n_cols=3
)
save_figure(fig, "results/figures/overview")
```

How regions are combined:
- **Exact:** counts and volumes are summed, and region means are `n_valid`-weighted, so both are exact.
- **Approximate:** a region's median is the `n_valid`-weighted median of its label medians.
- **Concentration:** ΔR1 / r1 in mM, and `total_amount` in mmol. A single r1 is applied to CSF and parenchyma alike, so parenchymal concentrations are approximate.

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest -n auto              # fast tests, in parallel (about 2 min)
uv run pytest -n auto --runslow    # everything, as CI runs it
```

- Package management: `uv add <package>` / `uv add --dev <package>`.
- Code must pass `pre-commit` (black, flake8, reorder-python-imports, add-trailing-comma, mypy).
- Tests live in `test/`; long ones are marked `slow` and run only with `--runslow`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## License

[MIT](LICENSE)
