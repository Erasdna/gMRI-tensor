# gMRI-tensor

Tools for tensor (CP / PARAFAC2) decomposition of gadolinium-enhanced MRI (gMRI) data: tracer-signal preprocessing from NIfTI images, decomposition and replicability analysis, and plotting utilities for the resulting subject/spatial/temporal modes.

[![MIT License](https://img.shields.io/github/license/scientificcomputing/reproducibility)](LICENSE)

## Installation

```bash
uv sync
```

## Package overview

- `gMRItensor.preprocessing` — compute tracer signal (T1map/R1map/T1w) from baseline and post-injection images, extract per-region/voxel values via masks and segmentations, and pivot results into a tensor ready for decomposition (`prepare_tensor`).
- `gMRItensor.decomposition` — CP and PARAFAC2 decomposition (`compute_CP_decomposition`, `compute_PARAFAC2_decomposition`), with multi-restart runners (`run_CP_decomposition_repeated`, `run_PARAFAC2_decomposition_repeated`) and a `setup_backend` helper for configuring TensorLy's PyTorch backend.
- `gMRItensor.replicability` — split-half and cross-validation engines (`HalfHalfEngine`, `CrossValidationEngine`) for assessing decomposition replicability, plus `evaluate_replicability_multiproc` for running them in parallel.
- `gMRItensor.jobs` — scatter/gather restarts (`plan_restarts`, `plan_replicability`, `job_slice`, `run_tasks`, `collect`, `DirectoryStore`) for spreading one fit or replicability analysis over many jobs, e.g. a SLURM array, with the same results as the centralised path.
- `gMRItensor.roi_groups` — FreeSurfer label presets (ventricles, grey/white matter, limbic system, CSF counterparts at `id + 10000`, …) and `resolve_roi_groups` for combining presets with custom regions.
- `gMRItensor.group_statistics` — group summaries (mean ± SEM) and per-time-point group comparisons (Mann-Whitney / Kruskal-Wallis, BH-FDR per facet) over time, shared by the evolving mode and the ROI analysis.
- `gMRItensor.plotting` — visualization of decomposition modes (subject-mode boxplots and correlations, spatial-mode brain overlays, mode grids, evolving-mode trajectories) and ROI tracer evolution (`plot_roi_evolution_rows`, `plot_roi_evolution_panels`).

## Usage

The steps below follow the pipeline order. Each step reads the output of the previous one from disk, so it can run as a separate job.

### Command line

Each stage runs on its own from one YAML config, which names its inputs and outputs. Paths are relative to the config file. `examples/*.yaml` are commented templates; every option is described in [docs/configuration.md](docs/configuration.md).

```bash
gmri preprocess  preprocessing.yaml       # images -> results/data/{tracer,roi_statistics}.parquet
gmri plot        plotting.yaml            # ROI statistics -> results/roi_analysis/ tables + results/figures/roi/
gmri decompose     run decomposition.yaml # tracer table -> rank_<r>.h5 + fits.csv
gmri replicability run replicability.yaml # tracer table -> replicability.csv (factor match scores)
```

Decomposition and replicability can also spread their restarts over many jobs, e.g. a SLURM array. `collect` writes the same files as a single-process `run`:

```bash
N=$(gmri decompose plan decomposition.yaml)          # number of jobs
# in each array job:
gmri decompose run decomposition.yaml --job $SLURM_ARRAY_TASK_ID
# after all jobs have finished:
gmri decompose collect decomposition.yaml
```

- **Help:** `gmri --help` and `gmri <command> [<action>] --help` describe every argument.
- **Errors:** unknown config keys are errors, and every error names the file and field.
- **Provenance:** each stage copies its config next to its outputs.

The sections below show the same steps through the Python API.

### Preprocessing

Each `args_list` entry describes one scan: the baseline, post-injection, mask and segmentation NIfTI paths on one grid, the `signal_type`, an aggregation `func`, and the `subject` and `time_point` it belongs to.

```python
import numpy as np
from gMRItensor.preprocessing import load_tensor_from_parquet, scale_tensor, write_preprocessed_data

args_list = [
    {
        "baseline_path": "sub-01/ses-00_T1map.nii.gz",
        "post_injection_path": "sub-01/ses-24_T1map.nii.gz",
        "mask_path": "sub-01/brain_mask.nii.gz",
        "segmentation_path": "sub-01/aparc+aseg.nii.gz",
        "signal_type": "T1map",  # or "R1map", "T1w"
        "func": np.nanmedian,     # one value per label; None keeps every voxel
        "subject": "sub-01",
        "time_point": 24,
    },
    # ... one entry per scan
]
paths = write_preprocessed_data(args_list, "results", n_procs=5)

slices, subjects, timepoints, labels, label_index = load_tensor_from_parquet(paths.tracer, "parafac2")
slices, mean, std = scale_tensor(slices)  # per-label scaling over subjects and time points
```

- Tracer signal is ΔR1 for T1map/R1map and the post/baseline ratio for T1w. Only voxels inside the mask with a nonzero segmentation label are used.
- Every image is read once. `results/data/` gets `tracer.parquet` (plus `tracer.coords.parquet` with voxel coordinates when `func=None`) and `roi_statistics.parquet`. `write_tracer_parquet` writes only the tracer file.
- `load_tensor_from_parquet` streams the file into a tensor without loading the long table into memory:
  - `"cp"` gives a regular `(subjects, time_points, labels)` array.
  - `"parafac2"` gives one `(n_timepoints_i, labels)` slice per subject.
  - Sessions with more than `max_invalid_fraction` non-finite values are dropped, and only labels finite in every kept session remain.
- `(labels, label_index)` identifies each tensor column. Use it to map spatial modes back onto voxels with the coords file.

### Decomposition

```python
import torch
from gMRItensor import run_CP_decomposition_repeated, run_PARAFAC2_decomposition_repeated, setup_backend

device = setup_backend()  # GPU if GMRITENSOR_USE_GPU=TRUE, else CPU with $CPUS_PER_TASK threads

model, error = run_PARAFAC2_decomposition_repeated(
    [torch.as_tensor(s) for s in slices], rank=3, init_repeats=50, device=device,
)
# model.subject_mode: (subjects, rank); model.evolving_states[i]: subject i's (n_timepoints_i, rank) time course;
# model.label_mode: (labels, rank)

tensor, *_ = load_tensor_from_parquet(paths.tracer, "cp")
weights, factors, error = run_CP_decomposition_repeated(torch.as_tensor(tensor), rank=3, device=device)
```

- Both runners fit from `init_repeats` random restarts and keep the best accepted fit. `restart_procs` spreads the restarts over processes.
- PARAFAC2 supports `solver="tensorly"` (default) or `"matcouply"` (AO-ADMM, with constraints passed through `aoadmm_options`). `nn_modes` chooses which modes are non-negative.
- CP non-negativity is set with `non_negative`, and `allow_nan_imputation=True` fits tensors that contain NaN rows.
- PARAFAC2 raises a `UserWarning` when most restarts fail for the same reason, naming the setting to change (e.g. `max_iter`).

### Replicability

```python
from gMRItensor.replicability import HalfHalfEngine, evaluate_replicability_multiproc

scores = evaluate_replicability_multiproc(
    HalfHalfEngine(repeats=20), [torch.as_tensor(s) for s in slices], rank=3, method="PARAFAC2",
    n_procs=4, init_repeats=20,
)
```

- `HalfHalfEngine` fits independent random halves of the subjects. `CrossValidationEngine(splits, repeats)` fits k-fold subsets.
- The result is a list of factor match scores between fits, one per comparison. The tuple layout depends on the engine.
- `stratification` keeps group proportions equal across splits.
- Keyword arguments beyond the engine's own (here `init_repeats`) are forwarded to the decomposition runner.

### Distributed restarts

`gMRItensor.jobs` splits the same work into `(group, seed)` tasks, so restarts can run as separate jobs (e.g. a SLURM array) that share a `ResultStore`:

```python
from gMRItensor.jobs import DirectoryStore, collect, job_slice, plan_replicability, run_tasks

plan = plan_replicability(HalfHalfEngine(repeats=20, seed=0), len(slices), n_restarts=10)
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
from gMRItensor.plotting import evolving_factors_to_numpy, plot_evolving_mode, plot_subject_mode

fig, axs = plot_subject_mode(model.subject_mode.numpy(), subjects, subject_info, "diagnosis", ["age"])
fig, axs, significance = plot_evolving_mode(
    evolving_factors_to_numpy(model.evolving_states), timepoints, subjects, subject_info, "diagnosis",
)
```

- `subject_info` has a `subjects` column plus the grouping and covariate columns.
- `plot_subject_mode` draws one row per component: a boxplot by group with significance annotations, then a scatter against each of the `plotting_variables`.
- `plot_subject_mode_correlation` draws a component-by-component matrix with group comparisons.
- `plot_evolving_mode` draws per-group mean ± SEM ribbons and individual subject curves. Its per-time-point group comparisons are returned as a table, not drawn.
- `plot_spatial_mode` overlays the spatial mode on brain slices, yielding one figure per region from `region_masks_from_segmentations`. `expand_roi_mode_to_voxels` broadcasts an ROI-level mode out to voxels.
- `plot_mode_grid` combines the spatial, time and subject modes in one figure.
- Figures size to a journal `page_width` in inches. All plotting uses the non-interactive Agg backend.

### ROI tracer evolution

Images are read once; every later step works from the saved tables, and plotting only draws them.

```python
from gMRItensor.group_statistics import compare_roi_groups, load_roi_statistics, summarize_roi_statistics
from gMRItensor.plotting import figure_path, plot_roi_evolution_panels, save_figure
from gMRItensor.preprocessing import write_preprocessed_data
from gMRItensor.roi_groups import resolve_roi_groups

groups = resolve_roi_groups(["ventricles", "white_matter"], custom={"my_region": [17, 53]})
paths = write_preprocessed_data(args_list, "results", roi_groups=groups)  # results/data/*.parquet

stats = load_roi_statistics(paths.roi_statistics, subject_info, "diagnosis", rois=list(groups))
summary = summarize_roi_statistics(stats, "median_concentration")
significance = compare_roi_groups(stats, "median_concentration")

fig, _ = plot_roi_evolution_panels(summary, significance, list(groups), "median_concentration", n_rows=1, n_cols=3, page_width="double")
save_figure(fig, figure_path("results", "overview", None, "median_concentration", "panels"))
```

`args_list` entries are the `write_tracer_parquet` arguments (image paths, `signal_type`, `func`, `subject`, `time_point`). For T1map/R1map, concentration is `ΔR1 / r1` in mM (defaults `relaxivity=3.2` 1/(mM·s) and `time_unit="ms"`), and `total_amount` is in mmol. A single r1 is applied to CSF and parenchyma alike, so parenchymal concentrations are approximate.

## Development

```bash
uv sync --all-extras
uv run pre-commit install
uv run pytest
```

- Package management: `uv add <package>` / `uv add --dev <package>`.
- Code must pass `pre-commit` (black, flake8, reorder-python-imports, add-trailing-comma, mypy).
- Tests live in `test/` and run with `uv run pytest`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## License

[MIT](LICENSE)
