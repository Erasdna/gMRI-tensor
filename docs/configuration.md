# Configuration reference

Each `gmri` stage reads one YAML file. That file names everything the stage reads and writes, so any stage can be run (or rerun) on its own:

| Command | Config | Reads | Writes |
|---|---|---|---|
| `gmri preprocess` | preprocessing | manifest CSV, NIfTI images | `<output_dir>/data/` |
| `gmri plot` | plotting | `roi_statistics.parquet`, subject info CSV | `<output_dir>/roi_analysis/`, `<output_dir>/figures/roi/` |
| `gmri decompose run\|plan\|collect` | decomposition | `tracer.parquet` | `rank_<r>.h5`, `fits.csv` |
| `gmri replicability run\|plan\|collect` | replicability | `tracer.parquet`, subject info CSV | `replicability.csv` |

Commented templates for all four are in [`examples/`](../examples). `gmri <command> --help` lists each command's arguments.

## General rules

- **Paths** are relative to the config file, not to the directory you run `gmri` from. Absolute paths also work.
- **Unknown keys are errors.** A misspelled key fails instead of being ignored. Every error names the file and the field, e.g. `plotting.yaml: grids[1].n_rows: must be >= 1`.
- **Defaults:** keys marked *required* have no default. All other keys can be left out.
- **Numbers:** write `1e-5` or `1.0e-5` freely; both are read as numbers.
- **Provenance:** each stage copies its config to its output directory as `<stage>.yaml`.

## Preprocessing (`gmri preprocess`)

Reads every scan once and writes the tracer table (decomposition input) and per-ROI statistics (plotting input). The optional `regions` block chooses which ROI groups, besides every single label, get statistics.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `manifest` | path | *required* | CSV with one row per scan (format below). Must exist. |
| `output_dir` | path | *required* | Files go to `<output_dir>/data/`. |
| `signal_type` | `T1map` \| `R1map` \| `T1w` | *required* | Image type. T1map/R1map give ΔR1 (and concentrations); T1w gives the post/baseline ratio. |
| `aggregation` | `median` \| `mean` \| `voxel` | `median` | One tracer value per label (median or mean over its voxels), or `voxel` for one row per voxel plus `tracer.coords.parquet` with voxel coordinates. |
| `relaxivity` | number \| `null` | `3.2` | Contrast agent relaxivity r1 in 1/(mM·s); concentration = ΔR1 / r1. `null` leaves the concentration columns NaN. Ignored for T1w. |
| `time_unit` | `ms` \| `s` | `ms` | Unit of the T1/R1 maps. ΔR1 is converted to 1/s before dividing by r1. |
| `n_procs` | integer ≥ 1 | `5` | Parallel worker processes for reading images. |
| `regions.presets` | list of names | `[]` | Predefined ROI groups to compute (list below). |
| `regions.custom` | mapping name → list of label ids | `{}` | Your own ROI groups, e.g. `hippocampus_amygdala: [17, 18, 53, 54]`. A name may not repeat a preset. |
| `regions.csf_offset` | integer | `10000` | Offset of CSF atlas labels relative to the parenchyma label they border. |

### Manifest CSV

```csv
subject,time_point,baseline_path,post_injection_path,mask_path,segmentation_path
sub-01,0,sub-01/T1map_ses-00.nii.gz,sub-01/T1map_ses-00.nii.gz,sub-01/mask.nii.gz,sub-01/aparc+aseg.nii.gz
sub-01,24,sub-01/T1map_ses-00.nii.gz,sub-01/T1map_ses-24.nii.gz,sub-01/mask.nii.gz,sub-01/aparc+aseg.nii.gz
```

- All six columns are required. `time_point` is an integer (hours after injection).
- Paths are relative to the CSV file. The four images of a row must be on the same grid.
- The manifest is checked before any image is read. Empty cells, non-integer time points, duplicate `(subject, time_point)` rows and missing files are each reported.

### ROI group presets

FreeSurfer label ids. CSF variants are the parenchyma ids plus `csf_offset`. Ventricle labels count as CSF with or without the offset.

| Preset | Labels |
|---|---|
| `ventricles` | 4, 5, 14, 15, 43, 44, 72 and the same + offset |
| `cortical_grey_matter` | 3, 42, 1001–1035, 2001–2035 |
| `subcortical_grey_matter` | 10–13, 17, 18, 26, 49–54, 58 |
| `grey_matter` | cortical ∪ subcortical grey matter |
| `white_matter` | 2, 41, 77, 251–255 |
| `cerebellum` | 7, 8, 46, 47 |
| `brainstem` | 16 |
| `basal_ganglia` | 11, 12, 13, 26, 50, 51, 52, 58 |
| `hippocampus`, `amygdala`, `thalamus` | 17, 53 · 18, 54 · 10, 49 |
| `cingulate`, `parahippocampal`, `entorhinal` | 1002, 1010, 1023, 1026 and the 2000-series counterparts · 1016, 2016 · 1006, 2006 |
| `limbic_system` | hippocampus ∪ amygdala ∪ cingulate ∪ parahippocampal ∪ entorhinal |
| `cortical_grey_matter_csf`, `cerebellum_csf`, `brainstem_csf`, `limbic_system_csf` | the matching preset + offset (`brainstem_csf` is the pontine cistern) |
| `all_csf` | every `*_csf` preset ∪ `ventricles` |

### Outputs

| File | Contents |
|---|---|
| `data/tracer.parquet` | Long table `subject, time_point, labels, label_index, values`, read by decomposition and replicability. |
| `data/tracer.coords.parquet` | Voxel coordinates per tensor column (only with `aggregation: voxel`). |
| `data/roi_statistics.parquet` | One row per subject, time point and ROI (each label, plus each configured group). |

Columns of `data/roi_statistics.parquet`:

| Column | Meaning |
|---|---|
| `roi`, `roi_type` | ROI name (a label id as text, or a group name) and `label` or `group`. |
| `n_voxels`, `n_valid` | Voxels in the ROI, and how many have a finite value. |
| `volume_mm3` | ROI volume. |
| `median`, `mean` | Tracer signal. |
| `median_concentration`, `mean_concentration` | Concentration in mM. |
| `total_amount` | Total contrast agent in mmol. |

## Plotting (`gmri plot`)

The stage runs in two steps:
1. **Tables:** for every statistic used by a figure or grid, it writes a per-group summary (mean ± SEM per time point) and per-time-point group comparisons to `roi_analysis/`.
2. **Figures:** it draws the figures from those saved tables.

Every listed ROI is checked against the statistics file before anything is written.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `roi_statistics` | path | *required* | `data/roi_statistics.parquet` from `gmri preprocess`. |
| `subject_info` | path | *required* | CSV with a `subjects` column and the `group_variable` column. Must exist. |
| `group_variable` | string | *required* | Column of `subject_info` to compare, e.g. `diagnosis`. |
| `output_dir` | path | *required* | Tables go to `roi_analysis/`, figures to `figures/roi/`. |
| `min_group_n` | integer ≥ 1 | `2` | A time point is tested only if every group has at least this many subjects there. |
| `alpha` | number in (0, 1) | `0.05` | Significance threshold on Benjamini–Hochberg adjusted p-values (FDR per ROI). Significant time points get a `*`. |
| `figures` | list | `[]` | Individual figures: one per ROI and statistic (keys below). |
| `grids` | list | `[]` | Several ROIs per figure, split over pages if needed (keys below). At least one figure or grid is required. |
| `formats` | list | `[pdf, png]` | File formats to save. |
| `dpi` | integer ≥ 1 | `300` | Resolution of raster formats. |

- **Statistics:** `median`, `mean`, `median_concentration`, `mean_concentration`, `total_amount`.
- **ROI names:** preset or custom group names as configured at preprocessing, or label ids as strings (e.g. `"4"`).
- **Group tests:** Mann–Whitney U for two groups, Kruskal–Wallis for more.

`figures[]` entries:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rois` | list | *required* | ROIs; each gets its own figure per statistic. |
| `statistics` | list | *required* | Statistics to plot. |
| `layout` | `rows` \| `panels` | *required* | `rows`: group ribbon plus one column of subject curves per group. `panels`: ribbon only. |
| `page_width` | `single` \| `onehalf` \| `double` \| inches | `double` | Figure width: 3.5, 5.5 or 7.0 inches, or a number. |

`grids[]` entries:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | *required* | Used for the directory and file names. |
| `rois`, `statistics`, `layout`, `page_width` | | | As for `figures`. |
| `n_rows` | integer ≥ 1 | *required* | Rows per page (`rows` layout: ROIs per page). |
| `n_cols` | integer ≥ 1 | `1` | Columns per page (`panels` layout only). |
| `sharey` | boolean | `false` | One y-range for every panel of a page. |

**Output files:**
- Tables: `roi_analysis/summary__<statistic>.parquet` and `significance__<statistic>.csv`.
- Individual figures: `figures/roi/single/<statistic>/<roi>__<layout>.<ext>`.
- Grids: `figures/roi/<name>/<name>__<statistic>__<layout>[__p<k>].<ext>`. `__p<k>` is added only when the grid needs more than one page.

## Decomposition (`gmri decompose`)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `input` | path | *required* | `data/tracer.parquet` from `gmri preprocess`. |
| `output_dir` | path | *required* | Where `rank_<r>.h5`, `fits.csv` and the restart store go. |
| `method` | `cp` \| `parafac2` | *required* | CP needs every subject at every time point (see `tensor.min_timepoints`). PARAFAC2 allows different time points per subject. |
| `ranks` | list of integers ≥ 1 | *required* | One fit per rank. |
| `tensor` | mapping | | How the tensor is built (below). |
| `fit` | mapping | | Fit settings (below). |
| `distributed` | mapping | | Settings for `plan` / `run --job` / `collect` (below). |

`tensor` (shared with replicability):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `scale` | boolean | `true` | Divide each label by its standard deviation over all subjects and time points before fitting. The scaling is saved with the model. |
| `min_timepoints` | integer \| `null` | `null` | Minimum valid time points to keep a subject. `null` requires all of them. For CP, a lower value leaves NaN rows (see `fit.options.allow_nan_imputation`). |
| `max_invalid_fraction` | number in [0, 1] | `0.9` | Drop a scan session whose fraction of non-finite values is larger. |

`fit` (shared with replicability):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `restarts` | integer ≥ 1 | `50` | Random initialisations per fit; the best accepted one is kept. |
| `max_iter` | integer ≥ 1 | `2000` | Iteration limit per restart. |
| `tolerance` | number > 0 | `1e-5` | Convergence tolerance on the relative reconstruction error. |
| `restart_procs` | integer ≥ 1 | `1` | Parallel processes for the restarts (decomposition only; replicability uses `n_procs`). |
| `options` | mapping | `{}` | Passed unchanged to the decomposition runner. |

`fit.options` examples:
- PARAFAC2: `solver: matcouply`, `nn_modes: auto`, `aoadmm_options: {...}`.
- CP: `non_negative: false`, `allow_nan_imputation: true`.
- Either method: `progress_bar: true`.

An unknown option fails with the runner's error.

`distributed` (shared with replicability):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `store` | string | `restarts` | Directory for per-restart results, relative to `output_dir`. |
| `tasks_per_job` | integer ≥ 1 | `1` | Restarts (fits) per `run --job` job. |

**Outputs:**
- `rank_<r>.h5`: the fitted factors, plus the subjects, time points, label ids and scaling needed to read them.
- `fits.csv`: `rank, error`.

Load a model with `gMRItensor.model_io.load_decomposition`.

## Replicability (`gmri replicability`)

Refits on subsets of subjects and scores how well the fits agree, using the factor match score (FMS). It needs no decomposition outputs.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `input` | path | *required* | `data/tracer.parquet` from `gmri preprocess`. |
| `output_dir` | path | *required* | Where `replicability.csv` and the restart store go. |
| `method` | `cp` \| `parafac2` | *required* | As for decomposition. |
| `ranks` | list of integers ≥ 1 | *required* | Ranks to score. |
| `engine` | `halfhalf` \| `cv` | *required* | `halfhalf`: fit two random halves of the subjects and compare them. `cv`: fit k-fold training sets and compare every pair within a repeat. |
| `repeats` | integer ≥ 1 | *required* | Random splits (`halfhalf`) or repetitions of the k-fold split (`cv`). |
| `splits` | integer ≥ 3 | `null` | Folds for `cv` (required there). Two folds would have no subjects in common. |
| `subject_info` | path | `null` | CSV with a `subjects` column; needed for `stratify_by`. |
| `stratify_by` | string | `null` | `subject_info` column whose proportions every split keeps. |
| `n_procs` | integer ≥ 1 | `1` | Parallel processes for the fits. Use 1 on GPU. |
| `seed` | integer ≥ 0 | `0` | Seed of the splits; every rank uses the same splits. |
| `tensor`, `fit`, `distributed` | | | As for decomposition. The scaling is applied to the full tensor before splitting. |

**Output:** `replicability.csv` has columns `rank, split, fms` for `halfhalf`, or `rank, fold_i, fold_j, n_common, fms` for `cv`.

## Distributed restarts

For decomposition and replicability, `run` without `--job` does everything in one process. To spread the restarts over many jobs (e.g. a SLURM array), split the same work into tasks. Each job stores its results in `<output_dir>/<distributed.store>/rank_<r>/`, and `collect` writes the same output files as a single-process run:

```bash
N=$(gmri decompose plan decomposition.yaml)     # prints only the number of jobs
sbatch --array=0-$((N-1)) job.sh                # job.sh: gmri decompose run decomposition.yaml --job $SLURM_ARRAY_TASK_ID
gmri decompose collect decomposition.yaml       # after every job has finished
```

Things to know before using it:
- **Resuming:** a job skips restarts that are already in the store, so a failed or pre-empted job can simply be re-queued.
- **Missing results:** `collect` refuses to write anything while restarts are missing, and says how many.
- **Changing the config:** don't change `ranks`, `fit.restarts`, `tensor`, the engine settings or `tasks_per_job` between `plan`, `run --job` and `collect`. Those keys define the task list. To start over, delete the store directory.
- **Store files:** they are kept after `collect`. Delete the store directory once you no longer need to resume.
