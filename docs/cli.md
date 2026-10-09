# Command line reference

`gmri` runs the pipeline one step at a time. Every setting is a command-line argument, and each command reads only files that earlier commands wrote, so any step can be rerun on its own from a shell or a job script. `gmri <command> [<action>] --help` lists every argument with its default.

| Command | Reads | Writes |
|---|---|---|
| `gmri preprocess` | manifest CSV, NIfTI images | `<output-dir>/data/` |
| `gmri plot statistics` | `roi_signal.parquet`, subject info CSV | `<output-dir>/roi_analysis/`, `<output-dir>/figures/roi/` |
| `gmri decompose run\|plan\|collect` | `roi_signal.parquet` or `voxels.parquet` | `rank_<r>.h5`, `fits.csv` |
| `gmri replicability run\|plan\|collect` | the same, plus the subject info CSV for stratification | `replicability.csv` |
| `gmri plot decomposition` | `rank_<r>.h5`, subject info CSV, optionally a segmentation | `<output-dir>/figures/decomposition/` |

- **Provenance:** every command records its settings and the package version in `<output-dir>/<command>.json`.
- **Errors:** an invalid argument or a missing input stops the command before it writes anything. The command prints a one-line message and exits with code 2.

## `gmri preprocess`

```bash
gmri preprocess --manifest scans.csv --output-dir results --input-type T1map --time-unit ms [--store-voxels] [--n-procs 5]
```

| Argument | Default | Meaning |
|---|---|---|
| `--manifest` | required | CSV with one row per scan (format below). |
| `--output-dir` | required | Files go to `<output-dir>/data/`. |
| `--input-type` | required | What the images are: `T1map` (T1 relaxation times), `R1map` (R1 relaxation rates) or `T1w` (T1-weighted intensities). |
| `--time-unit` | required for T1map/R1map | Unit of the maps: T1 in `ms`/`s`, R1 in 1/`ms` or 1/`s`. There is no default, because a wrong unit scales ΔR1 by 1000. Ignored for T1w. |
| `--store-voxels` | off | Also store every voxel's value, for a voxel-wise decomposition. All scans must be on one common template. |
| `--n-procs` | `5` | Images read in parallel. |

**Signal:**
- **T1map:** converted to R1 = 1/T1. The signal is ΔR1 = R1_post − R1_baseline, always stored in 1/s whatever `--time-unit` is.
- **R1map:** used directly, giving the same ΔR1 in 1/s.
- **T1w:** the signal is the post/baseline intensity ratio, which has no unit and gives no concentrations.
- **Which voxels:** only voxels inside the mask with a nonzero segmentation label are used. Concentrations are not computed here, but later by `gmri plot statistics`.

**Manifest:**

```csv
subject,time_point,baseline_path,post_injection_path,mask_path,segmentation_path
sub-01,0,sub-01/T1map_ses-00.nii.gz,sub-01/T1map_ses-00.nii.gz,sub-01/mask.nii.gz,sub-01/aparc+aseg.nii.gz
sub-01,24,sub-01/T1map_ses-00.nii.gz,sub-01/T1map_ses-24.nii.gz,sub-01/mask.nii.gz,sub-01/aparc+aseg.nii.gz
```

- `time_point` is an integer, the number of hours after injection.
- Paths are relative to the CSV, and the four images of a row must share one grid.
- The manifest is checked before any image is read. Empty cells, non-integer time points, duplicate `(subject, time_point)` rows and missing files are each reported.

**Outputs** (each file records its kind, input type, signal and unit; read them with `gMRItensor.preprocessing.read_signal_metadata`):

| File | Contents |
|---|---|
| `data/roi_signal.parquet` | One row per subject × time point × label: `median`, `mean`, `n_voxels`, `n_valid` (finite voxels), `voxel_volume_mm3`. |
| `data/voxels.parquet` | With `--store-voxels`: one row per scan × labeled voxel (`labels`, `label_index`, `values`). |
| `data/voxels.coords.parquet` | Each voxel's `(labels, label_index)` key and `(i, j, k)` index. The template `shape` and `affine` are stored in the file metadata. |

## `gmri plot statistics`

```bash
gmri plot statistics --roi-signal results/data/roi_signal.parquet --subject-info subjects.csv \
    --group-variable diagnosis --output-dir results \
    [--region cortical_grey_matter --region mine=17,53] [--rois all | --rois 4 17] \
    [--statistics median median_concentration total_amount] [--relaxivity 3.2]
```

The command combines labels into ROIs and computes the statistics. It then compares the groups at every time point, prints a table of the significant differences and draws one figure per ROI and statistic.

| Argument | Default | Meaning |
|---|---|---|
| `--roi-signal` | required | `roi_signal.parquet` from `gmri preprocess`. |
| `--subject-info` | required | CSV with a `subjects` column and the group column. |
| `--group-variable` | required | Column to compare, e.g. `diagnosis`. Two groups use a Mann–Whitney U test, more use Kruskal–Wallis. P-values are Benjamini–Hochberg adjusted per ROI. |
| `--output-dir` | required | Results directory. |
| `--region SPEC` | every preset in the data | One ROI per region: a preset name (table below) or `name=ids`, e.g. `mine=17,53,1001-1035`. Repeatable. |
| `--rois` | none | `all`, or label ids to plot one by one. |
| `--statistics` | `median` | Any of `median`, `mean`, `median_concentration`, `mean_concentration`, `total_amount`; each gets its own tables. |
| `--relaxivity` | `3.2` | r1 in 1/(mM·s) for concentrations. It is an error for T1w input. |
| `--alpha` | `0.05` | Threshold on adjusted p-values. Significant time points get a `*` in the figures. |
| `--min-group-n` | `2` | A time point is tested only if every group has at least this many subjects there. |
| `--layout` | `rows` | `rows`: mean ± SEM ribbon plus one column of subject curves per group. `panels`: ribbon only. |
| `--page-width`, `--formats`, `--dpi` | `double`, `pdf png`, `300` | Figure width (`single` 3.5", `onehalf` 5.5", `double` 7.0", or inches), file formats and resolution. |

**How ROI values are combined:**
- **Single labels:** the values are copied from `roi_signal`.
- **Region counts:** `n_voxels`, `n_valid` and the volume are summed.
- **Region `mean`:** a mean weighted by `n_valid`, which is exact.
- **Region `median`:** the `n_valid`-weighted median of the label medians. This approximates the median of all the region's voxels, and is exact for one label.
- **Concentrations:** ΔR1 / r1 in mM, exact because the conversion is linear. `total_amount` is mean concentration × valid volume, in mmol. One r1 is applied to every region, so parenchymal concentrations are approximate.

**Outputs:**
- `roi_analysis/roi_statistics.parquet`: every ROI's values per scan.
- For each statistic, `roi_analysis/summary__<stat>.parquet` (mean, SEM and n per group and time point) and `roi_analysis/significance__<stat>.csv`.
- `roi_analysis/summary.csv`: for each ROI and statistic, the significant time points, the smallest adjusted p-value and the group with the higher mean there.
- `figures/roi/single/<stat>/<roi>__<layout>.<ext>`.

### ROI presets

FreeSurfer label ids. CSF labels are the parenchyma ids + 10000. Ventricles count as CSF with or without the offset.

| Preset | Labels |
|---|---|
| `ventricles` | 4, 5, 14, 15, 43, 44, 72 and the same + 10000 |
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
| `cortical_grey_matter_csf`, `cerebellum_csf`, `brainstem_csf`, `limbic_system_csf` | the matching preset + 10000 (`brainstem_csf` is the pontine cistern) |
| `all_csf` | every `*_csf` preset ∪ `ventricles` |

## `gmri decompose`

```bash
gmri decompose run --input results/data/roi_signal.parquet --output-dir results/parafac2 \
    --method parafac2 --ranks 2 3 4 [--statistic median] [--restarts 50] [--non-negative-modes auto]
```

| Argument | Default | Meaning |
|---|---|---|
| `--input` | required | `roi_signal.parquet` for a per-ROI fit (one column per label), or `voxels.parquet` for a per-voxel fit. |
| `--output-dir` | required | Results directory. |
| `--method` | required | `cp` (one shared time mode; needs every subject at every time point) or `parafac2` (each subject has its own time course). |
| `--ranks` | required | One fit per rank. |
| `--statistic` | `median` | Per-ROI statistic to decompose (`roi_signal` input only). |
| `--no-scale` | scale | Do not divide each column by its standard deviation over subjects and time points. |
| `--center` | off | Subtract each column's mean. Centered data is negative, so this needs `--non-negative-modes none`. |
| `--min-timepoints` | all | Minimum valid time points to keep a subject. For CP, a lower value leaves NaN rows (see `--fit-option allow_nan_imputation=true`). |
| `--max-invalid-fraction` | `0.9` | Drop a scan whose fraction of non-finite values is larger. |
| `--restarts` | `50` | Random initialisations per fit; the best accepted one is kept. |
| `--max-iter`, `--tolerance` | `2000`, `1e-5` | Iteration limit and convergence tolerance per restart. |
| `--restart-procs` | `1` | Parallel processes for the restarts. |
| `--non-negative-modes` | `auto` | Modes 0 = subject, 1 = time, 2 = label. See the table below. |
| `--solver` | `tensorly` | PARAFAC2 solver: `tensorly` (ALS) or `matcouply` (AO-ADMM, which can also constrain mode 1). |
| `--fit-option KEY=VALUE` | | Any other `run_*_decomposition_repeated` option, with VALUE read as JSON, e.g. `--fit-option progress_bar=true` or `--fit-option 'aoadmm_options={"l2_penalty": 0.1}'`. Repeatable. |
| `--tasks-per-job`, `--store` | `1`, `restarts` | Distributed runs (below). |

| `--non-negative-modes` | PARAFAC2 | CP |
|---|---|---|
| `auto` | the solver's default: `0,2` (tensorly) or `0,1,2` (matcouply) | all modes |
| `none` | unconstrained | unconstrained |
| e.g. `0,2` | exactly those modes (mode 1 needs `--solver matcouply`) | only `0,1,2`: CP constrains all modes or none |

**Outputs:**
- `rank_<r>.h5`: the fitted factors, plus the subjects, time points, label ids and scaling needed to read them. Voxel models also store each voxel's template coordinates. Load a model with `gMRItensor.model_io.load_decomposition`.
- `fits.csv`: `rank, error`.

## `gmri replicability`

`gmri replicability run|plan|collect` takes the same arguments as `decompose`, plus:

| Argument | Default | Meaning |
|---|---|---|
| `--engine` | required | `halfhalf`: fit two random halves of the subjects and compare them. `cv`: fit k-fold training sets and compare every pair within a repeat. |
| `--repeats` | required | Random splits (`halfhalf`) or repetitions of the k-fold split (`cv`). |
| `--splits` | | Folds for `cv`, at least 3. Two folds would share no subjects. |
| `--subject-info`, `--stratify-by` | | Keep the proportions of a subject-info column, e.g. `diagnosis`, equal across splits. |
| `--n-procs` | `1` | Parallel fits. Use 1 on a GPU. |
| `--seed` | `0` | Seed of the splits; every rank uses the same splits. |

The output is `replicability.csv`. It has columns `rank, split, fms` for `halfhalf`, or `rank, fold_i, fold_j, n_common, fms` for `cv`, where fms is the factor match score. Scaling and centering are applied to the full tensor before splitting.

## Distributed restarts (SLURM)

For `decompose` and `replicability`, `run` without `--job` fits everything in one process. To spread the restarts over many jobs instead:

```bash
N=$(gmri decompose plan --input results/data/roi_signal.parquet --output-dir results/parafac2 \
        --method parafac2 --ranks 2 3 4 --restarts 50 --tasks-per-job 4)    # saves plan.json, prints 38
sbatch --array=0-$((N-1)) job.sh
gmri decompose collect --output-dir results/parafac2                         # after every job has finished
```

with `job.sh` running

```bash
gmri decompose run --output-dir results/parafac2 --job $SLURM_ARRAY_TASK_ID
```

- **Settings:** `plan` saves all arguments to `<output-dir>/plan.json`, so `run --job` and `collect` need only `--output-dir`. All jobs therefore use the same settings.
- **Tasks:** each restart of each fit is one task, and a job runs `--tasks-per-job` of them, one after another or in parallel with `--restart-procs` (decompose) / `--n-procs` (replicability). Request `--cpus-per-task` to match.

| Command | Tasks | Example, `--tasks-per-job 4` |
|---|---|---|
| decompose | `ranks × restarts` | 3 ranks × 50 restarts = 150 tasks → 38 jobs |
| replicability, halfhalf | `ranks × 2 × repeats × restarts` | 3 × 40 halves × 20 = 2400 tasks → 600 jobs |
| replicability, cv | `ranks × splits × repeats × restarts` | |

- **Array size limits:** many clusters cap array size (`MaxArraySize`, often 1001). If `plan` prints more jobs than that, raise `--tasks-per-job`.
- **Resuming:** results are kept in `<output-dir>/<store>/rank_<r>/`. A re-queued job skips restarts that are already there, so failed jobs can simply be resubmitted.
- **Collecting:** `collect` refuses to write while restarts are missing, and says how many. It writes the same files as a single-process `run`.
- **Starting over:** run `plan` again and delete the store directory.

## `gmri plot decomposition`

```bash
gmri plot decomposition --model results/parafac2/rank_3.h5 --subject-info subjects.csv \
    --group-variable diagnosis --output-dir results --segmentation template_seg.nii.gz \
    [--mode-grid] [--subject-mode] [--time] [--spatial] [--background template_T1.nii.gz] [--slices 90 110 80]
```

| Argument | Default | Meaning |
|---|---|---|
| `--model` | required | `rank_<r>.h5` from `gmri decompose`. |
| `--subject-info`, `--group-variable` | required | Subjects and the column to colour and compare by. |
| `--output-dir` | required | Figures go to `figures/decomposition/<model>__<part>.<ext>`. |
| `--mode-grid` | | Time, subject and spatial modes in one figure, with CSF and parenchyma shown separately. |
| `--subject-mode` | | Group boxplot per component, plus a scatter against each `--covariates` column. |
| `--time` | | CP: the shared time mode. PARAFAC2: group ribbons and subject curves of the evolving mode; its group tests are saved as `<model>__evolving_mode_significance.csv`. |
| `--spatial` | | The spatial mode on three slices, one figure for parenchyma and one for CSF. |
| `--segmentation` | | Label volume in the decomposition's label space (one merged file, CSF ids + 10000). Needed for the spatial parts of a per-ROI model; per-voxel models carry their own coordinates. |
| `--background` | segmentation mask | Image to draw the spatial maps on, on the template grid. |
| `--slices I J K` | centre of mass | Slice indices. The mode grid uses `I` (sagittal). |
| `--covariates` | none | Subject-info columns to scatter the subject mode against. |
| `--alpha`, `--min-group-n` | `0.05`, `2` | Group tests of the evolving mode. |
| `--page-width`, `--formats`, `--dpi` | `double`, `pdf png`, `300` | As for `plot statistics`. |

- **Part flags:** without any, every part is drawn.
- **Missing segmentation:** if a per-ROI model is plotted without `--segmentation`, the spatial parts are skipped with a note. Asking for one explicitly without a segmentation is an error.

## Custom figures from scripts

Figure grids and other layouts are built in Python from the saved tables:

```python
import pandas as pd
from gMRItensor.plotting import plot_roi_evolution_panels, save_figure

summary = pd.read_parquet("results/roi_analysis/summary__median.parquet")
significance = pd.read_csv("results/roi_analysis/significance__median.csv", dtype={"roi": str})
rois = ["ventricles", "white_matter", "cortical_grey_matter", "thalamus"]
fig, axs = plot_roi_evolution_panels(summary, significance, rois, "median", n_rows=2, n_cols=2, page_width="double")
save_figure(fig, "results/figures/overview")
```

- **ROI rows layout:** `plot_roi_evolution_rows` draws several ROIs as rows. It also takes the per-scan values (`roi_statistics.parquet`) for the subject curves.
- **Decomposition figures:** use `gMRItensor.model_io.load_decomposition` with `plot_mode_grid`, `plot_subject_mode`, `plot_evolving_mode`, `gMRItensor.plotting.time_mode.plot_time_mode` and `plot_spatial_mode`.
- **Building blocks:** `gMRItensor.pipeline` exposes each step for scripts, e.g. `run_statistics_plots(StatisticsPlotOptions(...))`.
