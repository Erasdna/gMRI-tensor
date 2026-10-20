# Command line reference

`gmri` runs the pipeline one step at a time. Every setting is a command-line argument, and each command reads only files that earlier commands wrote, so any step can be rerun on its own from a shell or a job script. `gmri <command> [<action>] --help` lists every argument with its default.

| Command | Reads | Writes |
|---|---|---|
| `gmri preprocess` | manifest CSV, NIfTI images | `<output-dir>/data/` |
| `gmri plot statistics` | `roi_signal.parquet`, subject info CSV | `<output-dir>/roi_analysis/`, `<output-dir>/figures/roi/` |
| `gmri decompose run\|plan\|collect` | `roi_signal.parquet` or `voxels.parquet` | one directory per model, `<method>_rank_<r>/` (`model.h5`, `decompose.json`), `fits.csv`, the restarts and their reproducibility in `<store>/<method>_rank_<r>/` |
| `gmri replicability run\|plan\|collect` | the same, plus the subject info CSV with `--stratify-by` | `replicability.csv` |
| `gmri plot decomposition` | a model directory, subject info CSV, optionally a segmentation; for `--outliers`, the data the model was fitted to | `<output-dir>/figures/decomposition/` |
| `gmri plot group-maps` | `roi_signal.parquet` (plus a segmentation) or `voxels.parquet`, subject info CSV | `<output-dir>/figures/group_maps/` |
| `gmri plot reproducibility` | a `reproducibility.csv` | the figure, next to the CSV or at `--output` |

- **Provenance:** every command records its settings and the package version as JSON, with absolute paths: `preprocess.json`, `plot_statistics.json`, `replicability.json`, `plot_decomposition.json` and `plot_group_maps.json` in `<output-dir>`, and each model's `decompose.json` in its own directory. `plan` saves `plan.json` instead; `run --job` and `plot reproducibility` write none.
- **Errors:** an invalid argument or a missing input stops the command before it writes anything. The command prints a one-line message and exits with code 2.
- **Arguments:** flags must be written out in full; abbreviations such as `--tol` are refused. An argument that would have no effect in its combination (e.g. `--splits` with `--engine halfhalf`) is an error rather than being ignored.

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
- **Invalid voxels:** a 0 in either map of a T1map or R1map pair marks a voxel the mapping failed on; it is not finite in the output and not counted in `n_valid`.
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
| `data/voxels.coords.parquet` | Each voxel's `(labels, label_index)` key and `(i, j, k)` index. The template `shape` and `affine` are stored in the file metadata, with `template`, the resolved path of the first scan's segmentation. That path is a record only: the template check compares labels and voxel coordinates, not files. |

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
| `--sd-bands` | off | Draw the cohort mean ± 1, 2 and 3 SD (all subjects, regardless of group) behind the subject curves, to spot outlying subjects. Needs `--layout rows`. |
| `--page-width`, `--formats`, `--dpi` | `double`, `pdf png`, `300` | Figure width (`single` 3.5", `onehalf` 5.5", `double` 7.0", or inches), file formats and resolution. |

**Subjects and tests:**
- A subject whose `--group-variable` cell is blank is left out, and the command names it. A subject missing from the subject info is an error.
- A time point where every value is tied (e.g. a t=0 scan used as its own baseline) is not tested, so it does not enter the Benjamini–Hochberg correction.

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
- With `--sd-bands`, for each statistic: `roi_analysis/cohort__<stat>.parquet` (cohort `mean`, `std` and `n` per ROI and time point) and `roi_analysis/outside_3sd__<stat>.csv` (each subject with a point beyond ±3 SD: its most extreme time point, value and `z`). With 10 or fewer subjects no point can lie beyond 3 SD.
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
| `parenchyma` | every label 1–9999 except the ventricles and FreeSurfer CSF (24) |

Figures label presets with short names (e.g. `white_matter` → "WM", `cortical_grey_matter_csf` → "Supratentorial CSF"); file names keep the preset name.

## `gmri decompose`

```bash
gmri decompose run --input results/data/roi_signal.parquet --output-dir results/parafac2 \
    --method parafac2 --ranks 2 3 4 [--statistic median] [--restarts 50] [--non-negative-modes auto]
```

| Argument | Default | Meaning |
|---|---|---|
| `--input` | required | `roi_signal.parquet` for a per-ROI fit (one column per label), or `voxels.parquet` for a per-voxel fit. |
| `--output-dir` | required | Results directory. |
| `--method` | required | `cp`, `parafac2` or `cmf` (see Methods below). |
| `--ranks` | required | One fit per rank. |
| `--statistic` | `median` | Per-ROI statistic to decompose (`roi_signal` input only). |
| `--no-scale` | scale | Do not divide each column by its standard deviation over subjects and time points. |
| `--center` | off | Subtract each column's mean. Centered data is negative, so this needs `--non-negative-modes none`. |
| `--min-timepoints` | all | Minimum valid time points to keep a subject. CP imputes the missing sessions of the subjects it keeps, with a warning. PARAFAC2 and CMF fit those subjects with shorter time courses. |
| `--max-invalid-fraction` | `0.9` | A scan whose fraction of non-finite values is larger counts as missing. In the scans that are kept, CP imputes the remaining non-finite values and drops only the columns that are never finite. PARAFAC2 and CMF drop every column that is non-finite in any kept scan. `0` imputes no scattered values, `1` imputes all it can. CP's reported error covers the observed entries only. |
| `--restarts` | `50` | Random initialisations per fit; the best accepted one is kept. |
| `--max-iter`, `--tolerance` | `2000`, see below | Iteration limit and convergence tolerance per restart. Without `--tolerance`: TensorLy's own `tol` for CP (`1e-8`, non-negative `1e-6`) and PARAFAC2 (`1e-8`); for matcouply (PARAFAC2 and CMF) `1e-7` on the relative change of its penalized objective (matcouply's own default is `1e-8`; set it with `--fit-option aoadmm_loss_tolerance=1e-8`). A TensorLy restart that has not converged by its `tol` within `--max-iter` is rejected; a matcouply one is kept, as matcouply returns it, with a warning. With matcouply, `--tolerance` adds a gate on the reconstruction-error change: restarts that settle within it are accepted, even at the iteration limit, and the rest are rejected. |
| `--restart-procs` | `1` | Parallel processes for the restarts (`decompose` only; `replicability` uses `--n-procs`). |
| `--non-negative-modes` | `auto` | Modes 0 = subject, 1 = time, 2 = label. See the table below. |
| `--solver` | `matcouply` | PARAFAC2 solver: `matcouply` (AO-ADMM, which can also constrain mode 1) or `tensorly` (ALS). CMF always uses matcouply. |
| `--fit-option KEY=VALUE` | | Any other `run_*_decomposition_repeated` option, with VALUE read as JSON (or else as plain text), e.g. `--fit-option progress_bar=true` or `--fit-option 'aoadmm_options={"l2_penalty": 0.1}'`. Repeatable. Options with their own flag, and those the pipeline sets (such as `compute_device`), are refused. Fits are never normalised: the scale stays in the factors and the weights are ones. The figures and `<model>__component_weights.csv` do the normalising. |
| `--tasks-per-job`, `--store` | `1`, `restarts` | Distributed runs (below); `--tasks-per-job` only for `plan`. |

| `--non-negative-modes` | PARAFAC2 | CMF | CP |
|---|---|---|---|
| `auto` | the solver's default: `0,1,2` (matcouply) or `0,2` (tensorly) | `1,2` | all modes |
| `none` | unconstrained | unconstrained | unconstrained |
| e.g. `0,2` | exactly those modes (mode 1 needs `--solver matcouply`) | exactly those modes (mode 0 is fixed, so it has no effect) | only `0,1,2`: CP constrains all modes or none |

**Methods:**
- **`cp`:** one time course per component, shared by all subjects. Needs every subject at every time point.
- **`parafac2`:** each subject has its own time course per component (`B_i`). The PARAFAC2 constraint couples them: `B_iᵀB_i` is the same for every subject.
- **`cmf`:** non-negative coupled matrix factorization, fitted with matcouply's `cmf_aoadmm`, as in Chatzis et al., *Revealing Subject-specific Temporal Patterns from Longitudinal Data*.
  - **Model:** subject i's data is `B_i Cᵀ`. The label mode `C` is shared, and the time courses `B_i` are free, so subjects may differ in shape as well as amplitude.
  - **Subject weights:** fixed at one during the fit, so each `B_i` carries its subject's amplitude.
  - **Saved subject mode:** derived from the time courses, not fitted. Each entry is the RMS of that subject's time course over its time points, and it is what the subject-mode plots show.
  - **Replicability:** the subject mode is not scored, because it is derived. `halfhalf` compares `C`; `cv` compares `C` and the stacked `B_i` of the shared subjects.
  - **Solver settings:** matcouply's own defaults, except the loss tolerance (`1e-7`, above). Penalties and solver settings go through `--fit-option 'aoadmm_options={...}'`, e.g. `{"inner_n_iter_max": 20, "absolute_tol": 1e-6, "feasibility_tol": 1e-5, "inner_tol": 1e-5}`.

**Outputs:**
- `<method>_rank_<r>/model.h5` (e.g. `cmf_rank_3/model.h5`): the fitted factors, components ordered by weight, largest first (`model_io.component_weights`: the weight times each mode's L2 norm), plus the subjects, time points, label ids and scaling needed to read them. Voxel models also store each voxel's template coordinates, the template's shape and affine, and the recorded template path. Load a model with `gMRItensor.model_io.load_decomposition`.
- `<method>_rank_<r>/decompose.json`: the settings that model was fitted with (input, statistic, tensor and fit options). `plot decomposition --outliers` reloads the data from it, so several methods and ranks can share one `--output-dir`.
- `fits.csv`: `method, rank, error, model`, one row per model in `--output-dir`. A later run replaces the rows of the methods and ranks it fits and keeps the others.
- **A rank that does not converge:** the ranks that did are still written; the command then exits with code 2 and names the failed ranks.
- `<store>/<method>_rank_<r>/full/seed_<NNNN>.pt`: every restart, failures included. Both `run` and `collect` keep them. A `run` without `--job` refits every restart and overwrites what the store held.
- `<store>/<method>_rank_<r>/reproducibility.csv` and `reproducibility.{pdf,png}`: how consistently the restarts reach the same minimum.
  - **Contents:** the factor match score between every pair of converged restarts. Rows are sorted by reconstruction error, and the columns are `seed, error`, then one column per seed in the same order.
  - **What to look for:** if the low-error restarts found the same minimum, they form a block of FMS close to 1 in the top-left corner.
  - **Scoring:** all modes are compared and weights are ignored. CMF's derived subject mode is not scored, as in replicability.
  - **Redraw:** use `gmri plot reproducibility`.

## `gmri replicability`

`gmri replicability run|plan|collect` takes the same arguments as `decompose`, plus:

| Argument | Default | Meaning |
|---|---|---|
| `--engine` | required | `halfhalf`: fit two random halves of the subjects and compare them. `cv`: fit k-fold training sets and compare every pair within a repeat. |
| `--repeats` | required | Random splits (`halfhalf`) or repetitions of the k-fold split (`cv`). |
| `--splits` | | Folds for `cv` only, at least 3. Two folds would share no subjects. |
| `--subject-info`, `--stratify-by` | | Keep the proportions of a subject-info column, e.g. `diagnosis`, equal across splits. Given together; a subject with a blank cell is an error. |
| `--n-procs` | `1` | Parallel fits (every split and restart in one pool). Use 1 on a GPU. `--restart-procs` does not apply. |
| `--seed` | `0` | Seed of the splits; every rank uses the same splits. |

The output is `replicability.csv`. It has columns `method, rank, split, fms` for `halfhalf`, or `method, rank, fold_i, fold_j, n_common, fms` for `cv`, where fms is the factor match score. Scaling and centering are applied to the full tensor before splitting. A single-process `run` keeps its fits in memory: only the distributed `plan` / `run --job` / `collect` path writes them to the store.

## CPU and GPU

- **CPU (default):** every fit runs on TensorLy's numpy backend in float64, using `CPUS_PER_TASK` threads (1 if unset) for both torch and numpy's BLAS. The solvers' inner loops make many small calls, which numpy runs several times faster than torch. `GMRITENSOR_FIT_BACKEND=torch` runs CPU fits on torch instead (in the input's precision, float32), as before 0.2.0.
- **GPU:** set `GMRITENSOR_USE_GPU=TRUE`. It applies to `decompose run` and `replicability run`, with or without `--job`, and to every method. If CUDA is available, the input tensor is moved to the GPU once it has been built, and every restart fits there, always on TensorLy's pytorch backend (`GMRITENSOR_FIT_BACKEND` has no effect). Without CUDA the fits stay on the CPU.
- **Memory:** on the GPU, the whole input tensor must fit in GPU memory.
- **Parallelism:** fits run one after another on the GPU, so keep `--restart-procs` / `--n-procs` at 1; larger values are refused.
- **Precision:** TF32 matrix products are enabled on the GPU, so float32 fits (CP, tensorly PARAFAC2) differ slightly from CPU fits. matcouply fits run in float64 everywhere.

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

- **Settings:** `plan` saves all arguments to `<output-dir>/plan.json`, so `run --job` and `collect` need only `--output-dir`. All jobs therefore use the same settings: `run --job` refuses any other argument, and `--fit-option` cannot override a setting that has its own flag (such as `max_iter`). Run `plan` again to change the settings.
- **Changed settings:** every stored restart records a fingerprint of the settings that change a fit (input, statistic, tensor and fit options). After `plan` with other such settings, `run --job` and `collect` refuse the restarts fitted under the old ones instead of reusing them: use a fresh `--store`, or delete the store. More `--restarts` or another `--tasks-per-job` keep what is stored.
- **Tasks:** each restart of each fit is one task, and a job runs `--tasks-per-job` of them, one after another or in parallel with `--restart-procs` (decompose) / `--n-procs` (replicability). Request `--cpus-per-task` to match.

| Command | Tasks | Example, `--tasks-per-job 4` |
|---|---|---|
| decompose | `ranks × restarts` | 3 ranks × 50 restarts = 150 tasks → 38 jobs |
| replicability, halfhalf | `ranks × 2 × repeats × restarts` | 3 × 40 halves × 20 = 2400 tasks → 600 jobs |
| replicability, cv | `ranks × splits × repeats × restarts` | |

- **Array size limits:** many clusters cap array size (`MaxArraySize`, often 1001). If `plan` prints more jobs than that, raise `--tasks-per-job`.
- **Resuming:** results are kept in `<output-dir>/<store>/<method>_rank_<r>/`. A re-queued job skips restarts that are already there, so failed jobs can simply be resubmitted. A single-process `decompose run` uses the same store, but refits every restart. A restart file that cannot be read (e.g. after a node crash) stops the command with its path; delete it to fit that restart again.
- **Collecting:** `collect` refuses to write while restarts are missing, and says how many. It writes the same files as a single-process `run`.
- **Starting over:** run `plan` again and delete the store directory.

## `gmri plot decomposition`

```bash
gmri plot decomposition --model results/parafac2/parafac2_rank_3 --subject-info subjects.csv \
    --group-variable diagnosis --output-dir results --segmentation template_seg.nii.gz \
    [--mode-grid] [--subject-mode] [--time] [--spatial] [--outliers] [--background template_T1.nii.gz] \
    [--slices 90 110 80] [--categorical sex apoe] [--numerical age mmse] [--subject-prefix basic]
```

| Argument | Default | Meaning |
|---|---|---|
| `--model` | required | A model directory `<method>_rank_<r>/` from `gmri decompose` (or its `model.h5`). Figures are named after it, e.g. `cmf_rank_3__mode_grid.png`. Components are drawn largest weight first, also for models saved before sorting existed. |
| `--subject-info`, `--group-variable` | required | Subjects and the column to colour and compare by. |
| `--output-dir` | required | Figures go to `figures/decomposition/<model>__<part>.<ext>`, with parts `mode_grid`, `subject_mode`, `time_mode` (CP) or `evolving_mode` (PARAFAC2, CMF), `spatial_parenchyma`, `spatial_csf` and `outliers`. Every run also writes `<model>__component_weights.csv` there: each component's amplitude (`weight`, and `relative_weight`, its share of the total). The figures draw every mode normalised, so this is the scale they leave out. |
| `--mode-grid` | | Time, subject and spatial modes in one figure, with CSF and parenchyma shown separately. A voxel is CSF if its label is ≥ 10000, a ventricle id or FreeSurfer's CSF (24); everything else is parenchyma. |
| `--subject-mode` | | One row per component. Boxplots by `--group-variable`, with an extra hollow `All` box of every subject, then by each `--categorical` column, drawn as light grey boxes with every subject as a dot in its group's colour (colour always means the `--group-variable` group); then a scatter against each `--numerical` column, with one fit line per group and a black one over all subjects (`All` in the legend). Each row shares one y-axis. |
| `--time` | | CP: the shared time mode. PARAFAC2 and CMF: group ribbons and subject curves of the evolving mode, each subject's curve normalised to its shape. A subject whose subject-mode value is below `--min-subject-coefficient` in absolute value is drawn as zero for that component and left out of its ribbon and group tests (also in `--mode-grid`). The group tests are saved as `<model>__evolving_mode_significance.csv`. |
| `--spatial` | | The spatial mode on three slices, one figure for parenchyma and one for CSF. For a per-ROI model, labels of the segmentation that the model has no row for (e.g. an ROI dropped for missing data) are left blank. |
| `--outliers` | | Leverage of each subject in the subject mode against its residual (mean squared, normalised to sum to 1), after Fida et al. (2022, §3.3), with Bonferroni-corrected cut-offs: leverage (Belsley's F test, Hotelling's T² at rank 1), residual (scaled χ², Box 1954) and a robust residual cut-off from the median and MAD. Writes `<model>__outliers.csv` (`subject, leverage, residual, high_leverage, high_residual, high_residual_robust`) and prints the flagged subjects ("Possible outliers: ..."). Every fitted subject is scored, with or without a group. The data is reloaded with the settings in the model's `decompose.json`; it must match the model (subjects, labels, time points, centring and scaling), or the part stops with an error. A subject mode without full column rank gets no leverage cut-off. |
| `--segmentation` | | Label volume in the decomposition's label space (one merged file, CSF ids + 10000). Needed for the spatial parts of a per-ROI model. Per-voxel models carry their own coordinates; a segmentation given for one must be on the model's template grid (shape and affine), or the command stops with an error. |
| `--background` | segmentation mask | Image to draw the spatial maps on (for a per-voxel model without a segmentation, the voxels the model covers). It must be on the template grid (the segmentation's, for a per-ROI model), or the command stops with an error. |
| `--slices I J K` | centre of mass | Voxel indices of the volume reoriented to RAS (nibabel's `as_closest_canonical`), not of the file as stored; each within the grid. The mode grid uses `I` (sagittal). |
| `--categorical` | none | Subject-info columns to box-plot the subject mode by. Every value is a category, numeric codes too (e.g. APOE 2/3/4). |
| `--numerical` | none | Subject-info columns to scatter the subject mode against. A value that is not a number is an error. |
| `--min-panel-width` | none | Inches. With many variables, the subject-mode panels get narrow; this widens that figure beyond `--page-width` until each panel is at least this wide, e.g. `2`. |
| `--subject-prefix` | none | Prepended as `<prefix>_` to the subject-mode figure name, e.g. `sleep_cmf_rank_3__subject_mode.png`. Other parts keep their names. |
| `--alpha`, `--min-group-n` | `0.05`, `2` | Group tests of the evolving mode. |
| `--outlier-alpha` | `0.01` | Family-wise p for the outlier cut-offs, Bonferroni-corrected over subjects (`p = alpha / n_subjects`). |
| `--input` | the recorded input | For `--outliers`: the data file the model was fitted to, if it has moved. Every other setting still comes from `decompose.json`. |
| `--min-subject-coefficient C` | `1e-6` | PARAFAC2 and CMF only (CP has no per-subject time curves, so it is ignored there). Cutoff on the raw subject mode, which fits never normalise: a subject whose subject-mode value is below C in absolute value for a component is drawn as zero in the evolving mode and left out of its ribbons and group tests, in `--time` and `--mode-grid` alike. With 0, only loadings the fit set exactly to zero are dropped. |
| `--page-width`, `--formats`, `--dpi` | `double`, `pdf png`, `300` | As for `plot statistics`. |

- **Part flags:** without any, every part is drawn.
- **Subject variables:** every column named by `--group-variable`, `--categorical` or `--numerical` must be in the subject info; otherwise the command stops before loading the model and lists the available columns. A blank cell leaves that subject out of that variable's panels only. A subject with a blank `--group-variable` cell, or no subject-info row, is left out of the mode grid, subject mode and evolving mode (the command names them); every group must keep at least 2 subjects, or the command stops with an error.
- **Group comparisons:** in each boxplot, every pair of categories gets a Mann–Whitney U test, and significant pairs are starred (no multiple-comparison correction). The `All` box shows which subjects are outliers for the whole dataset rather than within their group, and is never tested.
- **Category order:** categories are sorted, so a group has the same colour in every figure.
- **Several variable sets:** run `--subject-mode` once per set, each with its own prefix, so the figures sit side by side:

  ```bash
  gmri plot decomposition ... --subject-mode --categorical sex apoe --numerical age --subject-prefix basic
  gmri plot decomposition ... --subject-mode --numerical sleep_efficiency waso --subject-prefix sleep
  ```
- **Missing segmentation:** if a per-ROI model is plotted without `--segmentation`, the spatial parts are skipped with a note. Asking for one explicitly without a segmentation is an error.
- **Outliers without the data:** if no part flag is given and the outlier part cannot reload the fitted data (no `decompose.json`, a moved input, or data that no longer matches the model), it is skipped with a note and the other parts are drawn. With `--outliers` given explicitly, this is an error.

## `gmri plot group-maps`

```bash
gmri plot group-maps --input results/data/voxels.parquet --subject-info subjects.csv \
    --group-variable diagnosis --output-dir results \
    [--aggregate trimmed|mean|median] [--trim 2] [--timepoints 6 24] [--background template_T1.nii.gz] [--slices 90 110 80]
```

At every time point, each group's subjects are aggregated per ROI or voxel. The maps are drawn like `plot decomposition --spatial`: one figure per time point and region (parenchyma, CSF), with one row per group.

| Argument | Default | Meaning |
|---|---|---|
| `--input` | required | `voxels.parquet` for per-voxel maps, or `roi_signal.parquet` for per-ROI maps painted onto `--segmentation`. |
| `--subject-info`, `--group-variable` | required | Subjects and the column that defines the groups. |
| `--output-dir` | required | Figures go to `figures/group_maps/<input>__<aggregate>__t<time point>__<region>.<ext>`. |
| `--aggregate` | `trimmed` | Across the subjects of a group: `trimmed` (the mean after dropping the `--trim` highest and lowest values), `mean` or `median`. Non-finite values are ignored. |
| `--trim` | `2` | Values dropped at each end for `trimmed`. With fewer than `2 × trim + 1` values, the median is used instead. |
| `--statistic` | `median` | Per-ROI statistic to aggregate (`roi_signal` input only). |
| `--segmentation` | | Label volume in `roi_signal`'s label space. Required for per-ROI input. For voxel input, it must be on the template grid. |
| `--background`, `--slices I J K` | segmentation mask, centre of mass | As for `plot decomposition`. |
| `--timepoints` | all | Time points to draw. Each must be in the input. |
| `--page-width`, `--formats`, `--dpi` | `double`, `pdf png`, `300` | As for `plot statistics`. |

- **Rows:** labelled `<group> (n=<subjects scanned at that time point>)`. A group with no scan at a time point keeps an empty row, so the rows match across figures. Subjects with a blank `--group-variable` cell are left out, and named.
- **Colours:** one range per region, the 5th–95th percentile of every group and time point, so the figures of a region are comparable.

## `gmri plot reproducibility`

```bash
gmri plot reproducibility results/parafac2/restarts/parafac2_rank_3/reproducibility.csv \
    [--output figures/reproducibility_rank_3] [--page-width single] [--formats pdf png] [--dpi 300]
```

Redraws a `reproducibility.csv` written by `decompose run` or `collect`. The figure is the FMS matrix, sorted by reconstruction error, with each restart's error above it. The colour range spans the FMS between distinct restarts, not 0–1, so differences stay visible when every score is close to 1. Read the colourbar before comparing figures. In Python, `plot_reproducibility(..., vmin=0, vmax=1)` fixes the range. It is saved next to the CSV unless `--output` (a path without extension) is given. `--page-width`, `--formats` and `--dpi` work as for `plot statistics`; the default width here is `double`, while the figure `decompose` writes is `single`. In Python, the equivalent is `plot_reproducibility(*read_reproducibility(csv))`.

## Custom figures from scripts

Figure grids and other layouts are built in Python from the saved tables:

```python
import pandas as pd
from gMRItensor.plotting import plot_roi_evolution_panels, save_figure

summary = pd.read_parquet("results/roi_analysis/summary__median.parquet")
significance = pd.read_csv(
    "results/roi_analysis/significance__median.csv", dtype={"roi": str}
)
rois = ["ventricles", "white_matter", "cortical_grey_matter", "thalamus"]
fig, axs = plot_roi_evolution_panels(
    summary, significance, rois, "median", n_rows=2, n_cols=2, page_width="double"
)
save_figure(fig, "results/figures/overview")
```

- **ROI rows layout:** `plot_roi_evolution_rows` draws several ROIs as rows. It also takes the per-scan values (`roi_statistics.parquet`) for the subject curves.
- **Group-average maps:** `gMRItensor.pipeline.group_averages` returns the aggregated values without plotting.
- **Decomposition figures:** use `gMRItensor.model_io.load_decomposition` with `plot_mode_grid`, `plot_subject_mode`, `plot_evolving_mode`, `gMRItensor.plotting.time_mode.plot_time_mode` and `plot_spatial_mode`.
- **Building blocks:** `gMRItensor.pipeline` exposes each step for scripts, e.g. `run_statistics_plots(StatisticsPlotOptions(...))`.
