# Changelog

## Unreleased

### Added
- **ROI overview:** `gmri plot overview` draws every label for every subject and session, one row per session (all parenchyma, all CSF, and one column per region), and flags values with |modified z-score| > 3.5 among all subjects for that label and session (`--z-cut`; Iglewicz & Hoaglin 1993). Each flagged value is ringed and named in the figures and listed in `roi_analysis/overview_flags__<statistic>.csv`. See `docs/cli.md`.
- **Excluding subjects:** `decompose` and `replicability` (`run` and `plan`) take `--exclude-subjects ID ...`; the subjects are left out as if absent from the input, and an unknown ID is an error. Every left-out subject, also those dropped for missing or unusable sessions (naming the time points), is listed with the reason in `excluded_subjects.csv`, per model directory for `decompose`.
- **Coefficient tables:** `plot decomposition` writes the subject and time coefficients exactly as drawn, `<model>__subject_coefficients.csv` (one row per subject) and `<model>__time_coefficients.csv` (per time point; per subject and time point for PARAFAC2 and CMF), so a subject can be looked up.
- **Negative cutoff:** `preprocess --negative-cutoff K` (off by default) estimates the study's ΔR1 noise level from the data (outside the brain mask, or per compartment inside it; `gMRItensor.noise`), makes values below −K noise SDs invalid, and makes an ROI NaN when more than `--max-invalid-voxel-fraction` (0.5) of its voxels are invalid, in the ROI and the voxel data alike. Each setting writes to its own folder, `data/k<K>[...]/`, with `noise.csv` (per-scan noise estimates, z-scores, cutoff, rejected voxels) and `invalid_rois.csv`. `roi_signal.parquet` gains an `n_rejected` column (0 without a cutoff).
- **`roi_signal.csv`:** `preprocess` also writes the ROI table as CSV next to `roi_signal.parquet`, for viewing (the voxel files get none).
- **Dropped columns listed:** `decompose` and `replicability` write `excluded_columns.csv` next to `excluded_subjects.csv`: per label, the columns left out of the fit (PARAFAC2/CMF drop any column non-finite in a kept scan) and the scans that caused it.
- **Restart timings:** every restart records its fit time, iterations, backend, threads, peak memory and host in the store, and `decompose run` / `collect` list them per rank in `<store>/<method>_rank_<r>/restarts.csv`, also when no restart converged. Restarts stored earlier load as before, with empty timing columns.
- **Performance settings as environment variables**, read at run time and not part of the settings fingerprint: `GMRITENSOR_FIT_BACKEND=auto` (numpy below `GMRITENSOR_AUTO_BACKEND_ENTRIES` data entries, default 5e7, torch above; the default stays `numpy`) and `GMRITENSOR_MEMORY_EFFICIENT_MTTKRP=0|1`. The `use_memory_efficient_khatri_rao` fit option no longer marks stored restarts stale.
- **`gmri predict`** (PARAFAC2 and CMF models): how well each component's time profile (curve × subject coefficient) predicts numerical clinical outcomes, with no, each single and all primary variables as covariates, each against the same model without the profile. A stratified train/test split, repeated k-fold cross-validation on the training part (R², RMSE, MAE), a single test-set score, AIC/BIC, and permutation p-values for the model against chance and for the profile's gain over its baseline (Freedman–Lane, FDR-corrected). Written to `<model>__curve_prediction.csv`; see `docs/curve-prediction.md` and, for every column and how it is computed, `docs/curve-prediction-columns.md`.
- **Guide for large inputs on CPU clusters:** `docs/cpu-scaling.md` covers backend, threads, restarts per node and job size, plus a scaling-test protocol.

### Changed
- **CP and TensorLy's PARAFAC2 run in float64 on every backend and on the GPU**, as matcouply's fits already did, so fits are comparable wherever they ran. Before, they ran in the input's float32 on torch and on the GPU, where TF32 also applied; such stored restarts are reused as they are. `compute_CP_decomposition` returns float64.
- **CPU pools share one float64 copy of the input:** a job casts it once, and its `--restart-procs` / `--n-procs` workers no longer each make their own (10.6 GB per worker at 30 × 4 × 11 M voxels). `/dev/shm` must hold the float64 input.
- CP on the CPU no longer goes through `torch.compile` (GPU fits still do).
- A job warns when its numpy-backend fits get more than one thread each.
- The loader's notes on dropped sessions and subjects go to stderr, so the stdout of `plan` stays the job count.
- **`total_amount`** is mean concentration × the full ROI volume (it was × the valid volume), so invalid voxels count at the ROI's mean concentration instead of lowering the total.
- For data preprocessed with `--negative-cutoff`, `plot statistics` makes a region NaN when its share of invalid voxels exceeds the `--max-invalid-voxel-fraction` used then.
- **Evolving-mode curves** (PARAFAC2, CMF; `--time` and `--mode-grid`): each subject's curve is its unit-norm shape times its scaled subject coefficient, so a subject that barely expresses a component is a near-flat line instead of a full-height curve. Every subject counts in the ribbons and group tests. A negative PARAFAC2 coefficient flips the curve.
- **CMF subject coefficient:** the figures use the L2 norm of each subject's time course, not the RMS the fit saves (they differ when subjects have different numbers of time points).
- **Mode-grid labels:** the time and subject columns are labelled "Time coef. (a.u.)" and "Subject coef. (a.u.)", each row is titled "Component k" in bold, and only the rightmost (CSF) colorbar is labelled. The spatial colorbars, here and in the spatial-mode figure, say "Spatial coef. (a.u.)" instead of "Loading (a.u.)".

### Removed
- `plot decomposition --min-subject-coefficient`: with curves scaled by the subject coefficient, no cutoff is needed.

## 0.2.0

### Added
- **`gmri` command line:** `preprocess`, `decompose`, `replicability` and `plot statistics | decomposition | group-maps | reproducibility`. Every setting is an argument; each command reads only files earlier commands wrote and records its settings as JSON. See `docs/cli.md`.
- **Distributed restarts:** `plan` / `run --job N` / `collect` for `decompose` and `replicability` (e.g. SLURM arrays), with a resumable restart store.
- **CMF:** non-negative coupled matrix factorization (matcouply's `cmf_aoadmm`), next to CP and PARAFAC2.
- **Restart reproducibility:** pairwise factor match scores between the restarts of every fit (`reproducibility.csv` and figure).
- **Outlier diagnostics:** `plot decomposition --outliers`, leverage against residual per subject with Bonferroni-corrected cut-offs (Fida et al. 2022).
- **SD bands:** `plot statistics --sd-bands` draws the cohort mean ± 1, 2 and 3 SD and lists subjects beyond 3 SD.
- **Group-average maps:** `plot group-maps`.

### Changed (behaviour)
- **Default PARAFAC2 solver:** matcouply (AO-ADMM), non-negative in all three modes; `--solver tensorly` keeps the previous ALS fit.
- **CPU fits run on TensorLy's numpy backend in float64.** Per restart (30 subjects × 4 time points, rank 3, one thread): matcouply PARAFAC2 116 s → 16 s, CMF 12 s → 2 s, TensorLy PARAFAC2 11 s → 5 s, CP on 80 ROIs 6.4 s → 0.4 s, with the same or lower errors. float32 fits used to stop early on rounding, so CP now reaches lower errors; a voxel-wise CP fit (20,000 voxels) takes longer (4.5 s → 12.5 s) because it now actually converges. With matcouply as the new default, a default PARAFAC2 restart takes about 1.5× as long as the old TensorLy default did. `GMRITENSOR_FIT_BACKEND=torch` restores the torch path; GPU fits always run on torch.
- **matcouply loss tolerance:** `1e-7` instead of matcouply's `1e-8`, which fits rarely met within their iteration budget although the error had stopped improving (`--fit-option aoadmm_loss_tolerance=1e-8` restores it).
- **Output layout:** each model is written to its own directory, `<method>_rank_<r>/model.h5` with its `decompose.json`; `fits.csv` lists every model of the output directory.
- **Stricter arguments:** abbreviated flags, and arguments that would have no effect (`replicability --restart-procs`, `--splits` with `halfhalf`, `--subject-info` without `--stratify-by`, `--tasks-per-job` without `--job`), are errors.
- **Moved:** `gMRItensor.decomposition` is a package (`models`, `cp`, `parafac2`, `cmf`, `restarts`); its public names are still importable from `gMRItensor.decomposition`. `scatter_to_volume`, `expand_roi_mode_to_voxels`, `merge_segmentations` and `region_masks_from_segmentations` moved from `gMRItensor.plotting.utils` to `gMRItensor.volumes`.
- **Removed:** the `normalize` fit option (fits keep their scale in the factors; figures normalise), and the legacy preprocessing API (`compute_tracer_from_image`, `compute_tracer_parallel`, `iter_tracer_results`, `write_tracer_parquet`, `prepare_tensor`), superseded by `gmri preprocess` and `load_tensor_from_parquet`.

### Fixed
- TensorLy PARAFAC2 accepted fits that had run out of iterations (its line search hid the limit from the iteration count).
- Re-planning with other settings reused the restarts fitted under the old ones. The jobs now refit and overwrite such stale restarts (and report what they fitted, reused and refitted); `collect` waits for them, naming the jobs to run; `plan --overwrite` starts the planned ranks from scratch.
- `plot decomposition --outliers` could reload the data with another model's settings when models shared an output directory, and failed from another working directory.
- A blank group cell emptied the significance tables (numeric groups) or crashed `plot statistics` (text groups); an all-tied time point turned every Benjamini–Hochberg p-value of its ROI into NaN.
- Subject-mode scatter colours shifted when a group had no values for a variable.
- ROI spatial maps drew labels the model lacks as the lowest loading; 0 in an R1 map counted as a valid voxel.
- One rank that did not converge discarded every other rank of the run.
