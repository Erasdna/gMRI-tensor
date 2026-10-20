# Changelog

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
- **Removed:** the `normalize` fit option (fits keep their scale in the factors; figures normalise), and the legacy preprocessing API (`compute_tracer_from_image`, `compute_tracer_parallel`, `iter_tracer_results`, `write_tracer_parquet`, `prepare_tensor`), superseded by `gmri preprocess` and `load_tensor_from_parquet`.

### Fixed
- TensorLy PARAFAC2 accepted fits that had run out of iterations (its line search hid the limit from the iteration count).
- Re-planning with other settings reused the restarts fitted under the old ones; stores now refuse them.
- `plot decomposition --outliers` could reload the data with another model's settings when models shared an output directory, and failed from another working directory.
- A blank group cell emptied the significance tables (numeric groups) or crashed `plot statistics` (text groups); an all-tied time point turned every Benjamini–Hochberg p-value of its ROI into NaN.
- Subject-mode scatter colours shifted when a group had no values for a variable.
- ROI spatial maps drew labels the model lacks as the lowest loading; 0 in an R1 map counted as a valid voxel.
- One rank that did not converge discarded every other rank of the run.
