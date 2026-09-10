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
- `gMRItensor.plotting` — visualization of decomposition modes: subject-mode boxplots and correlations, spatial-mode brain overlays, mode grids, and evolving (longitudinal) mode trajectories.

## Development

```bash
uv sync --all-extras
uv run pre-commit install
uv run pytest
```

- Package management: `uv add <package>` / `uv add --dev <package>`.
- Code must pass `pre-commit` (Ruff formatting/linting, Mypy).
- Tests live in `test/` and run with `uv run pytest`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## License

[MIT](LICENSE)
