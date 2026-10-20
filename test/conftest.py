"""Shared fixtures for the stage (pipeline/CLI) tests."""
import os
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

TIME_POINTS = (0, 6, 24)
LABELS = (4, 10, 49, 2, 41)

if os.environ.get("PYTEST_XDIST_WORKER"):
    # One BLAS/OpenMP thread per xdist worker, set before numpy or torch start
    # their thread pools: the workers already use every core between them.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run the tests marked slow",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


class SyntheticStudy(NamedTuple):
    root: Path
    manifest: Path
    subject_info: Path


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit on the CPU unless a test opts in, and undo any test's own
    `GMRITENSOR_USE_GPU` change, so the flag never leaks between tests."""
    monkeypatch.setenv("GMRITENSOR_USE_GPU", "FALSE")


@pytest.fixture(autouse=True)
def quick_reproducibility_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decomposition runs save each rank's reproducibility figure; in tests
    a small PNG is enough to check it is written."""
    from gMRItensor import pipeline

    monkeypatch.setattr(
        pipeline,
        "REPRODUCIBILITY_FIGURE",
        {"formats": ("png",), "dpi": 50},
    )


@pytest.fixture
def synthetic_study(tmp_path: Path) -> SyntheticStudy:
    """6 subjects (3 PD, 3 Control) x 3 time points of 6^3 T1 maps [ms].

    The subject info also has `sex` (F/M), `apoe` (numeric codes 2/3/4) and
    `age`, blank for sub-00.

    Baseline T1 is ~1500 ms; enhancement peaks at 24 h with ΔR1 ~1e-4/ms,
    stronger in the ventricles (label 4) and more so for PD, so
    concentrations are finite and nonzero. Manifest paths are relative to
    the manifest.
    """
    rng = np.random.default_rng(0)
    shape = (6, 6, 6)
    affine = np.eye(4)
    images = tmp_path / "images"
    images.mkdir()
    segmentation = rng.choice(LABELS, size=shape).astype(float)
    nib.save(nib.Nifti1Image(segmentation, affine), images / "seg.nii")
    nib.save(nib.Nifti1Image(np.ones(shape), affine), images / "mask.nii")

    rows, subject_rows = [], []
    for s in range(6):
        subject = f"sub-{s:02d}"
        group = "PD" if s < 3 else "Control"
        subject_rows.append(
            {
                "subjects": subject,
                "diagnosis": group,
                "sex": "FM"[s % 2],
                "apoe": (2, 3, 4)[s % 3],
                "age": "" if s == 0 else 55 + 3 * s,  # one blank cell
            },
        )
        baseline = 1500.0 + rng.normal(0, 10, shape)
        nib.save(nib.Nifti1Image(baseline, affine), images / f"{subject}_base.nii")
        for t in TIME_POINTS:
            enhancement = np.exp(-((t - 24) ** 2) / 200)
            boost = np.where(segmentation == 4, 3.0 + (s < 3), 1.0)
            delta_r1 = 1e-4 * enhancement * boost + rng.normal(0, 5e-6, shape)
            post = 1.0 / (1.0 / baseline + delta_r1)
            name = f"{subject}_t{t}.nii"
            nib.save(nib.Nifti1Image(post, affine), images / name)
            rows.append(
                {
                    "subject": subject,
                    "time_point": t,
                    "baseline_path": f"images/{subject}_base.nii",
                    "post_injection_path": f"images/{name}",
                    "mask_path": "images/mask.nii",
                    "segmentation_path": "images/seg.nii",
                },
            )

    manifest = tmp_path / "scans.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    subject_info = tmp_path / "subjects.csv"
    pd.DataFrame(subject_rows).to_csv(subject_info, index=False)
    return SyntheticStudy(tmp_path, manifest, subject_info)
