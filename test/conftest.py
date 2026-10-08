"""Shared fixtures for the stage (pipeline/CLI) tests."""
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

TIME_POINTS = (0, 6, 24)
LABELS = (4, 10, 49, 2, 41)


class SyntheticStudy(NamedTuple):
    root: Path
    manifest: Path
    subject_info: Path


@pytest.fixture
def synthetic_study(tmp_path: Path) -> SyntheticStudy:
    """6 subjects (3 PD, 3 Control) x 3 time points of 6^3 T1 maps [ms].

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
        subject_rows.append({"subjects": subject, "diagnosis": group})
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
