"""`gmri preprocess --negative-cutoff`: noise estimation, rejected values,
invalidated ROIs and the per-setting output folder, on `head_study`."""
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from gMRItensor import preprocessing
from gMRItensor.options import PreprocessingOptions
from gMRItensor.options import StatisticsPlotOptions
from gMRItensor.pipeline import run_preprocessing
from gMRItensor.pipeline import run_statistics_plots
from gMRItensor.preprocessing import PreprocessedPaths
from gMRItensor.preprocessing import read_signal_metadata

NOISE_COLUMNS = [
    "subject",
    "time_point",
    "set",
    "n",
    "mode",
    "sigma",
    "mode_z",
    "sigma_z",
    "used_in_reference",
    "cutoff",
    "n_rejected",
    "rejected_share",
    "n_invalid_rois",
]


def preprocess(study: Any, **overrides: Any) -> PreprocessedPaths:
    values: dict[str, Any] = {
        "output_dir": study.root / "results",
        "input_type": "T1map",
        "time_unit": "ms",
        "n_procs": 1,
        **overrides,
    }
    return run_preprocessing(PreprocessingOptions(study.manifest, **values))


def _scan(frame: pd.DataFrame, subject: str, time_point: int) -> pd.DataFrame:
    return frame[(frame["subject"] == subject) & (frame["time_point"] == time_point)]


def test_without_a_cutoff_preprocessing_is_unchanged(head_study: Any) -> None:
    paths = preprocess(head_study)

    data = head_study.root / "results" / "data"
    assert paths.roi_signal == data / "roi_signal.parquet"
    assert paths.noise is None and paths.invalid_rois is None
    assert sorted(p.name for p in data.iterdir()) == ["roi_signal.parquet"]
    assert (head_study.root / "results" / "preprocess.json").exists()
    roi = pd.read_parquet(paths.roi_signal)
    assert (roi["n_rejected"] == 0).all()
    # The planted artifacts are still there.
    assert _scan(roi, "sub-1", 1).set_index("label").loc[10, "median"] < 0


def test_cutoff_writes_its_own_folder_with_the_qc_tables(
    head_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = preprocess(head_study, negative_cutoff=4.0)

    folder = head_study.root / "results" / "data" / "k4"
    assert paths.roi_signal == folder / "roi_signal.parquet"
    assert (paths.noise, paths.invalid_rois) == (
        folder / "noise.csv",
        folder / "invalid_rois.csv",
    )
    assert (folder / "preprocess.json").exists()
    assert not (head_study.root / "results" / "preprocess.json").exists()
    assert not (head_study.root / "results" / "data" / "roi_signal.parquet").exists()
    err = capsys.readouterr().err
    assert "noise source: outside" in err
    # Only the shifted scan is reported (in every set it shifts), not small
    # differences between scans.
    unusual = [line for line in err.splitlines() if line.startswith("unusual scan")]
    assert unusual and all("sub-2, t=2," in line for line in unusual)
    assert any("sub-2, t=2, outside" in line for line in unusual)

    noise = pd.read_csv(paths.noise)
    assert list(noise.columns) == NOISE_COLUMNS
    assert len(noise) == 6 * 3  # scans x sets
    compartments = noise[noise["set"] != "outside"]
    assert compartments["cutoff"].nunique() == 1
    assert compartments["cutoff"].iloc[0] == pytest.approx(
        -4 * head_study.noise_sd,
        rel=0.15,
    )
    outside = noise[noise["set"] == "outside"].set_index(["subject", "time_point"])
    assert outside["mode_z"].idxmin() == ("sub-2", 2)  # the shifted scan
    assert outside["mode_z"].min() < -3.5
    assert outside["n_rejected"].isna().all()  # measured, never thresholded
    parenchyma = noise[noise["set"] == "parenchyma"].set_index(
        ["subject", "time_point"],
    )
    assert parenchyma.loc[("sub-0", 1), "n_rejected"] == head_study.artifact_voxels
    assert parenchyma.loc[("sub-1", 1), "n_invalid_rois"] == 1

    invalid = pd.read_csv(paths.invalid_rois)
    assert invalid[["subject", "time_point", "label"]].values.tolist() == [
        ["sub-1", 1, 10],
    ]
    row = invalid.iloc[0]
    assert row["compartment"] == "parenchyma"
    assert row["n_rejected"] == head_study.artifact_roi_voxels
    assert row["invalid_share"] == pytest.approx(head_study.artifact_roi_voxels / 27)

    roi = pd.read_parquet(paths.roi_signal)
    cube = _scan(roi, "sub-1", 1).set_index("label").loc[10]
    assert np.isnan(cube["median"]) and cube["n_valid"] == 0
    assert cube["n_rejected"] == head_study.artifact_roi_voxels
    label_2 = _scan(roi, "sub-0", 1).set_index("label").loc[2]
    assert label_2["n_rejected"] == head_study.artifact_voxels
    assert label_2["median"] > 0
    metadata = read_signal_metadata(paths.roi_signal)
    assert metadata["negative_cutoff_k"] == "4"
    assert metadata["noise_source"] == "outside"
    assert metadata["max_invalid_voxel_fraction"] == "0.5"
    assert set(json.loads(metadata["negative_cutoff"])) == {"parenchyma", "csf"}


def test_cutoff_applies_to_the_stored_voxels(head_study: Any) -> None:
    raw = preprocess(head_study, store_voxels=True)
    paths = preprocess(head_study, negative_cutoff=4.0, store_voxels=True)

    assert paths.voxels is not None and paths.voxels.parent.name == "k4"
    assert (paths.voxels.parent / "voxels.coords.parquet").exists()
    assert raw.voxels is not None
    before, after = pd.read_parquet(raw.voxels), pd.read_parquet(paths.voxels)
    assert len(after) == len(before)
    newly_nan = after["values"].isna() & before["values"].notna()
    lost = after[newly_nan].groupby(["subject", "time_point", "labels"]).size()
    assert lost.to_dict() == {
        ("sub-0", 1, 2): head_study.artifact_voxels,
        ("sub-1", 1, 10): 27,  # the invalidated ROI: every voxel
    }


def test_auto_falls_back_to_inside_estimates(
    skull_stripped_head_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = preprocess(skull_stripped_head_study, negative_cutoff=4.0)

    err = capsys.readouterr().err
    assert "noise source: inside" in err
    assert "no csf scan has a noise estimate" in err  # 268 CSF voxels only
    assert paths.noise is not None
    noise = pd.read_csv(paths.noise)
    parenchyma = noise[noise["set"] == "parenchyma"]
    assert parenchyma["cutoff"].iloc[0] == pytest.approx(
        -4 * skull_stripped_head_study.noise_sd,
        rel=0.2,
    )
    assert noise.loc[noise["set"] == "csf", "cutoff"].isna().all()
    assert noise.loc[noise["set"] == "outside", "sigma"].isna().all()


def test_explicit_outside_needs_voxels_outside_the_mask(
    skull_stripped_head_study: Any,
) -> None:
    with pytest.raises(ValueError, match="outside"):
        preprocess(
            skull_stripped_head_study,
            negative_cutoff=4.0,
            noise_source="outside",
        )
    assert not (skull_stripped_head_study.root / "results" / "data" / "k4").exists()


def test_a_fixed_noise_sd_skips_the_noise_pass(
    head_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_noise_pass(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the noise pass ran")

    monkeypatch.setattr(preprocessing, "_noise_worker", no_noise_pass)

    paths = preprocess(head_study, negative_cutoff=4.0, noise_sd=0.01)

    assert paths.roi_signal.parent.name == "k4_sd0.01"
    assert paths.noise is not None
    noise = pd.read_csv(paths.noise)
    assert noise["sigma"].isna().all()
    assert noise["cutoff"].dropna().unique().tolist() == [pytest.approx(-0.04)]
    metadata = read_signal_metadata(paths.roi_signal)
    assert metadata["noise_source"] == "fixed"


def test_parallel_matches_sequential(head_study: Any) -> None:
    results = head_study.root / "results"
    one = preprocess(head_study, negative_cutoff=4.0, output_dir=results / "one")
    two = preprocess(
        head_study,
        negative_cutoff=4.0,
        output_dir=results / "two",
        n_procs=2,
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(one.roi_signal),
        pd.read_parquet(two.roi_signal),
    )
    assert one.noise is not None and two.noise is not None
    pd.testing.assert_frame_equal(pd.read_csv(one.noise), pd.read_csv(two.noise))


def test_a_rerun_says_what_changed(
    head_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preprocess(head_study, negative_cutoff=4.0)
    manifest = pd.read_csv(head_study.manifest)
    manifest.iloc[:-1].to_csv(head_study.manifest, index=False)
    capsys.readouterr()

    preprocess(head_study, negative_cutoff=4.0)

    err = capsys.readouterr().err
    assert "replacing data/k4" in err.replace("\\", "/")
    assert "scans: 6 -> 5" in err


def test_other_settings_go_to_another_folder(head_study: Any) -> None:
    first = preprocess(head_study, negative_cutoff=4.0)
    second = preprocess(head_study, negative_cutoff=4.0, noise_source="inside")

    assert first.roi_signal.parent.name == "k4"
    assert second.roi_signal.parent.name == "k4_inside"
    assert first.roi_signal.exists() and second.roi_signal.exists()


def test_statistics_apply_the_roi_rule_to_regions(head_study: Any) -> None:
    paths = preprocess(head_study, negative_cutoff=4.0)
    info = head_study.root / "subjects.csv"
    pd.DataFrame(
        {"subjects": ["sub-0", "sub-1", "sub-2"], "diagnosis": ["A", "A", "B"]},
    ).to_csv(info, index=False)
    output = head_study.root / "statistics"

    run_statistics_plots(
        StatisticsPlotOptions(
            roi_signal=paths.roi_signal,
            subject_info=info,
            group_variable="diagnosis",
            output_dir=output,
            regions=("cube=10", "pair=10,11", "both=2,10"),
            statistics=("median", "total_amount"),
            formats=("png",),
            dpi=50,
        ),
    )

    stats = pd.read_parquet(output / "roi_analysis" / "roi_statistics.parquet")
    scan = stats[(stats["subject"] == "sub-1") & (stats["timepoint"] == 1)]
    cube = scan.set_index("roi").loc["cube"]
    pair = scan.set_index("roi").loc["pair"]
    both = scan.set_index("roi").loc["both"]
    # The invalid label alone, and with the 8 voxels of label 11 (27 of 35
    # invalid), is invalid; with label 2 it is not, and its median comes
    # from label 2 only.
    assert np.isnan(cube["median"]) and cube["n_valid"] == 0
    assert np.isnan(cube["total_amount"])
    assert np.isnan(pair["median"]) and pair["n_valid"] == 0
    label_2 = pd.read_parquet(paths.roi_signal)
    label_2 = label_2[
        (label_2["subject"] == "sub-1")
        & (label_2["time_point"] == 1)
        & (label_2["label"] == 2)
    ].iloc[0]
    assert both["median"] == pytest.approx(label_2["median"])
    # The total counts every voxel of the region at its mean.
    assert both["total_amount"] == pytest.approx(
        both["mean"] / 3.2 * both["n_voxels"] * 1e-6,
    )
