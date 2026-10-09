from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from gMRItensor import pipeline
from gMRItensor import preprocessing
from gMRItensor.model_io import load_decomposition
from gMRItensor.options import DecompositionOptions
from gMRItensor.options import DecompositionPlotOptions
from gMRItensor.options import DistributedOptions
from gMRItensor.options import FitOptions
from gMRItensor.options import PreprocessingOptions
from gMRItensor.options import ReplicabilityOptions
from gMRItensor.options import StatisticsPlotOptions
from gMRItensor.options import TensorOptions
from gMRItensor.pipeline import collect_decomposition
from gMRItensor.pipeline import collect_replicability
from gMRItensor.pipeline import format_significance_summary
from gMRItensor.pipeline import plan_decomposition
from gMRItensor.pipeline import plan_replicability_jobs
from gMRItensor.pipeline import read_manifest
from gMRItensor.pipeline import run_decomposition
from gMRItensor.pipeline import run_decomposition_job
from gMRItensor.pipeline import run_decomposition_plots
from gMRItensor.pipeline import run_preprocessing
from gMRItensor.pipeline import run_replicability
from gMRItensor.pipeline import run_replicability_job
from gMRItensor.pipeline import run_statistics_plots

QUICK_FIT = FitOptions(restarts=2, max_iter=50)


def preprocess(study: Any, **overrides: Any) -> Path:
    """Run `gmri preprocess` on the study; return the results directory."""
    values: dict[str, Any] = {"n_procs": 1, "time_unit": "ms", **overrides}
    options = PreprocessingOptions(
        study.manifest,
        study.root / "results",
        values.pop("input_type", "T1map"),
        **values,
    )
    run_preprocessing(options)
    return options.output_dir


def statistics_options(study: Any, **overrides: Any) -> StatisticsPlotOptions:
    values: dict[str, Any] = {
        "roi_signal": study.root / "results" / "data" / "roi_signal.parquet",
        "subject_info": study.subject_info,
        "group_variable": "diagnosis",
        "output_dir": study.root / "results",
        "formats": ("png",),
        "dpi": 50,
        **overrides,
    }
    return StatisticsPlotOptions(**values)


def decomposition_options(study: Any, **overrides: Any) -> DecompositionOptions:
    values: dict[str, Any] = {
        "input": study.root / "results" / "data" / "roi_signal.parquet",
        "output_dir": study.root / "results" / "decompositions" / "test",
        "method": "parafac2",
        "ranks": (1, 2),
        "fit": QUICK_FIT,
        **overrides,
    }
    return DecompositionOptions(**values)


def replicability_options(study: Any, **overrides: Any) -> ReplicabilityOptions:
    values: dict[str, Any] = {
        "input": study.root / "results" / "data" / "roi_signal.parquet",
        "output_dir": study.root / "results" / "replicability" / "test",
        "method": "parafac2",
        "ranks": (1,),
        "engine": "halfhalf",
        "repeats": 2,
        "subject_info": study.subject_info,
        "stratify_by": "diagnosis",
        "fit": QUICK_FIT,
        **overrides,
    }
    return ReplicabilityOptions(**values)


# Preprocessing ------------------------------------------------------------


def test_run_preprocessing_writes_roi_signal_and_provenance(
    synthetic_study: Any,
) -> None:
    results = preprocess(synthetic_study)

    roi = pd.read_parquet(results / "data" / "roi_signal.parquet")
    assert set(roi["label"]) == {2, 4, 10, 41, 49}
    assert len(roi) == 5 * 6 * 3  # labels x subjects x time points
    assert (roi["median"] > 0).all()
    assert (results / "preprocess.json").exists()
    assert not (results / "data" / "voxels.parquet").exists()


def test_run_preprocessing_store_voxels(synthetic_study: Any) -> None:
    results = preprocess(synthetic_study, store_voxels=True)

    voxels = pd.read_parquet(results / "data" / "voxels.parquet")
    assert len(voxels) == 6**3 * 6 * 3  # every voxel is labeled in the study


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda df: pd.concat([df, df.iloc[[0]]]), "duplicate"),
        (lambda df: df.assign(time_point=df["time_point"] + 0.5), "time_point"),
        (lambda df: df.drop(columns="mask_path"), "mask_path"),
        (lambda df: df.assign(mask_path=[""] + list(df["mask_path"][1:])), "mask_path"),
        (lambda df: df.assign(subject=[""] + list(df["subject"][1:])), "subject"),
    ],
)
def test_read_manifest_rejects_bad_rows(
    synthetic_study: Any,
    edit: Any,
    match: str,
) -> None:
    manifest = pd.read_csv(synthetic_study.manifest)
    edit(manifest).to_csv(synthetic_study.manifest, index=False)
    options = PreprocessingOptions(
        synthetic_study.manifest,
        synthetic_study.root,
        "T1map",
        "ms",
    )

    with pytest.raises(ValueError, match=match):
        read_manifest(options)


def test_run_preprocessing_reports_missing_images_before_loading(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = pd.read_csv(synthetic_study.manifest)
    manifest.loc[3, "post_injection_path"] = "images/missing.nii"
    manifest.to_csv(synthetic_study.manifest, index=False)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an image was loaded")

    monkeypatch.setattr(preprocessing, "_load_labeled_tracer_voxels", fail)

    with pytest.raises(ValueError, match="missing.nii"):
        preprocess(synthetic_study)


# Statistics ---------------------------------------------------------------


def test_run_statistics_plots_defaults_to_present_presets(synthetic_study: Any) -> None:
    preprocess(synthetic_study)

    result = run_statistics_plots(statistics_options(synthetic_study))

    tables = synthetic_study.root / "results" / "roi_analysis"
    rois = set(pd.read_parquet(tables / "roi_statistics.parquet")["roi"])
    # Presets with a label in the data (4, 10, 49, 2, 41); the rest are skipped.
    assert {"ventricles", "thalamus", "white_matter", "all_csf"} <= rois
    assert "cerebellum" not in rois
    assert set(result.summary["roi"]) == rois
    assert (tables / "summary__median.parquet").exists()
    assert (tables / "significance__median.csv").exists()
    assert (tables / "summary.csv").exists()
    single = synthetic_study.root / "results" / "figures" / "roi" / "single" / "median"
    assert sorted(result.figures) == sorted(single / f"{roi}__rows.png" for roi in rois)


def test_run_statistics_plots_labels_regions_and_concentration(
    synthetic_study: Any,
) -> None:
    preprocess(synthetic_study)
    options = statistics_options(
        synthetic_study,
        regions=("lateral=4,43",),
        rois=("4", "10"),
        statistics=("median", "total_amount"),
        layout="panels",
    )

    result = run_statistics_plots(options)

    stats = pd.read_parquet(
        synthetic_study.root / "results" / "roi_analysis" / "roi_statistics.parquet",
    )
    assert set(stats["roi"]) == {"4", "10", "lateral"}
    lateral = stats[stats["roi"] == "lateral"]
    four = stats[stats["roi"] == "4"]
    # Label 43 is not in the data, so the region equals label 4.
    np.testing.assert_allclose(lateral["median"], four["median"])
    np.testing.assert_allclose(
        stats["median_concentration"],
        stats["median"] / pipeline.DEFAULT_RELAXIVITY,
    )
    assert len(result.figures) == 3 * 2
    assert set(result.summary["statistic"]) == {"median", "total_amount"}


def test_run_statistics_plots_rejects_concentration_for_t1w(
    synthetic_study: Any,
) -> None:
    preprocess(synthetic_study, input_type="T1w")

    with pytest.raises(ValueError, match="T1w"):
        run_statistics_plots(
            statistics_options(synthetic_study, statistics=("total_amount",)),
        )
    with pytest.raises(ValueError, match="relaxivity"):
        run_statistics_plots(statistics_options(synthetic_study, relaxivity=3.0))
    assert not (synthetic_study.root / "results" / "roi_analysis").exists()


def test_run_statistics_plots_unknown_label_writes_nothing(
    synthetic_study: Any,
) -> None:
    preprocess(synthetic_study)

    with pytest.raises(ValueError, match="99"):
        run_statistics_plots(statistics_options(synthetic_study, rois=("99",)))
    assert not (synthetic_study.root / "results" / "roi_analysis").exists()


def test_format_significance_summary() -> None:
    summary = pd.DataFrame(
        {
            "statistic": ["median", "median"],
            "roi": ["ventricles", "thalamus"],
            "n_tested": [3, 3],
            "significant_timepoints": ["24", ""],
            "min_p_adj": [0.01, 0.4],
            "higher_group": ["PD", "PD"],
        },
    )

    text = format_significance_summary(summary, 0.05)

    assert "ventricles" in text and "thalamus" not in text
    assert "No significant" in format_significance_summary(summary.iloc[1:], 0.05)


def test_run_statistics_plots_before_preprocessing(synthetic_study: Any) -> None:
    with pytest.raises(FileNotFoundError, match="gmri preprocess"):
        run_statistics_plots(statistics_options(synthetic_study))


# Decomposition ------------------------------------------------------------


@pytest.mark.parametrize("statistic", ["median", "mean"])
def test_run_decomposition_from_roi_signal(
    synthetic_study: Any,
    statistic: str,
) -> None:
    preprocess(synthetic_study)
    options = decomposition_options(synthetic_study, statistic=statistic)

    written = run_decomposition(options)

    out = options.output_dir
    assert written == [out / "rank_1.h5", out / "rank_2.h5"]
    assert list(pd.read_csv(out / "fits.csv")["rank"]) == [1, 2]
    saved = load_decomposition(out / "rank_2.h5")
    assert saved.method == "parafac2"
    assert saved.evolving_states is not None and len(saved.evolving_states) == 6
    assert saved.label_mode.shape == (5, 2)
    np.testing.assert_array_equal(saved.labels, [2, 4, 10, 41, 49])
    assert saved.voxel_coords is None
    assert (out / "decompose.json").exists()


def test_run_decomposition_inputs_follow_the_statistic(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []

    def fake_parafac2(slices: list, rank: int, **kwargs: Any) -> tuple:
        seen.append(np.stack([s.numpy() for s in slices]))
        model = pipeline.PARAFAC2Model(
            weights=torch.ones(rank),
            subject_mode=torch.rand(len(slices), rank),
            evolving_states=[torch.rand(s.shape[0], rank) for s in slices],
            label_mode=torch.rand(slices[0].shape[1], rank),
        )
        return model, torch.tensor(0.5)

    monkeypatch.setattr(pipeline, "run_PARAFAC2_decomposition_repeated", fake_parafac2)
    preprocess(synthetic_study)
    unscaled = TensorOptions(scale=False)
    for statistic in ("median", "mean"):
        run_decomposition(
            decomposition_options(
                synthetic_study,
                ranks=(1,),
                statistic=statistic,
                tensor=unscaled,
            ),
        )

    roi = pd.read_parquet(
        synthetic_study.root / "results" / "data" / "roi_signal.parquet",
    )
    first = roi.query("subject == 'sub-00' and time_point == 0").sort_values("label")
    np.testing.assert_allclose(seen[0][0, 0], first["median"], rtol=1e-6)
    np.testing.assert_allclose(seen[1][0, 0], first["mean"], rtol=1e-6)


def test_run_decomposition_from_voxels_keeps_coordinates(synthetic_study: Any) -> None:
    preprocess(synthetic_study, store_voxels=True)
    options = decomposition_options(
        synthetic_study,
        input=synthetic_study.root / "results" / "data" / "voxels.parquet",
        ranks=(2,),
    )

    run_decomposition(options)

    saved = load_decomposition(options.output_dir / "rank_2.h5")
    assert saved.label_mode.shape == (6**3, 2)
    assert saved.voxel_coords is not None and saved.voxel_coords.shape == (6**3, 3)
    np.testing.assert_array_equal(saved.template_shape, [6, 6, 6])
    np.testing.assert_array_equal(saved.template_affine, np.eye(4))


def test_cp_wiring_and_center(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the CP fit: the real one triggers torch.compile, which is slow.
    calls = []

    def fake_cp(tensor: torch.Tensor, rank: int, **kwargs: Any) -> tuple:
        calls.append((tensor.numpy().copy(), kwargs))
        factors = [torch.rand(n, rank) for n in tensor.shape]
        return torch.ones(rank), factors, torch.tensor(0.25)

    monkeypatch.setattr(pipeline, "run_CP_decomposition_repeated", fake_cp)
    preprocess(synthetic_study)
    options = decomposition_options(
        synthetic_study,
        method="cp",
        ranks=(3,),
        tensor=TensorOptions(center=True, scale=False),
        fit=FitOptions(restarts=4, max_iter=10, non_negative_modes=None),
    )

    run_decomposition(options)

    ((tensor, kwargs),) = calls
    assert tensor.shape == (6, 3, 5)
    np.testing.assert_allclose(tensor.mean(axis=(0, 1)), 0.0, atol=1e-5)
    assert kwargs["init_repeats"] == 4 and kwargs["non_negative"] is False
    saved = load_decomposition(options.output_dir / "rank_3.h5")
    assert saved.time_mode is not None and saved.time_mode.shape == (3, 3)
    assert saved.centered is True and saved.scale_std is None


@pytest.mark.parametrize(
    "fit, expected",
    [
        (FitOptions(restarts=1), {"nn_modes": "auto"}),
        (
            FitOptions(restarts=1, non_negative_modes=(0, 2), solver="matcouply"),
            {"nn_modes": (0, 2), "solver": "matcouply"},
        ),
    ],
)
def test_parafac2_options_reach_the_runner(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    fit: FitOptions,
    expected: dict,
) -> None:
    calls = []

    def fake_parafac2(slices: list, rank: int, **kwargs: Any) -> tuple:
        calls.append(kwargs)
        model = pipeline.PARAFAC2Model(
            weights=torch.ones(rank),
            subject_mode=torch.rand(len(slices), rank),
            evolving_states=[torch.rand(s.shape[0], rank) for s in slices],
            label_mode=torch.rand(slices[0].shape[1], rank),
        )
        return model, torch.tensor(0.5)

    monkeypatch.setattr(pipeline, "run_PARAFAC2_decomposition_repeated", fake_parafac2)
    preprocess(synthetic_study)

    run_decomposition(decomposition_options(synthetic_study, ranks=(2,), fit=fit))

    (kwargs,) = calls
    assert {key: kwargs.get(key) for key in expected} == expected
    assert "non_negative" not in kwargs


def test_distributed_decomposition_matches_centralised(synthetic_study: Any) -> None:
    preprocess(synthetic_study)
    central = decomposition_options(synthetic_study)
    run_decomposition(central)
    distributed = decomposition_options(
        synthetic_study,
        output_dir=synthetic_study.root / "results" / "decompositions" / "distributed",
        distributed=DistributedOptions(tasks_per_job=3),
    )

    n_jobs = plan_decomposition(distributed)
    assert n_jobs == 2  # 2 ranks x 2 restarts in blocks of 3
    for job in range(n_jobs):
        run_decomposition_job(distributed.output_dir, job)
    written = collect_decomposition(distributed.output_dir)

    assert [path.name for path in written] == ["rank_1.h5", "rank_2.h5"]
    pd.testing.assert_frame_equal(
        pd.read_csv(distributed.output_dir / "fits.csv"),
        pd.read_csv(central.output_dir / "fits.csv"),
    )
    for name in ("rank_1.h5", "rank_2.h5"):
        got = load_decomposition(distributed.output_dir / name)
        expected = load_decomposition(central.output_dir / name)
        np.testing.assert_allclose(got.label_mode, expected.label_mode, rtol=1e-5)
    assert (distributed.output_dir / "plan.json").exists()


def test_distributed_jobs_need_a_plan_and_all_restarts(synthetic_study: Any) -> None:
    preprocess(synthetic_study)
    options = decomposition_options(
        synthetic_study,
        distributed=DistributedOptions(tasks_per_job=3),
    )

    with pytest.raises(FileNotFoundError, match="plan"):
        run_decomposition_job(options.output_dir, 0)
    plan_decomposition(options)
    with pytest.raises(ValueError, match="job 2 .* 2 jobs"):
        run_decomposition_job(options.output_dir, 2)
    run_decomposition_job(options.output_dir, 0)
    with pytest.raises(ValueError, match="1 of 2 restarts missing"):
        collect_decomposition(options.output_dir)
    with pytest.raises(ValueError, match="DecompositionOptions plan"):
        run_replicability_job(options.output_dir, 0)


def test_distributed_plan_is_independent_of_the_working_directory(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # Regression test: plan.json stored relative paths, so array jobs started
    # from another directory could not find the input or wrote the restarts
    # under their own working directory.
    preprocess(synthetic_study)
    monkeypatch.chdir(synthetic_study.root)
    options = decomposition_options(
        synthetic_study,
        input=Path("results/data/roi_signal.parquet"),
        output_dir=Path("results/relative"),
        ranks=(1,),
    )
    n_jobs = plan_decomposition(options)

    output_dir = synthetic_study.root / "results" / "relative"
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    for job in range(n_jobs):
        run_decomposition_job(output_dir, job)
    written = collect_decomposition(output_dir)

    assert written == [output_dir / "rank_1.h5"]


# Replicability ------------------------------------------------------------


def test_run_replicability_halfhalf(synthetic_study: Any) -> None:
    preprocess(synthetic_study)

    path = run_replicability(replicability_options(synthetic_study))

    scores = pd.read_csv(path)
    assert list(scores.columns) == ["rank", "split", "fms"]
    assert list(scores["split"]) == [0, 1]
    assert scores["fms"].between(0, 1).all()
    assert (path.parent / "replicability.json").exists()


def test_run_replicability_cv_columns(synthetic_study: Any) -> None:
    preprocess(synthetic_study)
    options = replicability_options(synthetic_study, engine="cv", splits=3, repeats=1)

    scores = pd.read_csv(run_replicability(options))

    assert list(scores.columns) == ["rank", "fold_i", "fold_j", "n_common", "fms"]
    assert len(scores) == 3


def test_run_replicability_missing_subject_info_row(synthetic_study: Any) -> None:
    preprocess(synthetic_study)
    subject_info = pd.read_csv(synthetic_study.subject_info)
    subject_info.iloc[1:].to_csv(synthetic_study.subject_info, index=False)

    with pytest.raises(ValueError, match="sub-00"):
        run_replicability(replicability_options(synthetic_study))


def test_distributed_replicability_matches_centralised(synthetic_study: Any) -> None:
    preprocess(synthetic_study)
    expected = pd.read_csv(run_replicability(replicability_options(synthetic_study)))
    distributed = replicability_options(
        synthetic_study,
        output_dir=synthetic_study.root / "results" / "replicability" / "distributed",
        distributed=DistributedOptions(tasks_per_job=2),
    )

    n_jobs = plan_replicability_jobs(distributed)
    assert n_jobs == 4  # 1 rank x 2 repeats x 2 halves x 2 restarts, blocks of 2
    for job in range(n_jobs):
        run_replicability_job(distributed.output_dir, job)
    got = pd.read_csv(collect_replicability(distributed.output_dir))

    pd.testing.assert_frame_equal(got, expected, rtol=1e-5)


def test_progress_bar_option_is_passed_through(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocess(synthetic_study)
    seen = []
    original: Any = pipeline.run_tasks

    def spy(*args: Any, progress_bar: bool = True, **kwargs: Any) -> None:
        seen.append(progress_bar)
        original(*args, progress_bar=progress_bar, **kwargs)

    monkeypatch.setattr(pipeline, "run_tasks", spy)
    fit = FitOptions(restarts=1, max_iter=20, extra={"progress_bar": True})
    options = decomposition_options(synthetic_study, ranks=(1,), fit=fit)
    plan_decomposition(options)

    run_decomposition_job(options.output_dir, 0)

    assert seen == [True]


# Decomposition figures -----------------------------------------------------


def plot_options(study: Any, model: Path, **overrides: Any) -> DecompositionPlotOptions:
    values: dict[str, Any] = {
        "model": model,
        "subject_info": study.subject_info,
        "group_variable": "diagnosis",
        "output_dir": study.root / "results",
        "formats": ("png",),
        "dpi": 50,
        **overrides,
    }
    return DecompositionPlotOptions(**values)


def _figures(study: Any) -> Path:
    return study.root / "results" / "figures" / "decomposition"


def test_plot_parafac2_roi_model_with_segmentation(synthetic_study: Any) -> None:
    preprocess(synthetic_study)
    options = decomposition_options(synthetic_study, ranks=(2,))
    (model,) = run_decomposition(options)

    written = run_decomposition_plots(
        plot_options(
            synthetic_study,
            model,
            segmentation=synthetic_study.root / "images" / "seg.nii",
        ),
    )

    names = sorted(path.name for path in written)
    assert names == [
        "rank_2__evolving_mode.png",
        "rank_2__mode_grid.png",
        "rank_2__spatial_csf.png",
        "rank_2__spatial_parenchyma.png",
        "rank_2__subject_mode.png",
    ]
    assert (
        _figures(synthetic_study) / "rank_2__evolving_mode_significance.csv"
    ).exists()


def test_plot_roi_model_without_segmentation(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))

    written = run_decomposition_plots(plot_options(synthetic_study, model))

    assert sorted(p.name for p in written) == [
        "rank_2__evolving_mode.png",
        "rank_2__subject_mode.png",
    ]
    assert "segmentation" in capsys.readouterr().out
    with pytest.raises(ValueError, match="segmentation"):
        run_decomposition_plots(
            plot_options(synthetic_study, model, parts=("spatial",)),
        )


def test_plot_cp_voxel_model_without_segmentation(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_cp(tensor: torch.Tensor, rank: int, **kwargs: Any) -> tuple:
        factors = [torch.rand(n, rank) for n in tensor.shape]
        return torch.ones(rank), factors, torch.tensor(0.25)

    monkeypatch.setattr(pipeline, "run_CP_decomposition_repeated", fake_cp)
    preprocess(synthetic_study, store_voxels=True)
    options = decomposition_options(
        synthetic_study,
        input=synthetic_study.root / "results" / "data" / "voxels.parquet",
        method="cp",
        ranks=(2,),
    )
    (model,) = run_decomposition(options)

    written = run_decomposition_plots(
        plot_options(synthetic_study, model, parts=("time", "spatial", "mode_grid")),
    )

    assert sorted(p.name for p in written) == [
        "rank_2__mode_grid.png",
        "rank_2__spatial_csf.png",
        "rank_2__spatial_parenchyma.png",
        "rank_2__time_mode.png",
    ]
    assert (synthetic_study.root / "results" / "plot_decomposition.json").exists()


def _segmentation_without(study: Any, labels: tuple[int, ...]) -> Path:
    """Copy of the study segmentation with `labels` set to background."""
    image = nib.load(study.root / "images" / "seg.nii")
    data = np.array(image.get_fdata())
    data[np.isin(data, labels)] = 0
    path = study.root / "images" / "seg_subset.nii"
    nib.save(nib.Nifti1Image(data, image.affine), path)
    return path


def test_plot_roi_model_with_a_segmentation_without_csf(synthetic_study: Any) -> None:
    # Regression test: an empty CSF region crashed the mode grid with an
    # IndexError from np.percentile on no values.
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))

    written = run_decomposition_plots(
        plot_options(
            synthetic_study,
            model,
            segmentation=_segmentation_without(synthetic_study, (4,)),
            parts=("mode_grid", "spatial"),
        ),
    )

    assert sorted(p.name for p in written) == [
        "rank_2__mode_grid.png",
        "rank_2__spatial_parenchyma.png",
    ]


def test_plot_roi_model_rejects_a_segmentation_in_another_label_space(
    synthetic_study: Any,
) -> None:
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    segmentation = _segmentation_without(synthetic_study, (2, 4, 10, 41, 49))

    with pytest.raises(ValueError, match="none of the model's labels"):
        run_decomposition_plots(
            plot_options(synthetic_study, model, segmentation=segmentation),
        )


# CMF ------------------------------------------------------------------------

CMF_FIT = FitOptions(restarts=2, max_iter=300, tolerance=1e-4)


def test_cmf_end_to_end(synthetic_study: Any) -> None:
    preprocess(synthetic_study)
    options = decomposition_options(
        synthetic_study,
        method="cmf",
        ranks=(2,),
        fit=CMF_FIT,
    )

    (model,) = run_decomposition(options)

    saved = load_decomposition(model)
    assert saved.method == "cmf"
    assert saved.evolving_states is not None and len(saved.evolving_states) == 6
    assert saved.label_mode.shape == (5, 2)
    # The subject mode is each subject's B_i amplitude (A is fixed at one).
    amplitude = np.stack(
        [np.sqrt((B**2).mean(axis=0)) for B in saved.evolving_states],
    )
    np.testing.assert_allclose(saved.subject_mode, amplitude, rtol=1e-5)

    distributed = decomposition_options(
        synthetic_study,
        output_dir=synthetic_study.root / "results" / "cmf_distributed",
        method="cmf",
        ranks=(2,),
        fit=CMF_FIT,
        distributed=DistributedOptions(tasks_per_job=1),
    )
    for job in range(plan_decomposition(distributed)):
        run_decomposition_job(distributed.output_dir, job)
    (gathered,) = collect_decomposition(distributed.output_dir)
    np.testing.assert_allclose(
        load_decomposition(gathered).label_mode,
        saved.label_mode,
        rtol=1e-5,
    )

    written = run_decomposition_plots(
        plot_options(
            synthetic_study,
            model,
            segmentation=synthetic_study.root / "images" / "seg.nii",
        ),
    )
    assert sorted(p.name for p in written) == [
        "rank_2__evolving_mode.png",
        "rank_2__mode_grid.png",
        "rank_2__spatial_csf.png",
        "rank_2__spatial_parenchyma.png",
        "rank_2__subject_mode.png",
    ]

    scores = pd.read_csv(
        run_replicability(
            replicability_options(synthetic_study, method="cmf", fit=CMF_FIT),
        ),
    )
    assert scores["fms"].between(0, 1).all()


def test_cmf_rejects_the_tensorly_solver(synthetic_study: Any) -> None:
    with pytest.raises(ValueError, match="solver"):
        decomposition_options(
            synthetic_study,
            method="cmf",
            fit=FitOptions(solver="tensorly"),
        )
