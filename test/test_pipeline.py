import dataclasses
import json
from pathlib import Path
from typing import Any
from typing import cast

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tensorly as tl
import torch
from gMRItensor import pipeline
from gMRItensor import preprocessing
from gMRItensor import replicability
from gMRItensor.decomposition import ConvergenceError
from gMRItensor.decomposition import restarts
from gMRItensor.jobs import FitTask
from gMRItensor.jobs import pending_tasks
from gMRItensor.jobs import RestartResult
from gMRItensor.jobs import TaskCounts
from gMRItensor.model_io import _read_decomposition
from gMRItensor.model_io import component_weights
from gMRItensor.model_io import load_decomposition
from gMRItensor.model_io import save_decomposition
from gMRItensor.model_io import select_subjects
from gMRItensor.model_io import sort_components
from gMRItensor.options import DecompositionOptions
from gMRItensor.options import DecompositionPlotOptions
from gMRItensor.options import DistributedOptions
from gMRItensor.options import FitOptions
from gMRItensor.options import GroupMapPlotOptions
from gMRItensor.options import options_to_json
from gMRItensor.options import OverviewPlotOptions
from gMRItensor.options import PreprocessingOptions
from gMRItensor.options import ReplicabilityOptions
from gMRItensor.options import StatisticsPlotOptions
from gMRItensor.options import TensorOptions
from gMRItensor.pipeline import collect_decomposition
from gMRItensor.pipeline import collect_replicability
from gMRItensor.pipeline import format_significance_summary
from gMRItensor.pipeline import group_averages
from gMRItensor.pipeline import load_plan
from gMRItensor.pipeline import plan_decomposition
from gMRItensor.pipeline import plan_replicability_jobs
from gMRItensor.pipeline import read_manifest
from gMRItensor.pipeline import run_decomposition
from gMRItensor.pipeline import run_decomposition_job
from gMRItensor.pipeline import run_decomposition_plots
from gMRItensor.pipeline import run_group_map_plots
from gMRItensor.pipeline import run_overview_plots
from gMRItensor.pipeline import run_preprocessing
from gMRItensor.pipeline import run_replicability
from gMRItensor.pipeline import run_replicability_job
from gMRItensor.pipeline import run_reproducibility_plot
from gMRItensor.pipeline import run_statistics_plots
from gMRItensor.reproducibility import read_reproducibility
from gMRItensor.reproducibility import write_reproducibility

QUICK_FIT = FitOptions(restarts=2, max_iter=50)


def fake_model(data: Any, rank: int, method: str = "CP", seed: int = 0) -> Any:
    """A random model shaped like `data` (a CP tensor or per-subject slices),
    the same for the same `seed`, so two runs over one plan agree."""
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator)

    if isinstance(data, torch.Tensor):
        return torch.ones(rank), [rand(n, rank) for n in data.shape]
    evolving = [rand(s.shape[0], rank) for s in data]
    label_mode = rand(data[0].shape[1], rank)
    if method == "CMF":
        return pipeline.CMFModel(
            weights=torch.ones(rank),
            subject_mode=torch.stack([b.pow(2).mean(0).sqrt() for b in evolving]),
            evolving_states=evolving,
            label_mode=label_mode,
        )
    return pipeline.PARAFAC2Model(
        weights=torch.ones(rank),
        subject_mode=rand(len(data), rank),
        evolving_states=evolving,
        label_mode=label_mode,
    )


def stub_fits(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, int, dict]]:
    """Store random models instead of fitting, for decompose and replicability
    alike: these tests check the pipeline, not the solvers (test_jobs and
    test_decomposition fit for real). Returns each `run_tasks` call's
    `(data, n_tasks, kwargs)`. Higher seeds get lower errors, so error order
    differs from seed order."""
    calls: list[tuple[Any, int, dict]] = []

    def fake_run_tasks(
        tasks: list,
        data: Any,
        rank: int,
        method: str,
        store: Any,
        **kwargs: Any,
    ) -> TaskCounts:
        calls.append((data, len(tasks), kwargs))
        pending, counts = pending_tasks(tasks, store, kwargs.get("resume", True))
        for task in pending:
            rows = list(task.indices)
            subset = (
                data[rows]
                if isinstance(data, torch.Tensor)
                else [data[i] for i in rows]
            )
            model = fake_model(subset, rank, method, seed=task.seed)
            store.save(RestartResult(task, model, 0.3 - 0.01 * task.seed))
        return counts

    monkeypatch.setattr(pipeline, "run_tasks", fake_run_tasks)
    monkeypatch.setattr(replicability, "run_tasks", fake_run_tasks)
    return calls


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


OVERVIEW_FIGURES = ("parenchyma", "csf", "parenchyma_regions", "csf_regions")
OVERVIEW_COLUMNS = [
    "time_point",
    "label",
    "region",
    "tissue",
    "subject",
    "group",
    "value",
    "robust_z",
]


def overview_options(study: Any, **overrides: Any) -> OverviewPlotOptions:
    values: dict[str, Any] = {
        "roi_signal": study.root / "results" / "data" / "roi_signal.parquet",
        "subject_info": study.subject_info,
        "group_variable": "diagnosis",
        "output_dir": study.root / "results",
        "formats": ("png",),
        "dpi": 50,
        **overrides,
    }
    return OverviewPlotOptions(**values)


def _edit_roi_signal(path: Path, edit: Any) -> None:
    """Rewrite `roi_signal.parquet` as `edit(frame)`, keeping its metadata."""
    table = pq.read_table(path)
    pq.write_table(
        pa.Table.from_pandas(
            edit(table.to_pandas()),
            preserve_index=False,
        ).replace_schema_metadata(table.schema.metadata),
        path,
    )


def test_run_overview_plots_figures_and_flags(synthetic_study: Any) -> None:
    results = preprocess(synthetic_study)

    def plant(signal: pd.DataFrame) -> pd.DataFrame:
        subject = signal["subject"] == "sub-02"
        # Far off in label 10 at 24 h; and no CSF label (4) at all, which
        # must not drop it from the parenchyma.
        far = subject & (signal["time_point"] == 24) & (signal["label"] == 10)
        signal.loc[far, "median"] *= 50
        return signal[~(subject & (signal["label"] == 4))]

    _edit_roi_signal(results / "data" / "roi_signal.parquet", plant)

    written = run_overview_plots(
        overview_options(synthetic_study, statistics=("median", "total_amount")),
    )

    for statistic in ("median", "total_amount"):
        overview = results / "figures" / "roi" / "overview" / statistic
        for name in OVERVIEW_FIGURES:
            assert overview / f"{name}.png" in written
    flags = pd.read_csv(results / "roi_analysis" / "overview_flags__median.csv")
    assert list(flags.columns) == OVERVIEW_COLUMNS
    planted = (flags["subject"] == "sub-02") & (flags["label"] == 10)
    assert flags.loc[planted, "time_point"].tolist() == [24]
    row = flags[planted].iloc[0]
    assert (row["region"], row["tissue"], row["group"]) == (
        "Subcortical GM",
        "parenchyma",
        "PD",
    )
    assert row["robust_z"] > 3.5
    total = pd.read_csv(results / "roi_analysis" / "overview_flags__total_amount.csv")
    assert list(total.columns) == OVERVIEW_COLUMNS
    # Only the overview: no per-ROI tables or figures, its own provenance.
    assert not (results / "roi_analysis" / "roi_statistics.parquet").exists()
    assert not (results / "figures" / "roi" / "single").exists()
    assert (results / "plot_overview.json").exists()
    assert not (results / "plot_statistics.json").exists()


def test_run_overview_plots_rejects_concentration_for_t1w(
    synthetic_study: Any,
) -> None:
    preprocess(synthetic_study, input_type="T1w")

    with pytest.raises(ValueError, match="T1w"):
        run_overview_plots(
            overview_options(synthetic_study, statistics=("total_amount",)),
        )
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    options = decomposition_options(synthetic_study, statistic=statistic)

    written = run_decomposition(options)

    out = options.output_dir
    assert written == [
        out / "parafac2_rank_1" / "model.h5",
        out / "parafac2_rank_2" / "model.h5",
    ]
    fits = pd.read_csv(out / "fits.csv")
    assert list(fits["rank"]) == [1, 2]
    assert list(fits["method"]) == ["parafac2", "parafac2"]
    assert list(fits["model"]) == [
        "parafac2_rank_1/model.h5",
        "parafac2_rank_2/model.h5",
    ]
    saved = load_decomposition(out / "parafac2_rank_2" / "model.h5")
    assert saved.method == "parafac2"
    assert saved.evolving_states is not None and len(saved.evolving_states) == 6
    assert saved.label_mode.shape == (5, 2)
    np.testing.assert_array_equal(saved.labels, [2, 4, 10, 41, 49])
    assert saved.voxel_coords is None
    record = json.loads((out / "parafac2_rank_2" / "decompose.json").read_text())
    assert record["options"]["ranks"] == [2]
    assert Path(record["options"]["input"]).is_absolute()
    assert not (out / "decompose.json").exists()
    _assert_saved_largest_first(out / "parafac2_rank_2" / "model.h5")


def test_run_decomposition_lists_every_left_out_subject(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub_fits(monkeypatch)
    _drop_session(synthetic_study, "sub-02", 6)
    preprocess(synthetic_study)
    options = decomposition_options(
        synthetic_study,
        tensor=TensorOptions(exclude_subjects=("sub-01",)),
    )

    run_decomposition(options)

    assert "Rejected at command line: 1 subject(s): sub-01" in capsys.readouterr().err
    for rank in (1, 2):
        model_dir = options.output_dir / f"parafac2_rank_{rank}"
        saved = load_decomposition(model_dir / "model.h5")
        assert list(saved.subjects) == ["sub-00", "sub-03", "sub-04", "sub-05"]
        excluded = pd.read_csv(model_dir / "excluded_subjects.csv")
        assert excluded.values.tolist() == [
            ["sub-01", "rejected at command line"],
            [
                "sub-02",
                "too few valid time points (2 of 3; needs 3): time point 6 missing",
            ],
        ]
        record = json.loads((model_dir / "decompose.json").read_text())
        assert record["options"]["tensor"]["exclude_subjects"] == ["sub-01"]

    # Rerun without exclusions: the files list only sub-02 now.
    run_decomposition(decomposition_options(synthetic_study))
    excluded = pd.read_csv(
        options.output_dir / "parafac2_rank_1" / "excluded_subjects.csv",
    )
    assert list(excluded["subject"]) == ["sub-02"]


def test_excluded_subjects_file_has_a_header_when_nobody_is_left_out(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    options = decomposition_options(synthetic_study, ranks=(1,))

    run_decomposition(options)

    path = options.output_dir / "parafac2_rank_1" / "excluded_subjects.csv"
    assert path.read_text().splitlines() == ["subject,reason"]


def test_exclude_subjects_rejects_unknown_ids_before_writing(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_fits(monkeypatch)
    preprocess(synthetic_study)
    options = decomposition_options(
        synthetic_study,
        tensor=TensorOptions(exclude_subjects=("sub-01", "sub-99")),
    )

    with pytest.raises(ValueError, match=r"not in the data: \['sub-99'\]"):
        run_decomposition(options)
    with pytest.raises(ValueError, match="sub-99"):
        plan_decomposition(options)
    assert not calls
    assert not options.output_dir.exists()


def test_fingerprint_unchanged_without_exclusions() -> None:
    """Stores fitted before `exclude_subjects` existed stay valid: the
    default leaves the fingerprint as it was (value from before the
    option), and an exclusion changes it."""
    options = DecompositionOptions(
        input=Path("/data/roi_signal.parquet"),
        output_dir=Path("/out"),
        method="parafac2",
        ranks=(2,),
    )
    excluded = dataclasses.replace(
        options,
        tensor=TensorOptions(exclude_subjects=("sub-01",)),
    )

    assert pipeline.settings_fingerprint(options) == "40cd606578e8cc98"
    assert pipeline.settings_fingerprint(excluded) != "40cd606578e8cc98"


def test_distributed_decomposition_with_exclusions(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    options = decomposition_options(
        synthetic_study,
        ranks=(1,),
        tensor=TensorOptions(exclude_subjects=("sub-04",)),
    )

    n_jobs = plan_decomposition(options)

    planned = pd.read_csv(options.output_dir / "excluded_subjects.csv")
    assert planned.values.tolist() == [["sub-04", "rejected at command line"]]
    for job in range(n_jobs):
        run_decomposition_job(options.output_dir, job)
    collect_decomposition(options.output_dir)
    model_dir = options.output_dir / "parafac2_rank_1"
    assert "sub-04" not in load_decomposition(model_dir / "model.h5").subjects
    assert (model_dir / "excluded_subjects.csv").read_text() == (
        options.output_dir / "excluded_subjects.csv"
    ).read_text()


def test_replicability_with_exclusions_needs_no_subject_info_row(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    info = pd.read_csv(synthetic_study.subject_info)
    trimmed = synthetic_study.root / "trimmed.csv"
    info[info["subjects"] != "sub-00"].to_csv(trimmed, index=False)
    options = replicability_options(
        synthetic_study,
        subject_info=trimmed,
        tensor=TensorOptions(exclude_subjects=("sub-00",)),
    )

    run_replicability(options)

    excluded = pd.read_csv(options.output_dir / "excluded_subjects.csv")
    assert excluded.values.tolist() == [["sub-00", "rejected at command line"]]
    assert pd.read_csv(options.output_dir / "excluded_columns.csv").empty


def test_decomposition_lists_the_dropped_columns(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    results = preprocess(synthetic_study)

    def blank(signal: pd.DataFrame) -> pd.DataFrame:
        scan = (signal["subject"] == "sub-02") & (signal["time_point"] == 24)
        signal.loc[scan & (signal["label"] == 10), "median"] = np.nan
        return signal

    _edit_roi_signal(results / "data" / "roi_signal.parquet", blank)
    parafac2 = decomposition_options(synthetic_study, ranks=(1,))
    cp = decomposition_options(
        synthetic_study,
        method="cp",
        ranks=(1,),
        output_dir=results / "decompositions" / "cp",
    )

    run_decomposition(parafac2)
    run_decomposition(cp)

    dropped = pd.read_csv(
        parafac2.output_dir / "parafac2_rank_1" / "excluded_columns.csv",
    )
    assert dropped.values.tolist() == [[10, 1, 1, "sub-02 t=24"]]
    imputed = pd.read_csv(cp.output_dir / "cp_rank_1" / "excluded_columns.csv")
    assert imputed.empty  # CP imputes the missing value


def test_negative_cutoff_reaches_voxel_wise_fits(
    head_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    paths = run_preprocessing(
        PreprocessingOptions(
            head_study.manifest,
            head_study.root / "results",
            "T1map",
            time_unit="ms",
            n_procs=1,
            negative_cutoff=4.0,
            store_voxels=True,
        ),
    )
    assert paths.voxels is not None
    voxels = pd.read_parquet(paths.voxels)
    first = voxels[(voxels["subject"] == "sub-0") & (voxels["time_point"] == 2)]
    n_label = first["labels"].value_counts()
    common = {"input": paths.voxels, "ranks": (1,)}
    parafac2 = decomposition_options(
        head_study,
        output_dir=head_study.root / "voxel_parafac2",
        **common,
    )
    cp = decomposition_options(
        head_study,
        method="cp",
        output_dir=head_study.root / "voxel_cp",
        **common,
    )

    run_decomposition(parafac2)
    run_decomposition(cp)

    dropped = pd.read_csv(
        parafac2.output_dir / "parafac2_rank_1" / "excluded_columns.csv",
    )
    assert dropped.values.tolist() == [
        [2, n_label[2], head_study.artifact_voxels, "sub-0 t=1"],
        [10, 27, 27, "sub-1 t=1"],  # the invalidated ROI: all its voxels
    ]
    assert pd.read_csv(cp.output_dir / "cp_rank_1" / "excluded_columns.csv").empty

    info = head_study.root / "subjects.csv"
    pd.DataFrame(
        {"subjects": ["sub-0", "sub-1", "sub-2"], "diagnosis": ["A", "A", "B"]},
    ).to_csv(info, index=False)
    written = run_group_map_plots(
        GroupMapPlotOptions(
            input=paths.voxels,
            subject_info=info,
            group_variable="diagnosis",
            output_dir=head_study.root / "maps",
            trim=0,
            formats=("png",),
            dpi=50,
        ),
    )
    assert written


def _assert_saved_largest_first(path: Path) -> None:
    """The file itself (not only `load_decomposition`) is sorted."""
    weights = component_weights(_read_decomposition(path))
    assert np.all(np.diff(weights) <= 0), weights


def test_run_decomposition_inputs_follow_the_statistic(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_fits(monkeypatch)
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
    seen = [np.stack([s.numpy() for s in data]) for data, _, _ in calls]
    np.testing.assert_allclose(seen[0][0, 0], first["median"], rtol=1e-6)
    np.testing.assert_allclose(seen[1][0, 0], first["mean"], rtol=1e-6)


def test_run_decomposition_from_voxels_keeps_coordinates(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study, store_voxels=True)
    options = decomposition_options(
        synthetic_study,
        input=synthetic_study.root / "results" / "data" / "voxels.parquet",
        ranks=(2,),
    )

    run_decomposition(options)

    saved = load_decomposition(options.output_dir / "parafac2_rank_2" / "model.h5")
    assert saved.label_mode.shape == (6**3, 2)
    assert saved.voxel_coords is not None and saved.voxel_coords.shape == (6**3, 3)
    np.testing.assert_array_equal(saved.template_shape, [6, 6, 6])
    np.testing.assert_array_equal(saved.template_affine, np.eye(4))


def test_cp_wiring_and_center(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_fits(monkeypatch)
    preprocess(synthetic_study)
    options = decomposition_options(
        synthetic_study,
        method="cp",
        ranks=(3,),
        tensor=TensorOptions(center=True, scale=False),
        fit=FitOptions(restarts=4, max_iter=10, non_negative_modes=None),
    )

    run_decomposition(options)

    ((data, n_tasks, kwargs),) = calls
    tensor = data.numpy()
    assert tensor.shape == (6, 3, 5)
    np.testing.assert_allclose(tensor.mean(axis=(0, 1)), 0.0, atol=1e-5)
    assert n_tasks == 4 and kwargs["non_negative"] is False
    assert kwargs["allow_nan_imputation"] is True
    saved = load_decomposition(options.output_dir / "cp_rank_3" / "model.h5")
    assert saved.time_mode is not None and saved.time_mode.shape == (3, 3)
    assert saved.centered is True and saved.scale_std is None


def _drop_session(study: Any, subject: str, time_point: int) -> None:
    """Remove one scan from the study's manifest."""
    manifest = pd.read_csv(study.manifest)
    keep = ~((manifest["subject"] == subject) & (manifest["time_point"] == time_point))
    manifest[keep].to_csv(study.manifest, index=False)


def test_cp_imputes_a_missing_session(synthetic_study: Any) -> None:
    # Regression test: CP used to drop (or fail on) a subject missing a scan.
    _drop_session(synthetic_study, "sub-01", 6)
    preprocess(synthetic_study)
    options = decomposition_options(
        synthetic_study,
        method="cp",
        ranks=(2,),
        tensor=TensorOptions(min_timepoints=2),
        fit=FitOptions(restarts=2, max_iter=2000, tolerance=1e-4),
    )

    with pytest.warns(UserWarning, match=r"1 missing session.*\('sub-01', 6\)"):
        run_decomposition(options)

    saved = load_decomposition(options.output_dir / "cp_rank_2" / "model.h5")
    assert "sub-01" in saved.subjects and len(saved.subjects) == 6
    assert saved.time_mode is not None
    for factor in (saved.subject_mode, saved.time_mode, saved.label_mode):
        assert np.isfinite(factor).all()
    assert np.isfinite(saved.error)


def test_parafac2_keeps_a_shorter_slice_for_a_missing_session(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_fits(monkeypatch)
    _drop_session(synthetic_study, "sub-01", 6)
    preprocess(synthetic_study)

    run_decomposition(
        decomposition_options(
            synthetic_study,
            ranks=(1,),
            tensor=TensorOptions(min_timepoints=2),
        ),
    )

    ((slices, _, kwargs),) = calls
    assert [len(s) for s in slices] == [3, 2, 3, 3, 3, 3]
    assert "allow_nan_imputation" not in kwargs


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
    calls = stub_fits(monkeypatch)
    preprocess(synthetic_study)

    run_decomposition(decomposition_options(synthetic_study, ranks=(2,), fit=fit))

    ((_, _, kwargs),) = calls
    assert {key: kwargs.get(key) for key in expected} == expected
    assert "non_negative" not in kwargs


def test_distributed_decomposition_matches_centralised(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same stored restarts -> same outputs; test_jobs checks real fits agree.
    stub_fits(monkeypatch)
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

    assert [path.parent.name for path in written] == [
        "parafac2_rank_1",
        "parafac2_rank_2",
    ]
    pd.testing.assert_frame_equal(
        pd.read_csv(distributed.output_dir / "fits.csv"),
        pd.read_csv(central.output_dir / "fits.csv"),
    )
    for name in ("parafac2_rank_1", "parafac2_rank_2"):
        got = load_decomposition(distributed.output_dir / name / "model.h5")
        expected = load_decomposition(central.output_dir / name / "model.h5")
        np.testing.assert_allclose(got.label_mode, expected.label_mode, rtol=1e-5)
    assert (distributed.output_dir / "plan.json").exists()
    # Regression: both paths keep every restart and score them the same.
    for rank in (1, 2):
        got_fms, got_errors = read_reproducibility(
            distributed.store_dir / f"parafac2_rank_{rank}" / "reproducibility.csv",
        )
        expected_fms, expected_errors = read_reproducibility(
            central.store_dir / f"parafac2_rank_{rank}" / "reproducibility.csv",
        )
        pd.testing.assert_frame_equal(got_fms, expected_fms, rtol=1e-4)
        pd.testing.assert_series_equal(got_errors, expected_errors, rtol=1e-5)


def test_run_decomposition_saves_restarts_and_reproducibility(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_fits(monkeypatch)
    # Both formats, as by default (conftest keeps only PNG for speed).
    monkeypatch.setattr(
        pipeline,
        "REPRODUCIBILITY_FIGURE",
        {"formats": ("pdf", "png"), "dpi": 50},
    )
    preprocess(synthetic_study)
    options = decomposition_options(synthetic_study, ranks=(2,))
    run_decomposition(options)

    ((_, _, kwargs),) = calls
    assert kwargs["resume"] is False  # a fresh run refits the whole store
    rank_dir = options.store_dir / "parafac2_rank_2"
    assert sorted(p.name for p in (rank_dir / "full").iterdir()) == [
        "seed_0000.pt",
        "seed_0001.pt",
    ]
    fms, errors = read_reproducibility(rank_dir / "reproducibility.csv")
    assert list(fms.index) == list(fms.columns) == [1, 0]  # by error
    assert list(errors) == pytest.approx([0.29, 0.30])
    assert pd.read_csv(options.output_dir / "fits.csv")["error"].tolist() == [
        pytest.approx(0.29),
    ]
    assert (rank_dir / "reproducibility.png").exists()
    assert (rank_dir / "reproducibility.pdf").exists()


def test_reproducibility_plot_redraws_a_saved_csv(tmp_path: Path) -> None:
    results = [
        RestartResult(
            FitTask("full", (0, 1, 2), seed),
            (torch.ones(2), [torch.rand(n, 2) for n in (3, 4, 5)]),
            0.1 * (seed + 1),
        )
        for seed in range(3)
    ]
    csv = write_reproducibility(results, tmp_path)

    written = run_reproducibility_plot(
        csv,
        tmp_path / "replot" / "fms",
        formats=("png",),
        dpi=50,
    )

    assert written == [tmp_path / "replot" / "fms.png"]
    assert written[0].exists()
    assert run_reproducibility_plot(csv, formats=("png",), dpi=50) == [
        tmp_path / "reproducibility.png",
    ]
    with pytest.raises(FileNotFoundError, match="decompose run"):
        run_reproducibility_plot(tmp_path / "missing.csv")


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
    with pytest.raises(ValueError, match=r"rank 2: 1 restart missing.* job 1 first"):
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

    assert written == [output_dir / "parafac2_rank_1" / "model.h5"]


# Replicability ------------------------------------------------------------


def test_run_replicability_halfhalf(synthetic_study: Any) -> None:
    preprocess(synthetic_study)

    path = run_replicability(replicability_options(synthetic_study))

    scores = pd.read_csv(path)
    assert list(scores.columns) == ["method", "rank", "split", "fms"]
    assert set(scores["method"]) == {"parafac2"}
    assert list(scores["split"]) == [0, 1]
    assert scores["fms"].between(0, 1).all()
    assert (path.parent / "replicability.json").exists()


def test_run_replicability_cv_columns(synthetic_study: Any) -> None:
    preprocess(synthetic_study)
    options = replicability_options(synthetic_study, engine="cv", splits=3, repeats=1)

    scores = pd.read_csv(run_replicability(options))

    assert list(scores.columns) == [
        "method",
        "rank",
        "fold_i",
        "fold_j",
        "n_common",
        "fms",
    ]
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
    # `collect` runs in a fresh process, which never sets the pytorch backend.
    previous = tl.get_backend()
    tl.set_backend("numpy")
    try:
        got = pd.read_csv(collect_replicability(distributed.output_dir))
    finally:
        tl.set_backend(previous)

    pd.testing.assert_frame_equal(got, expected, rtol=1e-5)


def test_progress_bar_option_is_passed_through(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocess(synthetic_study)
    seen = []
    original: Any = pipeline.run_tasks

    def spy(*args: Any, progress_bar: bool = True, **kwargs: Any) -> Any:
        seen.append(progress_bar)
        return original(*args, progress_bar=progress_bar, **kwargs)

    monkeypatch.setattr(pipeline, "run_tasks", spy)
    fit = FitOptions(restarts=1, max_iter=20, extra={"progress_bar": True})
    options = decomposition_options(synthetic_study, ranks=(1,), fit=fit)
    plan_decomposition(options)

    run_decomposition_job(options.output_dir, 0)

    assert seen == [True]


# GPU (GMRITENSOR_USE_GPU) --------------------------------------------------

#: `--method` -> the `compute_*` function every restart goes through.
COMPUTE_FUNCTIONS = {
    "cp": "compute_CP_decomposition",
    "parafac2": "compute_PARAFAC2_decomposition",
    "cmf": "compute_CMF_decomposition",
}
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA GPU",
)


def _devices(data: Any) -> set[torch.device]:
    """Device(s) of a tensor or of every slice of a slice list."""
    tensors = [data] if isinstance(data, torch.Tensor) else list(data)
    return {t.device for t in tensors}


def spy_fit_devices(monkeypatch: pytest.MonkeyPatch, method: str) -> list[str]:
    """Record the device type of every fit's input, then fit as usual."""
    name = COMPUTE_FUNCTIONS[method]
    # Looked up where `_fit_one` calls it.
    original = getattr(restarts, name)
    seen: list[str] = []

    def spy(data: Any, *args: Any, **kwargs: Any) -> Any:
        seen.extend(device.type for device in _devices(data))
        return original(data, *args, **kwargs)

    monkeypatch.setattr(restarts, name, spy)
    return seen


def test_on_device_moves_a_tensor_and_every_slice() -> None:
    meta = torch.device("meta")

    assert _devices(pipeline._on_device(torch.ones(2, 3), meta)) == {meta}
    slices = [torch.ones(2, 3), torch.ones(4, 3)]
    assert _devices(pipeline._on_device(slices, meta)) == {meta}


@pytest.mark.parametrize("method", list(COMPUTE_FUNCTIONS))
def test_decomposition_fits_on_the_backend_device(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    preprocess(synthetic_study)
    calls = stub_fits(monkeypatch)
    monkeypatch.setattr(pipeline, "setup_backend", lambda: torch.device("meta"))

    run_decomposition(decomposition_options(synthetic_study, method=method))

    assert calls and all(_devices(data) == {torch.device("meta")} for data, *_ in calls)


@pytest.mark.parametrize("method", list(COMPUTE_FUNCTIONS))
def test_replicability_fits_on_the_backend_device(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    preprocess(synthetic_study)
    seen: list[set[torch.device]] = []

    def fake_evaluate(engine: Any, tensor: Any, *args: Any, **kwargs: Any) -> list:
        seen.append(_devices(tensor))
        return []

    monkeypatch.setattr(pipeline, "evaluate_replicability_multiproc", fake_evaluate)
    monkeypatch.setattr(pipeline, "setup_backend", lambda: torch.device("meta"))

    run_replicability(replicability_options(synthetic_study, method=method))

    assert seen == [{torch.device("meta")}]


@pytest.mark.parametrize("method", list(COMPUTE_FUNCTIONS))
def test_flag_off_fits_on_cpu(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    preprocess(synthetic_study)
    seen = spy_fit_devices(monkeypatch, method)
    fit = FitOptions(restarts=1, max_iter=50)

    run_decomposition(
        decomposition_options(synthetic_study, method=method, ranks=(1,), fit=fit),
    )

    assert seen and set(seen) == {"cpu"}


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA fallback")
def test_flag_on_without_cuda_falls_back_to_cpu(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocess(synthetic_study)
    monkeypatch.setenv("GMRITENSOR_USE_GPU", "TRUE")
    seen = spy_fit_devices(monkeypatch, "parafac2")
    fit = FitOptions(restarts=1, max_iter=50)

    run_decomposition(decomposition_options(synthetic_study, ranks=(1,), fit=fit))

    assert seen and set(seen) == {"cpu"}


@pytest.fixture
def gpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GMRITENSOR_USE_GPU=TRUE`, with the TF32 switch `setup_backend` flips
    restored afterwards."""
    monkeypatch.setenv("GMRITENSOR_USE_GPU", "TRUE")
    monkeypatch.setattr(
        torch.backends.cuda.matmul,
        "allow_tf32",
        torch.backends.cuda.matmul.allow_tf32,
    )


@needs_cuda
@pytest.mark.parametrize("method", list(COMPUTE_FUNCTIONS))
def test_flag_on_fits_decomposition_on_the_gpu(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    gpu_backend: None,
    method: str,
) -> None:
    preprocess(synthetic_study)
    seen = spy_fit_devices(monkeypatch, method)
    torch.cuda.reset_peak_memory_stats()
    fit = FitOptions(restarts=2, max_iter=50)
    options = decomposition_options(
        synthetic_study,
        method=method,
        ranks=(1,),
        fit=fit,
    )

    (path,) = run_decomposition(options)

    assert seen and set(seen) == {"cuda"}
    assert torch.cuda.max_memory_allocated() > 0
    saved = load_decomposition(path)
    assert isinstance(saved.label_mode, np.ndarray)
    assert np.isfinite(saved.error)


@needs_cuda
@pytest.mark.parametrize("method", list(COMPUTE_FUNCTIONS))
def test_flag_on_fits_replicability_on_the_gpu(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    gpu_backend: None,
    method: str,
) -> None:
    preprocess(synthetic_study)
    seen = spy_fit_devices(monkeypatch, method)

    path = run_replicability(replicability_options(synthetic_study, method=method))

    assert seen and set(seen) == {"cuda"}
    assert len(pd.read_csv(path)) == 2


@needs_cuda
@pytest.mark.parametrize("method", list(COMPUTE_FUNCTIONS))
def test_gpu_fit_error_matches_the_cpu_fit(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    gpu_backend: None,
    method: str,
) -> None:
    preprocess(synthetic_study)
    fit = FitOptions(restarts=2, max_iter=200)
    errors = {}
    for flag in ("FALSE", "TRUE"):
        monkeypatch.setenv("GMRITENSOR_USE_GPU", flag)
        options = decomposition_options(
            synthetic_study,
            method=method,
            ranks=(1,),
            fit=fit,
            output_dir=synthetic_study.root / "results" / flag,
        )
        (path,) = run_decomposition(options)
        errors[flag] = load_decomposition(path).error

    assert errors["TRUE"] == pytest.approx(errors["FALSE"], rel=1e-2, abs=1e-3)


@needs_cuda
def test_gpu_refuses_parallel_restarts(
    synthetic_study: Any,
    gpu_backend: None,
) -> None:
    preprocess(synthetic_study)
    fit = FitOptions(restarts=2, max_iter=50, restart_procs=2)

    with pytest.raises(ValueError, match="CUDA"):
        run_decomposition(decomposition_options(synthetic_study, fit=fit))


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


def test_plot_parafac2_roi_model_with_segmentation(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
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
        "parafac2_rank_2__evolving_mode.png",
        "parafac2_rank_2__mode_grid.png",
        "parafac2_rank_2__outliers.png",
        "parafac2_rank_2__spatial_csf.png",
        "parafac2_rank_2__spatial_parenchyma.png",
        "parafac2_rank_2__subject_mode.png",
    ]
    assert (
        _figures(synthetic_study) / "parafac2_rank_2__evolving_mode_significance.csv"
    ).exists()


def test_plot_roi_model_without_segmentation(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))

    written = run_decomposition_plots(plot_options(synthetic_study, model))

    assert sorted(p.name for p in written) == [
        "parafac2_rank_2__evolving_mode.png",
        "parafac2_rank_2__outliers.png",
        "parafac2_rank_2__subject_mode.png",
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
    stub_fits(monkeypatch)
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
        "cp_rank_2__mode_grid.png",
        "cp_rank_2__spatial_csf.png",
        "cp_rank_2__spatial_parenchyma.png",
        "cp_rank_2__time_mode.png",
    ]
    assert (synthetic_study.root / "results" / "plot_decomposition.json").exists()


def _segmentation_without(study: Any, labels: tuple[int, ...]) -> Path:
    """Copy of the study segmentation with `labels` set to background."""
    image = cast(nib.Nifti1Image, nib.load(study.root / "images" / "seg.nii"))
    data = np.array(image.get_fdata())
    data[np.isin(data, labels)] = 0
    path = study.root / "images" / "seg_subset.nii"
    nib.save(nib.Nifti1Image(data, image.affine), path)
    return path


def test_plot_roi_model_with_a_segmentation_without_csf(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test: an empty CSF region crashed the mode grid with an
    # IndexError from np.percentile on no values.
    stub_fits(monkeypatch)
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
        "parafac2_rank_2__mode_grid.png",
        "parafac2_rank_2__spatial_parenchyma.png",
    ]


def test_plot_roi_model_rejects_a_segmentation_in_another_label_space(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    segmentation = _segmentation_without(synthetic_study, (2, 4, 10, 41, 49))

    with pytest.raises(ValueError, match="none of the model's labels"):
        run_decomposition_plots(
            plot_options(synthetic_study, model, segmentation=segmentation),
        )


# Group-average maps --------------------------------------------------------


def group_map_options(study: Any, **overrides: Any) -> GroupMapPlotOptions:
    values: dict[str, Any] = {
        "input": study.root / "results" / "data" / "voxels.parquet",
        "subject_info": study.subject_info,
        "group_variable": "diagnosis",
        "output_dir": study.root / "results",
        "formats": ("png",),
        "dpi": 50,
        **overrides,
    }
    return GroupMapPlotOptions(**values)


@pytest.mark.parametrize("numeric", [False, True])
def test_statistics_leave_out_subjects_without_a_group(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
    numeric: bool,
) -> None:
    """Regression: a blank group cell became a group of its own with no
    data, so nothing was tested (numeric groups) or sorting the groups
    raised (text groups)."""
    preprocess(synthetic_study, store_voxels=True)
    info = pd.read_csv(synthetic_study.subject_info)
    if numeric:
        info["diagnosis"] = (info["diagnosis"] == "PD").astype(float)
    info.loc[0, "diagnosis"] = None
    info.to_csv(synthetic_study.subject_info, index=False)
    capsys.readouterr()

    run_statistics_plots(statistics_options(synthetic_study))

    note = "Leaving out 1 subject(s) without a 'diagnosis' value: sub-00"
    assert note in capsys.readouterr().out
    tables = synthetic_study.root / "results" / "roi_analysis"
    assert not pd.read_csv(tables / "significance__median.csv").empty
    frame = pd.read_parquet(tables / "roi_statistics.parquet")
    assert "sub-00" not in set(frame["subject"])
    averages = group_averages(group_map_options(synthetic_study))
    assert averages.groups == (["0.0", "1.0"] if numeric else ["Control", "PD"])
    np.testing.assert_array_equal(averages.n_subjects[:, -1], [2, 2, 2])


def test_group_averages_of_voxels_match_a_direct_mean(synthetic_study: Any) -> None:
    results = preprocess(synthetic_study, store_voxels=True)

    averages = group_averages(group_map_options(synthetic_study, aggregate="mean"))

    assert averages.timepoints == [0, 6, 24]
    assert averages.groups == ["Control", "PD"]
    np.testing.assert_array_equal(averages.n_subjects, np.full((3, 2), 3))
    voxels = pd.read_parquet(results / "data" / "voxels.parquet")
    pd_subjects = [f"sub-{s:02d}" for s in range(3)]
    first = voxels[
        (voxels["labels"] == averages.labels[0])
        & (voxels["label_index"] == averages.label_index[0])
        & (voxels["time_point"] == 24)
        & voxels["subject"].isin(pd_subjects)
    ]
    assert averages.values[0, 2, 1] == pytest.approx(first["values"].mean())
    assert averages.voxel_coords is not None
    assert averages.values.shape == (len(averages.voxel_coords), 3, 2)


def test_group_averages_of_rois_trimmed_falls_back_to_the_median(
    synthetic_study: Any,
) -> None:
    results = preprocess(synthetic_study)
    options = group_map_options(
        synthetic_study,
        input=results / "data" / "roi_signal.parquet",
        statistic="mean",
        timepoints=(6, 24),
    )

    averages = group_averages(options)

    # Three subjects per group: too few to trim two at each end.
    roi = pd.read_parquet(results / "data" / "roi_signal.parquet")
    roi["group"] = np.where(roi["subject"] < "sub-03", "PD", "Control")
    expected = roi.groupby(["label", "time_point", "group"])["mean"].median()
    assert averages.timepoints == [6, 24]
    assert averages.voxel_coords is None
    for column, label in enumerate(averages.labels):
        for t, timepoint in enumerate(averages.timepoints):
            for g, group in enumerate(averages.groups):
                assert averages.values[column, t, g] == pytest.approx(
                    expected[(label, timepoint, group)],
                )


def test_run_group_map_plots_from_voxels(synthetic_study: Any) -> None:
    preprocess(synthetic_study, store_voxels=True)

    written = run_group_map_plots(group_map_options(synthetic_study))

    assert sorted(p.name for p in written) == sorted(
        f"voxels__trimmed__t{t}__{region}.png"
        for t in (0, 6, 24)
        for region in ("csf", "parenchyma")
    )
    assert all(p.parent.name == "group_maps" for p in written)
    assert (synthetic_study.root / "results" / "plot_group_maps.json").exists()


def test_run_group_map_plots_from_rois_needs_a_segmentation(
    synthetic_study: Any,
) -> None:
    results = preprocess(synthetic_study)
    roi_signal = results / "data" / "roi_signal.parquet"

    with pytest.raises(ValueError, match="segmentation"):
        run_group_map_plots(group_map_options(synthetic_study, input=roi_signal))
    assert not (results / "figures" / "group_maps").exists()

    written = run_group_map_plots(
        group_map_options(
            synthetic_study,
            input=roi_signal,
            aggregate="median",
            segmentation=synthetic_study.root / "images" / "seg.nii",
            timepoints=(24,),
        ),
    )

    assert sorted(p.name for p in written) == [
        "roi_signal__median__t24__csf.png",
        "roi_signal__median__t24__parenchyma.png",
    ]


def test_group_averages_reject_an_unknown_time_point(synthetic_study: Any) -> None:
    preprocess(synthetic_study, store_voxels=True)

    with pytest.raises(ValueError, match="time point"):
        group_averages(group_map_options(synthetic_study, timepoints=(12,)))


# CMF ------------------------------------------------------------------------

CMF_FIT = FitOptions(restarts=2, max_iter=300, tolerance=1e-4)


def test_cmf_end_to_end(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
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
    _assert_saved_largest_first(model)

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
        "cmf_rank_2__evolving_mode.png",
        "cmf_rank_2__mode_grid.png",
        "cmf_rank_2__outliers.png",
        "cmf_rank_2__spatial_csf.png",
        "cmf_rank_2__spatial_parenchyma.png",
        "cmf_rank_2__subject_mode.png",
    ]

    scores = pd.read_csv(
        run_replicability(
            replicability_options(synthetic_study, method="cmf", fit=CMF_FIT),
        ),
    )
    assert scores["fms"].between(0, 1 + 1e-6).all()


def test_cmf_rejects_the_tensorly_solver(synthetic_study: Any) -> None:
    with pytest.raises(ValueError, match="solver"):
        decomposition_options(
            synthetic_study,
            method="cmf",
            fit=FitOptions(solver="tensorly"),
        )


# Voxel templates -------------------------------------------------------------


def _voxel_cp_model(study: Any, monkeypatch: pytest.MonkeyPatch) -> Path:
    stub_fits(monkeypatch)
    preprocess(study, store_voxels=True)
    options = decomposition_options(
        study,
        input=study.root / "results" / "data" / "voxels.parquet",
        method="cp",
        ranks=(2,),
    )
    (model,) = run_decomposition(options)
    return model


def test_voxel_model_records_its_template(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _voxel_cp_model(synthetic_study, monkeypatch)

    segmentation = synthetic_study.root / "images" / "seg.nii"
    assert load_decomposition(model).template_path == str(segmentation.resolve())
    written = run_decomposition_plots(
        plot_options(
            synthetic_study,
            model,
            segmentation=segmentation,
            parts=("spatial",),
        ),
    )
    assert [p.name for p in written] == [
        "cp_rank_2__spatial_parenchyma.png",
        "cp_rank_2__spatial_csf.png",
    ]


def test_plot_images_must_be_on_the_template_grid(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Regression test: an image on another grid was drawn silently (or
    # crashed with an IndexError) instead of being refused.
    model = _voxel_cp_model(synthetic_study, monkeypatch)
    image = cast(
        nib.Nifti1Image,
        nib.load(synthetic_study.root / "images" / "seg.nii"),
    )
    small = tmp_path / "small.nii"
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), image.affine), small)
    shifted_affine = image.affine.copy()
    shifted_affine[0, 3] += 5.0
    shifted = tmp_path / "shifted.nii"
    nib.save(nib.Nifti1Image(np.asarray(image.get_fdata()), shifted_affine), shifted)

    for overrides in ({"segmentation": small}, {"background": shifted}):
        with pytest.raises(ValueError, match="grid"):
            run_decomposition_plots(
                plot_options(synthetic_study, model, parts=("spatial",), **overrides),
            )
    with pytest.raises(ValueError, match="slices"):
        run_decomposition_plots(
            plot_options(synthetic_study, model, parts=("spatial",), slices=(99, 0, 0)),
        )


def test_csf_is_offset_ventricles_and_label_24(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _voxel_cp_model(synthetic_study, monkeypatch)
    saved = load_decomposition(model)
    # 10002 (CSF beside white matter) is in no preset, but is CSF.
    labels = np.select(
        [saved.labels == 2, saved.labels == 41],
        [10002, 24],
        saved.labels,
    )
    relabelled = dataclasses.replace(saved, labels=labels)

    spatial = pipeline._spatial_inputs(
        relabelled.label_mode,
        pipeline._model_label_space(relabelled),
        None,
        None,
        None,
    )

    assert spatial is not None
    # Labels 4 (a ventricle), 10002 and 24 are CSF; 10 and 49 are not.
    np.testing.assert_array_equal(spatial.csf, np.isin(labels, [4, 10002, 24]))


def test_roi_spatial_maps_leave_unmodelled_labels_blank(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: voxels of a label the model has no row for (e.g. an ROI
    dropped for missing data) were drawn as loading 0, like no uptake."""
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    saved = load_decomposition(model)
    keep = saved.labels != 4
    fewer = dataclasses.replace(
        saved,
        labels=saved.labels[keep],
        label_index=saved.label_index[keep],
        label_mode=saved.label_mode[keep],
    )
    segmentation = synthetic_study.root / "images" / "seg.nii"

    spatial = pipeline._spatial_inputs(
        fewer.label_mode,
        pipeline._model_label_space(fewer),
        segmentation,
        None,
        None,
    )

    assert spatial is not None
    labels = np.rint(cast(nib.Nifti1Image, nib.load(segmentation)).get_fdata())
    labels = labels.astype(int)
    placed = labels[tuple(spatial.index_list.T)]
    assert 4 not in placed
    assert len(placed) == np.count_nonzero(np.isin(labels, fewer.labels))


def test_plots_sort_models_saved_unsorted(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model written before sorting existed is plotted largest first."""
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    saved = load_decomposition(model)
    assert saved.evolving_states is not None
    flipped = dataclasses.replace(
        saved,
        weights=saved.weights[::-1],
        subject_mode=saved.subject_mode[:, ::-1],
        label_mode=saved.label_mode[:, ::-1],
        evolving_states=[B[:, ::-1] for B in saved.evolving_states],
    )
    old = model.with_name("old_rank_2.h5")
    save_decomposition(old, flipped)
    assert np.all(np.diff(component_weights(_read_decomposition(old))) >= 0)

    drawn: dict[str, Any] = {}

    def fake_subject_mode(subject_mode: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        drawn["subject_mode"] = subject_mode
        return plt.figure(), None

    def fake_evolving_mode(states: list[np.ndarray], *args: Any, **kwargs: Any) -> Any:
        drawn["evolving_states"] = states
        drawn["evolving_subject_mode"] = kwargs["subject_mode"]
        return plt.figure(), None, pd.DataFrame()

    monkeypatch.setattr(pipeline, "plot_subject_mode", fake_subject_mode)
    monkeypatch.setattr(pipeline, "plot_evolving_mode", fake_evolving_mode)
    run_decomposition_plots(
        plot_options(synthetic_study, old, parts=("subject_mode", "time")),
    )

    expected = sort_components(flipped)
    np.testing.assert_array_equal(drawn["subject_mode"], expected.subject_mode)
    # Each subject's own B_i: the subject weights only decide which are drawn.
    assert expected.evolving_states is not None
    for got, want in zip(drawn["evolving_states"], expected.evolving_states):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(
        drawn["evolving_subject_mode"],
        expected.subject_mode,
    )
    np.testing.assert_array_equal(drawn["subject_mode"], saved.subject_mode)


def test_plots_share_the_subject_coefficient_cutoff(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evolving mode and the mode grid drop the same subjects."""
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    seen: dict[str, float] = {}

    def fake_mode_grid(*args: Any, **kwargs: Any) -> Any:
        seen["mode_grid"] = kwargs["negligible_atol"]
        return plt.figure(), None

    def fake_evolving_mode(*args: Any, **kwargs: Any) -> Any:
        seen["time"] = kwargs["negligible_atol"]
        return plt.figure(), None, pd.DataFrame()

    monkeypatch.setattr(pipeline, "plot_mode_grid", fake_mode_grid)
    monkeypatch.setattr(pipeline, "plot_evolving_mode", fake_evolving_mode)
    run_decomposition_plots(
        plot_options(
            synthetic_study,
            model,
            parts=("mode_grid", "time"),
            segmentation=synthetic_study.root / "images" / "seg.nii",
            min_subject_coefficient=1e-3,
        ),
    )

    assert seen == {"mode_grid": 1e-3, "time": 1e-3}


def test_subject_prefix_renames_only_the_subject_mode_figure(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(
        decomposition_options(synthetic_study, method="cp", ranks=(2,)),
    )
    parts = ("subject_mode", "time")

    plain = run_decomposition_plots(plot_options(synthetic_study, model, parts=parts))
    prefixed = run_decomposition_plots(
        plot_options(synthetic_study, model, parts=parts, subject_prefix="sleep"),
    )

    assert sorted(p.name for p in plain) == [
        "cp_rank_2__subject_mode.png",
        "cp_rank_2__time_mode.png",
    ]
    assert sorted(p.name for p in prefixed) == [
        "cp_rank_2__time_mode.png",
        "sleep_cp_rank_2__subject_mode.png",
    ]


def _subject_info_with_gaps(study: Any, drop: list[str], blank: list[str]) -> Path:
    """The study's subject info without the `drop` rows and with a blank
    diagnosis for `blank`."""
    info = pd.read_csv(study.subject_info, dtype={"subjects": str})
    info = info[~info["subjects"].isin(drop)]
    info.loc[info["subjects"].isin(blank), "diagnosis"] = None
    path = study.root / "subjects_with_gaps.csv"
    info.to_csv(path, index=False)
    return path


def test_plots_leave_out_subjects_without_a_group(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    info = pd.read_csv(synthetic_study.subject_info, dtype={"subjects": str})
    pd_subjects = list(info.loc[info["diagnosis"] == "PD", "subjects"])
    control = list(info.loc[info["diagnosis"] == "Control", "subjects"])
    gaps = _subject_info_with_gaps(
        synthetic_study,
        drop=[control[0]],
        blank=[pd_subjects[0]],
    )
    drawn: dict[str, Any] = {}
    real_subject_mode = pipeline.plot_subject_mode

    def spy(
        subject_mode: np.ndarray,
        subjects: list[str],
        *args: Any,
        **kw: Any,
    ) -> Any:
        drawn["subjects"] = subjects
        drawn["rows"] = subject_mode.shape[0]
        return real_subject_mode(subject_mode, subjects, *args, **kw)

    monkeypatch.setattr(pipeline, "plot_subject_mode", spy)
    written = run_decomposition_plots(
        plot_options(
            synthetic_study,
            model,
            subject_info=gaps,
            parts=("mode_grid", "subject_mode", "time"),
            segmentation=synthetic_study.root / "images" / "seg.nii",
        ),
    )

    out = capsys.readouterr().out
    assert "Leaving out 2 subject(s)" in out
    assert pd_subjects[0] in out and control[0] in out
    assert sorted(drawn["subjects"]) == sorted(pd_subjects[1:] + control[1:])
    assert drawn["rows"] == 4
    assert {p.name for p in written} >= {
        "parafac2_rank_2__mode_grid.png",
        "parafac2_rank_2__subject_mode.png",
        "parafac2_rank_2__evolving_mode.png",
    }


@pytest.mark.parametrize("method", ["cp", "parafac2", "cmf"])
def test_plots_report_component_weights(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """The amplitude of each component, once its modes are normalised, of
    the full fit: leaving out subjects without a group does not change it."""
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    fit = CMF_FIT if method == "cmf" else QUICK_FIT
    (model,) = run_decomposition(
        decomposition_options(synthetic_study, method=method, ranks=(2,), fit=fit),
    )
    info = pd.read_csv(synthetic_study.subject_info, dtype={"subjects": str})
    gaps = _subject_info_with_gaps(
        synthetic_study,
        drop=[info["subjects"][0]],
        blank=[],
    )

    run_decomposition_plots(
        plot_options(
            synthetic_study,
            model,
            subject_info=gaps,
            # Both parts group subjects, so every method drops one first.
            parts=("subject_mode", "time"),
        ),
    )

    table = pd.read_csv(
        _figures(synthetic_study) / f"{method}_rank_2__component_weights.csv",
    )
    expected = component_weights(load_decomposition(model))
    assert list(table.columns) == ["component", "weight", "relative_weight"]
    assert list(table["component"]) == [1, 2]
    np.testing.assert_allclose(table["weight"], expected, rtol=1e-6)
    np.testing.assert_allclose(table["relative_weight"], expected / expected.sum())
    assert table["weight"].is_monotonic_decreasing


def test_plots_need_two_subjects_per_group(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    info = pd.read_csv(synthetic_study.subject_info, dtype={"subjects": str})
    control = list(info.loc[info["diagnosis"] == "Control", "subjects"])
    gaps = _subject_info_with_gaps(
        synthetic_study,
        drop=control[:1],
        blank=control[1:2],
    )

    with pytest.raises(ValueError, match="'Control'.*1 subject"):
        run_decomposition_plots(
            plot_options(
                synthetic_study,
                model,
                subject_info=gaps,
                parts=("subject_mode",),
            ),
        )


def test_old_plans_with_a_normalize_option_ask_to_plan_again(tmp_path: Path) -> None:
    """Plans saved before fits stopped normalising hold `normalize` in
    `fit.extra`; they now fail with the usual re-plan hint."""
    options = DecompositionOptions(
        input=Path("x.parquet"),
        output_dir=tmp_path,
        method="parafac2",
        ranks=(2,),
    )
    record = options_to_json(options)
    record["options"]["fit"]["extra"] = {"normalize": False}
    (tmp_path / "plan.json").write_text(json.dumps(record))

    with pytest.raises(ValueError, match="run `plan` again"):
        load_plan(tmp_path, DecompositionOptions)


def test_component_weights_table_handles_an_all_zero_fit() -> None:
    table = pipeline._component_weights_table(np.zeros(2))
    assert list(table["relative_weight"]) == [0.0, 0.0]
    table = pipeline._component_weights_table(np.array([3.0, 1.0]))
    assert list(table["relative_weight"]) == [0.75, 0.25]


# Outlier diagnostics -------------------------------------------------------


@pytest.mark.parametrize("method", ["cp", "parafac2", "cmf"])
def test_plot_outliers_part_writes_figure_and_table(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(
        decomposition_options(synthetic_study, method=method, ranks=(2,)),
    )

    written = run_decomposition_plots(
        plot_options(synthetic_study, model, parts=("outliers",)),
    )

    assert [p.name for p in written] == [f"{method}_rank_2__outliers.png"]
    table = pd.read_csv(
        _figures(synthetic_study) / f"{method}_rank_2__outliers.csv",
        dtype={"subject": str},
    )
    saved = load_decomposition(model)
    assert list(table["subject"]) == [str(s) for s in saved.subjects]
    assert table["leverage"].sum() == pytest.approx(2)
    assert table["residual"].sum() == pytest.approx(1)
    assert "Possible outliers" in capsys.readouterr().out


def test_plot_outliers_reloads_a_fit_with_excluded_subjects(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(
        decomposition_options(
            synthetic_study,
            ranks=(2,),
            tensor=TensorOptions(exclude_subjects=("sub-03",)),
        ),
    )

    run_decomposition_plots(plot_options(synthetic_study, model, parts=("outliers",)))

    table = pd.read_csv(
        _figures(synthetic_study) / "parafac2_rank_2__outliers.csv",
        dtype={"subject": str},
    )
    assert "sub-03" not in set(table["subject"])
    assert len(table) == 5


def test_plot_outliers_needs_the_fitted_data(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    results = preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    record = model.with_name("decompose.json")
    data = json.loads(record.read_text())
    data["options"]["input"] = str(synthetic_study.root / "moved.parquet")
    record.write_text(json.dumps(data))

    # Chosen explicitly: an error that names the way out.
    with pytest.raises(FileNotFoundError, match="--input"):
        run_decomposition_plots(
            plot_options(synthetic_study, model, parts=("outliers",)),
        )
    # --input points at the data instead.
    written = run_decomposition_plots(
        plot_options(
            synthetic_study,
            model,
            parts=("outliers",),
            input=results / "data" / "roi_signal.parquet",
        ),
    )
    assert [p.name for p in written] == ["parafac2_rank_2__outliers.png"]
    # By default: skipped with a note.
    record.unlink()
    capsys.readouterr()
    written = run_decomposition_plots(plot_options(synthetic_study, model))
    assert "parafac2_rank_2__outliers.png" not in [p.name for p in written]
    assert "Skipping the outlier part" in capsys.readouterr().out


def test_plot_outliers_rejects_data_with_other_subjects(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    saved = load_decomposition(model)
    fewer = select_subjects(saved, list(range(len(saved.subjects) - 1)))

    with pytest.raises(ValueError, match=r"differs from what .* \(subjects"):
        pipeline.load_model_data(model, fewer)


def test_replanning_with_other_settings_refits_the_stored_restarts(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`plan` again with other settings: the jobs refit the stored restarts
    and overwrite them (never reuse them); `collect` waits for all of them."""
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    options = decomposition_options(synthetic_study, ranks=(1,))
    for job in range(plan_decomposition(options)):
        run_decomposition_job(options.output_dir, job)

    # More restarts or another job size keep what is stored.
    more = dataclasses.replace(
        options,
        fit=dataclasses.replace(options.fit, restarts=3),
        distributed=DistributedOptions(tasks_per_job=2),
    )
    capsys.readouterr()
    n_jobs = plan_decomposition(more)
    assert "2 restarts already stored with these settings will be reused" in (
        capsys.readouterr().err
    )
    for job in range(n_jobs):
        run_decomposition_job(options.output_dir, job)
    out = capsys.readouterr().out
    assert "rank 1: 0 fitted, 2 reused, 0 refitted" in out
    assert "rank 1: 1 fitted, 0 reused, 0 refitted" in out
    collect_decomposition(options.output_dir)

    changed = dataclasses.replace(
        more,
        fit=dataclasses.replace(more.fit, max_iter=more.fit.max_iter + 1),
    )
    plan_decomposition(changed)
    assert "3 restarts in" in capsys.readouterr().err
    with pytest.raises(
        ValueError,
        match=r"rank 1: 3 restarts fitted with other.*jobs 0, 1 first",
    ):
        collect_decomposition(options.output_dir)
    run_decomposition_job(options.output_dir, 0)
    assert "rank 1: 0 fitted, 0 reused, 2 refitted" in capsys.readouterr().out
    with pytest.raises(ValueError, match=r"1 restart fitted with other.* job 1 first"):
        collect_decomposition(options.output_dir)
    run_decomposition_job(options.output_dir, 1)
    collect_decomposition(options.output_dir)


def test_plan_overwrite_removes_the_planned_ranks_restarts(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    options = decomposition_options(synthetic_study, ranks=(1, 2))
    for job in range(plan_decomposition(options)):
        run_decomposition_job(options.output_dir, job)
    store = options.store_dir

    capsys.readouterr()
    plan_decomposition(dataclasses.replace(options, ranks=(1,)), overwrite=True)

    assert "Removed 2 stored restarts" in capsys.readouterr().err
    assert not (store / "parafac2_rank_1").exists()
    assert (store / "parafac2_rank_2").exists()  # not planned: kept
    run_decomposition_job(options.output_dir, 0)
    assert "rank 1: 1 fitted, 0 reused, 0 refitted" in capsys.readouterr().out
    run_decomposition_job(options.output_dir, 0)  # a re-queued job resumes
    assert "rank 1: 0 fitted, 1 reused, 0 refitted" in capsys.readouterr().out


def test_a_rank_that_does_not_converge_keeps_the_others(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    options = decomposition_options(synthetic_study, ranks=(1, 2))
    finish_rank = pipeline._finish_rank

    def rank_2_fails(plan: Any, options: Any, rank: int, *args: Any) -> Any:
        if rank == 2:
            raise ConvergenceError(f"No decomposition converged at rank {rank}")
        return finish_rank(plan, options, rank, *args)

    monkeypatch.setattr(pipeline, "_finish_rank", rank_2_fails)

    with pytest.raises(ConvergenceError, match="rank 2.*other ranks were written"):
        run_decomposition(options)
    assert (options.output_dir / "parafac2_rank_1" / "model.h5").exists()
    assert list(pd.read_csv(options.output_dir / "fits.csv")["rank"]) == [1]


def test_plan_refuses_an_impossible_parafac2_rank(synthetic_study: Any) -> None:
    preprocess(synthetic_study)

    with pytest.raises(ValueError, match="rank"):
        plan_decomposition(decomposition_options(synthetic_study, ranks=(2, 50)))
    assert not (
        synthetic_study.root / "results" / "decompositions" / "test" / "plan.json"
    ).exists()


def test_stratifying_by_a_column_with_blanks_is_refused(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    info = pd.read_csv(synthetic_study.subject_info)
    info.loc[0, "diagnosis"] = None
    info.to_csv(synthetic_study.subject_info, index=False)

    with pytest.raises(ValueError, match="without a 'diagnosis' value.*sub-00"):
        run_replicability(replicability_options(synthetic_study))


def test_models_in_one_output_dir_keep_their_own_record(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: one shared decompose.json made outliers of the first
    model reload the data with the second model's settings."""
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (parafac2,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    (cp,) = run_decomposition(
        decomposition_options(
            synthetic_study,
            method="cp",
            ranks=(2,),
            tensor=TensorOptions(scale=False),
        ),
    )

    fits = pd.read_csv(parafac2.parent.parent / "fits.csv")
    assert list(zip(fits["method"], fits["rank"])) == [("cp", 2), ("parafac2", 2)]
    for model in (parafac2, cp):
        # Each reloads its own data: the scaling checks would raise otherwise.
        pipeline.load_model_data(model, load_decomposition(model))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method", "cp", "records a cp fit"),
        ("ranks", [3], r"at rank\(s\) \[3\]"),
        ("tensor", {"scale": True, "center": True}, "centred"),
    ],
)
def test_plot_outliers_rejects_a_record_that_does_not_match_the_fit(
    synthetic_study: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    (model,) = run_decomposition(decomposition_options(synthetic_study, ranks=(2,)))
    record = model.with_name("decompose.json")
    data = json.loads(record.read_text())
    data["options"][field] = (
        {**data["options"][field], **value} if isinstance(value, dict) else value
    )
    record.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=message):
        run_decomposition_plots(
            plot_options(synthetic_study, model, parts=("outliers",)),
        )
    # By default the other parts are still drawn.
    capsys.readouterr()
    names = [
        p.name for p in run_decomposition_plots(plot_options(synthetic_study, model))
    ]
    assert "parafac2_rank_2__outliers.png" not in names
    assert "parafac2_rank_2__subject_mode.png" in names
    assert "Skipping the outlier part" in capsys.readouterr().out


def test_plot_outliers_from_another_working_directory(
    synthetic_study: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a relative --input was recorded as given, so plotting
    from another directory could not find the data."""
    stub_fits(monkeypatch)
    preprocess(synthetic_study)
    monkeypatch.chdir(synthetic_study.root)
    options = dataclasses.replace(
        decomposition_options(synthetic_study, ranks=(2,)),
        input=Path("results/data/roi_signal.parquet"),
        output_dir=Path("results/decompositions/test"),
    )
    (model,) = run_decomposition(options)
    model_dir = (synthetic_study.root / model).parent
    elsewhere = synthetic_study.root / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    written = run_decomposition_plots(
        plot_options(synthetic_study, model_dir, parts=("outliers",)),
    )

    assert [p.name for p in written] == ["parafac2_rank_2__outliers.png"]


def test_run_statistics_plots_sd_bands_write_the_cohort_table(
    synthetic_study: Any,
) -> None:
    preprocess(synthetic_study)

    run_statistics_plots(statistics_options(synthetic_study, sd_bands=True))

    cohort = pd.read_parquet(
        synthetic_study.root / "results" / "roi_analysis" / "cohort__median.parquet",
    )
    assert list(cohort.columns) == ["roi", "timepoint", "mean", "std", "n"]
    # Six subjects: nothing can lie beyond 3 SD, but the table is written.
    outside = pd.read_csv(
        synthetic_study.root / "results" / "roi_analysis" / "outside_3sd__median.csv",
    )
    assert outside.empty and "z" in outside.columns
    with pytest.raises(ValueError, match="--sd-bands needs --layout rows"):
        statistics_options(synthetic_study, sd_bands=True, layout="panels")
