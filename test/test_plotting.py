import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from gMRItensor.plotting.evolving_mode import plot_evolving_mode
from gMRItensor.plotting.subject_mode import _prepare_plotting_dataframe
from gMRItensor.plotting.subject_mode import make_subject_boxplot
from gMRItensor.plotting.subject_mode import make_variable_correlation
from gMRItensor.plotting.subject_mode import plot_subject_mode
from gMRItensor.plotting.subject_mode import plot_subject_mode_correlation
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import create_colorbar_with_offset
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import scale_mode

matplotlib.use("Agg")


# TODO: gMRItensor.plotting.spatial_mode and gMRItensor.plotting.mode_grid
# still need coverage; they require realistic 3D volumes/masks to exercise
# meaningfully, which is a larger effort than fits here.


def make_subject_info(n_per_group: int = 6):
    n = 2 * n_per_group
    rng = np.random.default_rng(0)
    subjects = [f"sub-{i:02d}" for i in range(n)]
    return (
        pd.DataFrame(
            {
                "subjects": subjects,
                "group": ["A"] * n_per_group + ["B"] * n_per_group,
                "age": rng.normal(50, 10, n),
            },
        ),
        subjects,
    )


# --- gMRItensor.plotting.utils ---


def test_scale_mode():
    arr = np.array([[3.0, 0.0], [4.0, 5.0]])
    scaled = scale_mode(arr)
    norms = np.linalg.norm(scaled, axis=0)
    assert np.allclose(norms, 1.0)


def test_compute_figsize_basic():
    width, height = compute_figsize(
        n_components=2,
        n_columns=1,
        page_width=7.0,
        width_to_height_ratio=1.618,
        add_title_margin=False,
    )
    assert width == 7.0
    expected_height = (7.0 / 1) / 1.618 * 2
    assert height == pytest.approx(expected_height)


def test_compute_figsize_title_margin():
    _, height_no_margin = compute_figsize(
        n_components=1,
        n_columns=1,
        page_width=7.0,
        add_title_margin=False,
    )
    _, height_with_margin = compute_figsize(
        n_components=1,
        n_columns=1,
        page_width=7.0,
        add_title_margin=True,
    )
    assert height_with_margin == pytest.approx(height_no_margin * 1.1)


def test_get_color_palette_length():
    for n in [1, 2, 5]:
        colors = get_color_palette(n)
        assert len(colors) == n


def test_create_colorbar_with_offset():
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.rand(4, 4))
    cax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    offset = create_colorbar_with_offset(fig, ax, im, cax, label="value")
    assert isinstance(offset, float)
    plt.close(fig)


# --- gMRItensor.plotting.subject_mode: _prepare_plotting_dataframe ---


def test_prepare_plotting_dataframe_valid():
    subject_info, subjects = make_subject_info()
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects), 3))
    df = _prepare_plotting_dataframe(subject_mode, subjects, subject_info, "group")
    assert len(df) == len(subjects)
    assert {"comp_0", "comp_1", "comp_2", "group"}.issubset(df.columns)


def test_prepare_plotting_dataframe_row_mismatch():
    subject_info, subjects = make_subject_info()
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects) - 1, 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(subject_mode, subjects, subject_info, "group")


def test_prepare_plotting_dataframe_missing_group_column():
    subject_info, subjects = make_subject_info()
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects), 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(
            subject_mode,
            subjects,
            subject_info,
            "does_not_exist",
        )


def test_prepare_plotting_dataframe_missing_subjects_column():
    subject_info, subjects = make_subject_info()
    subject_info = subject_info.rename(columns={"subjects": "id"})
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects), 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(subject_mode, subjects, subject_info, "group")


def test_prepare_plotting_dataframe_empty_merge():
    subject_info, _ = make_subject_info()
    other_subjects = ["not-a-subject"] * len(subject_info)
    subject_mode = np.random.default_rng(0).normal(size=(len(other_subjects), 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(subject_mode, other_subjects, subject_info, "group")


def test_prepare_plotting_dataframe_single_group():
    subject_info, subjects = make_subject_info()
    subject_info = subject_info.copy()
    subject_info["group"] = "A"
    subject_mode = np.random.default_rng(0).normal(size=(len(subjects), 3))
    with pytest.raises(ValueError):
        _prepare_plotting_dataframe(subject_mode, subjects, subject_info, "group")


# --- gMRItensor.plotting.subject_mode: make_subject_boxplot / make_variable_correlation ---


def make_boxplot_df(n_per_group: int = 8):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "group": ["A"] * n_per_group + ["B"] * n_per_group,
            "value": np.concatenate(
                [rng.normal(0, 1, n_per_group), rng.normal(1, 1, n_per_group)],
            ),
        },
    )


def test_make_subject_boxplot_boxplot():
    df = make_boxplot_df()
    fig, ax = plt.subplots()
    ylim, required_xlim = make_subject_boxplot(
        ax,
        df,
        x_column="group",
        y_column="value",
        legend=True,
    )
    assert len(ylim) == 2
    assert required_xlim is not None
    plt.close(fig)


def test_make_subject_boxplot_no_legend():
    df = make_boxplot_df()
    fig, ax = plt.subplots()
    _, required_xlim = make_subject_boxplot(
        ax,
        df,
        x_column="group",
        y_column="value",
        legend=False,
    )
    assert required_xlim is None
    plt.close(fig)


def test_make_variable_correlation():
    rng = np.random.default_rng(2)
    n = 20
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": x * 2 + rng.normal(scale=0.1, size=n),
            "group": ["A"] * (n // 2) + ["B"] * (n // 2),
        },
    )
    fig, ax = plt.subplots()
    make_variable_correlation(ax, df, x_column="x", y_column="y", category="group")
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    plt.close(fig)


# --- gMRItensor.plotting.subject_mode: full figure smoke tests ---


def test_plot_subject_mode():
    subject_info, subjects = make_subject_info()
    n_components = 3
    subject_mode = np.random.default_rng(3).normal(size=(len(subjects), n_components))

    fig, axs = plot_subject_mode(
        subject_mode,
        subjects,
        subject_info,
        group_variable="group",
        plotting_variables=["age"],
    )
    assert axs.shape == (n_components, 2)
    plt.close(fig)


def test_plot_subject_mode_correlation():
    subject_info, subjects = make_subject_info()
    n_components = 2
    subject_mode = np.random.default_rng(4).normal(size=(len(subjects), n_components))

    fig, axs = plot_subject_mode_correlation(
        subject_mode,
        subjects,
        subject_info,
        group_variable="group",
    )
    assert axs.shape == (n_components, n_components)
    plt.close(fig)


def test_plot_subject_mode_single_component():
    # Regression test: matplotlib squeezes plt.subplots' axes array when any
    # grid dimension is 1, which used to break indexing like ax[0]/ax[i].
    subject_info, subjects = make_subject_info()
    n_components = 1
    subject_mode = np.random.default_rng(5).normal(size=(len(subjects), n_components))

    fig, axs = plot_subject_mode(
        subject_mode,
        subjects,
        subject_info,
        group_variable="group",
        plotting_variables=["age"],
    )
    assert axs.shape == (n_components, 2)
    plt.close(fig)


def test_plot_subject_mode_correlation_single_component():
    # Regression test for the same axes-squeeze hazard as above.
    subject_info, subjects = make_subject_info()
    n_components = 1
    subject_mode = np.random.default_rng(6).normal(size=(len(subjects), n_components))

    fig, axs = plot_subject_mode_correlation(
        subject_mode,
        subjects,
        subject_info,
        group_variable="group",
    )
    assert axs.shape == (n_components, n_components)
    plt.close(fig)


# --- gMRItensor.plotting.evolving_mode ---


def make_evolving_factors(n_subjects_per_group=3, n_components=2, seed=0):
    rng = np.random.default_rng(seed)
    subjects = [f"sub-{i:02d}" for i in range(2 * n_subjects_per_group)]
    subject_info = pd.DataFrame(
        {
            "subjects": subjects,
            "group": ["A"] * n_subjects_per_group + ["B"] * n_subjects_per_group,
        },
    )
    # Ragged: each subject gets a different number of time points.
    evolving_factors = []
    timepoints_per_subject = []
    for i in range(len(subjects)):
        n_timepoints = 4 + (i % 3)
        evolving_factors.append(rng.normal(size=(n_timepoints, n_components)))
        timepoints_per_subject.append(np.arange(n_timepoints))
    return evolving_factors, timepoints_per_subject, subjects, subject_info


def test_plot_evolving_mode():
    evolving_factors, timepoints, subjects, subject_info = make_evolving_factors()
    fig, axs = plot_evolving_mode(
        evolving_factors,
        timepoints,
        subjects,
        subject_info,
        group_variable="group",
    )
    assert axs.shape == (2, 1)
    plt.close(fig)


def test_plot_evolving_mode_single_component():
    # Regression test for the same axes-squeeze hazard exercised elsewhere in
    # this file (n_components == 1).
    evolving_factors, timepoints, subjects, subject_info = make_evolving_factors(
        n_components=1,
    )
    fig, axs = plot_evolving_mode(
        evolving_factors,
        timepoints,
        subjects,
        subject_info,
        group_variable="group",
    )
    assert axs.shape == (1, 1)
    plt.close(fig)


def test_plot_evolving_mode_length_mismatch():
    evolving_factors, timepoints, subjects, subject_info = make_evolving_factors()
    with pytest.raises(ValueError):
        plot_evolving_mode(
            evolving_factors,
            timepoints,
            subjects[:-1],
            subject_info,
            group_variable="group",
        )


def test_plot_evolving_mode_missing_subject():
    evolving_factors, timepoints, subjects, subject_info = make_evolving_factors()
    subject_info = subject_info[subject_info["subjects"] != subjects[0]]
    with pytest.raises(ValueError):
        plot_evolving_mode(
            evolving_factors,
            timepoints,
            subjects,
            subject_info,
            group_variable="group",
        )
