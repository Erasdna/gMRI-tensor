import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from gMRItensor.group_statistics import compare_groups_over_time
from gMRItensor.group_statistics import summarize_cohort_over_time
from gMRItensor.group_statistics import summarize_groups_over_time
from gMRItensor.plotting.roi_evolution import figure_path
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_panels
from gMRItensor.plotting.roi_evolution import plot_roi_evolution_rows
from gMRItensor.plotting.roi_evolution import SD_BAND_LABEL
from gMRItensor.plotting.roi_evolution import statistic_label
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import JOURNAL_WIDTHS
from gMRItensor.plotting.utils import resolve_page_width
from gMRItensor.plotting.utils import save_figure

matplotlib.use("Agg")

ROIS = ("ventricles", "white_matter", "thalamus")


def make_roi_statistics(n_per_group: int = 6, seed: int = 0) -> pd.DataFrame:
    """Per-scan ROI frame: group B is shifted in `ventricles`
    at time points 2 and 3 only, so exactly those are significant."""
    rng = np.random.default_rng(seed)
    rows = []
    for roi in ROIS:
        for i in range(2 * n_per_group):
            group = "A" if i < n_per_group else "B"
            for timepoint in (0, 1, 2, 3):
                value = rng.normal()
                if roi == "ventricles" and group == "B" and timepoint >= 2:
                    value += 10.0
                rows.append(
                    {
                        "subject": f"s{i}",
                        "group": group,
                        "roi": roi,
                        "timepoint": timepoint,
                        "median": value,
                    },
                )
    return pd.DataFrame(rows)


@pytest.fixture
def roi_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = make_roi_statistics()
    summary = summarize_groups_over_time(stats, facet="roi", value="median")
    significance = compare_groups_over_time(stats, ["A", "B"], "roi", "median")
    significance["significant"] = significance["p_adj"] < 0.05
    return stats, summary, significance


def _stars(ax: matplotlib.axes.Axes) -> list[float]:
    return sorted(text.xy[0] for text in ax.texts if text.get_text() == "*")


def test_rows_layout_shape_stars_and_colors(roi_tables):
    stats, summary, significance = roi_tables

    fig, axs = plot_roi_evolution_rows(
        summary,
        stats,
        significance,
        rois=["ventricles", "white_matter"],
        statistic="median",
        page_width="double",
    )

    assert axs.shape == (2, 3)  # ribbon + one column per group
    assert fig.get_size_inches()[0] == pytest.approx(JOURNAL_WIDTHS["double"])
    assert _stars(axs[0, 0]) == [2, 3]
    assert _stars(axs[1, 0]) == []
    # One curve per subject in each group column, in that group's color.
    color_a, color_b = get_color_palette(2)
    assert len(axs[0, 1].lines) == len(axs[0, 2].lines) == 6
    assert matplotlib.colors.same_color(axs[0, 1].lines[0].get_color(), color_a)
    assert matplotlib.colors.same_color(axs[0, 2].lines[0].get_color(), color_b)
    # Each row shares one y-range, so ribbon and subject columns compare.
    assert axs[0, 0].get_ylim() == axs[0, 1].get_ylim() == axs[0, 2].get_ylim()
    # Rows link their y axes; only the left column labels its ticks.
    shared = axs[0, 0].get_shared_y_axes()
    assert shared.joined(axs[0, 0], axs[0, 2])
    assert not shared.joined(axs[0, 0], axs[1, 0])
    assert axs[0, 0].yaxis.majorTicks[0].label1.get_visible()
    assert not axs[0, 1].yaxis.majorTicks[0].label1.get_visible()
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in axs.flat)
    assert axs[0, 0].get_ylabel() == "Ventricles"
    assert fig.get_supylabel() == "Median ΔR1 (1/s)"
    plt.close(fig)


def test_panels_layout_hides_unused_axes_and_has_one_legend(roi_tables):
    _, summary, significance = roi_tables

    fig, axs = plot_roi_evolution_panels(
        summary,
        significance,
        rois=list(ROIS),
        statistic="median",
        n_rows=2,
        n_cols=2,
        page_width=5.0,
    )

    assert axs.shape == (2, 2)
    assert [ax.get_visible() for ax in axs.flat] == [True, True, True, False]
    assert [ax.get_title() for ax in axs.flat[:3]] == ["Ventricles", "WM", "Thalamus"]
    assert _stars(axs[0, 0]) == [2, 3]
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in axs.flat)
    assert fig.get_size_inches()[0] == pytest.approx(5.0)
    plt.close(fig)


def test_panels_layout_sharey(roi_tables):
    _, summary, significance = roi_tables

    fig, axs = plot_roi_evolution_panels(
        summary,
        significance,
        rois=list(ROIS),
        statistic="median",
        n_rows=1,
        n_cols=3,
        sharey=True,
    )

    assert len({ax.get_ylim() for ax in axs.flat}) == 1
    plt.close(fig)


def test_layouts_reject_bad_rois(roi_tables):
    stats, summary, significance = roi_tables

    with pytest.raises(ValueError, match="not_a_roi"):
        plot_roi_evolution_rows(summary, stats, significance, ["not_a_roi"], "median")
    with pytest.raises(ValueError, match="2 x 1"):
        plot_roi_evolution_panels(summary, significance, list(ROIS), "median", 2, 1)


def test_layouts_without_significance(roi_tables):
    stats, summary, significance = roi_tables

    fig, axs = plot_roi_evolution_rows(
        summary,
        stats,
        significance.iloc[0:0],
        ["ventricles"],
        "median",
    )

    assert _stars(axs[0, 0]) == []
    plt.close(fig)


def test_statistic_label_units():
    assert statistic_label("median") == "Median ΔR1 (1/s)"
    assert statistic_label("mean_concentration") == "Mean concentration (mM)"
    assert statistic_label("total_amount") == "Total amount (mmol)"
    assert statistic_label("custom") == "custom"


def test_resolve_page_width():
    assert resolve_page_width("single") == 3.5
    assert resolve_page_width("onehalf") == 5.5
    assert resolve_page_width(6.0) == 6.0
    with pytest.raises(ValueError, match="page_width"):
        resolve_page_width("poster")


def test_figure_path(tmp_path):
    assert (
        figure_path(tmp_path, "ventricles", "median", "rows")
        == tmp_path / "figures" / "roi" / "single" / "median" / "ventricles__rows"
    )


def test_save_figure_writes_all_formats_and_closes(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    stem = tmp_path / "nested" / "ventricles__rows"

    paths = save_figure(fig, stem, formats=("pdf", "png"), dpi=50)

    assert paths == [
        stem.with_name("ventricles__rows.pdf"),
        stem.with_name("ventricles__rows.png"),
    ]
    assert all(path.stat().st_size > 0 for path in paths)
    assert not plt.fignum_exists(fig.number)


def test_statistic_label_follows_the_signal():
    assert statistic_label("median", signal="ratio") == "Median signal ratio"
    assert statistic_label("mean", signal="delta_R1") == "Mean ΔR1 (1/s)"


def test_plot_time_mode_one_row_per_component():
    from gMRItensor.plotting.time_mode import plot_time_mode

    rng = np.random.default_rng(0)
    fig, axs = plot_time_mode(rng.random((4, 3)), [0, 6, 24, 48], page_width="single")

    assert axs.shape == (3,)
    assert [len(ax.lines) for ax in axs] == [1, 1, 1]
    np.testing.assert_array_equal(axs[0].lines[0].get_xdata(), [0, 6, 24, 48])
    assert axs[2].get_ylabel() == "Component 3"
    plt.close(fig)


def test_rows_layout_sd_bands_in_subject_columns_only(roi_tables):
    stats, summary, significance = roi_tables
    cohort = summarize_cohort_over_time(stats, facet="roi", value="median")

    fig, axs = plot_roi_evolution_rows(
        summary,
        stats,
        significance,
        rois=["ventricles"],
        statistic="median",
        cohort=cohort,
    )

    ribbon_fills = len(axs[0, 0].collections)
    assert ribbon_fills == 2  # one SEM ribbon per group, no bands
    for ax in axs[0, 1:]:
        assert len(ax.collections) == 3  # +/-1, 2, 3 SD
        assert len(ax.lines) == 6 + 1  # subject curves + cohort mean
    # The +/-3 SD band fits inside the shared row range.
    ventricles = cohort[cohort["roi"] == "ventricles"]
    low = (ventricles["mean"] - 3 * ventricles["std"]).min()
    high = (ventricles["mean"] + 3 * ventricles["std"]).max()
    ymin, ymax = axs[0, 1].get_ylim()
    assert ymin <= low and high <= ymax
    legend = [text.get_text() for text in fig.legends[0].get_texts()]
    assert legend == ["A", "B", SD_BAND_LABEL]
    plt.close(fig)


def test_rows_layout_without_cohort_has_no_bands(roi_tables):
    stats, summary, significance = roi_tables

    fig, axs = plot_roi_evolution_rows(
        summary,
        stats,
        significance,
        rois=["ventricles"],
        statistic="median",
    )

    assert all(len(ax.collections) == 0 for ax in axs[0, 1:])
    assert len(fig.legends[0].get_texts()) == 2
    plt.close(fig)


def test_rows_layout_names_subjects_outside_3_sd(roi_tables):
    stats, summary, significance = roi_tables
    cohort = summarize_cohort_over_time(stats, facet="roi", value="median")
    outside = pd.DataFrame(
        {
            "roi": ["ventricles", "ventricles", "thalamus"],
            "subject": ["s1", "s8", "s2"],
            "group": ["A", "B", "A"],
            "timepoint": [1, 2, 0],
            "median": [9.0, -9.0, 9.0],
            "z": [3.5, -3.5, 3.5],
        },
    )

    fig, axs = plot_roi_evolution_rows(
        summary,
        stats,
        significance,
        rois=["ventricles"],
        statistic="median",
        cohort=cohort,
        outside=outside,
    )

    names_a = {t.get_text(): t for t in axs[0, 1].texts}
    names_b = {t.get_text(): t for t in axs[0, 2].texts}
    assert set(names_a) == {"s1"}  # thalamus is not drawn
    assert set(names_b) == {"s8"}
    assert names_a["s1"].xy == (1, 9.0)
    assert names_a["s1"].get_va() == "bottom" and names_b["s8"].get_va() == "top"
    plt.close(fig)
