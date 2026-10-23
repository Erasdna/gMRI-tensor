from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from gMRItensor.group_statistics import flag_roi_values
from gMRItensor.plotting.roi_overview import plot_region_columns
from gMRItensor.plotting.roi_overview import plot_roi_overview
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import save_figure
from gMRItensor.roi_groups import overview_blocks
from gMRItensor.roi_groups import overview_regions
from matplotlib.colors import to_rgba

matplotlib.use("Agg")

LABELS = (10, 49, 2, 41, 16, 1001, 1002, 2001, 4, 43, 10016, 11001)


def make_label_values(n_per_group: int = 5) -> pd.DataFrame:
    """Per-label rows of `median` for groups A and B at 4 h and 24 h; the
    24 h values are 10x larger, s1 has no 24 h scan, and s3 is far off in
    label 41 at 24 h (the one flagged value)."""
    rows = []
    for i in range(2 * n_per_group):
        for t, scale in ((4, 1.0), (24, 10.0)):
            if (i, t) == (1, 24):
                continue
            for label in LABELS:
                value = scale * (1.0 + 0.01 * ((3 * i + label) % (2 * n_per_group)))
                if (i, t, label) == (3, 24, 41):
                    value = 100.0
                rows.append(
                    {
                        "subject": f"s{i}",
                        "group": "AB"[i // n_per_group],
                        "timepoint": t,
                        "label": label,
                        "median": value,
                    },
                )
    return pd.DataFrame(rows)


def _ring_offsets(fig: matplotlib.figure.Figure) -> np.ndarray:
    rings = [
        collection.get_offsets()
        for ax in fig.axes
        for collection in ax.collections
        if collection.get_gid() == "flags"
    ]
    return np.concatenate(rings) if rings else np.empty((0, 2))


def test_overview_rows_are_sessions_with_their_own_y_scale() -> None:
    values = make_label_values()
    parenchyma, _ = overview_blocks(np.array(LABELS))

    fig, axs = plot_roi_overview(
        values,
        parenchyma,
        "median",
        flag_roi_values(values, "median"),
        "Median",
    )

    assert axs.shape == (2, 1)
    # Each row fits its own values (and the zero line).
    high_4h, high_24h = (ax.get_ylim()[1] for ax in axs[:, 0])
    assert high_4h < 2
    assert high_24h > 100


def test_overview_rings_every_flag_with_its_name_in_block_order() -> None:
    values = make_label_values()
    flags = flag_roi_values(values, "median")
    parenchyma, _ = overview_blocks(np.array(LABELS))
    order = [label for _, ids in parenchyma for label in ids]

    fig, axs = plot_roi_overview(values, parenchyma, "median", flags, "Median")

    assert len(flags) == 1
    rings = _ring_offsets(fig)
    assert len(rings) == len(flags)
    assert rings[0, 1] == 100.0
    assert order.index(41) - 0.5 < rings[0, 0] < order.index(41) + 0.5
    assert "s3" in [text.get_text() for text in axs[1, 0].texts]
    ticks = [tick.get_text() for tick in axs[-1, 0].get_xticklabels()]
    assert ticks == [str(label) for label in order]
    assert order[:4] == [10, 49, 2, 41]  # Subcortical GM, then WM


def test_overview_colours_groups_with_the_palette() -> None:
    values = make_label_values()
    parenchyma, _ = overview_blocks(np.array(LABELS))

    fig, axs = plot_roi_overview(
        values,
        parenchyma,
        "median",
        flag_roi_values(values, "median"),
        "Median",
    )

    colours = [
        tuple(collection.get_facecolor()[0][:3])
        for collection in axs[0, 0].collections
        if collection.get_gid() != "flags"
    ]
    assert colours == [to_rgba(c)[:3] for c in get_color_palette(2)]


def test_region_columns_one_column_per_region_sized_by_label_count() -> None:
    values = make_label_values()
    flags = flag_roi_values(values, "median")
    parenchyma, csf = overview_regions(np.array(LABELS))

    fig, axs = plot_region_columns(values, parenchyma, "median", flags, "Median")

    assert [name for name, _ in parenchyma] == ["Subcortical GM", "WM", "Brainstem"]
    assert axs.shape == (2, 3)
    assert [ax.get_title() for ax in axs[0]] == ["Subcortical GM", "WM", "Brainstem"]
    # Width per label, at least 3 so a single-label region stays readable;
    # the last column holds the row notes.
    ratios = axs[0, 0].get_gridspec().get_width_ratios()
    assert ratios[:3] == [3, 3, 3]
    assert len(_ring_offsets(fig)) == 1  # label 41 is in WM
    csf_fig, csf_axs = plot_region_columns(values, csf, "median", flags, "Median")
    assert len(_ring_offsets(csf_fig)) == 0
    assert csf_axs.shape == (2, len(csf))


def test_overview_figures_save_with_a_missing_session(tmp_path: Path) -> None:
    values = make_label_values()
    values = values[~((values["timepoint"] == 24) & (values["label"] >= 10000))]
    flags = flag_roi_values(values, "median")
    _, csf = overview_blocks(np.array(LABELS))
    _, csf_regions = overview_regions(np.array(LABELS))

    fig, axs = plot_roi_overview(
        values,
        csf,
        "median",
        flags,
        "Median",
        timepoints=[4, 24, 48],
    )
    region_fig, _ = plot_region_columns(values, csf_regions, "median", flags, "Median")

    assert axs.shape == (3, 1)
    assert save_figure(fig, tmp_path / "csf", ("pdf",))[0].stat().st_size > 0
    assert save_figure(region_fig, tmp_path / "regions", ("pdf",))[0].exists()
