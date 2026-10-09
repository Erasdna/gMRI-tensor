import numpy as np
import pandas as pd
import pytest
from gMRItensor.roi_groups import add_concentration
from gMRItensor.roi_groups import aggregate_roi_signal
from gMRItensor.roi_groups import get_roi_presets
from gMRItensor.roi_groups import parse_region
from gMRItensor.roi_groups import resolve_roi_groups

VENTRICLE_IDS = (4, 5, 14, 15, 43, 44, 72)
PARENCHYMA_PRESETS = (
    "cortical_grey_matter",
    "subcortical_grey_matter",
    "grey_matter",
    "white_matter",
    "cerebellum",
    "brainstem",
    "basal_ganglia",
    "hippocampus",
    "amygdala",
    "thalamus",
    "cingulate",
    "parahippocampal",
    "entorhinal",
    "limbic_system",
)
CSF_PRESETS = {
    "cortical_grey_matter_csf": "cortical_grey_matter",
    "cerebellum_csf": "cerebellum",
    "brainstem_csf": "brainstem",
    "limbic_system_csf": "limbic_system",
}


def test_presets_are_sorted_and_unique():
    for name, ids in get_roi_presets().items():
        assert list(ids) == sorted(set(ids)), name
        assert len(ids) > 0, name


def test_known_preset_contents():
    presets = get_roi_presets()
    assert presets["brainstem"] == (16,)
    assert presets["thalamus"] == (10, 49)
    assert presets["cerebellum"] == (7, 8, 46, 47)
    assert presets["white_matter"] == (2, 41, 77, 251, 252, 253, 254, 255)
    cortical = presets["cortical_grey_matter"]
    assert {3, 42, 1001, 1035, 2001, 2035} <= set(cortical)
    assert len(cortical) == 2 + 2 * 35


def test_unions():
    presets = get_roi_presets()
    assert set(presets["grey_matter"]) == set(presets["cortical_grey_matter"]) | set(
        presets["subcortical_grey_matter"],
    )
    limbic = set(presets["limbic_system"])
    for part in ("hippocampus", "amygdala", "cingulate", "parahippocampal"):
        assert set(presets[part]) <= limbic
    assert set(presets["entorhinal"]) <= limbic
    assert not set(presets["thalamus"]) & limbic


def test_ventricles_include_bare_and_offset_ids():
    presets = get_roi_presets(csf_offset=10000)
    expected = set(VENTRICLE_IDS) | {i + 10000 for i in VENTRICLE_IDS}
    assert set(presets["ventricles"]) == expected
    assert "ventricles_csf" not in presets


@pytest.mark.parametrize("csf_offset", [10000, 20000])
def test_csf_presets_are_parenchyma_ids_plus_offset(csf_offset):
    presets = get_roi_presets(csf_offset=csf_offset)
    for csf_name, parenchyma_name in CSF_PRESETS.items():
        expected = tuple(i + csf_offset for i in presets[parenchyma_name])
        assert presets[csf_name] == expected
    assert presets["brainstem_csf"] == (16 + csf_offset,)


def test_all_csf_is_union_of_csf_presets_and_ventricles():
    presets = get_roi_presets()
    expected = set(presets["ventricles"])
    for csf_name in CSF_PRESETS:
        expected |= set(presets[csf_name])
    assert set(presets["all_csf"]) == expected


def test_no_ventricle_id_in_parenchyma_presets():
    presets = get_roi_presets()
    ventricle_ids = set(presets["ventricles"])
    for name in PARENCHYMA_PRESETS:
        assert not set(presets[name]) & ventricle_ids, name


def test_resolve_roi_groups_presets_and_custom():
    groups = resolve_roi_groups(
        presets=["thalamus", "brainstem_csf"],
        custom={"my_region": [17, 10, 17]},
    )
    assert list(groups) == ["thalamus", "brainstem_csf", "my_region"]
    np.testing.assert_array_equal(groups["thalamus"], [10, 49])
    np.testing.assert_array_equal(groups["brainstem_csf"], [10016])
    np.testing.assert_array_equal(groups["my_region"], [10, 17])
    assert all(ids.dtype == np.int64 for ids in groups.values())


def test_resolve_roi_groups_empty_by_default():
    assert resolve_roi_groups() == {}


def test_resolve_roi_groups_passes_csf_offset():
    groups = resolve_roi_groups(presets=["brainstem_csf"], csf_offset=5000)
    np.testing.assert_array_equal(groups["brainstem_csf"], [5016])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"presets": ["not_a_preset"]}, "Unknown ROI preset"),
        ({"presets": ["thalamus", "thalamus"]}, "more than once"),
        (
            {"presets": ["thalamus"], "custom": {"thalamus": [10]}},
            "more than once",
        ),
        ({"custom": {"empty": []}}, "no label ids"),
    ],
)
def test_resolve_roi_groups_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        resolve_roi_groups(**kwargs)


def _roi_signal() -> pd.DataFrame:
    """Two scans; labels 1 and 2 (label 3 only in the first, all-NaN)."""
    return pd.DataFrame(
        {
            "subject": ["s0", "s0", "s0", "s1", "s1"],
            "time_point": [0, 0, 0, 0, 0],
            "label": [1, 2, 3, 1, 2],
            "median": [1.0, 10.0, np.nan, 2.0, 4.0],
            "mean": [1.5, 12.0, np.nan, 2.0, 5.0],
            "n_voxels": [4, 2, 3, 2, 6],
            "n_valid": [3, 1, 0, 2, 6],
            "voxel_volume_mm3": [2.0, 2.0, 2.0, 1.0, 1.0],
        },
    )


def test_parse_region_presets_ids_and_ranges() -> None:
    name, ids = parse_region("thalamus")
    assert name == "thalamus"
    np.testing.assert_array_equal(ids, [10, 49])

    name, ids = parse_region("mine=17, 53,1001-1003")
    assert name == "mine"
    np.testing.assert_array_equal(ids, [17, 53, 1001, 1002, 1003])

    _, ids = parse_region("brainstem_csf", csf_offset=5000)
    np.testing.assert_array_equal(ids, [5016])


@pytest.mark.parametrize("spec", ["nope", "=17", "x=", "x=a,b", "x=5-2"])
def test_parse_region_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="region"):
        parse_region(spec)


def test_aggregate_roi_signal_groups_are_exact_sums_and_weighted_means() -> None:
    frame = aggregate_roi_signal(_roi_signal(), {"g": np.array([1, 2, 3])})

    s0 = frame.set_index("subject").loc["s0"]
    assert s0["roi"] == "g" and s0["roi_type"] == "group"
    assert (s0["n_voxels"], s0["n_valid"]) == (9, 4)
    assert s0["volume_mm3"] == pytest.approx(18.0)
    # Means weighted by n_valid: (1.5 * 3 + 12 * 1) / 4.
    assert s0["mean"] == pytest.approx(16.5 / 4)
    # Weighted median of label medians: weights 3 (median 1) vs 1 -> 1.
    assert s0["median"] == pytest.approx(1.0)
    s1 = frame.set_index("subject").loc["s1"]
    # Weights 2 (median 2) and 6 (median 4) -> 4.
    assert s1["median"] == pytest.approx(4.0)
    assert s1["mean"] == pytest.approx((2.0 * 2 + 5.0 * 6) / 8)


def test_aggregate_roi_signal_single_labels_match_label_rows() -> None:
    roi_signal = _roi_signal()

    frame = aggregate_roi_signal(roi_signal, labels="all")

    assert set(frame["roi"]) == {"1", "2", "3"}
    assert (frame["roi_type"] == "label").all()
    one = frame[(frame["roi"] == "2") & (frame["subject"] == "s1")].iloc[0]
    assert (one["median"], one["mean"], one["n_valid"]) == (4.0, 5.0, 6)
    assert one["volume_mm3"] == pytest.approx(6.0)
    only = aggregate_roi_signal(roi_signal, labels=[2])
    assert set(only["roi"]) == {"2"}


def test_aggregate_roi_signal_rejects_unknown_labels_and_empty_regions() -> None:
    with pytest.raises(ValueError, match="99"):
        aggregate_roi_signal(_roi_signal(), labels=[99])
    with pytest.raises(ValueError, match="region 'empty'"):
        aggregate_roi_signal(_roi_signal(), {"empty": np.array([99])})


def test_add_concentration() -> None:
    frame = aggregate_roi_signal(_roi_signal(), labels=[1])

    out = add_concentration(frame, relaxivity=2.0)

    s0 = out.set_index("subject").loc["s0"]
    assert s0["median_concentration"] == pytest.approx(0.5)  # 1 1/s / 2 1/(mM s)
    assert s0["mean_concentration"] == pytest.approx(0.75)
    # mM * n_valid * mm^3 * 1e-6 = mmol.
    assert s0["total_amount"] == pytest.approx(0.75 * 3 * 2.0 * 1e-6)
    with pytest.raises(ValueError, match="relaxivity"):
        add_concentration(frame, relaxivity=0.0)
