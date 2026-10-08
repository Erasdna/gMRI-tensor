import numpy as np
import pytest
from gMRItensor.roi_groups import get_roi_presets
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
