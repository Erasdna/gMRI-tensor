"""Named groups of FreeSurfer segmentation labels (ROI presets).

Parenchyma uses FreeSurfer LUT ids. The CSF atlas reuses those ids plus
`csf_offset`: CSF next to a cortical parcel is the parcel id + offset,
cerebellar CSF is the cerebellar ids + offset and the pontine cistern is
brainstem (16) + offset. Ventricle labels are CSF whether or not they carry
the offset, so `ventricles` holds both and no parenchyma preset contains
either.
"""
from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np

VENTRICLE_IDS = (4, 5, 14, 15, 43, 44, 72)

_CORTICAL_GREY_MATTER = (3, 42, *range(1001, 1036), *range(2001, 2036))
_SUBCORTICAL_GREY_MATTER = (10, 11, 12, 13, 17, 18, 26, *range(49, 55), 58)
_HIPPOCAMPUS = (17, 53)
_AMYGDALA = (18, 54)
_CINGULATE = (1002, 1010, 1023, 1026, 2002, 2010, 2023, 2026)
_PARAHIPPOCAMPAL = (1016, 2016)
_ENTORHINAL = (1006, 2006)

_PARENCHYMA_PRESETS: dict[str, tuple[int, ...]] = {
    "cortical_grey_matter": _CORTICAL_GREY_MATTER,
    "subcortical_grey_matter": _SUBCORTICAL_GREY_MATTER,
    "grey_matter": _CORTICAL_GREY_MATTER + _SUBCORTICAL_GREY_MATTER,
    "white_matter": (2, 41, 77, *range(251, 256)),
    "cerebellum": (7, 8, 46, 47),
    "brainstem": (16,),
    "basal_ganglia": (11, 12, 13, 26, 50, 51, 52, 58),
    "hippocampus": _HIPPOCAMPUS,
    "amygdala": _AMYGDALA,
    "thalamus": (10, 49),
    "cingulate": _CINGULATE,
    "parahippocampal": _PARAHIPPOCAMPAL,
    "entorhinal": _ENTORHINAL,
    "limbic_system": (
        _HIPPOCAMPUS + _AMYGDALA + _CINGULATE + _PARAHIPPOCAMPAL + _ENTORHINAL
    ),
}

# CSF preset name -> parenchyma preset whose ids + offset it covers.
_CSF_COUNTERPARTS = {
    "cortical_grey_matter_csf": "cortical_grey_matter",
    "cerebellum_csf": "cerebellum",
    "brainstem_csf": "brainstem",
    "limbic_system_csf": "limbic_system",
}


def _sorted_unique(ids: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(ids)))


def get_roi_presets(csf_offset: int = 10000) -> dict[str, tuple[int, ...]]:
    """All ROI presets as `name -> sorted unique label ids`."""
    presets = {name: _sorted_unique(ids) for name, ids in _PARENCHYMA_PRESETS.items()}
    presets["ventricles"] = _sorted_unique(
        VENTRICLE_IDS + tuple(i + csf_offset for i in VENTRICLE_IDS),
    )
    for csf_name, parenchyma_name in _CSF_COUNTERPARTS.items():
        presets[csf_name] = tuple(i + csf_offset for i in presets[parenchyma_name])
    presets["all_csf"] = _sorted_unique(
        [i for name in (*_CSF_COUNTERPARTS, "ventricles") for i in presets[name]],
    )
    return presets


def resolve_roi_groups(
    presets: Sequence[str] = (),
    custom: Mapping[str, Sequence[int]] | None = None,
    csf_offset: int = 10000,
) -> dict[str, np.ndarray]:
    """Build `name -> int64 label ids` from preset names and custom regions.

    Presets come first, then custom regions, each in the given order.
    Raises `ValueError` for an unknown preset, a name given more than once
    (also across presets and custom) or a custom region without ids.
    """
    available = get_roi_presets(csf_offset)
    groups: dict[str, np.ndarray] = {}

    def add(name: str, ids: Sequence[int]) -> None:
        if name in groups:
            raise ValueError(f"ROI group '{name}' given more than once")
        if len(ids) == 0:
            raise ValueError(f"ROI group '{name}' has no label ids")
        groups[name] = np.unique(np.asarray(ids, dtype=np.int64))

    for name in presets:
        if name not in available:
            raise ValueError(
                f"Unknown ROI preset '{name}', choose from {sorted(available)}",
            )
        add(name, available[name])
    for name, ids in (custom or {}).items():
        add(name, ids)
    return groups
