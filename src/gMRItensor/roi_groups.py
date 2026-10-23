"""Named groups of FreeSurfer segmentation labels (ROI presets).

Parenchyma uses FreeSurfer LUT ids. The CSF atlas reuses those ids plus
`csf_offset`: CSF next to a cortical parcel is the parcel id + offset,
cerebellar CSF is the cerebellar ids + offset and the pontine cistern is
brainstem (16) + offset. Ventricle labels are CSF whether or not they carry
the offset, so `ventricles` holds both and no parenchyma preset contains
either. `parenchyma` is every label below the offset that is not CSF.
"""
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

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

_PARENCHYMA_PRESETS_NAMES: dict[str, str] = {
    "cortical_grey_matter": "Cortical GM",
    "subcortical_grey_matter": "Subcortical GM",
    "grey_matter": "GM",
    "white_matter": "WM",
    "cerebellum": "Cerebellum",
    "brainstem": "Brainstem",
    "basal_ganglia": "Basal ganglia",
    "hippocampus": "Hippocampus",
    "amygdala": "Amygdala",
    "thalamus": "Thalamus",
    "cingulate": "Cingulate",
    "parahippocampal": "Parahippocampal",
    "entorhinal": "Entorhinal",
    "limbic_system": "Limbic system",
    "parenchyma": "Parenchyma",
}

# CSF preset name -> parenchyma preset whose ids + offset it covers.
_CSF_COUNTERPARTS = {
    "cortical_grey_matter_csf": "cortical_grey_matter",
    "cerebellum_csf": "cerebellum",
    "brainstem_csf": "brainstem",
    "limbic_system_csf": "limbic_system",
}
_CSF_PRESETS_NAMES: dict[str, str] = {
    "cortical_grey_matter_csf": "Supratentorial CSF",
    "cerebellum_csf": "Infratentorial CSF",
    "brainstem_csf": "Pontine cistern",
    "limbic_system_csf": "Limbic system CSF",
    "ventricles": "Ventricles",
    "all_csf": "All CSF",
}


def _sorted_unique(ids: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(ids)))


#: FreeSurfer's own CSF label, outside the ventricles.
CSF_ID = 24


def get_roi_presets(csf_offset: int = 10000) -> dict[str, tuple[int, ...]]:
    """All ROI presets as `name -> sorted unique label ids`."""
    presets = {name: _sorted_unique(ids) for name, ids in _PARENCHYMA_PRESETS.items()}
    presets["parenchyma"] = tuple(
        i for i in range(1, csf_offset) if i not in (*VENTRICLE_IDS, CSF_ID)
    )
    presets["ventricles"] = _sorted_unique(
        VENTRICLE_IDS + tuple(i + csf_offset for i in VENTRICLE_IDS),
    )
    for csf_name, parenchyma_name in _CSF_COUNTERPARTS.items():
        presets[csf_name] = tuple(i + csf_offset for i in presets[parenchyma_name])
    presets["all_csf"] = _sorted_unique(
        [i for name in (*_CSF_COUNTERPARTS, "ventricles") for i in presets[name]],
    )
    return presets


def roi_display_name(roi: str) -> str:
    """Figure label for an ROI: the preset's short name, else `roi` itself."""
    return {**_PARENCHYMA_PRESETS_NAMES, **_CSF_PRESETS_NAMES}.get(roi, roi)


def is_csf(labels: np.ndarray, csf_offset: int = 10000) -> np.ndarray:
    """Which `labels` are CSF: a ventricle, FreeSurfer CSF (24), or any id
    carrying `csf_offset`. Everything else is parenchyma."""
    labels = np.asarray(labels)
    return (labels >= csf_offset) | np.isin(labels, (*VENTRICLE_IDS, CSF_ID))


#: A named, sorted list of label ids, as `overview_blocks` returns them.
Block = tuple[str, list[int]]


def _blocks(
    candidates: Sequence[tuple[str, Sequence[int]]],
    labels: set[int],
) -> list[Block]:
    """The `labels` in each candidate, in order, each label in its first
    candidate only; empty blocks are left out. Assigned labels are removed
    from `labels`."""
    blocks = []
    for name, ids in candidates:
        present = sorted(labels.intersection(ids))
        labels.difference_update(present)
        if present:
            blocks.append((name, present))
    return blocks


def _other(labels: set[int]) -> list[Block]:
    return [("Other", sorted(labels))] if labels else []


def overview_blocks(
    labels: np.ndarray,
    csf_offset: int = 10000,
) -> tuple[list[Block], list[Block]]:
    """The ROI overview's x-axis order: `(parenchyma, csf)` blocks of `labels`.

    Parenchyma: Subcortical GM, WM, Cerebellum, Brainstem, Other, Cortex lh
    (1000s), Cortex rh (2000s). CSF (`is_csf`): Ventricles + CSF (24),
    Infratentorial, Pons, Cortical CSF lh and rh (the cortical ranges +
    `csf_offset`), Other. Ids are sorted within a block; empty blocks are
    left out.
    """
    labels = np.unique(np.asarray(labels)).astype(int)
    csf_mask = is_csf(labels, csf_offset)
    presets = get_roi_presets(csf_offset)
    remaining = set(labels[~csf_mask].tolist())
    cortex = _blocks(
        [("Cortex lh", range(1000, 2000)), ("Cortex rh", range(2000, 3000))],
        remaining,
    )
    parenchyma = _blocks(
        [
            ("Subcortical GM", presets["subcortical_grey_matter"]),
            ("WM", presets["white_matter"]),
            ("Cerebellum", presets["cerebellum"]),
            ("Brainstem", presets["brainstem"]),
        ],
        remaining,
    )
    parenchyma += _other(remaining) + cortex

    remaining = set(labels[csf_mask].tolist())
    csf = _blocks(
        [
            ("Ventricles + CSF", (*presets["ventricles"], CSF_ID)),
            ("Infratentorial", presets["cerebellum_csf"]),
            ("Pons", presets["brainstem_csf"]),
            ("Cortical CSF lh", range(csf_offset + 1000, csf_offset + 2000)),
            ("Cortical CSF rh", range(csf_offset + 2000, csf_offset + 3000)),
        ],
        remaining,
    )
    return parenchyma, csf + _other(remaining)


def overview_regions(
    labels: np.ndarray,
    csf_offset: int = 10000,
) -> tuple[list[Block], list[Block]]:
    """The ROI overview's region columns: `(parenchyma, csf)` presets
    restricted to `labels`, empty ones left out. The cortex has no
    parenchyma column (the all-ROI figure shows it)."""
    presets = get_roi_presets(csf_offset)
    present = set(np.unique(np.asarray(labels)).astype(int).tolist())
    parenchyma = _blocks(
        [
            (roi_display_name(name), presets[name])
            for name in (
                "subcortical_grey_matter",
                "white_matter",
                "cerebellum",
                "brainstem",
            )
        ],
        set(present),
    )
    csf = _blocks(
        [
            (
                roi_display_name("cortical_grey_matter_csf"),
                presets["cortical_grey_matter_csf"],
            ),
            ("Ventricles + CSF (24)", (*presets["ventricles"], CSF_ID)),
            (roi_display_name("cerebellum_csf"), presets["cerebellum_csf"]),
            (roi_display_name("brainstem_csf"), presets["brainstem_csf"]),
        ],
        set(present),
    )
    return parenchyma, csf


def parse_region(spec: str, csf_offset: int = 10000) -> tuple[str, np.ndarray]:
    """A `--region` value: a preset name, or `name=ids` with ids such as
    `17,53,1001-1035` (comma-separated, `a-b` inclusive ranges)."""
    presets = get_roi_presets(csf_offset)
    if "=" not in spec:
        if spec not in presets:
            raise ValueError(
                f"Unknown region {spec!r}: use a preset ({sorted(presets)}) or "
                "name=ids",
            )
        return spec, np.asarray(presets[spec], dtype=np.int64)

    name, _, id_text = spec.partition("=")
    name = name.strip()
    if not name:
        raise ValueError(f"region {spec!r} has no name before '='")
    if "/" in name or "\\" in name or name.startswith("."):
        # The name becomes a figure file name.
        raise ValueError(
            f"region name {name!r} may not contain '/' or '\\' or start with '.'",
        )
    ids: list[int] = []
    try:
        for part in id_text.split(","):
            part = part.strip()
            if not part:
                continue
            low, dash, high = part.partition("-")
            if dash:
                first, last = int(low), int(high)
                if last < first:
                    raise ValueError
                ids.extend(range(first, last + 1))
            else:
                ids.append(int(part))
    except ValueError:
        raise ValueError(
            f"region {spec!r}: ids must be integers or a-b ranges",
        ) from None
    if not ids:
        raise ValueError(f"region {spec!r} has no label ids")
    return name, np.unique(np.asarray(ids, dtype=np.int64))


_ROI_COLUMNS = [
    "subject",
    "time_point",
    "roi",
    "roi_type",
    "n_voxels",
    "n_valid",
    "volume_mm3",
    "voxel_volume_mm3",
    "median",
    "mean",
]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    usable = np.isfinite(values) & (weights > 0)
    if not usable.any():
        return np.nan
    order = np.argsort(values[usable])
    cumulative = np.cumsum(weights[usable][order])
    index = np.searchsorted(cumulative, cumulative[-1] / 2)
    return float(values[usable][order][index])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    usable = np.isfinite(values) & (weights > 0)
    if not usable.any():
        return np.nan
    return float(np.sum(values[usable] * weights[usable]) / np.sum(weights[usable]))


def _combine_labels(scan: pd.DataFrame) -> pd.Series:
    """One region's row for one scan, from its labels' `roi_signal` rows."""
    n_valid = scan["n_valid"].to_numpy()
    voxel_volume = float(scan["voxel_volume_mm3"].iloc[0])
    return pd.Series(
        {
            "n_voxels": int(scan["n_voxels"].sum()),
            "n_valid": int(n_valid.sum()),
            "volume_mm3": float(scan["n_voxels"].sum()) * voxel_volume,
            "voxel_volume_mm3": voxel_volume,
            "median": _weighted_median(scan["median"].to_numpy(), n_valid),
            "mean": _weighted_mean(scan["mean"].to_numpy(), n_valid),
        },
    )


def aggregate_roi_signal(
    roi_signal: pd.DataFrame,
    regions: Mapping[str, Sequence[int] | np.ndarray] | None = None,
    labels: Sequence[int] | Literal["all"] | None = None,
    max_invalid_voxel_fraction: float | None = None,
) -> pd.DataFrame:
    """Per-scan ROI rows from `write_preprocessed_data`'s `roi_signal` table.

    `labels` ("all" or ids) become `roi_type="label"` rows named `str(id)`,
    copied from their label rows. Each `regions` entry becomes a
    `roi_type="group"` row combining its labels per scan: counts and
    `volume_mm3` are sums and `mean` is the `n_valid`-weighted mean, all
    exact; `median` is the `n_valid`-weighted median of the label medians,
    an approximation of the median of the pooled voxels (exact for a single
    label).

    With `max_invalid_voxel_fraction` (the value `gmri preprocess
    --negative-cutoff` applied to the labels, from the file's metadata), a
    region with a larger share of invalid voxels, 1 - sum(n_valid) /
    sum(n_voxels), gets NaN `median`/`mean` and `n_valid` 0 as an invalid
    label does, so it does not silently become the median of its remaining
    labels.

    Raises `ValueError` for requested labels, or regions with no labels, not
    in the table.
    """
    frames = []
    present = set(roi_signal["label"])
    if labels is not None:
        if isinstance(labels, str):
            selected = roi_signal
        else:
            missing = sorted(set(labels) - present)
            if missing:
                raise ValueError(f"label(s) not in the data: {missing}")
            selected = roi_signal[roi_signal["label"].isin(list(labels))]
        frames.append(
            selected.assign(
                roi=selected["label"].astype(str),
                roi_type="label",
                volume_mm3=selected["n_voxels"] * selected["voxel_volume_mm3"],
            )[_ROI_COLUMNS],
        )

    for name, ids in (regions or {}).items():
        member = roi_signal[roi_signal["label"].isin(list(ids))]
        if member.empty:
            raise ValueError(f"region {name!r}: none of its labels are in the data")
        combined = (
            member.groupby(["subject", "time_point"])
            .apply(_combine_labels, include_groups=False)
            .reset_index()
        )
        combined["roi"] = name
        combined["roi_type"] = "group"
        if max_invalid_voxel_fraction is not None:
            share = 1 - combined["n_valid"] / combined["n_voxels"]
            invalid = share > max_invalid_voxel_fraction
            combined.loc[invalid, ["median", "mean"]] = np.nan
            combined.loc[invalid, "n_valid"] = 0
        frames.append(combined[_ROI_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=_ROI_COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    return frame.astype({"n_voxels": np.int64, "n_valid": np.int64})


# mM * mm^3 = (1e-3 mol / 1e-3 m^3) * 1e-9 m^3 = 1e-9 mol = 1e-6 mmol.
_MM_MM3_TO_MMOL = 1e-6


def add_concentration(frame: pd.DataFrame, relaxivity: float) -> pd.DataFrame:
    """Add concentration columns to `aggregate_roi_signal` output (ΔR1 in 1/s).

    `median_concentration` / `mean_concentration` = ΔR1 / `relaxivity`
    [mM], exact for both because the conversion is linear; `total_amount`
    = mean concentration x the full ROI volume `volume_mm3` [mmol], so
    invalid voxels count at the ROI's mean concentration. `relaxivity` is r1 in
    1/(mM s); one value is applied to every region, so parenchymal
    concentrations are approximate.
    """
    if not relaxivity > 0:
        raise ValueError(f"relaxivity must be > 0, got {relaxivity}")
    out = frame.copy()
    out["median_concentration"] = out["median"] / relaxivity
    out["mean_concentration"] = out["mean"] / relaxivity
    out["total_amount"] = (
        out["mean_concentration"] * out["volume_mm3"] * _MM_MM3_TO_MMOL
    )
    return out
