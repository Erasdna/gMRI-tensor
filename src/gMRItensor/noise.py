"""The noise level of ΔR1, and the cutoff below which a value is not noise.

ΔR1 of a voxel without tracer is noise around 0, so some values are
negative; values far below 0 are artifacts (misregistration, partial
volume, a bad baseline). `noise_scale` estimates the noise of one set of
voxels; `summarize_noise` combines the per-scan estimates into one
reference for the study; `reject_negative` marks the values below
`-k * sigma`. No image is judged against its own estimate: the per-scan
estimates only feed the reference and the QC z-scores.
"""
from collections.abc import Mapping
from typing import Literal
from typing import NamedTuple

import numpy as np
import pandas as pd
from gMRItensor.group_statistics import MAD_TO_SD

#: Fewer finite values than this give no estimate.
MIN_NOISE_VOXELS = 1000
#: Voxel sets with an estimate per scan.
NOISE_SETS = ("outside", "parenchyma", "csf")
COMPARTMENTS = ("parenchyma", "csf")
NoiseSource = Literal["auto", "outside", "inside"]
NOISE_SOURCES = ("auto", "outside", "inside")
SCAN_NOISE_COLUMNS = ["subject", "time_point", "set", "n", "mode", "sigma"]
# Outside and inside estimates further apart than this are warned about.
_DISAGREEMENT = 2.0


def half_sample_mode(values: np.ndarray) -> float:
    """The half-sample mode of Bickel & Frühwirth (2006): repeatedly keep
    the shortest interval holding half of the sorted values."""
    x = np.sort(np.asarray(values, dtype=float))
    while len(x) > 3:
        half = (len(x) + 1) // 2
        first, last = half - 1, len(x) - half + 1
        start = int(np.argmin(x[first:] - x[:last]))
        end = start + half
        x = x[start:end]
    return float(np.median(x))


def noise_scale(values: np.ndarray) -> tuple[float, float]:
    """`(mode, sigma)` of the noise of `values`.

    `sigma = 1.4826 * median(mode - x | x < mode)`: the half below the
    half-sample mode, mirrored. Positive values may be real tracer, so
    only the lower half is used, and anchoring at the mode instead of 0
    keeps both an overall shift and a uniform low uptake out of `sigma`.
    It errs high when the concentrations themselves vary a lot. NaN with
    fewer than `MIN_NOISE_VOXELS` finite values, and `sigma` NaN when no
    value lies below the mode.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_NOISE_VOXELS:
        return np.nan, np.nan
    mode = half_sample_mode(x)
    below = mode - x[x < mode]
    sigma = MAD_TO_SD * float(np.median(below)) if len(below) else np.nan
    return mode, sigma


class NoiseReference(NamedTuple):
    """`summarize_noise` output."""

    source: Literal["outside", "inside"]
    reason: str
    #: Reference noise SD per compartment (1/s); NaN = no cutoff there.
    sigma: dict[str, float]
    #: The per-scan estimates with `used_in_reference`, `mode_z`, `sigma_z`.
    table: pd.DataFrame
    warnings: list[str]


def _robust_z(values: pd.Series) -> pd.Series:
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return pd.Series(np.nan, index=values.index)
    scale = MAD_TO_SD * (finite - finite.median()).abs().median()
    if scale == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - finite.median()) / scale


def _usable(table: pd.DataFrame, set_name: str) -> pd.Series:
    return (table["set"] == set_name) & np.isfinite(table["sigma"])


def summarize_noise(
    scan_noise: pd.DataFrame,
    source: NoiseSource,
) -> NoiseReference:
    """One noise level for the study from per-scan estimates.

    `scan_noise` has `SCAN_NOISE_COLUMNS`, one row per scan and set
    (`outside` the brain mask, `parenchyma`, `csf`). With the `outside`
    source the reference is the median over scans of the outside estimates,
    used for both compartments (no tracer is expected there, so the
    estimates are not inflated); with `inside` it is the lower quartile
    over scans per compartment, which favours the scans least inflated by
    real concentration spread. `auto` uses `outside` when at least half
    the scans have an outside estimate.

    The returned table adds `used_in_reference` and the robust z-scores of
    the mode and of log sigma among the scans of the same set and time
    point, to flag shifted or unusually noisy scans. Raises `ValueError` for an explicit `outside`
    without a usable scan.
    """
    if source not in NOISE_SOURCES:
        raise ValueError(f"noise source must be one of {NOISE_SOURCES}")
    table = scan_noise.copy()
    n_scans = len(table[["subject", "time_point"]].drop_duplicates())
    n_outside = int(_usable(table, "outside").sum())
    if source == "outside" and n_outside == 0:
        raise ValueError(
            "--noise-source outside: no scan has enough valid voxels outside "
            "the mask (are the maps zero or skull-stripped there?); use "
            "--noise-source inside or --noise-sd",
        )
    used: Literal["outside", "inside"]
    if source == "auto":
        used = "outside" if n_outside >= n_scans / 2 else "inside"
        reason = (
            f"{n_outside} of {n_scans} scans have enough valid voxels outside "
            "the mask"
        )
    else:
        used = source
        reason = "requested"

    def lower_quartile(compartment: str) -> float:
        values = table.loc[_usable(table, compartment), "sigma"]
        return float(np.quantile(values, 0.25)) if len(values) else np.nan

    outside = table.loc[_usable(table, "outside"), "sigma"]
    outside_sigma = float(outside.median()) if len(outside) else np.nan
    inside_sigma = {c: lower_quartile(c) for c in COMPARTMENTS}
    if used == "outside":
        sigma = {c: outside_sigma for c in COMPARTMENTS}
        table["used_in_reference"] = _usable(table, "outside")
    else:
        sigma = inside_sigma
        table["used_in_reference"] = _usable(table, "parenchyma") | _usable(
            table,
            "csf",
        )

    warnings = [
        f"no {c} scan has a noise estimate: no cutoff for {c}"
        for c in COMPARTMENTS
        if not np.isfinite(sigma[c])
    ]
    ratio = outside_sigma / inside_sigma["parenchyma"]
    if np.isfinite(ratio) and not 1 / _DISAGREEMENT <= ratio <= _DISAGREEMENT:
        warnings.append(
            f"the noise outside the mask ({outside_sigma:.3g} 1/s) and in the "
            f"parenchyma ({inside_sigma['parenchyma']:.3g} 1/s) differ by more "
            f"than a factor {_DISAGREEMENT:g}: the reference may not represent "
            "the brain; compare the sets in noise.csv",
        )

    # Among the scans of one set and time point: inside the brain the mode
    # and the estimate follow the tracer, which differs between sessions.
    groups = table.groupby(["set", "time_point"], group_keys=False)
    table["mode_z"] = groups["mode"].apply(_robust_z)
    table["sigma_z"] = groups["sigma"].apply(lambda s: _robust_z(np.log(s)))
    return NoiseReference(used, reason, sigma, table, warnings)


def cutoffs_from_sigma(sigma: Mapping[str, float], k: float) -> dict[str, float]:
    """The cutoff per compartment, `-k * sigma` in 1/s (NaN: no cutoff)."""
    return {compartment: -k * value for compartment, value in sigma.items()}


def reject_negative(
    signal: np.ndarray,
    csf: np.ndarray,
    cutoffs: Mapping[str, float],
) -> np.ndarray:
    """Which values lie below their compartment's cutoff (`csf` marks the
    CSF voxels). A value equal to the cutoff is kept, NaN values are not
    rejected (they are invalid already), and a NaN cutoff rejects nothing."""
    threshold = np.where(csf, cutoffs["csf"], cutoffs["parenchyma"])
    with np.errstate(invalid="ignore"):
        return np.asarray(signal < threshold)
