from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from gMRItensor import noise
from gMRItensor.noise import cutoffs_from_sigma
from gMRItensor.noise import half_sample_mode
from gMRItensor.noise import noise_scale
from gMRItensor.noise import reject_negative
from gMRItensor.noise import summarize_noise


def _sample(mu: np.ndarray, seed: int = 0) -> np.ndarray:
    """`mu` plus N(0, 1) noise, with 1% artifacts far below zero."""
    rng = np.random.default_rng(seed)
    x = mu + rng.standard_normal(len(mu))
    artifacts = rng.random(len(mu)) < 0.01
    x[artifacts] = -rng.uniform(5, 20, artifacts.sum())
    return x


N = 20_000
RNG = np.random.default_rng(1)


@pytest.mark.parametrize(
    "mu",
    [
        np.zeros(N),  # no tracer
        np.full(N, 1.12),  # uniform low uptake
        np.where(RNG.random(N) < 0.5, 0.0, RNG.exponential(2.0, N)),  # partial
    ],
    ids=["none", "uniform-low", "partial"],
)
def test_noise_scale_recovers_the_noise_sd(mu: np.ndarray) -> None:
    _, sigma = noise_scale(_sample(mu))

    assert sigma == pytest.approx(1.0, rel=0.1)


def test_noise_scale_ignores_a_global_shift() -> None:
    x = _sample(np.zeros(N))

    mode, sigma = noise_scale(x - 3.0)

    assert mode == pytest.approx(-3.0, abs=0.15)
    assert sigma == pytest.approx(noise_scale(x)[1], rel=0.02)


def test_noise_scale_needs_enough_values(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.r_[_sample(np.zeros(500)), np.nan]

    assert np.isnan(noise_scale(x)).all()  # 500 < MIN_NOISE_VOXELS
    monkeypatch.setattr(noise, "MIN_NOISE_VOXELS", 100)
    assert np.isfinite(noise_scale(x)).all()
    # No value below the mode (every value tied): no estimate.
    assert np.isnan(noise_scale(np.zeros(2000))[1])


def test_half_sample_mode_finds_the_densest_point() -> None:
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(5.0, 0.1, 3000), rng.uniform(-50, 50, 2000)]

    assert half_sample_mode(x) == pytest.approx(5.0, abs=0.05)


def _scan_noise(rows: Sequence[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["subject", "time_point", "set", "n", "mode", "sigma"],
    )


def _study(outside: list[float], parenchyma: list[float]) -> pd.DataFrame:
    rows = []
    for i, (out, par) in enumerate(zip(outside, parenchyma)):
        rows.append((f"s{i}", 4, "outside", 5000, 0.0, out))
        rows.append((f"s{i}", 4, "parenchyma", 9000, 0.02, par))
        rows.append((f"s{i}", 4, "csf", 3000, 0.4, 3 * par))
    return _scan_noise(rows)


def test_summarize_noise_outside_uses_the_median_for_both_compartments() -> None:
    table = _study([0.01, 0.012, 0.011, np.nan], [0.012, 0.013, 0.02, 0.011])

    reference = summarize_noise(table, "auto")

    assert reference.source == "outside"
    assert reference.sigma == {"parenchyma": 0.011, "csf": 0.011}
    used = reference.table.set_index(["subject", "set"])["used_in_reference"]
    assert used[("s0", "outside")] and not used[("s3", "outside")]
    assert not used[("s0", "parenchyma")]


def test_summarize_noise_inside_uses_the_lower_quartile_per_compartment() -> None:
    table = _study([np.nan] * 4, [0.010, 0.012, 0.014, 0.030])

    reference = summarize_noise(table, "auto")

    assert reference.source == "inside"
    assert "outside" in reference.reason
    assert reference.sigma["parenchyma"] == pytest.approx(
        np.quantile([0.010, 0.012, 0.014, 0.030], 0.25),
    )
    assert reference.sigma["csf"] == pytest.approx(
        3 * np.quantile([0.010, 0.012, 0.014, 0.030], 0.25),
    )


def test_summarize_noise_explicit_outside_needs_usable_scans() -> None:
    with pytest.raises(ValueError, match="no scan has"):
        summarize_noise(_study([np.nan] * 3, [0.01] * 3), "outside")


def test_summarize_noise_compartment_without_usable_scans_has_no_cutoff() -> None:
    table = _study([np.nan] * 3, [0.01] * 3)
    table.loc[table["set"] == "csf", "sigma"] = np.nan

    reference = summarize_noise(table, "inside")

    assert np.isnan(reference.sigma["csf"])
    assert any("csf" in warning for warning in reference.warnings)


def test_summarize_noise_warns_when_outside_and_inside_disagree() -> None:
    agree = summarize_noise(_study([0.01] * 4, [0.012] * 4), "auto")
    differ = summarize_noise(_study([0.01] * 4, [0.03] * 4), "auto")

    assert not agree.warnings
    assert any("factor" in warning for warning in differ.warnings)


def test_summarize_noise_z_scores_flag_a_shifted_scan() -> None:
    table = _study([0.010, 0.011, 0.012, 0.010, 0.011], [0.012] * 5)
    outside = table["set"] == "outside"
    # s4 is shifted far below the others' spread of about 0.001.
    table.loc[outside, "mode"] = [0.001, -0.001, 0.002, 0.0, -0.03]

    z = summarize_noise(table, "auto").table.set_index(["subject", "set"])

    assert z.loc[("s4", "outside"), "mode_z"] < -3.5
    assert abs(z.loc[("s0", "outside"), "mode_z"]) < 3.5
    assert np.isfinite(z.loc[("s2", "outside"), "sigma_z"])


def test_cutoffs_and_rejection_per_compartment() -> None:
    cutoffs = cutoffs_from_sigma({"parenchyma": 0.01, "csf": np.nan}, k=4)
    signal = np.array([-0.05, -0.04, -0.03, np.nan, -0.5, 0.2])
    csf = np.array([False, False, False, False, True, False])

    rejected = reject_negative(signal, csf, cutoffs)

    assert cutoffs["parenchyma"] == pytest.approx(-0.04)
    # Below the cutoff only; equal is kept; NaN stays NaN (not rejected);
    # a compartment without a cutoff keeps everything.
    assert list(rejected) == [True, False, False, False, False, False]
