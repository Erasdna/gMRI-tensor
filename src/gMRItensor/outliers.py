"""Subject-mode outlier diagnostics: leverage and per-subject residual.

Following Fida et al. (2022, Sec. 3.3): a subject is influential if its
subject-mode leverage is high and poorly described if its residual is high;
each gets a Bonferroni-corrected cut-off. Leverage and its cut-off, and
the paper's residual cut-off, come from `tlviz.outliers`. The residual
itself is computed here because tlviz has no mask and assumes equally
sized slabs, while CP slices carry NaN for missing sessions and
PARAFAC2/CMF slices differ in length; a robust residual cut-off is added
because tlviz's is masked by the very outlier it should find.
"""
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from gMRItensor.model_io import SavedDecomposition
from scipy.stats import chi2
from tlviz.outliers import compute_leverage
from tlviz.outliers import get_leverage_outlier_threshold
from tlviz.outliers import get_slabwise_sse_outlier_threshold


def reconstruct_slices(saved: SavedDecomposition) -> list[np.ndarray]:
    """Each subject's model estimate `X̂_i`, `(n_timepoints_i, n_labels)`.

    CP: `B diag(w * a_i) Cᵀ`; PARAFAC2: `B_i diag(w * a_i) Cᵀ`; CMF:
    `B_i diag(w) Cᵀ` (its `subject_mode` is derived from the `B_i`, not a
    factor of the model).
    """
    weights = np.asarray(saved.weights, dtype=float)
    subject_mode = np.asarray(saved.subject_mode, dtype=float)
    label_mode = np.asarray(saved.label_mode, dtype=float)
    if saved.method == "cp":
        assert saved.time_mode is not None
        time_mode = np.asarray(saved.time_mode, dtype=float)
        return [(time_mode * (weights * a)) @ label_mode.T for a in subject_mode]
    states = [np.asarray(B, dtype=float) for B in saved.evolving_states or []]
    if saved.method == "parafac2":
        return [
            (B * (weights * a)) @ label_mode.T for B, a in zip(states, subject_mode)
        ]
    return [(B * weights) @ label_mode.T for B in states]


def data_slices(data: torch.Tensor | Sequence[torch.Tensor]) -> list[np.ndarray]:
    """`DecompositionInput.data` as one numpy array per subject."""
    if isinstance(data, torch.Tensor):
        return [s.numpy() for s in data.detach().cpu()]
    return [s.detach().cpu().numpy() for s in data]


def subject_residuals(
    observed: Sequence[np.ndarray],
    estimated: Sequence[np.ndarray],
) -> np.ndarray:
    """Mean squared residual per subject over its observed (non-NaN)
    entries, normalised to sum to 1.

    The mean, not the sum, so subjects with fewer observed entries (missing
    sessions, shorter slices) are not scored as better fitted.
    """
    mse = np.array(
        [
            np.nanmean((np.asarray(x, dtype=float) - x_hat) ** 2)
            for x, x_hat in zip(observed, estimated)
        ],
    )
    total = mse.sum()
    return mse / total if total > 0 else mse


def _leverage_cutoff(leverage: np.ndarray, rank: int, p_value: float) -> float | None:
    """Belsley's F-based cut-off, or Hotelling's T² at rank 1 (the former
    needs rank > 1); None if there are too few subjects for either."""
    n = len(leverage)
    if rank > 1 and n > rank:
        method = "p-value"
    elif n > rank + 1:
        method = "hotelling"
    else:
        return None
    return float(
        get_leverage_outlier_threshold(leverage, method=method, p_value=p_value),
    )


def _residual_cutoff(residual: np.ndarray, p_value: float) -> float | None:
    """tlviz's scaled-χ² cut-off (Box 1954), its moments from all subjects;
    None without spread. One strongly outlying subject inflates the variance
    it is judged against, so it can hide itself (see `_robust_residual_cutoff`).
    """
    if np.std(residual) == 0:
        return None
    return float(
        get_slabwise_sse_outlier_threshold(residual, method="p-value", p_value=p_value),
    )


def _robust_residual_cutoff(residual: np.ndarray, p_value: float) -> float | None:
    """The same scaled χ² `g·χ²(h)` with `g = σ²/(2μ)`, `h = μ/g`, but `μ`
    and `σ` from the median and `1.4826·MAD`, so a few outlying subjects do
    not move the cut-off; None without spread."""
    centre = float(np.median(residual))
    spread = 1.4826 * float(np.median(np.abs(residual - centre)))
    if spread == 0 or centre <= 0:
        return None
    scale = spread**2 / (2 * centre)
    return float(chi2.isf(p_value, centre / scale) * scale)


class OutlierCutoffs(NamedTuple):
    """Cut-offs of `outlier_table`; None where one cannot be computed."""

    leverage: float | None
    residual: float | None
    robust_residual: float | None


def _above(values: np.ndarray, cutoff: float | None) -> np.ndarray:
    if cutoff is None:
        return np.zeros(len(values), dtype=bool)
    return values > cutoff


def outlier_table(
    saved: SavedDecomposition,
    observed: Sequence[np.ndarray],
    alpha: float = 0.01,
) -> tuple[pd.DataFrame, OutlierCutoffs]:
    """Leverage and residual per subject, with their cut-offs.

    Cut-offs use `p = alpha / n_subjects` (Bonferroni). The residual gets
    two: tlviz's (as in Fida et al. 2022) and a robust one
    (`_robust_residual_cutoff`). For CMF the leverage is that of the
    derived subject amplitudes, which still marks subjects with an unusual
    component profile.

    Returns `(table, cutoffs)`, the table with columns `subject`,
    `leverage`, `residual`, `high_leverage`, `high_residual`,
    `high_residual_robust`; nothing is flagged by a missing cut-off.
    """
    subject_mode = np.asarray(saved.subject_mode, dtype=float)
    n_subjects, rank = subject_mode.shape
    if len(observed) != n_subjects:
        raise ValueError(
            f"{len(observed)} data slices for a model of {n_subjects} subjects",
        )
    p_value = alpha / n_subjects
    leverage = np.asarray(compute_leverage(subject_mode), dtype=float)
    residual = subject_residuals(observed, reconstruct_slices(saved))
    cutoffs = OutlierCutoffs(
        leverage=_leverage_cutoff(leverage, rank, p_value),
        residual=_residual_cutoff(residual, p_value),
        robust_residual=_robust_residual_cutoff(residual, p_value),
    )
    table = pd.DataFrame(
        {
            "subject": [str(s) for s in saved.subjects],
            "leverage": leverage,
            "residual": residual,
            "high_leverage": _above(leverage, cutoffs.leverage),
            "high_residual": _above(residual, cutoffs.residual),
            "high_residual_robust": _above(residual, cutoffs.robust_residual),
        },
    )
    return table, cutoffs
