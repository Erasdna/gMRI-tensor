"""Subject-mode outlier diagnostics: leverage and per-subject residual.

Following Fida et al. (2022, Sec. 3.3): a subject is influential if its
subject-mode leverage is high and poorly described if its residual is high;
each gets a Bonferroni-corrected cut-off. The leverage cut-off and the
paper's residual cut-off come from `tlviz.outliers`; the leverage itself
is computed here with a pseudo-inverse, so a rank-deficient subject mode
does not fail. The residual
itself is computed here because tlviz has no mask and assumes equally
sized slabs, while CP slices carry NaN for missing sessions and
PARAFAC2/CMF slices differ in length; a robust residual cut-off is added
because tlviz's is masked by the very outlier it should find.
"""
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from gMRItensor.model_io import SavedDecomposition
from scipy.stats import chi2
from tlviz.outliers import get_leverage_outlier_threshold
from tlviz.outliers import get_slabwise_sse_outlier_threshold


#: The boolean columns of `outlier_table`, one per cut-off.
FLAG_COLUMNS = ("high_leverage", "high_residual", "high_residual_robust")


def iter_reconstructions(saved: SavedDecomposition) -> Iterator[np.ndarray]:
    """Each subject's model estimate `X̂_i`, `(n_timepoints_i, n_labels)`,
    one at a time.

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
        for a in subject_mode:
            yield (time_mode * (weights * a)) @ label_mode.T
        return
    for i, state in enumerate(saved.evolving_states or []):
        scale = weights * subject_mode[i] if saved.method == "parafac2" else weights
        yield (np.asarray(state, dtype=float) * scale) @ label_mode.T


def reconstruct_slices(saved: SavedDecomposition) -> list[np.ndarray]:
    """Every subject's `X̂_i` at once (see `iter_reconstructions`)."""
    return list(iter_reconstructions(saved))


def data_slices(data: torch.Tensor | Sequence[torch.Tensor]) -> list[np.ndarray]:
    """`DecompositionInput.data` as one numpy array per subject."""
    if isinstance(data, torch.Tensor):
        return [s.numpy() for s in data.detach().cpu()]
    return [s.detach().cpu().numpy() for s in data]


def subject_residuals(
    observed: Sequence[np.ndarray],
    estimated: Iterable[np.ndarray],
) -> np.ndarray:
    """Mean squared residual per subject over its observed (non-NaN)
    entries, normalised to sum to 1.

    The mean, not the sum, so subjects with fewer observed entries (missing
    sessions, shorter slices) are not scored as better fitted. `estimated`
    is consumed one subject at a time, so it can be `iter_reconstructions`.
    Raises `ValueError` if a subject's slices differ in shape.
    """
    mse = []
    for i, (x, x_hat) in enumerate(zip(observed, estimated)):
        if np.shape(x) != np.shape(x_hat):
            raise ValueError(
                f"subject {i}: data of shape {np.shape(x)} for a model "
                f"estimate of shape {np.shape(x_hat)}",
            )
        mse.append(np.nanmean((np.asarray(x, dtype=float) - x_hat) ** 2))
    residual = np.array(mse)
    total = residual.sum()
    return residual / total if total > 0 else residual


def leverage(subject_mode: np.ndarray) -> np.ndarray:
    """Each subject's leverage, `diag(A A⁺)`.

    Equals the usual `diag(A (AᵀA)⁻¹ Aᵀ)` when `A` has full column rank,
    and stays defined when it does not (more components than subjects, or
    a component no subject expresses).
    """
    A = np.asarray(subject_mode, dtype=float)
    return np.einsum("ij,ji->i", A, np.linalg.pinv(A))


def _leverage_cutoff(
    leverages: np.ndarray,
    rank: int,
    p_value: float,
) -> float | None:
    """Belsley's F-based cut-off, or Hotelling's T² at rank 1 (the former
    needs rank > 1); None if there are too few subjects for either, or the
    subject mode does not have full column rank (`rank` is then its
    numerical rank, smaller than the model's)."""
    n = len(leverages)
    if rank > 1 and n > rank:
        method = "p-value"
    elif n > rank + 1:
        method = "hotelling"
    else:
        return None
    return float(
        get_leverage_outlier_threshold(leverages, method=method, p_value=p_value),
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
    `leverage`, `residual` and the `FLAG_COLUMNS`; nothing is flagged by a
    missing cut-off. A subject mode without full column rank gets no
    leverage cut-off.
    """
    subject_mode = np.asarray(saved.subject_mode, dtype=float)
    n_subjects, rank = subject_mode.shape
    if len(observed) != n_subjects:
        raise ValueError(
            f"{len(observed)} data slices for a model of {n_subjects} subjects",
        )
    p_value = alpha / n_subjects
    leverages = leverage(subject_mode)
    residual = subject_residuals(observed, iter_reconstructions(saved))
    full_rank = np.linalg.matrix_rank(subject_mode) == rank
    cutoffs = OutlierCutoffs(
        leverage=_leverage_cutoff(leverages, rank, p_value) if full_rank else None,
        residual=_residual_cutoff(residual, p_value),
        robust_residual=_robust_residual_cutoff(residual, p_value),
    )
    table = pd.DataFrame(
        {
            "subject": [str(s) for s in saved.subjects],
            "leverage": leverages,
            "residual": residual,
            "high_leverage": _above(leverages, cutoffs.leverage),
            "high_residual": _above(residual, cutoffs.residual),
            "high_residual_robust": _above(residual, cutoffs.robust_residual),
        },
    )
    return table, cutoffs
