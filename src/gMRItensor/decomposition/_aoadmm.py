"""matcouply's AO-ADMM, shared by PARAFAC2 and CMF: options, the
acceptance gate and zeroing of negligible loadings.
"""
import math
from typing import Any
from typing import Literal

import torch

from .models import ConvergenceError
from .models import PARAFAC2Diagnostics
from .models import PARAFAC2Solver


#: `parafac2_aoadmm` arguments callers may set via `aoadmm_options`.
_AOADMM_PASSTHROUGH_OPTIONS: frozenset[str] = frozenset(
    {
        "l1_penalty",
        "l2_penalty",
        "tv_penalty",
        "unimodal",
        "generalized_l2_penalty",
        "l2_norm_bound",
        "lower_bound",
        "upper_bound",
        "regs",
        "feasibility_penalty_scale",
        "constant_feasibility_penalty",
        "aux_init",
        "dual_init",
        "svd",
        "init_params",
        "absolute_tol",
        "feasibility_tol",
        "inner_tol",
        "inner_n_iter_max",
        "update_A",
        "update_B_is",
        "update_C",
        "mask",
    },
)


#: Arguments this wrapper owns, mapped to the parameter that sets them.
#: Passing one via `aoadmm_options` raises rather than silently overriding.
_AOADMM_MANAGED_OPTIONS: dict[str, str] = {
    "matrices": "pass the data as `tensor_slices`",
    "rank": "use `rank`",
    "init": "use `init`",
    "n_iter_max": "use `PARAFAC2_max_iter`",
    "tol": "use `aoadmm_loss_tolerance`",
    "random_state": "use `random_state`",
    "non_negative": "use `nn_modes`",
    "verbose": "use `PARAFAC2_verbose_level`",
    "return_errors": "always enabled",
    "return_admm_vars": "always enabled",
    "parafac2": "always enabled",
}


#: Overridable defaults applied before `aoadmm_options`. The feasibility
#: penalty converges markedly faster on ragged per-subject slices.
_AOADMM_DEFAULTS: dict[str, Any] = {"constant_feasibility_penalty": True}


#: `cmf_aoadmm` arguments `compute_CMF_decomposition` owns. CMF sets no
#: defaults of its own: matcouply's apply unless `aoadmm_options` says so.
_CMF_MANAGED_OPTIONS: dict[str, str] = {
    **_AOADMM_MANAGED_OPTIONS,
    "n_iter_max": "use `CMF_max_iter`",
    "verbose": "use `CMF_verbose_level`",
    "return_admm_vars": "not used",
    "parafac2": "always disabled; use the PARAFAC2 method for that constraint",
    "update_A": "always disabled: the subject weights are fixed at one",
}


#: matcouply's own default, used when reporting the feasibility gap.
_AOADMM_DEFAULT_FEASIBILITY_TOL: float = 1e-4


#: AO-ADMM's tolerance on the relative change of the penalized objective
#: (matcouply's own default is 1e-8, which fits rarely meet within their
#: iteration budget although their error stopped improving long before).
AOADMM_DEFAULT_LOSS_TOL: float = 1e-7


#: Iterations of error history used to extrapolate a suggested `max_iter`.
_DECAY_FIT_WINDOW: int = 50


#: Cap on how far past the available history `_suggest_max_iter` extrapolates.
_MAX_EXTRAPOLATION_FACTOR: int = 50


def _resolve_nn_modes(
    nn_modes: tuple[int, ...] | None | Literal["auto"],
    solver: PARAFAC2Solver,
) -> tuple[int, ...] | None:
    """Resolve the `"auto"` sentinel to the solver's default `nn_modes`.

    `(0, 2)` for TensorLy, whose ALS cannot constrain mode 1, and
    `(0, 1, 2)` for matcouply, whose AO-ADMM can. A sentinel is needed
    because `None` already means "unconstrained".
    """
    if nn_modes != "auto":
        return nn_modes
    return (0, 2) if solver == "tensorly" else (0, 1, 2)


def _nn_modes_to_non_negative(
    nn_modes: tuple[int, ...] | None,
    method_label: str = "PARAFAC2",
) -> dict[int, bool] | None:
    """Translate `nn_modes` into matcouply's per-mode `non_negative` dict."""
    if not nn_modes:
        return None
    invalid = sorted(set(nn_modes) - {0, 1, 2})
    if invalid:
        raise ValueError(
            f"nn_modes contains invalid mode index/indices {invalid}; {method_label} "
            "has exactly three modes, so valid entries are 0 (subject), "
            "1 (evolving/time) and 2 (label/region).",
        )
    return {mode: True for mode in sorted(set(nn_modes))}


def _build_aoadmm_options(
    aoadmm_options: dict[str, Any] | None,
    managed: dict[str, str] = _AOADMM_MANAGED_OPTIONS,
    defaults: dict[str, Any] = _AOADMM_DEFAULTS,
    owner: str = "compute_PARAFAC2_decomposition",
) -> dict[str, Any]:
    """Validate and merge caller-supplied AO-ADMM options over `defaults`.

    Unknown and wrapper-managed keys raise rather than being forwarded, so a
    typo surfaces immediately instead of being swallowed by matcouply.
    """
    options = dict(defaults)
    if not aoadmm_options:
        return options

    for key in aoadmm_options:
        if key in managed:
            raise ValueError(
                f"aoadmm_options[{key!r}] is managed by {owner} -- {managed[key]}.",
            )
        if key not in _AOADMM_PASSTHROUGH_OPTIONS:
            raise ValueError(
                f"Unknown aoadmm_options key {key!r}. Valid keys are: "
                f"{sorted(_AOADMM_PASSTHROUGH_OPTIONS)}.",
            )
    options.update(aoadmm_options)
    return options


def _zero_negligible_loadings(
    factor: torch.Tensor,
    rtol: float,
) -> torch.Tensor:
    """Zero entries too small to mean anything, per component.

    A loading this far below its component's largest means the subject (or
    region) does not express that component. Leaving it as solver residue is
    not harmless: AO-ADMM returns the primal, which sits within the
    feasibility gap of non-negative, so such an entry can be ~-1e-08. Using
    it as a multiplier propagates that sign, and normalizing the result
    restores it to full amplitude as a mirrored curve.

    Scaled per component rather than absolutely, so it adapts to however the
    fit distributed scale between the modes. This only removes residue,
    relative to its own column (sign safety): a column that is small
    throughout keeps its largest entries here.
    """
    if rtol <= 0:
        return factor
    scale = factor.abs().max(dim=0, keepdim=True).values
    return torch.where(factor.abs() <= rtol * scale, torch.zeros_like(factor), factor)


def _zero_negligible_evolving(
    evolving_states: list[torch.Tensor],
    rtol: float,
) -> list[torch.Tensor]:
    """Zero negligible entries across the ragged evolving mode.

    Scaled per component against the largest value over *all* subjects, not
    each subject's own maximum -- otherwise a subject whose whole time course
    is residue would be rescaled against its own noise and kept. Residue
    cleanup only, as `_zero_negligible_loadings`.
    """
    if rtol <= 0 or not evolving_states:
        return evolving_states
    scale = (
        torch.stack(
            [B.abs().max(dim=0).values for B in evolving_states],
        )
        .max(dim=0)
        .values
    )
    return [
        torch.where(B.abs() <= rtol * scale, torch.zeros_like(B), B)
        for B in evolving_states
    ]


def _zero_negligible_modes(
    subject_mode: torch.Tensor | None,
    evolving_states: list[torch.Tensor],
    label_mode: torch.Tensor,
    nn_modes: tuple[int, ...] | None,
    rtol: float,
) -> tuple[torch.Tensor | None, list[torch.Tensor], torch.Tensor]:
    """Zero the negligible entries of each mode in `nn_modes` (0 subject,
    1 evolving, 2 label): a negligible entry means "not expressed", and its
    residue is unsafe to leave (see `_zero_negligible_loadings`). A None
    `subject_mode` (CMF, which derives it) is left as None."""
    modes = set(nn_modes or ())
    if subject_mode is not None and 0 in modes:
        subject_mode = _zero_negligible_loadings(subject_mode, rtol)
    if 1 in modes:
        evolving_states = _zero_negligible_evolving(evolving_states, rtol)
    if 2 in modes:
        label_mode = _zero_negligible_loadings(label_mode, rtol)
    return subject_mode, evolving_states, label_mode


def _as_optional_bool(value: Any) -> bool | None:
    """Coerce matcouply's `np.True_`/`np.False_` flags to plain `bool`.

    These compare equal to Python bools but fail the `is True` / `is False`
    identity checks this module and its callers use.
    """
    return None if value is None else bool(value)


def _max_feasibility_gap(feasibility_gaps: Any) -> float:
    """Largest gap across all modes and penalties at the final iteration.

    matcouply reports these as a 3-tuple (one entry per mode) of lists (one
    per penalty on that mode), mixing torch scalars and numpy floats.
    """
    final = feasibility_gaps[-1]
    gaps = [float(gap) for mode_gaps in final for gap in mode_gaps]
    return max(gaps) if gaps else 0.0


def _suggest_max_iter(
    rec_errors: list[float],
    tolerance: float,
    safety_factor: float = 2.0,
) -> int | None:
    """Extrapolate how many iterations would reach `tolerance`.

    `abs(rec[-2] - rec[-1])` decays roughly log-linearly for AO-ADMM, so a
    least-squares line through `log|delta|` over the last
    `_DECAY_FIT_WINDOW` iterations can be solved for the crossing point. The
    estimate is biased low when extrapolated early, hence `safety_factor` on
    the extra iterations.

    Returns None when the decay cannot be extrapolated: too little history,
    an error that is not decreasing, or a crossing further away than
    `_MAX_EXTRAPOLATION_FACTOR` times the available history.
    """
    deltas = [
        abs(rec_errors[i + 1] - rec_errors[i]) for i in range(len(rec_errors) - 1)
    ]
    # Need several points to fit a slope, and positive deltas to take logs of.
    usable = [delta for delta in deltas[-_DECAY_FIT_WINDOW:] if delta > 0]
    if len(usable) < 5:
        return None

    log_deltas = [math.log(delta) for delta in usable]
    n = len(log_deltas)
    mean_x = (n - 1) / 2
    mean_y = sum(log_deltas) / n
    variance = sum((i - mean_x) ** 2 for i in range(n))
    if variance == 0:
        return None
    slope = (
        sum((i - mean_x) * (y - mean_y) for i, y in enumerate(log_deltas)) / variance
    )
    if slope >= 0:
        # Not decaying -- more iterations will not get there.
        return None

    extra = (math.log(tolerance) - log_deltas[-1]) / slope
    if extra <= 0:
        return None

    observed = len(rec_errors) - 1
    # Also stops a near-flat sequence, whose slope is negative only through
    # floating-point noise, from suggesting an astronomical number.
    if extra > _MAX_EXTRAPOLATION_FACTOR * max(observed, 1):
        return None

    suggested = observed + extra * safety_factor
    # Round up rather than quoting a false precision.
    magnitude = 10 ** max(1, int(math.log10(suggested)) - 1)
    return int(math.ceil(suggested / magnitude) * magnitude)


def _accept_aoadmm_fit(
    diagnostics: Any,
    max_iter: int,
    tolerance: float | None,
    aoadmm_loss_tolerance: float,
    feasibility_tol: float,
    nn_modes: tuple[int, ...] | None,
    method_label: str,
) -> tuple[list[torch.Tensor], PARAFAC2Diagnostics]:
    """Gate an AO-ADMM fit on matcouply's `diagnostics`; raise if rejected.

    Shared by the PARAFAC2 and CMF fits. Rejects an infeasible fit, and one
    whose reconstruction error is still moving by `tolerance`. With
    `tolerance=None` a fit that ran out of iterations before matcouply's own
    criterion was met is kept, as matcouply returns it. Returns the
    per-iteration relative reconstruction errors and what the fit achieved.
    """
    rec_errors = [float(error) for error in diagnostics.rec_errors]
    losses = [float(loss) for loss in diagnostics.regularized_loss]
    max_gap = _max_feasibility_gap(diagnostics.feasibility_gaps)
    n_iter = int(diagnostics.n_iter)
    loss_change = (
        abs(losses[-2] - losses[-1]) / abs(losses[-2])
        if len(losses) >= 2 and losses[-2] != 0
        else None
    )
    # Coerce before comparing: `is False` against a raw numpy bool_ never fires.
    feasible = _as_optional_bool(diagnostics.satisfied_feasibility_condition)
    loss_converged = _as_optional_bool(diagnostics.satisfied_stopping_condition)

    if feasible is False:
        raise ConvergenceError(
            f"AO-ADMM constraints are not satisfied: max feasibility gap "
            f"{max_gap:.3e} > feasibility_tol {feasibility_tol:.3e} after "
            f"{n_iter}/{max_iter} iterations. The constraints requested via "
            f"nn_modes={nn_modes} are therefore not actually enforced on this "
            f"fit. Increase {method_label}_max_iter, or relax feasibility_tol via "
            f"aoadmm_options if this gap is acceptable for your use.",
            reason="feasibility",
            level_reached=max_gap,
        )

    delta = abs(rec_errors[-2] - rec_errors[-1]) if len(rec_errors) >= 2 else math.inf
    # Without a tolerance a fit that ran out of iterations is kept, as
    # matcouply itself returns it; `_warn_if_accepted_at_iteration_limit`
    # reports it.
    if tolerance is not None and delta >= tolerance:
        suggested = _suggest_max_iter(rec_errors, tolerance)
        suggestion = (
            f" Extrapolating the observed decay, about {suggested} iterations "
            f"would be needed."
            if suggested is not None and suggested > max_iter
            else ""
        )
        raise ConvergenceError(
            f"AO-ADMM reconstruction error has not converged: "
            f"|delta rel. reconstruction error| = {delta:.3e} >= tolerance "
            f"{tolerance:.3e} after {n_iter}/{max_iter} iterations "
            f"(rel. reconstruction error {rec_errors[-1]:.6g})."
            f"{suggestion} Increase {method_label}_max_iter, loosen {method_label}_tolerance, "
            f"or lower aoadmm_loss_tolerance if AO-ADMM stopped early on the "
            f"penalized objective ({diagnostics.message}).",
            reason="reconstruction",
            level_reached=delta,
            suggested_max_iter=suggested,
        )

    parafac2_diagnostics = PARAFAC2Diagnostics(
        solver="matcouply",
        n_iter=n_iter,
        max_iter=max_iter,
        reached_max_iter=n_iter >= max_iter,
        relative_reconstruction_error=rec_errors[-1],
        reconstruction_error_change=delta,
        reconstruction_tolerance=tolerance,
        nn_modes=nn_modes,
        loss_converged=loss_converged,
        loss_tolerance=aoadmm_loss_tolerance,
        final_relative_loss_change=loss_change,
        feasible=feasible,
        max_feasibility_gap=max_gap,
        feasibility_tol=feasibility_tol,
        message=str(diagnostics.message),
    )
    errors = [torch.tensor(error, dtype=torch.float64) for error in rec_errors]
    return errors, parafac2_diagnostics
