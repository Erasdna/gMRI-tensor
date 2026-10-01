import gc
import math
import os
import sys
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import current_process
from multiprocessing import get_context
from typing import Any
from typing import Callable
from typing import Literal
from typing import overload

import tensorly as tl
import torch
from matcouply.decomposition import parafac2_aoadmm
from tensorly.parafac2_tensor import Parafac2Tensor
from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao_memory
from tlviz.factor_tools import degeneracy_score
from tqdm import tqdm

#: Which library fits the PARAFAC2 model. Not to be confused with TensorLy's
#: *compute* backend (numpy/pytorch), which `setup_backend` configures.
PARAFAC2Solver = Literal["tensorly", "matcouply"]

#: Why a single restart was rejected. Tallied by `_repeat_with_restarts` so
#: `run_PARAFAC2_decomposition_repeated` can tell the caller what to change.
ConvergenceFailureReason = Literal[
    "max_iter",
    "reconstruction",
    "feasibility",
    "degenerate",
]


class ConvergenceError(Exception):
    """Raised when a single decomposition attempt fails to converge.

    `reason` classifies the failure so `_repeat_with_restarts` can tally it
    across restarts and `run_PARAFAC2_decomposition_repeated` can give
    actionable advice (see `_build_restart_advisory`) rather than just
    reporting that nothing converged. `suggested_max_iter` is set only for
    `reason="reconstruction"`, where the observed error decay lets us
    extrapolate how many iterations would actually have been needed.
    """

    def __init__(
        self,
        message: str,
        reason: ConvergenceFailureReason | None = None,
        level_reached: float | None = None,
        suggested_max_iter: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.level_reached = level_reached
        self.suggested_max_iter = suggested_max_iter


@dataclass(frozen=True)
class PARAFAC2Diagnostics:
    """What a PARAFAC2 fit actually achieved, populated for both solvers.

    Some fields mean the same thing for `solver="tensorly"` and
    `solver="matcouply"` and some do not -- mixing the two up is the easiest
    way to draw a wrong conclusion when comparing solvers, so they are
    separated here deliberately.

    **Comparable across solvers**

    `relative_reconstruction_error` is ``||X - X_hat|| / ||X||`` under the
    same definition in both (TensorLy's `_parafac2_reconstruction_error`
    divided by the tensor norm; matcouply's `rec_errors`, divided by
    `_root_sum_squared_list(matrices)`). `reconstruction_error_change`
    against `reconstruction_tolerance` is the *acceptance gate*, applied with
    TensorLy's criterion (`abs(rec[-2] - rec[-1]) < tol`) to both solvers --
    so "this fit was accepted" means the same thing either way.

    **Not comparable across solvers**

    `loss_converged` is matcouply's own stopping condition, measured on the
    *penalized objective*, which has no TensorLy analogue. A matcouply fit
    with `loss_converged=False` is routinely better than a TensorLy fit that
    reports convergence -- never gate a comparison on it. `n_iter` is not
    comparable either: an ALS sweep and an AO-ADMM iteration are not the same
    unit of work. `feasible`/`max_feasibility_gap` are matcouply-only, since
    TensorLy has no constraint-satisfaction notion to compare against.

    There is deliberately no single `converged` field: it would invite
    exactly the cross-solver comparison that does not hold.
    """

    solver: PARAFAC2Solver
    n_iter: int
    max_iter: int
    reached_max_iter: bool
    # Comparable across solvers.
    relative_reconstruction_error: float
    reconstruction_error_change: float
    reconstruction_tolerance: float
    # matcouply only; None for tensorly.
    loss_converged: bool | None = None
    loss_tolerance: float | None = None
    final_relative_loss_change: float | None = None
    feasible: bool | None = None
    max_feasibility_gap: float | None = None
    feasibility_tol: float | None = None
    message: str = ""


#: matcouply `parafac2_aoadmm` keyword arguments the caller may set through
#: `aoadmm_options`. Everything AO-ADMM-specific that this wrapper does not
#: manage itself -- regularization, ADMM internals, init schemes.
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

#: matcouply keyword arguments this wrapper owns. Passing one through
#: `aoadmm_options` is an error naming the parameter that controls it,
#: rather than a silent override of the wrapper's own bookkeeping.
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

#: Overridable defaults applied before `aoadmm_options`.
#: `constant_feasibility_penalty=True` converges markedly faster on the
#: ragged per-subject slices this package produces.
_AOADMM_DEFAULTS: dict[str, Any] = {"constant_feasibility_penalty": True}

#: matcouply's own default, used when reporting the feasibility gap.
_AOADMM_DEFAULT_FEASIBILITY_TOL: float = 1e-4

#: Iterations of error history used to extrapolate a suggested `max_iter`.
_DECAY_FIT_WINDOW: int = 50

#: Never suggest a `max_iter` more than this many times the history we have.
#: Extrapolating further is guesswork, and it is what keeps a near-flat
#: error sequence from producing an astronomical suggestion.
_MAX_EXTRAPOLATION_FACTOR: int = 50


@torch.no_grad()
@torch.compile(dynamic=True)
def non_negative_parafac_compiled(tensor, **kwargs):
    return tl.decomposition.non_negative_parafac(tensor, **kwargs)


@torch.no_grad()
@torch.compile(dynamic=True)
def parafac_compiled(tensor, **kwargs):
    return tl.decomposition.parafac(tensor, **kwargs)


def compute_CP_decomposition(
    tensor: torch.Tensor,
    rank: int,
    CP_max_iter: int = 500,
    random_state: int = 0,
    init: str = "random",
    CP_verbose_level: int = 0,
    CP_tolerance: float = 1e-5,
    normalize_factors: bool = False,
    allow_nan_imputation: bool = False,
    non_negative: bool = True,
):
    """Compute a single CP/PARAFAC decomposition attempt.

    Notes
    -----
    If `allow_nan_imputation` is True, any NaN entries in `tensor` (e.g. from
    `prepare_tensor(..., require_regular=True)`, where a subject's missing
    time point becomes a NaN row) are treated as missing and imputed from the
    model's own reconstruction at each iteration, via TensorLy's `mask`
    support -- unlike PARAFAC2, where neither solver is wired up for
    imputation here (see `compute_PARAFAC2_decomposition`; TensorLy's
    `parafac2` has no mask support at all in this version, and while
    matcouply's `parafac2_aoadmm` does accept a `mask`, it is not plumbed
    through). If False (the default) and `tensor` contains NaN, a
    `ValueError` is raised rather than silently fitting on/propagating NaN.

    `non_negative` defaults to True (non-negative CP, appropriate for a
    tracer signal that should physically be non-negative). Set it to False to
    run plain, unconstrained CP instead.
    """
    mask = None
    if allow_nan_imputation:
        mask = (~torch.isnan(tensor)).to(tensor.dtype)
        tensor = torch.nan_to_num(tensor, nan=0.0)
    elif torch.isnan(tensor).any():
        raise ValueError(
            "tensor contains NaN values; pass allow_nan_imputation=True to let "
            "TensorLy impute them during fitting, or remove/fill them yourself "
            "first.",
        )

    decomposition_fn = (
        non_negative_parafac_compiled if non_negative else parafac_compiled
    )
    decomp, errors = decomposition_fn(
        tensor,
        rank=rank,
        n_iter_max=CP_max_iter,
        tol=CP_tolerance,  # Computing this tensor decomp is quite expensive...
        return_errors=True,
        random_state=random_state,
        verbose=CP_verbose_level,
        init=init,
        normalize_factors=normalize_factors,
        mask=mask,
    )
    if len(errors) > CP_max_iter - 1:
        raise ConvergenceError(
            "Decomposition did not converge within the maximum iteration count",
        )

    w, f = decomp
    w = w.float()
    f = [ff.float() for ff in f]
    if degeneracy_score((w, f)) < -0.85:
        raise ConvergenceError("Decomposition is degenerate")

    return decomp, errors


@contextmanager
def _matcouply_numeric_context(device: torch.device) -> Iterator[None]:
    """Make torch's defaults float64-on-`device` for the duration of a fit.

    matcouply builds its factor, auxiliary and dual variables with a mix of
    `tl.tensor(numpy_array)` (which, under TensorLy's pytorch backend,
    produces a **float64 CPU** tensor regardless of torch's defaults) and
    `tl.eye`/`tl.zeros` (which follow torch's *default* dtype and device). It
    never threads `tl.context(matrices[0])` through, so with this package's
    usual float32 CUDA setup the two kinds of tensor disagree and matcouply
    fails with either

        RuntimeError: expected m1 and m2 to have the same dtype,
        but got: float != double

    or, on GPU,

        RuntimeError: Expected all tensors to be on the same device,
        but got mat2 is on cpu, different from other tensors on cuda:0

    Setting torch's defaults to float64 and to the input's device makes both
    kinds agree. float32 is not reachable without monkeypatching matcouply
    itself; results are cast back to float32 on the way out (see
    `run_PARAFAC2_decomposition_repeated`'s `to_cpu`).

    Scoped per fit rather than set globally in `setup_backend`: float64 would
    otherwise apply to the CP path too, where it doubles memory and makes the
    `torch.set_float32_matmul_precision("high")`/TF32 tuning meaningless.
    Spawned restart workers pick this up automatically, since they reach it
    via `_restart_worker` -> `compute_PARAFAC2_decomposition`, so
    `_init_restart_worker_backend` needs no matcouply-specific setup.

    Note these are process-global torch settings. That is safe here because
    the restart pool and `evaluate_replicability_multiproc`'s pool are both
    process-parallel; it would not be safe if fits were ever run on threads
    within a single process.
    """
    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_default_device(previous_device)


def _resolve_nn_modes(
    nn_modes: tuple[int, ...] | None | Literal["auto"],
    solver: PARAFAC2Solver,
) -> tuple[int, ...] | None:
    """Resolve the `"auto"` sentinel to the solver's default `nn_modes`.

    The default is solver-dependent -- `(0, 2)` for TensorLy, whose ALS
    cannot enforce non-negativity on mode 1, and `(0, 1, 2)` for matcouply,
    whose AO-ADMM can (the reason this solver exists here). A sentinel is
    needed rather than `None` as the default because `None` already has a
    meaning: fit with no non-negativity constraint at all.
    """
    if nn_modes != "auto":
        return nn_modes
    return (0, 2) if solver == "tensorly" else (0, 1, 2)


def _nn_modes_to_non_negative(
    nn_modes: tuple[int, ...] | None,
) -> dict[int, bool] | None:
    """Translate `nn_modes` into matcouply's per-mode `non_negative` dict."""
    if not nn_modes:
        return None
    invalid = sorted(set(nn_modes) - {0, 1, 2})
    if invalid:
        raise ValueError(
            f"nn_modes contains invalid mode index/indices {invalid}; PARAFAC2 "
            "has exactly three modes, so valid entries are 0 (subject), "
            "1 (evolving/time) and 2 (label/region).",
        )
    return {mode: True for mode in sorted(set(nn_modes))}


def _build_aoadmm_options(
    aoadmm_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and merge caller-supplied AO-ADMM options over the defaults.

    Unknown keys and keys this wrapper manages itself are rejected rather
    than forwarded, so a typo surfaces immediately instead of being silently
    swallowed by matcouply, and so a caller cannot quietly override the
    bookkeeping (`return_admm_vars`, `n_iter_max`, ...) the wrapper depends
    on.
    """
    options = dict(_AOADMM_DEFAULTS)
    if not aoadmm_options:
        return options

    for key in aoadmm_options:
        if key in _AOADMM_MANAGED_OPTIONS:
            raise ValueError(
                f"aoadmm_options[{key!r}] is managed by "
                f"compute_PARAFAC2_decomposition -- {_AOADMM_MANAGED_OPTIONS[key]}.",
            )
        if key not in _AOADMM_PASSTHROUGH_OPTIONS:
            raise ValueError(
                f"Unknown aoadmm_options key {key!r}. Valid keys are: "
                f"{sorted(_AOADMM_PASSTHROUGH_OPTIONS)}.",
            )
    options.update(aoadmm_options)
    return options


def _normalize_parafac2_factors(
    weights: torch.Tensor,
    factors: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Scale each factor's columns to unit norm, folding norms into `weights`.

    matcouply has no `normalize_factors` option of its own, so this
    reproduces TensorLy's semantics. It is well defined for PARAFAC2 because
    the projections are orthonormal: `||P_i @ B[:, r]|| == ||B[:, r]||`, so
    normalizing the shared coordinate matrix `B` rescales every subject's
    reconstructed time curve by the same factor.
    """
    scaled_weights = weights.clone()
    scaled_factors = []
    for factor in factors:
        norms = torch.linalg.norm(factor, dim=0)
        # Leave all-zero columns alone rather than dividing by zero; their
        # weight contribution is zero either way.
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        scaled_factors.append(factor / safe_norms)
        scaled_weights = scaled_weights * safe_norms
    return scaled_weights, scaled_factors


def _matcouply_to_parafac2_tensor(
    cmf: Any,
    admm_vars: Any,
    rank: int,
    normalize_factors: bool,
) -> Parafac2Tensor:
    """Adapt matcouply's output to TensorLy's `Parafac2Tensor` contract.

    matcouply returns `(weights, (A, B_is, C))` where `weights` is None and
    `B_is` is a *ragged list* of per-slice `(J_i, rank)` matrices -- not the
    `(weights, [A, B, C], projections)` form the rest of this package (and
    `gMRItensor.plotting.evolving_mode.reconstruct_evolving_factors`, and
    `gMRItensor.replicability`) is written against.

    The PARAFAC2 constraint is imposed through
    `matcouply.penalties.Parafac2`, whose auxiliary variable is exactly the
    Kiers parametrization `B_i = P_i @ B`: a list of orthonormal basis
    matrices and the shared `rank x rank` coordinate matrix. Reading those
    back out recovers TensorLy's representation faithfully -- in testing,
    `||B_is[i] - P_i @ B|| / ||B_is[i]||` is ~1e-7 -- which is what lets
    every downstream consumer work unchanged for both solvers.

    Note `B` (the coordinate matrix, i.e. `factors[1]`) is NOT non-negative
    even when mode 1 is constrained; the non-negativity holds on
    `projections[i] @ factors[1]`. See `compute_PARAFAC2_decomposition`.
    """
    _, (A, _B_is, C) = cmf
    mode_1_auxes = admm_vars.auxes[1]
    layout_error = (
        "Could not read the PARAFAC2 basis/coordinate matrices out of "
        "matcouply's ADMM auxiliary variables. This wrapper relies on "
        "matcouply.decomposition._parse_mode_penalties prepending a "
        "penalties.Parafac2 instance for mode 1, so that auxes[1][0] is the "
        "(basis_matrices, coordinate_matrix) pair. That layout is internal to "
        "matcouply and appears to have changed"
    )
    if not mode_1_auxes or not isinstance(mode_1_auxes[0], tuple):
        raise RuntimeError(f"{layout_error}.")

    parafac2_aux = mode_1_auxes[0]
    if len(parafac2_aux) != 2:
        raise RuntimeError(f"{layout_error} (expected a 2-tuple).")
    basis_matrices, coordinate_matrix = parafac2_aux

    if tuple(coordinate_matrix.shape) != (rank, rank):
        raise RuntimeError(
            f"{layout_error}: expected a {rank}x{rank} coordinate matrix, got "
            f"{tuple(coordinate_matrix.shape)}.",
        )
    if len(basis_matrices) != A.shape[0]:
        raise RuntimeError(
            f"{layout_error}: expected {A.shape[0]} basis matrices, got "
            f"{len(basis_matrices)}.",
        )
    for index, (basis, B_i) in enumerate(zip(basis_matrices, _B_is)):
        if tuple(basis.shape) != (B_i.shape[0], rank):
            raise RuntimeError(
                f"{layout_error}: basis matrix {index} has shape "
                f"{tuple(basis.shape)}, expected {(B_i.shape[0], rank)}.",
            )

    weights = torch.ones(rank, dtype=A.dtype, device=A.device)
    factors = [A, coordinate_matrix, C]
    if normalize_factors:
        weights, factors = _normalize_parafac2_factors(weights, factors)

    return Parafac2Tensor((weights, factors, list(basis_matrices)))


def _as_optional_bool(value: Any) -> bool | None:
    """Coerce matcouply's numpy bool_ flags to plain `bool` (or None).

    matcouply returns `np.True_`/`np.False_` for its stopping and feasibility
    conditions. Those compare equal to Python bools but fail `is True` /
    `is False` identity checks, which both this module and callers use.
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

    `abs(rec[-2] - rec[-1])` decays close to log-linearly for AO-ADMM, so a
    least-squares line through `log|delta|` over the last
    `_DECAY_FIT_WINDOW` iterations can be solved for where it crosses
    `tolerance`. Validated against a run carried out to 3000 iterations whose
    true crossing of `tolerance=1e-5` was iteration 552: extrapolating from
    iteration 200 estimated 399, from 300 estimated 497 and from 500
    estimated 549.

    The estimate is therefore good but biased low when extrapolated from
    early in the run, hence `safety_factor` on the *extra* iterations.
    Returns None rather than guessing when the decay cannot be extrapolated:
    too little history, an error that isn't decreasing (more iterations are
    not the answer), or a decay so slow that reaching `tolerance` is further
    away than this much history can honestly speak to. That last guard
    matters -- a flat error sequence has a slope that is only
    infinitesimally negative through floating-point noise, which without it
    extrapolates to absurdities like 5e18 iterations.
    """
    deltas = [
        abs(rec_errors[i + 1] - rec_errors[i]) for i in range(len(rec_errors) - 1)
    ]
    # Need a couple of points to fit a slope, and strictly positive deltas to
    # take logs of.
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
    # Refuse to extrapolate more than _MAX_EXTRAPOLATION_FACTOR beyond the
    # history we actually have. Besides being dishonest, this is what stops a
    # near-flat sequence -- whose fitted slope is negative only through
    # floating-point noise -- from suggesting an astronomically large number.
    if extra > _MAX_EXTRAPOLATION_FACTOR * max(observed, 1):
        return None

    suggested = observed + extra * safety_factor
    # Round up to something human-sized rather than quoting a false precision.
    magnitude = 10 ** max(1, int(math.log10(suggested)) - 1)
    return int(math.ceil(suggested / magnitude) * magnitude)


def _compute_PARAFAC2_matcouply(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int,
    random_state: int,
    init: str,
    verbose_level: int,
    tolerance: float,
    normalize_factors: bool,
    nn_modes: tuple[int, ...] | None,
    aoadmm_options: dict[str, Any] | None,
    aoadmm_loss_tolerance: float,
) -> tuple[Parafac2Tensor, list[torch.Tensor], PARAFAC2Diagnostics]:
    """Fit PARAFAC2 with matcouply's AO-ADMM and adapt it to TensorLy's shape.

    See `compute_PARAFAC2_decomposition` for the two-threshold convergence
    scheme this implements.
    """
    options = _build_aoadmm_options(aoadmm_options)
    feasibility_tol = options.get("feasibility_tol", _AOADMM_DEFAULT_FEASIBILITY_TOL)

    slices = (
        list(tensor_slices)
        if isinstance(tensor_slices, list)
        else [tensor_slices[i] for i in range(tensor_slices.shape[0])]
    )
    device = slices[0].device

    with _matcouply_numeric_context(device):
        cmf, admm_vars, diagnostics = parafac2_aoadmm(
            [s.double() for s in slices],
            rank=rank,
            init=init,
            n_iter_max=max_iter,
            tol=aoadmm_loss_tolerance,
            random_state=random_state,
            non_negative=_nn_modes_to_non_negative(nn_modes),
            verbose=verbose_level,
            return_errors=True,
            return_admm_vars=True,
            **options,
        )
        result = _matcouply_to_parafac2_tensor(
            cmf,
            admm_vars,
            rank,
            normalize_factors,
        )

    # Everything below runs with torch's defaults already restored, so
    # neither the float64 dtype nor the device leaks into what we hand back
    # -- including on the error paths.
    rec_errors = [float(error) for error in diagnostics.rec_errors]
    losses = [float(loss) for loss in diagnostics.regularized_loss]
    max_gap = _max_feasibility_gap(diagnostics.feasibility_gaps)
    n_iter = int(diagnostics.n_iter)
    loss_change = (
        abs(losses[-2] - losses[-1]) / abs(losses[-2])
        if len(losses) >= 2 and losses[-2] != 0
        else None
    )
    # Coerce before comparing: matcouply reports these as numpy bool_, which
    # compares equal to a Python bool but is not identical to one, so an
    # `is False` check against the raw value would silently never fire.
    feasible = _as_optional_bool(diagnostics.satisfied_feasibility_condition)
    loss_converged = _as_optional_bool(diagnostics.satisfied_stopping_condition)

    if feasible is False:
        raise ConvergenceError(
            f"AO-ADMM constraints are not satisfied: max feasibility gap "
            f"{max_gap:.3e} > feasibility_tol {feasibility_tol:.3e} after "
            f"{n_iter}/{max_iter} iterations. The constraints requested via "
            f"nn_modes={nn_modes} are therefore not actually enforced on this "
            f"fit. Increase PARAFAC2_max_iter, or relax feasibility_tol via "
            f"aoadmm_options if this gap is acceptable for your use.",
            reason="feasibility",
            level_reached=max_gap,
        )

    delta = abs(rec_errors[-2] - rec_errors[-1]) if len(rec_errors) >= 2 else math.inf
    if delta >= tolerance:
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
            f"{suggestion} Increase PARAFAC2_max_iter, loosen PARAFAC2_tolerance, "
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
        loss_converged=loss_converged,
        loss_tolerance=aoadmm_loss_tolerance,
        final_relative_loss_change=loss_change,
        feasible=feasible,
        max_feasibility_gap=max_gap,
        feasibility_tol=feasibility_tol,
        message=str(diagnostics.message),
    )
    errors = [torch.tensor(error, dtype=torch.float64) for error in rec_errors]
    return result, errors, parafac2_diagnostics


def _compute_PARAFAC2_tensorly(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int,
    random_state: int,
    init: str,
    verbose_level: int,
    tolerance: float,
    normalize_factors: bool,
    nn_modes: tuple[int, ...] | None,
) -> tuple[Parafac2Tensor, list[torch.Tensor], PARAFAC2Diagnostics]:
    """Fit PARAFAC2 with TensorLy's ALS. Behaviour unchanged from before."""
    result, errors = tl.decomposition.parafac2(
        tensor_slices,
        rank=rank,
        n_iter_max=max_iter,
        tol=tolerance,
        return_errors=True,
        random_state=random_state,
        verbose=verbose_level,
        init=init,
        normalize_factors=normalize_factors,
        nn_modes=list(nn_modes) if nn_modes else None,
    )
    if len(errors) > max_iter - 1:
        raise ConvergenceError(
            "Decomposition did not converge within the maximum iteration count",
            reason="max_iter",
        )

    rec_errors = [float(error) for error in errors]
    delta = abs(rec_errors[-2] - rec_errors[-1]) if len(rec_errors) >= 2 else math.inf
    diagnostics = PARAFAC2Diagnostics(
        solver="tensorly",
        n_iter=len(errors),
        max_iter=max_iter,
        reached_max_iter=len(errors) > max_iter - 1,
        relative_reconstruction_error=rec_errors[-1],
        reconstruction_error_change=delta,
        reconstruction_tolerance=tolerance,
        message="converged",
    )
    return result, errors, diagnostics


def compute_PARAFAC2_decomposition(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    PARAFAC2_max_iter: int = 500,
    random_state: int = 0,
    init: str = "random",
    PARAFAC2_verbose_level: int = 0,
    PARAFAC2_tolerance: float = 1e-5,
    normalize_factors: bool = False,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = "auto",
    solver: PARAFAC2Solver = "tensorly",
    aoadmm_options: dict[str, Any] | None = None,
    aoadmm_loss_tolerance: float = 1e-10,
) -> tuple[Parafac2Tensor, list[torch.Tensor], PARAFAC2Diagnostics]:
    """Compute a single PARAFAC2 decomposition attempt.

    PARAFAC2 relaxes CP/PARAFAC by allowing one mode (here: time) to have a
    different size per slice (here: per subject) -- its "evolving factor".
    `tensor_slices` may be a single regular 3D tensor or a list of 2D slices
    with a shared number of columns but a varying number of rows (e.g. one
    (n_timepoints_i, n_labels) array per subject).

    The returned `factors = [A, B, C]` are always regular, fixed-shape
    matrices: `A` (subjects x rank), `B` (rank x rank, the shared evolving-mode
    basis), `C` (labels x rank). The actual subject-specific time pattern is
    `projections[i] @ B` -- see `gMRItensor.plotting.evolving_mode.
    reconstruct_evolving_factors`. This holds for **both** solvers: the
    matcouply result is adapted to exactly this shape (see
    `_matcouply_to_parafac2_tensor`), so downstream consumers never branch on
    `solver`.

    Choosing a solver
    -----------------
    `solver="tensorly"` (default) uses `tensorly.decomposition.parafac2`
    (ALS/HALS). `solver="matcouply"` uses
    `matcouply.decomposition.parafac2_aoadmm` (AO-ADMM), which exists here
    for one main reason: **it can enforce non-negativity on mode 1**, the
    evolving/time mode, which TensorLy's ALS cannot. It also offers L1/L2/TV,
    unimodality and bound constraints via `aoadmm_options`.

    `nn_modes` therefore defaults to `"auto"`, which resolves to `(0, 2)` for
    TensorLy (subject and region modes only) and `(0, 1, 2)` for matcouply.
    Pass `None` for a wholly unconstrained fit. Asking TensorLy for mode 1
    raises: it accepts the request but only warns and silently leaves the
    mode unconstrained, which is worse than refusing.

    Caveat specific to matcouply: `factors[1]` is AO-ADMM's *coordinate
    matrix*, and it carries negative entries even when mode 1 is fully
    constrained (measured min ~-0.93 on a fit whose per-subject factors were
    non-negative to ~5e-8). The non-negativity holds on
    `projections[i] @ factors[1]`, which is the quantity with a physical
    meaning. Assert there, not on `factors[1]`.

    matcouply also runs in float64 regardless of `setup_backend`'s float32
    setup -- see `_matcouply_numeric_context` for why -- so the TF32 and
    `set_float32_matmul_precision` tuning does not apply to it. Results are
    cast back to float32 by `run_PARAFAC2_decomposition_repeated`.

    matcouply's `init` vocabulary is `"random"`, `"svd"`, `"threshold_svd"`,
    `"parafac2_als"`, `"cp_als"` and `"cp_hals"`. Note the non-random ones
    ignore `random_state`, which makes every restart identical.

    Convergence: AO-ADMM has two thresholds
    ---------------------------------------
    TensorLy's `tol` tests the change in *relative reconstruction error*;
    matcouply's own `tol` tests the change in the *penalized objective*. The
    same number is not the same stopping rule, so matcouply's own
    convergence flag cannot be used to decide whether a fit is acceptable
    (empirically it is routinely `False` on fits that are better than a
    TensorLy fit reporting success).

    So the two are separated:

    - `aoadmm_loss_tolerance` is handed to matcouply as its `tol` and governs
      only *when AO-ADMM stops iterating*. It defaults to 1e-10, tighter than
      matcouply's own 1e-8, because a loose value lets AO-ADMM stop while the
      reconstruction error is still moving -- which the gate below then
      rejects, wasting the whole restart.
    - `PARAFAC2_tolerance` is the **acceptance gate**, and means the same
      thing for both solvers: `abs(rec[-2] - rec[-1]) < tolerance`, TensorLy's
      own criterion, applied by this function to matcouply's reconstruction
      errors. Failing it raises `ConvergenceError`.

    Returns
    -------
    tuple[Parafac2Tensor, list[torch.Tensor], PARAFAC2Diagnostics]
        The fit, its per-iteration relative reconstruction errors, and what
        the fit actually achieved. See `PARAFAC2Diagnostics` for which of its
        fields may be compared across solvers and which may not.

    Notes
    -----
    Unlike `compute_CP_decomposition`, this is not wrapped in `torch.compile`:
    TensorLy's `parafac2` has per-iteration convergence/linesearch checks and
    an inherently ragged per-slice Python loop that cannot be traced into one
    graph (confirmed to produce dozens of graph breaks on trivial inputs), so
    compiling it adds overhead without a real speedup.

    Raises
    ------
    ValueError
        If any slice contains NaN; if `solver` is unknown; if an
        AO-ADMM-only option is passed with `solver="tensorly"`; or if
        `nn_modes` includes mode 1 with `solver="tensorly"`.
    ConvergenceError
        If the fit does not meet the acceptance gate above, if matcouply's
        constraints are left infeasible, or if the result is degenerate.
    """
    slices_to_check = (
        tensor_slices if isinstance(tensor_slices, list) else [tensor_slices]
    )
    if any(torch.isnan(s).any() for s in slices_to_check):
        # Neither path imputes: TensorLy's parafac2 has no mask support in
        # this version at all, and while matcouply does accept a `mask`, it
        # is not wired up here.
        raise ValueError(
            "tensor_slices contains NaN values, but neither PARAFAC2 solver is "
            "set up for NaN imputation here -- remove or impute missing values "
            "first (e.g. via prepare_tensor(..., require_regular=False), which "
            "never introduces NaN gaps).",
        )

    if solver not in ("tensorly", "matcouply"):
        raise ValueError(
            f"Unknown PARAFAC2 solver {solver!r}; expected 'tensorly' or "
            "'matcouply'.",
        )
    if solver != "matcouply":
        if aoadmm_options is not None:
            raise ValueError(
                f"aoadmm_options only applies to solver='matcouply', not "
                f"{solver!r}.",
            )
        if aoadmm_loss_tolerance != 1e-10:
            raise ValueError(
                f"aoadmm_loss_tolerance only applies to solver='matcouply', not "
                f"{solver!r}. TensorLy has no penalized objective; use "
                "PARAFAC2_tolerance.",
            )

    resolved_nn_modes = _resolve_nn_modes(nn_modes, solver)
    if solver == "tensorly" and resolved_nn_modes and 1 in resolved_nn_modes:
        raise ValueError(
            "TensorLy's PARAFAC2 ALS cannot enforce non-negativity on mode 1 "
            "(the evolving/time mode) -- it accepts the request but only warns "
            "and leaves the mode unconstrained. Pass solver='matcouply' to fit "
            "with AO-ADMM, which can enforce it.",
        )

    if solver == "matcouply":
        result, errors, diagnostics = _compute_PARAFAC2_matcouply(
            tensor_slices,
            rank=rank,
            max_iter=PARAFAC2_max_iter,
            random_state=random_state,
            init=init,
            verbose_level=PARAFAC2_verbose_level,
            tolerance=PARAFAC2_tolerance,
            normalize_factors=normalize_factors,
            nn_modes=resolved_nn_modes,
            aoadmm_options=aoadmm_options,
            aoadmm_loss_tolerance=aoadmm_loss_tolerance,
        )
    else:
        result, errors, diagnostics = _compute_PARAFAC2_tensorly(
            tensor_slices,
            rank=rank,
            max_iter=PARAFAC2_max_iter,
            random_state=random_state,
            init=init,
            verbose_level=PARAFAC2_verbose_level,
            tolerance=PARAFAC2_tolerance,
            normalize_factors=normalize_factors,
            nn_modes=resolved_nn_modes,
        )

    # Shared across solvers: the matcouply adapter returns a real
    # Parafac2Tensor with real weights, so this needs no branching.
    w = result.weights.float()
    f = [ff.float() for ff in result.factors]
    if degeneracy_score((w, f)) < -0.85:
        raise ConvergenceError(
            "Decomposition is degenerate",
            reason="degenerate",
        )

    return result, errors, diagnostics


def _restart_worker(
    args: tuple[Literal["CP", "PARAFAC2"], int, Any, dict[str, Any]],
) -> tuple[Any, Any, PARAFAC2Diagnostics | None, ConvergenceError | None]:
    """Pool worker: run a single random-restart attempt.

    Only used by `_repeat_with_restarts`'s parallel (CPU-only, `restart_procs
    >= 2`) path -- the sequential path calls `compute_CP_decomposition`/
    `compute_PARAFAC2_decomposition` directly, in-process, so it also works
    with a GPU tensor and reuses the `torch.compile` cache across restarts.

    Returns `(decomp, errors, diagnostics, None)` on success and
    `(None, None, None, exc)` on `ConvergenceError`, rather than raising, so
    one failed restart doesn't kill the whole `Pool.imap_unordered` --
    matching the sequential loop's "skip and continue" behavior. The
    exception is handed back rather than discarded so the parallel path can
    tally failure reasons exactly like the sequential one; a
    `ConvergenceError` pickles fine across the process boundary. Failure and
    diagnostics get their own slots so neither is ever read as the other.

    `diagnostics` is a `PARAFAC2Diagnostics` for PARAFAC2 and None for CP.
    """
    method, random_state, payload, kwargs = args
    try:
        if method == "CP":
            decomp, errors = compute_CP_decomposition(
                payload, random_state=random_state, **kwargs
            )
            return decomp, errors, None, None
        decomp, errors, diagnostics = compute_PARAFAC2_decomposition(
            payload,
            random_state=random_state,
            **kwargs,
        )
        return decomp, errors, diagnostics, None
    except ConvergenceError as error:
        return None, None, None, error


def _init_restart_worker_backend(num_threads: int) -> None:
    """Pool initializer: set up TensorLy's backend and cap this worker's threads.

    `tl.set_backend("pytorch")` is needed with the "spawn" start method used
    here (see `gMRItensor.replicability._init_worker_backend`, which does the
    same thing for the same reason).

    `torch.set_num_threads(num_threads)` guards against thread
    oversubscription: `torch.set_num_threads` is process-local state, so it
    does *not* carry over from the parent into a freshly "spawn"ed worker --
    left unset, each of the `restart_procs` worker processes falls back to
    its own default intra-op thread pool (often sized to *all* visible
    cores), so `restart_procs` processes each also fanning out into a full
    thread pool massively oversubscribes the CPU (this is what made the
    parallel path effectively not work: `restart_procs` workers x each
    worker's own large thread pool, rather than `restart_procs` total
    threads of work). `_repeat_with_restarts` computes `num_threads` so that
    `restart_procs * num_threads` stays close to the parent's own
    `torch.get_num_threads()` (e.g. set via `setup_backend`'s
    `CPUS_PER_TASK` handling).
    """
    tl.set_backend("pytorch")
    torch.set_num_threads(num_threads)


def _in_worker_process() -> bool:
    """True if already running inside a multiprocessing worker.

    `multiprocessing.current_process()` is the `"MainProcess"` only for the
    process that was never handed off into a `Pool` -- a worker spawned by
    `evaluate_replicability_multiproc`'s own `Pool` has some other name
    (e.g. `"SpawnPoolWorker-1"`). Used to refuse restart-level
    multiprocessing there: spawning a second layer of processes from inside
    an already-parallel worker just multiplies process-startup/backend-init
    overhead without adding real parallelism (the outer pool is already
    using all the requested workers).
    """
    return current_process().name != "MainProcess"


class RestartTally:
    """What happened across a run's random restarts.

    Previously a restart that raised `ConvergenceError` was silently skipped,
    so a run where most restarts ran out of iterations looked identical to a
    healthy one. Recording *why* each restart was rejected is what lets
    `_build_restart_advisory` tell the caller which knob to turn.
    """

    def __init__(self, attempted: int) -> None:
        self.attempted = attempted
        self.succeeded = 0
        self.reasons: Counter[str] = Counter()
        self.levels: dict[str, list[float]] = {}
        self.suggested_max_iters: list[int] = []
        self.hit_iteration_limit = 0
        self.last_failure: ConvergenceError | None = None

    def record_success(self, diagnostics: Any) -> None:
        self.succeeded += 1
        if isinstance(diagnostics, PARAFAC2Diagnostics) and (
            diagnostics.reached_max_iter
        ):
            self.hit_iteration_limit += 1

    def record_failure(self, error: ConvergenceError | None) -> None:
        if error is None:
            # Defensive: a worker should always hand the exception back.
            self.reasons["unknown"] += 1
            return
        self.last_failure = error
        reason = error.reason or "unknown"
        self.reasons[reason] += 1
        if error.level_reached is not None:
            self.levels.setdefault(reason, []).append(error.level_reached)
        if error.suggested_max_iter is not None:
            self.suggested_max_iters.append(error.suggested_max_iter)

    @property
    def failed(self) -> int:
        return sum(self.reasons.values())

    def median_level(self, reason: str) -> float | None:
        levels = sorted(self.levels.get(reason, []))
        if not levels:
            return None
        return levels[len(levels) // 2]

    def suggested_max_iter(self) -> int | None:
        if not self.suggested_max_iters:
            return None
        ordered = sorted(self.suggested_max_iters)
        return ordered[len(ordered) // 2]

    def failure_summary(self) -> str:
        if not self.reasons:
            return ""
        breakdown = ", ".join(
            f"{count} {reason}" for reason, count in self.reasons.most_common()
        )
        summary = f" (rejected: {breakdown})"
        if self.last_failure is not None:
            summary += f". Last failure: {self.last_failure}"
        return summary


def _build_restart_advisory(
    tally: RestartTally,
    solver: PARAFAC2Solver,
    max_iter: int,
    tolerance: float,
    aoadmm_loss_tolerance: float,
) -> str | None:
    """Advise on thresholds when restarts are *systematically* struggling.

    Deliberately only fires on a pattern (at least half the restarts), not on
    a single unlucky restart -- an advisory that cries wolf gets filtered out
    mentally, which defeats the point. Returns None when the run looks
    healthy.
    """
    if tally.attempted == 0:
        return None
    half = tally.attempted / 2

    reconstruction_failures = tally.reasons.get("reconstruction", 0)
    if reconstruction_failures >= half:
        level = tally.median_level("reconstruction")
        suggested = tally.suggested_max_iter()
        message = (
            f"PARAFAC2(solver={solver!r}): {reconstruction_failures} of "
            f"{tally.attempted} restarts were rejected because the "
            f"reconstruction error had not converged within max_iter="
            f"{max_iter}."
        )
        if level is not None:
            message += (
                f" Median level reached: |delta rel. reconstruction error| = "
                f"{level:.2e} vs tolerance {tolerance:.1e}"
            )
            # Within an order of magnitude means the fit is nearly there and
            # more iterations will close it; far off points at the model
            # (rank, init, over-constraint) instead, where more iterations
            # would just burn time.
            if level < 10 * tolerance:
                message += " -- close, so this is an iteration-budget issue."
            else:
                message += (
                    " -- far from the tolerance, which usually points at the "
                    "rank, the initialisation or an over-constrained fit "
                    "rather than at the iteration budget."
                )
        if suggested is not None and suggested > max_iter:
            message += f" Suggested max_iter ~= {suggested} (extrapolated)."
        message += (
            f" Alternatives: loosen PARAFAC2_tolerance if {tolerance:.1e} is "
            f"stricter than you need"
        )
        if solver == "matcouply":
            message += (
                f", or lower aoadmm_loss_tolerance (currently "
                f"{aoadmm_loss_tolerance:.1e}) if AO-ADMM is stopping early on "
                f"the penalized objective"
            )
        return message + "."

    feasibility_failures = tally.reasons.get("feasibility", 0)
    if feasibility_failures >= half:
        level = tally.median_level("feasibility")
        level_text = (
            f" Median max feasibility gap: {level:.2e}." if level is not None else ""
        )
        return (
            f"PARAFAC2(solver={solver!r}): {feasibility_failures} of "
            f"{tally.attempted} restarts were rejected because AO-ADMM left "
            f"the constraints infeasible within max_iter={max_iter}."
            f"{level_text} Raise PARAFAC2_max_iter, or relax feasibility_tol "
            "via aoadmm_options if that gap is acceptable. "
            "aoadmm_options={'constant_feasibility_penalty': True} (the "
            "default here) also helps convergence markedly."
        )

    if tally.succeeded and tally.hit_iteration_limit >= tally.succeeded / 2:
        return (
            f"PARAFAC2(solver={solver!r}): {tally.hit_iteration_limit} of "
            f"{tally.succeeded} accepted restarts ran to the iteration limit "
            f"(max_iter={max_iter}). The fits met the reconstruction "
            f"tolerance {tolerance:.1e}, so they are usable, but every restart "
            "is paying the full iteration budget and more iterations would "
            "still improve them."
        )

    return None


def _repeat_with_restarts(
    method: Literal["CP", "PARAFAC2"],
    payload: Any,
    kwargs: dict[str, Any],
    to_cpu: Callable[[Any], Any],
    init_repeats: int,
    device: torch.device,
    verbose_level: int,
    progress_bar: bool,
    restart_procs: int = 1,
) -> tuple[Any, torch.Tensor, Any, "RestartTally"]:
    """Run repeated random restarts and keep the best result.

    Shared restart/error-tracking/GPU-memory-management skeleton used by both
    `run_CP_decomposition_repeated` and `run_PARAFAC2_decomposition_repeated`.

    Parameters
    ----------
    method : Literal["CP", "PARAFAC2"]
        Which of `compute_CP_decomposition`/`compute_PARAFAC2_decomposition`
        to call for each restart.
    payload : Any
        The `tensor`/`tensor_slices` positional argument to pass to that
        function.
    kwargs : dict[str, Any]
        The rest of that function's arguments (everything except
        `random_state`, which is filled in per restart).
    to_cpu : Callable[[Any], Any]
        Moves the winning `decomp` to CPU/float precision.
    init_repeats : int
        Number of random restarts to try.
    device : torch.device
        Device the input tensor(s) live on (used for CUDA memory management,
        and to reject `restart_procs >= 2`, which isn't safe on CUDA -- see
        Raises).
    verbose_level : int
        If > 0, prints each `ConvergenceError` encountered.
    progress_bar : bool
        Whether to show a tqdm progress bar over the restarts.
    restart_procs : int, optional
        Number of worker processes to run restarts in parallel with. By
        default 1 (sequential, in-process, unchanged behavior). Only takes
        effect when `device.type == "cpu"` and this isn't already running
        inside another worker process (see `_in_worker_process`) -- e.g. one
        of `evaluate_replicability_multiproc`'s own workers -- in which case
        it silently falls back to 1 to avoid nesting process pools.

        Each worker process is pinned to `torch.get_num_threads() //
        restart_procs` intra-op threads (see
        `_init_restart_worker_backend`), so the *total* CPU budget stays
        close to whatever the calling process's own `torch.get_num_threads()`
        already is (e.g. as set by `setup_backend` from `CPUS_PER_TASK`) --
        pick `restart_procs` as a number of workers to split that budget
        across, not as extra CPUs on top of it.

    Returns
    -------
    tuple[Any, torch.Tensor, Any, RestartTally]
        `(best_decomp, best_error, best_extra, tally)`, with `best_decomp`
        already moved to CPU. `best_extra` is whatever third value the
        compute function returned for the winning restart (a
        `PARAFAC2Diagnostics` for PARAFAC2, None for CP). `tally` records
        what happened across all restarts, so the caller can tell the user
        *why* restarts were rejected rather than just how many.

    Raises
    ------
    ConvergenceError
        If no restart converged. The last failure is chained as `__cause__`
        and summarised in the message, since that is where the numbers saying
        what to change actually live.
    ValueError
        If `restart_procs >= 2` and `device` is CUDA.
    """
    if restart_procs >= 2 and device.type == "cuda":
        raise ValueError(
            f"restart_procs={restart_procs} requests multiprocessing, but the "
            "input is on CUDA. Running multiple worker processes against a "
            "CUDA context is unsafe/unreliable across GPU driver setups -- "
            "pass restart_procs=1 to run sequentially on the GPU, or move "
            "the input to CPU first to use multiple processes.",
        )
    if restart_procs >= 2 and _in_worker_process():
        if verbose_level > 0:
            print(
                f"restart_procs={restart_procs} requested, but already running "
                "inside a worker process (e.g. evaluate_replicability_multiproc's "
                "own pool) -- falling back to restart_procs=1 to avoid nesting "
                "process pools.",
            )
        restart_procs = 1

    best_error: torch.Tensor | float = torch.inf
    best_decomp: Any = None
    best_extra: Any = None
    tally = RestartTally(attempted=init_repeats)

    if restart_procs < 2:
        for i in tqdm(range(init_repeats), disable=not progress_bar):
            try:
                if method == "CP":
                    decomp, errors = compute_CP_decomposition(
                        payload,
                        random_state=i,
                        **kwargs,
                    )
                    extra = None
                else:
                    decomp, errors, extra = compute_PARAFAC2_decomposition(
                        payload,
                        random_state=i,
                        **kwargs,
                    )
            except ConvergenceError as e:
                tally.record_failure(e)
                if verbose_level > 0:
                    print(e)
                continue

            tally.record_success(extra)
            if errors[-1] < best_error:
                best_error = errors[-1]
                # Move the best result to CPU immediately to free up GPU VRAM
                best_decomp = to_cpu(decomp)
                best_extra = extra

            del decomp, errors

            # Reduce some memory issues by clearing cache when memory usage is high
            if device.type == "cuda":
                mem_reserved = torch.cuda.memory_reserved(device)
                total_mem = torch.cuda.get_device_properties(device).total_memory
                if mem_reserved / total_mem > 0.85:
                    torch.cuda.empty_cache()
            sys.stdout.flush()
    else:
        task_args = [(method, i, payload, kwargs) for i in range(init_repeats)]
        # Split the parent's own thread budget across restart_procs workers
        # rather than letting each worker default to its own (often much
        # larger) thread pool -- see _init_restart_worker_backend.
        threads_per_proc = max(1, torch.get_num_threads() // restart_procs)
        with get_context("spawn").Pool(
            restart_procs,
            initializer=_init_restart_worker_backend,
            initargs=(threads_per_proc,),
        ) as pool:
            for decomp, errors, extra, failure in tqdm(
                pool.imap_unordered(_restart_worker, task_args),
                total=init_repeats,
                disable=not progress_bar,
            ):
                if decomp is None:
                    tally.record_failure(failure)
                    if verbose_level > 0:
                        print(failure)
                    continue
                tally.record_success(extra)
                if errors[-1] < best_error:
                    best_error = errors[-1]
                    best_decomp = to_cpu(decomp)
                    best_extra = extra
                del decomp, errors

    gc.collect()
    # Force PyTorch to release its internal cached memory back to the OS/GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if best_decomp is None:
        # Chain the last failure rather than swallowing it: its message is
        # where the numbers saying what to change (iteration counts,
        # feasibility gaps, a suggested max_iter) actually live.
        raise ConvergenceError(
            f"No decomposition converged within {init_repeats} repeats"
            f"{tally.failure_summary()}",
        ) from tally.last_failure
    assert isinstance(best_error, torch.Tensor)  # guaranteed once best_decomp is set

    return best_decomp, best_error.float().cpu(), best_extra, tally


def _maybe_register_memory_efficient_khatri_rao(enabled: bool) -> None:
    """Register TensorLy's memory-efficient MTTKRP backend method, if enabled.

    `tl.tenalg.register_backend_method` is a global TensorLy backend
    registration, not specific to any one decomposition -- shared by
    `run_CP_decomposition_repeated` and `run_PARAFAC2_decomposition_repeated`,
    since both algorithms' ALS iterations rely on the same underlying
    MTTKRP operation.
    """
    if enabled:
        tl.tenalg.register_backend_method(
            "unfolding_dot_khatri_rao",
            unfolding_dot_khatri_rao_memory,
        )
        tl.tenalg.use_dynamic_dispatch()


def run_CP_decomposition_repeated(
    tensor: torch.Tensor,
    rank: int,
    max_iter: int = 5000,
    init_repeats: int = 50,
    device: torch.device = torch.device("cpu"),
    use_memory_efficient_khatri_rao: bool = True,
    verbose_level: int = 0,
    tolerance: float = 1e-5,
    progress_bar: bool = True,
    normalize: bool = False,
    allow_nan_imputation: bool = False,
    non_negative: bool = True,
    restart_procs: int = 1,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Repeatedly fit a CP/PARAFAC decomposition from random restarts.

    See `compute_CP_decomposition` for the meaning of `allow_nan_imputation`
    and `non_negative`, and `_repeat_with_restarts` for `restart_procs`.

    Notes
    -----
    Shares its option names (`max_iter`, `init_repeats`, `verbose_level`,
    `tolerance`, `normalize`, `use_memory_efficient_khatri_rao`,
    `progress_bar`, `device`, `rank`, `restart_procs`) with
    `run_PARAFAC2_decomposition_repeated` -- see that function's docstring
    for the options it doesn't share (`nn_modes` instead of `non_negative`;
    the PARAFAC2-only `solver`, `aoadmm_options`, `aoadmm_loss_tolerance`
    and `return_diagnostics`; no `allow_nan_imputation`). Kept in sync so a
    single `**kwargs` dict of shared options (e.g.
    `gMRItensor.replicability.evaluate_replicability_multiproc`'s
    `CP_kwargs`) can be forwarded to either function.
    """
    _maybe_register_memory_efficient_khatri_rao(use_memory_efficient_khatri_rao)

    kwargs = {
        "rank": rank,
        "CP_max_iter": max_iter,
        "CP_verbose_level": verbose_level,
        "CP_tolerance": tolerance,
        "normalize_factors": normalize,
        "allow_nan_imputation": allow_nan_imputation,
        "non_negative": non_negative,
    }

    def to_cpu(decomp):
        weights, factors = decomp
        return weights.float().cpu(), [f.float().cpu() for f in factors]

    (best_weights, best_factors), best_error, _, _ = _repeat_with_restarts(
        "CP",
        tensor,
        kwargs,
        to_cpu,
        init_repeats,
        device,
        verbose_level,
        progress_bar,
        restart_procs=restart_procs,
    )

    return best_weights, best_factors, best_error


def _warn_if_accepted_at_iteration_limit(
    diagnostics: PARAFAC2Diagnostics | None,
) -> None:
    """Explain an accepted fit whose own solver reports it didn't converge.

    For matcouply this combination is the normal case, not a red flag: its
    stopping criterion is on the penalized objective, which routinely keeps
    inching along after the reconstruction error has settled. Saying so
    explicitly, with the levels reached, is better than either staying silent
    (the caller never learns the iteration budget was exhausted) or raising
    (which would reject fits that are in fact good).
    """
    if diagnostics is None or not diagnostics.reached_max_iter:
        return
    if diagnostics.loss_converged is not False:
        return

    gap_text = ""
    if diagnostics.max_feasibility_gap is not None:
        gap_text = (
            f", max feasibility gap {diagnostics.max_feasibility_gap:.2e} <= "
            f"feasibility_tol {diagnostics.feasibility_tol:.1e}"
        )
    loss_text = ""
    if diagnostics.final_relative_loss_change is not None:
        loss_text = (
            f" -- relative loss change "
            f"{diagnostics.final_relative_loss_change:.1e} >= "
            f"aoadmm_loss_tolerance {diagnostics.loss_tolerance:.1e}"
        )
    warnings.warn(
        f"PARAFAC2(solver={diagnostics.solver!r}) accepted at the iteration "
        f"limit ({diagnostics.n_iter}/{diagnostics.max_iter}). The "
        f"reconstruction error has converged -- |delta rel. reconstruction "
        f"error| = {diagnostics.reconstruction_error_change:.3e} < tolerance "
        f"{diagnostics.reconstruction_tolerance:.1e}, final rel. "
        f"reconstruction error "
        f"{diagnostics.relative_reconstruction_error:.3e}{gap_text} -- but "
        f"AO-ADMM's own penalized-objective criterion was not met{loss_text}. "
        "The penalized objective is not comparable to TensorLy's criterion; "
        "acceptance is based on the reconstruction error, which is.",
        stacklevel=3,
    )


PARAFAC2Result = tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
]
PARAFAC2ResultWithDiagnostics = tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    PARAFAC2Diagnostics,
]


# Overloaded on `return_diagnostics` so callers that leave it off keep the
# plain 4-tuple type -- without this, every existing call site (e.g.
# `gMRItensor.replicability._decomposition_worker`) would have to narrow a
# union before unpacking.
@overload
def run_PARAFAC2_decomposition_repeated(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int = ...,
    init_repeats: int = ...,
    device: torch.device = ...,
    use_memory_efficient_khatri_rao: bool = ...,
    verbose_level: int = ...,
    tolerance: float = ...,
    progress_bar: bool = ...,
    normalize: bool = ...,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = ...,
    restart_procs: int = ...,
    solver: PARAFAC2Solver = ...,
    aoadmm_options: dict[str, Any] | None = ...,
    aoadmm_loss_tolerance: float = ...,
    return_diagnostics: Literal[False] = ...,
) -> PARAFAC2Result:
    ...


@overload
def run_PARAFAC2_decomposition_repeated(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int = ...,
    init_repeats: int = ...,
    device: torch.device = ...,
    use_memory_efficient_khatri_rao: bool = ...,
    verbose_level: int = ...,
    tolerance: float = ...,
    progress_bar: bool = ...,
    normalize: bool = ...,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = ...,
    restart_procs: int = ...,
    solver: PARAFAC2Solver = ...,
    aoadmm_options: dict[str, Any] | None = ...,
    aoadmm_loss_tolerance: float = ...,
    *,
    return_diagnostics: Literal[True],
) -> PARAFAC2ResultWithDiagnostics:
    ...


def run_PARAFAC2_decomposition_repeated(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int = 2000,
    init_repeats: int = 50,
    device: torch.device = torch.device("cpu"),
    use_memory_efficient_khatri_rao: bool = True,
    verbose_level: int = 0,
    tolerance: float = 1e-5,
    progress_bar: bool = True,
    normalize: bool = False,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = "auto",
    restart_procs: int = 1,
    solver: PARAFAC2Solver = "tensorly",
    aoadmm_options: dict[str, Any] | None = None,
    aoadmm_loss_tolerance: float = 1e-10,
    return_diagnostics: bool = False,
) -> PARAFAC2Result | PARAFAC2ResultWithDiagnostics:
    """Repeatedly fit a PARAFAC2 decomposition from random restarts.

    See `compute_PARAFAC2_decomposition` for `solver`, `nn_modes`,
    `aoadmm_options`, `aoadmm_loss_tolerance`, the two-threshold convergence
    scheme, and why this is not `torch.compile`-wrapped; and
    `_repeat_with_restarts` for `restart_procs`.

    The return contract is identical for both solvers -- same arity, same
    types, same shapes, same float32 CPU tensors -- so calling code never has
    to branch on `solver`.

    Parameters
    ----------
    return_diagnostics : bool, optional
        If True, append the winning restart's `PARAFAC2Diagnostics` to the
        returned tuple. Off by default so the 4-tuple contract that
        `gMRItensor.replicability._decomposition_worker` unpacks is
        unchanged.

    Notes
    -----
    Shares its option names (`max_iter`, `init_repeats`, `verbose_level`,
    `tolerance`, `normalize`, `use_memory_efficient_khatri_rao`,
    `progress_bar`, `device`, `rank`, `restart_procs`) with
    `run_CP_decomposition_repeated` -- see that function's `Notes`. The
    options that aren't shared: `nn_modes` replaces CP's flat `non_negative`
    bool (it's strictly more expressive -- it picks *which* modes are
    constrained); `solver`, `aoadmm_options`, `aoadmm_loss_tolerance` and
    `return_diagnostics` are PARAFAC2-only; and there's no
    `allow_nan_imputation`, since neither PARAFAC2 solver is wired up for
    imputation here -- unlike CP, NaN input always raises.

    This function emits `UserWarning`s rather than staying silent when the
    solver is struggling: once if the winning fit was accepted at the
    iteration limit, and once more if *most* restarts were rejected for the
    same reason (see `_build_restart_advisory`). Both carry the levels
    actually reached and name the threshold to change.

    Returns
    -------
    tuple
        `(best_weights, best_factors, best_projections, best_error)`, plus
        `best_diagnostics` when `return_diagnostics=True`.
        `best_factors = [A, B, C]` (subject, shared evolving-mode basis,
        region); `best_projections[i]` is the per-subject orthonormal
        projection needed to reconstruct that subject's own time pattern
        (`projections[i] @ best_factors[1]`). `best_error` is the relative
        reconstruction error, defined identically for both solvers and hence
        directly comparable between them.
    """
    _maybe_register_memory_efficient_khatri_rao(use_memory_efficient_khatri_rao)

    kwargs = {
        "rank": rank,
        "PARAFAC2_max_iter": max_iter,
        "PARAFAC2_verbose_level": verbose_level,
        "PARAFAC2_tolerance": tolerance,
        "normalize_factors": normalize,
        "nn_modes": nn_modes,
        "solver": solver,
        "aoadmm_options": aoadmm_options,
        "aoadmm_loss_tolerance": aoadmm_loss_tolerance,
    }

    def to_cpu(result):
        weights = result.weights.float().cpu()
        factors = [f.float().cpu() for f in result.factors]
        projections = [p.float().cpu() for p in result.projections]
        return weights, factors, projections

    (
        (best_weights, best_factors, best_projections),
        best_error,
        best_diagnostics,
        tally,
    ) = _repeat_with_restarts(
        "PARAFAC2",
        tensor_slices,
        kwargs,
        to_cpu,
        init_repeats,
        device,
        verbose_level,
        progress_bar,
        restart_procs=restart_procs,
    )

    _warn_if_accepted_at_iteration_limit(best_diagnostics)
    advisory = _build_restart_advisory(
        tally,
        solver,
        max_iter,
        tolerance,
        aoadmm_loss_tolerance,
    )
    if advisory is not None:
        warnings.warn(advisory, stacklevel=2)

    if return_diagnostics:
        return (
            best_weights,
            best_factors,
            best_projections,
            best_error,
            best_diagnostics,
        )
    return best_weights, best_factors, best_projections, best_error


def setup_backend():
    # Check if use gpu flag is passed
    # Note that if variable is not defined this will be false
    use_gpu = True if os.environ.get("GMRITENSOR_USE_GPU") == "TRUE" else False

    # Use pytorch backend from openMP + GPU support
    tl.set_backend("pytorch")
    torch.set_float32_matmul_precision("high")

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"Running on: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        slurm_cpus = os.environ.get("CPUS_PER_TASK")
        # Run sequential if number of CPUs is not made explicit
        torch.set_num_threads(int(slurm_cpus) if slurm_cpus else 1)
        print(f"Running on: Multi-CPU ({torch.get_num_threads()} threads)")
    return device
