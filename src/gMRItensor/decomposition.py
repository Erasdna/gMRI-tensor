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

#: Library used to fit PARAFAC2. Distinct from TensorLy's compute backend
#: (numpy/pytorch), which `setup_backend` configures.
PARAFAC2Solver = Literal["tensorly", "matcouply"]

ConvergenceFailureReason = Literal[
    "max_iter",
    "reconstruction",
    "feasibility",
    "degenerate",
]


class ConvergenceError(Exception):
    """Raised when a single decomposition attempt fails to converge.

    `reason` is tallied across restarts to produce advice on what to change
    (`_build_restart_advisory`). `suggested_max_iter` is set only for
    `reason="reconstruction"`.
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
    """What a PARAFAC2 fit achieved. Populated for both solvers.

    Comparable across solvers: `relative_reconstruction_error`
    (``||X - X_hat|| / ||X||``, same definition in both) and
    `reconstruction_error_change` against `reconstruction_tolerance`, the
    acceptance gate.

    NOT comparable: `loss_converged` measures matcouply's penalized
    objective, which has no TensorLy analogue -- a fit with
    `loss_converged=False` is routinely better than a TensorLy fit reporting
    convergence. `n_iter` counts different units of work per solver.
    `feasible`/`max_feasibility_gap` are matcouply-only.

    There is no single `converged` field, since it would invite the
    cross-solver comparison that does not hold.
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

#: matcouply's own default, used when reporting the feasibility gap.
_AOADMM_DEFAULT_FEASIBILITY_TOL: float = 1e-4

#: Iterations of error history used to extrapolate a suggested `max_iter`.
_DECAY_FIT_WINDOW: int = 50

#: Cap on how far past the available history `_suggest_max_iter` extrapolates.
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

    `allow_nan_imputation` treats NaN entries as missing and imputes them
    from the model's own reconstruction each iteration, via TensorLy's
    `mask`. Off by default, where NaN input raises instead. PARAFAC2 has no
    equivalent (see `compute_PARAFAC2_decomposition`).

    `non_negative` defaults to True, appropriate for a tracer signal that is
    physically non-negative.
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

    matcouply mixes `tl.tensor(numpy_array)` (always float64 CPU under the
    pytorch backend) with `tl.eye`/`tl.zeros` (which follow torch's
    defaults), and never threads `tl.context` through. Under this package's
    float32 CUDA setup the two disagree and matcouply raises a dtype or
    device mismatch. float32 is unreachable without patching matcouply;
    results are cast back on the way out.

    Scoped per fit rather than set in `setup_backend`, which would push the
    CP path to float64 too. Spawned restart workers reach it automatically
    via `_restart_worker`.

    These are process-global torch settings, safe only because restarts are
    process-parallel rather than threaded.
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

    `(0, 2)` for TensorLy, whose ALS cannot constrain mode 1, and
    `(0, 1, 2)` for matcouply, whose AO-ADMM can. A sentinel is needed
    because `None` already means "unconstrained".
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

    Unknown and wrapper-managed keys raise rather than being forwarded, so a
    typo surfaces immediately instead of being swallowed by matcouply.
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

    Reproduces TensorLy's `normalize_factors` semantics, which matcouply
    lacks. Well defined for PARAFAC2 because the projections are orthonormal
    (`||P_i @ B[:, r]|| == ||B[:, r]||`), so normalizing the shared
    coordinate matrix rescales every subject's time curve equally.
    """
    scaled_weights = weights.clone()
    scaled_factors = []
    for factor in factors:
        norms = torch.linalg.norm(factor, dim=0)
        # All-zero columns contribute no weight either way; avoid dividing by 0.
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

    matcouply returns `(weights, (A, B_is, C))` with `weights=None` and
    `B_is` a ragged list of per-slice `(J_i, rank)` matrices, rather than the
    `(weights, [A, B, C], projections)` form the rest of this package expects.

    `matcouply.penalties.Parafac2`'s auxiliary variable is exactly the Kiers
    parametrization `B_i = P_i @ B` -- orthonormal basis matrices plus the
    shared `rank x rank` coordinate matrix -- so reading it back recovers
    TensorLy's representation to ~1e-7 relative error. This is what lets
    downstream consumers work unchanged for both solvers.

    `factors[1]` (the coordinate matrix) is NOT non-negative even when mode 1
    is constrained; non-negativity holds on `projections[i] @ factors[1]`.
    """
    _, (A, _B_is, C) = cmf
    mode_1_auxes = admm_vars.auxes[1]
    # auxes[1][0] holding the (basis_matrices, coordinate_matrix) pair is
    # internal matcouply layout, so check it rather than trusting it.
    layout_error = (
        "Could not read the PARAFAC2 basis/coordinate matrices out of "
        "matcouply's ADMM auxiliary variables; auxes[1][0] is expected to be "
        "the (basis_matrices, coordinate_matrix) pair. This is internal "
        "matcouply layout and appears to have changed"
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

    # Below here torch's defaults are restored, so neither float64 nor the
    # device leaks into what we return -- including on the error paths.
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

    PARAFAC2 allows one mode (here: time) to vary in size per slice (here:
    per subject). `tensor_slices` may be a regular 3D tensor or a list of 2D
    slices sharing their column count.

    `factors = [A, B, C]` are always regular matrices: `A` (subjects x rank),
    `B` (rank x rank, shared evolving-mode basis), `C` (labels x rank). The
    subject-specific time pattern is `projections[i] @ B` -- see
    `gMRItensor.plotting.evolving_mode.reconstruct_evolving_factors`. Both
    solvers return this same shape, so downstream code never branches on
    `solver`.

    Choosing a solver
    -----------------
    `solver="tensorly"` (default) uses ALS/HALS. `solver="matcouply"` uses
    AO-ADMM, which can enforce non-negativity on mode 1 (the evolving/time
    mode) where TensorLy's ALS cannot, and offers L1/L2/TV, unimodality and
    bound constraints via `aoadmm_options`.

    `nn_modes="auto"` resolves to `(0, 2)` for TensorLy and `(0, 1, 2)` for
    matcouply; pass `None` for an unconstrained fit. Asking TensorLy for
    mode 1 raises, since it would otherwise only warn and leave the mode
    unconstrained.

    With matcouply, `factors[1]` is AO-ADMM's coordinate matrix and carries
    negative entries even under a fully constrained fit; non-negativity holds
    on `projections[i] @ factors[1]`. matcouply also runs in float64
    regardless of `setup_backend` (see `_matcouply_numeric_context`), and its
    non-random `init` options ignore `random_state`, making restarts
    identical.

    Convergence: AO-ADMM has two thresholds
    ---------------------------------------
    TensorLy's `tol` tests the relative reconstruction error; matcouply's
    tests the penalized objective. These are not the same stopping rule, and
    matcouply's own convergence flag is routinely False on fits better than a
    TensorLy fit reporting success -- so it cannot gate acceptance. Hence:

    - `aoadmm_loss_tolerance` becomes matcouply's `tol` and governs only when
      AO-ADMM stops iterating. Defaults tighter than matcouply's own 1e-8,
      since a loose value lets it stop while the reconstruction error is
      still moving, wasting the restart on the gate below.
    - `PARAFAC2_tolerance` is the acceptance gate and means the same thing
      for both solvers: `abs(rec[-2] - rec[-1]) < tolerance`.

    Unlike `compute_CP_decomposition`, neither solver is `torch.compile`-d:
    the ragged per-slice loop cannot be traced into one graph, so compiling
    adds overhead without a speedup.

    Returns
    -------
    tuple[Parafac2Tensor, list[torch.Tensor], PARAFAC2Diagnostics]
        The fit, its per-iteration relative reconstruction errors, and what
        it achieved. See `PARAFAC2Diagnostics` for which fields are
        comparable across solvers.

    Raises
    ------
    ValueError
        NaN input, unknown `solver`, an AO-ADMM-only option passed with
        `solver="tensorly"`, or `nn_modes` including mode 1 for TensorLy.
    ConvergenceError
        The acceptance gate failed, matcouply's constraints were left
        infeasible, or the result is degenerate.
    """
    slices_to_check = (
        tensor_slices if isinstance(tensor_slices, list) else [tensor_slices]
    )
    if any(torch.isnan(s).any() for s in slices_to_check):
        # TensorLy's parafac2 has no mask support; matcouply accepts a `mask`
        # but it is not wired up here.
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

    # No branching needed: the matcouply adapter returns a real
    # Parafac2Tensor with real weights.
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

    Only used by `_repeat_with_restarts`'s parallel (`restart_procs >= 2`)
    path; the sequential path calls the compute functions in-process, so it
    also works on GPU and reuses the `torch.compile` cache across restarts.

    Returns `(decomp, errors, diagnostics, None)` on success and
    `(None, None, None, exc)` on `ConvergenceError` rather than raising, so
    one failed restart does not kill the whole `Pool.imap_unordered`. The
    exception is returned so the parallel path can tally failure reasons like
    the sequential one; failure and diagnostics get separate slots so neither
    is read as the other. `diagnostics` is None for CP.
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
    """Pool initializer: set TensorLy's backend and cap this worker's threads.

    Both settings are process-local and so do not survive "spawn".
    Without the thread cap each worker falls back to a thread pool sized to
    all visible cores, so `restart_procs` workers oversubscribe the CPU
    badly. `_repeat_with_restarts` picks `num_threads` to keep
    `restart_procs * num_threads` near the parent's `torch.get_num_threads()`.
    """
    tl.set_backend("pytorch")
    torch.set_num_threads(num_threads)


def _in_worker_process() -> bool:
    """True if already running inside a multiprocessing worker.

    Used to refuse restart-level multiprocessing inside e.g.
    `evaluate_replicability_multiproc`'s pool: a second layer of processes
    adds startup and backend-init overhead without real parallelism, since
    the outer pool already uses all requested workers.
    """
    return current_process().name != "MainProcess"


class RestartTally:
    """What happened across a run's random restarts.

    Records *why* each restart was rejected so `_build_restart_advisory` can
    tell the caller which knob to turn, rather than a run that mostly ran out
    of iterations looking identical to a healthy one.
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
    """Advise on thresholds when restarts are systematically struggling.

    Fires only on a pattern (at least half the restarts), never on a single
    unlucky one, so the advisory stays worth reading. Returns None when the
    run looks healthy.
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
            # Within an order of magnitude, more iterations will close it;
            # far off points at the model instead (rank, init, constraints).
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
        Which compute function to call for each restart.
    payload : Any
        The `tensor`/`tensor_slices` argument for that function.
    kwargs : dict[str, Any]
        Its remaining arguments; `random_state` is filled in per restart.
    to_cpu : Callable[[Any], Any]
        Moves the winning `decomp` to CPU/float precision.
    init_repeats : int
        Number of random restarts to try.
    device : torch.device
        Device the input lives on. Used for CUDA memory management and to
        reject `restart_procs >= 2`.
    verbose_level : int
        If > 0, prints each `ConvergenceError` encountered.
    progress_bar : bool
        Whether to show a tqdm progress bar over the restarts.
    restart_procs : int, optional
        Worker processes to spread restarts across; 1 (default) runs
        sequentially in-process. Falls back to 1 inside another worker
        process, to avoid nesting pools.

        Each worker is pinned to `torch.get_num_threads() // restart_procs`
        threads, so this splits the caller's existing CPU budget rather than
        adding to it.

    Returns
    -------
    tuple[Any, torch.Tensor, Any, RestartTally]
        `(best_decomp, best_error, best_extra, tally)`, `best_decomp` already
        on CPU. `best_extra` is the winning restart's `PARAFAC2Diagnostics`,
        or None for CP.

    Raises
    ------
    ConvergenceError
        If no restart converged. The last failure is chained as `__cause__`,
        since that carries the numbers saying what to change.
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

            # Clear the cache when VRAM is running tight.
            if device.type == "cuda":
                mem_reserved = torch.cuda.memory_reserved(device)
                total_mem = torch.cuda.get_device_properties(device).total_memory
                if mem_reserved / total_mem > 0.85:
                    torch.cuda.empty_cache()
            sys.stdout.flush()
    else:
        task_args = [(method, i, payload, kwargs) for i in range(init_repeats)]
        # Split the parent's thread budget rather than letting each worker
        # default to its own pool -- see _init_restart_worker_backend.
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
    # Release PyTorch's cached memory back to the OS/GPU.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if best_decomp is None:
        # Chain the last failure: its message carries the numbers saying what
        # to change (iteration counts, feasibility gaps, suggested max_iter).
        raise ConvergenceError(
            f"No decomposition converged within {init_repeats} repeats"
            f"{tally.failure_summary()}",
        ) from tally.last_failure
    assert isinstance(best_error, torch.Tensor)  # guaranteed once best_decomp is set

    return best_decomp, best_error.float().cpu(), best_extra, tally


def _maybe_register_memory_efficient_khatri_rao(enabled: bool) -> None:
    """Register TensorLy's memory-efficient MTTKRP backend method, if enabled.

    A global TensorLy registration shared by both decompositions, since both
    rely on the same underlying MTTKRP operation.
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

    See `compute_CP_decomposition` for `allow_nan_imputation` and
    `non_negative`, and `_repeat_with_restarts` for `restart_procs`.

    Shared option names are kept in sync with
    `run_PARAFAC2_decomposition_repeated` so one `**kwargs` dict can be
    forwarded to either (as `evaluate_replicability_multiproc` does). Only
    `non_negative` and `allow_nan_imputation` are CP-specific.
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

    Normal for matcouply, whose criterion is on the penalized objective and
    keeps inching along after the reconstruction error has settled. Reporting
    it beats staying silent (the caller never learns the budget was
    exhausted) or raising (which would reject good fits).
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
# plain 4-tuple type instead of having to narrow a union before unpacking.
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
    `aoadmm_options`, `aoadmm_loss_tolerance` and the two-threshold
    convergence scheme; `_repeat_with_restarts` for `restart_procs`.

    The return contract is identical for both solvers -- same arity, types,
    shapes and float32 CPU tensors -- so calling code never branches on
    `solver`. `return_diagnostics=True` appends the winning restart's
    `PARAFAC2Diagnostics`; it is off by default so the 4-tuple contract
    `gMRItensor.replicability` unpacks stays unchanged.

    Option names are kept in sync with `run_CP_decomposition_repeated` so one
    `**kwargs` dict routes to either. Not shared: `nn_modes` replaces CP's
    `non_negative` (it picks *which* modes are constrained); `solver`,
    `aoadmm_options`, `aoadmm_loss_tolerance` and `return_diagnostics` are
    PARAFAC2-only; and there is no `allow_nan_imputation`, since NaN input
    always raises here.

    Emits a `UserWarning` rather than staying silent when the solver is
    struggling: once if the winning fit was accepted at the iteration limit,
    and once if most restarts were rejected for the same reason (see
    `_build_restart_advisory`).

    Returns
    -------
    tuple
        `(best_weights, best_factors, best_projections, best_error)`, plus
        `best_diagnostics` when `return_diagnostics=True`.
        `best_factors = [A, B, C]` (subject, shared evolving-mode basis,
        region); `best_projections[i] @ best_factors[1]` reconstructs subject
        `i`'s time pattern. `best_error` is the relative reconstruction
        error, defined identically for both solvers and so comparable
        between them.
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
    """Configure TensorLy's compute backend and return the device to use.

    Reads `GMRITENSOR_USE_GPU` and, on CPU, `CPUS_PER_TASK` (SLURM).
    """
    use_gpu = True if os.environ.get("GMRITENSOR_USE_GPU") == "TRUE" else False

    # pytorch backend, for OpenMP + GPU support.
    tl.set_backend("pytorch")
    torch.set_float32_matmul_precision("high")

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"Running on: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        slurm_cpus = os.environ.get("CPUS_PER_TASK")
        # Sequential unless the CPU count is made explicit.
        torch.set_num_threads(int(slurm_cpus) if slurm_cpus else 1)
        print(f"Running on: Multi-CPU ({torch.get_num_threads()} threads)")
    return device
