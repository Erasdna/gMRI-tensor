"""PARAFAC2 fits, by TensorLy (ALS/HALS) or matcouply (AO-ADMM): one time
course per subject and component, coupled by the PARAFAC2 constraint.
"""
import math
from typing import Any
from typing import Literal

import tensorly as tl
import torch
from matcouply.decomposition import parafac2_aoadmm
from tensorly.parafac2_tensor import apply_parafac2_projections
from tlviz.factor_tools import degeneracy_score

from ._aoadmm import _accept_aoadmm_fit
from ._aoadmm import _AOADMM_DEFAULT_FEASIBILITY_TOL
from ._aoadmm import _build_aoadmm_options
from ._aoadmm import _nn_modes_to_non_negative
from ._aoadmm import _resolve_nn_modes
from ._aoadmm import _zero_negligible_modes
from ._aoadmm import AOADMM_DEFAULT_LOSS_TOL
from ._backend import _check_tolerances
from ._backend import _entries
from ._backend import _fit_device
from ._backend import _FitBackend
from ._backend import _matcouply_numeric_context
from ._backend import _solver_default
from ._backend import _to_solver_slices
from .models import ConvergenceError
from .models import DEGENERACY_THRESHOLD
from .models import PARAFAC2Diagnostics
from .models import PARAFAC2Model
from .models import PARAFAC2Solver


def _matcouply_kiers_form(
    admm_vars: Any,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Read `(coordinate_matrix, projections)` out of matcouply's ADMM state.

    `matcouply.penalties.Parafac2`'s auxiliary is exactly the Kiers
    parametrization. Only consulted when the caller asks for the compact
    form; the default path never touches it, since rebuilding `B_i` from it
    is what introduced per-subject sign flips.

    `auxes[1][0]` is internal matcouply layout, so it is checked rather than
    trusted.
    """
    layout_error = (
        "Could not read the PARAFAC2 basis/coordinate matrices out of "
        "matcouply's ADMM auxiliary variables; auxes[1][0] is expected to be "
        "the (basis_matrices, coordinate_matrix) pair. This is internal "
        "matcouply layout and appears to have changed"
    )
    mode_1_auxes = admm_vars.auxes[1]
    if not mode_1_auxes or not isinstance(mode_1_auxes[0], tuple):
        raise RuntimeError(f"{layout_error}.")
    if len(mode_1_auxes[0]) != 2:
        raise RuntimeError(f"{layout_error} (expected a 2-tuple).")

    basis_matrices, coordinate_matrix = mode_1_auxes[0]
    return coordinate_matrix, list(basis_matrices)


def _evolving_cross_product(evolving_states: list[torch.Tensor]) -> torch.Tensor:
    """`B_i.T @ B_i`, the PARAFAC2 invariant, averaged over subjects.

    The constraint makes this identical for every subject, so it is the only
    subject-independent summary of the evolving mode. Averaged rather than
    taken from subject 0 because matcouply's primal satisfies the constraint
    only to within the feasibility gap.
    """
    grams = torch.stack([B.T @ B for B in evolving_states])
    return grams.mean(dim=0)


def _cross_product_factor(evolving_states: list[torch.Tensor]) -> torch.Tensor:
    """A `(rank, rank)` stand-in for the ragged evolving mode.

    Any matrix whose Gram equals `B_i.T @ B_i` has the same component
    cosines as every `B_i`, which is all `degeneracy_score` needs. Built with
    `eigh` and a clamp rather than Cholesky: a collapsed component makes the
    cross-product singular, which Cholesky rejects outright.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(
        _evolving_cross_product(evolving_states),
    )
    return (eigenvectors * torch.sqrt(eigenvalues.clamp(min=0.0))).T


def _compute_PARAFAC2_matcouply(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int,
    random_state: int,
    init: str,
    verbose_level: int,
    tolerance: float | None,
    nn_modes: tuple[int, ...] | None,
    aoadmm_options: dict[str, Any] | None,
    aoadmm_loss_tolerance: float | None,
    return_projections: bool,
    negligible_rtol: float,
    compute_device: torch.device | None,
) -> tuple[PARAFAC2Model, list[torch.Tensor], PARAFAC2Diagnostics]:
    """Fit PARAFAC2 with matcouply's AO-ADMM.

    Returns matcouply's primal `cmf` essentially as-is. It is deliberately
    NOT converted to TensorLy's `(A, Delta, C) + projections` form: rebuilding
    `B_i = P_i @ Delta` runs an orthogonal Procrustes fit whose solution is
    weakly determined for a subject that barely expresses a component, and can
    land on a reflected `P_i` -- flipping that subject's whole curve.

    See `compute_PARAFAC2_decomposition` for the two-threshold convergence
    scheme this implements.
    """
    options = _build_aoadmm_options(aoadmm_options)
    feasibility_tol = options.get("feasibility_tol", _AOADMM_DEFAULT_FEASIBILITY_TOL)
    if aoadmm_loss_tolerance is None:
        aoadmm_loss_tolerance = AOADMM_DEFAULT_LOSS_TOL

    slices = (
        list(tensor_slices)
        if isinstance(tensor_slices, list)
        else [tensor_slices[i] for i in range(tensor_slices.shape[0])]
    )
    device = _fit_device(compute_device, slices[0].device)

    backend = _FitBackend(device, _entries(slices))
    with _matcouply_numeric_context(device):
        cmf, admm_vars, diagnostics = backend.output(
            backend.call(
                parafac2_aoadmm,
                backend.input(_to_solver_slices(slices, device)),
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
            ),
            torch.float64,
        )
        # matcouply leaves `weights` as None, keeping the scale in the
        # factors; the weights are ones, as for every fit here.
        _, (A, evolving_states, C) = cmf
        subject_mode, evolving_states, C = _zero_negligible_modes(
            A,
            list(evolving_states),
            C,
            nn_modes,
            negligible_rtol,
        )
        assert subject_mode is not None
        A = subject_mode
        result = PARAFAC2Model(
            weights=torch.ones(rank, dtype=A.dtype, device=A.device),
            subject_mode=A,
            evolving_states=list(evolving_states),
            label_mode=C,
            kiers=_matcouply_kiers_form(admm_vars) if return_projections else None,
        )

    # Below here torch's defaults are restored, so neither float64 nor the
    # device leaks into what we return -- including on the error paths.
    errors, parafac2_diagnostics = _accept_aoadmm_fit(
        diagnostics,
        max_iter,
        tolerance,
        aoadmm_loss_tolerance,
        feasibility_tol,
        nn_modes,
        "PARAFAC2",
    )
    return result, errors, parafac2_diagnostics


def _compute_PARAFAC2_tensorly(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int,
    random_state: int,
    init: str,
    verbose_level: int,
    tolerance: float | None,
    nn_modes: tuple[int, ...] | None,
    return_projections: bool,
) -> tuple[PARAFAC2Model, list[torch.Tensor], PARAFAC2Diagnostics]:
    """Fit PARAFAC2 with TensorLy's ALS, converted to coupled-matrix form,
    in float64 on every backend and device.

    `apply_parafac2_projections` is TensorLy's own public converter and
    already returns exactly matcouply's native `(weights, (A, B_is, C))`
    structure, so both solvers share one contract.
    """
    if tolerance is None:
        tolerance = _solver_default(tl.decomposition.parafac2)
    # float64 on every backend and device, as the matcouply solver runs.
    tensor_slices = (
        [s.double() for s in tensor_slices]
        if isinstance(tensor_slices, list)
        else tensor_slices.double()
    )
    first = tensor_slices[0]
    backend = _FitBackend(first.device, _entries(tensor_slices))
    result, errors = backend.output(
        backend.call(
            tl.decomposition.parafac2,
            backend.input(tensor_slices),
            rank=rank,
            n_iter_max=max_iter,
            tol=tolerance,
            return_errors=True,
            random_state=random_state,
            verbose=verbose_level,
            init=init,
            nn_modes=list(nn_modes) if nn_modes else None,
        ),
        first.dtype,
    )
    rec_errors = [float(error) for error in errors]
    # TensorLy stops exactly when two successive recorded errors differ by
    # less than `tol`, so that is the convergence test. Counting the errors
    # is not: its line-search iterations replace the last error instead of
    # adding one, so a fit can exhaust `max_iter` with far fewer recorded.
    delta = abs(rec_errors[-2] - rec_errors[-1]) if len(rec_errors) >= 2 else math.inf
    if not delta < tolerance:
        raise ConvergenceError(
            "Decomposition did not converge within the maximum iteration count",
            reason="max_iter",
            n_iter=len(errors),
        )
    diagnostics = PARAFAC2Diagnostics(
        solver="tensorly",
        n_iter=len(errors),
        max_iter=max_iter,
        reached_max_iter=False,
        relative_reconstruction_error=rec_errors[-1],
        reconstruction_error_change=delta,
        reconstruction_tolerance=tolerance,
        nn_modes=nn_modes,
        message="converged",
    )
    weights, (A, evolving_states, C) = apply_parafac2_projections(result)
    model = PARAFAC2Model(
        weights=weights,
        subject_mode=A,
        evolving_states=list(evolving_states),
        label_mode=C,
        kiers=(result.factors[1], list(result.projections))
        if return_projections
        else None,
    )
    return model, errors, diagnostics


def compute_PARAFAC2_decomposition(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    PARAFAC2_max_iter: int = 500,
    random_state: int = 0,
    init: str = "random",
    PARAFAC2_verbose_level: int = 0,
    PARAFAC2_tolerance: float | None = None,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = "auto",
    solver: PARAFAC2Solver = "matcouply",
    aoadmm_options: dict[str, Any] | None = None,
    aoadmm_loss_tolerance: float | None = None,
    return_projections: bool = False,
    negligible_rtol: float = 1e-6,
    compute_device: torch.device | None = None,
) -> tuple[PARAFAC2Model, list[torch.Tensor], PARAFAC2Diagnostics]:
    """Compute a single PARAFAC2 decomposition attempt.

    PARAFAC2 allows one mode (here: time) to vary in size per slice (here:
    per subject). `tensor_slices` may be a regular 3D tensor or a list of 2D
    slices sharing their column count.

    Returns a `PARAFAC2Model` in coupled-matrix form: `subject_mode`
    (subjects x rank), `evolving_states[i]` (that subject's own
    `(n_timepoints_i, rank)` time course) and `label_mode` (labels x rank).
    Both solvers return this same form, so downstream code never branches on
    `solver`. `return_projections=True` additionally fills `kiers` with the
    compact `(coordinate_matrix, projections)` parametrization.

    Choosing a solver
    -----------------
    `solver="matcouply"` (default) uses AO-ADMM, which can enforce
    non-negativity on mode 1 (the evolving/time mode) where TensorLy's ALS
    cannot, and offers L1/L2/TV, unimodality and bound constraints via
    `aoadmm_options`. `solver="tensorly"` uses ALS/HALS.

    `nn_modes="auto"` resolves to `(0, 2)` for TensorLy and `(0, 1, 2)` for
    matcouply; pass `None` for an unconstrained fit. Asking TensorLy for
    mode 1 raises, since it would otherwise only warn and leave the mode
    unconstrained.

    Tolerances
    ----------
    For TensorLy `PARAFAC2_tolerance` is its `tol` (None: TensorLy's
    default). For matcouply `aoadmm_loss_tolerance` is its `tol` on the
    penalized objective (None: `AOADMM_DEFAULT_LOSS_TOL`); a fit that runs
    out of iterations before meeting it is kept, as matcouply returns it,
    with a warning. Passing `PARAFAC2_tolerance` instead gates on the
    reconstruction error, accepting fits whose error has settled to within
    it -- even at the iteration limit -- and rejecting those that have not.

    `compute_device` is the device the fit runs on; None (default) uses the
    input's own device. It lets CPU-resident slices be fitted on the GPU
    without a host-side copy -- for matcouply, which needs float64, the cast
    happens on `compute_device` (see `_to_solver_slices`). A CPU fit runs on
    TensorLy's numpy backend in float64 (`_fits_on_numpy`); the returned
    tensors are torch either way.

    Returns
    -------
    tuple[Parafac2Tensor, list[torch.Tensor], PARAFAC2Diagnostics]
        The fit, its per-iteration relative reconstruction errors, and what
        it achieved. See `PARAFAC2Diagnostics` for which fields are
        comparable across solvers.

    Raises
    ------
    ValueError
        NaN input, `rank` above the label count or a subject's time-point
        count (see `_check_parafac2_rank`), unknown `solver`, an
        AO-ADMM-only option passed with `solver="tensorly"`, or `nn_modes`
        including mode 1 for TensorLy.
    ConvergenceError
        The acceptance gate failed, matcouply's constraints were left
        infeasible, or the result is degenerate.
    """
    _check_tolerances(
        PARAFAC2_tolerance=PARAFAC2_tolerance,
        aoadmm_loss_tolerance=aoadmm_loss_tolerance,
    )
    _check_parafac2_rank(tensor_slices, rank)
    slices_to_check = (
        tensor_slices if isinstance(tensor_slices, list) else [tensor_slices]
    )
    if any(torch.isnan(s).any() for s in slices_to_check):
        # TensorLy's parafac2 has no mask support; matcouply accepts a `mask`
        # but it is not wired up here.
        raise ValueError(
            "tensor_slices contains NaN values, but neither PARAFAC2 solver is "
            "set up for NaN imputation here -- remove or impute missing values "
            "first (e.g. load_tensor_from_parquet(..., model='parafac2'), "
            "which drops non-finite columns instead).",
        )

    if solver not in ("tensorly", "matcouply"):
        raise ValueError(
            f"Unknown PARAFAC2 solver {solver!r}; expected 'tensorly' or 'matcouply'.",
        )
    if solver != "matcouply":
        if aoadmm_options is not None:
            raise ValueError(
                f"aoadmm_options only applies to solver='matcouply', not {solver!r}.",
            )
        if aoadmm_loss_tolerance is not None:
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
            nn_modes=resolved_nn_modes,
            aoadmm_options=aoadmm_options,
            aoadmm_loss_tolerance=aoadmm_loss_tolerance,
            return_projections=return_projections,
            negligible_rtol=negligible_rtol,
            compute_device=compute_device,
        )
    else:
        if compute_device is not None:
            tensor_slices = (
                [s.to(compute_device) for s in tensor_slices]
                if isinstance(tensor_slices, list)
                else tensor_slices.to(compute_device)
            )
        result, errors, diagnostics = _compute_PARAFAC2_tensorly(
            tensor_slices,
            rank=rank,
            max_iter=PARAFAC2_max_iter,
            random_state=random_state,
            init=init,
            verbose_level=PARAFAC2_verbose_level,
            tolerance=PARAFAC2_tolerance,
            nn_modes=resolved_nn_modes,
            return_projections=return_projections,
        )

    # `degeneracy_score` needs a regular mode 1, but the evolving states are
    # ragged. Component cosines depend only on the Gram matrix, and PARAFAC2
    # makes `B_i.T @ B_i` subject-independent, so a factor built from that
    # cross-product is an exact stand-in.
    w = result.weights.float()
    f = [
        result.subject_mode.float(),
        _cross_product_factor(result.evolving_states).float(),
        result.label_mode.float(),
    ]
    if degeneracy_score((w, f)) < DEGENERACY_THRESHOLD:
        raise ConvergenceError(
            "Decomposition is degenerate",
            reason="degenerate",
            n_iter=diagnostics.n_iter,
        )

    return result, errors, diagnostics


def _check_parafac2_rank(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
) -> None:
    """Raise if `rank` is too large for PARAFAC2 on these slices.

    Each subject's `(n_timepoints_i, n_labels)` slice has rank at most
    `min(n_timepoints_i, n_labels)`, and its projection `P_i` needs `rank`
    orthonormal columns of length `n_timepoints_i`. So `rank` may exceed
    neither the label count nor any subject's time-point count. Both solvers
    would otherwise fail inside the library with a shape error, or (matcouply
    with too few labels) return a non-unique fit.

    Raises
    ------
    ValueError
        Naming the limit that was exceeded.
    """
    if isinstance(tensor_slices, list):
        n_timepoints = [s.shape[0] for s in tensor_slices]
        n_labels = tensor_slices[0].shape[1]
    else:
        n_timepoints = [tensor_slices.shape[1]] * tensor_slices.shape[0]
        n_labels = tensor_slices.shape[2]
    if rank > n_labels:
        raise ValueError(
            f"PARAFAC2 rank ({rank}) exceeds the number of labels ({n_labels}); "
            f"use rank <= {n_labels}.",
        )
    shortest = min(range(len(n_timepoints)), key=n_timepoints.__getitem__)
    if rank > n_timepoints[shortest]:
        raise ValueError(
            f"PARAFAC2 rank ({rank}) exceeds the time-point count of subject "
            f"{shortest}, which has only {n_timepoints[shortest]} time points; "
            f"use rank <= {n_timepoints[shortest]}, or drop short subjects "
            "with --min-timepoints.",
        )
