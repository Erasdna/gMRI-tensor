"""Types of the decomposition API: models, diagnostics, errors and the
restart tally.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Any
from typing import Literal

import torch


#: Library used to fit PARAFAC2. Distinct from TensorLy's compute backend
#: (numpy/pytorch), which `setup_backend` configures.
PARAFAC2Solver = Literal["tensorly", "matcouply"]


#: A decomposition method, as the restart machinery names it.
Method = Literal["CP", "PARAFAC2", "CMF"]


#: A fit whose degeneracy score (`tlviz.factor_tools.degeneracy_score`, the
#: most negative congruence between two components over all modes) is below
#: this is rejected: two components that nearly cancel each other.
DEGENERACY_THRESHOLD = -0.85

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
    `reason="reconstruction"`. `n_iter` is how many iterations the rejected
    fit ran, where the raiser knows it; it is recorded with the restart's
    timing.
    """

    def __init__(
        self,
        message: str,
        reason: ConvergenceFailureReason | None = None,
        level_reached: float | None = None,
        suggested_max_iter: int | None = None,
        n_iter: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.level_reached = level_reached
        self.suggested_max_iter = suggested_max_iter
        self.n_iter = n_iter


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
    #: The gate on the reconstruction-error change; None when matcouply's own
    #: criterion was the only gate.
    reconstruction_tolerance: float | None
    #: Non-negativity actually applied, with `"auto"` already resolved.
    nn_modes: tuple[int, ...] | None = None
    #: Largest per-component non-negativity violation in the *returned*
    #: factors, keyed by mode, relative to that component's own scale.
    max_nn_violation: dict[int, float] | None = None
    # matcouply only; None for tensorly.
    loss_converged: bool | None = None
    loss_tolerance: float | None = None
    final_relative_loss_change: float | None = None
    feasible: bool | None = None
    max_feasibility_gap: float | None = None
    feasibility_tol: float | None = None
    message: str = ""


@dataclass(frozen=True)
class PARAFAC2Model:
    """A fitted PARAFAC2 model, in coupled-matrix form.

    `evolving_states[i]` is subject `i`'s own `(n_timepoints_i, rank)` time
    course -- directly interpretable, and what both solvers return. This
    is matcouply's native form; TensorLy is converted into it with its public
    `apply_parafac2_projections`.

    `kiers` holds the compact `(coordinate_matrix, projections)`
    parametrization with `evolving_states[i] == projections[i] @
    coordinate_matrix`. It is populated only when the caller asks for it, and
    is deliberately not the primary form: `coordinate_matrix` is identifiable
    only up to an orthogonal rotation (`Delta -> Q Delta` with `P_i -> P_i
    Q.T` leaves the model unchanged), so it must never be compared across
    fits.

    `weights` are always ones: no fit is normalised, so the scale stays in
    the factors. Plotting and post-processing normalise (`scale_mode`,
    `model_io.component_weights`).
    """

    weights: torch.Tensor
    subject_mode: torch.Tensor
    evolving_states: list[torch.Tensor]
    label_mode: torch.Tensor
    kiers: tuple[torch.Tensor, list[torch.Tensor]] | None = None


@dataclass(frozen=True)
class CMFModel:
    """A fitted non-negative coupled matrix factorization (CMF).

    Subject `i`'s `(n_timepoints_i, n_labels)` slice is modelled as
    `evolving_states[i] @ label_mode.T`. This is `PARAFAC2Model`'s form
    without the PARAFAC2 constraint: nothing ties the subjects' time courses
    together, so they may differ in shape, not only in amplitude.

    The subject weights are fixed at one during the fit
    (`cmf_aoadmm(update_A=False)`): without the constraint their scale is
    interchangeable with the `evolving_states[i]`, which therefore carry each
    subject's amplitude. `subject_mode` is derived, not fitted:
    `subject_mode[i, r]` is the RMS of `evolving_states[i][:, r]` over that
    subject's time points, so it compares subjects scanned a different number
    of times. `weights` are always ones, as for `PARAFAC2Model`.
    """

    weights: torch.Tensor
    subject_mode: torch.Tensor
    evolving_states: list[torch.Tensor]
    label_mode: torch.Tensor


#: Models whose mode 1 is a ragged list of per-subject time courses.
EVOLVING_MODELS = (PARAFAC2Model, CMFModel)


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
        """Count an accepted restart (and whether it hit the iteration limit)."""
        self.succeeded += 1
        if isinstance(diagnostics, PARAFAC2Diagnostics) and (
            diagnostics.reached_max_iter
        ):
            self.hit_iteration_limit += 1

    def record_failure(self, error: ConvergenceError | None) -> None:
        """Count a rejected restart under its reason, keeping its numbers."""
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
        """Rejected restarts, all reasons together."""
        return sum(self.reasons.values())

    def median_level(self, reason: str) -> float | None:
        """Median level reached by the restarts rejected for `reason`."""
        levels = sorted(self.levels.get(reason, []))
        if not levels:
            return None
        return levels[len(levels) // 2]

    def suggested_max_iter(self) -> int | None:
        """Median of the rejected restarts' suggested `max_iter`."""
        if not self.suggested_max_iters:
            return None
        ordered = sorted(self.suggested_max_iters)
        return ordered[len(ordered) // 2]

    def failure_summary(self) -> str:
        """`" (rejected: N reason, ...)"` plus the last failure, for messages;
        empty if nothing failed."""
        if not self.reasons:
            return ""
        breakdown = ", ".join(
            f"{count} {reason}" for reason, count in self.reasons.most_common()
        )
        summary = f" (rejected: {breakdown})"
        if self.last_failure is not None:
            summary += f". Last failure: {self.last_failure}"
        return summary


PARAFAC2Result = tuple[PARAFAC2Model, torch.Tensor]


PARAFAC2ResultWithDiagnostics = tuple[
    PARAFAC2Model,
    torch.Tensor,
    PARAFAC2Diagnostics,
]


CMFResult = tuple[CMFModel, torch.Tensor]


CMFResultWithDiagnostics = tuple[CMFModel, torch.Tensor, PARAFAC2Diagnostics]
