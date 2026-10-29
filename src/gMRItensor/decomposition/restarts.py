"""Repeated random restarts: one fit per seed (`_fit_one`), the best of
them, the warnings about them, and the `run_*_repeated` runners.
"""
import gc
import inspect
import math
import sys
import warnings
from collections.abc import Callable
from dataclasses import replace
from multiprocessing import get_context
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import overload

import torch
from tqdm import tqdm

from ._backend import _fit_device
from ._backend import _float64_on_cpu
from ._backend import _format_tolerance
from ._backend import _in_worker_process
from ._backend import _init_restart_worker_backend
from ._backend import _maybe_register_memory_efficient_khatri_rao
from ._backend import _release_cuda_cache
from ._backend import _to_cpu
from .cmf import compute_CMF_decomposition
from .cp import compute_CP_decomposition
from .models import CMFModel
from .models import CMFResult
from .models import CMFResultWithDiagnostics
from .models import ConvergenceError
from .models import Method
from .models import PARAFAC2Diagnostics
from .models import PARAFAC2Model
from .models import PARAFAC2Result
from .models import PARAFAC2ResultWithDiagnostics
from .models import PARAFAC2Solver
from .models import RestartTally
from .parafac2 import _check_parafac2_rank
from .parafac2 import compute_PARAFAC2_decomposition


def _resolve_options(
    method: Method,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Every `run_*_repeated` option, `options` filled in with its defaults.

    Bound against the public signature, so the defaults have one source and
    an unknown option raises `TypeError` exactly as calling it would.
    """
    runners: dict[str, Callable[..., Any]] = {
        "CP": run_CP_decomposition_repeated,
        "PARAFAC2": run_PARAFAC2_decomposition_repeated,
        "CMF": run_CMF_decomposition_repeated,
    }
    if method not in runners:
        raise ValueError(f"Unknown decomposition method: {method!r}")
    function = runners[method]
    bound = inspect.signature(function).bind_partial(**options)
    bound.apply_defaults()
    return dict(bound.arguments)


def _compute_kwargs(
    method: Method,
    rank: int,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    """Map resolved `run_*_repeated` options to `compute_*` arguments."""
    if method == "CP":
        return {
            "rank": rank,
            "CP_max_iter": resolved["max_iter"],
            "CP_verbose_level": resolved["verbose_level"],
            "CP_tolerance": resolved["tolerance"],
            "allow_nan_imputation": resolved["allow_nan_imputation"],
            "non_negative": resolved["non_negative"],
        }
    if method == "CMF":
        return {
            "rank": rank,
            "CMF_max_iter": resolved["max_iter"],
            "CMF_verbose_level": resolved["verbose_level"],
            "CMF_tolerance": resolved["tolerance"],
            "nn_modes": resolved["nn_modes"],
            "aoadmm_options": resolved["aoadmm_options"],
            "aoadmm_loss_tolerance": resolved["aoadmm_loss_tolerance"],
            "negligible_rtol": resolved["negligible_rtol"],
            "compute_device": resolved["compute_device"],
        }
    return {
        "rank": rank,
        "PARAFAC2_max_iter": resolved["max_iter"],
        "PARAFAC2_verbose_level": resolved["verbose_level"],
        "PARAFAC2_tolerance": resolved["tolerance"],
        "nn_modes": resolved["nn_modes"],
        "solver": resolved["solver"],
        "aoadmm_options": resolved["aoadmm_options"],
        "aoadmm_loss_tolerance": resolved["aoadmm_loss_tolerance"],
        "return_projections": resolved["return_projections"],
        "negligible_rtol": resolved["negligible_rtol"],
        "compute_device": resolved["compute_device"],
    }


class FitOutcome(NamedTuple):
    """One restart's result, as `_fit_one` returns it.

    `model` is None, and `failure` set, when the fit was rejected. `n_iter`
    is how many iterations it ran (None when unknown), for the restart's
    timing.
    """

    model: Any
    error: float | None
    diagnostics: PARAFAC2Diagnostics | None
    failure: ConvergenceError | None
    n_iter: int | None


def _fit_one(args: tuple[Method, int, Any, dict[str, Any]]) -> FitOutcome:
    """Run a single random-restart attempt, returning failure as a value.

    The one place a seed is fitted: both paths of `_repeat_with_restarts`
    and `jobs.fit_task` go through it, so they cannot drift apart.

    On success `model` is already CPU float32 and `error` is the final
    relative reconstruction error. A `ConvergenceError` is returned as
    `failure` rather than raised, so one failed restart does not kill a
    whole `Pool.imap_unordered`. For PARAFAC2 `diagnostics.max_nn_violation`
    is measured on the returned CPU factors -- what the caller will plot --
    and `diagnostics` is None for CP.
    """
    method, random_state, payload, kwargs = args
    try:
        if method == "CP":
            decomp, errors = compute_CP_decomposition(
                payload,
                random_state=random_state,
                **kwargs,
            )
            weights, factors = decomp
            return FitOutcome(
                (_to_cpu(weights), _to_cpu(list(factors))),
                float(errors[-1]),
                None,
                None,
                len(errors),
            )
        model: PARAFAC2Model | CMFModel
        if method == "CMF":
            cmf, errors, diagnostics = compute_CMF_decomposition(
                payload,
                random_state=random_state,
                **kwargs,
            )
            model = _to_cpu(cmf)
        else:
            result, errors, diagnostics = compute_PARAFAC2_decomposition(
                payload,
                random_state=random_state,
                **kwargs,
            )
            model = _to_cpu(result)
    except ConvergenceError as error:
        return FitOutcome(None, None, None, error, error.n_iter)

    diagnostics = replace(
        diagnostics,
        max_nn_violation=_nn_violations(model, diagnostics.nn_modes),
    )
    return FitOutcome(model, float(errors[-1]), diagnostics, None, diagnostics.n_iter)


class _BestOfRestarts:
    """Keep the lowest-error restart and tally the rest.

    The one selection rule, shared by `_repeat_with_restarts` and
    `jobs.collect` so in-process and gathered results pick the same winner.
    Strictly lower wins, so on a tie the first restart offered is kept --
    which makes the choice deterministic when restarts are offered in seed
    order.
    """

    def __init__(self, attempted: int) -> None:
        self.tally = RestartTally(attempted=attempted)
        self.best_error = math.inf
        self.best: Any = None
        self.best_diagnostics: PARAFAC2Diagnostics | None = None

    def offer(
        self,
        error: float,
        item: Any,
        diagnostics: PARAFAC2Diagnostics | None,
    ) -> None:
        self.tally.record_success(diagnostics)
        if error < self.best_error:
            self.best_error = error
            self.best = item
            self.best_diagnostics = diagnostics

    def fail(self, error: ConvergenceError | None) -> None:
        self.tally.record_failure(error)


def _build_restart_advisory(
    tally: RestartTally,
    solver: PARAFAC2Solver,
    max_iter: int,
    tolerance: float | None,
    aoadmm_loss_tolerance: float | None,
    method_label: str = "PARAFAC2",
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
            f"{method_label}(solver={solver!r}): {reconstruction_failures} of "
            f"{tally.attempted} restarts were rejected because the "
            f"reconstruction error had not converged within max_iter="
            f"{max_iter}."
        )
        if level is not None and tolerance is not None:
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
        tolerance_text = _format_tolerance(tolerance)
        loss_tolerance_text = _format_tolerance(aoadmm_loss_tolerance)
        message += (
            f" Alternatives: loosen {method_label}_tolerance if "
            f"{tolerance_text} is stricter than you need"
        )
        if solver == "matcouply":
            message += (
                f", or lower aoadmm_loss_tolerance, currently "
                f"{loss_tolerance_text}, if AO-ADMM is stopping early on the "
                "penalized objective"
            )
        return message + "."

    feasibility_failures = tally.reasons.get("feasibility", 0)
    if feasibility_failures >= half:
        level = tally.median_level("feasibility")
        level_text = (
            f" Median max feasibility gap: {level:.2e}." if level is not None else ""
        )
        default_note = " (the default here)" if method_label == "PARAFAC2" else ""
        return (
            f"{method_label}(solver={solver!r}): {feasibility_failures} of "
            f"{tally.attempted} restarts were rejected because AO-ADMM left "
            f"the constraints infeasible within max_iter={max_iter}."
            f"{level_text} Raise {method_label}_max_iter, or relax "
            "feasibility_tol via aoadmm_options if that gap is acceptable. "
            "aoadmm_options={'constant_feasibility_penalty': True}"
            f"{default_note} also helps convergence markedly."
        )

    if tally.succeeded and tally.hit_iteration_limit >= tally.succeeded / 2:
        if tolerance is None:
            met = "were kept as the solver returned them"
        else:
            met = f"met the convergence tolerance of {tolerance:.1e}"
        return (
            f"{method_label}(solver={solver!r}): {tally.hit_iteration_limit} of "
            f"{tally.succeeded} accepted restarts ran to the iteration limit "
            f"(max_iter={max_iter}). The fits {met}, so they are usable, but "
            "every restart is paying the full iteration budget and more "
            "iterations would still improve them."
        )

    return None


def _repeat_with_restarts(
    method: Method,
    payload: Any,
    kwargs: dict[str, Any],
    init_repeats: int,
    device: torch.device,
    verbose_level: int,
    progress_bar: bool,
    restart_procs: int = 1,
    memory_efficient: bool = False,
) -> tuple[Any, torch.Tensor, Any, "RestartTally"]:
    """Run repeated random restarts and keep the best result.

    Shared restart/error-tracking/GPU-memory-management skeleton used by both
    `run_CP_decomposition_repeated` and `run_PARAFAC2_decomposition_repeated`.

    Parameters
    ----------
    method : Method
        Which compute function to call for each restart.
    payload : Any
        The `tensor`/`tensor_slices` argument for that function.
    kwargs : dict[str, Any]
        Its remaining arguments; `random_state` is filled in per restart.
    init_repeats : int
        Number of random restarts to try.
    device : torch.device
        Device the fit runs on (the input's, unless a `compute_device` moves
        it). Used for CUDA memory management and to reject
        `restart_procs >= 2`.
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
        adding to it. Results are compared in seed order either way, so an
        exact tie picks the same restart as `jobs.collect`.
    memory_efficient : bool, optional
        Whether the caller registered the memory-efficient MTTKRP; workers
        register it too.

    Returns
    -------
    tuple[Any, torch.Tensor, Any, RestartTally]
        `(best_decomp, best_error, best_extra, tally)`, `best_decomp` already
        CPU float32 (see `_fit_one`). `best_extra` is the winning restart's `PARAFAC2Diagnostics`,
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
            "fit runs on CUDA. Running multiple worker processes against a "
            "CUDA context is unsafe/unreliable across GPU driver setups -- "
            "pass restart_procs=1 to run sequentially on the GPU, or fit on "
            "CPU (input and compute_device) to use multiple processes.",
        )
    if restart_procs >= 2 and _in_worker_process():
        if verbose_level > 0:
            print(
                f"restart_procs={restart_procs} requested, but already running "
                "inside a worker process -- falling back to restart_procs=1 to "
                "avoid nesting process pools.",
            )
        restart_procs = 1

    reducer = _BestOfRestarts(attempted=init_repeats)
    # Once, so pool workers share one float64 copy (see `_float64_on_cpu`).
    payload = _float64_on_cpu(payload, device)
    task_args = [(method, i, payload, kwargs) for i in range(init_repeats)]

    def record(outcome: FitOutcome) -> None:
        decomp, error, extra, failure, _ = outcome
        if decomp is None:
            reducer.fail(failure)
            if verbose_level > 0:
                print(failure)
            return
        assert error is not None  # set whenever a model is
        reducer.offer(error, decomp, extra)

    if restart_procs < 2:
        for args in tqdm(task_args, disable=not progress_bar):
            record(_fit_one(args))
            _release_cuda_cache(device)
            sys.stdout.flush()
    else:
        # Split the parent's thread budget rather than letting each worker
        # default to its own pool -- see _init_restart_worker_backend.
        threads_per_proc = max(1, torch.get_num_threads() // restart_procs)
        with get_context("spawn").Pool(
            restart_procs,
            initializer=_init_restart_worker_backend,
            initargs=(threads_per_proc, memory_efficient),
        ) as pool:
            for outcome in tqdm(
                pool.imap(_fit_one, task_args),
                total=init_repeats,
                disable=not progress_bar,
            ):
                record(outcome)

    gc.collect()
    # Release PyTorch's cached memory back to the OS/GPU.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tally = reducer.tally
    if reducer.best is None:
        # Chain the last failure: its message carries the numbers saying what
        # to change (iteration counts, feasibility gaps, suggested max_iter).
        raise ConvergenceError(
            f"No decomposition converged within {init_repeats} repeats"
            f"{tally.failure_summary()}",
        ) from tally.last_failure

    best_error = torch.tensor(reducer.best_error, dtype=torch.float32)
    return reducer.best, best_error, reducer.best_diagnostics, tally


def run_CP_decomposition_repeated(
    tensor: torch.Tensor,
    rank: int,
    max_iter: int = 5000,
    init_repeats: int = 50,
    device: torch.device = torch.device("cpu"),
    use_memory_efficient_khatri_rao: bool = True,
    verbose_level: int = 0,
    tolerance: float | None = None,
    progress_bar: bool = True,
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

    kwargs = _compute_kwargs(
        "CP",
        rank,
        {
            "max_iter": max_iter,
            "verbose_level": verbose_level,
            "tolerance": tolerance,
            "allow_nan_imputation": allow_nan_imputation,
            "non_negative": non_negative,
        },
    )

    (best_weights, best_factors), best_error, _, _ = _repeat_with_restarts(
        "CP",
        tensor,
        kwargs,
        init_repeats,
        device,
        verbose_level,
        progress_bar,
        restart_procs=restart_procs,
        memory_efficient=use_memory_efficient_khatri_rao,
    )

    return best_weights, best_factors, best_error


def _nn_violations(
    model: "PARAFAC2Model | CMFModel",
    nn_modes: tuple[int, ...] | None,
) -> dict[int, float]:
    """Largest per-component non-negativity violation in the returned model.

    Scored **relative to each component's own scale**, because an absolute
    threshold misses the case that matters: a loading of -1e-07 is invisible
    against a total norm of order 1, yet it flips the sign of everything it
    multiplies.
    """
    if not nn_modes:
        return {}

    by_mode = {
        0: [model.subject_mode],
        1: model.evolving_states,
        2: [model.label_mode],
    }
    violations: dict[int, float] = {}
    for mode in sorted(set(nn_modes)):
        columns = by_mode[mode]
        worst = 0.0
        for rank_index in range(model.subject_mode.shape[1]):
            values = torch.cat([c[:, rank_index].reshape(-1) for c in columns])
            scale = float(values.abs().max())
            if scale == 0.0:
                continue
            worst = max(worst, -min(0.0, float(values.min())) / scale)
        violations[mode] = worst
    return violations


def _warn_if_nn_modes_violated(
    model: "PARAFAC2Model | CMFModel",
    nn_modes: tuple[int, ...] | None,
    violations: dict[int, float],
    solver: PARAFAC2Solver,
    stacklevel: int = 3,
    method_label: str = "PARAFAC2",
) -> None:
    """Warn when a constrained mode still carries negative entries.

    Triggered by the *presence* of negatives, not their size. Magnitude is
    the wrong test here: AO-ADMM enforces hard constraints on its auxiliary
    variables, so the returned primal sits within the feasibility gap of
    non-negative -- a relative violation of ~1e-08. That is numerically
    negligible and yet sign-decisive, because any negative entry flips every
    curve it multiplies. A caller scaling one mode by another (a time course
    by its subject loading, say) turns 1e-08 of residue into a
    full-amplitude mirrored curve once the result is normalized.
    """
    if not nn_modes:
        return

    by_mode = {
        0: [model.subject_mode],
        1: model.evolving_states,
        2: [model.label_mode],
    }
    offenders = []
    for mode in sorted(set(nn_modes)):
        negatives = sum(int((part < 0).sum()) for part in by_mode[mode])
        if not negatives:
            continue
        smallest = min(float(part.min()) for part in by_mode[mode])
        offenders.append(
            f"mode {mode} ({negatives} entries, min {smallest:.2e}, "
            f"{violations.get(mode, 0.0):.1e} relative)",
        )
    if not offenders:
        return

    warnings.warn(
        f"{method_label}(solver={solver!r}): the returned factors carry negative "
        f"entries in constrained {', '.join(offenders)}. These sit within "
        "AO-ADMM's feasibility gap -- the hard constraint holds on its "
        "auxiliary variables, not on the primal returned here -- so they are "
        "numerically negligible but NOT sign-negligible: multiplying one mode "
        "by another propagates the sign, and per-curve normalization then "
        "restores it to full amplitude as a mirrored curve. Clip with "
        "`factor.clamp(min=0)` before using a constrained mode as a "
        "multiplier.",
        stacklevel=stacklevel,
    )


def _warn_if_accepted_at_iteration_limit(
    diagnostics: PARAFAC2Diagnostics | None,
    stacklevel: int = 3,
    method_label: str = "PARAFAC2",
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
        loss_tolerance_text = _format_tolerance(diagnostics.loss_tolerance)
        loss_text = (
            f" -- relative loss change "
            f"{diagnostics.final_relative_loss_change:.1e} >= "
            f"aoadmm_loss_tolerance {loss_tolerance_text}"
        )
    if diagnostics.reconstruction_tolerance is None:
        warnings.warn(
            f"{method_label}(solver={diagnostics.solver!r}) stopped at the "
            f"iteration limit ({diagnostics.n_iter}/{diagnostics.max_iter}) "
            f"before meeting matcouply's own criterion{loss_text}. The fit is "
            f"kept, as matcouply returns it (final rel. reconstruction error "
            f"{diagnostics.relative_reconstruction_error:.3e}{gap_text}). Raise "
            "max_iter to let it converge, or pass a tolerance to accept only "
            "fits whose reconstruction error has settled to within it.",
            stacklevel=stacklevel,
        )
        return
    tolerance_text = _format_tolerance(diagnostics.reconstruction_tolerance)
    warnings.warn(
        f"{method_label}(solver={diagnostics.solver!r}) accepted at the iteration "
        f"limit ({diagnostics.n_iter}/{diagnostics.max_iter}). The "
        f"reconstruction error has converged -- |delta rel. reconstruction "
        f"error| = {diagnostics.reconstruction_error_change:.3e} < tolerance "
        f"{tolerance_text}, final rel. "
        f"reconstruction error "
        f"{diagnostics.relative_reconstruction_error:.3e}{gap_text} -- but "
        f"AO-ADMM's own penalized-objective criterion was not met{loss_text}. "
        "The penalized objective is not comparable to TensorLy's criterion; "
        "acceptance is based on the reconstruction error, which is.",
        stacklevel=stacklevel,
    )


def _warn_parafac2_outcome(
    model: "PARAFAC2Model | CMFModel | None",
    diagnostics: PARAFAC2Diagnostics | None,
    tally: RestartTally,
    solver: PARAFAC2Solver,
    max_iter: int,
    tolerance: float | None,
    aoadmm_loss_tolerance: float | None,
    stacklevel: int = 3,
    method_label: str = "PARAFAC2",
) -> None:
    """Emit every PARAFAC2/CMF warning about one best-of-restarts outcome.

    Shared by `run_PARAFAC2_decomposition_repeated`,
    `run_CMF_decomposition_repeated` and `jobs.collect`, so a
    gathered fit is reported exactly like an in-process one. `stacklevel`
    points at the caller of whichever of those two called this.
    `max_nn_violation` is read from `diagnostics`, where `_fit_one` measured
    it on the CPU float32 factors the caller actually receives.
    """
    if model is not None and diagnostics is not None:
        _warn_if_nn_modes_violated(
            model,
            diagnostics.nn_modes,
            diagnostics.max_nn_violation or {},
            solver,
            stacklevel=stacklevel + 1,
            method_label=method_label,
        )
        _warn_if_accepted_at_iteration_limit(
            diagnostics,
            stacklevel=stacklevel + 1,
            method_label=method_label,
        )
    advisory = _build_restart_advisory(
        tally,
        solver,
        max_iter,
        tolerance,
        aoadmm_loss_tolerance,
        method_label=method_label,
    )
    if advisory is not None:
        warnings.warn(advisory, stacklevel=stacklevel)


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
    tolerance: float | None = ...,
    progress_bar: bool = ...,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = ...,
    restart_procs: int = ...,
    solver: PARAFAC2Solver = ...,
    aoadmm_options: dict[str, Any] | None = ...,
    aoadmm_loss_tolerance: float | None = ...,
    return_projections: bool = ...,
    negligible_rtol: float = ...,
    compute_device: torch.device | None = ...,
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
    tolerance: float | None = ...,
    progress_bar: bool = ...,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = ...,
    restart_procs: int = ...,
    solver: PARAFAC2Solver = ...,
    aoadmm_options: dict[str, Any] | None = ...,
    aoadmm_loss_tolerance: float | None = ...,
    return_projections: bool = ...,
    negligible_rtol: float = ...,
    compute_device: torch.device | None = ...,
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
    tolerance: float | None = None,
    progress_bar: bool = True,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = "auto",
    restart_procs: int = 1,
    solver: PARAFAC2Solver = "matcouply",
    aoadmm_options: dict[str, Any] | None = None,
    aoadmm_loss_tolerance: float | None = None,
    return_projections: bool = False,
    negligible_rtol: float = 1e-6,
    compute_device: torch.device | None = None,
    return_diagnostics: bool = False,
) -> PARAFAC2Result | PARAFAC2ResultWithDiagnostics:
    """Repeatedly fit a PARAFAC2 decomposition from random restarts.

    See `compute_PARAFAC2_decomposition` for `solver`, `nn_modes`,
    `aoadmm_options`, `aoadmm_loss_tolerance` and the two-threshold
    convergence scheme; `_repeat_with_restarts` for `restart_procs`.

    The return contract is identical for both solvers, so calling code never
    branches on `solver`. `return_diagnostics=True` appends the winning
    restart's `PARAFAC2Diagnostics`.

    Option names are kept in sync with `run_CP_decomposition_repeated` so one
    `**kwargs` dict routes to either. Not shared: `nn_modes` replaces CP's
    `non_negative` (it picks *which* modes are constrained); `solver`,
    `aoadmm_options`, `aoadmm_loss_tolerance`, `return_projections` and
    `return_diagnostics` are PARAFAC2-only; and there is no
    `allow_nan_imputation`, since NaN input always raises here.

    `compute_device` (see `compute_PARAFAC2_decomposition`) is separate from
    `device`, which only describes where the input lives: keep a large input
    on CPU and pass `compute_device=torch.device("cuda")` to fit it on the
    GPU. Each restart then moves it there afresh, and frees it afterwards.

    Emits a `UserWarning` rather than staying silent when the solver is
    struggling: once if the winning fit was accepted at the iteration limit,
    and once if most restarts were rejected for the same reason (see
    `_build_restart_advisory`).

    Returns
    -------
    tuple
        `(model, error)`, plus `diagnostics` when `return_diagnostics=True`.
        `model` is a `PARAFAC2Model` whose `evolving_states[i]` is subject
        `i`'s own time course. `error` is the relative reconstruction error,
        defined identically for both solvers and so comparable between them.
    """
    # Once up front, not in every restart (and worker).
    _check_parafac2_rank(tensor_slices, rank)
    _maybe_register_memory_efficient_khatri_rao(use_memory_efficient_khatri_rao)

    kwargs = _compute_kwargs(
        "PARAFAC2",
        rank,
        {
            "max_iter": max_iter,
            "verbose_level": verbose_level,
            "tolerance": tolerance,
            "nn_modes": nn_modes,
            "solver": solver,
            "aoadmm_options": aoadmm_options,
            "aoadmm_loss_tolerance": aoadmm_loss_tolerance,
            "return_projections": return_projections,
            "negligible_rtol": negligible_rtol,
            "compute_device": compute_device,
        },
    )

    (best_model, best_error, best_diagnostics, tally) = _repeat_with_restarts(
        "PARAFAC2",
        tensor_slices,
        kwargs,
        init_repeats,
        # Where the fit runs is what the CUDA guard and cache clearing need.
        _fit_device(compute_device, device),
        verbose_level,
        progress_bar,
        restart_procs=restart_procs,
        memory_efficient=use_memory_efficient_khatri_rao,
    )

    _warn_parafac2_outcome(
        best_model,
        best_diagnostics,
        tally,
        solver,
        max_iter,
        tolerance,
        aoadmm_loss_tolerance,
    )

    if return_diagnostics:
        return best_model, best_error, best_diagnostics
    return best_model, best_error


@overload
def run_CMF_decomposition_repeated(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int = ...,
    init_repeats: int = ...,
    device: torch.device = ...,
    use_memory_efficient_khatri_rao: bool = ...,
    verbose_level: int = ...,
    tolerance: float | None = ...,
    progress_bar: bool = ...,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = ...,
    restart_procs: int = ...,
    aoadmm_options: dict[str, Any] | None = ...,
    aoadmm_loss_tolerance: float | None = ...,
    negligible_rtol: float = ...,
    compute_device: torch.device | None = ...,
    return_diagnostics: Literal[False] = ...,
) -> CMFResult:
    ...


@overload
def run_CMF_decomposition_repeated(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int = ...,
    init_repeats: int = ...,
    device: torch.device = ...,
    use_memory_efficient_khatri_rao: bool = ...,
    verbose_level: int = ...,
    tolerance: float | None = ...,
    progress_bar: bool = ...,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = ...,
    restart_procs: int = ...,
    aoadmm_options: dict[str, Any] | None = ...,
    aoadmm_loss_tolerance: float | None = ...,
    negligible_rtol: float = ...,
    compute_device: torch.device | None = ...,
    *,
    return_diagnostics: Literal[True],
) -> CMFResultWithDiagnostics:
    ...


def run_CMF_decomposition_repeated(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    max_iter: int = 2000,
    init_repeats: int = 50,
    device: torch.device = torch.device("cpu"),
    use_memory_efficient_khatri_rao: bool = True,
    verbose_level: int = 0,
    tolerance: float | None = None,
    progress_bar: bool = True,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = "auto",
    restart_procs: int = 1,
    aoadmm_options: dict[str, Any] | None = None,
    aoadmm_loss_tolerance: float | None = None,
    negligible_rtol: float = 1e-6,
    compute_device: torch.device | None = None,
    return_diagnostics: bool = False,
) -> CMFResult | CMFResultWithDiagnostics:
    """Repeatedly fit a non-negative CMF from random restarts.

    See `compute_CMF_decomposition` for the model, `nn_modes`,
    `aoadmm_options` and the tolerances, and `_repeat_with_restarts` for
    `restart_procs`. Option names match `run_PARAFAC2_decomposition_repeated`
    (without `solver`, since only matcouply fits a CMF), so one `**kwargs`
    dict routes to either; `use_memory_efficient_khatri_rao` is accepted for
    that reason and has no effect here.

    Warns like the PARAFAC2 runner when the solver is struggling.

    Returns
    -------
    tuple
        `(model, error)`, plus the winning restart's diagnostics when
        `return_diagnostics=True`. `model` is a `CMFModel`; `error` is the
        relative reconstruction error.
    """
    kwargs = _compute_kwargs(
        "CMF",
        rank,
        {
            "max_iter": max_iter,
            "verbose_level": verbose_level,
            "tolerance": tolerance,
            "nn_modes": nn_modes,
            "aoadmm_options": aoadmm_options,
            "aoadmm_loss_tolerance": aoadmm_loss_tolerance,
            "negligible_rtol": negligible_rtol,
            "compute_device": compute_device,
        },
    )

    best_model, best_error, best_diagnostics, tally = _repeat_with_restarts(
        "CMF",
        tensor_slices,
        kwargs,
        init_repeats,
        _fit_device(compute_device, device),
        verbose_level,
        progress_bar,
        restart_procs=restart_procs,
        memory_efficient=False,
    )

    _warn_parafac2_outcome(
        best_model,
        best_diagnostics,
        tally,
        "matcouply",
        max_iter,
        tolerance,
        aoadmm_loss_tolerance,
        method_label="CMF",
    )

    if return_diagnostics:
        return best_model, best_error, best_diagnostics
    return best_model, best_error
