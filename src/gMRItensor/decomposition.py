import gc
import os
import sys
from multiprocessing import current_process
from multiprocessing import get_context
from typing import Any
from typing import Callable
from typing import Literal

import tensorly as tl
import torch
from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao_memory
from tlviz.factor_tools import degeneracy_score
from tqdm import tqdm


class ConvergenceError(Exception):
    """Custom exception raised when TensorLy's PARAFAC fails to converge."""

    pass


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
    support -- unlike PARAFAC2, which has no such support in this TensorLy
    version (see `compute_PARAFAC2_decomposition`). If False (the default)
    and `tensor` contains NaN, a `ValueError` is raised rather than silently
    fitting on/propagating NaN.

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


def compute_PARAFAC2_decomposition(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    PARAFAC2_max_iter: int = 500,
    random_state: int = 0,
    init: str = "random",
    PARAFAC2_verbose_level: int = 0,
    PARAFAC2_tolerance: float = 1e-5,
    normalize_factors: bool = False,
    nn_modes: tuple[int, ...] | None = (0, 2),
):
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
    reconstruct_evolving_factors`.

    Notes
    -----
    `nn_modes` defaults to `(0, 2)` (subject and region modes constrained
    non-negative) rather than including mode 1 (the evolving/time mode):
    TensorLy's ALS solver cannot truly enforce non-negativity on mode 1 of a
    PARAFAC2 decomposition and warns if asked to.

    Unlike `compute_CP_decomposition`, this is not wrapped in `torch.compile`:
    TensorLy's `parafac2` has per-iteration convergence/linesearch checks and
    an inherently ragged per-slice Python loop that cannot be traced into one
    graph (confirmed to produce dozens of graph breaks on trivial inputs), so
    compiling it adds overhead without a real speedup.

    Raises
    ------
    ValueError
        If any slice contains NaN. Unlike `compute_CP_decomposition`,
        TensorLy's `parafac2` has no `mask`/imputation support in this
        version, so there is no `allow_nan_imputation` option here -- NaNs
        must be removed or imputed before calling this function (e.g. by
        using `prepare_tensor(..., require_regular=False)`, which never
        introduces NaN gaps in the first place).
    """
    slices_to_check = (
        tensor_slices if isinstance(tensor_slices, list) else [tensor_slices]
    )
    if any(torch.isnan(s).any() for s in slices_to_check):
        raise ValueError(
            "tensor_slices contains NaN values, but PARAFAC2 has no "
            "NaN-imputation/masking support in this TensorLy version -- "
            "remove or impute missing values first (e.g. via "
            "prepare_tensor(..., require_regular=False), which never "
            "introduces NaN gaps).",
        )

    result, errors = tl.decomposition.parafac2(
        tensor_slices,
        rank=rank,
        n_iter_max=PARAFAC2_max_iter,
        tol=PARAFAC2_tolerance,
        return_errors=True,
        random_state=random_state,
        verbose=PARAFAC2_verbose_level,
        init=init,
        normalize_factors=normalize_factors,
        nn_modes=list(nn_modes) if nn_modes else None,
    )
    if len(errors) > PARAFAC2_max_iter - 1:
        raise ConvergenceError(
            "Decomposition did not converge within the maximum iteration count",
        )

    w = result.weights.float()
    f = [ff.float() for ff in result.factors]
    if degeneracy_score((w, f)) < -0.85:
        raise ConvergenceError("Decomposition is degenerate")

    return result, errors


def _restart_worker(
    args: tuple[Literal["CP", "PARAFAC2"], int, Any, dict[str, Any]],
) -> tuple[Any, torch.Tensor] | tuple[None, None]:
    """Pool worker: run a single random-restart attempt.

    Only used by `_repeat_with_restarts`'s parallel (CPU-only, `restart_procs
    >= 2`) path -- the sequential path calls `compute_CP_decomposition`/
    `compute_PARAFAC2_decomposition` directly, in-process, so it also works
    with a GPU tensor and reuses the `torch.compile` cache across restarts.

    Returns `(None, None)` rather than raising on `ConvergenceError`, so one
    failed restart doesn't kill the whole `Pool.imap_unordered` -- matching
    the sequential loop's "skip and continue" behavior.
    """
    method, random_state, payload, kwargs = args
    try:
        if method == "CP":
            return compute_CP_decomposition(
                payload, random_state=random_state, **kwargs
            )
        return compute_PARAFAC2_decomposition(
            payload,
            random_state=random_state,
            **kwargs,
        )
    except ConvergenceError:
        return None, None


def _init_restart_worker_backend() -> None:
    """Pool initializer: set up TensorLy's "pytorch" backend in each worker.

    Needed with the "spawn" start method used here (see
    `gMRItensor.replicability._init_worker_backend`, which does the same
    thing for the same reason).
    """
    tl.set_backend("pytorch")


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
) -> tuple[Any, torch.Tensor]:
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

    Returns
    -------
    tuple[Any, torch.Tensor]
        `(best_decomp, best_error)`, with `best_decomp` already moved to CPU.

    Raises
    ------
    ConvergenceError
        If no restart converged.
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

    if restart_procs < 2:
        for i in tqdm(range(init_repeats), disable=not progress_bar):
            try:
                if method == "CP":
                    decomp, errors = compute_CP_decomposition(
                        payload,
                        random_state=i,
                        **kwargs,
                    )
                else:
                    decomp, errors = compute_PARAFAC2_decomposition(
                        payload,
                        random_state=i,
                        **kwargs,
                    )
            except ConvergenceError as e:
                if verbose_level > 0:
                    print(e)
                continue

            if errors[-1] < best_error:
                best_error = errors[-1]
                # Move the best result to CPU immediately to free up GPU VRAM
                best_decomp = to_cpu(decomp)

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
        with get_context("spawn").Pool(
            restart_procs,
            initializer=_init_restart_worker_backend,
        ) as pool:
            for decomp, errors in tqdm(
                pool.imap_unordered(_restart_worker, task_args),
                total=init_repeats,
                disable=not progress_bar,
            ):
                if decomp is None:
                    continue
                if errors[-1] < best_error:
                    best_error = errors[-1]
                    best_decomp = to_cpu(decomp)
                del decomp, errors

    gc.collect()
    # Force PyTorch to release its internal cached memory back to the OS/GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if best_decomp is None:
        raise ConvergenceError(
            f"No decomposition converged within {init_repeats} repeats",
        )
    assert isinstance(best_error, torch.Tensor)  # guaranteed once best_decomp is set

    return best_decomp, best_error.float().cpu()


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
    for the options it doesn't share (`nn_modes` instead of
    `non_negative`; no `allow_nan_imputation`). Kept in sync so a single
    `**kwargs` dict of shared options (e.g.
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

    (best_weights, best_factors), best_error = _repeat_with_restarts(
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
    nn_modes: tuple[int, ...] | None = (0, 2),
    restart_procs: int = 1,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Repeatedly fit a PARAFAC2 decomposition from random restarts.

    See `compute_PARAFAC2_decomposition` for the meaning of `nn_modes` and
    why this is not `torch.compile`-wrapped, and `_repeat_with_restarts` for
    `restart_procs`.

    Notes
    -----
    Shares its option names (`max_iter`, `init_repeats`, `verbose_level`,
    `tolerance`, `normalize`, `use_memory_efficient_khatri_rao`,
    `progress_bar`, `device`, `rank`, `restart_procs`) with
    `run_CP_decomposition_repeated` -- see that function's `Notes`. Two
    options aren't shared: `nn_modes` replaces CP's flat `non_negative`
    bool (it's strictly more expressive -- it picks *which* modes are
    constrained, defaulting to `(0, 2)`; pass `None` for an unconstrained
    fit); and there's no `allow_nan_imputation` here, since this TensorLy
    version has no PARAFAC2 mask/imputation support at all (see
    `compute_PARAFAC2_decomposition`) -- unlike CP, NaN input always
    raises regardless of any parameter.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]
        `(best_weights, best_factors, best_projections, best_error)`.
        `best_factors = [A, B, C]` (subject, shared evolving-mode basis,
        region); `best_projections[i]` is the per-subject orthonormal
        projection needed to reconstruct that subject's own time pattern
        (`projections[i] @ best_factors[1]`).
    """
    _maybe_register_memory_efficient_khatri_rao(use_memory_efficient_khatri_rao)

    kwargs = {
        "rank": rank,
        "PARAFAC2_max_iter": max_iter,
        "PARAFAC2_verbose_level": verbose_level,
        "PARAFAC2_tolerance": tolerance,
        "normalize_factors": normalize,
        "nn_modes": nn_modes,
    }

    def to_cpu(result):
        weights = result.weights.float().cpu()
        factors = [f.float().cpu() for f in result.factors]
        projections = [p.float().cpu() for p in result.projections]
        return weights, factors, projections

    (best_weights, best_factors, best_projections), best_error = _repeat_with_restarts(
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
