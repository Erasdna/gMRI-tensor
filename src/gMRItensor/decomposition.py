import gc
import os
import sys
from typing import Any
from typing import Callable

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


def compute_CP_decomposition(
    tensor: torch.Tensor,
    rank: int,
    CP_max_iter: int = 500,
    random_state: int = 0,
    init: str = "random",
    CP_verbose_level: int = 0,
    CP_tolerance: float = 1e-5,
    normalize_factors: bool = False,
):
    decomp, errors = non_negative_parafac_compiled(
        tensor,
        rank=rank,
        n_iter_max=CP_max_iter,
        tol=CP_tolerance,  # Computing this tensor decomp is quite expensive...
        return_errors=True,
        random_state=random_state,
        verbose=CP_verbose_level,
        init=init,
        normalize_factors=normalize_factors,
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
    """
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


def _repeat_with_restarts(
    attempt: Callable[[int], tuple[Any, torch.Tensor]],
    to_cpu: Callable[[Any], Any],
    init_repeats: int,
    device: torch.device,
    verbose_level: int,
    progress_bar: bool,
) -> tuple[Any, torch.Tensor]:
    """Run `attempt` with repeated random restarts and keep the best result.

    Shared restart/error-tracking/GPU-memory-management skeleton used by both
    `run_CP_decomposition_repeated` and `run_PARAFAC2_decomposition_repeated`.

    Parameters
    ----------
    attempt : Callable[[int], tuple[Any, torch.Tensor]]
        Called with a `random_state` index; should return `(decomp, errors)`
        for that restart, raising `ConvergenceError` if it failed. `decomp`
        is an opaque, decomposition-specific result and `errors` is the list
        of per-iteration reconstruction errors.
    to_cpu : Callable[[Any], Any]
        Moves the winning `decomp` to CPU/float precision.
    init_repeats : int
        Number of random restarts to try.
    device : torch.device
        Device the input tensor(s) live on (used for CUDA memory management).
    verbose_level : int
        If > 0, prints each `ConvergenceError` encountered.
    progress_bar : bool
        Whether to show a tqdm progress bar over the restarts.

    Returns
    -------
    tuple[Any, torch.Tensor]
        `(best_decomp, best_error)`, with `best_decomp` already moved to CPU.

    Raises
    ------
    ConvergenceError
        If no restart converged.
    """
    best_error: torch.Tensor | float = torch.inf
    best_decomp: Any = None

    for i in tqdm(range(init_repeats), disable=not progress_bar):
        try:
            decomp, errors = attempt(i)
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


def run_CP_decomposition_repeated(
    tensor: torch.Tensor,
    rank: int,
    CP_max_iter: int = 5000,
    CP_init_repeats: int = 50,
    device: torch.device = torch.device("cpu"),
    use_memory_efficient_khatri_rao: bool = True,
    CP_verbose_level: int = 0,
    CP_tolerance: float = 1e-5,
    progress_bar: bool = True,
    normalize: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    if use_memory_efficient_khatri_rao:
        tl.tenalg.register_backend_method(
            "unfolding_dot_khatri_rao",
            unfolding_dot_khatri_rao_memory,
        )
        tl.tenalg.use_dynamic_dispatch()

    def attempt(random_state: int):
        return compute_CP_decomposition(
            tensor,
            rank,
            CP_max_iter,
            random_state=random_state,
            CP_verbose_level=CP_verbose_level,
            CP_tolerance=CP_tolerance,
            normalize_factors=normalize,
        )

    def to_cpu(decomp):
        weights, factors = decomp
        return weights.float().cpu(), [f.float().cpu() for f in factors]

    (best_weights, best_factors), best_error = _repeat_with_restarts(
        attempt,
        to_cpu,
        CP_init_repeats,
        device,
        CP_verbose_level,
        progress_bar,
    )

    return best_weights, best_factors, best_error


def run_PARAFAC2_decomposition_repeated(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    PARAFAC2_max_iter: int = 2000,
    PARAFAC2_init_repeats: int = 50,
    device: torch.device = torch.device("cpu"),
    PARAFAC2_verbose_level: int = 0,
    PARAFAC2_tolerance: float = 1e-5,
    progress_bar: bool = True,
    nn_modes: tuple[int, ...] | None = (0, 2),
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Repeatedly fit a PARAFAC2 decomposition from random restarts.

    See `compute_PARAFAC2_decomposition` for the meaning of `nn_modes` and
    why this is not `torch.compile`-wrapped.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]
        `(best_weights, best_factors, best_projections, best_error)`.
        `best_factors = [A, B, C]` (subject, shared evolving-mode basis,
        region); `best_projections[i]` is the per-subject orthonormal
        projection needed to reconstruct that subject's own time pattern
        (`projections[i] @ best_factors[1]`).
    """

    def attempt(random_state: int):
        return compute_PARAFAC2_decomposition(
            tensor_slices,
            rank,
            PARAFAC2_max_iter,
            random_state=random_state,
            PARAFAC2_verbose_level=PARAFAC2_verbose_level,
            PARAFAC2_tolerance=PARAFAC2_tolerance,
            nn_modes=nn_modes,
        )

    def to_cpu(result):
        weights = result.weights.float().cpu()
        factors = [f.float().cpu() for f in result.factors]
        projections = [p.float().cpu() for p in result.projections]
        return weights, factors, projections

    (best_weights, best_factors, best_projections), best_error = _repeat_with_restarts(
        attempt,
        to_cpu,
        PARAFAC2_init_repeats,
        device,
        PARAFAC2_verbose_level,
        progress_bar,
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
