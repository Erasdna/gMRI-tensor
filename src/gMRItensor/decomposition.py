import gc
import os
import sys

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
):
    if use_memory_efficient_khatri_rao:
        tl.tenalg.register_backend_method(
            "unfolding_dot_khatri_rao",
            unfolding_dot_khatri_rao_memory,
        )
        tl.tenalg.use_dynamic_dispatch()

    best_error: torch.Tensor | float = torch.inf
    best_factors: list[torch.Tensor] | None = None
    best_weights: torch.Tensor | None = None

    for i in tqdm(range(CP_init_repeats), disable=not progress_bar):
        try:
            decomp, error = compute_CP_decomposition(
                tensor,
                rank,
                CP_max_iter,
                random_state=i,
                CP_verbose_level=CP_verbose_level,
                CP_tolerance=CP_tolerance,
                normalize_factors=normalize,
            )
        except ConvergenceError as e:
            if CP_verbose_level > 0:
                print(e)
            continue

        if error[-1] < best_error:
            best_error = error[-1]
            # Move the best factors to CPU immediately to free up GPU VRAM
            weights, factors = decomp
            best_weights = weights.float().cpu()
            best_factors = [f.float().cpu() for f in factors]

        del decomp, error

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

    if best_weights is None or best_factors is None:
        raise ConvergenceError(
            f"No decomposition converged within {CP_init_repeats} repeats",
        )
    assert isinstance(best_error, torch.Tensor)  # guaranteed once best_weights is set

    return best_weights, best_factors, best_error.float().cpu()


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
