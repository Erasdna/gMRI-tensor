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


def compute_CP_decomposition(
    tensor: torch.Tensor,
    rank: int,
    CP_max_iter: int = 500,
    random_state: int = 0,
    func=tl.decomposition.non_negative_parafac,
    init="random",
    CP_verbose_level=0,
    CP_tolerance=1e-5,
):
    decomp, errors = func(
        tensor,
        rank=rank,
        n_iter_max=CP_max_iter,
        tol=CP_tolerance,  # Computing this tensor decomp is quite expensive...
        return_errors=True,
        random_state=random_state,
        verbose=CP_verbose_level,
        init=init,
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
    CP_verbose_level=0,
    CP_tolerance=1e-5,
):
    if use_memory_efficient_khatri_rao:
        tl.tenalg.register_backend_method(
            "unfolding_dot_khatri_rao",
            unfolding_dot_khatri_rao_memory,
        )
        tl.tenalg.use_dynamic_dispatch()

    best_error = torch.inf
    best_factors = None
    best_weights = None

    func = torch.compile(tl.decomposition.non_negative_parafac, dynamic=True)
    for i in tqdm(range(CP_init_repeats)):
        try:
            decomp, error = compute_CP_decomposition(
                tensor,
                rank,
                CP_max_iter,
                i,
                func,
                CP_verbose_level=CP_verbose_level,
                CP_tolerance=CP_tolerance,
            )
        except ConvergenceError as e:
            print(e)
            continue

        if error[-1] < best_error:
            best_error = error[-1]
            # Move the best factors to CPU immediately to free up GPU VRAM
            weights, factors = decomp
            best_weights = weights.float().cpu()
            best_factors = [f.float().cpu() for f in factors]

        del decomp, error
        gc.collect()
        # Force PyTorch to release its internal cached memory back to the OS/GPU
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sys.stdout.flush()
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
