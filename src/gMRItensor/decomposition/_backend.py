"""Where a fit runs: TensorLy's numpy backend in float64 for CPU fits,
pytorch on the GPU, thread limits and restart worker processes.
"""
import dataclasses
import inspect
import os
import warnings
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import current_process
from typing import Any

import numpy as np
import tensorly as tl
import torch
from matcouply.coupled_matrices import CoupledMatrixFactorization
from tensorly.cp_tensor import CPTensor
from tensorly.parafac2_tensor import Parafac2Tensor
from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao_memory
from threadpoolctl import threadpool_limits


def _check_tolerances(**tolerances: float | None) -> None:
    """Raise `ValueError` unless each tolerance is None (the solver's
    default) or > 0: at 0 no fit can converge."""
    for name, value in tolerances.items():
        if value is not None and not value > 0:
            raise ValueError(f"{name} must be > 0 (or None), got {value}")


def _solver_default(function: Callable[..., Any], name: str = "tol") -> float:
    """`function`'s own default for `name`, read from its signature.

    Tolerances default to None throughout this module, meaning "whatever the
    underlying library uses"; reading it here keeps that true across library
    upgrades instead of copying the number.
    """
    return inspect.signature(inspect.unwrap(function)).parameters[name].default


def _format_tolerance(tolerance: float | None) -> str:
    return "the solver's own criterion" if tolerance is None else f"{tolerance:.1e}"


@contextmanager
def _matcouply_numeric_context(device: torch.device) -> Iterator[None]:
    """Make torch's defaults float64-on-`device` for the duration of a fit.
    Necessary to be compatible with matcouply, which does not accept float32.

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


def _to_solver_slices(
    slices: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    """Move each slice to `device`, then cast it to float64 there.

    Moving before casting keeps the float64 copy off the source device: for
    CPU-resident voxel data fitted on the GPU, casting first would put a full
    float64 copy -- twice the float32 input -- in host RAM. Done per slice, so
    at most one float32 slice-sized temporary exists on `device` at a time.
    """
    return [s.to(device).double() for s in slices]


#: Environment variable choosing where CPU fits run: "numpy" (default),
#: "torch" or "auto" (by size, see `AUTO_BACKEND_VARIABLE`). GPU fits always
#: run on pytorch.
FIT_BACKEND_VARIABLE = "GMRITENSOR_FIT_BACKEND"


#: Environment variable: with `GMRITENSOR_FIT_BACKEND=auto`, CPU fits of at
#: least this many data entries run on torch, smaller ones on numpy.
AUTO_BACKEND_VARIABLE = "GMRITENSOR_AUTO_BACKEND_ENTRIES"


#: `AUTO_BACKEND_VARIABLE`'s default: between the sizes where numpy was
#: still faster (30 x 4 x 1e5 entries) and torch already was (30 x 4 x 1e6).
AUTO_BACKEND_DEFAULT_ENTRIES = 50_000_000


#: Environment variable overriding `use_memory_efficient_khatri_rao`: "1"
#: registers TensorLy's memory-efficient MTTKRP, "0" keeps its default one.
MEMORY_EFFICIENT_MTTKRP_VARIABLE = "GMRITENSOR_MEMORY_EFFICIENT_MTTKRP"


#: Containers the solvers return, rebuilt part by part by `_solver_output`.
_FACTORIZATIONS = (CPTensor, Parafac2Tensor, CoupledMatrixFactorization)


def _fit_device(
    compute_device: torch.device | None,
    default: torch.device,
) -> torch.device:
    """The device a fit runs on: `compute_device` if given, else `default`
    (the input's)."""
    return compute_device if compute_device is not None else default


def _auto_backend_entries() -> int:
    """`AUTO_BACKEND_VARIABLE`, or its default; raises unless an int > 0."""
    value = os.environ.get(AUTO_BACKEND_VARIABLE)
    if value is None:
        return AUTO_BACKEND_DEFAULT_ENTRIES
    try:
        entries = int(float(value))
    except ValueError:
        entries = 0
    if entries <= 0:
        raise ValueError(
            f"{AUTO_BACKEND_VARIABLE} must be a number of entries > 0, got {value!r}",
        )
    return entries


def _fits_on_numpy(device: torch.device, entries: int) -> bool:
    """Whether a fit of `entries` data entries on `device` runs on
    TensorLy's numpy backend.

    Every CPU fit does, in float64, unless `GMRITENSOR_FIT_BACKEND` says
    otherwise: `torch`, or `auto` for a fit of at least
    `GMRITENSOR_AUTO_BACKEND_ENTRIES` entries. The solvers' inner loops make
    many small calls, each several times cheaper in numpy than in torch, but
    on large data torch's multithreaded elementwise operations win. A GPU
    fit always runs on pytorch; asking for numpy there is ignored with a
    warning.
    """
    choice = os.environ.get(FIT_BACKEND_VARIABLE, "numpy")
    if choice not in ("numpy", "torch", "auto"):
        raise ValueError(
            f"{FIT_BACKEND_VARIABLE} must be numpy, torch or auto, got {choice!r}",
        )
    if device.type != "cpu":
        if FIT_BACKEND_VARIABLE in os.environ and choice == "numpy":
            warnings.warn(
                f"{FIT_BACKEND_VARIABLE}=numpy is ignored for a fit on "
                f"{device}: GPU fits always run on pytorch.",
                stacklevel=3,
            )
        return False
    if choice == "auto":
        return entries < _auto_backend_entries()
    return choice == "numpy"


def _fit_backend_name(device: torch.device, entries: int) -> str:
    """Where a fit of `entries` entries on `device` runs, as recorded with
    each restart: "numpy", "torch", or the GPU device type."""
    if device.type != "cpu":
        return device.type
    return "numpy" if _fits_on_numpy(device, entries) else "torch"


def _entries(data: list[torch.Tensor] | torch.Tensor) -> int:
    """How many entries a fit's data has: a tensor's, or all its slices'."""
    if isinstance(data, torch.Tensor):
        return data.numel()
    return sum(s.numel() for s in data)


def _float64_on_cpu(
    data: list[torch.Tensor] | torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor] | torch.Tensor:
    """`data` (a tensor or slices) in float64 when the fit runs on the CPU.

    Every CPU fit runs in float64, so casting once here, before the restarts
    are spread over a pool, lets the workers share this one copy: their own
    casts are then no-ops. Otherwise each worker made a private float64 copy
    of the whole input. GPU data is left as it is.
    """
    if device.type != "cpu":
        return data
    if isinstance(data, torch.Tensor):
        return data.double()
    return [s.double() for s in data]


def _to_cpu(value: Any) -> Any:
    """`value` with every tensor as CPU float32, through lists, tuples and
    the model dataclasses: how every restart's fit is returned and stored."""
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            f.name: _to_cpu(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
        return dataclasses.replace(value, **fields)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _solver_input(value: Any) -> Any:
    """`value` with every torch tensor as a float64 numpy array, through
    lists and tuples."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    if isinstance(value, (list, tuple)):
        return type(value)(_solver_input(item) for item in value)
    return value


def _solver_output(value: Any, dtype: torch.dtype) -> Any:
    """`value` back from the numpy backend as CPU torch: arrays as `dtype`,
    numeric scalars as float64 (they are errors and losses, compared to
    tolerances), through lists, tuples, named tuples and the factorization
    containers. Other values are returned as they are."""
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return torch.from_numpy(np.ascontiguousarray(value)).to(dtype)
    if isinstance(value, (np.ndarray, np.generic)):
        item = value.item()
        if isinstance(item, (bool, int)) or not isinstance(item, float):
            return item
        return torch.tensor(item, dtype=torch.float64)
    if isinstance(value, _FACTORIZATIONS):
        return type(value)(tuple(_solver_output(part, dtype) for part in value))
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_solver_output(item, dtype) for item in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_solver_output(item, dtype) for item in value)
    return value


class _FitBackend:
    """Where one solver call runs (see `_fits_on_numpy`).

    `input` prepares the solver's arguments, `call` runs it and `output`
    turns its result back into torch. On the pytorch backend all three
    leave their values as they are. The conversion back happens after the
    numpy backend is left, so TensorLy validates the rebuilt containers on
    pytorch.
    """

    def __init__(self, device: torch.device, entries: int) -> None:
        self.numpy = _fits_on_numpy(device, entries)

    def input(self, value: Any) -> Any:
        """`value` as the solver takes it."""
        return _solver_input(value) if self.numpy else value

    def call(self, solver: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """`solver(*args, **kwargs)` on this backend."""
        if not self.numpy:
            return solver(*args, **kwargs)
        with tl.backend_context("numpy"):
            return solver(*args, **kwargs)

    def output(self, value: Any, dtype: torch.dtype) -> Any:
        """The solver's result as torch, arrays as `dtype`."""
        return _solver_output(value, dtype) if self.numpy else value


def _as_slices(tensor_slices: list[torch.Tensor] | torch.Tensor) -> list[torch.Tensor]:
    if isinstance(tensor_slices, list):
        return list(tensor_slices)
    return [tensor_slices[i] for i in range(tensor_slices.shape[0])]


def _release_cuda_cache(device: torch.device) -> None:
    """Clear PyTorch's CUDA cache when VRAM is running tight."""
    if device.type != "cuda":
        return
    mem_reserved = torch.cuda.memory_reserved(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    if mem_reserved / total_mem > 0.85:
        torch.cuda.empty_cache()


def _limit_threads(num_threads: int) -> None:
    """Cap torch's threads and numpy's BLAS/OpenMP pools at `num_threads`;
    numpy fits (`_fits_on_numpy`) do not follow `torch.set_num_threads`."""
    torch.set_num_threads(num_threads)
    threadpool_limits(num_threads)


def _init_restart_worker_backend(
    num_threads: int,
    memory_efficient: bool = False,
) -> None:
    """Pool initializer: set TensorLy's backend, cap this worker's threads
    and register the memory-efficient MTTKRP if the parent did.

    All three are process-local and so do not survive "spawn".
    Without the thread cap each worker falls back to a thread pool sized to
    all visible cores, so `restart_procs` workers oversubscribe the CPU
    badly. `_repeat_with_restarts` picks `num_threads` to keep
    `restart_procs * num_threads` near the parent's `torch.get_num_threads()`.
    """
    tl.set_backend("pytorch")
    _limit_threads(num_threads)
    _maybe_register_memory_efficient_khatri_rao(memory_efficient)


def _in_worker_process() -> bool:
    """True if already running inside a multiprocessing worker.

    Used to refuse restart-level multiprocessing inside another pool (a
    caller's own): a second layer of processes adds startup and
    backend-init overhead without real parallelism, since the outer pool
    already uses all requested workers.
    """
    return current_process().name != "MainProcess"


def _maybe_register_memory_efficient_khatri_rao(enabled: bool) -> None:
    """Register TensorLy's memory-efficient MTTKRP backend method, if enabled.

    A global TensorLy registration shared by both decompositions, since both
    rely on the same underlying MTTKRP operation. `GMRITENSOR_MEMORY_EFFICIENT_MTTKRP`
    ("0" or "1"), when set, overrides `enabled`. Only a performance setting:
    both MTTKRPs compute the same product.
    """
    override = os.environ.get(MEMORY_EFFICIENT_MTTKRP_VARIABLE)
    if override is not None:
        if override not in ("0", "1"):
            raise ValueError(
                f"{MEMORY_EFFICIENT_MTTKRP_VARIABLE} must be 0 or 1, got {override!r}",
            )
        enabled = override == "1"
    if enabled:
        tl.tenalg.register_backend_method(
            "unfolding_dot_khatri_rao",
            unfolding_dot_khatri_rao_memory,
        )
        tl.tenalg.use_dynamic_dispatch()


def setup_backend() -> torch.device:
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
        _limit_threads(int(slurm_cpus) if slurm_cpus else 1)
        print(
            f"Running on: Multi-CPU ({torch.get_num_threads()} threads, "
            f"{FIT_BACKEND_VARIABLE}={os.environ.get(FIT_BACKEND_VARIABLE, 'numpy')})",
        )
    return device
