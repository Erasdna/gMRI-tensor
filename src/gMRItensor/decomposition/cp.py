"""CP (PARAFAC) fits: one time course per component, shared by all
subjects.
"""
from typing import Any

import tensorly as tl
import torch
from tlviz.factor_tools import degeneracy_score

from ._backend import _check_tolerances
from ._backend import _FitBackend
from ._backend import _solver_default
from .models import ConvergenceError
from .models import DEGENERACY_THRESHOLD


@torch.no_grad()
@torch.compile(dynamic=True)
def non_negative_parafac_compiled(tensor: torch.Tensor, **kwargs: Any) -> Any:
    """TensorLy's `non_negative_parafac`, compiled; used for GPU fits only
    (on the CPU, compiling costs start-up and recompiles without gain)."""
    return tl.decomposition.non_negative_parafac(tensor, **kwargs)


@torch.no_grad()
@torch.compile(dynamic=True)
def parafac_compiled(tensor: torch.Tensor, **kwargs: Any) -> Any:
    """TensorLy's `parafac`, compiled; used for GPU fits only."""
    return tl.decomposition.parafac(tensor, **kwargs)


def compute_CP_decomposition(
    tensor: torch.Tensor,
    rank: int,
    CP_max_iter: int = 500,
    random_state: int = 0,
    init: str = "random",
    CP_verbose_level: int = 0,
    CP_tolerance: float | None = None,
    allow_nan_imputation: bool = False,
    non_negative: bool = True,
) -> tuple[Any, list[torch.Tensor]]:
    """Compute a single CP/PARAFAC decomposition attempt.

    `allow_nan_imputation` treats NaN entries as missing and imputes them
    from the model's own reconstruction each iteration, via TensorLy's
    `mask`; a tensor without NaN is fitted unmasked, as with it off. The
    last returned error is then measured on the observed entries only
    (TensorLy's own includes the imputed ones). Off by default, where NaN
    input raises instead. PARAFAC2 has no equivalent (see
    `compute_PARAFAC2_decomposition`).

    `non_negative` defaults to True, appropriate for a tracer signal that is
    physically non-negative. `CP_tolerance=None` uses TensorLy's own default
    `tol` for the chosen solver. The factors are not normalised, so the
    weights are ones (see `PARAFAC2Model`).

    The fit runs in float64 on every backend and device, as PARAFAC2 and CMF
    do, so fits are comparable wherever they ran; the result is float64.
    """
    _check_tolerances(CP_tolerance=CP_tolerance)
    tensor = tensor.double()
    mask = None
    has_nan = bool(torch.isnan(tensor).any())
    if allow_nan_imputation and has_nan:
        mask = (~torch.isnan(tensor)).to(tensor.dtype)
        tensor = torch.nan_to_num(tensor, nan=0.0)
    elif has_nan:
        raise ValueError(
            "tensor contains NaN values; pass allow_nan_imputation=True to let "
            "TensorLy impute them during fitting, or remove/fill them yourself "
            "first.",
        )

    library_fn = (
        tl.decomposition.non_negative_parafac
        if non_negative
        else tl.decomposition.parafac
    )
    if CP_tolerance is None:
        CP_tolerance = _solver_default(library_fn)
    backend = _FitBackend(tensor.device, tensor.numel())
    # Compiled for the GPU; CPU fits (numpy or torch) call TensorLy directly.
    compiled_fn = non_negative_parafac_compiled if non_negative else parafac_compiled
    decomp, errors = backend.output(
        backend.call(
            compiled_fn if tensor.device.type == "cuda" else library_fn,
            backend.input(tensor),
            rank=rank,
            n_iter_max=CP_max_iter,
            tol=CP_tolerance,
            return_errors=True,
            random_state=random_state,
            verbose=CP_verbose_level,
            init=init,
            mask=backend.input(mask),
        ),
        tensor.dtype,
    )
    if len(errors) > CP_max_iter - 1:
        raise ConvergenceError(
            "Decomposition did not converge within the maximum iteration count",
            reason="max_iter",
            n_iter=len(errors),
        )

    w, f = decomp
    w = w.float()
    f = [ff.float() for ff in f]
    if degeneracy_score((w, f)) < DEGENERACY_THRESHOLD:
        raise ConvergenceError(
            "Decomposition is degenerate",
            reason="degenerate",
            n_iter=len(errors),
        )

    if mask is not None:
        errors = [*errors[:-1], _observed_error(tensor, decomp, mask)]
    return decomp, errors


def _observed_error(
    tensor: torch.Tensor,
    decomp: Any,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Relative reconstruction error of `decomp` over the entries `mask`
    marks as observed: `||M*(X - X_hat)|| / ||M*X||`."""
    residual = mask * (tensor - tl.cp_to_tensor(decomp))
    return torch.linalg.norm(residual) / torch.linalg.norm(mask * tensor)
