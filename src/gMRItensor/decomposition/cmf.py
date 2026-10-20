"""Non-negative coupled matrix factorization (matcouply's cmf_aoadmm):
free per-subject time courses on a shared label mode.
"""
from typing import Any
from typing import Literal

import tensorly as tl
import torch
from matcouply._utils import get_svd
from matcouply.decomposition import cmf_aoadmm
from matcouply.decomposition import initialize_cmf

from ._aoadmm import _accept_aoadmm_fit
from ._aoadmm import _build_aoadmm_options
from ._aoadmm import _CMF_MANAGED_OPTIONS
from ._aoadmm import _nn_modes_to_non_negative
from ._aoadmm import _zero_negligible_modes
from ._aoadmm import AOADMM_DEFAULT_LOSS_TOL
from ._backend import _as_slices
from ._backend import _check_tolerances
from ._backend import _fit_device
from ._backend import _FitBackend
from ._backend import _matcouply_numeric_context
from ._backend import _solver_default
from ._backend import _to_solver_slices
from .models import CMFModel
from .models import PARAFAC2Diagnostics


def _subject_amplitudes(evolving_states: list[torch.Tensor]) -> torch.Tensor:
    """`(n_subjects, rank)` RMS of each subject's time course per component."""
    return torch.stack([B.pow(2).mean(dim=0).sqrt() for B in evolving_states])


def compute_CMF_decomposition(
    tensor_slices: list[torch.Tensor] | torch.Tensor,
    rank: int,
    CMF_max_iter: int = 2000,
    random_state: int = 0,
    init: str = "random",
    CMF_verbose_level: int = 0,
    CMF_tolerance: float | None = None,
    nn_modes: tuple[int, ...] | None | Literal["auto"] = "auto",
    aoadmm_options: dict[str, Any] | None = None,
    aoadmm_loss_tolerance: float | None = None,
    negligible_rtol: float = 1e-6,
    compute_device: torch.device | None = None,
) -> tuple[CMFModel, list[torch.Tensor], PARAFAC2Diagnostics]:
    """Compute a single non-negative coupled matrix factorization attempt.

    Each subject's `(n_timepoints_i, n_labels)` slice is modelled as
    `B_i @ C.T`: a subject-specific time course `B_i` per component, and a
    label mode `C` shared by all subjects. Unlike PARAFAC2 nothing ties the
    `B_i` together (see `CMFModel`).

    Fitted with matcouply's `cmf_aoadmm`, following Chatzis et al.,
    "Revealing Subject-specific Temporal Patterns from Longitudinal Data":
    the subject weights are fixed at one (`update_A=False`) and each `B_i`
    carries its subject's amplitude. Every restart starts from matcouply's
    `init` with those weights replaced by ones.

    `nn_modes="auto"` constrains the time (1) and label (2) modes; mode 0 is
    fixed, so constraining it has no effect. `aoadmm_options` takes
    `cmf_aoadmm`'s penalties and solver settings (see
    `_AOADMM_PASSTHROUGH_OPTIONS`); unlike the PARAFAC2 wrapper none are set
    here, so matcouply's own defaults apply. Tolerances work as for the
    matcouply PARAFAC2 solver (see `compute_PARAFAC2_decomposition`), with
    `CMF_tolerance` in place of `PARAFAC2_tolerance`.

    There is no degeneracy check: the PARAFAC2 one relies on the constraint
    making `B_i.T @ B_i` the same for every subject.

    Raises
    ------
    ValueError
        NaN input, or an unknown or wrapper-managed `aoadmm_options` key.
    ConvergenceError
        The constraints were left infeasible, or the fit did not converge.
    """
    _check_tolerances(
        CMF_tolerance=CMF_tolerance,
        aoadmm_loss_tolerance=aoadmm_loss_tolerance,
    )
    slices = _as_slices(tensor_slices)
    if any(torch.isnan(s).any() for s in slices):
        raise ValueError(
            "tensor_slices contains NaN values, which the CMF fit does not "
            "impute -- remove or impute missing values first.",
        )
    nn_modes = (1, 2) if nn_modes == "auto" else nn_modes
    options = _build_aoadmm_options(
        aoadmm_options,
        managed=_CMF_MANAGED_OPTIONS,
        defaults={},
        owner="compute_CMF_decomposition",
    )
    feasibility_tol = options.get(
        "feasibility_tol",
        _solver_default(cmf_aoadmm, "feasibility_tol"),
    )
    if aoadmm_loss_tolerance is None:
        aoadmm_loss_tolerance = AOADMM_DEFAULT_LOSS_TOL
    device = _fit_device(compute_device, slices[0].device)

    backend = _FitBackend(device)
    non_negative = _nn_modes_to_non_negative(nn_modes, "CMF")

    def fit(matrices: list[Any]) -> Any:
        # One generator for the initialisation and the ADMM variables, so a
        # seed fixes the whole fit.
        rng = tl.check_random_state(random_state)
        _, (A, B_is, C) = initialize_cmf(
            matrices,
            rank,
            init,
            svd_fun=get_svd(options.get("svd", _solver_default(cmf_aoadmm, "svd"))),
            random_state=rng,
            init_params=options.get("init_params"),
        )
        return cmf_aoadmm(
            matrices,
            rank=rank,
            init=(None, (tl.ones(tl.shape(A), **tl.context(A)), B_is, C)),
            n_iter_max=CMF_max_iter,
            tol=aoadmm_loss_tolerance,
            random_state=rng,
            non_negative=non_negative,
            verbose=CMF_verbose_level,
            update_A=False,
            return_errors=True,
            **options,
        )

    with _matcouply_numeric_context(device):
        cmf, diagnostics = backend.output(
            backend.call(fit, backend.input(_to_solver_slices(slices, device))),
            torch.float64,
        )
        _, (_, evolving_states, C) = cmf
        _, evolving_states, C = _zero_negligible_modes(
            None,
            list(evolving_states),
            C,
            nn_modes,
            negligible_rtol,
        )
        result = CMFModel(
            weights=torch.ones(rank, dtype=C.dtype, device=C.device),
            subject_mode=_subject_amplitudes(evolving_states),
            evolving_states=evolving_states,
            label_mode=C,
        )

    errors, cmf_diagnostics = _accept_aoadmm_fit(
        diagnostics,
        CMF_max_iter,
        CMF_tolerance,
        aoadmm_loss_tolerance,
        feasibility_tol,
        nn_modes,
        "CMF",
    )
    return result, errors, cmf_diagnostics
