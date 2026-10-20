"""CP, PARAFAC2 and CMF fits with repeated random restarts.

The public names of the modules below, re-exported:

- `models`: Types of the decomposition API: models, diagnostics, errors and the
- `_backend`: Where a fit runs: TensorLy's numpy backend in float64 for CPU fits,
- `_aoadmm`: matcouply's AO-ADMM, shared by PARAFAC2 and CMF: options, the
- `cp`: CP (PARAFAC) fits: one time course per component, shared by all
- `parafac2`: PARAFAC2 fits, by TensorLy (ALS/HALS) or matcouply (AO-ADMM): one time
- `cmf`: Non-negative coupled matrix factorization (matcouply's cmf_aoadmm):
- `restarts`: Repeated random restarts: one fit per seed (`_fit_one`), the best of
"""
from ._aoadmm import AOADMM_DEFAULT_LOSS_TOL
from ._backend import FIT_BACKEND_VARIABLE
from ._backend import setup_backend
from .cmf import compute_CMF_decomposition
from .cp import compute_CP_decomposition
from .models import CMFModel
from .models import CMFResult
from .models import CMFResultWithDiagnostics
from .models import ConvergenceError
from .models import ConvergenceFailureReason
from .models import EVOLVING_MODELS
from .models import Method
from .models import PARAFAC2Diagnostics
from .models import PARAFAC2Model
from .models import PARAFAC2Result
from .models import PARAFAC2ResultWithDiagnostics
from .models import PARAFAC2Solver
from .models import RestartTally
from .parafac2 import compute_PARAFAC2_decomposition
from .restarts import run_CMF_decomposition_repeated
from .restarts import run_CP_decomposition_repeated
from .restarts import run_PARAFAC2_decomposition_repeated

__all__ = [
    "CMFModel",
    "CMFResult",
    "CMFResultWithDiagnostics",
    "ConvergenceError",
    "ConvergenceFailureReason",
    "EVOLVING_MODELS",
    "Method",
    "PARAFAC2Diagnostics",
    "PARAFAC2Model",
    "PARAFAC2Result",
    "PARAFAC2ResultWithDiagnostics",
    "PARAFAC2Solver",
    "RestartTally",
    "FIT_BACKEND_VARIABLE",
    "setup_backend",
    "AOADMM_DEFAULT_LOSS_TOL",
    "compute_CP_decomposition",
    "compute_PARAFAC2_decomposition",
    "compute_CMF_decomposition",
    "run_CMF_decomposition_repeated",
    "run_CP_decomposition_repeated",
    "run_PARAFAC2_decomposition_repeated",
]
