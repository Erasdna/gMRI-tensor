"""Tensor decompositions of gadolinium-enhanced MRI time series.

The `gmri` command line (`gMRItensor.cli`) runs the pipeline step by step;
this namespace exports the decomposition and restart machinery for use from
Python. See README.md and docs/cli.md.
"""
import importlib.metadata

from . import _third_party_warnings  # noqa: F401
from .decomposition import CMFModel
from .decomposition import compute_CMF_decomposition
from .decomposition import compute_CP_decomposition
from .decomposition import compute_PARAFAC2_decomposition
from .decomposition import ConvergenceError
from .decomposition import PARAFAC2Diagnostics
from .decomposition import PARAFAC2Model
from .decomposition import PARAFAC2Solver
from .decomposition import RestartTally
from .decomposition import run_CMF_decomposition_repeated
from .decomposition import run_CP_decomposition_repeated
from .decomposition import run_PARAFAC2_decomposition_repeated
from .decomposition import setup_backend
from .jobs import collect
from .jobs import DirectoryStore
from .jobs import fit_task
from .jobs import FitTask
from .jobs import GroupSummary
from .jobs import InMemoryStore
from .jobs import job_slice
from .jobs import n_jobs
from .jobs import plan_replicability
from .jobs import plan_restarts
from .jobs import RestartResult
from .jobs import ResultStore
from .jobs import run_tasks

__version__ = importlib.metadata.version(__package__)


__all__ = [
    "CMFModel",
    "compute_CMF_decomposition",
    "compute_CP_decomposition",
    "compute_PARAFAC2_decomposition",
    "ConvergenceError",
    "PARAFAC2Diagnostics",
    "PARAFAC2Model",
    "PARAFAC2Solver",
    "RestartTally",
    "run_CMF_decomposition_repeated",
    "run_CP_decomposition_repeated",
    "run_PARAFAC2_decomposition_repeated",
    "setup_backend",
    "collect",
    "DirectoryStore",
    "fit_task",
    "FitTask",
    "GroupSummary",
    "InMemoryStore",
    "job_slice",
    "n_jobs",
    "plan_replicability",
    "plan_restarts",
    "RestartResult",
    "ResultStore",
    "run_tasks",
]
