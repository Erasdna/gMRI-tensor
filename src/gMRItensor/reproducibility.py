"""Reproducibility: do repeated restarts on the same data reach the same
minimum?

Every successful restart is compared with every other one by FMS, and the
restarts are ordered by reconstruction error. If the low-error restarts have
found the same minimum, they form a high-FMS block in the top-left corner of
the matrix.
"""
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .jobs import RestartResult
from .replicability import _weights_and_factors
from .replicability import ComparableFit
from .replicability import compute_FMS

#: The file `write_reproducibility` writes and `read_reproducibility` reads.
REPRODUCIBILITY_FILE = "reproducibility.csv"


def _restart_fit(result: RestartResult) -> ComparableFit:
    """A restart as `(weights, factors)` that `compute_FMS` can score.

    Restarts fit the same subjects in the same order, so no alignment is
    needed. Only a ragged evolving mode (PARAFAC2's and CMF's `B_i`) is
    stacked into one `(sum_i J_i, rank)` matrix, since `factor_match_score`
    rejects lists. `compute_FMS` converts the fit to numpy itself.
    """
    weights, factors = _weights_and_factors(result.model)
    subject_mode, evolving, label_mode = factors
    if not isinstance(evolving, torch.Tensor):
        evolving = torch.cat(list(evolving), dim=0)
    return weights, [subject_mode, evolving, label_mode]


def restart_fms_matrix(
    results: Sequence[RestartResult],
) -> tuple[pd.DataFrame, pd.Series]:
    """Pairwise FMS between the successful restarts, ordered by error.

    Failed restarts are left out. Ties in error keep seed order.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        The symmetric FMS matrix (diagonal 1), indexed by seed on both axes,
        and each restart's reconstruction error, in the same order.
    """
    succeeded = sorted(
        (result for result in results if result.ok),
        key=lambda result: (result.error, result.task.seed),
    )
    seeds = [result.task.seed for result in succeeded]
    fits = [_restart_fit(result) for result in succeeded]

    fms = np.eye(len(fits))
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            fms[i, j] = fms[j, i] = compute_FMS(fits[i], fits[j])

    index = pd.Index(seeds, name="seed")
    errors = pd.Series(
        [float(result.error) for result in succeeded],  # type: ignore[arg-type]
        index=index,
        name="error",
    )
    return pd.DataFrame(fms, index=index, columns=index), errors


def write_reproducibility(
    results: Sequence[RestartResult],
    directory: Path,
) -> Path:
    """Write `<directory>/reproducibility.csv`; return its path.

    One row per successful restart, sorted by error: `seed`, `error`, then
    the FMS against every restart, one column per seed in the same order.
    """
    fms, errors = restart_fms_matrix(results)
    table = fms.copy()
    table.columns = [str(seed) for seed in fms.columns]
    table.insert(0, "error", errors)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REPRODUCIBILITY_FILE
    table.to_csv(path)
    return path


def read_reproducibility(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """The FMS matrix and errors `write_reproducibility` saved to `path`."""
    table = pd.read_csv(path, index_col="seed")
    errors = table.pop("error")
    table.columns = pd.Index([int(seed) for seed in table.columns], name="seed")
    return table, errors
