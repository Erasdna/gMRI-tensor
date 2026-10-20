"""CPU fits run on TensorLy's numpy backend in float64 (`_fits_on_numpy`).

These tests hold that path to the pytorch one: the same fits, returned as
the same torch types, stored and reloaded the same way, and never chosen
for a GPU fit.
"""
import itertools
from collections.abc import Callable
from typing import Any
from typing import cast

import numpy as np
import pytest
import tensorly as tl
import torch
from gMRItensor import compute_CMF_decomposition
from gMRItensor import compute_CP_decomposition
from gMRItensor import compute_PARAFAC2_decomposition
from gMRItensor import DirectoryStore
from gMRItensor import fit_task
from gMRItensor import FitTask
from gMRItensor import run_PARAFAC2_decomposition_repeated
from gMRItensor import setup_backend
from gMRItensor.decomposition import ConvergenceError
from gMRItensor.decomposition import FIT_BACKEND_VARIABLE
from gMRItensor.decomposition._backend import _FitBackend
from gMRItensor.decomposition._backend import _fits_on_numpy
from gMRItensor.decomposition._backend import _init_restart_worker_backend
from threadpoolctl import threadpool_info

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


def _slices(seed: int = 0) -> list[torch.Tensor]:
    """Ragged, non-negative, rank-2 PARAFAC2-like slices with a little noise."""
    rng = np.random.default_rng(seed)
    curves = np.abs(np.sin(np.linspace(0.3, 2.5, 6)[:, None] * [1.0, 2.0])) + 0.05
    amplitudes = rng.uniform(0.5, 1.5, (6, 2))
    labels = rng.random((12, 2))
    return [
        torch.from_numpy(
            (curves[:n] * amplitudes[i]) @ labels.T + 0.01 * rng.random((n, 12)),
        )
        for i, n in enumerate((4, 5, 6, 4, 5, 6))
    ]


def _cube(seed: int = 0, missing: bool = False) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    a, b, c = rng.random((6, 2)), rng.random((4, 2)), rng.random((12, 2))
    cube = np.einsum("ir,jr,kr->ijk", a, b, c) + 0.01 * rng.random((6, 4, 12))
    if missing:
        cube[0, 1, :] = np.nan
    return torch.from_numpy(cube)


def _fms(first: list[np.ndarray], second: list[np.ndarray]) -> float:
    """Factor match score: mean over components of the product of absolute
    cosines across modes, under the best component permutation."""

    def unit(matrix: np.ndarray) -> np.ndarray:
        return matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-300)

    congruence = np.prod(
        [np.abs(unit(x).T @ unit(y)) for x, y in zip(first, second)],
        axis=0,
    )
    rank = congruence.shape[0]
    return max(
        float(np.mean([congruence[i, j] for i, j in enumerate(order)]))
        for order in itertools.permutations(range(rank))
    )


def _cp(
    missing: bool = False,
    **kwargs: Any,
) -> Callable[[], tuple[float, list[np.ndarray]]]:
    def fit() -> tuple[float, list[np.ndarray]]:
        (_, factors), errors = compute_CP_decomposition(
            _cube(missing=missing),
            2,
            CP_max_iter=500,
            CP_tolerance=1e-6,
            random_state=1,
            **kwargs,
        )
        return float(errors[-1]), [f.double().numpy() for f in factors]

    return fit


def _coupled(compute: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    def fit() -> tuple[float, list[np.ndarray]]:
        model, errors, _ = compute(_slices(), 2, random_state=1, **kwargs)
        modes = [model.subject_mode, model.label_mode, *model.evolving_states]
        return float(errors[-1]), [m.double().numpy() for m in modes[:2]]

    return fit


FITS: dict[str, Callable[[], tuple[float, list[np.ndarray]]]] = {
    "cp-non-negative": _cp(),
    "cp-unconstrained": _cp(non_negative=False),
    "cp-imputing-nan": _cp(allow_nan_imputation=True, missing=True),
    "parafac2-tensorly": _coupled(
        compute_PARAFAC2_decomposition,
        solver="tensorly",
        PARAFAC2_max_iter=500,
        PARAFAC2_tolerance=1e-6,
    ),
    "parafac2-matcouply": _coupled(
        compute_PARAFAC2_decomposition,
        solver="matcouply",
        PARAFAC2_max_iter=300,
        return_projections=True,
    ),
    "cmf": _coupled(compute_CMF_decomposition, CMF_max_iter=300),
}


# The torch side of these two is slow: TensorLy's HALS loop on torch, and
# compiling CP.
_SLOW_ON_TORCH = ("cp-non-negative", "parafac2-tensorly")


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=[pytest.mark.slow] if name in _SLOW_ON_TORCH else [])
        for name in FITS
    ],
)
def test_numpy_fits_match_float64_torch_fits(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same seed, same data: the numpy path reproduces the torch one (both
    initialise from the same numpy generator)."""
    setup_backend()
    monkeypatch.setenv(FIT_BACKEND_VARIABLE, "numpy")
    numpy_error, numpy_factors = FITS[name]()
    monkeypatch.setenv(FIT_BACKEND_VARIABLE, "torch")
    torch_error, torch_factors = FITS[name]()

    assert numpy_error == pytest.approx(torch_error, rel=1e-8)
    assert _fms(numpy_factors, torch_factors) > 0.9999


def _all_tensors(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return [item for part in value for item in _all_tensors(part)]
    return [value] if value is not None else []


def test_numpy_fits_return_torch_and_restore_the_backend() -> None:
    setup_backend()

    model, errors, diagnostics = compute_PARAFAC2_decomposition(
        _slices(),
        2,
        PARAFAC2_max_iter=300,
        return_projections=True,
    )

    assert tl.get_backend() == "pytorch"
    parts = [model.weights, model.subject_mode, model.label_mode]
    parts += [*model.evolving_states, *_all_tensors(model.kiers), *errors]
    assert all(isinstance(part, torch.Tensor) for part in parts)
    assert {part.device for part in parts} == {CPU}
    for name, value in vars(diagnostics).items():
        assert not isinstance(value, (np.ndarray, np.generic)), name


def test_a_failed_numpy_fit_restores_the_backend() -> None:
    setup_backend()

    with pytest.raises(ConvergenceError):
        compute_PARAFAC2_decomposition(
            _slices(),
            2,
            PARAFAC2_max_iter=3,
            PARAFAC2_tolerance=1e-12,
        )

    assert tl.get_backend() == "pytorch"


@pytest.mark.parametrize("method", ["CP", "PARAFAC2", "CMF"])
def test_numpy_fits_round_trip_through_the_store(tmp_path: Any, method: str) -> None:
    """The store loads with `weights_only=True`, which refuses numpy values:
    any left in a result would fail here."""
    setup_backend()
    data: Any = _cube().float() if method == "CP" else [s.float() for s in _slices()]
    task = FitTask("full", tuple(range(6)), 0)
    result = fit_task(task, data, 2, cast(Any, method), max_iter=300, tolerance=1e-4)
    assert result.ok, result.failure_message
    store = DirectoryStore(tmp_path)

    store.save(result)
    loaded = store.load(task)

    assert loaded is not None and loaded.error == result.error
    assert loaded.diagnostics == result.diagnostics


def test_gpu_fits_always_run_on_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FIT_BACKEND_VARIABLE, raising=False)
    assert _fits_on_numpy(CPU)
    assert not _fits_on_numpy(CUDA)

    monkeypatch.setenv(FIT_BACKEND_VARIABLE, "torch")
    assert not _fits_on_numpy(CPU)

    monkeypatch.setenv(FIT_BACKEND_VARIABLE, "numpy")
    with pytest.warns(UserWarning, match="GPU fits always run on pytorch"):
        assert not _fits_on_numpy(CUDA)

    monkeypatch.setenv(FIT_BACKEND_VARIABLE, "jax")
    with pytest.raises(ValueError, match="numpy or torch"):
        _fits_on_numpy(CPU)


def test_a_gpu_fit_backend_converts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """On the GPU every value passes through untouched: no numpy conversion
    and no change of TensorLy backend."""
    from gMRItensor.decomposition import _backend

    def no_conversion(*args: Any) -> Any:
        raise AssertionError("converted")

    monkeypatch.setattr(_backend, "_solver_input", no_conversion)
    monkeypatch.setattr(_backend, "_solver_output", no_conversion)
    backend = _FitBackend(CUDA)
    value = torch.ones(2)
    # The patch is live: a CPU backend does convert, and so trips it.
    with pytest.raises(AssertionError, match="converted"):
        _FitBackend(CPU).input(value)

    assert backend.input(value) is value
    assert backend.call(lambda: tl.get_backend()) == tl.get_backend()
    assert backend.output(value, torch.float32) is value


def test_restart_workers_cap_numpy_threads_too() -> None:
    """numpy's BLAS ignores `torch.set_num_threads`: without its own cap
    every restart worker would use all cores."""
    original = torch.get_num_threads()
    try:
        _init_restart_worker_backend(1)
        assert torch.get_num_threads() == 1
        blas = [info for info in threadpool_info() if info["user_api"] == "blas"]
        assert blas and all(info["num_threads"] == 1 for info in blas)
    finally:
        torch.set_num_threads(original)


def test_numpy_restarts_in_processes_match_sequential_ones() -> None:
    device = setup_backend()
    common: dict[str, Any] = dict(
        rank=2,
        max_iter=300,
        init_repeats=3,
        device=device,
        progress_bar=False,
    )

    sequential, error = run_PARAFAC2_decomposition_repeated(_slices(), **common)
    parallel, parallel_error = run_PARAFAC2_decomposition_repeated(
        _slices(),
        restart_procs=2,
        **common,
    )

    assert float(parallel_error) == float(error)
    torch.testing.assert_close(parallel.subject_mode, sequential.subject_mode)
