"""Scatter/gather random restarts, centralised or distributed.

A *plan* is a flat list of `FitTask`s -- one per `(group, seed)` -- where a
group is a set of subjects fitted together: `"full"` for a plain restart
run, or a replicability engine's split/fold id. The same plan can be run

- centralised: `run_tasks(plan, ..., store=InMemoryStore())`, in-process or
  over a process pool, or
- distributed: each job runs `run_tasks(job_slice(plan, i, k), ...)` into a
  shared `ResultStore`, and a final step calls `collect(plan, store)`.

Both paths fit each seed through the same `_fit_one` and select winners with
the same `_BestOfRestarts` as `run_*_decomposition_repeated`, so they give
identical results. Nothing here knows about schedulers; the only file
format is `DirectoryStore`, a default for callers without their own.
"""
import dataclasses
import math
import os
import re
import warnings
from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Protocol
from typing import TYPE_CHECKING
from typing import Union

import torch
from numpy.typing import ArrayLike
from tqdm import tqdm

from .decomposition import _BestOfRestarts
from .decomposition import _compute_kwargs
from .decomposition import _fit_one
from .decomposition import _init_restart_worker_backend
from .decomposition import _maybe_register_memory_efficient_khatri_rao
from .decomposition import _release_cuda_cache
from .decomposition import _resolve_options
from .decomposition import _warn_parafac2_outcome
from .decomposition import ConvergenceError
from .decomposition import PARAFAC2Diagnostics
from .decomposition import PARAFAC2Model
from .decomposition import RestartTally

if TYPE_CHECKING:
    from .replicability import ReplicabilityEngine

Method = Literal["CP", "PARAFAC2"]
#: A regular tensor (samples x ...), or a list of per-subject slices.
Data = Union[torch.Tensor, list[torch.Tensor]]
#: A CP fit as `(weights, factors)`, CPU float32.
CPModel = tuple[torch.Tensor, list[torch.Tensor]]
#: `transform(subset) -> subset`, applied once per group after row selection.
Transform = Callable[[Any], Any]


@dataclass(frozen=True)
class FitTask:
    """One restart of one group: fit `data[indices]` from `seed`."""

    group: Hashable
    indices: tuple[int, ...]
    seed: int


def plan_restarts(n_samples: int, n_restarts: int) -> list[FitTask]:
    """Plan `n_restarts` seeds of a fit on all `n_samples` subjects."""
    _check_positive("n_restarts", n_restarts)
    indices = tuple(range(n_samples))
    return [FitTask("full", indices, seed) for seed in range(n_restarts)]


def plan_replicability(
    engine: "ReplicabilityEngine",
    n_samples: int,
    n_restarts: int,
    stratification: ArrayLike | None = None,
) -> list[FitTask]:
    """Plan `n_restarts` seeds for each of `engine`'s splits or folds.

    Ordered by group, then seed, so a contiguous `job_slice` mostly stays
    within one group and `run_tasks` builds few subsets. The engine's splits
    depend only on its own `seed`, so every job regenerates the same plan.
    """
    _check_positive("n_restarts", n_restarts)
    return [
        FitTask(group, tuple(int(i) for i in indices), seed)
        for group, indices in engine.generate_tasks(n_samples, stratification)
        for seed in range(n_restarts)
    ]


def n_jobs(plan: Sequence[FitTask], tasks_per_job: int) -> int:
    """Number of jobs needed to run `plan` in blocks of `tasks_per_job`."""
    _check_positive("tasks_per_job", tasks_per_job)
    return math.ceil(len(plan) / tasks_per_job)


def job_slice(
    plan: Sequence[FitTask],
    job_index: int,
    tasks_per_job: int,
) -> list[FitTask]:
    """The tasks job `job_index` runs; the slices cover `plan` exactly once."""
    total = n_jobs(plan, tasks_per_job)
    if not 0 <= job_index < total:
        raise IndexError(
            f"job_index={job_index} is out of range for {len(plan)} tasks in "
            f"blocks of {tasks_per_job} ({total} jobs).",
        )
    start = job_index * tasks_per_job
    stop = start + tasks_per_job
    return list(plan[start:stop])


@dataclass(frozen=True)
class RestartResult:
    """What one `FitTask` produced: a model, or why it was rejected.

    Failures are stored as plain values rather than an exception, so any
    `ResultStore` can serialise them; `failure()` rebuilds the
    `ConvergenceError` for tallying.
    """

    task: FitTask
    model: PARAFAC2Model | CPModel | None = None
    error: float | None = None
    diagnostics: PARAFAC2Diagnostics | None = None
    failure_reason: str | None = None
    failure_message: str | None = None
    failure_level: float | None = None
    failure_suggested_max_iter: int | None = None

    @property
    def ok(self) -> bool:
        return self.model is not None

    def failure(self) -> ConvergenceError | None:
        if self.ok:
            return None
        return ConvergenceError(
            self.failure_message or "",
            reason=self.failure_reason,  # type: ignore[arg-type]
            level_reached=self.failure_level,
            suggested_max_iter=self.failure_suggested_max_iter,
        )


def _result_from_outcome(
    task: FitTask,
    outcome: tuple[Any, float | None, PARAFAC2Diagnostics | None, Any],
) -> RestartResult:
    model, error, diagnostics, failure = outcome
    if model is not None:
        return RestartResult(task, model, error, diagnostics)
    return RestartResult(
        task,
        failure_reason=failure.reason,
        failure_message=str(failure),
        failure_level=failure.level_reached,
        failure_suggested_max_iter=failure.suggested_max_iter,
    )


def _subset(data: Data, indices: tuple[int, ...], method: Method) -> Data:
    """Rows `indices` of `data`: a tensor for CP, a slice list for PARAFAC2."""
    if method == "CP":
        if not isinstance(data, torch.Tensor):
            raise TypeError("method='CP' needs a regular torch.Tensor.")
        return data[list(indices)]
    return [data[i] for i in indices]


def _data_device(data: Data) -> torch.device:
    """Device the input lives on (the first slice's, for a slice list)."""
    return data.device if isinstance(data, torch.Tensor) else data[0].device


def _run_task(
    args: tuple[FitTask, Data, Method, dict[str, Any], bool],
) -> RestartResult:
    """Fit one task on its already-built subset. Picklable pool worker."""
    task, subset, method, compute_kwargs, memory_efficient = args
    _maybe_register_memory_efficient_khatri_rao(memory_efficient)
    outcome = _fit_one((method, task.seed, subset, compute_kwargs))
    return _result_from_outcome(task, outcome)


def fit_task(
    task: FitTask,
    data: Data,
    rank: int,
    method: Method,
    transform: Transform | None = None,
    **kwargs: Any,
) -> RestartResult:
    """Fit a single task, returning a non-convergence as a value.

    `kwargs` are `run_*_decomposition_repeated` options, with the same
    defaults; restart-loop options (`init_repeats`, `restart_procs`, ...) are
    accepted and ignored. Invalid options raise as they would there.
    """
    resolved = _resolve_options(method, kwargs)
    subset = _subset(data, task.indices, method)
    if transform is not None:
        subset = transform(subset)
    return _run_task(
        (
            task,
            subset,
            method,
            _compute_kwargs(method, rank, resolved),
            resolved["use_memory_efficient_khatri_rao"],
        ),
    )


class ResultStore(Protocol):
    """Where `run_tasks` puts results and `collect` reads them."""

    def save(self, result: RestartResult) -> None:
        ...

    def load(self, task: FitTask) -> RestartResult | None:
        """The stored result for `task`, or None if it is missing."""
        ...


class InMemoryStore:
    """A `ResultStore` in a dict, for centralised runs."""

    def __init__(self) -> None:
        self.results: dict[FitTask, RestartResult] = {}

    def save(self, result: RestartResult) -> None:
        self.results[result.task] = result

    def load(self, task: FitTask) -> RestartResult | None:
        return self.results.get(task)


_SLUG_PART = re.compile(r"^[A-Za-z0-9.-]+$")


def _group_slug(group: Hashable) -> str:
    """A directory name for `group`: `"full"`, `3`, or `(2, 1)` -> `2_1`."""
    parts = group if isinstance(group, tuple) else (group,)
    slugs = []
    for part in parts:
        if isinstance(part, bool) or not isinstance(part, (int, str)):
            raise TypeError(
                f"DirectoryStore cannot name group {group!r}: groups must be "
                "an int, a str or a tuple of them.",
            )
        if not _SLUG_PART.match(str(part)):
            raise ValueError(
                f"DirectoryStore cannot name group {group!r}: {part!r} is not "
                "made of letters, digits, '.' and '-'.",
            )
        slugs.append(str(part))
    return "_".join(slugs)


def _model_record(model: PARAFAC2Model | CPModel | None) -> dict[str, Any] | None:
    if model is None:
        return None
    if isinstance(model, PARAFAC2Model):
        return {"kind": "PARAFAC2", **dataclasses.asdict(model)}
    weights, factors = model
    return {"kind": "CP", "weights": weights, "factors": list(factors)}


def _model_from_record(
    record: dict[str, Any] | None,
) -> PARAFAC2Model | CPModel | None:
    if record is None:
        return None
    if record["kind"] == "CP":
        return record["weights"], list(record["factors"])
    fields = {k: v for k, v in record.items() if k != "kind"}
    if fields["kiers"] is not None:
        coordinate_matrix, projections = fields["kiers"]
        fields["kiers"] = (coordinate_matrix, list(projections))
    return PARAFAC2Model(**fields)


class DirectoryStore:
    """A `ResultStore` with one `torch.save` file per task.

    Files live at `root/<group>/seed_<NNNN>.pt` and hold only tensors and
    plain values, so they load with `weights_only=True`. Writes go to a
    temporary file first and are renamed into place, so a job killed
    mid-write never leaves a half-written result for `collect` to read.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def path(self, task: FitTask) -> Path:
        return self.root / _group_slug(task.group) / f"seed_{task.seed:04d}.pt"

    def save(self, result: RestartResult) -> None:
        path = self.path(result.task)
        path.parent.mkdir(parents=True, exist_ok=True)
        task = result.task
        record = {
            "task": {
                "group": task.group,
                "indices": list(task.indices),
                "seed": task.seed,
            },
            "model": _model_record(result.model),
            "error": result.error,
            "diagnostics": (
                None
                if result.diagnostics is None
                else dataclasses.asdict(result.diagnostics)
            ),
            "failure_reason": result.failure_reason,
            "failure_message": result.failure_message,
            "failure_level": result.failure_level,
            "failure_suggested_max_iter": result.failure_suggested_max_iter,
        }
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(record, tmp)
        os.replace(tmp, path)

    def load(self, task: FitTask) -> RestartResult | None:
        path = self.path(task)
        if not path.exists():
            return None
        record = torch.load(path, weights_only=True)
        stored = record["task"]
        stored_task = FitTask(
            stored["group"],
            tuple(stored["indices"]),
            stored["seed"],
        )
        if stored_task != task:
            raise ValueError(
                f"{path} holds {stored_task}, not the requested {task}. The "
                "store was written by a different plan (other splits, subject "
                "set or n_restarts); use a fresh directory.",
            )
        diagnostics = record["diagnostics"]
        return RestartResult(
            task=task,
            model=_model_from_record(record["model"]),
            error=record["error"],
            diagnostics=(
                None if diagnostics is None else PARAFAC2Diagnostics(**diagnostics)
            ),
            failure_reason=record["failure_reason"],
            failure_message=record["failure_message"],
            failure_level=record["failure_level"],
            failure_suggested_max_iter=record["failure_suggested_max_iter"],
        )


def _with_subsets(
    tasks: Sequence[FitTask],
    data: Data,
    method: Method,
    transform: Transform | None,
) -> Iterator[tuple[FitTask, Data]]:
    """Pair each task with its subset, built once per run of equal groups."""
    current: tuple[Hashable, tuple[int, ...]] | None = None
    subset: Data | None = None
    for task in tasks:
        key = (task.group, task.indices)
        if key != current:
            subset = _subset(data, task.indices, method)
            if transform is not None:
                subset = transform(subset)
            current = key
        assert subset is not None
        yield task, subset


def run_tasks(
    tasks: Sequence[FitTask],
    data: Data,
    rank: int,
    method: Method,
    store: ResultStore,
    n_procs: int = 1,
    transform: Transform | None = None,
    progress_bar: bool = True,
    **kwargs: Any,
) -> None:
    """Fit `tasks` and save every result, failures included, to `store`.

    Tasks already in `store` are skipped, so a re-queued job resumes rather
    than refitting. Each group's subset is built (and `transform`ed) once, in
    this process, as the tasks reach it.

    `kwargs` are `run_*_decomposition_repeated` options -- see `fit_task`.
    `n_procs >= 2` spreads the tasks over a spawned process pool, each worker
    pinned to its share of `torch.get_num_threads()`.

    Raises
    ------
    ValueError
        If `n_procs >= 2` and the fit runs on CUDA. Worker processes against
        a CUDA context are unreliable across driver setups, so this is
        rejected rather than silently run sequentially.
    """
    resolved = _resolve_options(method, kwargs)
    compute_device = resolved.get("compute_device")
    fit_device = compute_device if compute_device is not None else _data_device(data)
    if n_procs >= 2 and fit_device.type == "cuda":
        raise ValueError(
            f"n_procs={n_procs} requests multiprocessing, but the fit runs on "
            "CUDA. Running multiple worker processes against a CUDA context "
            "is unsafe/unreliable across GPU driver setups -- pass n_procs=1 "
            "to run sequentially on the GPU, or fit on CPU (data and "
            "compute_device) to use multiple processes.",
        )

    pending = [task for task in tasks if store.load(task) is None]
    compute_kwargs = _compute_kwargs(method, rank, resolved)
    memory_efficient = resolved["use_memory_efficient_khatri_rao"]
    task_args = (
        (task, subset, method, compute_kwargs, memory_efficient)
        for task, subset in _with_subsets(pending, data, method, transform)
    )

    if n_procs < 2:
        for args in tqdm(
            task_args,
            total=len(pending),
            disable=not progress_bar,
            desc="Fitting restarts (sequential)",
        ):
            store.save(_run_task(args))
            _release_cuda_cache(fit_device)
        return

    threads_per_proc = max(1, torch.get_num_threads() // n_procs)
    with get_context("spawn").Pool(
        n_procs,
        initializer=_init_restart_worker_backend,
        initargs=(threads_per_proc,),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_run_task, task_args),
            total=len(pending),
            disable=not progress_bar,
            desc=f"Fitting restarts (parallel, {n_procs} procs)",
        ):
            store.save(result)


@dataclass(frozen=True)
class GroupSummary:
    """The gathered restarts of one group.

    `best` is the lowest-error successful restart, or None if none
    succeeded. `missing` lists seeds with no stored result; they are not
    counted in `tally`, whose `attempted` is the number actually found.
    """

    best: RestartResult | None
    tally: RestartTally
    missing: list[int]
    indices: tuple[int, ...]


def collect(
    plan: Sequence[FitTask],
    store: ResultStore,
    warn: bool = True,
    method: Method | None = None,
    **fit_kwargs: Any,
) -> dict[Hashable, GroupSummary]:
    """Gather `plan`'s results from `store`, picking each group's best.

    Restarts are compared in seed order, so the winner is the same however
    the tasks were scheduled, and the same as `run_*_decomposition_repeated`
    would pick.

    With `warn`, a group where no stored restart converged gets a
    `UserWarning`. Passing `method="PARAFAC2"` and the fit's options as
    `fit_kwargs` also emits the warnings `run_PARAFAC2_decomposition_repeated`
    would (non-negativity residue, acceptance at the iteration limit, and
    the restart advisory).
    """
    resolved: dict[str, Any] | None = None
    if method is not None:
        resolved = _resolve_options(method, fit_kwargs)
    elif fit_kwargs:
        raise TypeError("collect() needs `method` to interpret fit_kwargs.")

    groups: dict[Hashable, list[FitTask]] = {}
    for task in plan:
        groups.setdefault(task.group, []).append(task)

    summaries: dict[Hashable, GroupSummary] = {}
    for group, tasks in groups.items():
        indices = tasks[0].indices
        if any(task.indices != indices for task in tasks):
            raise ValueError(
                f"Group {group!r} has tasks with different subject indices.",
            )

        present: list[RestartResult] = []
        missing: list[int] = []
        for task in sorted(tasks, key=lambda t: t.seed):
            result = store.load(task)
            if result is None:
                missing.append(task.seed)
            else:
                present.append(result)

        reducer = _BestOfRestarts(attempted=len(present))
        for result in present:
            if result.ok:
                assert result.error is not None
                reducer.offer(result.error, result, result.diagnostics)
            else:
                reducer.fail(result.failure())

        best: RestartResult | None = reducer.best
        summaries[group] = GroupSummary(best, reducer.tally, missing, indices)

        if not warn or not present:
            continue
        if best is None:
            warnings.warn(
                f"Group {group!r}: none of {len(present)} restarts converged"
                f"{reducer.tally.failure_summary()}",
                stacklevel=2,
            )
        if method == "PARAFAC2" and resolved is not None:
            model = best.model if best is not None else None
            assert model is None or isinstance(model, PARAFAC2Model)
            _warn_parafac2_outcome(
                model,
                best.diagnostics if best is not None else None,
                reducer.tally,
                resolved["solver"],
                resolved["max_iter"],
                resolved["tolerance"],
                resolved["aoadmm_loss_tolerance"],
            )
    return summaries


def _check_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
