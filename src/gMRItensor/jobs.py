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
import resource
import socket
import sys
import time
import uuid
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
from typing import NamedTuple
from typing import Protocol
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

import pandas as pd
import torch
from numpy.typing import ArrayLike
from tqdm import tqdm

from .decomposition import CMFModel
from .decomposition import ConvergenceError
from .decomposition import EVOLVING_MODELS
from .decomposition import Method
from .decomposition import PARAFAC2Diagnostics
from .decomposition import PARAFAC2Model
from .decomposition import RestartTally
from .decomposition._backend import _entries
from .decomposition._backend import _fit_backend_name
from .decomposition._backend import _fits_on_numpy
from .decomposition._backend import _float64_on_cpu
from .decomposition._backend import _init_restart_worker_backend
from .decomposition._backend import _maybe_register_memory_efficient_khatri_rao
from .decomposition._backend import _release_cuda_cache
from .decomposition.parafac2 import _check_parafac2_rank
from .decomposition.restarts import _BestOfRestarts
from .decomposition.restarts import _compute_kwargs
from .decomposition.restarts import _fit_one
from .decomposition.restarts import _resolve_options
from .decomposition.restarts import _warn_parafac2_outcome
from .decomposition.restarts import FitOutcome

if TYPE_CHECKING:
    from .replicability import ReplicabilityEngine

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


#: A plan's item: a `FitTask`, or anything a caller schedules alongside it.
_Item = TypeVar("_Item")


def n_jobs(plan: Sequence[Any], tasks_per_job: int) -> int:
    """Number of jobs needed to run `plan` in blocks of `tasks_per_job`."""
    _check_positive("tasks_per_job", tasks_per_job)
    return math.ceil(len(plan) / tasks_per_job)


def job_slice(
    plan: Sequence[_Item],
    job_index: int,
    tasks_per_job: int,
) -> list[_Item]:
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
class RestartTiming:
    """What one restart cost and where it ran, recorded by `_run_task`.

    `n_iter` is None when the solver did not report it. `peak_rss_gb` is the
    peak resident memory of the process that fitted it, up to the end of
    this fit: in a pool, a worker's earlier restarts count too. `backend` is
    "numpy", "torch" or the GPU device type (see `_fit_backend_name`).
    """

    fit_seconds: float
    n_iter: int | None
    backend: str
    threads: int
    peak_rss_gb: float
    host: str

    @property
    def seconds_per_iter(self) -> float | None:
        return self.fit_seconds / self.n_iter if self.n_iter else None


def _peak_rss_gb() -> float:
    """This process's peak resident memory so far, in GB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


@dataclass(frozen=True)
class RestartResult:
    """What one `FitTask` produced: a model, or why it was rejected.

    Failures are stored as plain values rather than an exception, so any
    `ResultStore` can serialise them; `failure()` rebuilds the
    `ConvergenceError` for tallying. `timing` is None for results stored
    before timings were recorded.
    """

    task: FitTask
    model: PARAFAC2Model | CMFModel | CPModel | None = None
    error: float | None = None
    diagnostics: PARAFAC2Diagnostics | None = None
    failure_reason: str | None = None
    failure_message: str | None = None
    failure_level: float | None = None
    failure_suggested_max_iter: int | None = None
    timing: RestartTiming | None = None

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
    outcome: FitOutcome,
    timing: RestartTiming | None = None,
) -> RestartResult:
    model, error, diagnostics, failure, _ = outcome
    if model is not None:
        return RestartResult(task, model, error, diagnostics, timing=timing)
    assert failure is not None
    return RestartResult(
        task,
        failure_reason=failure.reason,
        failure_message=str(failure),
        failure_level=failure.level_reached,
        failure_suggested_max_iter=failure.suggested_max_iter,
        timing=timing,
    )


def _subset(data: Data, indices: tuple[int, ...], method: Method) -> Data:
    """Rows `indices` of `data`: a tensor for CP, a slice list otherwise."""
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
    """Fit one task on its already-built subset, timing it. Picklable pool
    worker."""
    task, subset, method, compute_kwargs, memory_efficient = args
    _maybe_register_memory_efficient_khatri_rao(memory_efficient)
    start = time.perf_counter()
    outcome = _fit_one((method, task.seed, subset, compute_kwargs))
    compute_device = compute_kwargs.get("compute_device")
    device = compute_device if compute_device is not None else _data_device(subset)
    timing = RestartTiming(
        fit_seconds=time.perf_counter() - start,
        n_iter=outcome.n_iter,
        backend=_fit_backend_name(device, _entries(subset)),
        threads=torch.get_num_threads(),
        peak_rss_gb=_peak_rss_gb(),
        host=socket.gethostname(),
    )
    return _result_from_outcome(task, outcome, timing)


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


#: What a store holds for a task: nothing, a usable result, or a result it
#: cannot use (fitted with other settings, for another task, or unreadable).
StoreStatus = Literal["missing", "current", "stale"]


class ResultStore(Protocol):
    """Where `run_tasks` puts results and `collect` reads them."""

    def save(self, result: RestartResult) -> None:
        """Store `result`, replacing any result for its task."""
        ...

    def load(self, task: FitTask) -> RestartResult | None:
        """The stored result for `task`, or None if it is missing."""
        ...

    def status(self, task: FitTask) -> StoreStatus:
        """Whether `task`'s result is missing, usable or stale."""
        ...


class InMemoryStore:
    """A `ResultStore` in a dict, for centralised runs."""

    def __init__(self) -> None:
        self.results: dict[FitTask, RestartResult] = {}

    def save(self, result: RestartResult) -> None:
        self.results[result.task] = result

    def load(self, task: FitTask) -> RestartResult | None:
        return self.results.get(task)

    def status(self, task: FitTask) -> StoreStatus:
        return "current" if task in self.results else "missing"


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


def _model_record(
    model: PARAFAC2Model | CMFModel | CPModel | None,
) -> dict[str, Any] | None:
    if model is None:
        return None
    if isinstance(model, CMFModel):
        return {"kind": "CMF", **dataclasses.asdict(model)}
    if isinstance(model, PARAFAC2Model):
        return {"kind": "PARAFAC2", **dataclasses.asdict(model)}
    weights, factors = model
    return {"kind": "CP", "weights": weights, "factors": list(factors)}


def _model_from_record(
    record: dict[str, Any] | None,
) -> PARAFAC2Model | CMFModel | CPModel | None:
    if record is None:
        return None
    if record["kind"] == "CP":
        return record["weights"], list(record["factors"])
    fields = {k: v for k, v in record.items() if k != "kind"}
    if record["kind"] == "CMF":
        fields["evolving_states"] = list(fields["evolving_states"])
        return CMFModel(**fields)
    if fields["kiers"] is not None:
        coordinate_matrix, projections = fields["kiers"]
        fields["kiers"] = (coordinate_matrix, list(projections))
    return PARAFAC2Model(**fields)


class DirectoryStore:
    """A `ResultStore` with one `torch.save` file per task.

    Files live at `root/<group>/seed_<NNNN>.pt` and hold only tensors and
    plain values, so they load with `weights_only=True`. Writes go to a
    temporary file named after this host and process, are synced to disk
    and then renamed into place, so neither a job killed mid-write nor two
    jobs writing the same task leave a half-written result for `collect`.

    `settings` (any string, e.g. a hash of the fit options) is saved with
    every result; a store with `settings` refuses results saved with other
    ones instead of reusing fits made under different options.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        settings: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.settings = settings

    def path(self, task: FitTask) -> Path:
        return self.root / _group_slug(task.group) / f"seed_{task.seed:04d}.pt"

    def save(self, result: RestartResult) -> None:
        path = self.path(result.task)
        path.parent.mkdir(parents=True, exist_ok=True)
        task = result.task
        record = {
            "settings": self.settings,
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
            "timing": (
                None if result.timing is None else dataclasses.asdict(result.timing)
            ),
        }
        writer = f"{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}"
        tmp = path.with_name(f".{path.name}.{writer}.tmp")
        with open(tmp, "wb") as file:
            torch.save(record, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)

    def _read(self, task: FitTask) -> tuple[dict[str, Any] | None, str | None]:
        """`task`'s stored record (None if missing or unreadable) and, if
        it cannot be used, why."""
        path = self.path(task)
        if not path.exists():
            return None, None
        try:
            record = torch.load(path, weights_only=True)
        except Exception as error:  # any unpickling failure: a damaged file
            # Only the type: torch's message spans lines and suggests
            # weights_only=False, which would run code from the file.
            return None, f"{path} cannot be read ({type(error).__name__})"
        if self.settings is not None and record.get("settings") != self.settings:
            return record, (
                f"{path} was fitted with other settings (input, tensor or fit "
                "options) than now"
            )
        stored = record["task"]
        stored_task = FitTask(stored["group"], tuple(stored["indices"]), stored["seed"])
        if stored_task != task:
            return record, (
                f"{path} holds {stored_task}, not the requested {task}: it was "
                "written by a different plan (other splits, subject set or "
                "n_restarts)"
            )
        return record, None

    def status(self, task: FitTask) -> StoreStatus:
        """`"missing"`, `"current"`, or `"stale"` for a result that cannot
        be used: unreadable, for another task, or saved with other
        `settings`. Never raises; `run_tasks` refits stale results."""
        if not self.path(task).exists():
            return "missing"
        return "stale" if self._read(task)[1] is not None else "current"

    def load(self, task: FitTask) -> RestartResult | None:
        """The stored result for `task`, or None if it is missing.

        Raises `ValueError` for a stale result (see `status`): it has to be
        fitted again, or deleted, before it can be collected.
        """
        record, problem = self._read(task)
        if problem is not None:
            raise ValueError(
                f"{problem}; fit it again (`run --job`), or delete it",
            )
        if record is None:
            return None
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
            # Absent from results stored before timings were recorded.
            timing=(
                None
                if record.get("timing") is None
                else RestartTiming(**record["timing"])
            ),
        )


def _with_subsets(
    tasks: Sequence[FitTask],
    data: Data,
    method: Method,
    transform: Transform | None,
    fit_device: torch.device,
) -> Iterator[tuple[FitTask, Data]]:
    """Pair each task with its subset, built once per run of equal groups.

    On the CPU the subset is cast to float64 here, once, so the restarts of
    a pool share it (see `_float64_on_cpu`).
    """
    current: tuple[Hashable, tuple[int, ...]] | None = None
    subset: Data | None = None
    for task in tasks:
        key = (task.group, task.indices)
        if key != current:
            subset = _subset(data, task.indices, method)
            if transform is not None:
                subset = transform(subset)
            subset = _float64_on_cpu(subset, fit_device)
            current = key
        assert subset is not None
        yield task, subset


class TaskCounts(NamedTuple):
    """What `run_tasks` did: tasks fitted for the first time, reused from
    the store, and fitted again over a stale result."""

    fitted: int
    reused: int
    refitted: int


def pending_tasks(
    tasks: Sequence[FitTask],
    store: ResultStore,
    resume: bool = True,
) -> tuple[list[FitTask], TaskCounts]:
    """The tasks to fit and what that means for the store. With `resume`,
    tasks with a current result are skipped and stale ones refitted;
    without it every task is fitted, as new."""
    if not resume:
        return list(tasks), TaskCounts(len(tasks), 0, 0)
    statuses = [store.status(task) for task in tasks]
    pending = [task for task, status in zip(tasks, statuses) if status != "current"]
    return pending, TaskCounts(
        fitted=statuses.count("missing"),
        reused=statuses.count("current"),
        refitted=statuses.count("stale"),
    )


def _warn_if_threads_are_wasted(
    data: Data,
    fit_device: torch.device,
    n_procs: int,
) -> None:
    """Warn when numpy fits get more than one thread each: only its BLAS
    calls use them, which gained at most 1.3x in benchmarks, while the
    same cores fit other restarts in parallel."""
    total = torch.get_num_threads()
    threads = total if n_procs < 2 else max(1, total // n_procs)
    if threads < 2 or fit_device.type != "cpu":
        return
    if not _fits_on_numpy(fit_device, _entries(data)):
        return
    warnings.warn(
        f"Each restart gets {threads} threads, but fits on the numpy backend "
        "use about one core: its threads only speed up the BLAS calls. Spread "
        "the cores over restarts instead (--restart-procs / --n-procs), or "
        "fit large data on torch (GMRITENSOR_FIT_BACKEND=torch or auto), "
        "whose fits do use several threads. See docs/cpu-scaling.md.",
        stacklevel=3,
    )


#: The columns of `restart_table` (and of `restarts.csv`).
RESTART_TABLE_COLUMNS = (
    "seed",
    "status",
    "error",
    "n_iter",
    "fit_seconds",
    "seconds_per_iter",
    "backend",
    "threads",
    "peak_rss_gb",
    "host",
)


def restart_table(results: Sequence[RestartResult]) -> pd.DataFrame:
    """One row per restart: its outcome, and what it cost (`RestartTiming`;
    empty for results stored before timings were recorded)."""
    rows = []
    for result in results:
        timing = result.timing
        rows.append(
            {
                "seed": result.task.seed,
                "status": "ok" if result.ok else (result.failure_reason or "unknown"),
                "error": result.error,
                "n_iter": None if timing is None else timing.n_iter,
                "fit_seconds": None if timing is None else timing.fit_seconds,
                "seconds_per_iter": (
                    None if timing is None else timing.seconds_per_iter
                ),
                "backend": None if timing is None else timing.backend,
                "threads": None if timing is None else timing.threads,
                "peak_rss_gb": None if timing is None else timing.peak_rss_gb,
                "host": None if timing is None else timing.host,
            },
        )
    return pd.DataFrame(rows, columns=RESTART_TABLE_COLUMNS)


def run_tasks(
    tasks: Sequence[FitTask],
    data: Data,
    rank: int,
    method: Method,
    store: ResultStore,
    n_procs: int = 1,
    transform: Transform | None = None,
    progress_bar: bool = True,
    resume: bool = True,
    **kwargs: Any,
) -> TaskCounts:
    """Fit `tasks` and save every result, failures included, to `store`.

    With `resume` (the default), tasks with a current result in `store`
    are skipped, so a re-queued job resumes rather than refitting; stale
    results (see `DirectoryStore.status`) are refitted and overwritten.
    Without it every task is refitted and overwrites what the store held.
    Returns what was fitted, reused and refitted (`TaskCounts`). Each group's subset is
    built (and `transform`ed) once, in this process, as the tasks reach it.

    `kwargs` are `run_*_decomposition_repeated` options -- see `fit_task`.
    `n_procs >= 2` spreads the tasks over a spawned process pool, each worker
    pinned to its share of `torch.get_num_threads()`.

    Raises
    ------
    ValueError
        If `n_procs >= 2` and the fit runs on CUDA. Worker processes against
        a CUDA context are unreliable across driver setups, so this is
        rejected rather than silently run sequentially. Also for a PARAFAC2
        `rank` the data cannot support (see `_check_parafac2_rank`), before
        any task is fitted.
    """
    if method == "PARAFAC2":
        # Subsets only drop subjects, so the full data sets the limits.
        _check_parafac2_rank(data, rank)
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

    _warn_if_threads_are_wasted(data, fit_device, n_procs)

    pending, counts = pending_tasks(tasks, store, resume)
    compute_kwargs = _compute_kwargs(method, rank, resolved)
    memory_efficient = resolved["use_memory_efficient_khatri_rao"]
    task_args = (
        (task, subset, method, compute_kwargs, memory_efficient)
        for task, subset in _with_subsets(
            pending,
            data,
            method,
            transform,
            fit_device,
        )
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
        return counts

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
    return counts


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
    `UserWarning`. Passing `method="PARAFAC2"` (or `"CMF"`) and the fit's
    options as `fit_kwargs` also emits the warnings that method's
    `run_*_decomposition_repeated` would (non-negativity residue,
    acceptance at the iteration limit, and the restart advisory).
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
        if method in ("PARAFAC2", "CMF") and resolved is not None:
            model = best.model if best is not None else None
            assert model is None or isinstance(model, EVOLVING_MODELS)
            _warn_parafac2_outcome(
                model,
                best.diagnostics if best is not None else None,
                reducer.tally,
                resolved.get("solver", "matcouply"),
                resolved["max_iter"],
                resolved["tolerance"],
                resolved["aoadmm_loss_tolerance"],
                method_label=method,
            )
    return summaries


def _check_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
