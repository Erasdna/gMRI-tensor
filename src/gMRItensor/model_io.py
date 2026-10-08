"""Store one fitted CP or PARAFAC2 decomposition per HDF5 file."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

_ARRAYS = ("weights", "subject_mode", "label_mode", "labels", "label_index")
_OPTIONAL_ARRAYS = ("scale_mean", "scale_std")


@dataclass(frozen=True)
class SavedDecomposition:
    """A fit plus what is needed to interpret it.

    CP has one shared `time_mode` over `timepoints`; PARAFAC2 has one
    `evolving_states[i]` per subject over `timepoints[i]`. `labels` and
    `label_index` identify `label_mode` rows, as returned by
    `load_tensor_from_parquet`; `scale_mean`/`scale_std` are the per-label
    scaling applied before fitting, or None if unscaled.
    """

    method: Literal["cp", "parafac2"]
    rank: int
    error: float
    weights: np.ndarray
    subject_mode: np.ndarray
    label_mode: np.ndarray
    subjects: np.ndarray
    timepoints: np.ndarray | list[np.ndarray]
    labels: np.ndarray
    label_index: np.ndarray
    time_mode: np.ndarray | None = None
    evolving_states: list[np.ndarray] | None = None
    scale_mean: np.ndarray | None = None
    scale_std: np.ndarray | None = None


def _write_ragged(file: h5py.File, name: str, arrays: list[np.ndarray]) -> None:
    group = file.create_group(name)
    for i, array in enumerate(arrays):
        group.create_dataset(str(i), data=np.asarray(array))


def _read_ragged(file: h5py.File, name: str) -> list[np.ndarray]:
    group = file[name]
    return [group[str(i)][()] for i in range(len(group))]


def save_decomposition(path: Path, saved: SavedDecomposition) -> None:
    """Write `saved` to `path`, replacing it only once fully written."""
    if saved.method == "parafac2" and saved.evolving_states is None:
        raise ValueError("A PARAFAC2 decomposition needs evolving_states")
    if saved.method == "cp" and saved.time_mode is None:
        raise ValueError("A CP decomposition needs time_mode")

    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with h5py.File(tmp_path, "w") as file:
            file.attrs["method"] = saved.method
            file.attrs["rank"] = saved.rank
            file.attrs["error"] = saved.error
            for name in _ARRAYS:
                file.create_dataset(name, data=np.asarray(getattr(saved, name)))
            for name in _OPTIONAL_ARRAYS:
                if getattr(saved, name) is not None:
                    file.create_dataset(name, data=np.asarray(getattr(saved, name)))
            file.create_dataset(
                "subjects",
                data=np.asarray(saved.subjects, dtype=object),
                dtype=h5py.string_dtype(),
            )
            if saved.method == "parafac2":
                _write_ragged(file, "timepoints", list(saved.timepoints))
                _write_ragged(
                    file,
                    "evolving_states",
                    list(saved.evolving_states or []),
                )
            else:
                file.create_dataset("timepoints", data=np.asarray(saved.timepoints))
                file.create_dataset("time_mode", data=np.asarray(saved.time_mode))
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_decomposition(path: Path) -> SavedDecomposition:
    """Read a `save_decomposition` file."""
    with h5py.File(path, "r") as file:
        method = str(file.attrs["method"])
        arrays = {name: file[name][()] for name in _ARRAYS}
        optional = {
            name: file[name][()] if name in file else None for name in _OPTIONAL_ARRAYS
        }
        subjects = np.asarray(file["subjects"].asstr()[()])
        if method == "parafac2":
            timepoints: np.ndarray | list[np.ndarray] = _read_ragged(file, "timepoints")
            evolving_states = _read_ragged(file, "evolving_states")
            time_mode = None
        else:
            timepoints = file["timepoints"][()]
            time_mode = file["time_mode"][()]
            evolving_states = None
        return SavedDecomposition(
            method=method,  # type: ignore[arg-type]
            rank=int(file.attrs["rank"]),
            error=float(file.attrs["error"]),
            subjects=subjects,
            timepoints=timepoints,
            time_mode=time_mode,
            evolving_states=evolving_states,
            **arrays,
            **optional,
        )
