"""Store one fitted CP, PARAFAC2 or CMF decomposition per HDF5 file."""
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Literal

import h5py
import numpy as np

_ARRAYS = ("weights", "subject_mode", "label_mode", "labels", "label_index")
_OPTIONAL_ARRAYS = (
    "scale_mean",
    "scale_std",
    "voxel_coords",
    "template_shape",
    "template_affine",
)


#: Methods saved with a ragged, per-subject evolving mode.
_EVOLVING_METHODS = ("parafac2", "cmf")

#: The model file in each model directory, `<method>_rank_<r>/`.
MODEL_FILE = "model.h5"


def resolve_model(model: Path) -> tuple[Path, str]:
    """`(model file, model name)` for a model directory, its `MODEL_FILE`,
    or any other saved `.h5` file, which is named after its stem."""
    model = Path(model)
    if model.is_dir():
        return model / MODEL_FILE, model.name
    if model.name == MODEL_FILE:
        return model, model.parent.name
    return model, model.stem


@dataclass(frozen=True)
class SavedDecomposition:
    """A fit plus what is needed to interpret it.

    CP has one shared `time_mode` over `timepoints`; PARAFAC2 and CMF have
    one `evolving_states[i]` per subject over `timepoints[i]`. `labels` and
    `label_index` identify `label_mode` rows, as returned by
    `load_tensor_from_parquet`. `scale_mean`/`scale_std` are the per-label
    statistics of the fitted data: `centered` says whether `scale_mean` was
    subtracted, and `scale_std` is None if the data was not divided by it.
    Voxel models also carry each `label_mode` row's `voxel_coords` (i, j, k)
    on the template grid (`template_shape`, `template_affine`), and
    `template_path`, the segmentation preprocessing recorded as the template
    (a record only; it is never read).

    Components are ordered by `component_weights`, largest first: saving
    through the pipeline sorts them and `load_decomposition` sorts again,
    so files written before sorting existed load in the same order.
    """

    method: Literal["cp", "parafac2", "cmf"]
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
    centered: bool = False
    voxel_coords: np.ndarray | None = None
    template_shape: np.ndarray | None = None
    template_affine: np.ndarray | None = None
    template_path: str | None = None


def _column_norms(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(matrix), axis=0)


def component_weights(saved: SavedDecomposition) -> np.ndarray:
    """Each component's size: `|weights|` times the L2 norms of its modes.

    Fits are never normalised (their weights are ones), so this is the
    product of the mode norms; `|weights|` covers models saved earlier with
    normalised factors. CP multiplies the subject, time and label norms.
    PARAFAC2 and CMF use the label norm and the evolving mode's norm over
    all subjects, each subject's `B_i` scaled by its subject weight for
    PARAFAC2; CMF's `subject_mode` is derived from the `B_i`, so it is not
    counted twice.
    """
    weights = np.abs(np.asarray(saved.weights, dtype=float))
    label = _column_norms(saved.label_mode)
    if saved.method == "cp":
        assert saved.time_mode is not None
        return (
            weights
            * _column_norms(saved.subject_mode)
            * _column_norms(saved.time_mode)
            * label
        )
    states = [np.asarray(B) for B in saved.evolving_states or []]
    if saved.method == "parafac2":
        states = [B * a for B, a in zip(states, np.asarray(saved.subject_mode))]
    evolving = np.sqrt(sum((B**2).sum(axis=0) for B in states))
    return weights * evolving * label


def subject_coefficients(saved: SavedDecomposition) -> np.ndarray:
    """`(n_subjects, rank)` subject coefficients the figures draw: the
    subject mode, except for CMF, whose `B_i` carry each subject's amplitude:
    there it is the L2 norm of each subject's `B_i` columns (the saved
    `subject_mode` is their RMS, which also depends on the number of time
    points)."""
    if saved.method == "cmf":
        return np.array([_column_norms(B) for B in saved.evolving_states or []])
    return np.asarray(saved.subject_mode)


def sort_components(saved: SavedDecomposition) -> SavedDecomposition:
    """`saved` with its components ordered by `component_weights`, largest
    first (stable for ties). Values are only permuted, never rescaled."""
    order = np.argsort(-component_weights(saved), kind="stable")
    return replace(
        saved,
        weights=np.asarray(saved.weights)[order],
        subject_mode=np.asarray(saved.subject_mode)[:, order],
        label_mode=np.asarray(saved.label_mode)[:, order],
        time_mode=(
            None if saved.time_mode is None else np.asarray(saved.time_mode)[:, order]
        ),
        evolving_states=(
            None
            if saved.evolving_states is None
            else [np.asarray(B)[:, order] for B in saved.evolving_states]
        ),
    )


def select_subjects(
    saved: SavedDecomposition,
    positions: Sequence[int],
) -> SavedDecomposition:
    """`saved` restricted to the subjects at `positions`: their
    `subject_mode` rows and ids, and for PARAFAC2/CMF their evolving states
    and time points. Shared modes are unchanged."""
    index = list(positions)
    states = saved.evolving_states
    return replace(
        saved,
        subject_mode=np.asarray(saved.subject_mode)[index],
        subjects=np.asarray(saved.subjects)[index],
        evolving_states=None if states is None else [states[i] for i in index],
        timepoints=(
            saved.timepoints if states is None else [saved.timepoints[i] for i in index]
        ),
    )


def _write_ragged(file: h5py.File, name: str, arrays: list[np.ndarray]) -> None:
    group = file.create_group(name)
    for i, array in enumerate(arrays):
        group.create_dataset(str(i), data=np.asarray(array))


def _read_ragged(file: h5py.File, name: str) -> list[np.ndarray]:
    group = file[name]
    return [group[str(i)][()] for i in range(len(group))]


def save_decomposition(path: Path, saved: SavedDecomposition) -> None:
    """Write `saved` to `path`, replacing it only once fully written."""
    if saved.method in _EVOLVING_METHODS and saved.evolving_states is None:
        raise ValueError(f"A {saved.method} decomposition needs evolving_states")
    if saved.method == "cp" and saved.time_mode is None:
        raise ValueError("A CP decomposition needs time_mode")

    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with h5py.File(tmp_path, "w") as file:
            file.attrs["method"] = saved.method
            file.attrs["rank"] = saved.rank
            file.attrs["error"] = saved.error
            file.attrs["centered"] = saved.centered
            if saved.template_path is not None:
                file.attrs["template_path"] = saved.template_path
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
            if saved.method in _EVOLVING_METHODS:
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
    """Read a `save_decomposition` file, components largest first
    (`sort_components`), however the file was written."""
    return sort_components(_read_decomposition(path))


def _read_decomposition(path: Path) -> SavedDecomposition:
    """A `save_decomposition` file exactly as stored."""
    with h5py.File(path, "r") as file:
        method = str(file.attrs["method"])
        arrays: dict[str, Any] = {name: file[name][()] for name in _ARRAYS}
        optional: dict[str, Any] = {
            name: file[name][()] if name in file else None for name in _OPTIONAL_ARRAYS
        }
        subjects = np.asarray(file["subjects"].asstr()[()])
        if method in _EVOLVING_METHODS:
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
            centered=bool(file.attrs.get("centered", False)),
            template_path=(
                str(file.attrs["template_path"])
                if "template_path" in file.attrs
                else None
            ),
            subjects=subjects,
            timepoints=timepoints,
            time_mode=time_mode,
            evolving_states=evolving_states,
            **arrays,
            **optional,
        )
