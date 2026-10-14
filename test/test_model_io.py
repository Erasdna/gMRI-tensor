from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gMRItensor.model_io import component_weights
from gMRItensor.model_io import load_decomposition
from gMRItensor.model_io import save_decomposition
from gMRItensor.model_io import SavedDecomposition
from gMRItensor.model_io import sort_components


def _common(rank: int, rng: np.random.Generator) -> dict[str, Any]:
    return {
        "rank": rank,
        "error": 0.125,
        "weights": rng.random(rank),
        "subject_mode": rng.random((3, rank)),
        "label_mode": rng.random((5, rank)),
        "subjects": np.array(["sub-01", "sub-02", "sub-03"]),
        "labels": np.array([2, 4, 10, 41, 49]),
        "label_index": np.zeros(5, dtype=int),
        "scale_mean": rng.random(5),
        "scale_std": rng.random(5),
    }


def _assert_equal(got: SavedDecomposition, expected: SavedDecomposition) -> None:
    for item in fields(SavedDecomposition):
        got_value = getattr(got, item.name)
        expected_value = getattr(expected, item.name)
        if isinstance(expected_value, list):
            assert len(got_value) == len(expected_value), item.name
            for got_part, expected_part in zip(got_value, expected_value):
                np.testing.assert_array_equal(got_part, expected_part)
        elif isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(got_value, expected_value)
        else:
            assert got_value == expected_value, item.name


def test_round_trip_cp(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    saved = SavedDecomposition(
        method="cp",
        timepoints=np.array([0, 6, 24]),
        time_mode=rng.random((3, 2)),
        **_common(2, rng),
    )

    save_decomposition(tmp_path / "rank_2.h5", saved)
    loaded = load_decomposition(tmp_path / "rank_2.h5")

    _assert_equal(loaded, sort_components(saved))
    assert loaded.evolving_states is None
    assert all(isinstance(subject, str) for subject in loaded.subjects)


def test_round_trip_parafac2_ragged(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    timepoints = [np.array([0, 24]), np.array([0, 6, 24]), np.array([6, 24])]
    saved = SavedDecomposition(
        method="parafac2",
        timepoints=timepoints,
        evolving_states=[rng.random((len(t), 3)) for t in timepoints],
        **_common(3, rng),
    )

    save_decomposition(tmp_path / "rank_3.h5", saved)
    loaded = load_decomposition(tmp_path / "rank_3.h5")

    _assert_equal(loaded, sort_components(saved))
    assert loaded.time_mode is None


def test_round_trip_cmf_ragged(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    timepoints = [np.array([0, 24]), np.array([0, 6, 24]), np.array([6, 24])]
    saved = SavedDecomposition(
        method="cmf",
        timepoints=timepoints,
        evolving_states=[rng.random((len(t), 2)) for t in timepoints],
        **_common(2, rng),
    )

    save_decomposition(tmp_path / "rank_2.h5", saved)
    loaded = load_decomposition(tmp_path / "rank_2.h5")

    _assert_equal(loaded, sort_components(saved))
    assert loaded.time_mode is None


def test_round_trip_without_scaling(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    saved = replace(
        SavedDecomposition(
            method="cp",
            timepoints=np.array([0, 6, 24]),
            time_mode=rng.random((3, 2)),
            **_common(2, rng),
        ),
        scale_mean=None,
        scale_std=None,
    )

    save_decomposition(tmp_path / "rank_2.h5", saved)
    loaded = load_decomposition(tmp_path / "rank_2.h5")

    assert loaded.scale_mean is None and loaded.scale_std is None


def test_save_is_atomic(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    saved = SavedDecomposition(
        method="cp",
        timepoints=np.array([0]),
        time_mode=rng.random((1, 2)),
        **_common(2, rng),
    )
    broken = replace(saved, weights=object())  # not storable

    with pytest.raises(TypeError):
        save_decomposition(tmp_path / "rank_2.h5", broken)  # type: ignore[arg-type]

    assert list(tmp_path.iterdir()) == []


def test_round_trip_centered_flag(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    saved = replace(
        SavedDecomposition(
            method="cp",
            timepoints=np.array([0, 6]),
            time_mode=rng.random((2, 2)),
            centered=True,
            **_common(2, rng),
        ),
        scale_std=None,
    )

    save_decomposition(tmp_path / "rank_2.h5", saved)
    loaded = load_decomposition(tmp_path / "rank_2.h5")

    assert loaded.centered is True
    assert loaded.scale_std is None
    np.testing.assert_array_equal(loaded.scale_mean, saved.scale_mean)


def test_round_trip_voxel_template(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    saved = replace(
        SavedDecomposition(
            method="cp",
            timepoints=np.array([0, 6]),
            time_mode=rng.random((2, 2)),
            **_common(2, rng),
        ),
        voxel_coords=rng.integers(0, 6, size=(5, 3)),
        template_shape=np.array([6, 6, 6]),
        template_affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        template_path="/data/template/seg.nii.gz",
    )

    save_decomposition(tmp_path / "rank_2.h5", saved)
    loaded = load_decomposition(tmp_path / "rank_2.h5")

    _assert_equal(loaded, sort_components(saved))
    plain = load_decomposition(_save_plain(tmp_path, rng))
    assert plain.voxel_coords is None and plain.template_shape is None
    assert plain.template_path is None


def _save_plain(tmp_path: Path, rng: np.random.Generator) -> Path:
    path = tmp_path / "plain.h5"
    save_decomposition(
        path,
        SavedDecomposition(
            method="cp",
            timepoints=np.array([0]),
            time_mode=rng.random((1, 2)),
            **_common(2, rng),
        ),
    )
    return path


TIMEPOINTS = [np.array([0, 24]), np.array([0, 6, 24]), np.array([6, 24])]


def _unsorted(method: str, rng: np.random.Generator) -> SavedDecomposition:
    """Rank 3, components scaled so their sizes are 1x, 100x and 10x."""
    scale = np.array([1.0, 100.0, 10.0])
    common = _common(3, rng)
    common["label_mode"] = common["label_mode"] * scale
    common["weights"] = rng.uniform(0.5, 1.0, 3)  # keep the scale's order
    if method == "cp":
        return SavedDecomposition(
            method="cp",
            timepoints=np.array([0, 6, 24]),
            time_mode=rng.random((3, 3)),
            **common,
        )
    return SavedDecomposition(
        method=method,  # type: ignore[arg-type]
        timepoints=TIMEPOINTS,
        evolving_states=[rng.random((len(t), 3)) for t in TIMEPOINTS],
        **common,
    )


def _reconstruction(saved: SavedDecomposition) -> list[np.ndarray]:
    """Each subject's (time, label) slice of the model."""
    w, A, C = saved.weights, saved.subject_mode, saved.label_mode
    if saved.method == "cp":
        assert saved.time_mode is not None
        return [(saved.time_mode * w * a) @ C.T for a in A]
    assert saved.evolving_states is not None
    if saved.method == "cmf":  # A is derived from the B_i, not a model factor
        return [(B * w) @ C.T for B in saved.evolving_states]
    return [(B * w * a) @ C.T for B, a in zip(saved.evolving_states, A)]


@pytest.mark.parametrize("method", ["cp", "parafac2", "cmf"])
def test_sort_components_largest_first(method: str) -> None:
    saved = _unsorted(method, np.random.default_rng(6))

    ordered = sort_components(saved)

    weights = component_weights(ordered)
    assert np.all(np.diff(weights) <= 0), weights  # component 1 is the largest
    order = [1, 2, 0]  # the 100x, 10x and 1x components
    np.testing.assert_allclose(weights, component_weights(saved)[order])
    np.testing.assert_array_equal(ordered.weights, saved.weights[order])
    np.testing.assert_array_equal(ordered.subject_mode, saved.subject_mode[:, order])
    np.testing.assert_array_equal(ordered.label_mode, saved.label_mode[:, order])
    if method == "cp":
        assert saved.time_mode is not None and ordered.time_mode is not None
        np.testing.assert_array_equal(ordered.time_mode, saved.time_mode[:, order])
    else:
        assert saved.evolving_states is not None
        assert ordered.evolving_states is not None
        for got, expected in zip(ordered.evolving_states, saved.evolving_states):
            np.testing.assert_array_equal(got, expected[:, order])
    for got, expected in zip(_reconstruction(ordered), _reconstruction(saved)):
        np.testing.assert_allclose(got, expected)
    _assert_equal(sort_components(ordered), ordered)


def test_component_weights_are_the_mode_norm_products() -> None:
    rng = np.random.default_rng(7)
    saved = _unsorted("cp", rng)
    assert saved.time_mode is not None
    norms = (
        np.linalg.norm(saved.subject_mode, axis=0)
        * np.linalg.norm(saved.time_mode, axis=0)
        * np.linalg.norm(saved.label_mode, axis=0)
    )
    np.testing.assert_allclose(component_weights(saved), saved.weights * norms)

    unit = replace(
        saved,
        subject_mode=saved.subject_mode / np.linalg.norm(saved.subject_mode, axis=0),
        time_mode=saved.time_mode / np.linalg.norm(saved.time_mode, axis=0),
        label_mode=saved.label_mode / np.linalg.norm(saved.label_mode, axis=0),
    )
    np.testing.assert_allclose(component_weights(unit), saved.weights)


def test_cmf_with_unit_weights_sorts_by_its_modes() -> None:
    """CMF stores weights of one and derives the subject mode from the B_i,
    so its size is ||C_r|| ||B_r|| over all subjects."""
    saved = replace(_unsorted("cmf", np.random.default_rng(8)), weights=np.ones(3))
    assert saved.evolving_states is not None
    evolving = np.sqrt(sum((B**2).sum(axis=0) for B in saved.evolving_states))
    expected = np.linalg.norm(saved.label_mode, axis=0) * evolving
    np.testing.assert_allclose(component_weights(saved), expected)
    # A large derived subject mode does not change the order.
    big_subject = replace(saved, subject_mode=saved.subject_mode * [1e6, 1, 1])
    np.testing.assert_allclose(component_weights(big_subject), expected)

    ordered = sort_components(saved)
    assert np.all(np.diff(component_weights(ordered)) <= 0)


def test_load_sorts_files_saved_unsorted(tmp_path: Path) -> None:
    """Models written before sorting existed still load largest first."""
    saved = _unsorted("parafac2", np.random.default_rng(9))
    save_decomposition(tmp_path / "old.h5", saved)

    loaded = load_decomposition(tmp_path / "old.h5")

    _assert_equal(loaded, sort_components(saved))
    assert np.all(np.diff(component_weights(loaded)) <= 0)
