from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gMRItensor.model_io import load_decomposition
from gMRItensor.model_io import save_decomposition
from gMRItensor.model_io import SavedDecomposition


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

    _assert_equal(loaded, saved)
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

    _assert_equal(loaded, saved)
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
    )

    save_decomposition(tmp_path / "rank_2.h5", saved)
    loaded = load_decomposition(tmp_path / "rank_2.h5")

    _assert_equal(loaded, saved)
    plain = load_decomposition(_save_plain(tmp_path, rng))
    assert plain.voxel_coords is None and plain.template_shape is None


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
