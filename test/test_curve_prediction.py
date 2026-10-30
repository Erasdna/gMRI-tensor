"""Predicting clinical variables from time profiles (`gMRItensor.curve_prediction`)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from gMRItensor import pipeline
from gMRItensor.curve_prediction import _evaluate
from gMRItensor.curve_prediction import covariate_sets
from gMRItensor.curve_prediction import cv_folds
from gMRItensor.curve_prediction import evaluate_curves
from gMRItensor.curve_prediction import oof_operator
from gMRItensor.curve_prediction import PREDICTION_COLUMNS
from gMRItensor.curve_prediction import scores
from gMRItensor.curve_prediction import split_subjects
from gMRItensor.model_io import SavedDecomposition
from gMRItensor.subject_statistics import add_fdr
from gMRItensor.subject_statistics import permutations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

DOCS = Path(__file__).parents[1] / "docs" / "curve-prediction-columns.md"
PRIMARY = ["diagnosis", "sex", "age"]
CATEGORICAL = {"diagnosis", "sex"}
TIMEPOINTS = [0, 6, 24, 48]


def _study(
    n: int = 30,
    seed: int = 0,
    outcome: str = "curve",
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Two components' curves for `n` subjects, component 1's amplitude
    higher in PD. `y` depends on component 2's curve at 24 h (`outcome =
    "curve"`), on diagnosis only (`"diagnosis"`) or on nothing (`"null"`)."""
    rng = np.random.default_rng(seed)
    subjects = [f"sub-{i:02d}" for i in range(n)]
    info = pd.DataFrame(
        {
            "diagnosis": np.resize(["Control", "PD"], n),
            "sex": rng.choice(["F", "M"], n),
            "age": rng.normal(65, 8, n),
        },
        index=subjects,
    )
    affected = (info["diagnosis"] == "PD").to_numpy(float)
    curves = {}
    for component in (1, 2):
        shape = np.exp(-((np.arange(len(TIMEPOINTS)) - component) ** 2) / 2)
        amplitude = rng.uniform(0.5, 1.5, n) + (affected if component == 1 else 0)
        values = amplitude[:, None] * shape + rng.normal(0, 0.05, (n, len(shape)))
        curves[component] = pd.DataFrame(values, index=subjects, columns=TIMEPOINTS)
    noise = rng.normal(0, 0.3, n)
    if outcome == "curve":
        info["y"] = 5 * curves[2][24] + 0.02 * info["age"] + noise
    elif outcome == "diagnosis":
        info["y"] = 2 * affected + noise
    else:
        info["y"] = noise
    return curves, info


def _evaluate_study(
    curves: dict[int, pd.DataFrame],
    info: pd.DataFrame,
    primary: list[str] = PRIMARY,
    **kwargs: object,
) -> pd.DataFrame:
    options: dict = {"n_permutations": 500, **kwargs}
    table = evaluate_curves(
        curves,
        ["y"],
        info,
        primary,
        CATEGORICAL,
        "diagnosis" if "diagnosis" in primary else None,
        **options,
    )
    return add_fdr(
        table,
        keys=("outcome", "component", "covariates"),
        p_column="p_gain",
    )


def test_oof_operator_matches_fold_by_fold_refits() -> None:
    rng = np.random.default_rng(0)
    design = np.column_stack([np.ones(20), rng.normal(size=(20, 3))])
    y = rng.normal(size=20)
    folds = list(KFold(5, shuffle=True, random_state=0).split(design))

    predicted = oof_operator(design, folds) @ y

    expected = np.empty(20)
    for fit, held_out in folds:
        model = LinearRegression(fit_intercept=False).fit(design[fit], y[fit])
        expected[held_out] = model.predict(design[held_out])
    np.testing.assert_allclose(predicted, expected)


def test_scores_match_sklearn() -> None:
    rng = np.random.default_rng(1)
    y, predicted = rng.normal(size=15), rng.normal(size=15)
    r2, rmse, mae = scores(y, predicted)

    assert r2 == pytest.approx(r2_score(y, predicted))
    assert rmse == pytest.approx(np.sqrt(mean_squared_error(y, predicted)))
    assert mae == pytest.approx(mean_absolute_error(y, predicted))
    columns = scores(np.column_stack([y, y]), np.column_stack([predicted, y]))[0]
    np.testing.assert_allclose(columns, [r2_score(y, predicted), 1.0])


def test_training_fit_matches_statsmodels() -> None:
    rng = np.random.default_rng(2)
    design = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
    y = design @ [1.0, 0.5, -0.3] + rng.normal(0, 0.5, 30)
    train, test = np.arange(24), np.arange(24, 30)
    folds, _ = cv_folds(24, None, 5, 3, 0)

    row, _ = _evaluate(y, design, train, test, folds, permutations(24, 100, 0))

    fit = sm.OLS(y[train], design[train]).fit()
    assert row["aic"] == pytest.approx(fit.aic)
    assert row["bic"] == pytest.approx(fit.bic)
    assert row["train_r2_adj"] == pytest.approx(fit.rsquared_adj)
    assert row["test_r2"] == pytest.approx(r2_score(y[test], fit.predict(design[test])))
    assert row["n_params"] == 3


def test_split_and_folds_are_stratified_disjoint_and_reproducible() -> None:
    strata = np.resize(["Control", "PD"], 30)
    train, test, note = split_subjects(30, strata, 0.2, seed=3)

    assert note == "" and len(test) == 6
    assert set(train).isdisjoint(test) and set(train) | set(test) == set(range(30))
    assert pd.Series(strata[test]).value_counts().tolist() == [3, 3]
    again = split_subjects(30, strata, 0.2, seed=3)
    np.testing.assert_array_equal(again[1], test)

    folds, note = cv_folds(len(train), strata[train], 5, 10, seed=3)
    assert note == "" and len(folds) == 10
    for repeat in folds:
        held_out = np.concatenate([part for _, part in repeat])
        assert sorted(held_out) == list(range(len(train)))


def test_unusable_strata_fall_back_with_a_note() -> None:
    strata = np.array(["A"] * 29 + ["B"])
    _, _, split_note = split_subjects(30, strata, 0.2, seed=0)
    _, fold_note = cv_folds(24, strata[:24], 5, 2, seed=0)

    assert "not stratified" in split_note and "not stratified" in fold_note


def test_covariate_sets() -> None:
    assert covariate_sets(["diagnosis"]) == [(), ("diagnosis",)]
    assert covariate_sets(PRIMARY) == [
        (),
        ("diagnosis",),
        ("sex",),
        ("age",),
        ("diagnosis", "sex", "age"),
    ]


def test_the_table_has_a_baseline_and_a_curve_row_per_set() -> None:
    curves, info = _study()
    table = _evaluate_study(curves, info)

    assert tuple(table.columns) == PREDICTION_COLUMNS
    assert len(table) == 5 * 3  # 5 sets x (baseline + 2 components)
    baselines = table[~table["curve"]]
    assert baselines["component"].isna().all()
    assert list(baselines["model"]) == [
        "intercept only",
        "diagnosis only",
        "sex only",
        "age only",
        "diagnosis + sex + age only",
    ]
    assert np.isnan(baselines["p_perm"].iloc[0])  # intercept only: no test
    assert baselines["p_gain"].isna().all()
    assert (table["n_train"] == 24).all() and (table["n_test"] == 6).all()


def test_a_planted_curve_is_found() -> None:
    curves, info = _study()
    table = _evaluate_study(curves, info)
    second = table[table["component"] == 2]
    first = table[table["component"] == 1]

    assert (second["cv_r2_gain"] > 0).all()
    assert (second["p_gain"] < 0.01).all() and second["significant"].all()
    assert (second["test_r2"] > 0.5).all()
    assert not first["significant"].any()


def test_a_confounded_curve_adds_nothing_beyond_diagnosis() -> None:
    curves, info = _study(outcome="diagnosis", seed=4)
    table = _evaluate_study(curves, info, primary=["diagnosis"]).set_index(
        ["model", "component"],
    )

    alone = table.loc[("curve", 1)]
    adjusted = table.loc[("diagnosis + curve", 1)]
    assert alone["p_perm"] < 0.05  # the curve tracks diagnosis
    assert adjusted["p_gain"] > 0.05  # but adds nothing beyond it


@pytest.mark.slow
def test_null_gain_rejection_rate_is_nominal() -> None:
    rejections = []
    for seed in range(200):
        curves, info = _study(seed=seed, outcome="null")
        table = evaluate_curves(
            {1: curves[1]},
            ["y"],
            info,
            ["diagnosis"],
            CATEGORICAL,
            "diagnosis",
            n_permutations=200,
            seed=seed,
        )
        rejections.append(bool((table["p_gain"].dropna() < 0.05).iloc[-1]))
    rate = np.mean(rejections)
    assert 0.01 < rate < 0.10  # 200 draws at 0.05: ~0.02 to ~0.08


def test_a_model_too_big_for_its_folds_is_skipped() -> None:
    curves, info = _study(n=12)
    wide = {c: pd.concat([frame] * 3, axis=1) for c, frame in curves.items()}
    table = evaluate_curves(wide, ["y"], info, PRIMARY, CATEGORICAL, None, folds=2)
    skipped = table[table["curve"] & (table["covariates"] == "diagnosis + sex + age")]

    assert skipped["cv_r2"].isna().all()
    assert skipped["note"].str.contains("smallest training fold").all()


def test_subjects_missing_a_value_or_a_time_point_are_left_out() -> None:
    curves, info = _study()
    info.loc["sub-00", "y"] = np.nan
    curves[1].loc["sub-01", 48] = np.nan

    table = _evaluate_study(curves, info)

    assert (table["n_train"] + table["n_test"] == 28).all()
    assert table["note"].str.contains("2 subject").all()


def test_a_zero_curve_is_skipped() -> None:
    curves, info = _study()
    curves[1].loc[:, :] = 0.0

    table = _evaluate_study(curves, info)
    zero = table[table["component"] == 1]

    assert zero["cv_r2"].isna().all()
    assert zero["note"].str.contains("curve is zero").all()


def test_same_seed_same_table() -> None:
    curves, info = _study()
    pd.testing.assert_frame_equal(
        _evaluate_study(curves, info, seed=5),
        _evaluate_study(curves, info, seed=5),
    )


def test_model_curves_are_the_saved_time_coefficients() -> None:
    rng = np.random.default_rng(0)
    timepoints = [np.array([0, 6, 24]), np.array([0, 24]), np.array([0, 6, 24])]
    saved = SavedDecomposition(
        method="cmf",
        rank=2,
        error=0.1,
        weights=np.ones(2),
        subject_mode=np.ones((3, 2)),
        label_mode=rng.uniform(size=(5, 2)),
        subjects=np.array(["a", "b", "c"]),
        timepoints=timepoints,
        labels=np.arange(5),
        label_index=np.zeros(5, dtype=int),
        evolving_states=[rng.uniform(size=(len(t), 2)) for t in timepoints],
    )

    curves = pipeline.model_curves(saved)

    table = pipeline._time_coefficients_table(
        saved,
        ["a", "b", "c"],
        pipeline.subject_coefficients(saved),
    )
    for component in (1, 2):
        frame = curves[component]
        assert list(frame.index) == ["a", "b", "c"]
        assert list(frame.columns) == [0, 6, 24]
        assert np.isnan(frame.loc["b", 6])  # b has no 6 h scan
        for _, row in table.iterrows():
            assert frame.loc[row["subject"], row["timepoint"]] == pytest.approx(
                row[f"component_{component}"],
            )


def test_docs_explain_every_column() -> None:
    documented = set(re.findall(r"^### `([a-z_0-9]+)`", DOCS.read_text(), re.MULTILINE))
    assert documented == set(PREDICTION_COLUMNS)
