"""Predicting clinical variables from the subjects' time profiles (no
plotting).

For a PARAFAC2 or CMF model every subject has its own time course per
component: its evolving curve times its subject coefficient, the values of
`<model>__time_coefficients.csv`. `evaluate_curves` measures how well one
component's curve predicts a numerical clinical outcome, as a model
evaluation: the subjects are split into a training and a test part, every
model is cross-validated on the training part (repeated k-fold) and then
fitted on all of it and scored once on the test part.

The models compare covariate sets S (none, each primary variable alone, all
of them), each as `outcome ~ S` (the baseline) and `outcome ~ S + curve`, so
the table shows what the curve adds beyond S. Ordinary least squares
throughout; its out-of-fold predictions are linear in the outcome, so the
permutation tests (`p_perm`: the model against chance; `p_gain`: the curve
against its baseline, Freedman & Lane 1983) cost one matrix product per
model. `docs/curve-prediction-columns.md` explains every column.
"""
from collections.abc import Collection
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

from .subject_statistics import _clean
from .subject_statistics import design_columns
from .subject_statistics import permutations

#: The columns of an `evaluate_curves` table.
PREDICTION_COLUMNS = (
    "outcome",
    "component",
    "model",
    "covariates",
    "curve",
    "n_train",
    "n_test",
    "n_params",
    "cv_r2",
    "cv_r2_sd",
    "cv_rmse",
    "cv_rmse_sd",
    "cv_mae",
    "cv_mae_sd",
    "test_r2",
    "test_rmse",
    "test_mae",
    "train_r2_adj",
    "aic",
    "bic",
    "cv_r2_gain",
    "p_perm",
    "p_gain",
    "p_fdr",
    "significant",
    "note",
)
#: A permuted score counts as at least the observed one within this.
_TIE = 1e-10

#: Training and test indices of a split, and folds of the training part:
#: per repeat, its `(fit, held_out)` index pairs.
Folds = list[list[tuple[np.ndarray, np.ndarray]]]


def covariate_sets(primary: Sequence[str]) -> list[tuple[str, ...]]:
    """The covariate sets to compare: none, each primary column alone, and
    all of them (without repeats, so one primary column gives two)."""
    return list(dict.fromkeys([(), *((p,) for p in primary), tuple(primary)]))


def _strata_usable(strata: np.ndarray | None, smallest: int) -> bool:
    if strata is None:
        return False
    counts = pd.Series(strata).value_counts()
    return len(counts) >= 2 and int(counts.min()) >= smallest


def split_subjects(
    n: int,
    strata: np.ndarray | None,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sorted training and test indices of `n` subjects, the test part
    `test_fraction` of them, stratified by `strata` when it is given and
    every level has 2 subjects; else unstratified, with a note saying why."""
    note = ""
    if strata is not None:
        if _strata_usable(strata, 2):
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_fraction,
                random_state=seed,
            )
            try:
                train, test = next(splitter.split(np.zeros(n), strata))
                return np.sort(train), np.sort(test), ""
            except ValueError as error:
                note = f"split not stratified ({error})"
        else:
            note = "split not stratified: a level has fewer than 2 subjects"
    splitter = ShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train, test = next(splitter.split(np.zeros(n)))
    return np.sort(train), np.sort(test), note


def cv_folds(
    n: int,
    strata: np.ndarray | None,
    folds: int,
    repeats: int,
    seed: int,
) -> tuple[Folds, str]:
    """`repeats` k-fold partitions of `n` subjects, stratified by `strata`
    when every level has at least `folds` subjects; else unstratified, with
    a note when `strata` was given."""
    note = ""
    if _strata_usable(strata, folds):
        splitter: Any = RepeatedStratifiedKFold(
            n_splits=folds,
            n_repeats=repeats,
            random_state=seed,
        )
        splits = list(splitter.split(np.zeros(n), strata))
    else:
        if strata is not None:
            note = f"folds not stratified: a level has fewer than {folds} subjects"
        splitter = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
        splits = list(splitter.split(np.zeros(n)))
    starts = range(0, len(splits), folds)
    return [splits[start:][:folds] for start in starts], note


def oof_operator(
    design: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """`M` with `M @ y` the out-of-fold OLS predictions of `y` on `design`
    (with its intercept column): each held-out row predicted from the fit
    on the others. Least squares through `pinv`, so a rank-deficient fold
    (a level missing from it) still predicts."""
    n = len(design)
    operator = np.zeros((n, n))
    for fit, held_out in folds:
        operator[np.ix_(held_out, fit)] = design[held_out] @ np.linalg.pinv(design[fit])
    return operator


def scores(
    y: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R², RMSE and MAE of `predictions` per column (or for one vector), R²
    against each column's own mean, as `sklearn.metrics.r2_score`."""
    y = np.asarray(y, dtype=float)
    observed = y.reshape(len(y), -1)
    residual = observed - np.asarray(predictions).reshape(observed.shape)
    ss_res = (residual**2).sum(0)
    ss_tot = ((observed - observed.mean(0)) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)
    rmse = np.sqrt((residual**2).mean(0))
    mae = np.abs(residual).mean(0)
    if y.ndim == 1:
        return r2[0], rmse[0], mae[0]
    return r2, rmse, mae


class _CrossValidated:
    """A design's out-of-fold operators, one per repeat of the folds."""

    def __init__(self, design: np.ndarray, folds: Folds) -> None:
        self.operators = [oof_operator(design, repeat) for repeat in folds]

    def scores(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per repeat, R², RMSE and MAE of the pooled out-of-fold
        predictions; for a 2-D `y` (one outcome per column) each
        `(repeats, columns)`."""
        per_repeat = [scores(y, operator @ y) for operator in self.operators]
        r2, rmse, mae = (np.array(values) for values in zip(*per_repeat))
        return r2, rmse, mae

    def cv_r2(self, y: np.ndarray) -> np.ndarray:
        """The mean over repeats of the pooled out-of-fold R²."""
        return self.scores(y)[0].mean(0)


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else np.nan


def _p(observed: float, permuted: np.ndarray) -> float:
    return float((1 + (permuted >= observed - _TIE).sum()) / (1 + len(permuted)))


def _evaluate(
    y: np.ndarray,
    design: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    folds: Folds,
    orders: np.ndarray,
) -> tuple[dict[str, Any], _CrossValidated | None]:
    """One model's scores: cross-validated on the `train` subjects, then
    fitted on all of them and scored on `test`; and its `p_perm` against
    the outcome permuted within the training part. Returns the row values
    and the cross-validation (None when skipped)."""
    fit_design, y_train = design[train], y[train]
    n_params = int(np.linalg.matrix_rank(fit_design))
    row: dict[str, Any] = {
        "n_train": len(train),
        "n_test": len(test),
        "n_params": n_params,
    }
    smallest = min(len(fit) for repeat in folds for fit, _ in repeat)
    if smallest <= n_params + 1:
        row["note"] = (
            f"{smallest} subjects in the smallest training fold for {n_params} "
            "model parameters"
        )
        return row, None
    cross_validated = _CrossValidated(fit_design, folds)
    r2, rmse, mae = cross_validated.scores(y_train)
    beta = np.linalg.pinv(fit_design) @ y_train
    test_r2, test_rmse, test_mae = scores(y[test], design[test] @ beta)
    ols = sm.OLS(y_train, fit_design).fit()
    row |= {
        "cv_r2": float(r2.mean()),
        "cv_r2_sd": _sd(r2),
        "cv_rmse": float(rmse.mean()),
        "cv_rmse_sd": _sd(rmse),
        "cv_mae": float(mae.mean()),
        "cv_mae_sd": _sd(mae),
        "test_r2": float(test_r2) if len(test) > 1 else np.nan,
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "train_r2_adj": float(ols.rsquared_adj) if design.shape[1] > 1 else 0.0,
        "aic": float(ols.aic),
        "bic": float(ols.bic),
    }
    if design.shape[1] > 1:
        permuted = cross_validated.cv_r2(y_train[orders].T)
        row["p_perm"] = _p(row["cv_r2"], permuted)
    return row, cross_validated


def _gain_p(
    y_train: np.ndarray,
    baseline_design: np.ndarray,
    baseline: _CrossValidated,
    curve: _CrossValidated,
    observed: float,
    orders: np.ndarray,
) -> float:
    """The p-value of the curve's CV R² gain over its baseline: Freedman-
    Lane, permuting the residuals of the outcome on the baseline design and
    adding its fit back, so the baseline's own predictive value is kept."""
    fitted = baseline_design @ (np.linalg.pinv(baseline_design) @ y_train)
    residual = y_train - fitted
    permuted = fitted[:, None] + residual[orders].T
    gains = curve.cv_r2(permuted) - baseline.cv_r2(permuted)
    return _p(observed, gains)


def _model_names(covariates: tuple[str, ...]) -> tuple[str, str, str]:
    """`(covariates text, baseline name, curve model name)`."""
    if not covariates:
        return "none", "intercept only", "curve"
    text = " + ".join(covariates)
    return text, f"{text} only", f"{text} + curve"


def evaluate_curves(
    curves: Mapping[int, pd.DataFrame],
    outcomes: Sequence[str],
    info: pd.DataFrame,
    primary: Sequence[str],
    categorical: Collection[str],
    stratify_by: str | None = None,
    test_fraction: float = 0.2,
    folds: int = 5,
    repeats: int = 10,
    n_permutations: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """How well each component's curve predicts each outcome, beyond each
    covariate set (`covariate_sets(primary)`).

    `curves` maps a component (1-based) to its curves, subjects x time
    points, indexed like `info` (subjects). Per outcome, the subjects with
    the outcome, every primary value and every time point are split once
    (`split_subjects`, stratified by `stratify_by`) and folded once
    (`cv_folds`), so every model of that outcome is scored on the same
    subjects. Terms in `categorical` are treatment coded; outcomes must be
    numerical. Rows have the `PREDICTION_COLUMNS` but `p_fdr` and
    `significant` (`subject_statistics.add_fdr` on `p_gain`), baselines
    first, then one row per component.
    """
    terms = list(dict.fromkeys([*outcomes, *primary]))
    data = _clean(info, terms, categorical)
    aligned = {c: frame.reindex(info.index) for c, frame in curves.items()}
    every_timepoint = np.logical_and.reduce(
        [frame.notna().all(axis=1).to_numpy() for frame in aligned.values()],
    )
    sets = covariate_sets(primary)
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        usable = (
            data[[outcome, *primary]].notna().all(axis=1).to_numpy() & every_timepoint
        )
        sample = data[usable]
        y = sample[outcome].to_numpy(dtype=float)
        notes = []
        if (~usable).sum():
            notes.append(
                f"{(~usable).sum()} subject(s) left out (no {outcome}, primary "
                "value or time point)",
            )
        outcome_rows = [
            {
                "outcome": outcome,
                "component": component,
                "model": names[2] if component else names[1],
                "covariates": names[0],
                "curve": bool(component),
            }
            for covariates in sets
            for names in [_model_names(covariates)]
            for component in [None, *aligned]
        ]
        skip = ""
        if len(sample) < 2 * folds or np.unique(y).size < 2:
            skip = (
                f"'{outcome}' is constant"
                if len(sample) >= 2 * folds
                else f"{len(sample)} subjects, too few for {folds} folds and a test part"
            )
        if skip:
            rows += [{**row, "note": "; ".join([*notes, skip])} for row in outcome_rows]
            continue

        strata = (
            sample[stratify_by].astype(str).to_numpy()
            if stratify_by is not None
            else None
        )
        train, test, split_note = split_subjects(
            len(sample),
            strata,
            test_fraction,
            seed,
        )
        fold_sets, fold_note = cv_folds(
            len(train),
            None if strata is None else strata[train],
            folds,
            repeats,
            seed,
        )
        notes += [note for note in (split_note, fold_note) if note]
        orders = permutations(len(train), n_permutations, seed)
        rows_by_set = iter(outcome_rows)
        for covariates in sets:
            covariate_columns, _ = design_columns(sample, list(covariates), categorical)
            baseline_design = np.column_stack(
                [np.ones(len(sample)), covariate_columns.to_numpy(dtype=float)],
            )
            baseline_row, baseline = _evaluate(
                y,
                baseline_design,
                train,
                test,
                fold_sets,
                orders,
            )
            rows.append({**next(rows_by_set), **baseline_row})
            for component, frame in aligned.items():
                row = next(rows_by_set)
                curve = frame[usable].to_numpy(dtype=float)
                if not np.any(curve):
                    rows.append(
                        {**row, "note": f"component {component}'s curve is zero"},
                    )
                    continue
                design = np.column_stack([baseline_design, curve])
                curve_row, model = _evaluate(y, design, train, test, fold_sets, orders)
                if model is not None and baseline is not None:
                    gain = curve_row["cv_r2"] - baseline_row["cv_r2"]
                    curve_row["cv_r2_gain"] = gain
                    curve_row["p_gain"] = _gain_p(
                        y[train],
                        baseline_design[train],
                        baseline,
                        model,
                        gain,
                        orders,
                    )
                rows.append({**row, **curve_row})
        first = len(rows) - len(outcome_rows)
        for row in rows[first:]:
            row["note"] = "; ".join(filter(None, [*notes, str(row.get("note", ""))]))
    table = pd.DataFrame(rows).reindex(columns=PREDICTION_COLUMNS)
    table["component"] = table["component"].astype("Int64")
    table["curve"] = table["curve"].astype(bool)
    table["note"] = table["note"].fillna("")
    return table
