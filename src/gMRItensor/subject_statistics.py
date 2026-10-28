"""Covariate-adjusted tests of the subject coefficients (no plotting).

Per component, a linear model of the coefficient tests each primary
variable (e.g. diagnosis, sex, age) with the other primary variables as
nuisance, and each clinical variable with all primary variables as
nuisance. p-values are Freedman-Lane permutation p-values of the term's
partial F (Freedman & Lane 1983; Winkler et al. 2014), BH-corrected per
table (`add_fdr`). `docs/subject-statistics.md` explains every column.
"""
from collections.abc import Collection
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

#: The columns of a `primary_effects` / `clinical_effects` table.
COLUMNS = (
    "component",
    "term",
    "level",
    "n",
    "covariates",
    "estimate",
    "ci_low",
    "ci_high",
    "standardised",
    "std_kind",
    "partial_r2",
    "partial_r",
    "vif",
    "r_unadjusted",
    "p_unadjusted",
    "p_parametric",
    "p_perm",
    "p_fdr",
    "significant",
    "note",
)
#: Permuted residuals are scored this many permutations at a time.
_CHUNK = 1000


def permutations(n: int, n_permutations: int, seed: int) -> np.ndarray:
    """`(n_permutations, n)` row orders, the same for the same `n` and
    `seed`, so every test on one sample shares them."""
    rng = np.random.default_rng(seed)
    return rng.permuted(np.tile(np.arange(n), (n_permutations, 1)), axis=1)


def _residual_maker(matrix: np.ndarray) -> np.ndarray:
    return np.eye(len(matrix)) - matrix @ np.linalg.pinv(matrix)


def freedman_lane(
    y: np.ndarray,
    nuisance: np.ndarray,
    effect: np.ndarray,
    orders: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial F of the `effect` columns given the `nuisance` columns, and
    its Freedman-Lane permutation p-value, per column of `y`.

    The residuals of `y` on `nuisance` alone are permuted by each row of
    `orders` (`permutations`), and F is recomputed for the full model;
    adding the reduced fit back is left out as it changes neither residual
    sum of squares. p = (1 + #{F* >= F}) / (1 + number of permutations).
    `nuisance` should hold the intercept. Returns `(F, p)`, one value per
    column of a 2-D `y` and 0-d for a 1-D `y`.
    """
    y = np.asarray(y, dtype=float)
    ys = y.reshape(len(y), -1)
    full = np.column_stack([nuisance, effect])
    residual_full = _residual_maker(full)
    residual_reduced = _residual_maker(nuisance)
    q = np.linalg.matrix_rank(full) - np.linalg.matrix_rank(nuisance)
    dfe = len(ys) - np.linalg.matrix_rank(full)

    def statistic(values: np.ndarray) -> np.ndarray:
        rss_full = (np.einsum("ij,...jk->...ik", residual_full, values) ** 2).sum(-2)
        rss_reduced = (np.einsum("ij,...jk->...ik", residual_reduced, values) ** 2).sum(
            -2,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return ((rss_reduced - rss_full) / q) / (rss_full / dfe)

    observed = statistic(ys)
    residuals = residual_reduced @ ys
    exceed = np.zeros(ys.shape[1])
    for start in range(0, len(orders), _CHUNK):
        permuted = residuals[orders[start:][:_CHUNK]]
        exceed += (statistic(permuted) >= observed * (1 - 1e-10)).sum(0)
    p = (1 + exceed) / (1 + len(orders))
    if y.ndim == 1:
        return observed[0], p[0]
    return observed, p


def _clean(
    info: pd.DataFrame,
    terms: Sequence[str],
    categorical: Collection[str],
) -> pd.DataFrame:
    """`terms` of `info`: numerical ones as floats, categorical ones as
    they are, except integer-valued floats as integers (codes read from a
    column with a blank cell). Raises `ValueError` naming a numerical
    column with a non-numeric value."""
    columns = {}
    for term in terms:
        values = info[term]
        if term in categorical:
            if pd.api.types.is_float_dtype(values) and (values.dropna() % 1 == 0).all():
                values = values.astype("Int64")
            columns[term] = values
            continue
        try:
            columns[term] = pd.to_numeric(values).astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"numerical column '{term}' has a non-numeric value ({error})",
            ) from error
    return pd.DataFrame(columns, index=info.index)


def _levels(values: pd.Series) -> list[Any]:
    return sorted(values.dropna().unique())


def design_columns(
    data: pd.DataFrame,
    terms: Sequence[str],
    categorical: Collection[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """The columns of `terms` (no intercept) and, per term, its columns
    with the level each one contrasts ("" for a numerical term).

    A categorical term is treatment coded: one 0/1 column per level but the
    first sorted one, the reference, labelled "<level> vs <reference>".
    """
    blocks, labels = [], {}
    for term in terms:
        if term not in categorical:
            blocks.append(data[[term]].astype(float))
            labels[term] = {term: ""}
            continue
        levels = _levels(data[term])
        names = {f"{term}: {level} vs {levels[0]}": level for level in levels[1:]}
        blocks.append(
            pd.DataFrame(
                {
                    name: (data[term] == level).astype(float)
                    for name, level in names.items()
                },
                index=data.index,
            ),
        )
        labels[term] = {name: name.split(": ", 1)[1] for name in names}
    design = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=data.index)
    return design, labels


def generalised_vif(design: pd.DataFrame, columns: Sequence[str]) -> float:
    """GVIF^(1/df) of `columns` against the other columns of `design`
    (Fox & Monette 1992): the ordinary VIF for one column, and on the same
    scale for a factor's columns, without inflation from its own dummies
    predicting each other. 1 when there are no other columns."""
    others = [c for c in design.columns if c not in columns]
    if not others:
        return 1.0
    corr = np.corrcoef(design.to_numpy(dtype=float), rowvar=False)
    tested = [design.columns.get_loc(c) for c in columns]
    rest = [design.columns.get_loc(c) for c in others]
    gvif = (
        np.linalg.det(corr[np.ix_(tested, tested)])
        * np.linalg.det(corr[np.ix_(rest, rest)])
        / np.linalg.det(corr)
    )
    return float(gvif ** (1 / len(columns)))


def _skip_reason(
    data: pd.DataFrame,
    term: str,
    categorical: Collection[str],
    n_params: int,
) -> str:
    """Why `term` cannot be tested in `data`, or ""."""
    n = len(data)
    if term in categorical:
        counts = data[term].value_counts()
        if len(counts) < 2:
            return f"'{term}' has one level in the {n} subjects with values"
        small = [str(level) for level, count in counts.items() if count < 2]
        if small:
            return f"'{term}' level(s) {', '.join(small)} have fewer than 2 subjects"
    elif data[term].nunique() < 2:
        return f"'{term}' is constant in the {n} subjects with values"
    if n <= n_params + 2:
        return f"{n} subjects for {n_params} model parameters"
    return ""


def _empty_rows(
    k: int,
    term: str,
    levels: Sequence[str],
    n: int,
    covariates: str,
    note: str,
) -> list[dict[str, Any]]:
    return [
        {
            "component": c + 1,
            "term": term,
            "level": level,
            "n": n,
            "covariates": covariates,
            "note": note,
        }
        for c in range(k)
        for level in levels or [""]
    ]


def _term_rows(
    y: np.ndarray,
    data: pd.DataFrame,
    term: str,
    nuisance: Sequence[str],
    categorical: Collection[str],
    orders: np.ndarray,
    note: str,
) -> list[dict[str, Any]]:
    """Rows of `term` tested against `nuisance` in the complete cases
    `data`, one per component (column of `y`) and contrast."""
    n, k = y.shape
    covariates = " + ".join(nuisance)
    design, labels = design_columns(data, [*nuisance, term], categorical)
    tested = list(labels[term])
    n_params = design.shape[1] + 1
    reason = _skip_reason(data, term, categorical, n_params)
    full = np.column_stack([np.ones(n), design.to_numpy()])
    if not reason and np.linalg.matrix_rank(full) < full.shape[1]:
        reason = f"'{term}' is collinear with {covariates or 'the intercept'}"
    if reason:
        levels = [labels[term][c] for c in tested]
        return _empty_rows(
            k,
            term,
            levels,
            n,
            covariates,
            "; ".join(filter(None, [note, reason])),
        )

    intercept = np.ones((n, 1))
    reduced = np.column_stack([intercept, design.drop(columns=tested).to_numpy()])
    effect = design[tested].to_numpy()
    f_value, p_perm = freedman_lane(y, reduced, effect, orders)
    _, p_unadjusted = freedman_lane(y, intercept, effect, orders)
    rss_full = ((_residual_maker(full) @ y) ** 2).sum(0)
    rss_reduced = ((_residual_maker(reduced) @ y) ** 2).sum(0)
    partial_r2 = (rss_reduced - rss_full) / rss_reduced
    vif = generalised_vif(design, tested)
    exog = pd.DataFrame(full, columns=["intercept", *design.columns])

    rows = []
    for c in range(k):
        fit = sm.OLS(y[:, c], exog).fit()
        sigma = np.sqrt(fit.mse_resid)
        for column in tested:
            estimate = float(fit.params[column])
            ci_low, ci_high = fit.conf_int().loc[column]
            row: dict[str, Any] = {
                "component": c + 1,
                "term": term,
                "level": labels[term][column],
                "n": n,
                "covariates": covariates,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "partial_r2": partial_r2[c],
                "vif": vif,
                "p_unadjusted": p_unadjusted[c],
                "p_parametric": float(fit.pvalues[column]),
                "p_perm": p_perm[c],
                "note": note,
            }
            if term in categorical:
                row["standardised"] = estimate / sigma
                row["std_kind"] = "d"
            else:
                x = data[term].to_numpy(dtype=float)
                row["standardised"] = estimate * x.std(ddof=1) / y[:, c].std(ddof=1)
                row["std_kind"] = "beta"
                row["partial_r"] = np.sign(estimate) * np.sqrt(partial_r2[c])
                row["r_unadjusted"] = np.corrcoef(x, y[:, c])[0, 1]
            rows.append(row)
    return rows


def _effects(
    coefficients: np.ndarray,
    info: pd.DataFrame,
    tests: Sequence[tuple[str, Sequence[str]]],
    categorical: Collection[str],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """One `_term_rows` per `(term, nuisance)` in `tests`, each in the
    subjects with every value it needs; a nuisance term that is constant
    there is left out with a note."""
    coefficients = np.asarray(coefficients, dtype=float)
    if len(coefficients) != len(info):
        raise ValueError(
            f"{len(coefficients)} coefficient rows but {len(info)} subject-info rows",
        )
    terms = list(
        dict.fromkeys(t for term, nuisance in tests for t in [term, *nuisance]),
    )
    data = _clean(info, terms, categorical)
    rows = []
    for term, nuisance in tests:
        complete = data[[term, *nuisance]].notna().all(axis=1).to_numpy()
        sample = data[complete]
        constant = [t for t in nuisance if sample[t].nunique() < 2]
        note = (
            f"not adjusted for {', '.join(constant)} (constant in these subjects)"
            if constant
            else ""
        )
        rows += _term_rows(
            coefficients[complete],
            sample,
            term,
            [t for t in nuisance if t not in constant],
            categorical,
            permutations(int(complete.sum()), n_permutations, seed),
            note,
        )
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    table = table.sort_values("component", kind="stable", ignore_index=True)
    table["note"] = table["note"].fillna("")
    table["std_kind"] = table["std_kind"].fillna("")
    return table


def primary_effects(
    coefficients: np.ndarray,
    info: pd.DataFrame,
    primary: Sequence[str],
    categorical: Collection[str],
    n_permutations: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Each `primary` term per component, adjusted for the other primary
    terms, in the subjects with every primary value.

    `coefficients` is `(n_subjects, n_components)` and `info` holds the
    same subjects in the same order. Terms in `categorical` are treatment
    coded against their first sorted level; the others must be numeric.
    Returns one row per component and term (per contrast for a factor) with
    the `COLUMNS` but `p_fdr` and `significant` (`add_fdr`).
    """
    tests = [(term, [t for t in primary if t != term]) for term in primary]
    return _effects(coefficients, info, tests, categorical, n_permutations, seed)


def clinical_effects(
    coefficients: np.ndarray,
    info: pd.DataFrame,
    clinical: Sequence[str],
    primary: Sequence[str],
    categorical: Collection[str],
    n_permutations: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Each `clinical` term per component, adjusted for all `primary`
    terms, in the subjects with that clinical value and every primary one.
    Arguments and rows as in `primary_effects`."""
    tests = [(term, list(primary)) for term in clinical]
    return _effects(coefficients, info, tests, categorical, n_permutations, seed)


def add_fdr(table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """`table` with `p_fdr`, Benjamini-Hochberg over its tested terms (a
    factor's contrast rows share one `p_perm` and count once; untested rows
    are left out), and `significant` = `p_fdr < alpha`."""
    table = table.copy()
    terms = table.dropna(subset=["p_perm"]).drop_duplicates(["component", "term"])
    adjusted: dict[tuple[Any, Any], float] = {}
    if len(terms):
        p_fdr = multipletests(terms["p_perm"], method="fdr_bh")[1]
        adjusted = dict(zip(zip(terms["component"], terms["term"]), p_fdr))
    table["p_fdr"] = [
        adjusted.get(key, np.nan) for key in zip(table["component"], table["term"])
    ]
    table["significant"] = table["p_fdr"] < alpha
    return table
