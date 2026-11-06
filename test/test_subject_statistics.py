"""Covariate-adjusted subject-coefficient tests (`gMRItensor.subject_statistics`)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from gMRItensor.subject_statistics import add_fdr
from gMRItensor.subject_statistics import add_interaction_fdr
from gMRItensor.subject_statistics import clinical_effects
from gMRItensor.subject_statistics import COLUMNS
from gMRItensor.subject_statistics import design_columns
from gMRItensor.subject_statistics import freedman_lane
from gMRItensor.subject_statistics import generalised_vif
from gMRItensor.subject_statistics import INTERACTION_COLUMNS
from gMRItensor.subject_statistics import interaction_effects
from gMRItensor.subject_statistics import permutations
from gMRItensor.subject_statistics import primary_effects
from scipy.stats import norm
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

DOCS = Path(__file__).parents[1] / "docs" / "subject-statistics.md"
PRIMARY = ["diagnosis", "sex", "age"]
CATEGORICAL = {"diagnosis", "sex", "apoe"}
CORRELATION_CIS = [
    "partial_r_ci_low",
    "partial_r_ci_high",
    "r_unadjusted_ci_low",
    "r_unadjusted_ci_high",
]


def _study(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Subject info and 3 components: 1 depends on diagnosis, 2 on age, 3 is
    noise. `egfr` falls with age; `apoe` has 3 levels."""
    rng = np.random.default_rng(seed)
    info = pd.DataFrame(
        {
            "diagnosis": np.repeat(["Control", "PD"], n // 2),
            "sex": np.tile(["F", "M"], n // 2),
            "age": rng.normal(65, 8, n),
            "apoe": rng.choice([2, 3, 4], n),
        },
    )
    info["egfr"] = 140 - 0.9 * info["age"] + rng.normal(0, 5, n)
    pd_ = (info["diagnosis"] == "PD").to_numpy(float)
    age = info["age"].to_numpy()
    coefficients = np.column_stack(
        [
            2 + 2 * pd_ + rng.normal(0, 0.5, n),
            2 + 0.1 * (age - 65) + rng.normal(0, 0.3, n),
            2 + rng.normal(0, 0.5, n),
        ],
    )
    return info, coefficients


def test_ols_columns_match_statsmodels_formula() -> None:
    info, coefficients = _study()
    table = primary_effects(coefficients, info, PRIMARY, CATEGORICAL, 200)

    frame = info.assign(y=coefficients[:, 0])
    fit = smf.ols("y ~ C(diagnosis) + C(sex) + age", frame).fit()
    rows = table[table["component"] == 1].set_index("term")
    for term, name in [
        ("diagnosis", "C(diagnosis)[T.PD]"),
        ("sex", "C(sex)[T.M]"),
        ("age", "age"),
    ]:
        assert rows.loc[term, "estimate"] == pytest.approx(fit.params[name])
        assert rows.loc[term, "ci_low"] == pytest.approx(fit.conf_int().loc[name, 0])
        assert rows.loc[term, "ci_high"] == pytest.approx(fit.conf_int().loc[name, 1])
        assert rows.loc[term, "p_parametric"] == pytest.approx(fit.pvalues[name])
    assert rows.loc["diagnosis", "level"] == "PD vs Control"
    assert rows.loc["age", "level"] == ""
    assert rows.loc["age", "covariates"] == "diagnosis + sex"
    assert rows.loc["age", "standardised"] == pytest.approx(
        fit.params["age"] * info["age"].std() / coefficients[:, 0].std(ddof=1),
    )
    assert rows.loc["diagnosis", "standardised"] == pytest.approx(
        fit.params["C(diagnosis)[T.PD]"] / np.sqrt(fit.mse_resid),
    )


def test_partial_f_matches_statsmodels_f_test() -> None:
    info, coefficients = _study()
    frame = info.assign(y=coefficients[:, 2])
    fit = smf.ols("y ~ C(apoe) + C(diagnosis) + age", frame).fit()
    expected = fit.f_test("C(apoe)[T.3] = 0, C(apoe)[T.4] = 0")

    design, labels = design_columns(info, ["diagnosis", "age", "apoe"], CATEGORICAL)
    nuisance = np.column_stack(
        [np.ones(len(info)), design[["diagnosis: PD vs Control", "age"]]],
    )
    f_value, _ = freedman_lane(
        coefficients[:, 2],
        nuisance,
        design[list(labels["apoe"])].to_numpy(),
        permutations(len(info), 10, 0),
    )
    assert f_value == pytest.approx(float(expected.fvalue))


def test_p_perm_matches_nilearn_permuted_ols() -> None:
    nilearn = pytest.importorskip("nilearn.mass_univariate")
    info, coefficients = _study(n=30, seed=3)
    y = coefficients[:, 1:2]
    confounds = np.column_stack(
        [(info["diagnosis"] == "PD").astype(float), info["sex"] == "M"],
    )
    tested = info[["egfr"]].to_numpy()

    expected = nilearn.permuted_ols(
        tested,
        y,
        confounding_vars=confounds.astype(float),
        n_perm=10000,
        random_state=0,
        output_type="dict",
    )
    nuisance = np.column_stack([np.ones(len(y)), confounds])
    f_value, p = freedman_lane(
        y[:, 0],
        nuisance,
        tested,
        permutations(len(y), 10000, 1),
    )

    assert f_value == pytest.approx(float(expected["t"][0, 0]) ** 2)
    assert p == pytest.approx(10 ** -float(expected["logp_max_t"][0, 0]), abs=0.02)


def test_planted_effects_are_found() -> None:
    info, coefficients = _study()
    table = add_fdr(primary_effects(coefficients, info, PRIMARY, CATEGORICAL, 2000))
    hits = table.loc[table["significant"], ["component", "term"]]
    significant = set(hits.itertuples(index=False, name=None))

    assert (1, "diagnosis") in significant
    assert (2, "age") in significant
    assert not any(component == 3 for component, _ in significant)
    assert (
        table.loc[
            (table["component"] == 1) & (table["term"] == "diagnosis"),
            "p_perm",
        ].item()
        < 0.01
    )


def _fisher_ci(r: float, dof: int) -> tuple[float, float]:
    """95% Fisher-z interval of a correlation, `dof` = n - 3 - the number
    of covariate columns."""
    half = norm.ppf(0.975) / np.sqrt(dof)
    return np.tanh(np.arctanh(r) - half), np.tanh(np.arctanh(r) + half)


def test_correlation_cis_are_fisher_z_intervals() -> None:
    info, coefficients = _study()
    table = clinical_effects(coefficients, info, ["egfr"], PRIMARY, CATEGORICAL, 200)
    covariates = "C(diagnosis) + C(sex) + age"
    for c, row in table.set_index("component").iterrows():
        frame = info.assign(y=coefficients[:, c - 1])
        unadjusted = pearsonr(frame["egfr"], frame["y"]).confidence_interval(0.95)
        assert row["r_unadjusted_ci_low"] == pytest.approx(unadjusted.low)
        assert row["r_unadjusted_ci_high"] == pytest.approx(unadjusted.high)

        partial = pearsonr(
            smf.ols(f"egfr ~ {covariates}", frame).fit().resid,
            smf.ols(f"y ~ {covariates}", frame).fit().resid,
        ).statistic
        low, high = _fisher_ci(partial, len(frame) - 3 - 3)
        assert row["partial_r"] == pytest.approx(partial)
        assert row["partial_r_ci_low"] == pytest.approx(low)
        assert row["partial_r_ci_high"] == pytest.approx(high)
        for r in ("partial_r", "r_unadjusted"):
            assert -1 < row[f"{r}_ci_low"] < row[r] < row[f"{r}_ci_high"] < 1


def test_partial_r_ci_without_covariates_is_pearsons() -> None:
    info, coefficients = _study()
    table = primary_effects(coefficients, info, ["age"], CATEGORICAL, 200)
    for c, row in table.set_index("component").iterrows():
        expected = pearsonr(info["age"], coefficients[:, c - 1])
        interval = expected.confidence_interval(0.95)
        assert row["partial_r"] == pytest.approx(expected.statistic)
        assert row["partial_r_ci_low"] == pytest.approx(interval.low)
        assert row["partial_r_ci_high"] == pytest.approx(interval.high)
        assert row["partial_r_ci_low"] == pytest.approx(row["r_unadjusted_ci_low"])
        assert row["partial_r_ci_high"] == pytest.approx(row["r_unadjusted_ci_high"])


def test_adjustment_removes_a_confounded_association() -> None:
    info, coefficients = _study(n=60, seed=0)
    table = clinical_effects(coefficients, info, ["egfr"], PRIMARY, CATEGORICAL, 2000)
    row = table[table["component"] == 2].iloc[0]

    assert row["p_unadjusted"] < 0.05
    assert row["r_unadjusted"] < 0
    assert row["p_perm"] > 0.05
    assert row["vif"] > 2  # egfr largely duplicates age
    assert row["covariates"] == "diagnosis + sex + age"


def test_factor_contrasts_share_one_test_and_count_once_in_fdr() -> None:
    info, coefficients = _study()
    table = add_fdr(
        clinical_effects(
            coefficients,
            info,
            ["apoe", "egfr"],
            PRIMARY,
            CATEGORICAL,
            500,
        ),
    )
    apoe = table[(table["term"] == "apoe") & (table["component"] == 1)]

    assert list(apoe["level"]) == ["3 vs 2", "4 vs 2"]
    assert apoe["p_perm"].nunique() == 1
    assert apoe["p_fdr"].nunique() == 1
    assert apoe["partial_r2"].nunique() == 1
    assert apoe["std_kind"].tolist() == ["d", "d"]
    assert apoe["partial_r"].isna().all() and apoe["r_unadjusted"].isna().all()
    assert apoe[CORRELATION_CIS].isna().all().all()

    terms = table.drop_duplicates(["component", "term"])
    from statsmodels.stats.multitest import multipletests

    expected = multipletests(terms["p_perm"], method="fdr_bh")[1]
    assert terms["p_fdr"].to_numpy() == pytest.approx(expected)


def test_fdr_is_per_table() -> None:
    info, coefficients = _study()
    primary = add_fdr(primary_effects(coefficients, info, PRIMARY, CATEGORICAL, 300))
    from statsmodels.stats.multitest import multipletests

    assert len(primary) == 9
    assert primary["p_fdr"].to_numpy() == pytest.approx(
        multipletests(primary["p_perm"], method="fdr_bh")[1],
    )


def test_generalised_vif() -> None:
    info, _ = _study()
    design, labels = design_columns(
        info,
        ["diagnosis", "age", "egfr", "apoe"],
        CATEGORICAL,
    )
    exog = np.column_stack([np.ones(len(design)), design])

    for index, column in enumerate(design.columns):
        if column in ("age", "egfr"):
            assert generalised_vif(design, [column]) == pytest.approx(
                variance_inflation_factor(exog, index + 1),
            )
    # A factor is not inflated by its own dummies predicting each other.
    apoe = list(labels["apoe"])
    assert generalised_vif(design, apoe) < variance_inflation_factor(exog, 4)
    corr = np.corrcoef(design.to_numpy(), rowvar=False)
    tested, rest = [3, 4], [0, 1, 2]
    gvif = (
        np.linalg.det(corr[np.ix_(tested, tested)])
        * np.linalg.det(corr[np.ix_(rest, rest)])
        / np.linalg.det(corr)
    )
    assert generalised_vif(design, apoe) == pytest.approx(np.sqrt(gvif))
    assert generalised_vif(design[apoe], apoe) == 1.0


def test_missing_values_constant_nuisance_and_skips() -> None:
    info, coefficients = _study()
    info["updrs"] = np.where(info["diagnosis"] == "PD", np.arange(len(info)), np.nan)
    info["one_level"] = "x"
    info.loc[[0, 1, 2], "age"] = np.nan
    table = add_fdr(
        clinical_effects(
            coefficients,
            info,
            ["updrs", "one_level", "egfr"],
            PRIMARY,
            CATEGORICAL | {"one_level"},
            200,
        ),
    )
    rows = table[table["component"] == 1].set_index("term")

    # updrs: PD only, so diagnosis is constant and left out.
    assert rows.loc["updrs", "n"] == 20
    assert rows.loc["updrs", "covariates"] == "sex + age"
    assert "not adjusted for diagnosis" in rows.loc["updrs", "note"]
    assert np.isfinite(rows.loc["updrs", "p_perm"])
    # Subjects without age are left out of every clinical model.
    assert rows.loc["egfr", "n"] == 37
    # An untestable term gets a reason and no p, and is not in the FDR family.
    assert "one level" in rows.loc["one_level", "note"]
    assert np.isnan(rows.loc["one_level", "p_perm"])
    assert np.isnan(rows.loc["one_level", "p_fdr"])
    assert not rows.loc["one_level", "significant"]


def test_too_few_subjects_and_rare_levels_are_skipped() -> None:
    info, coefficients = _study(n=8)
    info["rare"] = ["a"] * 7 + ["b"]
    table = clinical_effects(
        coefficients,
        info,
        ["rare"],
        PRIMARY,
        CATEGORICAL | {"rare"},
        100,
    )
    assert table["note"].str.contains("fewer than 2 subjects").all()

    small = primary_effects(coefficients[:6], info.iloc[:6], PRIMARY, CATEGORICAL, 100)
    assert small["note"].str.contains("model parameters").all()
    assert small["p_perm"].isna().all()


def test_non_numeric_numerical_column_is_an_error() -> None:
    info, coefficients = _study()
    info["age"] = info["age"].astype(object)
    info.loc[0, "age"] = "old"
    with pytest.raises(ValueError, match="numerical column 'age'"):
        primary_effects(coefficients, info, PRIMARY, CATEGORICAL, 100)


def test_integer_codes_with_blank_cells_stay_integer_levels() -> None:
    info, coefficients = _study()
    info["apoe"] = info["apoe"].astype(float)
    info.loc[0, "apoe"] = np.nan
    table = clinical_effects(coefficients, info, ["apoe"], PRIMARY, CATEGORICAL, 100)
    assert set(table["level"]) == {"3 vs 2", "4 vs 2"}


def test_same_seed_same_output() -> None:
    info, coefficients = _study()
    first = primary_effects(coefficients, info, PRIMARY, CATEGORICAL, 300, seed=4)
    second = primary_effects(coefficients, info, PRIMARY, CATEGORICAL, 300, seed=4)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [c for c in COLUMNS]


@pytest.mark.slow
def test_null_rejection_rate_is_nominal() -> None:
    rng = np.random.default_rng(5)
    n, repeats = 30, 400
    rejected = 0
    for _ in range(repeats):
        nuisance = np.column_stack([np.ones(n), rng.normal(size=n)])
        effect = rng.normal(size=(n, 1)) + nuisance[:, 1:]
        y = 0.8 * nuisance[:, 1] + rng.standard_t(3, n)
        _, p = freedman_lane(
            y,
            nuisance,
            effect,
            permutations(n, 500, rng.integers(1e9)),
        )
        rejected += p < 0.05
    # 95% binomial interval around 0.05 for 400 tests: about [0.029, 0.071].
    assert 0.025 <= rejected / repeats <= 0.075


def _interaction_study(
    n: int = 40,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Subject info and 3 components: 1 rises with `egfr` in PD only (case
    A), 2 with `duration`, which is 0 in every Control (case B), 3 is noise.
    `apoe` has 3 levels."""
    rng = np.random.default_rng(seed)
    info = pd.DataFrame(
        {
            "diagnosis": np.repeat(["Control", "PD"], n // 2),
            "sex": rng.choice(["F", "M"], n),
            "age": rng.normal(65, 8, n),
            "apoe": np.resize([2, 3, 4], n),
            "egfr": rng.normal(90, 15, n),
        },
    )
    affected = (info["diagnosis"] == "PD").to_numpy(float)
    info["duration"] = affected * rng.uniform(1, 15, n)
    coefficients = np.column_stack(
        [
            2 + 0.03 * (info["egfr"] - 90) * affected + rng.normal(0, 0.1, n),
            2 + 0.3 * affected + 0.05 * info["duration"] + rng.normal(0, 0.1, n),
            2 + rng.normal(0, 0.5, n),
        ],
    )
    return info, coefficients


def _interactions(
    pairs: list[tuple[str, str]],
    primary: list[str] = PRIMARY,
    n_permutations: int = 500,
    **kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    info, coefficients = _interaction_study(**kwargs)  # type: ignore[arg-type]
    table = interaction_effects(
        coefficients,
        info,
        pairs,
        primary,
        CATEGORICAL,
        n_permutations,
    )
    return table, info, coefficients


def _rows(table: pd.DataFrame, component: int, test: str) -> pd.DataFrame:
    rows = table[(table["component"] == component) & (table["test"] == test)]
    return rows.set_index("level")


def test_slopes_per_group_match_statsmodels() -> None:
    table, info, coefficients = _interactions([("egfr", "diagnosis")])
    frame = info.assign(y=coefficients[:, 0])
    fit = smf.ols("y ~ C(diagnosis) + C(sex) + age + egfr:C(diagnosis)", frame).fit()
    slopes = _rows(table, 1, "slope")

    assert list(slopes.index) == ["slope in Control", "slope in PD"]
    for group in ("Control", "PD"):
        row, name = slopes.loc[f"slope in {group}"], f"egfr:C(diagnosis)[{group}]"
        within = frame[frame["diagnosis"] == group]
        assert row["estimate"] == pytest.approx(fit.params[name])
        assert row["ci_low"] == pytest.approx(fit.conf_int().loc[name, 0])
        assert row["ci_high"] == pytest.approx(fit.conf_int().loc[name, 1])
        assert row["p_parametric"] == pytest.approx(fit.pvalues[name])
        assert row["standardised"] == pytest.approx(
            fit.params[name] * within["egfr"].std() / within["y"].std(),
        )
        assert row["r_unadjusted"] == pytest.approx(
            np.corrcoef(within["egfr"], within["y"])[0, 1],
        )
        unadjusted = pearsonr(within["egfr"], within["y"]).confidence_interval(0.95)
        assert row["r_unadjusted_ci_low"] == pytest.approx(unadjusted.low)
        assert row["r_unadjusted_ci_high"] == pytest.approx(unadjusted.high)
        # All 40 subjects; diagnosis, sex, age and the other slope: 4 columns.
        low, high = _fisher_ci(row["partial_r"], 40 - 3 - 4)
        assert row["partial_r_ci_low"] == pytest.approx(low)
        assert row["partial_r_ci_high"] == pytest.approx(high)
        assert row["n"] == 20
    assert set(slopes["term"]) == {"egfr × diagnosis"}
    assert set(slopes["covariates"]) == {"diagnosis + sex + age"}
    assert set(slopes["std_kind"]) == {"beta"}
    assert (np.sign(slopes["partial_r"]) == np.sign(slopes["estimate"])).all()


def test_slope_difference_matches_the_statsmodels_interaction_term() -> None:
    table, info, coefficients = _interactions([("egfr", "diagnosis")])
    frame = info.assign(y=coefficients[:, 0])
    fit = smf.ols("y ~ egfr * C(diagnosis) + C(sex) + age", frame).fit()
    name = "egfr:C(diagnosis)[T.PD]"
    row = _rows(table, 1, "difference").loc["PD vs Control"]

    assert row["estimate"] == pytest.approx(fit.params[name])
    assert row["ci_low"] == pytest.approx(fit.conf_int().loc[name, 0])
    assert row["p_parametric"] == pytest.approx(fit.pvalues[name])
    assert row["covariates"] == "egfr + diagnosis + sex + age"
    assert row["n"] == 40
    assert np.isnan(row["standardised"]) and np.isnan(row["r_unadjusted"])
    assert row[CORRELATION_CIS].isna().all()


def test_a_planted_group_specific_slope_is_found() -> None:
    table, _, _ = _interactions([("egfr", "diagnosis")], n_permutations=2000)
    table = add_interaction_fdr(table)
    slopes, difference = _rows(table, 1, "slope"), _rows(table, 1, "difference")

    assert slopes.loc["slope in PD", "significant"]
    assert not slopes.loc["slope in Control", "significant"]
    assert difference.loc["PD vs Control", "significant"]
    assert not table.loc[table["component"] == 3, "significant"].any()


def test_a_variable_constant_in_one_group_gives_only_the_other_slope() -> None:
    table, info, coefficients = _interactions([("duration", "diagnosis")])
    clinical = clinical_effects(
        coefficients,
        info,
        ["duration"],
        PRIMARY,
        CATEGORICAL,
        500,
    )
    slopes = _rows(table, 2, "slope")

    control = slopes.loc["slope in Control"]
    assert np.isnan(control["estimate"])
    assert "duration is constant in Control" in control["note"]
    assert _rows(table, 2, "difference").empty
    expected = clinical[clinical["component"] == 2].iloc[0]
    assert slopes.loc["slope in PD", "estimate"] == pytest.approx(expected["estimate"])
    assert slopes.loc["slope in PD", "p_perm"] == pytest.approx(expected["p_perm"])


def test_three_levels_give_a_slope_each_and_one_shared_difference_test() -> None:
    primary = [*PRIMARY, "apoe"]
    table, _, _ = _interactions([("egfr", "apoe")], primary)
    table = add_interaction_fdr(table)
    slopes, differences = _rows(table, 1, "slope"), _rows(table, 1, "difference")

    assert list(slopes.index) == ["slope in 2", "slope in 3", "slope in 4"]
    assert list(differences.index) == ["3 vs 2", "4 vs 2"]
    for column in ("p_perm", "p_fdr", "partial_r2"):
        assert differences[column].nunique() == 1
    assert set(differences["covariates"]) == {"egfr + diagnosis + sex + age + apoe"}


def test_a_level_with_too_few_subjects_has_no_slope() -> None:
    info, coefficients = _interaction_study()
    info["apoe"] = np.resize([3, 4], len(info))
    info.loc[[0, 1], "apoe"] = 2
    table = interaction_effects(
        coefficients,
        info,
        [("egfr", "apoe")],
        [*PRIMARY, "apoe"],
        CATEGORICAL,
        200,
    )
    slopes = _rows(table, 1, "slope")

    assert np.isnan(slopes.loc["slope in 2", "estimate"])
    assert "fewer than 3 subjects" in slopes.loc["slope in 2", "note"]
    assert slopes.loc[["slope in 3", "slope in 4"], "estimate"].notna().all()
    assert _rows(table, 1, "difference").empty


def test_a_pair_whose_group_has_one_level_is_skipped() -> None:
    info, coefficients = _interaction_study()
    info.loc[info["diagnosis"] == "Control", "duration"] = np.nan
    table = interaction_effects(
        coefficients,
        info,
        [("duration", "diagnosis")],
        PRIMARY,
        CATEGORICAL,
        200,
    )

    assert table["estimate"].isna().all()
    assert table["note"].str.contains("'diagnosis' has one level").all()


def test_interaction_fdr_is_per_family() -> None:
    table, _, _ = _interactions([("egfr", "diagnosis"), ("duration", "diagnosis")])
    table = add_interaction_fdr(table)

    slopes = table[(table["test"] == "slope") & table["p_perm"].notna()]
    assert slopes["p_fdr"].to_numpy() == pytest.approx(
        multipletests(slopes["p_perm"], method="fdr_bh")[1],
    )
    differences = table[table["test"] == "difference"]
    assert differences["p_fdr"].to_numpy() == pytest.approx(
        multipletests(differences["p_perm"], method="fdr_bh")[1],
    )


def test_interactions_same_seed_same_output() -> None:
    first, _, _ = _interactions([("egfr", "diagnosis")], n_permutations=200)
    second, _, _ = _interactions([("egfr", "diagnosis")], n_permutations=200)
    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == INTERACTION_COLUMNS


def test_docs_explain_every_column() -> None:
    documented = set(re.findall(r"^### `([a-z_0-9]+)`", DOCS.read_text(), re.MULTILINE))
    assert documented == set(INTERACTION_COLUMNS) > set(COLUMNS)
