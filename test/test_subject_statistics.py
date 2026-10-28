"""Covariate-adjusted subject-coefficient tests (`gMRItensor.subject_statistics`)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from gMRItensor.subject_statistics import add_fdr
from gMRItensor.subject_statistics import clinical_effects
from gMRItensor.subject_statistics import COLUMNS
from gMRItensor.subject_statistics import design_columns
from gMRItensor.subject_statistics import freedman_lane
from gMRItensor.subject_statistics import generalised_vif
from gMRItensor.subject_statistics import permutations
from gMRItensor.subject_statistics import primary_effects
from statsmodels.stats.outliers_influence import variance_inflation_factor

DOCS = Path(__file__).parents[1] / "docs" / "subject-statistics.md"
PRIMARY = ["diagnosis", "sex", "age"]
CATEGORICAL = {"diagnosis", "sex", "apoe"}


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


def test_docs_explain_every_column() -> None:
    documented = set(re.findall(r"^### `([a-z_0-9]+)`", DOCS.read_text(), re.MULTILINE))
    assert documented == set(COLUMNS)
