# Reading the subject-coefficient statistics

`gmri stats` tests whether the **subject coefficients** of a decomposition
differ between groups of subjects or go along with clinical measurements. It
writes two tables to `<output-dir>/statistics/`:

- `<model>__subject_stats_primary.csv`: the primary variables (for example
  diagnosis, sex and age), each adjusted for the others.
- `<model>__subject_stats_clinical.csv`: each clinical variable (for example
  kidney clearance or a sleep score), adjusted for all primary variables.
  It is only written when clinical columns are given.

This page explains what is tested, every column of the two tables, which
p-value to use, and what to watch out for. The command line itself is
described in [cli.md](cli.md#gmri-stats).

## 1. What is tested

Every component of the decomposition gives each subject one number, its
subject coefficient: how strongly that component's pattern is present in the
subject. These are the values in `<model>__subject_coefficients.csv`. Each
component is analysed separately.

**Primary table.** For each component a linear model is fitted:

```
coefficient ~ diagnosis + sex + age
```

Each row tests one of these variables *with the others held fixed*. The
diagnosis row answers: "do PD and Control subjects of the same sex and age
differ in this component?". The age row answers: "does the coefficient change
with age, comparing subjects with the same diagnosis and sex?".

**Clinical table.** For each component and each clinical variable X, one model
is fitted:

```
coefficient ~ X + diagnosis + sex + age
```

The row tests X only. It answers: "beyond what diagnosis, sex and age already
explain, does the coefficient go along with X?".

**What "adjusted" means.** Suppose kidney clearance (eGFR) falls with age and
the coefficient rises with age. Then eGFR and the coefficient are correlated
even if eGFR has nothing to do with the component: older subjects have both a
lower eGFR and a higher coefficient. Comparing subjects *of the same age*
removes this: if eGFR then no longer matters, the association was due to age.
The `p_unadjusted` column shows the association before adjustment, so the
two can be compared.

**Categorical and numerical variables.** Every column is declared as
categorical (groups such as diagnosis, sex, APOE genotype) or numerical
(measurements such as age, eGFR, a sleep score). A categorical variable is
compared against its first level in sorted order, the *reference*:
"Control" before "PD", "F" before "M", 2 before 3 before 4. A variable with
three or more levels gets one row per level ("e3 vs e2", "e4 vs e2"), and one
overall test of whether any level differs.

**Missing values.** Each model uses the subjects who have every value it
needs, so `n` can differ between rows. A subject without an age, for example,
is left out of every model.

## 2. Column reference

The tables have one row per component and variable, or per component and
level for a categorical variable with three or more levels.

### `component`

The component number, as in the figures and the coefficient CSV. Component 1
is the largest.

### `term`

The variable tested in this row.

### `level`

For a categorical variable, the comparison in this row, such as
`PD vs Control`: the first group compared with the reference. Empty for a
numerical variable.

### `n`

The number of subjects in this model: those with a value for the tested
variable and for every covariate.

### `covariates`

The variables this row is adjusted for, such as `diagnosis + sex + age`. A
covariate that has only one value in this row's subjects cannot be adjusted
for and is left out (see `note`). Example: a motor score recorded only for PD
subjects leaves diagnosis out.

### `estimate`

The adjusted effect in coefficient units.

- Numerical variable: the change in the coefficient per unit of the variable
  (per year of age, per mL/min of eGFR), with the covariates fixed.
- Categorical variable: the adjusted difference between the group in `level`
  and the reference, for example PD minus Control.

Positive means a higher coefficient for higher values of the variable, or for
the group named first in `level`. The coefficients are scaled to unit length
per component, so their units are arbitrary: compare `standardised` across
components, not `estimate`.

### `ci_low`

Lower end of the 95% confidence interval of `estimate`.

### `ci_high`

Upper end of the 95% confidence interval of `estimate`. An interval that
excludes 0 corresponds to `p_parametric` < 0.05.

### `standardised`

The effect on a scale without units, comparable between components and
variables. `std_kind` says which scale is used.

- `beta` (numerical variables): the change in the coefficient, in standard
  deviations, per standard deviation of the variable. As a rough guide,
  0.1 is small, 0.3 medium and 0.5 large.
- `d` (categorical variables): the adjusted group difference divided by the
  residual standard deviation, a covariate-adjusted Cohen's d. As a rough
  guide, 0.2 is small, 0.5 medium and 0.8 large.

### `std_kind`

`beta` or `d`: which scale `standardised` is on.

### `partial_r2`

Of the variation in the coefficient that the covariates leave unexplained,
the share the tested variable explains (partial η²). For a categorical variable with
three or more levels it covers all levels together, so all its rows show the
same value. As a rough guide, 0.01 is small, 0.06 medium and 0.14 large.

### `partial_r`

For a numerical variable, the partial correlation: the correlation between
the variable and the coefficient once the covariates are accounted for, from
−1 to 1, with the sign of `estimate`. Its square is `partial_r2`. Empty for a
categorical variable.

### `vif`

How much the tested variable overlaps with the covariates (variance inflation
factor).

- 1 means no overlap.
- Above about 5, the variable is largely predicted by the covariates: it
  carries little information of its own, the estimate is uncertain and the
  test has little power. A non-significant result then says little.

For a categorical variable with three or more levels this is the generalised
VIF (Fox & Monette 1992), raised to the power 1/(number of levels − 1), so the
same rule of thumb applies.

### `r_unadjusted`

For a numerical variable, the plain (Pearson) correlation with the
coefficient, without any adjustment, in the same subjects. Compare it with
`partial_r` to see what the adjustment changed. Empty for a categorical
variable.

### `p_unadjusted`

The permutation p-value of the tested variable *without* any adjustment, in
the same subjects. It is only there for comparison with `p_perm`: a small
`p_unadjusted` with a large `p_perm` means the association is explained by
the covariates. Do not report it as a finding.

### `p_parametric`

The classical p-value of this row from the linear model (t-test), which
assumes normally distributed residuals. For a categorical variable with
three or more levels it tests this one level against the reference, whereas
`p_perm` tests the variable as a whole. It is a cross-check: if it differs a
lot from `p_perm`, the normality assumption is doubtful and `p_perm` is the
one to trust.

### `p_perm`

The p-value of the test, from 10,000 Freedman–Lane permutations by default.
It does not assume normally distributed data. For a categorical variable with
three or more levels it is one overall test of whether any level differs,
shared by all its rows. This is the evidence for one test on its own,
before correcting for the number of tests.

### `p_fdr`

`p_perm` corrected for the number of tests in this table (Benjamini–Hochberg
false discovery rate). The correction counts every component and variable in
the table, and a categorical variable once, however many levels it has. The
primary and clinical tables are corrected separately.

### `significant`

`True` when `p_fdr` is below `--alpha` (default 0.05). **This is the column to
decide on.**

### `note`

Empty, or why a row is incomplete:

- a covariate left out because it has one value in these subjects;
- a variable that could not be tested (only one value, a level with fewer
  than 2 subjects, too few subjects for the model, or completely determined
  by the covariates). These rows have no p-values and are not counted in the
  correction.

## 3. Which p-value to use

| column | use it for |
|---|---|
| `significant`, `p_fdr` | Deciding and reporting: corrected for all tests in the table. |
| `p_perm` | The evidence from one test on its own. |
| `p_parametric` | A cross-check of `p_perm`; a single level of a multi-level variable. |
| `p_unadjusted` | Only for seeing what the adjustment changed. |

Report effects with `standardised` (or `estimate` with `ci_low` and
`ci_high`) and `p_fdr`. A significant p-value with a tiny `standardised` is
rarely meaningful, and a large `standardised` that is not significant may be
worth a larger study.

## 4. Worked example

These tables come from made-up data for 42 subjects and 3 components. The
following effects were built in:

- component 1 is higher in PD;
- component 2 rises with age and with a sleep score (`psqi`);
- component 3 is noise;
- eGFR falls with age, and `psqi` is higher in PD;
- three subjects have no `psqi` value.

```
gmri stats --model out/cp_rank_3 --subject-info info.csv \
    --primary-categorical diagnosis sex --primary-numerical age \
    --clinical-categorical apoe --clinical-numerical egfr psqi
```

The tables are wide, so each one is shown below in three parts:

1. which test each row is;
2. how large the effect is;
3. how sure we can be.

The three parts are the same rows of one CSV, and `component`, `term` and
`level` are repeated in every part so the rows can be matched. Values are
rounded to 3 significant digits. An empty cell is empty in the CSV too: the
column does not apply to that row.

### Primary table: `cp_rank_3__subject_stats_primary.csv`

Part 1, which test:

```
component  term       level          n   covariates
1          diagnosis  PD vs Control  42  sex + age
1          sex        M vs F         42  diagnosis + age
1          age                       42  diagnosis + sex
2          diagnosis  PD vs Control  42  sex + age
2          sex        M vs F         42  diagnosis + age
2          age                       42  diagnosis + sex
3          diagnosis  PD vs Control  42  sex + age
3          sex        M vs F         42  diagnosis + age
3          age                       42  diagnosis + sex
```

Part 2, effect sizes:

```
component  term       level          estimate   ci_low     ci_high   standardised  std_kind  partial_r2  partial_r  vif
1          diagnosis  PD vs Control   0.0374     0.0163    0.0584     1.11         d         0.254                  1.00
1          sex        M vs F         -0.00773   -0.0292    0.0137    -0.23         d         0.0138                 1.03
1          age                       -0.000433  -0.00182   0.000957  -0.0889       beta      0.0104      -0.102     1.03
2          diagnosis  PD vs Control   0.0243     0.00352   0.0452     0.731        d         0.129                  1.00
2          sex        M vs F          0.000782  -0.0205    0.022      0.0235       d         0.000146               1.03
2          age                        0.00348    0.0021    0.00485    0.617        beta      0.408        0.639     1.03
3          diagnosis  PD vs Control   0.0185    -0.00256   0.0396     0.55         d         0.0768                 1.00
3          sex        M vs F          0.0223     0.000782  0.0438     0.661        d         0.104                  1.03
3          age                       -0.000938  -0.00233   0.000453  -0.2          beta      0.0467      -0.216     1.03
```

Part 3, p-values (`note` is empty in every row, so it is left out):

```
component  term       level          r_unadjusted  p_unadjusted  p_parametric  p_perm  p_fdr   significant
1          diagnosis  PD vs Control                0.0008        0.000923      0.001   0.0045  True
1          sex        M vs F                       0.485         0.471         0.474   0.596   False
1          age                       -0.0615       0.7           0.532         0.53    0.596   False
2          diagnosis  PD vs Control                0.0582        0.0231        0.023   0.069   False
2          sex        M vs F                       0.472         0.941         0.941   0.941   False
2          age                        0.621        0.0001        9.08e-06      0.0001  0.0009  True
3          diagnosis  PD vs Control                0.133         0.0833        0.0851  0.153   False
3          sex        M vs F                       0.0316        0.0426        0.0423  0.0952  False
3          age                       -0.249        0.114         0.18          0.183   0.274   False
```

### Clinical table: `cp_rank_3__subject_stats_clinical.csv`

Part 1, which test (every row is adjusted for diagnosis, sex and age):

```
component  term  level     n   covariates
1          egfr            42  diagnosis + sex + age
1          psqi            39  diagnosis + sex + age
1          apoe  e3 vs e2  42  diagnosis + sex + age
1          apoe  e4 vs e2  42  diagnosis + sex + age
2          egfr            42  diagnosis + sex + age
2          psqi            39  diagnosis + sex + age
2          apoe  e3 vs e2  42  diagnosis + sex + age
2          apoe  e4 vs e2  42  diagnosis + sex + age
3          egfr            42  diagnosis + sex + age
3          psqi            39  diagnosis + sex + age
3          apoe  e3 vs e2  42  diagnosis + sex + age
3          apoe  e4 vs e2  42  diagnosis + sex + age
```

Part 2, effect sizes:

```
component  term  level     estimate   ci_low     ci_high    standardised  std_kind  partial_r2  partial_r  vif
1          egfr            -0.000204  -0.00184    0.00143   -0.0528       beta      0.00172     -0.0415    2.21
1          psqi             0.00212   -0.00356    0.00781    0.138        beta      0.0166       0.129     1.58
1          apoe  e3 vs e2  -0.0124    -0.0477     0.0228    -0.365        d         0.0322                 1.05
1          apoe  e4 vs e2  -0.0196    -0.0566     0.0173    -0.577        d         0.0322                 1.05
2          egfr             0.000657  -0.000946   0.00226    0.147        beta      0.0183       0.135     2.21
2          psqi             0.0104     0.00614    0.0147     0.594        beta      0.419        0.647     1.58
2          apoe  e3 vs e2  -0.00315   -0.0384     0.0321    -0.0928       d         0.012                  1.05
2          apoe  e4 vs e2  -0.00984   -0.0468     0.0271    -0.29         d         0.012                  1.05
3          egfr             0.00113   -0.000462   0.00273    0.304        beta      0.053        0.23      2.21
3          psqi             0.00386   -0.000862   0.00859    0.302        beta      0.0751       0.274     1.58
3          apoe  e3 vs e2  -0.0184    -0.0521     0.0153    -0.566        d         0.118                  1.05
3          apoe  e4 vs e2  -0.0357    -0.071     -0.000407  -1.1          d         0.118                  1.05
```

Part 3, p-values (`note` is empty in every row, so it is left out):

```
component  term  level     r_unadjusted  p_unadjusted  p_parametric  p_perm  p_fdr   significant
1          egfr             0.0627       0.689         0.802         0.799   0.803   False
1          psqi             0.39         0.0147        0.454         0.447   0.671   False
1          apoe  e3 vs e2                0.798         0.48          0.55    0.707   False
1          apoe  e4 vs e2                0.798         0.289         0.55    0.707   False
2          egfr            -0.343        0.0264        0.411         0.406   0.671   False
2          psqi             0.484        0.0017        2e-05         0.0001  0.0009  True
2          apoe  e3 vs e2                0.684         0.857         0.803   0.803   False
2          apoe  e4 vs e2                0.684         0.592         0.803   0.803   False
3          egfr             0.391        0.0107        0.158         0.164   0.368   False
3          psqi             0.293        0.0671        0.106         0.106   0.32    False
3          apoe  e3 vs e2                0.271         0.275         0.107   0.32    False
3          apoe  e4 vs e2                0.271         0.0476        0.107   0.32    False
```

### Reading the effect sizes

Four rows from the tables above, one of each kind.

**A two-group variable: diagnosis on component 1 (primary table).**

```
level          estimate  ci_low  ci_high  standardised  std_kind  partial_r2  vif
PD vs Control  0.0374    0.0163  0.0584   1.11          d         0.254       1.00
```

- `estimate`: PD subjects have a component-1 coefficient 0.037 higher than
  Control subjects of the same sex and age.
- `ci_low`, `ci_high`: the true difference is plausibly between 0.016 and
  0.058. The interval excludes 0.
- `standardised` with `std_kind` = `d`: the difference is 1.11 residual
  standard deviations, a large effect (above 0.8).
- `partial_r2`: diagnosis explains 25% of the variation that sex and age leave
  unexplained, a large effect (above 0.14).
- `vif` = 1.00: diagnosis does not overlap with sex or age.
- Report as: *PD > Control, d = 1.11 (95% CI of the difference 0.016 to
  0.058), FDR p = 0.0045.*

**A numerical variable: age on component 2 (primary table).**

```
level  estimate  ci_low  ci_high  standardised  std_kind  partial_r2  partial_r  vif
       0.00348   0.0021  0.00485  0.617         beta      0.408       0.639      1.03
```

- `estimate`: the coefficient rises by 0.0035 per year of age, for subjects
  with the same diagnosis and sex.
- `standardised` with `std_kind` = `beta`: one standard deviation older (about 8
  years here) means 0.62 standard deviations higher, a large effect (above
  0.5).
- `partial_r` = 0.64: the correlation between age and the coefficient once
  diagnosis and sex are accounted for. Its square is `partial_r2` = 0.41.
- Report as: *β = 0.62, partial r = 0.64, FDR p = 0.0009.*

**A clinical variable: psqi on component 2 (clinical table).**

```
level  n   estimate  ci_low   ci_high  standardised  std_kind  partial_r2  partial_r  vif   r_unadjusted
       39  0.0104    0.00614  0.0147   0.594         beta      0.419       0.647      1.58  0.484
```

- `n` = 39: three subjects have no sleep score.
- `partial_r` = 0.65 against `r_unadjusted` = 0.48: the association is
  *stronger* once diagnosis, sex and age are accounted for, because age adds
  variation that hides it.
- `vif` = 1.58: the sleep score overlaps somewhat with the covariates
  (it is higher in PD), but not enough to matter.
- Report as: *β = 0.59, partial r = 0.65 adjusted for diagnosis, sex and age
  (n = 39), FDR p = 0.0009.*

**A variable with three levels: APOE on component 3 (clinical table).**

```
level     estimate  ci_low   ci_high    standardised  std_kind  partial_r2  vif
e3 vs e2  -0.0184   -0.0521   0.0153    -0.566        d         0.118       1.05
e4 vs e2  -0.0357   -0.071   -0.000407  -1.1          d         0.118       1.05
```

- Each row compares one genotype with the reference, e2: e4 carriers are
  1.1 residual standard deviations lower than e2 carriers, e3 carriers 0.57
  lower.
- `partial_r2` = 0.118 is shared: APOE as a whole explains 12% of the
  remaining variation. It belongs to the overall test, as does `p_perm`.
- `partial_r` and `r_unadjusted` are empty: they only exist for numerical
  variables.
- Here the overall test is not significant (`p_fdr` 0.32), so the large
  e4-vs-e2 `d` is not evidence on its own. See the last point below.

How to read the whole example:

- **The built-in effects are found.** Component 1 is higher in PD (d = 1.11,
  `significant`). Component 2 rises with age (β = 0.62), and with the sleep
  score beyond diagnosis, sex and age (partial r = 0.65, n = 39 because of
  the missing values).
- **Adjusting for diagnosis removes a misleading association.** On its own,
  the sleep score correlates with component 1 (`r_unadjusted` 0.39,
  `p_unadjusted` 0.015). That is only because PD subjects sleep worse *and*
  have a higher component 1. Among subjects with the same diagnosis there is
  no association (`p_perm` 0.45).
- **Adjusting for age removes a misleading association.** eGFR correlates
  with components 2 and 3, but only because it falls with age. Adjusted for
  age, nothing remains. `vif` = 2.2 shows that eGFR partly duplicates age.
- **The correction removes a chance finding.** Sex on component 3 has
  `p_perm` 0.042, but with 9 tests in the table `p_fdr` is 0.095. Component 3
  is pure noise, so this is the false positive the correction is for.
- **Total and direct effects differ.** Diagnosis on component 2 is borderline
  (`p_fdr` 0.069) in the primary table: PD subjects sleep worse, and sleep
  acts on component 2. The primary table shows the *total* effect of
  diagnosis, including any route through sleep; the clinical table shows
  what sleep adds *beyond* diagnosis.
- **One level and the whole variable can disagree.** For APOE on component 3,
  "e4 vs e2" alone has `p_parametric` 0.048. The overall test of APOE
  (`p_perm` 0.107) is the one to rely on: it does not pick the most extreme
  of several comparisons.

## 5. Pitfalls

- **Direction of a component.** A higher coefficient means more of that
  component's pattern. Read the sign of `estimate` together with the
  component's spatial and time modes. The units are arbitrary per component.
- **Adjusting for a consequence of diagnosis.** If a clinical variable is
  itself caused by the disease, adjusting for diagnosis removes much of its
  variation. A non-significant clinical row can then mean "nothing beyond
  diagnosis", not "no relation". A high `vif` warns of this.
- **Different subjects per row.** `n` can differ between rows and between the
  two tables, because each model uses the subjects with complete values.
- **Read the `note` column.** It says when a covariate was left out or a
  variable could not be tested.
- **Unequal spread between groups.** The permutation test assumes the
  residuals are comparable between subjects. If one group varies much more
  than another *and* the groups differ a lot in size, p-values can be
  somewhat too small.
- **Separate runs are corrected separately.** Tables for different ranks,
  models or `--subject-prefix` variable sets each have their own correction.
  Running many variants and reporting the best one inflates false positives.
- **Association, not causation.** A significant row shows that the
  coefficient goes along with the variable, adjusted for the covariates. It
  does not show that one causes the other.

## Methods (for a paper)

For each component, subject coefficients (scaled to unit norm per component)
were analysed with linear models. Primary variables were tested jointly, each
adjusted for the others. Each clinical variable was tested in a separate model
adjusted for all primary variables. Categorical variables were treatment
coded. Subjects with missing values were excluded per model. p-values were
obtained with the Freedman–Lane permutation procedure (Freedman & Lane 1983;
Winkler et al. 2014), using the partial F statistic and 10,000 permutations.
They were corrected for multiple comparisons with the Benjamini–Hochberg
procedure across all components and variables of a table. Effect sizes are
reported as standardised regression coefficients, covariate-adjusted Cohen's
d and partial η²; collinearity is reported as the (generalised) variance
inflation factor (Fox & Monette 1992).

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate.
  *J. R. Stat. Soc. B* 57, 289–300.
- Fox, J. & Monette, G. (1992). Generalized collinearity diagnostics.
  *J. Am. Stat. Assoc.* 87, 178–183.
- Freedman, D. & Lane, D. (1983). A nonstochastic interpretation of reported
  significance levels. *J. Bus. Econ. Stat.* 1, 292–298.
- Winkler, A. M. et al. (2014). Permutation inference for the general linear
  model. *NeuroImage* 92, 381–397.
