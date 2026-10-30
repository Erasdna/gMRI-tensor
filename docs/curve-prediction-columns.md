# The curve-prediction table, column by column

`gmri predict` writes one table per model:
`<output-dir>/statistics/<model>__curve_prediction.csv`. When
`--subject-prefix` is set, `<prefix>_` comes first. This page covers each
column of that table: what it means and exactly how it is computed. It ends
with a worked example.

Two other pages cover the rest:

- what the models are for and how to compare them:
  [curve-prediction.md](curve-prediction.md);
- the command line: [cli.md](cli.md#gmri-predict).

Every number can be recomputed from three files:

- the model;
- the subject-information CSV;
- `predict.json` in the output directory, which records the options used,
  including `--seed`.

An empty cell means "not computed". The `note` column says why.

## 1. The rows

Each **outcome** (`--outcomes`) gets a block of rows. The block has one group
of rows per **covariate set**:

1. none;
2. each primary variable on its own, in the order given on the command line;
3. all primary variables together. With a single primary variable this set
   is the same as step 2, so it is not repeated.

Each group holds the **baseline** (the covariates alone) and then one
**profile model** per component (the covariates plus that component's time
profile). With 3 primary variables and a rank-3 model, an outcome has
5 sets × (1 + 3) = 20 rows.

## 2. What every row of an outcome shares

### 2.1 The subjects

The subjects of the model are matched to the `subjects` column of
`--subject-info`. Subjects without a row there are left out, and the terminal
lists them.

For each outcome, a subject is used only if they have all of:

- a value for the outcome;
- a value for **every** primary variable, even those a given model does not
  use;
- a value of every component's profile at **every** time point of the model.

Every row of the outcome therefore uses the same subjects, which makes the
rows comparable. Time points are matched by their value. A subject scanned at
23 h instead of 24 h lacks the 24 h value, and is left out.

### 2.2 The time profiles (the predictors)

A subject's profile for a component is the component's curve for that subject
(the evolving mode `B_i`), multiplied by the subject's coefficient. These are
the values in `<model>__time_coefficients.csv` and in the figures. For
subject `i`, component `r` and time point `t`:

```
profile[i, r, t] = B_i[t, r] / ‖B_i[:, r]‖  ×  c[i, r] / ‖c[:, r]‖
```

- The first factor is the subject's curve, scaled to unit length.
- The second factor is the subject coefficient `c`, scaled to unit length
  over all subjects.
- The subject coefficient `c` depends on the model:
  - **PARAFAC2:** the subject mode;
  - **CMF:** the length of the subject's curve, `c[i, r] = ‖B_i[:, r]‖`. The
    profile then simplifies to `B_i[t, r] / √(Σ_j ‖B_j[:, r]‖²)`.

Two consequences of this scaling:

- **The scaling does not change any score.** Multiplying a predictor by a
  constant changes its fitted coefficient, but not the predictions. The
  scaling only makes the profiles comparable with the figures.
- **A negligible component gives zero profiles.** If a component's
  coefficients are vanishingly small next to the other components', its
  profiles are set to zero. The row is then not scored (see `note`).

### 2.3 The design matrix

Each model is a linear regression. Its design matrix has one row per subject
and these columns:

```
[ 1 | covariates | profile(t_1) … profile(t_T) ]
```

- `1` is the intercept.
- **Numerical covariate:** its values, as they are.
- **Categorical covariate:** treatment coded. There is one 0/1 column per
  level except the reference level. The reference is the first level in
  sorted order, for example `Control`.
- **Profile:** one column per time point. A baseline has no profile columns.

The coding is set once for the outcome's whole sample.

### 2.4 Fitting

Every fit is ordinary least squares, `β = X⁺ y`, where `X⁺` is the
Moore–Penrose pseudoinverse of the design matrix `X`.

- When the columns are linearly independent, this is the usual least-squares
  fit.
- When they are not, it is the minimum-norm least-squares solution. Its
  predictions are still well defined. Two examples:
  - a covariate level that is missing from a training fold;
  - a time point where every subject's profile is 0.

### 2.5 The split into a training and a test part

The subjects are split once, with scikit-learn's `StratifiedShuffleSplit`:

- `test_size = --test-fraction` (default 0.2);
- `random_state = --seed`.

The sizes are:

```
n_test  = ⌈test_fraction × n⌉        n_train = n − n_test
```

The split keeps the shares of the levels of the stratification variable about
the same in both parts. That variable is `--stratify-by`, by default the first
`--primary-categorical` column.

If a level has fewer than 2 subjects, the split is not stratified
(`ShuffleSplit`), and `note` says so. Without any categorical primary
variable, the split is never stratified.

### 2.6 The cross-validation folds

The training subjects are cross-validated with scikit-learn's
`RepeatedStratifiedKFold`, stratified by the same variable:

- `n_splits = --cv-folds` (default 5);
- `n_repeats = --cv-repeats` (default 10);
- `random_state = --seed`.

Each repeat divides the training subjects into `k` groups (folds) in a new
way. If a level has fewer than `k` training subjects, the folds are not
stratified (`RepeatedKFold`), and `note` says so.

**Out-of-fold predictions.** Within one repeat, each fold in turn is predicted
by a model fitted (§2.4) on the other `k − 1` folds. Every training subject
thus gets exactly one prediction from a model that did not see them.

### 2.7 The permutations

`--n-permutations` (default 1000) random reorderings of the training
subjects are drawn from `--seed`. Every model of the outcome uses the same
reorderings.

A permutation p-value counts how often a permuted score reaches the real one:

```
p = (1 + number of permuted scores ≥ the real score) / (1 + number of permutations)
```

The smallest possible value is therefore `1 / (1 + N)`, which is 0.000999 for
1000 permutations. Ties within 10⁻¹⁰ count as reaching the real score.

### 2.8 The three scores

For observed outcomes `y_i` and predictions `ŷ_i` over `n` subjects:

```
R²   = 1 − Σ (y_i − ŷ_i)² / Σ (y_i − ȳ)²
RMSE = √( Σ (y_i − ŷ_i)² / n )
MAE  = Σ |y_i − ŷ_i| / n
```

- `ȳ` is the mean of the same `n` observed values.
- These are the definitions of scikit-learn's `r2_score`, the root of
  `mean_squared_error`, and `mean_absolute_error`.
- R² has no unit. RMSE and MAE are in the outcome's units.

## 3. Which model the row is

### `outcome`

The clinical column predicted, as named in `--outcomes`.

### `component`

The component whose profile the model uses, numbered as in
`<model>__time_coefficients.csv`. Empty for a baseline.

### `model`

The model's name, built from the covariates:

- **Baseline:** the covariates followed by `only`, for example
  `diagnosis only`. Without covariates it is `intercept only`.
- **Profile model:** the covariates followed by `+ curve`, for example
  `diagnosis + curve`. Without covariates it is `curve`.

### `covariates`

The covariates, joined with ` + `, for example `diagnosis + sex + age`. A
model without covariates shows `none`. The baseline and the profile models of
a set share this value.

### `curve`

`True` for a profile model, `False` for a baseline.

## 4. Subjects and parameters

### `n_train`

The number of subjects in the training part (§2.5). The same in every row of
an outcome.

### `n_test`

The number of subjects in the test part, `⌈test_fraction × n⌉`. The same in
every row of an outcome.

### `n_params`

The number of parameters the model can actually estimate: the rank of its
design matrix on the training part.

Normally this is the number of design columns:

- 1 for the intercept;
- 1 per numerical covariate;
- the number of levels minus 1 per categorical covariate;
- 1 per time point of the profile.

It is lower when columns are linearly dependent. An example is a time point
at which every subject's profile is 0: that column adds nothing.

The row is not scored when the smallest training fold of the
cross-validation has `n_params + 1` subjects or fewer. Such a fit is
(nearly) exact and predicts nothing. `note` then says so.

## 5. Cross-validated scores (training part)

Each repeat of the folds gives one set of out-of-fold predictions for all
`n_train` training subjects (§2.6). The three scores (§2.8) are computed
**over all of them together**, with `ȳ` the mean of all training
outcomes. This gives one value per repeat. The columns are the mean and the
standard deviation (`ddof = 1`) over the `--cv-repeats` repeats.

Scoring all subjects together, rather than fold by fold, matters because a
fold has only ~5 subjects. An R² on 5 subjects is unstable, and so would be
an average of such values.

### `cv_r2`

The mean over repeats of the out-of-fold R². This is the main score.

- 1 is a perfect prediction.
- **Below 0** means worse than predicting the training mean for everybody.
  This happens when a model fits noise.

The intercept-only model scores a little below 0. It predicts each held-out
subject with the mean of the other folds, not with the mean of all training
subjects that the denominator uses.

### `cv_r2_sd`

The standard deviation of the out-of-fold R² over the repeats. Empty when
`--cv-repeats` is 1.

It shows only how much the score depends on how the training subjects happen
to be grouped. It does **not** include the uncertainty from having this
particular set of subjects, so the real uncertainty of `cv_r2` is larger.
Whether a difference between models is real is tested by `p_gain`.

### `cv_rmse`

The mean over repeats of the out-of-fold root-mean-square error, in the
outcome's units: the typical size of a prediction error.

### `cv_rmse_sd`

The standard deviation of the out-of-fold RMSE over the repeats.

### `cv_mae`

The mean over repeats of the out-of-fold mean absolute error, in the
outcome's units. It is less influenced by a few large errors than
`cv_rmse`.

### `cv_mae_sd`

The standard deviation of the out-of-fold MAE over the repeats.

## 6. Test-set scores

The model is fitted once on all `n_train` training subjects (§2.4) and
predicts the `n_test` test subjects. The test subjects are used for nothing
else. These scores are computed from those `n_test` predictions alone.

### `test_r2`

R² on the test part (§2.8). `ȳ` is the **test subjects' own mean**, as in
scikit-learn's `r2_score`.

The intercept-only model's `test_r2` is therefore never above 0. Its single
prediction, the training mean, differs from the test mean.

Empty when there is only one test subject, or when all test subjects have
the same outcome. With ~6 test subjects this is a noisy number: read it as a
check that `cv_r2` holds up.

### `test_rmse`

The root-mean-square error of the test predictions, in the outcome's units.

### `test_mae`

The mean absolute error of the test predictions, in the outcome's units.

## 7. The fit on the training part

These come from the same fit as the test predictions: ordinary least squares
on all `n_train` training subjects, as computed by statsmodels `OLS`. Here
`n = n_train`, `k = n_params`, and `RSS` is the residual sum of squares of
that fit.

### `train_r2_adj`

The adjusted R² of the training fit:

```
R²_train = 1 − RSS / Σ (y_i − ȳ)²
adj. R²  = 1 − (1 − R²_train) × (n − 1) / (n − k)
```

It describes how well the model fits the subjects it was fitted on,
penalised for its number of parameters. It is 0 for the intercept-only
model.

It is usually higher than `cv_r2`. A large gap between the two means the
model overfits.

### `aic`

The Akaike information criterion of the training fit, with the Gaussian
log-likelihood:

```
AIC = n × ln(2π × RSS / n) + n + 2k
```

As in statsmodels, the residual variance is not counted in `k`. This shifts
every AIC by the same 2, which does not change differences.

**Lower is better.** Only differences between models of the **same outcome**
mean something, because those models share the training subjects. Rough
guide: a difference below ~2 is negligible, 4–7 is clear, and above 10 is
strong.

### `bic`

The Bayesian information criterion of the training fit:

```
BIC = n × ln(2π × RSS / n) + n + ln(n) × k
```

Read it as `aic`. It penalises each parameter by `ln(n)`, about 3.2 for 24
subjects, instead of 2. It therefore favours smaller models.

## 8. What the profile adds, and the tests

### `cv_r2_gain`

Profile models only:

```
cv_r2_gain = cv_r2(profile model) − cv_r2(baseline of the same covariate set)
```

Both models were scored on the same subjects and folds, so this is a paired
difference. Positive means the profile improves the prediction beyond the
covariates.

Read it together with the baseline's own `cv_r2`. A large gain over a poor
baseline can mean that the covariates hurt more than the profile helps.

### `p_perm`

Whether the model, covariates included, predicts better than chance. For
each permutation (§2.7):

1. the outcome values are reassigned among the training subjects;
2. the designs, the folds and everything else stay as they were;
3. the whole repeated cross-validation is rerun, giving a permuted `cv_r2`.

`p_perm` is the share of permutations whose `cv_r2` reaches the real one
(§2.7 formula). The test part is not used.

Its null hypothesis is that the outcome is unrelated to **all** of the
model's predictors. A profile model's `p_perm` can therefore be small because
of its covariates alone. Empty for `intercept only`, which has nothing to
test.

### `p_gain`

Profile models only. It tests whether the profile adds predictive value
beyond the covariates, with a Freedman–Lane permutation test:

```
fitted   = the baseline's fit of the outcome on all training subjects
residual = outcome − fitted
for each permutation π:
    outcome*  = fitted + residual reordered by π
    gain*     = cv_r2(profile model, outcome*) − cv_r2(baseline, outcome*)
p_gain = (1 + number of gain* ≥ cv_r2_gain) / (1 + number of permutations)
```

Only the part of the outcome that the covariates do not explain is shuffled.
The relation between the covariates and the outcome is kept, so the null
hypothesis is "the profile adds nothing beyond these covariates". The test
uses the same permutations as `p_perm`.

### `p_fdr`

`p_gain` corrected for multiple testing with the Benjamini–Hochberg false
discovery rate (statsmodels `multipletests`, `fdr_bh`). All profile models of
the table that have a `p_gain` form one family, across every outcome,
covariate set and component. With `m` such p-values sorted from smallest to
largest, `p_(1) ≤ … ≤ p_(m)`:

```
p_fdr of p_(i) = min over j ≥ i of  min(1, p_(j) × m / j)
```

Empty for baselines. Each `gmri predict` run is corrected on its own. Tables
of other models or other runs are not included.

### `significant`

`True` when `p_fdr` is below `--alpha` (default 0.05): the profile adds
predictive value beyond its covariates.

`False` for baselines and unscored rows, which were not tested.

## 9. Notes

### `note`

Empty, or one or more remarks separated by `; `. The remarks are:

| remark | applies to | consequence |
|---|---|---|
| `k subject(s) left out (no <outcome>, primary value or time point)` | every row of the outcome | §2.1; the scores use the others |
| `'<outcome>' is constant` | every row of the outcome | nothing scored |
| `n subjects, too few for k folds and a test part` | every row of the outcome | nothing scored; needs at least `2 × --cv-folds` subjects |
| `split not stratified …` | every row of the outcome | §2.5 |
| `folds not stratified: a level has fewer than k subjects` | every row of the outcome | §2.6 |
| `s subjects in the smallest training fold for k model parameters` | that row | not scored (`n_params`) |
| `component c's curve is zero` | that row | not scored (§2.2) |

## 10. Worked example

The example is a synthetic study, made so that the right answer is known:

- 30 subjects (15 Control, 15 PD), each scanned at 0, 6, 24 and 48 h;
- a rank-3 CMF model;
- primary variables diagnosis, sex and age.

The sleep score `psqi` was generated from component 1's profile at 24 h,
plus a small age effect and noise. Components 2 and 3 have nothing to do with
it. Component 1's curve is 0 at 0 h for every subject.

Here are the rows of two of the covariate sets: none, and all three.

| model | comp. | n_params | cv_r2 | cv_r2_sd | cv_rmse | test_r2 | train_r2_adj | aic | cv_r2_gain | p_perm | p_gain | p_fdr | significant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intercept only | | 1 | −0.09 | 0.08 | 1.98 | −0.92 | 0.00 | 100.7 | | | | | False |
| curve | 1 | 4 | 0.76 | 0.04 | 0.91 | 0.73 | 0.83 | 60.2 | 0.86 | 0.001 | 0.001 | 0.009 | True |
| curve | 2 | 5 | −0.65 | 0.22 | 2.42 | −0.91 | −0.11 | 106.7 | −0.55 | 0.87 | 0.89 | 1.00 | False |
| curve | 3 | 5 | −0.61 | 0.18 | 2.39 | −1.23 | −0.02 | 104.5 | −0.51 | 0.70 | 0.71 | 1.00 | False |
| diagnosis + sex + age only | | 4 | −0.54 | 0.19 | 2.34 | −0.90 | −0.15 | 106.6 | | 0.97 | | | False |
| diagnosis + sex + age + curve | 1 | 7 | 0.82 | 0.04 | 0.79 | 0.56 | 0.90 | 50.6 | 1.36 | 0.001 | 0.001 | 0.009 | True |
| diagnosis + sex + age + curve | 2 | 8 | −1.30 | 0.37 | 2.86 | −1.80 | −0.27 | 111.7 | −0.77 | 0.92 | 0.82 | 1.00 | False |
| diagnosis + sex + age + curve | 3 | 8 | −1.09 | 0.30 | 2.73 | −1.55 | −0.14 | 109.0 | −0.55 | 0.72 | 0.51 | 1.00 | False |

Every row has `n_train` = 24 and `n_test` = 6: ⌈0.2 × 30⌉ = 6 test
subjects.

### Reading the `intercept only` row

- `cv_r2` is −0.09, not 0. Each held-out subject is predicted with the mean of
  the other folds, not with the overall training mean (§5).
- `test_r2` is −0.92. The training mean of `psqi` is some way off the mean of
  the 6 test subjects, and an intercept-only model can never score above 0
  there.

### Reading component 1 without covariates (`curve`, 1)

- **`n_params` = 4, not 5.** The 0 h column is 0 for everybody, so the
  design has rank 4.
- **The model predicts well, and holds up.** `cv_r2` is 0.76 ± 0.04, and
  `test_r2` on the unseen subjects is 0.73.
- **The typical error halves.** `cv_rmse` is 0.91 psqi points, against
  1.98 for the intercept-only model.
- **AIC clearly prefers it.** Its AIC is 60.2, against 100.7 for the
  intercept-only model: a difference of 40.
- **The permutation tests are at their floor.** `p_perm` and `p_gain` are
  both 0.000999 = 1/1001: none of the 1000 permutations reached the real
  score.
- **`p_fdr` is 0.009.** The table has m = 45 profile models (3 outcomes ×
  5 sets × 3 components). The 5 smallest p-values are all 0.000999, so
  `p_fdr` = 0.000999 × 45 / 5 = 0.009.

### Reading components 2 and 3

- `cv_r2` is negative: worse than the mean. These models fit 4 noise
  predictors on ~19 subjects per training fold.
- `train_r2_adj` is about 0, and `p_gain` is large. The profiles add nothing.

### Reading the rows with all three covariates

- **The covariates alone do not predict `psqi`.** Their baseline has
  `cv_r2` −0.54, and its `p_perm` of 0.97 says no better than chance. The
  age effect is small, and the 3 extra parameters cost more than they bring.
- **The large gain mostly makes up for that poor baseline.** Component 1
  adds 1.36, but the model's own `cv_r2` of 0.82 is close to the 0.76 of
  the profile alone. The covariates add little.
- **Check the gain against the baseline's `cv_r2`.** Always read
  `cv_r2_gain` together with the baseline's own `cv_r2`.
