# Predicting clinical variables from the time profiles

`gmri predict` asks how well a component's **time profile** predicts a
clinical measurement, such as kidney clearance (eGFR) or a sleep score. It is
for PARAFAC2 and CMF models, where every subject has their own time course
per component. CP models have one time course shared by all subjects, so
their profiles add nothing to the subject coefficients: test those with
`gmri stats` ([subject-statistics.md](subject-statistics.md)).

It writes one table, `<output-dir>/statistics/<model>__curve_prediction.csv`.
This page explains how the models are built and scored, how to compare them,
and what to watch out for. Every column, and exactly how it is computed, is
in [curve-prediction-columns.md](curve-prediction-columns.md); the command
line is described in [cli.md](cli.md#gmri-predict).

## 1. What is predicted, and from what

A subject's **time profile** for a component is the component's curve for
that subject, multiplied by the subject's coefficient: the values in
`<model>__time_coefficients.csv` and in the figures. With 4 scans, each
subject has 4 numbers per component, one per time point. These are the
predictors of one model.

Each **outcome** (`--outcomes`) is a numerical clinical column. Each
component is tried on its own, and every model is an ordinary linear
regression:

```
outcome ~ covariates + profile at 0 h + profile at 6 h + profile at 24 h + profile at 48 h
```

## 2. The models compared

The primary variables (`--primary-categorical`, `--primary-numerical`; for
example diagnosis, sex and age) can explain part of an outcome on their own.
To see what the profile adds, every outcome gets these **covariate sets**:

- none;
- each primary variable on its own;
- all primary variables together.

For each set there are two kinds of model:

| model | example name | answers |
|---|---|---|
| baseline | `diagnosis only` (`intercept only` for none) | how well the covariates alone predict the outcome |
| profile model | `diagnosis + curve` (`curve` for none) | how well the covariates and one component's profile predict it |

The difference between the two is what the profile adds. With 3 primary
variables and a rank-3 model there are 5 sets × (1 baseline + 3 components)
= 20 rows per outcome.

## 3. How the models are scored

1. **Split.** The subjects are split once into a **training part** (80 %,
   `--test-fraction`) and a **test part** (20 %). The split keeps the
   proportions of `--stratify-by` (default: the first primary categorical
   variable, e.g. diagnosis) about equal in both parts.
2. **Cross-validation on the training part.** The training subjects are
   divided into 5 groups (`--cv-folds`). Each group in turn is predicted by a
   model fitted on the other four, so every training subject gets one
   prediction from a model that never saw them. This is repeated 10 times
   with different groupings (`--cv-repeats`), giving a mean and a spread.
3. **Final test.** Each model is fitted once on the whole training part and
   used to predict the test subjects. These subjects were not used for
   anything else, so this is the most honest check, but with ~6 subjects it
   is a noisy one.
4. **Fit on the training part.** The same fit also gives AIC and BIC, which
   balance how well the model fits against how many parameters it uses.

Every model of an outcome uses the same subjects, the same split and the same
groupings, so their scores can be compared directly.

## 4. The columns

[curve-prediction-columns.md](curve-prediction-columns.md) explains every
column of the table: what it means, exactly how it is computed, and a worked
example. The ones to start with are `cv_r2` (with `cv_r2_sd`), `cv_r2_gain`,
`significant` and `test_r2`; the next section says how to use them.

## 5. How to compare the models

1. **Does the profile add anything?** Look at `significant` and
   `cv_r2_gain`. A significant row means the profile improves the prediction
   beyond those covariates, more than chance would.
2. **Beyond which covariates?** Compare the sets. A profile that adds a lot
   to `intercept only` but nothing to `diagnosis only` predicts the outcome
   only through diagnosis: it tells you which group a subject is in, and the
   group predicts the outcome.
3. **How well, in practice?** `cv_rmse` and `cv_mae` give the typical error
   in the outcome's units, which is often easier to judge than R².
4. **Does it hold up?** `test_r2` should be roughly in line with `cv_r2`. A
   much lower test score is a warning, but with ~6 test subjects it can also
   be bad luck.
5. **Simpler or richer?** AIC and BIC compare models of the same outcome on
   the training fit. When a profile model does not lower them clearly (by
   more than ~2), the simpler model is as good.

`p_perm` answers a weaker question: whether the model, covariates included,
predicts better than chance at all. A baseline such as `diagnosis only` can
have a small `p_perm` without any profile.

## 6. Pitfalls

- **Few subjects.** With ~30 subjects, each training group of the
  cross-validation has ~19, and a model with all covariates and a 4-point
  profile has ~8 parameters. Scores are noisy: rely on `cv_r2` with its
  `cv_r2_sd`, not on small differences.
- **Time points move together.** A profile's values at neighbouring time
  points are usually strongly correlated. The model as a whole is fine, but it
  cannot say which time point matters, which is why the table has no
  per-time-point coefficients.
- **Missing scans.** A subject missing any scan is left out of that outcome's
  models (see `note`). Many missing scans shrink the sample.
- **Sign and scale of a component.** A component's profile has an arbitrary
  sign and scale. This does not change the scores, but read the direction of
  a relation together with the component's figures.
- **Picking the best of many models.** The table has many rows. Reporting only
  the best-scoring one overstates how well it predicts; the test set guards
  against this only once, and `p_fdr` corrects the p-values for all rows.
- **Prediction is not explanation.** A profile that predicts an outcome is
  associated with it; that does not show that one causes the other.

## Methods (for a paper)

For PARAFAC2/CMF decompositions, each subject's time profile per component
(the component's subject-specific evolving curve scaled by the subject
coefficient) was used to predict clinical variables with ordinary least
squares. For each outcome and component, models with the profile values at
all time points were compared with baselines without them, for covariate sets
of no, each single and all primary variables (categorical variables treatment
coded). Subjects were split once into a training (80 %) and a test (20 %)
part, stratified by diagnosis. Models were evaluated on the training part by
repeated stratified 5-fold cross-validation (10 repeats; R², RMSE and MAE of
the pooled out-of-fold predictions, mean and SD over repeats), refitted on the
whole training part and scored once on the test part; AIC and BIC were taken
from the training fit. The added predictive value of a profile (the gain in
cross-validated R² over its baseline) was tested with 1,000 Freedman–Lane
permutations of the baseline residuals (Freedman & Lane 1983; Winkler et al.
2014), and p-values were corrected with the Benjamini–Hochberg procedure
across components, outcomes and covariate sets.

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate.
  *J. R. Stat. Soc. B* 57, 289–300.
- Freedman, D. & Lane, D. (1983). A nonstochastic interpretation of reported
  significance levels. *J. Bus. Econ. Stat.* 1, 292–298.
- Winkler, A. M. et al. (2014). Permutation inference for the general linear
  model. *NeuroImage* 92, 381–397.
