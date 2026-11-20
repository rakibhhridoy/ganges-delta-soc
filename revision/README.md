# Analyses added in response to the referees (v4)

These scripts reproduce the numbers in the authors' replies to the two referees on the public discussion of
egusphere-2026-4799 (RC1, RC2). They read only `../data/` and write to `outputs/`. Run each from this folder,
for example `python3 r1_same_target.py v4`. Random seed 42 throughout. Every model is gradient boosting
(scikit-learn GradientBoostingRegressor) with the standardiser fitted inside the training folds.

| Script | What it does | Referee point |
|---|---|---|
| `r1_same_target.py v4 v4_wosis_only polygon` | Cross-delta transfer scored against a local model and a mean baseline on the same held-out sites; sites (identical coordinates) are the CV groups | RC1-1, RC2 G1 |
| `r1_fewshot.py` | Local-only versus external-plus-local models given the same k local sites, scored on the same test sites | RC1-1 |
| `r1_bias_rmse.py` | Splits transfer error into mean offset (bias) and scatter (SDE); offset-corrected R² | RC2 G1 |
| `r2_site_level.py` | Table 1 tests at location level (n = 20 site means), mixed models with a location random intercept and zone fixed effect | RC1-3, RC2 G5 |
| `r2_stocks_and_ablation.py` | 0-100 cm profile stocks, Tidal/non-Tidal ratio with a location bootstrap, predictor ablation (with and without bulk density), high-SOC underprediction | RC1-2, RC1-4 |
| `r3_chloride_bound.py` | Worst-case Walkley-Black chloride bias from EC (all salt as NaCl); the laboratory added Ag₂SO₄, so this is an upper bound | RC1-3, RC2 G5 |
| `r4_ml_validation.py` | Validation schemes compared on identical inputs; 100 random location partitions; zone-stratified folds; engineered features with in-fold selection | RC1-4, RC2 G4 |
| `r4_bootstrap_ci.py` | Location-bootstrap 95 % interval for the location-grouped R² | RC1-4 |

The deep-learning comparison under repeated location partitions is in `../scripts/dl_reconstruction.py`
(v4: ensemble weights cross-fitted, so no test labels are used).

`r1_bias_rmse.py` and `r1_fewshot.py` import `r1_same_target.py` from this folder; the panel can be
switched with the environment variable `P1_PANEL` (default `v4`).
