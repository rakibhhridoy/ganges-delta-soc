"""
Leak-free spatial cross-validation for the SOC ML models (addresses R1-13, R1-25, R3-13).

The 100 samples are 5 depths x 20 field locations (pseudoreplication). Random k-fold CV
splits same-location samples across train and test, leaking information and inflating R^2.
This script quantifies that optimism by comparing, for the same model and an in-fold
preprocessing pipeline (StandardScaler fit only on the training fold):

  * random 5-fold CV          (the leaky scheme used originally)
  * location-grouped 5-fold   (GroupKFold by location -- leak-free spatial)
  * Leave-Location-Out (LOLO) (20-fold, group = location)
  * Leave-Zone-Out (LZO)      (4-fold, group = geomorphological zone)

Pooled R^2 is computed over the concatenated held-out predictions (matching how the
manuscript reports CV R^2). Models: GBR (best traditional model) and two simple
baselines (EC-only OLS, full OLS).

Run:  python3 ml_spatial_cv.py   ->  ml_spatial_cv_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ---- data ------------------------------------------------------------------
feat = pd.read_csv(DATA / "features_preprocessed.csv")
y = pd.read_csv(DATA / "target.csv").iloc[:, 0].values.astype(float)
soc = pd.read_csv(DATA / "GangesSOC.csv"); soc.columns = [c.strip() for c in soc.columns]

loc = soc["Location"].values
loc_groups = pd.factorize(loc)[0]

zone_map = {}
for z, f in [("Tidal", "Tidal.csv"), ("Active", "Active.csv"),
             ("Mature", "Mature.csv"), ("Moribund", "Moribund.csv")]:
    zf = pd.read_csv(DATA / f); zf.columns = [c.strip() for c in zf.columns]
    for L in zf["Location"].unique():
        zone_map[L] = z
zone = np.array([zone_map.get(L, "?") for L in loc])
zone_groups = pd.factorize(zone)[0]

# Drop target-derived / leakage columns: OM (r=0.996 with SOC, per manuscript), and
# any SOC component if present. The manuscript states all models exclude OM.
LEAK = [c for c in feat.columns if c.strip().upper() in
        {"OM", "SOC", "SOCC", "SOCD", "SOC STOCK", "SOC_STOCK"}]
feat_model = feat.drop(columns=LEAK)
print(f"dropped leakage columns: {LEAK}  ->  {feat_model.shape[1]} predictors")

X_full = feat_model.values
X_ec = feat[["EC"]].values

# ---- helpers ---------------------------------------------------------------
def pooled_cv(make_model, X, y, splitter, groups=None):
    """Return pooled R^2 and RMSE over concatenated held-out predictions."""
    preds = np.full(len(y), np.nan)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        m = make_model()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return r2_score(y, preds), float(np.sqrt(mean_squared_error(y, preds)))

def gbr():  return make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
def ols():  return make_pipeline(StandardScaler(), LinearRegression())

random5 = KFold(n_splits=5, shuffle=True, random_state=42)
group5  = GroupKFold(n_splits=5)
logo    = LeaveOneGroupOut()

# ---- runs ------------------------------------------------------------------
rows = []
def add(model, scheme, r2, rmse, ngroups=""):
    rows.append({"model": model, "cv_scheme": scheme, "R2": round(r2, 3),
                 "RMSE": round(rmse, 1), "n_folds_or_groups": ngroups})

# GBR across all four schemes
r, e = pooled_cv(gbr, X_full, y, random5);                       add("GBR (79 feat)", "random 5-fold (leaky)", r, e, 5)
r, e = pooled_cv(gbr, X_full, y, group5, loc_groups);           add("GBR (79 feat)", "location-grouped 5-fold", r, e, 5)
r, e = pooled_cv(gbr, X_full, y, logo, loc_groups);             add("GBR (79 feat)", "Leave-Location-Out (LOLO)", r, e, 20)
r, e = pooled_cv(gbr, X_full, y, logo, zone_groups);            add("GBR (79 feat)", "Leave-Zone-Out (LZO)", r, e, 4)

# baselines under random vs grouped
r, e = pooled_cv(ols, X_ec, y, random5);                        add("EC-only OLS", "random 5-fold (leaky)", r, e, 5)
r, e = pooled_cv(ols, X_ec, y, group5, loc_groups);            add("EC-only OLS", "location-grouped 5-fold", r, e, 5)
r, e = pooled_cv(ols, X_full, y, random5);                      add("Full OLS", "random 5-fold (leaky)", r, e, 5)
r, e = pooled_cv(ols, X_full, y, group5, loc_groups);          add("Full OLS", "location-grouped 5-fold", r, e, 5)

res = pd.DataFrame(rows)
res.to_csv(Path(__file__).resolve().parent / "ml_spatial_cv_results.csv", index=False)

# ---- report ----------------------------------------------------------------
print(res.to_string(index=False))
gbr_rand = res[(res.model.str.startswith("GBR")) & (res.cv_scheme.str.contains("random"))]["R2"].iloc[0]
gbr_grp  = res[(res.model.str.startswith("GBR")) & (res.cv_scheme.str.contains("location-grouped"))]["R2"].iloc[0]
print(f"\nGBR optimism from pseudoreplication: random {gbr_rand:.3f} -> location-grouped {gbr_grp:.3f} "
      f"(drop {gbr_rand-gbr_grp:+.3f})")
print(f"n samples={len(y)}  n locations={len(np.unique(loc_groups))}  n zones={len(np.unique(zone_groups))}")
