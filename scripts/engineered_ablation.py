"""
Reconstruct the domain-informed engineered features and test their contribution
under leak-free (location-grouped) cross-validation. Replaces the unverifiable
MoE "+0.42" ablation (original DL code + engineered columns are gone) with a
reproducible delta-R^2 on models that can actually be re-run.

Engineered features are rebuilt from the Methods recipe (H1-informed interactions,
ratios, polynomial expansions, and log transforms; the Methods enumerate examples
rather than all 29, so this is a faithful reconstruction of the described
categories, documented here).

Run:  python3 engineered_ablation.py  ->  engineered_ablation_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

d = pd.read_csv(DATA / "GangesSOC.csv"); d.columns = [c.strip() for c in d.columns]
y = pd.read_csv(DATA / "target.csv").iloc[:, 0].values.astype(float)
assert np.allclose(np.sort(d["SOC Stock"].values), np.sort(y)), "row alignment check"
groups = pd.factorize(d["Location"].values)[0]

SOIL = ["Bulk Density", "Moisture", "Sand", "Silt", "Clay", "pH", "EC", "TN", "CEC"]
base = d[SOIL].copy()

# ---- reconstruct engineered features (Methods recipe) -----------------------
eps = 1e-6
eng = pd.DataFrame(index=d.index)
# interaction terms (H1-informed)
eng["EC_x_CEC"]      = d["EC"] * d["CEC"]
eng["Clay_x_Moist"]  = d["Clay"] * d["Moisture"]
eng["pH_x_EC"]       = d["pH"] * d["EC"]
eng["CEC_x_pH"]      = d["CEC"] * d["pH"]
eng["EC_x_Moist"]    = d["EC"] * d["Moisture"]
eng["Clay_x_CEC"]    = d["Clay"] * d["CEC"]
eng["Moist_x_CEC"]   = d["Moisture"] * d["CEC"]
# ratio features
eng["TN_over_EC"]    = d["TN"] / (d["EC"] + eps)
eng["EC_over_CEC"]   = d["EC"] / (d["CEC"] + eps)
eng["Sand_over_Clay"]= d["Sand"] / (d["Clay"] + eps)
eng["TN_over_CEC"]   = d["TN"] / (d["CEC"] + eps)
eng["Clay_over_Sand"]= d["Clay"] / (d["Sand"] + eps)
eng["Silt_over_Clay"]= d["Silt"] / (d["Clay"] + eps)
# polynomial expansions
eng["EC2"]  = d["EC"] ** 2
eng["EC3"]  = d["EC"] ** 3
eng["TN2"]  = d["TN"] ** 2
eng["CEC2"] = d["CEC"] ** 2
eng["Clay2"] = d["Clay"] ** 2
eng["Moist2"] = d["Moisture"] ** 2
# log transforms (per H4.3 top features: log(EC), log(CEC))
eng["log_EC"]  = np.log(d["EC"] + eps)
eng["log_CEC"] = np.log(d["CEC"] + eps)
eng["log_TN"]  = np.log1p(d["TN"])
eng["log_Clay"]= np.log1p(d["Clay"])
print(f"reconstructed {eng.shape[1]} engineered features")

# full (78) feature matrix from preprocessed (minus OM), for the full-model test
feat = pd.read_csv(DATA / "features_preprocessed.csv")
full = feat.drop(columns=[c for c in feat.columns if c.strip().upper() == "OM"]).reset_index(drop=True)

# ---- leak-free location-grouped pooled R^2 ---------------------------------
g5 = GroupKFold(5)
def pooled(make_model, X):
    preds = np.full(len(y), np.nan)
    for tr, te in g5.split(X, y, groups):
        m = make_model(); m.fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
    return r2_score(y, preds)

def gbr(): return make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
def mlp(): return make_pipeline(StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True,
                         random_state=42))

base_X      = base.values
base_eng_X  = pd.concat([base, eng], axis=1).values
full_X      = full.values
full_eng_X  = np.hstack([full.values, eng.values])

rows = []
for name, mk in [("GBR", gbr), ("MLP", mlp)]:
    b   = pooled(mk, base_X)
    be  = pooled(mk, base_eng_X)
    f   = pooled(mk, full_X)
    fe  = pooled(mk, full_eng_X)
    rows.append({"model": name, "soil9": round(b,3), "soil9+eng": round(be,3),
                 "dR2_eng_on_soil": round(be-b,3),
                 "full78": round(f,3), "full78+eng": round(fe,3),
                 "dR2_eng_on_full": round(fe-f,3)})

res = pd.DataFrame(rows)
res.to_csv(Path(__file__).resolve().parent / "engineered_ablation_results.csv", index=False)
print(res.to_string(index=False))
