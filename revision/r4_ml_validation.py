"""RC2: (1) why LOLO R2 0.87 vs location-grouped 5-fold 0.64; (2) engineered features with selection
inside the folds. Same data, model (GBR default, rs 42, in-fold scaling) and pooled-R2 definition as the
shipped ml_spatial_cv.py / engineered_ablation.py. Read-only."""
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"))

import numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
DATA = (Path(__file__).resolve().parent.parent / "data")
feat = pd.read_csv(DATA / "features_preprocessed.csv")
X14 = feat.drop(columns=[c for c in feat.columns if c.strip().upper() == "OM"]).values
y = pd.read_csv(DATA / "target.csv").iloc[:, 0].values.astype(float)
d = pd.read_csv(DATA / "GangesSOC.csv"); d.columns = [c.strip() for c in d.columns]
assert np.allclose(d["SOC Stock"].values, y), "row order"
loc = d.Location.values; g = pd.factorize(loc)[0]
zmap = {}
for z in ["Tidal", "Active", "Mature", "Moribund"]:
    t = pd.read_csv(DATA / f"{z}.csv"); t.columns = [c.strip() for c in t.columns]
    for L in t.Location.unique(): zmap[L] = z
zone = np.array([zmap[L] for L in loc])
eps = 1e-6; e = pd.DataFrame(index=d.index)
for a, b in [("EC","CEC"),("Clay","Moisture"),("pH","EC"),("CEC","pH"),("EC","Moisture"),("Clay","CEC"),("Moisture","CEC")]: e[f"{a}x{b}"] = d[a]*d[b]
for a, b in [("TN","EC"),("EC","CEC"),("Sand","Clay"),("TN","CEC"),("Clay","Sand"),("Silt","Clay")]: e[f"{a}/{b}"] = d[a]/(d[b]+eps)
for c, p in [("EC",2),("EC",3),("TN",2),("CEC",2),("Clay",2),("Moisture",2)]: e[f"{c}^{p}"] = d[c]**p
e["logEC"] = np.log(d.EC+eps); e["logCEC"] = np.log(d.CEC+eps); e["logTN"] = np.log1p(d.TN); e["logClay"] = np.log1p(d.Clay)
X37 = np.hstack([X14, e.values])
gbr = lambda: make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
sel = lambda k: (lambda: make_pipeline(StandardScaler(), SelectKBest(f_regression, k=k), GradientBoostingRegressor(random_state=42)))
def pooled(mk, X, folds):
    p = np.zeros(len(y))
    for i in np.unique(folds):
        tr, te = folds != i, folds == i; p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    return r2_score(y, p), p
def rand_folds(k, seed):
    u = np.random.default_rng(seed).permutation(np.unique(g)); f = dict(zip(u, np.arange(len(u)) % k)); return np.array([f[x] for x in g])

print("== (1) LOLO vs location-grouped 5-fold ==")
gk = np.zeros(len(y), int)
for i, (_, te) in enumerate(GroupKFold(5).split(X14, y, g)): gk[te] = i
r_gk, p_gk = pooled(gbr, X14, gk); r_lo, p_lo = pooled(gbr, X14, g)
print(f"reproduced: GroupKFold(5) pooled R2 {r_gk:.3f}   LOLO pooled R2 {r_lo:.3f}")
print("published GroupKFold folds, by zone (locations per fold):")
for i in range(5):
    m = gk == i; zs = pd.Series(zone[m]).groupby(loc[m]).first().value_counts().to_dict()
    print(f"  fold {i}: {zs}   fold R2 {r2_score(y[m], p_gk[m]):+.2f}")
tid = zone == "Tidal"
print(f"Tidal-row error: GroupKFold MAE {np.abs(p_gk-y)[tid].mean():.1f}, LOLO MAE {np.abs(p_lo-y)[tid].mean():.1f} t C/ha;"
      f" non-Tidal {np.abs(p_gk-y)[~tid].mean():.1f} vs {np.abs(p_lo-y)[~tid].mean():.1f}")
print("random location partitions, pooled R2 by number of folds (100 partitions each; k=20 is LOLO):")
for k in [4, 5, 10]:
    rs = [pooled(gbr, X14, rand_folds(k, s))[0] for s in range(100)]
    print(f"  k={k:2d}: mean {np.mean(rs):.3f}  median {np.median(rs):.3f}  5-95% [{np.percentile(rs,5):.2f}, {np.percentile(rs,95):.2f}]"
          f"  published-0.64-or-lower in {100*np.mean(np.array(rs)<=r_gk+1e-9):.0f} %")
# stratified by zone: each fold gets one location per zone (4 folds can't hold 5 per zone evenly -> 5 folds x 4 zones)
def strat_folds(seed):
    rng = np.random.default_rng(seed); f = {}
    for z in ["Tidal","Active","Mature","Moribund"]:
        Ls = rng.permutation(np.unique(g[zone == z]))
        for i, L in enumerate(Ls): f[L] = i
    return np.array([f[x] for x in g])
rs = [pooled(gbr, X14, strat_folds(s))[0] for s in range(100)]
print(f"  zone-stratified 5-fold (one location per zone per fold): mean {np.mean(rs):.3f}  5-95% [{np.percentile(rs,5):.2f}, {np.percentile(rs,95):.2f}]")

print("\n== (2) engineered features, selection inside folds (zone-stratified 5-fold, 100 partitions) ==")
cfg = {"base 14": (gbr, X14), "base 14 + 23 eng (no selection)": (gbr, X37),
       "37, top-14 selected in fold": (sel(14), X37), "37, top-7 selected in fold": (sel(7), X37),
       "14, top-7 selected in fold": (sel(7), X14)}
out = {k: [] for k in cfg}
for s in range(100):
    f = strat_folds(s)
    for k, (mk, X) in cfg.items(): out[k].append(pooled(mk, X, f)[0])
b = np.array(out["base 14"])
for k, v in out.items():
    v = np.array(v); dv = v - b
    print(f"  {k:34s} mean R2 {v.mean():.3f}  5-95% [{np.percentile(v,5):.2f}, {np.percentile(v,95):.2f}]   vs base: mean {dv.mean():+.3f}, better in {100*np.mean(dv>0):.0f} %")
# baselines
zm = np.zeros(len(y)); f = strat_folds(0)
rz = []
for s in range(100):
    f = strat_folds(s); p = np.zeros(len(y))
    for i in range(5):
        tr, te = f != i, f == i
        means = pd.Series(y[tr]).groupby(zone[tr]).mean(); p[te] = [means[z] for z in zone[te]]
    rz.append(r2_score(y, p))
re_ = [pooled(lambda: make_pipeline(StandardScaler(), LinearRegression()), d[["EC"]].values, strat_folds(s))[0] for s in range(100)]
print(f"  baseline zone mean: mean R2 {np.mean(rz):.3f}   EC-only OLS: {np.mean(re_):.3f}")
