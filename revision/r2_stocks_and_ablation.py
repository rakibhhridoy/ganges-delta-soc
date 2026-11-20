"""Re-check RC1 point 2 (stock accounting) and point 4 (predictor ablation) numbers from RC1_NOTES.txt."""
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"))

import numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
DATA = (Path(__file__).resolve().parent.parent / "data")
d = pd.read_csv(DATA / "GangesSOC.csv"); d.columns = [c.strip() for c in d.columns]
zmap = {}
for z in ["Tidal","Active","Mature","Moribund"]:
    t = pd.read_csv(DATA / f"{z}.csv"); t.columns = [c.strip() for c in t.columns]
    for L in t.Location.unique(): zmap[L] = z
d["zone"] = d.Location.map(zmap)
print("== Point 2 ==")
calc = d.SOCC * d["Bulk Density"] * 20
print(f"stock = SOCC x BD x 20: max abs diff {np.abs(calc - d['SOC Stock']).max():.4f}")
print(f"layer stock mean {d['SOC Stock'].mean():.1f}, range {d['SOC Stock'].min():.1f}-{d['SOC Stock'].max():.1f} t C/ha (per 20 cm)")
prof = d.groupby(["zone","Location"])["SOC Stock"].agg(["sum","count"])
print("layers per location:", prof["count"].unique())
print(f"0-100 cm mean {prof['sum'].mean():.1f}; by zone:", prof.groupby("zone")["sum"].mean().round(1).to_dict())
print("0-100 cm range", round(prof["sum"].min(),1), "-", round(prof["sum"].max(),1))
print("layer stock by depth:", d.groupby("Depth")["SOC Stock"].mean().round(1).to_dict())
br, fw = d[d.EC > 0.5]["SOC Stock"], d[d.EC <= 0.5]["SOC Stock"]
print(f"EC>0.5 vs <=0.5 layer ratio {br.mean()/fw.mean():.2f} (n {len(br)} layers, {d[d.EC>0.5].Location.nunique()} locations)")
rng = np.random.default_rng(42); locs = d.Location.unique(); rs = []
for _ in range(2000):
    s = rng.choice(locs, len(locs), replace=True); q = pd.concat([d[d.Location == L] for L in s])
    a, b = q[q.EC > 0.5]["SOC Stock"], q[q.EC <= 0.5]["SOC Stock"]
    if len(a) and len(b): rs.append(a.mean() / b.mean())
print(f"  location-bootstrap 95% CI {np.percentile(rs,2.5):.1f}-{np.percentile(rs,97.5):.1f}")

print("\n== Point 4 (stock target; location-grouped 5-fold as published, plus 100 random location partitions) ==")
feat = pd.read_csv(DATA / "features_preprocessed.csv"); feat = feat.drop(columns=[c for c in feat.columns if c.strip().upper()=="OM"])
y = pd.read_csv(DATA / "target.csv").iloc[:,0].values.astype(float); g = pd.factorize(d.Location)[0]
gbr = lambda: make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
def pooled(cols, folds):
    X = feat[cols].values; p = np.zeros(len(y))
    for i in np.unique(folds): tr, te = folds != i, folds == i; p[te] = gbr().fit(X[tr], y[tr]).predict(X[te])
    return r2_score(y, p), p
gk = np.zeros(len(y), int)
for i, (_, te) in enumerate(GroupKFold(5).split(feat, y, g)): gk[te] = i
def rf(seed):
    u = np.random.default_rng(seed).permutation(np.unique(g)); f = dict(zip(u, np.arange(20) % 5)); return np.array([f[x] for x in g])
allc = list(feat.columns); bd = [c for c in allc if c.strip() == "Bulk Density"]; tn = [c for c in allc if c.strip() == "TN"]
sets = {"all 14": allc, "without BD": [c for c in allc if c not in bd], "without BD and TN": [c for c in allc if c not in bd + tn],
        "coords + terrain only": [c for c in allc if c.strip() in ("Lat","Long","Elevation","Slope","Aspect")]}
for k, cols in sets.items():
    r0, _ = pooled(cols, gk); rl, _ = pooled(cols, g); rr = [pooled(cols, rf(s))[0] for s in range(100)]
    print(f"  {k:22s} GroupKFold {r0:.2f}  LOLO {rl:.2f}  random-partition mean {np.mean(rr):.2f}")
_, p = pooled(allc, gk); top = y >= np.percentile(y, 90)
print(f"  top decile observed mean {y[top].mean():.1f}, predicted {p[top].mean():.1f} (GroupKFold)")
