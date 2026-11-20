"""Location-bootstrap 95 % CI of the published location-grouped 5-fold R2 (0.64): resample the 20 locations
with replacement over the fixed out-of-fold predictions; 2000 resamples, seed 42."""
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"))

import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
DATA = (Path(__file__).resolve().parent.parent / "data")
f = pd.read_csv(DATA/"features_preprocessed.csv"); X = f.drop(columns=[c for c in f.columns if c.strip().upper()=="OM"]).values
y = pd.read_csv(DATA/"target.csv").iloc[:,0].values.astype(float)
d = pd.read_csv(DATA/"GangesSOC.csv"); d.columns=[c.strip() for c in d.columns]; g = pd.factorize(d.Location)[0]
p = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, g): p[te] = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42)).fit(X[tr], y[tr]).predict(X[te])
print(f"point estimate {r2_score(y,p):.3f}")
rng = np.random.default_rng(42); u = np.unique(g); rs = []
for _ in range(2000):
    idx = np.concatenate([np.where(g==L)[0] for L in rng.choice(u, len(u), replace=True)]); rs.append(r2_score(y[idx], p[idx]))
print(f"location bootstrap 95 % CI {np.percentile(rs,2.5):.2f} to {np.percentile(rs,97.5):.2f}; median {np.median(rs):.2f}")
