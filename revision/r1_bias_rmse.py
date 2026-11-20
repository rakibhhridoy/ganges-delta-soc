"""RC2: is negative cross-delta R2 just a mean offset? Corrected panel from RC1 Point 1 (spliced WoSIS rows
replaced, exact duplicates dropped), identical models (GBR 200 trees, rs 42, in-fold scaling, pH + texture).
Diagonal: site-grouped 5-fold, 5 repeats (3 for deltas > 2000 rows). Off-diagonal: model trained on the
whole source delta, scored on the whole target delta.
Per pair: R2, bias (mean pred - obs), RMSE, SDE = sqrt(RMSE^2 - bias^2), share of MSE from bias, and
R2 after subtracting the mean bias (offset-corrected), i.e. the best a constant shift could do."""
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"))

import sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.argv = ["x"]; warnings.filterwarnings("ignore")
import r1_same_target as P1
from sklearn.metrics import r2_score
import os; PANEL = os.environ.get("P1_PANEL", "v4")
D, _ = P1.panel(PANEL); names = sorted(D.delta.unique()); data = {t: D[D.delta == t] for t in names}
def mets(y, p):
    b = float(np.mean(p - y)); mse = float(np.mean((p - y) ** 2)); vy = float(np.var(y))
    return dict(R2=r2_score(y, p), bias=b, RMSE=mse ** .5, SDE=max(mse - b * b, 0) ** .5,
                bias_share=b * b / mse if mse > 0 else np.nan, R2_offset=1 - (mse - b * b) / vy, sd_obs=vy ** .5)
src = {t: P1.gb().fit(data[t][P1.P].values, data[t].soc.values) for t in names}
rows = []
for t in names:
    q = data[t]; X, y, S = q[P1.P].values, q.soc.values, q.site.values
    R = 3 if len(y) > 2000 else 5; acc = []
    for seed in range(R):
        f, k = P1.site_folds(S, 5, seed); p = np.zeros(len(y))
        for i in range(k): p[f == i] = P1.gb().fit(X[f != i], y[f != i]).predict(X[f == i])
        acc.append(mets(y, p))
    rows.append(dict(kind="local", source=t, target=t, n=len(y), **{k2: float(np.mean([a[k2] for a in acc])) for k2 in acc[0]}))
    for s in names:
        if s != t: rows.append(dict(kind="transfer", source=s, target=t, n=len(y), **mets(y, src[s].predict(X))))
T = pd.DataFrame(rows); T.to_csv(f"r1_bias_pairs_{PANEL}.csv", index=False)
L, X_ = T[T.kind == "local"], T[T.kind == "transfer"]
pd.set_option("display.width", 200)
print(f"deltas {len(names)}, transfer pairs {len(X_)}")
print("\nTRANSFER (off-diagonal), medians:")
print(f"  R2 {X_.R2.median():+.2f}   R2 after removing mean offset {X_.R2_offset.median():+.2f}")
print(f"  pairs R2 > 0: {100*(X_.R2>0).mean():.1f} %   offset-corrected R2 > 0: {100*(X_.R2_offset>0).mean():.1f} %")
print(f"  |bias| {X_.bias.abs().median():.1f} g/kg   RMSE {X_.RMSE.median():.1f}   SDE {X_.SDE.median():.1f}   bias share of MSE {X_.bias_share.median():.2f}")
print(f"  pairs where bias share > 0.5: {100*(X_.bias_share>0.5).mean():.1f} %")
print("\nLOCAL (diagonal, site-grouped), medians:")
print(f"  R2 {L.R2.median():+.2f}   offset-corrected {L.R2_offset.median():+.2f}   |bias| {L.bias.abs().median():.2f}   RMSE {L.RMSE.median():.1f}   SDE {L.SDE.median():.1f}")
print("\nRatio transfer / local for the same target (median over pairs): RMSE %.2f  SDE %.2f" % (
    np.median([r.RMSE / L[L.target == r.target].RMSE.iloc[0] for r in X_.itertuples()]),
    np.median([r.SDE / L[L.target == r.target].SDE.iloc[0] for r in X_.itertuples()])))
print("\nPer target delta:")
g = X_.groupby("target").agg(tr_R2=("R2", "median"), tr_R2_off=("R2_offset", "median"), tr_bias=("bias", "median"),
                             tr_RMSE=("RMSE", "median"), tr_SDE=("SDE", "median"), bias_share=("bias_share", "median"))
g = g.join(L.set_index("target")[["n", "R2", "R2_offset", "RMSE", "SDE", "sd_obs"]].rename(columns=lambda c: "loc_" + c))
print(g.round(2).to_string())
