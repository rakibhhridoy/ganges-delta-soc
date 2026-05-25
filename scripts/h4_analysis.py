"""
H4 cross-delta transferability — definitive analysis.

Self-contained: reads h4_delta_samples.csv (4,504 complete-case top-metre profiles
across 14 global deltas, extracted from harmonised WoSIS profiles, Batjes 2024) plus
the in-house Ganges data. Predictor set = pH + clay + silt + sand (the variables
co-measured across deltas in WoSIS). SOC target in g/kg.

Produces:
  - transfer matrix (train delta i -> test delta j; diagonal = within-delta 5-fold CV)
  - forward (Ganges -> each delta) and reverse (pooled deltas -> Ganges) zero-shot
  - few-shot calibration curves; Leave-One-Delta-Out CV
  -> h4_results.json

Run:  python3 h4_analysis.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"   # bundle layout: reproducibility/{scripts,data}
PREDICTORS = ["ph", "clay", "silt", "sand"]


def gbr():
    return make_pipeline(StandardScaler(),
                         GradientBoostingRegressor(random_state=42, n_estimators=200))


def main():
    df = pd.read_csv(DATA / "h4_delta_samples.csv")
    data = {d: (g[PREDICTORS].values, g["soc"].values)
            for d, g in df.groupby("delta")}

    # Ganges (SOC concentration % -> g/kg to match WoSIS orgc)
    g = pd.read_csv(DATA / "GangesSOC.csv"); g.columns = [c.strip() for c in g.columns]
    gx = g[["pH", "Clay", "Silt", "Sand"]].rename(
        columns={"pH": "ph", "Clay": "clay", "Silt": "silt", "Sand": "sand"}).dropna()
    gy = g.loc[gx.index, "SOCC"].values * 10.0
    data["Ganges"] = (gx[PREDICTORS].values, gy)

    names = list(data.keys())
    counts = {k: int(len(v[1])) for k, v in data.items()}
    print("samples:", counts, "| total ext =", sum(v for k, v in counts.items() if k != "Ganges"))

    # ---- transfer matrix ----
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for tr in names:
        Xtr, ytr = data[tr]
        m = gbr().fit(Xtr, ytr)
        for te in names:
            Xte, yte = data[te]
            if tr == te:
                if len(yte) >= 10:
                    p = np.full(len(yte), np.nan)
                    for a, b in KFold(5, shuffle=True, random_state=42).split(Xte):
                        p[b] = gbr().fit(Xte[a], yte[a]).predict(Xte[b])
                    M.loc[tr, te] = r2_score(yte, p)
            else:
                M.loc[tr, te] = r2_score(yte, m.predict(Xte))
    M.to_csv(HERE / "h4_transfer_matrix.csv")

    diag = np.array([M.loc[d, d] for d in names if not pd.isna(M.loc[d, d])])
    off = M.values[~np.eye(len(names), dtype=bool)].astype(float)
    off = off[~np.isnan(off)]

    res = {"source": "Harmonised WoSIS profiles (Batjes 2024)",
           "predictors": PREDICTORS, "n_deltas_external": len(names) - 1,
           "n_total_external": int(sum(v for k, v in counts.items() if k != "Ganges")),
           "sample_counts": counts,
           "within_delta_median_R2": round(float(np.median(diag)), 3),
           "within_delta_range": [round(float(diag.min()), 2), round(float(diag.max()), 2)],
           "cross_delta_median_R2": round(float(np.median(off)), 3),
           "cross_delta_frac_positive": round(float((off > 0).mean()), 3),
           "cross_delta_n_pairs": int(len(off))}

    # ---- forward / reverse zero-shot ----
    gX, gY = data["Ganges"]
    gmodel = gbr().fit(gX, gY)
    res["forward_ganges_to_delta"] = {
        d: round(float(r2_score(data[d][1], gmodel.predict(data[d][0]))), 3)
        for d in names if d != "Ganges"}

    ext = [d for d in names if d != "Ganges"]
    PX = np.vstack([data[d][0] for d in ext]); PY = np.concatenate([data[d][1] for d in ext])
    rev = gbr().fit(PX, PY)
    res["reverse_pooled_to_ganges"] = {
        "train_n": int(len(PY)), "R2": round(float(r2_score(gY, rev.predict(gX))), 3),
        "r": round(float(np.corrcoef(gY, rev.predict(gX))[0, 1]), 3)}

    # reverse few-shot
    rng = np.random.default_rng(7); curve = {}
    for nloc in (0, 5, 10, 15, 20):
        r2s = []
        for _ in range(20):
            idx = rng.permutation(len(gY)); loc, rest = idx[:nloc], idx[nloc:]
            mm = rev if nloc == 0 else gbr().fit(np.vstack([PX, gX[loc]]),
                                                 np.concatenate([PY, gY[loc]]))
            r2s.append(r2_score(gY[rest], mm.predict(gX[rest])))
        curve[nloc] = round(float(np.mean(r2s)), 3)
    res["reverse_few_shot"] = curve

    # ---- LODO ----
    grp = np.concatenate([np.full(len(data[d][1]), i) for i, d in enumerate(ext)])
    lodo = {}
    for i, d in enumerate(ext):
        tr = grp != i; te = grp == i
        if te.sum() >= 5:
            lodo[d] = round(float(r2_score(PY[te], gbr().fit(PX[tr], PY[tr]).predict(PX[te]))), 3)
    res["lodo"] = lodo

    (HERE / "h4_results.json").write_text(json.dumps(res, indent=2))
    print("\n=== TRANSFER MATRIX (rows=train, cols=test) ===")
    print(M.round(2).to_string())
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in res.items() if k != "sample_counts"}, indent=2))


if __name__ == "__main__":
    main()
