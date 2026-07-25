"""
H4 robustness check — delta delineation by geomorphological polygon vs bounding box.

A reviewer may ask whether the cross-delta transfer failure (main text Fig. 5,
Table S6) is an artefact of the rectangular bounding boxes used to delineate each
delta, which can admit non-deltaic upland soils. To test this we re-extracted the
external panel from WoSIS using the published active-delta polygons of Nienhuis
et al. (2020, Nature; the 100 largest global deltas), buffered outward by 0.5 deg.
Senegal is not among the 100 mapped deltas (a 0.5-deg-buffered bounding box was
used instead); the Yellow River polygon retained n<10 (below the within-delta CV
threshold). The BIS Nederland rows augmenting Rhine-Meuse were re-clipped to the
buffered Rhine-Meuse polygon.

This script reads the polygon-extracted panel (data/h4_delta_samples_POLY.csv,
shipped) plus the in-house Ganges data and reproduces the transfer summary
reported in Supplementary Table S11. The polygons are shipped as
data/delta_polygons.geojson; the extraction from raw WoSIS is documented in
provenance/polygon_extraction.py.

Result: cross-delta median R2 = -0.66 (5.2% of 210 pairs > 0), reverse
pooled->Ganges R2 = -0.74, LODO negative for 13/14 deltas; within-delta median
attenuates to +0.05. The cross-delta transfer failure is therefore robust to how
deltas are delineated.

Run:  python3 h4_polygon_robustness.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
PREDICTORS = ["ph", "clay", "silt", "sand"]


def gbr():
    return make_pipeline(StandardScaler(),
                         GradientBoostingRegressor(random_state=42, n_estimators=200))


def main():
    df = pd.read_csv(DATA / "h4_delta_samples_POLY.csv")
    data = {d: (g[PREDICTORS].values, g["soc"].values) for d, g in df.groupby("delta")}

    g = pd.read_csv(DATA / "GangesSOC.csv"); g.columns = [c.strip() for c in g.columns]
    gx = g[["pH", "Clay", "Silt", "Sand"]].rename(
        columns={"pH": "ph", "Clay": "clay", "Silt": "silt", "Sand": "sand"}).dropna()
    data["Ganges"] = (gx[PREDICTORS].values, g.loc[gx.index, "SOCC"].values * 10.0)

    names = list(data.keys())
    counts = {k: int(len(v[1])) for k, v in data.items()}

    # ---- transfer matrix ----
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for tr in names:
        Xtr, ytr = data[tr]; m = gbr().fit(Xtr, ytr)
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
    M.to_csv(HERE / "h4_transfer_matrix_POLY.csv")

    diag = np.array([M.loc[d, d] for d in names if not pd.isna(M.loc[d, d])])
    off = M.values[~np.eye(len(names), dtype=bool)].astype(float); off = off[~np.isnan(off)]

    ext = [d for d in names if d != "Ganges"]
    gX, gY = data["Ganges"]
    PX = np.vstack([data[d][0] for d in ext]); PY = np.concatenate([data[d][1] for d in ext])
    rev = gbr().fit(PX, PY)
    grp = np.concatenate([np.full(len(data[d][1]), i) for i, d in enumerate(ext)])
    lodo = {}
    for i, d in enumerate(ext):
        tr = grp != i; te = grp == i
        if te.sum() >= 5:
            lodo[d] = round(float(r2_score(PY[te], gbr().fit(PX[tr], PY[tr]).predict(PX[te]))), 3)

    res = {"delineation": "0.5deg-buffered active-delta polygons (Nienhuis et al. 2020); "
                          "Senegal = buffered bounding box (not in 100-delta set)",
           "n_total_external": int(sum(v for k, v in counts.items() if k != "Ganges")),
           "sample_counts": counts,
           "within_delta_median_R2": round(float(np.median(diag)), 3),
           "cross_delta_median_R2": round(float(np.median(off)), 3),
           "cross_delta_frac_positive": round(float((off > 0).mean()), 3),
           "cross_delta_n_pairs": int(len(off)),
           "reverse_pooled_to_ganges_R2": round(float(r2_score(gY, rev.predict(gX))), 3),
           "lodo_negative_of_14": int(sum(v < 0 for v in lodo.values())),
           "lodo": lodo}
    (HERE / "h4_polygon_robustness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "sample_counts"}, indent=2))


if __name__ == "__main__":
    main()
