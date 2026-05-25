"""
H4 expanded analyses (2026-05-25), on the BIS-augmented 15-delta panel:

  (1) Distance-decay of transfer  — does cross-delta transfer R^2 worsen with
      geographic and feature-space (and latitudinal/climate-proxy) distance between
      deltas? Quantifies the "ecological heterogeneity" interpretation.
  (2) Feature-importance universality — per-delta GBR importance over pH/clay/silt/
      sand vs the Ganges ranking (Kendall tau, top-feature agreement). Shows *why*
      transfer fails: deltas weight the predictors differently.
  (3) Few-shot recovery curves for every delta — pooled(all other deltas) + n local
      samples -> R^2 on held-out local samples, n = {0,5,10,15,20}. Shows that local
      sampling is irreplaceable across the board, not just for the Ganges.

Reads h4_delta_samples.csv (+ ../reproducibility/data/GangesSOC.csv) and
h4_transfer_matrix.csv (for the transfer R^2 values). Writes h4_expanded_results.json
and h4_distance_decay.csv / h4_feature_importance.csv / h4_fewshot_curves.csv.

Run: python3 h4_expanded_analysis.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats

HERE = Path(__file__).resolve().parent
PRED = ["ph", "clay", "silt", "sand"]


def gbr():
    return make_pipeline(StandardScaler(),
                         GradientBoostingRegressor(random_state=42, n_estimators=200))


def load_panel():
    d = pd.read_csv(HERE.parent / "data" / "h4_delta_samples.csv")
    cols = ["delta", "lat", "lon", "soc"] + PRED
    d = d[[c for c in cols if c in d.columns]]
    g = pd.read_csv(HERE.parent / "data" / "GangesSOC.csv")
    g.columns = [c.strip() for c in g.columns]
    gx = g[["pH", "Clay", "Silt", "Sand", "Lat", "Long"]].rename(
        columns={"pH": "ph", "Clay": "clay", "Silt": "silt", "Sand": "sand",
                 "Lat": "lat", "Long": "lon"}).copy()
    gx["soc"] = g["SOCC"] * 10.0
    gx["delta"] = "Ganges"
    gx = gx.dropna(subset=["soc"] + PRED)
    return pd.concat([d, gx[["delta", "lat", "lon", "soc"] + PRED]], ignore_index=True)


def haversine(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dphi = np.radians(la2 - la1); dl = np.radians(lo2 - lo1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_decay(panel, M):
    deltas = [d for d in M.index if d in panel.delta.unique()]
    cen = {d: (panel[panel.delta == d].lat.mean(), panel[panel.delta == d].lon.mean()) for d in deltas}
    # standardized feature means per delta (feature-space domain distance)
    sc = StandardScaler().fit(panel[PRED].values)
    fmean = {d: sc.transform(panel[panel.delta == d][PRED].values).mean(axis=0) for d in deltas}
    rows = []
    for i in deltas:
        for j in deltas:
            if i == j or pd.isna(M.loc[i, j]):
                continue
            rows.append({
                "train": i, "test": j, "transfer_R2": float(M.loc[i, j]),
                "geo_km": haversine(*cen[i], *cen[j]),
                "feat_dist": float(np.linalg.norm(fmean[i] - fmean[j])),
                "abs_dlat": abs(cen[i][0] - cen[j][0]),
            })
    df = pd.DataFrame(rows)
    # Spearman vs transfer R^2 (rank-based: robust to the large negative R^2 tail)
    out = {}
    for k in ["geo_km", "feat_dist", "abs_dlat"]:
        rho, p = stats.spearmanr(df[k], df["transfer_R2"])
        out[k] = {"spearman_rho": round(float(rho), 3), "p": float(f"{p:.2e}")}
    df.to_csv(HERE / "h4_distance_decay.csv", index=False)
    return out, df


def feature_universality(panel):
    deltas = [d for d in panel.delta.unique() if (panel.delta == d).sum() >= 30]
    imp = {}
    for d in deltas:
        sub = panel[panel.delta == d]
        m = GradientBoostingRegressor(random_state=42, n_estimators=200).fit(sub[PRED].values, sub["soc"].values)
        imp[d] = pd.Series(m.feature_importances_, index=PRED)
    gi = imp["Ganges"]
    rows = []
    for d in deltas:
        if d == "Ganges":
            continue
        tau, p = stats.kendalltau(gi.values, imp[d].values)
        rows.append({"delta": d, "top_feature": imp[d].idxmax(),
                     "matches_ganges_top": imp[d].idxmax() == gi.idxmax(),
                     "kendall_tau_vs_ganges": round(float(tau), 3), "p": round(float(p), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "h4_feature_importance.csv", index=False)
    summary = {"ganges_top_feature": gi.idxmax(),
               "mean_kendall_tau": round(float(df["kendall_tau_vs_ganges"].mean()), 3),
               "frac_sharing_ganges_top": round(float(df["matches_ganges_top"].mean()), 3),
               "n_deltas": int(len(df))}
    return summary, df


def fewshot_all(panel, sizes=(0, 5, 10, 15, 20), repeats=20, min_n=40):
    deltas = [d for d in panel.delta.unique() if (panel.delta == d).sum() >= min_n]
    data = {d: (panel[panel.delta == d][PRED].values, panel[panel.delta == d]["soc"].values) for d in panel.delta.unique()}
    rng = np.random.default_rng(7)
    rows = []
    for tgt in deltas:
        Xo = np.vstack([data[o][0] for o in panel.delta.unique() if o != tgt])
        yo = np.concatenate([data[o][1] for o in panel.delta.unique() if o != tgt])
        Xt, yt = data[tgt]
        # within-delta CV ceiling
        p = np.full(len(yt), np.nan)
        for a, b in KFold(5, shuffle=True, random_state=42).split(Xt):
            p[b] = gbr().fit(Xt[a], yt[a]).predict(Xt[b])
        ceiling = r2_score(yt, p)
        for n in sizes:
            r2s = []
            for _ in range(repeats):
                idx = rng.permutation(len(yt)); loc, rest = idx[:n], idx[n:]
                if len(rest) < 5:
                    continue
                if n == 0:
                    m = gbr().fit(Xo, yo)
                else:
                    m = gbr().fit(np.vstack([Xo, Xt[loc]]), np.concatenate([yo, yt[loc]]))
                r2s.append(r2_score(yt[rest], m.predict(Xt[rest])))
            rows.append({"delta": tgt, "n_local": n, "R2": round(float(np.mean(r2s)), 3),
                         "ceiling_within_R2": round(float(ceiling), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "h4_fewshot_curves.csv", index=False)
    # how many reach >=50% of within-delta ceiling by n=20
    piv = df.pivot_table(index="delta", columns="n_local", values="R2")
    ceil = df.groupby("delta")["ceiling_within_R2"].first()
    reach = sum(1 for d in piv.index if ceil[d] > 0 and piv.loc[d, 20] >= 0.5 * ceil[d])
    pos20 = sum(1 for d in piv.index if piv.loc[d, 20] > 0)
    summary = {"n_deltas": int(len(piv)),
               "reach_50pct_ceiling_by_n20": int(reach),
               "positive_R2_by_n20": int(pos20)}
    return summary, df


def main():
    panel = load_panel()
    M = pd.read_csv(HERE / "h4_transfer_matrix.csv", index_col=0).astype(float)
    dd, _ = distance_decay(panel, M)
    fu, _ = feature_universality(panel)
    fs, _ = fewshot_all(panel)
    res = {"distance_decay": dd, "feature_universality": fu, "fewshot_recovery": fs}
    (HERE / "h4_expanded_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
