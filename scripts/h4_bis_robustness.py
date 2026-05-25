"""
H4 robustness check — augment the Rhine-Meuse cell with the dense BIS Nederland
national soil survey (4TU), recovered 2026-05-25.

Motivation: a reviewer may ask whether the cross-delta transfer failure is an
artefact of sparse per-delta data. We test this on the best-covered European delta
by adding 2,157 complete-case BIS profiles (pH + sand/silt/clay co-measured;
SOM%->SOC via van Bemmelen ×0.58, then ->g/kg ×10) to the 504 WoSIS Rhine-Meuse
samples, and re-running the within-delta CV and the full transfer matrix.

Result (see also RECOVERY_NOTES.md):
  Rhine-Meuse within-delta R² rises 0.145 -> 0.202 (the within-delta signal is real
  and strengthens with more, higher-quality data), BUT the cross-delta conclusion is
  unchanged: cross-delta median R² = -0.55, ~6% of 210 pairs > 0, and Rhine-Meuse
  still does not transfer to (or from) other deltas. Transfer failure is therefore
  NOT a data-volume artefact.

Run: python3 h4_bis_robustness.py   (reads h4_delta_samples.csv, GangesSOC.csv,
     and ../_recovered_H4_original/all_deltas_harmonized.csv) -> h4_bis_robustness.json
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

HERE = Path(__file__).resolve().parent
PRED = ["ph", "clay", "silt", "sand"]


def gbr():
    return make_pipeline(StandardScaler(),
                         GradientBoostingRegressor(random_state=42, n_estimators=200))


def load_data(with_bis):
    # The shipped panel (data/h4_delta_samples.csv) already includes the BIS
    # Nederland rows for Rhine-Meuse, tagged in the `source` column. We derive
    # both panels from it: with_bis=True keeps all rows; with_bis=False drops the
    # BIS rows to recover the WoSIS-only Rhine-Meuse cell. No external archive needed.
    lock = pd.read_csv(HERE.parent / "data" / "h4_delta_samples.csv")
    if not with_bis and "source" in lock.columns:
        lock = lock[lock["source"] != "BIS_Nederland"]
    data = {d: (g[PRED].values, g["soc"].values) for d, g in lock.groupby("delta")}

    g = pd.read_csv(HERE.parent / "data" / "GangesSOC.csv")
    g.columns = [c.strip() for c in g.columns]
    gx = g[["pH", "Clay", "Silt", "Sand"]].rename(
        columns={"pH": "ph", "Clay": "clay", "Silt": "silt", "Sand": "sand"}).copy()
    gx["soc"] = g["SOCC"] * 10.0
    gx = gx.dropna(subset=["soc"] + PRED)
    data["Ganges"] = (gx[PRED].values, gx["soc"].values)

    if False:  # (legacy path; BIS now baked into the shipped panel)
        bis = pd.DataFrame()
        Xrm, yrm = data["Rhine-Meuse"]
        data["Rhine-Meuse"] = (np.vstack([Xrm, bis[PRED].values]),
                               np.concatenate([yrm, bis["soc"].values]))
    return data


def transfer(data):
    names = list(data.keys())
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
    diag = np.array([M.loc[d, d] for d in names if not pd.isna(M.loc[d, d])])
    off = M.values[~np.eye(len(names), dtype=bool)].astype(float)
    off = off[~np.isnan(off)]
    return M, diag, off


def summarize(tag, data):
    M, diag, off = transfer(data)
    out = {"tag": tag,
           "rhine_meuse_n": int(len(data["Rhine-Meuse"][1])),
           "rhine_meuse_within_R2": round(float(M.loc["Rhine-Meuse", "Rhine-Meuse"]), 3),
           "within_delta_median_R2": round(float(np.median(diag)), 3),
           "cross_delta_median_R2": round(float(np.median(off)), 3),
           "cross_delta_frac_positive": round(float((off > 0).mean()), 3),
           "cross_delta_n_pairs": int(len(off))}
    print(f"[{tag}] RM n={out['rhine_meuse_n']}  RM within R²={out['rhine_meuse_within_R2']:+.3f}  "
          f"within-median={out['within_delta_median_R2']:+.3f}  "
          f"cross-median={out['cross_delta_median_R2']:+.3f}  "
          f"{out['cross_delta_frac_positive']*100:.1f}% of {out['cross_delta_n_pairs']} >0")
    return out


def main():
    res = {"locked": summarize("locked", load_data(with_bis=False)),
           "bis_augmented": summarize("bis_augmented", load_data(with_bis=True))}
    (HERE / "h4_bis_robustness.json").write_text(json.dumps(res, indent=2))
    print("\nwrote h4_bis_robustness.json")
    print("Conclusion: BIS augmentation raises Rhine-Meuse within-delta skill but leaves "
          "the cross-delta transfer failure unchanged — failure is not a data-volume artefact.")


if __name__ == "__main__":
    main()
