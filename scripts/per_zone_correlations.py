"""
Per-zone correlation analysis (addresses Reviewer 3, comment 5): because the four
geomorphological zones are distinct biogeochemical settings, soil property--SOC
correlations are computed separately within each zone, not only pooled. Also
substantiates the "EC paradox" (strong between-zone, weak within-zone) and the
TN sign-reversal that produces TN's near-zero pooled correlation.

Correlations use SOC concentration (consistent with the H1 choice; see A5/R3-1).
Run:  python3 per_zone_correlations.py  ->  per_zone_correlations.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA = Path(__file__).resolve().parent.parent / "data"
ZONES = {"Tidal Active": "Tidal.csv", "Active Delta": "Active.csv",
         "Mature Delta": "Mature.csv", "Moribund Delta": "Moribund.csv"}
PREDS = ["EC", "CEC", "Moisture", "Clay", "pH", "TN"]
Y = "SOCC"  # SOC concentration

frames = []
for zname, f in ZONES.items():
    d = pd.read_csv(DATA / f); d.columns = [c.strip() for c in d.columns]; d["zone"] = zname
    frames.append(d)
alld = pd.concat(frames, ignore_index=True)

rows = []
for p in PREDS:
    rec = {"predictor": p}
    r, pv = stats.pearsonr(alld[p], alld[Y])
    rec["Overall"] = f"{r:+.2f}{'*' if pv < 0.05 else ''}"
    for zname in ZONES:
        sub = alld[alld["zone"] == zname]
        r, pv = stats.pearsonr(sub[p], sub[Y])
        rec[zname] = f"{r:+.2f}{'*' if pv < 0.05 else ''}"
    rows.append(rec)

res = pd.DataFrame(rows)
res.to_csv(Path(__file__).resolve().parent / "per_zone_correlations.csv", index=False)
print(res.to_string(index=False))
print("\n* p < 0.05; n = 100 overall, n = 25 per zone; correlations vs SOC concentration.")
