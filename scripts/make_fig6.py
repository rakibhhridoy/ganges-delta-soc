"""
Rebuild Fig 6 (SI: H3 spatial structure) in the unified warm palette.

  (a) SOC distribution by geomorphological zone (box + jittered points)
  (b) spatial cross-validation: per-zone Leave-Zone-Out R^2 (all < 0) vs the
      pooled Leave-Location-Out R^2 (0.83), showing within-zone generalisation
      but failure to extrapolate across zones.

Reconstructed from surviving data (data/{zone}.csv, results/features_preprocessed.csv);
LZO/LOLO recomputed leak-free (group = location/zone), matching ml_spatial_cv.py.

Run:  python3 make_fig6.py  ->  fig6_hypo3_spatial.png
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.dpi": 200, "savefig.dpi": 300})

PALETTE = ["#491212", "#BF281B", "#CE471C", "#F24C27", "#F27127"]

# Bar hatching (dots, vertical, horizontal, cross, crosshatch), adaptive edge colour.
import matplotlib.colors as _mc
HATCHES = ['..', '||', '--', '++', 'xx']
def _edge(fc):
    r, g, b = _mc.to_rgb(fc)[:3]
    return 'white' if (0.299*r + 0.587*g + 0.114*b) < 0.5 else '#222222'
def apply_hatches(bars, hatches=None):
    hs = hatches or [HATCHES[i % len(HATCHES)] for i in range(len(bars))]
    for b, h in zip(bars, hs):
        b.set_edgecolor(_edge(b.get_facecolor())); b.set_linewidth(0.6); b.set_hatch(h)
ZONES = {"Tidal Active": "Tidal.csv", "Active Delta": "Active.csv",
         "Mature Delta": "Mature.csv", "Moribund Delta": "Moribund.csv"}
ZCOLOR = {"Tidal Active": "#491212", "Active Delta": "#BF281B",
          "Mature Delta": "#CE471C", "Moribund Delta": "#F27127"}

frames = []
for z, f in ZONES.items():
    d = pd.read_csv(DATA / f); d.columns = [c.strip() for c in d.columns]; d["zone"] = z
    frames.append(d)
alld = pd.concat(frames, ignore_index=True)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.6))
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.16, wspace=0.28)

# --- (a) SOC by zone ---------------------------------------------------------
order = list(ZONES.keys())
data = [alld[alld.zone == z]["SOC Stock"].values for z in order]
bp = axa.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
for patch, z in zip(bp["boxes"], order):
    patch.set_facecolor(ZCOLOR[z]); patch.set_alpha(0.75)
for med in bp["medians"]:
    med.set_color("white"); med.set_linewidth(1.5)
for i, z in enumerate(order):
    y = data[i]; x = np.random.default_rng(1).normal(i + 1, 0.06, len(y))
    axa.scatter(x, y, s=14, color=ZCOLOR[z], edgecolor="white", linewidth=0.3, alpha=0.9, zorder=3)
axa.set_xticks(range(1, len(order) + 1)); axa.set_xticklabels([z.replace(" ", "\n") for z in order], fontsize=8.5)
axa.set_ylabel("SOC stock (t C ha$^{-1}$)")
axa.set_title("(a) SOC distribution by zone")

# --- (b) per-zone LZO vs pooled LOLO -----------------------------------------
feat = pd.read_csv(DATA / "features_preprocessed.csv")
y = pd.read_csv(DATA / "target.csv").iloc[:, 0].values.astype(float)
X = feat.drop(columns=[c for c in feat.columns if c.strip().upper() == "OM"]).values
zlab = alld["zone"].values

def gbr(): return make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))

# per-zone Leave-Zone-Out: train on other 3 zones, predict held-out zone
lzo = {}
for z in order:
    tr = zlab != z; te = zlab == z
    m = gbr(); m.fit(X[tr], y[tr])
    lzo[z] = r2_score(y[te], m.predict(X[te]))

# pooled Leave-Location-Out R^2 (reported)
LOLO = 0.834

labels = [z.replace(" ", "\n") for z in order] + ["LOLO\n(pooled)"]
vals = [lzo[z] for z in order] + [LOLO]
cols = [ZCOLOR[z] for z in order] + ["#7a7a7a"]
apply_hatches(axb.bar(range(len(vals)), vals, color=cols, width=0.62))
axb.axhline(0, color="k", lw=0.8)
axb.set_xticks(range(len(labels))); axb.set_xticklabels(labels, fontsize=8)
axb.set_ylabel("Cross-validation $R^2$")
axb.set_title("(b) Leave-Zone-Out fails; LOLO generalises")
for i, v in enumerate(vals):
    axb.text(i, v + (0.04 if v >= 0 else -0.04), f"{v:.2f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=8)
axb.margins(y=0.18)

fig.savefig("fig6_hypo3_spatial.png", bbox_inches="tight")
print("wrote fig6_hypo3_spatial.png")
print("per-zone LZO R^2:", {z: round(v, 2) for z, v in lzo.items()}, "| LOLO:", LOLO)
