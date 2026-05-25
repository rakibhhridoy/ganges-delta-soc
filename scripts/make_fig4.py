"""
Regenerate Fig 4 (H3 ML/DL prediction) from the leak-free analysis.

The original Fig 4 was built from a pipeline that (i) included the OM leakage column
(OM ~0.98 feature importance, r=0.996 with SOC) and (ii) used random k-fold CV. Both
are corrected here (see ml_spatial_cv.py / R1-13, R1-25, R3-13). All panels are
reproducible from the surviving feature matrix; deep-learning architectures are not
re-fit (their code is gone), so this figure reports the gradient-boosting model under
honest, leak-free validation rather than the optimistic multi-model comparison.

  (a) GBR R^2 under random vs leak-free CV schemes (+ EC-only baseline)
  (b) observed vs predicted SOC, location-grouped out-of-fold
  (c) GBR feature importance (OM excluded), top 12, coloured by feature group
  (d) feature-group ablation under location-grouped CV

Run:  python3 make_fig4.py   ->  fig4_ml_dl.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.dpi": 200, "savefig.dpi": 300})

# ---- data ------------------------------------------------------------------
feat = pd.read_csv(DATA / "features_preprocessed.csv")
y = pd.read_csv(DATA / "target.csv").iloc[:, 0].values.astype(float)
soc = pd.read_csv(DATA / "GangesSOC.csv"); soc.columns = [c.strip() for c in soc.columns]
loc_groups = pd.factorize(soc["Location"].values)[0]

zone_map = {}
for z, f in [("Tidal", "Tidal.csv"), ("Active", "Active.csv"),
             ("Mature", "Mature.csv"), ("Moribund", "Moribund.csv")]:
    zf = pd.read_csv(DATA / f); zf.columns = [c.strip() for c in zf.columns]
    for L in zf["Location"].unique():
        zone_map[L] = z
zone_groups = pd.factorize(np.array([zone_map.get(L, "?") for L in soc["Location"].values]))[0]

feat_model = feat.drop(columns=[c for c in feat.columns if c.strip().upper() == "OM"])

SOIL = ["Bulk Density", "Moisture", "Sand", "Silt", "Clay", "pH", "EC", "TN", "CEC"]
SPATIAL = ["Lat", "Long", "Elevation", "Slope", "Aspect"]
def fgroup(c):
    if c in SOIL: return "Soil property"
    if c in SPATIAL: return "Spatial/topographic"
    return "ViT embedding"

X = feat_model.values
def gbr(): return make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
def ols(): return make_pipeline(StandardScaler(), LinearRegression())

def pooled(make_model, Xm, splitter, groups=None):
    preds = np.full(len(y), np.nan)
    it = splitter.split(Xm, y, groups) if groups is not None else splitter.split(Xm, y)
    for tr, te in it:
        m = make_model(); m.fit(Xm[tr], y[tr]); preds[te] = m.predict(Xm[te])
    return preds

random5 = KFold(5, shuffle=True, random_state=42)
group5 = GroupKFold(5)
logo = LeaveOneGroupOut()

# Discrete viridis (user request: Figs 4 & 5 in viridis). Sample away from the
# extreme dark/bright ends for legible bars + readable value labels.
def viridis_n(n):
    return [plt.cm.viridis_r(x) for x in np.linspace(0.12, 0.86, n)]

_v3 = viridis_n(3)
C_SOIL, C_SPA, C_EMB = _v3[0], _v3[1], _v3[2]   # 3 feature groups (panel c)
C_POINT = plt.cm.viridis_r(0.45)                # panel b scatter

# Bar hatching (dots, vertical, horizontal, cross, crosshatch) with adaptive
# edge colour so the pattern is legible on both light and dark fills.
import matplotlib.colors as _mc
HATCHES = ['..', '||', '--', '++', 'xx']
def _edge(fc):
    r, g, b = _mc.to_rgb(fc)[:3]
    return 'white' if (0.299*r + 0.587*g + 0.114*b) < 0.5 else '#222222'
def apply_hatches(bars, hatches=None):
    hs = hatches or [HATCHES[i % len(HATCHES)] for i in range(len(bars))]
    for b, h in zip(bars, hs):
        b.set_edgecolor(_edge(b.get_facecolor())); b.set_linewidth(0.6); b.set_hatch(h)

fig = plt.figure(figsize=(11, 8.5))
gs = gridspec.GridSpec(2, 2, hspace=0.34, wspace=0.27, left=0.09, right=0.97, top=0.93, bottom=0.09)

# ---- (a) CV schemes ---------------------------------------------------------
axa = fig.add_subplot(gs[0, 0])
schemes = [("Random\n5-fold", pooled(gbr, X, random5)),
           ("Location-\ngrouped 5-fold", pooled(gbr, X, group5, loc_groups)),
           ("Leave-Location-\nOut", pooled(gbr, X, logo, loc_groups)),
           ("Leave-Zone-\nOut", pooled(gbr, X, logo, zone_groups))]
labels = [s[0] for s in schemes]
r2s = [r2_score(y, s[1]) for s in schemes]
bars = axa.bar(range(len(labels)), r2s, color=viridis_n(len(labels)), width=0.62)
apply_hatches(bars)
axa.axhline(0, color="k", lw=0.8)
axa.set_xticks(range(len(labels))); axa.set_xticklabels(labels, fontsize=8)
axa.set_ylabel("Pooled $R^2$"); axa.set_title("(a) GBR under leak-free CV schemes")
axa.set_ylim(min(r2s) - 0.16, max(r2s) + 0.16)
for i, v in enumerate(r2s):
    axa.text(i, v + (0.03 if v >= 0 else -0.05), f"{v:.2f}", ha="center", fontsize=8,
             va="bottom" if v >= 0 else "top")

# ---- (b) observed vs predicted (location-grouped OOF) -----------------------
axb = fig.add_subplot(gs[0, 1])
pred_grp = pooled(gbr, X, group5, loc_groups)
axb.scatter(y, pred_grp, s=28, color=C_POINT, alpha=0.7, edgecolor="white", linewidth=0.4)
lim = [0, max(y.max(), pred_grp.max()) * 1.05]
axb.plot(lim, lim, "--", color="0.4", lw=1)
axb.set_xlim(lim); axb.set_ylim(lim)
axb.set_xlabel("Observed SOC (t C ha$^{-1}$)"); axb.set_ylabel("Predicted SOC (t C ha$^{-1}$)")
axb.set_title("(b) Observed vs predicted (location-grouped)")
axb.text(0.05, 0.93, f"$R^2$ = {r2_score(y, pred_grp):.2f}\nleak-free OOF", transform=axb.transAxes,
         fontsize=9, va="top", bbox=dict(boxstyle="round", fc="#fff4e6", ec="0.6"))

# ---- (c) feature importance (OM excluded), top 12 ---------------------------
axc = fig.add_subplot(gs[1, 0])
m = GradientBoostingRegressor(random_state=42).fit(StandardScaler().fit_transform(X), y)
imp = pd.Series(m.feature_importances_, index=feat_model.columns).sort_values(ascending=False).head(12)[::-1]
gcolors = {"Soil property": C_SOIL, "Spatial/topographic": C_SPA, "ViT embedding": C_EMB}
ghatch = {"Soil property": "..", "Spatial/topographic": "||", "ViT embedding": "++"}
bar_cols = [gcolors[fgroup(c)] for c in imp.index]
cbars = axc.barh(range(len(imp)), imp.values, color=bar_cols)
apply_hatches(cbars, [ghatch[fgroup(c)] for c in imp.index])
axc.set_yticks(range(len(imp))); axc.set_yticklabels(imp.index, fontsize=8)
axc.set_xlabel("Gini importance"); axc.set_title("(c) GBR feature importance (OM excluded)")
axc.legend(handles=[Patch(facecolor=v, edgecolor=_edge(v), hatch=ghatch[k], label=k)
                    for k, v in gcolors.items()], fontsize=7.5, loc="lower right", framealpha=0.95)

# ---- (d) feature-group ablation (location-grouped CV) -----------------------
axd = fig.add_subplot(gs[1, 1])
sets = [("All\n(78 feat.)", feat_model.columns.tolist()),
        ("No\nembeddings", SOIL + SPATIAL),
        ("Soil\nonly (9)", SOIL),
        ("Spatial\nonly (5)", SPATIAL),
        ("EC\nonly (1)", ["EC"])]
abl_r2 = []
for _, cols_set in sets:
    Xs = feat_model[cols_set].values
    abl_r2.append(r2_score(y, pooled(gbr, Xs, group5, loc_groups)))
dbars = axd.bar(range(len(sets)), abl_r2, color=viridis_n(len(sets)), width=0.62)
apply_hatches(dbars)
axd.axhline(0, color="k", lw=0.8)
axd.set_xticks(range(len(sets))); axd.set_xticklabels([s[0] for s in sets], fontsize=8)
axd.set_ylabel("Location-grouped CV $R^2$"); axd.set_title("(d) Feature-group ablation")
axd.set_ylim(min(0, min(abl_r2)) - 0.04, max(abl_r2) + 0.10)
for i, v in enumerate(abl_r2):
    axd.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:.2f}", ha="center", fontsize=8,
             va="bottom" if v >= 0 else "top")

fig.savefig("fig4_ml_dl.png", bbox_inches="tight")
print("wrote fig4_ml_dl.png")
print("(a) CV:", {l.replace(chr(10), ' '): round(r, 2) for l, r in zip(labels, r2s)})
print("(c) top feature:", imp.index[-1], round(imp.values[-1], 3))
print("(d) ablation:", {s[0].replace(chr(10), ' '): round(r, 2) for s, r in zip(sets, abl_r2)})
