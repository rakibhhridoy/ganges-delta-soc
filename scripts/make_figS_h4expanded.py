"""
Supplementary figure for the expanded H4 analyses (2026-05-25).

(a) Distance-decay: cross-delta transfer R^2 vs feature-space distance between deltas
    (Spearman rho annotated); transfer collapses as environmental distance grows.
(b) Feature-importance universality: Kendall tau of each delta's GBR importance ranking
    vs the Ganges ranking (mostly near zero / negative = predictors are weighted
    differently per delta -> no universal feature structure).
(c) Few-shot recovery: pooled(all other deltas)+n local samples -> R^2 on held-out
    local samples, per delta. Curves stay at/below zero until local data is added.

Reads h4_distance_decay.csv, h4_feature_importance.csv, h4_fewshot_curves.csv
(produced by h4_expanded_analysis.py). Run: python3 make_figS_h4expanded.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "figure.dpi": 200, "savefig.dpi": 300})

dd = pd.read_csv(HERE / "h4_distance_decay.csv")
fi = pd.read_csv(HERE / "h4_feature_importance.csv")
fs = pd.read_csv(HERE / "h4_fewshot_curves.csv")

fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(13.5, 4.2))
fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.16, wspace=0.30)

# ---- (a) distance-decay -----------------------------------------------------
y = np.clip(dd["transfer_R2"], -3, 1)
axa.scatter(dd["feat_dist"], y, s=14, alpha=0.5, color="#b3322c", edgecolor="none")
rho, p = stats.spearmanr(dd["feat_dist"], dd["transfer_R2"])
rho_g, p_g = stats.spearmanr(dd["geo_km"], dd["transfer_R2"])
axa.axhline(0, color="k", lw=0.7, ls=":")
axa.set_xlabel("feature-space distance between deltas")
axa.set_ylabel("cross-delta transfer $R^2$ (clipped at $-3$)")
axa.set_title("(a) Transfer failure is distance-independent")
axa.text(0.96, 0.06, f"feature dist: $\\rho$={rho:.2f} ($p$={p:.2f})\n"
                     f"geographic:  $\\rho$={rho_g:.2f} ($p$={p_g:.2f})",
         transform=axa.transAxes, ha="right", va="bottom", fontsize=7.5,
         bbox=dict(boxstyle="round", fc="white", ec="0.7"))

# ---- (b) feature universality ----------------------------------------------
fi2 = fi.sort_values("kendall_tau_vs_ganges")
colors = ["#1b7f7a" if t > 0 else "#b3322c" for t in fi2["kendall_tau_vs_ganges"]]
axb.barh(fi2["delta"], fi2["kendall_tau_vs_ganges"], color=colors)
axb.axvline(0, color="k", lw=0.8)
axb.set_xlabel("Kendall $\\tau$ of feature ranking vs Ganges")
axb.set_title("(b) Same predictors matter, mapping differs")
axb.tick_params(axis="y", labelsize=7)
axb.text(0.96, 0.04, f"mean $\\tau$ = {fi['kendall_tau_vs_ganges'].mean():.2f}",
         transform=axb.transAxes, ha="right", va="bottom", fontsize=8,
         bbox=dict(boxstyle="round", fc="white", ec="0.7"))

# ---- (c) few-shot recovery curves ------------------------------------------
cmap = plt.cm.viridis(np.linspace(0, 0.92, fs["delta"].nunique()))
for (d, g), c in zip(fs.groupby("delta"), cmap):
    g = g.sort_values("n_local")
    axc.plot(g["n_local"], np.clip(g["R2"], -2, 1), "o-", ms=3, lw=1.0, color=c, label=d)
    # within-delta CV ceiling marker (where the dedicated local model lands)
    axc.plot(22, np.clip(g["ceiling_within_R2"].iloc[0], -2, 1), "*", ms=7, color=c)
axc.axhline(0, color="k", lw=0.8, ls=":")
axc.text(22, 1.02, "within-\ndelta CV", fontsize=5.5, ha="center", va="bottom")
axc.set_xlabel("local samples added to pooled model ($n$)")
axc.set_ylabel("held-out $R^2$ (clipped at $-2$)")
axc.set_title("(c) Pooling cannot replace local data")
axc.legend(fontsize=5.5, ncol=2, frameon=False, loc="lower right")

fig.savefig(HERE / "figS_h4_expanded.png", bbox_inches="tight")
print("wrote figS_h4_expanded.png")
