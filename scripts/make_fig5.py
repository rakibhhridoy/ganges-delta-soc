"""
Fig 5 — cross-delta transferability of SOC prediction (rebuilt from WoSIS, 14 deltas).

(a) Transfer matrix: train on delta i (row), test on delta j (col). Diagonal (boxed) =
    within-delta 5-fold CV; off-diagonal = zero-shot transfer R^2. Colour clipped to
    [-1, 1] (R^2 < -1 = no skill) so the within-vs-across contrast is legible.
(b) Distribution of within-delta vs cross-delta R^2 (the headline contrast), with an
    inset of the reverse transfer (pooled 14-delta model -> Ganges) few-shot curve.

Reads h4_results.json + h4_transfer_matrix.csv (produced by h4_analysis.py).
Run:  python3 make_fig5.py  ->  fig5_transfer.png
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "figure.dpi": 200, "savefig.dpi": 300})

M = pd.read_csv(HERE / "h4_transfer_matrix.csv", index_col=0).astype(float)
res = json.load(open(HERE / "h4_results.json"))

order = ["Mississippi", "Rhine-Meuse", "Niger", "Yellow River", "Limpopo", "Chao Phraya",
         "Parana-Plata", "Sacramento", "Indus", "Mackenzie", "Danube", "Amazon",
         "Senegal", "Pearl", "Ganges"]
order = [d for d in order if d in M.index]
M = M.loc[order, order]

fig = plt.figure(figsize=(11, 5.6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.45, 1.0], wspace=0.30,
                       left=0.13, right=0.97, top=0.90, bottom=0.16)

# ---- (a) transfer-matrix heatmap -------------------------------------------
axa = fig.add_subplot(gs[0, 0])
Mc = M.clip(-1, 1)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = axa.imshow(Mc.values, cmap="RdBu", norm=norm, aspect="auto")
axa.set_xticks(range(len(order))); axa.set_xticklabels(order, rotation=90, fontsize=7)
axa.set_yticks(range(len(order))); axa.set_yticklabels(order, fontsize=7)
axa.set_xlabel("Tested on"); axa.set_ylabel("Trained on")
axa.set_title("(a) Cross-delta transfer matrix ($R^2$)", pad=18)
axa.text(0.5, 1.015, "diagonal (boxed) = within-delta CV; off-diagonal = zero-shot transfer",
         transform=axa.transAxes, ha="center", va="bottom",
         fontsize=6.8, style="italic", color="0.3")
for i in range(len(order)):
    axa.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.2))
cb = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.02, extend="min")
cb.set_label("$R^2$ (clipped at $-1$)", fontsize=8)

# ---- (b) within vs cross distribution --------------------------------------
axb = fig.add_subplot(gs[0, 1])
diag = np.array([M.loc[d, d] for d in order if not np.isnan(M.loc[d, d])])
off = M.values[~np.eye(len(order), dtype=bool)]
off = off[~np.isnan(off)]
bins = np.linspace(-2, 1, 25)
axb.hist(np.clip(off, -2, 1), bins=bins, color="#b3322c", alpha=0.75,
         label=f"cross-delta (n={len(off)}; {(off > 0).mean() * 100:.0f}% $>$0)")
axb.hist(np.clip(diag, -2, 1), bins=bins, color="#1b7f7a", alpha=0.85,
         label=f"within-delta (median {np.median(diag):+.2f})")
axb.axvline(0, color="k", lw=0.8, ls=":")
axb.set_xlabel("$R^2$ (clipped at $-2$)"); axb.set_ylabel("count")
axb.set_title("(b) Within vs cross-delta skill")
axb.legend(frameon=False, fontsize=7, loc="upper left")

axc = axb.inset_axes([0.58, 0.52, 0.38, 0.40])
fs = res["reverse_few_shot"]
xs = sorted(int(k) for k in fs); ys = [fs[str(k)] for k in xs]
axc.plot(xs, ys, "o-", color="#3a3a3a", lw=1.4, ms=3)
axc.axhline(0, color="0.6", lw=0.7, ls=":")
axc.set_title("pooled$\\rightarrow$Ganges", fontsize=6.5, fontweight="normal")
axc.set_xlabel("local n", fontsize=6); axc.set_ylabel("$R^2$", fontsize=6)
axc.tick_params(labelsize=5.5)

fig.savefig(HERE / "fig5_transfer.png", bbox_inches="tight")
print("wrote fig5_transfer.png")
print(f"within median={np.median(diag):+.2f}  cross median={np.median(off):+.2f}  cross %>0={(off > 0).mean() * 100:.0f}%")
