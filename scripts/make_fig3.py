"""
Regenerate Fig 3 (H2 climate dynamics) from the rebuilt first-order G-RothC.

The original Fig 3 and its plotting code were not preserved, and the empirical
temperature-trend data (old panel a) is gone. This figure is therefore rebuilt
entirely from reproducible model output (grothc_model.py):

  (a) mean SOC trajectories 2000-2050 under three key scenarios
  (b) projected 2050 mean SOC under the six scenarios
  (c) SOC change rate by salinity class (brackish vs freshwater)
  (d) parameter sensitivity of 2050 SOC to k0 and Q10

Run:  python3 make_fig3.py   ->  fig3_climate.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

import grothc_model as g

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.dpi": 200, "savefig.dpi": 300})

s = g.load_samples()
C_in, D0 = g.calibrate_input(s)
base_mean = s["SOC_obs"].mean()

# Single warm palette (fishualize::Acanthostracion_polygonius_y); roles chosen for
# lightness contrast within the maroon->orange ramp.
PALETTE = ["#491212", "#BF281B", "#CE471C", "#F24C27", "#F27127"]
C_TEAL, C_ORANGE, C_GREEN, C_RED = "#F27127", "#BF281B", "#CE471C", "#491212"

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

fig = plt.figure(figsize=(11, 8.5))
gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.26,
                       left=0.08, right=0.97, top=0.93, bottom=0.08)

# ---- (a) trajectories -------------------------------------------------------
axa = fig.add_subplot(gs[0, 0])
scen = [("RCP4.5", 0.0, C_TEAL, "RCP 4.5 BAU"),
        ("RCP8.5", 0.0, C_RED, "RCP 8.5 BAU"),
        ("RCP4.5", 0.50, C_GREEN, "RCP 4.5 + 50% mangrove")]
for pw, frac, col, lab in scen:
    yrs, traj = g.simulate_trajectory(s, C_in, pw, frac)
    axa.plot(yrs, traj, color=col, lw=2.2, label=lab)
axa.axhline(base_mean, color="grey", ls=":", lw=1, label=f"2000 baseline ({base_mean:.1f})")
axa.set_xlabel("Year"); axa.set_ylabel("Mean SOC (t C ha$^{-1}$)")
axa.set_title("(a) Projected SOC trajectories")
axa.legend(frameon=False, fontsize=8, loc="upper left")
axa.margins(x=0.02)

# ---- (b) 2050 projections, six scenarios ------------------------------------
axb = fig.add_subplot(gs[0, 1])
land = ["BAU", "10pct_mangrove", "50pct_mangrove"]
land_lab = ["BAU", "10% mangrove", "50% mangrove"]
pathways = ["RCP4.5", "RCP8.5"]
vals = {pw: [np.mean(g.simulate(s, C_in, pw, {"BAU": 0, "10pct_mangrove": .1, "50pct_mangrove": .5}[l]))
             for l in land] for pw in pathways}
x = np.arange(len(land)); w = 0.38
apply_hatches(axb.bar(x - w/2, vals["RCP4.5"], w, color=C_TEAL, label="RCP 4.5"), [".."] * len(x))
apply_hatches(axb.bar(x + w/2, vals["RCP8.5"], w, color=C_RED, label="RCP 8.5"), ["++"] * len(x))
axb.axhline(base_mean, color="grey", ls=":", lw=1)
axb.set_xticks(x); axb.set_xticklabels(land_lab)
axb.set_ylabel("2050 mean SOC (t C ha$^{-1}$)")
axb.set_title("(b) 2050 projections by scenario")
axb.legend(frameon=False, fontsize=8)
for xi, v in zip(x - w/2, vals["RCP4.5"]):
    axb.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)
for xi, v in zip(x + w/2, vals["RCP8.5"]):
    axb.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)

# ---- (c) change rate by salinity class --------------------------------------
axc = fig.add_subplot(gs[1, 0])
soc_bau = g.simulate(s, C_in, "RCP4.5", 0.0)
rate = (soc_bau - s["SOC_obs"].values) / (g.YEAR_END - g.YEAR0)
classes = ["Freshwater", "Brackish"]
means = [np.mean(rate[s["salinity_class"].values == c]) for c in classes]
cols = [C_TEAL, C_ORANGE]
apply_hatches(axc.bar(classes, means, color=cols, width=0.55))
axc.axhline(0, color="k", lw=0.8)
axc.set_ylabel("SOC change rate (t C ha$^{-1}$ yr$^{-1}$)")
axc.set_title("(c) Change rate by salinity class (RCP 4.5 BAU)")
for i, v in enumerate(means):
    axc.text(i, v + (0.004 if v >= 0 else -0.004), f"{v:+.3f}",
             ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
axc.margins(y=0.2)

# ---- (d) parameter sensitivity ----------------------------------------------
axd = fig.add_subplot(gs[1, 1])
k0_range = np.linspace(0.01, 0.05, 9)
q10_range = np.linspace(1.5, 3.5, 9)
k0_means = [np.mean(g.simulate(s, g.calibrate_input_for(s, k0=k)[0], "RCP4.5", 0.0, k0=k)) for k in k0_range]
q10_means = [np.mean(g.simulate(s, g.calibrate_input_for(s, q10=q)[0], "RCP4.5", 0.0, q10=q)) for q in q10_range]
axd.plot(k0_range / k0_range.max(), k0_means, "o-", color=C_ORANGE, lw=2, label="$k_0$ (0.01–0.05)")
axd.plot(q10_range / q10_range.max(), q10_means, "s-", color=C_TEAL, lw=2, label="$Q_{10}$ (1.5–3.5)")
axd.set_xlabel("Parameter (fraction of max)")
axd.set_ylabel("2050 mean SOC (t C ha$^{-1}$)")
axd.set_title("(d) Parameter sensitivity")
axd.legend(frameon=False, fontsize=8)

fig.savefig("fig3_climate.png", bbox_inches="tight")
print("wrote fig3_climate.png")
print(f"(a) trajectories; (b) 2050 six-scenario; (c) rates F={means[0]:+.3f} B={means[1]:+.3f}; "
      f"(d) k0 span {(max(k0_means)-min(k0_means))/np.mean(k0_means)*100:.1f}% vs "
      f"Q10 span {(max(q10_means)-min(q10_means))/np.mean(q10_means)*100:.1f}%")
