"""
H1 geochemical hypothesis tests -- reproduces Table 1 (tab:hypo1_summary) and the
associated H1 numbers in the Results text, end-to-end from the four zone CSVs.

Outputs, for each sub-hypothesis:
  H1.1-H1.7  Pearson r (+ p, Fisher-z 95% CI, bootstrap 95% CI, Spearman rho)
  H1.3       pH quadratic regression R^2 (+ F-test p)
  H1.8-H1.10 hierarchical-regression interaction delta-R^2 (+ F, p)
plus the salinity contrast (brackish EC>0.5 vs freshwater: ratio, Cohen's d,
Kruskal-Wallis) and the GBR residual diagnostics (Shapiro-Wilk, Breusch-Pagan).

Targets follow the manuscript: SOC stock for H1.1-H1.6 and the interactions;
SOC concentration for H1.7 (bulk density) to avoid the algebraic stock-BD link.
Seed 42; bootstrap = 2000 resamples.  Run:  python3 h1_hypothesis_tests.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42; rng = np.random.default_rng(SEED)
ZONES = {"Tidal Active": "Tidal.csv", "Active Delta": "Active.csv",
         "Mature Delta": "Mature.csv", "Moribund Delta": "Moribund.csv"}

frames = []
for z, f in ZONES.items():
    d = pd.read_csv(DATA / f); d.columns = [c.strip() for c in d.columns]; d["zone"] = z
    frames.append(d)
a = pd.concat(frames, ignore_index=True)
STOCK, CONC = "SOC Stock", "SOCC"
print(f"n = {len(a)}; mean SOC stock = {a[STOCK].mean():.1f} t C/ha\n")

def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return tuple(np.tanh([z - 1.96 * se, z + 1.96 * se]))

def boot_ci(x, y, n_boot=2000):
    idx = np.arange(len(x)); rs = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if np.std(x[s]) > 0 and np.std(y[s]) > 0:
            rs.append(np.corrcoef(x[s], y[s])[0, 1])
    return np.percentile(rs, [2.5, 97.5])

print("=== H1.1-H1.7 univariate correlations (reported r in parens) ===")
rows = [("H1.1 Clay", "Clay", STOCK, 0.323), ("H1.2 Moisture", "Moisture", STOCK, 0.552),
        ("H1.4 TN", "TN", STOCK, 0.045), ("H1.5 CEC", "CEC", STOCK, 0.636),
        ("H1.6 EC", "EC", STOCK, 0.664), ("H1.7 BD (vs conc.)", "Bulk Density", CONC, -0.511)]
for name, col, tgt, rep in rows:
    x, y = a[col].values.astype(float), a[tgt].values.astype(float)
    r, p = stats.pearsonr(x, y); rho, _ = stats.spearmanr(x, y)
    flo, fhi = fisher_ci(r, len(x)); blo, bhi = boot_ci(x, y)
    print(f"  {name:20s} r={r:+.3f} (rep {rep:+.3f})  p={p:.4g}  "
          f"Fisher[{flo:+.2f},{fhi:+.2f}] boot[{blo:+.2f},{bhi:+.2f}]  rho={rho:+.2f}")

print("\n=== H1.3 pH quadratic regression (reported R^2=0.108; dash = non-sig quadratic term) ===")
pH = a["pH"].values.astype(float); y = a[STOCK].values.astype(float)
mq = sm.OLS(y, sm.add_constant(np.column_stack([pH, pH**2]))).fit()       # linear + quadratic
ml = sm.OLS(y, sm.add_constant(pH)).fit()                                 # linear only
n = len(y); dR2 = mq.rsquared - ml.rsquared
F_inc = dR2 / ((1 - mq.rsquared) / (n - 3)); p_inc = 1 - stats.f.cdf(F_inc, 1, n - 3)
print(f"  quadratic-model R^2={mq.rsquared:.3f}  |  incremental pH^2 term: F={F_inc:.2f} p={p_inc:.3f} "
      f"({'non-significant -> dash OK' if p_inc >= 0.05 else 'SIGNIFICANT -> dash wrong'})")

print("\n=== H1.8-H1.10 hierarchical interaction delta-R^2 (reported) ===")
def hier(v1, v2, rep_dr2, tag):
    x1, x2, y = a[v1].values.astype(float), a[v2].values.astype(float), a[STOCK].values.astype(float)
    base = sm.OLS(y, sm.add_constant(np.column_stack([x1, x2]))).fit()
    full = sm.OLS(y, sm.add_constant(np.column_stack([x1, x2, x1 * x2]))).fit()
    dR2 = full.rsquared - base.rsquared
    n = len(y); F = dR2 / ((1 - full.rsquared) / (n - 4))
    p = 1 - stats.f.cdf(F, 1, n - 4)
    print(f"  {tag:22s} dR2={dR2:+.3f} (rep {rep_dr2:+.3f})  F={F:.2f}  p={p:.4g}")
hier("Clay", "Moisture", 0.068, "H1.8 Clay x Moisture")
hier("CEC", "pH", 0.095, "H1.9 CEC x pH (F=20.75)")
hier("EC", "Moisture", 0.028, "H1.10 EC x Moisture")

print("\n=== Salinity contrast (reported 6.5x, d=1.87, H=50.2) ===")
br = a[a.EC > 0.5][STOCK].values; fw = a[a.EC <= 0.5][STOCK].values
nx, ny = len(br), len(fw)
sp = np.sqrt(((nx-1)*br.var(ddof=1) + (ny-1)*fw.var(ddof=1)) / (nx+ny-2))
H, pkw = stats.kruskal(br, fw)
print(f"  brackish mean={br.mean():.1f} (n={nx}) vs freshwater={fw.mean():.1f} (n={ny}); "
      f"ratio={br.mean()/fw.mean():.2f}x  Cohen d={(br.mean()-fw.mean())/sp:.2f}  KW H={H:.1f} p={pkw:.2g}")

print("\n=== GBR residual diagnostics (reported Shapiro W=0.813, BP p=0.002) ===")
feat = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "features_preprocessed.csv")
feat = feat.drop(columns=[c for c in feat.columns if c.strip().upper() == "OM"])
yv = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "target.csv").iloc[:, 0].values.astype(float)
soc = pd.read_csv(DATA / "GangesSOC.csv"); soc.columns = [c.strip() for c in soc.columns]
loc = pd.factorize(soc["Location"].values)[0]
Xf = feat.values; pred = np.full(len(yv), np.nan)
for tr, te in GroupKFold(5).split(Xf, yv, loc):
    mdl = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=SEED)).fit(Xf[tr], yv[tr])
    pred[te] = mdl.predict(Xf[te])
res = yv - pred
W, pw = stats.shapiro(res)
bp_p = het_breuschpagan(res, sm.add_constant(pred))[1]
print(f"  Shapiro-Wilk W={W:.3f} p={pw:.4g}  |  Breusch-Pagan p={bp_p:.4g}  (location-grouped OOF residuals)")
