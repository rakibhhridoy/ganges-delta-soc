"""RC2: redo Table 1 (H1.1-H1.10) without treating the 100 depth samples as independent.
Same data, targets and variables as the shipped h1_hypothesis_tests.py (stock for all except H1.7, which
uses concentration). For each test:
  pooled    as published, n = 100 (reproduction check)
  sitemean  location means, n = 20
  mixed     random intercept per location (ML), Wald p on the slope / quadratic / interaction term
  mixed+zone  as mixed, plus zone as a fixed effect: is there an association beyond the zone contrast?
  nonTidal  location means excluding Tidal, n = 15
Read-only on the data."""
import os as _os; _os.makedirs(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"), exist_ok=True); _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs"))

import numpy as np, pandas as pd, warnings
from pathlib import Path
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
DATA = (Path(__file__).resolve().parent.parent / "data")
Z = {"Tidal": "Tidal.csv", "Active": "Active.csv", "Mature": "Mature.csv", "Moribund": "Moribund.csv"}
fr = []
for z, f in Z.items():
    d = pd.read_csv(DATA / f); d.columns = [c.strip() for c in d.columns]; d["zone"] = z; fr.append(d)
a = pd.concat(fr, ignore_index=True)
a = a.rename(columns={"SOC Stock": "stock", "Bulk Density": "BD"})
a["loc"] = a.zone + "_" + a.Location
a["pH2"] = a.pH ** 2
print("rows", len(a), "locations", a["loc"].nunique())

def mixed(formula, term, data):
    try:
        m = smf.mixedlm(formula, data, groups=data["loc"]).fit(reml=False, method="lbfgs")
        return m.params[term], m.pvalues[term]
    except Exception as e:
        return np.nan, np.nan

def std(df, cols):
    df = df.copy()
    for c in cols: df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df

tests = [("H1.1", "Clay", "stock"), ("H1.2", "Moisture", "stock"), ("H1.4", "TN", "stock"),
         ("H1.5", "CEC", "stock"), ("H1.6", "EC", "stock"), ("H1.7", "BD", "SOCC")]
rows = []
for tid, x, y in tests:
    r, p = stats.pearsonr(a[x], a[y])
    m = a.groupby("loc").agg(x=(x, "mean"), y=(y, "mean"), zone=("zone", "first"))
    rs, ps = stats.pearsonr(m.x, m.y)
    mn = m[m.zone != "Tidal"]; rn, pn = stats.pearsonr(mn.x, mn.y)
    s = std(a, [x, y])
    b, pm = mixed(f"{y} ~ {x}", x, s)
    bz, pz = mixed(f"{y} ~ {x} + C(zone)", x, s)
    rows.append(dict(test=tid, var=x, target=y, pooled_r=r, pooled_p=p, site_r=rs, site_p=ps,
                     mixed_beta=b, mixed_p=pm, mixedzone_beta=bz, mixedzone_p=pz, nonTidal_site_r=rn, nonTidal_site_p=pn))

# H1.3 pH quadratic
s = std(a, ["pH", "stock"]); s["pH2"] = s.pH ** 2
ml = sm.OLS(a.stock, sm.add_constant(a[["pH", "pH2"]])).fit()
m = a.groupby("loc").agg(pH=("pH", "mean"), stock=("stock", "mean"), zone=("zone", "first")); m["pH2"] = m.pH ** 2
ms = sm.OLS(m.stock, sm.add_constant(m[["pH", "pH2"]])).fit()
_, pm = mixed("stock ~ pH + pH2", "pH2", s); _, pz = mixed("stock ~ pH + pH2 + C(zone)", "pH2", s)
rows.append(dict(test="H1.3", var="pH^2", target="stock", pooled_r=np.nan, pooled_p=ml.pvalues["pH2"],
                 site_r=np.nan, site_p=ms.pvalues["pH2"], mixed_beta=np.nan, mixed_p=pm, mixedzone_beta=np.nan, mixedzone_p=pz,
                 nonTidal_site_r=np.nan, nonTidal_site_p=np.nan))

# H1.8-H1.10 interactions
for tid, x1, x2 in [("H1.8", "Clay", "Moisture"), ("H1.9", "CEC", "pH"), ("H1.10", "EC", "Moisture")]:
    s = std(a, [x1, x2, "stock"]); s["ia"] = s[x1] * s[x2]
    full = sm.OLS(a.stock, sm.add_constant(np.column_stack([a[x1], a[x2], a[x1] * a[x2]]))).fit()
    m = a.groupby("loc").agg(**{x1: (x1, "mean"), x2: (x2, "mean"), "stock": ("stock", "mean")})
    ms = sm.OLS(m.stock, sm.add_constant(np.column_stack([m[x1], m[x2], m[x1] * m[x2]]))).fit()
    b, pm = mixed(f"stock ~ {x1} + {x2} + ia", "ia", s); bz, pz = mixed(f"stock ~ {x1} + {x2} + ia + C(zone)", "ia", s)
    rows.append(dict(test=tid, var=f"{x1}x{x2}", target="stock", pooled_r=np.nan, pooled_p=full.pvalues[3],
                     site_r=np.nan, site_p=ms.pvalues[3], mixed_beta=b, mixed_p=pm, mixedzone_beta=bz, mixedzone_p=pz,
                     nonTidal_site_r=np.nan, nonTidal_site_p=np.nan))
T = pd.DataFrame(rows).set_index("test").loc[["H1.1","H1.2","H1.3","H1.4","H1.5","H1.6","H1.7","H1.8","H1.9","H1.10"]]
for c in [c for c in T.columns if c.endswith("_p")]: T[c + "<.005"] = T[c] < 0.005
pd.set_option("display.width", 250)
print(T[["var","pooled_r","pooled_p","site_r","site_p","mixed_beta","mixed_p","mixedzone_beta","mixedzone_p","nonTidal_site_r","nonTidal_site_p"]].round(4).to_string())
print("\nPass Bonferroni 0.05/10 = 0.005:")
for c in ["pooled_p","site_p","mixed_p","mixedzone_p"]: print(f"  {c:12s}", int((T[c] < 0.005).sum()), "of 10:", list(T.index[T[c] < 0.005]))
print("Pass 0.002 (published threshold), pooled:", int((T.pooled_p < 0.002).sum()), list(T.index[T.pooled_p < 0.002]))
T.to_csv("a_site_level_table1.csv")

# EC within zones: mixed model with zone-specific EC slopes; and site means within zone (n = 5)
print("\nEC slope on stock within each zone (mixed, random intercept per location; unstandardized, t C/ha per dS/m):")
m = smf.mixedlm("stock ~ C(zone) + C(zone):EC - 1", a, groups=a["loc"]).fit(reml=False, method="lbfgs")
for z in Z:
    k = f"C(zone)[{z}]:EC"; ci = m.conf_int().loc[k]
    sm5 = a[a.zone == z].groupby("loc")[["EC", "stock"]].mean(); r5 = stats.pearsonr(sm5.EC, sm5.stock)
    print(f"  {z:9s} slope {m.params[k]:8.1f}  95% CI [{ci[0]:7.1f}, {ci[1]:7.1f}]  p {m.pvalues[k]:.3f}   site-mean r (n=5) {r5[0]:+.2f} p {r5[1]:.2f}")
