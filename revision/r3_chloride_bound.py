"""Upper bound on Walkley-Black chloride interference, from EC alone (RC1 point 3, RC2 G5).
The laboratory added Ag2SO4 to the digestion, which precipitates chloride; this script bounds the effect
had it not. Worst case: dissolved salt = 640 mg/L per dS/m in the 1:5 extract, all of it NaCl
(35.453/58.443 Cl by mass): 388 mg Cl/L per dS/m, x5 dilution = 0.194 % Cl in soil per dS/m. This exceeds
the ~0.15 % implied by NaCl's own molar conductivity, so it is the more conservative bound.
Each g Cl oxidised to Cl2 consumes dichromate equivalent to 12.011/(4*35.453) = 0.0847 g C; the
Walkley-Black recovery factor 1.32 is applied, as it is to SOC."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(D / "GangesSOC.csv"); df.columns = [c.strip() for c in df.columns]
zone = {}
for z in ["Active", "Mature", "Moribund", "Tidal"]:
    t = pd.read_csv(D / f"{z}.csv"); t.columns = [c.strip() for c in t.columns]
    for s, l in zip(t.Sample, t.Location): zone[(s, l)] = z
df["Zone"] = [zone[(s, l)] for s, l in zip(df.Sample, df.Location)]
CL_PER_EC = 640 * (35.453 / 58.443) * 5 / 1e4   # % Cl per dS/m = 0.194
C_PER_CL = 12.011 / (4 * 35.453)
df["Cl_max"] = CL_PER_EC * df.EC
df["bias"] = 1.32 * C_PER_CL * df.Cl_max          # % C (same unit as SOCC)
df["bias_pct"] = 100 * df.bias / df.SOCC
df["SOCC_corr"] = df.SOCC - df.bias
print("Worst-case bias as % of SOCC, by zone:")
print(df.groupby("Zone").bias_pct.agg(["mean", "max"]).round(1).to_string())
print("Negative corrected SOCC:", int((df.SOCC_corr <= 0).sum()))
for col in ["SOCC", "SOCC_corr"]:
    print(f"EC vs {col:9s} r = {stats.pearsonr(df.EC, df[col])[0]:.3f}")
