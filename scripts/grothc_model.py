"""
G-RothC v2 — corrected first-order single-pool SOC process model
================================================================
Rebuild of the Ganges-Delta process model for the STOTEN-rejection revision.

WHY THIS EXISTS
---------------
The original G-RothC implementation and its climate-forcing time series were not
preserved in the project tree (verified 2026-05-22). This module re-implements the
model from the equation and parameters reported in the manuscript, fixing the
reviewer-identified defects:

  * R3#4  decay is now genuinely FIRST ORDER  (loss term proportional to SOC)
  * R3#11 management acts on BOTH carbon input and decomposition (author's choice)
  * R3#6  C_in is a calibrated carbon-input FLUX (t C ha^-1 yr^-1), back-calculated
          from the steady-state condition, not a raw dimensionless NDVI value
  * R3#7  no circular "hindcast R^2" is reported; the model is initialised to the
          observed stocks (single-date data) and used for scenario / sensitivity
          analysis only. This limitation is stated explicitly in the manuscript.

GOVERNING EQUATION (per sample i)
---------------------------------
    dSOC_i/dt = I_i(t)  -  D_i(t) * SOC_i

    I_i(t) = C_in,i * f_moist_i * f_EC(zone_i) * f_mgmt_in(scenario, t)        [input]
    D_i(t) = k0 * f_mgmt_dec(scenario, t) * Q10**((T(t) - T_ref)/10)          [rate]

Steady-state calibration at t0 (year 2000, BAU climate, BAU management):
    0 = I0_i - D0_i * SOC_obs_i   ->   C_in,i = D0_i * SOC_obs_i / (f_moist_i*f_EC_i)
    with D0_i = k0 * Q10**((T_base - T_ref)/10)

Because f_moist, f_EC are constant across our (climate x management) scenarios they
cancel in the calibrated forward runs: the only drivers of change are temperature
(via Q10) and management (via the two f_mgmt factors). The model therefore cleanly
isolates the climate vs management contributions that the manuscript reports.

Author parameters are taken from the manuscript; climate-forcing increments are
documented IPCC AR6 South Asia mid-century values (the original site series are gone).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
DATA = Path(__file__).resolve().parent.parent / "data"     # .../Ganges/data
OUT  = Path(__file__).resolve().parent                  # StotenRejectUpdate/

# ----------------------------------------------------------------------------- params
K0       = 0.05      # base decomposition rate (yr^-1)            [manuscript]
Q10      = 2.4       # temperature sensitivity                    [manuscript, Lloyd1994]
T_REF    = 25.0      # reference temperature (deg C)              [manuscript]
T_BASE   = 26.0      # study-area mean annual temperature (deg C) [manuscript study area]

# Warming applied by 2050 relative to 2000 baseline (deg C), IPCC AR6 South Asia.
WARMING_2050 = {"RCP4.5": 1.0, "RCP8.5": 2.0}
YEAR0, YEAR_HIND, YEAR_END = 2000, 2026, 2050

# Management factors (both within the manuscript's 0.7-1.2 envelope), acting on
# BOTH input and decomposition (author's choice):
MGMT = {
    #             f_in   f_dec
    "BAU":       (1.00,  1.00),
    "mangrove":  (1.15,  0.80),   # more litter input + anoxic suppression of decomposition
    "aquaculture": (0.85, 1.15),  # reduced input + enhanced (oxic/disturbed) decomposition
}

# Zone-level salinity preservation factor on the input/retention term (descriptive;
# cancels in the calibrated runs because salinity is held fixed across scenarios).
F_EC = {"Tidal": 1.30, "Active": 1.00, "Mature": 0.95, "Moribund": 0.95}

ZONE_FILES = {"Tidal": "Tidal.csv", "Active": "Active.csv",
              "Mature": "Mature.csv", "Moribund": "Moribund.csv"}


# ----------------------------------------------------------------------------- data
def load_samples():
    """Load the 100 samples with authoritative geomorphological zone labels."""
    frames = []
    for zone, fname in ZONE_FILES.items():
        df = pd.read_csv(DATA / fname)
        df.columns = [c.strip() for c in df.columns]
        df = df[["SOC Stock", "EC", "Moisture", "Location", "Depth"]].copy()
        df["zone"] = zone
        frames.append(df)
    s = pd.concat(frames, ignore_index=True)
    s = s.rename(columns={"SOC Stock": "SOC_obs"})
    s = s.dropna(subset=["SOC_obs", "Moisture"])
    s["SOC_obs"] = s["SOC_obs"].astype(float)
    # moisture factor, normalised to mean 1 (saturating, bounded)
    s["f_moist"] = np.clip(s["Moisture"] / s["Moisture"].mean(), 0.6, 1.4)
    s["f_EC"] = s["zone"].map(F_EC)
    # brackish vs freshwater split (manuscript EC>0.5 dS/m), for reporting only
    s["salinity_class"] = np.where(s["EC"] > 0.5, "Brackish", "Freshwater")
    return s.reset_index(drop=True)


# ----------------------------------------------------------------------------- model
def temperature(year, pathway):
    """Linear warming from T_BASE at YEAR0 to T_BASE+WARMING_2050 at YEAR_END."""
    frac = (year - YEAR0) / (YEAR_END - YEAR0)
    return T_BASE + WARMING_2050[pathway] * frac


def calibrate_input(s):
    """Back-calculate the steady-state carbon-input flux C_in (t C ha^-1 yr^-1)."""
    D0 = K0 * Q10 ** ((T_BASE - T_REF) / 10.0)            # baseline decay rate
    C_in = D0 * s["SOC_obs"].values / (s["f_moist"].values * s["f_EC"].values)
    return C_in, D0


def simulate(s, C_in, pathway, mgmt_fraction, k0=K0, q10=Q10, dt=1.0):
    """Euler-integrate SOC for every sample to YEAR_END. Returns final SOC array.

    mgmt_fraction: fraction of samples converted to mangrove (rest BAU). The most
    saline samples are restored first (mangrove restoration targets saline/aquaculture
    land), matching the manuscript's framing.
    """
    n = len(s)
    soc = s["SOC_obs"].values.astype(float).copy()
    # assign management per sample
    order = np.argsort(-s["EC"].values)          # most saline first
    n_mangrove = int(round(mgmt_fraction * n))
    is_mangrove = np.zeros(n, dtype=bool)
    is_mangrove[order[:n_mangrove]] = True
    f_in  = np.where(is_mangrove, MGMT["mangrove"][0], MGMT["BAU"][0])
    f_dec = np.where(is_mangrove, MGMT["mangrove"][1], MGMT["BAU"][1])

    base_input = C_in * s["f_moist"].values * s["f_EC"].values   # = D0*SOC_obs
    years = np.arange(YEAR0, YEAR_END + 1)
    for yr in years[1:]:
        T = temperature(yr, pathway)
        D = k0 * f_dec * q10 ** ((T - T_REF) / 10.0)
        I = base_input * f_in
        soc = soc + dt * (I - D * soc)
        soc = np.clip(soc, 0.0, None)
    return soc


def simulate_trajectory(s, C_in, pathway, mgmt_fraction, k0=K0, q10=Q10):
    """Like simulate() but returns (years, mean_SOC_per_year) for plotting."""
    n = len(s)
    soc = s["SOC_obs"].values.astype(float).copy()
    order = np.argsort(-s["EC"].values)
    is_m = np.zeros(n, dtype=bool)
    is_m[order[:int(round(mgmt_fraction * n))]] = True
    f_in = np.where(is_m, MGMT["mangrove"][0], MGMT["BAU"][0])
    f_dec = np.where(is_m, MGMT["mangrove"][1], MGMT["BAU"][1])
    base_input = C_in * s["f_moist"].values * s["f_EC"].values
    years = np.arange(YEAR0, YEAR_END + 1)
    traj = [float(soc.mean())]
    for yr in years[1:]:
        T = temperature(yr, pathway)
        D = k0 * f_dec * q10 ** ((T - T_REF) / 10.0)
        soc = np.clip(soc + (base_input * f_in - D * soc), 0.0, None)
        traj.append(float(soc.mean()))
    return years, np.array(traj)


# ----------------------------------------------------------------------------- runs
def main():
    s = load_samples()
    C_in, D0 = calibrate_input(s)
    base_mean = s["SOC_obs"].mean()

    pathways = ["RCP4.5", "RCP8.5"]
    fractions = {"BAU": 0.0, "10pct_mangrove": 0.10, "50pct_mangrove": 0.50}

    rows = []
    grid = {}
    for pw in pathways:
        for label, frac in fractions.items():
            soc_final = simulate(s, C_in, pw, frac)
            m = float(np.mean(soc_final))
            grid[(pw, label)] = soc_final
            rows.append({"pathway": pw, "land_use": label,
                         "mean_SOC_2050": round(m, 2),
                         "pct_change_vs_baseline": round(100 * (m - base_mean) / base_mean, 2)})
    res = pd.DataFrame(rows)

    # management vs climate effect (hold one fixed)
    mgmt_effect = np.mean(grid[("RCP4.5", "50pct_mangrove")]) - np.mean(grid[("RCP4.5", "BAU")])
    clim_effect = np.mean(grid[("RCP4.5", "BAU")]) - np.mean(grid[("RCP8.5", "BAU")])
    ratio = abs(mgmt_effect) / abs(clim_effect) if clim_effect else float("nan")

    # per salinity-class change rate (t C ha^-1 yr^-1) under RCP4.5 BAU
    soc_bau = grid[("RCP4.5", "BAU")]
    rate = (soc_bau - s["SOC_obs"].values) / (YEAR_END - YEAR0)
    by_class = pd.DataFrame({"salinity_class": s["salinity_class"], "rate": rate}) \
        .groupby("salinity_class")["rate"].mean()

    # sensitivity: 2050 RCP4.5 BAU mean SOC across k0 and Q10 ranges
    k0_range = np.linspace(0.01, 0.05, 9)
    q10_range = np.linspace(1.5, 3.5, 9)
    k0_means = [np.mean(simulate(s, calibrate_input_for(s, k0=k)[0], "RCP4.5", 0.0, k0=k)) for k in k0_range]
    q10_means = [np.mean(simulate(s, calibrate_input_for(s, q10=q)[0], "RCP4.5", 0.0, q10=q)) for q in q10_range]
    k0_span = (max(k0_means) - min(k0_means)) / np.mean(k0_means) * 100
    q10_span = (max(q10_means) - min(q10_means)) / np.mean(q10_means) * 100

    # 100-member parameter ensemble -> 90% PI for RCP4.5 BAU 2050 mean SOC
    ens = []
    for _ in range(100):
        k = RNG.uniform(0.01, 0.05); q = RNG.uniform(1.5, 3.5)
        C, _d = calibrate_input_for(s, k0=k, q10=q)
        ens.append(np.mean(simulate(s, C, "RCP4.5", 0.0, k0=k, q10=q)))
    pi = np.percentile(ens, [5, 95])

    # also management-positive fraction across ensemble
    pos = 0
    for _ in range(100):
        k = RNG.uniform(0.01, 0.05); q = RNG.uniform(1.5, 3.5)
        C, _d = calibrate_input_for(s, k0=k, q10=q)
        d = np.mean(simulate(s, C, "RCP4.5", 0.50, k0=k, q10=q)) - np.mean(simulate(s, C, "RCP4.5", 0.0, k0=k, q10=q))
        pos += d > 0

    # multi-delta management:climate ratios (H4.5)
    md = multidelta_ratios(S0=50.0)
    md_check = multidelta_ratios(S0=150.0)   # confirm independence of initial SOC
    md_vals = list(md.values())

    # ----- write outputs
    res.to_csv(OUT / "grothc_results.csv", index=False)
    pd.DataFrame([{"delta": k, "mgmt_climate_ratio": round(v, 2)} for k, v in md.items()]) \
        .to_csv(OUT / "grothc_multidelta_ratios.csv", index=False)
    summary = []
    summary.append("G-RothC v2 (corrected first-order) — results")
    summary.append("=" * 48)
    summary.append(f"n samples            : {len(s)}")
    summary.append(f"baseline mean SOC    : {base_mean:.2f} t C ha^-1")
    summary.append(f"baseline decay D0    : {D0:.4f} yr^-1   (mean residence ~{1/D0:.1f} yr)")
    summary.append(f"calibrated C_in flux : mean {np.mean(C_in):.2f} (range {np.min(C_in):.2f}-{np.max(C_in):.2f}) t C ha^-1 yr^-1")
    summary.append("")
    summary.append("2050 mean SOC by scenario (t C ha^-1):")
    for _, r in res.iterrows():
        summary.append(f"  {r['pathway']:7s} {r['land_use']:16s} : {r['mean_SOC_2050']:7.2f}  ({r['pct_change_vs_baseline']:+.2f}% vs baseline)")
    summary.append("")
    summary.append(f"Management effect (50% mangrove vs BAU, RCP4.5) : {mgmt_effect:+.2f} t C ha^-1")
    summary.append(f"Climate effect    (RCP4.5 vs RCP8.5, BAU)       : {clim_effect:+.2f} t C ha^-1")
    summary.append(f"Management : Climate ratio                       : {ratio:.2f}x")
    summary.append("")
    summary.append("Change rate by salinity class under RCP4.5 BAU (t C ha^-1 yr^-1):")
    for cls, v in by_class.items():
        summary.append(f"  {cls:11s}: {v:+.3f}")
    summary.append("")
    summary.append(f"Sensitivity (RCP4.5 BAU 2050 mean SOC):")
    summary.append(f"  k0  (0.01-0.05) span : {k0_span:.1f}% of mean")
    summary.append(f"  Q10 (1.5-3.5)  span : {q10_span:.1f}% of mean")
    summary.append(f"  -> dominant driver  : {'k0' if k0_span>q10_span else 'Q10'}")
    summary.append("")
    summary.append(f"100-member ensemble 90% PI (RCP4.5 BAU 2050 mean SOC): {pi[0]:.1f}-{pi[1]:.1f} t C ha^-1")
    summary.append(f"Management signal positive in {pos}/100 ensemble members")
    summary.append("")
    summary.append("Multi-delta management:climate ratio (H4.5; per-delta IPCC AR6 forcing):")
    for k, v in md.items():
        summary.append(f"  {k:13s}: {v:.2f}x")
    summary.append(f"  range across 5 deltas : {min(md_vals):.1f}-{max(md_vals):.1f}x")
    summary.append(f"  (stock-independence check: S0=50 vs S0=150 identical = "
                   f"{all(abs(md[k]-md_check[k])<1e-6 for k in md)})")
    text = "\n".join(summary)
    (OUT / "grothc_results_summary.txt").write_text(text + "\n")
    print(text)


def calibrate_input_for(s, k0=K0, q10=Q10):
    """Calibration with overridable k0/Q10 (for sensitivity & ensemble runs)."""
    D0 = k0 * q10 ** ((T_BASE - T_REF) / 10.0)
    C_in = D0 * s["SOC_obs"].values / (s["f_moist"].values * s["f_EC"].values)
    return C_in, D0


# --------------------------------------------------------------- multi-delta (H4.5)
# Per-delta climate forcing: baseline mean annual T and mid-century (2050) warming
# under RCP4.5 / RCP8.5, from IPCC AR6 regional projections. The management:climate
# ratio is independent of absolute initial SOC (both effects scale with the stock),
# so only the climate forcing and the shared management mechanism differ across deltas.
DELTAS = {
    "Ganges":       dict(T_base=26.0, w45=1.0, w85=2.0),  # tropical monsoon, S Asia
    "Mississippi":  dict(T_base=20.0, w45=1.2, w85=2.5),  # subtropical humid, N America
    "Mekong":       dict(T_base=27.0, w45=1.0, w85=1.9),  # tropical monsoon, SE Asia
    "Rhine-Meuse":  dict(T_base=10.0, w45=1.3, w85=2.8),  # temperate maritime, W Europe
    "Yellow River": dict(T_base=14.0, w45=1.5, w85=3.0),  # temperate continental, E Asia
}


def _project_single(S0, T_base, warming, f_in, f_dec, k0=K0, q10=Q10):
    """Integrate one representative stock to YEAR_END under given forcing/management."""
    D0 = k0 * q10 ** ((T_base - T_REF) / 10.0)
    base_input = D0 * S0                     # steady-state input at baseline
    soc = float(S0)
    for yr in range(YEAR0 + 1, YEAR_END + 1):
        T = T_base + warming * (yr - YEAR0) / (YEAR_END - YEAR0)
        D = k0 * f_dec * q10 ** ((T - T_REF) / 10.0)
        soc += (base_input * f_in) - D * soc
    return max(soc, 0.0)


def multidelta_ratios(S0=50.0):
    """Management:climate ratio per delta (ratio is independent of S0)."""
    fin_m, fdec_m = MGMT["mangrove"]
    out = {}
    for name, c in DELTAS.items():
        bau45 = _project_single(S0, c["T_base"], c["w45"], 1.0, 1.0)
        bau85 = _project_single(S0, c["T_base"], c["w85"], 1.0, 1.0)
        man45 = _project_single(S0, c["T_base"], c["w45"], fin_m, fdec_m)
        mgmt_eff = man45 - bau45
        clim_eff = bau45 - bau85
        out[name] = abs(mgmt_eff) / abs(clim_eff) if clim_eff else float("nan")
    return out


if __name__ == "__main__":
    main()
