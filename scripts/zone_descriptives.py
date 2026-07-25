#!/usr/bin/env python3
"""Descriptive statistics of key soil properties by geomorphological zone.

Reproduces Supplementary Table S2 and the between-zone EC contrasts quoted in
Section 2.1 of the manuscript. Writes zone_descriptives.csv and prints a LaTeX
tabular body so the table cannot drift from the data again.

Run from scripts/:  python3 zone_descriptives.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent

ZONES = {
    "Tidal Active": "Tidal.csv",
    "Active Delta": "Active.csv",
    "Mature Delta": "Mature.csv",
    "Moribund Delta": "Moribund.csv",
}

# source column -> (display label, decimal places)
PROPS = {
    "SOC Stock": ("SOC (t C ha$^{-1}$)", 1),
    "EC": ("EC (dS m$^{-1}$)", 2),
    "CEC": ("CEC (cmol$_c$ kg$^{-1}$)", 1),
    "TN": ("TN (\\%)", 2),
    "pH": ("pH", 2),
    "Bulk Density": ("BD (g cm$^{-3}$)", 2),
    "Moisture": ("Moisture (\\%)", 1),
    "Clay": ("Clay (\\%)", 1),
}


def load(fname: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / fname)
    d.columns = [c.strip() for c in d.columns]
    return d


def main() -> None:
    frames = {z: load(f) for z, f in ZONES.items()}
    n = {z: len(d) for z, d in frames.items()}

    records = []
    for src, (label, dp) in PROPS.items():
        row = {"property": label}
        for z, d in frames.items():
            v = pd.to_numeric(d[src], errors="coerce").dropna()
            row[f"{z} mean"] = round(v.mean(), dp)
            row[f"{z} sd"] = round(v.std(ddof=1), dp)
            row[f"{z} min"] = round(v.min(), dp)
            row[f"{z} max"] = round(v.max(), dp)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(OUT / "zone_descriptives.csv", index=False)

    print("n per zone:", n)
    print("\n% --- LaTeX tabular body for Supplementary Table S2 ---")
    for src, (label, dp) in PROPS.items():
        cells = []
        for z, d in frames.items():
            v = pd.to_numeric(d[src], errors="coerce").dropna()
            cells.append(f"${v.mean():.{dp}f} \\pm {v.std(ddof=1):.{dp}f}$")
        print(f"{label:26s} & " + " & ".join(cells) + " \\\\")

    print("\n% --- EC by zone (Section 2.1 contrasts) ---")
    for z, d in frames.items():
        v = pd.to_numeric(d["EC"], errors="coerce").dropna()
        print(f"  {z:16s} mean {v.mean():5.2f}   range {v.min():.2f}--{v.max():.2f} dS/m")

    print("\nwrote zone_descriptives.csv")


if __name__ == "__main__":
    main()
