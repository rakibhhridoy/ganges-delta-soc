"""
PROVENANCE: how data/h4_delta_samples.csv was produced from WoSIS 2023 December.

This is NOT needed to run the analyses (the extracted CSV is cached in ../data/).
It documents the exact, reproducible extraction so a reader can regenerate the
cross-delta input from the raw public snapshot.

Steps to reproduce from scratch:
  1. Download WoSIS 2023 December (456 MB, CC BY 4.0, Batjes et al. 2024) from
     https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip
     (sha256 3115c6be84613dc960c600768a9999dd6d6ad5207017065196b7e356d3c50f3a)
  2. Unzip so that the .tsv tables sit in WOSIS_DIR (set below).
  3. python3 wosis_extraction.py   ->  ../data/h4_delta_samples.csv

WoSIS units: orgc value_avg = g/kg (organic carbon); clay/silt/sand = g/100 g (%);
phaq = pH (water). Depths in cm. We keep complete-case top-metre (<100 cm) layers
with SOC + pH + clay + silt + sand, within each delta bounding box.
"""
from pathlib import Path
import pandas as pd

WOSIS_DIR = Path("WoSIS_2023_December")  # point at the unzipped snapshot
OUT = Path(__file__).resolve().parent.parent / "data" / "h4_delta_samples.csv"

# 14 deltas with adequate co-measured predictor coverage (lat0, lat1, lon0, lon1).
DELTAS = {
    "Mississippi": (28, 33, -93.5, -88), "Rhine-Meuse": (50.5, 53.5, 3, 7.5),
    "Niger": (4, 6.5, 4.5, 7.5), "Yellow River": (34, 40, 116, 122),
    "Limpopo": (-25.5, -24.5, 33, 34.2), "Chao Phraya": (13, 15, 100, 101),
    "Parana-Plata": (-35, -32, -59, -57), "Sacramento": (37.5, 38.5, -122.3, -121.3),
    "Indus": (23, 25.5, 66.5, 69), "Mackenzie": (67.5, 70.5, -137, -132),
    "Danube": (44.3, 45.6, 28, 30.2), "Amazon": (-2.5, 1.5, -52, -48),
    "Senegal": (15.5, 16.5, -16.7, -15.8), "Pearl": (21.8, 23.5, 112.5, 114.2),
}
PREDICTORS = ["ph", "clay", "silt", "sand"]


def load_attr(name, col):
    d = pd.read_csv(WOSIS_DIR / f"wosis_202312_{name}.tsv", sep="\t", low_memory=False,
                    usecols=["profile_id", "layer_id", "value_avg"]).rename(columns={"value_avg": col})
    d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.dropna(subset=[col])


def main():
    sites = pd.read_csv(WOSIS_DIR / "wosis_202312_sites.tsv", sep="\t", low_memory=False,
                        usecols=["site_id", "longitude", "latitude"])
    layers = pd.read_csv(WOSIS_DIR / "wosis_202312_layers.tsv", sep="\t", low_memory=False,
                         usecols=["profile_id", "layer_id", "site_id", "upper_depth", "lower_depth"]).merge(
        sites, on="site_id", how="left")
    layers["upper_depth"] = pd.to_numeric(layers["upper_depth"], errors="coerce")

    attrs = {"soc": load_attr("orgc", "soc"), "ph": load_attr("phaq", "ph"),
             "clay": load_attr("clay", "clay"), "silt": load_attr("silt", "silt"),
             "sand": load_attr("sand", "sand")}
    base = layers[["profile_id", "layer_id", "latitude", "longitude", "upper_depth", "lower_depth"]].copy()
    for k, t in attrs.items():
        base = base.merge(t, on=["profile_id", "layer_id"], how="left")
    base = base[base["upper_depth"] < 100]

    rows = []
    for nm, (a0, a1, o0, o1) in DELTAS.items():
        sub = base[(base.latitude.between(a0, a1)) & (base.longitude.between(o0, o1))]
        sub = sub.dropna(subset=["soc"] + PREDICTORS).copy()
        sub["delta"] = nm
        rows.append(sub.rename(columns={"latitude": "lat", "longitude": "lon",
                                        "upper_depth": "depth_top_cm", "lower_depth": "depth_bot_cm"})
                    [["delta", "lat", "lon", "depth_top_cm", "depth_bot_cm", "soc", "ph", "clay", "silt", "sand"]])
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} : {len(out)} samples across {out.delta.nunique()} deltas")
    print(out.delta.value_counts().to_dict())


if __name__ == "__main__":
    main()
