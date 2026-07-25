"""
PROVENANCE: how data/h4_delta_samples_POLY.csv was produced (H4 delta-delineation
robustness check; Supplementary Table S11).

NOT needed to run the robustness analysis (the extracted CSV is cached in ../data/
and analysed by scripts/h4_polygon_robustness.py). This documents the exact,
reproducible extraction from the raw public WoSIS snapshot and the published delta
polygons.

Delta regions:
  - 13 of 14 external deltas: the active-delta polygon from Nienhuis et al. (2020,
    Nature 577:514-518; file land_area_change/GlobalDeltaMax100_poly.kml in the
    GlobalDeltaChange repository, https://github.com/jhnienhuis/GlobalDeltaChange),
    matched to each named delta by spatial overlap with its bounding box, then
    buffered outward by 0.5 deg. The 14 matched+buffered polygons are shipped as
    ../data/delta_polygons.geojson.
  - Senegal: absent from the 100-delta set -> 0.5-deg-buffered bounding box.

Steps to reproduce from scratch:
  1. Obtain WoSIS 2023 December (see provenance/wosis_extraction.py for URL + sha256)
     and unzip so the .tsv tables sit in WOSIS_DIR.
  2. Download GlobalDeltaMax100_poly.kml (above) and build the matched, buffered
     polygons -> ../data/delta_polygons.geojson (spatial match: for each delta,
     take the KML polygon whose intersection with the delta bounding box is largest).
  3. python3 polygon_extraction.py  ->  ../data/h4_delta_samples_POLY.csv

Units and filtering are identical to the main (bounding-box) panel: WoSIS orgc
value_avg = g/kg; clay/silt/sand = g/100 g; phaq = pH(water); complete-case
top-metre (<100 cm) layers with SOC + pH + clay + silt + sand. BIS Nederland rows
augmenting Rhine-Meuse (see wosis_extraction.py / h4_bis_robustness.py) are taken
from the main panel and re-clipped to the buffered Rhine-Meuse polygon.
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

HERE = Path(__file__).resolve().parent
WOSIS_DIR = Path("WoSIS_2023_December")            # unzipped snapshot
POLY = HERE.parent / "data" / "delta_polygons.geojson"
MAIN_PANEL = HERE.parent / "data" / "h4_delta_samples.csv"   # for BIS rows
OUT = HERE.parent / "data" / "h4_delta_samples_POLY.csv"
BUF = 0.5
PREDICTORS = ["ph", "clay", "silt", "sand"]
BAD = {"Senegal"}  # box fallback (not in the 100-delta polygon set)


def attr(tbl, col):
    d = pd.read_csv(WOSIS_DIR / f"wosis_202312_{tbl}.tsv", sep="\t", low_memory=False,
                    usecols=["profile_id", "layer_id", "value_avg"]).rename(columns={"value_avg": col})
    d[col] = pd.to_numeric(d[col], errors="coerce"); return d.dropna(subset=[col])


def main():
    reg = gpd.read_file(POLY).set_index("delta")
    reg["geometry"] = [g if d in BAD else g.buffer(BUF) for d, g in reg.geometry.items()]
    reg = reg.reset_index()

    sites = pd.read_csv(WOSIS_DIR / "wosis_202312_sites.tsv", sep="\t", low_memory=False,
                        usecols=["site_id", "longitude", "latitude"])
    layers = pd.read_csv(WOSIS_DIR / "wosis_202312_layers.tsv", sep="\t", low_memory=False,
                         usecols=["profile_id", "layer_id", "site_id", "upper_depth", "lower_depth"]).merge(
        sites, on="site_id", how="left")
    layers["upper_depth"] = pd.to_numeric(layers["upper_depth"], errors="coerce")
    base = layers[["profile_id", "layer_id", "latitude", "longitude", "upper_depth", "lower_depth"]].copy()
    for tbl, col in [("orgc", "soc"), ("phaq", "ph"), ("clay", "clay"), ("silt", "silt"), ("sand", "sand")]:
        base = base.merge(attr(tbl, col), on=["profile_id", "layer_id"], how="left")
    base = base[base["upper_depth"] < 100].dropna(subset=["soc"] + PREDICTORS).copy()

    pts = gpd.GeoDataFrame(base, geometry=gpd.points_from_xy(base.longitude, base.latitude), crs="EPSG:4326")
    j = gpd.sjoin(pts, reg[["delta", "geometry"]], predicate="within", how="inner").rename(
        columns={"latitude": "lat", "longitude": "lon", "upper_depth": "depth_top_cm", "lower_depth": "depth_bot_cm"})
    wos = j[["delta", "lat", "lon", "depth_top_cm", "depth_bot_cm", "soc", "ph", "clay", "silt", "sand"]].copy()
    wos["source"] = "WoSIS"
    wos = wos.drop_duplicates(subset=["lat", "lon", "depth_top_cm", "soc"])

    # BIS rows re-clipped to buffered Rhine-Meuse polygon
    old = pd.read_csv(MAIN_PANEL)
    bis = old[old.source == "BIS_Nederland"].copy()
    rm = reg.loc[reg.delta == "Rhine-Meuse", "geometry"].iloc[0]
    bis_pts = gpd.GeoDataFrame(bis, geometry=gpd.points_from_xy(bis.lon, bis.lat), crs="EPSG:4326")
    bis_in = bis[bis_pts.within(rm).values][wos.columns].copy()

    out = pd.concat([wos, bis_in], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} profiles across {out.delta.nunique()} deltas")
    print(out.delta.value_counts().to_dict())


if __name__ == "__main__":
    main()
