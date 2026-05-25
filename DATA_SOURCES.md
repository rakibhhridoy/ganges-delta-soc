# Data sources and provenance

All data underlying the manuscript *"Geochemistry-informed prediction and layered
transferability of soil organic carbon in the Ganges Delta"* (Hasan et al.).

## 1. Primary field data (this study)
| File | Description | n |
|------|-------------|---|
| `data/GangesSOC.csv` | 100 soil samples, 20 locations × 5 depths (0–100 cm), four geomorphological zones of the Ganges Delta, dry season Jan–Feb 2025. 14 physicochemical properties (SOC stock, OM, EC, CEC, TN, pH, BD, moisture, P, K, sand/silt/clay). | 100 |
| `data/Tidal.csv`, `data/Active.csv`, `data/Mature.csv`, `data/Moribund.csv` | The same samples split by geomorphological zone (25 each). Authoritative zone assignment (geomorphology, not EC). | 4×25 |
| `data/LocationMerged.csv` | Location-level means used for the study-area figure. | 20 |
| `data/target.csv` | SOC stock target vector (row-aligned to the feature matrix). | 100 |
| `data/features_preprocessed.csv` | Engineered ML feature matrix (soil + spatial + ViT embeddings). **Note:** the OM column is present but is dropped in all modelling scripts (r = 0.996 with SOC = leakage). | 100 |

**Collection:** soil pits (1 × 1 × 1 m), cores + Edelman auger; laboratory methods in
Supplementary Table S4. Coordinates in `data/gis/` shapefiles and `GangesSOC.csv`.

## 2. Geospatial layers (this study + open boundary)
| File | Source | Licence |
|------|--------|---------|
| `data/gis/TidalActiveD1.shp`, `ActiveDeltaD4.shp`, `MoribundDeltaD3.shp`, `GangesDeltaD.shp` | Geomorphological delta zones (upazila polygons), delineated after Hossain et al. (2015) over Bangladesh administrative boundaries (HDX/Bangladesh ADM3). Mature Delta is reconstructed in-script from `GangesDeltaD.shp` minus the other three zones (the standalone `MatureDeltaD2.shp` was corrupted). | Administrative boundaries: Humanitarian Data Exchange, CC BY |
| `data/gis/bgd_adm0.geojson` | Bangladesh national boundary (Fig 1 inset). | **geoBoundaries** gbOpen BGD ADM0, CC BY 4.0 (Runfola et al. 2020), https://www.geoboundaries.org |
| `data/gis/DEM.tif` | Elevation (study-area figure). SRTM-derived. | NASA SRTM, public domain |

## 3. Cross-delta transferability data (H4) — external, public
| Item | Source | Access |
|------|--------|--------|
| `data/h4_delta_samples.csv` | 6,652 complete-case top-metre soil profiles (SOC, pH, sand, silt, clay) across 14 global river deltas, primarily from harmonised WoSIS profiles filtered to delta bounding boxes, with the BIS Nederland national survey augmenting the Rhine-Meuse cell (tagged in the `source` column). **This is the exact analysis input** (derived/cached so the analysis runs without the raw downloads). | derived here |
| WoSIS 2023 December snapshot (primary source of `h4_delta_samples.csv`) | **Batjes et al. (2024)**, World Soil Information Service, ISRIC. | https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip ; DOI 10.17027/isric-wdc-2023-12. CC BY 4.0 |
| BIS Nederland (Rhine-Meuse augmentation) | **Wageningen Environmental Research**, Bodemkundig Informatiesysteem reference profiles. SOM→SOC via van Bemmelen ×0.58, rescaled to g kg⁻¹, RD-New (EPSG:28992)→WGS84, deduplicated against WoSIS. | 4TU.ResearchData, DOI 10.4121/c90215b3-bdc6-4633-b721-4c4a0259d6dc. CC BY 4.0 |

The 14 deltas and their bounding boxes are defined in `provenance/wosis_extraction.py`
(and re-stated in the manuscript Methods / Supplementary). To regenerate
`h4_delta_samples.csv` from scratch: download the WoSIS snapshot, extract, and run
`provenance/wosis_extraction.py`. The raw snapshot (456 MB) is **not** redistributed
here — it is obtained from ISRIC under its own DOI and licence.

**Deltas (n complete-case):** Mississippi 2043, Rhine-Meuse 504, Niger 327,
Yellow River 315, Limpopo 268, Chao Phraya 204, Paraná-Plata 179, Sacramento 178,
Indus 167, Mackenzie 77, Danube 68, Amazon 62, Senegal 57, Pearl 55.
The tropical Asian monsoon deltas most analogous to the Ganges (Mekong, Irrawaddy,
Godavari, Krishna, Red River) are **absent**: WoSIS holds SOC values there but no
co-measured texture/pH, so they cannot enter a feature-based transfer test.

## 4. Remote sensing (study area / context; not redistributed)
Acquired via Google Earth Engine, publicly accessible: Landsat-8, Sentinel-2, MODIS
(NDVI/LST), CHIRPS (precipitation). SoilGrids (Hengl et al. 2017), https://soilgrids.org.
IPCC AR6 regional mid-century warming increments parameterise the G-RothC climate
scenarios (cited in Methods).

## 5. Process-model parameters (G-RothC)
First-order single-pool model; parameters (k0, Q10, T_ref, management factors) and
per-delta warming are listed in `scripts/grothc_model.py` and the manuscript Methods.
