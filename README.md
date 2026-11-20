# Ganges Delta SOC: code and data (version 4)

Code and data for *"Geochemical predictability quantification and spatial transferability of soil organic
carbon in Ganges Delta"* (Hasan et al.), in public discussion at **SOIL** as egusphere-2026-4799.

Version 4 accompanies the authors' replies to the referees. What changed from version 3 is listed in
`CHANGELOG_v4.md`. In short: the cross-delta panel was rebuilt (layers wholly within 0-100 cm, 954 rows that
the documented extraction does not produce removed, BIS rows with impossible texture removed, exact
duplicates removed); the process model (G-RothC) was removed; the deep-learning ensemble is now weighted
without using test labels; and the analyses requested by the referees were added in `revision/`.

## Deposit
- **Code: GitHub** (this repository): <https://github.com/rakibhhridoy/ganges-delta-soc>. Everything except the
  large `data/gis/DEM.tif`.
- **Code and data: Zenodo v4** (DOI <https://doi.org/10.5281/zenodo.24398898>): the full bundle including the DEM.
  `scripts/zone_descriptives.py` (SI Table S2) and `CITATION.cff` are in this repository but were not included in
  the Zenodo v4 archive.

## Environment
```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # Python 3.11; versions pinned
```

## Reproduce the analyses
Run from `scripts/` (each writes its outputs there):
```
cd scripts
python3 h1_hypothesis_tests.py     # H1 Table 1: correlations, interactions, pH quadratic, salinity, diagnostics
python3 zone_descriptives.py       # SI Table S2: per-zone descriptive statistics
python3 ml_spatial_cv.py           # H3 location-grouped spatial CV -> ml_spatial_cv_results.csv
python3 engineered_ablation.py     # H3 engineered-feature ablation
python3 per_zone_correlations.py   # H1 per-zone correlations -> Supplementary Table S10
python3 dl_reconstruction.py       # H3 deep-learning reconstruction (needs torch + torch_geometric; see note)
python3 h4_analysis.py             # H4 transfer matrix, forward/reverse/LODO -> h4_results.json
python3 h4_bis_robustness.py       # H4 robustness: WoSIS-only vs BIS-augmented Rhine-Meuse
python3 h4_expanded_analysis.py    # H4 distance-decay, feature universality, few-shot recovery
python3 h4_polygon_robustness.py   # H4 robustness: delta polygons vs bounding boxes
python3 make_fig4.py               # Fig 4 (ML/DL)
python3 make_fig5.py               # Fig 5 (transferability) [needs h4_analysis.py outputs]
python3 make_figS_h4expanded.py    # SI figure [needs h4_expanded_analysis.py outputs]
python3 make_fig6.py               # SI spatial figure
python3 create_study_area_figure.py  # Fig 1 (study area) [needs data/gis/, incl. DEM.tif]
```
The analyses added for the referees are in `revision/`; see `revision/README.md` for the mapping from
each script to the referee point it answers.

**`dl_reconstruction.py`: provenance.** The original deep-learning training code was not preserved; this
script is an independent re-implementation from the architecture description in the Methods. Deep models
are high-variance at n = 100, so the script reports results over 20 random location partitions. In v4 the
ensemble weights applied to each test fold are fitted on the other folds only, and the verdict is computed
from the results.

## Cross-delta panel
`data/h4_delta_samples.csv` (4,789 samples, 14 deltas) combines WoSIS 2023 (Batjes et al., 2024) and the
BIS Nederland survey for Rhine-Meuse. It is rebuilt from the sources by
`provenance/wosis_extraction.py` (needs the WoSIS snapshot; writes `data/wosis_layers_0_100cm.csv`) followed
by `provenance/assemble_panel.py` (adds `data/bis_nederland_samples.csv`, drops BIS samples whose
sand + silt + clay lies outside 98-102 %, and drops exact duplicates). Units were harmonised across sources;
laboratory methods were not.

## Notebooks
`notebooks/01_run_all.ipynb` regenerates the analyses with the code shown inline; `02_tutorial.ipynb` adds
explanatory narrative. Both are built by `notebooks/_build_notebooks.py` and shipped executed on v4 data.

## Key results (v4 data)
| Result | Value | Script |
|--------|-------|--------|
| Location-grouped CV R² (fixed split) | 0.64 (random 5-fold 0.91; LOLO 0.87; leave-zone-out −0.37) | ml_spatial_cv.py |
| Location-grouped 5-fold, 100 random location partitions | median about 0.73 (0.47-0.89) | revision/r4_ml_validation.py |
| Cross-delta transfer (210 pairs) | median R² −0.53; 7.6 % positive | h4_analysis.py |
| Same-target: transfer vs local model, MAE ratio | 1.38; transfer better in 20 % of pairs | revision/r1_same_target.py |
| Within-delta skill, sites held out (median) | R² about 0 | revision/r1_same_target.py |
| Reverse pooled (4,789) → Ganges | R² −0.16 | h4_analysis.py |
| Delta polygons vs boxes (robustness) | cross-delta median −0.73; 5.2 % positive | h4_polygon_robustness.py |
| Deep learning vs GBR, 20 location partitions | no model or ensemble beats GBR | dl_reconstruction.py |

## Notes
- `data/features_preprocessed.csv` holds the 14 base predictors (9 soil + 5 spatial). The v3 backup matrix
  with unverifiable embedding columns has been removed.
- Fig 2 (geochemistry correlations) has no generator script; it is a static image in the manuscript.
- Random seeds are fixed (`random_state = 42`).
- Data provenance and licences: `DATA_SOURCES.md`.
