# Changes from version 3 (10.5281/zenodo.21564960) to version 4

## Data
1. **WoSIS rows rebuilt.** v3's `data/h4_delta_samples.csv` held 4,504 WoSIS rows, but the documented
   extraction (`provenance/wosis_extraction.py`) produces 3,528 from the same snapshot. The 954 extra rows
   (Mississippi 834, Indus 60, Rhine-Meuse 60) were removed; WoSIS rows now carry `profile_id`.
2. **Depth.** 532 WoSIS layers began within the top metre but extended below 100 cm (to 250 cm). v4 keeps only
   layers lying wholly within 0-100 cm, as the Methods state.
3. **BIS Nederland.** 157 samples (all Rhine-Meuse) with sand + silt + clay outside 98-102 % were dropped.
   The 2,148 v3 BIS samples are shipped unchanged as `data/bis_nederland_samples.csv`.
4. **Duplicates.** 198 exact duplicates (one profile stored under two WoSIS IDs) were dropped.
5. Result: 4,789 external samples (2,798 WoSIS, 1,991 BIS) at 2,647 sites; v3 had 6,652.
   The polygon panel (`h4_delta_samples_POLY.csv`) received the same depth and texture rules: 1,449 → 1,263.
6. `data/features_preprocessed_WITH_embeddings_backup.csv` (unverifiable columns, not used) was removed.
7. `DATA_SOURCES.md`: corrected the description of `GangesSOC.csv` (it contains no P or K columns).

## Provenance
8. `provenance/wosis_extraction.py` applies the 0-100 cm rule and writes `data/wosis_layers_0_100cm.csv`;
   the new `provenance/assemble_panel.py` adds the BIS samples, applies the texture rule and drops
   duplicates. Together they rebuild `data/h4_delta_samples.csv` exactly.

## Scripts
9. **Process model removed** (`grothc_model.py`, its outputs, `make_fig3.py`, `fig3_climate.png`).
10. **`dl_reconstruction.py`.** v3 fitted the ensemble weights on all out-of-fold predictions, which uses the
    test labels, and printed "gradient boosting is not surpassed" unconditionally. v4 fits the weights for
    each test fold on the other folds only, computes the verdict, and adds 20 random location partitions.
    On v3's single split the v3 ensemble scored 0.66 against GBR 0.60; with cross-fitted weights it scores
    0.52, and over 20 partitions it never beats GBR (median 0.72 vs 0.76).
11. **`revision/`** (new): the analyses requested by the referees; see `revision/README.md`.
12. Cross-delta outputs (`h4_*`, `fig5_transfer.png`, `figS_h4_expanded.png`) regenerated on v4 data. Outputs
    that use only the Ganges field data are unchanged.

## Notebooks
13. Rebuilt and executed on v4 data, with the process-model sections removed and statements that no longer
    hold corrected (within-delta +0.25; engineered-feature gain; "consistent analytical methods").
