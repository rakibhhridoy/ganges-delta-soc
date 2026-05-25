"""Build two notebooks from the reproducibility scripts:

  01_reproducibility.ipynb — clean run-all for peer review / Zenodo
  02_tutorial.ipynb        — same analyses with didactic narrative +
                              extensible exploration cells at the end

Both notebooks run from /scripts paths (so a reader can place this file
anywhere under reproducibility/notebooks/ and run it).
"""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent

def md(text):  return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

SCRIPTS_DIR = HERE.parent / "scripts"

def inline_script(fname, call_main=False, display=None):
    """Embed a script's source verbatim as a code cell, so the analysis is visible
    *inline* rather than hidden behind `import <script>`. Rewrites the script's
    `Path(__file__)`-relative DATA/OUT so they resolve from the notebook's cwd
    (the preamble chdir's to scripts/), drops the standalone `__main__` guard, and
    optionally calls main() / appends a display expression for a rich table."""
    src = (SCRIPTS_DIR / fname).read_text()
    src = src.replace("from __future__ import annotations\n", "")
    src = src.replace("Path(__file__).resolve().parent.parent", "Path.cwd().parent")
    src = src.replace("Path(__file__).resolve().parent", "Path.cwd()")
    src = src.split("\nif __name__ ==")[0].rstrip()
    if call_main:
        src += "\n\nmain()"
    if display:
        src += f"\n\n# --- display the result table inline ---\n{display}"
    return src

# ────────────────────────────────────────────────── shared analysis cells
# Analysis cells embed the real script source (single source of truth, no drift);
# figures are rendered by the plotting scripts via fig_cell().

PREAMBLE_CODE = """\
# Environment + path setup. The notebook lives in reproducibility/notebooks/;
# inputs are in ../data/ and the analysis scripts are in ../scripts/.
import sys, os, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove for inline interactive figures
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
while ROOT.name and not (ROOT / "scripts").is_dir() and not (ROOT / "data").is_dir():
    ROOT = ROOT.parent
SCRIPTS = ROOT / "scripts"
DATA    = ROOT / "data"
print(f"ROOT    = {ROOT}")
print(f"DATA    = {DATA}")
print(f"SCRIPTS = {SCRIPTS}")

# Make scripts importable AND chdir there so the scripts' own relative paths
# resolve as if they were being run from scripts/.
sys.path.insert(0, str(SCRIPTS))
os.chdir(SCRIPTS)
print(f"cwd     = {os.getcwd()}")

# Pin deterministic state
np.random.seed(42)

# Common plotting defaults
plt.rcParams.update({"font.size": 10, "figure.dpi": 110, "savefig.dpi": 200,
                     "axes.titleweight": "bold"})
"""

LOAD_DATA_CODE = """\
# Load the primary field dataset (100 samples × 14 properties).
soc = pd.read_csv(DATA / "GangesSOC.csv")
soc.columns = [c.strip() for c in soc.columns]
print(f"GangesSOC.csv: {len(soc)} samples × {len(soc.columns)} variables")
print("first columns:", list(soc.columns)[:10])

# Zone-level files (5 locations per zone, 25 samples per zone, 4 zones)
zones = {}
for z, f in [("Tidal Active","Tidal.csv"), ("Active Delta","Active.csv"),
             ("Mature Delta","Mature.csv"), ("Moribund Delta","Moribund.csv")]:
    zc = pd.read_csv(DATA / f); zc.columns = [c.strip() for c in zc.columns]
    zones[z] = zc
print("samples per zone:", {z: len(d) for z, d in zones.items()})
"""

GROTHC_RUN = inline_script("grothc_model.py", call_main=True,
                           display='pd.read_csv("grothc_results.csv")')

ML_CV_RUN = inline_script("ml_spatial_cv.py",
                          display='pd.read_csv("ml_spatial_cv_results.csv")')

ENG_ABL_RUN = inline_script("engineered_ablation.py",
                            display='pd.read_csv("engineered_ablation_results.csv")')

PERZONE_RUN = inline_script("per_zone_correlations.py",
                            display='pd.read_csv("per_zone_correlations.csv")')

H4_RUN = inline_script("h4_analysis.py", call_main=True,
                       display='pd.read_csv("h4_transfer_matrix.csv", index_col=0).round(2)')

def fig_cell(script_name, png_name, caption):
    """Run a figure script and embed its PNG."""
    return code(f"""\
# {caption}
import importlib, {script_name[:-3]}
importlib.reload({script_name[:-3]})
from IPython.display import Image
Image("{png_name}", width=720)""")

# ──────────────────────────────────────────────── notebook 1: reproducibility

n1 = nbf.v4.new_notebook()
n1.cells = [
    md("# 01 — Reproducibility notebook\n\n"
       "*Companion to:* **Geochemistry-informed prediction and layered transferability of "
       "soil organic carbon in the Ganges Delta** (Uddin et al.).\n\n"
       "This notebook regenerates every reported number and every figure in the manuscript "
       "from the cached inputs in `../data/` and the analysis scripts in `../scripts/`. "
       "It is intended for the Zenodo deposit and for peer review; the same code is also "
       "available as standalone Python scripts.\n\n"
       "**Runtime:** ≈ 1 minute end-to-end on a laptop. **Outputs:** CSV/JSON tables "
       "and PNG figures dropped alongside `scripts/`.\n\n"
       "**Random seed:** 42 throughout."),
    md("## 0. Environment and data"),
    code(PREAMBLE_CODE),
    code(LOAD_DATA_CODE),

    md("## 1. H2 — first-order G-RothC process model\n\n"
       "Rebuilt from scratch as proper first-order kinetics; carbon input back-calculated "
       "to a steady-state flux; management acts on both input and decomposition. "
       "Reproduces the headline numbers: baseline mean SOC 45.1 t C ha⁻¹, "
       "50% mangrove restoration +26.6%, management : climate ratio 5.6× (Ganges), "
       "4.6–7.2× across 5 deltas."),
    code(GROTHC_RUN),
    fig_cell("make_fig3.py", "fig3_climate.png", "Figure 3 — H2 climate dynamics."),

    md("## 2. H3 — leak-free spatial cross-validation\n\n"
       "Because the 100 samples are five depths nested within 20 locations, random "
       "k-fold CV leaks spatial information across train/test. Location-grouped CV is "
       "the authoritative metric: gradient-boosting performance drops from R² ≈ 0.91 "
       "(random) to R² ≈ 0.69 (location-grouped). OM is excluded (r = 0.996 with SOC). "
       "Reproduces R1.13 / R1.25 / R3.13 fixes."),
    code(ML_CV_RUN),

    md("### 2b. H3 — engineered-feature ablation\n\n"
       "The 23 H1-informed engineered features (interaction, ratio, polynomial, log "
       "transforms) reconstructed from the Methods recipe; their leak-free contribution "
       "to GBR is +0.05 to +0.11. (Replaces the unverifiable +0.42 MoE figure.)"),
    code(ENG_ABL_RUN),
    fig_cell("make_fig4.py", "fig4_ml_dl.png", "Figure 4 — ML/DL prediction under leak-free CV."),

    md("## 3. H1 — per-zone geochemical correlations\n\n"
       "Within each of the four geomorphological zones (Tidal/Active/Mature/Moribund). "
       "Sharpens the 'EC paradox' (EC collapses in saline zones, persists in freshwater) "
       "and exposes sign reversals masked by pooling (e.g., TN +0.73 in Active vs. "
       "−0.79 in Tidal). Becomes Supplementary Table S10."),
    code(PERZONE_RUN),

    md("## 4. H4 — cross-delta transferability (14 deltas + Ganges)\n\n"
       "Rebuilt from WoSIS 2023 December (Batjes 2024; CC BY 4.0). 4,504 complete-case "
       "top-metre profiles across 14 global deltas (Mississippi, Rhine-Meuse, Niger, "
       "Yellow River, Limpopo, Chao Phraya, Paraná-Plata, Sacramento, Indus, Mackenzie, "
       "Danube, Amazon, Senegal, Pearl) plus the 100 Ganges samples. The harmonised "
       "extract is cached in `../data/h4_delta_samples.csv` so the analysis runs without "
       "re-downloading the 0.5 GB WoSIS snapshot.\n\n"
       "**Key result:** 210 cross-delta pairs, only 6% R² > 0 (median −0.55) vs. "
       "within-delta median +0.25 — models generalise within a delta but not between."),
    code(H4_RUN),
    fig_cell("make_fig5.py", "fig5_transfer.png", "Figure 5 — H4 transfer matrix."),
    fig_cell("make_fig6.py", "fig6_hypo3_spatial.png", "Figure 6 — SI: H3 spatial structure."),

    md("## 5. Figure 1 — study area\n\n"
       "Reads root shapefiles + DEM in `../data/gis/`. The DEM (183 MB) is omitted from "
       "the GitHub release; install it from the Zenodo archive to regenerate this figure."),
    code('# Run the study-area figure generator (skips gracefully if DEM not present)\n'
         'import importlib\n'
         'try:\n'
         '    import create_study_area_figure\n'
         '    importlib.reload(create_study_area_figure)\n'
         '    from IPython.display import Image\n'
         '    Image("fig1_study_area.png", width=720)\n'
         'except FileNotFoundError as e:\n'
         '    print(f"Skipped: {e}\\n(Install data/gis/DEM.tif from the Zenodo archive.)")'),

    md("## Done.\n\n"
       "All outputs (`grothc_results.csv`, `ml_spatial_cv_results.csv`, "
       "`engineered_ablation_results.csv`, `per_zone_correlations.csv`, "
       "`h4_results.json`, `h4_transfer_matrix.csv`, `fig3_climate.png`, "
       "`fig4_ml_dl.png`, `fig5_transfer.png`, `fig6_hypo3_spatial.png`) "
       "now sit in `../scripts/`. They are the exact numbers cited in the manuscript."),
]
nbf.write(n1, HERE / "01_reproducibility.ipynb")
print("wrote", HERE / "01_reproducibility.ipynb")

# ─────────────────────────────────────────────────── notebook 2: tutorial

n2 = nbf.v4.new_notebook()
n2.cells = [
    md("# 02 — Tutorial notebook\n\n"
       "*Companion to:* **Geochemistry-informed prediction and layered transferability of "
       "soil organic carbon in the Ganges Delta**.\n\n"
       "This notebook walks through the analyses **with explanatory narrative**, so a "
       "reader unfamiliar with the design can understand *why* each step exists. It ends "
       "with empty exploration cells for extending the work (try other models, deltas, "
       "or sensitivity sweeps).\n\n"
       "**Audience:** authors onboarding new collaborators, students, or anyone learning "
       "spatial cross-validation, process-modelling, or cross-delta transferability for SOC.\n\n"
       "**Prereqs:** Python 3.11, `requirements.txt` in the bundle root."),

    md("## 0. Setup\n\n"
       "We work from `reproducibility/`; the scripts and data live one level above this "
       "notebook. Everything below is deterministic (random seed = 42)."),
    code(PREAMBLE_CODE),

    md("## 1. The field data\n\n"
       "We sampled 20 sites across four geomorphological zones of the Ganges Delta "
       "(Hossain et al. 2015) — Tidal Active, Active Delta, Mature Delta, Moribund Delta "
       "— five sites per zone, five depth intervals per site (0–20, 20–40, …, 80–100 cm), "
       "yielding 100 samples. Each was analysed for 14 physicochemical properties.\n\n"
       "Why we kept the data design as nested replicates rather than pseudo-independent "
       "samples: because the 5 depths within a pit are spatially autocorrelated, mixing "
       "them across train/test inflates apparent model skill. (See section 3.)"),
    code(LOAD_DATA_CODE),
    code('# Quick descriptive look at SOC stock by zone\n'
         'import pandas as pd\n'
         'frames = []\n'
         'for z, d in zones.items():\n'
         '    frames.append(pd.DataFrame({"zone": z, "SOC_stock": d["SOC Stock"]}))\n'
         'long = pd.concat(frames)\n'
         'long.groupby("zone")["SOC_stock"].describe().round(1)'),

    md("## 2. H1 — geochemical controls and the 'EC paradox'\n\n"
       "Pearson correlations on the pooled 100-sample dataset show electrical "
       "conductivity (EC), cation exchange capacity (CEC), and soil moisture as the "
       "strongest geochemical correlates of SOC. But the *between-zone* contrast is "
       "very different from the *within-zone* one — the EC paradox.\n\n"
       "We run the per-zone analysis to expose this. Sign reversals (TN positive in one "
       "zone, negative in another) demonstrate why pooling correlations can be "
       "misleading in heterogeneous systems."),
    code(PERZONE_RUN),
    md("**Notice** rows where the pooled and per-zone signs disagree (TN, pH). These are "
       "the strongest argument for zone-stratified analysis rather than pooling."),

    md("## 3. H3 — honest cross-validation for nested-sample data\n\n"
       "A random 5-fold split puts samples from the same site in both training and test "
       "folds. Because nearby samples are correlated, this lets the model 'memorise' the "
       "site rather than learn the underlying relationship — apparent R² is inflated.\n\n"
       "The fix is **GroupKFold by location**: whole sites are held out so the test set "
       "is genuinely spatially separated. We compare four schemes:\n"
       "* Random 5-fold (the leaky one),\n"
       "* Location-grouped 5-fold (the authoritative one),\n"
       "* Leave-Location-Out (20-fold; less conservative because one site at a time),\n"
       "* Leave-Zone-Out (4-fold; the strictest test — generalisation across zones).\n\n"
       "We also drop OM (Walkley-Black organic matter) up front: it correlates r = 0.996 "
       "with SOC because they measure essentially the same thing. Including OM made the "
       "model look good (R² ≈ 0.98) for the wrong reason."),
    code(ML_CV_RUN),
    md("The drop from R² ≈ 0.91 (random) → 0.69 (location-grouped) quantifies the "
       "pseudoreplication inflation. The Leave-Zone-Out failure (R² < 0) tells you the "
       "model has no out-of-zone skill — which is consistent with the cross-delta "
       "transfer failure we'll see in section 6."),

    md("### 3b. What do the engineered features actually add?\n\n"
       "The H1 correlations and pH/CEC/EC interactions suggest specific feature "
       "transformations (products, ratios, polynomials, logs). The Methods recipe "
       "enumerates 29 of these; we reconstruct and test them under the same leak-free CV."),
    code(ENG_ABL_RUN),
    md("So the engineered features add ~+0.05 to +0.11 R² to GBR. Modest but reproducible "
       "— in line with the manuscript's claim, and notably without the spuriously large "
       "+0.42 that arose under hold-out CV for unstable deep models."),

    md("## 4. H2 — first-order G-RothC process model\n\n"
       "RothC is a first-order soil carbon decomposition model. We rebuilt it from "
       "scratch so the implementation matches the equation:\n\n"
       "$$\\frac{dC}{dt} = I - k_0 \\cdot f_T(T) \\cdot f_m(M) \\cdot f_\\mathrm{mgmt} \\cdot C$$\n\n"
       "(decay proportional to current stock, *not* a constant zero-order term). The "
       "input $I$ is back-calculated per location from the steady-state condition so the "
       "model initialises in equilibrium with each sample's observed stock. Management "
       "factors operate on both input (mangrove restoration adds litter) and "
       "decomposition (anoxia suppresses microbial activity)."),
    code(GROTHC_RUN),
    code('# Pull the summary into a clean table\n'
         'pd.read_csv("grothc_results.csv")'),
    md("The headline policy result — **management interventions outweigh climate "
       "forcing by ~5×** — is robust because both effects scale with the stock at risk; "
       "the ratio is therefore largely independent of absolute SOC."),

    md("## 5. Figures 3 and 4 — climate dynamics and ML/DL prediction"),
    fig_cell("make_fig3.py", "fig3_climate.png", "Figure 3."),
    fig_cell("make_fig4.py", "fig4_ml_dl.png", "Figure 4."),

    md("## 6. H4 — cross-delta transferability\n\n"
       "The motivating question: if a model trained on the Ganges learns 'SOC depends on "
       "EC, CEC, moisture, etc.', does that knowledge transfer to another delta?\n\n"
       "We test this with a 15 × 15 train-on-delta-*i*, test-on-delta-*j* matrix. "
       "External data come from WoSIS 2023 (Batjes 2024) — a single harmonised public "
       "source, so units, depths, and analytical methods are consistent. The common "
       "predictor set is pH + clay + silt + sand (other variables are too sparsely "
       "co-measured in WoSIS).\n\n"
       "Important null result: Asian monsoon deltas (Mekong, Irrawaddy, Godavari, "
       "Krishna) have SOC values but no co-measured texture/pH and so cannot enter a "
       "feature-based transfer test — itself a finding about the global data gap."),
    code(H4_RUN),
    fig_cell("make_fig5.py", "fig5_transfer.png", "Figure 5 — transfer matrix."),
    md("**Read the matrix as:** the diagonal (boxed in Fig 5a) is within-delta CV — "
       "the model genuinely learns SOC where data is adequate. Off-diagonal cells are "
       "near-uniformly dark red (R² < 0): training on one delta does not generalise to "
       "another. Forward (Ganges → world) fails for all 14; reverse "
       "(pooled 14 → Ganges) reaches only R² ≈ 0.03. **Pooling does not enable "
       "cross-delta transfer.** This is a strong, expected pedometric result and the "
       "reason the manuscript advocates *layered transferability*: only higher-level "
       "structures (methodology, process-model form) transfer; learned weights do not."),
    fig_cell("make_fig6.py", "fig6_hypo3_spatial.png", "Figure 6 — SI: H3 spatial structure."),

    md("## 7. Figure 1 — study area\n\n"
       "Requires the GIS layers in `../data/gis/`; the DEM (~183 MB) is in the Zenodo "
       "archive only. Skipped gracefully if absent."),
    code('try:\n'
         '    import importlib, create_study_area_figure\n'
         '    importlib.reload(create_study_area_figure)\n'
         '    from IPython.display import Image\n'
         '    Image("fig1_study_area.png", width=720)\n'
         'except FileNotFoundError as e:\n'
         '    print(f"Skipped: {e}")'),

    md("---\n# 8. Exploration cells\n\n"
       "Everything below is yours to extend. Suggested directions:\n\n"
       "* Try a different ML model on the leak-free CV (Random Forest, LightGBM).\n"
       "* Add other deltas to the H4 panel (the bounding-box dictionary is in "
       "`scripts/h4_analysis.py` / `provenance/wosis_extraction.py`).\n"
       "* Re-run G-RothC with different management scenarios "
       "  (mangrove fraction, k₀ sensitivity, Q₁₀ range).\n"
       "* Add a Bayesian uncertainty quantification on the GBR predictions.\n"
       "* Investigate which features carry signal *within* a delta but not *between* "
       "  — the explanation for the transfer-failure asymmetry.\n"),
    code('# YOUR EXPLORATION HERE\n'),
    code('# YOUR EXPLORATION HERE\n'),
    code('# YOUR EXPLORATION HERE\n'),
]
nbf.write(n2, HERE / "02_tutorial.ipynb")
print("wrote", HERE / "02_tutorial.ipynb")
