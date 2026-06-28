# IMC-FL: Spatial Proteomics of Follicular Lymphoma

Analysis code for the manuscript *"Spatial Proteomics Reveals Dual-Compartment
Immune Evasion Architecture and Core Lymphomagenic Stromal Niches in Follicular
Lymphoma"* (Elemento et al.).

The study characterizes the follicular lymphoma tumor microenvironment using
Hyperion imaging mass cytometry (IMC) across four tissue microarrays and two
39-marker antibody panels (~4.2 million cells), identifying spatial immune-evasion
programs, CD14-high follicular dendritic cell niches, and a VISTA-dominated myeloid
compartment.

## Repository structure

- `scripts/` — analysis and figure-generation scripts (segmentation, cell-type
  annotation, spatial-compartment/UTAG analysis, survival, figure assembly)
- `src/` — shared modules (e.g., clinical linkage)
- `docs/` — dataset description (`DATASET.md`), pipeline overview (`PIPELINE.md`),
  and key literature (`LITERATURE.md`)
- `pyproject.toml` — Python environment

The agent-based model used for the treatment-scenario simulations is in a separate
repository: https://github.com/oelemento/FL-ABM

## Environment

Python 3.11. Install dependencies from `pyproject.toml` (e.g., `pip install -e .`).

## Regenerating the figures

The paper's figure/table PDF is assembled by `scripts/assemble_paper_figures.py`,
which lays out the individual figure PNGs and renders the tables. Regeneration runs
in three stages; data files are passed as command-line arguments (see each script's
`--help`), not hardcoded.

**1. Build the single-cell tables (pipeline).** Starting from the raw IMC data,
segment cells, recombine raw intensities, harmonize across the four TMAs, annotate
cell types, and call spatial compartments:

```
batch_process.py / batch_process_S.py     # cell segmentation (T and S panels)
recombine_raw.py                          # attach raw marker intensities
cross_tma_global.py                       # cross-TMA integration/embedding
reannotate.py                             # v8 cell-type annotation
utag_celltype_all_T.py / _S.py            # UTAG spatial domains (one-hot features)
utag_merge_domains.py                     # hierarchical merge to ~15 compartments
utag_name_compartments.py                 # name compartments
compute_marker_qc_stats.py               # per-marker QC (writes marker_qc_stats.npz)
```

This produces the annotated `.h5ad` files (deposited at Zenodo, see below) plus the
QC stats consumed downstream.

**2. Run the core analyses** that write the intermediate tables the figures read:

```
survival_analysis.py / survival_relapse_post_rchemo.py   # -> survival_covariates.csv
vista_fl_vs_tonsil.py                                     # -> vista_fl_vs_tonsil.csv
cd14_validation.py                                        # CD14/mutation association tables
```

**3. Generate the figure panels, then assemble.** Each `fig_*.py` script (and the
shared helpers `figure_style.py`, `immune_evasion.py`, `compartment_figures.py`,
`fig_signaling_architecture.py`, `fig_tonsil_comparison.py`,
`generate_model_figure_svg.py`, `qc_figures.py`) writes one figure PNG. Once the
PNGs exist, build the combined PDF:

```
python scripts/assemble_paper_figures.py
```

Table 1 (cohort), Table S1 (antibody panels, read from the supplementary workbook),
and Table S2 (ABM parameters) are rendered by `assemble_paper_figures.py` itself and
placed after the figures.

The agent-based-model figure (`fig_abm_treatment.png`, main Fig 7) is produced by the
separate FL-ABM repository linked below and dropped into the assembly.

## Data availability

Processed single-cell data (both panels) are deposited at Zenodo:
**DOI 10.5281/zenodo.20612591** (CC-BY 4.0; released openly upon publication).
Raw imaging data and patient-level clinical data are described in the manuscript's
Data Availability statement.

## Notes

Scripts take input data files as command-line arguments (data are not bundled in
this repository). Paths in `docs/` referencing an internal compute environment are
shown as placeholders (`<DATA_ROOT>`, `<PROJECT_ROOT>`).

## Citation

Please cite the associated manuscript (and its bioRxiv preprint). Contact:
Olivier Elemento, Englander Institute for Precision Medicine, Weill Cornell Medicine.
