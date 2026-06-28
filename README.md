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
