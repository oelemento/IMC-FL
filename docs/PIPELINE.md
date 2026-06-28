# Pipeline & Scripts

## Pipeline Dependency Chain

```
Raw TXT pixels
  └─► batch_process.py / batch_process_S.py  →  per-ROI h5ad (segmented cells)
        └─► recombine_raw.py  →  *_raw_combined.h5ad (raw counts, concatenated)
              └─► cross_tma_global.py --version TAG  →  all_TMA_*_global_TAG.h5ad  ← RE-ANNOTATION POINT
                    ├─► utag_celltype_all_*.py  →  UTAG domains (depends on cell type labels)
                    │     └─► utag_merge_domains.py  →  *_utag_ct_merged.h5ad
                    │           └─► All spatial compartment hypotheses (H1a, H2a, H6c, H9a-g, etc.)
                    ├─► All cell-type-level hypotheses (H5e/f/g, H8c, H3c, H4c, etc.)
                    ├─► All QC/annotation figures (qc_figures.py, compartment_figures.py)
                    └─► cross_panel_figure.py (uses both T and S annotated files)
```

**Annotation version is a parameter, not a script name.** Input data paths must always be CLI arguments — never hardcoded constants. Every hypothesis script must accept `--t-panel` and/or `--s-panel` arguments for its input h5ad files. Output paths should be derived from the input filename (e.g., extract `v8` from the input path and use it in the output). This way, re-annotation only requires re-running the same scripts pointing at new files, not copying or editing source code.

> **Tech debt**: Several existing scripts (`run_h1a.py`, `run_h1c.py`, `run_h3c.py`, `run_h4c.py`, `run_h6b.py`, `run_h8c.py`, `visualize_fdc_raw.py`, and the UTAG scripts) still hardcode `v8` paths as constants. These need to be refactored to accept CLI arguments before any re-annotation run. New scripts must always use CLI arguments from the start.

## What Needs Re-Running When

| Change | Must re-run | Safe to skip |
|--------|-------------|--------------|
| Re-segmentation (Cellpose params) | Everything from `batch_process` down | Nothing |
| Re-annotation (v8 → v9) | `cross_tma_global.py --version v9` + all UTAG + all hypothesis scripts + all figures | `batch_process`, `recombine_raw` (segmentation unchanged) |
| UTAG re-clustering (new resolution) | `utag_merge_domains.py` + compartment hypotheses | Non-spatial hypothesis tests, annotation figures |
| New hypothesis script | Just that script | Everything else |

**Convention**: Every NOTEBOOK.md entry should name its input files explicitly (e.g., `all_TMA_T_global_v8.h5ad`) so that a grep for `v8` reveals all analyses that depend on v8 annotations and need re-running if the annotation changes.

---

## Script Inventory

### Pipeline (run on Cayuga, produce h5ad files)

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| `batch_process.py` | Raw TXT | Per-ROI h5ad | T-panel, one ROI at a time, hybrid segmentation |
| `batch_process_S.py` | Raw TXT | Per-ROI h5ad | S-panel |
| `recombine_raw.py` | Per-ROI h5ad | `*_raw_combined.h5ad` | Extracts `.raw` before concat (fixes double-transform) |
| `cross_tma_global.py` | raw_combined x 4 | `all_TMA_*_global_v8.h5ad` | Cross-TMA embed + cluster + v8 annotate |
| `reannotate.py` | v8 h5ad | Updated h5ad | Re-run annotation without re-embedding |

### UTAG (run on Cayuga, produce tissue domains)

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| `utag_celltype_all_T.py` | v8 h5ad | UTAG h5ad | Cell-type feature UTAG (T-panel) |
| `utag_celltype_all_S.py` | v8 h5ad | UTAG h5ad | Cell-type feature UTAG (S-panel) |
| `utag_merge_domains.py` | UTAG h5ad | `*_utag_ct_merged.h5ad` | Hierarchical merge to 15 named compartments |
| `utag_name_compartments.py` | Merged h5ad | Named compartments | Auto-name by dominant cell type |

### Hypothesis testing & figures (run locally)

| Script | Hypotheses | Output |
|--------|-----------|--------|
| `run_hypotheses_v2.py` | H6a, H2a, H2e, H6c, H2b | `output/hypotheses_v8/*.png` |
| `spatial_interactions.py` | H6b (nhood enrichment + compartment-conditioned) | `fig_nhood_enrichment_T.png` |
| `cellular_neighborhoods.py` | H6b (cellular neighborhoods, 10 clusters) | `fig_cellular_neighborhoods_T.png` |
| `spatial_functional.py` | H2c, H2d, H2f (exhaustion × proximity) | `fig_functional_proximity_T.png` |
| `spatial_covariation.py` | H6b (co-variation network + boundary) | `fig_spatial_covariation_T.png` |
| `qc_figures.py` | Data description | `output/qc/*.png` |
| `compartment_figures.py` | Compartment analysis | `output/*.png` |
| `compute_marker_qc_stats.py` | Marker QC stats | `output/marker_qc_stats.npz` |
| `marker_qc.py` | Per-marker QC | `output/*.png` |
| `survival_analysis.py` | Spatial metrics vs OS/PFS/transformation | `fig_survival.png`, `survival_covariates.csv` |
| `immune_evasion.py` | H9a-h (all immune evasion hypotheses) | `fig_ie_*.png` |

### Cross-panel analysis (local)

| Script | Purpose |
|--------|---------|
| `cross_panel_concordance.py` | T/S panel concordance analysis (v7, exploratory) |
| `cross_panel_figure.py` | Publication cross-panel concordance figure (v8) |
| `cross_panel_transfer.py` | Transfer labels between panels |
| `register_paired_rois.py` / `_v2.py` | Register paired serial section ROIs |

### Utilities (`src/`)

| Module | Purpose |
|--------|---------|
| `src/data_loader.py` | Load raw TXT pixel data, extract markers |
| `src/clinical_linkage.py` | Clinical data loading, ROI name normalization (`ROI_00X`→`FLX`, `FL0X`→`FLX`), clinical-to-obs merge |

### Historical (keep for reference, not actively used)

Segmentation development: `test_segmentation.py`, `test_cellpose_v2.py`, `test_membrane_seg.py`, `test_stardist.py`, `tune_cellpose.py`, `tune_watershed.py`, `hybrid_segmentation.py`, `fast_segment.py`, `segment_dense.py`, `segment_patch.py`, `validate_segmentation.py`, `validate_marker_homogeneity.py`

QC/diagnostic: `qc_b1_trace.py`, `qc_b1_trace2.py`, `qc_cross_tma.py`, `qc_cross_tma_light.py`, `check_s_panel_clusters.py`, `check_s_panel_other.py`

Early pipeline: `combine_results.py` (superseded by `recombine_raw.py`), `global_analysis.py`, `cluster_cells.py`, `cluster_heatmap.py`

UTAG exploration: `test_utag.py`, `test_utag_single_roi.py`, `utag_maxdist_sweep.py`, `run_utag_b1.py`, `run_utag_b1_S.py`, `utag_per_tma_T.py`, `utag_per_tma_S.py`
