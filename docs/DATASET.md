# IMC-FL Dataset Card

## Overview

Hyperion Imaging Mass Cytometry (IMC) of follicular lymphoma (FL) tissue microarrays (TMAs). Four TMAs (A1, B1, C1, Biomax) were stained with two antibody panels on serial sections: a T-cell panel (T) focused on T cell subsets, checkpoints, and exhaustion markers, and a stromal panel (S) focused on myeloid, B cell, stromal, and vascular markers. Each panel has 39 biological markers. Total: ~4.2 million cells across ~170 ROIs per panel.

**Data locations:**
- Raw pixel exports (READ-ONLY): `<DATA_ROOT>/`
- Processed data: `<PROJECT_ROOT>/`
- Analysis code + local outputs: `<PROJECT_ROOT>/`

**Clinical metadata:** ~130 patients, most received rituximab-based therapy. Patient IDs, FL grade, treatment history, and outcomes to be obtained.

---

## Tissue Microarrays

| TMA | Source | ROIs (T/S) | Cells (T/S) | Controls | Notes |
|-----|--------|-----------|-------------|----------|-------|
| A1 | In-house FL | 55/54 | 664,084 / 665,909 | Prostate, Tonsil | First FL TMA |
| B1 | In-house FL | 50/47 | 631,429 / 562,361 | Kidney, Tonsil | Second FL TMA |
| C1 | In-house FL | 49/49 | 566,570 / 546,002 | Spleen, Tonsil | Third FL TMA |
| Biomax | Commercial | 28/27 | 314,793 / 284,751 | Tonsil (`_Ton_`), Adrenal (`_Adr_`) | Mixed FL + controls |
| **Total** | | **182/177** | **2,176,876 / 2,059,023** | | |

### Control exclusion rules

Exclude ROIs matching (case-insensitive):
- Any TMA: `tonsil`, `prostate`, `kidney`, `spleen`, `adrenal`
- Biomax-specific: `_Ton_` (tonsil), `_Adr_` (adrenal)
- Biomax FL tumor cores: `_Lym_` pattern (include these)

~98K T-panel and ~87K S-panel cells excluded as controls.

---

## Antibody Panels

### T-panel (39 markers)

QC status per TMA from arcsinh(raw/5) p99: `+` good (p99 > 1.0), `~` weak (0.3-1.0), `-` dead (< 0.3)

| Marker | Category | A1 | B1 | C1 | Biomax | p99 range | Notes |
|--------|----------|----|----|----|----|-----------|-------|
| CD3 | T cell lineage | `~` | `~` | `~` | `~` | 0.71-0.92 | Working but moderate signal |
| CD4 | T cell lineage | `~` | `~` | `~` | `~` | 0.55-0.67 | Working |
| CD8a | T cell lineage | `+` | `+` | `+` | `~` | 0.96-1.23 | Good |
| CD45RO | T cell lineage | `+` | `+` | `+` | `+` | 1.11-1.31 | Good |
| CD127 | T cell lineage | `~` | `~` | `~` | `~` | 0.48-0.76 | Working |
| CD20 | B cell | `+` | `+` | `+` | `~` | 0.92-2.41 | Primary B cell marker |
| CD68 | Myeloid | `+` | `+` | `+` | `+` | 1.40-2.53 | Primary myeloid marker |
| FoxP3 | Treg | `-` | `-` | `-` | `-` | 0.11-0.29 | Weak but detectable above background |
| TOX | Exhaustion | `~` | `~` | `~` | `-` | 0.18-0.61 | Key exhaustion marker; works in A1/B1/C1 |
| PD_1 | Checkpoint | `-` | `~` | `-` | `-` | 0.07-0.35 | Weak; B1 best |
| CXCR5 | Tfh | `~` | `~` | `~` | `~` | 0.50-0.99 | Continuous distribution, not bimodal. Use p90 threshold (>2.0) |
| GranzymeB | Cytotoxic | `+` | `~` | `-` | `~` | 0.23-1.16 | Variable; A1 best |
| CD38 | Activation | `~` | `-` | `-` | `~` | 0.19-0.72 | Variable |
| IRF4 | Transcription factor | `~` | `~` | `-` | `-` | 0.21-0.34 | Weak |
| cMYC_p67 | Proliferation | `~` | `+` | `~` | `+` | 0.90-1.11 | Working |
| pCREB | Signaling | `+` | `+` | `+` | `+` | 1.06-1.37 | Good |
| pS6 | Signaling | `+` | `+` | `+` | `~` | 0.60-1.51 | Good in A1/B1/C1 |
| CD47 | Immune evasion | `~` | `~` | `~` | `-` | 0.27-0.45 | Weak |
| CD31 | Endothelial | `-` | `-` | `-` | `~` | 0.19-0.32 | Weak |
| CD27 | Costimulation | `~` | `~` | `~` | `-` | 0.20-0.57 | Variable |
| CD57 | Senescence | `~` | `~` | `~` | `~` | 0.32-0.64 | Weak-moderate |
| CD39 | Ectoenzyme | `~` | `~` | `-` | `~` | 0.28-0.48 | Variable |
| CD86 | Costimulation | `-` | `~` | `~` | `-` | 0.15-0.39 | Mostly weak |
| TIM3 | Checkpoint | `~` | `~` | `~` | `~` | 0.33-0.45 | Diffuse, no bimodal separation; not usable as gate |
| ICOS | Costimulation | `-` | `-` | `-` | `-` | 0.07-0.20 | Dead |
| PD_L1 | Checkpoint | `-` | `-` | `-` | `-` | 0.09-0.23 | Dead |
| LAG3 | Checkpoint | `-` | `-` | `-` | `-` | 0.04-0.08 | Dead |
| T_Bet | Transcription factor | `-` | `-` | `-` | `-` | 0.08-0.11 | Dead |
| CTLA4 | Checkpoint | `-` | `-` | `-` | `-` | 0.08-0.18 | Dead |
| GATA3 | Transcription factor | `-` | `-` | `-` | `-` | 0.14-0.18 | Dead |
| p53 | Tumor suppressor | `-` | `-` | `-` | `-` | 0.08-0.14 | Dead |
| pSTAT3 | Signaling | `-` | `-` | `-` | `-` | 0.08-0.15 | Dead |
| Cleaved_caspase_3 | Apoptosis | `-` | `-` | `-` | `-` | 0.13-0.17 | Dead |
| Carbonic_Anhydrase_IX | Hypoxia | `-` | `-` | `-` | `~` | 0.14-0.34 | Dead except Biomax |
| HistoneH3 | Structural | `+` | `+` | `+` | `+` | 2.39-3.18 | Nuclear marker (excluded from analysis) |
| p_H3s28 | Structural | `~` | `~` | `~` | `~` | 0.54-0.94 | Structural (excluded) |
| H3K27me3 | Structural | `+` | `+` | `+` | `+` | 2.10-3.78 | Structural (excluded) |
| DNA1 | Structural | `+` | `+` | `+` | `+` | 3.52-4.10 | Iridium 191 (excluded) |
| DNA2 | Structural | `+` | `+` | `+` | `+` | 4.11-4.67 | Iridium 193 (excluded) |

### S-panel (39 markers)

| Marker | Category | A1 | B1 | C1 | Biomax | p99 range | Notes |
|--------|----------|----|----|----|----|-----------|-------|
| CD20 | B cell | `+` | `+` | `+` | `~` | 0.89-2.00 | Primary B cell marker |
| PAX5 | B cell | `~` | `~` | `+` | `~` | 0.39-1.09 | B lineage; C1 best |
| BCL_2 | B cell / FL | `+` | `+` | `+` | `~` | 0.41-1.48 | FL hallmark |
| CD21 | B cell / FDC | `+` | `+` | `+` | `~` | 0.69-2.47 | FDC marker |
| CD68 | Myeloid | `+` | `+` | `+` | `+` | 1.02-2.35 | Primary myeloid marker |
| CD163 | M2 macrophage | `-` | `-` | `-` | `~` | 0.09-0.72 | Dead in A1/B1; functional only in Biomax |
| CD206 | M2 macrophage | `-` | `-` | `~` | `+` | 0.13-1.27 | Dead in A1/B1; functional in Biomax/C1 |
| CD14 | Monocyte | `~` | `~` | `~` | `+` | 0.34-1.03 | Biomax best |
| CD11c | Myeloid/DC | `+` | `+` | `+` | `~` | 0.61-1.47 | Good in A1/B1/C1 |
| CD11b | Myeloid | `-` | `-` | `-` | `-` | 0.20-0.28 | Weak across all |
| S100A9 | Myeloid | `~` | `+` | `+` | `+` | 0.96-3.05 | Variable; good in B1/C1/Biomax |
| CD4 | T cell | `~` | `~` | `~` | `~` | 0.41-0.76 | Working |
| CD8a | T cell | `~` | `~` | `+` | `~` | 0.79-1.38 | C1 best |
| HLA_DR | Antigen presentation | `+` | `+` | `+` | `+` | 1.96-3.15 | Good |
| HLA_Class_I | Antigen presentation | `+` | `+` | `+` | `+` | 2.28-3.03 | Good |
| Vimentin | Stromal | `+` | `+` | `+` | `+` | 1.38-2.16 | Good |
| Fibronectin | Stromal/ECM | `-` | `-` | `~` | `-` | 0.13-0.42 | Mostly weak |
| PDPN | Stromal/FRC | `~` | `~` | `~` | `-` | 0.14-0.45 | Weak |
| CD31 | Endothelial | `-` | `-` | `-` | `~` | 0.16-0.31 | Weak |
| CD34 | Endothelial | `~` | `~` | `~` | `+` | 0.58-1.24 | Biomax best |
| CD146 | Endothelial | `-` | `-` | `~` | `-` | 0.19-0.32 | Weak |
| SOX9 | Stromal | `+` | `+` | `+` | `~` | 0.88-1.21 | Good |
| CD44 | Adhesion | `~` | `~` | `~` | `+` | 0.73-1.01 | Moderate-good |
| Ki-67 | Proliferation | `~` | `~` | `+` | `-` | 0.20-1.19 | C1 good; Biomax dead |
| CD123 | pDC | `-` | `-` | `-` | `~` | 0.09-0.33 | Marginal in Biomax/C1 only |
| CD209 | DC-SIGN | `-` | `-` | `~` | `-` | 0.14-0.36 | Weak |
| CD1a | DC | `-` | `-` | `-` | `-` | 0.13-0.25 | Dead |
| CD49a | Integrin | `~` | `-` | `~` | `~` | 0.20-0.48 | Variable |
| CXCL13 | Chemokine | `-` | `-` | `-` | `-` | 0.07-0.23 | Dead |
| CXCL12 | Chemokine | `-` | `-` | `-` | `-` | 0.15-0.26 | Dead |
| CCL21 | Chemokine | `-` | `-` | `~` | `~` | 0.20-0.52 | Weak; C1/Biomax slightly better |
| IDO | Immune regulation | `-` | `-` | `-` | `-` | 0.09-0.22 | Dead |
| VISTA | Immune regulation | `-` | `-` | `-` | `-` | 0.12-0.25 | Dead |
| PD_L1 | Checkpoint | `-` | `-` | `-` | `-` | 0.07-0.11 | Dead |
| BCL_6 | Transcription factor | `-` | `-` | `-` | `-` | 0.05-0.11 | Dead |
| HistoneH3 | Structural | `+` | `+` | `+` | `+` | 2.60-3.01 | Structural (excluded) |
| p_H3s28 | Structural | `~` | `~` | `~` | `~` | 0.58-0.81 | Structural (excluded) |
| DNA1 | Structural | `+` | `+` | `+` | `+` | 3.71-4.08 | Iridium 191 (excluded) |
| DNA2 | Structural | `+` | `+` | `+` | `+` | 4.28-4.65 | Iridium 193 (excluded) |

### Markers excluded from PCA/clustering

DNA1, DNA2, HistoneH3, p_H3s28, H3K27me3 (structural markers, not informative for cell phenotyping).

---

## Processing Pipeline

| Step | Method | Key Parameters |
|------|--------|---------------|
| 1. Raw data | Pixel-level TXT exports | Tab-separated; columns: X, Y, Z + markers with metal tags |
| 2. Segmentation | Cellpose `cyto3` hybrid | `flow_threshold=0.8`, nuclear filter (`min_distance=2`, `sigma=1.0`) |
| 3. Feature extraction | Mean marker intensity per cell | Per-cell, produces AnnData with `obs`: cell_id, sample_id, centroid_x/y, area |
| 4. Transform | arcsinh(X / 5) | Cofactor = 5, standard for mass cytometry |
| 5. Scaling | `sc.pp.scale(max_value=10)` | Per-marker z-score, clipped at 10 |
| 6. PCA | 30 components | On non-structural markers |
| 7. Neighbors | k=15 | Using 15 PCs |
| 8. Clustering | Leiden, resolution=0.5 | igraph backend |
| 9. Cell type annotation | Rule-based per-cell gating (v8) | See below |
| 10. UTAG | Cell-type one-hot features | `max_dist=50`, Leiden `res=0.5`, merged to 15 compartments |

### Segmentation details

**Hybrid method**: (1) Detect nuclei from DNA1+DNA2 (local maxima on smoothed signal), (2) Run Cellpose `cyto3` with `flow_threshold=0.8` for cell boundaries, (3) Keep only Cellpose cells containing at least 1 detected nucleus. This removes ~22% of Cellpose cells (cytoplasm fragments, artifacts).

`flow_threshold=0.8` is critical for dense lymphoma tissue — the default 0.4 under-segments by ~50%.

---

## Cell Type Annotations (v8)

Per-cell rule-based gating on arcsinh-transformed marker values. Cluster-level annotation fails for FL because B cells dominate every cluster, inflating cluster-mean CD20 and preventing T cell identification. Per-cell gating resolves this.

### T-panel cell types

| Cell Type | Gating Rule | Priority |
|-----------|------------|----------|
| CD8 T exhausted | CD3>0.5, CD8>CD4, CD8>CD20, TOX>0.8, PD1>0.5 | 1 |
| CD8 T pre-exhausted (TOX+) | CD3>0.5, CD8>CD4, CD8>CD20, TOX>0.8 | 1 |
| CD8 T cells | CD3>0.5, CD8>CD4, CD8>CD20 | 1 |
| Treg | CD3>0.5, CD4>CD8, CD4>CD20, FoxP3>threshold | 1 |
| CD4 T cells | CD3>0.5, CD4>CD8, CD4>CD20 | 1 |
| T cells | CD3>0.5, CD4>CD20 or CD8>CD20 | 1 |
| GC B cells | CD20>CD3, CD20>CD68, CD20>=1.5, CXCR5>2.0, CD20>8.0 | 2 |
| Activated B / Plasmablast | CD20>CD3, CD20>CD68, CD20>=1.5, CD38>1.0, IRF4>0.5 | 2 |
| B cells (CD20hi) | CD20>CD3, CD20>CD68, CD20>=1.5, CD20>6.0 | 2 |
| B cells (CXCR5hi) | CD20>CD3, CD20>CD68, CD20>=1.5, CXCR5>1.5 | 2 |
| B cells (TOXhi) | CD20>CD3, CD20>CD68, CD20>=1.5, TOX>1.5 | 2 |
| B cells | CD20>CD3, CD20>CD68, CD20>=1.5 | 2 |
| Cytotoxic (GzmB+) | CD68>2.0, GzmB>2.0 | 3 |
| Macrophages | CD68>2.0, CD68>CD3 | 3 |
| B cells (weak CD20) | CD20>=0.5, CD20>2*CD3, CD20>2*CD68 | 4 |
| Mixed / Border cells | CD20>1.0, CD3>0.5 | 5 |
| Low quality / Unassigned | CD20<1.5, CD3<0.8, CD68<1.5 | 6 |

### S-panel cell types

| Cell Type | Gating Rule | Priority |
|-----------|------------|----------|
| FDC | CD21>5.0; or CD21>2.0, CXCL13>0.3, CD20<CD21 | 1 |
| FRC (PDPN+) | PDPN>1.5, CD20<2.0, CD68<2.0 | 2 |
| Endothelial | Vimentin>3.0 (no lineage), CD31>1.5 or CD34>1.5 | 3 |
| Stromal / CAF | Vimentin>3.0, no strong lineage markers | 3 |
| CD4 T cells | CD4>CD20, CD4>CD68, CD4>0.5 | 4 |
| CD8 T cells | CD8>CD20, CD8>CD68, CD8>0.5 | 4 |
| B cells (BCL2+) | CD20>1.5, CD20>CD68, BCL2>2.0 | 5 |
| B cells (PAX5+) | CD20>1.5, CD20>CD68, PAX5>1.0 | 5 |
| B cells | CD20>1.5, CD20>CD68 | 5 |
| Myeloid (S100A9+) | S100A9>5.0 | 8 |
| M2 Macrophages | CD68>2.0, CD163>1.5 or CD206>1.5 | 8 |
| M1 Macrophages | CD68>2.0, CD11c>1.0 or HLA_DR>2.0 | 8 |
| Macrophages | CD68>2.0; or CD14>2.0, CD14>CD20 | 8 |
| Dendritic cells | CD11c>1.5, HLA_DR>1.5, CD14<1.0 | 8 |
| pDC | CD123>1.5 (unreliable in A1/B1) | 8 |
| Endothelial | CD31>2.0 or CD34>1.2 | 9 |
| Histiocytes (CD44hi) | CD44>5.0 | 10 |
| Mixed / Border cells | CD20>1.0, CD68>1.0 | 11 |
| Low quality / Unassigned | CD20<1.0, CD68<1.0, CD4<0.5, Vimentin<1.5, PAX5<1.0 | 12 |

### Display consolidation

For figures and composition analysis, B cell subtypes are merged:

| Raw annotation | Display label |
|----------------|--------------|
| B cells (CD20hi), B cells (CXCR5hi), B cells (TOXhi), B cells (weak CD20), B cells (BCL2+), B cells (PAX5+) | B cells |
| T cells (residual) | Other |

Source: `CONSOLIDATE_MAP` in `scripts/qc_figures.py`

---

## UTAG Tissue Compartments

Tissue domains identified using UTAG (Unsupervised Tissue Architecture Graph) with cell-type one-hot features as input (not raw marker intensities). Each cell's neighborhood composition (within `max_dist=50` pixels) is encoded, then Leiden clustering at resolution 0.5 groups cells into spatial domains, which are hierarchically merged to 15 named compartments.

Compartments are classified as:
- **Follicular**: >50% B cells by v8 annotation (keywords: GC B, Follicle core, Follicle mantle, Activated B, B cell zone; S-panel also: FDC, BCL2+, PAX5+)
- **Interfollicular**: T cell, macrophage, stromal-dominated domains (keywords: T cell zone, Treg, Macrophage, Cytotoxic, interface, Stromal, Endothelial, Dendritic)

Three UTAG resolution versions are stored: `leiden_0.01`, `leiden_0.015`, `leiden_0.02`. Primary analysis uses 0.015.

---

## Known Issues

| Issue | Details | Affected TMAs |
|-------|---------|---------------|
| CD163 dead | p99 < 0.10 in arcsinh | A1, B1 (functional only in Biomax/C1) |
| CD206 dead | p99 < 0.14 | A1, B1 (functional only in Biomax/C1) |
| PD-L1 dead | p99 < 0.23 in T-panel, < 0.11 in S-panel | All TMAs, both panels |
| LAG3 dead | p99 < 0.08 | All TMAs |
| T-bet dead | p99 < 0.11 | All TMAs |
| CTLA4 dead | p99 < 0.18 | All TMAs |
| BCL6 dead | p99 < 0.11 | All TMAs (S-panel) |
| CXCR5 not bimodal | Continuous distribution without clear positive/negative separation | All TMAs — use p90 threshold (>2.0), not standard 0.5 |
| Double-transform bug | Historical: `ad.concat()` discards `.raw`; old pipeline saved transformed data then transformed again. Fixed by `recombine_raw.py` which extracts raw from per-ROI files before concatenation. | All TMAs (fixed) |
| TIM3 diffuse | mean=0.52, 54% of CD8 T cells "positive" at 0.5 — no bimodal separation | All TMAs — not usable as gate |

---

## Raw Data Format

### TXT pixel exports

Tab-separated values, one file per ROI. Rows = pixels, columns = metadata + markers.

| Column type | Examples |
|-------------|---------|
| Acquisition metadata | `Start_push`, `End_push`, `Pushes_duration` |
| Coordinates | `X`, `Y`, `Z` |
| Markers | `CD3(Er170Di)`, `CD20(Sm147Di)`, etc. (marker name + metal tag) |

Files range from ~100 MB to ~1 GB per ROI depending on tissue area.

### File naming convention

```
YYYYMMDD_CT14_XX_TMA_panel_N_ROI_NNN_N.txt
```

- `YYYYMMDD`: Acquisition date
- `TMA`: A1, B1, C1
- `panel`: Tcellpanel or stromalpanel
- `ROI_NNN` or `FLNN`: ROI/sample identifier
