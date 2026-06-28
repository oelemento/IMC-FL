#!/usr/bin/env python3
"""Assemble main paper figures into a single PDF.

Narrative order (7 main + 12 supplementary — supplementary inline after related main):
  Table 1  — Cohort description
  Figure 1 — Cell type annotation (with dataset schematic)
  Figure S1 — Segmentation & marker QC (T + S panels) [after Fig 1]
  Figure S2 — Cross-panel concordance [after S1, before Fig 2]
  Figure S3 — Mutation × cell type associations (negative result) [after S2]
  Figure 2 — Spatial compartments + follicular biology (10 panels, v2)
  Figure S4 — Compartment ROI examples (3x3 gallery) [after Fig 2]
  Figure S5 — S-panel spatial tissue compartments [after S4]
  Figure S6 — FDC network zone vs raw CD21 signal validation [after S5]
  Figure S7 — Tfh enrichment, exhaustion, and spatial co-localization [after S6]
  Figure S8 — Grade-associated proliferation, myeloid, follicle architecture [after S7, before Fig 3]
  Figure 3 — Survival & POD24 prediction (+ multivariate Cox)  [was Fig 4]
  Figure 4 — CD14+ FDC: discovery, characterization, and tumor support  [was Fig 5]
  Figure S9 — CD14+ FDC validation and extended characterization [after Fig 4]
  Figure 5 — Myeloid ecosystem  [was Fig 6]
  Figure S10 — M2 Macrophages in FDC network zone [after Fig 5]
  Figure S11 — S100A9+ MDSC-like myeloid characterization [after S10]
  Figure S12 — VISTA targeting + checkpoint signaling [after S11]
  Figure 6 — Agent-based model: treatment scenario simulations [promoted from S13]
  Figure 7 — Dual-compartment immune evasion model (capstone)
  Table S2 — ABM parameters [last, supplementary table]
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("output")
HYP = OUT / "hypotheses_v8"
QC  = OUT / "qc"

# ── Figure list with captions ──────────────────────────────────────────
FIGURES = [
    # (path, figure_number, title, caption)

    # ── Cell type annotation ──
    (QC / "fig_cell_type_annotation.png", 1,
     "Dataset overview and cell type annotation",
     "(a) Study design: follicular lymphoma tissue on 4 TMAs, ~150 tumor ROIs, "
     "imaged by mass cytometry with T-panel (immune) and S-panel (stromal/myeloid), "
     "yielding ~4.2M cells. "
     "(b-c) UMAP embeddings colored by cell type for T-panel and S-panel. "
     "(d-e) Mean marker expression heatmaps (z-scored) validate cell type identity. "
     "(f-g) Cell type composition across tumor cores."),

    # ── S1 inline: QC (related to Fig 1) ──
    (QC / "fig_qc_combined.png", "S1",
     "Segmentation and marker quality control",
     "Supplementary Figure S1, related to Figure 1. "
     "(a) T-panel marker dynamic range (p99 arcsinh expression, pooled tumor cores): "
     "green = good (>1.0), orange = average (0.3–1.0), red = low (<0.3). "
     "Low-dynamic-range markers: LAG3, T-Bet, CTLA4, p53, pSTAT3, PD-L1, FoxP3, ICOS, GATA3. "
     "(b) S-panel marker dynamic range: low-dynamic-range markers include BCL6, PD-L1. "
     "(c) Cellpose hybrid segmentation illustration (350×350 µm crop). "
     "(d) Cell area distribution (T-panel, median 68 µm²). "
     "(e) Cell area by cell type (T-panel violin). "
     "(f) Overall cell type composition: FL tumor vs normal tonsil (stacked bar)."),

    # ── S2 inline: cross-panel concordance (related to Fig 1) ──
    (QC / "fig_cross_panel_concordance.png", "S2",
     "Cross-panel concordance: T-panel vs S-panel on serial sections",
     "Supplementary Figure S2, related to Figure 1. "
     "(a) DNA-based serial-section registration: fluorescence overlay (green=S, magenta=T) "
     "confirms high spatial alignment (FL32 concordant example). "
     "(b) Smoothed CD20 expression maps from S-panel and T-panel on the same tissue section "
     "(r=0.903). "
     "(c) Global cell type composition in both panels (broad categories). "
     "(d) Per-ROI scatter: T-panel vs S-panel fractions for CD4 T, CD8 T, B cells, Myeloid "
     "(n=115 paired ROIs; Pearson r=0.63-0.75 for CD4 T, CD8 T, and Myeloid; r=0.46 for B cells). "
     "(e) B cell concordance improves from r=0.46 (all ROIs) to r=0.75 (good ROIs) "
     "when excluding low-quality ROIs."),

    # ── S3: mutation × cell type (negative result, part of annotation chapter) ──
    (HYP / "fig_mutation_celltype_heatmap.png", "S3",
     "Mutation × cell type fraction associations",
     "Supplementary Figure S3, related to Figure 1. Heatmap of log₂ fold change "
     "in cell type fractions between mutant and wild-type patients for 15 recurrently "
     "mutated genes. Stars indicate nominal significance (*p<0.05, **p<0.01, "
     "***p<0.001, Mann-Whitney U). None of 555 tests survive FDR correction "
     "(all q>0.07). Mutations in FL do not have detectable large-scale effects on "
     "cell type composition as measured by IMC."),

    # ── Compartments + follicular biology (new Fig 2, v2) ──
    (HYP / "fig2_compartments_v2.png", 2,
     "Spatial tissue compartments and follicular biology",
     "UTAG tissue domain analysis using one-hot cell-type features (max_dist=50 µm). "
     "(a) T-panel cell type composition per compartment (heatmap; red sidebar = follicular, "
     "blue = interfollicular). "
     "(b) Representative ROI (C1_FL34) colored by compartment. "
     "(c) Cell type composition gradient across 9 compartments (follicle → T zone). "
     "(d) CD8 T cell exhaustion fraction (TOX+PD-1+): 73.6% at GC core vs 7.2% in T zone. "
     "(e) Exhaustion marker expression (TOX, PD-1, CD39) on CD8 T cells. "
     "(f) FL vs tonsil CD8 exhaustion comparison. "
     "(g) Compartment spatial adjacency (row-normalized neighbor fraction within 50 µm, "
     "diagonal masked): neighbors concentrate on gradient-adjacent compartments, ordering "
     "them concentrically. "
     "(h) Treg vs CD8 effector fraction: Treg dominates follicular zones. "
     "(i) FL vs tonsil Treg distribution: FL-specific Treg enrichment. "
     "(j) Pairwise interaction enrichment (permutation z-scores, K=10, 200 perms): "
     "Mac–CD8 T and Mac–exhausted-CD8 T enriched follicularly (GC core z=+15, +27), depleted in T cell zone (z=−13, −7)."),

    # ── S4: compartment ROI examples (related to Fig 2) ──
    (HYP / "fig2_compartments_v2_supp.png", "S4",
     "Compartment spatial examples across representative ROIs",
     "Supplementary Figure S4, related to Figure 2. "
     "Gallery of 9 key T-panel compartments. Each pair shows the compartment "
     "highlighted in a representative ROI (left) and cell types within that "
     "compartment only (right). Follicular: GC core, Follicle core, "
     "Follicle mantle, B cell follicle, B cell zone. "
     "Interfollicular: T cell zone, Treg-enriched zone, Macrophage-rich zone, "
     "Follicle-T zone interface."),

    # ── S5: S-panel compartments (related to Fig 2) ──
    (HYP / "fig_compartments_combined_S.png", "S5",
     "S-panel spatial tissue compartments",
     "Supplementary Figure S5, related to Figure 2. "
     "S-panel UTAG tissue compartments. "
     "(a) S-panel cell type composition per compartment (heatmap). "
     "(b-c) Representative ROIs from two TMAs colored by compartment. "
     "(d-i) Six key compartments: each pair shows compartment highlighted (left) "
     "and cell types within that compartment only (right). "
     "Follicular compartments: B cell zone (BCL2+/PAX5+), FDC network zone, FDC/myeloid zone. "
     "Interfollicular: T cell zone, Stromal/CAF zone."),

    # ── S6: FDC zone validation (related to Fig 2) ──
    (HYP / "fig_fdc_zone_vs_cd21_raw.png", "S6",
     "FDC network zone validation: concordance with raw CD21 signal",
     "Supplementary Figure S6, related to Figure 2. "
     "Spatial concordance between UTAG-defined FDC network zone "
     "and raw CD21 IMC signal across 4 B1 ROIs with decreasing FDC zone fraction "
     "(49%, 27%, 14%, 4%). "
     "Left: raw CD21 pixel intensities (inferno colormap, per-ROI 99.5th percentile clip). "
     "Center: cell-level CD21 expression (z-scored, after segmentation and transformation). "
     "Right: UTAG compartment map with FDC network zone highlighted in red and individual "
     "FDC cells as gold dots. "
     "The FDC network zone tracks dense CD21 meshworks faithfully."),

    # ── S7: Tfh enrichment (related to Fig 2) ──
    (HYP / "fig_tfh_enrichment.png", "S7",
     "TFR composition, PD-1-hi Tfh niches, and spatial co-localization",
     "Supplementary Figure S7, related to Figure 2. "
     "(a) TFR (FoxP3+CXCR5>2) vs classical Treg composition across compartments: "
     "TFR fraction increases from <2% in T cell zones to 70% at the GC core, "
     "indicating that intrafollicular Tregs are predominantly T follicular regulatory cells. "
     "(b) PD-1-hi (>1.5) Tfh density (% of all cells) by compartment: highest at GC core. "
     "PD-1-lo Tfh at boundary compartments likely include Tfr and CXCR5 spillover artifacts. "
     "Tfh are rare (median 3/ROI, 0.03% of typed cells) and do not predict survival "
     "(PFS rho=-0.07, p=0.46), transformation (p=0.61), or EZH2 status (p=0.59). "
     "(c) Marker profile: PD-1-hi Tfh vs CD4 non-Tfh — PD-1-hi Tfh show elevated "
     "CXCR5, PD-1, TOX, and FOXP3 compared to bulk CD4 T cells. "
     "(d) Permutation-based neighborhood enrichment (K=10, 500 permutations): "
     "PD-1-hi Tfh at GC core are enriched for Tregs (z=+22.3), B cells (z=+10.6), "
     "and CD4 T cells (z=+11.3), forming a structured multi-cell hub. "
     "(e) Full ROI scatter with PD-1-hi Tfh (gold stars), Tregs (purple), "
     "B cells (blue), CD4 T (green); zoom rectangle indicates region shown in (f). "
     "(f-g) Two zoom examples of GC-center Tfh-Treg-B cell hubs, "
     "showing spatial co-localization within follicular domains (red tint)."),

    # ── S8: Grade-associated architecture (related to Fig 2, before Fig 3) ──
    (HYP / "fig_grade_supplementary.png", "S8",
     "Grade-associated changes in proliferation, myeloid composition, and follicle architecture",
     "Supplementary Figure S8, related to Figure 2. "
     "Patient-level metrics (ROIs averaged per patient) stratified by centrally reviewed "
     "histologic grade (FOLL1, FOLL2, FOLL3A). Box: median, IQR, 1.5xIQR whiskers; points "
     "are individual patients. P-values: Kruskal-Wallis across grades (*p<0.05, **p<0.01, "
     "***p<0.001). (a) Ki-67+ B-cell fraction (S-panel) — internal consistency check that "
     "single-cell B-cell typing recovers the centroblast-count gradient defining grade. "
     "(b) M1 Macrophage fraction (S-panel). (c) S100A9+ MDSC-like myeloid fraction (S-panel). "
     "(d) Macrophage fraction (T-panel). (e) Macrophage-rich-zone UTAG compartment fraction "
     "(T-panel; absent in FOLL1/FOLL2, emerges at FOLL3A). (f) Macrophage spatial clustering "
     "via Besag's L at r=25 um (T-panel; L=0 under spatial randomness, lower = more dispersed): "
     "macrophages decompartmentalize at higher grade. (g) Mean follicle compactness (4*pi*A/P^2, "
     "T-panel; 1 = perfect circle, lower = more irregular boundary). (h) Representative FOLL1 ROI "
     "(B1_FL41): discrete circular follicles, sparse clustered macrophages. (i) Representative "
     "FOLL3A ROI (B1_FL14): single irregular follicular mass, abundant dispersed macrophages. "
     "n=110 patients S-panel (47/44/19), n=99 T-panel (45/36/18). Grade-associated architectural "
     "changes are large in magnitude but do not independently predict outcome (Cox, Figure 3), "
     "indicating the CD14 prognostic signal operates on an axis partly orthogonal to grade."),

    # ── Survival analysis (was Fig 4, now Fig 3) ──
    (HYP / "fig_survival_v2.png", 3,
     "Survival analysis, biomarker discovery, and multivariate validation",
     "Comprehensive survival screen and CD14 biomarker validation in treated patients. "
     "(a) Cell-type fraction forest — PFS (univariate Cox, HR per SD): M2 Mac and FDC predict shorter PFS. "
     "(b) Cell-type fraction forest — OS: M2 Mac, FDC, S100A9+ predict shorter OS. "
     "(c) Transformation forest (Mann-Whitney, all patients): S100A9+ (3.4x), "
     "M1 Mac (1.9x), M2 Mac (15.6x) enriched in transformers. "
     "(d) S-panel marker forest — PFS: CD14 (HR=1.59, P<0.0001) is the strongest predictor. "
     "(e) KM: CD14 high vs low splits PFS. "
     "(f) Multivariate Cox: CD14 HR remains significant across progressive adjustment "
     "(univariate → +FLIPI → +grade → +stage+age; full model HR=1.50, P=0.0007). "
     "(g) Full model forest: CD14 is the only independent predictor of PFS "
     "(FLIPI P=0.088, grade/stage/age all NS). "
     "(h) POD24 prediction: CD14+FLIPI (AUC=0.77) vs FLIPI-only ROC."),

    # ── CD14+ FDC biology (consolidated, was Fig 5, now Fig 4) ──
    (HYP / "fig_fdc_cd14_main.png", 4,
     "CD14+ FDCs: discovery, characterization, and tumor support",
     "(a) CD14 expression by cell type (IMC): FDC is second highest after myeloid. "
     "(b) Compartment localization: CD14-high FDCs 89% follicular vs 65% for CD14-low "
     "(chi-sq=5655, P<0.001). "
     "(c) Compartment-specific survival: follicular FDC CD14 predicts PFS (HR=1.52, P=0.008) "
     "and OS (HR=1.81, P=0.0009); interfollicular FDC CD14 does not (PFS HR=1.00, OS HR=0.74, both NS). "
     "(d) Intrafollicular marker profile (follicular compartments only): CD14-high FDCs "
     "upregulate CD21 +2.2, HLA-I +1.4, CXCL13 +1.1, and VISTA. "
     "(e) Intrafollicular neighbors (k=10): CD14-high FDCs enriched for BCL2+ B (+9.3pp), "
     "M1 Mac (+9.3pp), CD8 T (+5.8pp); depleted for PAX5+ B (-20.6pp). "
     "(f) Representative tissue microenvironment (B1_FL8): segmented cell scatter "
     "with raw IMC composite inset (CD21=green, CD14=red, CD68=magenta, CD8=cyan; "
     "CD21+CD14 overlap appears yellow) showing CD14-high FDCs co-localizing with "
     "macrophages and CD8 T cells."),

    # ── S9: FDC validation (related to Fig 4) ──
    (HYP / "fig_fdc_cd14_suppl.png", "S9",
     "CD14+ FDC validation and signaling characterization",
     "Supplementary Figure S9, related to Figure 4. "
     "(a) CD14+ cell composition (pie: myeloid 30%, FDC 25%, spillover 45%) and "
     "distance-dependent spillover gradient (rho=-0.115). "
     "(b) scRNA-seq validation (Han 2022): CD14 mRNA highest in myeloid, second in FDC "
     "(19% FDCs positive), confirming IMC protein finding. "
     "(c) CD14 expression on FDCs: FL tumor vs normal tonsil (Mann-Whitney); FL FDCs "
     "have significantly elevated CD14 (p=1.4e-44). "
     "(d) FDC transcriptional activity: CD14+ FDCs have 1.6x higher UMI counts per cell "
     "(median 3,009 vs 1,922, P=6.0e-7), confirming CD14 marks a transcriptionally "
     "hyperactive state rather than myeloid lineage. "
     "(e) B cell proliferation: follicular B cells within 30 px of CD14-high FDCs "
     "have higher Ki-67 (mean=0.483 vs 0.301, P=2.9e-32). "
     "(f) B cell HLA expression near CD14+ FDCs. "
     "(g) B cell survival signals (BAFF, APRIL, TGF-β1, IL-6) "
     "in CD14+ vs CD14- FDCs; TGF-β1 (*) and IL-6 (**) significantly elevated; "
     "APRIL trending (p=0.13), BAFF unchanged. "
     "(h) Signaling molecule expression across cell types (scRNA-seq dot plot): "
     "FDCs are the dominant source of CXCL13 and CCL21; BAFF and APRIL are "
     "predominantly myeloid-derived; myeloid cells also dominate VISTA. "
     "(i) CXCL13-CD21 per-ROI concordance (ρ=0.564, p=1.3e-12), confirming "
     "FDC-derived CXCL13 organizes the follicular niche. "
     "(j) IMC protein heatmap (z-scored): 10 signaling markers × 8 cell types; "
     "VISTA peaks on M2 Mac, CXCL13 on FDC, CD14/S100A9 on MDSC-like cells."),

    # ── Myeloid ecosystem (was Fig 6, now Fig 5) ──
    (HYP / "fig_macrophage_biology.png", 5,
     "Myeloid ecosystem: compartmentalization and spatial interactions",
     "Spatial organization and functional specialization of M1, M2, and S100A9+ "
     "myeloid subtypes. "
     "(a) Functional marker profiles. "
     "(b) Compartment distribution: M2 Mac uniquely follicular (56.5%); M1, S100A9+ "
     "predominantly interfollicular. "
     "(c) Specific compartment localization: M2 Mac concentrates in the FDC network zone "
     "(41%); M1 and S100A9+ distribute across T cell and other interfollicular zones. "
     "(d) Permutation-based neighborhood z-scores (k=10 nearest, 200 perms) "
     "within the FDC network zone; values clipped at ±50 for visibility "
     "(M1/M2 self-enrichment reaches +83/+122). "
     "(e) Myeloid-lymphocyte distances. "
     "(f) VISTA checkpoint expression by myeloid subtype. "
     "(g) EP300 mutation associated with VISTA upregulation, strongest in M2 Mac (P=0.004). "
     "(h) Marker drivers of myeloid-rich microenvironment. "
     "See Figs S10-S12 for detailed M2 Mac, S100A9+, and VISTA analyses."),

    # ── S10: M2 Mac in FDC network zone (related to Fig 5) ──
    (HYP / "fig_m2_mac_fdc_zone.png", "S10",
     "M2 Macrophages in FDC network zone: representative ROI with CD21 validation",
     "Supplementary Figure S10, related to Figure 5. "
     "(a) Representative ROI (C1_FL12, 46 M2 in FDC zone). Left: cell types in FDC network "
     "zone with M2 Mac as red stars; inset zooms on the densest M2 niche. Non-FDC zone cells "
     "shown in gray. Right: CD21 signal (inferno colormap) validates that UTAG-defined FDC "
     "network zone corresponds to CD21-bright FDC meshwork. M2 Macs populate the "
     "FDC network zone but remain spatially separated from CD21-bright FDCs at "
     "the single-cell level, consistent with the neighborhood depletion quantified "
     "in Fig 5d."),

    # ── S11: S100A9+ MDSC-like myeloid (related to Fig 5) ──
    (HYP / "fig_s100a9_myeloid.png", "S11",
     "S100A9+ MDSC-like myeloid cells in follicular lymphoma",
     "Supplementary Figure S11, related to Figure 5. "
     "(a) Neighborhood enrichment z-scores (K=10, 200 permutations): S100A9+ cells are "
     "strongly enriched near M1 macrophages (z=+72), endothelial cells (z=+61), and CD8 T "
     "cells (z=+46), while depleted near B cells. "
     "(b) Representative ROI (C1_FL10): S100A9+ (brown), M1 Mac (red), CD8 T (gold). "
     "(c) Co-occurrence ecology: per-ROI Spearman correlation of S100A9+ fraction with "
     "other cell types — S100A9+ co-occurs with M1 Mac (ρ=+0.53), GzmB+ macrophages "
     "(ρ=+0.43 cross-panel), and FDC (+0.32), marking inflamed microenvironments. "
     "(d) Compartment-specific phenotype: follicular S100A9+ express dramatically higher "
     "VISTA (3.17 vs 1.82), CD68, and CD11b than interfollicular, consistent "
     "with an elevated suppressive phenotype; CD14 unchanged across compartments. "
     "(e) scRNA-seq validation (Han 2022, 108 vs 781 myeloid cells): calprotectin program "
     "(S100A8/A9/A12), FCN1, VCAN, TYROBP upregulated; CD14, CD11b, VISTA concordant "
     "with IMC; HLA-DRA unchanged."),

    # ── S12: VISTA targeting + checkpoint signaling (related to Fig 5) ──
    (HYP / "fig_vista_targeting.png", "S12",
     "VISTA targeting landscape and checkpoint signaling",
     "Supplementary Figure S12, related to Figure 5. "
     "(a) VISTA+ vs VISTA− myeloid gene expression (scRNA-seq): VISTA+ cells "
     "co-express CD163, TGF-β1, TIM-3, Gal-9, confirming M2-skewed suppressive "
     "phenotype. "
     "(b) Checkpoint stacking: 94% of VISTA+ myeloid co-express ≥2 additional "
     "checkpoints (PD-L1, TIM-3, IDO1, SIGLEC10, Gal-9, PD-L2). "
     "(c) Druggable target heatmap: % cells expressing surface molecules across "
     "myeloid, FDC, and other cell types. "
     "(d) Checkpoint landscape at the transcript level (scRNA-seq, Han 2022): "
     "VISTA (VSIR) dominates PD-L1 (CD274) in myeloid cells (56% vs 12% positive), "
     "confirming that VISTA — not PD-L1 — is the operative myeloid checkpoint in FL. "
     "(e) VISTA+ fractions (% cells > 0.5 scaled) by compartment and cell type "
     "(IMC). Per-cell hierarchy in follicle: M2 Mac 91%, S100A9+ 88%, CD14+ FDC "
     "32%, CD14- FDC 15%; all populations substantially reduced in "
     "interfollicular compartments. "
     "(f) Absolute VISTA+ cell counts in the follicular compartment (pooled "
     "across tumor cores, IMC): CD14+ FDCs are the largest VISTA+ source "
     "population in the follicle, reflecting their high abundance. "
     "(g) VISTA in FL vs normal tonsil (IMC, S-panel). Bars = pooled "
     "per-cell mean +/- SEM; stars from per-ROI Mann-Whitney (one-sided, "
     "FL > tonsil): *p<0.05, **p<0.01, ***p<0.001. Per-cell fold change "
     "FL/tonsil: M2 Mac 3.8x, S100A9+ 3.4x, CD14+ FDC 2.1x, "
     "CD14- FDC 1.3x (ns). "
     "See also Figure S9(e-h) for additional signaling characterization."),

    # ── Figure 6: ABM treatment simulations (promoted to main figure) ──
    (HYP / "fig_abm_treatment.png", 6,
     "Agent-based model: treatment scenario simulations",
     "Agent-based model (Mesa framework) encoding the dual-compartment evasion "
     "architecture: 8 cell types, 4 diffusive chemokine fields, concentric zone "
     "geometry (100×100 grid, circular tissue r=45). "
     "Left column: population trajectories over 2000 ticks. "
     "Right column: final spatial snapshot (t=2000). "
     "Model includes a 3% CD20-negative tumor subclone with 20% slower "
     "proliferation (fitness cost of CD20 loss). Rituximab is fully blocked by "
     "CD20 loss; modern bispecifics retain partial (~50%) activity against "
     "CD20-low cells. "
     "(a-b) Baseline: tumor B grows to ~2460, CD8 T excluded from follicle. "
     "(c-d) Anti-VISTA monotherapy: lifts suppression but tumor persists. "
     "(e-f) Rituximab monotherapy: deep durable response (nadir ~66 at "
     "t=777), slow CD20-negative relapse (final ~155). "
     "(g-h) Bispecific CD20-CD3 monotherapy: slow clearance (t=766). "
     "(i-j) Anti-VISTA + Rituximab: CD20-negative residual (final ~193). "
     "(k-l) Anti-VISTA + Bispecific CD20-CD3: clearance at t=405, nearly "
     "twice as fast as bispecific alone. "
     "(m-n) Triple combination: clearance at t=177, modest speed improvement "
     "over the two-drug combination. "
     "Immune cell turnover (death 0.3%/tick + homeostatic immigration) maintains "
     "CD8 at 112-120. Model code: github.com/oelemento/FL-ABM."),

    # ── Synthesis model (Fig 7, capstone) ──
    (HYP / "fig_model_immune_evasion.png", 7,
     "Dual-compartment immune evasion model in follicular lymphoma",
     "Proposed model integrating all spatial and genetic findings. "
     "Intrafollicular sanctuary: immune exclusion (2.0% CD8 T, 73.6% exhausted), "
     "VISTA+ CD14+ FDCs organize the niche, M2 Mac in FDC zone (41%). "
     "Treg barrier at boundary (Treg:CD8 = 1.34) with CD39+ U-shaped gradient. "
     "Interfollicular suppression: VISTA+ on 56% of myeloid (vs PD-L1 12%), "
     "S100A9+ MDSC-like cells (3.6x in transformers), perivascular niches. "
     "Biomarker: CD14 predicts PFS (HR=1.50, p=0.0007) independent of FLIPI, "
     "grade, stage, age. EP300 mutation associated with VISTA+ M2 Mac (p=0.003), which "
     "associates with transformation (OR=4.48). POD24: CD14+FLIPI AUC=0.77."),
]


def make_key_findings_page(pdf):
    """Summary page: key discoveries and new hypotheses."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
    ax.axis("off")

    y = 0.97
    fs = 9
    dy_title = 0.042
    dy_head = 0.030
    dy_line = 0.025
    dy_section = 0.016

    def _title(text, yy):
        ax.text(0.0, yy, text, fontsize=13, fontweight="bold",
                transform=ax.transAxes, va="top")
        return yy - dy_title

    def _head(text, yy):
        ax.text(0.0, yy, text, fontsize=fs, fontweight="bold",
                transform=ax.transAxes, va="top")
        return yy - dy_head

    def _text(text, yy):
        for line in text.split("\n"):
            ax.text(0.02, yy, line, fontsize=fs,
                    transform=ax.transAxes, va="top")
            yy -= dy_line
        return yy

    y = _title("Key Discoveries", y)

    y = _head("1. Dual-compartment immune evasion architecture", y)
    y = _text("\u2022  Intrafollicular sanctuary: CD8 T cells excluded (2.0% at GC core vs 28.3% in T zone);\n"
              "   the few that penetrate are rapidly exhausted (73.6% TOX+PD-1+ vs 7.2% in T zone)\n"
              "\u2022  Interfollicular suppression: VISTA+ on 56% of myeloid cells (vs PD-L1 12%), 94% co-express\n"
              "   2+ checkpoints; effector CD8 T cells (74.7% interfollicular) in direct contact with myeloid\n"
              "\u2022  Treg barrier at follicle boundary (Treg:CD8 = 1.34), U-shaped CD39+ gradient (GC 67%,\n"
              "   B zone 8%, Treg zone 41%)", y)
    y -= dy_section

    y = _head("2. CD14+ FDCs as active immune organizers (not passive scaffolds)", y)
    y = _text("\u2022  Co-express VISTA + IDO, secrete CXCL13, sit in tumor B cell nests\n"
              "\u2022  Best spatial biomarker: CD14 predicts PFS (HR=2.10, p=0.002) and POD24 (AUC=0.77)\n"
              "\u2022  scRNA-seq confirms CD14 marks a hyperactive state (4.7\u00d7 more UMIs)", y)
    y -= dy_section

    y = _head("3. VISTA, not PD-L1, is the dominant checkpoint in FL", y)
    y = _text("\u2022  VISTA 56% vs PD-L1 12% on myeloid cells; completely independent (\u03c1=+0.02)\n"
              "\u2022  IDO is NOT independent of VISTA in survival models\n"
              "\u2022  Implies anti-PD-1/PD-L1 therapies miss the dominant suppressive axis in FL", y)
    y -= dy_section

    y = _head("4. Spatial validation of Dave et al. 2004 IR2 signature", y)
    y = _text("\u2022  IR2 (macrophage/FDC genes) predicted poor survival 20 years ago \u2014 we show its spatial\n"
              "   correlate at single-cell resolution\n"
              "\u2022  CD14 integrates signal from macrophages (~35%), CD14-high FDCs (~25%), and proximity-\n"
              "   dependent spillover (~40%)", y)
    y -= dy_section * 1.5

    y = _title("New Hypotheses for Future Work", y)

    y = _text("\u2022  Anti-VISTA as primary checkpoint therapy in FL \u2014 PD-1/PD-L1 blockade may target the wrong\n"
              "   axis; ABM simulations support VISTA blockade + bispecific CD3-CD20\n"
              "\u2022  CD14 as a clinical biomarker \u2014 simple IHC-based CD14 scoring could stratify patients for\n"
              "   transformation risk and POD24\n"
              "\u2022  FDC activation state as therapeutic target \u2014 disrupting CD14+ FDC activation could collapse\n"
              "   the intrafollicular sanctuary\n"
              "\u2022  EP300 mutation \u2192 VISTA+ M2 Mac pathway \u2014 EP300 mutants have 4.48\u00d7 more VISTA+ M2 Macs,\n"
              "   linking genetics to microenvironment remodeling", y)

    pdf.savefig(fig, dpi=400)
    plt.close(fig)


def make_cohort_table(pdf):
    """Create Table 1: cohort description with All / Treated / Observation columns.

    Restricted to patients with matched IMC data (in-house TMAs only).
    """
    import sys, re, h5py
    sys.path.insert(0, "src")
    from clinical_linkage import normalize_sample_id, EXCLUDE_ROIS

    # Get IMC slide_IDs from T-panel
    f = h5py.File("output/all_TMA_T_utag_ct_merged.h5ad", "r")
    ds = f["obs"]["sample_id"]
    cats = [c.decode() if isinstance(c, bytes) else str(c)
            for c in ds["categories"][:]]
    sids = [cats[c] for c in ds["codes"][:]]
    f.close()
    exclude = re.compile(r"(?i)tonsil|prostate|kidney|spleen|adrenal|_ton_|_adr_")
    tumor_sids = sorted(set(s for s in sids
                            if not exclude.search(s) and not s.startswith("Biomax")))
    imc_slide_ids = set(normalize_sample_id(s) for s in tumor_sids)

    df = pd.read_csv("data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    df = df[df["slide_ID"].isin(imc_slide_ids)]  # restrict to IMC-matched
    # Merge grade (DIAG) from master_clinical_ezh2.csv
    try:
        grade_df = pd.read_csv("output/cd14_validation/master_clinical_ezh2.csv")
        grade_map = dict(zip(grade_df["slide_ID"], grade_df["DIAG"]))
        df["DIAG"] = df["slide_ID"].map(grade_map)
    except Exception:
        df["DIAG"] = None
    pts = df.drop_duplicates("Patient_ID")

    treated = pts[pts["INITIAL OBSERVATION"] != "Yes"]
    obs = pts[pts["INITIAL OBSERVATION"] == "Yes"]

    def grp_stats(g):
        """Compute summary stats for a patient group."""
        n = len(g)
        if n == 0:
            return {"n": 0}
        male = (g["SEX"] == "M").sum()
        female = (g["SEX"] == "F").sum()
        age_med = g["AGE"].median()
        age_q1 = g["AGE"].quantile(0.25)
        age_q3 = g["AGE"].quantile(0.75)
        stage34 = (g["ANN ARBOR STAGE"] >= 3).sum()
        flipi_h = (g["FLIPI.1"] == "HIGH").sum()
        trans = (g["Transformation"] == "Yes").sum()
        grade1 = (g["DIAG"] == "FOLL1").sum()
        grade2 = (g["DIAG"] == "FOLL2").sum()
        grade3a = (g["DIAG"] == "FOLL3A").sum()
        os_ev = int(g["CODE_OS"].sum())
        pfs_ev = int(g["CODE_PFS"].sum())
        med_os = g["Overall survival (y)"].median()
        med_pfs = g["Progression free survival (y)"].median()
        return dict(n=n, male=male, female=female,
                    age_med=age_med, age_q1=age_q1, age_q3=age_q3,
                    stage34=stage34, flipi_h=flipi_h, trans=trans,
                    grade1=grade1, grade2=grade2, grade3a=grade3a,
                    os_ev=os_ev, pfs_ev=pfs_ev, med_os=med_os, med_pfs=med_pfs)

    a = grp_stats(pts)
    t = grp_stats(treated)
    o = grp_stats(obs)

    def pct(num, denom):
        return f"{num} ({100*num/denom:.0f}%)" if denom > 0 else "–"

    def age_str(s):
        return f"{s['age_med']:.0f} ({s['age_q1']:.0f}–{s['age_q3']:.0f})"

    page_w, page_h = 8.5, 11
    title_top = 0.99
    caption_bot = 0.02

    caption = ("Table 1. Cohort characteristics. "
               "Demographics, disease features, and survival outcomes for patients "
               "with matched IMC data on in-house TMAs (A1, B1, C1). "
               "Biomax TMA (commercial, 28 ROIs) has no clinical data and is excluded. "
               "IQR, interquartile range; FLIPI, Follicular Lymphoma International "
               "Prognostic Index; OS, overall survival; PFS, progression-free survival.")
    caption_lines = max(1, len(caption) // 110 + 1)
    caption_h_frac = 0.011 * caption_lines + 0.008

    fig = plt.figure(figsize=(page_w, page_h))

    # Title at top (same style as figure pages)
    fig.text(0.5, title_top, "Table 1. Cohort Characteristics",
             ha="center", va="top", fontsize=12, fontweight="bold")

    # Table axes
    ax = fig.add_axes([0.05, caption_h_frac + 0.01, 0.90, 0.88])
    ax.axis("off")

    rows = [
        ["", f"All (N={a['n']})", f"Treated (N={t['n']})", f"Obs. (N={o['n']})"],
        ["Demographics", "", "", ""],
        ["  Male / Female",
         f"{a['male']} / {a['female']}",
         f"{t['male']} / {t['female']}",
         f"{o['male']} / {o['female']}"],
        ["  Age, median (IQR)",
         age_str(a), age_str(t), age_str(o)],
        ["Disease", "", "", ""],
        ["  Grade 1",
         pct(a["grade1"], a["n"]),
         pct(t["grade1"], t["n"]),
         pct(o["grade1"], o["n"])],
        ["  Grade 2",
         pct(a["grade2"], a["n"]),
         pct(t["grade2"], t["n"]),
         pct(o["grade2"], o["n"])],
        ["  Grade 3A",
         pct(a["grade3a"], a["n"]),
         pct(t["grade3a"], t["n"]),
         pct(o["grade3a"], o["n"])],
        ["  Stage III–IV",
         pct(a["stage34"], a["n"]),
         pct(t["stage34"], t["n"]),
         pct(o["stage34"], o["n"])],
        ["  FLIPI High",
         pct(a["flipi_h"], a["n"]),
         pct(t["flipi_h"], t["n"]),
         pct(o["flipi_h"], o["n"])],
        ["  Transformation",
         pct(a["trans"], a["n"]),
         pct(t["trans"], t["n"]),
         pct(o["trans"], o["n"])],
        ["Outcomes", "", "", ""],
        ["  OS events",
         pct(a["os_ev"], a["n"]),
         pct(t["os_ev"], t["n"]),
         pct(o["os_ev"], o["n"])],
        ["  PFS events",
         pct(a["pfs_ev"], a["n"]),
         pct(t["pfs_ev"], t["n"]),
         pct(o["pfs_ev"], o["n"])],
        ["  Median OS (yr)",
         f"{a['med_os']:.1f}", f"{t['med_os']:.1f}", f"{o['med_os']:.1f}"],
        ["  Median PFS (yr)",
         f"{a['med_pfs']:.1f}", f"{t['med_pfs']:.1f}", f"{o['med_pfs']:.1f}"],
    ]

    n_rows = len(rows)
    table = ax.table(
        cellText=rows, colLabels=None, cellLoc="left", loc="upper center",
        bbox=[0.0, 0.40, 1.0, 0.55],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    col_widths = [0.30, 0.24, 0.24, 0.22]
    for (row, col), cell in table.get_celld().items():
        cell.set_width(col_widths[col])
        cell.set_edgecolor("white")
        cell.set_linewidth(0)
        text = cell.get_text().get_text()

        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=12)
            cell.set_facecolor("#E8E8E8")
            cell.set_edgecolor("#333333")
            cell.set_linewidth(1.5)
            cell.visible_edges = "BT"
        elif text in ["Demographics", "Disease", "Outcomes"]:
            cell.set_text_props(fontweight="bold", fontstyle="italic", fontsize=12)
            cell.set_facecolor("#F5F5F5")
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(0.5)
            cell.visible_edges = "T"
        else:
            cell.set_facecolor("white")
            cell.visible_edges = ""

        if col > 0:
            cell.get_text().set_ha("center")

    for col in range(4):
        cell = table.get_celld()[(n_rows - 1, col)]
        cell.set_edgecolor("#333333")
        cell.set_linewidth(1)
        cell.visible_edges = "B"

    pdf.savefig(fig, dpi=400)
    plt.close(fig)


def make_antibody_panel_table(pdf):
    """Create Table S1: antibody reagent panels (T-panel and S-panel).

    Renders the full reagent list (metal tag, target, dilution, clone, vendor,
    catalog, lot) from the antibody workbook, one page per panel.
    """
    xlsx = "manuscript/supplementary/FL_Manuscript_IMC_antibody_list_2026.xlsx"
    panels = [
        ("T-panel (immune)", "T B NK panel", "a"),
        ("S-panel (stromal/myeloid)", "Stromal Myeloid panel", "b"),
    ]
    # relative widths (sum ~1.0): Metal, Antibody, Dilution, Clone, Vendor, Catalog, Lot
    col_w = [0.11, 0.21, 0.10, 0.17, 0.15, 0.14, 0.12]
    page_w, page_h = 8.5, 11
    title_top = 0.99

    for panel_name, sheet, label in panels:
        raw = pd.read_excel(xlsx, sheet_name=sheet, header=None, usecols="B:H")
        raw = raw.dropna(how="all").reset_index(drop=True)
        import datetime as _dt
        def _fmt(v):
            if pd.isna(v):
                return ""
            if isinstance(v, _dt.time):  # Excel turned "1:50" dilutions into times
                return f"{v.hour}:{v.minute:02d}"
            return str(v).strip()
        header = [_fmt(x) for x in raw.iloc[0].tolist()]
        body = [[_fmt(v) for v in r.tolist()] for _, r in raw.iloc[1:].iterrows()]

        fig = plt.figure(figsize=(page_w, page_h))
        fig.text(0.5, title_top, f"Table S1. Antibody Panel: {panel_name}",
                 ha="center", va="top", fontsize=12, fontweight="bold")
        ax = fig.add_axes([0.04, 0.03, 0.92, 0.93])
        ax.axis("off")
        ax.text(-0.02, 1.02, label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom")

        table_data = [header] + body
        n_cols = len(header)
        colors = [["#E5E7EB"] * n_cols] + [["white"] * n_cols for _ in body]
        tbl = ax.table(cellText=table_data, cellColours=colors,
                       cellLoc="left", loc="upper center", bbox=[0.0, 0.0, 1.0, 1.0])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)
        n_rows = len(table_data)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_width(col_w[col])
            cell.set_edgecolor("none")
            cell.set_linewidth(0)
            cell.PAD = 0.02
            if row == 0:
                cell.set_text_props(fontweight="bold")
        for col in range(n_cols):
            top = tbl.get_celld()[(0, col)]
            top.set_edgecolor("#333333"); top.set_linewidth(1); top.visible_edges = "T"
            if n_rows > 1:
                fb = tbl.get_celld()[(1, col)]
                fb.set_edgecolor("#333333"); fb.set_linewidth(0.8); fb.visible_edges = "T"
            bot = tbl.get_celld()[(n_rows - 1, col)]
            bot.set_edgecolor("#333333"); bot.set_linewidth(1); bot.visible_edges = "B"

        pdf.savefig(fig, dpi=400)
        plt.close(fig)


def _render_param_page(pdf, title, rows, col_widths, footnote=None):
    """Render one page of a parameter table (harmonized with Table S1 style)."""
    fig = plt.figure(figsize=(8.5, 11))

    fig.text(0.5, 0.96, title,
             ha="center", va="top", fontsize=12, fontweight="bold")

    ax = fig.add_axes([0.04, 0.03, 0.92, 0.91])
    ax.axis("off")

    n_cols = len(rows[0])
    # Per-cell colors: gray header, light gray category rows, white else
    cell_colors = []
    for r_idx, row_data in enumerate(rows):
        if r_idx == 0:
            cell_colors.append(["#E5E7EB"] * n_cols)
        elif row_data[0] and not row_data[0].startswith("  "):
            cell_colors.append(["#F5F5F5"] * n_cols)
        else:
            cell_colors.append(["white"] * n_cols)

    table = ax.table(
        cellText=rows, cellColours=cell_colors,
        cellLoc="left", loc="upper center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    n_rows = len(rows)
    for (row, col), cell in table.get_celld().items():
        cell.set_width(col_widths[col])
        cell.set_edgecolor("none")
        cell.set_linewidth(0)
        cell.PAD = 0.03

        text = cell.get_text().get_text()
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=7)
        elif col == 0 and text and not text.startswith("  "):
            cell.set_text_props(fontweight="bold", fontstyle="italic", fontsize=7)

    # Top/bottom rules only (no internal horizontal separators)
    for col in range(n_cols):
        top = table.get_celld()[(0, col)]
        top.set_edgecolor("#333333")
        top.set_linewidth(1)
        top.visible_edges = "T"
        header_bot = table.get_celld()[(0, col)]
        # Header bottom border: need a second cell reference for it to draw
        if n_rows > 1:
            first_body = table.get_celld()[(1, col)]
            first_body.set_edgecolor("#333333")
            first_body.set_linewidth(0.8)
            first_body.visible_edges = "T"
        bot = table.get_celld()[(n_rows - 1, col)]
        bot.set_edgecolor("#333333")
        bot.set_linewidth(1)
        bot.visible_edges = "B"

    pdf.savefig(fig, dpi=400)
    plt.close(fig)


def make_parameter_table(pdf):
    """Create Table S1: ABM parameter justification (2 pages)."""
    col_widths = [0.22, 0.10, 0.36, 0.32]

    # ── Page 1: Geometry, cell counts, tumor B, CD8, suppression ──
    page1 = [
        ["Parameter", "Value", "Justification", "Reference"],
        # Geometry
        ["Spatial geometry", "", "", ""],
        ["  Grid cell size", "~10 µm", "Lymphocyte diameter 7-10 µm", "BioNumbers BNID 100507"],
        ["  FDC zone radius", "15 cells", "FDC network inner 1/3 of follicle", "Willard-Mack 2006"],
        ["  B zone radius", "30 cells", "FL follicle ~500 µm diameter", "Willard-Mack 2006"],
        ["  Boundary width", "5 cells", "Treg enrichment at T-B border", "Sayin 2018, JEM"],
        ["  Tissue radius", "45 cells", "TMA core radius", "This study"],
        # Counts
        ["Initial cell counts", "", "", ""],
        ["  FDC", "40", "Rare stromal network", "Park 2005, Immunology"],
        ["  Tumor B", "600", "Dominant (>75%) in follicle", "FL histopathology"],
        ["  CD8 T", "120", "Yields ~2% follicular, ~28% T zone", "This study, Fig 2"],
        ["  CD4 T / Tfh", "80", "CD4:CD8 > 1; Tfh enriched", "Pangault 2010, Leukemia 24:2080"],
        ["  Treg", "60", "Treg:CD8 = 1.34 at boundary", "This study, Fig 2"],
        ["  M2 Mac", "80", "~41% in FDC zone", "This study, Fig 5"],
        ["  M1 Mac", "50", "Interfollicular, pro-inflammatory", "This study, Fig 5"],
        ["  S100A9+ MDSC", "30", "~3% of total; 3.6x in transformers", "This study, Fig S11"],
        # Tumor B
        ["Tumor B proliferation", "", "", ""],
        ["  Proliferation rate", "0.005/tick", "Ki-67 median ~16% in FL", "Koster 2007, Haematologica 92:184"],
        ["  Survival dependence", "f(signal)", "FDC niche required (BAFF, IL-15)", "Laurent 2024, Blood 143:1080"],
        ["  FDC survival emission", "0.5", "Primary anti-apoptotic source", "Aguzzi 2014, Trends Immunol"],
        # CD8
        ["CD8 T cell killing", "", "", ""],
        ["  Base kill rate", "0.08", "2-16 kills/CTL/day in vivo", "Halle 2016, Immunity"],
        ["  Activation bonus", "0.25", "Additive sublethal cytotoxicity", "Weigelin 2021, Nat Commun"],
        ["  Exhaustion rate", "0.02", "Calibrated to 73.6% in GC", "This study, Fig 2"],
        ["  VISTA exhaustion", "0.008", "Contact-dependent quiescence", "Johnston 2019, Nature"],
        ["  IDO exhaustion", "0.006", "Diffusible metabolite-driven", "Liu 2018, Cancer Cell"],
        # Suppression
        ["Suppression", "", "", ""],
        ["  Background", "0.05", "Baseline immunosuppression in FL", "This study, Fig 2"],
        ["  Treg strength", "0.3", "CTLA-4, IL-10, TGF-β contact", "Vignali 2008, Nat Rev Immunol"],
        ["  M2 Mac strength", "0.25", "Arginase, IL-10 suppression", "Mantovani 2002, Trends Immunol; Lines 2014, Cancer Res"],
        ["  MDSC strength", "0.35", "Potent immunosuppressive activity", "Veglia 2021, Nat Rev Immunol"],
    ]

    footnote1 = (" Emergent calibration: parameter tuned so model "
                 "reproduces observed spatial phenotype. Kill formula: "
                 "base × (1−exhaustion) × (1−suppression) × (1 + activation×bonus).")

    _render_param_page(pdf, "Table S2. Agent-Based Model Parameters (1/2)",
                       page1, col_widths, footnote=footnote1)

    # ── Page 2: Chemokines, VISTA, chemotaxis, turnover, interventions ──
    page2 = [
        ["Parameter", "Value", "Justification", "Reference"],
        # Chemokines
        ["Chemokine fields", "", "", ""],
        ["  CXCL13 emission (FDC)", "0.4", "FDCs primary follicular source", "Ansel 2000, Nature"],
        ["  CXCL13 decay", "0.01", "Slow (heparan sulfate binding)", "Miller 2018, Front Immunol"],
        ["  CXCL9/10 emission (M1)", "0.05×act.", "IFN-γ inducible", "Groom 2011, Immunol Cell Biol"],
        ["  CXCL9/10 decay", "0.02", "Faster: inducible, transient", "Groom 2011"],
        ["  IDO emission (FDC zone)", "0.3", "Field-level; IMC FDC IDO partly M2 Mac spillover", "Munn 2016, Trends Immunol"],
        ["  IDO emission (M2)", "0.2", "IFN-γ inducible in M2", "Munn 1999, J Exp Med"],
        ["  IDO emission (MDSC)", "0.25", "Strongest per-cell IDO", "This study, Fig S11"],
        ["  IDO decay", "0.08", "Fast: kynurenine t½ ~hours", "Badawy 2017, Int J Tryp Res"],
        # VISTA
        ["VISTA (contact-dependent)", "", "", ""],
        ["  FDC vista level", "0.6", "CD14+ FDCs VISTA+ at protein", "This study, Fig 5"],
        ["  M2 Mac vista level", "0.5", "56% myeloid VISTA+", "This study; Lines 2014, Cancer Res"],
        ["  MDSC vista level", "0.7", "Highest per-cell VISTA", "This study, Fig S11"],
        # Chemotaxis
        ["Chemotaxis", "", "", ""],
        ["  CD8 → CXCL9/10", "p=1.0", "CXCR3-driven, highly motile", "Groom 2011"],
        ["  CD4 → CXCL13", "p=0.8", "CXCR5 follicle homing", "Breitfeld 2000, J Exp Med"],
        ["  Treg → CXCL13", "p=0.7", "Moderate CXCR5; noise → boundary", "Sayin 2018, JEM"],
        ["  M2 → CXCL13", "p=0.6", "Follicle-attracted", "This study, Fig 5"],
        ["  FDC / Tumor B", "static", "Sessile stromal / follicular cells", "Allen 2008, Semin Immunol"],
        # Turnover
        ["Immune turnover", "", "", ""],
        ["  Death rate", "0.003/tick", "T cell t½ 3-42 days (subpopulation-dep.)", "Gossel 2017, eLife"],
        ["  Spawn probability", "0.15/deficit", "Homeostatic replenishment", "Bousso 2003, Nat Immunol"],
        ["  Spawn location", "r = 0.7-1.0 R", "Perivascular immigration", "This study, Fig 2"],
        # Interventions
        ["Anti-VISTA", "", "", ""],
        ["  VISTA block", "70%", "Direct VISTA-PSGL1 blockade", "Iadonato 2023, Front Immunol"],
        ["  IDO reduction", "30%", "Indirect via VISTA signaling", "Thisted 2024, Nat Commun"],
        ["  Kill bonus", "+0.15", "Restored CD8 effector function", "HMBD-002, ASCO 2021"],
        ["Rituximab", "", "", ""],
        ["  Kill rate", "0.012/tick", "ADCC/CDC of CD20+ B cells", "Cartron 2002, Blood"],
        ["  Penetration decay", "0.03", "Binding-site barrier", "Fujimori 1990, J Nucl Med"],
        ["  CD20-neg fraction", "0.03", "Clinical CD20 loss under treatment", "Czuczman 2008, Clin Cancer Res"],
        ["  CD20-neg prolif factor", "0.8", "Fitness cost of CD20 loss", "Design: clinical PFS ~12-24 mo"],
        ["Bispecific CD20-CD3", "", "", ""],
        ["  Kill bonus", "+0.10", "T cell redirection (CR 60%)", "Budde 2022, Lancet Oncol"],
        ["  Exhaustion bypass", "30%", "Forced synapse overcomes TOX", "Bacac 2018; Hutchings 2021"],
        ["  Seek probability", "0.4/tick", "Bispecific bridging redirects", "Krupka 2016, Leukemia"],
        ["  CD20-low activity", "0.5×", "Partial engagement (higher avidity)", "Bacac 2018; Hutchings 2021"],
    ]

    footnote2 = ("VISTA is transmembrane (contact-dependent, not diffusible). "
                 "IDO produces diffusible kynurenines. "
                 "Diffusion kernel: 3×3 cross-shaped, center-weighted (4/8). "
                 "Model code: github.com/oelemento/FL-ABM.")

    _render_param_page(pdf, "Table S2. Agent-Based Model Parameters (2/2)",
                       page2, col_widths, footnote=footnote2)


def _make_caption_page(fig_num, title, caption):
    """Create a matplotlib figure with just title at top (no caption).

    Returns a PDF bytes buffer for the caption overlay page.
    The title wraps to multiple lines so long titles do not overflow the
    page width (3% side margins => ~94% usable width at fontsize 12).
    """
    import io
    import textwrap
    page_w, page_h = 8.5, 11
    fig = plt.figure(figsize=(page_w, page_h))
    # ~88 chars of bold 12pt fit across the usable page width; wrap longer titles.
    full = f"Figure {fig_num}. {title}"
    wrapped = "\n".join(textwrap.wrap(full, width=80, break_long_words=False))
    fig.text(0.5, 0.99, wrapped,
             ha="center", va="top", fontsize=12, fontweight="bold",
             linespacing=1.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=400, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_figure_legends_pages(figures):
    """Generate pages collecting all figure legends.

    Returns a list of PDF byte buffers (one per page).
    """
    import io
    import textwrap

    page_w, page_h = 8.5, 11
    margin_x = 0.08
    text_width = 1.0 - 2 * margin_x
    bufs = []

    # Collect all legends
    legends = []
    for img_path, fig_num, title, caption in figures:
        if isinstance(fig_num, int) or (isinstance(fig_num, str) and not fig_num.startswith("S")):
            prefix = f"Figure {fig_num}. {title}."
        else:
            prefix = f"Figure {fig_num}. {title}."
        legends.append((prefix, caption))

    # Render legends onto pages
    fig = plt.figure(figsize=(page_w, page_h))
    fig.text(0.5, 0.97, "Figure Legends",
             ha="center", va="top", fontsize=14, fontweight="bold")
    y = 0.93

    for prefix, caption in legends:
        full_text = f"{prefix} {caption}"
        # Estimate height needed (~60 chars per line at fontsize 9, ~0.016 per line)
        n_lines = max(1, len(full_text) // 75 + 1)
        block_h = n_lines * 0.016 + 0.02  # padding between entries

        if y - block_h < 0.05:
            # Start new page
            buf = io.BytesIO()
            fig.savefig(buf, format="pdf", dpi=400, facecolor="white")
            plt.close(fig)
            buf.seek(0)
            bufs.append(buf)
            fig = plt.figure(figsize=(page_w, page_h))
            fig.text(0.5, 0.97, "Figure Legends (continued)",
                     ha="center", va="top", fontsize=14, fontweight="bold")
            y = 0.93

        fig.text(margin_x, y, prefix,
                 ha="left", va="top", fontsize=9, fontweight="bold",
                 transform=fig.transFigure)
        fig.text(margin_x, y - 0.015, caption,
                 ha="left", va="top", fontsize=8, color="#333333",
                 wrap=True, transform=fig.transFigure,
                 linespacing=1.4)
        y -= block_h

    # Save last page
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=400, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    bufs.append(buf)

    return bufs


def add_figure_page_vector(out_pdf, img_path, fig_num, title, caption):
    """Add a figure page using the vector PDF version if available.

    Embeds the figure's vector PDF scaled into a letter page with
    title at top and caption at bottom. Falls back to rasterized PNG
    if no PDF exists.
    """
    import pikepdf

    pdf_path = Path(str(img_path).replace(".png", ".pdf"))
    if not pdf_path.exists():
        _append_mpl_pages(out_pdf,
            lambda pdf, *a: add_figure_page_raster(pdf, *a),
            img_path, fig_num, title, caption)
        return

    # Create caption overlay (letter page with just title + caption text)
    caption_buf = _make_caption_page(fig_num, title, caption)
    caption_pdf = pikepdf.Pdf.open(caption_buf)

    # Add caption page to output (this becomes the base page)
    page_idx = len(out_pdf.pages)
    out_pdf.pages.extend(caption_pdf.pages)
    out_page = out_pdf.pages[page_idx]

    # Open the figure's vector PDF (take first page)
    fig_pdf = pikepdf.Pdf.open(str(pdf_path))
    fig_page = fig_pdf.pages[0]

    # Get figure dimensions (in PDF points, 72 pt/inch)
    fig_box = fig_page.mediabox
    fig_w = float(fig_box[2]) - float(fig_box[0])
    fig_h = float(fig_box[3]) - float(fig_box[1])

    # Letter page in points
    page_w_pt = 8.5 * 72   # 612
    page_h_pt = 11 * 72    # 792

    # Available area: below title (top 4%), small bottom margin (2%)
    avail_top = page_h_pt * 0.96
    avail_bot = page_h_pt * 0.02
    avail_h = avail_top - avail_bot
    avail_w = page_w_pt * 0.94  # 3% margin each side
    margin_x = page_w_pt * 0.03

    # Scale figure to fit available area
    scale = min(avail_w / fig_w, avail_h / fig_h)
    scaled_w = fig_w * scale
    scaled_h = fig_h * scale

    # Center horizontally, top-align vertically
    x_offset = margin_x + (avail_w - scaled_w) / 2
    y_offset = avail_top - scaled_h

    # Create form XObject from figure page
    fig_form = pikepdf.Page(fig_page).as_form_xobject()
    fig_form_foreign = out_pdf.copy_foreign(fig_form)

    # Stamp the figure onto the caption page
    resources = out_page.get("/Resources", pikepdf.Dictionary())
    xobjects = resources.get("/XObject", pikepdf.Dictionary())
    xobj_name = pikepdf.Name("/FigImg")
    xobjects[xobj_name] = fig_form_foreign
    resources[pikepdf.Name("/XObject")] = xobjects
    out_page[pikepdf.Name("/Resources")] = resources

    # Build content stream: save state, translate+scale, draw, restore
    stamp_stream = (
        f"q {scale:.6f} 0 0 {scale:.6f} {x_offset:.2f} {y_offset:.2f} cm "
        f"/FigImg Do Q"
    )
    # Append to existing page content
    existing = out_page.get("/Contents")
    if existing is not None:
        if isinstance(existing, pikepdf.Array):
            streams = list(existing)
        else:
            streams = [existing]
    else:
        streams = []
    new_stream = out_pdf.make_stream(stamp_stream.encode())
    streams.append(new_stream)
    out_page[pikepdf.Name("/Contents")] = pikepdf.Array(streams)

    fig_pdf.close()
    caption_pdf.close()


def add_figure_page_raster(pdf, img_path, fig_num, title, caption):
    """Add one figure page using rasterized PNG (fallback)."""
    img = imread(str(img_path))
    h, w = img.shape[:2]
    aspect = w / h

    page_w, page_h = 8.5, 11
    title_top_frac = 0.99
    img_top_frac = title_top_frac - 0.02
    img_bot_min = 0.02

    fig_h_max = (img_top_frac - img_bot_min) * page_h
    fig_w_max = page_w - 0.1

    if aspect > fig_w_max / fig_h_max:
        display_w = fig_w_max
        display_h = display_w / aspect
    else:
        display_h = fig_h_max
        display_w = display_h * aspect

    fig = plt.figure(figsize=(page_w, page_h))
    fig.text(0.5, title_top_frac, f"Figure {fig_num}. {title}",
             ha="center", va="top", fontsize=12, fontweight="bold")

    img_left = (page_w - display_w) / 2 / page_w
    img_bottom = img_top_frac - display_h / page_h
    ax = fig.add_axes([img_left, max(img_bottom, img_bot_min),
                       display_w / page_w, display_h / page_h])
    ax.imshow(img)
    ax.axis("off")

    pdf.savefig(fig, dpi=400)
    plt.close(fig)


def _save_mpl_page(fig):
    """Save a matplotlib figure to a pikepdf-ready PDF bytes buffer."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=400, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _append_mpl_buf(out_pdf, buf):
    """Append pages from a matplotlib PDF buffer to pikepdf output."""
    import pikepdf
    src = pikepdf.Pdf.open(buf)
    out_pdf.pages.extend(src.pages)


def _append_mpl_pages(out_pdf, mpl_pdf_func, *args):
    """Run a function that writes to PdfPages, then append to pikepdf output."""
    import io, pikepdf
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        mpl_pdf_func(pdf, *args)
    buf.seek(0)
    src = pikepdf.Pdf.open(buf)
    out_pdf.pages.extend(src.pages)


def main():
    import pikepdf

    out_path = HYP / "paper_main_figures.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_pdf = pikepdf.Pdf.new()

    # Title page
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.55,
             "Spatial Proteomics Reveals\n"
             "Dual-Compartment Immune Evasion Architecture\n"
             "and Core Lymphomagenic Stromal Niches\n"
             "in Follicular Lymphoma",
             ha="center", va="center", fontsize=16, fontweight="bold",
             linespacing=1.6)
    fig.text(0.5, 0.38,
             "Main Figures",
             ha="center", va="center", fontsize=14, color="#555555")
    fig.text(0.5, 0.30,
             "Imaging Mass Cytometry · 4 TMAs · 2 Panels · ~4.2M Cells",
             ha="center", va="center", fontsize=11, color="#888888")
    _append_mpl_buf(out_pdf, _save_mpl_page(fig))

    # All figures first (ABM is now main Figure 6; dual-compartment model is Figure 7, capstone)
    missing = []
    for img_path, fig_num, title, caption in FIGURES:
        if img_path.exists():
            pdf_path = Path(str(img_path).replace(".png", ".pdf"))
            if pdf_path.exists():
                print(f"  Adding Figure {fig_num}: {title} [vector PDF]")
                add_figure_page_vector(out_pdf, img_path, fig_num, title, caption)
            else:
                print(f"  Adding Figure {fig_num}: {title} [raster PNG]")
                _append_mpl_pages(out_pdf,
                    lambda pdf, *a: add_figure_page_raster(pdf, *a),
                    img_path, fig_num, title, caption)
        else:
            missing.append((fig_num, title, str(img_path)))
            print(f"  MISSING Figure {fig_num}: {img_path}")

    # All tables placed after the figures, in order: Table 1, Table S1, Table S2
    print("  Adding Table 1: Cohort characteristics")
    _append_mpl_pages(out_pdf, make_cohort_table)
    print("  Adding Table S1: Antibody panels (2 pages)")
    _append_mpl_pages(out_pdf, make_antibody_panel_table)
    print("  Adding Table S2: ABM parameters (2 pages)")
    _append_mpl_pages(out_pdf, make_parameter_table)

    # Figure legends live in the manuscript, not the figure PDF
    out_pdf.save(str(out_path))
    out_pdf.close()
    print(f"\nSaved: {out_path}")
    if missing:
        print(f"\n{len(missing)} figures missing:")
        for num, title, path in missing:
            print(f"  Figure {num}: {title} — {path}")


if __name__ == "__main__":
    main()
