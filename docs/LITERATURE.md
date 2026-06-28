# Literature Review: Spatial Profiling of the Follicular Lymphoma TME

## Background

Follicular lymphoma (FL) is the most common indolent non-Hodgkin lymphoma, characterized by developmentally blocked germinal center B cells with a t(14;18) translocation leading to BCL2 overexpression. Despite median overall survival exceeding 10 years, 15-20% of patients experience early progression (POD24) or histologic transformation to aggressive DLBCL. The tumor microenvironment (TME) plays a central role in FL pathogenesis, but the spatial architecture of immune-tumor interactions remains incompletely characterized.

This project uses Hyperion Imaging Mass Cytometry (IMC) with ~39 markers per panel on FL tissue microarrays, providing simultaneous protein quantification at single-cell resolution while preserving spatial context.

---

## Key Literature

### Malignant B Cell Heterogeneity

**Wang/Nissen et al. (Nat Commun 2022)** — CyTOF on 155 diagnostic FL biopsies. Identified two recurrent FL subtypes: **Type A** (GCB-like, IgG+HLA-DR+CD22+) and **Type B** (memory B-like, IgM+CD79B+CD44+). Mutual exclusivity within tumors, distinct mutational landscapes. Type B: transformation HR=2.8 (p=0.043). Intratumoral phenotypic entropy independently predicted transformation (HR=5.7, p=0.011).

**Rouland/Sarrabay et al. (Blood 2024, ASH)** — CITE-seq on 107 FL samples, 10 malignant B cell archetypes. Archetype 6 (stemness markers) enriched in POD24. Two distinct TME ecosystems: (1) diagnostic/treatment-naive (Tfh, Treg, effector enriched), (2) relapsed/refractory (naive T, myeloid enriched).

**Gap**: Both used dissociated cells. Spatial distribution of B cell subtypes within tissue architecture unknown.

### TME Co-evolution During Transformation

**Sarkozy et al. (Cancer Cell 2024)** — Integrated single-cell analysis of FL transformation using paired pre/post biopsies. Progressive loss of Tcm and Tfh, replaced by exhausted and effector T cells. CD8+LAG3+ exhausted T cells predict transformation (multivariate HR=4.03). CD70-CD27 signaling enriched in tFL-DLBCL. Validated spatially: CD8+LAG3+CD27+ cells closer to CD70+ B cells (177.8 vs 305.8 um, p=0.02).

**Gap**: Spatial validation used only 6-7 marker IF. IMC with 39 markers can map full exhaustion phenotype simultaneously.

### Spatial Compartmentalization

**Ge et al. (J Hematol Oncol 2022)** — First IMC spatial analysis in FL POD24 (36-marker, 13 paired biopsies). Peri-follicular regions act as immune barriers. During POD24: increased CD163- macrophages with PD-L1/L2 upregulation, decreased CD8+ T cells with LAG-3 upregulation.

**Pelcovits et al. (Hematological Oncology 2025)** — Spatial transcriptomics identifying 4 consistent FL domains: neoplastic follicle/GC, border zone (BZ), interfollicular (IF), stromal. Border zone has highest CD8+ density, LAMP3+ DCs, CCR7-CCL19/21 signaling.

**Gap**: Small sample sizes (n=13). Our TMA design with ~170 ROIs enables population-level spatial analysis.

### Stromal Remodeling

**Radtke et al. (Cancer Cell 2024)** — Multi-omic profiling (40-plex DIVE-IBEX) of FL lymph nodes. Architectural changes in early relapsers: smaller/irregular follicles, stromal desmoplasia (desmin+vimentin+ FRCs, lumican+ ECM), DC-SIGN+ macrophage networks intrafollicularly. 15 cellular neighborhoods, 1.8M cells. Stromal communities predicted early relapse.

**Radtke & Roschewski (Blood 2024)** — Review: FL follicles are unpolarized (no LZ/DZ), mantle zone attenuated, Tregs enriched within/around follicles, DC-SIGN+ macrophages in interfollicular area/sinuses.

### T Cell Infiltration Patterns

**Roider/Gaydosik et al. (Nat Cell Biol 2024)** — Multimodal T cell reference map across nodal B-NHL (CITE-seq + TCR-seq + mIF, 101 lymph nodes). PD1+TCF7- cytotoxic T cells associated with poor survival. Distinct T cell infiltration patterns across lymphoma entities.

### Tregs and Prognosis

**Farinha et al. (Blood 2010)** — FOXP3+ T cell architectural pattern predicts survival: follicular Treg pattern = poor survival, interfollicular = better prognosis. Independent of FLIPI.

### CD14+ FDCs and Transformation

**Smeltzer et al. (Clin Cancer Res 2014)** — IHC + flow cytometry + multicolor IHC on 58 FL patients who later transformed to DLBCL. Key findings: (1) CD14+ intratumoral cells are FDCs (co-express CD21/CD23, NOT CD68/CD163), not monocytes/macrophages; (2) The *pattern* (follicular vs nonfollicular) of CD14+ cells predicted TTT (HR=3.0, P=0.004 on multivariate including FLIPI), but quantity did not; (3) Tonsil-derived bulk FDCs (not CD14+-specific) promote FL B cell viability in co-culture (42% vs 19%, P=0.003) — the paper extrapolates this to CD14+ FDCs but did not test CD14+ vs CD14− FDCs separately; (4) Serial biopsies showed CD14+ pattern transitions from nonfollicular to follicular as transformation approaches; (5) PD1+ cells in follicle = Tfh (CD3+CXCR5+, TIM3-), associated with better outcomes; PD1+ cells outside follicle = exhausted T cells (TIM3+), worse outcomes. Independent of each other (P=0.36 for correlation).

**Relevance to our data**: Directly validates CD14-high FDCs as a real, clinically relevant population. Our IMC data can test the follicular vs nonfollicular CD14+ FDC pattern as transformation predictor with true single-cell spatial resolution. Our finding that CD14-high FDCs are slightly less follicular (56% vs 61%) and that top-CD14 ROIs show disrupted architecture (33% follicular) may reflect the transition state described in serial biopsies.

### Germinal Center Sub-Compartments

**Kennedy et al. (Nat Immunol 2020)** — Identified a novel "gray zone" (GZ) within normal germinal centers beyond the classic LZ/DZ binary. Using proteomic spatial profiling, found a "DZ-proliferating" (DZp) sub-compartment spatially distinct from "DZ-differentiating" (DZd) cells. The GZ showed enrichment for metabolism-related proteins and was intermediate between LZ and DZ. This finding establishes that GC internal organization is more complex than the two-zone model, though it has not been extended to FL.

**Relevance to our data**: Our UTAG analysis reveals that FL follicles, despite lacking LZ/DZ polarity, have their own form of concentric sub-compartmentalization (GC B center → Follicle core → Mantle → Activated B zone). Kennedy et al.'s finding that normal GCs have sub-zones beyond LZ/DZ supports the broader principle that follicular architecture is more organized than previously appreciated.

### Stromal Biology

**Amé-Thomas & Tarte (Blood 2024)** — Tfh provide CD40L/IL-4 support to malignant B cells intrafollicularly. FL B cells interact with CD49a+ FRCs.

**Mourcin et al. (Immunity 2021)** — FL reprograms FRCs and FDCs via TNF/TGF-beta. FL FRCs and FDCs overexpress CXCL12, CCL19, CCL21.

---

## Knowledge Gaps Addressable by IMC

| Gap | Why IMC | Prior data |
|-----|---------|------------|
| Spatial distribution of B cell subtypes | CyTOF lost spatial info | Wang 2022 |
| Multi-cellular interaction hubs | Need >6 markers simultaneously | Sarkozy 2024 (6-marker IF) |
| Follicular/peri-follicular gradients at scale | Need large n | Ge 2022 (n=13) |
| Stromal-immune-tumor tripartite niches | Must see all 3 compartments | Radtke 2024 (whole sections, few patients) |
| TME ecosystem spatial territories | Dissociated approaches lose context | Rouland 2024 |
| T cell exhaustion spatial trajectories | Predicted but not mapped | Multiple studies |

---

## Literature Validation of Our Results

### Cell type proportions (Feb 10, 2025)

Compared v5 cross-TMA T-panel proportions against 5 publications (Wang 2022, Radtke 2024, Sarkozy 2024, Sarrabay 2024, BIDIFLY). All major cell types consistent:

- **B cells (47-61%)**: Matches FL as B-cell malignancy. Wang CyTOF: ~50% malignant B cells.
- **T cells (10-14% of total)**: Consistent after platform adjustment. As % of non-B non-LQ cells: 48-76%, plausible given IMC captures stromal populations missed by dissociation.
- **CD4 > CD8 ratio**: Consistent with FL Tfh/Treg enrichment (Wang, Radtke). B1 inverted (CD4 2.5% vs CD8 5.4%) — inter-patient heterogeneity.
- **CD8 T exhausted (0.7-2.3%)**: Matches literature. As fraction of total CD8: 11-27%.
- **Macrophages (0.6-2.1% in T-panel)**: Expected to be low — T-panel has only CD68 for myeloid.
- **Biomax reactive LN (48% CD4 T, 11% CD8, 15% B)**: Matches normal LN architecture.

### Tissue compartment validation (Feb 8, 2025)

Our 15 UTAG compartments validated against Pelcovits 2025, Gaydosik 2024, Radtke 2024, Farinha 2010, Mourcin 2021. All major compartments match known FL spatial architecture:

- GC B center, follicle core, follicle mantle → neoplastic follicle zones
- Follicle-T zone interface → border zone (Pelcovits BZ)
- T cell zone, Treg-enriched T zone → paracortex
- Macrophage/M2 zones → interfollicular macrophage niches
- FDC/FRC/Stromal zones → stromal/reticular compartments

**Novel contributions**: (1) entropy gradient quantification across compartments, (2) Treg-enriched T zone as distinct compartment (29% Treg), (3) per-ROI per-compartment statistical framework, (4) UTAG cell-type feature method for B cell-dominant tissues.

---

## References

1. Sarkozy C et al. Integrated single cell analysis reveals co-evolution of malignant B cells and tumor micro-environment in transformed follicular lymphoma. *Cancer Cell* 42, 1003-1017 (2024).
2. Wang X, Nissen M et al. Single-cell profiling reveals a memory B cell-like subtype of follicular lymphoma with increased transformation risk. *Nat Commun* 13, 6772 (2022).
3. Radtke AJ & Roschewski M. The follicular lymphoma tumor microenvironment at single-cell and spatial resolution. *Blood* 143, 1069-1079 (2024).
4. Rouland S, Sarrabay A et al. A Single Cell Atlas of Follicular Lymphoma across Clinical Stages. *Blood* 144, 740-741 (2024).
5. Ge Z et al. Revealing the evolution of the tumor immune microenvironment in FL patients progressing within 24 months using single-cell IMC. *J Hematol Oncol* 15, 112 (2022).
6. Radtke AJ et al. Multi-omic profiling of follicular lymphoma reveals changes in tissue architecture and enhanced stromal remodeling in high-risk patients. *Cancer Cell* 42, 444-463 (2024).
7. Roider T et al. Multimodal and spatially resolved profiling identifies distinct patterns of T cell infiltration in nodal B cell lymphoma entities. *Nat Cell Biol* 26, 478-492 (2024).
8. Pelcovits A et al. Spatial Transcriptomics of FL and MZL Identifies Domains of CD8+ T Cell Recruitment and Exclusion. *Hematological Oncology* (2025).
9. Farinha P et al. Architectural pattern of FOXP3+ T cells in FL is independent predictor of survival and transformation. *Blood* 116, 5764-5772 (2010).
10. Mourcin F et al. FL triggers phenotypic and functional remodeling of the lymphoid stromal cell landscape. *Immunity* 54, 1788-1801 (2021).
11. Amé-Thomas P & Tarte K. Cell cross-talk within the lymphoma TME: FL as a paradigm. *Blood* 143, 1060-1068 (2024).
12. Smeltzer JP et al. Pattern of CD14+ follicular dendritic cells and PD1+ T cells independently predicts time to transformation in follicular lymphoma. *Clin Cancer Res* 20, 2862-2872 (2014).
13. Kennedy DE et al. Novel specialized cell state and spatial compartments within the germinal center. *Nat Immunol* 21, 660-670 (2020).
