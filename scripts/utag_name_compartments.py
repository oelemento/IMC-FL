#!/usr/bin/env python
"""Assign biologically meaningful names to merged UTAG compartments.

Reads the merged h5ad files, computes composition, assigns names based
on dominant cell types and biological context, re-plots everything.
"""
import time
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTDIR = "<PROJECT_ROOT>/output"

ct_colors_T = {
    "B cells": "#FFB347", "B cells (CXCR5hi)": "#4A90D9", "B cells (CD20hi)": "#87CEEB",
    "B (weak CD20)": "#DEB887", "GC B cells": "#FF8C00", "Activated B": "#FF6347",
    "B (TOXhi)": "#CD853F", "CD4 T cells": "#9370DB", "CD8 T cells": "#90EE90",
    "CD8 T exhausted": "#FFB6C1", "CD8 T pre-exhausted (TOX+)": "#FF69B4",
    "Treg": "#DC143C", "Macrophages": "#228B22",
    "Low quality / Unassigned": "#D3D3D3",
    "Mixed / Border cells": "#A9A9A9", "Other": "#C0C0C0",
    "Macrophages (GzmB+)": "#006400", "T cells": "#B0C4DE",
}

ct_colors_S = {
    "B cells": "#FFB347", "B cells (BCL2+)": "#E8A020", "B cells (PAX5+)": "#F4C430",
    "FDC": "#8B4513", "FRC (PDPN+)": "#D2691E",
    "Macrophages": "#228B22", "M1 Macrophages": "#006400", "M2 Macrophages": "#32CD32",
    "Histiocytes (CD44hi)": "#556B2F",
    "CD4 T cells": "#9370DB", "CD8 T cells": "#90EE90",
    "Dendritic cells": "#FF8C00", "pDC": "#FFD700",
    "Endothelial": "#4169E1", "Stromal / CAF": "#DDA0DD",
    "Myeloid (S100A9+)": "#2E8B57", "Mixed / Border cells": "#A9A9A9",
    "Low quality / Unassigned": "#D3D3D3", "Other": "#C0C0C0",
}


def name_T_compartment(row):
    """Assign biological name to T-panel compartment based on composition."""
    # Get top cell types
    top = row.sort_values(ascending=False)

    # Sum B lineage fractions
    b_markers = ["B cells", "B cells (CXCR5hi)", "B cells (CD20hi)",
                 "GC B cells", "Activated B / Plasmablast", "B cells (TOXhi)",
                 "B cells (weak CD20)"]
    b_total = sum(row.get(m, 0) for m in b_markers)

    # Sum T lineage fractions
    t_markers = ["CD4 T cells", "CD8 T cells", "CD8 T exhausted",
                 "CD8 T pre-exhausted (TOX+)", "Treg", "T cells"]
    t_total = sum(row.get(m, 0) for m in t_markers)

    gc_b = row.get("GC B cells", 0)
    cxcr5 = row.get("B cells (CXCR5hi)", 0)
    cd20hi = row.get("B cells (CD20hi)", 0)
    b_gen = row.get("B cells", 0)
    act_b = row.get("Activated B / Plasmablast", 0)
    weak_cd20 = row.get("B cells (weak CD20)", 0)
    lq = row.get("Low quality / Unassigned", 0)
    mac = row.get("Macrophages", 0)
    cd4 = row.get("CD4 T cells", 0)
    cd8 = row.get("CD8 T cells", 0)
    cd8_exh = row.get("CD8 T exhausted", 0)
    cd8_preexh = row.get("CD8 T pre-exhausted (TOX+)", 0)
    treg = row.get("Treg", 0)
    cytotoxic = row.get("Macrophages (GzmB+)", 0)

    # GC B cell center
    if gc_b >= 0.50:
        return "GC core"
    # Follicle core with mixed GC/CD20hi/CXCR5hi
    if gc_b >= 0.15 and (cd20hi + cxcr5) >= 0.40:
        return "Follicle core (GC/CD20hi/CXCR5hi)"
    # CXCR5hi-dominant zone (mantle zone)
    if cxcr5 >= 0.40:
        return "Follicle mantle (CXCR5hi)"
    # Activated B / CXCR5hi
    if act_b >= 0.20 and cxcr5 >= 0.20:
        return "Activated B / CXCR5hi zone"
    # B cell follicle (mixed B subtypes)
    if b_total >= 0.65 and lq < 0.10:
        if cd20hi >= 0.10:
            return "B cell follicle (CD20hi/CXCR5hi)"
        return "B cell zone"
    # Weak CD20 / LQ
    if weak_cd20 >= 0.40:
        return "Weak CD20 / LQ border"
    # LQ-dominated
    if lq >= 0.60:
        return "Unidentified zone"
    # LQ with some B
    if lq >= 0.30 and b_total + weak_cd20 >= 0.20:
        return "LQ / B transitional"
    # Macrophage-rich
    if mac >= 0.35:
        return "Macrophage-rich zone"
    # T cell zone
    if t_total >= 0.50 and treg < 0.20:
        return "T cell zone (CD4/CD8)"
    # Treg-enriched T zone
    if treg >= 0.20 and t_total >= 0.50:
        return "Treg-enriched T zone"
    # Cytotoxic niche
    if cytotoxic >= 0.15:
        return "Cytotoxic / LQ niche"
    # Immune interface (mixed T/Mac/B at follicle edge)
    if mac >= 0.10 and t_total >= 0.20 and b_total >= 0.15:
        return "Follicle-T zone interface"
    # Mixed B/T/Mac
    if b_total >= 0.20 and t_total >= 0.15 and mac >= 0.10:
        return "B/T/Mac mixed zone"
    # LQ/weak CD20 with cytotoxic
    if lq >= 0.25 and cytotoxic >= 0.10:
        return "Cytotoxic / LQ niche"

    # Fallback
    return f"Mixed ({top.index[0][:12]} {top.iloc[0]:.0%})"


def name_S_compartment(row):
    """Assign biological name to S-panel compartment based on composition."""
    top = row.sort_values(ascending=False)

    b_markers = ["B cells", "B cells (BCL2+)", "B cells (PAX5+)"]
    b_total = sum(row.get(m, 0) for m in b_markers)

    t_markers = ["CD4 T cells", "CD8 T cells"]
    t_total = sum(row.get(m, 0) for m in t_markers)

    b_gen = row.get("B cells", 0)
    bcl2 = row.get("B cells (BCL2+)", 0)
    pax5 = row.get("B cells (PAX5+)", 0)
    fdc = row.get("FDC", 0)
    frc = row.get("FRC (PDPN+)", 0)
    mac = row.get("Macrophages", 0)
    m1 = row.get("M1 Macrophages", 0)
    m2 = row.get("M2 Macrophages", 0)
    histo = row.get("Histiocytes (CD44hi)", 0)
    cd4 = row.get("CD4 T cells", 0)
    cd8 = row.get("CD8 T cells", 0)
    lq = row.get("Low quality / Unassigned", 0)
    strom = row.get("Stromal / CAF", 0)
    endo = row.get("Endothelial", 0)
    myeloid = row.get("Myeloid (S100A9+)", 0)
    pdc = row.get("pDC", 0)
    dc = row.get("Dendritic cells", 0)
    other = row.get("Other", 0)

    # FDC-rich
    if fdc >= 0.35:
        return "FDC network zone"
    # FDC + myeloid
    if fdc >= 0.10 and (m1 + mac) >= 0.10:
        return "FDC / myeloid zone"
    # Histiocyte zone
    if histo >= 0.50:
        return "Histiocyte zone"
    # B cell zone (BCL2+)
    if bcl2 >= 0.40:
        return "B cell zone (BCL2+)"
    # B cell zone (PAX5+)
    if pax5 >= 0.40:
        return "B cell zone (PAX5+)"
    # Mixed B cell
    if b_total >= 0.60:
        if bcl2 >= 0.15 and pax5 >= 0.15:
            return "B cell follicle (BCL2+/PAX5+)"
        return "B cell zone"
    # T cell zone
    if t_total >= 0.30 and b_total < 0.20:
        return "T cell zone"
    # Stromal
    if strom >= 0.30:
        return "Stromal / CAF zone"
    # pDC-enriched
    if pdc >= 0.30:
        return "pDC-enriched zone"
    # LQ
    if lq >= 0.50:
        return "Unidentified zone"
    # Myeloid-enriched
    if myeloid >= 0.25:
        return "Myeloid (S100A9+) zone"
    # Mixed B/T
    if b_total >= 0.20 and t_total >= 0.15:
        return "B/T mixed zone"
    # Mixed with other/myeloid
    if other >= 0.15 or myeloid >= 0.15:
        return "Other / myeloid zone"
    # Mixed with M1
    if m1 >= 0.15:
        return "M1 macrophage zone"

    return f"Mixed ({top.index[0][:12]} {top.iloc[0]:.0%})"


def process_panel(input_path, panel_name, ct_colors, name_fn):
    """Load, name, plot, save."""
    prefix = f"all_TMA_{panel_name}_utag_ct"

    print(f"\n{'='*60}")
    print(f"  {panel_name}-panel: Naming compartments")
    print(f"{'='*60}")

    adata = ad.read_h5ad(input_path)
    print(f"Loaded {adata.shape}")

    # Composition per compartment
    comp = pd.crosstab(
        adata.obs["tissue_compartment"],
        adata.obs["cell_type"],
        normalize="index",
    )

    # Assign names
    names = {}
    for c in comp.index:
        names[c] = name_fn(comp.loc[c])
    adata.obs["compartment_name"] = adata.obs["tissue_compartment"].map(names)

    print(f"\n=== Named Compartments ===")
    sizes = adata.obs["tissue_compartment"].value_counts()
    for c in sorted(names.keys(), key=lambda x: int(x)):
        n = sizes.get(c, 0)
        print(f"  C{c}: {names[c]:40s} {n:>8,} cells ({n/len(adata)*100:.1f}%)")

    # Re-sort composition by biological category
    # Order: B cell zones first, then mixed, then T zones, then myeloid, then LQ
    cat_order = []
    for c in sorted(names.keys(), key=lambda x: int(x)):
        nm = names[c]
        if "GC B" in nm or "Follicle core" in nm:
            cat_order.append((0, c))
        elif "Follicle mantle" in nm or "CXCR5" in nm:
            cat_order.append((1, c))
        elif "Activated B" in nm:
            cat_order.append((2, c))
        elif "B cell" in nm and "LQ" not in nm:
            cat_order.append((3, c))
        elif "FDC" in nm:
            cat_order.append((4, c))
        elif "Follicle" in nm and "interface" in nm:
            cat_order.append((5, c))
        elif "B/T" in nm:
            cat_order.append((6, c))
        elif "T cell" in nm or "Treg" in nm:
            cat_order.append((7, c))
        elif "Macrophage" in nm or "M1" in nm or "Myeloid" in nm:
            cat_order.append((8, c))
        elif "Cytotoxic" in nm:
            cat_order.append((9, c))
        elif "Histiocyte" in nm:
            cat_order.append((10, c))
        elif "Stromal" in nm or "pDC" in nm:
            cat_order.append((11, c))
        elif "Weak CD20" in nm or "LQ" in nm or "transitional" in nm:
            cat_order.append((12, c))
        elif "Unidentified" in nm:
            cat_order.append((13, c))
        else:
            cat_order.append((14, c))

    sorted_comps = [c for _, c in sorted(cat_order)]
    comp_sorted = comp.loc[sorted_comps]

    # ---- Plot 1: Named composition heatmap ----
    n_comp = len(comp_sorted)
    fig, ax = plt.subplots(figsize=(16, max(7, n_comp * 0.55)))
    sns.heatmap(comp_sorted, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Fraction"}, annot_kws={"size": 8})

    ylabels = [f"{names[c]} (n={sizes.get(c,0):,})" for c in sorted_comps]
    ax.set_yticklabels(ylabels, fontsize=9, rotation=0)
    ax.set_title(f"{panel_name}-panel v8: Tissue Compartments (n={n_comp})\n"
                 f"UTAG cell-type features, max_dist=50, merged from res=0.5",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{prefix}_named_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {prefix}_named_composition.png")

    # ---- Plot 2: FL11 spatial with named compartments ----
    roi = adata[adata.obs["sample_id"] == "B1_FL11"].copy()
    if len(roi) > 0:
        x = roi.obsm["spatial"][:, 0] if "spatial" in roi.obsm else roi.obs["centroid_x"].values
        y = roi.obsm["spatial"][:, 1] if "spatial" in roi.obsm else roi.obs["centroid_y"].values

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # Panel a: Cell types
        ax = axes[0]
        for ct in roi.obs["cell_type"].unique():
            m = roi.obs["cell_type"] == ct
            ax.scatter(x[m], y[m], c=ct_colors.get(ct, "#808080"),
                       s=2, alpha=0.7, label=ct, rasterized=True)
        ax.set_title("(a) Cell Types (v8)", fontweight="bold", fontsize=12)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        ct_counts = roi.obs["cell_type"].value_counts()
        order = [labels.index(ct) for ct in ct_counts.index if ct in labels]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="upper left", fontsize=6, markerscale=2, framealpha=0.8)

        # Panel b: UTAG domains (unmerged)
        ax = axes[1]
        col = "utag_ct_0.5"
        domains = roi.obs[col].astype(str)
        unique_d = sorted(domains.unique(), key=lambda d: int(d))
        n_d = len(unique_d)
        cmap_d = plt.colormaps.get_cmap("tab20" if n_d <= 20 else "gist_ncar")
        colors_d = {d: cmap_d(j / max(n_d - 1, 1)) for j, d in enumerate(unique_d)}
        for d in unique_d:
            m = domains == d
            ax.scatter(x[m.values], y[m.values], c=[colors_d[d]], s=2, alpha=0.7, rasterized=True)
        ax.set_title(f"(b) UTAG Domains (res=0.5, {n_d} in FL11)",
                     fontweight="bold", fontsize=12)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])

        # Panel c: Named compartments
        ax = axes[2]
        comp_names_in_roi = roi.obs["compartment_name"].unique()
        # Use a good qualitative colormap
        n_c = len(comp_names_in_roi)
        cmap_c = plt.colormaps.get_cmap("tab20")
        # Sort by frequency for consistent coloring
        comp_counts = roi.obs["compartment_name"].value_counts()
        colors_c = {nm: cmap_c(j % 20) for j, nm in enumerate(comp_counts.index)}

        for nm in comp_counts.index:
            m = roi.obs["compartment_name"] == nm
            ax.scatter(x[m.values], y[m.values], c=[colors_c[nm]], s=2, alpha=0.7,
                       label=nm, rasterized=True)
        ax.set_title(f"(c) Tissue Compartments ({n_c} in FL11)",
                     fontweight="bold", fontsize=12)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper left", fontsize=5.5, markerscale=2, framealpha=0.8)

        plt.suptitle(f"UTAG Cell-Type Features → Named Compartments — B1_FL11 ({panel_name}-panel v8)",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(f"{OUTDIR}/{prefix}_FL11_named.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {prefix}_FL11_named.png")

    # Save
    adata.write_h5ad(input_path)  # overwrite merged file
    print(f"Updated {input_path}")

    return adata


# ====== Process both panels ======
t0 = time.time()

adata_T = process_panel(
    f"{OUTDIR}/all_TMA_T_utag_ct_merged.h5ad",
    "T", ct_colors_T, name_T_compartment
)
del adata_T

adata_S = process_panel(
    f"{OUTDIR}/all_TMA_S_utag_ct_merged.h5ad",
    "S", ct_colors_S, name_S_compartment
)
del adata_S

print(f"\n=== ALL DONE ({time.time()-t0:.0f}s) ===")
