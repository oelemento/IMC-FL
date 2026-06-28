#!/usr/bin/env python
"""Merge UTAG cell-type domains via hierarchical clustering.

Takes the UTAG cell-type feature results (res=0.5) and merges domains
by cell-type composition similarity using cosine distance + Ward linkage.
Targets ~15 tissue compartments per panel.
"""
import time
import anndata as ad
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTDIR = "<PROJECT_ROOT>/output"
SOURCE_RES = 0.5  # starting UTAG resolution

ct_colors_T = {
    "B cells": "#FFB347", "B cells (CXCR5hi)": "#4A90D9", "B cells (CD20hi)": "#87CEEB",
    "B (weak CD20)": "#DEB887", "GC B cells": "#FF8C00", "Activated B": "#FF6347",
    "B (TOXhi)": "#CD853F", "CD4 T cells": "#9370DB", "CD8 T cells": "#90EE90",
    "CD8 T exhausted": "#FFB6C1", "CD8 T pre-exhausted (TOX+)": "#FF69B4",
    "Treg": "#DC143C", "Macrophages": "#228B22", "LQ": "#D3D3D3",
    "Low quality / Unassigned": "#D3D3D3",
    "Mixed": "#A9A9A9", "Other": "#C0C0C0", "Cytotoxic": "#006400",
}

ct_colors_S = {
    "B cells": "#FFB347", "B cells (BCL2+)": "#E8A020", "B cells (PAX5+)": "#F4C430",
    "FDC": "#8B4513", "FDC (PDPN+)": "#A0522D", "FRC (PDPN+)": "#D2691E",
    "Macrophages": "#228B22", "M1 Macrophages": "#006400", "M2 Macrophages": "#32CD32",
    "Histiocytes (CD44hi)": "#556B2F",
    "CD4 T cells": "#9370DB", "CD8 T cells": "#90EE90",
    "Treg": "#DC143C", "Plasma cells": "#FF1493",
    "Endothelial": "#4169E1", "Stromal / CAF": "#DDA0DD",
    "Dendritic cells": "#FF8C00", "pDC": "#FFD700",
    "Myeloid (S100A9+)": "#2E8B57", "Mixed / Border cells": "#A9A9A9",
    "Low quality / Unassigned": "#D3D3D3",
    "Other": "#C0C0C0",
}


def merge_domains(adata, panel_name, ct_colors, n_compartments=15):
    """Merge UTAG domains into tissue compartments via hierarchical clustering."""

    col = f"utag_ct_{SOURCE_RES}"
    prefix = f"all_TMA_{panel_name}_utag_ct"

    print(f"\n{'='*60}")
    print(f"  {panel_name}-panel: Merging UTAG domains into {n_compartments} compartments")
    print(f"{'='*60}")

    n_domains = adata.obs[col].nunique()
    print(f"Starting domains (res={SOURCE_RES}): {n_domains}")

    # ---- Filter out tiny domains (< 50 cells) before clustering ----
    domain_sizes = adata.obs[col].value_counts()
    small_domains = domain_sizes[domain_sizes < 50].index
    large_domains = domain_sizes[domain_sizes >= 50].index
    print(f"Domains >= 50 cells: {len(large_domains)}")
    print(f"Domains < 50 cells: {len(small_domains)} ({domain_sizes[small_domains].sum()} cells)")

    # ---- Composition matrix (large domains only) ----
    mask_large = adata.obs[col].isin(large_domains)
    comp = pd.crosstab(
        adata.obs.loc[mask_large, col],
        adata.obs.loc[mask_large, "cell_type"],
        normalize="index",
    )
    print(f"Composition matrix: {comp.shape}")

    # ---- Hierarchical clustering ----
    dist = pdist(comp.values, metric="cosine")
    Z = linkage(dist, method="ward")
    compartment_labels = fcluster(Z, t=n_compartments, criterion="maxclust")

    domain_to_compartment = dict(zip(comp.index, compartment_labels))

    # Assign small domains to nearest compartment by composition similarity
    if len(small_domains) > 0:
        comp_small = pd.crosstab(
            adata.obs.loc[~mask_large, col],
            adata.obs.loc[~mask_large, "cell_type"],
            normalize="index",
        )
        # Align columns
        comp_small = comp_small.reindex(columns=comp.columns, fill_value=0)

        # Compute compartment centroids
        compartment_centroids = {}
        for c in range(1, n_compartments + 1):
            domains_in_c = [d for d, lbl in domain_to_compartment.items() if lbl == c]
            if domains_in_c:
                compartment_centroids[c] = comp.loc[domains_in_c].mean(axis=0).values

        # Assign each small domain to nearest compartment
        for sd in comp_small.index:
            best_c, best_dist = None, float("inf")
            for c, centroid in compartment_centroids.items():
                from scipy.spatial.distance import cosine
                d = cosine(comp_small.loc[sd].values, centroid)
                if d < best_dist:
                    best_dist = d
                    best_c = c
            domain_to_compartment[sd] = best_c

    # Map to cells
    adata.obs["tissue_compartment"] = adata.obs[col].map(domain_to_compartment).astype(str)
    n_actual = adata.obs["tissue_compartment"].nunique()
    print(f"Tissue compartments: {n_actual}")

    # ---- Compartment composition ----
    comp_final = pd.crosstab(
        adata.obs["tissue_compartment"],
        adata.obs["cell_type"],
        normalize="index",
    )
    # Sort by dominant cell type for readability
    comp_final = comp_final.loc[
        comp_final.idxmax(axis=1).sort_values().index
    ]

    # ---- Name compartments by dominant composition ----
    compartment_names = {}
    for c in comp_final.index:
        row = comp_final.loc[c]
        top1 = row.idxmax()
        top1_frac = row.max()
        top2 = row.drop(top1).idxmax()
        top2_frac = row.drop(top1).max()

        if top1_frac > 0.7:
            name = f"C{c}: {top1} zone"
        elif top1_frac > 0.4:
            name = f"C{c}: {top1}/{top2}"
        else:
            name = f"C{c}: Mixed"
        compartment_names[c] = name
    adata.obs["compartment_name"] = adata.obs["tissue_compartment"].map(compartment_names)

    print(f"\n=== Compartment Summary ===")
    sizes = adata.obs["tissue_compartment"].value_counts().sort_index()
    for c in comp_final.index:
        n = sizes.get(c, 0)
        print(f"  {compartment_names[c]}: {n:,} cells ({n/len(adata)*100:.1f}%)")

    # ---- Plot 1: Dendrogram + composition heatmap ----
    fig, (ax_dend, ax_heat) = plt.subplots(
        1, 2, figsize=(18, max(8, len(comp) * 0.12)),
        gridspec_kw={"width_ratios": [1, 3]}
    )

    # Color dendrogram by compartment
    dend = dendrogram(Z, orientation="left", ax=ax_dend,
                      labels=[str(d) for d in comp.index],
                      leaf_font_size=4, color_threshold=0, above_threshold_color="gray")
    ax_dend.set_title("Domain Dendrogram", fontsize=10)

    # Reorder heatmap
    order = dend["leaves"]
    comp_reordered = comp.iloc[order]

    sns.heatmap(comp_reordered, cmap="YlOrRd", ax=ax_heat,
                linewidths=0.1, cbar_kws={"label": "Fraction"})
    ax_heat.set_title(f"{panel_name}-panel: Domain Composition (res={SOURCE_RES})\n"
                      f"Ordered by hierarchical clustering → {n_compartments} compartments",
                      fontsize=10)
    ax_heat.set_ylabel("")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{prefix}_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {prefix}_dendrogram.png")

    # ---- Plot 2: Merged compartment composition ----
    fig, ax = plt.subplots(figsize=(14, max(6, n_actual * 0.5)))
    sns.heatmap(comp_final, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Fraction"}, annot_kws={"size": 8})
    # Use compartment names as y-tick labels
    ax.set_yticklabels([compartment_names.get(c, c) for c in comp_final.index],
                       fontsize=9)
    ax.set_title(f"{panel_name}-panel v8: Tissue Compartments (n={n_actual})\n"
                 f"UTAG cell-type features, max_dist=50, merged from res={SOURCE_RES}",
                 fontweight="bold")
    ax.set_ylabel("Tissue Compartment")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{prefix}_merged_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {prefix}_merged_composition.png")

    # ---- Plot 3: Per-TMA compartment distribution ----
    tma_comp = pd.crosstab(adata.obs["tma"], adata.obs["tissue_compartment"], normalize="index")
    fig, ax = plt.subplots(figsize=(max(10, n_actual * 0.6), 5))
    sns.heatmap(tma_comp, cmap="YlOrRd", ax=ax, linewidths=0.5,
                annot=True, fmt=".2f", cbar_kws={"label": "Fraction"})
    ax.set_title(f"{panel_name}-panel v8: Compartment Distribution per TMA", fontweight="bold")
    ax.set_ylabel("TMA")
    ax.set_xlabel("Tissue Compartment")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{prefix}_tma_compartments.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {prefix}_tma_compartments.png")

    # ---- Plot 4: FL11 spatial — cell types + compartments ----
    roi = adata[adata.obs["sample_id"] == "B1_FL11"].copy()
    if len(roi) > 0:
        x = roi.obsm["spatial"][:, 0] if "spatial" in roi.obsm else roi.obs["centroid_x"].values
        y = roi.obsm["spatial"][:, 1] if "spatial" in roi.obsm else roi.obs["centroid_y"].values

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        # Panel a: Cell types
        ax = axes[0]
        for ct in roi.obs["cell_type"].unique():
            m = roi.obs["cell_type"] == ct
            ax.scatter(x[m], y[m], c=ct_colors.get(ct, "#808080"),
                       s=2, alpha=0.7, label=ct, rasterized=True)
        ax.set_title(f"B1_FL11 — Cell Types (v8)", fontweight="bold", fontsize=11)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        ct_counts = roi.obs["cell_type"].value_counts()
        order = [labels.index(ct) for ct in ct_counts.index if ct in labels]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="upper left", fontsize=5, markerscale=2, framealpha=0.8)

        # Panel b: UTAG domains (unmerged)
        ax = axes[1]
        domains = roi.obs[col].astype(str)
        unique_d = sorted(domains.unique(), key=lambda d: int(d))
        n_d = len(unique_d)
        cmap_d = plt.colormaps.get_cmap("tab20" if n_d <= 20 else "gist_ncar")
        colors_d = {d: cmap_d(j / max(n_d - 1, 1)) for j, d in enumerate(unique_d)}
        for d in unique_d:
            m = domains == d
            ax.scatter(x[m.values], y[m.values], c=[colors_d[d]], s=2, alpha=0.7, rasterized=True)
        ax.set_title(f"UTAG Domains (res={SOURCE_RES}, {n_d} domains)",
                     fontweight="bold", fontsize=11)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])

        # Panel c: Merged compartments
        ax = axes[2]
        comps = roi.obs["tissue_compartment"].astype(str)
        unique_c = sorted(comps.unique(), key=lambda x: int(x))
        n_c = len(unique_c)
        cmap_c = plt.colormaps.get_cmap("tab20")
        colors_c = {c: cmap_c(j % 20) for j, c in enumerate(unique_c)}
        for c in unique_c:
            m = comps == c
            short_name = compartment_names.get(c, f"C{c}")
            ax.scatter(x[m.values], y[m.values], c=[colors_c[c]], s=2, alpha=0.7,
                       label=short_name, rasterized=True)
        ax.set_title(f"Tissue Compartments ({n_c} in FL11)",
                     fontweight="bold", fontsize=11)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper left", fontsize=5, markerscale=2, framealpha=0.8)

        plt.suptitle(f"UTAG Cell-Type Features → Merged Compartments — B1_FL11 ({panel_name}-panel v8)",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(f"{OUTDIR}/{prefix}_FL11_merged.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {prefix}_FL11_merged.png")

    return adata


# ====== T-panel ======
print("=== Loading T-panel ===")
t0 = time.time()
adata_T = ad.read_h5ad(f"{OUTDIR}/all_TMA_T_utag_ct.h5ad")
print(f"T-panel: {adata_T.shape}")

adata_T = merge_domains(adata_T, "T", ct_colors_T, n_compartments=15)

adata_T.write_h5ad(f"{OUTDIR}/all_TMA_T_utag_ct_merged.h5ad")
print(f"Saved all_TMA_T_utag_ct_merged.h5ad")

del adata_T

# ====== S-panel ======
print("\n=== Loading S-panel ===")
adata_S = ad.read_h5ad(f"{OUTDIR}/all_TMA_S_utag_ct.h5ad")
print(f"S-panel: {adata_S.shape}")

adata_S = merge_domains(adata_S, "S", ct_colors_S, n_compartments=15)

adata_S.write_h5ad(f"{OUTDIR}/all_TMA_S_utag_ct_merged.h5ad")
print(f"Saved all_TMA_S_utag_ct_merged.h5ad")

total = time.time() - t0
print(f"\n=== ALL DONE (total: {total:.0f}s / {total/60:.1f} min) ===")
