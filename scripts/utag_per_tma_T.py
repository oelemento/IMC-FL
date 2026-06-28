#!/usr/bin/env python
"""UTAG per-TMA + meta-clustering for T-panel v8."""
import time
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTDIR = "<PROJECT_ROOT>/output"
INPUT = f"{OUTDIR}/all_TMA_T_global_v8.h5ad"
PREFIX = "all_TMA_T_utag_v2"
UTAG_RES = [0.3, 0.5, 1.0]
PRIMARY_RES = 0.5
N_COMPARTMENTS = 15  # target number of tissue compartments

print("=== Loading T-panel v8 ===")
t0 = time.time()
adata = ad.read_h5ad(INPUT)
print(f"Loaded {adata.shape} in {time.time()-t0:.0f}s")

from utag import utag as run_utag

# ---- Level 1: Per-TMA UTAG ----
print("\n=== Level 1: Per-TMA UTAG ===")
tmas = sorted(adata.obs["tma"].unique())
utag_col = f"UTAG Label_leiden_{PRIMARY_RES}"
all_domain_labels = pd.Series(index=adata.obs_names, dtype=str)

for tma in tmas:
    t1 = time.time()
    mask = adata.obs["tma"] == tma
    sub = adata[mask].copy()
    n_rois = sub.obs["sample_id"].nunique()
    print(f"\n--- {tma}: {sub.shape[0]} cells, {n_rois} ROIs ---")

    result = run_utag(
        sub,
        slide_key="sample_id",
        max_dist=20,
        normalization_mode="l2_norm",
        apply_clustering=True,
        clustering_method="leiden",
        resolutions=UTAG_RES,
        leiden_kwargs=dict(flavor="igraph", n_iterations=2, directed=False),
    )

    # Prefix domain labels with TMA name to make unique
    domain_labels = tma + "_" + result.obs[utag_col].astype(str)
    all_domain_labels[result.obs_names] = domain_labels

    n_domains = result.obs[utag_col].nunique()
    elapsed = time.time() - t1
    print(f"  {n_domains} domains in {elapsed:.0f}s")

    # Save per-TMA UTAG columns back (all resolutions)
    for res in UTAG_RES:
        col = f"UTAG Label_leiden_{res}"
        if col in result.obs.columns:
            adata.obs.loc[mask, f"utag_{tma}_{res}"] = result.obs[col].values

# Store the primary per-TMA domain labels
adata.obs["utag_domain"] = all_domain_labels
print(f"\nTotal unique domains: {adata.obs['utag_domain'].nunique()}")

# ---- Level 2: Meta-clustering ----
print("\n=== Level 2: Meta-clustering by composition ===")

# Compute cell-type composition per domain
ct_comp = pd.crosstab(
    adata.obs["utag_domain"],
    adata.obs["cell_type"],
    normalize="index",
)
print(f"Domain composition matrix: {ct_comp.shape}")

# Also compute domain sizes
domain_sizes = adata.obs["utag_domain"].value_counts()

# Hierarchical clustering of domains by composition
Z = linkage(pdist(ct_comp.values, metric="cosine"), method="ward")
compartment_labels = fcluster(Z, t=N_COMPARTMENTS, criterion="maxclust")

# Map domain -> compartment
domain_to_compartment = dict(zip(ct_comp.index, compartment_labels))
adata.obs["tissue_compartment"] = adata.obs["utag_domain"].map(domain_to_compartment).astype(str)

n_comp = adata.obs["tissue_compartment"].nunique()
print(f"Tissue compartments: {n_comp}")

# Summarize compartment composition
comp_composition = pd.crosstab(
    adata.obs["tissue_compartment"],
    adata.obs["cell_type"],
    normalize="index",
)
print(f"\n=== Tissue compartment composition ===")
print(comp_composition.round(3).to_string())

# Compartment sizes
comp_sizes = adata.obs["tissue_compartment"].value_counts().sort_index()
print(f"\n=== Compartment sizes ===")
for c, n in comp_sizes.items():
    print(f"  Compartment {c}: {n:,} cells ({n/len(adata)*100:.1f}%)")

# Per-TMA compartment distribution
tma_comp = pd.crosstab(
    adata.obs["tma"],
    adata.obs["tissue_compartment"],
    normalize="index",
)
print(f"\n=== Compartment distribution per TMA ===")
print(tma_comp.round(3).to_string())

# ---- Save ----
outpath = f"{OUTDIR}/{PREFIX}.h5ad"
adata.write_h5ad(outpath)
print(f"\nSaved to {outpath}")

# ---- Plots ----
print("\n=== Generating plots ===")

# 1. Compartment composition heatmap
fig, ax = plt.subplots(figsize=(14, max(6, n_comp * 0.5)))
sns.heatmap(comp_composition, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
            linewidths=0.5, cbar_kws={"label": "Fraction"}, annot_kws={"size": 8})
ax.set_title(f"T-panel v8: Cell Type Composition per Tissue Compartment\n(n={N_COMPARTMENTS}, from per-TMA UTAG res={PRIMARY_RES})",
             fontweight="bold")
ax.set_ylabel("Tissue Compartment")
ax.set_xlabel("")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/{PREFIX}_compartment_composition.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {PREFIX}_compartment_composition.png")

# 2. Dendrogram + composition heatmap of individual domains
fig, (ax_dend, ax_heat) = plt.subplots(1, 2, figsize=(20, max(12, len(ct_comp) * 0.15)),
                                        gridspec_kw={"width_ratios": [1, 3]})
from scipy.cluster.hierarchy import dendrogram
dend = dendrogram(Z, orientation="left", ax=ax_dend, labels=ct_comp.index.tolist(),
                  leaf_font_size=5, color_threshold=0)
ax_dend.set_title("Domain Dendrogram", fontsize=10)

# Reorder heatmap by dendrogram
order = dend["leaves"]
ct_reordered = ct_comp.iloc[order]
# Color bar for compartment assignment
compartment_colors = [domain_to_compartment[d] for d in ct_reordered.index]

sns.heatmap(ct_reordered, cmap="YlOrRd", ax=ax_heat, linewidths=0.1,
            cbar_kws={"label": "Fraction"})
ax_heat.set_title(f"Per-TMA UTAG Domains — Cell Type Composition\n(ordered by hierarchical clustering)", fontsize=10)
ax_heat.set_ylabel("")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/{PREFIX}_domain_dendrogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {PREFIX}_domain_dendrogram.png")

# 3. Per-TMA compartment distribution
fig, ax = plt.subplots(figsize=(max(10, n_comp * 0.6), 5))
sns.heatmap(tma_comp, cmap="YlOrRd", ax=ax, linewidths=0.5,
            annot=True, fmt=".2f", cbar_kws={"label": "Fraction"})
ax.set_title(f"T-panel v8: Tissue Compartment Distribution per TMA", fontweight="bold")
ax.set_ylabel("TMA")
ax.set_xlabel("Tissue Compartment")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/{PREFIX}_tma_compartments.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {PREFIX}_tma_compartments.png")

# 4. B1_FL11 spatial plot: cell types + compartments
roi = adata[adata.obs["sample_id"] == "B1_FL11"].copy()
if len(roi) > 0:
    x = roi.obsm["spatial"][:, 0] if "spatial" in roi.obsm else roi.obs["centroid_x"].values
    y = roi.obsm["spatial"][:, 1] if "spatial" in roi.obsm else roi.obs["centroid_y"].values

    ct_colors = {
        "B cells": "#FFB347", "B cells (CXCR5hi)": "#4A90D9", "B cells (CD20hi)": "#87CEEB",
        "B (weak CD20)": "#DEB887", "GC B cells": "#FF8C00", "Activated B": "#FF6347",
        "B (TOXhi)": "#CD853F", "CD4 T cells": "#9370DB", "CD8 T cells": "#90EE90",
        "CD8 T exhausted": "#FFB6C1", "CD8 T pre-exhausted (TOX+)": "#FF69B4",
        "Treg": "#DC143C", "Macrophages": "#228B22", "LQ": "#D3D3D3",
        "Mixed": "#A9A9A9", "Other": "#C0C0C0", "Cytotoxic": "#006400",
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Cell types
    ax = axes[0]
    for ct in roi.obs["cell_type"].unique():
        m = roi.obs["cell_type"] == ct
        ax.scatter(x[m], y[m], c=ct_colors.get(ct, "#808080"), s=1, alpha=0.6, label=ct, rasterized=True)
    ax.set_title("B1_FL11 — Cell Types (v8)", fontweight="bold")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    ct_counts = roi.obs["cell_type"].value_counts()
    order = [labels.index(ct) for ct in ct_counts.index if ct in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", fontsize=6, markerscale=3, framealpha=0.8)

    # Per-TMA UTAG domains
    ax = axes[1]
    domains = roi.obs["utag_domain"].astype(str)
    unique_d = sorted(domains.unique())
    cmap = plt.colormaps.get_cmap("tab20")
    colors = {d: cmap(j % 20) for j, d in enumerate(unique_d)}
    for d in unique_d:
        m = domains == d
        short = d.split("_", 1)[1] if "_" in d else d
        ax.scatter(x[m], y[m], c=[colors[d]], s=1, alpha=0.6, label=short, rasterized=True)
    ax.set_title(f"B1_FL11 — UTAG Domains (res={PRIMARY_RES}, {len(unique_d)} domains)",
                 fontweight="bold")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=5, markerscale=3, framealpha=0.8, ncol=2)

    # Tissue compartments
    ax = axes[2]
    comps = roi.obs["tissue_compartment"].astype(str)
    unique_c = sorted(comps.unique(), key=lambda x: int(x))
    cmap2 = plt.colormaps.get_cmap("Set3")
    colors2 = {c: cmap2(j % 12) for j, c in enumerate(unique_c)}
    for c in unique_c:
        m = comps == c
        ax.scatter(x[m], y[m], c=[colors2[c]], s=1, alpha=0.6, label=f"Comp {c}", rasterized=True)
    ax.set_title(f"B1_FL11 — Tissue Compartments ({len(unique_c)} compartments)",
                 fontweight="bold")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=7, markerscale=3, framealpha=0.8)

    plt.suptitle("UTAG Per-TMA + Meta-clustering — B1_FL11 (T-panel v8)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{PREFIX}_FL11_spatial.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PREFIX}_FL11_spatial.png")

total = time.time() - t0
print(f"\n=== DONE (total: {total:.0f}s / {total/60:.1f} min) ===")
