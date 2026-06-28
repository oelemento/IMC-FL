#!/usr/bin/env python
"""Run UTAG on full B1 T-panel (631K cells, all 50 ROIs together)."""
import time
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "<PROJECT_ROOT>/output"
INPUT = f"{OUTDIR}/batch/B1_T_raw_combined.h5ad"

print("=== Loading B1 T-panel data ===")
t0 = time.time()
adata = ad.read_h5ad(INPUT)
print(f"Loaded {adata.shape} in {time.time()-t0:.0f}s")
print(f"ROIs: {adata.obs['sample_id'].nunique()}")
print(f"X range: [{float(adata.X.min()):.1f}, {float(adata.X.max()):.1f}]")

# Run UTAG
print("\n=== Running UTAG (all 50 ROIs, slide_key='sample_id') ===")
from utag import utag as run_utag

t1 = time.time()
utag_results = run_utag(
    adata,
    slide_key="sample_id",
    max_dist=20,
    normalization_mode="l2_norm",
    apply_clustering=True,
    clustering_method="leiden",
    resolutions=[0.3, 0.5, 1.0],
    leiden_kwargs=dict(flavor="igraph", n_iterations=2, directed=False),
)
elapsed = time.time() - t1
print(f"\nUTAG completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# Results summary
utag_cols = [c for c in utag_results.obs.columns if "UTAG" in c]
print(f"\nUTAG columns: {utag_cols}")
for col in utag_cols:
    n = utag_results.obs[col].nunique()
    print(f"  {col}: {n} domains")

# Cross-tabulate with cell types at each resolution
for col in utag_cols:
    ct = pd.crosstab(
        utag_results.obs[col],
        utag_results.obs["cell_type"],
        normalize="index",
    )
    print(f"\n=== Cell type composition per domain ({col}) ===")
    print(ct.round(3).to_string())

# Per-ROI domain distribution at res 0.5
mid_col = "UTAG Label_leiden_0.5"
if mid_col in utag_results.obs.columns:
    roi_domain = pd.crosstab(
        utag_results.obs["sample_id"],
        utag_results.obs[mid_col],
        normalize="index",
    )
    print(f"\n=== Domain distribution per ROI ({mid_col}) ===")
    print(roi_domain.round(3).to_string())

# Save
outpath = f"{OUTDIR}/B1_T_utag_full.h5ad"
utag_results.write_h5ad(outpath)
print(f"\nSaved to {outpath}")

# Visualizations
print("\n=== Generating plots ===")

# 1. Spatial plot of UTAG domains (sample of ROIs for readability)
rois = adata.obs["sample_id"].value_counts().head(6).index.tolist()
for col in utag_cols:
    n_rois = len(rois)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, roi in enumerate(rois):
        mask = utag_results.obs["sample_id"] == roi
        sub = utag_results[mask]
        sc.pl.embedding(sub, basis="spatial", color=col, ax=axes[i], show=False,
                        title=f"{roi} (n={mask.sum()})", size=3, palette="tab20")
        axes[i].set_aspect("equal")
    plt.suptitle(f"UTAG Domains — {col}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    res_tag = col.split("_")[-1]
    fig.savefig(f"{OUTDIR}/B1_T_utag_spatial_{res_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved B1_T_utag_spatial_{res_tag}.png")

# 2. Composition heatmap
if mid_col in utag_results.obs.columns:
    ct = pd.crosstab(
        utag_results.obs[mid_col],
        utag_results.obs["cell_type"],
        normalize="index",
    )
    fig, ax = plt.subplots(figsize=(10, max(6, len(ct) * 0.4)))
    import seaborn as sns
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Fraction"})
    ax.set_title(f"Cell Type Composition per UTAG Domain\n({mid_col})", fontweight="bold")
    ax.set_ylabel("UTAG Domain")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/B1_T_utag_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved B1_T_utag_composition.png")

# 3. Domain proportion per ROI heatmap
if mid_col in utag_results.obs.columns:
    roi_domain = pd.crosstab(
        utag_results.obs["sample_id"],
        utag_results.obs[mid_col],
        normalize="index",
    )
    fig, ax = plt.subplots(figsize=(max(8, len(roi_domain.columns) * 0.5), max(10, len(roi_domain) * 0.3)))
    sns.heatmap(roi_domain, cmap="YlOrRd", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Fraction"})
    ax.set_title(f"UTAG Domain Distribution per ROI\n({mid_col})", fontweight="bold")
    ax.set_ylabel("ROI")
    ax.set_xlabel("UTAG Domain")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/B1_T_utag_roi_domains.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved B1_T_utag_roi_domains.png")

total = time.time() - t0
print(f"\n=== DONE (total: {total:.0f}s / {total/60:.1f} min) ===")
