#!/usr/bin/env python
"""Run UTAG on all TMAs S-panel v8 (2.18M cells, 170 ROIs)."""
import time
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTDIR = "<PROJECT_ROOT>/output"
INPUT = f"{OUTDIR}/all_TMA_S_global_v8.h5ad"
PREFIX = "all_TMA_S_utag"

print("=== Loading S-panel v8 ===")
t0 = time.time()
adata = ad.read_h5ad(INPUT)
print(f"Loaded {adata.shape} in {time.time()-t0:.0f}s")
print(f"ROIs: {adata.obs['sample_id'].nunique()}")
print(f"TMAs: {adata.obs['tma'].value_counts().to_string()}")
print(f"X range: [{float(adata.X.min()):.1f}, {float(adata.X.max()):.1f}]")

# Run UTAG
print("\n=== Running UTAG ===")
from utag import utag as run_utag

t1 = time.time()
utag_results = run_utag(
    adata,
    slide_key="sample_id",
    max_dist=20,
    normalization_mode="l2_norm",
    apply_clustering=True,
    clustering_method="leiden",
    resolutions=[0.05, 0.1, 0.2],
    leiden_kwargs=dict(flavor="igraph", n_iterations=2, directed=False),
)
elapsed = time.time() - t1
print(f"\nUTAG completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# Results summary
utag_cols = [c for c in utag_results.obs.columns if "UTAG" in c]
print(f"UTAG columns: {utag_cols}")
for col in utag_cols:
    print(f"  {col}: {utag_results.obs[col].nunique()} domains")

# Cross-tabulate with cell types
mid_col = "UTAG Label_leiden_0.1"
if mid_col not in utag_results.obs.columns:
    mid_col = utag_cols[len(utag_cols) // 2] if utag_cols else None

if mid_col:
    ct = pd.crosstab(
        utag_results.obs[mid_col],
        utag_results.obs["cell_type"],
        normalize="index",
    )
    print(f"\n=== Cell type composition per domain ({mid_col}) ===")
    print(ct.round(3).to_string())

    # Per-TMA domain distribution
    tma_domain = pd.crosstab(
        utag_results.obs["tma"],
        utag_results.obs[mid_col],
        normalize="index",
    )
    print(f"\n=== Domain distribution per TMA ({mid_col}) ===")
    print(tma_domain.round(3).to_string())

# Save h5ad
outpath = f"{OUTDIR}/{PREFIX}.h5ad"
utag_results.write_h5ad(outpath)
print(f"\nSaved to {outpath}")

# --- Plots ---
print("\n=== Generating plots ===")

# 1. Composition heatmap
if mid_col:
    ct = pd.crosstab(
        utag_results.obs[mid_col],
        utag_results.obs["cell_type"],
        normalize="index",
    )
    fig, ax = plt.subplots(figsize=(14, max(8, len(ct) * 0.4)))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.3, cbar_kws={"label": "Fraction"},
                annot_kws={"size": 6})
    ax.set_title(f"S-panel v8: Cell Type Composition per UTAG Domain\n({mid_col})", fontweight="bold")
    ax.set_ylabel("UTAG Domain")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{PREFIX}_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PREFIX}_composition.png")

# 2. Per-TMA domain distribution
if mid_col:
    tma_domain = pd.crosstab(
        utag_results.obs["tma"],
        utag_results.obs[mid_col],
        normalize="index",
    )
    fig, ax = plt.subplots(figsize=(max(10, len(tma_domain.columns) * 0.4), 5))
    sns.heatmap(tma_domain, cmap="YlOrRd", ax=ax, linewidths=0.5,
                annot=True, fmt=".2f", cbar_kws={"label": "Fraction"})
    ax.set_title(f"S-panel v8: UTAG Domain Distribution per TMA\n({mid_col})", fontweight="bold")
    ax.set_ylabel("TMA")
    ax.set_xlabel("UTAG Domain")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{PREFIX}_tma_domains.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PREFIX}_tma_domains.png")

# 3. Per-ROI domain distribution (sampled for readability)
if mid_col:
    roi_domain = pd.crosstab(
        utag_results.obs["sample_id"],
        utag_results.obs[mid_col],
        normalize="index",
    )
    fig, ax = plt.subplots(figsize=(max(10, len(roi_domain.columns) * 0.4), max(12, len(roi_domain) * 0.15)))
    sns.heatmap(roi_domain, cmap="YlOrRd", ax=ax, linewidths=0.1,
                cbar_kws={"label": "Fraction"})
    ax.set_title(f"S-panel v8: UTAG Domain Distribution per ROI\n({mid_col})", fontweight="bold")
    ax.set_ylabel("ROI")
    ax.set_xlabel("UTAG Domain")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{PREFIX}_roi_domains.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PREFIX}_roi_domains.png")

total = time.time() - t0
print(f"\n=== DONE (total: {total:.0f}s / {total/60:.1f} min) ===")
