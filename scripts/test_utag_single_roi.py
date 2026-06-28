#!/usr/bin/env python
"""Test UTAG on a single ROI from B1 T-panel."""
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=== Loading B1 T-panel data ===")
adata = ad.read_h5ad(
    "<PROJECT_ROOT>/output/batch/B1_T_raw_combined.h5ad"
)
print(f"Full B1: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"ROIs: {adata.obs['sample_id'].nunique()}")

# Pick the largest ROI for a good test
roi_sizes = adata.obs["sample_id"].value_counts()
print(f"\nROI sizes (top 5):\n{roi_sizes.head()}")
test_roi = roi_sizes.index[0]
print(f"\nUsing ROI: {test_roi} ({roi_sizes[test_roi]} cells)")

adata_roi = adata[adata.obs["sample_id"] == test_roi].copy()
print(f"ROI shape: {adata_roi.shape}")
print(f"Spatial range X: {adata_roi.obsm['spatial'][:,0].min():.1f} - {adata_roi.obsm['spatial'][:,0].max():.1f}")
print(f"Spatial range Y: {adata_roi.obsm['spatial'][:,1].min():.1f} - {adata_roi.obsm['spatial'][:,1].max():.1f}")
print(f"X stats - min: {float(adata_roi.X.min()):.3f}, max: {float(adata_roi.X.max()):.3f}, mean: {float(adata_roi.X.mean()):.3f}")

if "cell_type" in adata_roi.obs.columns:
    print(f"\nCell types:\n{adata_roi.obs['cell_type'].value_counts()}")

# Run UTAG on single ROI
print("\n=== Running UTAG on single ROI ===")
from utag import utag as run_utag
import time

t0 = time.time()
utag_results = run_utag(
    adata_roi,
    slide_key=None,  # single ROI, no need for slide_key
    max_dist=20,
    normalization_mode="l2_norm",
    apply_clustering=True,
    clustering_method="leiden",
    resolutions=[0.3, 0.5, 1.0],
)
elapsed = time.time() - t0
print(f"UTAG completed in {elapsed:.1f} seconds")

print(f"\nUTAG results shape: {utag_results.shape}")
utag_cols = [c for c in utag_results.obs.columns if "UTAG" in c]
print(f"UTAG columns: {utag_cols}")

for col in utag_cols:
    n_domains = utag_results.obs[col].nunique()
    print(f"\n{col}: {n_domains} domains")
    print(utag_results.obs[col].value_counts().head(10))

# Cross-tabulate UTAG domains with cell types
if "cell_type" in utag_results.obs.columns and utag_cols:
    mid_col = utag_cols[len(utag_cols) // 2] if len(utag_cols) > 1 else utag_cols[0]
    ct = pd.crosstab(
        utag_results.obs[mid_col],
        utag_results.obs["cell_type"],
        normalize="index",
    )
    print(f"\n=== Cell type composition per UTAG domain ({mid_col}) ===")
    print(ct.round(3).to_string())

# Save results
outdir = "<PROJECT_ROOT>/output"
outpath = f"{outdir}/B1_T_{test_roi}_utag.h5ad"
utag_results.write_h5ad(outpath)
print(f"\nSaved to {outpath}")

# Visualization
fig, axes = plt.subplots(1, min(3, len(utag_cols)), figsize=(6 * min(3, len(utag_cols)), 5))
if len(utag_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, utag_cols[:3]):
    sc.pl.embedding(
        utag_results, basis="spatial", color=col, ax=ax, show=False,
        title=col, size=2,
    )
plt.tight_layout()
figpath = f"{outdir}/B1_T_{test_roi}_utag_spatial.png"
fig.savefig(figpath, dpi=150, bbox_inches="tight")
print(f"Saved spatial plot to {figpath}")

# Also plot cell types for comparison
fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
sc.pl.embedding(
    utag_results, basis="spatial", color="cell_type", ax=ax2, show=False,
    title="Cell types", size=2,
)
fig2.savefig(f"{outdir}/B1_T_{test_roi}_celltypes_spatial.png", dpi=150, bbox_inches="tight")
print(f"Saved cell type plot")

print("\n=== DONE ===")
