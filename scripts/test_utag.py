#!/usr/bin/env python
"""Test UTAG on B1 T-panel data."""
import sys
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=== Loading B1 T-panel data ===")
adata_b1 = ad.read_h5ad(
    "<PROJECT_ROOT>/output/batch/B1_T_raw_combined.h5ad"
)
print(f"B1 T-panel: {adata_b1.shape}")
print(f"obs columns: {list(adata_b1.obs.columns)}")
print(f"obsm keys: {list(adata_b1.obsm.keys())}")

# Verify spatial coordinates exist
assert "spatial" in adata_b1.obsm, "No spatial key in obsm!"
print(f"Spatial coords shape: {adata_b1.obsm['spatial'].shape}")
print(f"Spatial range X: {adata_b1.obsm['spatial'][:,0].min():.1f} - {adata_b1.obsm['spatial'][:,0].max():.1f}")
print(f"Spatial range Y: {adata_b1.obsm['spatial'][:,1].min():.1f} - {adata_b1.obsm['spatial'][:,1].max():.1f}")

# slide_key = sample_id to keep ROIs separate
sample_col = "sample_id"
print(f"ROIs: {adata_b1.obs[sample_col].nunique()} unique values")
print(f"ROI sizes:\n{adata_b1.obs[sample_col].value_counts().describe()}")

if "cell_type" in adata_b1.obs.columns:
    print(f"\nCell types:\n{adata_b1.obs['cell_type'].value_counts()}")

# Run UTAG
print("\n=== Running UTAG ===")
from utag import utag as run_utag

print(f"X min: {float(adata_b1.X.min()):.3f}, max: {float(adata_b1.X.max()):.3f}, mean: {float(adata_b1.X.mean()):.3f}")

slide_key = sample_col
print(f"Using slide_key: {slide_key}")

utag_results = run_utag(
    adata_b1,
    slide_key=slide_key,
    max_dist=20,
    normalization_mode="l2_norm",
    apply_clustering=True,
    clustering_method="leiden",
    resolutions=[0.3, 0.5, 1.0],
)

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
outpath = "<PROJECT_ROOT>/output/B1_T_utag.h5ad"
utag_results.write_h5ad(outpath)
print(f"\nSaved to {outpath}")

# Visualization
fig, axes = plt.subplots(1, min(3, len(utag_cols)), figsize=(6 * min(3, len(utag_cols)), 5))
if len(utag_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, utag_cols[:3]):
    sc.pl.embedding(
        utag_results, basis="spatial", color=col, ax=ax, show=False,
        title=col, size=0.5,
    )
plt.tight_layout()
figpath = "<PROJECT_ROOT>/output/B1_T_utag_spatial.png"
fig.savefig(figpath, dpi=150, bbox_inches="tight")
print(f"Saved spatial plot to {figpath}")

print("\n=== DONE ===")
