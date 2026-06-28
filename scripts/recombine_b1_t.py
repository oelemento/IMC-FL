#!/usr/bin/env python3
"""Re-combine B1 T-panel per-ROI files preserving raw counts.
Same logic as recombine_raw.py but filtered to B1 Tcellpanel files only.
"""
import sys
import scanpy as sc
import anndata as ad
from pathlib import Path

BASE = Path("<PROJECT_ROOT>")
input_dir = BASE / "output" / "batch"

# Only B1 Tcellpanel files
h5ad_files = sorted([
    f for f in input_dir.glob("*B1*Tcell*.h5ad")
    if "combined" not in f.name and "TMA" not in f.name and "global" not in f.name
])

print(f"Found {len(h5ad_files)} B1 T-panel ROI files in {input_dir}")

adatas = []
for f in h5ad_files:
    adata = sc.read_h5ad(f)
    if adata.raw is not None:
        adata_raw = adata.raw.to_adata()
        adata_raw.obs = adata.obs.copy()
        adata = adata_raw
    else:
        if adata.X.min() < 0:
            print(f"  WARNING: {f.name} has no .raw and X has negatives — skipping")
            continue

    sample_id = adata.obs["sample_id"].iloc[0] if "sample_id" in adata.obs.columns else f.stem
    print(f"  {sample_id}: {adata.n_obs} cells, X range [{adata.X.min():.1f}, {adata.X.max():.1f}]")
    adatas.append(adata)

print(f"\nConcatenating {len(adatas)} ROIs...")
combined = ad.concat(adatas, join="outer")
combined.obs_names_make_unique()

print(f"X range: [{combined.X.min():.2f}, {combined.X.max():.2f}]")
assert combined.X.min() >= 0, f"ERROR: X has negative values ({combined.X.min():.2f})"

outfile = input_dir / "B1_T_raw_combined.h5ad"
combined.write(outfile)
print(f"\nSaved: {outfile}")
print(f"  {combined.n_obs:,} cells x {combined.n_vars} markers")
print(f"  X range: [{combined.X.min():.2f}, {combined.X.max():.2f}]")
