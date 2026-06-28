#!/usr/bin/env python3
"""Fix B1 T-panel: extract raw counts from .raw of the combined file.

TMA_B1_T_combined.h5ad has 631K cells with:
  .X  = already transformed (arcsinh+scaled, range [-9, 10])
  .raw = true raw ion counts (range [0, 1161])

This script extracts .raw into .X and saves as B1_T_raw_combined.h5ad,
matching the format of all other *_raw_combined.h5ad files.
"""
import anndata as ad

BASE = "<PROJECT_ROOT>"
SRC = f"{BASE}/output/batch/TMA_B1_T_combined.h5ad"
DST = f"{BASE}/output/batch/B1_T_raw_combined.h5ad"

print(f"Loading {SRC}...")
adata = ad.read_h5ad(SRC)
print(f"  Shape: {adata.shape}")
print(f"  .X range: [{float(adata.X.min()):.3f}, {float(adata.X.max()):.3f}]")
print(f"  .raw range: [{float(adata.raw.X.min()):.3f}, {float(adata.raw.X.max()):.3f}]")

# Extract raw counts
raw_adata = adata.raw.to_adata()
raw_adata.obs = adata.obs.copy()
print(f"\nExtracted .raw:")
print(f"  Shape: {raw_adata.shape}")
print(f"  .X range: [{float(raw_adata.X.min()):.3f}, {float(raw_adata.X.max()):.3f}]")

assert raw_adata.X.min() >= 0, f"ERROR: .X has negative values ({raw_adata.X.min():.3f})"
assert raw_adata.shape[0] == adata.shape[0], f"ERROR: cell count mismatch"

raw_adata.write(DST)
print(f"\nSaved: {DST}")
print(f"  {raw_adata.n_obs:,} cells x {raw_adata.n_vars} markers")
print(f"  .X range: [{float(raw_adata.X.min()):.3f}, {float(raw_adata.X.max()):.3f}]")
print("Done.")
