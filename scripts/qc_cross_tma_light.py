#!/usr/bin/env python3
"""Lightweight QC - only reads obs metadata, not full matrix.
Gets TMA crosstab and .raw per-TMA min from a single file.
"""
import sys
import anndata as ad
import pandas as pd
import numpy as np

path = sys.argv[1]
print(f"Loading {path} (backed mode for .raw check)...")

# Read full for .raw check but only need obs + raw.X
adata = ad.read_h5ad(path)

# TMA × cell_type crosstab
if "cell_type" in adata.obs.columns and "TMA" in adata.obs.columns:
    print(f"\nCell type × TMA (%):")
    ct_counts = adata.obs["cell_type"].value_counts()
    ct = pd.crosstab(adata.obs["cell_type"], adata.obs["TMA"], normalize="columns") * 100
    ct = ct.loc[ct_counts.index]
    print(ct.round(1).to_string())

# .raw min per TMA
if adata.raw is not None and "TMA" in adata.obs.columns:
    print(f"\n.raw min per TMA:")
    for tma in sorted(adata.obs["TMA"].unique()):
        mask = adata.obs["TMA"] == tma
        tma_raw_min = float(adata.raw.X[mask.values].min())
        tma_raw_max = float(adata.raw.X[mask.values].max())
        flag = " *** NEGATIVE ***" if tma_raw_min < 0 else ""
        print(f"  {tma}: [{tma_raw_min:.3f}, {tma_raw_max:.3f}]{flag}")

print("\nDone.")
