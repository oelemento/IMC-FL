#!/usr/bin/env python3
"""Build inventory of all h5ad files in output/.
For each file: shape, .X range, .raw range, obs columns, leiden columns, cell_type presence, TMA label.
Outputs TSV to stdout for easy reading.
"""
import os
import sys
import glob
import anndata as ad
import numpy as np

BASE = os.environ.get("BASE_DIR", "<PROJECT_ROOT>")
OUT_DIR = f"{BASE}/output"

# Find all h5ad files
h5ad_files = sorted(glob.glob(f"{OUT_DIR}/**/*.h5ad", recursive=True))
# Also check output/ root
h5ad_files += sorted(glob.glob(f"{OUT_DIR}/*.h5ad"))
# Deduplicate
h5ad_files = sorted(set(h5ad_files))

print(f"Found {len(h5ad_files)} h5ad files\n")

# Header
print(f"{'File':<75} {'Shape':>15} {'X_min':>8} {'X_max':>8} {'raw_min':>8} {'raw_max':>8} {'has_raw':>7} {'has_ct':>6} {'leiden_cols':>30} {'tma':>6}")
print("-" * 190)

for path in h5ad_files:
    rel = path.replace(OUT_DIR + "/", "")
    try:
        adata = ad.read_h5ad(path)
        shape = f"{adata.shape[0]:,}x{adata.shape[1]}"
        x_min = f"{float(adata.X.min()):.2f}"
        x_max = f"{float(adata.X.max()):.2f}"

        if adata.raw is not None:
            raw_min = f"{float(adata.raw.X.min()):.2f}"
            raw_max = f"{float(adata.raw.X.max()):.2f}"
            has_raw = "yes"
        else:
            raw_min = raw_max = "n/a"
            has_raw = "no"

        has_ct = "yes" if "cell_type" in adata.obs.columns else "no"
        leiden_cols = [c for c in adata.obs.columns if "leiden" in c.lower()]
        leiden_str = ",".join(leiden_cols) if leiden_cols else "none"

        # TMA label
        tma = ""
        if "tma" in adata.obs.columns:
            tma = ",".join(sorted(adata.obs["tma"].unique()))
        elif "TMA" in adata.obs.columns:
            tma = ",".join(sorted(adata.obs["TMA"].unique()))

        print(f"{rel:<75} {shape:>15} {x_min:>8} {x_max:>8} {raw_min:>8} {raw_max:>8} {has_raw:>7} {has_ct:>6} {leiden_str:>30} {tma:>6}")

    except Exception as e:
        print(f"{rel:<75} ERROR: {e}")

print()
print("=== Legend ===")
print("X_min/X_max: range of .X matrix (negative = transformed/scaled)")
print("raw_min/raw_max: range of .raw matrix (should be >=0 for true raw ion counts)")
print("has_ct: has cell_type annotation in .obs")
print("tma: TMA labels if cross-TMA file")
