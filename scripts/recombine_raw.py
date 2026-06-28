#!/usr/bin/env python3
"""Re-combine per-ROI h5ad files preserving RAW counts in .X

The original combine_results.py saved transformed (arcsinh+scaled) data.
This script loads .raw from each per-ROI h5ad (which has original ion counts)
and saves a combined file with raw counts in .X, suitable for global_analysis.py.

Usage:
    python scripts/recombine_raw.py --input-dir output/A1_T --panel A1_T
    python scripts/recombine_raw.py --input-dir output/batch --panel B1_T
"""

import sys
import os
import scanpy as sc
import anndata as ad
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Re-combine ROI h5ad files with raw counts')
parser.add_argument('--input-dir', required=True, help='Directory with individual h5ad files')
parser.add_argument('--panel', required=True, help='Panel name for output file')
args = parser.parse_args()

input_dir = Path(args.input_dir)

# Find all individual ROI h5ad files (exclude combined/global files)
h5ad_files = sorted([
    f for f in input_dir.glob('*.h5ad')
    if 'combined' not in f.name and 'TMA' not in f.name and 'global' not in f.name
])

print(f"Found {len(h5ad_files)} ROI files in {input_dir}")

if not h5ad_files:
    print("No files to combine.")
    sys.exit(0)

adatas = []
skipped = 0
for f in h5ad_files:
    adata = sc.read_h5ad(f)

    # Extract raw counts if available
    if adata.raw is not None:
        adata_raw = adata.raw.to_adata()
        # Keep obs metadata from processed version
        adata_raw.obs = adata.obs.copy()
        adata = adata_raw
    else:
        # Check if data looks already transformed (has negative values)
        if adata.X.min() < 0:
            print(f"  WARNING: {f.name} has no .raw and X has negative values — skipping")
            skipped += 1
            continue

    sample_id = adata.obs['sample_id'].iloc[0] if 'sample_id' in adata.obs.columns else f.stem
    print(f"  {sample_id}: {adata.n_obs} cells, X range [{adata.X.min():.1f}, {adata.X.max():.1f}]")
    adatas.append(adata)

if not adatas:
    print("ERROR: No valid files to combine!")
    sys.exit(1)

print(f"\nConcatenating {len(adatas)} ROIs (skipped {skipped})...")

# Use file stems as keys to avoid duplicate sample_id issues
keys = [f.stem for f in h5ad_files[:len(adatas)+skipped] if 'combined' not in f.name]
# Actually just use indices to guarantee uniqueness
adata_combined = ad.concat(adatas, join='outer')
adata_combined.obs_names_make_unique()

# Verify raw counts
print(f"X range: [{adata_combined.X.min():.2f}, {adata_combined.X.max():.2f}]")
assert adata_combined.X.min() >= 0, f"ERROR: X has negative values ({adata_combined.X.min():.2f}), not raw counts!"

outfile = input_dir / f'{args.panel}_raw_combined.h5ad'
adata_combined.write(outfile)
print(f"\nSaved: {outfile}")
print(f"  {adata_combined.n_obs:,} cells x {adata_combined.n_vars} markers")
print(f"  X range: [{adata_combined.X.min():.2f}, {adata_combined.X.max():.2f}]")
