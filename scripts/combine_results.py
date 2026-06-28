#!/usr/bin/env python3
"""Combine individual ROI h5ad files into a single TMA-level AnnData."""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import scanpy as sc
import anndata as ad
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Combine batch-processed ROI results')
parser.add_argument('--input-dir', required=True, help='Directory with individual h5ad files')
parser.add_argument('--panel', default='T', help='Panel name for output file')
args = parser.parse_args()

input_dir = Path(args.input_dir)

# Find all individual ROI h5ad files (exclude combined files)
h5ad_files = sorted([
    f for f in input_dir.glob('*.h5ad')
    if 'combined' not in f.name and 'TMA' not in f.name
])

print(f"Found {len(h5ad_files)} ROI files in {input_dir}")

if not h5ad_files:
    print("No files to combine.")
    sys.exit(0)

# Load all
adatas = []
for f in h5ad_files:
    adata = sc.read_h5ad(f)
    sample_id = f.stem
    print(f"  {sample_id}: {adata.n_obs} cells")
    adatas.append(adata)

# Concatenate
print(f"\nConcatenating {len(adatas)} ROIs...")
adata_combined = ad.concat(
    adatas, join='outer', label='sample_id',
    keys=[a.obs['sample_id'].iloc[0] for a in adatas]
)
adata_combined.obs_names_make_unique()

outfile = input_dir / f'TMA_B1_{args.panel}_combined.h5ad'
adata_combined.write(outfile)
print(f"Saved: {outfile} ({adata_combined.n_obs} cells)")

# Summary
print(f"\n=== Summary ===")
print(f"Total ROIs: {len(adatas)}")
print(f"Total cells: {adata_combined.n_obs}")

if 'cell_type' in adata_combined.obs.columns:
    print(f"\n=== Cell Type Composition ===")
    comp = adata_combined.obs['cell_type'].value_counts()
    for ct, n in comp.items():
        pct = n / adata_combined.n_obs * 100
        print(f"  {ct}: {n} ({pct:.1f}%)")
