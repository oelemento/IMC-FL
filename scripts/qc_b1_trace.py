#!/usr/bin/env python3
"""Trace B1 T-panel .raw negative values back to source."""
import anndata as ad
import numpy as np
import glob

BASE = "<PROJECT_ROOT>"

# 1. Check the raw_combined file (used by cross_tma_global.py)
print("=== B1 T raw_combined (input to cross-TMA) ===")
a = ad.read_h5ad(f"{BASE}/output/batch/TMA_B1_T_raw_combined.h5ad")
print(f"Shape: {a.shape}")
print(f".X range: [{float(a.X.min()):.3f}, {float(a.X.max()):.3f}]")
if a.raw is not None:
    print(f".raw range: [{float(a.raw.X.min()):.3f}, {float(a.raw.X.max()):.3f}]")
    print(f".raw var_names[:5]: {list(a.raw.var_names[:5])}")
else:
    print(".raw: None")
print(f"obs cols: {list(a.obs.columns)}")
print()

# 2. Check the original combined file (before recombine fix)
print("=== B1 T original combined ===")
b = ad.read_h5ad(f"{BASE}/output/batch/TMA_B1_T_combined.h5ad")
print(f"Shape: {b.shape}")
print(f".X range: [{float(b.X.min()):.3f}, {float(b.X.max()):.3f}]")
if b.raw is not None:
    print(f".raw range: [{float(b.raw.X.min()):.3f}, {float(b.raw.X.max()):.3f}]")
else:
    print(".raw: None")
print()

# 3. Check per-ROI files for B1 T
roi_files = sorted(glob.glob(f"{BASE}/output/batch/B1_T/*.h5ad"))
print(f"=== B1 T per-ROI files: {len(roi_files)} ===")
if roi_files:
    for rf in roi_files[:3]:
        r = ad.read_h5ad(rf)
        raw_info = "None"
        if r.raw is not None:
            raw_info = f"[{float(r.raw.X.min()):.3f}, {float(r.raw.X.max()):.3f}]"
        print(f"  {rf.split('/')[-1]}: .X=[{float(r.X.min()):.3f}, {float(r.X.max()):.3f}], .raw={raw_info}")
else:
    print("  No per-ROI files found in output/batch/B1_T/")
    # Check alternative locations
    for pattern in ["output/B1_T/*.h5ad", "output/batch/TMA_B1_T_*.h5ad"]:
        alt = sorted(glob.glob(f"{BASE}/{pattern}"))
        if alt:
            print(f"  Found at {pattern}: {len(alt)} files")

# 4. Check what recombine_raw.py would have used
print()
print("=== Checking recombine_raw.py source directory ===")
import os
for d in ["output/batch/B1_T", "output/B1_T"]:
    full = f"{BASE}/{d}"
    if os.path.isdir(full):
        files = [f for f in os.listdir(full) if f.endswith(".h5ad")]
        print(f"  {d}: {len(files)} h5ad files")
    else:
        print(f"  {d}: does not exist")
