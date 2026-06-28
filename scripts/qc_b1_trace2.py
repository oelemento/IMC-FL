#!/usr/bin/env python3
"""Trace B1 T-panel .raw issue."""
import anndata as ad
import numpy as np
import glob

BASE = "<PROJECT_ROOT>"

# 1. The file cross_tma uses for B1 T
path = f"{BASE}/output/batch/TMA_B1_T_combined.h5ad"
print(f"=== {path} ===")
a = ad.read_h5ad(path)
print(f"Shape: {a.shape}")
print(f".X range: [{float(a.X.min()):.3f}, {float(a.X.max()):.3f}]")
if a.raw is not None:
    print(f".raw shape: {a.raw.shape}")
    print(f".raw range: [{float(a.raw.X.min()):.3f}, {float(a.raw.X.max()):.3f}]")
    print(f".raw var_names[:5]: {list(a.raw.var_names[:5])}")
else:
    print(".raw: None")
print(f"obs cols: {list(a.obs.columns)}")
print(f"var cols: {list(a.var.columns)}")
print()

# 2. Check per-ROI batch files for B1 T
roi_files = sorted(glob.glob(f"{BASE}/output/batch/*B1*Tcell*.h5ad"))
print(f"=== B1 T per-ROI batch files: {len(roi_files)} ===")
for rf in roi_files[:3]:
    r = ad.read_h5ad(rf)
    raw_info = "None"
    if r.raw is not None:
        raw_info = f"[{float(r.raw.X.min()):.3f}, {float(r.raw.X.max()):.3f}]"
    print(f"  {rf.split('/')[-1]}: shape={r.shape}, .X=[{float(r.X.min()):.3f}, {float(r.X.max()):.3f}], .raw={raw_info}")
print()

# 3. Check the original local B1 analysis (in B1_T dir if it exists)
import os
for d in ["output/B1_T", "output/batch/B1_T"]:
    full = f"{BASE}/{d}"
    if os.path.isdir(full):
        files = [f for f in os.listdir(full) if f.endswith(".h5ad")]
        print(f"=== {d}: {len(files)} h5ad files ===")
        for f in files[:3]:
            fp = f"{full}/{f}"
            adata = ad.read_h5ad(fp)
            raw_info = "None"
            if adata.raw is not None:
                raw_info = f"[{float(adata.raw.X.min()):.3f}, {float(adata.raw.X.max()):.3f}]"
            print(f"  {f}: shape={adata.shape}, .X=[{float(adata.X.min()):.3f}, {float(adata.X.max()):.3f}], .raw={raw_info}")

# 4. Key question: does cross_tma_global.py transform .X again?
# If B1 combined already has transformed .X (arcsinh+scaled), and the embed step
# transforms again, that would make .raw = already-transformed .X = negative values
print()
print("=== Key diagnostic ===")
print(f"B1 combined .X min: {float(a.X.min()):.3f}")
print(f"B1 combined .X max: {float(a.X.max()):.3f}")
has_negative_x = float(a.X.min()) < 0
print(f".X has negatives: {has_negative_x}")
print(f"If .X has negatives, it is ALREADY transformed (arcsinh+scaled).")
print(f"cross_tma_global.py stores .raw = current .X before re-transforming.")
print(f"So .raw would contain the already-transformed values = NEGATIVE.")
