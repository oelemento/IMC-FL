#!/usr/bin/env python3
"""Tune Cellpose parameters for dense lymphoma tissue."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import segmentation
from scipy import ndimage
import time

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import prepare_nuclear_image, segment_cells_local_maxima

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load ROI
files = list_rois(DATA_DIR)
image, markers, metadata = load_roi_txt(files[0])
sample_id = extract_sample_id(files[0].name)

# Prepare nuclear image - use smaller patch for faster iteration
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])
h, w = nuclear.shape
patch_size = 300  # smaller for faster testing
y0, x0 = (h - patch_size) // 2, (w - patch_size) // 2
patch = nuclear[y0:y0+patch_size, x0:x0+patch_size]

print(f"Sample: {sample_id}")
print(f"Patch: {patch_size}x{patch_size}")

# Reference: local maxima
masks_ref = segment_cells_local_maxima(patch, sigma=1.0, min_distance=2, expansion='voronoi')
n_ref = masks_ref.max()
print(f"\nReference (local maxima): {n_ref} cells")

# Cellpose parameter sweep
from cellpose import models

model = models.CellposeModel(gpu=False)
img = (patch * 255).astype(np.uint8)

# Parameters to test
# cellprob_threshold: lower = more cells (can be negative)
# flow_threshold: higher = more permissive
# diameter: cell size (None = auto, or specify)

param_sets = [
    # Default
    {'cellprob_threshold': 0.0, 'flow_threshold': 0.4, 'diameter': None, 'name': 'Default'},
    # More permissive cellprob
    {'cellprob_threshold': -1.0, 'flow_threshold': 0.4, 'diameter': None, 'name': 'cellprob=-1'},
    {'cellprob_threshold': -2.0, 'flow_threshold': 0.4, 'diameter': None, 'name': 'cellprob=-2'},
    {'cellprob_threshold': -3.0, 'flow_threshold': 0.4, 'diameter': None, 'name': 'cellprob=-3'},
    # More permissive flow
    {'cellprob_threshold': 0.0, 'flow_threshold': 0.8, 'diameter': None, 'name': 'flow=0.8'},
    {'cellprob_threshold': 0.0, 'flow_threshold': 1.0, 'diameter': None, 'name': 'flow=1.0'},
    # Combined
    {'cellprob_threshold': -2.0, 'flow_threshold': 0.8, 'diameter': None, 'name': 'cp=-2,fl=0.8'},
    {'cellprob_threshold': -3.0, 'flow_threshold': 1.0, 'diameter': None, 'name': 'cp=-3,fl=1.0'},
    # Smaller diameter (for small cells)
    {'cellprob_threshold': -2.0, 'flow_threshold': 0.4, 'diameter': 15, 'name': 'd=15,cp=-2'},
    {'cellprob_threshold': -2.0, 'flow_threshold': 0.4, 'diameter': 10, 'name': 'd=10,cp=-2'},
]

results = []

print("\nTesting Cellpose parameters...")
for params in param_sets:
    name = params.pop('name')
    print(f"  {name}...", end=' ', flush=True)

    t0 = time.time()
    masks, _, _ = model.eval(img, **params)
    t = time.time() - t0
    n = masks.max()

    results.append({
        'name': name,
        'n_cells': n,
        'time': t,
        'masks': masks,
        **params
    })
    print(f"{n} cells in {t:.1f}s")

# Find best result (closest to reference)
best = min(results, key=lambda x: abs(x['n_cells'] - n_ref))
print(f"\nBest match to reference: {best['name']} with {best['n_cells']} cells")
print(f"Reference: {n_ref} cells")

# Visualization
n_results = len(results)
cols = 4
rows = (n_results + 1) // cols + 1

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.flatten()

# Reference
boundaries_ref = segmentation.find_boundaries(masks_ref, mode='outer')
axes[0].imshow(patch, cmap='gray')
axes[0].imshow(boundaries_ref, cmap='Greens', alpha=0.7)
axes[0].set_title(f'Reference (local maxima)\nn={n_ref}', fontsize=10)
axes[0].axis('off')

# Results
for i, r in enumerate(results):
    ax = axes[i + 1]
    boundaries = segmentation.find_boundaries(r['masks'], mode='outer')
    ax.imshow(patch, cmap='gray')
    ax.imshow(boundaries, cmap='Reds', alpha=0.7)

    diff = r['n_cells'] - n_ref
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    ax.set_title(f"{r['name']}\nn={r['n_cells']} ({diff_str})", fontsize=9)
    ax.axis('off')

# Hide unused axes
for i in range(len(results) + 1, len(axes)):
    axes[i].axis('off')

plt.suptitle(f'{sample_id}: Cellpose Parameter Sweep ({patch_size}x{patch_size})', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_cellpose_tuning.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_cellpose_tuning.png")

# Zoomed comparison of best vs reference
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))

z = 75
cy, cx = patch_size // 2, patch_size // 2
y1, y2 = cy - z, cy + z
x1, x2 = cx - z, cx + z

axes2[0].imshow(patch[y1:y2, x1:x2], cmap='gray')
axes2[0].set_title('Nuclear')
axes2[0].axis('off')

axes2[1].imshow(patch[y1:y2, x1:x2], cmap='gray')
axes2[1].imshow(boundaries_ref[y1:y2, x1:x2], cmap='Greens', alpha=0.7)
axes2[1].set_title(f'Reference: {n_ref} cells')
axes2[1].axis('off')

best_boundaries = segmentation.find_boundaries(best['masks'], mode='outer')
axes2[2].imshow(patch[y1:y2, x1:x2], cmap='gray')
axes2[2].imshow(best_boundaries[y1:y2, x1:x2], cmap='Reds', alpha=0.7)
axes2[2].set_title(f"Best Cellpose ({best['name']}): {best['n_cells']} cells")
axes2[2].axis('off')

plt.suptitle(f'{sample_id}: Best Cellpose vs Reference (zoomed)', fontsize=12)
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / f'{sample_id}_cellpose_best.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {OUTPUT_DIR}/{sample_id}_cellpose_best.png")

import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_cellpose_tuning.png')])

print("\n=== Summary ===")
print(f"{'Name':<20} {'Cells':>8} {'Diff':>8} {'Time':>8}")
print("-" * 46)
for r in sorted(results, key=lambda x: -x['n_cells']):
    diff = r['n_cells'] - n_ref
    print(f"{r['name']:<20} {r['n_cells']:>8} {diff:>+8} {r['time']:>7.1f}s")
print("-" * 46)
print(f"{'Reference':<20} {n_ref:>8}")
