#!/usr/bin/env python3
"""Cross-reference nuclear cell centers with Cellpose segmentation."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import segmentation
from skimage.feature import peak_local_max
from scipy import ndimage

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import prepare_nuclear_image

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load ROI
files = list_rois(DATA_DIR)
image, markers, metadata = load_roi_txt(files[0])
sample_id = extract_sample_id(files[0].name)

# Prepare nuclear image
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])
h, w = nuclear.shape
patch_size = 300
y0, x0 = (h - patch_size) // 2, (w - patch_size) // 2
patch = nuclear[y0:y0+patch_size, x0:x0+patch_size]

print(f"Sample: {sample_id}")
print(f"Patch: {patch_size}x{patch_size}")

# Detect nuclei centers using local maxima on DNA signal
smoothed = ndimage.gaussian_filter(patch, sigma=1.0)
nuclei_coords = peak_local_max(smoothed, min_distance=2, threshold_rel=0.05)
n_nuclei = len(nuclei_coords)
print(f"\nNuclei detected (DNA signal): {n_nuclei}")

# Run Cellpose with tuned parameters
from cellpose import models

print("\nRunning Cellpose (flow=0.8)...")
model = models.CellposeModel(gpu=False)
img = (patch * 255).astype(np.uint8)
masks_cp, _, _ = model.eval(img, diameter=None, flow_threshold=0.8, cellprob_threshold=0.0)
n_cellpose = masks_cp.max()
print(f"Cellpose cells: {n_cellpose}")

# Cross-reference: which nuclei are captured by which Cellpose cells?
nuclei_to_cell = {}  # nuclei index -> cell ID
cell_to_nuclei = {}  # cell ID -> list of nuclei indices

for i, (y, x) in enumerate(nuclei_coords):
    cell_id = masks_cp[y, x]
    nuclei_to_cell[i] = cell_id

    if cell_id > 0:
        if cell_id not in cell_to_nuclei:
            cell_to_nuclei[cell_id] = []
        cell_to_nuclei[cell_id].append(i)

# Analysis
nuclei_captured = sum(1 for cell_id in nuclei_to_cell.values() if cell_id > 0)
nuclei_missed = n_nuclei - nuclei_captured

cells_with_nuclei = len(cell_to_nuclei)
cells_without_nuclei = n_cellpose - cells_with_nuclei

# Cells with multiple nuclei (potential under-segmentation in Cellpose)
cells_multi_nuclei = sum(1 for nuclei_list in cell_to_nuclei.values() if len(nuclei_list) > 1)

print(f"\n=== Cross-Reference Analysis ===")
print(f"Nuclei (DNA peaks): {n_nuclei}")
print(f"Cellpose cells: {n_cellpose}")
print(f"\nNuclei captured by Cellpose: {nuclei_captured} ({nuclei_captured/n_nuclei*100:.1f}%)")
print(f"Nuclei missed (in background): {nuclei_missed} ({nuclei_missed/n_nuclei*100:.1f}%)")
print(f"\nCellpose cells with ≥1 nucleus: {cells_with_nuclei}")
print(f"Cellpose cells with 0 nuclei: {cells_without_nuclei} (potential over-segmentation)")
print(f"Cellpose cells with >1 nuclei: {cells_multi_nuclei} (potential under-segmentation)")

# Detailed breakdown
print(f"\n=== Nuclei per Cellpose cell ===")
nuclei_counts = [len(v) for v in cell_to_nuclei.values()]
for count in sorted(set(nuclei_counts), reverse=True):
    n = sum(1 for c in nuclei_counts if c == count)
    print(f"  {count} nuclei: {n} cells")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Overview
# Nuclear with detected centers
axes[0, 0].imshow(patch, cmap='gray')
axes[0, 0].scatter(nuclei_coords[:, 1], nuclei_coords[:, 0], c='red', s=3, alpha=0.7)
axes[0, 0].set_title(f'Nuclei centers (DNA): {n_nuclei}')
axes[0, 0].axis('off')

# Cellpose boundaries
boundaries = segmentation.find_boundaries(masks_cp, mode='outer')
axes[0, 1].imshow(patch, cmap='gray')
axes[0, 1].imshow(boundaries, cmap='Blues', alpha=0.7)
axes[0, 1].set_title(f'Cellpose (flow=0.8): {n_cellpose} cells')
axes[0, 1].axis('off')

# Overlay: nuclei on Cellpose
axes[0, 2].imshow(patch, cmap='gray')
axes[0, 2].imshow(boundaries, cmap='Blues', alpha=0.5)
axes[0, 2].scatter(nuclei_coords[:, 1], nuclei_coords[:, 0], c='red', s=3, alpha=0.7)
axes[0, 2].set_title(f'Overlay: {nuclei_captured}/{n_nuclei} nuclei captured')
axes[0, 2].axis('off')

# Row 2: Problem cases
# Missed nuclei (in background)
missed_nuclei = np.array([nuclei_coords[i] for i, cell_id in nuclei_to_cell.items() if cell_id == 0])
axes[1, 0].imshow(patch, cmap='gray')
axes[1, 0].imshow(boundaries, cmap='Blues', alpha=0.5)
if len(missed_nuclei) > 0:
    axes[1, 0].scatter(missed_nuclei[:, 1], missed_nuclei[:, 0], c='red', s=20, marker='x')
axes[1, 0].set_title(f'Missed nuclei: {nuclei_missed} (red X)')
axes[1, 0].axis('off')

# Cells with multiple nuclei (under-segmentation)
multi_nuclei_cells = [cell_id for cell_id, nuclei_list in cell_to_nuclei.items() if len(nuclei_list) > 1]
multi_mask = np.isin(masks_cp, multi_nuclei_cells)
axes[1, 1].imshow(patch, cmap='gray')
axes[1, 1].imshow(multi_mask, cmap='Reds', alpha=0.5)
# Show the nuclei in these cells
multi_nuclei_coords = []
for cell_id in multi_nuclei_cells:
    for ni in cell_to_nuclei[cell_id]:
        multi_nuclei_coords.append(nuclei_coords[ni])
if multi_nuclei_coords:
    multi_nuclei_coords = np.array(multi_nuclei_coords)
    axes[1, 1].scatter(multi_nuclei_coords[:, 1], multi_nuclei_coords[:, 0], c='yellow', s=10)
axes[1, 1].set_title(f'Cells with >1 nucleus: {cells_multi_nuclei} (under-seg)')
axes[1, 1].axis('off')

# Cells without nuclei (potential over-segmentation)
all_cells = set(range(1, n_cellpose + 1))
cells_with = set(cell_to_nuclei.keys())
cells_without = list(all_cells - cells_with)
no_nuclei_mask = np.isin(masks_cp, cells_without)
axes[1, 2].imshow(patch, cmap='gray')
axes[1, 2].imshow(no_nuclei_mask, cmap='Greens', alpha=0.5)
axes[1, 2].set_title(f'Cells with 0 nuclei: {cells_without_nuclei} (over-seg?)')
axes[1, 2].axis('off')

plt.suptitle(f'{sample_id}: Nuclei vs Cellpose Cross-Reference', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_crossref.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_crossref.png")

import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_crossref.png')])

# Summary
print("\n=== Summary ===")
print(f"Nuclear detection found {n_nuclei} cell centers")
print(f"Cellpose found {n_cellpose} cells")
print(f"  - {nuclei_captured} cells match a nucleus ({nuclei_captured/n_nuclei*100:.1f}%)")
print(f"  - {cells_without_nuclei} Cellpose cells have no nucleus (over-segmentation)")
print(f"  - {cells_multi_nuclei} Cellpose cells have >1 nucleus (under-segmentation)")
print(f"  - {nuclei_missed} nuclei are missed by Cellpose")
