#!/usr/bin/env python3
"""Hybrid segmentation: Cellpose boundaries filtered by nuclear detection."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import segmentation, measure
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


def hybrid_segmentation(nuclear_img, flow_threshold=0.8, sigma=1.0, min_distance=2):
    """
    Hybrid segmentation approach:
    1. Detect nuclei from DNA signal (local maxima)
    2. Run Cellpose for cell boundaries
    3. Keep only Cellpose cells that contain ≥1 nucleus

    Returns:
        filtered_masks: Cellpose masks filtered to only cells with nuclei
        nuclei_coords: detected nuclei coordinates
        stats: dict with statistics
    """
    from cellpose import models

    # Step 1: Detect nuclei from DNA signal
    smoothed = ndimage.gaussian_filter(nuclear_img, sigma=sigma)
    nuclei_coords = peak_local_max(smoothed, min_distance=min_distance, threshold_rel=0.05)
    n_nuclei = len(nuclei_coords)

    # Step 2: Run Cellpose
    model = models.CellposeModel(gpu=False)
    img = (nuclear_img * 255).astype(np.uint8)
    masks_cp, _, _ = model.eval(img, diameter=None, flow_threshold=flow_threshold, cellprob_threshold=0.0)
    n_cellpose = masks_cp.max()

    # Step 3: Find which cells contain nuclei
    cells_with_nuclei = set()
    for y, x in nuclei_coords:
        cell_id = masks_cp[y, x]
        if cell_id > 0:
            cells_with_nuclei.add(cell_id)

    # Step 4: Filter masks - keep only cells with nuclei
    filtered_masks = np.zeros_like(masks_cp)
    new_id = 1
    for old_id in sorted(cells_with_nuclei):
        filtered_masks[masks_cp == old_id] = new_id
        new_id += 1

    n_filtered = filtered_masks.max()
    n_removed = n_cellpose - len(cells_with_nuclei)

    stats = {
        'n_nuclei': n_nuclei,
        'n_cellpose_raw': n_cellpose,
        'n_cells_with_nuclei': len(cells_with_nuclei),
        'n_removed': n_removed,
        'n_final': n_filtered,
    }

    return filtered_masks, nuclei_coords, masks_cp, stats


print("\nRunning hybrid segmentation...")
filtered_masks, nuclei_coords, raw_masks, stats = hybrid_segmentation(patch)

print(f"\n=== Results ===")
print(f"Nuclei detected (DNA): {stats['n_nuclei']}")
print(f"Cellpose raw: {stats['n_cellpose_raw']}")
print(f"Cells with nuclei: {stats['n_cells_with_nuclei']}")
print(f"Cells removed (no nucleus): {stats['n_removed']}")
print(f"Final cell count: {stats['n_final']}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: The process
# Nuclei detection
axes[0, 0].imshow(patch, cmap='gray')
axes[0, 0].scatter(nuclei_coords[:, 1], nuclei_coords[:, 0], c='red', s=5, alpha=0.7)
axes[0, 0].set_title(f'1. Nuclei detection\n{stats["n_nuclei"]} nuclei')
axes[0, 0].axis('off')

# Raw Cellpose
boundaries_raw = segmentation.find_boundaries(raw_masks, mode='outer')
axes[0, 1].imshow(patch, cmap='gray')
axes[0, 1].imshow(boundaries_raw, cmap='Reds', alpha=0.7)
axes[0, 1].set_title(f'2. Cellpose raw\n{stats["n_cellpose_raw"]} cells')
axes[0, 1].axis('off')

# Filtered result
boundaries_filtered = segmentation.find_boundaries(filtered_masks, mode='outer')
axes[0, 2].imshow(patch, cmap='gray')
axes[0, 2].imshow(boundaries_filtered, cmap='Greens', alpha=0.7)
axes[0, 2].set_title(f'3. Filtered (has nucleus)\n{stats["n_final"]} cells')
axes[0, 2].axis('off')

# Row 2: Comparison
# What was removed
removed_mask = (raw_masks > 0) & (filtered_masks == 0)
axes[1, 0].imshow(patch, cmap='gray')
axes[1, 0].imshow(removed_mask, cmap='Reds', alpha=0.5)
axes[1, 0].set_title(f'Removed cells (no nucleus)\n{stats["n_removed"]} cells')
axes[1, 0].axis('off')

# Final with nuclei overlay
axes[1, 1].imshow(patch, cmap='gray')
axes[1, 1].imshow(boundaries_filtered, cmap='Greens', alpha=0.5)
axes[1, 1].scatter(nuclei_coords[:, 1], nuclei_coords[:, 0], c='red', s=3, alpha=0.7)
axes[1, 1].set_title(f'Final: boundaries + nuclei')
axes[1, 1].axis('off')

# Zoomed view
z = 75
cy, cx = patch_size // 2, patch_size // 2
y1, y2 = cy - z, cy + z
x1, x2 = cx - z, cx + z

axes[1, 2].imshow(patch[y1:y2, x1:x2], cmap='gray')
axes[1, 2].imshow(boundaries_filtered[y1:y2, x1:x2], cmap='Greens', alpha=0.5)
# Nuclei in this region
nuclei_in_region = nuclei_coords[
    (nuclei_coords[:, 0] >= y1) & (nuclei_coords[:, 0] < y2) &
    (nuclei_coords[:, 1] >= x1) & (nuclei_coords[:, 1] < x2)
]
axes[1, 2].scatter(nuclei_in_region[:, 1] - x1, nuclei_in_region[:, 0] - y1, c='red', s=10, alpha=0.7)
axes[1, 2].set_title('Zoomed: each cell has a nucleus')
axes[1, 2].axis('off')

plt.suptitle(f'{sample_id}: Hybrid Segmentation (Cellpose + Nuclear Filter)', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_hybrid_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_hybrid_segmentation.png")

import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_hybrid_segmentation.png')])

print("\n=== Summary ===")
print(f"Hybrid approach: Cellpose boundaries + nuclear validation")
print(f"  - Detect nuclei from DNA signal: {stats['n_nuclei']}")
print(f"  - Run Cellpose (flow=0.8): {stats['n_cellpose_raw']} cells")
print(f"  - Filter: keep cells with ≥1 nucleus")
print(f"  - Final: {stats['n_final']} validated cells")
print(f"  - Removed {stats['n_removed']} spurious cells ({stats['n_removed']/stats['n_cellpose_raw']*100:.1f}%)")
