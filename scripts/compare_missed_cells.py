#!/usr/bin/env python3
"""Visualize where Cellpose misses cells compared to local maxima."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import segmentation
from scipy import ndimage

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import prepare_nuclear_image, segment_cells_local_maxima

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load ROI
files = list_rois(DATA_DIR)
image, markers, metadata = load_roi_txt(files[0])
sample_id = extract_sample_id(files[0].name)

# Prepare nuclear image
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])

# Take patch
h, w = nuclear.shape
patch_size = 400
y0, x0 = (h - patch_size) // 2, (w - patch_size) // 2
patch = nuclear[y0:y0+patch_size, x0:x0+patch_size]

print("Running segmentation...")

# Cellpose
from cellpose import models
model = models.CellposeModel(gpu=False)
img = (patch * 255).astype(np.uint8)
masks_cp, _, _ = model.eval(img, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)
n_cp = masks_cp.max()
print(f"Cellpose: {n_cp} cells")

# Local maxima
masks_local = segment_cells_local_maxima(patch, sigma=1.0, min_distance=2, expansion='voronoi')
n_local = masks_local.max()
print(f"Local maxima: {n_local} cells")

# Find cell centers for both
def get_centroids(masks):
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]
    centroids = ndimage.center_of_mass(masks > 0, masks, cell_ids)
    return np.array(centroids)

centroids_cp = get_centroids(masks_cp)
centroids_local = get_centroids(masks_local)

print(f"\nCellpose centroids: {len(centroids_cp)}")
print(f"Local maxima centroids: {len(centroids_local)}")

# Find which local maxima cells are "missed" by Cellpose
# A cell is missed if no Cellpose centroid is within 5 pixels
from scipy.spatial import distance

if len(centroids_cp) > 0 and len(centroids_local) > 0:
    # Distance from each local maxima centroid to nearest Cellpose centroid
    dists = distance.cdist(centroids_local, centroids_cp)
    min_dists = dists.min(axis=1)

    # Cells missed by Cellpose (no match within 5 pixels)
    missed_mask = min_dists > 5
    missed_centroids = centroids_local[missed_mask]
    matched_centroids = centroids_local[~missed_mask]

    print(f"\nMatched cells: {len(matched_centroids)}")
    print(f"Missed by Cellpose: {len(missed_centroids)}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Full patch comparison
axes[0, 0].imshow(patch, cmap='gray')
axes[0, 0].set_title(f'Nuclear ({patch_size}x{patch_size})')
axes[0, 0].axis('off')

# Cellpose
boundaries_cp = segmentation.find_boundaries(masks_cp, mode='outer')
axes[0, 1].imshow(patch, cmap='gray')
axes[0, 1].imshow(boundaries_cp, cmap='Reds', alpha=0.7)
axes[0, 1].set_title(f'Cellpose: {n_cp} cells')
axes[0, 1].axis('off')

# Local maxima
boundaries_local = segmentation.find_boundaries(masks_local, mode='outer')
axes[0, 2].imshow(patch, cmap='gray')
axes[0, 2].imshow(boundaries_local, cmap='Blues', alpha=0.7)
axes[0, 2].set_title(f'Local maxima: {n_local} cells')
axes[0, 2].axis('off')

# Row 2: Show missed cells
# Overlay showing where cells are missed
axes[1, 0].imshow(patch, cmap='gray')
# Plot matched (green) and missed (red) centroids
if len(matched_centroids) > 0:
    axes[1, 0].scatter(matched_centroids[:, 1], matched_centroids[:, 0],
                       c='green', s=5, alpha=0.5, label=f'Matched ({len(matched_centroids)})')
if len(missed_centroids) > 0:
    axes[1, 0].scatter(missed_centroids[:, 1], missed_centroids[:, 0],
                       c='red', s=10, alpha=0.8, label=f'Missed ({len(missed_centroids)})')
axes[1, 0].legend(loc='upper right', fontsize=8)
axes[1, 0].set_title('Cells missed by Cellpose (red)')
axes[1, 0].axis('off')

# Zoomed region showing missed cells
z = 80
# Find a region with missed cells
if len(missed_centroids) > 0:
    # Pick region around first missed cell
    cy, cx = int(missed_centroids[0][0]), int(missed_centroids[0][1])
    cy = max(z, min(patch_size - z, cy))
    cx = max(z, min(patch_size - z, cx))
else:
    cy, cx = patch_size // 2, patch_size // 2

y1, y2 = cy - z, cy + z
x1, x2 = cx - z, cx + z

axes[1, 1].imshow(patch[y1:y2, x1:x2], cmap='gray')
axes[1, 1].imshow(boundaries_cp[y1:y2, x1:x2], cmap='Reds', alpha=0.7)
# Show missed cells in this region
missed_in_region = missed_centroids[
    (missed_centroids[:, 0] >= y1) & (missed_centroids[:, 0] < y2) &
    (missed_centroids[:, 1] >= x1) & (missed_centroids[:, 1] < x2)
]
if len(missed_in_region) > 0:
    axes[1, 1].scatter(missed_in_region[:, 1] - x1, missed_in_region[:, 0] - y1,
                       c='yellow', s=50, marker='o', facecolors='none', linewidths=2)
axes[1, 1].set_title(f'Cellpose (zoomed) - circles = missed')
axes[1, 1].axis('off')

axes[1, 2].imshow(patch[y1:y2, x1:x2], cmap='gray')
axes[1, 2].imshow(boundaries_local[y1:y2, x1:x2], cmap='Blues', alpha=0.7)
axes[1, 2].set_title('Local maxima (zoomed)')
axes[1, 2].axis('off')

plt.suptitle(f'{sample_id}: Cellpose misses {len(missed_centroids)} cells ({len(missed_centroids)/n_local*100:.0f}%)', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_missed_cells.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_missed_cells.png")

import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_missed_cells.png')])

print(f"\n=== Analysis ===")
print(f"Cellpose detects {n_cp} cells")
print(f"Local maxima detects {n_local} cells")
print(f"Cellpose misses ~{len(missed_centroids)} cells ({len(missed_centroids)/n_local*100:.0f}%)")
print(f"\nThis suggests Cellpose may be under-segmenting densely packed cells.")
