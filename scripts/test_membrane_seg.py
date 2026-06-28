#!/usr/bin/env python3
"""Compare membrane-guided vs voronoi segmentation."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import segmentation
import time

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import (
    prepare_nuclear_image,
    prepare_membrane_image,
    segment_cells_local_maxima,
    segment_cells_membrane,
    segment_roi,
)

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load ROI
files = list_rois(DATA_DIR)
image, markers, metadata = load_roi_txt(files[0])
sample_id = extract_sample_id(files[0].name)

print(f"Sample: {sample_id}")
print(f"Image: {image.shape}")

# Prepare images
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])
membrane = prepare_membrane_image(image, markers, ['CD45RO', 'CD3', 'CD20'])

print(f"\nNuclear image: {nuclear.shape}, range [{nuclear.min():.2f}, {nuclear.max():.2f}]")
print(f"Membrane image: {membrane.shape}, range [{membrane.min():.2f}, {membrane.max():.2f}]")

# Check which membrane markers were found
found_markers = []
for m in ['CD45RO', 'CD3', 'CD20', 'CD38', 'CD4', 'CD8a']:
    if m in markers:
        found_markers.append(m)
print(f"Available membrane markers: {found_markers}")

# Compare segmentation methods
print("\nRunning segmentation comparison...")

# Method 1: Voronoi (current best)
t0 = time.time()
masks_voronoi = segment_cells_local_maxima(nuclear, sigma=1.0, min_distance=2, expansion='voronoi')
t_voronoi = time.time() - t0
n_voronoi = masks_voronoi.max()
print(f"  Voronoi: {n_voronoi} cells in {t_voronoi:.1f}s")

# Method 2: Membrane-guided
t0 = time.time()
masks_membrane = segment_cells_membrane(nuclear, membrane, sigma=1.0, min_distance=2, membrane_weight=0.7)
t_membrane = time.time() - t0
n_membrane = masks_membrane.max()
print(f"  Membrane: {n_membrane} cells in {t_membrane:.1f}s")

# Also try different membrane weights
membrane_weights = [0.3, 0.5, 0.7, 0.9]
masks_weights = {}
for w in membrane_weights:
    masks_weights[w] = segment_cells_membrane(nuclear, membrane, sigma=1.0, min_distance=2, membrane_weight=w)

# Visualize comparison
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Full images
# Nuclear
axes[0, 0].imshow(nuclear, cmap='gray')
axes[0, 0].set_title('Nuclear (DNA1+DNA2)')
axes[0, 0].axis('off')

# Membrane
axes[0, 1].imshow(membrane, cmap='hot')
axes[0, 1].set_title('Membrane (CD45RO+CD3+CD20)')
axes[0, 1].axis('off')

# Voronoi boundaries
boundaries_v = segmentation.find_boundaries(masks_voronoi, mode='outer')
axes[0, 2].imshow(nuclear, cmap='gray')
axes[0, 2].imshow(boundaries_v, cmap='Reds', alpha=0.7)
axes[0, 2].set_title(f'Voronoi: {n_voronoi} cells')
axes[0, 2].axis('off')

# Membrane boundaries
boundaries_m = segmentation.find_boundaries(masks_membrane, mode='outer')
axes[0, 3].imshow(nuclear, cmap='gray')
axes[0, 3].imshow(boundaries_m, cmap='Blues', alpha=0.7)
axes[0, 3].set_title(f'Membrane: {n_membrane} cells')
axes[0, 3].axis('off')

# Row 2: Zoomed center region
h, w = nuclear.shape
y1, y2 = h//3, 2*h//3
x1, x2 = w//3, 2*w//3

axes[1, 0].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
axes[1, 0].set_title('Nuclear (zoomed)')
axes[1, 0].axis('off')

axes[1, 1].imshow(membrane[y1:y2, x1:x2], cmap='hot')
axes[1, 1].set_title('Membrane (zoomed)')
axes[1, 1].axis('off')

axes[1, 2].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
axes[1, 2].imshow(boundaries_v[y1:y2, x1:x2], cmap='Reds', alpha=0.7)
axes[1, 2].set_title('Voronoi (zoomed)')
axes[1, 2].axis('off')

axes[1, 3].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
axes[1, 3].imshow(boundaries_m[y1:y2, x1:x2], cmap='Blues', alpha=0.7)
axes[1, 3].set_title('Membrane (zoomed)')
axes[1, 3].axis('off')

plt.suptitle(f'{sample_id}: Voronoi vs Membrane-guided Segmentation', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_membrane_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_membrane_comparison.png")

# Also create a figure showing membrane weight sweep
fig2, axes2 = plt.subplots(2, len(membrane_weights), figsize=(4*len(membrane_weights), 8))

for i, w in enumerate(membrane_weights):
    masks = masks_weights[w]
    n_cells = masks.max()
    boundaries = segmentation.find_boundaries(masks, mode='outer')

    # Full
    axes2[0, i].imshow(nuclear, cmap='gray')
    axes2[0, i].imshow(boundaries, cmap='Greens', alpha=0.7)
    axes2[0, i].set_title(f'w={w}, n={n_cells}')
    axes2[0, i].axis('off')

    # Zoomed
    axes2[1, i].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
    axes2[1, i].imshow(boundaries[y1:y2, x1:x2], cmap='Greens', alpha=0.7)
    axes2[1, i].set_title('Zoomed')
    axes2[1, i].axis('off')

plt.suptitle(f'{sample_id}: Membrane Weight Sweep (0=nuclear only, 1=membrane only)', fontsize=12)
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / f'{sample_id}_membrane_weights.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {OUTPUT_DIR}/{sample_id}_membrane_weights.png")

# Open results
import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_membrane_comparison.png')])

print("\n=== Summary ===")
print(f"Both methods detect same cell centers (from nuclear signal)")
print(f"Difference is in boundary placement:")
print(f"  - Voronoi: boundaries at equal distance between cell centers")
print(f"  - Membrane: boundaries follow membrane marker signal")
print(f"\nHigher membrane_weight (0.7-0.9) = boundaries follow membrane more closely")
print(f"Lower membrane_weight (0.3-0.5) = smoother boundaries, less affected by membrane noise")
