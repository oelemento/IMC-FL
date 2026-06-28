#!/usr/bin/env python3
"""Tune watershed parameters."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from skimage import segmentation

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import prepare_nuclear_image, segment_cells_watershed

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load first ROI
files = list_rois(DATA_DIR)
roi_file = files[0]
sample_id = extract_sample_id(roi_file.name)

print(f"Loading {sample_id}...")
image, markers, metadata = load_roi_txt(roi_file)

# Prepare nuclear image
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])

# Test different min_distance values
distances = [3, 5, 7, 10]

fig, axes = plt.subplots(2, len(distances), figsize=(16, 8))

for i, dist in enumerate(distances):
    print(f"Testing min_distance={dist}...")
    masks = segment_cells_watershed(nuclear, min_distance=dist, min_size=30)
    n_cells = masks.max()

    # Get cell areas
    areas = []
    for cell_id in range(1, n_cells + 1):
        areas.append((masks == cell_id).sum())
    areas = np.array(areas) if areas else np.array([0])

    # Top row: full image with boundaries
    boundaries = segmentation.find_boundaries(masks, mode='outer')
    axes[0, i].imshow(nuclear, cmap='gray')
    axes[0, i].imshow(boundaries, cmap='Reds', alpha=0.7)
    axes[0, i].set_title(f'd={dist}, n={n_cells}', fontsize=11)
    axes[0, i].axis('off')

    # Bottom row: zoomed center
    h, w = nuclear.shape
    y1, y2 = h//3, 2*h//3
    x1, x2 = w//3, 2*w//3

    axes[1, i].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
    axes[1, i].imshow(boundaries[y1:y2, x1:x2], cmap='Reds', alpha=0.7)
    median_area = np.median(areas) if len(areas) > 0 else 0
    axes[1, i].set_title(f'median area: {median_area:.0f}px', fontsize=10)
    axes[1, i].axis('off')

plt.suptitle(f'{sample_id}: Watershed Parameter Sweep', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_watershed_sweep.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_watershed_sweep.png")

# Open result
import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_watershed_sweep.png')])

print("\nLook at the segmentation overlay:")
print("- Too few cells / merged cells → decrease min_distance")
print("- Too many cells / over-segmented → increase min_distance")
