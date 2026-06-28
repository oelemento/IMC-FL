#!/usr/bin/env python3
"""Test StarDist segmentation from ElementoLab IMC package."""

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
from src.segmentation import prepare_nuclear_image

# Import StarDist from the imc package
sys.path.insert(0, '/Users/ole2001/PROGRAMS/imc-pipeline')
from imc.segmentation import stardist_segment_nuclei, normalize

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load ROI
files = list_rois(DATA_DIR)
image, markers, metadata = load_roi_txt(files[0])
sample_id = extract_sample_id(files[0].name)

print(f"Sample: {sample_id}")
print(f"Image: {image.shape}")

# Prepare nuclear image
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])
print(f"Nuclear image: {nuclear.shape}, range [{nuclear.min():.2f}, {nuclear.max():.2f}]")

# Normalize for StarDist (expects 0-1 range with histogram equalization)
nuclear_norm = normalize(nuclear)
print(f"Normalized: range [{nuclear_norm.min():.2f}, {nuclear_norm.max():.2f}]")

# Run StarDist segmentation
print("\nRunning StarDist segmentation...")
t0 = time.time()
masks_stardist = stardist_segment_nuclei(nuclear_norm)
t_stardist = time.time() - t0
n_stardist = masks_stardist.max()
print(f"  StarDist: {n_stardist} cells in {t_stardist:.1f}s")

# Compare with our local_maxima method
from src.segmentation import segment_cells_local_maxima

print("\nRunning local_maxima for comparison...")
t0 = time.time()
masks_local = segment_cells_local_maxima(nuclear, sigma=1.0, min_distance=2, expansion='voronoi')
t_local = time.time() - t0
n_local = masks_local.max()
print(f"  Local maxima: {n_local} cells in {t_local:.1f}s")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Full images
axes[0, 0].imshow(nuclear, cmap='gray')
axes[0, 0].set_title('Nuclear (DNA1+DNA2)')
axes[0, 0].axis('off')

boundaries_s = segmentation.find_boundaries(masks_stardist, mode='outer')
axes[0, 1].imshow(nuclear, cmap='gray')
axes[0, 1].imshow(boundaries_s, cmap='Reds', alpha=0.7)
axes[0, 1].set_title(f'StarDist: {n_stardist} cells ({t_stardist:.1f}s)')
axes[0, 1].axis('off')

boundaries_l = segmentation.find_boundaries(masks_local, mode='outer')
axes[0, 2].imshow(nuclear, cmap='gray')
axes[0, 2].imshow(boundaries_l, cmap='Blues', alpha=0.7)
axes[0, 2].set_title(f'Local maxima: {n_local} cells ({t_local:.1f}s)')
axes[0, 2].axis('off')

# Row 2: Zoomed center
h, w = nuclear.shape
y1, y2 = h//3, 2*h//3
x1, x2 = w//3, 2*w//3

axes[1, 0].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
axes[1, 0].set_title('Zoomed center')
axes[1, 0].axis('off')

axes[1, 1].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
axes[1, 1].imshow(boundaries_s[y1:y2, x1:x2], cmap='Reds', alpha=0.7)
axes[1, 1].set_title('StarDist (zoomed)')
axes[1, 1].axis('off')

axes[1, 2].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
axes[1, 2].imshow(boundaries_l[y1:y2, x1:x2], cmap='Blues', alpha=0.7)
axes[1, 2].set_title('Local maxima (zoomed)')
axes[1, 2].axis('off')

plt.suptitle(f'{sample_id}: StarDist vs Local Maxima Segmentation', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_stardist_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_stardist_comparison.png")

# Open result
import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_stardist_comparison.png')])

print("\n=== Summary ===")
print(f"StarDist: {n_stardist} cells in {t_stardist:.1f}s (deep learning)")
print(f"Local maxima: {n_local} cells in {t_local:.1f}s (classical)")
print(f"Difference: {abs(n_stardist - n_local)} cells ({(n_stardist/n_local - 1)*100:+.1f}%)")
