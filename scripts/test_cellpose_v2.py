#!/usr/bin/env python3
"""Test Cellpose segmentation."""

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
from src.segmentation import prepare_nuclear_image, segment_cells_local_maxima

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

# Take a smaller patch for faster Cellpose testing
h, w = nuclear.shape
patch_size = 400
y0, x0 = (h - patch_size) // 2, (w - patch_size) // 2
patch = nuclear[y0:y0+patch_size, x0:x0+patch_size]

print(f"Testing on {patch_size}x{patch_size} patch...")

# Run Cellpose
print("\nRunning Cellpose...")
try:
    from cellpose import models

    t0 = time.time()
    model = models.CellposeModel(gpu=False)
    img = (patch * 255).astype(np.uint8)
    masks_cp, flows, styles = model.eval(img, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)
    t_cp = time.time() - t0
    n_cp = masks_cp.max()
    print(f"  Cellpose: {n_cp} cells in {t_cp:.1f}s")
    cellpose_ok = True
except Exception as e:
    print(f"  Cellpose failed: {e}")
    cellpose_ok = False

# Run local_maxima for comparison
print("\nRunning local_maxima...")
t0 = time.time()
masks_local = segment_cells_local_maxima(patch, sigma=1.0, min_distance=2, expansion='voronoi')
t_local = time.time() - t0
n_local = masks_local.max()
print(f"  Local maxima: {n_local} cells in {t_local:.1f}s")

# Visualization
if cellpose_ok:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Row 1
    axes[0, 0].imshow(patch, cmap='gray')
    axes[0, 0].set_title('Nuclear patch')
    axes[0, 0].axis('off')

    boundaries_cp = segmentation.find_boundaries(masks_cp, mode='outer')
    axes[0, 1].imshow(patch, cmap='gray')
    axes[0, 1].imshow(boundaries_cp, cmap='Reds', alpha=0.7)
    axes[0, 1].set_title(f'Cellpose: {n_cp} cells ({t_cp:.1f}s)')
    axes[0, 1].axis('off')

    boundaries_l = segmentation.find_boundaries(masks_local, mode='outer')
    axes[0, 2].imshow(patch, cmap='gray')
    axes[0, 2].imshow(boundaries_l, cmap='Blues', alpha=0.7)
    axes[0, 2].set_title(f'Local maxima: {n_local} cells ({t_local:.1f}s)')
    axes[0, 2].axis('off')

    # Row 2: zoomed
    z = 100
    zy, zx = patch_size//2 - z, patch_size//2 - z

    axes[1, 0].imshow(patch[zy:zy+2*z, zx:zx+2*z], cmap='gray')
    axes[1, 0].set_title('Zoomed')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(patch[zy:zy+2*z, zx:zx+2*z], cmap='gray')
    axes[1, 1].imshow(boundaries_cp[zy:zy+2*z, zx:zx+2*z], cmap='Reds', alpha=0.7)
    axes[1, 1].set_title('Cellpose (zoomed)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(patch[zy:zy+2*z, zx:zx+2*z], cmap='gray')
    axes[1, 2].imshow(boundaries_l[zy:zy+2*z, zx:zx+2*z], cmap='Blues', alpha=0.7)
    axes[1, 2].set_title('Local maxima (zoomed)')
    axes[1, 2].axis('off')

    plt.suptitle(f'{sample_id}: Cellpose vs Local Maxima ({patch_size}x{patch_size} patch)', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{sample_id}_cellpose_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_cellpose_comparison.png")

    # Open
    import subprocess
    subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_cellpose_comparison.png')])

    print("\n=== Summary ===")
    print(f"Cellpose: {n_cp} cells in {t_cp:.1f}s (deep learning)")
    print(f"Local maxima: {n_local} cells in {t_local:.1f}s (classical)")
else:
    print("\nCellpose not available, skipping comparison.")
