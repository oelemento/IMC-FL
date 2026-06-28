#!/usr/bin/env python3
"""Fast segmentation testing on a small patch."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import segmentation, filters, morphology, measure
from scipy import ndimage

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import prepare_nuclear_image

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load ROI
files = list_rois(DATA_DIR)
roi_file = files[0]
sample_id = extract_sample_id(roi_file.name)

print(f"Loading {sample_id}...")
image, markers, metadata = load_roi_txt(roi_file)

# Prepare nuclear image
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])
print(f"Full image: {nuclear.shape}")

# Crop a 300x300 patch from center
h, w = nuclear.shape
patch_size = 300
y0 = (h - patch_size) // 2
x0 = (w - patch_size) // 2
patch = nuclear[y0:y0+patch_size, x0:x0+patch_size]
print(f"Patch: {patch.shape} from center")

def segment_watershed_v2(img, min_distance=5, threshold_factor=0.5, min_size=20):
    """Improved watershed with adjustable threshold."""
    from skimage.feature import peak_local_max

    # Smooth less aggressively
    smoothed = ndimage.gaussian_filter(img, sigma=0.5)

    # Use lower threshold to capture more cells
    thresh = filters.threshold_otsu(smoothed) * threshold_factor
    binary = smoothed > thresh

    # Clean up
    binary = morphology.remove_small_objects(binary, min_size=min_size)

    # Distance transform
    distance = ndimage.distance_transform_edt(binary)

    # Find peaks with smaller min_distance
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)

    # Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)
    labels = measure.label(labels > 0)

    return labels, binary, distance

# Test different parameters
print("\nTesting parameters on patch...")

params = [
    {'min_distance': 3, 'threshold_factor': 0.3},
    {'min_distance': 3, 'threshold_factor': 0.5},
    {'min_distance': 5, 'threshold_factor': 0.3},
    {'min_distance': 5, 'threshold_factor': 0.5},
    {'min_distance': 3, 'threshold_factor': 0.2},
    {'min_distance': 2, 'threshold_factor': 0.3},
]

fig, axes = plt.subplots(3, len(params), figsize=(3*len(params), 9))

for i, p in enumerate(params):
    print(f"  Testing d={p['min_distance']}, t={p['threshold_factor']}...")
    masks, binary, distance = segment_watershed_v2(patch, **p)
    n_cells = masks.max()

    # Row 1: Nuclear with boundaries
    boundaries = segmentation.find_boundaries(masks, mode='outer')
    axes[0, i].imshow(patch, cmap='gray')
    axes[0, i].imshow(boundaries, cmap='Reds', alpha=0.8)
    axes[0, i].set_title(f'd={p["min_distance"]}, t={p["threshold_factor"]}\nn={n_cells}', fontsize=9)
    axes[0, i].axis('off')

    # Row 2: Binary mask (what's being segmented)
    axes[1, i].imshow(binary, cmap='gray')
    axes[1, i].set_title('Binary mask', fontsize=9)
    axes[1, i].axis('off')

    # Row 3: Colored cells
    np.random.seed(42)
    colors = np.random.rand(n_cells + 1, 3)
    colors[0] = [0, 0, 0]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    axes[2, i].imshow(masks, cmap=cmap, interpolation='nearest')
    axes[2, i].set_title(f'Cell labels', fontsize=9)
    axes[2, i].axis('off')

plt.suptitle(f'{sample_id}: 300x300 patch segmentation tests', fontsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_patch_test.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_patch_test.png")

# Also show the patch location on full image
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(nuclear, cmap='gray')
rect = plt.Rectangle((x0, y0), patch_size, patch_size,
                       linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.set_title(f'{sample_id}: Patch location (red box)')
ax.axis('off')
fig.savefig(OUTPUT_DIR / f'{sample_id}_patch_location.png', dpi=100, bbox_inches='tight')
plt.close()

# Open results
import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_patch_test.png')])

print("\nLook at the patch test:")
print("- Row 1: Nuclear image with cell boundaries (red)")
print("- Row 2: Binary mask (what counts as 'cell')")
print("- Row 3: Individual cells colored")
print("\nAdjust 'threshold_factor' (lower = more tissue detected)")
print("Adjust 'min_distance' (lower = more cells separated)")
