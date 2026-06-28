#!/usr/bin/env python3
"""Segmentation for densely packed cells (like lymphoma)."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import segmentation, filters, morphology, measure
from skimage.feature import blob_log, peak_local_max
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

# Crop patch
h, w = nuclear.shape
patch_size = 300
y0, x0 = (h - patch_size) // 2, (w - patch_size) // 2
patch = nuclear[y0:y0+patch_size, x0:x0+patch_size]
print(f"Patch: {patch.shape}")


def segment_local_maxima(img, sigma=2, min_distance=3, threshold_rel=0.1):
    """Find cells as local maxima after Gaussian smoothing."""
    # Smooth to reduce noise
    smoothed = ndimage.gaussian_filter(img, sigma=sigma)

    # Find local maxima
    coords = peak_local_max(smoothed, min_distance=min_distance,
                            threshold_rel=threshold_rel)

    # Create markers
    markers = np.zeros(img.shape, dtype=int)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i

    # Expand markers using watershed on inverted image
    # Use the smoothed image gradient as the "landscape"
    gradient = filters.sobel(smoothed)
    labels = segmentation.watershed(gradient, markers)

    return labels, len(coords)


def segment_blob_log(img, min_sigma=2, max_sigma=6, threshold=0.02):
    """Detect cells as blobs using Laplacian of Gaussian."""
    # Detect blobs
    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                     threshold=threshold, overlap=0.5)

    # blobs format: (y, x, sigma)
    # Create markers at blob centers
    markers = np.zeros(img.shape, dtype=int)
    for i, (y, x, sigma) in enumerate(blobs, 1):
        y, x = int(y), int(x)
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            markers[y, x] = i

    # Expand using watershed
    gradient = filters.sobel(ndimage.gaussian_filter(img, sigma=1))
    labels = segmentation.watershed(gradient, markers)

    return labels, len(blobs)


def segment_h_maxima(img, h=0.05, min_distance=3):
    """Use h-maxima transform to find cell centers."""
    from skimage.morphology import h_maxima, local_maxima

    # Smooth
    smoothed = ndimage.gaussian_filter(img, sigma=1)

    # h-maxima suppresses all maxima below height h
    h_max = h_maxima(smoothed, h=h)

    # Find remaining peaks
    coords = peak_local_max(h_max, min_distance=min_distance)

    # Create markers
    markers = np.zeros(img.shape, dtype=int)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i

    # Watershed
    gradient = filters.sobel(smoothed)
    labels = segmentation.watershed(gradient, markers)

    return labels, len(coords)


# Test different methods
print("Testing segmentation methods on patch...")

methods = [
    ('Local maxima\nσ=2, d=3', lambda p: segment_local_maxima(p, sigma=2, min_distance=3)),
    ('Local maxima\nσ=1.5, d=2', lambda p: segment_local_maxima(p, sigma=1.5, min_distance=2)),
    ('Local maxima\nσ=1, d=2', lambda p: segment_local_maxima(p, sigma=1, min_distance=2)),
    ('Blob LoG\nσ=2-5', lambda p: segment_blob_log(p, min_sigma=2, max_sigma=5, threshold=0.01)),
    ('Blob LoG\nσ=1-4', lambda p: segment_blob_log(p, min_sigma=1, max_sigma=4, threshold=0.01)),
    ('H-maxima\nh=0.03, d=2', lambda p: segment_h_maxima(p, h=0.03, min_distance=2)),
]

fig, axes = plt.subplots(2, len(methods), figsize=(3*len(methods), 6))

for i, (name, func) in enumerate(methods):
    print(f"  {name.replace(chr(10), ' ')}...")
    labels, n_cells = func(patch)

    # Top: boundaries overlay
    boundaries = segmentation.find_boundaries(labels, mode='outer')
    axes[0, i].imshow(patch, cmap='gray')
    axes[0, i].imshow(boundaries, cmap='Reds', alpha=0.8)
    axes[0, i].set_title(f'{name}\nn={n_cells}', fontsize=9)
    axes[0, i].axis('off')

    # Bottom: colored labels
    np.random.seed(42)
    colors = np.random.rand(labels.max() + 1, 3)
    colors[0] = [0, 0, 0]
    from matplotlib.colors import ListedColormap
    axes[1, i].imshow(labels, cmap=ListedColormap(colors), interpolation='nearest')
    axes[1, i].axis('off')

plt.suptitle(f'{sample_id}: Dense tissue segmentation (300x300 patch)', fontsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_dense_test.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUT_DIR}/{sample_id}_dense_test.png")

# Open
import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_dense_test.png')])

print("\nMethods tested:")
print("- Local maxima: finds bright peaks after smoothing")
print("- Blob LoG: Laplacian of Gaussian blob detection")
print("- H-maxima: suppresses shallow peaks, keeps strong ones")
