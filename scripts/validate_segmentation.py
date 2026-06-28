#!/usr/bin/env python3
"""Validate segmentation quality: membrane vs voronoi."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import segmentation, measure

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import (
    prepare_nuclear_image,
    prepare_membrane_image,
    segment_cells_local_maxima,
    segment_cells_membrane,
)

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load ROI
files = list_rois(DATA_DIR)
image, markers, metadata = load_roi_txt(files[0])
sample_id = extract_sample_id(files[0].name)

print(f"Sample: {sample_id}")

# Prepare images
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])
membrane = prepare_membrane_image(image, markers, ['CD45RO', 'CD3', 'CD20'])

# Get individual membrane channels for analysis
cd45ro_idx = markers.index('CD45RO')
cd3_idx = markers.index('CD3')
cd20_idx = markers.index('CD20')

cd45ro = image[:, :, cd45ro_idx].astype(np.float32)
cd3 = image[:, :, cd3_idx].astype(np.float32)
cd20 = image[:, :, cd20_idx].astype(np.float32)

# Run both segmentation methods
print("\nRunning segmentation...")
masks_voronoi = segment_cells_local_maxima(nuclear, sigma=1.0, min_distance=2, expansion='voronoi')
masks_membrane = segment_cells_membrane(nuclear, membrane, sigma=1.0, min_distance=2, membrane_weight=0.7)

print(f"  Voronoi: {masks_voronoi.max()} cells")
print(f"  Membrane: {masks_membrane.max()} cells")


def analyze_boundary_quality(masks, membrane_img, name):
    """Analyze how well boundaries align with membrane signal."""
    # Find boundaries
    boundaries = segmentation.find_boundaries(masks, mode='inner')
    interior = (masks > 0) & ~boundaries

    # Membrane intensity at boundaries vs interior
    boundary_intensity = membrane_img[boundaries].mean()
    interior_intensity = membrane_img[interior].mean()

    # Ratio: higher = boundaries are at membrane signal
    ratio = boundary_intensity / (interior_intensity + 1e-6)

    return {
        'name': name,
        'boundary_membrane_intensity': boundary_intensity,
        'interior_membrane_intensity': interior_intensity,
        'boundary_to_interior_ratio': ratio,
    }


def analyze_marker_distribution(masks, marker_img, marker_name):
    """Analyze marker distribution within cells."""
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]

    # For each cell, compare edge vs center intensity
    edge_intensities = []
    center_intensities = []

    for cell_id in cell_ids[:500]:  # Sample for speed
        cell_mask = masks == cell_id
        if cell_mask.sum() < 10:
            continue

        # Erode to get center
        eroded = ndimage.binary_erosion(cell_mask, iterations=2)
        edge = cell_mask & ~eroded

        if edge.sum() > 0 and eroded.sum() > 0:
            edge_intensities.append(marker_img[edge].mean())
            center_intensities.append(marker_img[eroded].mean())

    edge_mean = np.mean(edge_intensities)
    center_mean = np.mean(center_intensities)

    return {
        'marker': marker_name,
        'edge_intensity': edge_mean,
        'center_intensity': center_mean,
        'edge_to_center_ratio': edge_mean / (center_mean + 1e-6),
    }


print("\n=== Validation Metrics ===\n")

# 1. Boundary-membrane alignment
print("1. Boundary-Membrane Alignment")
print("   (Higher ratio = boundaries align with membrane signal)")
print()

for masks, name in [(masks_voronoi, 'Voronoi'), (masks_membrane, 'Membrane')]:
    result = analyze_boundary_quality(masks, membrane, name)
    print(f"   {name}:")
    print(f"     Boundary intensity: {result['boundary_membrane_intensity']:.4f}")
    print(f"     Interior intensity: {result['interior_membrane_intensity']:.4f}")
    print(f"     Ratio: {result['boundary_to_interior_ratio']:.2f}x")
    print()

# 2. Membrane marker edge localization
print("2. Membrane Marker Edge Localization")
print("   (Membrane markers SHOULD be higher at cell edges)")
print()

membrane_markers = [('CD45RO', cd45ro), ('CD3', cd3), ('CD20', cd20)]

for marker_name, marker_img in membrane_markers:
    print(f"   {marker_name}:")
    for masks, seg_name in [(masks_voronoi, 'Voronoi'), (masks_membrane, 'Membrane')]:
        result = analyze_marker_distribution(masks, marker_img, marker_name)
        print(f"     {seg_name}: edge/center = {result['edge_to_center_ratio']:.2f}x")
    print()

# 3. Nuclear marker center localization (control - should be similar for both)
print("3. Nuclear Marker Center Localization (control)")
print("   (Nuclear markers SHOULD be higher at cell centers)")
print()

dna1_idx = markers.index('DNA1')
dna1 = image[:, :, dna1_idx].astype(np.float32)

for masks, seg_name in [(masks_voronoi, 'Voronoi'), (masks_membrane, 'Membrane')]:
    result = analyze_marker_distribution(masks, dna1, 'DNA1')
    print(f"   {seg_name}: edge/center = {result['edge_to_center_ratio']:.2f}x")

# 4. Cell size distribution
print("\n4. Cell Size Distribution")

for masks, name in [(masks_voronoi, 'Voronoi'), (masks_membrane, 'Membrane')]:
    props = measure.regionprops(masks)
    areas = [p.area for p in props]
    print(f"   {name}:")
    print(f"     Mean area: {np.mean(areas):.1f} px")
    print(f"     Median area: {np.median(areas):.1f} px")
    print(f"     Std: {np.std(areas):.1f} px")

# Visualization
print("\n\nGenerating visualization...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Sample a small region for detailed view
h, w = nuclear.shape
cy, cx = h // 2, w // 2
size = 150
y1, y2 = cy - size, cy + size
x1, x2 = cx - size, cx + size

# Row 1: Voronoi
boundaries_v = segmentation.find_boundaries(masks_voronoi, mode='outer')

axes[0, 0].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
axes[0, 0].set_title('Nuclear')
axes[0, 0].axis('off')

axes[0, 1].imshow(membrane[y1:y2, x1:x2], cmap='hot')
axes[0, 1].set_title('Membrane (combined)')
axes[0, 1].axis('off')

axes[0, 2].imshow(membrane[y1:y2, x1:x2], cmap='hot')
axes[0, 2].imshow(boundaries_v[y1:y2, x1:x2], cmap='Greens', alpha=0.8)
axes[0, 2].set_title('VORONOI boundaries\non membrane')
axes[0, 2].axis('off')

# Overlay on CD45RO specifically
axes[0, 3].imshow(cd45ro[y1:y2, x1:x2], cmap='hot', vmax=np.percentile(cd45ro, 99))
axes[0, 3].imshow(boundaries_v[y1:y2, x1:x2], cmap='Greens', alpha=0.8)
axes[0, 3].set_title('VORONOI on CD45RO')
axes[0, 3].axis('off')

# Row 2: Membrane-guided
boundaries_m = segmentation.find_boundaries(masks_membrane, mode='outer')

axes[1, 0].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
axes[1, 0].set_title('Nuclear')
axes[1, 0].axis('off')

axes[1, 1].imshow(membrane[y1:y2, x1:x2], cmap='hot')
axes[1, 1].set_title('Membrane (combined)')
axes[1, 1].axis('off')

axes[1, 2].imshow(membrane[y1:y2, x1:x2], cmap='hot')
axes[1, 2].imshow(boundaries_m[y1:y2, x1:x2], cmap='Blues', alpha=0.8)
axes[1, 2].set_title('MEMBRANE boundaries\non membrane')
axes[1, 2].axis('off')

axes[1, 3].imshow(cd45ro[y1:y2, x1:x2], cmap='hot', vmax=np.percentile(cd45ro, 99))
axes[1, 3].imshow(boundaries_m[y1:y2, x1:x2], cmap='Blues', alpha=0.8)
axes[1, 3].set_title('MEMBRANE on CD45RO')
axes[1, 3].axis('off')

plt.suptitle(f'{sample_id}: Segmentation Validation (300x300 region)', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_segmentation_validation.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {OUTPUT_DIR}/{sample_id}_segmentation_validation.png")

# Open result
import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_segmentation_validation.png')])

print("\n=== Interpretation ===")
print("""
BETTER segmentation should show:
1. Higher boundary-to-interior ratio (boundaries at membrane signal)
2. Higher edge-to-center ratio for membrane markers (CD45RO, CD3, CD20)
3. Lower edge-to-center ratio for nuclear markers (DNA1) - this is a control

If membrane-guided gives higher ratios for membrane markers, it's doing
a better job of placing boundaries where the actual cell membranes are.
""")
