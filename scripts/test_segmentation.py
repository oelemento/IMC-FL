#!/usr/bin/env python3
"""Test segmentation on a single ROI with parameter sweep."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import (
    prepare_nuclear_image,
    segment_cells_cellpose,
    segment_roi,
    visualize_segmentation,
    parameter_sweep,
)

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Load first ROI
files = list_rois(DATA_DIR)
roi_file = files[0]
sample_id = extract_sample_id(roi_file.name)

print(f"Loading {sample_id}: {roi_file.name}")
image, markers, metadata = load_roi_txt(roi_file)
print(f"Image shape: {image.shape}")

# Parameter sweep to find optimal diameter
print("\n=== Parameter Sweep ===")
print("Testing different cell diameters...")
fig = parameter_sweep(
    image, markers,
    diameters=[20, 30, 40, 50],
    nuclear_markers=['DNA1', 'DNA2'],
)
fig.savefig(OUTPUT_DIR / f'{sample_id}_diameter_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/{sample_id}_diameter_sweep.png")

# Run segmentation with chosen diameter
print("\n=== Running Segmentation (diameter=30) ===")
masks, adata = segment_roi(
    image, markers, sample_id,
    diameter=30,
    nuclear_markers=['DNA1', 'DNA2'],
)
print(f"Segmented {adata.n_obs} cells")

# Visualize result
print("\nGenerating visualization...")
fig = visualize_segmentation(image, markers, masks)
fig.savefig(OUTPUT_DIR / f'{sample_id}_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/{sample_id}_segmentation.png")

# Save AnnData
adata_path = OUTPUT_DIR / f'{sample_id}_cells.h5ad'
adata.write(adata_path)
print(f"Saved: {adata_path}")

# Print summary statistics
print("\n=== Summary ===")
print(f"Sample: {sample_id}")
print(f"Image size: {metadata['width']} x {metadata['height']} pixels")
print(f"Total cells: {adata.n_obs}")
print(f"Cell density: {adata.n_obs / (metadata['width'] * metadata['height'] / 1e6):.0f} cells/mm² (approx)")
print(f"Mean cell area: {adata.obs['area'].mean():.1f} pixels")
print(f"Median cell area: {adata.obs['area'].median():.1f} pixels")

# Show marker expression summary
print(f"\nMarker expression (mean ± std across cells):")
import numpy as np
key_markers = ['CD3', 'CD4', 'CD8a', 'CD20', 'PD_1', 'FoxP3']
for marker in key_markers:
    if marker in markers:
        idx = markers.index(marker)
        vals = adata.X[:, idx]
        print(f"  {marker}: {vals.mean():.2f} ± {vals.std():.2f}")

print(f"\n✓ Segmentation complete!")
print(f"\nView results: open {OUTPUT_DIR}/{sample_id}_segmentation.png")
