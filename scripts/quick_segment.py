#!/usr/bin/env python3
"""Quick segmentation on a single ROI."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import (
    prepare_nuclear_image,
    segment_roi,
    visualize_segmentation,
)

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load first ROI
files = list_rois(DATA_DIR)
roi_file = files[0]
sample_id = extract_sample_id(roi_file.name)

print(f"Loading {sample_id}: {roi_file.name}")
image, markers, metadata = load_roi_txt(roi_file)
print(f"Image shape: {image.shape}")

# Run segmentation with diameter=30 (typical for IMC nuclei)
print("\nRunning Cellpose segmentation (diameter=30)...")
print("This may take 2-5 minutes on CPU...")

masks, adata = segment_roi(
    image, markers, sample_id,
    diameter=30,
    nuclear_markers=['DNA1', 'DNA2'],
)

print(f"\n✓ Segmented {adata.n_obs} cells")

# Visualize
print("\nGenerating visualization...")
fig = visualize_segmentation(image, markers, masks)
fig.savefig(OUTPUT_DIR / f'{sample_id}_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/{sample_id}_segmentation.png")

# Save AnnData
adata_path = OUTPUT_DIR / f'{sample_id}_cells.h5ad'
adata.write(adata_path)
print(f"Saved: {adata_path}")

# Summary
print("\n=== Summary ===")
print(f"Sample: {sample_id}")
print(f"Image: {metadata['width']} x {metadata['height']} pixels")
print(f"Cells: {adata.n_obs}")
print(f"Mean area: {adata.obs['area'].mean():.1f} px")
print(f"Markers: {len(markers)}")

# Open result
import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_segmentation.png')])
