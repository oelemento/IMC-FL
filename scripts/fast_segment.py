#!/usr/bin/env python3
"""Fast segmentation using watershed (runs in seconds)."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import segment_roi, visualize_segmentation

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

# Run watershed segmentation (fast!)
print("\nRunning watershed segmentation...")
t0 = time.time()

masks, adata = segment_roi(
    image, markers, sample_id,
    method='watershed',
    nuclear_markers=['DNA1', 'DNA2'],
    min_distance=8,  # adjust for cell density
)

elapsed = time.time() - t0
print(f"✓ Segmented {adata.n_obs} cells in {elapsed:.1f} seconds")

# Visualize
print("\nGenerating visualization...")
fig = visualize_segmentation(image, markers, masks)
plt.suptitle(f'{sample_id}: {adata.n_obs} cells (watershed)', y=1.02)
fig.savefig(OUTPUT_DIR / f'{sample_id}_segmentation_watershed.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/{sample_id}_segmentation_watershed.png")

# Save AnnData
adata_path = OUTPUT_DIR / f'{sample_id}_cells.h5ad'
adata.write(adata_path)
print(f"Saved: {adata_path}")

# Summary stats
print("\n=== Summary ===")
print(f"Sample: {sample_id}")
print(f"Image: {metadata['width']} x {metadata['height']} pixels")
print(f"Cells: {adata.n_obs}")
print(f"Mean cell area: {adata.obs['area'].mean():.1f} pixels")
print(f"Median cell area: {adata.obs['area'].median():.1f} pixels")

# Cell density
area_mm2 = (metadata['width'] * metadata['height']) / 1e6  # assuming 1µm/pixel
print(f"Cell density: ~{adata.n_obs / area_mm2:.0f} cells/mm²")

# Marker expression preview
print("\nMean expression (top markers):")
key_markers = ['CD3', 'CD4', 'CD8a', 'CD20', 'PD_1', 'FoxP3', 'GranzymeB', 'CXCR5']
for marker in key_markers:
    if marker in markers:
        idx = markers.index(marker)
        mean_val = adata.X[:, idx].mean()
        pos_pct = (adata.X[:, idx] > mean_val).sum() / adata.n_obs * 100
        print(f"  {marker}: mean={mean_val:.2f}, {pos_pct:.1f}% above mean")

# Open result
import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_segmentation_watershed.png')])

print("\n✓ Done! View the segmentation overlay to check quality.")
print("\nNext: If segmentation looks good, we can batch process all ROIs.")
print("      If not, adjust min_distance parameter (smaller = more cells).")
