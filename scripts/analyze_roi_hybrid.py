#!/usr/bin/env python3
"""Analyze a single ROI with hybrid segmentation (full image)."""

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
    segment_roi,
    prepare_nuclear_image,
    visualize_segmentation,
)

# Paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load first ROI
files = list_rois(DATA_DIR)
filepath = files[0]
sample_id = extract_sample_id(filepath.name)

print(f"Loading {sample_id} from {filepath.name}...")
image, markers, metadata = load_roi_txt(filepath)
print(f"Image: {image.shape} ({image.shape[0]}x{image.shape[1]}, {len(markers)} markers)")

# Run hybrid segmentation
print(f"\nRunning hybrid segmentation on full {image.shape[0]}x{image.shape[1]} image...")
print("  Step 1: Detecting nuclei from DNA signal...")
print("  Step 2: Running Cellpose (flow=0.8) — this will take a few minutes on CPU...")

t0 = time.time()
masks, adata = segment_roi(
    image, markers, sample_id,
    method='hybrid',
    flow_threshold=0.8,
    min_distance=2,
    sigma=1.0,
)
total_time = time.time() - t0

n_cells = adata.n_obs
print(f"\n✓ {n_cells} cells segmented in {total_time:.0f}s")
print(f"\nAnnData: {adata}")
print(f"  Markers: {list(adata.var_names[:5])}...{list(adata.var_names[-3:])}")
print(f"  Cell areas: mean={adata.obs['area'].mean():.1f}, median={adata.obs['area'].median():.1f} px")

# Save AnnData
h5ad_path = OUTPUT_DIR / f'{sample_id}_hybrid.h5ad'
adata.write(h5ad_path)
print(f"\nSaved: {h5ad_path}")

# Visualization
print("\nGenerating figures...")

# 1. Segmentation overlay
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])
boundaries = segmentation.find_boundaries(masks, mode='outer')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(nuclear, cmap='gray')
axes[0].set_title(f'Nuclear (DNA1+DNA2)')
axes[0].axis('off')

axes[1].imshow(nuclear, cmap='gray')
axes[1].imshow(boundaries, cmap='Greens', alpha=0.7)
axes[1].set_title(f'Hybrid segmentation: {n_cells} cells')
axes[1].axis('off')

# Zoomed center
h, w = nuclear.shape
z = 150
y1, y2 = h//2 - z, h//2 + z
x1, x2 = w//2 - z, w//2 + z
axes[2].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
axes[2].imshow(boundaries[y1:y2, x1:x2], cmap='Greens', alpha=0.7)
axes[2].set_title('Zoomed center (300x300)')
axes[2].axis('off')

plt.suptitle(f'{sample_id}: Hybrid Segmentation (Cellpose + Nuclear Filter)', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_hybrid_full.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Marker expression heatmap (top expressed markers)
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))

# Show key T-cell markers overlaid with segmentation
show_markers = ['CD3', 'CD4', 'CD8a', 'CD20', 'FoxP3', 'PD_1', 'GranzymeB', 'CD45RO']
show_markers = [m for m in show_markers if m in markers][:8]

for i, marker in enumerate(show_markers):
    ax = axes2[i // 4, i % 4]
    idx = markers.index(marker)
    ch = image[:, :, idx].astype(np.float32)
    vmax = np.percentile(ch, 99.5)

    ax.imshow(ch[y1:y2, x1:x2], cmap='hot', vmax=vmax)
    ax.imshow(boundaries[y1:y2, x1:x2], cmap='Greens', alpha=0.3)
    ax.set_title(marker, fontsize=11)
    ax.axis('off')

plt.suptitle(f'{sample_id}: Marker Channels with Cell Boundaries (zoomed)', fontsize=13)
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / f'{sample_id}_hybrid_markers.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Cell area distribution
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

areas = adata.obs['area'].values
axes3[0].hist(areas, bins=100, range=(0, np.percentile(areas, 99)), color='steelblue', edgecolor='none')
axes3[0].set_xlabel('Cell area (pixels)')
axes3[0].set_ylabel('Count')
axes3[0].set_title(f'Cell area distribution (n={n_cells})')
axes3[0].axvline(np.median(areas), color='red', linestyle='--', label=f'median={np.median(areas):.0f}')
axes3[0].legend()

# Mean expression per marker
mean_expr = adata.X.mean(axis=0)
sorted_idx = np.argsort(mean_expr)[::-1]
top_n = 20
axes3[1].barh(range(top_n), mean_expr[sorted_idx[:top_n]], color='steelblue')
axes3[1].set_yticks(range(top_n))
axes3[1].set_yticklabels([adata.var_names[i] for i in sorted_idx[:top_n]], fontsize=9)
axes3[1].set_xlabel('Mean intensity')
axes3[1].set_title('Top 20 markers by mean expression')
axes3[1].invert_yaxis()

plt.tight_layout()
fig3.savefig(OUTPUT_DIR / f'{sample_id}_hybrid_stats.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {OUTPUT_DIR}/{sample_id}_hybrid_full.png")
print(f"Saved: {OUTPUT_DIR}/{sample_id}_hybrid_markers.png")
print(f"Saved: {OUTPUT_DIR}/{sample_id}_hybrid_stats.png")

import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_hybrid_full.png')])

print(f"\n=== Done ===")
print(f"Sample: {sample_id}")
print(f"Cells: {n_cells}")
print(f"Time: {total_time:.0f}s")
print(f"Output: {h5ad_path}")
