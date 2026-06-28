#!/usr/bin/env python3
"""Quick visualization of IMC data."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.visualization import (
    plot_single_channel, plot_composite, quick_view,
    create_composite, normalize_image
)

# Data paths
DATA_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw')
OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Load first T-cell panel ROI
t_files = list_rois(DATA_DIR / 'TMA_B1_T')
print(f"Found {len(t_files)} T-cell panel ROIs")

roi_file = t_files[0]
sample_id = extract_sample_id(roi_file.name)
print(f"\nLoading {sample_id}: {roi_file.name}")

image, markers, metadata = load_roi_txt(roi_file)
print(f"Image shape: {image.shape}")
print(f"Markers: {markers[:10]}...")

# 1. Quick overview
print("\nGenerating quick overview...")
fig = quick_view(image, markers, figsize=(16, 5))
plt.suptitle(f'ROI: {sample_id} ({metadata["width"]}x{metadata["height"]})', y=1.02)
fig.savefig(OUTPUT_DIR / f'{sample_id}_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/{sample_id}_overview.png")

# 2. T-cell composite
print("\nGenerating T-cell composite...")
fig, ax = plt.subplots(figsize=(12, 12))
composite_def = {}
if 'CD4' in markers:
    composite_def['CD4'] = 'green'
if 'CD8a' in markers:
    composite_def['CD8a'] = 'red'
if 'FoxP3' in markers:
    composite_def['FoxP3'] = 'cyan'
if 'DNA1' in markers:
    composite_def['DNA1'] = 'blue'

if composite_def:
    plot_composite(image, markers, composite_def, ax=ax)
    ax.set_title(f'{sample_id}: CD4 (green), CD8 (red), FoxP3 (cyan), DNA (blue)', fontsize=14)
fig.savefig(OUTPUT_DIR / f'{sample_id}_tcell_composite.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/{sample_id}_tcell_composite.png")

# 3. Immune checkpoint composite
print("\nGenerating checkpoint composite...")
fig, ax = plt.subplots(figsize=(12, 12))
checkpoint_def = {}
if 'PD_1' in markers:
    checkpoint_def['PD_1'] = 'red'
if 'TIM3' in markers:
    checkpoint_def['TIM3'] = 'green'
if 'LAG3' in markers:
    checkpoint_def['LAG3'] = 'cyan'
if 'CD3' in markers:
    checkpoint_def['CD3'] = 'white'

if checkpoint_def:
    plot_composite(image, markers, checkpoint_def, ax=ax)
    ax.set_title(f'{sample_id}: PD-1 (red), TIM3 (green), LAG3 (cyan), CD3 (white)', fontsize=14)
fig.savefig(OUTPUT_DIR / f'{sample_id}_checkpoint_composite.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/{sample_id}_checkpoint_composite.png")

# 4. Key markers grid
print("\nGenerating markers grid...")
key_markers = ['DNA1', 'CD3', 'CD4', 'CD8a', 'CD20', 'PD_1', 'FoxP3', 'GranzymeB',
               'CXCR5', 'TIM3', 'LAG3', 'ICOS']
present_markers = [m for m in key_markers if m in markers]

ncols = 4
nrows = (len(present_markers) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
axes = axes.flatten()

for i, marker in enumerate(present_markers):
    plot_single_channel(image, markers, marker, ax=axes[i], colorbar=False)

for i in range(len(present_markers), len(axes)):
    axes[i].axis('off')

plt.suptitle(f'{sample_id}: Key Markers', y=1.01, fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_markers_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/{sample_id}_markers_grid.png")

# 5. Compare multiple ROIs
print("\nComparing multiple ROIs...")
n_rois = min(6, len(t_files))
fig, axes = plt.subplots(2, n_rois, figsize=(3*n_rois, 6))

for i, f in enumerate(t_files[:n_rois]):
    img, mkrs, meta = load_roi_txt(f)
    sid = extract_sample_id(f.name)

    # DNA
    idx = mkrs.index('DNA1')
    dna = normalize_image(img[:, :, idx])
    axes[0, i].imshow(dna, cmap='gray')
    axes[0, i].set_title(f'{sid}\n{meta["width"]}x{meta["height"]}', fontsize=9)
    axes[0, i].axis('off')

    # CD3
    if 'CD3' in mkrs:
        idx = mkrs.index('CD3')
        cd3 = normalize_image(img[:, :, idx])
        axes[1, i].imshow(cd3, cmap='Greens')
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('DNA1', fontsize=10)
axes[1, 0].set_ylabel('CD3', fontsize=10)
plt.suptitle('ROI Comparison', y=1.02, fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'roi_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/roi_comparison.png")

print(f"\n✓ All visualizations saved to {OUTPUT_DIR}/")
print("\nTo view: open output/*.png")
