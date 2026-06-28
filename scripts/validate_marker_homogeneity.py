#!/usr/bin/env python3
"""Validate segmentation by marker expression homogeneity within cells."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import ndimage

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
print(f"Markers: {len(markers)}")

# Prepare images
nuclear = prepare_nuclear_image(image, markers, ['DNA1', 'DNA2'])
membrane = prepare_membrane_image(image, markers, ['CD45RO', 'CD3', 'CD20'])

# Run both segmentation methods
print("\nRunning segmentation...")
masks_voronoi = segment_cells_local_maxima(nuclear, sigma=1.0, min_distance=2, expansion='voronoi')
masks_membrane = segment_cells_membrane(nuclear, membrane, sigma=1.0, min_distance=2, membrane_weight=0.7)

n_cells = masks_voronoi.max()
print(f"  Cells: {n_cells}")

# Key markers to analyze (cell-type specific)
key_markers = [
    # T-cell markers
    'CD3', 'CD4', 'CD8a',
    # B-cell markers
    'CD20',
    # Other immune
    'CD45RO', 'FoxP3', 'PD_1', 'ICOS',
    # Functional markers
    'GranzymeB', 'TIM3', 'LAG3', 'CTLA4',
    # Transcription factors
    'T_Bet', 'GATA3', 'TOX',
    # Nuclear (control)
    'DNA1', 'HistoneH3',
]

# Filter to available markers
key_markers = [m for m in key_markers if m in markers]
print(f"\nAnalyzing {len(key_markers)} markers: {key_markers}")


def compute_within_cell_cv(masks, marker_img, min_pixels=10):
    """Compute coefficient of variation within each cell.

    Lower CV = more homogeneous expression within cells = better segmentation.
    """
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]

    cvs = []
    means = []

    for cell_id in cell_ids:
        cell_mask = masks == cell_id
        if cell_mask.sum() < min_pixels:
            continue

        values = marker_img[cell_mask]
        mean_val = values.mean()
        std_val = values.std()

        if mean_val > 0:
            cv = std_val / mean_val
            cvs.append(cv)
            means.append(mean_val)

    return np.array(cvs), np.array(means)


def compute_expression_stats(masks, marker_img):
    """Compute mean expression per cell."""
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]

    means = ndimage.mean(marker_img, masks, cell_ids)
    return np.array(means)


# Analyze each marker
print("\n=== Within-Cell Homogeneity (CV) ===")
print("Lower CV = more homogeneous = better segmentation\n")

results = []

for marker in key_markers:
    idx = markers.index(marker)
    marker_img = image[:, :, idx].astype(np.float32)

    # Compute CV for both methods
    cv_voronoi, means_v = compute_within_cell_cv(masks_voronoi, marker_img)
    cv_membrane, means_m = compute_within_cell_cv(masks_membrane, marker_img)

    # Compare median CVs
    median_cv_v = np.median(cv_voronoi)
    median_cv_m = np.median(cv_membrane)

    # Which is better?
    better = "Membrane" if median_cv_m < median_cv_v else "Voronoi"
    improvement = (median_cv_v - median_cv_m) / median_cv_v * 100

    results.append({
        'marker': marker,
        'cv_voronoi': median_cv_v,
        'cv_membrane': median_cv_m,
        'better': better,
        'improvement': improvement,
    })

    print(f"{marker:15s}  Voronoi: {median_cv_v:.3f}  Membrane: {median_cv_m:.3f}  → {better} ({improvement:+.1f}%)")

# Summary
print("\n=== Summary ===")
membrane_wins = sum(1 for r in results if r['better'] == 'Membrane')
voronoi_wins = len(results) - membrane_wins
print(f"Membrane better for {membrane_wins}/{len(results)} markers")
print(f"Voronoi better for {voronoi_wins}/{len(results)} markers")

avg_improvement = np.mean([r['improvement'] for r in results])
print(f"Average CV improvement with membrane: {avg_improvement:+.1f}%")


# Visualization
print("\n\nGenerating visualization...")

# 1. CV comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Bar chart of CVs
marker_names = [r['marker'] for r in results]
cv_v = [r['cv_voronoi'] for r in results]
cv_m = [r['cv_membrane'] for r in results]

x = np.arange(len(marker_names))
width = 0.35

axes[0, 0].bar(x - width/2, cv_v, width, label='Voronoi', color='green', alpha=0.7)
axes[0, 0].bar(x + width/2, cv_m, width, label='Membrane', color='blue', alpha=0.7)
axes[0, 0].set_ylabel('Median CV (lower = better)')
axes[0, 0].set_title('Within-Cell Coefficient of Variation')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(marker_names, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

# Improvement chart
improvements = [r['improvement'] for r in results]
colors = ['blue' if imp > 0 else 'green' for imp in improvements]
axes[0, 1].bar(x, improvements, color=colors, alpha=0.7)
axes[0, 1].set_ylabel('% Improvement with Membrane')
axes[0, 1].set_title('CV Improvement (positive = membrane better)')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(marker_names, rotation=45, ha='right')
axes[0, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

# 2. Expression distribution comparison for key markers
# Pick a few markers to show distributions
show_markers = ['CD3', 'CD20', 'CD4', 'GranzymeB']
show_markers = [m for m in show_markers if m in markers]

for i, marker in enumerate(show_markers[:2]):
    ax = axes[1, i]
    idx = markers.index(marker)
    marker_img = image[:, :, idx].astype(np.float32)

    expr_v = compute_expression_stats(masks_voronoi, marker_img)
    expr_m = compute_expression_stats(masks_membrane, marker_img)

    # Histogram
    bins = np.linspace(0, np.percentile(np.concatenate([expr_v, expr_m]), 99), 50)
    ax.hist(expr_v, bins=bins, alpha=0.5, label='Voronoi', color='green', density=True)
    ax.hist(expr_m, bins=bins, alpha=0.5, label='Membrane', color='blue', density=True)
    ax.set_xlabel(f'{marker} mean intensity')
    ax.set_ylabel('Density')
    ax.set_title(f'{marker} Expression Distribution')
    ax.legend()

plt.suptitle(f'{sample_id}: Marker Homogeneity Comparison', fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / f'{sample_id}_marker_homogeneity.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {OUTPUT_DIR}/{sample_id}_marker_homogeneity.png")


# 3. Detailed comparison for T-cell markers
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))

t_cell_markers = ['CD3', 'CD4', 'CD8a', 'FoxP3', 'PD_1', 'GranzymeB', 'ICOS', 'TIM3']
t_cell_markers = [m for m in t_cell_markers if m in markers][:8]

for i, marker in enumerate(t_cell_markers):
    row, col = i // 4, i % 4
    ax = axes2[row, col]

    idx = markers.index(marker)
    marker_img = image[:, :, idx].astype(np.float32)

    expr_v = compute_expression_stats(masks_voronoi, marker_img)
    expr_m = compute_expression_stats(masks_membrane, marker_img)

    # Log scale for better visualization
    expr_v_log = np.log1p(expr_v)
    expr_m_log = np.log1p(expr_m)

    bins = np.linspace(0, max(expr_v_log.max(), expr_m_log.max()), 40)
    ax.hist(expr_v_log, bins=bins, alpha=0.5, label='Voronoi', color='green', density=True)
    ax.hist(expr_m_log, bins=bins, alpha=0.5, label='Membrane', color='blue', density=True)
    ax.set_xlabel('log(intensity + 1)')
    ax.set_title(marker)
    if i == 0:
        ax.legend()

plt.suptitle(f'{sample_id}: T-cell Marker Distributions (Voronoi vs Membrane)', fontsize=14)
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / f'{sample_id}_tcell_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {OUTPUT_DIR}/{sample_id}_tcell_distributions.png")

# Open results
import subprocess
subprocess.run(['open', str(OUTPUT_DIR / f'{sample_id}_marker_homogeneity.png')])

print("\n=== Interpretation ===")
print("""
Lower within-cell CV means marker expression is more uniform within each cell,
which indicates the segmentation is capturing real cell boundaries better.

If membrane-guided consistently shows lower CV across cell-type markers,
it's doing a better job of:
1. Not mixing signal from adjacent cells
2. Capturing the full cell body for accurate quantification
3. Providing cleaner data for downstream clustering/phenotyping
""")
