#!/usr/bin/env python3
"""Visualize raw CD21 pixel signal vs segmented FDC centroids.

Shows how cell-based segmentation misses the FDC meshwork.
Picks 2 ROIs from B1 S-panel: one high-FDC, one low-FDC.
"""
import sys, os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'scripts'))
from run_hypotheses_v2 import is_tumor_core, load_array

V8_PATH = os.path.join(BASE, 'output', 'all_TMA_S_global_v8.h5ad')
RAW_DIR = os.path.join(BASE, 'data', 'raw', 'TMA_B1_S')

LQ_TYPES = {'Unassigned', 'Low quality', 'Unknown'}

def load_raw_image(txt_path, marker_col='CD21(Er170Di)'):
    """Load raw TXT and reconstruct pixel image for a marker."""
    import pandas as pd
    df = pd.read_csv(txt_path, sep='\t')
    x = df['X'].values.astype(int)
    y = df['Y'].values.astype(int)
    vals = df[marker_col].values

    img = np.zeros((y.max() + 1, x.max() + 1), dtype=np.float32)
    img[y, x] = vals
    return img

def find_b1_rois_by_fdc(f):
    """Find B1 S-panel ROIs ranked by FDC fraction."""
    sample_ids = load_array(f, 'sample_id')
    cell_types = load_array(f, 'cell_type')
    tma_arr = load_array(f, 'tma')

    # B1 tumor ROIs only
    b1_mask = tma_arr == 'B1'
    tumor_mask = np.array([is_tumor_core(s) for s in sample_ids])
    mask = b1_mask & tumor_mask

    rois = np.unique(sample_ids[mask])
    roi_fdc = {}
    for roi in rois:
        roi_mask = mask & (sample_ids == roi)
        ct = cell_types[roi_mask]
        typed = ct[~np.isin(ct, list(LQ_TYPES))]
        if len(typed) < 5000:
            continue
        fdc_frac = np.sum(typed == 'FDC') / len(typed)
        roi_fdc[roi] = fdc_frac

    return roi_fdc

def find_raw_file(roi_id, raw_dir):
    """Find the raw TXT file matching an ROI ID."""
    # ROI IDs look like "B1_FL1" etc
    # Raw files: 20210518_CT14_09_B1_Stromalpanel_1_FL1_L_3.txt
    import re
    fl_part = roi_id.replace('B1_', '')  # e.g., "FL1"
    # Match FL1_ but not FL10_, FL11_ etc
    pattern = re.compile(rf'_{re.escape(fl_part)}_')
    for fname in os.listdir(raw_dir):
        if pattern.search(fname) and fname.endswith('.txt'):
            return os.path.join(raw_dir, fname)
    return None

def main():
    print("Loading S-panel v8 data...")
    f = h5py.File(V8_PATH, 'r')
    sample_ids = load_array(f, 'sample_id')
    cell_types = load_array(f, 'cell_type')
    cx = f['obs']['centroid_x'][...]
    cy = f['obs']['centroid_y'][...]
    tma_arr = load_array(f, 'tma')

    # Find ROIs ranked by FDC fraction
    roi_fdc = find_b1_rois_by_fdc(f)
    sorted_rois = sorted(roi_fdc.items(), key=lambda x: x[1])

    print(f"\nB1 S-panel tumor ROIs: {len(sorted_rois)}")
    print(f"FDC fraction range: {sorted_rois[0][1]:.4f} - {sorted_rois[-1][1]:.4f}")

    # Pick low-FDC and high-FDC ROIs that have raw files
    low_roi, high_roi = None, None
    for roi_id, fdc_frac in sorted_rois:
        raw_file = find_raw_file(roi_id, RAW_DIR)
        if raw_file and fdc_frac < 0.02:
            low_roi = (roi_id, fdc_frac, raw_file)
            break

    for roi_id, fdc_frac in reversed(sorted_rois):
        raw_file = find_raw_file(roi_id, RAW_DIR)
        if raw_file and fdc_frac > 0.10:
            high_roi = (roi_id, fdc_frac, raw_file)
            break

    if not low_roi or not high_roi:
        print("Could not find suitable ROIs with raw files")
        # Fall back: just use first and last
        for roi_id, fdc_frac in sorted_rois[:5]:
            raw_file = find_raw_file(roi_id, RAW_DIR)
            if raw_file:
                low_roi = (roi_id, fdc_frac, raw_file)
                break
        for roi_id, fdc_frac in sorted_rois[-5:]:
            raw_file = find_raw_file(roi_id, RAW_DIR)
            if raw_file:
                high_roi = (roi_id, fdc_frac, raw_file)
                break

    print(f"\nLow FDC:  {low_roi[0]} (FDC={low_roi[1]:.3f})")
    print(f"High FDC: {high_roi[0]} (FDC={high_roi[1]:.3f})")

    # Create figure: 2 rows x 3 cols
    # Row 1: high FDC ROI, Row 2: low FDC ROI
    # Col 1: raw CD21 pixel signal
    # Col 2: segmented cell centroids (FDC highlighted)
    # Col 3: overlay (raw CD21 + FDC centroids)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for row, (roi_id, fdc_frac, raw_file) in enumerate([high_roi, low_roi]):
        print(f"\nProcessing {roi_id}...")

        # Load raw CD21 image
        cd21_img = load_raw_image(raw_file, 'CD21(Er170Di)')
        # Also load CD20 for context
        cd20_img = load_raw_image(raw_file, 'CD20(Dy161Di)')
        # And DNA for tissue mask
        dna_img = load_raw_image(raw_file, 'DNA1(Ir191Di)')

        # Get cell centroids for this ROI
        tumor_mask = np.array([is_tumor_core(s) for s in sample_ids])
        roi_mask = (sample_ids == roi_id) & tumor_mask
        roi_cx = cx[roi_mask]
        roi_cy = cy[roi_mask]
        roi_ct = cell_types[roi_mask]

        fdc_mask = roi_ct == 'FDC'
        b_mask = np.isin(roi_ct, ['B cells', 'B cells (BCL2+)', 'B cells (PAX5+)',
                                    'B cells (CXCR5hi)', 'B cells (weak CD20)',
                                    'GC B cells', 'Activated B / Plasmablast'])
        other_mask = ~fdc_mask & ~b_mask & ~np.isin(roi_ct, list(LQ_TYPES))
        unassigned_mask = np.isin(roi_ct, list(LQ_TYPES))

        n_fdc = np.sum(fdc_mask)
        n_typed = np.sum(~np.isin(roi_ct, list(LQ_TYPES)))

        # Arcsinh transform for display
        cd21_display = np.arcsinh(cd21_img / 5)
        cd20_display = np.arcsinh(cd20_img / 5)

        # Col 1: Raw CD21 pixel signal
        ax = axes[row, 0]
        vmax = np.percentile(cd21_display[cd21_display > 0], 99) if np.any(cd21_display > 0) else 1
        ax.imshow(cd21_display, cmap='magma', vmin=0, vmax=vmax, origin='lower')
        ax.set_title(f'Raw CD21 pixel signal\n{roi_id} (FDC={fdc_frac:.1%})', fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # Col 2: Segmented cell centroids
        ax = axes[row, 1]
        # Gray background from DNA
        dna_display = np.arcsinh(dna_img / 5)
        ax.imshow(dna_display, cmap='gray', alpha=0.3, origin='lower',
                  vmin=0, vmax=np.percentile(dna_display[dna_display > 0], 99) if np.any(dna_display > 0) else 1)
        # Plot cells
        if np.sum(unassigned_mask) > 0:
            ax.scatter(roi_cx[unassigned_mask], roi_cy[unassigned_mask],
                      s=1, c='#D3D3D3', alpha=0.2, zorder=1)
        if np.sum(other_mask) > 0:
            ax.scatter(roi_cx[other_mask], roi_cy[other_mask],
                      s=3, c='#888888', alpha=0.4, zorder=2)
        if np.sum(b_mask) > 0:
            ax.scatter(roi_cx[b_mask], roi_cy[b_mask],
                      s=4, c='#4477AA', alpha=0.5, zorder=3, label=f'B cells ({np.sum(b_mask)})')
        if np.sum(fdc_mask) > 0:
            ax.scatter(roi_cx[fdc_mask], roi_cy[fdc_mask],
                      s=40, c='#FFD700', edgecolors='black', linewidth=0.5,
                      alpha=0.9, zorder=5, label=f'FDC ({n_fdc})')
        ax.set_title(f'Segmented cells\n{n_fdc} FDC of {n_typed} typed', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, cd21_img.shape[1]); ax.set_ylim(0, cd21_img.shape[0])

        # Col 3: Overlay — raw CD21 + FDC centroids
        ax = axes[row, 2]
        ax.imshow(cd21_display, cmap='magma', vmin=0, vmax=vmax, origin='lower')
        if np.sum(fdc_mask) > 0:
            ax.scatter(roi_cx[fdc_mask], roi_cy[fdc_mask],
                      s=60, facecolors='none', edgecolors='#00FF00', linewidth=1.5,
                      alpha=0.9, zorder=5, label=f'FDC centroids ({n_fdc})')
        ax.set_title(f'CD21 signal + FDC centroids\n(green circles = segmented FDC)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # Row labels
    axes[0, 0].text(-0.05, 0.5, 'High FDC ROI', transform=axes[0, 0].transAxes,
                     fontsize=13, fontweight='bold', rotation=90, va='center', ha='right')
    axes[1, 0].text(-0.05, 0.5, 'Low FDC ROI', transform=axes[1, 0].transAxes,
                     fontsize=13, fontweight='bold', rotation=90, va='center', ha='right')

    plt.suptitle('FDC Meshwork: Raw CD21 Signal vs Segmented Cell Centroids\n'
                 'Cell segmentation captures FDC soma but misses reticular processes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(BASE, 'output', 'fdc_raw_vs_segmented.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    f.close()

if __name__ == '__main__':
    main()
