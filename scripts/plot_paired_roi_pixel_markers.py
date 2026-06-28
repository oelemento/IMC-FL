#!/usr/bin/env python3
"""Plot pixel-level raw marker expression for paired ROIs across T and S panels."""
import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from src.data_loader import load_roi_txt

# --- Config ---
DATA_T = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
DATA_S = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_S')

ROIS = {
    'B1_FL32': {
        'T': '20220118_CT14_09_B1_Tcellpanel_4_FL32_R_5.txt',
        'S': '20210518_CT14_09_B1_Stromalpanel_5_FL32_R_2.txt',
        'label': 'B1_FL32 (concordant, B diff=1.7%)',
    },
    'B1_FL18': {
        'T': '20220118_CT14_09_B1_Tcellpanel_2_FL18_L_12.txt',
        'S': '20210518_CT14_09_B1_Stromalpanel_2_FL18_L_6.txt',
        'label': 'B1_FL18 (moderate, B diff=22%)',
    },
    'B1_FL10': {
        'T': '20220118_CT14_09_B1_Tcellpanel_2_FL10_R_3.txt',
        'S': '20210518_CT14_09_B1_Stromalpanel_2_FL10_R_3.txt',
        'label': 'B1_FL10 (moderate, B diff=23%)',
    },
    'B1_FL26': {
        'T': '20220118_CT14_09_B1_Tcellpanel_3_FL26_L_11.txt',
        'S': '20210518_CT14_09_B1_Stromalpanel_4_FL26_L_5.txt',
        'label': 'B1_FL26 (discordant, B diff=88%)',
    },
}

# Shared markers present in both panels
SHARED_MARKERS = ['CD20', 'CD4', 'CD8a', 'CD68']
# Panel-specific markers
S_SPECIFIC = ['PAX5', 'BCL_2', 'Vimentin']
T_SPECIFIC = ['CD3', 'FoxP3', 'GranzymeB']

ALL_MARKERS = SHARED_MARKERS + S_SPECIFIC + T_SPECIFIC
n_rois = len(ROIS)

# -----------------------------------------------------------
# Figure 1: Shared markers — S vs T pixel images side by side
# -----------------------------------------------------------
print("=== Figure 1: Shared markers, S vs T ===")
n_shared = len(SHARED_MARKERS)
# Layout: rows = markers * 2 (S row, T row), cols = ROIs
fig, axes = plt.subplots(n_shared * 2, n_rois, figsize=(n_rois * 4, n_shared * 2 * 3.2))

for j, (roi_name, roi_info) in enumerate(ROIS.items()):
    for panel_idx, (panel, data_dir) in enumerate([('S', DATA_S), ('T', DATA_T)]):
        txt_file = data_dir / roi_info[panel]
        print(f"  Loading {panel} {roi_name}: {txt_file.name}")
        image, markers, meta = load_roi_txt(txt_file)

        for mi, marker in enumerate(SHARED_MARKERS):
            row = mi * 2 + panel_idx  # S=even rows, T=odd rows
            ax = axes[row, j]

            if marker in markers:
                ch_idx = markers.index(marker)
                img = image[:, :, ch_idx]

                # Clip at 99th percentile for contrast
                vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1.0
                vmax = max(vmax, 0.1)

                ax.imshow(img, cmap='magma', vmin=0, vmax=vmax,
                         interpolation='nearest', aspect='equal')
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')

            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(f'{marker}\n({panel}-panel)', fontsize=10, fontweight='bold')
            if row == 0:
                ax.set_title(roi_info['label'], fontsize=9, fontweight='bold')

fig.suptitle('Shared markers — pixel-level raw ion counts (S-panel vs T-panel)',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('output/paired_roi_pixel_shared.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: output/paired_roi_pixel_shared.png")


# -----------------------------------------------------------
# Figure 2: All markers per panel (S-panel view)
# -----------------------------------------------------------
print("\n=== Figure 2: S-panel all markers ===")
s_markers = SHARED_MARKERS + S_SPECIFIC
n_s = len(s_markers)
fig, axes = plt.subplots(n_s, n_rois, figsize=(n_rois * 4, n_s * 3.2))

for j, (roi_name, roi_info) in enumerate(ROIS.items()):
    txt_file = DATA_S / roi_info['S']
    print(f"  Loading S {roi_name}")
    image, markers, meta = load_roi_txt(txt_file)

    for i, marker in enumerate(s_markers):
        ax = axes[i, j]
        if marker in markers:
            ch_idx = markers.index(marker)
            img = image[:, :, ch_idx]
            vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1.0
            vmax = max(vmax, 0.1)
            ax.imshow(img, cmap='magma', vmin=0, vmax=vmax,
                     interpolation='nearest', aspect='equal')
        else:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(marker, fontsize=10, fontweight='bold')
        if i == 0:
            ax.set_title(roi_info['label'], fontsize=9, fontweight='bold')

fig.suptitle('S-panel — pixel-level raw ion counts',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('output/paired_roi_pixel_spanel.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: output/paired_roi_pixel_spanel.png")


# -----------------------------------------------------------
# Figure 3: All markers per panel (T-panel view)
# -----------------------------------------------------------
print("\n=== Figure 3: T-panel all markers ===")
t_markers = SHARED_MARKERS + T_SPECIFIC
n_t = len(t_markers)
fig, axes = plt.subplots(n_t, n_rois, figsize=(n_rois * 4, n_t * 3.2))

for j, (roi_name, roi_info) in enumerate(ROIS.items()):
    txt_file = DATA_T / roi_info['T']
    print(f"  Loading T {roi_name}")
    image, markers, meta = load_roi_txt(txt_file)

    for i, marker in enumerate(t_markers):
        ax = axes[i, j]
        if marker in markers:
            ch_idx = markers.index(marker)
            img = image[:, :, ch_idx]
            vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1.0
            vmax = max(vmax, 0.1)
            ax.imshow(img, cmap='magma', vmin=0, vmax=vmax,
                     interpolation='nearest', aspect='equal')
        else:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(marker, fontsize=10, fontweight='bold')
        if i == 0:
            ax.set_title(roi_info['label'], fontsize=9, fontweight='bold')

fig.suptitle('T-panel — pixel-level raw ion counts',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('output/paired_roi_pixel_tpanel.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: output/paired_roi_pixel_tpanel.png")


# -----------------------------------------------------------
# Figure 4: DNA overlay for tissue context
# -----------------------------------------------------------
print("\n=== Figure 4: DNA composite + CD20 overlay ===")
fig, axes = plt.subplots(3, n_rois, figsize=(n_rois * 4, 3 * 3.5))

for j, (roi_name, roi_info) in enumerate(ROIS.items()):
    # DNA from S-panel (row 0)
    img_s, markers_s, _ = load_roi_txt(DATA_S / roi_info['S'])
    img_t, markers_t, _ = load_roi_txt(DATA_T / roi_info['T'])

    # Row 0: DNA composite (S-panel)
    ax = axes[0, j]
    dna1_idx = markers_s.index('DNA1') if 'DNA1' in markers_s else None
    dna2_idx = markers_s.index('DNA2') if 'DNA2' in markers_s else None
    if dna1_idx is not None and dna2_idx is not None:
        dna = img_s[:, :, dna1_idx] + img_s[:, :, dna2_idx]
        vmax = np.percentile(dna[dna > 0], 99) if (dna > 0).any() else 1.0
        ax.imshow(dna, cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
        ax.set_ylabel('DNA (S-panel)', fontsize=10, fontweight='bold')
    ax.set_title(roi_info['label'], fontsize=9, fontweight='bold')

    # Row 1: CD20 from S-panel
    ax = axes[1, j]
    if 'CD20' in markers_s:
        cd20_s = img_s[:, :, markers_s.index('CD20')]
        vmax = np.percentile(cd20_s[cd20_s > 0], 99) if (cd20_s > 0).any() else 1.0
        ax.imshow(cd20_s, cmap='magma', vmin=0, vmax=max(vmax, 0.1), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
        ax.set_ylabel('CD20 (S-panel)', fontsize=10, fontweight='bold')

    # Row 2: CD20 from T-panel
    ax = axes[2, j]
    if 'CD20' in markers_t:
        cd20_t = img_t[:, :, markers_t.index('CD20')]
        vmax = np.percentile(cd20_t[cd20_t > 0], 99) if (cd20_t > 0).any() else 1.0
        ax.imshow(cd20_t, cmap='magma', vmin=0, vmax=max(vmax, 0.1), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
        ax.set_ylabel('CD20 (T-panel)', fontsize=10, fontweight='bold')

fig.suptitle('DNA tissue context + CD20 comparison (pixel-level)',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('output/paired_roi_pixel_dna_cd20.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: output/paired_roi_pixel_dna_cd20.png")

print("\nDone! All figures saved.")
