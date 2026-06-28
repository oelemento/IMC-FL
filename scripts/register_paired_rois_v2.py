#!/usr/bin/env python3
"""Register paired ROIs v2: center-of-mass alignment + coarse-grain comparison.

Serial sections don't have pixel-to-pixel correspondence, so we:
1. Align tissue centers of mass
2. Smooth images to ~20µm resolution (IMC pixel = 1µm)
3. Compare spatial patterns in overlapping tissue at coarse scale
"""
import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import shift as ndi_shift, gaussian_filter, center_of_mass
from src.data_loader import load_roi_txt

DATA_T = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_T')
DATA_S = Path('/Users/ole2001/PROGRAMS/IMC-FL/data/raw/TMA_B1_S')

ROIS = {
    'B1_FL32': {
        'T': '20220118_CT14_09_B1_Tcellpanel_4_FL32_R_5.txt',
        'S': '20210518_CT14_09_B1_Stromalpanel_5_FL32_R_2.txt',
        'label': 'FL32 (concordant)',
    },
    'B1_FL18': {
        'T': '20220118_CT14_09_B1_Tcellpanel_2_FL18_L_12.txt',
        'S': '20210518_CT14_09_B1_Stromalpanel_2_FL18_L_6.txt',
        'label': 'FL18 (moderate)',
    },
    'B1_FL10': {
        'T': '20220118_CT14_09_B1_Tcellpanel_2_FL10_R_3.txt',
        'S': '20210518_CT14_09_B1_Stromalpanel_2_FL10_R_3.txt',
        'label': 'FL10 (moderate)',
    },
    'B1_FL26': {
        'T': '20220118_CT14_09_B1_Tcellpanel_3_FL26_L_11.txt',
        'S': '20210518_CT14_09_B1_Stromalpanel_4_FL26_L_5.txt',
        'label': 'FL26 (discordant)',
    },
}

SHARED_MARKERS = ['CD20', 'CD4', 'CD8a', 'CD68']
SMOOTH_SIGMA = 20  # ~20µm Gaussian smoothing for pattern comparison


def get_dna_composite(image, markers):
    channels = []
    for m in ['DNA1', 'DNA2']:
        if m in markers:
            channels.append(image[:, :, markers.index(m)])
    return sum(channels)


def tissue_mask(dna, sigma=5, quantile=0.15):
    smoothed = gaussian_filter(dna.astype(float), sigma=sigma)
    thresh = np.quantile(smoothed[smoothed > 0], quantile) if (smoothed > 0).any() else 0
    return smoothed > thresh


def apply_shift_2d(img, shift_yx):
    return ndi_shift(img, shift_yx, order=1, mode='constant', cval=0)


def apply_shift_3d(img, shift_yx):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = apply_shift_2d(img[:, :, c], shift_yx)
    return out


# Process all ROIs
results = {}

for roi_name, roi_info in ROIS.items():
    print(f"\n=== {roi_name} ({roi_info['label']}) ===")

    img_s, markers_s, meta_s = load_roi_txt(DATA_S / roi_info['S'])
    img_t, markers_t, meta_t = load_roi_txt(DATA_T / roi_info['T'])
    print(f"  S: {img_s.shape[1]}x{img_s.shape[0]}, T: {img_t.shape[1]}x{img_t.shape[0]}")

    dna_s = get_dna_composite(img_s, markers_s)
    dna_t = get_dna_composite(img_t, markers_t)

    # Pad to same canvas
    h = max(dna_s.shape[0], dna_t.shape[0]) + 100  # extra margin for shift
    w = max(dna_s.shape[1], dna_t.shape[1]) + 100

    def pad(img, target_h, target_w):
        if img.ndim == 3:
            return np.pad(img, ((50, target_h - img.shape[0] - 50),
                                (50, target_w - img.shape[1] - 50),
                                (0, 0)))
        return np.pad(img, ((50, target_h - img.shape[0] - 50),
                            (50, target_w - img.shape[1] - 50)))

    dna_s_p = pad(dna_s, h, w)
    dna_t_p = pad(dna_t, h, w)
    img_s_p = pad(img_s, h, w)
    img_t_p = pad(img_t, h, w)

    # Tissue masks
    mask_s = tissue_mask(dna_s_p)
    mask_t = tissue_mask(dna_t_p)

    # Center-of-mass alignment
    com_s = np.array(center_of_mass(mask_s.astype(float)))
    com_t = np.array(center_of_mass(mask_t.astype(float)))
    shift_yx = com_s - com_t
    print(f"  Center-of-mass shift: dy={shift_yx[0]:.1f}, dx={shift_yx[1]:.1f}")

    # Apply shift to T-panel
    dna_t_reg = apply_shift_2d(dna_t_p, shift_yx)
    img_t_reg = apply_shift_3d(img_t_p, shift_yx)
    mask_t_reg = apply_shift_2d(mask_t.astype(float), shift_yx) > 0.5

    # Overlap
    overlap = mask_s & mask_t_reg
    area_s = mask_s.sum()
    area_t = mask_t_reg.sum()
    area_overlap = overlap.sum()
    pct_s = area_overlap / area_s * 100 if area_s > 0 else 0
    pct_t = area_overlap / area_t * 100 if area_t > 0 else 0
    print(f"  Overlap: {area_overlap:,}px ({pct_s:.1f}% of S, {pct_t:.1f}% of T)")

    # Coarse-grain marker comparison
    print(f"  Marker pattern correlations (σ={SMOOTH_SIGMA}µm smoothing):")
    marker_corrs = {}
    marker_imgs = {}
    for marker in SHARED_MARKERS:
        if marker in markers_s and marker in markers_t:
            ch_s = img_s_p[:, :, markers_s.index(marker)].astype(float)
            ch_t = img_t_reg[:, :, markers_t.index(marker)].astype(float)

            # Smooth
            ch_s_smooth = gaussian_filter(ch_s, sigma=SMOOTH_SIGMA)
            ch_t_smooth = gaussian_filter(ch_t, sigma=SMOOTH_SIGMA)

            # Correlate in overlap
            vals_s = ch_s_smooth[overlap]
            vals_t = ch_t_smooth[overlap]

            if vals_s.std() > 0 and vals_t.std() > 0:
                corr = np.corrcoef(vals_s, vals_t)[0, 1]
            else:
                corr = 0.0
            marker_corrs[marker] = corr
            marker_imgs[marker] = (ch_s_smooth, ch_t_smooth)
            print(f"    {marker}: r={corr:.3f}")

    results[roi_name] = {
        'shift': shift_yx, 'overlap': overlap,
        'mask_s': mask_s, 'mask_t_reg': mask_t_reg,
        'dna_s': dna_s_p, 'dna_t_reg': dna_t_reg,
        'img_s': img_s_p, 'img_t_reg': img_t_reg,
        'markers_s': markers_s, 'markers_t': markers_t,
        'marker_corrs': marker_corrs, 'marker_imgs': marker_imgs,
        'pct_s': pct_s, 'pct_t': pct_t,
    }

# ========================================================
# Figure 1: Registration + overlap + CD20 pattern comparison
# ========================================================
print("\n=== Generating figures ===")
n_rois = len(ROIS)

fig, axes = plt.subplots(5, n_rois, figsize=(n_rois * 5, 5 * 4))

for j, (roi_name, r) in enumerate(results.items()):
    h, w = r['dna_s'].shape

    # Row 0: DNA overlay (green=S, magenta=T)
    ax = axes[0, j]
    rgb = np.zeros((h, w, 3))
    s_n = r['dna_s'] / (np.percentile(r['dna_s'][r['dna_s'] > 0], 99) + 1e-8)
    t_n = r['dna_t_reg'] / (np.percentile(r['dna_t_reg'][r['dna_t_reg'] > 0], 99) + 1e-8)
    rgb[:, :, 0] = np.clip(t_n, 0, 1)
    rgb[:, :, 1] = np.clip(s_n, 0, 1)
    rgb[:, :, 2] = np.clip(t_n, 0, 1)
    ax.imshow(rgb)
    ax.set_title(f"{ROIS[roi_name]['label']}\nshift=({r['shift'][0]:.0f},{r['shift'][1]:.0f})",
                fontsize=9, fontweight='bold')
    if j == 0:
        ax.set_ylabel('DNA overlay\n(green=S, magenta=T)', fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # Row 1: Overlap mask
    ax = axes[1, j]
    mask_rgb = np.zeros((h, w, 3))
    mask_rgb[r['mask_s'] & ~r['mask_t_reg']] = [0, 1, 0]
    mask_rgb[r['mask_t_reg'] & ~r['mask_s']] = [1, 0, 1]
    mask_rgb[r['overlap']] = [1, 1, 1]
    ax.imshow(mask_rgb)
    ax.set_title(f"overlap: {r['pct_s']:.0f}% S, {r['pct_t']:.0f}% T", fontsize=8)
    if j == 0:
        ax.set_ylabel('Tissue mask\n(white=overlap)', fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2: Smoothed CD20 S (overlap only)
    ax = axes[2, j]
    if 'CD20' in r['marker_imgs']:
        cd20_s_sm, cd20_t_sm = r['marker_imgs']['CD20']
        display = cd20_s_sm.copy()
        display[~r['overlap']] = 0
        vmax = np.percentile(display[r['overlap']], 99) if r['overlap'].any() else 1
        ax.imshow(display, cmap='magma', vmin=0, vmax=max(vmax, 0.01))
    if j == 0:
        ax.set_ylabel(f'CD20 (S) smoothed\nσ={SMOOTH_SIGMA}µm', fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # Row 3: Smoothed CD20 T (overlap only)
    ax = axes[3, j]
    if 'CD20' in r['marker_imgs']:
        display = cd20_t_sm.copy()
        display[~r['overlap']] = 0
        vmax = np.percentile(display[r['overlap']], 99) if r['overlap'].any() else 1
        ax.imshow(display, cmap='magma', vmin=0, vmax=max(vmax, 0.01))
    corr = r['marker_corrs'].get('CD20', 0)
    ax.set_title(f'CD20 r={corr:.3f}', fontsize=9)
    if j == 0:
        ax.set_ylabel(f'CD20 (T) smoothed\nσ={SMOOTH_SIGMA}µm', fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # Row 4: Pixel-level scatter CD20 S vs T in overlap
    ax = axes[4, j]
    if 'CD20' in r['markers_s'] and 'CD20' in r['markers_t']:
        cd20_s_raw = r['img_s'][:, :, r['markers_s'].index('CD20')]
        cd20_t_raw = r['img_t_reg'][:, :, r['markers_t'].index('CD20')]
        # Downsample: average in 20x20 blocks within overlap
        block = 20
        rows = h // block
        cols = w // block
        s_blocks = []
        t_blocks = []
        for bi in range(rows):
            for bj in range(cols):
                patch_mask = r['overlap'][bi*block:(bi+1)*block, bj*block:(bj+1)*block]
                if patch_mask.sum() > block * block * 0.5:  # >50% tissue
                    s_val = cd20_s_raw[bi*block:(bi+1)*block, bj*block:(bj+1)*block][patch_mask].mean()
                    t_val = cd20_t_raw[bi*block:(bi+1)*block, bj*block:(bj+1)*block][patch_mask].mean()
                    s_blocks.append(s_val)
                    t_blocks.append(t_val)
        s_blocks = np.array(s_blocks)
        t_blocks = np.array(t_blocks)
        if len(s_blocks) > 10 and s_blocks.std() > 0 and t_blocks.std() > 0:
            block_corr = np.corrcoef(s_blocks, t_blocks)[0, 1]
            ax.scatter(s_blocks, t_blocks, s=3, alpha=0.3, color='steelblue')
            mx = max(s_blocks.max(), t_blocks.max())
            ax.plot([0, mx], [0, mx], 'r--', alpha=0.5, linewidth=1)
            ax.set_title(f'CD20 block r={block_corr:.3f} (n={len(s_blocks)})', fontsize=9)
            ax.set_xlabel('S-panel', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'n={len(s_blocks)}', transform=ax.transAxes, ha='center')
    if j == 0:
        ax.set_ylabel(f'CD20 S vs T\n(20×20µm blocks)', fontsize=9, fontweight='bold')

fig.suptitle('Cross-panel registration (center-of-mass) + coarse CD20 pattern comparison',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('output/paired_roi_registered_v2_cd20.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: output/paired_roi_registered_v2_cd20.png")


# ========================================================
# Figure 2: All shared markers — smoothed S vs T in overlap
# ========================================================
n_markers = len(SHARED_MARKERS)
fig, axes = plt.subplots(n_markers, n_rois * 2, figsize=(n_rois * 2 * 3.5, n_markers * 3.5))

for j, (roi_name, r) in enumerate(results.items()):
    for mi, marker in enumerate(SHARED_MARKERS):
        if marker not in r['marker_imgs']:
            continue
        sm_s, sm_t = r['marker_imgs'][marker]
        corr = r['marker_corrs'][marker]

        # S-panel column
        ax = axes[mi, j * 2]
        display = sm_s.copy()
        display[~r['overlap']] = 0
        vmax = np.percentile(display[r['overlap']], 99) if r['overlap'].any() else 1
        ax.imshow(display, cmap='magma', vmin=0, vmax=max(vmax, 0.01))
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.set_title(f"{ROIS[roi_name]['label']}\nS-panel", fontsize=8, fontweight='bold')
        if j == 0:
            ax.set_ylabel(marker, fontsize=10, fontweight='bold')

        # T-panel column
        ax = axes[mi, j * 2 + 1]
        display = sm_t.copy()
        display[~r['overlap']] = 0
        vmax = np.percentile(display[r['overlap']], 99) if r['overlap'].any() else 1
        ax.imshow(display, cmap='magma', vmin=0, vmax=max(vmax, 0.01))
        ax.set_xticks([]); ax.set_yticks([])
        if mi == 0:
            ax.set_title(f"T-panel\nr={corr:.3f}", fontsize=8, fontweight='bold')
        else:
            ax.set_title(f"r={corr:.3f}", fontsize=8)

fig.suptitle(f'Registered shared markers (σ={SMOOTH_SIGMA}µm smoothing, overlap region only)',
             fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('output/paired_roi_registered_v2_all_markers.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: output/paired_roi_registered_v2_all_markers.png")


# Summary
print("\n=== Summary (coarse-grain pattern correlations, σ=20µm) ===")
print(f"{'ROI':>12} {'Shift':>14} {'Overlap':>8}", end='')
for m in SHARED_MARKERS:
    print(f' {m:>8}', end='')
print()
print('-' * 72)
for roi_name, r in results.items():
    s = f"({r['shift'][0]:.0f},{r['shift'][1]:.0f})"
    o = f"{min(r['pct_s'], r['pct_t']):.0f}%"
    print(f"{roi_name:>12} {s:>14} {o:>8}", end='')
    for m in SHARED_MARKERS:
        print(f" {r['marker_corrs'].get(m, 0):>8.3f}", end='')
    print()

print("\nDone!")
