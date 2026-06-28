#!/usr/bin/env python3
"""Register paired ROIs between S and T panels using DNA signal,
then compare shared markers in the overlapping tissue region."""
import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift, gaussian_filter
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


def get_dna_composite(image, markers):
    """Sum DNA1 + DNA2 channels."""
    channels = []
    for m in ['DNA1', 'DNA2']:
        if m in markers:
            channels.append(image[:, :, markers.index(m)])
    return sum(channels)


def tissue_mask(dna, sigma=3, quantile=0.1):
    """Binary tissue mask from DNA signal."""
    smoothed = gaussian_filter(dna.astype(float), sigma=sigma)
    thresh = np.quantile(smoothed[smoothed > 0], quantile) if (smoothed > 0).any() else 0
    return smoothed > thresh


def pad_to_match(img_a, img_b):
    """Pad smaller image to match larger image dimensions."""
    h = max(img_a.shape[0], img_b.shape[0])
    w = max(img_a.shape[1], img_b.shape[1])

    def pad(img):
        ph = h - img.shape[0]
        pw = w - img.shape[1]
        if img.ndim == 3:
            return np.pad(img, ((0, ph), (0, pw), (0, 0)))
        return np.pad(img, ((0, ph), (0, pw)))

    return pad(img_a), pad(img_b)


def register_dna(dna_ref, dna_mov):
    """Register moving DNA to reference DNA using phase cross-correlation.
    Returns (shift_y, shift_x) to apply to moving image."""
    # Smooth for robust registration
    ref = gaussian_filter(dna_ref.astype(float), sigma=5)
    mov = gaussian_filter(dna_mov.astype(float), sigma=5)

    # Normalize
    ref = ref / (ref.max() + 1e-8)
    mov = mov / (mov.max() + 1e-8)

    shift_est, error, diffphase = phase_cross_correlation(
        ref, mov, upsample_factor=10
    )
    print(f"    Registration shift: dy={shift_est[0]:.1f}, dx={shift_est[1]:.1f}, error={error:.4f}")
    return shift_est


def apply_shift(image, shift_yx):
    """Apply sub-pixel shift to 2D or 3D image."""
    if image.ndim == 3:
        shifted = np.zeros_like(image)
        for c in range(image.shape[2]):
            shifted[:, :, c] = ndi_shift(image[:, :, c], shift_yx, order=1, mode='constant')
        return shifted
    return ndi_shift(image, shift_yx, order=1, mode='constant')


# ========================================================
# Process each ROI
# ========================================================
results = {}

for roi_name, roi_info in ROIS.items():
    print(f"\n=== {roi_name} ({roi_info['label']}) ===")

    # Load both panels
    img_s, markers_s, meta_s = load_roi_txt(DATA_S / roi_info['S'])
    img_t, markers_t, meta_t = load_roi_txt(DATA_T / roi_info['T'])
    print(f"  S-panel: {img_s.shape[1]}x{img_s.shape[0]}, T-panel: {img_t.shape[1]}x{img_t.shape[0]}")

    # DNA composites
    dna_s = get_dna_composite(img_s, markers_s)
    dna_t = get_dna_composite(img_t, markers_t)

    # Pad to same size
    dna_s_p, dna_t_p = pad_to_match(dna_s, dna_t)
    img_s_p, img_t_p = pad_to_match(img_s, img_t)

    # Register T-panel onto S-panel (S = reference)
    print("  Registering T→S via DNA...")
    shift_yx = register_dna(dna_s_p, dna_t_p)

    # Apply shift to T-panel
    dna_t_reg = apply_shift(dna_t_p, shift_yx)
    img_t_reg = apply_shift(img_t_p, shift_yx)

    # Tissue masks
    mask_s = tissue_mask(dna_s_p)
    mask_t = tissue_mask(dna_t_reg)
    overlap = mask_s & mask_t

    area_s = mask_s.sum()
    area_t = mask_t.sum()
    area_overlap = overlap.sum()
    pct_overlap_s = area_overlap / area_s * 100 if area_s > 0 else 0
    pct_overlap_t = area_overlap / area_t * 100 if area_t > 0 else 0
    print(f"  Tissue area — S: {area_s:,}px, T: {area_t:,}px, overlap: {area_overlap:,}px")
    print(f"  Overlap coverage — {pct_overlap_s:.1f}% of S, {pct_overlap_t:.1f}% of T")

    # Compare shared markers in overlap region
    print("  Marker correlations in overlap region:")
    marker_corrs = {}
    for marker in SHARED_MARKERS:
        if marker in markers_s and marker in markers_t:
            ch_s = img_s_p[:, :, markers_s.index(marker)]
            ch_t = img_t_reg[:, :, markers_t.index(marker)]

            vals_s = ch_s[overlap]
            vals_t = ch_t[overlap]

            if vals_s.std() > 0 and vals_t.std() > 0:
                corr = np.corrcoef(vals_s, vals_t)[0, 1]
            else:
                corr = 0.0
            marker_corrs[marker] = corr
            print(f"    {marker}: r={corr:.3f} (n={overlap.sum():,} pixels)")

    results[roi_name] = {
        'shift': shift_yx,
        'area_s': area_s, 'area_t': area_t, 'area_overlap': area_overlap,
        'pct_overlap_s': pct_overlap_s, 'pct_overlap_t': pct_overlap_t,
        'marker_corrs': marker_corrs,
        'img_s': img_s_p, 'img_t_reg': img_t_reg, 'markers_s': markers_s,
        'markers_t': markers_t, 'dna_s': dna_s_p, 'dna_t_reg': dna_t_reg,
        'mask_s': mask_s, 'mask_t': mask_t, 'overlap': overlap,
    }


# ========================================================
# Figure 1: Registration diagnostic — DNA overlay + overlap mask
# ========================================================
print("\n=== Generating figures ===")
n_rois = len(ROIS)

fig, axes = plt.subplots(4, n_rois, figsize=(n_rois * 5, 4 * 4))

for j, (roi_name, r) in enumerate(results.items()):
    # Row 0: DNA S (green) + DNA T registered (magenta) overlay
    ax = axes[0, j]
    h, w = r['dna_s'].shape
    rgb = np.zeros((h, w, 3))
    s_norm = r['dna_s'] / (np.percentile(r['dna_s'][r['dna_s'] > 0], 99) + 1e-8)
    t_norm = r['dna_t_reg'] / (np.percentile(r['dna_t_reg'][r['dna_t_reg'] > 0], 99) + 1e-8)
    s_norm = np.clip(s_norm, 0, 1)
    t_norm = np.clip(t_norm, 0, 1)
    rgb[:, :, 0] = t_norm   # T = magenta (R)
    rgb[:, :, 1] = s_norm   # S = green (G)
    rgb[:, :, 2] = t_norm   # T = magenta (B)
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(f"{ROIS[roi_name]['label']}\nshift=({r['shift'][0]:.0f},{r['shift'][1]:.0f})", fontsize=9, fontweight='bold')
    if j == 0:
        ax.set_ylabel('DNA overlay\n(green=S, magenta=T)', fontsize=9, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Row 1: Overlap mask
    ax = axes[1, j]
    mask_rgb = np.zeros((h, w, 3))
    mask_rgb[r['mask_s'] & ~r['mask_t']] = [0, 1, 0]      # S only = green
    mask_rgb[r['mask_t'] & ~r['mask_s']] = [1, 0, 1]      # T only = magenta
    mask_rgb[r['overlap']] = [1, 1, 1]                      # overlap = white
    ax.imshow(mask_rgb, interpolation='nearest')
    ax.set_title(f"overlap: {r['pct_overlap_s']:.0f}% of S, {r['pct_overlap_t']:.0f}% of T", fontsize=8)
    if j == 0:
        ax.set_ylabel('Tissue mask\n(white=overlap)', fontsize=9, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Row 2: CD20 S in overlap
    ax = axes[2, j]
    if 'CD20' in r['markers_s']:
        cd20_s = r['img_s'][:, :, r['markers_s'].index('CD20')].copy()
        cd20_s[~r['overlap']] = 0
        vmax = np.percentile(cd20_s[r['overlap']], 99) if r['overlap'].any() else 1
        ax.imshow(cd20_s, cmap='magma', vmin=0, vmax=max(vmax, 0.1), interpolation='nearest')
    if j == 0:
        ax.set_ylabel('CD20 (S-panel)\noverlap only', fontsize=9, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Row 3: CD20 T registered in overlap
    ax = axes[3, j]
    if 'CD20' in r['markers_t']:
        cd20_t = r['img_t_reg'][:, :, r['markers_t'].index('CD20')].copy()
        cd20_t[~r['overlap']] = 0
        vmax = np.percentile(cd20_t[r['overlap']], 99) if r['overlap'].any() else 1
        ax.imshow(cd20_t, cmap='magma', vmin=0, vmax=max(vmax, 0.1), interpolation='nearest')
    if j == 0:
        ax.set_ylabel('CD20 (T-panel)\noverlap only', fontsize=9, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('Cross-panel registration (DNA-based) + CD20 in overlap region',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('output/paired_roi_registered_overview.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: output/paired_roi_registered_overview.png")


# ========================================================
# Figure 2: All shared markers in overlap — S vs T
# ========================================================
n_markers = len(SHARED_MARKERS)
fig, axes = plt.subplots(n_markers * 2, n_rois, figsize=(n_rois * 5, n_markers * 2 * 3))

for j, (roi_name, r) in enumerate(results.items()):
    for mi, marker in enumerate(SHARED_MARKERS):
        row_s = mi * 2
        row_t = mi * 2 + 1

        for row, panel, img_data, markers_list in [
            (row_s, 'S', r['img_s'], r['markers_s']),
            (row_t, 'T', r['img_t_reg'], r['markers_t']),
        ]:
            ax = axes[row, j]
            if marker in markers_list:
                ch = img_data[:, :, markers_list.index(marker)].copy()
                ch[~r['overlap']] = 0
                vmax = np.percentile(ch[r['overlap']], 99) if r['overlap'].any() else 1
                ax.imshow(ch, cmap='magma', vmin=0, vmax=max(vmax, 0.1), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                corr = r['marker_corrs'].get(marker, 0)
                ax.set_ylabel(f'{marker} ({panel})\nr={corr:.2f}' if row == row_s else f'{marker} ({panel})',
                             fontsize=9, fontweight='bold')
            if row == 0:
                ax.set_title(f"{ROIS[roi_name]['label']}", fontsize=9, fontweight='bold')

fig.suptitle('Registered shared markers in overlap region (S vs T)',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('output/paired_roi_registered_markers.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: output/paired_roi_registered_markers.png")


# ========================================================
# Summary table
# ========================================================
print("\n=== Summary ===")
print(f"{'ROI':>12} {'Shift':>12} {'Overlap%S':>10} {'Overlap%T':>10} ", end='')
for m in SHARED_MARKERS:
    print(f'{m:>8}', end='')
print()
print('-' * 70)
for roi_name, r in results.items():
    shift_str = f"({r['shift'][0]:.0f},{r['shift'][1]:.0f})"
    print(f"{roi_name:>12} {shift_str:>12} {r['pct_overlap_s']:>9.1f}% {r['pct_overlap_t']:>9.1f}% ", end='')
    for m in SHARED_MARKERS:
        corr = r['marker_corrs'].get(m, 0)
        print(f'{corr:>8.3f}', end='')
    print()

print("\nDone!")
