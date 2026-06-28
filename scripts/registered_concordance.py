#!/usr/bin/env python3
"""Registration-aware cross-panel concordance.

For each paired ROI (same tissue core, serial sections):
1. Load raw TXT pixel data for both T and S panels
2. Register via DNA center-of-mass alignment
3. Compute overlap mask
4. Filter cells whose centroids fall in the overlap region
5. Recompute per-ROI cell type proportions using only overlapping cells
6. Save results to CSV for figure generation

Usage (on Cayuga):
    python scripts/registered_concordance.py \
        --t-panel output/all_TMA_T_global_v8.h5ad \
        --s-panel output/all_TMA_S_global_v8.h5ad \
        --output output/registered_concordance.csv
"""

import argparse, os, sys, time, glob
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.ndimage import shift as ndi_shift
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_roi_txt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DATA_ROOT = Path('<DATA_ROOT>')

# TMA → raw data directory mapping (T and S panels)
TMA_DIRS = {
    'A1': {
        'T': RAW_DATA_ROOT / 'March_11_2021_FL_TMA_A1_T',
        'S': RAW_DATA_ROOT / 'March_9_2021_FL_TMA_A1_S',
    },
    'B1': {
        'T': RAW_DATA_ROOT / 'Jan 18 2022_FL_TMA_B1_T',
        'S': RAW_DATA_ROOT / 'May_18_2021_FL_TMA_B1_S',
    },
    'C1': {
        'T': RAW_DATA_ROOT / 'Jan_28_2022_FL_TMA_C1_T',
        'S': RAW_DATA_ROOT / 'Dec21_2021_FL_TMA_C1_S',
    },
}

SHARED_MARKERS = ['CD20', 'CD4', 'CD8a', 'CD68']
SMOOTH_SIGMA = 20
LQ = 'Low quality / Unassigned'

CONSOLIDATE_MAP = {
    'B cells (CD20hi)': 'B cells', 'B cells (CXCR5hi)': 'B cells',
    'B cells (weak CD20)': 'B cells', 'B cells (TOXhi)': 'B cells',
    'B cells (BCL2+)': 'B cells', 'B cells (PAX5+)': 'B cells',
    'B cells (dim)': 'B cells', 'T cells': 'Other',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_array(f, key):
    ds = f['obs'][key]
    if isinstance(ds, h5py.Group) and 'categories' in ds:
        cats = ds['categories'][:]
        codes = ds['codes'][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


def load_numeric(f, key):
    return f['obs'][key][:]


def consolidate_cell_types(cell_types):
    return np.array([CONSOLIDATE_MAP.get(ct, ct) for ct in cell_types])


def broad_map(ct):
    ct = str(ct)
    if 'CD4 T' in ct or 'Treg' in ct:
        return 'CD4 T'
    if 'CD8 T' in ct:
        return 'CD8 T'
    if any(x in ct for x in ['Macrophage', 'Myeloid', 'Dendritic', 'Histiocyte', 'pDC', 'M1 ', 'M2 ']):
        return 'Myeloid'
    if 'B cell' in ct or 'GC B' in ct or 'PAX5' in ct or 'BCL2' in ct or 'Activated B' in ct:
        return 'B cells'
    if 'FDC' in ct or 'Stromal' in ct or 'Endothelial' in ct or 'FRC' in ct:
        return 'Stromal'
    if 'Low quality' in ct or 'Unassigned' in ct:
        return 'Unidentified'
    return 'Other'


def is_tumor_core(sample_id):
    s_lower = sample_id.lower()
    if '_ton_' in s_lower or '_adr_' in s_lower:
        return False
    for tissue in ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal']:
        if tissue in s_lower:
            return False
    if sample_id == 'Biomax_ROI_006':
        return False
    return True


def get_tma(sid):
    for prefix in ['A1', 'B1', 'C1']:
        if sid.startswith(prefix + '_'):
            return prefix
    return None


def extract_core_name(sid):
    """Extract core name (e.g., 'FL32') from sample_id like 'B1_FL32'."""
    parts = sid.split('_', 1)
    if len(parts) == 2:
        return parts[1]
    return sid


def find_txt_file(data_dir, core_name):
    """Find the TXT file matching a core name in a directory."""
    pattern = f'*{core_name}*'
    matches = list(data_dir.glob(pattern))
    # Filter to .txt files only
    txt_matches = [m for m in matches if m.suffix == '.txt']
    if len(txt_matches) == 1:
        return txt_matches[0]
    elif len(txt_matches) > 1:
        # Prefer exact match (core name followed by _ or .)
        for m in txt_matches:
            fname = m.stem
            # Check core_name appears as a word boundary
            idx = fname.find(core_name)
            if idx >= 0:
                after = idx + len(core_name)
                if after >= len(fname) or fname[after] in ('_', '.', ' '):
                    return m
        return txt_matches[0]  # fallback: first match
    return None


def tissue_mask(dna, sigma=5, quantile=0.15):
    smoothed = gaussian_filter(dna.astype(float), sigma=sigma)
    thresh = np.quantile(smoothed[smoothed > 0], quantile) if (smoothed > 0).any() else 0
    return smoothed > thresh


def get_dna_composite(image, markers):
    channels = []
    for m in ['DNA1', 'DNA2']:
        if m in markers:
            channels.append(image[:, :, markers.index(m)])
    return sum(channels)


def register_and_overlap(txt_t, txt_s):
    """Register two ROIs and return overlap mask + shift.

    Returns dict with: overlap_mask_s (in S coords), shift_yx, img shapes,
    and marker correlations.
    """
    img_s, markers_s, _ = load_roi_txt(txt_s)
    img_t, markers_t, _ = load_roi_txt(txt_t)

    dna_s = get_dna_composite(img_s, markers_s)
    dna_t = get_dna_composite(img_t, markers_t)

    # Pad to common canvas
    h = max(dna_s.shape[0], dna_t.shape[0]) + 100
    w = max(dna_s.shape[1], dna_t.shape[1]) + 100

    def pad2d(img, th, tw):
        return np.pad(img, ((50, th - img.shape[0] - 50),
                            (50, tw - img.shape[1] - 50)))

    def pad3d(img, th, tw):
        return np.pad(img, ((50, th - img.shape[0] - 50),
                            (50, tw - img.shape[1] - 50), (0, 0)))

    dna_s_p = pad2d(dna_s, h, w)
    dna_t_p = pad2d(dna_t, h, w)

    mask_s = tissue_mask(dna_s_p)
    mask_t = tissue_mask(dna_t_p)

    # Center-of-mass alignment
    com_s = np.array(center_of_mass(mask_s.astype(float)))
    com_t = np.array(center_of_mass(mask_t.astype(float)))
    shift_yx = com_s - com_t

    mask_t_reg = ndi_shift(mask_t.astype(float), shift_yx, order=1, mode='constant', cval=0) > 0.5
    overlap = mask_s & mask_t_reg

    area_s = mask_s.sum()
    area_t = mask_t_reg.sum()
    area_overlap = overlap.sum()
    pct_overlap = area_overlap / min(area_s, area_t) * 100 if min(area_s, area_t) > 0 else 0

    # Marker correlations (smoothed)
    img_s_p = pad3d(img_s, h, w)
    img_t_p = pad3d(img_t, h, w)

    # Shift T-panel image
    def shift_3d(img, s):
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = ndi_shift(img[:, :, c], s, order=1, mode='constant', cval=0)
        return out

    img_t_reg = shift_3d(img_t_p, shift_yx)

    marker_corrs = {}
    for marker in SHARED_MARKERS:
        if marker in markers_s and marker in markers_t:
            ch_s = gaussian_filter(img_s_p[:, :, markers_s.index(marker)].astype(float), sigma=SMOOTH_SIGMA)
            ch_t = gaussian_filter(img_t_reg[:, :, markers_t.index(marker)].astype(float), sigma=SMOOTH_SIGMA)
            vals_s = ch_s[overlap]
            vals_t = ch_t[overlap]
            if len(vals_s) > 10 and vals_s.std() > 0 and vals_t.std() > 0:
                marker_corrs[marker] = float(np.corrcoef(vals_s, vals_t)[0, 1])
            else:
                marker_corrs[marker] = np.nan

    return {
        'shift_yx': shift_yx,
        'overlap_padded': overlap,
        'mask_s_padded': mask_s,
        'pct_overlap': pct_overlap,
        'img_shape_s': img_s.shape[:2],
        'img_shape_t': img_t.shape[:2],
        'marker_corrs': marker_corrs,
        'pad_offset': 50,  # how much padding was added
    }


def cell_in_overlap(cx, cy, overlap_mask, pad_offset=50):
    """Check which cells (by centroid) fall in the overlap region.

    Centroids are in original image coordinates; overlap_mask is in padded coords.
    """
    # Shift centroids to padded coordinates
    px = (cx + pad_offset).astype(int)
    py = (cy + pad_offset).astype(int)

    h, w = overlap_mask.shape
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    in_overlap = np.zeros(len(cx), dtype=bool)
    in_overlap[valid] = overlap_mask[py[valid], px[valid]]
    return in_overlap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--t-panel', required=True, help='T-panel v8 h5ad')
    parser.add_argument('--s-panel', required=True, help='S-panel v8 h5ad')
    parser.add_argument('--output', default='output/registered_concordance.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # --- Load h5ad cell data ---
    print('Loading h5ad data ...')
    f_t = h5py.File(args.t_panel, 'r')
    f_s = h5py.File(args.s_panel, 'r')

    t_sids = load_array(f_t, 'sample_id')
    s_sids = load_array(f_s, 'sample_id')
    t_cts = consolidate_cell_types(load_array(f_t, 'cell_type'))
    s_cts = consolidate_cell_types(load_array(f_s, 'cell_type'))
    t_cx = load_numeric(f_t, 'centroid_x')
    t_cy = load_numeric(f_t, 'centroid_y')
    s_cx = load_numeric(f_s, 'centroid_x')
    s_cy = load_numeric(f_s, 'centroid_y')

    t_broad = np.array([broad_map(ct) for ct in t_cts])
    s_broad = np.array([broad_map(ct) for ct in s_cts])

    f_t.close()
    f_s.close()

    # --- Find paired ROIs ---
    t_tumor_sids = set(s for s in set(t_sids) if is_tumor_core(s))
    s_tumor_sids = set(s for s in set(s_sids) if is_tumor_core(s))
    common = sorted(t_tumor_sids & s_tumor_sids)
    print(f'  Paired tumor ROIs: {len(common)}')

    # --- Build TXT file mapping ---
    print('Building TXT file mapping ...')
    txt_map = {}  # sample_id → {'T': Path, 'S': Path}
    skipped = []

    for sid in common:
        tma = get_tma(sid)
        if tma is None or tma not in TMA_DIRS:
            skipped.append((sid, 'no TMA dir'))
            continue

        core = extract_core_name(sid)
        t_txt = find_txt_file(TMA_DIRS[tma]['T'], core)
        s_txt = find_txt_file(TMA_DIRS[tma]['S'], core)

        if t_txt is None:
            skipped.append((sid, f'no T-panel TXT for {core}'))
            continue
        if s_txt is None:
            skipped.append((sid, f'no S-panel TXT for {core}'))
            continue

        txt_map[sid] = {'T': t_txt, 'S': s_txt}

    print(f'  Mapped: {len(txt_map)}, Skipped: {len(skipped)}')
    if skipped:
        for sid, reason in skipped[:10]:
            print(f'    SKIP: {sid} — {reason}')
        if len(skipped) > 10:
            print(f'    ... and {len(skipped) - 10} more')

    # --- Process each paired ROI ---
    broad_cats = ['B cells', 'CD4 T', 'CD8 T', 'Myeloid', 'Stromal', 'Unidentified', 'Other']
    results = []

    for i, sid in enumerate(sorted(txt_map.keys())):
        t0 = time.time()
        tma = get_tma(sid)
        print(f'  [{i+1}/{len(txt_map)}] {sid} ...', end=' ', flush=True)

        try:
            reg = register_and_overlap(txt_map[sid]['T'], txt_map[sid]['S'])
        except Exception as e:
            print(f'FAILED: {e}')
            continue

        # Filter T-panel cells in this ROI
        t_mask = t_sids == sid
        t_in_overlap = cell_in_overlap(t_cx[t_mask], t_cy[t_mask],
                                        reg['overlap_padded'], reg['pad_offset'])

        # Filter S-panel cells — S coords don't need shifting (overlap is in S-padded space)
        s_mask = s_sids == sid
        s_in_overlap = cell_in_overlap(s_cx[s_mask], s_cy[s_mask],
                                        reg['overlap_padded'], reg['pad_offset'])

        # Count cells
        n_t_total = t_mask.sum()
        n_t_overlap = t_in_overlap.sum()
        n_s_total = s_mask.sum()
        n_s_overlap = s_in_overlap.sum()

        # Proportions — all cells
        t_broad_roi = t_broad[t_mask]
        s_broad_roi = s_broad[s_mask]
        t_counts_all = Counter(t_broad_roi)
        s_counts_all = Counter(s_broad_roi)

        # Proportions — overlap only
        t_broad_overlap = t_broad_roi[t_in_overlap]
        s_broad_overlap = s_broad_roi[s_in_overlap]
        t_counts_ovl = Counter(t_broad_overlap)
        s_counts_ovl = Counter(s_broad_overlap)

        row = {
            'sample_id': sid,
            'tma': tma,
            'shift_dy': reg['shift_yx'][0],
            'shift_dx': reg['shift_yx'][1],
            'pct_overlap': reg['pct_overlap'],
            'n_t_total': n_t_total,
            'n_t_overlap': n_t_overlap,
            'n_s_total': n_s_total,
            'n_s_overlap': n_s_overlap,
        }
        # All-cell proportions
        for cat in broad_cats:
            row[f't_{cat}_all'] = t_counts_all.get(cat, 0) / max(n_t_total, 1)
            row[f's_{cat}_all'] = s_counts_all.get(cat, 0) / max(n_s_total, 1)
        # Overlap proportions
        for cat in broad_cats:
            row[f't_{cat}_ovl'] = t_counts_ovl.get(cat, 0) / max(n_t_overlap, 1)
            row[f's_{cat}_ovl'] = s_counts_ovl.get(cat, 0) / max(n_s_overlap, 1)
        # Marker correlations
        for marker in SHARED_MARKERS:
            row[f'corr_{marker}'] = reg['marker_corrs'].get(marker, np.nan)

        results.append(row)
        dt = time.time() - t0
        print(f'overlap={reg["pct_overlap"]:.0f}%, T={n_t_overlap}/{n_t_total}, '
              f'S={n_s_overlap}/{n_s_total}, {dt:.1f}s')

    # --- Save results ---
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f'\nSaved: {args.output} ({len(df)} ROIs)')

    # --- Summary statistics ---
    print('\n=== SUMMARY ===')
    print(f'ROIs processed: {len(df)}')
    print(f'Mean overlap: {df["pct_overlap"].mean():.1f}%')
    print(f'Median shift: dy={df["shift_dy"].median():.1f}, dx={df["shift_dx"].median():.1f}')

    for suffix, label in [('_all', 'All cells'), ('_ovl', 'Overlap only')]:
        print(f'\nPer-ROI correlations ({label}):')
        for cat in ['B cells', 'CD4 T', 'CD8 T', 'Myeloid']:
            t_vals = df[f't_{cat}{suffix}'].values
            s_vals = df[f's_{cat}{suffix}'].values
            if len(t_vals) >= 5:
                rp, pp = pearsonr(t_vals, s_vals)
                rs, _ = spearmanr(t_vals, s_vals)
                print(f'  {cat:>10s}: Pearson r={rp:.3f} (p={pp:.4f}), Spearman ρ={rs:.3f}')

    # B cell + Unidentified
    for suffix, label in [('_all', 'All cells'), ('_ovl', 'Overlap only')]:
        t_b = df[f't_B cells{suffix}'].values
        s_b = df[f's_B cells{suffix}'].values
        t_u = df[f't_Unidentified{suffix}'].values
        s_u = df[f's_Unidentified{suffix}'].values
        t_bu = t_b + t_u
        s_bu = s_b + s_u
        rp_b, _ = pearsonr(t_b, s_b)
        rp_bu, _ = pearsonr(t_bu, s_bu)
        print(f'\nB cell concordance ({label}): B-only r={rp_b:.3f}, B+Unident r={rp_bu:.3f}')


if __name__ == '__main__':
    main()
