#!/usr/bin/env python3
"""Cross-panel concordance figure: T-panel vs S-panel on serial sections.

Demonstrates that both antibody panels see the same biology by comparing:
  - Spatial marker patterns via DNA-based registration
  - Cell type proportions at global and per-ROI level

Output: output/qc/fig_cross_panel_concordance.png

Panels (cached-panel composite):
  (a) DNA overlay — registered serial sections
  (b) CD20 spatial concordance (S-panel vs T-panel, smoothed)
  (c) Global cell type composition bars
  (d) Per-ROI scatter: 2×2 grid (CD4 T, CD8 T, B cells, Myeloid)
  (e) B cell concordance improves with Unidentified

Usage:
    python scripts/cross_panel_figure.py \
        --t-panel output/all_TMA_T_global_v8.h5ad \
        --s-panel output/all_TMA_S_global_v8.h5ad
"""

import argparse, os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from collections import Counter
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.ndimage import shift as ndi_shift
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_roi_txt

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TMA_COLORS = {'A1': '#4e79a7', 'B1': '#f28e2b', 'C1': '#e15759', 'Biomax': '#76b7b2'}
TMA_ORDER = ['A1', 'B1', 'C1', 'Biomax']

BROAD_COLORS = {
    'B cells': '#1f77b4', 'CD4 T': '#ff7f0e', 'CD8 T': '#d62728',
    'Myeloid': '#7f7f7f', 'Stromal': '#9467bd', 'Unidentified': '#D3D3D3',
    'Other': '#c49c94',
}

LQ = 'Low quality / Unassigned'

# Registration ROIs (B1 only — we have raw TXT for these)
DATA_T = Path('data/raw/TMA_B1_T')
DATA_S = Path('data/raw/TMA_B1_S')
REG_ROIS = {
    'B1_FL32': {
        'T': '20220118_CT14_09_B1_Tcellpanel_4_FL32_R_5.txt',
        'S': '20210518_CT14_09_B1_Stromalpanel_5_FL32_R_2.txt',
        'label': 'FL32 (concordant)',
    },
}
SHARED_MARKERS = ['CD20', 'CD4', 'CD8a', 'CD68']
SMOOTH_SIGMA = 20

# Panel width for uniform font scaling
PW = 10


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


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, f'$\\bf{{{letter}}}$', transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')


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


CONSOLIDATE_MAP = {
    'B cells (CD20hi)': 'B cells', 'B cells (CXCR5hi)': 'B cells',
    'B cells (weak CD20)': 'B cells', 'B cells (TOXhi)': 'B cells',
    'B cells (BCL2+)': 'B cells', 'B cells (PAX5+)': 'B cells',
    'B cells (dim)': 'B cells', 'T cells': 'Other',
}


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


def get_tma(sid):
    sid = str(sid)
    if sid.startswith('A1_'):
        return 'A1'
    elif sid.startswith('B1_'):
        return 'B1'
    elif sid.startswith('C1_'):
        return 'C1'
    elif 'biomax' in sid.lower() or 'Biomax' in sid:
        return 'Biomax'
    return 'Unknown'


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

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


def register_roi(roi_info):
    img_s, markers_s, _ = load_roi_txt(DATA_S / roi_info['S'])
    img_t, markers_t, _ = load_roi_txt(DATA_T / roi_info['T'])

    dna_s = get_dna_composite(img_s, markers_s)
    dna_t = get_dna_composite(img_t, markers_t)

    h = max(dna_s.shape[0], dna_t.shape[0]) + 100
    w = max(dna_s.shape[1], dna_t.shape[1]) + 100

    def pad(img, th, tw):
        if img.ndim == 3:
            return np.pad(img, ((50, th - img.shape[0] - 50),
                                (50, tw - img.shape[1] - 50), (0, 0)))
        return np.pad(img, ((50, th - img.shape[0] - 50),
                            (50, tw - img.shape[1] - 50)))

    dna_s_p = pad(dna_s, h, w)
    dna_t_p = pad(dna_t, h, w)
    img_s_p = pad(img_s, h, w)
    img_t_p = pad(img_t, h, w)

    mask_s = tissue_mask(dna_s_p)
    mask_t = tissue_mask(dna_t_p)

    com_s = np.array(center_of_mass(mask_s.astype(float)))
    com_t = np.array(center_of_mass(mask_t.astype(float)))
    shift_yx = com_s - com_t

    dna_t_reg = ndi_shift(dna_t_p, shift_yx, order=1, mode='constant', cval=0)
    mask_t_reg = ndi_shift(mask_t.astype(float), shift_yx, order=1, mode='constant', cval=0) > 0.5

    def shift_3d(img, s):
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = ndi_shift(img[:, :, c], s, order=1, mode='constant', cval=0)
        return out

    img_t_reg = shift_3d(img_t_p, shift_yx)
    overlap = mask_s & mask_t_reg

    marker_corrs = {}
    marker_imgs = {}
    for marker in SHARED_MARKERS:
        if marker in markers_s and marker in markers_t:
            ch_s = gaussian_filter(img_s_p[:, :, markers_s.index(marker)].astype(float), sigma=SMOOTH_SIGMA)
            ch_t = gaussian_filter(img_t_reg[:, :, markers_t.index(marker)].astype(float), sigma=SMOOTH_SIGMA)
            vals_s = ch_s[overlap]
            vals_t = ch_t[overlap]
            if vals_s.std() > 0 and vals_t.std() > 0:
                corr = float(np.corrcoef(vals_s, vals_t)[0, 1])
            else:
                corr = 0.0
            marker_corrs[marker] = corr
            marker_imgs[marker] = (ch_s, ch_t)

    return {
        'dna_s': dna_s_p, 'dna_t_reg': dna_t_reg,
        'mask_s': mask_s, 'mask_t_reg': mask_t_reg, 'overlap': overlap,
        'marker_corrs': marker_corrs, 'marker_imgs': marker_imgs,
        'shift': shift_yx,
    }


# ---------------------------------------------------------------------------
# Panel render + paste helpers
# ---------------------------------------------------------------------------

def _render_panel(panel_id, plot_fn, plot_args, figsize, cache_dir, force=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{panel_id}.png"
    if path.exists() and not force:
        print(f"    [{panel_id}] cached")
        return path
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax, *plot_args)
    panel_label(ax, panel_id)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    fig.savefig(str(path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [{panel_id}] rendered → {path.name}")
    return path


def _render_panel_fig(panel_id, plot_fn, plot_args, figsize, cache_dir, force=False):
    """Render a figure-level panel (plot_fn receives fig, not ax)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{panel_id}.png"
    if path.exists() and not force:
        print(f"    [{panel_id}] cached")
        return path
    fig = plt.figure(figsize=figsize)
    plot_fn(fig, *plot_args)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    fig.savefig(str(path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [{panel_id}] rendered → {path.name}")
    return path


def _paste_panel(ax, png_path):
    img = mpimg.imread(str(png_path))
    ax.imshow(img)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Individual panel plot functions
# ---------------------------------------------------------------------------

def _plot_dna_overlay(ax, reg, roi_info):
    """(a) DNA overlay — registered serial sections."""
    h, w = reg['dna_s'].shape
    rgb = np.zeros((h, w, 3))
    s_n = reg['dna_s'] / (np.percentile(reg['dna_s'][reg['dna_s'] > 0], 99) + 1e-8)
    t_n = reg['dna_t_reg'] / (np.percentile(reg['dna_t_reg'][reg['dna_t_reg'] > 0], 99) + 1e-8)
    rgb[:, :, 0] = np.clip(t_n, 0, 1)   # magenta = T
    rgb[:, :, 1] = np.clip(s_n, 0, 1)   # green = S
    rgb[:, :, 2] = np.clip(t_n, 0, 1)
    ax.imshow(rgb)
    ax.set_title(f'DNA overlay: {roi_info["label"]}\n(green=S, magenta=T)', fontsize=14,
                 fontweight='medium')
    ax.axis('off')


def _plot_cd20_concordance(fig, reg):
    """(b) CD20 spatial concordance — side-by-side S-panel vs T-panel."""
    cd20_s, cd20_t = reg['marker_imgs']['CD20']
    cd20_corr = reg['marker_corrs']['CD20']

    gs = GridSpec(1, 2, figure=fig, wspace=0.02, left=0.05, right=0.95,
                  top=0.88, bottom=0.02)

    for si, (img_data, pname) in enumerate([
        (cd20_s, 'S-panel'), (cd20_t, 'T-panel'),
    ]):
        ax = fig.add_subplot(gs[0, si])
        display = img_data.copy()
        display[~reg['overlap']] = 0
        vmax = np.percentile(display[reg['overlap']], 99) if reg['overlap'].any() else 1
        ax.imshow(display, cmap='magma', vmin=0, vmax=max(vmax, 0.01))
        ax.set_title(f'CD20 {pname} (σ=20µm)', fontsize=14, fontweight='medium')
        ax.axis('off')
        if si == 1:
            ax.text(0.5, -0.03, f'r = {cd20_corr:.3f}', transform=ax.transAxes,
                    ha='center', fontsize=13, fontweight='bold')
    # Panel label on first subplot
    fig.axes[0].text(-0.02, 1.02, '$\\bf{b}$', transform=fig.axes[0].transAxes,
                     fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')


def _plot_global_composition(ax, t_pcts, s_pcts, display_cats):
    """(c) Global cell type composition bars."""
    x = np.arange(len(display_cats))
    w = 0.35
    ax.bar(x - w / 2, t_pcts, w, color=[BROAD_COLORS[c] for c in display_cats],
           edgecolor='black', linewidth=0.5, label='T-panel')
    ax.bar(x + w / 2, s_pcts, w, color=[BROAD_COLORS[c] for c in display_cats],
           edgecolor='black', linewidth=0.5, alpha=0.5, label='S-panel')
    ax.set_xticks(x)
    ax.set_xticklabels(display_cats, fontsize=TICK_SIZE, rotation=35, ha='right')
    ax.set_ylabel('% of tumor cells', fontsize=LABEL_SIZE)
    ax.set_title('Global cell type composition', fontsize=TITLE_SIZE, fontweight='medium')
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_per_roi_scatter(fig, t_props, s_props, common, tma_labels):
    """(d) Per-ROI scatter: 2×2 grid."""
    gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45,
                  left=0.10, right=0.95, top=0.92, bottom=0.08)
    scatter_types = ['CD4 T', 'CD8 T', 'B cells', 'Myeloid']

    for i, ct in enumerate(scatter_types):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        t_vals = np.array([t_props[s].get(ct, 0) * 100 for s in common])
        s_vals = np.array([s_props[s].get(ct, 0) * 100 for s in common])

        for tma in TMA_ORDER:
            mask = np.array([tma_labels[s] == tma for s in common])
            if mask.sum() > 0:
                ax.scatter(t_vals[mask], s_vals[mask], c=TMA_COLORS[tma],
                           s=25, alpha=0.7, edgecolors='none', label=tma)

        rp, _ = pearsonr(t_vals, s_vals)
        maxv = max(t_vals.max(), s_vals.max(), 1) * 1.05
        ax.plot([0, maxv], [0, maxv], 'k--', alpha=0.3, lw=1.0)
        ax.set_title(f'{ct} (r={rp:.2f})', fontsize=13)
        ax.set_xlabel('T-panel %', fontsize=LABEL_SIZE)
        ax.set_ylabel('S-panel %', fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 0:
            ax.legend(fontsize=LEGEND_SIZE, loc='upper left', markerscale=1.5)

    # Panel label on first subplot
    fig.axes[0].text(-0.02, 1.02, r'$\bf{d}$', transform=fig.axes[0].transAxes,
                     fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')


def _plot_bcell_concordance(ax, t_props, s_props, common, lq_threshold):
    """(e) B cell concordance improves with Unidentified."""
    t_b = np.array([t_props[s].get('B cells', 0) for s in common])
    s_b = np.array([s_props[s].get('B cells', 0) for s in common])
    t_lq = np.array([t_props[s].get('Unidentified', 0) for s in common])
    s_lq = np.array([s_props[s].get('Unidentified', 0) for s in common])
    high_lq = (t_lq > lq_threshold) | (s_lq > lq_threshold)
    good = ~high_lq

    r_all, _ = pearsonr(t_b, s_b)
    r_good, _ = pearsonr(t_b[good], s_b[good]) if good.sum() >= 5 else (r_all, None)
    t_blq = t_b + t_lq
    s_blq = s_b + s_lq
    r_blq, _ = pearsonr(t_blq, s_blq)

    labels_c = ['B cells\n(all ROIs)', f'B cells\n(good ROIs\nn={good.sum()})', 'B + Unident.\n(all ROIs)']
    vals_c = [r_all, r_good, r_blq]
    colors_c = ['#1f77b4', '#2ca02c', '#ff7f0e']
    bars = ax.bar(range(3), vals_c, color=colors_c, edgecolor='black', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, vals_c):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'r={val:.2f}',
                ha='center', fontsize=13, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels_c, fontsize=TICK_SIZE)
    ax.set_ylabel('Pearson r', fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.0)
    ax.set_title('B cell concordance\nimproves with Unidentified', fontsize=TITLE_SIZE,
                 fontweight='medium')
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.axhline(0.5, color='gray', ls=':', lw=1.0, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Cross-panel concordance figure')
    parser.add_argument('--t-panel', required=True, help='T-panel v8 h5ad')
    parser.add_argument('--s-panel', required=True, help='S-panel v8 h5ad')
    parser.add_argument('--output-dir', default='output/qc', help='Output directory')
    parser.add_argument('--lq-threshold', type=float, default=0.30)
    parser.add_argument('--no-cache', action='store_true', help='Force re-render all panels')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = Path(args.output_dir) / '_cache_fig_s2'
    force = args.no_cache

    # --- Load cell type data ---
    print('Loading data ...')
    f_t = h5py.File(args.t_panel, 'r')
    f_s = h5py.File(args.s_panel, 'r')

    t_sids = load_array(f_t, 'sample_id')
    s_sids = load_array(f_s, 'sample_id')
    t_cts = consolidate_cell_types(load_array(f_t, 'cell_type'))
    s_cts = consolidate_cell_types(load_array(f_s, 'cell_type'))

    t_tumor = np.array([is_tumor_core(s) for s in t_sids])
    s_tumor = np.array([is_tumor_core(s) for s in s_sids])

    t_broad = np.array([broad_map(ct) for ct in t_cts])
    s_broad = np.array([broad_map(ct) for ct in s_cts])

    # Per-ROI proportions
    broad_cats = ['B cells', 'CD4 T', 'CD8 T', 'Myeloid', 'Stromal', 'Unidentified', 'Other']

    def roi_proportions(sids, broads, tumor_mask):
        props = {}
        for sid in sorted(set(sids[tumor_mask])):
            m = (sids == sid) & tumor_mask
            n = m.sum()
            if n < 50:
                continue
            bc = Counter(broads[m])
            props[sid] = {cat: bc.get(cat, 0) / n for cat in broad_cats}
        return props

    t_props = roi_proportions(t_sids, t_broad, t_tumor)
    s_props = roi_proportions(s_sids, s_broad, s_tumor)
    common = sorted(set(t_props.keys()) & set(s_props.keys()))
    tma_labels = {s: get_tma(s) for s in common}
    print(f'  Paired tumor ROIs: {len(common)}')

    # Global composition
    display_cats = ['B cells', 'CD4 T', 'CD8 T', 'Myeloid', 'Stromal', 'Unidentified']
    t_global = Counter(t_broad[t_tumor])
    s_global = Counter(s_broad[s_tumor])
    n_t = t_tumor.sum()
    n_s = s_tumor.sum()
    t_pcts = [t_global.get(c, 0) / n_t * 100 for c in display_cats]
    s_pcts = [s_global.get(c, 0) / n_s * 100 for c in display_cats]

    f_t.close()
    f_s.close()

    # --- Registration ---
    has_raw = DATA_T.exists() and DATA_S.exists()
    reg = None
    if has_raw:
        print('  Registering paired ROIs ...')
        roi_name = 'B1_FL32'
        roi_info = REG_ROIS[roi_name]
        reg = register_roi(roi_info)
        print(f'    {roi_name}: shift=({reg["shift"][0]:.0f},{reg["shift"][1]:.0f})')
        for m, r in reg['marker_corrs'].items():
            print(f'      {m}: r={r:.3f}')
    else:
        print('  WARNING: Raw TXT files not found, skipping spatial panels (a, b)')

    # --- Render panels ---
    print('Rendering panels ...')
    panel_paths = {}

    # (a) DNA overlay — same height as CD20 panels
    if reg is not None:
        panel_paths['a'] = _render_panel(
            'a', _plot_dna_overlay, [reg, REG_ROIS['B1_FL32']],
            (8, 8), cache_dir, force=force)

    # (b) CD20 concordance (figure-level: two sub-axes, tighter spacing)
    if reg is not None:
        panel_paths['b'] = _render_panel_fig(
            'b', _plot_cd20_concordance, [reg],
            (16, 8), cache_dir, force=force)

    # (c) Global composition
    panel_paths['c'] = _render_panel(
        'c', _plot_global_composition, [t_pcts, s_pcts, display_cats],
        (PW, 7), cache_dir, force=force)

    # (d) Per-ROI scatter (figure-level: 2×2 grid)
    panel_paths['d'] = _render_panel_fig(
        'd', _plot_per_roi_scatter, [t_props, s_props, common, tma_labels],
        (PW, 8), cache_dir, force=force)

    # (e) B cell concordance
    panel_paths['e'] = _render_panel(
        'e', _plot_bcell_concordance, [t_props, s_props, common, args.lq_threshold],
        (PW, 7), cache_dir, force=force)

    # Print concordance stats
    scatter_types = ['CD4 T', 'CD8 T', 'B cells', 'Myeloid']
    print(f'  Per-ROI correlations (n={len(common)}):')
    for ct in scatter_types:
        t_v = np.array([t_props[s].get(ct, 0) for s in common])
        s_v = np.array([s_props[s].get(ct, 0) for s in common])
        rp, pp = pearsonr(t_v, s_v)
        rs, ps = spearmanr(t_v, s_v)
        print(f'    {ct:>10s}: Pearson r={rp:.3f} (p={pp:.4f}), Spearman ρ={rs:.3f}')

    # --- Composite assembly ---
    print('Assembling composite ...')

    # Row heights: a=8, b=8, c=7, d=8, e=7
    # Row 1: a + b (side by side, but b is 2x wide — use 1:2 ratio)
    # Row 2: c + d (side by side, d is scatter grid)
    # Row 3: e alone (half width)
    # But a is square-ish image, b is wide → put them in same row with width_ratios

    fig = plt.figure(figsize=(20, 24))

    gap = 0.008
    usable = 0.99
    # Row heights: row1 (a+b)=8, row2 (c+d)=8, row3 (e)=7
    total_h = 8 + 8 + 7
    h1 = usable * 8 / total_h
    h2 = usable * 8 / total_h
    h3 = usable * 7 / total_h

    top1 = 0.995
    bot1 = top1 - h1
    top2 = bot1 - gap
    bot2 = top2 - h2
    top3 = bot2 - gap
    bot3 = top3 - h3

    # Row 1: (a) DNA overlay + (b) CD20 concordance
    if 'a' in panel_paths and 'b' in panel_paths:
        gs1 = GridSpec(1, 2, figure=fig, left=0.005, right=0.995,
                       top=top1, bottom=bot1, wspace=0.02, width_ratios=[1, 2])
        ax = fig.add_subplot(gs1[0, 0])
        _paste_panel(ax, panel_paths['a'])
        ax = fig.add_subplot(gs1[0, 1])
        _paste_panel(ax, panel_paths['b'])

    # Row 2: (c) global composition + (d) per-ROI scatter
    gs2 = GridSpec(1, 2, figure=fig, left=0.005, right=0.995,
                   top=top2, bottom=bot2, wspace=0.02)
    ax = fig.add_subplot(gs2[0, 0])
    _paste_panel(ax, panel_paths['c'])
    ax = fig.add_subplot(gs2[0, 1])
    _paste_panel(ax, panel_paths['d'])

    # Row 3: (e) B cell concordance (left half only)
    gs3 = GridSpec(1, 2, figure=fig, left=0.005, right=0.995,
                   top=top3, bottom=bot3, wspace=0.02)
    ax = fig.add_subplot(gs3[0, 0])
    _paste_panel(ax, panel_paths['e'])
    ax_empty = fig.add_subplot(gs3[0, 1])
    ax_empty.axis('off')

    out = Path(args.output_dir) / 'fig_cross_panel_concordance.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(str(out).replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')

    import subprocess
    subprocess.run(['open', '-a', 'Preview', str(out)])


if __name__ == '__main__':
    main()
