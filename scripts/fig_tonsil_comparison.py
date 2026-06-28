#!/usr/bin/env python3
"""Compare FL tumor cores vs normal tonsil controls for key evasion metrics.

Generates a 2x3 supplementary figure:
  (a) CD8 T cell density by compartment — FL vs tonsil
  (b) Exhaustion fraction (TOX+PD-1+) by compartment — FL vs tonsil
  (c) Treg fraction by compartment — FL vs tonsil
  (d) CD14 expression on FDCs — FL vs tonsil (S-panel)
  (e) Compartment composition — FL vs tonsil (stacked bars)
  (f) FDC fraction by compartment — FL vs tonsil (S-panel)

Usage:
    .venv/bin/python scripts/fig_tonsil_comparison.py \
        --t-utag output/all_TMA_T_utag_ct_merged.h5ad \
        --s-utag output/all_TMA_S_utag_ct_merged.h5ad \
        --output-dir output/hypotheses_v8
"""

import argparse
import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu


from figure_style import (TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
                          ANNOT_SIZE, PANEL_LABEL_SIZE)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_array(f, key):
    ds = f['obs'][key]
    if isinstance(ds, h5py.Group) and 'categories' in ds:
        cats = ds['categories'][:]
        codes = ds['codes'][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c)
                             for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v)
                     for v in vals])


def get_marker_names(f):
    key = '_index' if '_index' in f['var'] else 'index'
    names = f['var'][key][:]
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def is_tonsil(sample_id):
    s = sample_id.lower()
    return 'tonsil' in s or '_ton_' in s


def is_tumor(sample_id):
    control_tags = ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal',
                    '_ton_', '_adr_']
    return not any(t in sample_id.lower() for t in control_tags)


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f'$\\bf{{{letter}}}$', transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')


def pval_str(p):
    if p < 0.0001:
        return f"p={p:.1e}"
    elif p < 0.001:
        return f"p={p:.4f}"
    else:
        return f"p={p:.3f}"


# ── T-panel compartment order ───────────────────────────────────────────────

GRADIENT_ORDER = [
    'GC core',
    'Follicle core (GC/CD20hi/CXCR5hi)',
    'Follicle mantle (CXCR5hi)',
    'B cell follicle (CD20hi/CXCR5hi)',
    'B cell zone',
    'Follicle-T zone interface',
    'Treg-enriched T zone',
    'T cell zone (CD4/CD8)',
    'Macrophage-rich zone',
]

GRADIENT_SHORT = [
    'GC\ncore', 'Follicle\ncore', 'Follicle\nmantle',
    'B cell\nfollicle', 'B cell\nzone', 'Foll-T\ninterface',
    'Treg\nT zone', 'T cell\nzone', 'Mac\nzone',
]

FOLL_INTER_BOUNDARY = 5

CD8_ALL = ['CD8 T cells', 'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)',
           'Macrophages (GzmB+)']
CD8_EXHAUSTED = ['CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)']


# ── S-panel compartment order ───────────────────────────────────────────────

S_COMPARTMENTS = [
    'B cell zone (BCL2+)', 'B cell zone (PAX5+)',
    'FDC network zone', 'FDC / myeloid zone',
    'T cell zone', 'Stromal / CAF zone',
]

S_SHORT = [
    'B zone\n(BCL2+)', 'B zone\n(PAX5+)', 'FDC\nnetwork',
    'FDC/\nmyeloid', 'T cell\nzone', 'Stromal\n/ CAF',
]


# ── Data extraction ─────────────────────────────────────────────────────────

def extract_t_panel(path):
    """Extract T-panel data for FL and tonsil."""
    with h5py.File(path, 'r') as f:
        sample_ids = load_array(f, 'sample_id')
        cell_types = load_array(f, 'cell_type')
        compartments = load_array(f, 'compartment_name')

        tonsil_mask = np.array([is_tonsil(s) for s in sample_ids])
        tumor_mask = np.array([is_tumor(s) for s in sample_ids])

        # Exclude Unassigned for composition
        typed_mask = ~np.isin(cell_types, ['Unassigned', 'Low quality / Unassigned'])

        data = {
            'sample_ids': sample_ids,
            'cell_types': cell_types,
            'compartments': compartments,
            'tonsil_mask': tonsil_mask,
            'tumor_mask': tumor_mask,
            'typed_mask': typed_mask,
        }

    return data


def extract_s_panel(path):
    """Extract S-panel data for FL and tonsil."""
    with h5py.File(path, 'r') as f:
        sample_ids = load_array(f, 'sample_id')
        cell_types = load_array(f, 'cell_type')
        compartments = load_array(f, 'compartment_name')

        markers = get_marker_names(f)
        X = f['X']

        cd14_idx = markers.index('CD14') if 'CD14' in markers else None

        tonsil_mask = np.array([is_tonsil(s) for s in sample_ids])
        tumor_mask = np.array([is_tumor(s) for s in sample_ids])
        typed_mask = ~np.isin(cell_types, ['Unassigned', 'Low quality / Unassigned'])

        data = {
            'sample_ids': sample_ids,
            'cell_types': cell_types,
            'compartments': compartments,
            'tonsil_mask': tonsil_mask,
            'tumor_mask': tumor_mask,
            'typed_mask': typed_mask,
        }

        # Load CD14 for FDC analysis
        if cd14_idx is not None:
            n = len(sample_ids)
            cd14_vals = np.zeros(n, dtype=np.float32)
            chunk = 50000
            for i in range(0, n, chunk):
                end = min(i + chunk, n)
                block = X[i:end, :]
                if hasattr(block, 'toarray'):
                    block = block.toarray()
                cd14_vals[i:end] = block[:, cd14_idx]
            data['cd14'] = cd14_vals

    return data


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_paired_bars(ax, fl_vals, ton_vals, short_labels, ylabel, title,
                     boundary=None):
    """Side-by-side bars for FL vs tonsil."""
    if boundary is None:
        boundary = FOLL_INTER_BOUNDARY
    x = np.arange(len(short_labels))
    w = 0.35
    ax.bar(x - w/2, fl_vals, w, color='#E41A1C', alpha=0.8,
           edgecolor='black', linewidth=0.5, label='FL tumor')
    ax.bar(x + w/2, ton_vals, w, color='#377EB8', alpha=0.8,
           edgecolor='black', linewidth=0.5, label='Normal tonsil')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.axvline(boundary - 0.5, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')

    # Add fold-change annotations for compartments where both are > 0.5%
    for i in range(len(fl_vals)):
        if fl_vals[i] is not None and ton_vals[i] is not None:
            if not np.isnan(fl_vals[i]) and not np.isnan(ton_vals[i]):
                if ton_vals[i] > 0.5 and fl_vals[i] > 0.5:
                    fold = fl_vals[i] / ton_vals[i]
                    if abs(fold - 1.0) > 0.2:
                        ymax = max(fl_vals[i], ton_vals[i])
                        ax.text(i, ymax + 0.3, f'{fold:.1f}x',
                                ha='center', va='bottom', fontsize=8,
                                color='black')


def compute_compartment_fracs(compartments, cell_types, typed_mask,
                              tissue_mask, target_types, order):
    """Compute fraction of target types per compartment."""
    vals = []
    for comp in order:
        mask = (compartments == comp) & tissue_mask & typed_mask
        n_total = mask.sum()
        if n_total < 20:
            vals.append(np.nan)
            continue
        n_target = (mask & np.isin(cell_types, target_types)).sum()
        vals.append(float(n_target) / n_total * 100)
    return vals


# ── Standalone importable plot functions ─────────────────────────────────────

def plot_tonsil_exclusion(ax, t_data):
    """Panel: CD8 T cell exclusion gradient — FL vs tonsil (paired bars)."""
    fl_cd8 = compute_compartment_fracs(
        t_data['compartments'], t_data['cell_types'], t_data['typed_mask'],
        t_data['tumor_mask'], CD8_ALL, GRADIENT_ORDER)
    ton_cd8 = compute_compartment_fracs(
        t_data['compartments'], t_data['cell_types'], t_data['typed_mask'],
        t_data['tonsil_mask'], CD8_ALL, GRADIENT_ORDER)
    plot_paired_bars(ax, fl_cd8, ton_cd8, GRADIENT_SHORT,
                     'CD8 T cells (% of typed)', 'Immune exclusion gradient')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fl_cd8, ton_cd8


def plot_tonsil_exhaustion(ax, t_data):
    """Panel: CD8 T cell exhaustion — FL vs tonsil (paired bars)."""
    fl_exh = []
    ton_exh = []
    for comp in GRADIENT_ORDER:
        # FL
        mask = ((t_data['compartments'] == comp) & t_data['tumor_mask'] &
                np.isin(t_data['cell_types'], CD8_ALL))
        n_cd8 = mask.sum()
        if n_cd8 < 10:
            fl_exh.append(np.nan)
        else:
            exh = ((t_data['compartments'] == comp) & t_data['tumor_mask'] &
                   np.isin(t_data['cell_types'], CD8_EXHAUSTED))
            fl_exh.append(float(exh.sum()) / n_cd8 * 100)
        # Tonsil
        mask_t = ((t_data['compartments'] == comp) & t_data['tonsil_mask'] &
                  np.isin(t_data['cell_types'], CD8_ALL))
        n_cd8_t = mask_t.sum()
        if n_cd8_t < 10:
            ton_exh.append(np.nan)
        else:
            exh_t = ((t_data['compartments'] == comp) & t_data['tonsil_mask'] &
                     np.isin(t_data['cell_types'], CD8_EXHAUSTED))
            ton_exh.append(float(exh_t.sum()) / n_cd8_t * 100)
    plot_paired_bars(ax, fl_exh, ton_exh, GRADIENT_SHORT,
                     'TOX+PD-1+ (% of CD8 T)',
                     'CD8 T cell exhaustion topography')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fl_exh, ton_exh


def plot_tonsil_treg(ax, t_data):
    """Panel: Treg distribution — FL vs tonsil (paired bars)."""
    fl_treg = compute_compartment_fracs(
        t_data['compartments'], t_data['cell_types'], t_data['typed_mask'],
        t_data['tumor_mask'], ['Treg'], GRADIENT_ORDER)
    ton_treg = compute_compartment_fracs(
        t_data['compartments'], t_data['cell_types'], t_data['typed_mask'],
        t_data['tonsil_mask'], ['Treg'], GRADIENT_ORDER)
    plot_paired_bars(ax, fl_treg, ton_treg, GRADIENT_SHORT,
                     'Treg (% of typed)', 'Treg distribution')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fl_treg, ton_treg


def plot_tonsil_cd14_fdc(ax, s_data):
    """Panel: CD14 expression on FDCs — FL vs tonsil (boxplot)."""
    fdc_types = ['FDC', 'FDC (CD14+)', 'FDC (CXCL13+)', 'FDC (CD21+)']
    s_cts_unique = np.unique(s_data['cell_types'])
    fdc_present = [t for t in fdc_types if t in s_cts_unique]
    if not fdc_present:
        fdc_present = [t for t in s_cts_unique if 'FDC' in t or 'fdc' in t.lower()]

    fl_fdc_mask = s_data['tumor_mask'] & np.isin(s_data['cell_types'], fdc_present)
    ton_fdc_mask = s_data['tonsil_mask'] & np.isin(s_data['cell_types'], fdc_present)

    fl_cd14_fdc = s_data['cd14'][fl_fdc_mask]
    ton_cd14_fdc = s_data['cd14'][ton_fdc_mask]

    bp = ax.boxplot([fl_cd14_fdc, ton_cd14_fdc],
                    labels=['FL tumor\nFDCs', 'Normal tonsil\nFDCs'],
                    patch_artist=True, widths=0.5, showfliers=False,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#E41A1C')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#377EB8')
    bp['boxes'][1].set_alpha(0.6)

    if len(fl_cd14_fdc) > 10 and len(ton_cd14_fdc) > 10:
        stat, p = mannwhitneyu(fl_cd14_fdc, ton_cd14_fdc, alternative='two-sided')
        # Bracket + P-value between the two boxes
        q75_fl = np.percentile(fl_cd14_fdc, 75)
        q75_ton = np.percentile(ton_cd14_fdc, 75)
        iqr_fl = q75_fl - np.percentile(fl_cd14_fdc, 25)
        ymax = max(q75_fl + 1.5 * iqr_fl,
                   q75_ton + 1.5 * (q75_ton - np.percentile(ton_cd14_fdc, 25)))
        ymax *= 1.05
        ax.plot([1, 1, 2, 2], [ymax, ymax * 1.02, ymax * 1.02, ymax],
                color='black', lw=1.2)
        ax.text(1.5, ymax * 1.03, pval_str(p),
                ha='center', va='bottom', fontsize=ANNOT_SIZE)

    ax.set_ylabel('CD14 intensity (z-scored)', fontsize=LABEL_SIZE)
    ax.set_title('CD14 expression on FDCs', fontsize=TITLE_SIZE, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fl_cd14_fdc, ton_cd14_fdc


def plot_tonsil_composition(ax, t_data):
    """Panel: Overall cell type composition — FL vs tonsil (stacked bars)."""
    broad_cats = {
        'B cells': lambda ct: any(x in ct for x in ['B cell', 'GC B', 'Activated B',
                                                      'Plasmablast']),
        'CD4 T': lambda ct: ct == 'CD4 T cells',
        'CD8 T': lambda ct: ct in CD8_ALL,
        'Treg': lambda ct: ct == 'Treg',
        'Macrophage': lambda ct: 'Mac' in ct or 'acrophage' in ct,
        'Other': lambda ct: True,
    }
    cat_colors = ['#4DAF4A', '#377EB8', '#E41A1C', '#984EA3',
                  '#A65628', '#999999']

    for tissue_label, tissue_mask, x_pos in [('FL tumor', t_data['tumor_mask'], 0),
                                              ('Tonsil', t_data['tonsil_mask'], 1)]:
        mask = tissue_mask & t_data['typed_mask']
        cts = t_data['cell_types'][mask]
        total = len(cts)
        if total == 0:
            continue

        bottom = 0
        for cat_name, cat_fn, color in zip(broad_cats.keys(),
                                            broad_cats.values(),
                                            cat_colors):
            if cat_name == 'Other':
                assigned = set()
                for cn, cf in list(broad_cats.items())[:-1]:
                    assigned.update([ct for ct in cts if cf(ct)])
                n_cat = sum(1 for ct in cts if ct not in assigned)
            else:
                n_cat = sum(1 for ct in cts if cat_fn(ct))
            frac = n_cat / total * 100
            ax.bar(x_pos, frac, bottom=bottom, color=color, edgecolor='white',
                   linewidth=0.5, width=0.35,
                   label=cat_name if x_pos == 0 else '')
            if frac > 3:
                ax.text(x_pos, bottom + frac/2, f'{frac:.0f}%',
                        ha='center', va='center', fontsize=8, color='white',
                        fontweight='bold')
            bottom += frac

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['FL tumor', 'Normal tonsil'], fontsize=TICK_SIZE)
    ax.set_ylabel('% of typed cells', fontsize=LABEL_SIZE)
    ax.set_title('Overall cell type composition', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlim(-0.5, 2.8)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right', frameon=False)
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Main figure ──────────────────────────────────────────────────────────────

def make_figure(t_data, s_data, output_dir):
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.06, right=0.96, top=0.93, bottom=0.06)

    fig.suptitle('FL Tumor vs Normal Tonsil: Key Evasion Metrics',
                 fontsize=14, fontweight='bold', y=0.98)

    # (a) CD8 exclusion
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, 'a')
    plot_tonsil_exclusion(ax_a, t_data)

    # (b) Exhaustion
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, 'b')
    plot_tonsil_exhaustion(ax_b, t_data)

    # (c) Treg
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, 'c')
    plot_tonsil_treg(ax_c, t_data)

    # (d) CD14 on FDCs
    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, 'd')
    plot_tonsil_cd14_fdc(ax_d, s_data)

    # (e) Composition
    ax_e = fig.add_subplot(gs[1, 1])
    panel_label(ax_e, 'e')
    plot_tonsil_composition(ax_e, t_data)

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, 'fig_tonsil_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FL vs tonsil comparison')
    parser.add_argument('--t-utag', required=True,
                        help='T-panel UTAG h5ad')
    parser.add_argument('--s-utag', required=True,
                        help='S-panel UTAG h5ad')
    parser.add_argument('--output-dir', default='output/hypotheses_v8')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading T-panel...")
    t_data = extract_t_panel(args.t_utag)
    n_ton_t = t_data['tonsil_mask'].sum()
    n_fl_t = t_data['tumor_mask'].sum()
    print(f"  T-panel: {n_fl_t:,} FL cells, {n_ton_t:,} tonsil cells")

    print("Loading S-panel...")
    s_data = extract_s_panel(args.s_utag)
    n_ton_s = s_data['tonsil_mask'].sum()
    n_fl_s = s_data['tumor_mask'].sum()
    print(f"  S-panel: {n_fl_s:,} FL cells, {n_ton_s:,} tonsil cells")

    print("Generating figure...")
    out = make_figure(t_data, s_data, args.output_dir)

    # Open in Preview
    os.system(f'open -a Preview "{out}"')


if __name__ == '__main__':
    main()
