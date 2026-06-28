#!/usr/bin/env python3
"""Generate figures for hypothesis testing results.

Usage:
    python scripts/hypothesis_figures.py \
        --t-panel output/all_TMA_T_global_v7.h5ad \
        --s-panel output/all_TMA_S_global_v7.h5ad \
        --output-dir output/hypotheses
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats


TMA_COLORS = {'A1': '#4e79a7', 'B1': '#f28e2b', 'C1': '#e15759', 'Biomax': '#76b7b2'}
TMA_ORDER = ['A1', 'B1', 'C1', 'Biomax']


def read_obs_column(f, col):
    g = f['obs'][col]
    if isinstance(g, h5py.Group) and 'categories' in g:
        cats = [x.decode() if isinstance(x, bytes) else x for x in g['categories'][:]]
        codes = g['codes'][:]
        return np.array([cats[c] if c >= 0 else '' for c in codes])
    arr = g[:]
    if arr.dtype.kind in ('S', 'O'):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
    return arr


def read_raw_marker(f, marker_name):
    raw_var = [x.decode() if isinstance(x, bytes) else x
               for x in f['raw']['var']['_index'][:]]
    # Try exact match, then underscore variant
    for name in [marker_name, marker_name.replace('-', '_')]:
        if name in raw_var:
            idx = raw_var.index(name)
            return f['raw']['X'][:, idx].astype(np.float32)
    return None


def shannon(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def load_panel(path):
    """Load cell_type, sample_id, tma from h5ad."""
    f = h5py.File(path, 'r')
    ct = read_obs_column(f, 'cell_type')
    sid = read_obs_column(f, 'sample_id')
    tma = read_obs_column(f, 'tma')
    return f, ct, sid, tma


# ===================================================================
# H6a: Phenotypic Entropy
# ===================================================================

def figure_h6a(t_path, s_path, out_dir):
    print("Generating H6a figures...")

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    panel_data = {}

    for col_offset, (panel, path) in enumerate(
            [('T-panel', t_path), ('S-panel', s_path)]):

        f, ct, sid, tma = load_panel(path)
        f.close()

        exclude = {'Low quality / Unassigned', 'Mixed / Border cells'}
        good = np.array([c not in exclude for c in ct], dtype=bool)
        ct_names = sorted(set(ct[good].tolist()))
        ct_all_names = sorted(set(ct.tolist()))

        rows = []
        for roi in sorted(set(sid.tolist())):
            rmask = sid == roi
            total = rmask.sum()
            gmask = rmask & good
            n_good = gmask.sum()
            if n_good < 50:
                continue
            lq_frac = 1 - n_good / total
            vc_clean = pd.Series(ct[gmask].tolist()).value_counts(normalize=True)
            vc_all = pd.Series(ct[rmask].tolist()).value_counts(normalize=True)

            props_clean = np.array([vc_clean.get(c, 0.0) for c in ct_names])
            props_all = np.array([vc_all.get(c, 0.0) for c in ct_all_names])

            row = {'roi': roi, 'tma': tma[rmask][0],
                   'n_cells': int(total), 'lq_frac': float(lq_frac),
                   'H_all': float(shannon(props_all)),
                   'H_clean': float(shannon(props_clean))}
            # Store individual proportions for correlation
            for c in ct_names:
                row[c] = vc_clean.get(c, 0.0)
            rows.append(row)

        df = pd.DataFrame(rows)
        panel_data[panel] = df

        # --- Panel A/B: Boxplot of entropy by TMA ---
        ax = fig.add_subplot(gs[0, col_offset])
        tmas = [t for t in TMA_ORDER if t in df['tma'].values]
        data = [df[df['tma'] == t]['H_clean'].values for t in tmas]
        bp = ax.boxplot(data, tick_labels=tmas, patch_artist=True,
                        widths=0.6, medianprops=dict(color='black', linewidth=1.5))
        for patch, t in zip(bp['boxes'], tmas):
            patch.set_facecolor(TMA_COLORS[t])
            patch.set_alpha(0.6)
        for i, (t, d) in enumerate(zip(tmas, data)):
            jitter = np.random.normal(i + 1, 0.06, len(d))
            ax.scatter(jitter, d, alpha=0.4, s=12, color=TMA_COLORS[t],
                       edgecolors='none')
        kw_groups = [df[df['tma'] == t]['H_clean'].values for t in tmas]
        _, kw_p = stats.kruskal(*kw_groups)
        ax.set_ylabel('Shannon Entropy (bits)', fontsize=11)
        ax.set_title(f'{panel}\nKruskal-Wallis p={kw_p:.4f}', fontsize=12, fontweight='bold')
        ax.set_ylim(bottom=0)
        for i, t in enumerate(tmas):
            n = len(df[df['tma'] == t])
            ax.text(i + 1, -0.15, f'n={n}', ha='center', fontsize=8, color='gray')

    # --- Panel C: T vs S entropy correlation ---
    ax = fig.add_subplot(gs[0, 2])
    df_t = panel_data['T-panel']
    df_s = panel_data['S-panel']
    merged = df_t[['roi', 'tma', 'H_clean']].merge(
        df_s[['roi', 'H_clean']], on='roi', suffixes=('_T', '_S'))
    for t in TMA_ORDER:
        sub = merged[merged['tma'] == t]
        if len(sub) > 0:
            ax.scatter(sub['H_clean_T'], sub['H_clean_S'], alpha=0.5, s=18,
                       color=TMA_COLORS[t], label=t, edgecolors='none')
    if len(merged) > 2:
        r, p = stats.pearsonr(merged['H_clean_T'], merged['H_clean_S'])
        ax.set_title(f'T vs S Entropy\nr={r:.2f}, p={p:.1e}', fontsize=12, fontweight='bold')
        # Fit line
        m, b = np.polyfit(merged['H_clean_T'], merged['H_clean_S'], 1)
        x_range = np.array([merged['H_clean_T'].min(), merged['H_clean_T'].max()])
        ax.plot(x_range, m * x_range + b, 'k--', alpha=0.3)
    ax.set_xlabel('T-panel Entropy (bits)', fontsize=11)
    ax.set_ylabel('S-panel Entropy (bits)', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_aspect('equal')
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.plot([0, lim], [0, lim], 'k:', alpha=0.2)

    # --- Row 2: Entropy drivers (top correlates) ---
    for col_offset, (panel, df) in enumerate(panel_data.items()):
        ax = fig.add_subplot(gs[1, col_offset])
        ct_cols = [c for c in df.columns
                   if c not in ('roi', 'tma', 'n_cells', 'lq_frac',
                                'H_all', 'H_clean')]
        corrs = []
        for c in ct_cols:
            if df[c].std() > 0.001:
                r_val, p_val = stats.pearsonr(df['H_clean'], df[c])
                corrs.append((c, r_val, p_val))
        corrs.sort(key=lambda x: x[1])
        # Show top 8 positive and top 4 negative
        top_neg = [x for x in corrs if x[1] < 0][:4]
        top_pos = [x for x in corrs if x[1] > 0][-8:]
        show = top_neg + top_pos

        names = [x[0][:25] for x in show]
        vals = [x[1] for x in show]
        colors_bar = ['#e15759' if v < 0 else '#4e79a7' for v in vals]
        y_pos = range(len(show))
        ax.barh(y_pos, vals, color=colors_bar, alpha=0.7, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Pearson r with Entropy', fontsize=10)
        ax.set_title(f'{panel}: Entropy Drivers', fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(-0.8, 0.7)

    # --- Panel E: LQ fraction vs entropy ---
    ax = fig.add_subplot(gs[1, 2])
    for panel, df in panel_data.items():
        marker = 'o' if panel == 'T-panel' else 's'
        for t in TMA_ORDER:
            sub = df[df['tma'] == t]
            if len(sub) > 0:
                ax.scatter(sub['lq_frac'] * 100, sub['H_clean'], alpha=0.3, s=10,
                           color=TMA_COLORS[t], marker=marker, edgecolors='none')
    ax.set_xlabel('LQ fraction (%)', fontsize=11)
    ax.set_ylabel('Clean Entropy (bits)', fontsize=11)
    ax.set_title('LQ Fraction vs Entropy\n(o=T, □=S)', fontsize=12, fontweight='bold')

    # --- Row 3: Distribution of entropy + low-entropy composition ---
    for col_offset, (panel, df) in enumerate(panel_data.items()):
        ax = fig.add_subplot(gs[2, col_offset])
        for t in TMA_ORDER:
            sub = df[df['tma'] == t]
            if len(sub) > 0:
                ax.hist(sub['H_clean'], bins=15, alpha=0.5, label=t,
                        color=TMA_COLORS[t], density=True)
        ax.set_xlabel('Shannon Entropy (bits)', fontsize=11)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{panel}: Entropy Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)

    # Low-entropy composition pie
    ax = fig.add_subplot(gs[2, 2])
    # Use T-panel low entropy ROIs
    df_t = panel_data['T-panel']
    ct_cols = [c for c in df_t.columns
               if c not in ('roi', 'tma', 'n_cells', 'lq_frac', 'H_all', 'H_clean')]
    low_ent = df_t[df_t['H_clean'] < 2.0]
    high_ent = df_t[df_t['H_clean'] >= 2.5]

    if len(low_ent) > 0 and len(high_ent) > 0:
        low_means = low_ent[ct_cols].mean().sort_values(ascending=False)
        high_means = high_ent[ct_cols].mean().sort_values(ascending=False)
        # Show top 6 cell types
        top_ct = list(dict.fromkeys(
            list(low_means.index[:4]) + list(high_means.index[:4])))[:8]

        x = np.arange(len(top_ct))
        w = 0.35
        ax.barh(x - w / 2, [low_means.get(c, 0) * 100 for c in top_ct],
                w, label=f'Low H (<2.0, n={len(low_ent)})',
                color='#e15759', alpha=0.7)
        ax.barh(x + w / 2, [high_means.get(c, 0) * 100 for c in top_ct],
                w, label=f'High H (≥2.5, n={len(high_ent)})',
                color='#4e79a7', alpha=0.7)
        ax.set_yticks(x)
        ax.set_yticklabels([c[:22] for c in top_ct], fontsize=8)
        ax.set_xlabel('Mean % of cells', fontsize=10)
        ax.set_title('T-panel: Low vs High Entropy\nComposition', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')

    plt.savefig(os.path.join(out_dir, 'fig_h6a_entropy.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.join(out_dir, 'fig_h6a_entropy.png')}")


# ===================================================================
# H2a: Tfh quantification
# ===================================================================

def figure_h2a(t_path, out_dir):
    print("Generating H2a figures...")

    f = h5py.File(t_path, 'r')
    ct = read_obs_column(f, 'cell_type')
    sid = read_obs_column(f, 'sample_id')
    tma = read_obs_column(f, 'tma')
    n = len(ct)

    cd3 = read_raw_marker(f, 'CD3')
    cd4 = read_raw_marker(f, 'CD4')
    cd8 = read_raw_marker(f, 'CD8a')
    cd20 = read_raw_marker(f, 'CD20')
    cxcr5 = read_raw_marker(f, 'CXCR5')
    pd1 = read_raw_marker(f, 'PD-1')
    cd57 = read_raw_marker(f, 'CD57')

    # Gates
    cd4_t = (cd3 > 0.5) & (cd4 > cd8) & (cd4 > cd20)

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # --- Panel A: CXCR5 distribution on CD4 T cells per TMA ---
    ax = fig.add_subplot(gs[0, 0])
    for t in TMA_ORDER:
        tmask = tma == t
        vals = cxcr5[cd4_t & tmask]
        if len(vals) > 0:
            bins = np.linspace(0, 5, 50)
            ax.hist(vals, bins=bins, alpha=0.4, density=True,
                    color=TMA_COLORS[t], label=t)
    ax.axvline(2.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (2.0)')
    ax.set_xlabel('CXCR5 intensity (raw)', fontsize=11)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('CXCR5 on CD4 T cells\n(non-bimodal → high threshold)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)

    # --- Panel B: Threshold sweep ---
    ax = fig.add_subplot(gs[0, 1])
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for t in TMA_ORDER:
        tmask = tma == t
        n_t = tmask.sum()
        fracs = []
        for th in thresholds:
            tfh = cd4_t & (cxcr5 > th) & tmask
            fracs.append(tfh.sum() / n_t * 100)
        ax.plot(thresholds, fracs, 'o-', color=TMA_COLORS[t], label=t,
                markersize=5, linewidth=1.5)
    ax.axvline(2.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('CXCR5 threshold', fontsize=11)
    ax.set_ylabel('Tfh fraction (% of all cells)', fontsize=10)
    ax.set_title('Tfh% vs CXCR5 Threshold', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 40)

    # --- Panel C: Tfh subtypes by TMA (stacked bar) ---
    ax = fig.add_subplot(gs[0, 2])
    tfh = cd4_t & (cxcr5 > 2.0)
    tfh_pd1 = tfh & (pd1 > 0.5)
    tfh_cd57 = tfh & (cd57 > 0.5)
    tfh_both = tfh & (pd1 > 0.5) & (cd57 > 0.5)
    tfh_other = tfh & ~tfh_pd1 & ~tfh_cd57

    x = np.arange(len(TMA_ORDER))
    w = 0.6
    bottom = np.zeros(len(TMA_ORDER))

    categories = [
        ('Tfh (other)', tfh_other, '#bab0ac'),
        ('Tfh PD-1hi', tfh_pd1 & ~tfh_both, '#e15759'),
        ('Tfh CD57+', tfh_cd57 & ~tfh_both, '#4e79a7'),
        ('Tfh PD-1hi CD57+', tfh_both, '#59a14f'),
    ]

    for label, mask, color in categories:
        vals = []
        for t in TMA_ORDER:
            tmask = tma == t
            n_t = tmask.sum()
            vals.append((mask & tmask).sum() / n_t * 100)
        ax.bar(x, vals, w, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(TMA_ORDER)
    ax.set_ylabel('% of all cells', fontsize=11)
    ax.set_title('Tfh Subtypes by TMA\n(CXCR5 > 2.0)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # --- Panel D: Per-ROI Tfh fraction ---
    ax = fig.add_subplot(gs[1, 0])
    roi_rows = []
    for roi in sorted(set(sid.tolist())):
        rmask = sid == roi
        n_r = rmask.sum()
        if n_r < 100:
            continue
        n_tfh = (tfh & rmask).sum()
        roi_rows.append({'roi': roi, 'tma': tma[rmask][0],
                         'tfh_pct': n_tfh / n_r * 100, 'n_cells': n_r})
    rdf = pd.DataFrame(roi_rows)

    tmas = [t for t in TMA_ORDER if t in rdf['tma'].values]
    data = [rdf[rdf['tma'] == t]['tfh_pct'].values for t in tmas]
    bp = ax.boxplot(data, tick_labels=tmas, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, t in zip(bp['boxes'], tmas):
        patch.set_facecolor(TMA_COLORS[t])
        patch.set_alpha(0.6)
    for i, (t, d) in enumerate(zip(tmas, data)):
        ax.scatter(np.random.normal(i + 1, 0.06, len(d)), d,
                   alpha=0.4, s=12, color=TMA_COLORS[t], edgecolors='none')
    ax.set_ylabel('Tfh % per ROI', fontsize=11)
    ax.set_title('Per-ROI Tfh Fraction', fontsize=12, fontweight='bold')

    kw_groups = [rdf[rdf['tma'] == t]['tfh_pct'].values for t in tmas]
    if all(len(g) > 0 for g in kw_groups):
        _, kw_p = stats.kruskal(*kw_groups)
        ax.text(0.02, 0.98, f'KW p={kw_p:.4f}', transform=ax.transAxes,
                va='top', fontsize=9)

    # --- Panel E: CXCR5 vs PD-1 scatter on CD4 T (sample) ---
    ax = fig.add_subplot(gs[1, 1])
    # Sample 50K CD4 T cells for scatter
    cd4_idx = np.where(cd4_t)[0]
    if len(cd4_idx) > 50000:
        sample_idx = np.random.choice(cd4_idx, 50000, replace=False)
    else:
        sample_idx = cd4_idx
    ax.scatter(cxcr5[sample_idx], pd1[sample_idx], alpha=0.02, s=1,
               color='gray', rasterized=True)
    # Highlight Tfh
    tfh_idx = np.where(tfh)[0]
    if len(tfh_idx) > 10000:
        tfh_sample = np.random.choice(tfh_idx, 10000, replace=False)
    else:
        tfh_sample = tfh_idx
    ax.scatter(cxcr5[tfh_sample], pd1[tfh_sample], alpha=0.1, s=2,
               color='#e15759', rasterized=True, label='Tfh')
    ax.axvline(2.0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('CXCR5', fontsize=11)
    ax.set_ylabel('PD-1', fontsize=11)
    ax.set_title('CD4 T: CXCR5 vs PD-1', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 3)

    # --- Panel F: Tfh vs exhausted CD8 T per ROI ---
    ax = fig.add_subplot(gs[1, 2])
    # Get exhausted CD8 T per ROI
    for roi_data in roi_rows:
        roi = roi_data['roi']
        rmask = sid == roi
        # Count exhausted CD8 T from cell_type
        ct_r = ct[rmask]
        n_exh = sum(1 for c in ct_r if 'exhausted' in c.lower() or 'tex' in c.lower())
        roi_data['exh_pct'] = n_exh / roi_data['n_cells'] * 100
    rdf = pd.DataFrame(roi_rows)

    for t in TMA_ORDER:
        sub = rdf[rdf['tma'] == t]
        if len(sub) > 0:
            ax.scatter(sub['tfh_pct'], sub['exh_pct'], alpha=0.5, s=18,
                       color=TMA_COLORS[t], label=t, edgecolors='none')
    if rdf['exh_pct'].std() > 0 and rdf['tfh_pct'].std() > 0:
        r, p = stats.pearsonr(rdf['tfh_pct'], rdf['exh_pct'])
        ax.set_title(f'Tfh vs Exhausted CD8 T\nr={r:.2f}, p={p:.3f}',
                     fontsize=12, fontweight='bold')
    else:
        ax.set_title('Tfh vs Exhausted CD8 T', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tfh % per ROI', fontsize=11)
    ax.set_ylabel('Exhausted CD8 T % per ROI', fontsize=11)
    ax.legend(fontsize=8)

    f.close()

    plt.savefig(os.path.join(out_dir, 'fig_h2a_tfh.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.join(out_dir, 'fig_h2a_tfh.png')}")


# ===================================================================
# H3a: Macrophage niches
# ===================================================================

def figure_h3a(s_path, out_dir):
    print("Generating H3a figures...")

    f = h5py.File(s_path, 'r')
    ct = read_obs_column(f, 'cell_type')
    sid = read_obs_column(f, 'sample_id')
    tma = read_obs_column(f, 'tma')
    n = len(ct)

    cd68 = read_raw_marker(f, 'CD68')
    cd163 = read_raw_marker(f, 'CD163')
    cd206 = read_raw_marker(f, 'CD206')
    cd14 = read_raw_marker(f, 'CD14')
    s100a9 = read_raw_marker(f, 'S100A9')
    cd11c = read_raw_marker(f, 'CD11c')
    hladr = read_raw_marker(f, 'HLA-DR')

    mac = cd68 > 2.0
    m2 = mac & (cd163 > 1.0) & (cd206 > 0.5)
    inflam = mac & (cd14 > 1.0) & (s100a9 > 0.5)
    dc = (cd11c > 1.0) & (hladr > 1.0)

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # --- Panel A: Per-TMA CD163 QC ---
    ax = fig.add_subplot(gs[0, 0])
    for t in TMA_ORDER:
        tmask = tma == t
        vals = cd163[tmask]
        bins = np.linspace(0, 3, 60)
        ax.hist(vals, bins=bins, alpha=0.4, density=True,
                color=TMA_COLORS[t], label=t)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Gate (1.0)')
    ax.set_xlabel('CD163 intensity (raw)', fontsize=11)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('CD163 per TMA\n(dead in A1/B1)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 3)

    # --- Panel B: Per-TMA marker p99 heatmap ---
    ax = fig.add_subplot(gs[0, 1])
    markers_to_show = ['CD68', 'CD163', 'CD206', 'CD14', 'S100A9', 'CD11c', 'HLA-DR']
    marker_data = {'CD68': cd68, 'CD163': cd163, 'CD206': cd206,
                   'CD14': cd14, 'S100A9': s100a9, 'CD11c': cd11c, 'HLA-DR': hladr}
    p99_matrix = np.zeros((len(markers_to_show), len(TMA_ORDER)))
    for i, m in enumerate(markers_to_show):
        for j, t in enumerate(TMA_ORDER):
            tmask = tma == t
            p99_matrix[i, j] = np.percentile(marker_data[m][tmask], 99)

    im = ax.imshow(p99_matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=max(5, p99_matrix.max()))
    ax.set_xticks(range(len(TMA_ORDER)))
    ax.set_xticklabels(TMA_ORDER)
    ax.set_yticks(range(len(markers_to_show)))
    ax.set_yticklabels(markers_to_show)
    for i in range(len(markers_to_show)):
        for j in range(len(TMA_ORDER)):
            val = p99_matrix[i, j]
            color = 'white' if val > 3 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=9, color=color)
    ax.set_title('Marker p99 by TMA\n(red = dead)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='p99 intensity')

    # --- Panel C: Macrophage composition by TMA (stacked bar) ---
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(TMA_ORDER))
    w = 0.6

    m2_only = m2 & ~inflam
    inflam_only = inflam & ~m2
    both = m2 & inflam
    other_mac = mac & ~m2 & ~inflam

    categories = [
        ('M2-like only', m2_only, '#4e79a7'),
        ('Inflammatory only', inflam_only, '#e15759'),
        ('Both', both, '#59a14f'),
        ('Other mac', other_mac, '#bab0ac'),
    ]

    bottom = np.zeros(len(TMA_ORDER))
    for label, mask, color in categories:
        vals = []
        for t in TMA_ORDER:
            tmask = tma == t
            n_mac = (mac & tmask).sum()
            vals.append((mask & tmask).sum() / n_mac * 100 if n_mac > 0 else 0)
        ax.bar(x, vals, w, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(TMA_ORDER)
    ax.set_ylabel('% of macrophages', fontsize=11)
    ax.set_title('Macrophage Polarization\nby TMA', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

    # --- Panel D: M2 vs Inflammatory per ROI (Biomax + C1 only) ---
    ax = fig.add_subplot(gs[1, 0])
    roi_rows = []
    for roi in sorted(set(sid.tolist())):
        rmask = sid == roi
        tma_r = tma[rmask][0]
        n_mac_r = (mac & rmask).sum()
        if n_mac_r < 10:
            continue
        roi_rows.append({
            'roi': roi, 'tma': tma_r, 'n_mac': n_mac_r,
            'm2_pct': (m2 & rmask).sum() / n_mac_r * 100,
            'inflam_pct': (inflam & rmask).sum() / n_mac_r * 100,
        })
    rdf = pd.DataFrame(roi_rows)

    # Only Biomax and C1 (where CD163 works)
    rdf_good = rdf[rdf['tma'].isin(['Biomax', 'C1'])]
    for t in ['Biomax', 'C1']:
        sub = rdf_good[rdf_good['tma'] == t]
        if len(sub) > 0:
            ax.scatter(sub['m2_pct'], sub['inflam_pct'], alpha=0.5, s=20,
                       color=TMA_COLORS[t], label=t, edgecolors='none')
    if len(rdf_good) > 2:
        r, p = stats.pearsonr(rdf_good['m2_pct'], rdf_good['inflam_pct'])
        ax.set_title(f'M2 vs Inflammatory (Biomax+C1)\nr={r:.2f}, p={p:.3f}',
                     fontsize=12, fontweight='bold')
    ax.set_xlabel('M2-like % of mac', fontsize=11)
    ax.set_ylabel('Inflammatory % of mac', fontsize=11)
    ax.legend(fontsize=8)

    # --- Panel E: All TMAs M2 vs Inflammatory (showing artifact) ---
    ax = fig.add_subplot(gs[1, 1])
    for t in TMA_ORDER:
        sub = rdf[rdf['tma'] == t]
        if len(sub) > 0:
            ax.scatter(sub['m2_pct'], sub['inflam_pct'], alpha=0.5, s=20,
                       color=TMA_COLORS[t], label=t, edgecolors='none')
    ax.set_xlabel('M2-like % of mac', fontsize=11)
    ax.set_ylabel('Inflammatory % of mac', fontsize=11)
    ax.set_title('M2 vs Inflammatory (all TMAs)\nA1/B1 clustered at M2≈0 (dead CD163)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    # Annotate the artifact
    ax.annotate('CD163 dead\nin A1/B1', xy=(1, 30), fontsize=9, color='red',
                fontstyle='italic')

    # --- Panel F: DC-like cells by TMA ---
    ax = fig.add_subplot(gs[1, 2])
    dc_rows = []
    for roi in sorted(set(sid.tolist())):
        rmask = sid == roi
        n_r = rmask.sum()
        if n_r < 100:
            continue
        dc_rows.append({
            'roi': roi, 'tma': tma[rmask][0],
            'dc_pct': (dc & rmask).sum() / n_r * 100,
            'mac_pct': (mac & rmask).sum() / n_r * 100,
        })
    ddf = pd.DataFrame(dc_rows)

    tmas = [t for t in TMA_ORDER if t in ddf['tma'].values]
    data = [ddf[ddf['tma'] == t]['dc_pct'].values for t in tmas]
    bp = ax.boxplot(data, tick_labels=tmas, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, t in zip(bp['boxes'], tmas):
        patch.set_facecolor(TMA_COLORS[t])
        patch.set_alpha(0.6)
    for i, (t, d) in enumerate(zip(tmas, data)):
        ax.scatter(np.random.normal(i + 1, 0.06, len(d)), d,
                   alpha=0.4, s=12, color=TMA_COLORS[t], edgecolors='none')
    ax.set_ylabel('DC-like % per ROI', fontsize=11)
    ax.set_title('DC-like (CD11c+HLA-DR+)\nby TMA', fontsize=12, fontweight='bold')

    f.close()

    plt.savefig(os.path.join(out_dir, 'fig_h3a_macrophages.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.join(out_dir, 'fig_h3a_macrophages.png')}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--t-panel', required=True)
    parser.add_argument('--s-panel', required=True)
    parser.add_argument('--output-dir', default='output/hypotheses')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(42)

    figure_h6a(args.t_panel, args.s_panel, args.output_dir)
    figure_h2a(args.t_panel, args.output_dir)
    figure_h3a(args.s_panel, args.output_dir)

    print("\nAll figures generated.")


if __name__ == '__main__':
    main()
