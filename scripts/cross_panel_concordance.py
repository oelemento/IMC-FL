#!/usr/bin/env python3
"""Cross-panel concordance analysis for v7 T-panel and S-panel annotations.

Computes per-ROI cell type proportions for both panels, calculates Pearson
and Spearman correlations, and generates scatter plots + per-TMA bar chart.

Usage:
    python scripts/cross_panel_concordance.py \
        --t-panel output/all_TMA_T_global_v7.h5ad \
        --s-panel output/all_TMA_S_global_v7.h5ad \
        --output output/v7_cross_panel_concordance.png
"""

import argparse
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def broad_map(ct: str) -> str:
    """Map fine cell types to broad categories for cross-panel comparison.

    GC B cells are merged into B cells because the S-panel cannot identify
    them (BCL6 antibody fails in IMC).
    """
    ct = str(ct)
    if 'CD4 T' in ct or 'Treg' in ct:
        return 'CD4 T'
    if 'CD8 T' in ct:
        return 'CD8 T'
    if any(x in ct for x in ['Macrophage', 'Myeloid', 'Dendritic', 'Histiocyte', 'pDC', 'M1 ', 'M2 ']):
        return 'Myeloid'
    if 'B cell' in ct or 'GC B' in ct or 'PAX5' in ct or 'BCL2' in ct:
        return 'B cells'
    if 'FDC' in ct:
        return 'FDC'
    if 'Stromal' in ct or 'Endothelial' in ct or 'FRC' in ct:
        return 'Stromal'
    if 'Low quality' in ct or 'Unassigned' in ct:
        return 'LQ'
    return 'Other'


def get_tma(sid: str) -> str:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--t-panel', required=True, help='T-panel v7 h5ad file')
    parser.add_argument('--s-panel', required=True, help='S-panel v7 h5ad file')
    parser.add_argument('--output', required=True, help='Output PNG path')
    parser.add_argument('--lq-threshold', type=float, default=0.30,
                        help='LQ fraction threshold for flagging bad ROIs (default: 0.30)')
    args = parser.parse_args()

    # --- Load ---
    print(f"Loading T-panel: {args.t_panel}")
    t = ad.read_h5ad(args.t_panel, backed='r')
    print(f"Loading S-panel: {args.s_panel}")
    s = ad.read_h5ad(args.s_panel, backed='r')

    t_obs = t.obs[['sample_id', 'cell_type']].copy()
    s_obs = s.obs[['sample_id', 'cell_type']].copy()

    t_obs['broad'] = t_obs['cell_type'].apply(broad_map)
    s_obs['broad'] = s_obs['cell_type'].apply(broad_map)

    # --- Per-ROI proportions ---
    t_props = pd.crosstab(t_obs['sample_id'], t_obs['broad'], normalize='index')
    s_props = pd.crosstab(s_obs['sample_id'], s_obs['broad'], normalize='index')

    common = t_props.index.intersection(s_props.index)
    print(f"Paired ROIs: {len(common)}")

    tma_labels = pd.Series([get_tma(s) for s in common], index=common)
    tma_colors = {'A1': '#1f77b4', 'B1': '#ff7f0e', 'C1': '#2ca02c', 'Biomax': '#d62728'}

    # --- Per-ROI correlations ---
    cell_types = ['CD4 T', 'CD8 T', 'B cells', 'Myeloid']
    print(f"\nPer-ROI correlations ({len(common)} paired ROIs, GC B merged into B cells):")
    print(f"{'Cell type':>12s}  {'Pearson r':>10s}  {'Spearman r':>10s}  {'p (Pearson)':>12s}")
    for ct in cell_types:
        t_vals = t_props.reindex(common).get(ct, pd.Series(0, index=common)).fillna(0).values
        s_vals = s_props.reindex(common).get(ct, pd.Series(0, index=common)).fillna(0).values
        rp, pp = pearsonr(t_vals, s_vals)
        rs, ps = spearmanr(t_vals, s_vals)
        print(f"{ct:>12s}  {rp:>10.3f}  {rs:>10.3f}  {pp:>12.4f}")

    # --- B cell discordance analysis ---
    t_lq = t_props.reindex(common).get('LQ', pd.Series(0, index=common)).fillna(0)
    s_lq = s_props.reindex(common).get('LQ', pd.Series(0, index=common)).fillna(0)
    high_lq = (t_lq > args.lq_threshold) | (s_lq > args.lq_threshold)

    t_b = t_props.reindex(common).get('B cells', pd.Series(0, index=common)).fillna(0).values
    s_b = s_props.reindex(common).get('B cells', pd.Series(0, index=common)).fillna(0).values
    t_blq = t_b + t_lq.values
    s_blq = s_b + s_lq.values

    good = common[~high_lq]

    print(f"\nB cell discordance analysis:")
    print(f"  ROIs with >{args.lq_threshold*100:.0f}% LQ in either panel: {high_lq.sum()}/{len(common)}")
    rp_all, _ = pearsonr(t_b, s_b)
    rs_all, _ = spearmanr(t_b, s_b)
    print(f"  B cells (all ROIs):          Pearson r={rp_all:.3f}, Spearman r={rs_all:.3f}")
    if len(good) >= 5:
        rp_good, _ = pearsonr(t_b[~high_lq], s_b[~high_lq])
        rs_good, _ = spearmanr(t_b[~high_lq], s_b[~high_lq])
        print(f"  B cells (good ROIs, n={len(good)}):  Pearson r={rp_good:.3f}, Spearman r={rs_good:.3f}")
    rp_blq, _ = pearsonr(t_blq, s_blq)
    rs_blq, _ = spearmanr(t_blq, s_blq)
    print(f"  B cells + LQ (all ROIs):     Pearson r={rp_blq:.3f}, Spearman r={rs_blq:.3f}")

    # --- Per-TMA correlations ---
    print(f"\nPer-TMA B cell correlations:")
    for tma in ['A1', 'B1', 'C1', 'Biomax']:
        mask = tma_labels == tma
        n = mask.sum()
        if n >= 5:
            rp, pp = pearsonr(t_b[mask], s_b[mask])
            rs, ps = spearmanr(t_b[mask], s_b[mask])
            print(f"  {tma} (n={n}): Pearson r={rp:.3f}, Spearman r={rs:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for i, ct in enumerate(cell_types):
        ax = axes_flat[i]
        t_vals = t_props.reindex(common).get(ct, pd.Series(0, index=common)).fillna(0).values * 100
        s_vals = s_props.reindex(common).get(ct, pd.Series(0, index=common)).fillna(0).values * 100

        for tma in ['A1', 'B1', 'C1', 'Biomax']:
            mask = tma_labels == tma
            if mask.sum() > 0:
                ax.scatter(t_vals[mask], s_vals[mask], c=tma_colors[tma],
                           label=tma, alpha=0.7, s=30, edgecolors='none')

        rp, pp = pearsonr(t_vals, s_vals)
        rs, ps = spearmanr(t_vals, s_vals)
        ax.set_title(f'{ct}  (r={rp:.2f}, ρ={rs:.2f})', fontsize=12)

        maxv = max(t_vals.max(), s_vals.max(), 1) * 1.05
        ax.plot([0, maxv], [0, maxv], 'k--', alpha=0.3, lw=1)
        ax.set_xlabel('T-panel (%)', fontsize=10)
        ax.set_ylabel('S-panel (%)', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')

    # Per-TMA bar chart
    ax = axes_flat[4]
    tmas = ['A1', 'B1', 'C1', 'Biomax']
    x = np.arange(len(tmas))
    width = 0.1
    colors_ct = {'CD4 T': '#1f77b4', 'CD8 T': '#ff7f0e', 'B cells': '#2ca02c', 'Myeloid': '#d62728'}

    for j, ct in enumerate(cell_types):
        t_means, s_means = [], []
        for tma in tmas:
            mask = tma_labels == tma
            if mask.sum() > 0:
                t_v = t_props.reindex(common[mask]).get(ct, pd.Series(0, index=common[mask])).fillna(0).values * 100
                s_v = s_props.reindex(common[mask]).get(ct, pd.Series(0, index=common[mask])).fillna(0).values * 100
                t_means.append(t_v.mean())
                s_means.append(s_v.mean())
            else:
                t_means.append(0)
                s_means.append(0)
        offset = (j - 1.5) * width
        ax.bar(x + offset - width / 2, t_means, width, color=colors_ct[ct], alpha=0.7)
        ax.bar(x + offset + width / 2, s_means, width, color=colors_ct[ct], alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(tmas)
    ax.set_ylabel('Mean %', fontsize=10)
    ax.set_title('Per-TMA means (solid=T, faded=S)', fontsize=12)
    legend_elements = [Patch(facecolor=colors_ct[ct], label=ct) for ct in cell_types]
    legend_elements += [Patch(facecolor='gray', alpha=0.7, label='T-panel'),
                        Patch(facecolor='gray', alpha=0.3, label='S-panel')]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    axes_flat[5].axis('off')

    fig.suptitle(f'v7 Cross-Panel Concordance (n={len(common)} paired ROIs, GC B merged into B cells)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
