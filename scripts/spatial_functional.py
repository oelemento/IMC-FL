#!/usr/bin/env python3
"""Spatial crosstalk #3 & #6: Functional state × proximity + distance distributions.

Tests whether a cell's functional state (exhaustion, cytotoxicity, signaling)
depends on its local neighborhood composition. Also computes full nearest-neighbor
distance distributions for key cell type pairs.

Output: output/hypotheses_v8/fig_functional_proximity_T.png
"""

import argparse, os
import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
# Data loading
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

DISPLAY_RENAME = {
    'Low quality / Unassigned': 'Unassigned',
    'B cells': 'Other B cells',
}

def rename_labels(arr):
    return np.array([DISPLAY_RENAME.get(v, v) for v in arr])

def get_tumor_mask(sample_ids):
    control_tags = ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal', '_ton_', '_adr_']
    return np.array([not any(t in s.lower() for t in control_tags) for s in sample_ids])


# B cell types (tumor)
B_TYPES = {'B cells (CXCR5hi)', 'Other B cells', 'B cells (CD20hi)',
           'B cells (weak CD20)', 'B cells (TOXhi)', 'GC B cells',
           'Activated B / Plasmablast'}

# Cell type groups for neighbor fractions
NEIGHBOR_GROUPS = {
    'B cell': B_TYPES,
    'Macrophage': {'Macrophages'},
    'CD4 T': {'CD4 T cells'},
    'Treg': {'Treg'},
    'CD8 T': {'CD8 T cells', 'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)', 'Macrophages (GzmB+)'},
}


def run_analysis(args):
    print("Loading T-panel data...")
    f_v8 = h5py.File(args.t_panel, 'r')
    sid = rename_labels(load_array(f_v8, 'sample_id'))
    ct = rename_labels(load_array(f_v8, 'cell_type'))
    cx = np.array(f_v8['obs']['centroid_x'])
    cy = np.array(f_v8['obs']['centroid_y'])
    tumor = get_tumor_mask(sid)

    # Load marker expression (z-scored but rank-preserving for Spearman)
    X = f_v8['X'][:]
    var_key = '_index' if '_index' in f_v8['var'] else 'index'
    markers = [v.decode() if isinstance(v, bytes) else str(v) for v in f_v8['var'][var_key][:]]
    marker_idx = {m: i for i, m in enumerate(markers)}

    unique_rois = sorted(set(sid[tumor]))
    print(f"  {len(unique_rois)} tumor ROIs, {len(markers)} markers")

    # --- Analysis 1: CD8 T exhaustion vs neighbor composition ---
    print("\n=== CD8 T exhaustion vs neighborhood ===")
    cd8_types = {'CD8 T cells', 'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)'}
    k = 20

    # Collect per-cell data across all ROIs
    cd8_tox = []
    cd8_pd1 = []
    cd8_nb_bcell = []
    cd8_nb_mac = []
    cd8_nb_treg = []

    for roi in unique_rois:
        m = np.where((sid == roi) & tumor)[0]
        if len(m) < k + 1:
            continue

        roi_ct = ct[m]
        roi_cx = cx[m]
        roi_cy = cy[m]
        roi_X = X[m]

        # Build kNN graph
        coords = np.column_stack([roi_cx, roi_cy])
        tree = cKDTree(coords)
        _, nb_idx = tree.query(coords, k=k + 1)
        nb_idx = nb_idx[:, 1:]

        # CD8 T cells in this ROI
        cd8_mask = np.array([c in cd8_types for c in roi_ct])
        cd8_local_idx = np.where(cd8_mask)[0]

        if len(cd8_local_idx) < 10:
            continue

        for li in cd8_local_idx:
            # Functional markers
            tox_val = roi_X[li, marker_idx['TOX']]
            pd1_val = roi_X[li, marker_idx['PD_1']]

            # Neighbor composition
            nb_ct = roi_ct[nb_idx[li]]
            frac_b = np.mean([c in B_TYPES for c in nb_ct])
            frac_mac = np.mean([c == 'Macrophages' for c in nb_ct])
            frac_treg = np.mean([c == 'Treg' for c in nb_ct])

            cd8_tox.append(tox_val)
            cd8_pd1.append(pd1_val)
            cd8_nb_bcell.append(frac_b)
            cd8_nb_mac.append(frac_mac)
            cd8_nb_treg.append(frac_treg)

    cd8_tox = np.array(cd8_tox)
    cd8_pd1 = np.array(cd8_pd1)
    cd8_nb_bcell = np.array(cd8_nb_bcell)
    cd8_nb_mac = np.array(cd8_nb_mac)
    cd8_nb_treg = np.array(cd8_nb_treg)
    print(f"  {len(cd8_tox)} CD8 T cells analyzed")

    # Spearman correlations
    corrs = {}
    for name, marker_vals in [('TOX', cd8_tox), ('PD-1', cd8_pd1)]:
        for nb_name, nb_vals in [('B cell neighbors', cd8_nb_bcell),
                                  ('Macrophage neighbors', cd8_nb_mac),
                                  ('Treg neighbors', cd8_nb_treg)]:
            rho, p = spearmanr(marker_vals, nb_vals)
            corrs[(name, nb_name)] = (rho, p)
            print(f"  {name} vs {nb_name}: rho={rho:.3f}, p={p:.2e}")

    # --- Analysis 2: Distance distributions for key cell type pairs ---
    print("\n=== Nearest-neighbor distance distributions ===")
    PAIRS = [
        ('CD4 T cells', B_TYPES, 'CD4 T → B cell'),
        ('Treg', {'Macrophages'}, 'Treg → Macrophage'),
        ('Treg', {'CD4 T cells'}, 'Treg → CD4 T'),
        ('Macrophages', B_TYPES, 'Mac → B cell'),
    ]

    pair_distances = {}
    for src_type, tgt_types, label in PAIRS:
        dists_all = []
        for roi in unique_rois:
            m = np.where((sid == roi) & tumor)[0]
            if len(m) < 50:
                continue
            roi_ct = ct[m]
            roi_cx = cx[m]
            roi_cy = cy[m]

            if isinstance(src_type, str):
                src_mask = roi_ct == src_type
            else:
                src_mask = np.array([c in src_type for c in roi_ct])
            tgt_mask = np.array([c in tgt_types for c in roi_ct])

            src_idx = np.where(src_mask)[0]
            tgt_idx = np.where(tgt_mask)[0]

            if len(src_idx) < 5 or len(tgt_idx) < 5:
                continue

            src_coords = np.column_stack([roi_cx[src_idx], roi_cy[src_idx]])
            tgt_coords = np.column_stack([roi_cx[tgt_idx], roi_cy[tgt_idx]])
            tgt_tree = cKDTree(tgt_coords)
            nn_dists, _ = tgt_tree.query(src_coords, k=1)
            dists_all.extend(nn_dists)

        pair_distances[label] = np.array(dists_all)
        print(f"  {label}: n={len(dists_all)}, median={np.median(dists_all):.1f} px, "
              f"mean={np.mean(dists_all):.1f} px")

    # --- Analysis 3: CD8 T exhausted vs non-exhausted: different neighborhoods? ---
    print("\n=== CD8 T exhausted vs non-exhausted neighborhood ===")
    exh_mask = ct == 'CD8 T exhausted'
    nonexh_mask = ct == 'CD8 T cells'

    exh_nb = {'B cell': [], 'Macrophage': [], 'CD4 T': [], 'Treg': []}
    nonexh_nb = {'B cell': [], 'Macrophage': [], 'CD4 T': [], 'Treg': []}

    for roi in unique_rois:
        m = np.where((sid == roi) & tumor)[0]
        if len(m) < k + 1:
            continue
        roi_ct = ct[m]
        roi_cx_arr = cx[m]
        roi_cy_arr = cy[m]

        coords = np.column_stack([roi_cx_arr, roi_cy_arr])
        tree = cKDTree(coords)
        _, nb_idx = tree.query(coords, k=k + 1)
        nb_idx = nb_idx[:, 1:]

        for li in range(len(m)):
            nb_ct = roi_ct[nb_idx[li]]
            cell_ct = roi_ct[li]
            if cell_ct not in ('CD8 T exhausted', 'CD8 T cells'):
                continue

            target = exh_nb if cell_ct == 'CD8 T exhausted' else nonexh_nb
            for grp, grp_types in NEIGHBOR_GROUPS.items():
                if grp in target:
                    target[grp].append(np.mean([c in grp_types for c in nb_ct]))

    for grp in exh_nb:
        e = np.array(exh_nb[grp])
        ne = np.array(nonexh_nb[grp])
        if len(e) > 0 and len(ne) > 0:
            from scipy.stats import mannwhitneyu
            u, p = mannwhitneyu(e, ne, alternative='two-sided')
            print(f"  {grp} neighbors: exhausted={np.mean(e):.3f}, non-exh={np.mean(ne):.3f}, "
                  f"MWU p={p:.2e}")

    # --- Figure: 3×3 grid ---
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    # (a) Concept cartoon
    ax_cartoon = fig.add_subplot(gs[0, 0])
    cartoon_path = os.path.join('output', 'hypothesis_cartoons', 'spatial_functional_proximity.png')
    if os.path.exists(cartoon_path):
        cartoon = mpimg.imread(cartoon_path)
        ax_cartoon.imshow(cartoon)
    ax_cartoon.axis('off')
    ax_cartoon.text(-0.02, 1.03, 'a', transform=ax_cartoon.transAxes,
                    fontsize=14, fontweight='bold', va='top')

    # (b) CD8 T TOX vs B cell neighbor fraction — hexbin density
    ax_a = fig.add_subplot(gs[0, 1])
    hb = ax_a.hexbin(cd8_nb_bcell, cd8_tox, gridsize=40, cmap='Blues', mincnt=1,
                      linewidths=0.2, rasterized=True)
    rho, p = corrs[('TOX', 'B cell neighbors')]
    ax_a.set_xlabel('Fraction B cell neighbors (k=20)', fontsize=9)
    ax_a.set_ylabel('TOX expression (z-score)', fontsize=9)
    ax_a.set_title(f'CD8 T: TOX vs B cell proximity\nρ={rho:.3f}, p={p:.1e}', fontsize=10)
    ax_a.text(-0.12, 1.05, 'b', transform=ax_a.transAxes, fontsize=14, fontweight='bold', va='top')
    plt.colorbar(hb, ax=ax_a, shrink=0.7, label='Count')

    # (c) CD8 T TOX vs Treg neighbor fraction — hexbin density
    ax_b = fig.add_subplot(gs[0, 2])
    hb2 = ax_b.hexbin(cd8_nb_treg, cd8_tox, gridsize=40, cmap='Reds', mincnt=1,
                        linewidths=0.2, rasterized=True)
    rho, p = corrs[('TOX', 'Treg neighbors')]
    ax_b.set_xlabel('Fraction Treg neighbors (k=20)', fontsize=9)
    ax_b.set_ylabel('TOX expression (z-score)', fontsize=9)
    ax_b.set_title(f'CD8 T: TOX vs Treg proximity\nρ={rho:.3f}, p={p:.1e}', fontsize=10)
    ax_b.text(-0.12, 1.05, 'c', transform=ax_b.transAxes, fontsize=14, fontweight='bold', va='top')
    plt.colorbar(hb2, ax=ax_b, shrink=0.7, label='Count')

    # (d) Exhausted vs non-exhausted neighborhood comparison + significance stars
    ax_c = fig.add_subplot(gs[1, 0])
    grp_names = [g for g in exh_nb if len(exh_nb[g]) > 0]
    x = np.arange(len(grp_names))
    exh_means = [np.mean(exh_nb[g]) for g in grp_names]
    nonexh_means = [np.mean(nonexh_nb[g]) for g in grp_names]
    ax_c.bar(x - 0.18, exh_means, width=0.35, color='#d6604d', label='CD8 T exhausted')
    ax_c.bar(x + 0.18, nonexh_means, width=0.35, color='#4393c3', label='CD8 T non-exhausted')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(grp_names, fontsize=8, rotation=30, ha='right')
    ax_c.set_ylabel('Mean fraction of neighbors', fontsize=9)
    ax_c.set_title('Neighborhood of exhausted vs\nnon-exhausted CD8 T', fontsize=10)
    ax_c.legend(fontsize=7, loc='upper right')
    from scipy.stats import mannwhitneyu as _mwu
    for gi, grp in enumerate(grp_names):
        e = np.array(exh_nb[grp])
        ne = np.array(nonexh_nb[grp])
        if len(e) > 0 and len(ne) > 0:
            _, pv = _mwu(e, ne, alternative='two-sided')
            sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'n.s.'
            y_max = max(exh_means[gi], nonexh_means[gi])
            ax_c.text(gi, y_max + 0.008, sig, ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_c.text(-0.12, 1.05, 'd', transform=ax_c.transAxes, fontsize=14, fontweight='bold', va='top')

    # (e) Summary correlation table as heatmap
    ax_d = fig.add_subplot(gs[1, 1])
    marker_names = ['TOX', 'PD-1']
    nb_names = ['B cell', 'Macrophage', 'Treg']
    nb_keys = ['B cell neighbors', 'Macrophage neighbors', 'Treg neighbors']
    corr_table = np.zeros((len(marker_names), len(nb_names)))
    for mi, m_name in enumerate(marker_names):
        for ni, nb_key in enumerate(nb_keys):
            corr_table[mi, ni] = corrs[(m_name, nb_key)][0]
    im_d = ax_d.imshow(corr_table, cmap='RdBu_r', vmin=-0.4, vmax=0.4, aspect='auto')
    ax_d.set_xticks(range(len(nb_names)))
    ax_d.set_xticklabels(nb_names, fontsize=9, rotation=30, ha='right')
    ax_d.set_yticks(range(len(marker_names)))
    ax_d.set_yticklabels(marker_names, fontsize=9)
    for mi in range(len(marker_names)):
        for ni in range(len(nb_names)):
            val = corr_table[mi, ni]
            pval = corrs[(marker_names[mi], nb_keys[ni])][1]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            color = 'white' if abs(val) > 0.2 else 'black'
            ax_d.text(ni, mi, f'{val:.3f}{sig}', ha='center', va='center', fontsize=10, color=color)
    ax_d.set_title('Exhaustion marker ×\nneighbor type (Spearman ρ)', fontsize=10)
    plt.colorbar(im_d, ax=ax_d, shrink=0.7, label='ρ')
    ax_d.text(-0.15, 1.05, 'e', transform=ax_d.transAxes, fontsize=14, fontweight='bold', va='top')

    # (f-i) Distance distributions — all 4 pairs
    dist_colors = ['#4393c3', '#92c5de', '#d6604d', '#f4a582']
    dist_items = list(pair_distances.items())
    # Place: (f) at [1,2], (g-i) at [2,0:3]
    dist_positions = [(1, 2), (2, 0), (2, 1), (2, 2)]
    for pi, (label, dists) in enumerate(dist_items):
        row, col = dist_positions[pi]
        ax = fig.add_subplot(gs[row, col])
        ax.hist(dists, bins=80, range=(0, 200), density=True, color=dist_colors[pi],
                alpha=0.7, edgecolor='white', linewidth=0.3)
        median = np.median(dists)
        ax.axvline(median, color='#333333', linewidth=2, linestyle='--',
                   label=f'Median={median:.1f} px')
        ax.set_xlabel('NN distance (pixels)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'{label}\n(n={len(dists):,})', fontsize=10)
        ax.legend(fontsize=8)
        lbl = chr(ord('f') + pi)
        ax.text(-0.12, 1.05, lbl, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    fig.suptitle('T-panel: Functional State × Spatial Proximity',
                 fontsize=13, fontweight='bold')

    out_path = os.path.join(args.output_dir, 'fig_functional_proximity_T.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")

    # --- Summary correlation table ---
    print("\n=== SUMMARY: All correlations ===")
    for (marker, nb_name), (rho, p) in sorted(corrs.items()):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f"  {marker:6s} vs {nb_name:25s}: rho={rho:+.3f} p={p:.2e} {sig}")

    f_v8.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t-panel', required=True)
    parser.add_argument('--output-dir', default='output/hypotheses_v8')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_analysis(args)


if __name__ == '__main__':
    main()
