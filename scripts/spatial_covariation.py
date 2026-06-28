#!/usr/bin/env python3
"""Spatial crosstalk #5 & #7: Interface/boundary analysis + co-variation network.

#5: Identify cells at compartment boundaries and profile their types.
#7: Per-ROI cell type fraction co-variation (correlation matrix).

Output: output/hypotheses_v8/fig_spatial_covariation_T.png
"""

import argparse, os
import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from collections import Counter

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
    'LQ / B transitional': 'B / Unassigned transitional',
    'Cytotoxic / LQ niche': 'Cytotoxic niche',
    'Weak CD20 / LQ border': 'Weak CD20 border',
}

def rename_labels(arr):
    return np.array([DISPLAY_RENAME.get(v, v) for v in arr])

def get_tumor_mask(sample_ids):
    control_tags = ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal', '_ton_', '_adr_']
    return np.array([not any(t in s.lower() for t in control_tags) for s in sample_ids])


BROAD_MAP = {
    'B cells (CXCR5hi)': 'B cells',
    'Other B cells': 'B cells',
    'B cells (CD20hi)': 'B cells',
    'B cells (weak CD20)': 'B cells',
    'B cells (TOXhi)': 'B cells',
    'GC B cells': 'GC B',
    'Activated B / Plasmablast': 'GC B',
    'CD4 T cells': 'CD4 T',
    'CD8 T cells': 'CD8 T',
    'CD8 T exhausted': 'CD8 T exh',
    'CD8 T pre-exhausted (TOX+)': 'CD8 T exh',
    'Treg': 'Treg',
    'Macrophages (GzmB+)': 'Cytotoxic',
    'Macrophages': 'Macrophages',
    'Unassigned': 'Unassigned',
    'Mixed / Border cells': 'Other',
    'Other': 'Other',
    'T cells': 'Other',
}

ANALYSIS_TYPES = ['B cells', 'GC B', 'CD4 T', 'CD8 T', 'CD8 T exh',
                  'Treg', 'Cytotoxic', 'Macrophages']

T_FOLL = {'GC core', 'Follicle core (GC/CD20hi/CXCR5hi)', 'Follicle mantle (CXCR5hi)',
           'Activated B / CXCR5hi zone', 'B cell follicle (CD20hi/CXCR5hi)', 'B cell zone'}
T_INTER = {'T cell zone (CD4/CD8)', 'Treg-enriched T zone', 'Macrophage-rich zone',
            'Follicle-T zone interface', 'Cytotoxic niche'}


def run_analysis(args):
    print("Loading T-panel data...")
    f_v8 = h5py.File(args.t_panel, 'r')
    sid = rename_labels(load_array(f_v8, 'sample_id'))
    ct_raw = rename_labels(load_array(f_v8, 'cell_type'))
    cx = np.array(f_v8['obs']['centroid_x'])
    cy = np.array(f_v8['obs']['centroid_y'])
    tumor = get_tumor_mask(sid)

    f_utag = h5py.File(args.t_utag, 'r')
    comp = rename_labels(load_array(f_utag, 'compartment_name'))
    f_utag.close()

    ct_broad = np.array([BROAD_MAP.get(c, 'Other') for c in ct_raw])

    # Classify compartments as foll/inter
    comp_class = np.array(['foll' if c in T_FOLL else 'inter' if c in T_INTER else 'excl'
                           for c in comp])

    unique_rois = sorted(set(sid[tumor]))
    n_types = len(ANALYSIS_TYPES)
    print(f"  {len(unique_rois)} tumor ROIs")

    # ===================================================================
    # Analysis #5: Boundary cells
    # ===================================================================
    print("\n=== BOUNDARY CELL ANALYSIS ===")
    k = 10
    boundary_ct = []
    interior_foll_ct = []
    interior_inter_ct = []

    for roi in unique_rois:
        m = np.where((sid == roi) & tumor)[0]
        if len(m) < k + 1:
            continue

        roi_cx = cx[m]
        roi_cy = cy[m]
        roi_comp = comp_class[m]
        roi_ct = ct_broad[m]

        # Only consider cells in foll/inter (not excluded)
        valid = (roi_comp == 'foll') | (roi_comp == 'inter')
        if np.sum(valid) < 50:
            continue

        coords = np.column_stack([roi_cx, roi_cy])
        tree = cKDTree(coords)
        _, nb_idx = tree.query(coords, k=k + 1)
        nb_idx = nb_idx[:, 1:]

        for li in range(len(m)):
            if not valid[li]:
                continue
            my_comp = roi_comp[li]
            nb_comps = roi_comp[nb_idx[li]]
            # Boundary = any neighbor in a different compartment class
            has_different = np.any((nb_comps != my_comp) & (nb_comps != 'excl'))
            cell_type = roi_ct[li]
            if has_different:
                boundary_ct.append(cell_type)
            elif my_comp == 'foll':
                interior_foll_ct.append(cell_type)
            elif my_comp == 'inter':
                interior_inter_ct.append(cell_type)

    print(f"  Boundary cells: {len(boundary_ct):,}")
    print(f"  Interior follicular: {len(interior_foll_ct):,}")
    print(f"  Interior interfollicular: {len(interior_inter_ct):,}")

    # Composition of each zone
    boundary_counter = Counter(boundary_ct)
    foll_counter = Counter(interior_foll_ct)
    inter_counter = Counter(interior_inter_ct)

    for zone_name, counter in [('Boundary', boundary_counter),
                                ('Interior foll', foll_counter),
                                ('Interior inter', inter_counter)]:
        total = sum(counter.values())
        print(f"\n  {zone_name} composition:")
        for ct_name in ANALYSIS_TYPES:
            frac = counter.get(ct_name, 0) / max(total, 1)
            print(f"    {ct_name:20s}: {frac:.1%}")

    # ===================================================================
    # Analysis #7: Per-ROI co-variation matrix
    # ===================================================================
    print("\n=== PER-ROI CELL TYPE CO-VARIATION ===")
    type_to_idx = {t: i for i, t in enumerate(ANALYSIS_TYPES)}
    roi_fracs = []

    for roi in unique_rois:
        m = (sid == roi) & tumor
        n = int(np.sum(m))
        if n < 500:
            continue
        roi_ct = ct_broad[m]
        fracs = np.array([np.mean(roi_ct == t) for t in ANALYSIS_TYPES])
        roi_fracs.append(fracs)

    roi_fracs = np.array(roi_fracs)
    print(f"  {len(roi_fracs)} ROIs × {n_types} cell types")

    # Spearman correlation matrix
    corr_mat = np.zeros((n_types, n_types))
    pval_mat = np.zeros((n_types, n_types))
    for i in range(n_types):
        for j in range(n_types):
            rho, p = spearmanr(roi_fracs[:, i], roi_fracs[:, j])
            corr_mat[i, j] = rho
            pval_mat[i, j] = p

    print("\n  Top positive co-variations:")
    pairs = [(corr_mat[i, j], ANALYSIS_TYPES[i], ANALYSIS_TYPES[j])
             for i in range(n_types) for j in range(i + 1, n_types)]
    pairs.sort(key=lambda x: -x[0])
    for rho, t1, t2 in pairs[:5]:
        p = pval_mat[ANALYSIS_TYPES.index(t1), ANALYSIS_TYPES.index(t2)]
        print(f"    {t1:20s} ↔ {t2:20s}: rho={rho:+.3f} p={p:.2e}")
    print("\n  Top negative co-variations:")
    pairs.sort(key=lambda x: x[0])
    for rho, t1, t2 in pairs[:5]:
        p = pval_mat[ANALYSIS_TYPES.index(t1), ANALYSIS_TYPES.index(t2)]
        print(f"    {t1:20s} ↔ {t2:20s}: rho={rho:+.3f} p={p:.2e}")

    # ===================================================================
    # Figure — 2×3 grid
    # ===================================================================
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (a) Concept cartoon
    ax_cartoon = fig.add_subplot(gs[0, 0])
    cartoon_path = os.path.join('output', 'hypothesis_cartoons', 'spatial_covariation_boundary.png')
    if os.path.exists(cartoon_path):
        cartoon = mpimg.imread(cartoon_path)
        ax_cartoon.imshow(cartoon)
    ax_cartoon.axis('off')
    ax_cartoon.text(-0.02, 1.03, 'a', transform=ax_cartoon.transAxes,
                    fontsize=14, fontweight='bold', va='top')

    # (b) Boundary vs interior composition
    ax_a = fig.add_subplot(gs[0, 1])
    x = np.arange(n_types)
    width = 0.25
    total_boundary = max(len(boundary_ct), 1)
    total_foll = max(len(interior_foll_ct), 1)
    total_inter = max(len(interior_inter_ct), 1)
    b_fracs = [boundary_counter.get(t, 0) / total_boundary for t in ANALYSIS_TYPES]
    f_fracs = [foll_counter.get(t, 0) / total_foll for t in ANALYSIS_TYPES]
    i_fracs = [inter_counter.get(t, 0) / total_inter for t in ANALYSIS_TYPES]
    ax_a.barh(x - width, f_fracs, height=width, color='#FFAAAA', label='Interior follicular')
    ax_a.barh(x, b_fracs, height=width, color='#888888', label='Boundary')
    ax_a.barh(x + width, i_fracs, height=width, color='#AAAAFF', label='Interior interfollicular')
    ax_a.set_yticks(x)
    ax_a.set_yticklabels(ANALYSIS_TYPES, fontsize=9)
    ax_a.set_xlabel('Fraction', fontsize=9)
    ax_a.set_title(f'Boundary vs Interior Cell Types\n(boundary: {len(boundary_ct):,} cells)',
                   fontsize=10, fontweight='bold')
    ax_a.legend(fontsize=7, loc='lower right')
    ax_a.invert_yaxis()
    ax_a.text(-0.12, 1.05, 'b', transform=ax_a.transAxes, fontsize=14, fontweight='bold', va='top')

    # (c) Boundary enrichment ratio (boundary / average of two interiors)
    ax_b_ratio = fig.add_subplot(gs[0, 2])
    avg_interior = [(f_fracs[i] + i_fracs[i]) / 2 for i in range(n_types)]
    enrichment = [b_fracs[i] / max(avg_interior[i], 1e-6) for i in range(n_types)]
    colors_enr = ['#d6604d' if e > 1.2 else '#4393c3' if e < 0.8 else '#888888' for e in enrichment]
    ax_b_ratio.barh(x, enrichment, color=colors_enr, height=0.5, edgecolor='white', linewidth=0.5)
    ax_b_ratio.axvline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax_b_ratio.set_yticks(x)
    ax_b_ratio.set_yticklabels(ANALYSIS_TYPES, fontsize=9)
    ax_b_ratio.set_xlabel('Boundary / Interior ratio', fontsize=9)
    ax_b_ratio.set_title('Boundary enrichment ratio\n(red = enriched, blue = depleted)', fontsize=10, fontweight='bold')
    ax_b_ratio.invert_yaxis()
    for i in range(n_types):
        ax_b_ratio.text(enrichment[i] + 0.02, i, f'{enrichment[i]:.2f}', va='center', fontsize=8)
    ax_b_ratio.text(-0.12, 1.05, 'c', transform=ax_b_ratio.transAxes, fontsize=14, fontweight='bold', va='top')

    # (d) Co-variation matrix
    ax_c = fig.add_subplot(gs[1, 0])
    im = ax_c.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax_c.set_xticks(range(n_types))
    ax_c.set_yticks(range(n_types))
    ax_c.set_xticklabels(ANALYSIS_TYPES, fontsize=8, rotation=45, ha='right')
    ax_c.set_yticklabels(ANALYSIS_TYPES, fontsize=8)
    for i in range(n_types):
        for j in range(n_types):
            val = corr_mat[i, j]
            sig = '*' if pval_mat[i, j] < 0.05 else ''
            color = 'white' if abs(val) > 0.5 else 'black'
            ax_c.text(j, i, f'{val:.2f}{sig}', ha='center', va='center',
                      fontsize=7, color=color)
    plt.colorbar(im, ax=ax_c, shrink=0.8, label='Spearman ρ')
    ax_c.set_title(f'Cell Type Co-variation\n(n={len(roi_fracs)} ROIs)',
                   fontsize=10, fontweight='bold')
    ax_c.text(-0.12, 1.05, 'd', transform=ax_c.transAxes, fontsize=14, fontweight='bold', va='top')

    # (e) Network visualization — spans 2 columns for more room
    ax_d = fig.add_subplot(gs[1, 1:3])
    theta = np.linspace(0, 2 * np.pi, n_types, endpoint=False) + np.pi / 2  # start at top
    node_x = np.cos(theta)
    node_y = np.sin(theta)

    # Draw edges for |rho| > 0.3 — thicker, with ρ labels on strongest
    edge_list = []
    for i in range(n_types):
        for j in range(i + 1, n_types):
            rho = corr_mat[i, j]
            if abs(rho) > 0.3 and pval_mat[i, j] < 0.05:
                edge_list.append((i, j, rho))

    for i, j, rho in edge_list:
        color = '#d6604d' if rho > 0 else '#4393c3'
        lw = abs(rho) * 5  # wider range for visibility
        ax_d.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]],
                 color=color, linewidth=lw, alpha=0.5, zorder=1)
        # Label strongest edges
        if abs(rho) > 0.5:
            mx = (node_x[i] + node_x[j]) / 2
            my = (node_y[i] + node_y[j]) / 2
            ax_d.text(mx, my, f'{rho:.2f}', fontsize=6, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none'))

    # Draw nodes — larger, with node size proportional to mean cell fraction
    node_colors = ['#FFAAAA', '#FF6666', '#AAAAFF', '#6666FF', '#9999FF',
                   '#66CCCC', '#FF9933', '#CC6633']
    mean_fracs = roi_fracs.mean(axis=0)
    node_sizes = 100 + mean_fracs / mean_fracs.max() * 400
    for i in range(n_types):
        ax_d.scatter(node_x[i], node_y[i], s=node_sizes[i], c=node_colors[i],
                    edgecolors='black', linewidth=1.5, zorder=2)
        # Label
        offset = 0.18
        ha = 'left' if node_x[i] > 0.1 else 'right' if node_x[i] < -0.1 else 'center'
        va = 'bottom' if node_y[i] > 0.1 else 'top' if node_y[i] < -0.1 else 'center'
        ax_d.text(node_x[i] * (1 + offset), node_y[i] * (1 + offset),
                 ANALYSIS_TYPES[i], fontsize=8, ha=ha, va=va, fontweight='bold')

    ax_d.set_xlim(-1.6, 1.6)
    ax_d.set_ylim(-1.6, 1.6)
    ax_d.set_aspect('equal')
    ax_d.axis('off')
    ax_d.set_title('Co-variation Network\n(|ρ|>0.3, p<0.05; red=+, blue=−)',
                   fontsize=10, fontweight='bold')
    # Add manual legend for edge colors
    from matplotlib.lines import Line2D
    edge_legend = [Line2D([0], [0], color='#d6604d', linewidth=3, label='Positive'),
                   Line2D([0], [0], color='#4393c3', linewidth=3, label='Negative')]
    ax_d.legend(handles=edge_legend, fontsize=8, loc='lower right')
    ax_d.text(-0.05, 1.05, 'e', transform=ax_d.transAxes, fontsize=14, fontweight='bold', va='top')

    fig.suptitle('T-panel: Spatial Co-variation & Boundary Analysis',
                 fontsize=13, fontweight='bold')

    out_path = os.path.join(args.output_dir, 'fig_spatial_covariation_T.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")

    f_v8.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t-panel', required=True)
    parser.add_argument('--t-utag', required=True)
    parser.add_argument('--output-dir', default='output/hypotheses_v8')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_analysis(args)


if __name__ == '__main__':
    main()
