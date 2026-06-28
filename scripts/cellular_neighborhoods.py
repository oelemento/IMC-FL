#!/usr/bin/env python3
"""Spatial crosstalk #4: Cellular neighborhood analysis.

For each cell, compute the cell type composition of its k nearest neighbors.
Cluster these composition vectors to identify recurring multicellular motifs
(cellular neighborhoods). Visualize on representative ROIs.

Output: output/hypotheses_v8/fig_cellular_neighborhoods_T.png
"""

import argparse, os
import numpy as np
import h5py
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.patches import Patch

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

# Broad types for neighborhood composition
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

CN_TYPES = ['B cells', 'GC B', 'CD4 T', 'CD8 T', 'CD8 T exh',
            'Treg', 'Cytotoxic', 'Macrophages', 'Unassigned', 'Other']

T_FOLL = {'GC core', 'Follicle core (GC/CD20hi/CXCR5hi)', 'Follicle mantle (CXCR5hi)',
           'Activated B / CXCR5hi zone', 'B cell follicle (CD20hi/CXCR5hi)', 'B cell zone'}
T_INTER = {'T cell zone (CD4/CD8)', 'Treg-enriched T zone', 'Macrophage-rich zone',
            'Follicle-T zone interface', 'Cytotoxic niche'}

CN_COLORS = plt.cm.tab20(np.linspace(0, 1, 12))


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
    type_to_idx = {t: i for i, t in enumerate(CN_TYPES)}
    n_types = len(CN_TYPES)
    ct_codes = np.array([type_to_idx.get(c, n_types - 1) for c in ct_broad])  # map unknown to Other

    unique_rois = sorted(set(sid[tumor]))
    print(f"  {len(unique_rois)} tumor ROIs")

    # --- Step 1: Compute per-cell neighborhood composition ---
    k = 20
    print(f"\nComputing k={k} neighborhood compositions per ROI...")
    # Process per-ROI, store compositions
    all_compositions = np.zeros((len(sid), n_types), dtype=np.float32)
    cell_indices = []  # indices into full array for tumor cells with neighborhoods

    for ri, roi in enumerate(unique_rois):
        m = np.where((sid == roi) & tumor)[0]
        n = len(m)
        if n < k + 1:
            continue

        coords = np.column_stack([cx[m], cy[m]])
        tree = cKDTree(coords)
        _, nb_idx = tree.query(coords, k=k + 1)
        nb_idx = nb_idx[:, 1:]  # exclude self

        roi_codes = ct_codes[m]
        # One-hot of neighbors
        for i in range(n):
            nb_codes = roi_codes[nb_idx[i]]
            counts = np.bincount(nb_codes, minlength=n_types)
            all_compositions[m[i]] = counts / k
            cell_indices.append(m[i])

        if (ri + 1) % 30 == 0:
            print(f"  [{ri+1}/{len(unique_rois)}]")

    cell_indices = np.array(cell_indices)
    compositions = all_compositions[cell_indices]
    print(f"  {len(cell_indices)} cells with neighborhoods computed")

    # --- Step 2: Cluster neighborhood compositions ---
    n_cn = 10
    print(f"\nClustering into {n_cn} cellular neighborhoods (MiniBatchKMeans)...")
    kmeans = MiniBatchKMeans(n_clusters=n_cn, random_state=42, batch_size=10000, n_init=3)
    cn_labels = kmeans.fit_predict(compositions)
    centers = kmeans.cluster_centers_

    # Assign labels back to all cells
    cn_full = np.full(len(sid), -1, dtype=np.int32)
    cn_full[cell_indices] = cn_labels

    # --- Step 3: Name neighborhoods ---
    cn_names = []
    for ci in range(n_cn):
        profile = centers[ci]
        top_idx = np.argsort(-profile)[:3]
        parts = []
        for ti in top_idx:
            if profile[ti] > 0.1:
                parts.append(f"{CN_TYPES[ti]} {profile[ti]:.0%}")
        name = f"CN{ci+1}: {' + '.join(parts[:2])}"
        cn_names.append(name)
        print(f"  {name} (n={np.sum(cn_labels == ci):,})")

    # --- Step 4: CN × compartment cross-tabulation ---
    cn_by_comp = {}
    for ci in range(n_cn):
        ci_mask = cn_full[cell_indices] == ci
        ci_comps = comp[cell_indices[ci_mask]]
        n_foll = np.sum([c in T_FOLL for c in ci_comps])
        n_inter = np.sum([c in T_INTER for c in ci_comps])
        n_total = len(ci_comps)
        cn_by_comp[ci] = {'foll': n_foll / max(n_total, 1),
                           'inter': n_inter / max(n_total, 1)}

    # --- Step 5: Figure ---
    # Load TMA info once for spatial maps
    tma = load_array(h5py.File(args.t_panel, 'r'), 'tma')

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.30, wspace=0.25)

    # (a) Concept cartoon
    ax_cartoon = fig.add_subplot(gs[0, 0])
    cartoon_path = os.path.join('output', 'hypothesis_cartoons', 'spatial_cellular_neighborhoods.png')
    if os.path.exists(cartoon_path):
        cartoon = mpimg.imread(cartoon_path)
        ax_cartoon.imshow(cartoon)
    ax_cartoon.axis('off')
    ax_cartoon.text(-0.02, 1.03, 'a', transform=ax_cartoon.transAxes,
                    fontsize=14, fontweight='bold', va='top')

    # (b) CN composition heatmap
    ax_hm = fig.add_subplot(gs[0, 1:3])
    im = ax_hm.imshow(centers, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    ax_hm.set_xticks(range(n_types))
    ax_hm.set_xticklabels(CN_TYPES, fontsize=9, rotation=45, ha='right')
    ax_hm.set_yticks(range(n_cn))
    # Short CN names with dominant type
    cn_short = []
    for ci in range(n_cn):
        top_i = np.argmax(centers[ci])
        cn_short.append(f'CN{ci+1}: {CN_TYPES[top_i]} ({np.sum(cn_labels==ci):,})')
    ax_hm.set_yticklabels(cn_short, fontsize=8)
    for i in range(n_cn):
        for j in range(n_types):
            val = centers[i, j]
            color = 'white' if val > 0.35 else 'black'
            ax_hm.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
    ax_hm.set_title('Cellular Neighborhood Composition', fontsize=11, fontweight='bold')
    ax_hm.text(-0.05, 1.03, 'b', transform=ax_hm.transAxes, fontsize=14, fontweight='bold', va='top')
    plt.colorbar(im, ax=ax_hm, shrink=0.7, label='Fraction of neighbors')

    # (c) CN × compartment enrichment
    ax_comp = fig.add_subplot(gs[0, 3])
    comp_data = np.array([[cn_by_comp[ci]['foll'], cn_by_comp[ci]['inter']] for ci in range(n_cn)])
    x = np.arange(n_cn)
    ax_comp.barh(x - 0.2, comp_data[:, 0], height=0.35, color='#FFAAAA', label='Follicular')
    ax_comp.barh(x + 0.2, comp_data[:, 1], height=0.35, color='#AAAAFF', label='Interfollicular')
    ax_comp.set_yticks(x)
    ax_comp.set_yticklabels([f'CN{i+1}' for i in range(n_cn)], fontsize=8)
    ax_comp.set_xlabel('Fraction of CN cells', fontsize=9)
    ax_comp.set_title('Compartment Distribution', fontsize=11, fontweight='bold')
    ax_comp.legend(fontsize=8, loc='lower right')
    ax_comp.text(-0.15, 1.03, 'c', transform=ax_comp.transAxes, fontsize=14, fontweight='bold', va='top')
    ax_comp.invert_yaxis()

    # (d-g) Spatial maps — pick 4 ROIs from DIFFERENT TMAs
    roi_cn_diversity = {}
    roi_tma_map = {}
    for roi in unique_rois:
        m = np.where((sid == roi) & tumor)[0]
        cn_vals = cn_full[m]
        cn_valid = cn_vals[cn_vals >= 0]
        if len(cn_valid) < 2000:
            continue
        roi_cn_diversity[roi] = len(set(cn_valid))
        roi_tma_map[roi] = tma[m[0]]

    # Sort by CN diversity, then pick one per TMA
    sorted_rois = sorted(roi_cn_diversity, key=roi_cn_diversity.get, reverse=True)
    selected_rois = []
    selected_tmas = set()
    for roi in sorted_rois:
        t = roi_tma_map[roi]
        if t not in selected_tmas:
            selected_rois.append(roi)
            selected_tmas.add(t)
        if len(selected_rois) == 4:
            break
    # If fewer than 4 TMAs, fill with best remaining
    if len(selected_rois) < 4:
        for roi in sorted_rois:
            if roi not in selected_rois:
                selected_rois.append(roi)
            if len(selected_rois) == 4:
                break

    for pi, roi in enumerate(selected_rois):
        ax_sp = fig.add_subplot(gs[1, pi])
        m = np.where((sid == roi) & tumor)[0]
        roi_cx = cx[m]
        roi_cy = cy[m]
        roi_cn = cn_full[m]

        # Plot cells colored by CN
        for ci in range(n_cn):
            ci_mask = roi_cn == ci
            if np.any(ci_mask):
                ax_sp.scatter(roi_cx[ci_mask], roi_cy[ci_mask], c=[CN_COLORS[ci]],
                              s=0.3, alpha=0.6, rasterized=True)
        # Unassigned CN
        unk_mask = roi_cn < 0
        if np.any(unk_mask):
            ax_sp.scatter(roi_cx[unk_mask], roi_cy[unk_mask], c='#DDDDDD',
                          s=0.1, alpha=0.2, rasterized=True)

        ax_sp.set_aspect('equal')
        ax_sp.invert_yaxis()
        ax_sp.set_xticks([])
        ax_sp.set_yticks([])
        roi_tma_val = roi_tma_map[roi]
        ax_sp.set_title(f'{roi} ({roi_tma_val})\n{len(m):,} cells, {roi_cn_diversity[roi]} CNs',
                        fontsize=9)
        label = chr(ord('d') + pi)
        ax_sp.text(-0.02, 1.05, label, transform=ax_sp.transAxes,
                   fontsize=14, fontweight='bold', va='top')

    # Legend for CNs — larger, across bottom
    legend_elements = [Patch(facecolor=CN_COLORS[i],
                              label=f'CN{ci+1}: {CN_TYPES[np.argmax(centers[ci])]}')
                       for ci, i in enumerate(range(n_cn))]
    fig.legend(handles=legend_elements, loc='lower center', fontsize=8, ncol=5,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle('T-panel: Cellular Neighborhoods (k=20, 10 clusters)',
                 fontsize=13, fontweight='bold')

    out_path = os.path.join(args.output_dir, 'fig_cellular_neighborhoods_T.png')
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
