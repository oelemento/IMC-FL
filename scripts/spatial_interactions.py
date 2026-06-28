#!/usr/bin/env python3
"""Spatial crosstalk #1 & #2: Neighborhood enrichment analysis.

Builds per-ROI kNN graphs, counts cell type pair frequencies at edges,
permutation test for enrichment/depletion. Also runs compartment-conditioned
version (follicular vs interfollicular subsets).

Output: output/hypotheses_v8/fig_nhood_enrichment_T.png
"""

import argparse, os, sys
import numpy as np
import h5py
from scipy.spatial import cKDTree
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
# Data loading (shared pattern)
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


# Broad cell type grouping for cleaner interaction matrix
BROAD_MAP = {
    'B cells (CXCR5hi)': 'B cells',
    'Other B cells': 'B cells',
    'B cells (CD20hi)': 'B cells',
    'B cells (weak CD20)': 'B cells',
    'B cells (TOXhi)': 'B cells',
    'GC B cells': 'GC B cells',
    'Activated B / Plasmablast': 'GC B cells',
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

# Interaction-relevant types (exclude Unassigned/Other for cleaner matrix)
INTERACTION_TYPES = ['B cells', 'GC B cells', 'CD4 T', 'CD8 T', 'CD8 T exh',
                     'Treg', 'Cytotoxic', 'Macrophages']

T_FOLL = ['GC core', 'Follicle core (GC/CD20hi/CXCR5hi)', 'Follicle mantle (CXCR5hi)',
          'Activated B / CXCR5hi zone', 'B cell follicle (CD20hi/CXCR5hi)', 'B cell zone']
T_INTER = ['T cell zone (CD4/CD8)', 'Treg-enriched T zone', 'Macrophage-rich zone',
           'Follicle-T zone interface', 'Cytotoxic niche']


def nhood_enrichment_roi(cx, cy, ct_codes, n_types, k=15, n_perm=200):
    """Compute neighborhood enrichment Z-scores for one ROI.

    Returns (n_types, n_types) Z-score matrix.
    """
    n = len(cx)
    if n < k + 1:
        return None

    coords = np.column_stack([cx, cy])
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)  # includes self
    neighbors = indices[:, 1:]  # exclude self

    # Observed counts: for each (source_type, neighbor_type) pair
    obs = np.zeros((n_types, n_types), dtype=np.float64)
    for i in range(n):
        src = ct_codes[i]
        if src < 0:
            continue
        for j_idx in range(k):
            nb = ct_codes[neighbors[i, j_idx]]
            if nb < 0:
                continue
            obs[src, nb] += 1

    # Permutation null
    perm_counts = np.zeros((n_perm, n_types, n_types), dtype=np.float64)
    for p in range(n_perm):
        perm_codes = np.random.permutation(ct_codes)
        for i in range(n):
            src = perm_codes[i]
            if src < 0:
                continue
            for j_idx in range(k):
                nb = perm_codes[neighbors[i, j_idx]]
                if nb < 0:
                    continue
                perm_counts[p, src, nb] += 1

    # Z-score
    perm_mean = perm_counts.mean(axis=0)
    perm_std = perm_counts.std(axis=0)
    perm_std[perm_std == 0] = 1  # avoid div by zero
    z = (obs - perm_mean) / perm_std
    return z


def nhood_enrichment_roi_fast(cx, cy, ct_codes, n_types, k=15, n_perm=200):
    """Vectorized neighborhood enrichment for one ROI."""
    n = len(cx)
    if n < k + 1:
        return None

    coords = np.column_stack([cx, cy])
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    neighbors = indices[:, 1:]

    # Mask for valid types
    valid = ct_codes >= 0

    # Observed: vectorized
    src_types = ct_codes  # (n,)
    nb_types = ct_codes[neighbors]  # (n, k)

    obs = np.zeros((n_types, n_types), dtype=np.float64)
    for s in range(n_types):
        s_mask = (src_types == s) & valid
        if not np.any(s_mask):
            continue
        nb_of_s = nb_types[s_mask]  # (n_s, k)
        nb_valid = nb_of_s[nb_of_s >= 0]
        if len(nb_valid) == 0:
            continue
        counts = np.bincount(nb_valid, minlength=n_types)
        obs[s, :] = counts[:n_types]

    # Permutation null (vectorized)
    perm_sums = np.zeros((n_types, n_types), dtype=np.float64)
    perm_sq_sums = np.zeros((n_types, n_types), dtype=np.float64)
    for p in range(n_perm):
        perm_codes = np.random.permutation(ct_codes)
        perm_valid = perm_codes >= 0
        perm_nb = perm_codes[neighbors]
        perm_obs = np.zeros((n_types, n_types), dtype=np.float64)
        for s in range(n_types):
            s_mask = (perm_codes == s) & perm_valid
            if not np.any(s_mask):
                continue
            nb_of_s = perm_nb[s_mask]
            nb_v = nb_of_s[nb_of_s >= 0]
            if len(nb_v) == 0:
                continue
            counts = np.bincount(nb_v, minlength=n_types)
            perm_obs[s, :] = counts[:n_types]
        perm_sums += perm_obs
        perm_sq_sums += perm_obs ** 2

    perm_mean = perm_sums / n_perm
    perm_var = perm_sq_sums / n_perm - perm_mean ** 2
    perm_std = np.sqrt(np.maximum(perm_var, 0))
    perm_std[perm_std == 0] = 1
    z = (obs - perm_mean) / perm_std
    return z


def run_analysis(args):
    print("Loading T-panel data...")
    f_v8 = h5py.File(args.t_panel, 'r')
    sid = rename_labels(load_array(f_v8, 'sample_id'))
    ct_raw = rename_labels(load_array(f_v8, 'cell_type'))
    cx = np.array(f_v8['obs']['centroid_x'])
    cy = np.array(f_v8['obs']['centroid_y'])
    tumor = get_tumor_mask(sid)

    # Load compartments from UTAG file
    f_utag = h5py.File(args.t_utag, 'r')
    comp = rename_labels(load_array(f_utag, 'compartment_name'))
    f_utag.close()

    # Map to broad types
    ct_broad = np.array([BROAD_MAP.get(c, 'Other') for c in ct_raw])

    # Encode
    type_to_idx = {t: i for i, t in enumerate(INTERACTION_TYPES)}
    n_types = len(INTERACTION_TYPES)
    ct_codes = np.array([type_to_idx.get(c, -1) for c in ct_broad])

    # Classify compartments
    comp_set = set(comp)
    is_foll = np.array([c in T_FOLL for c in comp])
    is_inter = np.array([c in T_INTER for c in comp])

    unique_rois = sorted(set(sid[tumor]))
    print(f"  {len(unique_rois)} tumor ROIs, {n_types} interaction types")

    # --- Run per-ROI enrichment ---
    z_all = []
    z_foll = []
    z_inter = []
    min_roi_cells = 200

    for ri, roi in enumerate(unique_rois):
        m = (sid == roi) & tumor
        n = int(np.sum(m))
        if n < min_roi_cells:
            continue

        roi_cx = cx[m]
        roi_cy = cy[m]
        roi_codes = ct_codes[m]
        roi_comp_foll = is_foll[m]
        roi_comp_inter = is_inter[m]

        # All cells
        z = nhood_enrichment_roi_fast(roi_cx, roi_cy, roi_codes, n_types, k=15, n_perm=200)
        if z is not None:
            z_all.append(z)

        # Follicular subset
        foll_mask = roi_comp_foll
        if np.sum(foll_mask) >= min_roi_cells:
            z_f = nhood_enrichment_roi_fast(
                roi_cx[foll_mask], roi_cy[foll_mask], roi_codes[foll_mask],
                n_types, k=10, n_perm=100)
            if z_f is not None:
                z_foll.append(z_f)

        # Interfollicular subset
        inter_mask = roi_comp_inter
        if np.sum(inter_mask) >= min_roi_cells:
            z_i = nhood_enrichment_roi_fast(
                roi_cx[inter_mask], roi_cy[inter_mask], roi_codes[inter_mask],
                n_types, k=10, n_perm=100)
            if z_i is not None:
                z_inter.append(z_i)

        if (ri + 1) % 20 == 0:
            print(f"  [{ri+1}/{len(unique_rois)}] {len(z_all)} all, {len(z_foll)} foll, {len(z_inter)} inter")

    f_v8.close()

    print(f"\nDone: {len(z_all)} ROIs (all), {len(z_foll)} (foll), {len(z_inter)} (inter)")

    # Aggregate: mean Z-score across ROIs
    z_mean_all = np.nanmean(z_all, axis=0)
    z_mean_foll = np.nanmean(z_foll, axis=0) if z_foll else np.zeros((n_types, n_types))
    z_mean_inter = np.nanmean(z_inter, axis=0) if z_inter else np.zeros((n_types, n_types))

    # --- Figure: 2×2 grid ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    # (a) Concept cartoon
    ax_cartoon = axes[0, 0]
    cartoon_path = os.path.join('output', 'hypothesis_cartoons', 'spatial_nhood_enrichment.png')
    if os.path.exists(cartoon_path):
        cartoon = mpimg.imread(cartoon_path)
        ax_cartoon.imshow(cartoon)
    ax_cartoon.axis('off')
    ax_cartoon.text(-0.02, 1.03, 'a', transform=ax_cartoon.transAxes,
                    fontsize=14, fontweight='bold', va='top')

    vmax = max(np.abs(z_mean_all).max(), np.abs(z_mean_foll).max(),
               np.abs(z_mean_inter).max(), 3)
    vmax = min(vmax, 15)

    # (b-d) Enrichment heatmaps
    hm_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, z_mat, title, label in [
        (hm_axes[0], z_mean_all, f'All cells (n={len(z_all)} ROIs)', 'b'),
        (hm_axes[1], z_mean_foll, f'Follicular (n={len(z_foll)} ROIs)', 'c'),
        (hm_axes[2], z_mean_inter, f'Interfollicular (n={len(z_inter)} ROIs)', 'd'),
    ]:
        im = ax.imshow(z_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_xticks(range(n_types))
        ax.set_yticks(range(n_types))
        ax.set_xticklabels(INTERACTION_TYPES, fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels(INTERACTION_TYPES, fontsize=8)
        ax.set_xlabel('Neighbor type', fontsize=9)
        if ax == hm_axes[0]:
            ax.set_ylabel('Source type', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

        # Annotate values
        for i in range(n_types):
            for j in range(n_types):
                val = z_mat[i, j]
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=6.5, color=color)

    plt.colorbar(im, ax=hm_axes, shrink=0.8, label='Mean Z-score (enrichment)')
    fig.suptitle('T-panel: Neighborhood Enrichment (kNN k=15, 200 permutations)',
                 fontsize=13, fontweight='bold')

    out_path = os.path.join(args.output_dir, 'fig_nhood_enrichment_T.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")

    # Print summary
    print("\n=== TOP ENRICHED INTERACTIONS (all cells) ===")
    flat = [(z_mean_all[i, j], INTERACTION_TYPES[i], INTERACTION_TYPES[j])
            for i in range(n_types) for j in range(n_types)]
    flat.sort(key=lambda x: -x[0])
    for z, src, nb in flat[:10]:
        print(f"  {src:20s} → {nb:20s}: Z={z:+.1f}")
    print("\n=== TOP DEPLETED INTERACTIONS (all cells) ===")
    flat.sort(key=lambda x: x[0])
    for z, src, nb in flat[:10]:
        print(f"  {src:20s} → {nb:20s}: Z={z:+.1f}")

    # Print compartment differences
    print("\n=== ENRICHMENT DIFFERENCE (Follicular - Interfollicular) ===")
    z_diff = z_mean_foll - z_mean_inter
    diff_flat = [(z_diff[i, j], INTERACTION_TYPES[i], INTERACTION_TYPES[j])
                 for i in range(n_types) for j in range(n_types)]
    diff_flat.sort(key=lambda x: -abs(x[0]))
    for z, src, nb in diff_flat[:10]:
        print(f"  {src:20s} → {nb:20s}: ΔZ={z:+.1f}")


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
