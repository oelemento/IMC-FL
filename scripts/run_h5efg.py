#!/usr/bin/env python3
"""H5e/H5f/H5g: HLA expression on tumor B cells and TME reshaping.

H5e: MHC-II (HLA-DR) on B cells varies across ROIs;
     HLA-DR-low ROIs have fewer CD4 T cells.  [S-panel]
H5f: HLA Class I on B cells correlates with CD8 T exhaustion;
     low MHC-I regions have less CD8 T exhaustion.  [Cross-panel: S→T]
H5g: Spatial proximity of CD4 T to B cells depends on B cell
     HLA-DR intensity (antigen presentation drives co-localization).  [S-panel]

HLA_DR and HLA_Class_I are S-panel markers. CD8 T exhausted is a T-panel
annotation. H5e and H5g use S-panel only. H5f correlates S-panel HLA-I
per ROI with T-panel CD8 T exhausted fraction per matched ROI.

Output: output/hypotheses_v8/fig_h5efg_T.png
"""

import argparse, os
import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, kruskal

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

# S-panel rename is simpler — 'B cells' stays as-is in S-panel
S_DISPLAY_RENAME = {
    'Low quality / Unassigned': 'Unassigned',
}

def rename_labels(arr, rename_map=None):
    if rename_map is None:
        rename_map = DISPLAY_RENAME
    return np.array([rename_map.get(v, v) for v in arr])

def get_tumor_mask(sample_ids):
    control_tags = ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal',
                    '_ton_', '_adr_']
    return np.array([not any(t in s.lower() for t in control_tags)
                     for s in sample_ids])


# S-panel cell type sets
S_B_CELL_TYPES = {'B cells (BCL2+)', 'B cells (PAX5+)', 'B cells'}
S_CD4_T_TYPES = {'CD4 T cells'}
S_CD8_T_TYPES = {'CD8 T cells'}
S_LQ_TYPES = {'Unassigned'}

# T-panel cell type sets (for cross-panel H5f)
T_CD8_EXH_TYPES = {'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)'}
T_LQ_TYPES = {'Unassigned'}

TMA_CMAP = plt.cm.Set1


# ---------------------------------------------------------------------------
# Spatial scatter helper
# ---------------------------------------------------------------------------

def plot_spatial(ax, roi_cx, roi_cy, roi_ct, roi_highlight, title):
    """Plot spatial scatter with highlighted CD4 T cells."""
    roi_lq = np.array([c in S_LQ_TYPES for c in roi_ct])

    # LQ cells
    lq_mask = roi_lq
    if np.any(lq_mask):
        ax.scatter(roi_cx[lq_mask], roi_cy[lq_mask], c='#D3D3D3',
                   s=0.3, alpha=0.4, rasterized=True, zorder=1)

    # Non-LQ, non-highlight
    other_mask = ~roi_lq & ~roi_highlight
    if np.any(other_mask):
        ax.scatter(roi_cx[other_mask], roi_cy[other_mask], c='#A0A0A0',
                   s=0.5, alpha=0.5, rasterized=True, zorder=2)

    # Highlighted cells (CD4 T) in gold
    if np.any(roi_highlight):
        ax.scatter(roi_cx[roi_highlight], roi_cy[roi_highlight], c='#FFD700',
                   s=8, alpha=0.9, edgecolors='black', linewidths=0.3,
                   rasterized=True, zorder=3)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)

    legend_elements = [
        Patch(facecolor='#FFD700', edgecolor='black', label='CD4 T cells'),
        Patch(facecolor='#A0A0A0', label='Other typed cells'),
        Patch(facecolor='#D3D3D3', label='Unassigned'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right',
              framealpha=0.8)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(args):
    # ===================================================================
    # Load S-panel (primary: HLA markers + CD4 T + spatial)
    # ===================================================================
    print("Loading S-panel data...")
    fs = h5py.File(args.s_panel, 'r')
    s_sid = rename_labels(load_array(fs, 'sample_id'), S_DISPLAY_RENAME)
    s_ct = rename_labels(load_array(fs, 'cell_type'), S_DISPLAY_RENAME)
    s_cx = np.array(fs['obs']['centroid_x'])
    s_cy = np.array(fs['obs']['centroid_y'])
    s_tma = load_array(fs, 'tma')
    s_tumor = get_tumor_mask(s_sid)

    # S-panel marker intensities
    var_key = '_index' if '_index' in fs['var'] else 'index'
    s_markers = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in fs['var'][var_key][:]]
    s_marker_idx = {m: i for i, m in enumerate(s_markers)}

    assert 'HLA_DR' in s_marker_idx, f"HLA_DR not in S-panel: {s_markers}"
    assert 'HLA_Class_I' in s_marker_idx, f"HLA_Class_I not in S-panel: {s_markers}"

    print("  Loading S-panel expression matrix...")
    s_X = fs['X'][:]
    s_hladr = s_X[:, s_marker_idx['HLA_DR']].astype(np.float32)
    s_hlai = s_X[:, s_marker_idx['HLA_Class_I']].astype(np.float32)
    del s_X

    # S-panel boolean masks
    s_is_b = np.array([c in S_B_CELL_TYPES for c in s_ct])
    s_is_cd4t = np.array([c in S_CD4_T_TYPES for c in s_ct])
    s_is_lq = np.array([c in S_LQ_TYPES for c in s_ct])

    # Encode S-panel cell types for fast counting
    s_all_ct = sorted(set(s_ct[s_tumor & ~s_is_lq]))
    s_ct_to_int = {c: i for i, c in enumerate(s_all_ct)}
    s_ct_enc = np.array([s_ct_to_int.get(c, -1) for c in s_ct])
    s_n_ct = len(s_all_ct)

    s_unique_rois = sorted(set(s_sid[s_tumor]))
    print(f"  {len(s_unique_rois)} S-panel tumor ROIs, {len(s_markers)} markers")
    print(f"  B cells: {np.sum(s_is_b & s_tumor):,}, "
          f"CD4 T: {np.sum(s_is_cd4t & s_tumor):,}")

    # ===================================================================
    # Load T-panel (for cross-panel H5f: CD8 T exhausted fraction)
    # ===================================================================
    print("\nLoading T-panel data (for H5f cross-panel)...")
    ft = h5py.File(args.t_panel, 'r')
    t_sid = rename_labels(load_array(ft, 'sample_id'))
    t_ct = rename_labels(load_array(ft, 'cell_type'))
    t_tumor = get_tumor_mask(t_sid)

    t_is_cd8exh = np.array([c in T_CD8_EXH_TYPES for c in t_ct])
    t_is_lq = np.array([c in T_LQ_TYPES for c in t_ct])

    # Per-ROI CD8 T exhausted fraction from T-panel
    t_unique_rois = sorted(set(t_sid[t_tumor]))
    t_roi_cd8exh = {}
    for roi in t_unique_rois:
        m = (t_sid == roi) & t_tumor
        n_nonlq = int(np.sum(m & ~t_is_lq))
        if n_nonlq < 200:
            continue
        t_roi_cd8exh[roi] = float(np.sum(m & t_is_cd8exh)) / n_nonlq

    ft.close()
    print(f"  {len(t_roi_cd8exh)} T-panel ROIs with CD8 exh data")

    # Find paired ROIs (same sample_id in both panels)
    paired_rois = sorted(set(s_unique_rois) & set(t_roi_cd8exh.keys()))
    print(f"  {len(paired_rois)} paired ROIs for cross-panel H5f")

    # ===================================================================
    # Step 1: Per-ROI metrics from S-panel
    # ===================================================================
    print("\nComputing per-ROI HLA metrics (S-panel)...")
    roi_data = []

    for roi in s_unique_rois:
        m = (s_sid == roi) & s_tumor
        m_idx = np.where(m)[0]
        n_nonlq = int(np.sum(~s_is_lq[m_idx]))
        if n_nonlq < 200:
            continue

        b_in_roi = s_is_b[m_idx]
        n_b = int(np.sum(b_in_roi))
        if n_b < 50:
            continue

        # Mean HLA on B cells
        b_global_idx = m_idx[b_in_roi]
        mean_hladr = float(s_hladr[b_global_idx].mean())
        mean_hlai = float(s_hlai[b_global_idx].mean())

        # Cell type fractions
        roi_ct_enc = s_ct_enc[m_idx]
        valid = roi_ct_enc >= 0
        counts = np.bincount(roi_ct_enc[valid], minlength=s_n_ct)
        fracs = counts / n_nonlq

        frac_cd4t = float(np.sum(s_is_cd4t[m_idx])) / n_nonlq

        # Cross-panel: T-panel CD8 T exhausted for this ROI (if available)
        frac_cd8exh_t = t_roi_cd8exh.get(roi, np.nan)

        roi_tma = s_tma[m_idx[0]]

        roi_data.append({
            'roi': roi, 'tma': roi_tma,
            'n_total': len(m_idx), 'n_nonlq': n_nonlq, 'n_b': n_b,
            'mean_hladr': mean_hladr, 'mean_hlai': mean_hlai,
            'frac_cd4t': frac_cd4t,
            'frac_cd8exh_t': frac_cd8exh_t,  # from T-panel
            'ct_fracs': fracs,
        })

    print(f"  {len(roi_data)} S-panel ROIs with >=200 non-LQ and >=50 B cells")

    # Vectorize
    tmas = [d['tma'] for d in roi_data]
    hladr_arr = np.array([d['mean_hladr'] for d in roi_data])
    hlai_arr = np.array([d['mean_hlai'] for d in roi_data])
    cd4t_arr = np.array([d['frac_cd4t'] for d in roi_data])
    cd8exh_arr = np.array([d['frac_cd8exh_t'] for d in roi_data])

    # ===================================================================
    # H5e: HLA-DR vs CD4 T fraction (S-panel within-panel)
    # ===================================================================
    rho_5e, p_5e = spearmanr(hladr_arr, cd4t_arr)
    print(f"\nH5e: HLA-DR vs CD4 T fraction (S-panel): "
          f"rho={rho_5e:.3f}, p={p_5e:.2e}")

    # ===================================================================
    # H5f: HLA-I vs CD8 T exhausted (cross-panel S→T)
    # ===================================================================
    # Filter to ROIs with paired T-panel data
    paired_mask = ~np.isnan(cd8exh_arr)
    n_paired = int(np.sum(paired_mask))
    hlai_paired = hlai_arr[paired_mask]
    cd8exh_paired = cd8exh_arr[paired_mask]

    if n_paired >= 10:
        rho_5f, p_5f = spearmanr(hlai_paired, cd8exh_paired)
    else:
        rho_5f, p_5f = np.nan, np.nan
    print(f"H5f: HLA-I vs CD8 T exh (cross-panel, n={n_paired}): "
          f"rho={rho_5f:.3f}, p={p_5f:.2e}")

    # ===================================================================
    # H5g: kNN analysis (HLA-DR quartiles → CD4 T neighbor fraction)
    # ===================================================================
    k = 20
    print(f"\nH5g: Computing k={k} neighborhoods for B cells per ROI...")

    quartile_cd4_fracs = {q: [] for q in range(4)}

    for ri, roi in enumerate(s_unique_rois):
        m_idx = np.where((s_sid == roi) & s_tumor)[0]
        n = len(m_idx)
        if n < k + 1:
            continue

        roi_is_b = s_is_b[m_idx]
        roi_is_cd4t = s_is_cd4t[m_idx]
        n_b_roi = int(np.sum(roi_is_b))
        if n_b_roi < 100:
            continue

        # kNN tree
        coords = np.column_stack([s_cx[m_idx], s_cy[m_idx]])
        tree = cKDTree(coords)
        _, nb_idx = tree.query(coords, k=k + 1)
        nb_idx = nb_idx[:, 1:]

        # Vectorized: B cell HLA-DR and CD4 T neighbor fractions
        b_local_idx = np.where(roi_is_b)[0]
        b_hladr = s_hladr[m_idx[b_local_idx]]
        b_nb_is_cd4t = roi_is_cd4t[nb_idx[b_local_idx]]  # (n_b, k)
        b_cd4t_frac = b_nb_is_cd4t.mean(axis=1)

        # Bin by within-ROI HLA-DR quartile
        quartile_edges = np.percentile(b_hladr, [25, 50, 75])
        q_assign = np.digitize(b_hladr, quartile_edges)  # 0,1,2,3

        for q in range(4):
            q_mask = q_assign == q
            if np.sum(q_mask) > 10:
                quartile_cd4_fracs[q].append(float(b_cd4t_frac[q_mask].mean()))

        if (ri + 1) % 30 == 0:
            print(f"  [{ri+1}/{len(s_unique_rois)}]")

    q_means = [np.mean(quartile_cd4_fracs[q]) for q in range(4)]
    q_sems = [np.std(quartile_cd4_fracs[q]) / np.sqrt(len(quartile_cd4_fracs[q]))
              for q in range(4)]
    q_ns = [len(quartile_cd4_fracs[q]) for q in range(4)]

    h_stat, p_5g = kruskal(*[quartile_cd4_fracs[q] for q in range(4)])
    print(f"\nH5g: CD4 T neighbor fraction by HLA-DR quartile:")
    for q in range(4):
        print(f"  Q{q+1}: mean={q_means[q]:.4f} +/- {q_sems[q]:.4f} "
              f"(n={q_ns[q]} ROIs)")
    print(f"  Kruskal-Wallis H={h_stat:.1f}, p={p_5g:.2e}")

    # ===================================================================
    # Leave-one-TMA-out sensitivity
    # ===================================================================
    print("\nLeave-one-TMA-out sensitivity...")
    unique_tmas = sorted(set(tmas))
    loto_5e = {}
    loto_5f = {}
    for t in unique_tmas:
        # H5e
        mask_e = np.array([tm != t for tm in tmas])
        if np.sum(mask_e) < 10:
            continue
        r_e, p_e = spearmanr(hladr_arr[mask_e], cd4t_arr[mask_e])
        loto_5e[t] = (r_e, p_e)

        # H5f (paired only)
        mask_f = mask_e & paired_mask
        if np.sum(mask_f) >= 10:
            r_f, p_f = spearmanr(hlai_arr[mask_f], cd8exh_arr[mask_f])
        else:
            r_f, p_f = np.nan, np.nan
        loto_5f[t] = (r_f, p_f)

        print(f"  Excl {t}: H5e rho={r_e:.3f} p={p_e:.2e}, "
              f"H5f rho={r_f:.3f} p={p_f:.2e}")

    # ===================================================================
    # Driver analysis: cell type fractions vs mean B-cell HLA-DR
    # ===================================================================
    print("\nDriver analysis: cell type fractions vs mean B-cell HLA-DR...")
    ct_frac_matrix = np.array([d['ct_fracs'] for d in roi_data])
    driver_rhos = {}
    for ci, ctype in enumerate(s_all_ct):
        rho, _ = spearmanr(hladr_arr, ct_frac_matrix[:, ci])
        driver_rhos[ctype] = rho
    driver_sorted = sorted(driver_rhos.items(), key=lambda x: -abs(x[1]))
    for ctype, rho in driver_sorted[:8]:
        print(f"  {ctype:35s}: rho={rho:+.3f}")

    # ===================================================================
    # Representative ROIs (15th and 85th percentile of mean B-cell HLA-DR)
    # ===================================================================
    print("\nSelecting representative ROIs...")
    plot_candidates = []
    for d in roi_data:
        min_cells = 5000 if 'biomax' in d['tma'].lower() else 8000
        if d['n_nonlq'] >= min_cells:
            plot_candidates.append(d)
    if len(plot_candidates) < 4:
        plot_candidates = [d for d in roi_data if d['n_nonlq'] >= 3000]

    cand_hladr = np.array([d['mean_hladr'] for d in plot_candidates])
    p15 = np.percentile(cand_hladr, 15)
    p85 = np.percentile(cand_hladr, 85)

    hi_idx = int(np.argmin(np.abs(cand_hladr - p85)))
    lo_idx = int(np.argmin(np.abs(cand_hladr - p15)))
    if hi_idx == lo_idx:
        si = np.argsort(cand_hladr)
        lo_idx, hi_idx = int(si[0]), int(si[-1])

    roi_hi = plot_candidates[hi_idx]
    roi_lo = plot_candidates[lo_idx]
    print(f"  High HLA-DR: {roi_hi['roi']} (mean={roi_hi['mean_hladr']:.2f}, "
          f"CD4 T={roi_hi['frac_cd4t']*100:.1f}%)")
    print(f"  Low  HLA-DR: {roi_lo['roi']} (mean={roi_lo['mean_hladr']:.2f}, "
          f"CD4 T={roi_lo['frac_cd4t']*100:.1f}%)")

    # ===================================================================
    # Figure — 3x3 grid
    # ===================================================================
    print("\nGenerating figure...")
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    tma_colors = {t: TMA_CMAP(i / max(len(unique_tmas) - 1, 1))
                  for i, t in enumerate(unique_tmas)}

    # -- (a) Concept cartoon --
    ax_a = fig.add_subplot(gs[0, 0])
    cartoon_path = os.path.join('output', 'hypothesis_cartoons',
                                'h5efg_hla_tme.png')
    if os.path.exists(cartoon_path):
        ax_a.imshow(mpimg.imread(cartoon_path))
    ax_a.axis('off')
    ax_a.text(-0.02, 1.03, 'a', transform=ax_a.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # -- (b) H5e scatter: HLA-DR vs CD4 T fraction (S-panel) --
    ax_b = fig.add_subplot(gs[0, 1])
    for t in unique_tmas:
        mask = np.array([tm == t for tm in tmas])
        ax_b.scatter(hladr_arr[mask], cd4t_arr[mask] * 100,
                     c=[tma_colors[t]], s=20, alpha=0.7, label=t,
                     edgecolors='none')
    z = np.polyfit(hladr_arr, cd4t_arr * 100, 1)
    xl = np.linspace(hladr_arr.min(), hladr_arr.max(), 100)
    ax_b.plot(xl, np.polyval(z, xl), 'k--', alpha=0.5, lw=1)
    ax_b.set_xlabel('Mean HLA-DR on B cells (z-score)', fontsize=10)
    ax_b.set_ylabel('CD4 T fraction (%, S-panel)', fontsize=10)
    pstr = f'p={p_5e:.1e}' if p_5e < 0.001 else f'p={p_5e:.3f}'
    ax_b.set_title(f'H5e: HLA-DR vs CD4 T density\n'
                   f'\u03c1={rho_5e:.3f}, {pstr} (n={len(hladr_arr)})',
                   fontsize=11, fontweight='bold')
    ax_b.legend(fontsize=7, loc='best', framealpha=0.7)
    ax_b.text(-0.12, 1.03, 'b', transform=ax_b.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # -- (c) H5f scatter: HLA-I vs CD8 T exh (cross-panel) --
    ax_c = fig.add_subplot(gs[0, 2])
    paired_tmas = [tmas[i] for i in range(len(tmas)) if paired_mask[i]]
    for t in unique_tmas:
        mask = np.array([tm == t for tm in paired_tmas])
        if not np.any(mask):
            continue
        ax_c.scatter(hlai_paired[mask], cd8exh_paired[mask] * 100,
                     c=[tma_colors[t]], s=20, alpha=0.7, label=t,
                     edgecolors='none')
    if n_paired > 2:
        z = np.polyfit(hlai_paired, cd8exh_paired * 100, 1)
        xl = np.linspace(hlai_paired.min(), hlai_paired.max(), 100)
        ax_c.plot(xl, np.polyval(z, xl), 'k--', alpha=0.5, lw=1)
    ax_c.set_xlabel('Mean HLA Class I on B cells (S-panel z-score)',
                    fontsize=10)
    ax_c.set_ylabel('CD8 T exhausted fraction (%, T-panel)', fontsize=10)
    pstr = f'p={p_5f:.1e}' if p_5f < 0.001 else f'p={p_5f:.3f}'
    ax_c.set_title(f'H5f: HLA-I vs CD8 T exh (cross-panel)\n'
                   f'\u03c1={rho_5f:.3f}, {pstr} (n={n_paired})',
                   fontsize=11, fontweight='bold')
    ax_c.legend(fontsize=7, loc='best', framealpha=0.7)
    ax_c.text(-0.12, 1.03, 'c', transform=ax_c.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # -- (d) H5g: quartile bar chart --
    ax_d = fig.add_subplot(gs[1, 0])
    q_labels = ['Q1\n(lowest)', 'Q2', 'Q3', 'Q4\n(highest)']
    q_colors = ['#4575B4', '#91BFDB', '#FC8D59', '#D73027']
    ax_d.bar(range(4), [m * 100 for m in q_means],
             yerr=[s * 100 for s in q_sems],
             color=q_colors, edgecolor='black', linewidth=0.5, capsize=4)
    ax_d.set_xticks(range(4))
    ax_d.set_xticklabels(q_labels, fontsize=9)
    ax_d.set_xlabel('B cell HLA-DR quartile (within-ROI)', fontsize=10)
    ax_d.set_ylabel('CD4 T neighbor fraction (%)', fontsize=10)
    pstr = f'p={p_5g:.1e}' if p_5g < 0.001 else f'p={p_5g:.3f}'
    ax_d.set_title(f'H5g: HLA-DR \u2192 CD4 T proximity\n'
                   f'Kruskal-Wallis {pstr}',
                   fontsize=11, fontweight='bold')
    ax_d.text(-0.12, 1.03, 'd', transform=ax_d.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # -- (e) Representative ROI: high HLA-DR --
    ax_e = fig.add_subplot(gs[1, 1])
    m_hi = np.where((s_sid == roi_hi['roi']) & s_tumor)[0]
    plot_spatial(ax_e, s_cx[m_hi], s_cy[m_hi], s_ct[m_hi],
                s_is_cd4t[m_hi],
                f"High HLA-DR: {roi_hi['roi']}\n"
                f"(mean={roi_hi['mean_hladr']:.2f}, "
                f"CD4 T={roi_hi['frac_cd4t']*100:.1f}%)")
    ax_e.text(-0.02, 1.05, 'e', transform=ax_e.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # -- (f) Representative ROI: low HLA-DR --
    ax_f = fig.add_subplot(gs[1, 2])
    m_lo = np.where((s_sid == roi_lo['roi']) & s_tumor)[0]
    plot_spatial(ax_f, s_cx[m_lo], s_cy[m_lo], s_ct[m_lo],
                s_is_cd4t[m_lo],
                f"Low HLA-DR: {roi_lo['roi']}\n"
                f"(mean={roi_lo['mean_hladr']:.2f}, "
                f"CD4 T={roi_lo['frac_cd4t']*100:.1f}%)")
    ax_f.text(-0.02, 1.05, 'f', transform=ax_f.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # -- (g) Driver analysis --
    ax_g = fig.add_subplot(gs[2, 0])
    top_n = min(12, len(driver_sorted))
    d_names = [d[0] for d in driver_sorted[:top_n]][::-1]
    d_rhos = [d[1] for d in driver_sorted[:top_n]][::-1]
    bar_colors = ['#D73027' if r > 0 else '#4575B4' for r in d_rhos]
    ax_g.barh(range(len(d_names)), d_rhos, color=bar_colors,
              edgecolor='black', linewidth=0.5)
    ax_g.set_yticks(range(len(d_names)))
    ax_g.set_yticklabels(d_names, fontsize=8)
    ax_g.set_xlabel('Spearman \u03c1 with mean B-cell HLA-DR', fontsize=10)
    ax_g.set_title('Driver: cell type \u2194 HLA-DR', fontsize=11,
                   fontweight='bold')
    ax_g.axvline(0, color='black', linewidth=0.5)
    ax_g.text(-0.15, 1.03, 'g', transform=ax_g.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # -- (h) LOTO H5e --
    ax_h = fig.add_subplot(gs[2, 1])
    loto_names = list(loto_5e.keys())
    loto_rhos_e = [loto_5e[t][0] for t in loto_names]
    loto_ps_e = [loto_5e[t][1] for t in loto_names]
    x_pos = np.arange(len(loto_names))
    ax_h.bar(x_pos, loto_rhos_e,
             color=[tma_colors[t] for t in loto_names],
             edgecolor='black', linewidth=0.5)
    ax_h.axhline(rho_5e, color='black', linestyle='--', linewidth=1,
                 label=f'All: \u03c1={rho_5e:.3f}')
    ax_h.set_xticks(x_pos)
    ax_h.set_xticklabels([f'excl\n{t}' for t in loto_names], fontsize=7)
    ax_h.set_ylabel('Spearman \u03c1', fontsize=10)
    ax_h.set_title('LOTO: H5e (HLA-DR \u2194 CD4 T)',
                   fontsize=11, fontweight='bold')
    ax_h.legend(fontsize=8, loc='best')
    for i, p in enumerate(loto_ps_e):
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        yoff = 0.01 if loto_rhos_e[i] >= 0 else -0.03
        ax_h.text(i, loto_rhos_e[i] + yoff, star, ha='center', fontsize=8)
    ax_h.text(-0.12, 1.03, 'h', transform=ax_h.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # -- (i) LOTO H5f --
    ax_i = fig.add_subplot(gs[2, 2])
    loto_rhos_f = [loto_5f[t][0] for t in loto_names]
    loto_ps_f = [loto_5f[t][1] for t in loto_names]
    ax_i.bar(x_pos, loto_rhos_f,
             color=[tma_colors[t] for t in loto_names],
             edgecolor='black', linewidth=0.5)
    ax_i.axhline(rho_5f, color='black', linestyle='--', linewidth=1,
                 label=f'All: \u03c1={rho_5f:.3f}')
    ax_i.set_xticks(x_pos)
    ax_i.set_xticklabels([f'excl\n{t}' for t in loto_names], fontsize=7)
    ax_i.set_ylabel('Spearman \u03c1', fontsize=10)
    ax_i.set_title('LOTO: H5f (HLA-I \u2194 CD8 exh, cross-panel)',
                   fontsize=11, fontweight='bold')
    ax_i.legend(fontsize=8, loc='best')
    for i, p in enumerate(loto_ps_f):
        if np.isnan(p):
            star = 'n/a'
        else:
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        yoff = 0.01 if (not np.isnan(loto_rhos_f[i]) and loto_rhos_f[i] >= 0) else -0.03
        ax_i.text(i, loto_rhos_f[i] + yoff if not np.isnan(loto_rhos_f[i]) else 0,
                  star, ha='center', fontsize=8)
    ax_i.text(-0.12, 1.03, 'i', transform=ax_i.transAxes,
              fontsize=14, fontweight='bold', va='top')

    fig.suptitle('HLA Expression on Tumor B Cells and TME Reshaping '
                 '(H5e/f/g)\nS-panel (H5e, H5g) + cross-panel S\u2192T (H5f)',
                 fontsize=14, fontweight='bold')

    out_path = os.path.join(args.output_dir, 'fig_h5efg_T.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"H5e: HLA-DR vs CD4 T fraction (S-panel):  "
          f"rho={rho_5e:+.3f}, p={p_5e:.2e}")
    status_5e = ("CONFIRMED" if p_5e < 0.05 and rho_5e > 0
                 else "NOT CONFIRMED" if p_5e >= 0.05
                 else "OPPOSITE DIRECTION")
    print(f"     Status: {status_5e}")

    print(f"H5f: HLA-I vs CD8 T exh (cross-panel, n={n_paired}):  "
          f"rho={rho_5f:+.3f}, p={p_5f:.2e}")
    status_5f = ("CONFIRMED" if p_5f < 0.05 and rho_5f > 0
                 else "NOT CONFIRMED" if p_5f >= 0.05
                 else "OPPOSITE DIRECTION")
    print(f"     Status: {status_5f}")

    print(f"H5g: HLA-DR quartile effect:  H={h_stat:.1f}, p={p_5g:.2e}")
    if q_means[3] > q_means[0] and p_5g < 0.05:
        status_5g = "CONFIRMED"
    elif p_5g >= 0.05:
        status_5g = "NOT CONFIRMED"
    else:
        status_5g = "UNEXPECTED DIRECTION"
    print(f"     Q1\u2192Q4: {q_means[0]*100:.2f}% \u2192 {q_means[3]*100:.2f}%")
    print(f"     Status: {status_5g}")

    fs.close()


def main():
    parser = argparse.ArgumentParser(
        description='H5e/f/g: HLA on B cells and TME reshaping')
    parser.add_argument('--s-panel', required=True,
                        help='S-panel v8 h5ad (has HLA_DR, HLA_Class_I)')
    parser.add_argument('--t-panel', required=True,
                        help='T-panel v8 h5ad (has CD8 T exhausted)')
    parser.add_argument('--output-dir', default='output/hypotheses_v8')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_analysis(args)


if __name__ == '__main__':
    main()
