#!/usr/bin/env python3
"""New Fig 2: Compartments + follicular biology (compact).

Layout:
  Row 1: (a) composition heatmap (short) + (b) two ROI scatters stacked
  Row 2: (c) cell type composition gradient + (d) CD8 exhaustion by zone + (e) distance boxplot

The bottom 3x3 compartment examples from old Fig 2 move to Fig 2S (supplementary).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread as mpl_imread

# Import from sibling scripts
sys.path.insert(0, str(Path(__file__).parent))
from compartment_figures import (
    load_array, get_marker_names, get_tumor_mask, rename_labels,
    T_COMPARTMENT_COLORS, T_FOLL, T_INTER, T_EXCL,
    select_representative_rois, plot_composition_heatmap, plot_spatial_map,
    _find_best_roi_for_compartment,
)
from fig_follicular_architecture import (
    FOLLICULAR_ZONES, FOLLICLE_CENTER, B_TYPES, T_CD4, T_CD8, TREG, MACRO,
)
# Use 9-compartment order (excludes "Activated B / CXCR5hi zone" — only 1 ROI)
from immune_evasion import (
    GRADIENT_ORDER, GRADIENT_SHORT, GRADIENT_COLORS, FOLL_INTER_BOUNDARY,
)
from fig_tonsil_comparison import extract_t_panel, plot_tonsil_exhaustion, plot_tonsil_treg


# Standardized font sizes (direct-render, no PNG scaling)
PW = 10
TITLE_SIZE = 28
LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 18
ANNOT_SIZE = 20
PANEL_LABEL_SIZE = 22

PANEL_STYLE = {
    "font.size": TICK_SIZE, "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
    "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "axes.titlesize": TITLE_SIZE,
}


def _harmonize_ax(ax):
    """Force consistent font sizes on an axes after plotting."""
    ax.title.set_fontsize(TITLE_SIZE)
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    for t in ax.get_xticklabels():
        t.set_fontsize(TICK_SIZE)
        t.set_rotation(40)
        t.set_ha('right')
        t.set_rotation_mode('anchor')
    for t in ax.get_yticklabels():
        t.set_fontsize(TICK_SIZE)
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontsize(LEGEND_SIZE)
        if leg.get_title():
            leg.get_title().set_fontsize(LEGEND_SIZE)


def _render_panel(name, plot_fn, figsize, cache_dir, force=False, harmonize=True):
    """Render a panel to cache if not already cached."""
    path = cache_dir / f"fig2v2_{name}.png"
    if path.exists() and not force:
        print(f"  Using cached: {path.name}")
        return path
    plt.rcParams.update(PANEL_STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax)
    if harmonize:
        _harmonize_ax(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    fig.savefig(str(path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Rendered: {path.name}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-panel", required=True, help="T-panel v8 h5ad")
    parser.add_argument("--t-utag", required=True, help="T-panel UTAG h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "_cache_fig2v2"
    cache_dir.mkdir(exist_ok=True, parents=True)
    force = args.no_cache

    # ── Load data ──
    print("Loading T-panel...")
    f_v8 = h5py.File(args.t_panel, "r")
    f_utag = h5py.File(args.t_utag, "r")

    sid = load_array(f_utag, "sample_id")
    tma = load_array(f_utag, "tma")
    comps = rename_labels(load_array(f_utag, "compartment_name"))
    ctypes = rename_labels(load_array(f_v8, "cell_type"))
    cx = f_utag["obs"]["centroid_x"][:].astype(float)
    cy = f_utag["obs"]["centroid_y"][:].astype(float)

    # Load expression matrix and marker indices for marker expression panel
    markers = get_marker_names(f_utag)
    marker_idx = {}
    for m in ['TOX', 'PD_1', 'CD39']:
        if m in markers:
            marker_idx[m] = markers.index(m)
    X_all = f_utag["X"][:]

    tumor = get_tumor_mask(sid)

    # Filter to tumor
    sid_t = sid[tumor]; tma_t = tma[tumor]; comp_t = comps[tumor]
    ct_t = ctypes[tumor]; cx_t = cx[tumor]; cy_t = cy[tumor]
    X_t = X_all[tumor]

    f_v8.close()
    f_utag.close()

    print(f"  Tumor cells: {len(sid_t):,}")

    # ── Compartment ordering (show only the 9 analyzed compartments) ──
    t_order = [c for c in GRADIENT_ORDER if c in set(comp_t)]
    # Sidebar: red for follicular (first 5), blue for interfollicular (last 4)
    t_sidebar = (['#e74c3c'] * min(FOLL_INTER_BOUNDARY, len(t_order)) +
                 ['#3498db'] * max(0, len(t_order) - FOLL_INTER_BOUNDARY))

    # ── ROI selection ──
    roi_t1, roi_t2 = select_representative_rois(
        sid, tma, comps, tumor, T_FOLL, T_INTER)

    # ═══════════════════════════════════════════════════════════════
    # Panel (a): Composition heatmap (rendered live — fast)
    # ═══════════════════════════════════════════════════════════════
    def plot_a(ax):
        plot_composition_heatmap(ax, comp_t, ct_t, np.ones(len(comp_t), dtype=bool),
                                 t_order, t_sidebar, "T-panel")
        ax.set_title("Compartment composition", fontsize=TITLE_SIZE, fontweight="medium")
        for t in ax.get_yticklabels():
            t.set_fontsize(TICK_SIZE)
        for t in ax.get_xticklabels():
            t.set_fontsize(TICK_SIZE)
        # Colorbar font
        for cb_ax in ax.figure.axes:
            if cb_ax is not ax:
                cb_ax.tick_params(labelsize=TICK_SIZE)
                if cb_ax.get_ylabel():
                    cb_ax.yaxis.label.set_fontsize(LABEL_SIZE)
    path_a = _render_panel("a_heatmap", plot_a, (18, 10), cache_dir, force, harmonize=False)

    # ═══════════════════════════════════════════════════════════════
    # Panel (b): Single representative ROI scatter
    # ═══════════════════════════════════════════════════════════════
    best_roi = roi_t2  # C1_FL34 — good follicular + interfollicular balance

    main_comps = T_FOLL + T_INTER  # 9 main compartments only

    def plot_b(ax):
        m = sid == best_roi
        m_t = m & tumor
        plot_spatial_map(ax, cx[m_t], cy[m_t], comps[m_t],
                         T_COMPARTMENT_COLORS, best_roi,
                         tma[m_t][0] if np.any(m_t) else "?",
                         s=12, show_legend=True, legend_fontsize=LEGEND_SIZE,
                         legend_comps=main_comps)
        ax.set_title(f"{best_roi} ({tma[m_t][0]})", fontsize=TITLE_SIZE, fontweight="medium")

    path_b = _render_panel("b_roi", plot_b, (10, 14), cache_dir, force)

    # ═══════════════════════════════════════════════════════════════
    # Panel (c): Cell type composition gradient (from Fig 3c)
    # ═══════════════════════════════════════════════════════════════
    # Compute composition fractions
    group_types = {
        "B cells": B_TYPES, "CD4 T": T_CD4, "CD8 T": T_CD8,
        "Treg": TREG, "Macrophages": MACRO,
    }
    comp_fracs = {}
    for gname, types in group_types.items():
        vals = []
        for c in GRADIENT_ORDER:
            m = comp_t == c
            n = m.sum()
            if n == 0:
                vals.append(0)
            else:
                vals.append(float(np.isin(ct_t[m], types).sum()) / n)
        comp_fracs[gname] = np.array(vals)

    def plot_c(ax):
        x = np.arange(len(GRADIENT_ORDER))
        group_colors = {
            "B cells": "#E74C3C", "CD4 T": "#27AE60", "CD8 T": "#3498DB",
            "Treg": "#9B59B6", "Macrophages": "#E67E22",
        }
        bottoms = np.zeros(len(GRADIENT_ORDER))
        for gname in ["B cells", "CD4 T", "CD8 T", "Treg", "Macrophages"]:
            vals = comp_fracs[gname]
            ax.bar(x, vals, bottom=bottoms, color=group_colors[gname],
                   width=0.75, label=gname, edgecolor="white", linewidth=0.3)
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels(GRADIENT_SHORT, rotation=40, ha='right', rotation_mode='anchor')
        ax.set_ylabel("Fraction of cells")
        ax.set_ylim(0, 1.05)
        ax.set_title("Cell type composition", fontsize=TITLE_SIZE, fontweight="medium")
        ax.legend(fontsize=LEGEND_SIZE, loc="upper right", framealpha=0.8)
        ax.axvline(x=FOLL_INTER_BOUNDARY - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    path_c = _render_panel("c_composition", plot_c, (PW, 7), cache_dir, force)

    # ═══════════════════════════════════════════════════════════════
    # Panel (d): CD8 exhaustion fraction by compartment (from S7c)
    # ═══════════════════════════════════════════════════════════════
    CD8_ALL = ["CD8 T cells", "CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]
    CD8_EXHAUSTED = ["CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]

    exh_frac = []
    exh_n = []
    for c in GRADIENT_ORDER:
        m = comp_t == c
        cd8_m = m & np.isin(ct_t, CD8_ALL)
        n_cd8 = cd8_m.sum()
        if n_cd8 < 10:
            exh_frac.append(np.nan)
        else:
            n_exh = (m & np.isin(ct_t, CD8_EXHAUSTED)).sum()
            exh_frac.append(float(n_exh) / n_cd8)
        exh_n.append(n_cd8)
    exh_frac = np.array(exh_frac)
    exh_n = np.array(exh_n)

    def plot_d(ax):
        x = np.arange(len(GRADIENT_ORDER))
        bar_colors = GRADIENT_COLORS
        ax.bar(x, exh_frac * 100, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.axvline(FOLL_INTER_BOUNDARY - 0.5, color="#999", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(GRADIENT_SHORT, rotation=40, ha='right', rotation_mode='anchor')
        ax.set_ylabel("% exhausted\n(TOX+PD-1+ / all CD8 T)")
        ax.set_title("CD8 T cell exhaustion", fontsize=TITLE_SIZE, fontweight="medium")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Print summary
        for i, c in enumerate(GRADIENT_ORDER):
            if not np.isnan(exh_frac[i]):
                print(f"    {c}: {exh_frac[i]*100:.1f}% exhausted (n={exh_n[i]:,})")

    path_d = _render_panel("d_exhaustion", plot_d, (PW, 7), cache_dir, force)

    # ═══════════════════════════════════════════════════════════════
    # Panel (e): Marker expression on CD8 T cells (from S7 panel d)
    #   TOX, PD-1, CD39 mean expression across compartments
    # ═══════════════════════════════════════════════════════════════
    from immune_evasion import CD8_ALL as IE_CD8_ALL, marker_expression_by_compartment
    n_comp = len(GRADIENT_ORDER)

    marker_expr = marker_expression_by_compartment(
        comp_t, ct_t, X_t, marker_idx, GRADIENT_ORDER, IE_CD8_ALL,
        ['TOX', 'PD_1', 'CD39'])

    def plot_e_markers(ax):
        x = np.arange(n_comp)
        mk_colors = {'TOX': '#6A5ACD', 'PD_1': '#CD5C5C', 'CD39': '#2E8B57'}
        mk_labels = {'TOX': 'TOX', 'PD_1': 'PD-1', 'CD39': 'CD39'}
        for mname, color in mk_colors.items():
            if mname in marker_expr:
                ax.plot(x, marker_expr[mname], 'o-', color=color, ms=7, lw=2,
                        label=mk_labels[mname], alpha=0.85)
        ax.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(GRADIENT_SHORT, rotation=40, ha='right', rotation_mode='anchor')
        ax.set_ylabel('Mean expression (z-scored)')
        ax.set_title('Exhaustion markers on CD8 T cells', fontsize=TITLE_SIZE, fontweight='medium')
        ax.legend(fontsize=LEGEND_SIZE, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    path_e = _render_panel("e_markers", plot_e_markers, (PW, 7), cache_dir, force)

    # ═══════════════════════════════════════════════════════════════
    # Panel (f): FL vs tonsil exhaustion comparison (from S7 panel e)
    # ═══════════════════════════════════════════════════════════════
    tonsil_t_data = extract_t_panel(args.t_utag)

    def plot_f_tonsil(ax):
        plot_tonsil_exhaustion(ax, tonsil_t_data)
        ax.set_title('CD8 exhaustion: FL vs tonsil', fontsize=TITLE_SIZE, fontweight='medium')

    path_f = _render_panel("f_tonsil_exh", plot_f_tonsil, (PW, 7), cache_dir, force)

    # ═══════════════════════════════════════════════════════════════
    # Panel (h): FL vs tonsil Treg comparison (from S7 panel h)
    # ═══════════════════════════════════════════════════════════════
    def plot_h_tonsil_treg(ax):
        plot_tonsil_treg(ax, tonsil_t_data)
        ax.set_title('Treg distribution: FL vs tonsil', fontsize=TITLE_SIZE, fontweight='medium')

    path_h = _render_panel("h_tonsil_treg", plot_h_tonsil_treg, (PW, 7), cache_dir, force)

    # ═══════════════════════════════════════════════════════════════
    # Panel (g): Compartment spatial adjacency — size-independent demonstration
    #   of concentric ordering (replaces the size-confounded radial-distance panel).
    #   Each compartment's neighbors concentrate on its gradient-adjacent
    #   compartments, so the mean neighbor index rises monotonically with position.
    # ═══════════════════════════════════════════════════════════════
    print("Computing compartment adjacency (50 um neighborhoods)...")
    from scipy.spatial import cKDTree as _cKDTree
    from scipy.stats import spearmanr as _spearmanr

    _adj_order = [c for c in GRADIENT_ORDER if c in set(comp_t)]
    _n_adj = len(_adj_order)
    _adj_idx = {c: i for i, c in enumerate(_adj_order)}
    _counts = np.zeros((_n_adj, _n_adj), dtype=np.int64)
    for _roi in np.unique(sid_t):
        _m = sid_t == _roi
        _rc = comp_t[_m]
        _known = np.array([c in _adj_idx for c in _rc])
        if _known.sum() < 100:
            continue
        _coords = np.column_stack([cx_t[_m][_known], cy_t[_m][_known]])
        _labels = _rc[_known]
        _tree = _cKDTree(_coords)
        for _i, _j in _tree.query_pairs(r=50.0):
            _ci = _adj_idx[_labels[_i]]; _cj = _adj_idx[_labels[_j]]
            _counts[_ci, _cj] += 1; _counts[_cj, _ci] += 1
    _rs = _counts.sum(axis=1, keepdims=True); _rs[_rs == 0] = 1
    _adj_frac = _counts / _rs
    # Mean neighbor gradient-index per compartment; a concentric sequence => monotonic
    _mean_nbr = (_adj_frac * np.arange(_n_adj)[None, :]).sum(axis=1)
    _rho_adj, _p_adj = _spearmanr(np.arange(_n_adj), _mean_nbr)
    print(f"  Adjacency ordering: Spearman rho={_rho_adj:.3f}, P={_p_adj:.1e}")

    def plot_g_dist(ax):
        # Mask the (dominant, expected) diagonal so the off-diagonal band is visible
        disp = _adj_frac.copy()
        np.fill_diagonal(disp, np.nan)
        vmax = np.nanmax(disp)
        cmap = plt.cm.viridis.copy(); cmap.set_bad("#EEEEEE")
        im = ax.imshow(disp, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        labs = [GRADIENT_SHORT[GRADIENT_ORDER.index(c)].replace("\n", " ")
                for c in _adj_order]
        cols = [GRADIENT_COLORS[GRADIENT_ORDER.index(c)] for c in _adj_order]
        ax.set_xticks(range(_n_adj)); ax.set_yticks(range(_n_adj))
        ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=TICK_SIZE)
        ax.set_yticklabels(labs, fontsize=TICK_SIZE)
        for t, c in zip(ax.get_xticklabels(), cols): t.set_color(c)
        for t, c in zip(ax.get_yticklabels(), cols): t.set_color(c)
        ax.set_xlabel("Neighbor compartment", fontsize=LABEL_SIZE)
        ax.set_ylabel("Compartment", fontsize=LABEL_SIZE)
        ax.set_title("Compartment spatial adjacency (50 um)",
                     fontsize=TITLE_SIZE, fontweight="medium")
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Neighbor fraction", fontsize=LABEL_SIZE)
        cb.ax.tick_params(labelsize=TICK_SIZE)

    path_g_dist = _render_panel("g_adjacency", plot_g_dist, (PW, 7), cache_dir,
                                force, harmonize=False)

    # ═══════════════════════════════════════════════════════════════
    # Panel (g): Pairwise interaction enrichment per compartment
    #   Permutation-based z-scores (K=10, 200 permutations)
    # ═══════════════════════════════════════════════════════════════
    from scipy.spatial import cKDTree

    # Cell type groups for interaction analysis
    INTERACT_TYPES = {
        "B cells": ["GC B cells", "B cells (CD20hi)", "B cells (CXCR5hi)", "Other B cells",
                     "B cells (TOXhi)", "Activated B / Plasmablast", "B cells (weak CD20)"],
        "CD4 T": T_CD4,
        "Treg": TREG,
        "CD8 T": ["CD8 T cells"],
        "CD8 exh": CD8_EXHAUSTED,
        "GzmB+": ["Macrophages (GzmB+)"],
        "Mac": MACRO,
    }
    ITYPE_NAMES = list(INTERACT_TYPES.keys())
    n_itypes = len(ITYPE_NAMES)

    # Map each individual cell type → group index
    ct_to_group = {}
    for gi, (gname, types) in enumerate(INTERACT_TYPES.items()):
        for t in types:
            ct_to_group[t] = gi

    K_NEIGH = 10
    N_PERM = 200
    MIN_CELLS = 50

    # Compute per-compartment pairwise interaction z-scores
    print(f"Computing pairwise interaction enrichment (K={K_NEIGH}, {N_PERM} perms)...")
    rng = np.random.default_rng(42)
    rois = np.unique(sid_t)

    # Sample ROIs for speed
    np.random.seed(42)
    sample_rois = np.random.choice(rois, min(60, len(rois)), replace=False)

    # For each compartment: observed and null pairwise counts
    z_matrices = {}  # comp -> (n_itypes, n_itypes) z-score matrix
    obs_matrices = {}  # comp -> (n_itypes, n_itypes) observed count matrix

    for comp in GRADIENT_ORDER:
        # Collect ROI-level data for this compartment
        roi_data = []
        for roi in sample_rois:
            rc_mask = (sid_t == roi) & (comp_t == comp)
            n_cells = rc_mask.sum()
            if n_cells < MIN_CELLS:
                continue

            rc_idx = np.where(rc_mask)[0]
            roi_cx_local = cx_t[rc_idx]
            roi_cy_local = cy_t[rc_idx]
            coords = np.column_stack([roi_cx_local, roi_cy_local])

            # Code cell types to group indices (-1 for unmatched)
            labels = np.array([ct_to_group.get(ct_t[j], -1) for j in rc_idx])

            tree = cKDTree(coords)
            k_q = min(K_NEIGH + 1, n_cells)
            _, indices = tree.query(coords, k=k_q)
            neigh_idx = indices[:, 1:]  # (n_cells, K)

            roi_data.append((labels, neigh_idx))

        total_cells = sum(len(lab) for lab, _ in roi_data)
        if total_cells < MIN_CELLS:
            print(f"  {comp}: {total_cells} cells, skipping")
            continue

        # Count observed pairwise interactions
        def count_pairs(labels_list):
            """Count source_type -> target_type neighbor pairs across ROIs."""
            mat = np.zeros((n_itypes, n_itypes))
            for labels, neigh_idx in labels_list:
                for si in range(n_itypes):
                    src_mask = labels == si
                    if not src_mask.any():
                        continue
                    neigh_labels = labels[neigh_idx[src_mask].ravel()]
                    for ti in range(n_itypes):
                        mat[si, ti] += (neigh_labels == ti).sum()
            return mat

        obs_mat = count_pairs(roi_data)

        # Permutation null
        null_mats = np.zeros((N_PERM, n_itypes, n_itypes))
        for p in range(N_PERM):
            shuffled_data = [(rng.permutation(labels), neigh_idx)
                             for labels, neigh_idx in roi_data]
            null_mats[p] = count_pairs(shuffled_data)

        # Z-scores
        null_mean = null_mats.mean(axis=0)
        null_std = null_mats.std(axis=0)
        z_mat = np.where(null_std > 0, (obs_mat - null_mean) / null_std, 0)

        z_matrices[comp] = z_mat
        obs_matrices[comp] = obs_mat
        # Print top interactions
        top_pairs = []
        for si in range(n_itypes):
            for ti in range(si, n_itypes):  # upper triangle
                z = z_mat[si, ti]
                if abs(z) > 3:
                    top_pairs.append((ITYPE_NAMES[si], ITYPE_NAMES[ti], z))
        top_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_str = "; ".join(f"{a}-{b}: z={z:+.1f}" for a, b, z in top_pairs[:5])
        print(f"  {comp}: {top_str}")

    # ── Build summary: rows = compartments, columns = key interaction pairs ──
    # Generate ALL cross-type pairs (no self-interactions)
    ALL_PAIRS = []
    for si in range(n_itypes):
        for ti in range(si + 1, n_itypes):
            ALL_PAIRS.append((ITYPE_NAMES[si], ITYPE_NAMES[ti]))

    comps_with_data = [c for c in GRADIENT_ORDER if c in z_matrices]

    # Compute z-scores for all pairs across compartments
    all_pair_zscores = np.full((len(comps_with_data), len(ALL_PAIRS)), np.nan)
    for ci, comp in enumerate(comps_with_data):
        zmat = z_matrices[comp]
        for pi, (a, b) in enumerate(ALL_PAIRS):
            ai = ITYPE_NAMES.index(a)
            bi = ITYPE_NAMES.index(b)
            all_pair_zscores[ci, pi] = (zmat[ai, bi] + zmat[bi, ai]) / 2

    # Filter: keep only pairs with strong POSITIVE enrichment in ≥1 compartment
    Z_THRESHOLD = 5
    keep_pairs = []
    for pi in range(len(ALL_PAIRS)):
        col = all_pair_zscores[:, pi]
        if np.nanmax(col) >= Z_THRESHOLD:
            keep_pairs.append(pi)

    PAIR_LIST = [ALL_PAIRS[pi] for pi in keep_pairs]
    pair_labels = [f"{a} –\n{b}" for a, b in PAIR_LIST]
    print(f"  Keeping {len(PAIR_LIST)}/{len(ALL_PAIRS)} cross-type pairs (max |z| >= {Z_THRESHOLD})")

    # Also build observed count matrix for gating
    all_pair_obs = np.full((len(comps_with_data), len(ALL_PAIRS)), 0.0)
    for ci, comp in enumerate(comps_with_data):
        omat = obs_matrices[comp]
        for pi, (a, b) in enumerate(ALL_PAIRS):
            ai = ITYPE_NAMES.index(a)
            bi = ITYPE_NAMES.index(b)
            all_pair_obs[ci, pi] = omat[ai, bi] + omat[bi, ai]

    summary_mat = all_pair_zscores[:, keep_pairs]
    summary_obs = all_pair_obs[:, keep_pairs]

    # Mask z-scores where observed interaction count is too low
    MIN_OBS = 100  # need at least 100 observed interactions
    summary_mat = np.where(summary_obs >= MIN_OBS, summary_mat, np.nan)

    # Re-filter: drop pairs where no compartment has z >= Z_THRESHOLD after masking
    keep2 = [j for j in range(summary_mat.shape[1])
             if np.nanmax(summary_mat[:, j]) >= Z_THRESHOLD]
    summary_mat = summary_mat[:, keep2]
    summary_obs = summary_obs[:, keep2]
    PAIR_LIST = [PAIR_LIST[j] for j in keep2]
    print(f"  After count gating: {len(PAIR_LIST)} pairs remain")

    comp_short = [GRADIENT_SHORT[GRADIENT_ORDER.index(c)] for c in comps_with_data]
    comp_colors_f = [GRADIENT_COLORS[GRADIENT_ORDER.index(c)] for c in comps_with_data]

    # Transpose: compartments on x-axis, pairs on y-axis
    summary_mat_T = summary_mat.T  # (n_pairs, n_comps)
    pair_labels_y = [f"{a} – {b}" for a, b in PAIR_LIST]

    def plot_f(ax):
        vmax = 15
        im = ax.imshow(summary_mat_T, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(comps_with_data)))
        ax.set_xticklabels(comp_short, rotation=45, ha="right")
        ax.set_yticks(range(len(PAIR_LIST)))
        ax.set_yticklabels(pair_labels_y)
        ax.set_title("Pairwise interaction enrichment (z-score)",
                      fontsize=TITLE_SIZE, fontweight="medium")

        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(comp_colors_f[i])
            label.set_fontweight("bold")

        # Follicular/interfollicular boundary (vertical now)
        n_foll = sum(1 for c in comps_with_data if c in GRADIENT_ORDER[:6])
        ax.axvline(x=n_foll - 0.5, color="black", linewidth=1.5)

        # Annotate significant values
        for i in range(summary_mat_T.shape[0]):
            for j in range(summary_mat_T.shape[1]):
                val = summary_mat_T[i, j]
                if np.isnan(val):
                    continue
                if abs(val) < 2:
                    continue  # skip non-significant
                txt = f"{val:+.0f}"
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=ANNOT_SIZE - 2, color=color)

        plt.colorbar(im, ax=ax, shrink=0.7, label="z-score")

    path_g = _render_panel("g_interactions", plot_f, (14, 8), cache_dir, force)

    # ═══════════════════════════════════════════════════════════════
    # Panel (i): Treg vs CD8 effector fraction by compartment
    # ═══════════════════════════════════════════════════════════════
    CD8_EFFECTOR = ["CD8 T cells", "Macrophages (GzmB+)"]

    treg_frac_arr = np.zeros(n_comp)
    cd8_eff_frac_arr = np.zeros(n_comp)
    for ci, comp in enumerate(GRADIENT_ORDER):
        mask = comp_t == comp
        n = mask.sum()
        if n == 0:
            continue
        treg_frac_arr[ci] = float((mask & (ct_t == "Treg")).sum()) / n
        cd8_eff_frac_arr[ci] = float((mask & np.isin(ct_t, CD8_EFFECTOR)).sum()) / n

    def plot_g(ax):
        x = np.arange(n_comp)
        w2 = 0.35
        ax.bar(x - w2/2, treg_frac_arr * 100, w2, color="#2E8B57", label="Treg",
               alpha=0.85, edgecolor="white")
        ax.bar(x + w2/2, cd8_eff_frac_arr * 100, w2, color="#4169E1",
               label="CD8 effector", alpha=0.85, edgecolor="white")
        ax.axvline(FOLL_INTER_BOUNDARY - 0.5, color="#999", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(GRADIENT_SHORT, rotation=40, ha='right', rotation_mode='anchor')
        ax.set_ylabel("% of cells in compartment")
        ax.set_title("Treg vs CD8 effector fraction", fontsize=TITLE_SIZE, fontweight="medium")
        ax.legend(fontsize=LEGEND_SIZE, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    path_i = _render_panel("i_treg_cd8", plot_g, (PW, 7), cache_dir, force)
    # ═══════════════════════════════════════════════════════════════
    # Composite assembly
    # ═══════════════════════════════════════════════════════════════
    print("\n  Assembling Fig 2 composite...")
    # Panel mapping: a=heatmap, b=ROI, c=composition, d=exhaustion,
    # e=markers, f=tonsil exh, g=distance, h=Treg vs CD8,
    # i=tonsil Treg, j=interactions
    imgs = {k: mpl_imread(str(p)) for k, p in [
        ("a", path_a), ("b", path_b), ("c", path_c), ("d", path_d),
        ("e", path_e), ("f", path_f), ("g", path_g_dist), ("h", path_i),
        ("i", path_h), ("j", path_g),
    ]}

    fig = plt.figure(figsize=(20, 24))

    # Proportional row heights based on panel content heights
    # Row 1: heatmap (h=8) + ROI (h=7) → use max = 8
    # Row 2: 3 panels at h=7
    # Row 3: 3 panels at h=7
    # Row 4: Treg (h=7) + interactions (h=8) → use max = 8
    h1, h2, h3, h4 = 10, 7, 7, 8
    total_h = h1 + h2 + h3 + h4  # = 29
    usable = 0.92
    gap = 0.015

    frac1 = usable * h1 / total_h
    frac2 = usable * h2 / total_h
    frac3 = usable * h3 / total_h
    frac4 = usable * h4 / total_h

    top1 = 0.98
    bot1 = top1 - frac1
    top2 = bot1 - gap
    bot2 = top2 - frac2
    top3 = bot2 - gap
    bot3 = top3 - frac3
    top4 = bot3 - gap
    bot4 = top4 - frac4

    # Fixed left margin for all rows
    L = 0.02
    R = 0.98
    LABEL_FS = PANEL_LABEL_SIZE

    # Row 1: (a) heatmap + (b) ROI — match heights by padding panel b
    ha, wa = imgs["a"].shape[:2]
    hb, wb = imgs["b"].shape[:2]
    if hb < ha:
        # Pad panel b (top+bottom) with white to match panel a height
        pad_total = ha - hb
        pad_top = pad_total // 2
        pad_bot = pad_total - pad_top
        white = np.ones((pad_top, wb, imgs["b"].shape[2]), dtype=imgs["b"].dtype)
        white_bot = np.ones((pad_bot, wb, imgs["b"].shape[2]), dtype=imgs["b"].dtype)
        imgs["b"] = np.concatenate([white, imgs["b"], white_bot], axis=0)
        hb = ha
        wb = imgs["b"].shape[1]
    gs1 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[wa, wb],
                            left=L, right=R, top=top1, bottom=bot1,
                            wspace=0.03)
    ax_a = fig.add_subplot(gs1[0, 0])
    ax_a.imshow(imgs["a"]); ax_a.axis("off")
    ax_b = fig.add_subplot(gs1[0, 1])
    ax_b.imshow(imgs["b"]); ax_b.axis("off")

    # Row 2: (c) composition + (d) exhaustion + (e) markers
    wc, wd, we = imgs["c"].shape[1], imgs["d"].shape[1], imgs["e"].shape[1]
    gs2 = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[wc, wd, we],
                            left=L, right=R, top=top2, bottom=bot2,
                            wspace=0.03)
    axes2 = []
    for idx, key in enumerate(["c", "d", "e"]):
        ax = fig.add_subplot(gs2[0, idx])
        ax.imshow(imgs[key]); ax.axis("off")
        axes2.append(ax)

    # Row 3: (f) tonsil exh + (g) distance + (h) Treg vs CD8
    wf, wg, wh = imgs["f"].shape[1], imgs["g"].shape[1], imgs["h"].shape[1]
    gs3 = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[wf, wg, wh],
                            left=L, right=R, top=top3, bottom=bot3,
                            wspace=0.03)
    axes3 = []
    for idx, key in enumerate(["f", "g", "h"]):
        ax = fig.add_subplot(gs3[0, idx])
        ax.imshow(imgs[key]); ax.axis("off")
        axes3.append(ax)

    # Row 4: (i) tonsil Treg + (j) interactions
    wi, wj = imgs["i"].shape[1], imgs["j"].shape[1]
    gs4 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[wi, wj],
                            left=L, right=R, top=top4, bottom=bot4,
                            wspace=0.03)
    axes4 = []
    for idx, key in enumerate(["i", "j"]):
        ax = fig.add_subplot(gs4[0, idx])
        ax.imshow(imgs[key]); ax.axis("off")
        axes4.append(ax)

    # Place panel labels at fixed figure x-coordinates so they align vertically
    # Column 1: left edge = L
    # Column 2: use axes position from 3-col rows (middle panel)
    # Column 3: use axes position from 3-col rows (right panel)
    fig.canvas.draw()

    def get_ax_left(ax):
        return ax.get_position().x0

    def get_ax_top(ax):
        return ax.get_position().y1

    # Use row 2 (3-col) as reference for x positions
    x1 = get_ax_left(axes2[0])   # left column
    x2 = get_ax_left(axes2[1])   # middle column
    x3 = get_ax_left(axes2[2])   # right column

    # Row 1 labels: a at x1, b at its own position
    fig.text(x1, get_ax_top(ax_a) + 0.005, r"$\bf{a}$",
             fontsize=LABEL_FS, va="bottom", ha="left")
    fig.text(get_ax_left(ax_b), get_ax_top(ax_b) + 0.005, r"$\bf{b}$",
             fontsize=LABEL_FS, va="bottom", ha="left")

    # Row 2 labels
    for ax, label in zip(axes2, ["c", "d", "e"]):
        fig.text(get_ax_left(ax), get_ax_top(ax) + 0.005, rf"$\bf{{{label}}}$",
                 fontsize=LABEL_FS, va="bottom", ha="left")

    # Row 3 labels
    for ax, label in zip(axes3, ["f", "g", "h"]):
        fig.text(get_ax_left(ax), get_ax_top(ax) + 0.005, rf"$\bf{{{label}}}$",
                 fontsize=LABEL_FS, va="bottom", ha="left")

    # Row 4 labels
    for ax, label in zip(axes4, ["i", "j"]):
        fig.text(get_ax_left(ax), get_ax_top(ax) + 0.005, rf"$\bf{{{label}}}$",
                 fontsize=LABEL_FS, va="bottom", ha="left")

    out_path = output_dir / "fig2_compartments_v2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # ═══════════════════════════════════════════════════════════════
    # Fig 2S: Supplementary (3x3 compartment examples)
    # ═══════════════════════════════════════════════════════════════
    from compartment_figures import T_CELLTYPE_COLORS
    from matplotlib.patches import Patch

    old_cache = output_dir / "_cache_compartments"
    if old_cache.exists():
        print("\n  Assembling Fig 2S (compartment examples) from existing cache...")

        KEY_COMPS = [
            'GC core',
            'Follicle core (GC/CD20hi/CXCR5hi)',
            'Follicle mantle (CXCR5hi)',
            'B cell follicle (CD20hi/CXCR5hi)',
            'B cell zone',
            'T cell zone (CD4/CD8)',
            'Treg-enriched T zone',
            'Macrophage-rich zone',
            'Follicle-T zone interface',
        ]
        COMP_SHORT = {
            'GC core': 'GC core',
            'Follicle core (GC/CD20hi/CXCR5hi)': 'Follicle core',
            'Follicle mantle (CXCR5hi)': 'Follicle mantle',
            'B cell follicle (CD20hi/CXCR5hi)': 'B cell follicle',
            'B cell zone': 'B cell zone',
            'T cell zone (CD4/CD8)': 'T cell zone',
            'Treg-enriched T zone': 'Treg-enriched zone',
            'Macrophage-rich zone': 'Macrophage-rich zone',
            'Follicle-T zone interface': 'Follicle-T interface',
        }

        # Check all cached panels exist (naming matches compartment_figures.py)
        # For each target, re-compute the ROI that was cached (same logic as compartment_figures)
        example_data = []
        for target in KEY_COMPS:
            safe = target[:20].replace('/', '_')
            comp_path = old_cache / f"ex_{safe}_comp.png"
            ct_path = old_cache / f"ex_{safe}_ct.png"
            if comp_path.exists() and ct_path.exists():
                best_roi, _ = _find_best_roi_for_compartment(
                    sid, tma, comps, tumor, target, excl_list=T_EXCL)
                example_data.append((target, comp_path, ct_path, best_roi))
            else:
                print(f"  Warning: missing cache for {target}")
                example_data.append((target, None, None, None))

        fig_s = plt.figure(figsize=(22, 20))
        gs_bot = gridspec.GridSpec(3, 6, figure=fig_s,
                                    hspace=0.40, wspace=0.15,
                                    left=0.02, right=0.98, top=0.94, bottom=0.08)

        for idx, (target, comp_path, ct_path, roi) in enumerate(example_data):
            row = idx // 3
            col_pair = (idx % 3) * 2

            ax_comp = fig_s.add_subplot(gs_bot[row, col_pair])
            ax_ct = fig_s.add_subplot(gs_bot[row, col_pair + 1])

            if comp_path is None:
                ax_comp.set_visible(False)
                ax_ct.set_visible(False)
                continue

            ax_comp.imshow(mpl_imread(str(comp_path)))
            ax_comp.axis('off')
            short_name = COMP_SHORT.get(target, target)
            ax_comp.set_title(short_name, fontsize=20, fontweight='bold', pad=12)
            label = chr(ord('a') + idx)
            ax_comp.text(-0.10, 1.02, rf"$\bf{{{label}}}$",
                         transform=ax_comp.transAxes, fontsize=22, va='bottom', ha='left')

            ax_ct.imshow(mpl_imread(str(ct_path)))
            ax_ct.axis('off')
            # Clean up ROI label (truncate Biomax IDs, remove prefix, limit length)
            roi_label = (roi or '').replace('Biomax_', 'Biomax ')
            if len(roi_label) > 14:
                roi_label = roi_label[:14] + '…'
            roi_label = roi_label.replace('_', ' ')
            ax_ct.set_title(f'Cell types\n{roi_label}', fontsize=20, pad=12)

        # Shared cell-type legend
        ct_items = [Patch(facecolor=c, label=name)
                    for name, c in T_CELLTYPE_COLORS.items()
                    if name != 'Unassigned']
        ax_leg = fig_s.add_axes([0.02, 0.01, 0.94, 0.05])
        ax_leg.axis('off')
        ax_leg.legend(handles=ct_items, loc='center', ncol=6, fontsize=LEGEND_SIZE,
                       frameon=False, handletextpad=0.4, columnspacing=1.0,
                       title='Cell types',
                       title_fontproperties={'weight': 'bold', 'size': 14})

        supp_path = output_dir / "fig2_compartments_v2_supp.png"
        fig_s.savefig(supp_path, dpi=150, bbox_inches='tight', facecolor='white')
        fig_s.savefig(str(supp_path).replace(".png", ".pdf"), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_s)
        print(f"  Saved: {supp_path} + PDF")
    else:
        print(f"\n  Warning: {old_cache} not found — run compartment_figures.py first for Fig 2S")

    return out_path


if __name__ == "__main__":
    main()
