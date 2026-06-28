#!/usr/bin/env python3
"""Supplementary figure: PD-1^hi Tfh (true GC-Tfh) spatial niches.

PD-1^hi (>1.5) Tfh = true GC-Tfh. They concentrate at the GC core (70.6%
PD-1^hi) and form structured multi-cell hubs with Tregs (z=+22.3),
B cells (z=+10.6), and CD4 T cells (z=+11.3). Boundary CXCR5+ cells are
likely Tfr (FOXP3=1.5); T zone CXCR5+ cells are spillover artifacts.

6-panel figure (3 rows):
  (a) PD-1^hi Tfh density by compartment
  (b) Marker profile: PD-1^hi Tfh vs CD4 non-Tfh
  (c) PD-1^hi Tfh neighborhood enrichment (who they co-localize with)
  (d) Full ROI scatter showing hub cell types
  (e-f) Two zoom examples of GC-center Tfh-Treg-B-CD4 hubs
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from collections import Counter
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency, mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle, Circle, ConnectionPatch


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_array(f, key):
    ds = f["obs"][key]
    if isinstance(ds, h5py.Group) and "categories" in ds:
        cats = ds["categories"][:]
        codes = ds["codes"][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


def get_marker_names(f):
    key = "_index" if "_index" in f["var"] else "index"
    names = f["var"][key][:]
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def get_tumor_mask(sample_ids):
    control_tags = ["tonsil", "prostate", "kidney", "spleen", "adrenal",
                    "_ton_", "_adr_"]
    mask = np.array([not any(t in s.lower() for t in control_tags)
                     for s in sample_ids])
    mask &= np.array([s != "Biomax_ROI_006" for s in sample_ids])
    return mask


DISPLAY_RENAME = {
    "Low quality / Unassigned": "Unassigned",
    "B cells": "Other B cells",
    "LQ / B transitional": "B / Unassigned transitional",
    "Cytotoxic / LQ niche": "Cytotoxic niche",
    "Weak CD20 / LQ border": "Weak CD20 border",
}

def rename_labels(arr):
    return np.array([DISPLAY_RENAME.get(v, v) for v in arr])


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"$\\bf{{{letter}}}$", transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE, va="bottom", ha="left")


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

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

GRADIENT_COLORS = [
    '#B22222', '#DC143C', '#E8734A', '#E06060', '#DAA520',
    '#6495ED', '#20B2AA', '#4169E1', '#191970',
]

FOLL_INTER_BOUNDARY = 5  # interface is first interfollicular

# Standardized font sizes (matches paper convention for 20" composite width)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22

B_TYPES = ['GC B cells', 'B cells (CD20hi)', 'B cells (CXCR5hi)',
           'Other B cells', 'B cells (TOXhi)', 'Activated B / Plasmablast',
           'B cells (weak CD20)']

# Tfh gating
TFH_CXCR5_THRESH = 2.0

# Exhaustion gates
TOX_THRESH = 0.8
PD1_THRESH = 0.5
CD39_THRESH = 0.5

# PD-1 high threshold for GC-Tfh vs pre-Tfh stratification
# Matches scRNA-seq: GC-Tfh PD-1 = 1.4-2.0, resting = 1.1
PD1_HI_THRESH = 1.5

# TMA name extraction
def get_tma(sid):
    if sid.startswith("A1_"): return "A1"
    if sid.startswith("B1_"): return "B1"
    if sid.startswith("C1_"): return "C1"
    if "biomax" in sid.lower() or "Biomax" in sid: return "Biomax"
    return "Other"

# Phenotype markers for heatmap
PHENOTYPE_MARKERS = [
    'CXCR5', 'PD_1', 'ICOS', 'CD57', 'TOX', 'CD39',
    'CD3', 'CD4',
]

# ── Permutation enrichment constants ──
CONSOL_MAP = {
    'B cells (CD20hi)': 'B cells', 'B cells (CXCR5hi)': 'B cells',
    'Other B cells': 'B cells', 'B cells (TOXhi)': 'B cells',
    'Activated B / Plasmablast': 'B cells', 'B cells (weak CD20)': 'B cells',
    'B / Unassigned transitional': 'B cells', 'Weak CD20 border': 'B cells',
    'CD8 T cells': 'CD8 T', 'CD8 T effector': 'CD8 T',
    'Cytotoxic niche': 'CD8 T', 'Macrophages (GzmB+)': 'CD8 T',
    'CD8 T pre-exhausted (TOX+)': 'CD8 T exhausted',
    'M1 Macrophages': 'Macrophages', 'M2 Macrophages': 'Macrophages',
    'Mac(generic)': 'Macrophages',
}

PERM_TYPES = ['GC B cells', 'B cells', 'CD4 T cells', 'Treg',
              'CD8 T', 'CD8 T exhausted', 'Macrophages']
PERM_DISPLAY = ['GC B', 'B cells', 'CD4 T\n(non-Tfh)', 'Treg',
                'CD8 T', 'CD8 T\nexh.', 'Mac']
# Compartment indices with enough Tfh (skip B cell zone [4] and Mac zone [8])
PERM_COMP_IDX = [0, 1, 2, 3, 5, 6, 7]
MIN_TFH_PERM = 20


# ═══════════════════════════════════════════════════════════════════════════
# Permutation enrichment
# ═══════════════════════════════════════════════════════════════════════════

def compute_permutation_enrichment(sample_ids, comps, cx, cy, tfh,
                                    ctypes_consol, K=10, N_PERM=500):
    """Permutation-based neighborhood enrichment for Tfh cells.

    For each compartment, finds K nearest neighbors of Tfh cells (restricted
    to cells within the same compartment+ROI), then shuffles cell type labels
    N_PERM times to build a null distribution. Returns z-scores.
    """
    rng = np.random.default_rng(42)
    perm_comps = [GRADIENT_ORDER[i] for i in PERM_COMP_IDX]
    n_types = len(PERM_TYPES)
    n_comps = len(perm_comps)
    type_to_idx = {t: i for i, t in enumerate(PERM_TYPES)}

    z_scores = np.full((n_types, n_comps), np.nan)

    for ci, comp in enumerate(perm_comps):
        comp_mask = comps == comp
        rois = np.unique(sample_ids[comp_mask & tfh])

        # Collect per-ROI neighbor index data (build trees once)
        roi_data = []
        total_tfh = 0
        for roi in rois:
            rc_mask = (sample_ids == roi) & comp_mask
            n_cells = rc_mask.sum()
            if n_cells < K + 5:
                continue

            tfh_local = tfh[rc_mask]
            n_tfh_local = tfh_local.sum()
            if n_tfh_local < 1:
                continue

            rc_idx = np.where(rc_mask)[0]
            coords = np.column_stack([cx[rc_idx], cy[rc_idx]])
            labels_coded = np.array([type_to_idx.get(ctypes_consol[j], n_types)
                                     for j in rc_idx])

            tree = cKDTree(coords)
            tfh_pos = np.where(tfh_local)[0]
            k_q = min(K + 1, n_cells)
            _, indices = tree.query(coords[tfh_pos], k=k_q)
            if k_q <= 1:
                continue
            neigh_idx = indices[:, 1:].ravel()

            roi_data.append((labels_coded, neigh_idx))
            total_tfh += n_tfh_local

        if total_tfh < MIN_TFH_PERM:
            print(f"  {comp}: {total_tfh} Tfh, skipping (< {MIN_TFH_PERM})")
            continue

        # Observed neighbor type counts
        obs = np.zeros(n_types)
        total = 0
        for labels_coded, neigh_idx in roi_data:
            neigh_coded = labels_coded[neigh_idx]
            counts = np.bincount(neigh_coded, minlength=n_types + 1)[:n_types]
            obs += counts
            total += len(neigh_idx)

        # Null distribution (shuffle labels within each ROI × compartment)
        null = np.zeros((N_PERM, n_types))
        for p in range(N_PERM):
            for labels_coded, neigh_idx in roi_data:
                shuffled = rng.permutation(labels_coded)
                neigh_coded = shuffled[neigh_idx]
                counts = np.bincount(neigh_coded, minlength=n_types + 1)[:n_types]
                null[p] += counts

        # Z-scores
        obs_frac = obs / total
        null_frac = null / total
        null_mean = null_frac.mean(axis=0)
        null_std = null_frac.std(axis=0)

        for ti in range(n_types):
            if null_std[ti] > 1e-10:
                z_scores[ti, ci] = (obs_frac[ti] - null_mean[ti]) / null_std[ti]
            else:
                z_scores[ti, ci] = 0

        top_ti = np.nanargmax(z_scores[:, ci])
        print(f"  {comp}: {total_tfh} Tfh, {total} neighbors — "
              f"top: {PERM_TYPES[top_ti]} (z={z_scores[top_ti, ci]:+.1f})")

    return z_scores, perm_comps


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-panel", required=True, help="T-panel h5ad")
    parser.add_argument("--t-utag", required=True, help="T-panel UTAG h5ad")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("Loading T-panel data...")
    f = h5py.File(args.t_panel, "r")
    sample_ids = load_array(f, "sample_id")
    ctypes_raw = load_array(f, "cell_type")
    ctypes = rename_labels(ctypes_raw)
    tumor = get_tumor_mask(sample_ids)

    marker_names = get_marker_names(f)
    midx = {m: i for i, m in enumerate(marker_names)}

    import scipy.sparse as sp
    X_raw = f["X"]
    if isinstance(X_raw, h5py.Group):
        from scipy.sparse import csr_matrix
        X = csr_matrix((X_raw["data"][:], X_raw["indices"][:], X_raw["indptr"][:]),
                       shape=(len(sample_ids), len(marker_names)))
    else:
        X = X_raw[:]

    cx = np.array(f["obs"]["centroid_x"][:], dtype=float)
    cy = np.array(f["obs"]["centroid_y"][:], dtype=float)
    f.close()

    print("Loading UTAG compartments...")
    f_utag = h5py.File(args.t_utag, "r")
    comps = rename_labels(load_array(f_utag, "compartment_name"))
    f_utag.close()

    # ── Gate cells ──
    print("\nGating cells...")
    def get_col(marker):
        if marker not in midx:
            alts = {'CD8a': 'CD8', 'PD_1': 'PD-1'}
            marker = alts.get(marker, marker)
        idx = midx[marker]
        if sp.issparse(X):
            return np.array(X[:, idx].toarray()).ravel()
        return np.array(X[:, idx]).ravel()

    cd3 = get_col('CD3'); cd4 = get_col('CD4')
    cd8a_key = 'CD8a' if 'CD8a' in midx else 'CD8'
    cd8a = get_col(cd8a_key); cd20 = get_col('CD20')
    cxcr5 = get_col('CXCR5')
    pd1 = get_col('PD_1' if 'PD_1' in midx else 'PD-1')
    tox = get_col('TOX'); cd39 = get_col('CD39')
    cd57 = get_col('CD57'); icos = get_col('ICOS')

    foxp3 = get_col('FoxP3' if 'FoxP3' in midx else 'FOXP3')

    cd4_t = tumor & (cd3 > 0.5) & (cd4 > 0.5) & (cd8a < 0.5) & (cd20 < 0.5)
    tfh = cd4_t & (cxcr5 > TFH_CXCR5_THRESH)
    tfh_pd1hi = tfh & (pd1 > PD1_THRESH)
    tfh_pd1_hi_strict = tfh & (pd1 > PD1_HI_THRESH)  # GC-Tfh proxy
    tfh_pd1_lo = tfh & (pd1 <= PD1_HI_THRESH)  # pre-Tfh / Tfr / spillover
    cd4_nontfh = cd4_t & ~tfh
    cd8_exh = tumor & (ctypes == 'CD8 T exhausted')
    tregs = tumor & (ctypes == 'Treg')

    # Exhaustion on Tfh
    tfh_exh = tfh & (tox > TOX_THRESH) & (pd1 > PD1_THRESH)

    print(f"  Tumor cells: {tumor.sum():,}")
    print(f"  CD4 T cells: {cd4_t.sum():,}")
    print(f"  Tfh: {tfh.sum():,} ({tfh.sum()/tumor.sum()*100:.2f}%)")
    print(f"  Tfh TOX+PD-1+ (exhausted): {tfh_exh.sum():,} ({tfh_exh.sum()/tfh.sum()*100:.1f}% of Tfh)")

    # ── Compute per-compartment data ──
    print("\n=== Computing compartment data ===")
    foll_comps = set(GRADIENT_ORDER[:FOLL_INTER_BOUNDARY])
    total_tfh = tfh.sum()
    total_tumor = tumor.sum()
    overall_tfh_rate = total_tfh / total_tumor  # overall Tfh density
    comp_data = {}
    for c in GRADIENT_ORDER:
        mask_c = (comps == c) & tumor
        n_all = mask_c.sum()
        n_cd4 = (mask_c & cd4_t).sum()
        n_tfh = (mask_c & tfh).sum()
        n_tfh_pd1 = (mask_c & tfh_pd1hi).sum()
        n_tfh_exh = (mask_c & tfh_exh).sum()
        tfh_pct_all = n_tfh / n_all * 100 if n_all > 0 else 0
        tfh_enrich = (n_tfh / n_all) / overall_tfh_rate if n_all > 0 else 0
        tfh_pct_of_total = n_tfh / total_tfh * 100 if total_tfh > 0 else 0
        exh_pct = n_tfh_exh / n_tfh * 100 if n_tfh > 3 else np.nan
        mean_tox = float(tox[mask_c & tfh].mean()) if n_tfh > 3 else np.nan
        mean_cd39 = float(cd39[mask_c & tfh].mean()) if n_tfh > 3 else np.nan
        mean_pd1 = float(pd1[mask_c & tfh].mean()) if n_tfh > 3 else np.nan

        # Also compute exhaustion for non-Tfh CD4 for comparison
        n_cd4nt = (mask_c & cd4_nontfh).sum()
        n_cd4nt_exh = (mask_c & cd4_nontfh & (tox > TOX_THRESH) & (pd1 > PD1_THRESH)).sum()
        cd4nt_exh_pct = n_cd4nt_exh / n_cd4nt * 100 if n_cd4nt > 10 else np.nan

        # PD-1 stratification
        n_pd1hi = (mask_c & tfh_pd1_hi_strict).sum()
        n_pd1lo = (mask_c & tfh_pd1_lo).sum()
        pd1hi_pct = n_pd1hi / n_tfh * 100 if n_tfh > 3 else np.nan
        mean_foxp3 = float(foxp3[mask_c & tfh].mean()) if n_tfh > 3 else np.nan

        comp_data[c] = {
            'n_all': n_all, 'n_cd4': n_cd4, 'n_tfh': n_tfh,
            'n_tfh_pd1': n_tfh_pd1, 'n_tfh_exh': n_tfh_exh,
            'n_pd1hi': n_pd1hi, 'n_pd1lo': n_pd1lo,
            'pd1hi_pct': pd1hi_pct, 'mean_foxp3': mean_foxp3,
            'tfh_pct_all': tfh_pct_all, 'tfh_enrich': tfh_enrich,
            'tfh_pct_of_total': tfh_pct_of_total,
            'exh_pct': exh_pct,
            'mean_tox': mean_tox, 'mean_cd39': mean_cd39, 'mean_pd1': mean_pd1,
            'cd4nt_exh_pct': cd4nt_exh_pct,
        }
        print(f"  {c:<40} n={n_all:>8,}  Tfh={n_tfh:>4} "
              f"PD1hi={pd1hi_pct:>5.1f}%  FOXP3={mean_foxp3:>5.3f}  "
              f"exh={exh_pct:>5.1f}%  TOX={mean_tox:>6.3f}")

    # Boundary vs deep-follicular enrichment
    boundary_comps = ['Follicle-T zone interface', 'Treg-enriched T zone']
    deep_foll_comps = GRADIENT_ORDER[:FOLL_INTER_BOUNDARY]
    boundary_tfh = sum(comp_data[c]['n_tfh'] for c in boundary_comps)
    boundary_all = sum(comp_data[c]['n_all'] for c in boundary_comps)
    deep_tfh = sum(comp_data[c]['n_tfh'] for c in deep_foll_comps)
    deep_all = sum(comp_data[c]['n_all'] for c in deep_foll_comps)
    boundary_rate = boundary_tfh / boundary_all if boundary_all > 0 else 0
    deep_rate = deep_tfh / deep_all if deep_all > 0 else 0
    boundary_enrichment = boundary_rate / overall_tfh_rate if overall_tfh_rate > 0 else 0
    deep_enrichment = deep_rate / overall_tfh_rate if overall_tfh_rate > 0 else 0
    table = np.array([[boundary_tfh, boundary_all - boundary_tfh],
                      [deep_tfh, deep_all - deep_tfh]])
    chi2, pval, _, _ = chi2_contingency(table)
    print(f"\n  Boundary (interface+Treg): {boundary_enrichment:.1f}x, "
          f"{boundary_tfh}/{boundary_all:,} ({boundary_tfh/total_tfh*100:.0f}% of Tfh)")
    print(f"  Deep follicular:          {deep_enrichment:.1f}x, "
          f"{deep_tfh}/{deep_all:,} ({deep_tfh/total_tfh*100:.0f}% of Tfh)")
    print(f"  Boundary vs deep: p={pval:.2e}")

    # Exhaustion: foll vs inter
    tfh_foll_mask = tfh & np.isin(comps, list(foll_comps))
    tfh_inter_mask = tfh & np.isin(comps, GRADIENT_ORDER[FOLL_INTER_BOUNDARY:])
    stat_tox, p_tox = mannwhitneyu(tox[tfh_foll_mask], tox[tfh_inter_mask])
    stat_cd39, p_cd39 = mannwhitneyu(cd39[tfh_foll_mask], cd39[tfh_inter_mask])
    foll_exh_pct = tfh_exh[np.isin(comps, list(foll_comps))].sum() / tfh_foll_mask.sum() * 100
    inter_exh_pct = tfh_exh[np.isin(comps, GRADIENT_ORDER[FOLL_INTER_BOUNDARY:])].sum() / tfh_inter_mask.sum() * 100
    print(f"  Tfh exhaustion: foll={foll_exh_pct:.1f}% vs inter={inter_exh_pct:.1f}%")
    print(f"  TOX foll vs inter: p={p_tox:.2e}")
    print(f"  CD39 foll vs inter: p={p_cd39:.2e}")

    # Per-TMA Tfh distribution across compartments
    print("\n=== Per-TMA Tfh distribution ===")
    tma_arr = np.array([get_tma(s) for s in sample_ids])
    tma_names = ['A1', 'B1', 'C1']
    tma_comp_data = {}
    for t in tma_names:
        mask_t = tumor & (tma_arr == t)
        t_total_tfh = (mask_t & tfh).sum()
        tma_comp_data[t] = {}
        for c in GRADIENT_ORDER:
            mc = mask_t & (comps == c)
            n_tfh_c = (mc & tfh).sum()
            pct = n_tfh_c / t_total_tfh * 100 if t_total_tfh > 0 else 0
            tma_comp_data[t][c] = {'n_tfh': n_tfh_c, 'pct_of_tma_tfh': pct}
        # Print top compartments
        top = sorted(GRADIENT_ORDER, key=lambda c: -tma_comp_data[t][c]['n_tfh'])[:3]
        top_str = ', '.join(f"{c.split('(')[0].strip()} {tma_comp_data[t][c]['pct_of_tma_tfh']:.0f}%"
                           for c in top)
        print(f"  {t} ({t_total_tfh} Tfh): {top_str}")

    # Marker heatmap data
    print("\n=== Marker heatmap ===")
    avail_markers = []
    for m in PHENOTYPE_MARKERS:
        if m in midx:
            avail_markers.append(m)
        else:
            alts = {'PD_1': 'PD-1'}
            alt = alts.get(m, None)
            if alt and alt in midx:
                avail_markers.append(alt)

    groups = {
        'PD-1$^{hi}$ Tfh': tfh_pd1_hi_strict,
        'CD4 T (non-Tfh)': cd4_nontfh,
    }
    heatmap_data = np.zeros((len(groups), len(avail_markers)))
    group_names = list(groups.keys())
    group_counts = []
    for i, (gname, gmask) in enumerate(groups.items()):
        group_counts.append(gmask.sum())
        for j, m in enumerate(avail_markers):
            idx = midx[m]
            if sp.issparse(X):
                vals = np.array(X[gmask, idx].toarray()).ravel()
            else:
                vals = np.array(X[gmask, idx]).ravel()
            heatmap_data[i, j] = np.mean(vals) if len(vals) > 0 else 0

    # Top 3 zoom windows — score by actual window contents (all hub members present)
    print("\n=== Selecting top 3 zoom windows (GC-center Tfh-Treg-B-CD4 hubs) ===")
    gc_comps = ['GC core', 'Follicle core (GC/CD20hi/CXCR5hi)']
    b_set_sel = set(B_TYPES + ['B / Unassigned transitional', 'Weak CD20 border'])
    rois = np.unique(sample_ids[tumor])
    window_half = 60  # 120×120px zoom windows
    all_windows = []  # (roi, center_x, center_y, n_tfh_hi, n_treg, n_cd4, n_b, score)
    for r in rois:
        if get_tma(r) == "Biomax":
            continue
        rmask = (sample_ids == r) & tumor
        r_cx_sel = cx[rmask]; r_cy_sel = cy[rmask]
        r_ct_sel = ctypes[rmask]; r_comp_sel = comps[rmask]
        r_pd1hi_sel = tfh_pd1_hi_strict[rmask]
        r_tregs_sel = tregs[rmask]
        r_cd4_sel = cd4_nontfh[rmask]
        r_gc_sel = np.isin(r_comp_sel, gc_comps)
        r_b_sel = np.array([ct in b_set_sel for ct in r_ct_sel])
        # Iterate over each GC PD-1^hi Tfh as potential window center
        gc_hi_idx = np.where(r_pd1hi_sel & r_gc_sel)[0]
        for idx in gc_hi_idx:
            wcx, wcy = r_cx_sel[idx], r_cy_sel[idx]
            in_w = ((np.abs(r_cx_sel - wcx) <= window_half) &
                    (np.abs(r_cy_sel - wcy) <= window_half))
            n_hi = (in_w & r_pd1hi_sel).sum()
            n_tr = (in_w & r_tregs_sel).sum()
            n_cd4 = (in_w & r_cd4_sel).sum()
            n_b = (in_w & r_b_sel).sum()
            # Require all hub members present
            if n_hi >= 2 and n_tr >= 2 and n_cd4 >= 1:
                score = n_hi + 0.5 * n_tr + 0.3 * n_cd4 + 0.1 * n_b
                all_windows.append((r, wcx, wcy, n_hi, n_tr, n_cd4, n_b, score))

    all_windows.sort(key=lambda x: -x[7])
    # Pick top 3, ensuring different ROIs when possible
    top_windows = []
    used_rois = set()
    for w in all_windows:
        if len(top_windows) >= 3:
            break
        r = w[0]
        # Prefer different ROIs; accept same ROI only if not enough diversity
        if r not in used_rois or len(top_windows) >= len(used_rois):
            # Check window doesn't overlap with already selected
            overlap = False
            for tw in top_windows:
                if tw[0] == r and abs(tw[1] - w[1]) < window_half and abs(tw[2] - w[2]) < window_half:
                    overlap = True
                    break
            if not overlap:
                top_windows.append(w)
                used_rois.add(r)
    # Fallback: if < 3, relax ROI diversity
    if len(top_windows) < 3:
        for w in all_windows:
            if len(top_windows) >= 3:
                break
            overlap = any(tw[0] == w[0] and abs(tw[1] - w[1]) < window_half
                          and abs(tw[2] - w[2]) < window_half for tw in top_windows)
            if not overlap and w not in top_windows:
                top_windows.append(w)

    for w in top_windows:
        print(f"  {w[0]}: center=({w[1]:.0f},{w[2]:.0f}), "
              f"{w[3]} Tfh^hi, {w[4]} Treg, {w[5]} CD4 T, {w[6]} B cells, score={w[7]:.1f}")

    top_rois = [w[0] for w in top_windows]
    best_roi = top_rois[0] if top_rois else None

    # ── Permutation-based neighborhood enrichment (PD-1^hi vs PD-1^lo) ──
    ctypes_consol = np.array([CONSOL_MAP.get(ct, ct) for ct in ctypes])
    ctypes_consol[tfh] = 'Tfh'  # Mark Tfh separately from CD4 T

    print("\n=== Permutation enrichment: PD-1^hi Tfh (K=10, N=500) ===")
    z_hi, perm_comps = compute_permutation_enrichment(
        sample_ids, comps, cx, cy, tfh_pd1_hi_strict, ctypes_consol)

    # Print z-score table
    perm_short = [GRADIENT_SHORT[i].replace('\n', ' ') for i in PERM_COMP_IDX]
    print(f"\n  PD-1^hi Tfh neighbors:")
    print(f"  {'Cell type':<20}", end='')
    for ps in perm_short:
        print(f"  {ps:>10}", end='')
    print()
    for ti, ttype in enumerate(PERM_TYPES):
        print(f"  {ttype:<20}", end='')
        for ci in range(len(perm_comps)):
            z = z_hi[ti, ci]
            sig = '*' if not np.isnan(z) and abs(z) > 2 else ' '
            print(f"  {z:>+8.1f}{sig}" if not np.isnan(z) else f"  {'N/A':>9}", end='')
        print()

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE: 3 rows × 2 cols
    # (a) PD-1^hi Tfh density by compartment
    # (b) Marker heatmap: PD-1^hi Tfh vs CD4 non-Tfh
    # (c) PD-1^hi Tfh neighborhood enrichment
    # (d) Full ROI scatter
    # (e-f) 2 zoom hub examples with connection lines from (d)
    # ═══════════════════════════════════════════════════════════════════════
    # ── Compute TFR fractions per compartment ──
    # Use raw CXCR5 (not scaled) for consistency with earlier analysis
    # The scaled cxcr5 variable uses z-scores where 2.0 is more stringent
    # than raw 2.0. Load raw from the h5ad.
    import anndata as ad
    _a = ad.read_h5ad(args.t_panel, backed='r')
    if _a.raw is not None:
        _raw_vars = list(_a.raw.var.index)
        _cxcr5_idx = _raw_vars.index('CXCR5')
        import scipy.sparse as _sp
        _col = _a.raw.X[:, _cxcr5_idx]
        cxcr5_raw = np.array(_col.todense()).flatten() if _sp.issparse(_col) else np.array(_col).flatten()
    else:
        cxcr5_raw = cxcr5  # fallback to scaled
    _a.file.close()

    tfr_fracs = []
    for c in GRADIENT_ORDER:
        cmask = (comps == c) & tregs
        n_treg = cmask.sum()
        if n_treg < 30:
            tfr_fracs.append(0.0)
        else:
            tfr_fracs.append(float((cxcr5_raw[cmask] > TFH_CXCR5_THRESH).sum()) / n_treg)
    tfr_fracs = np.array(tfr_fracs)
    print(f"  TFR fractions (raw CXCR5>{TFH_CXCR5_THRESH}): {dict(zip(GRADIENT_ORDER, tfr_fracs.round(3)))}")

    print("\n=== Generating figure ===")
    with plt.rc_context({'font.size': TICK_SIZE, 'axes.titlesize': TITLE_SIZE,
                         'axes.labelsize': LABEL_SIZE, 'xtick.labelsize': TICK_SIZE,
                         'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE}):
      pass  # context applies below
    plt.rcParams.update({'font.size': TICK_SIZE, 'axes.titlesize': TITLE_SIZE,
                         'axes.labelsize': LABEL_SIZE, 'xtick.labelsize': TICK_SIZE,
                         'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE})
    fig = plt.figure(figsize=(20, 28))
    gs_outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.28,
                                 left=0.06, right=0.96, top=0.97, bottom=0.03,
                                 height_ratios=[0.7, 0.6, 1.2])
    gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0],
                                                wspace=0.30)
    gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[1:],
                                          hspace=0.35, wspace=0.30)

    n_comp = len(GRADIENT_ORDER)
    x_pos = np.arange(n_comp)

    # ── (a) TFR vs classical Treg composition ──
    ax_tfr = fig.add_subplot(gs_row0[0])
    w_tfr = 0.65
    ax_tfr.bar(x_pos, tfr_fracs, w_tfr, color='#e6550d', edgecolor='white',
               linewidth=0.5, label='TFR (FoxP3+ CXCR5>2)')
    ax_tfr.bar(x_pos, 1.0 - tfr_fracs, w_tfr, bottom=tfr_fracs, color='#3182bd',
               edgecolor='white', linewidth=0.5,
               label=u'Classical Treg (FoxP3+ CXCR5\u22642)')
    ax_tfr.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_tfr.set_xticks(x_pos)
    ax_tfr.set_xticklabels(GRADIENT_SHORT, fontsize=11, rotation=40, ha='right', rotation_mode='anchor')
    ax_tfr.set_ylabel('Fraction of Tregs', fontsize=LABEL_SIZE)
    ax_tfr.set_ylim(0, 1.08)
    ax_tfr.set_title('TFR enriched in follicular compartments', fontsize=TITLE_SIZE,
                     fontweight='medium')
    ax_tfr.legend(fontsize=LEGEND_SIZE, loc='center right', frameon=True,
                  edgecolor='#cccccc', fancybox=False)
    ax_tfr.spines['top'].set_visible(False)
    ax_tfr.spines['right'].set_visible(False)
    ax_tfr.tick_params(axis='y', labelsize=TICK_SIZE)
    ax_tfr.text(2, 1.04, 'Follicular', ha='center',
                fontsize=ANNOT_SIZE, color='#B22222', fontstyle='italic')
    ax_tfr.text(7, 1.04, 'Interfollicular', ha='center',
                fontsize=ANNOT_SIZE, color='#4169E1', fontstyle='italic')
    panel_label(ax_tfr, 'a', x=-0.08)

    # ── (b) PD-1^hi Tfh density by compartment (single bars) ──
    ax_a = fig.add_subplot(gs_row0[1])
    pd1hi_density = [comp_data[c]['n_pd1hi'] / comp_data[c]['n_all'] * 100
                     if comp_data[c]['n_all'] > 0 else 0 for c in GRADIENT_ORDER]

    ax_a.bar(x_pos, pd1hi_density, 0.6,
             color='#DAA520', edgecolor='white', zorder=3)

    ax_a.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)

    ymax_a = max(pd1hi_density) * 1.2
    ax_a.set_ylim(0, ymax_a)

    ax_a.text(2, ymax_a * 0.98, 'Follicular', ha='center',
              fontsize=ANNOT_SIZE, color='#B22222', fontstyle='italic')
    ax_a.text(7, ymax_a * 0.98, 'Interfollicular', ha='center',
              fontsize=ANNOT_SIZE, color='#4169E1', fontstyle='italic')

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(GRADIENT_SHORT, fontsize=11, rotation=40, ha='right', rotation_mode='anchor')
    ax_a.set_ylabel('PD-1$^{hi}$ Tfh (% of all cells)', fontsize=LABEL_SIZE)
    ax_a.set_title('PD-1$^{hi}$ Tfh concentrate at GC core', fontsize=TITLE_SIZE,
                   fontweight='medium')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.tick_params(axis='y', labelsize=11)
    panel_label(ax_a, 'b')

    # ── (c) Marker heatmap: PD-1^hi Tfh vs CD4 non-Tfh ──
    ax_b = fig.add_subplot(gs_row0[2])
    # Show raw values side by side as grouped horizontal bars
    marker_y = np.arange(len(avail_markers))
    bar_h = 0.35
    vals_tfh = heatmap_data[0, :]   # PD-1^hi Tfh
    vals_cd4 = heatmap_data[1, :]   # CD4 non-Tfh

    ax_b.barh(marker_y - bar_h / 2, vals_tfh, bar_h,
              color='#DAA520', edgecolor='white', label=f'PD-1$^{{hi}}$ Tfh (n={group_counts[0]:,})')
    ax_b.barh(marker_y + bar_h / 2, vals_cd4, bar_h,
              color='#7FB3D8', edgecolor='white', label=f'CD4 T non-Tfh (n={group_counts[1]:,})')

    ax_b.set_yticks(marker_y)
    ax_b.set_yticklabels(avail_markers, fontsize=TICK_SIZE)
    ax_b.set_xlabel('Mean expression (z-scored)', fontsize=LABEL_SIZE)
    ax_b.set_title('PD-1$^{hi}$ Tfh marker profile', fontsize=TITLE_SIZE, fontweight='medium')
    ax_b.legend(fontsize=LEGEND_SIZE, loc='lower right')
    ax_b.tick_params(axis='x', labelsize=11)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.invert_yaxis()
    panel_label(ax_b, 'c')

    # ── (d) PD-1^hi Tfh neighborhood enrichment ──
    ax_c = fig.add_subplot(gs[0, :])
    perm_comp_short = [GRADIENT_SHORT[i] for i in PERM_COMP_IDX]
    z_vmax = 15
    z_clip = np.clip(z_hi, -z_vmax, z_vmax)
    im_c = ax_c.imshow(z_clip, aspect='auto', cmap='RdBu_r',
                       vmin=-z_vmax, vmax=z_vmax)

    ax_c.set_xticks(np.arange(len(perm_comps)))
    ax_c.set_xticklabels(perm_comp_short, fontsize=TICK_SIZE)
    ax_c.set_yticks(np.arange(len(PERM_TYPES)))
    ax_c.set_yticklabels(PERM_DISPLAY, fontsize=TICK_SIZE)

    for i in range(z_hi.shape[0]):
        for j in range(z_hi.shape[1]):
            val = z_hi[i, j]
            if np.isnan(val):
                ax_c.text(j, i, 'N/A', ha='center', va='center',
                          fontsize=ANNOT_SIZE, color='gray')
                continue
            sig = '*' if abs(val) > 2 else ''
            color = 'white' if abs(val) > 6 else 'black'
            weight = 'bold' if abs(val) > 2 else 'normal'
            ax_c.text(j, i, f'{val:+.1f}{sig}', ha='center', va='center',
                      fontsize=ANNOT_SIZE, color=color, fontweight=weight)

    ax_c.axvline(3.5, color='black', ls='--', lw=1.5, alpha=0.7)
    # Follicular/Interfollicular above heatmap, title pushed up to avoid overlap
    ax_c.text(1.5, -0.7, 'Follicular', ha='center', fontsize=ANNOT_SIZE,
              color='#B22222', fontstyle='italic')
    ax_c.text(5.0, -0.7, 'Interfollicular', ha='center', fontsize=ANNOT_SIZE,
              color='#4169E1', fontstyle='italic')

    cbar_c = fig.colorbar(im_c, ax=ax_c, shrink=0.6, pad=0.02, extend='both')
    cbar_c.set_label('z-score', fontsize=LABEL_SIZE)
    cbar_c.ax.tick_params(labelsize=11)
    ax_c.set_title('PD-1$^{hi}$ Tfh neighborhood enrichment (K=10, 500 permutations)',
                   fontsize=TITLE_SIZE, fontweight='medium', pad=35)
    panel_label(ax_c, 'd', x=-0.04)

    # ── (d) Full ROI scatter  +  (e-f) 2 GC-center hub zoom panels ──
    ax_d = fig.add_subplot(gs[1, 0])
    # Nested 2×1 gridspec for 2 zoom panels in right column
    gs_zoom = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1, 1], hspace=0.30)

    b_set = set(B_TYPES + ['B / Unassigned transitional', 'Weak CD20 border'])
    zoom_axes = [fig.add_subplot(gs_zoom[i]) for i in range(2)]
    zoom_letters = ['f', 'g']
    zoom_colors = ['#CC0000', '#006600']

    # --- Panel (d): Full ROI (best_roi) ---
    if best_roi:
        roi_mask = (sample_ids == best_roi) & tumor
        roi_cx_c = cx[roi_mask]; roi_cy_c = cy[roi_mask]
        roi_ct_c = ctypes[roi_mask]; roi_comp_c = comps[roi_mask]
        roi_tfh_c = tfh[roi_mask]
        roi_pd1hi_c = tfh_pd1_hi_strict[roi_mask]
        roi_tregs_c = tregs[roi_mask]
        roi_cd4nt_c = cd4_nontfh[roi_mask]
        foll_mask_c = np.isin(roi_comp_c, list(foll_comps))

        # Follicular background tint
        if foll_mask_c.sum() > 10:
            ax_d.scatter(roi_cx_c[foll_mask_c], roi_cy_c[foll_mask_c],
                         s=2, c='#FFDDDD', alpha=0.15, zorder=0, rasterized=True)

        # Gray base (everything not highlighted)
        non_hl = ~roi_tfh_c & ~roi_tregs_c & ~roi_cd4nt_c
        b_mask_c = np.array([ct in b_set for ct in roi_ct_c]) & non_hl
        other_c = non_hl & ~b_mask_c
        ax_d.scatter(roi_cx_c[other_c], roi_cy_c[other_c],
                     s=3, c='#D3D3D3', alpha=0.3, zorder=1, rasterized=True)

        # B cells (subtle)
        ax_d.scatter(roi_cx_c[b_mask_c], roi_cy_c[b_mask_c],
                     s=4, c='#B0C4DE', alpha=0.4, zorder=2, rasterized=True)

        # CD4 T non-Tfh (green, subtle)
        ax_d.scatter(roi_cx_c[roi_cd4nt_c], roi_cy_c[roi_cd4nt_c],
                     s=5, c='#90EE90', alpha=0.4, zorder=2, rasterized=True)

        # Tregs (purple)
        ax_d.scatter(roi_cx_c[roi_tregs_c], roi_cy_c[roi_tregs_c],
                     s=20, c='#9370DB', alpha=0.7, zorder=3, rasterized=True)

        # PD-1^hi Tfh (gold stars)
        ax_d.scatter(roi_cx_c[roi_pd1hi_c], roi_cy_c[roi_pd1hi_c],
                     s=80, c='#DAA520', edgecolor='black', linewidth=0.8,
                     marker='*', zorder=5)

        ax_d.set_aspect('equal')
        ax_d.invert_yaxis()
        ax_d.set_title(f'{best_roi}', fontsize=TITLE_SIZE, fontweight='medium')
        ax_d.set_xlabel('x (px)', fontsize=LABEL_SIZE)
        ax_d.set_ylabel('y (px)', fontsize=LABEL_SIZE)
        ax_d.tick_params(labelsize=11)

        legend_d = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#DAA520',
                       markeredgecolor='black', ms=10, label='PD-1$^{hi}$ Tfh'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9370DB',
                       ms=6, label='Treg'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90',
                       ms=5, label='CD4 T'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#B0C4DE',
                       ms=5, label='B cells'),
        ]
        ax_d.legend(handles=legend_d, fontsize=LEGEND_SIZE, loc='lower right', framealpha=0.8)

    panel_label(ax_d, 'e')

    # Sort top 2 windows by y-coordinate so upper window → upper zoom panel
    # (smaller y = visually higher with inverted y-axis)
    top_windows_draw = sorted(top_windows[:2], key=lambda w: w[2])

    # --- Panels (e-f): 2 zoom examples from pre-computed windows ---
    for win_data, ax_z, zletter, rect_color in zip(
            top_windows_draw, zoom_axes, zoom_letters, zoom_colors):

        zoom_roi, zcx, zcy = win_data[0], win_data[1], win_data[2]
        zx0 = zcx - window_half; zy0 = zcy - window_half
        zx1 = zcx + window_half; zy1 = zcy + window_half

        rmask = (sample_ids == zoom_roi) & tumor
        r_cx = cx[rmask]; r_cy = cy[rmask]
        r_ct = ctypes[rmask]; r_comp = comps[rmask]
        r_tfh = tfh[rmask]
        r_pd1hi = tfh_pd1_hi_strict[rmask]
        r_tregs = tregs[rmask]
        r_cd4nt = cd4_nontfh[rmask]

        # Draw rectangle + connection lines on full ROI panel (d) if same ROI
        if zoom_roi == best_roi:
            for lw_val, ec, ls in [(2.5, 'white', '-'), (1.5, rect_color, '--')]:
                rect = Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                                  linewidth=lw_val, edgecolor=ec, facecolor='none',
                                  linestyle=ls, zorder=10)
                ax_d.add_patch(rect)
            # ConnectionPatch: rectangle right edge → zoom panel left edge
            # Upper line: top-right of rect → top-left of zoom
            con_top = ConnectionPatch(
                xyA=(zx1, zy0), coordsA=ax_d.transData,
                xyB=(0, 1), coordsB=ax_z.transAxes,
                color=rect_color, lw=1.0, ls='--', alpha=0.6, zorder=9)
            fig.add_artist(con_top)
            # Lower line: bottom-right of rect → bottom-left of zoom
            con_bot = ConnectionPatch(
                xyA=(zx1, zy1), coordsA=ax_d.transData,
                xyB=(0, 0), coordsB=ax_z.transAxes,
                color=rect_color, lw=1.0, ls='--', alpha=0.6, zorder=9)
            fig.add_artist(con_bot)

        # Clip to zoom window
        in_z = ((r_cx >= zx0) & (r_cx <= zx1) & (r_cy >= zy0) & (r_cy <= zy1))
        z_cx = r_cx[in_z]; z_cy = r_cy[in_z]
        z_ct = r_ct[in_z]; z_comp = r_comp[in_z]
        z_pd1hi = r_pd1hi[in_z]
        z_tregs = r_tregs[in_z]
        z_cd4nt = r_cd4nt[in_z]
        z_tfh = r_tfh[in_z]

        # Follicular domain tint
        z_foll = np.isin(z_comp, list(foll_comps))
        if z_foll.sum() > 3:
            ax_z.scatter(z_cx[z_foll], z_cy[z_foll],
                         s=40, c='#FFDDDD', alpha=0.3, zorder=0, rasterized=True)

        # B cells (light blue)
        z_b = np.array([ct in b_set for ct in z_ct]) & ~z_tfh & ~z_tregs & ~z_cd4nt
        ax_z.scatter(z_cx[z_b], z_cy[z_b],
                     s=40, c='#6495ED', alpha=0.6, zorder=1, rasterized=True)

        # CD4 T non-Tfh (green)
        ax_z.scatter(z_cx[z_cd4nt], z_cy[z_cd4nt],
                     s=50, c='#2E8B57', edgecolor='black', linewidth=0.3,
                     alpha=0.7, zorder=2)

        # Other cells (gray)
        z_other = ~z_b & ~z_tfh & ~z_tregs & ~z_cd4nt
        ax_z.scatter(z_cx[z_other], z_cy[z_other],
                     s=20, c='#D3D3D3', alpha=0.4, zorder=1, rasterized=True)

        # Tregs (purple)
        ax_z.scatter(z_cx[z_tregs], z_cy[z_tregs],
                     s=100, c='#9370DB', edgecolor='black', linewidth=0.5,
                     alpha=0.8, zorder=3)

        # PD-1^hi Tfh (gold stars)
        ax_z.scatter(z_cx[z_pd1hi], z_cy[z_pd1hi],
                     s=300, c='#DAA520', edgecolor='black', linewidth=1.0,
                     marker='*', zorder=5)

        # 30µm proximity circles
        for xi, yi in zip(z_cx[z_pd1hi], z_cy[z_pd1hi]):
            circ = Circle((xi, yi), 30, fill=False, edgecolor='#DAA520',
                          linewidth=1.2, linestyle=':', alpha=0.7, zorder=4)
            ax_z.add_patch(circ)

        ax_z.set_xlim(zx0, zx1)
        ax_z.set_ylim(zy1, zy0)  # inverted
        ax_z.set_aspect('equal')

        # Counts annotation
        n_hi = z_pd1hi.sum(); n_tr = z_tregs.sum()
        n_cd4 = z_cd4nt.sum(); n_b = z_b.sum()
        ax_z.set_title(f'{zoom_roi}', fontsize=TITLE_SIZE, fontweight='medium')
        ax_z.annotate(f'{n_hi} Tfh$^{{hi}}$  {n_tr} Treg\n'
                      f'{n_cd4} CD4 T  {n_b} B',
                      xy=(0.02, 0.97), xycoords='axes fraction',
                      fontsize=ANNOT_SIZE, ha='left', va='top',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                                edgecolor='gray', alpha=0.9))
        ax_z.set_xticks([]); ax_z.set_yticks([])
        panel_label(ax_z, zletter, x=-0.12)

    # Shared legend for zoom panels
    zoom_legend = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#DAA520',
                   markeredgecolor='black', ms=10, label='PD-1$^{hi}$ Tfh'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9370DB',
                   markeredgecolor='black', ms=7, label='Treg'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E8B57',
                   markeredgecolor='black', ms=6, label='CD4 T'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6495ED',
                   ms=6, label='B cells'),
        Patch(facecolor='#FFDDDD', edgecolor='none', alpha=0.5,
              label='Follicular'),
    ]
    zoom_axes[-1].legend(handles=zoom_legend, fontsize=LEGEND_SIZE, loc='lower right',
                         framealpha=0.8)

    # ── Save ──
    # Save both PNG (for assembly) and PDF (vectorized for publication)
    out_path = out_dir / "fig_tfh_enrichment.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    out_pdf = out_dir / "fig_tfh_enrichment.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
