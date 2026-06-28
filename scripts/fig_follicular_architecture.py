#!/usr/bin/env python3
"""Follicular sub-architecture analysis: intra-follicular zonation in FL.

FL follicles are classically described as "unpolarized" (loss of LZ/DZ).
Our UTAG domain analysis reveals a *different* form of organization:
concentric B cell phenotype zones from GC core → follicle edge.

This script validates that these zones are:
  1. Spatially organized (radial distance analysis)
  2. Compositionally distinct (cell type gradients)
  3. Contiguous (spatial adjacency)
  4. Reproducible (across ROIs and TMAs)

Output: 6-panel publication figure + stats CSV.
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from collections import Counter
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (from compartment_figures.py / immune_evasion.py)
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
    # Also exclude Biomax_ROI_006 (known problem)
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

# Full gradient from follicle center → interfollicular
GRADIENT_ORDER = [
    "GC core",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)",
    "Activated B / CXCR5hi zone",
    "B cell follicle (CD20hi/CXCR5hi)",
    "B cell zone",
    "Follicle-T zone interface",
    "Treg-enriched T zone",
    "T cell zone (CD4/CD8)",
    "Macrophage-rich zone",
]

GRADIENT_SHORT = [
    "GC\ncore", "Follicle\ncore", "Follicle\nmantle",
    "Activated\nB zone", "B cell\nfollicle", "B cell\nzone",
    "Foll-T\ninterface", "Treg\nT zone", "T cell\nzone", "Mac\nzone",
]

GRADIENT_COLORS = [
    "#8B0000", "#B22222", "#DC143C", "#E8734A", "#E06060", "#DAA520",
    "#6495ED", "#20B2AA", "#4169E1", "#191970",
]

# Follicular sub-zones (the novel finding)
FOLLICULAR_ZONES = GRADIENT_ORDER[:6]
FOLLICLE_CENTER = ["GC core", "Follicle core (GC/CD20hi/CXCR5hi)"]

# Cell type groups
B_TYPES = ["GC B cells", "B cells (CD20hi)", "B cells (CXCR5hi)",
           "Other B cells", "B cells (TOXhi)", "Activated B / Plasmablast",
           "B cells (weak CD20)"]
T_CD4 = ["CD4 T cells"]
T_CD8 = ["CD8 T cells", "CD8 T exhausted", "CD8 T pre-exhausted (TOX+)",
          "Macrophages (GzmB+)"]
TREG = ["Treg"]
MACRO = ["Macrophages"]

# Markers for gradient profiles
PROFILE_MARKERS = ["CD20", "CXCR5", "CD3", "CD8a", "GranzymeB", "TOX",
                   "FoxP3", "CD68"]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data(t_panel_path, t_utag_path):
    """Load T-panel v8 + UTAG merged data."""
    print(f"Loading T-panel: {t_panel_path}")
    f_v8 = h5py.File(t_panel_path, "r")
    print(f"Loading T-UTAG: {t_utag_path}")
    f_utag = h5py.File(t_utag_path, "r")

    sid = load_array(f_utag, "sample_id")
    tma = load_array(f_utag, "tma")
    comps = rename_labels(load_array(f_utag, "compartment_name"))
    ctypes = rename_labels(load_array(f_v8, "cell_type"))
    cx = f_utag["obs"]["centroid_x"][:].astype(float)
    cy = f_utag["obs"]["centroid_y"][:].astype(float)
    tumor = get_tumor_mask(sid)

    # Marker expression
    markers = get_marker_names(f_v8)
    marker_idx = {m: markers.index(m) for m in PROFILE_MARKERS if m in markers}
    X = f_v8["X"][:]

    f_v8.close()
    f_utag.close()

    print(f"  Total cells: {len(sid):,}, Tumor: {tumor.sum():,}")

    return {
        "sid": sid[tumor], "tma": tma[tumor],
        "comps": comps[tumor], "ctypes": ctypes[tumor],
        "cx": cx[tumor], "cy": cy[tumor],
        "X": X[tumor], "marker_idx": marker_idx,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_radial_distances(data):
    """For each individually detected follicle (DBSCAN), compute distance of
    surrounding cells to that follicle's centroid (nearest-follicle assignment).
    Returns per-cell distances and qualifying ROI list."""
    from sklearn.cluster import DBSCAN

    sid, comps, cx, cy = data["sid"], data["comps"], data["cx"], data["cy"]
    rois = np.unique(sid)

    all_distances = []  # (compartment, distance, roi, tma)
    qualifying_rois = []
    n_follicles_total = 0

    for roi in rois:
        roi_mask = sid == roi
        roi_comps = comps[roi_mask]
        roi_cx = cx[roi_mask]
        roi_cy = cy[roi_mask]
        roi_tma = data["tma"][roi_mask][0]

        # DBSCAN on GC core + Follicle core cells
        center_mask = np.isin(roi_comps, FOLLICLE_CENTER)
        n_center = center_mask.sum()
        if n_center < 50:
            continue

        coords = np.column_stack([roi_cx[center_mask], roi_cy[center_mask]])
        db = DBSCAN(eps=120, min_samples=20).fit(coords)
        labels = db.labels_

        # Collect valid follicle centroids in this ROI
        centroids = []
        for lbl in np.unique(labels):
            if lbl == -1:
                continue
            fl = labels == lbl
            if fl.sum() < 40:
                continue
            centroids.append((coords[fl, 0].mean(), coords[fl, 1].mean()))

        if not centroids:
            # Fallback: composite centroid
            centroids = [(roi_cx[center_mask].mean(), roi_cy[center_mask].mean())]

        n_follicles_total += len(centroids)
        qualifying_rois.append((roi, roi_tma, n_center))

        # Assign each cell to its nearest follicle centroid (Voronoi)
        cell_coords = np.column_stack([roi_cx, roi_cy])
        centroid_arr = np.array(centroids)
        # Distance matrix: cells × centroids
        diff = cell_coords[:, None, :] - centroid_arr[None, :, :]  # N×K×2
        dist_to_centroids = np.sqrt((diff ** 2).sum(axis=2))        # N×K
        nearest = dist_to_centroids.argmin(axis=1)                   # N

        # For each cell, record its distance to its assigned follicle centroid
        dist_to_nearest = dist_to_centroids[np.arange(len(roi_comps)), nearest]

        for i in range(len(roi_comps)):
            if roi_comps[i] in GRADIENT_ORDER:
                all_distances.append((roi_comps[i], dist_to_nearest[i], roi, roi_tma))

    print(f"\n  Qualifying ROIs (≥50 center cells): {len(qualifying_rois)}")
    print(f"  Individual follicles used as centroids: {n_follicles_total}")

    return all_distances, qualifying_rois


def compute_adjacency(data, max_dist=50.0):
    """Compute compartment adjacency: for cells in each compartment,
    what fraction of neighbors (within max_dist) are in each other compartment?"""
    sid, comps, cx, cy = data["sid"], data["comps"], data["cx"], data["cy"]
    rois = np.unique(sid)

    # Only count compartments in GRADIENT_ORDER
    comp_set = set(GRADIENT_ORDER)
    n_comps = len(GRADIENT_ORDER)
    comp_to_idx = {c: i for i, c in enumerate(GRADIENT_ORDER)}

    # Accumulate neighbor counts
    neighbor_counts = np.zeros((n_comps, n_comps), dtype=np.int64)

    n_rois_used = 0
    for roi in rois:
        roi_mask = sid == roi
        roi_comps = comps[roi_mask]
        roi_cx = cx[roi_mask]
        roi_cy = cy[roi_mask]

        # Filter to cells in known compartments
        known = np.array([c in comp_set for c in roi_comps])
        if known.sum() < 100:
            continue

        coords = np.column_stack([roi_cx[known], roi_cy[known]])
        comp_labels = roi_comps[known]

        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=max_dist)

        for i, j in pairs:
            ci = comp_to_idx.get(comp_labels[i])
            cj = comp_to_idx.get(comp_labels[j])
            if ci is not None and cj is not None:
                neighbor_counts[ci, cj] += 1
                neighbor_counts[cj, ci] += 1

        n_rois_used += 1

    print(f"  Adjacency computed across {n_rois_used} ROIs")

    # Normalize: row sums to 1 (fraction of neighbors)
    row_sums = neighbor_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_frac = neighbor_counts / row_sums

    return adj_frac


def compute_composition_gradient(data):
    """Cell type fractions per compartment in gradient order."""
    comps, ctypes = data["comps"], data["ctypes"]

    groups = {
        "B cells": B_TYPES,
        "CD4 T": T_CD4,
        "CD8 T": T_CD8,
        "Treg": TREG,
        "Macrophages": MACRO,
    }

    result = {}
    counts = []
    for c in GRADIENT_ORDER:
        mask = comps == c
        n = mask.sum()
        counts.append(n)
        ct = ctypes[mask]
        for gname, gtypes in groups.items():
            if gname not in result:
                result[gname] = []
            frac = np.isin(ct, gtypes).sum() / max(n, 1)
            result[gname].append(frac)

    for gname in result:
        result[gname] = np.array(result[gname])
    result["_counts"] = np.array(counts)
    return result


def compute_marker_gradient(data):
    """Median marker expression per compartment (z-scored across compartments)."""
    comps = data["comps"]
    X = data["X"]
    marker_idx = data["marker_idx"]

    mat = np.zeros((len(GRADIENT_ORDER), len(PROFILE_MARKERS)))
    for j, mk in enumerate(PROFILE_MARKERS):
        if mk not in marker_idx:
            continue
        midx = marker_idx[mk]
        for i, c in enumerate(GRADIENT_ORDER):
            mask = comps == c
            if mask.sum() < 10:
                mat[i, j] = np.nan
            else:
                mat[i, j] = float(np.median(X[mask, midx]))

    # Z-score per marker (column)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            mu, sd = col[valid].mean(), col[valid].std()
            if sd > 0:
                mat[valid, j] = (col[valid] - mu) / sd

    return mat


def compute_per_roi_zone_counts(data):
    """For each tumor ROI, count how many follicular sub-zones are detected."""
    sid, comps = data["sid"], data["comps"]
    rois = np.unique(sid)

    roi_stats = []
    for roi in rois:
        roi_mask = sid == roi
        roi_comps = comps[roi_mask]
        tma = data["tma"][roi_mask][0]

        foll_zones = set()
        for z in FOLLICULAR_ZONES:
            if (roi_comps == z).sum() >= 30:  # min 30 cells
                foll_zones.add(z)
        roi_stats.append({
            "roi": roi, "tma": tma,
            "n_foll_zones": len(foll_zones),
            "zones": foll_zones,
        })

    return roi_stats


def detect_individual_follicles(data):
    """Use DBSCAN on GC core + Follicle core cells to identify individual
    follicles within each ROI. Returns a list of dicts with per-follicle stats."""
    from sklearn.cluster import DBSCAN

    sid, comps, ctypes = data["sid"], data["comps"], data["ctypes"]
    cx, cy = data["cx"], data["cy"]

    B_TYPES = ["GC B cells", "B cells (CD20hi)", "B cells (CXCR5hi)",
               "Other B cells", "B cells (TOXhi)", "Activated B / Plasmablast",
               "B cells (weak CD20)"]
    T_TYPES = ["CD4 T cells", "CD8 T cells", "CD8 T exhausted",
               "CD8 T pre-exhausted (TOX+)", "Macrophages (GzmB+)", "Treg"]

    follicles = []
    for roi in np.unique(sid):
        m = sid == roi
        rc = comps[m]; rx = cx[m]; ry = cy[m]; rt = ctypes[m]
        tma_val = data["tma"][m][0]

        center_mask = np.isin(rc, FOLLICLE_CENTER)
        if center_mask.sum() < 50:
            continue

        # DBSCAN to separate individual follicle GC cores
        coords = np.column_stack([rx[center_mask], ry[center_mask]])
        db = DBSCAN(eps=120, min_samples=20).fit(coords)
        labels = db.labels_

        n_follicles_in_roi = (np.unique(labels[labels >= 0])).size

        for lbl in np.unique(labels):
            if lbl == -1:
                continue
            fl = labels == lbl
            n_gc = fl.sum()
            if n_gc < 40:
                continue
            fc_x = coords[fl, 0].mean()
            fc_y = coords[fl, 1].mean()

            # Follicle radius: std of GC cell coordinates
            r_gc = np.sqrt(np.var(coords[fl, 0]) + np.var(coords[fl, 1]))

            # Characterize concentric zones around this follicle centroid
            dists = np.sqrt((rx - fc_x)**2 + (ry - fc_y)**2)
            inner  = dists < 150
            outer  = (dists >= 400) & (dists < 700)
            n_inner = inner.sum()
            if n_inner < 30:
                continue

            treg_inner  = (rt[inner] == "Treg").sum() / n_inner
            b_inner     = np.isin(rt[inner], B_TYPES).sum() / n_inner
            t_outer_n   = np.isin(rt[outer], T_TYPES).sum()

            # Check gradient zones are present around this follicle
            roi_comps_all = rc
            n_foll_zones = sum(1 for z in FOLLICULAR_ZONES
                               if (roi_comps_all == z).sum() >= 30)
            has_tzone = (roi_comps_all == "T cell zone (CD4/CD8)").sum() >= 30
            has_interface = (roi_comps_all == "Follicle-T zone interface").sum() >= 20

            # Penalise ROIs where follicular cells dominate (>60% of all cells)
            n_total_roi = rc.shape[0]
            n_foll_cells = sum((roi_comps_all == z).sum() for z in FOLLICULAR_ZONES)
            foll_frac = n_foll_cells / n_total_roi if n_total_roi > 0 else 1.0
            domination_penalty = max(0.0, foll_frac - 0.6) * 2.0

            # Score: reward clean center, penalise Treg contamination,
            # reward T zone + gradient. Penalise very large follicles (r_gc > 200px),
            # follicle-dominated ROIs, and reward ROIs with multiple distinct follicles.
            size_penalty = max(0.0, (r_gc - 200) / 200) * 0.5
            multi_reward = min(n_follicles_in_roi - 1, 3) * 0.15
            score = (b_inner
                     - 5.0 * treg_inner
                     - size_penalty
                     - domination_penalty
                     + multi_reward
                     + (t_outer_n > 200) * 0.3
                     + has_tzone * 0.2
                     + has_interface * 0.1
                     + n_foll_zones * 0.05)

            follicles.append({
                "roi": roi, "tma": tma_val,
                "fc_x": fc_x, "fc_y": fc_y,
                "n_gc": n_gc, "r_gc": r_gc,
                "n_foll_in_roi": n_follicles_in_roi,
                "score": score,
                "treg_inner": treg_inner, "b_inner": b_inner,
                "t_outer": t_outer_n,
            })

    follicles.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  Individual follicles detected: {len(follicles)}")
    print(f"  {'Score':>6}  {'ROI':<15} {'nGC':>5}  {'r_gc':>6}  {'nFoll':>5}  "
          f"{'Treg%':>6}  {'B%':>6}  {'T_out':>6}")
    for f in follicles[:10]:
        print(f"  {f['score']:>6.3f}  {f['roi']:<15} {f['n_gc']:>5}  "
              f"{f['r_gc']:>6.0f}  {f['n_foll_in_roi']:>5}  "
              f"{f['treg_inner']*100:>5.1f}%  {f['b_inner']*100:>5.1f}%  "
              f"{f['t_outer']:>6}")

    return follicles


def select_best_roi(data, qualifying_rois, force_roi=None):
    """Select the best individual follicle using the follicle detector.
    If force_roi is given, pick the highest-scoring follicle within that ROI."""
    follicles = detect_individual_follicles(data)
    if not follicles:
        # Fallback: original zone-count heuristic
        sid, comps = data["sid"], data["comps"]
        best_score = -1
        best_roi = None
        for roi, tma, n_center in qualifying_rois:
            roi_mask = sid == roi
            roi_comps = comps[roi_mask]
            n_zones = sum(1 for z in FOLLICULAR_ZONES
                          if (roi_comps == z).sum() >= 30)
            has_tzone = (roi_comps == "T cell zone (CD4/CD8)").sum() >= 30
            has_interface = (roi_comps == "Follicle-T zone interface").sum() >= 20
            score = n_zones * 2 + has_tzone + has_interface + (n_center > 100)
            if score > best_score:
                best_score = score
                best_roi = (roi, tma)
        return best_roi[0], best_roi[1], None, None, []

    if force_roi is not None:
        candidates = [f for f in follicles if f["roi"] == force_roi]
        if candidates:
            best = candidates[0]  # already sorted by score
        else:
            best = follicles[0]
            print(f"  Warning: {force_roi} not found in detected follicles, using best overall")
    else:
        best = follicles[0]

    # Collect all follicle centroids in the selected ROI
    roi_follicles = [f for f in follicles if f["roi"] == best["roi"]]
    all_centroids = [(f["fc_x"], f["fc_y"]) for f in roi_follicles]

    print(f"\n  Selected ROI: {best['roi']} ({best['tma']}), "
          f"{len(all_centroids)} follicle(s) detected")
    for i, f in enumerate(roi_follicles):
        print(f"    [{i+1}] score={f['score']:.3f}, "
              f"Treg={f['treg_inner']*100:.1f}%, B={f['b_inner']*100:.1f}%")
    return best["roi"], best["tma"], best["fc_x"], best["fc_y"], all_centroids


# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════

def _render_panel(panel_id, plot_fn, figsize, cache_dir, force=False):
    """Render a single panel to PNG cache. Returns path."""
    path = Path(cache_dir) / f"{panel_id}.png"
    if path.exists() and not force:
        return path
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax)
    panel_label(ax, panel_id)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.close(fig)
    return path


def make_figure(data, all_distances, qualifying_rois, adj_frac,
                comp_fracs, marker_mat, roi_stats, output_dir,
                no_cache=False):
    """6-panel cached-panel composite figure."""
    from matplotlib.image import imread as mpl_imread

    cache_dir = Path(output_dir) / "_cache_follicular"
    cache_dir.mkdir(exist_ok=True)
    force = no_cache

    color_map = dict(zip(GRADIENT_ORDER, GRADIENT_COLORS))

    # ── Select best ROI (needed for panel a) ──
    best_roi, best_tma, fc_x, fc_y, all_centroids = select_best_roi(
        data, qualifying_rois)

    # ── Panel (a): Representative ROI spatial map ──
    def plot_a(ax):
        roi_mask = data["sid"] == best_roi
        roi_cx = data["cx"][roi_mask]
        roi_cy = data["cy"][roi_mask]
        roi_comps = data["comps"][roi_mask]

        other = ~np.isin(roi_comps, GRADIENT_ORDER)
        if other.any():
            ax.scatter(roi_cx[other], roi_cy[other], c="#E0E0E0", s=4,
                       alpha=0.3, edgecolors="none", rasterized=True)
        for comp in reversed(GRADIENT_ORDER):
            m = roi_comps == comp
            if m.any():
                ax.scatter(roi_cx[m], roi_cy[m], c=color_map[comp], s=6,
                           alpha=0.8, edgecolors="none", rasterized=True,
                           label=comp if m.sum() >= 30 else None)
        if all_centroids:
            xs = [c[0] for c in all_centroids]
            ys = [c[1] for c in all_centroids]
            ax.scatter(xs, ys, c="gold", s=350, marker="*",
                       edgecolors="black", linewidths=0.8, zorder=10,
                       label=f"Follicle centroid (n={len(all_centroids)})")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{best_roi} ({best_tma})", fontsize=16, fontweight="medium")
        ax.legend(fontsize=LEGEND_SIZE, loc="lower left", framealpha=0.8,
                  markerscale=1.5, handletextpad=0.3)
        if all_centroids:
            pad = 600
            xs_c = np.array([c[0] for c in all_centroids])
            ys_c = np.array([c[1] for c in all_centroids])
            ax.set_xlim(xs_c.min() - pad, xs_c.max() + pad)
            ax.set_ylim(ys_c.max() + pad, ys_c.min() - pad)

    path_a = _render_panel("a", plot_a, (14, 12), cache_dir, force)

    # ── Panel (b): Radial distance boxplot ──
    def plot_b(ax):
        comp_dists = {c: [] for c in GRADIENT_ORDER}
        for comp, dist, _roi, _tma in all_distances:
            comp_dists[comp].append(dist)
        plot_comps = [c for c in GRADIENT_ORDER if len(comp_dists[c]) >= 100]
        plot_data = [comp_dists[c] for c in plot_comps]
        plot_colors = [color_map[c] for c in plot_comps]
        plot_labels = [GRADIENT_SHORT[GRADIENT_ORDER.index(c)] for c in plot_comps]

        bp = ax.boxplot(plot_data, positions=range(len(plot_comps)),
                        patch_artist=True, showfliers=False, widths=0.65,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(len(plot_comps)))
        ax.set_xticklabels(plot_labels, fontsize=TICK_SIZE)
        ax.set_ylabel("Distance to follicle centroid (px)", fontsize=15)
        ax.set_title("Radial distance from follicle center", fontsize=TITLE_SIZE,
                     fontweight="medium")
        ax.tick_params(axis="y", labelsize=TICK_SIZE)

        medians = [np.median(d) for d in plot_data]
        print("\n  Median distance to follicle centroid:")
        for c, med, n in zip(plot_comps, medians, [len(d) for d in plot_data]):
            print(f"    {c}: {med:.1f} px (n={n:,})")

        boundary_idx = None
        for i, c in enumerate(plot_comps):
            if c == "Follicle-T zone interface":
                boundary_idx = i
                break
        if boundary_idx is not None:
            ax.axvline(x=boundary_idx - 0.5, color="gray", linestyle="--",
                       linewidth=1, alpha=0.7)
            ax.text(boundary_idx - 0.6, ax.get_ylim()[1] * 0.95,
                    "follicular | interfollicular",
                    fontsize=12, color="gray", ha="center", rotation=90,
                    va="top")

        from scipy.stats import spearmanr, linregress
        indices = list(range(len(plot_comps)))
        slope, intercept, _, _, _ = linregress(indices, medians)
        x_line = np.array([0, len(plot_comps) - 1])
        ax.plot(x_line, slope * x_line + intercept, color="black",
                linewidth=1.5, linestyle="--", alpha=0.8, zorder=5,
                label="Linear trend")
        rho, p = spearmanr(indices, medians)
        ax.text(0.95, 0.95, f"Spearman rho={rho:.3f}\nP={p:.1e}",
                transform=ax.transAxes, fontsize=13, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8))

    path_b = _render_panel("b", plot_b, (10, 8), cache_dir, force)

    # ── Panel (c): Composition gradient ──
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
        ax.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
        ax.set_ylabel("Fraction of cells", fontsize=LABEL_SIZE)
        ax.set_ylim(0, 1.05)
        ax.set_title("Cell type composition gradient", fontsize=TITLE_SIZE,
                     fontweight="medium")
        ax.legend(fontsize=LEGEND_SIZE, loc="upper right", framealpha=0.8)
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
        ax.axvline(x=5.5, color="gray", linestyle="--", linewidth=1,
                   alpha=0.7)

    path_c = _render_panel("c", plot_c, (10, 7), cache_dir, force)

    # ── Panel (d): Marker expression heatmap ──
    def plot_d(ax):
        im = ax.imshow(marker_mat, aspect="auto", cmap="RdBu_r",
                       vmin=-2, vmax=2)
        ax.set_yticks(range(len(GRADIENT_ORDER)))
        ax.set_yticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE)
        ax.set_xticks(range(len(PROFILE_MARKERS)))
        display_mk = [m.replace("_", "-") for m in PROFILE_MARKERS]
        ax.set_xticklabels(display_mk, fontsize=TICK_SIZE, rotation=45, ha="right")
        ax.set_title("Marker expression (z-score)", fontsize=16,
                     fontweight="medium")
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_color(GRADIENT_COLORS[i])
            label.set_fontweight("bold")
        ax.axhline(y=5.5, color="black", linewidth=1)
        plt.colorbar(im, ax=ax, shrink=0.7, label="z-score")

    path_d = _render_panel("d", plot_d, (10, 8), cache_dir, force)

    # ── Panel (e): Spatial adjacency heatmap ──
    def plot_e(ax):
        im2 = ax.imshow(adj_frac, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=0.5)
        ax.set_xticks(range(len(GRADIENT_ORDER)))
        ax.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(GRADIENT_ORDER)))
        ax.set_yticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE)
        ax.set_title("Spatial adjacency (neighbor fraction)", fontsize=16,
                     fontweight="medium")
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_color(GRADIENT_COLORS[i])
            label.set_fontweight("bold")
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(GRADIENT_COLORS[i])
            label.set_fontweight("bold")
        ax.axhline(y=5.5, color="black", linewidth=1)
        ax.axvline(x=5.5, color="black", linewidth=1)
        plt.colorbar(im2, ax=ax, shrink=0.7, label="Fraction")

    path_e = _render_panel("e", plot_e, (10, 8), cache_dir, force)

    # ── Panel (f): Zone diversity histogram ──
    def plot_f(ax):
        max_zones = max(r["n_foll_zones"] for r in roi_stats)
        bins = np.arange(0, max_zones + 2) - 0.5
        all_vals = [r["n_foll_zones"] for r in roi_stats]
        ax.hist(all_vals, bins=bins, alpha=0.7, color="#5B9BD5",
                edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Number of follicular sub-zones detected", fontsize=LABEL_SIZE)
        ax.set_ylabel("Number of ROIs", fontsize=LABEL_SIZE)
        ax.set_title("Zone diversity per ROI", fontsize=TITLE_SIZE,
                     fontweight="medium")
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.set_xticks(range(max_zones + 1))

    path_f = _render_panel("f", plot_f, (10, 7), cache_dir, force)

    # Print summary stats
    n_with_3plus = sum(1 for r in roi_stats if r["n_foll_zones"] >= 3)
    n_total = len(roi_stats)
    print(f"\n  ROIs with ≥3 follicular sub-zones: {n_with_3plus}/{n_total} "
          f"({100*n_with_3plus/n_total:.0f}%)")

    # ===== Composite assembly =====
    print("  Assembling composite from cached panels...")
    imgs = {k: mpl_imread(str(p)) for k, p in
            [("a", path_a), ("b", path_b), ("c", path_c),
             ("d", path_d), ("e", path_e), ("f", path_f)]}

    # Row 1: a (wider) + b — use width ratios from actual panel sizes
    ha, wa = imgs["a"].shape[:2]
    hb, wb = imgs["b"].shape[:2]
    hc, wc = imgs["c"].shape[:2]
    hd, wd = imgs["d"].shape[:2]
    he, we = imgs["e"].shape[:2]
    hf, wf = imgs["f"].shape[:2]

    # Proportional row heights
    PW = 10  # reference panel width for uniform font scaling
    h_row1 = max(ha, hb)
    h_row2 = max(hc, hd)
    h_row3 = max(he, hf)
    total_h = h_row1 + h_row2 + h_row3

    fig = plt.figure(figsize=(20, 24))

    usable = 0.92  # vertical space (top=0.96, bottom=0.04)
    gap = 0.03
    frac1 = usable * h_row1 / total_h - gap
    frac2 = usable * h_row2 / total_h - gap
    frac3 = usable * h_row3 / total_h - gap

    top1 = 0.96
    bot1 = top1 - frac1
    top2 = bot1 - gap
    bot2 = top2 - frac2
    top3 = bot2 - gap
    bot3 = top3 - frac3

    # Row 1: a (wider) + b
    wr1 = [wa, wb]
    gs1 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=wr1,
                            left=0.02, right=0.98, top=top1, bottom=bot1,
                            wspace=0.03)
    ax1a = fig.add_subplot(gs1[0, 0])
    ax1a.imshow(imgs["a"]); ax1a.axis("off")
    ax1b = fig.add_subplot(gs1[0, 1])
    ax1b.imshow(imgs["b"]); ax1b.axis("off")

    # Row 2: c + d (equal width)
    gs2 = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.02, right=0.98, top=top2, bottom=bot2,
                            wspace=0.03)
    ax2c = fig.add_subplot(gs2[0, 0])
    ax2c.imshow(imgs["c"]); ax2c.axis("off")
    ax2d = fig.add_subplot(gs2[0, 1])
    ax2d.imshow(imgs["d"]); ax2d.axis("off")

    # Row 3: e + f (equal width)
    gs3 = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.02, right=0.98, top=top3, bottom=bot3,
                            wspace=0.03)
    ax3e = fig.add_subplot(gs3[0, 0])
    ax3e.imshow(imgs["e"]); ax3e.axis("off")
    ax3f = fig.add_subplot(gs3[0, 1])
    ax3f.imshow(imgs["f"]); ax3f.axis("off")

    fig.suptitle("Follicular Sub-Architecture in FL: Concentric Zonation",
                 fontsize=22, fontweight="bold", y=0.99)

    out_path = Path(output_dir) / "fig_follicular_architecture.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved: {out_path}")
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Gallery figure: 12 ROI examples
# ═══════════════════════════════════════════════════════════════════════════

def select_gallery_rois(data, qualifying_rois, n_per_tma=3, total=12):
    """Select top ROIs for gallery, prioritizing clear concentric zonation.

    Key improvement: scores each ROI by how well its zones are concentrically
    arranged (Spearman rho of zone index vs median radial distance), not just
    by how many zones are present.
    """
    from scipy.stats import spearmanr

    sid, comps, cx, cy = data["sid"], data["comps"], data["cx"], data["cy"]
    comp_to_idx = {c: i for i, c in enumerate(GRADIENT_ORDER)}

    scored = []
    for roi, tma, n_center in qualifying_rois:
        roi_mask = sid == roi
        roi_comps = comps[roi_mask]
        roi_cx = cx[roi_mask]
        roi_cy = cy[roi_mask]
        n_total = roi_mask.sum()

        # Follicle centroid
        center_mask = np.isin(roi_comps, FOLLICLE_CENTER)
        fc_x = roi_cx[center_mask].mean()
        fc_y = roi_cy[center_mask].mean()
        dists = np.sqrt((roi_cx - fc_x)**2 + (roi_cy - fc_y)**2)

        # Count follicular sub-zones with ≥30 cells + compute median distance
        zone_indices = []
        zone_med_dists = []
        for z in GRADIENT_ORDER:
            nz = (roi_comps == z).sum()
            if nz >= 30:
                zone_indices.append(comp_to_idx[z])
                zone_med_dists.append(float(np.median(dists[roi_comps == z])))

        n_zones_grad = len(zone_indices)
        n_foll_zones = sum(1 for zi in zone_indices if zi < 6)

        # Concentricity: Spearman of zone gradient index vs median distance
        if n_zones_grad >= 4:
            rho, _ = spearmanr(zone_indices, zone_med_dists)
        else:
            rho = 0.0

        # Want interface and T zone too for full gradient illustration
        n_tzone = (roi_comps == "T cell zone (CD4/CD8)").sum()
        n_interface = (roi_comps == "Follicle-T zone interface").sum()
        n_mantle = sum((roi_comps == z).sum() for z in [
            "Follicle mantle (CXCR5hi)", "B cell follicle (CD20hi/CXCR5hi)"])
        has_tzone = n_tzone >= 30
        has_interface = n_interface >= 20

        # Visual diversity: penalize ROIs where any single zone > 60%
        zone_counts = []
        for z in GRADIENT_ORDER:
            nz = (roi_comps == z).sum()
            if nz >= 30:
                zone_counts.append(nz)
        total_in_zones = sum(zone_counts) if zone_counts else 1
        max_frac = max(zone_counts) / total_in_zones if zone_counts else 1
        diversity_penalty = max(0, max_frac - 0.5) * 8  # penalty if >50% one zone

        # Color contrast: need visible T zone (blue) around follicular (red)
        frac_tzone_ish = (n_tzone + n_interface) / max(n_total, 1)
        frac_mantle = n_mantle / max(n_total, 1)
        color_contrast = min(frac_tzone_ish, 0.3) * 10 + min(frac_mantle, 0.3) * 5

        # Spatial compactness of center: prefer single dominant follicle
        center_std = np.sqrt(
            np.std(roi_cx[center_mask])**2 + np.std(roi_cy[center_mask])**2)
        roi_radius = np.sqrt(np.std(roi_cx)**2 + np.std(roi_cy)**2)
        compactness = max(0, 1 - center_std / max(roi_radius, 1)) * 3

        # Composite: concentricity + visual contrast + diversity
        score = (rho * 8                # concentricity (max ~8)
                 + n_foll_zones * 1.0   # follicular zone diversity
                 + has_tzone * 2        # need T zone for context
                 + has_interface        # interface is nice
                 + color_contrast       # visible blue/cyan rings (max ~4.5)
                 + compactness          # single follicle center (max 3)
                 - diversity_penalty    # penalize one-zone ROIs
                 + (n_center > 200)     # decent follicle size
                 + (n_total > 8000))    # decent ROI size

        scored.append({
            "roi": roi, "tma": tma, "n_center": n_center,
            "n_zones": n_foll_zones, "n_grad_zones": n_zones_grad,
            "rho": rho, "score": score, "n_total": n_total,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Pick top n_per_tma from each TMA
    selected = []
    tma_counts = {}
    for s in scored:
        t = s["tma"]
        if tma_counts.get(t, 0) >= n_per_tma:
            continue
        selected.append(s)
        tma_counts[t] = tma_counts.get(t, 0) + 1
        if len(selected) >= total:
            break

    # Fill remaining with best overall
    if len(selected) < total:
        used = {s["roi"] for s in selected}
        for s in scored:
            if s["roi"] not in used:
                selected.append(s)
                if len(selected) >= total:
                    break

    # Sort by TMA then score for nice layout
    selected.sort(key=lambda x: (x["tma"], -x["score"]))

    print(f"\n  Gallery: {len(selected)} ROIs selected")
    for s in selected:
        print(f"    {s['roi']} ({s['tma']}): {s['n_zones']} foll zones, "
              f"rho={s['rho']:.2f}, score={s['score']:.1f}, n={s['n_total']:,}")

    return selected


def find_follicle_cutouts(data, qualifying_rois, crop_radius=300, total=12,
                          n_per_tma=4, min_foll_zones=2):
    """Find individual follicle centers and score them for the gallery.

    For each qualifying ROI, identify the follicle centroid, crop to a window,
    and score by: concentricity within the window, zone diversity, and
    visual color contrast. Returns the best individual follicle cutouts.

    min_foll_zones: minimum number of follicular sub-zones (out of 6) required.
    """
    from scipy.stats import spearmanr

    sid, comps, cx, cy = data["sid"], data["comps"], data["cx"], data["cy"]
    comp_to_idx = {c: i for i, c in enumerate(GRADIENT_ORDER)}

    cutouts = []
    for roi, tma, n_center in qualifying_rois:
        roi_mask = sid == roi
        roi_comps = comps[roi_mask]
        roi_cx = cx[roi_mask]
        roi_cy = cy[roi_mask]

        # Follicle centroid
        center_mask = np.isin(roi_comps, FOLLICLE_CENTER)
        fc_x = float(roi_cx[center_mask].mean())
        fc_y = float(roi_cy[center_mask].mean())

        # Crop: cells within crop_radius of follicle centroid
        dists = np.sqrt((roi_cx - fc_x)**2 + (roi_cy - fc_y)**2)
        crop = dists <= crop_radius

        crop_comps = roi_comps[crop]
        crop_dists = dists[crop]
        n_crop = crop.sum()
        if n_crop < 200:
            continue

        # Count zones in the crop
        zone_indices = []
        zone_med_dists = []
        zone_names = []
        for z in GRADIENT_ORDER:
            nz = (crop_comps == z).sum()
            if nz >= 20:
                zone_indices.append(comp_to_idx[z])
                zone_med_dists.append(float(np.median(crop_dists[crop_comps == z])))
                zone_names.append(z)

        n_zones = len(zone_indices)
        n_foll = sum(1 for zi in zone_indices if zi < 6)

        # Concentricity within the crop
        if n_zones >= 4:
            rho, _ = spearmanr(zone_indices, zone_med_dists)
        elif n_zones >= 3:
            rho, _ = spearmanr(zone_indices, zone_med_dists)
            rho *= 0.8  # slight penalty for fewer points
        else:
            rho = 0.0

        # Visual contrast: need both inner (follicular) and outer (T/interface)
        n_inner = sum((crop_comps == z).sum() for z in FOLLICLE_CENTER)
        n_outer = sum((crop_comps == z).sum() for z in [
            "T cell zone (CD4/CD8)", "Follicle-T zone interface",
            "Treg-enriched T zone"])
        n_middle = sum((crop_comps == z).sum() for z in [
            "Follicle mantle (CXCR5hi)", "B cell follicle (CD20hi/CXCR5hi)",
            "B cell zone"])
        frac_inner = n_inner / max(n_crop, 1)
        frac_outer = n_outer / max(n_crop, 1)
        frac_middle = n_middle / max(n_crop, 1)

        # Best cutouts have all three rings visible
        ring_balance = (min(frac_inner, 0.3) + min(frac_outer, 0.3)
                        + min(frac_middle, 0.2))

        # Skip cutouts with too few follicular zones
        if n_foll < min_foll_zones:
            continue

        score = (rho * 8
                 + n_foll * 1.0
                 + ring_balance * 12
                 + (n_zones >= 5) * 2
                 + (n_crop > 1000) * 1)

        cutouts.append({
            "roi": roi, "tma": tma,
            "fc_x": fc_x, "fc_y": fc_y, "crop_radius": crop_radius,
            "n_zones": n_foll, "n_grad_zones": n_zones,
            "rho": rho, "score": score, "n_crop": n_crop,
            "zones": zone_names,
        })

    cutouts.sort(key=lambda x: x["score"], reverse=True)

    # Pick top per TMA
    selected = []
    tma_counts = {}
    for c in cutouts:
        t = c["tma"]
        if tma_counts.get(t, 0) >= n_per_tma:
            continue
        selected.append(c)
        tma_counts[t] = tma_counts.get(t, 0) + 1
        if len(selected) >= total:
            break

    # Fill remaining
    if len(selected) < total:
        used = {s["roi"] for s in selected}
        for c in cutouts:
            if c["roi"] not in used:
                selected.append(c)
                if len(selected) >= total:
                    break

    selected.sort(key=lambda x: (x["tma"], -x["score"]))

    print(f"\n  Follicle cutouts: {len(selected)} selected")
    for s in selected:
        print(f"    {s['roi']} ({s['tma']}): {s['n_zones']} foll zones, "
              f"{s['n_grad_zones']} total, rho={s['rho']:.2f}, "
              f"score={s['score']:.1f}, n={s['n_crop']:,}")

    return selected


def select_best_gallery_rois(data, qualifying_rois, total=6, must_include=None):
    """Select the best whole-core ROIs for the gallery.

    Scores by: number of visible follicular sub-zones, clear follicle-to-T zone
    gradient, large enough ROI. Picks best regardless of TMA.
    must_include: list of ROI names to always include (if they qualify).
    """
    from scipy.stats import spearmanr

    sid, comps, cx, cy = data["sid"], data["comps"], data["cx"], data["cy"]
    comp_to_idx = {c: i for i, c in enumerate(GRADIENT_ORDER)}

    scored = []
    for roi, tma, n_center in qualifying_rois:
        roi_mask = sid == roi
        roi_comps = comps[roi_mask]
        roi_cx = cx[roi_mask]
        roi_cy = cy[roi_mask]
        n_total = roi_mask.sum()

        if n_total < 5000:
            continue
        # Skip known artifact ROIs
        if roi in ("C1_FL36",):
            continue

        # Follicle centroid
        center_mask = np.isin(roi_comps, FOLLICLE_CENTER)
        fc_x = roi_cx[center_mask].mean()
        fc_y = roi_cy[center_mask].mean()
        dists = np.sqrt((roi_cx - fc_x)**2 + (roi_cy - fc_y)**2)

        # Count zones with ≥30 cells + median distance
        zone_indices = []
        zone_med_dists = []
        zone_names = []
        for z in GRADIENT_ORDER:
            nz = (roi_comps == z).sum()
            if nz >= 30:
                zone_indices.append(comp_to_idx[z])
                zone_med_dists.append(float(np.median(dists[roi_comps == z])))
                zone_names.append(z)

        n_foll = sum(1 for zi in zone_indices if zi < 6)
        n_total_zones = len(zone_indices)

        if n_foll < 3:
            continue

        # Concentricity
        if n_total_zones >= 4:
            rho, _ = spearmanr(zone_indices, zone_med_dists)
        else:
            rho = 0.0

        # Follicular vs interfollicular balance (need contrast to see round follicles)
        foll_mask = np.isin(roi_comps, FOLLICULAR_ZONES)
        n_foll_cells = foll_mask.sum()
        n_inter_cells = sum((roi_comps == z).sum() for z in GRADIENT_ORDER[6:])
        n_known = n_foll_cells + n_inter_cells
        if n_known > 0:
            balance = min(n_foll_cells, n_inter_cells) / n_known
        else:
            balance = 0

        # Skip cores with <15% interfollicular — no visual contrast
        # (unless in must_include list)
        must_set = set(must_include) if must_include else set()
        if balance < 0.15 and roi not in must_set:
            continue

        # Roundness: measure how compact (circular) the GC core cluster is
        # iso_ratio = ratio of eigenvalues of center cell positions (1.0 = round)
        circularity = 0.0
        if n_center >= 100:
            cx_c = roi_cx[center_mask] - fc_x
            cy_c = roi_cy[center_mask] - fc_y
            cov = np.cov(cx_c, cy_c)
            evals = np.linalg.eigvalsh(cov)
            if evals[1] > 0:
                circularity = evals[0] / evals[1]  # 1.0 = round, 0 = elongated

        # Score: visible round follicles with interfollicular contrast
        score = (circularity * 8                  # round follicle center
                 + n_foll * 2                     # more sub-zones
                 + min(n_total / 5000, 3) * 2    # bigger ROI
                 + balance * 10                   # red-vs-blue contrast
                 + (n_total_zones >= 6) * 2
                 + max(rho, 0) * 3)

        scored.append({
            "roi": roi, "tma": tma, "n_center": n_center,
            "n_foll_zones": n_foll, "n_total_zones": n_total_zones,
            "rho": rho, "circ": circularity, "score": score,
            "n_total": n_total, "zones": zone_names,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Prioritize must-include ROIs, then fill with top-scored
    if must_include:
        must_set = set(must_include)
        forced = [s for s in scored if s["roi"] in must_set]
        rest = [s for s in scored if s["roi"] not in must_set]
        selected = forced + rest[:max(0, total - len(forced))]
    else:
        selected = scored[:total]

    print(f"\n  Gallery: {len(selected)} ROIs selected (best whole cores)")
    for s in selected:
        print(f"    {s['roi']} ({s['tma']}): {s['n_foll_zones']} foll zones, "
              f"{s['n_total_zones']} total, circ={s['circ']:.2f}, rho={s['rho']:.2f}, "
              f"score={s['score']:.1f}, n={s['n_total']:,}")

    return selected


def make_gallery_figure(data, gallery_rois, output_dir):
    """Whole-core gallery: 2 rows × 3 cols, each panel a full TMA core
    colored by compartment with the standard red→blue palette.
    """
    # Same palette as compartment_figures.py
    COMP_COLORS = {
        'GC core': '#8B0000',
        'Follicle core (GC/CD20hi/CXCR5hi)': '#B22222',
        'Follicle mantle (CXCR5hi)': '#DC143C',
        'Activated B / CXCR5hi zone': '#E8734A',
        'B cell follicle (CD20hi/CXCR5hi)': '#E06060',
        'B cell zone': '#DAA520',
        'Follicle-T zone interface': '#6495ED',
        'Treg-enriched T zone': '#20B2AA',
        'T cell zone (CD4/CD8)': '#4169E1',
        'Macrophage-rich zone': '#191970',
    }

    n = len(gallery_rois)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    fig.suptitle("Follicular Sub-Architecture: Representative Cores",
                 fontsize=16, fontweight="bold", y=0.98)

    if n_rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    letters = "abcdefghijklmnop"

    for idx, info in enumerate(gallery_rois):
        ax = axes_flat[idx]
        roi_name = info["roi"]
        tma = info["tma"]

        roi_mask = data["sid"] == roi_name
        roi_cx = data["cx"][roi_mask]
        roi_cy = data["cy"][roi_mask]
        roi_comps = data["comps"][roi_mask]

        # Unassigned / other cells first (gray)
        other = ~np.isin(roi_comps, GRADIENT_ORDER)
        if other.any():
            ax.scatter(roi_cx[other], roi_cy[other], c="#D3D3D3", s=6,
                       alpha=0.3, edgecolors="none", rasterized=True)

        # Gradient compartments: outer zones first, center on top
        for comp in reversed(GRADIENT_ORDER):
            m = roi_comps == comp
            if m.any():
                ax.scatter(roi_cx[m], roi_cy[m],
                           c=COMP_COLORS.get(comp, "#808080"), s=6,
                           alpha=0.85, edgecolors="none", rasterized=True)

        # Draw boundary contours for follicular domains
        from scipy.ndimage import gaussian_filter
        xmin, xmax = roi_cx.min(), roi_cx.max()
        ymin, ymax = roi_cy.min(), roi_cy.max()
        grid_res = 200  # grid bins
        xbins = np.linspace(xmin, xmax, grid_res)
        ybins = np.linspace(ymin, ymax, grid_res)

        # Boundary between follicular and interfollicular
        foll_mask_r = np.isin(roi_comps, FOLLICULAR_ZONES)
        if foll_mask_r.sum() >= 50:
            H, _, _ = np.histogram2d(roi_cx[foll_mask_r], roi_cy[foll_mask_r],
                                     bins=[xbins, ybins])
            H_smooth = gaussian_filter(H.T, sigma=3)
            threshold = H_smooth.max() * 0.15
            ax.contour(xbins[:-1], ybins[:-1], H_smooth,
                       levels=[threshold], colors="black",
                       linewidths=1.5, linestyles="-")

        # Inner boundary: GC core
        gc_mask = np.isin(roi_comps, FOLLICLE_CENTER)
        if gc_mask.sum() >= 50:
            H2, _, _ = np.histogram2d(roi_cx[gc_mask], roi_cy[gc_mask],
                                      bins=[xbins, ybins])
            H2_smooth = gaussian_filter(H2.T, sigma=3)
            threshold2 = H2_smooth.max() * 0.15
            ax.contour(xbins[:-1], ybins[:-1], H2_smooth,
                       levels=[threshold2], colors="#4B0000",
                       linewidths=1.0, linestyles="--")

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        # Truncate long Biomax names
        display = roi_name
        if "Biomax" in roi_name and len(roi_name) > 20:
            # Extract the tissue identifier (e.g., "A3_Lym_3")
            parts = roi_name.replace(".txt", "").split("_")
            display = "Biomax_" + "_".join(parts[-3:])
        ax.set_title(f"{display} ({tma})", fontsize=12, fontweight="bold")
        panel_label(ax, letters[idx], x=-0.02, y=1.02)

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for comp in GRADIENT_ORDER:
        short = comp.split("(")[0].strip()
        if len(short) > 25:
            short = short[:22] + "..."
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COMP_COLORS.get(comp, "#808080"),
                   markersize=8, label=short))
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=LEGEND_SIZE, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_path = Path(output_dir) / "fig_follicular_architecture_gallery.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved gallery: {out_path}")
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Follicular sub-architecture analysis")
    parser.add_argument("--t-panel", required=True, help="T-panel v8 h5ad")
    parser.add_argument("--t-utag", required=True, help="T-panel UTAG merged h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-render all panels")
    args = parser.parse_args()

    # Load
    data = load_data(args.t_panel, args.t_utag)

    # Analysis
    print("\n=== Radial distance analysis ===")
    all_distances, qualifying_rois = compute_radial_distances(data)

    print("\n=== Spatial adjacency ===")
    adj_frac = compute_adjacency(data, max_dist=50.0)

    print("\n=== Composition gradient ===")
    comp_fracs = compute_composition_gradient(data)

    print("\n=== Marker gradient ===")
    marker_mat = compute_marker_gradient(data)

    print("\n=== Per-ROI zone diversity ===")
    roi_stats = compute_per_roi_zone_counts(data)

    # Figure
    print("\n=== Generating figure ===")
    out_path = make_figure(data, all_distances, qualifying_rois, adj_frac,
                           comp_fracs, marker_mat, roi_stats, args.output_dir,
                           no_cache=args.no_cache)

    # Print adjacency diagonal check
    print("\n=== Adjacency diagonal structure ===")
    for i, c in enumerate(GRADIENT_ORDER):
        top3_idx = np.argsort(adj_frac[i])[::-1][:3]
        top3 = [(GRADIENT_ORDER[j], f"{adj_frac[i,j]:.2f}") for j in top3_idx]
        print(f"  {c[:30]:30s} → {top3}")

    # Gallery figure: whole cores, best 6
    # Include known-good examples + fill with top-scored
    print("\n=== Generating gallery figure (whole cores) ===")
    gallery_rois = select_best_gallery_rois(data, qualifying_rois, total=6,
                                            must_include=["A1_FL35", "A1_FL40",
                                                          "C1_FL34", "C1_FL41"])
    make_gallery_figure(data, gallery_rois, args.output_dir)


if __name__ == "__main__":
    main()
