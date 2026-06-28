#!/usr/bin/env python3
"""CD14+ FDC localization and interactions in follicular lymphoma.

Panels:
  (a) Compartment distribution: CD14-high vs CD14-low FDC fraction per S-panel compartment
  (b) Spatial scatter: representative ROI — FDCs by CD14, macrophages, compartment tint
  (c) FDC→nearest macrophage distance by CD14 level
  (d) Macrophage subtype neighbor fractions for CD14-high vs CD14-low FDCs
  (e) Cross-panel: per-ROI FDC CD14 (S) vs CD8 exhaustion fraction (T)
  (f) Macrophage dependency: marker differences persist in mac-low ROIs?

Usage:
    .venv/bin/python scripts/fig_fdc_localization.py \
        --s-utag output/all_TMA_S_utag_ct_merged.h5ad \
        --t-panel output/all_TMA_T_global_v8.h5ad \
        --output-dir output/hypotheses_v8
"""

import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_array(f, key):
    ds = f["obs"][key]
    if isinstance(ds, h5py.Group) and "categories" in ds:
        cats = ds["categories"][:]
        codes = ds["codes"][:]
        cats_str = np.array(
            [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
        )
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


def is_tumor_core(sample_id):
    s = sample_id.lower()
    if "_ton_" in s or "_adr_" in s:
        return False
    for tissue in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if tissue in s:
            return False
    if sample_id == "Biomax_ROI_006":
        return False
    return True


def panel_label(ax, letter, x=-0.08, y=1.05):
    ax.text(
        x, y, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top", ha="left",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S_COMP_ORDER = [
    "B cell zone (BCL2+)", "B cell zone (PAX5+)",
    "FDC network zone", "FDC / myeloid zone",
    "T cell zone", "Stromal / CAF zone",
]
S_COMP_SHORT = [
    "B zone\n(BCL2+)", "B zone\n(PAX5+)", "FDC\nnetwork",
    "FDC/\nmyeloid", "T cell\nzone", "Stromal\n/ CAF",
]
S_COMP_COLORS = [
    "#B22222", "#DC143C", "#E8734A", "#FF8C00",
    "#4169E1", "#20B2AA",
]

MAC_SUBTYPES = [
    "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Myeloid (S100A9+)", "Dendritic cells",
]
MAC_SHORT = {
    "M1 Macrophages": "M1 Mac", "M2 Macrophages": "M2 Mac",
    "Macrophages": "Mac", "Myeloid (S100A9+)": "S100A9+",
    "Dendritic cells": "DC",
}
MAC_COLORS = {
    "M1 Mac": "#E41A1C", "M2 Mac": "#984EA3", "Mac": "#FF7F00",
    "S100A9+": "#A65628", "DC": "#4DAF4A",
}

CD8_ALL = ["CD8 T cells", "CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]
CD8_EXHAUSTED = ["CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]

SKIP_MARKERS = {"DNA1", "DNA2", "HistoneH3"}

# Markers to test for macrophage dependency
DEPENDENCY_MARKERS = [
    "VISTA", "CXCL13", "CD68", "HLA_Class_I", "CD11c",
    "IDO", "CXCL12", "CCL21", "CD21",
]
MARKER_DISPLAY = {
    "HLA_Class_I": "HLA-I", "HLA_DR": "HLA-DR", "BCL_2": "BCL-2",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_s_data(s_utag_path):
    """Load S-panel UTAG merged data (has compartments + expression)."""
    print("Loading S-panel UTAG merged h5ad...")
    f = h5py.File(s_utag_path, "r")
    X = f["X"][:]
    markers = [v.decode() if isinstance(v, bytes) else str(v)
               for v in f["var"]["_index"][:]]
    cell_types = load_array(f, "cell_type")
    sample_ids = load_array(f, "sample_id")
    comps = load_array(f, "compartment_name")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()

    marker_idx = {m: i for i, m in enumerate(markers)}

    # Filter to tumor cores (excl Biomax)
    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS
        and not s.startswith("Biomax")
        for s in sample_ids
    ])
    X = X[tumor_mask]
    cell_types = cell_types[tumor_mask]
    sample_ids = sample_ids[tumor_mask]
    comps = comps[tumor_mask]
    cx = cx[tumor_mask]
    cy = cy[tumor_mask]
    print(f"  {len(X):,} tumor cells")

    return {
        "X": X, "markers": markers, "marker_idx": marker_idx,
        "cell_types": cell_types, "sample_ids": sample_ids,
        "comps": comps, "cx": cx, "cy": cy,
    }


def extract_t_data(t_panel_path):
    """Load T-panel data — only need cell_type and sample_id for exhaustion."""
    print("Loading T-panel h5ad...")
    f = h5py.File(t_panel_path, "r")
    cell_types = load_array(f, "cell_type")
    sample_ids = load_array(f, "sample_id")
    f.close()

    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS
        and not s.startswith("Biomax")
        for s in sample_ids
    ])
    cell_types = cell_types[tumor_mask]
    sample_ids = sample_ids[tumor_mask]
    print(f"  {len(cell_types):,} tumor cells (T-panel)")

    return {"cell_types": cell_types, "sample_ids": sample_ids}


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_compartment_distribution(s):
    """Panel (a): compartment distribution by CD14 level."""
    print("\n--- Compartment distribution ---")
    fdc_mask = s["cell_types"] == "FDC"
    cd14_col = s["marker_idx"]["CD14"]
    fdc_cd14 = s["X"][fdc_mask, cd14_col]
    q25, q75 = np.percentile(fdc_cd14, [25, 75])

    fdc_comps = s["comps"][fdc_mask]
    hi_mask = fdc_cd14 >= q75
    lo_mask = fdc_cd14 <= q25

    results = {}
    for comp in S_COMP_ORDER:
        n_hi = (fdc_comps[hi_mask] == comp).sum()
        n_lo = (fdc_comps[lo_mask] == comp).sum()
        frac_hi = n_hi / hi_mask.sum() if hi_mask.sum() > 0 else 0
        frac_lo = n_lo / lo_mask.sum() if lo_mask.sum() > 0 else 0
        results[comp] = {
            "frac_hi": frac_hi, "frac_lo": frac_lo,
            "n_hi": int(n_hi), "n_lo": int(n_lo),
        }
        print(f"  {comp}: hi={frac_hi:.3f} lo={frac_lo:.3f} "
              f"(n_hi={n_hi}, n_lo={n_lo})")

    # Also report FDCs not in any named compartment
    named = set(S_COMP_ORDER)
    n_hi_other = sum(1 for c in fdc_comps[hi_mask] if c not in named)
    n_lo_other = sum(1 for c in fdc_comps[lo_mask] if c not in named)
    print(f"  Other/unnamed: hi={n_hi_other} lo={n_lo_other}")

    return {
        "comp_results": results, "q25": q25, "q75": q75,
        "n_hi": int(hi_mask.sum()), "n_lo": int(lo_mask.sum()),
    }


def analyze_fdc_mac_distance(s):
    """Panel (c): FDC→nearest macrophage distance by CD14 level."""
    print("\n--- FDC-macrophage distances ---")
    cd14_col = s["marker_idx"]["CD14"]
    fdc_mask = s["cell_types"] == "FDC"
    mac_mask = np.isin(s["cell_types"], MAC_SUBTYPES)

    dists_hi = []
    dists_lo = []

    for roi in np.unique(s["sample_ids"]):
        rmask = s["sample_ids"] == roi
        roi_fdc = np.where(rmask & fdc_mask)[0]
        roi_mac = np.where(rmask & mac_mask)[0]
        if len(roi_fdc) < 10 or len(roi_mac) < 10:
            continue

        mac_tree = KDTree(np.column_stack([
            s["cx"][roi_mac], s["cy"][roi_mac]
        ]))
        fdc_coords = np.column_stack([
            s["cx"][roi_fdc], s["cy"][roi_fdc]
        ])
        d, _ = mac_tree.query(fdc_coords, k=1)

        fdc_cd14 = s["X"][roi_fdc, cd14_col]
        q25_r, q75_r = np.percentile(fdc_cd14, [25, 75])
        for j in range(len(roi_fdc)):
            if fdc_cd14[j] >= q75_r:
                dists_hi.append(d[j])
            elif fdc_cd14[j] <= q25_r:
                dists_lo.append(d[j])

    dists_hi = np.array(dists_hi)
    dists_lo = np.array(dists_lo)
    stat, p = stats.mannwhitneyu(dists_hi, dists_lo, alternative="two-sided")
    print(f"  CD14-high FDC→mac: median={np.median(dists_hi):.1f}µm (n={len(dists_hi):,})")
    print(f"  CD14-low  FDC→mac: median={np.median(dists_lo):.1f}µm (n={len(dists_lo):,})")
    print(f"  Mann-Whitney p={p:.2e}")

    return {"dists_hi": dists_hi, "dists_lo": dists_lo, "p": p}


def analyze_mac_neighbor_fractions(s):
    """Panel (d): myeloid subtype neighbor fractions."""
    print("\n--- Myeloid neighbor fractions ---")
    cd14_col = s["marker_idx"]["CD14"]
    fdc_mask = s["cell_types"] == "FDC"

    fdc_counts_per_roi = Counter(s["sample_ids"][fdc_mask])
    candidate_rois = [r for r, c in fdc_counts_per_roi.items() if c >= 200]

    nbr_hi = Counter()
    nbr_lo = Counter()
    n_hi_total = 0
    n_lo_total = 0

    for roi in candidate_rois:
        rmask = s["sample_ids"] == roi
        roi_idx = np.where(rmask)[0]
        roi_ct = s["cell_types"][roi_idx]
        roi_cx = s["cx"][roi_idx]
        roi_cy = s["cy"][roi_idx]
        roi_cd14 = s["X"][roi_idx, cd14_col]

        fdc_local = np.where(roi_ct == "FDC")[0]
        non_fdc_local = np.where(roi_ct != "FDC")[0]
        if len(fdc_local) < 10 or len(non_fdc_local) < 20:
            continue

        tree = KDTree(np.column_stack([roi_cx[non_fdc_local], roi_cy[non_fdc_local]]))
        fdc_coords = np.column_stack([roi_cx[fdc_local], roi_cy[fdc_local]])
        _, idxs = tree.query(fdc_coords, k=10)

        fdc_cd14_vals = roi_cd14[fdc_local]
        q25_r, q75_r = np.percentile(fdc_cd14_vals, [25, 75])

        for j in range(len(fdc_local)):
            nbr_cts = roi_ct[non_fdc_local[idxs[j]]]
            # Only count macrophage subtypes
            mac_nbrs = [ct for ct in nbr_cts if ct in MAC_SUBTYPES]
            if fdc_cd14_vals[j] >= q75_r:
                nbr_hi.update(mac_nbrs)
                n_hi_total += 10  # total neighbors (all types)
            elif fdc_cd14_vals[j] <= q25_r:
                nbr_lo.update(mac_nbrs)
                n_lo_total += 10

    results = {}
    for mt in MAC_SUBTYPES:
        short = MAC_SHORT[mt]
        hi_frac = nbr_hi.get(mt, 0) / n_hi_total if n_hi_total > 0 else 0
        lo_frac = nbr_lo.get(mt, 0) / n_lo_total if n_lo_total > 0 else 0
        results[short] = {"hi": hi_frac, "lo": lo_frac}
        print(f"  {short}: hi={hi_frac*100:.2f}% lo={lo_frac*100:.2f}%")

    return {"mac_nbr": results, "n_hi": n_hi_total, "n_lo": n_lo_total}


def analyze_cross_panel(s, t):
    """Panel (e): per-ROI FDC CD14 (S) vs CD8 exhaustion (T)."""
    print("\n--- Cross-panel: FDC CD14 vs CD8 exhaustion ---")
    cd14_col = s["marker_idx"]["CD14"]
    fdc_mask = s["cell_types"] == "FDC"

    # S-panel: per-ROI mean FDC CD14
    roi_fdc_cd14 = {}
    for roi in np.unique(s["sample_ids"]):
        rmask = s["sample_ids"] == roi
        roi_fdc = rmask & fdc_mask
        if roi_fdc.sum() >= 10:
            roi_fdc_cd14[roi] = float(s["X"][roi_fdc, cd14_col].mean())

    # T-panel: per-ROI CD8 exhaustion fraction
    roi_exh_frac = {}
    for roi in np.unique(t["sample_ids"]):
        rmask = t["sample_ids"] == roi
        roi_ct = t["cell_types"][rmask]
        n_cd8_all = sum(1 for ct in roi_ct if ct in CD8_ALL)
        n_cd8_exh = sum(1 for ct in roi_ct if ct in CD8_EXHAUSTED)
        if n_cd8_all >= 10:
            roi_exh_frac[roi] = n_cd8_exh / n_cd8_all

    # Match ROIs across panels
    common = sorted(set(roi_fdc_cd14.keys()) & set(roi_exh_frac.keys()))
    print(f"  {len(common)} paired ROIs")

    if len(common) < 10:
        print("  WARNING: too few paired ROIs for correlation")
        return {"fdc_cd14": np.array([]), "exh_frac": np.array([]),
                "rho": np.nan, "p": np.nan, "n": 0}

    fdc_arr = np.array([roi_fdc_cd14[r] for r in common])
    exh_arr = np.array([roi_exh_frac[r] for r in common])
    rho, p = stats.spearmanr(fdc_arr, exh_arr)
    print(f"  Spearman rho={rho:.3f}, p={p:.2e}")

    return {"fdc_cd14": fdc_arr, "exh_frac": exh_arr,
            "rho": rho, "p": p, "n": len(common)}


def analyze_mac_dependency(s):
    """Panel (f): does CD14-high FDC phenotype depend on macrophage density?"""
    print("\n--- Macrophage dependency test ---")
    cd14_col = s["marker_idx"]["CD14"]
    fdc_mask = s["cell_types"] == "FDC"
    mac_mask = np.isin(s["cell_types"], MAC_SUBTYPES)

    # Per-ROI macrophage fraction
    roi_mac_frac = {}
    for roi in np.unique(s["sample_ids"]):
        rmask = s["sample_ids"] == roi
        n = rmask.sum()
        if n < 500:
            continue
        roi_mac_frac[roi] = mac_mask[rmask].sum() / n

    if not roi_mac_frac:
        return {"results": {}, "n_mac_hi": 0, "n_mac_lo": 0}

    rois_sorted = sorted(roi_mac_frac, key=roi_mac_frac.get)
    n_rois = len(rois_sorted)
    tertile = n_rois // 3
    mac_lo_rois = set(rois_sorted[:tertile])
    mac_hi_rois = set(rois_sorted[-tertile:])

    print(f"  Mac-low ROIs: {len(mac_lo_rois)} "
          f"(max frac={max(roi_mac_frac[r] for r in mac_lo_rois)*100:.1f}%)")
    print(f"  Mac-high ROIs: {len(mac_hi_rois)} "
          f"(min frac={min(roi_mac_frac[r] for r in mac_hi_rois)*100:.1f}%)")

    # For each group: compute CD14-high vs CD14-low FDC marker differences
    results = {}
    for marker in DEPENDENCY_MARKERS:
        if marker not in s["marker_idx"]:
            continue
        mi = s["marker_idx"][marker]
        display = MARKER_DISPLAY.get(marker, marker)

        diffs = {}
        for label, roi_set in [("mac_lo", mac_lo_rois), ("mac_hi", mac_hi_rois)]:
            # Collect all FDCs in these ROIs
            roi_mask = np.isin(s["sample_ids"], list(roi_set))
            fdc_in = roi_mask & fdc_mask
            if fdc_in.sum() < 50:
                diffs[label] = np.nan
                continue
            fdc_cd14 = s["X"][fdc_in, cd14_col]
            q25, q75 = np.percentile(fdc_cd14, [25, 75])
            hi = fdc_cd14 >= q75
            lo = fdc_cd14 <= q25
            fdc_X = s["X"][fdc_in, mi]
            diff = float(fdc_X[hi].mean() - fdc_X[lo].mean())
            diffs[label] = diff

        results[display] = diffs
        print(f"  {display}: mac-lo diff={diffs.get('mac_lo', np.nan):.3f}, "
              f"mac-hi diff={diffs.get('mac_hi', np.nan):.3f}")

    return {"results": results,
            "n_mac_lo": len(mac_lo_rois), "n_mac_hi": len(mac_hi_rois)}


def select_rep_roi(s):
    """Select representative ROI for spatial scatter."""
    print("\n--- Selecting representative ROI ---")
    fdc_mask = s["cell_types"] == "FDC"
    mac_mask = np.isin(s["cell_types"], MAC_SUBTYPES)

    roi_scores = {}
    for roi in np.unique(s["sample_ids"]):
        rmask = s["sample_ids"] == roi
        n_fdc = (rmask & fdc_mask).sum()
        n_mac = (rmask & mac_mask).sum()
        if n_fdc >= 200 and n_mac >= 50:
            roi_scores[roi] = (n_fdc * n_mac) ** 0.5

    if not roi_scores:
        # Fallback: most FDCs
        fdc_per_roi = Counter(s["sample_ids"][fdc_mask])
        rep_roi = fdc_per_roi.most_common(1)[0][0]
    else:
        rep_roi = max(roi_scores, key=roi_scores.get)

    print(f"  Representative ROI: {rep_roi}")
    rmask = s["sample_ids"] == rep_roi
    return {
        "roi": rep_roi,
        "x": s["cx"][rmask], "y": s["cy"][rmask],
        "ct": s["cell_types"][rmask],
        "cd14": s["X"][rmask, s["marker_idx"]["CD14"]],
        "comps": s["comps"][rmask],
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(comp_dist, fdc_mac_dist, mac_nbr, cross_panel,
                mac_dep, rep_data, output_dir):
    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.35,
                  left=0.09, right=0.95, top=0.97, bottom=0.04,
                  height_ratios=[1.0, 1.0, 1.0])

    # ── (a) Compartment distribution ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")

    comp_res = comp_dist["comp_results"]
    comps_present = [c for c in S_COMP_ORDER if c in comp_res]
    x_pos = np.arange(len(comps_present))
    width = 0.35

    hi_fracs = [comp_res[c]["frac_hi"] * 100 for c in comps_present]
    lo_fracs = [comp_res[c]["frac_lo"] * 100 for c in comps_present]

    ax_a.bar(x_pos - width/2, hi_fracs, width, label="CD14-high (Q75+)",
             color="#FFD700", edgecolor="black", linewidth=0.5)
    ax_a.bar(x_pos + width/2, lo_fracs, width, label="CD14-low (Q25-)",
             color="#4393C3", edgecolor="black", linewidth=0.5)

    short_labels = [S_COMP_SHORT[S_COMP_ORDER.index(c)] for c in comps_present]
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(short_labels, fontsize=8)
    ax_a.set_ylabel("% of FDCs in compartment")
    ax_a.set_title(
        f"FDC compartment distribution by CD14 level\n"
        f"(n_hi={comp_dist['n_hi']:,}, n_lo={comp_dist['n_lo']:,})",
        fontsize=11,
    )
    ax_a.legend(fontsize=9, loc="upper right")

    # Add significance markers
    for i, c in enumerate(comps_present):
        r = comp_res[c]
        # 2-proportion z-test
        n1, n2 = comp_dist["n_hi"], comp_dist["n_lo"]
        p1, p2 = r["frac_hi"], r["frac_lo"]
        p_pool = (r["n_hi"] + r["n_lo"]) / (n1 + n2)
        if 0 < p_pool < 1:
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z = (p1 - p2) / se if se > 0 else 0
            pval = 2 * stats.norm.sf(abs(z))
            if pval < 0.001:
                stars = "***"
            elif pval < 0.01:
                stars = "**"
            elif pval < 0.05:
                stars = "*"
            else:
                stars = ""
            if stars:
                ymax = max(hi_fracs[i], lo_fracs[i])
                ax_a.text(i, ymax + 0.5, stars, ha="center", fontsize=9,
                          color="#333333")

    # ── (b) Spatial scatter ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    rd = rep_data
    x, y = rd["x"], rd["y"]
    ct = rd["ct"]
    cd14 = rd["cd14"]
    rcomps = rd["comps"]
    is_fdc = ct == "FDC"
    is_mac = np.isin(ct, MAC_SUBTYPES)

    # Background tint by compartment
    comp_color_map = dict(zip(S_COMP_ORDER, S_COMP_COLORS))
    other_mask = ~is_fdc & ~is_mac
    ax_b.scatter(x[other_mask], y[other_mask], c="#D3D3D3", s=0.3, alpha=0.2,
                 rasterized=True, zorder=1)

    # Macrophages
    ax_b.scatter(x[is_mac], y[is_mac], c="#E41A1C", s=4, alpha=0.7,
                 marker="^", edgecolors="black", linewidth=0.2,
                 label="Macrophage", rasterized=True, zorder=3)

    # FDCs by CD14 level
    fdc_x, fdc_y = x[is_fdc], y[is_fdc]
    fdc_cd14 = cd14[is_fdc]
    q25_local = np.percentile(fdc_cd14, 25)
    q75_local = np.percentile(fdc_cd14, 75)

    lo = fdc_cd14 <= q25_local
    mid = (~lo) & (fdc_cd14 < q75_local)
    hi = fdc_cd14 >= q75_local

    ax_b.scatter(fdc_x[lo], fdc_y[lo], c="#4393C3", s=4, alpha=0.6,
                 edgecolors="black", linewidth=0.2, label="FDC CD14-low",
                 rasterized=True, zorder=4)
    ax_b.scatter(fdc_x[mid], fdc_y[mid], c="#FDDBC7", s=3, alpha=0.4,
                 rasterized=True, zorder=2)
    ax_b.scatter(fdc_x[hi], fdc_y[hi], c="#FFD700", s=8, alpha=0.9,
                 edgecolors="black", linewidth=0.3, label="FDC CD14-high",
                 rasterized=True, zorder=5)

    ax_b.set_aspect("equal")
    ax_b.invert_yaxis()
    ax_b.set_title(f"FDC localization — {rd['roi']}", fontsize=11)
    ax_b.set_xlabel("x (µm)")
    ax_b.set_ylabel("y (µm)")
    ax_b.legend(fontsize=8, loc="upper right", markerscale=2)

    n_fdc_roi = is_fdc.sum()
    n_mac_roi = is_mac.sum()
    ax_b.text(0.02, 0.02,
              f"{n_fdc_roi:,} FDCs, {n_mac_roi:,} myeloid",
              transform=ax_b.transAxes, fontsize=8,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── (c) FDC→macrophage distance ──
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")

    dhi = fdc_mac_dist["dists_hi"]
    dlo = fdc_mac_dist["dists_lo"]
    # Clip for display
    clip = 200
    dhi_c = np.clip(dhi, 0, clip)
    dlo_c = np.clip(dlo, 0, clip)

    parts = ax_c.violinplot([dhi_c, dlo_c], positions=[0, 1],
                             showmedians=True, showextrema=False)
    colors_v = ["#FFD700", "#4393C3"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_v[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["CD14-high\nFDC", "CD14-low\nFDC"], fontsize=10)
    ax_c.set_ylabel("Distance to nearest macrophage (µm)")
    ax_c.set_title("FDC → nearest macrophage distance", fontsize=11)

    med_hi = np.median(dhi)
    med_lo = np.median(dlo)
    p_dist = fdc_mac_dist["p"]
    stars = "***" if p_dist < 0.001 else ("**" if p_dist < 0.01 else
            ("*" if p_dist < 0.05 else "n.s."))
    ax_c.text(0.5, 0.95,
              f"Median: {med_hi:.1f} vs {med_lo:.1f} µm\n"
              f"p={p_dist:.1e} {stars}",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── (d) Macrophage neighbor fractions ──
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")

    mnb = mac_nbr["mac_nbr"]
    mac_types = [k for k in ["M1 Mac", "M2 Mac", "S100A9+", "DC", "Mac"]
                 if k in mnb]
    x_pos_d = np.arange(len(mac_types))
    width_d = 0.35

    hi_vals = [mnb[mt]["hi"] * 100 for mt in mac_types]
    lo_vals = [mnb[mt]["lo"] * 100 for mt in mac_types]

    ax_d.bar(x_pos_d - width_d/2, hi_vals, width_d,
             label="CD14-high FDC", color="#FFD700",
             edgecolor="black", linewidth=0.5)
    ax_d.bar(x_pos_d + width_d/2, lo_vals, width_d,
             label="CD14-low FDC", color="#4393C3",
             edgecolor="black", linewidth=0.5)

    ax_d.set_xticks(x_pos_d)
    ax_d.set_xticklabels(mac_types, fontsize=9)
    ax_d.set_ylabel("% of k=10 neighbors")
    ax_d.set_title(
        f"Myeloid neighbors of FDCs\n"
        f"(n_hi={mac_nbr['n_hi']//10:,} FDCs, "
        f"n_lo={mac_nbr['n_lo']//10:,} FDCs)",
        fontsize=11,
    )
    ax_d.legend(fontsize=9, loc="upper right")

    # ── (e) Cross-panel: FDC CD14 vs CD8 exhaustion ──
    ax_e = fig.add_subplot(gs[2, 0])
    panel_label(ax_e, "e")

    if cross_panel["n"] > 0:
        fdc_arr = cross_panel["fdc_cd14"]
        exh_arr = cross_panel["exh_frac"] * 100
        ax_e.scatter(fdc_arr, exh_arr, c="#6A5ACD", alpha=0.5, s=25,
                     edgecolors="white", linewidth=0.3)
        # Regression line
        m_fit, b_fit = np.polyfit(fdc_arr, exh_arr, 1)
        x_range = np.linspace(fdc_arr.min(), fdc_arr.max(), 50)
        ax_e.plot(x_range, m_fit * x_range + b_fit, "r--",
                  linewidth=1.5, alpha=0.7)
        ax_e.set_xlabel("Per-ROI mean FDC CD14 (S-panel)")
        ax_e.set_ylabel("CD8 T exhaustion fraction (%, T-panel)")
        ax_e.set_title(
            f"Cross-panel: FDC CD14 vs CD8 exhaustion\n"
            f"({cross_panel['n']} paired ROIs)",
            fontsize=11,
        )
        ax_e.text(0.05, 0.95,
                  f"Spearman ρ={cross_panel['rho']:.3f}\n"
                  f"p={cross_panel['p']:.2e}",
                  transform=ax_e.transAxes, va="top", fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            alpha=0.8))
    else:
        ax_e.text(0.5, 0.5, "Insufficient paired ROIs",
                  transform=ax_e.transAxes, ha="center", fontsize=12)
        ax_e.set_title("Cross-panel: FDC CD14 vs CD8 exhaustion", fontsize=11)

    # ── (f) Macrophage dependency ──
    ax_f = fig.add_subplot(gs[2, 1])
    panel_label(ax_f, "f")

    dep = mac_dep["results"]
    if dep:
        markers_show = [m for m in dep if not (np.isnan(dep[m].get("mac_lo", np.nan))
                        or np.isnan(dep[m].get("mac_hi", np.nan)))]
        if markers_show:
            y_pos_f = np.arange(len(markers_show))
            mac_lo_diffs = [dep[m]["mac_lo"] for m in markers_show]
            mac_hi_diffs = [dep[m]["mac_hi"] for m in markers_show]
            width_f = 0.35

            ax_f.barh(y_pos_f - width_f/2, mac_hi_diffs, width_f,
                      label=f"Mac-high ROIs (n={mac_dep['n_mac_hi']})",
                      color="#E41A1C", edgecolor="white", linewidth=0.5,
                      alpha=0.7)
            ax_f.barh(y_pos_f + width_f/2, mac_lo_diffs, width_f,
                      label=f"Mac-low ROIs (n={mac_dep['n_mac_lo']})",
                      color="#377EB8", edgecolor="white", linewidth=0.5,
                      alpha=0.7)
            ax_f.axvline(0, color="black", linewidth=0.5)
            ax_f.set_yticks(y_pos_f)
            ax_f.set_yticklabels(markers_show, fontsize=9)
            ax_f.set_xlabel("Marker diff (CD14-high − CD14-low FDC)")
            ax_f.set_title(
                "Macrophage dependency test:\nCD14-high FDC phenotype by context",
                fontsize=11,
            )
            ax_f.legend(fontsize=8, loc="lower right")

            # Annotate: count how many markers agree in direction
            n_agree = sum(
                1 for i in range(len(markers_show))
                if abs(mac_lo_diffs[i]) > 0.05  # skip near-zero
                and (mac_lo_diffs[i] > 0) == (mac_hi_diffs[i] > 0)
            )
            n_tested = sum(
                1 for i in range(len(markers_show))
                if abs(mac_lo_diffs[i]) > 0.05
            )
            persists = n_agree >= n_tested - 1  # allow 1 exception
            verdict = (f"Phenotype PERSISTS ({n_agree}/{n_tested} markers)\n"
                       f"in macrophage-poor ROIs"
                       if persists else
                       f"Phenotype CONTEXT-DEPENDENT\n"
                       f"({n_tested - n_agree}/{n_tested} markers flip)")
            ax_f.text(0.98, 0.02, verdict,
                      transform=ax_f.transAxes, ha="right", va="bottom",
                      fontsize=9, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="#E8F5E9" if persists else "#FFEBEE",
                                alpha=0.8))
    else:
        ax_f.text(0.5, 0.5, "Insufficient data",
                  transform=ax_f.transAxes, ha="center", fontsize=12)

    out = Path(output_dir) / "fig_fdc_localization.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s-utag",
                        default="output/all_TMA_S_utag_ct_merged.h5ad",
                        help="S-panel UTAG merged h5ad (has compartment_name)")
    parser.add_argument("--t-panel",
                        default="output/all_TMA_T_global_v8.h5ad",
                        help="T-panel annotated h5ad (for CD8 exhaustion)")
    parser.add_argument("--output-dir",
                        default="output/hypotheses_v8",
                        help="Output directory for figure")
    args = parser.parse_args()

    s = extract_s_data(args.s_utag)
    t = extract_t_data(args.t_panel)

    comp_dist = analyze_compartment_distribution(s)
    fdc_mac_dist = analyze_fdc_mac_distance(s)
    mac_nbr = analyze_mac_neighbor_fractions(s)
    cross_panel = analyze_cross_panel(s, t)
    mac_dep = analyze_mac_dependency(s)
    rep_data = select_rep_roi(s)

    make_figure(comp_dist, fdc_mac_dist, mac_nbr, cross_panel,
                mac_dep, rep_data, args.output_dir)


if __name__ == "__main__":
    main()
