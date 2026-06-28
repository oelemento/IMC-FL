#!/usr/bin/env python3
"""Cross-panel spatial proximity: CD14+ FDCs (S-panel) and exhausted CD8 T cells (T-panel).

Tests whether CD14-high FDCs are in spatial contact with exhausted CD8 T cells
using serial section coordinates.

Panels:
  (a) Distance from CD8 T cells to nearest FDC, stratified by exhaustion status
  (b) CD8 exhaustion fraction as a function of distance to nearest CD14-high FDC
  (c) CD8 exhaustion fraction near CD14-high vs CD14-low FDCs (within 50 µm)
  (d) Per-ROI scatter: local CD14-high FDC density vs CD8 exhaustion fraction
  (e) Representative ROI: spatial map of FDCs + exhausted CD8 T cells

Usage:
    .venv/bin/python scripts/fig_fdc_cd8_proximity.py \
        --s-utag output/all_TMA_S_utag_ct_merged.h5ad \
        --t-utag output/all_TMA_T_utag_ct_merged.h5ad \
        --output-dir output/hypotheses_v8
"""

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    ax.text(x, y, f"$\\bf{{{letter}}}$",
            transform=ax.transAxes, fontsize=14, va="top", ha="left")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CD8_ALL = ["CD8 T cells", "CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]
CD8_EXHAUSTED = ["CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]

PROXIMITY_RADIUS = 50  # µm — define "in contact"
DISTANCE_BINS = [0, 25, 50, 100, 200, 500]  # for distance-dependent analysis

# Follicular compartments in T-panel UTAG
T_FOLLICULAR = [
    "GC core", "Follicle core", "Follicle mantle",
    "B cell zone", "Follicle-T interface", "Mixed B/T zone",
]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_s_data(path):
    """Load S-panel UTAG: FDC positions + CD14."""
    print("Loading S-panel UTAG...")
    f = h5py.File(path, "r")
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

    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS
        and not s.startswith("Biomax")
        for s in sample_ids
    ])
    return {
        "X": X[tumor_mask], "markers": markers, "marker_idx": marker_idx,
        "cell_types": cell_types[tumor_mask],
        "sample_ids": sample_ids[tumor_mask],
        "comps": comps[tumor_mask],
        "cx": cx[tumor_mask], "cy": cy[tumor_mask],
    }


def extract_t_data(path):
    """Load T-panel UTAG: CD8 T positions + exhaustion status."""
    print("Loading T-panel UTAG...")
    f = h5py.File(path, "r")
    cell_types = load_array(f, "cell_type")
    sample_ids = load_array(f, "sample_id")
    comps = load_array(f, "compartment_name")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()

    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS
        and not s.startswith("Biomax")
        for s in sample_ids
    ])
    return {
        "cell_types": cell_types[tumor_mask],
        "sample_ids": sample_ids[tumor_mask],
        "comps": comps[tumor_mask],
        "cx": cx[tumor_mask], "cy": cy[tumor_mask],
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_proximity(s, t):
    """Core analysis: spatial proximity between FDCs (S) and CD8 T cells (T)."""
    cd14_col = s["marker_idx"]["CD14"]
    fdc_mask_s = s["cell_types"] == "FDC"
    fdc_cd14_all = s["X"][fdc_mask_s, cd14_col]
    q75 = np.percentile(fdc_cd14_all, 75)
    q25 = np.percentile(fdc_cd14_all, 25)
    print(f"CD14 thresholds: Q25={q25:.3f}, Q75={q75:.3f}")

    cd8_mask_t = np.isin(t["cell_types"], CD8_ALL)
    exh_set = set(CD8_EXHAUSTED)

    s_rois = set(np.unique(s["sample_ids"]))
    t_rois = set(np.unique(t["sample_ids"]))
    common_rois = sorted(s_rois & t_rois)
    print(f"Common ROIs: {len(common_rois)}")

    # Collect per-cell results
    all_cd8_dist_to_fdc = []      # distance from each CD8 T to nearest FDC
    all_cd8_dist_to_hi = []       # distance to nearest CD14-high FDC
    all_cd8_dist_to_lo = []       # distance to nearest CD14-low FDC
    all_cd8_is_exh = []           # whether each CD8 T is exhausted
    all_cd8_roi = []              # ROI label

    # Per-ROI summaries
    roi_results = []

    n_rois_used = 0
    for roi in common_rois:
        # S-panel FDCs in this ROI
        s_roi = s["sample_ids"] == roi
        fdc_in_roi = s_roi & fdc_mask_s
        n_fdc = fdc_in_roi.sum()
        if n_fdc < 5:
            continue

        # T-panel CD8 T cells in this ROI
        t_roi = t["sample_ids"] == roi
        cd8_in_roi = t_roi & cd8_mask_t
        n_cd8 = cd8_in_roi.sum()
        if n_cd8 < 5:
            continue

        # FDC positions and CD14
        fdc_x = s["cx"][fdc_in_roi]
        fdc_y = s["cy"][fdc_in_roi]
        fdc_cd14 = s["X"][fdc_in_roi, cd14_col]
        fdc_hi = fdc_cd14 >= q75
        fdc_lo = fdc_cd14 <= q25

        # CD8 T positions and exhaustion
        cd8_x = t["cx"][cd8_in_roi]
        cd8_y = t["cy"][cd8_in_roi]
        cd8_ct = t["cell_types"][cd8_in_roi]
        cd8_exh = np.array([ct in exh_set for ct in cd8_ct])

        # KDTree for all FDCs
        fdc_tree = KDTree(np.column_stack([fdc_x, fdc_y]))
        cd8_pts = np.column_stack([cd8_x, cd8_y])
        dists_all, idxs_all = fdc_tree.query(cd8_pts, k=1)

        all_cd8_dist_to_fdc.extend(dists_all)
        all_cd8_is_exh.extend(cd8_exh)
        all_cd8_roi.extend([roi] * len(cd8_exh))

        # KDTree for CD14-high FDCs
        if fdc_hi.sum() >= 3:
            hi_tree = KDTree(np.column_stack([fdc_x[fdc_hi], fdc_y[fdc_hi]]))
            dists_hi, _ = hi_tree.query(cd8_pts, k=1)
            all_cd8_dist_to_hi.extend(dists_hi)
        else:
            all_cd8_dist_to_hi.extend([np.nan] * len(cd8_exh))

        # KDTree for CD14-low FDCs
        if fdc_lo.sum() >= 3:
            lo_tree = KDTree(np.column_stack([fdc_x[fdc_lo], fdc_y[fdc_lo]]))
            dists_lo, _ = lo_tree.query(cd8_pts, k=1)
            all_cd8_dist_to_lo.extend(dists_lo)
        else:
            all_cd8_dist_to_lo.extend([np.nan] * len(cd8_exh))

        # Per-ROI summary
        near_fdc = dists_all <= PROXIMITY_RADIUS
        n_near = near_fdc.sum()
        exh_near = cd8_exh[near_fdc].sum() if n_near > 0 else 0
        exh_far = cd8_exh[~near_fdc].sum()
        n_far = (~near_fdc).sum()

        # Which FDC is nearest to each CD8? Is it CD14-high?
        nearest_fdc_cd14 = fdc_cd14[idxs_all]
        nearest_is_hi = nearest_fdc_cd14 >= q75

        roi_results.append({
            "roi": roi,
            "n_fdc": int(n_fdc), "n_cd8": int(n_cd8),
            "n_fdc_hi": int(fdc_hi.sum()),
            "exh_frac_near": exh_near / n_near if n_near > 5 else np.nan,
            "exh_frac_far": exh_far / n_far if n_far > 5 else np.nan,
            "n_near": int(n_near), "n_far": int(n_far),
            "median_dist_exh": float(np.median(dists_all[cd8_exh])) if cd8_exh.sum() > 0 else np.nan,
            "median_dist_nonexh": float(np.median(dists_all[~cd8_exh])) if (~cd8_exh).sum() > 0 else np.nan,
        })
        n_rois_used += 1

    print(f"ROIs used: {n_rois_used}")

    results = {
        "dist_to_fdc": np.array(all_cd8_dist_to_fdc),
        "dist_to_hi": np.array(all_cd8_dist_to_hi),
        "dist_to_lo": np.array(all_cd8_dist_to_lo),
        "is_exh": np.array(all_cd8_is_exh),
        "roi": np.array(all_cd8_roi),
        "roi_results": roi_results,
        "q75": q75, "q25": q25,
    }

    # Print summary stats
    valid = ~np.isnan(results["dist_to_fdc"])
    exh = results["is_exh"][valid]
    dist = results["dist_to_fdc"][valid]

    med_exh = np.median(dist[exh])
    med_nonexh = np.median(dist[~exh])
    u_stat, u_p = stats.mannwhitneyu(dist[exh], dist[~exh], alternative="two-sided")
    print(f"\nDistance to nearest FDC:")
    print(f"  Exhausted CD8: median={med_exh:.1f} µm (n={exh.sum():,})")
    print(f"  Non-exhausted CD8: median={med_nonexh:.1f} µm (n={(~exh).sum():,})")
    print(f"  Mann-Whitney P={u_p:.2e}")

    # Near CD14-high vs CD14-low FDCs
    hi_valid = ~np.isnan(results["dist_to_hi"])
    lo_valid = ~np.isnan(results["dist_to_lo"])
    both_valid = hi_valid & lo_valid

    if both_valid.sum() > 100:
        near_hi = results["dist_to_hi"][both_valid] <= PROXIMITY_RADIUS
        near_lo = results["dist_to_lo"][both_valid] <= PROXIMITY_RADIUS
        exh_bv = results["is_exh"][both_valid]

        exh_near_hi = exh_bv[near_hi].mean() if near_hi.sum() > 10 else np.nan
        exh_near_lo = exh_bv[near_lo].mean() if near_lo.sum() > 10 else np.nan
        exh_neither = exh_bv[~near_hi & ~near_lo].mean() if (~near_hi & ~near_lo).sum() > 10 else np.nan

        print(f"\nCD8 exhaustion rate by FDC proximity (within {PROXIMITY_RADIUS} µm):")
        print(f"  Near CD14-high FDC: {exh_near_hi:.1%} (n={near_hi.sum():,})")
        print(f"  Near CD14-low FDC:  {exh_near_lo:.1%} (n={near_lo.sum():,})")
        print(f"  Near neither:       {exh_neither:.1%} (n={(~near_hi & ~near_lo).sum():,})")

        results["exh_near_hi"] = exh_near_hi
        results["exh_near_lo"] = exh_near_lo
        results["exh_neither"] = exh_neither
        results["n_near_hi"] = int(near_hi.sum())
        results["n_near_lo"] = int(near_lo.sum())

    # Distance-binned exhaustion fraction
    bins = DISTANCE_BINS + [9999]
    bin_exh = []
    bin_n = []
    for i in range(len(bins) - 1):
        lo_b, hi_b = bins[i], bins[i + 1]
        # Use distance to nearest CD14-high FDC
        if hi_valid.sum() > 0:
            in_bin = (results["dist_to_hi"] >= lo_b) & (results["dist_to_hi"] < hi_b) & hi_valid
        else:
            in_bin = (dist >= lo_b) & (dist < hi_b)
        n = in_bin.sum()
        if n > 10:
            bin_exh.append(results["is_exh"][in_bin].mean())
        else:
            bin_exh.append(np.nan)
        bin_n.append(int(n))
    results["bin_exh"] = bin_exh
    results["bin_n"] = bin_n
    results["bin_edges"] = bins

    return results


def select_rep_roi(s, t, results):
    """Pick a representative ROI with both FDCs and exhausted CD8 T cells."""
    # Choose ROI with most CD8 T cells near FDCs
    best_roi = None
    best_n = 0
    for rr in results["roi_results"]:
        if rr["n_near"] > best_n and rr["n_fdc_hi"] >= 5:
            best_n = rr["n_near"]
            best_roi = rr["roi"]
    print(f"Representative ROI: {best_roi} ({best_n} CD8 T near FDCs)")
    return best_roi


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(s, t, results, output_dir):
    """5-panel figure."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.97, top=0.92, bottom=0.08)

    # ── (a) Distance to nearest FDC: exhausted vs non-exhausted ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")

    valid = ~np.isnan(results["dist_to_fdc"])
    exh = results["is_exh"][valid]
    dist = results["dist_to_fdc"][valid]

    # Clip for visualization
    clip = 300
    dist_clip = np.clip(dist, 0, clip)

    ax_a.hist(dist_clip[exh], bins=50, range=(0, clip), alpha=0.6,
              color="#D32F2F", density=True, label="Exhausted CD8")
    ax_a.hist(dist_clip[~exh], bins=50, range=(0, clip), alpha=0.6,
              color="#1976D2", density=True, label="Non-exhausted CD8")
    ax_a.axvline(PROXIMITY_RADIUS, color="gray", ls="--", lw=1, alpha=0.7)
    ax_a.set_xlabel("Distance to nearest FDC (µm)")
    ax_a.set_ylabel("Density")

    med_exh = np.median(dist[exh])
    med_nonexh = np.median(dist[~exh])
    u_stat, u_p = stats.mannwhitneyu(dist[exh], dist[~exh])
    ax_a.set_title(
        f"CD8 T → nearest FDC distance\n"
        f"(exh median={med_exh:.0f}, non-exh={med_nonexh:.0f} µm, P={u_p:.1e})",
        fontsize=10)
    ax_a.legend(fontsize=8)

    # ── (b) Exhaustion fraction vs distance to nearest CD14-high FDC ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")

    bin_labels = []
    for i in range(len(DISTANCE_BINS)):
        if i < len(DISTANCE_BINS) - 1:
            bin_labels.append(f"{DISTANCE_BINS[i]}–{DISTANCE_BINS[i+1]}")
        else:
            bin_labels.append(f">{DISTANCE_BINS[i]}")

    valid_bins = [(i, results["bin_exh"][i]) for i in range(len(bin_labels))
                  if not np.isnan(results["bin_exh"][i])]
    if valid_bins:
        x_pos = [v[0] for v in valid_bins]
        y_vals = [v[1] * 100 for v in valid_bins]
        n_vals = [results["bin_n"][v[0]] for v in valid_bins]
        bars = ax_b.bar(x_pos, y_vals, color="#E64A19", alpha=0.8, width=0.7)
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels([bin_labels[i] for i in x_pos], fontsize=8, rotation=30)
        # Add n labels
        for bar, n in zip(bars, n_vals):
            ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f"n={n:,}", ha="center", va="bottom", fontsize=7)
    ax_b.set_xlabel("Distance to nearest CD14-high FDC (µm)")
    ax_b.set_ylabel("CD8 exhaustion rate (%)")
    ax_b.set_title("Exhaustion rate by distance\nto CD14-high FDC", fontsize=10)

    # ── (c) Exhaustion near CD14-high vs CD14-low FDCs ──
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c")

    if "exh_near_hi" in results:
        cats = ["Near CD14-hi\nFDC", "Near CD14-lo\nFDC", "Far from\nboth"]
        vals = [results.get("exh_near_hi", 0) * 100,
                results.get("exh_near_lo", 0) * 100,
                results.get("exh_neither", 0) * 100]
        ns = [results.get("n_near_hi", 0),
              results.get("n_near_lo", 0),
              0]  # don't have exact n for neither stored separately
        colors = ["#D32F2F", "#1976D2", "#757575"]
        bars = ax_c.bar(range(3), vals, color=colors, alpha=0.85, width=0.6)
        ax_c.set_xticks(range(3))
        ax_c.set_xticklabels(cats, fontsize=9)
        ax_c.set_ylabel("CD8 exhaustion rate (%)")
        ax_c.set_title(
            f"CD8 exhaustion within {PROXIMITY_RADIUS} µm\n"
            f"of FDC by CD14 status", fontsize=10)
        for bar, n in zip(bars[:2], ns[:2]):
            if n > 0:
                ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          f"n={n:,}", ha="center", va="bottom", fontsize=8)
    else:
        ax_c.text(0.5, 0.5, "Insufficient data", transform=ax_c.transAxes,
                  ha="center", va="center")

    # ── (d) Per-ROI: FDC density near exhausted CD8 ──
    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, "d")

    rr = results["roi_results"]
    exh_near = [r["exh_frac_near"] for r in rr if not np.isnan(r.get("exh_frac_near", np.nan))]
    exh_far = [r["exh_frac_far"] for r in rr if not np.isnan(r.get("exh_frac_far", np.nan))]

    if exh_near and exh_far:
        # Paired ROI comparison: exhaustion near vs far from FDCs
        paired_rois = [r for r in rr
                       if not np.isnan(r.get("exh_frac_near", np.nan))
                       and not np.isnan(r.get("exh_frac_far", np.nan))]
        near_vals = np.array([r["exh_frac_near"] for r in paired_rois]) * 100
        far_vals = np.array([r["exh_frac_far"] for r in paired_rois]) * 100

        ax_d.scatter(far_vals, near_vals, alpha=0.4, s=20, c="#333")
        lim = max(max(near_vals), max(far_vals)) * 1.05
        ax_d.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        ax_d.set_xlabel(f"CD8 exhaustion rate (%) — far from FDC (>{PROXIMITY_RADIUS} µm)")
        ax_d.set_ylabel(f"CD8 exhaustion rate (%) — near FDC (≤{PROXIMITY_RADIUS} µm)")

        # Wilcoxon signed-rank for paired comparison
        if len(near_vals) >= 10:
            w_stat, w_p = stats.wilcoxon(near_vals, far_vals)
            diff = np.mean(near_vals - far_vals)
            ax_d.set_title(
                f"Per-ROI: CD8 exhaustion near vs far from FDC\n"
                f"(mean diff={diff:+.1f}%, Wilcoxon P={w_p:.2e}, n={len(paired_rois)} ROIs)",
                fontsize=10)
        else:
            ax_d.set_title(f"Per-ROI: CD8 exhaustion near vs far from FDC", fontsize=10)

    # ── (e) Representative ROI spatial map ──
    ax_e = fig.add_subplot(gs[1, 1:])
    panel_label(ax_e, "e")

    rep_roi = select_rep_roi(s, t, results)
    if rep_roi:
        cd14_col = s["marker_idx"]["CD14"]
        q75 = results["q75"]

        # S-panel cells in ROI
        s_roi = s["sample_ids"] == rep_roi
        s_cx, s_cy = s["cx"][s_roi], s["cy"][s_roi]
        s_ct = s["cell_types"][s_roi]
        s_cd14 = s["X"][s_roi, cd14_col]
        s_fdc = s_ct == "FDC"

        # T-panel cells in ROI
        t_roi = t["sample_ids"] == rep_roi
        t_cx, t_cy = t["cx"][t_roi], t["cy"][t_roi]
        t_ct = t["cell_types"][t_roi]
        t_cd8 = np.isin(t_ct, CD8_ALL)
        t_exh = np.isin(t_ct, CD8_EXHAUSTED)

        # Background: all cells in gray
        ax_e.scatter(s_cx, s_cy, s=0.5, c="#E0E0E0", alpha=0.2, rasterized=True)

        # FDCs: CD14-high in gold, CD14-low in teal
        fdc_hi = s_fdc & (s_cd14 >= q75)
        fdc_lo = s_fdc & (s_cd14 < q75)
        ax_e.scatter(s_cx[fdc_lo], s_cy[fdc_lo], s=25, c="#26A69A",
                     edgecolors="black", linewidths=0.3, zorder=3,
                     label=f"CD14-low FDC (n={fdc_lo.sum()})")
        ax_e.scatter(s_cx[fdc_hi], s_cy[fdc_hi], s=35, c="#FFD700",
                     edgecolors="black", linewidths=0.5, zorder=4,
                     label=f"CD14-high FDC (n={fdc_hi.sum()})")

        # CD8 T cells: exhausted in red, non-exhausted in blue
        cd8_nonexh = t_cd8 & ~t_exh
        ax_e.scatter(t_cx[cd8_nonexh], t_cy[cd8_nonexh], s=12, c="#42A5F5",
                     alpha=0.6, marker="^", zorder=2,
                     label=f"CD8 T non-exh (n={cd8_nonexh.sum()})")
        ax_e.scatter(t_cx[t_exh], t_cy[t_exh], s=18, c="#D32F2F",
                     edgecolors="black", linewidths=0.3, marker="^", zorder=3,
                     label=f"CD8 T exhausted (n={t_exh.sum()})")

        # Draw proximity circles around CD14-high FDCs
        for x, y in zip(s_cx[fdc_hi], s_cy[fdc_hi]):
            circ = plt.Circle((x, y), PROXIMITY_RADIUS, fill=False,
                              edgecolor="#FFD700", alpha=0.15, lw=0.5)
            ax_e.add_patch(circ)

        ax_e.set_aspect("equal")
        ax_e.invert_yaxis()
        ax_e.set_xlabel("x (µm)")
        ax_e.set_ylabel("y (µm)")
        ax_e.set_title(f"Representative ROI: {rep_roi}\n"
                       f"(S-panel FDCs + T-panel CD8 T cells, serial sections)",
                       fontsize=10)
        ax_e.legend(fontsize=7, loc="upper right", markerscale=1.5)

    fig.suptitle(
        "Cross-panel spatial proximity: CD14+ FDCs and exhausted CD8 T cells",
        fontsize=14, fontweight="bold", y=0.98)

    out_path = Path(output_dir) / "fig_fdc_cd8_proximity.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s-utag",
                        default="output/all_TMA_S_utag_ct_merged.h5ad",
                        help="S-panel UTAG merged h5ad")
    parser.add_argument("--t-utag",
                        default="output/all_TMA_T_utag_ct_merged.h5ad",
                        help="T-panel UTAG merged h5ad")
    parser.add_argument("--output-dir",
                        default="output/hypotheses_v8",
                        help="Output directory")
    args = parser.parse_args()

    s = extract_s_data(args.s_utag)
    t = extract_t_data(args.t_utag)

    results = analyze_proximity(s, t)
    make_figure(s, t, results, args.output_dir)


if __name__ == "__main__":
    main()
