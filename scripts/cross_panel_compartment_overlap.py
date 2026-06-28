#!/usr/bin/env python3
"""Cross-panel compartment overlap heatmap.

For each ROI present in both T-panel and S-panel UTAG files,
assign each T-panel cell to its nearest S-panel cell (and vice versa)
using spatial coordinates on serial sections. Build a co-occurrence
matrix: rows = T-panel compartments, columns = S-panel compartments.

Output: heatmap showing how T and S compartments overlap spatially.
"""

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import Counter


def load_array(f, key):
    ds = f["obs"][key]
    if isinstance(ds, h5py.Group) and "categories" in ds:
        cats = ds["categories"][:]
        codes = ds["codes"][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


EXCLUDE_PAT = ["tonsil", "prostate", "kidney", "spleen", "adrenal", "_ton_", "_adr_"]

def is_control(sid):
    s = sid.lower()
    return any(t in s for t in EXCLUDE_PAT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-utag", required=True)
    parser.add_argument("--s-utag", required=True)
    parser.add_argument("--max-dist", type=float, default=30.0,
                        help="Max distance (px) for cross-panel cell matching")
    parser.add_argument("--output", default="output/hypotheses_v8/cross_panel_compartment_overlap.png")
    args = parser.parse_args()

    print("Loading T-UTAG...")
    ft = h5py.File(args.t_utag, "r")
    t_sid = load_array(ft, "sample_id")
    t_comp = load_array(ft, "compartment_name")
    t_cx = ft["obs"]["centroid_x"][:].astype(float)
    t_cy = ft["obs"]["centroid_y"][:].astype(float)
    ft.close()

    print("Loading S-UTAG...")
    fs = h5py.File(args.s_utag, "r")
    s_sid = load_array(fs, "sample_id")
    s_comp = load_array(fs, "compartment_name")
    s_cx = fs["obs"]["centroid_x"][:].astype(float)
    s_cy = fs["obs"]["centroid_y"][:].astype(float)
    fs.close()

    # Find shared ROIs (excluding controls)
    t_rois = set(np.unique(t_sid))
    s_rois = set(np.unique(s_sid))
    shared = sorted([r for r in t_rois & s_rois if not is_control(r)])
    print(f"Shared tumor ROIs: {len(shared)}")

    # Build co-occurrence matrix
    cooccur = Counter()
    n_matched = 0
    n_total = 0

    for roi in shared:
        tm = t_sid == roi
        sm = s_sid == roi

        tc = t_comp[tm]
        sc = s_comp[sm]
        tx, ty = t_cx[tm], t_cy[tm]
        sx, sy = s_cx[sm], s_cy[sm]

        if len(tx) < 100 or len(sx) < 100:
            continue

        # For each T cell, find nearest S cell
        s_tree = cKDTree(np.column_stack([sx, sy]))
        dists, indices = s_tree.query(np.column_stack([tx, ty]), k=1)

        for i in range(len(tc)):
            n_total += 1
            if dists[i] <= args.max_dist:
                n_matched += 1
                cooccur[(tc[i], sc[indices[i]])] += 1

    print(f"Matched cells: {n_matched:,} / {n_total:,} ({100*n_matched/max(n_total,1):.1f}%)")

    # ── Ordered compartment lists: follicular → interfollicular → other ──
    T_ORDER = [
        # Follicular
        "GC core",
        "Follicle core (GC/CD20hi/CXCR5hi)",
        "Follicle mantle (CXCR5hi)",
        "Activated B / CXCR5hi zone",
        "B cell follicle (CD20hi/CXCR5hi)",
        "B cell zone",
        # Interfollicular
        "Follicle-T zone interface",
        "Treg-enriched T zone",
        "T cell zone (CD4/CD8)",
        "Macrophage-rich zone",
        "Cytotoxic / LQ niche",
        # Other
        "LQ / B transitional",   # displayed as "B transitional zone"
        "Weak CD20 / LQ border", # displayed as "Weak CD20 border"
        "Unidentified zone",
    ]
    S_ORDER = [
        # Follicular
        "B cell zone (BCL2+)",
        "B cell zone (PAX5+)",
        "FDC network zone",
        "FDC / myeloid zone",
        "Mixed (B cells (PAX 27%))",
        # Interfollicular
        "T cell zone",
        "Other / myeloid zone",
        "Mixed (M2 Macrophag 26%)",
        # Other
        "Stromal / CAF zone",
        "B/T mixed zone",
        "Unidentified zone",
    ]

    # Category boundaries for separator lines
    T_BOUNDARIES = [6, 11]   # after follicular, after interfollicular
    S_BOUNDARIES = [5, 8]

    T_LABELS = ["Follicular", "Interfollicular", "Other"]
    S_LABELS = ["Follicular", "Interfollicular", "Other"]

    # Filter to compartments present in data
    all_t = set(k[0] for k in cooccur)
    all_s = set(k[1] for k in cooccur)
    t_comps_all = [c for c in T_ORDER if c in all_t]
    # Append any not in our order list
    t_comps_all += sorted(all_t - set(T_ORDER))
    s_comps_all = [c for c in S_ORDER if c in all_s]
    s_comps_all += sorted(all_s - set(S_ORDER))

    # Recompute boundary positions based on filtered lists
    def get_boundaries(ordered, full_order, raw_boundaries):
        boundaries = []
        for b in raw_boundaries:
            # Find last item from full_order[:b] that's in ordered
            items_before = [c for c in full_order[:b] if c in ordered]
            if items_before:
                boundaries.append(len(items_before))
        return boundaries

    t_bounds = get_boundaries(t_comps_all, T_ORDER, T_BOUNDARIES)
    s_bounds = get_boundaries(s_comps_all, S_ORDER, S_BOUNDARIES)

    # Build matrix
    mat = np.zeros((len(t_comps_all), len(s_comps_all)))
    for (tc, sc), count in cooccur.items():
        if tc in t_comps_all and sc in s_comps_all:
            i = t_comps_all.index(tc)
            j = s_comps_all.index(sc)
            mat[i, j] = count

    # Normalize rows (each T compartment sums to 1)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_norm = mat / row_sums

    # ── Rename display labels: drop "LQ", shorten long names ──
    def clean_label(name):
        name = name.replace("LQ / B transitional", "B transitional zone")
        name = name.replace("Cytotoxic / LQ niche", "Cytotoxic niche")
        name = name.replace("Weak CD20 / LQ border", "Weak CD20 border")
        name = name.replace("Follicle core (GC/CD20hi/CXCR5hi)", "Follicle core (GC/CD20hi)")
        name = name.replace("B cell follicle (CD20hi/CXCR5hi)", "B cell follicle (CD20hi)")
        name = name.replace("Mixed (B cells (PAX 27%))", "Mixed (B cells 27%)")
        name = name.replace("Mixed (M2 Macrophag 26%)", "Mixed (M2 Mac 26%)")
        return name

    t_display = [clean_label(c) for c in t_comps_all]
    s_display = [clean_label(c) for c in s_comps_all]

    # ── Plot (square, manual layout) ──
    n_rows, n_cols = len(t_comps_all), len(s_comps_all)

    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_axes([0.28, 0.15, 0.50, 0.50])
    im = ax.imshow(mat_norm, aspect="equal", cmap="YlOrRd", vmin=0, vmax=0.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(s_display, rotation=45, ha="right", fontsize=16)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(t_display, fontsize=16)

    ax.set_xlabel("S-panel compartment", fontsize=19, labelpad=12)
    ax.set_ylabel("T-panel compartment", fontsize=19, labelpad=8)

    ax_pos = ax.get_position()
    fig.text((ax_pos.x0 + ax_pos.x1) / 2, ax_pos.y1 + 0.10,
             "Cross-panel compartment overlap",
             ha="center", fontsize=22, fontweight="medium")
    fig.text((ax_pos.x0 + ax_pos.x1) / 2, ax_pos.y1 + 0.08,
             "(fraction of T-panel cells mapping to each S-panel compartment)",
             ha="center", fontsize=17, color="#444444")

    # Separator lines
    for b in t_bounds:
        ax.axhline(y=b - 0.5, color="black", linewidth=2)
    for b in s_bounds:
        ax.axvline(x=b - 0.5, color="black", linewidth=2)

    # ── Color bars in figure coordinates ──
    BAR_COLORS = ["#E8A0A0", "#A0A0E8", "#BBBBBB"]
    BAR_W = 0.014

    # Y-axis bar (RIGHT side, between heatmap and colorbar)
    bar_x = ax_pos.x1 + 0.012
    y_starts = [0] + t_bounds
    y_ends = t_bounds + [n_rows]
    for start, end, label, col in zip(y_starts, y_ends, T_LABELS, BAR_COLORS):
        y0 = ax_pos.y0 + ax_pos.height * (1.0 - end / n_rows)
        y1 = ax_pos.y0 + ax_pos.height * (1.0 - start / n_rows)
        fig.patches.append(plt.Rectangle(
            (bar_x, y0), BAR_W, y1 - y0,
            color=col, clip_on=False, transform=fig.transFigure))
        fig.text(bar_x + BAR_W / 2, (y0 + y1) / 2, label, ha="center", va="center",
                 fontsize=14, rotation=90, fontweight="bold", color="#333333")

    # X-axis bar (above axes, with gap)
    bar_y = ax_pos.y1 + 0.012
    x_starts = [0] + s_bounds
    x_ends = s_bounds + [n_cols]
    for start, end, label, col in zip(x_starts, x_ends, S_LABELS, BAR_COLORS):
        x0 = ax_pos.x0 + ax_pos.width * (start / n_cols)
        x1 = ax_pos.x0 + ax_pos.width * (end / n_cols)
        fig.patches.append(plt.Rectangle(
            (x0, bar_y), x1 - x0, BAR_W,
            color=col, clip_on=False, transform=fig.transFigure))
        fig.text((x0 + x1) / 2, bar_y + BAR_W + 0.006, label,
                 ha="center", va="bottom",
                 fontsize=14, fontweight="bold", color="#333333")

    # ── Annotate ALL cells ──
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat_norm[i, j]
            if val < 0.005:
                txt = "<1%"
                color = "#AAAAAA"
                fs = 11
            else:
                txt = f"{val:.2f}"
                color = "white" if val > 0.25 else "black"
                fs = 14
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=fs, color=color)

    # Colorbar (compact, right of domain bar)
    cbar_x = bar_x + BAR_W + 0.012
    cbar_h = ax_pos.height * 0.5
    cbar_y = ax_pos.y0 + (ax_pos.height - cbar_h) / 2
    cbar_ax = fig.add_axes([cbar_x, cbar_y, 0.012, cbar_h])
    fig.colorbar(im, cax=cbar_ax, label="Fraction")

    fig.savefig(args.output, dpi=150, facecolor="white")
    fig.savefig(str(args.output).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output} + PDF")
    plt.close(fig)


if __name__ == "__main__":
    main()
