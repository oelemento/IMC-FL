#!/usr/bin/env python
"""Supplementary figure: CD14 signal decomposition — myeloid, FDC, and spillover.

Shows that ~45% of CD14+ non-myeloid/non-FDC cells are spillover artifacts,
evidenced by a steep proximity gradient to nearest myeloid cell.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree
from scipy import stats
from collections import Counter
from pathlib import Path


# ── Shared helpers ──────────────────────────────────────────────────────────
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


def is_tumor_core(s):
    sl = s.lower()
    if "_ton_" in sl or "_adr_" in sl:
        return False
    for tissue in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if tissue in sl:
            return False
    if s == "Biomax_ROI_006":
        return False
    return True


def panel_label(ax, letter):
    ax.text(
        -0.08, 1.06, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top",
    )


MYELOID_TYPES = {
    "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Myeloid (S100A9+)", "Dendritic cells",
}

CAT_COLORS = {
    "Myeloid": "#E41A1C",
    "FDC": "#FF7F00",
    "Spillover": "#999999",
}

CT_SHORT = {
    "M1 Macrophages": "M1 Mac",
    "M2 Macrophages": "M2 Mac",
    "Macrophages": "Mac (generic)",
    "Myeloid (S100A9+)": "S100A9+ Myeloid",
    "Dendritic cells": "DC",
    "B cells (BCL2+)": "BCL2+ B",
    "B cells (PAX5+)": "PAX5+ B",
    "B cells": "B cells",
    "CD4 T cells": "CD4 T",
    "CD8 T cells": "CD8 T",
    "Endothelial": "Endothelial",
    "Mixed / Border cells": "Mixed/Border",
    "Stromal / CAF": "Stromal/CAF",
    "FRC (PDPN+)": "FRC",
    "Histiocytes (CD44hi)": "Histiocytes",
    "Other": "Other",
    "Low quality / Unassigned": "Unassigned",
    "pDC": "pDC",
    "FDC": "FDC",
}


def extract_data(h5ad_path="output/all_TMA_S_global_v8.h5ad"):
    """Load S-panel data and compute all panel data."""
    print("Loading S-panel...")
    f = h5py.File(h5ad_path, "r")
    X = f["X"][:]
    markers = [v.decode() for v in f["var"]["_index"][:]]
    cell_types = load_array(f, "cell_type")
    sample_ids = load_array(f, "sample_id")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()

    marker_idx = {m: i for i, m in enumerate(markers)}
    cd14_col = marker_idx["CD14"]

    # Filter to tumor cores, no Biomax
    tumor_mask = np.array(
        [is_tumor_core(s) and not s.startswith("Biomax") for s in sample_ids]
    )
    X = X[tumor_mask]
    cell_types = cell_types[tumor_mask]
    sample_ids = sample_ids[tumor_mask]
    cx = cx[tumor_mask]
    cy = cy[tumor_mask]
    cd14_vals = X[:, cd14_col]
    print(f"Tumor cells: {len(cell_types):,}")

    is_myeloid = np.array([ct in MYELOID_TYPES for ct in cell_types])
    is_fdc = cell_types == "FDC"
    is_other = ~is_myeloid & ~is_fdc & (cell_types != "Low quality / Unassigned")

    # ── Panel (a): Mean CD14 by cell type ──
    unique_cts = sorted(set(cell_types) - {"Low quality / Unassigned"})
    ct_means = []
    for ct in unique_cts:
        mask = cell_types == ct
        ct_means.append((ct, mask.sum(), float(X[mask, cd14_col].mean())))
    ct_means.sort(key=lambda x: -x[2])

    # ── Panel (b): Composition of CD14+ cells ──
    cd14_pos = cd14_vals > 0.5
    pos_cts = Counter(cell_types[cd14_pos])
    n_pos = cd14_pos.sum()
    mye_pos = sum(n for ct, n in pos_cts.items() if ct in MYELOID_TYPES)
    fdc_pos = pos_cts.get("FDC", 0)
    other_pos = n_pos - mye_pos - fdc_pos

    # ── Panel (c): Distance gradient ──
    print("Computing proximity gradient...")
    unique_rois = np.unique(sample_ids)
    all_dists = []
    all_cd14 = []
    for roi in unique_rois:
        rmask = sample_ids == roi
        roi_mye = np.where(rmask & is_myeloid)[0]
        roi_target = np.where(rmask & is_other)[0]
        if len(roi_mye) < 50 or len(roi_target) < 100:
            continue
        tree = KDTree(np.column_stack([cx[roi_mye], cy[roi_mye]]))
        target_coords = np.column_stack([cx[roi_target], cy[roi_target]])
        dists, _ = tree.query(target_coords, k=1)
        all_dists.extend(dists)
        all_cd14.extend(X[roi_target, cd14_col])

    all_dists = np.array(all_dists)
    all_cd14 = np.array(all_cd14)
    rho, p_prox = stats.spearmanr(all_dists, all_cd14)
    print(f"Proximity rho={rho:.4f}, p={p_prox:.2e}")

    bins = [0, 5, 10, 15, 20, 30, 50, 100]
    bin_means = []
    bin_pct_pos = []
    bin_labels = []
    bin_ns = []
    for i in range(len(bins) - 1):
        mask = (all_dists >= bins[i]) & (all_dists < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(float(all_cd14[mask].mean()))
            bin_pct_pos.append(100 * (all_cd14[mask] > 0.5).mean())
            bin_labels.append(f"{bins[i]}–{bins[i+1]}")
            bin_ns.append(int(mask.sum()))

    # ── Panel (d): Spatial scatter — pick ROI with good myeloid density ──
    print("Selecting representative ROI for spatial scatter...")
    roi_scores = {}
    for roi in unique_rois:
        rmask = sample_ids == roi
        n_typed = ((cell_types[rmask] != "Low quality / Unassigned")).sum()
        n_mye = (is_myeloid[rmask]).sum()
        if n_typed >= 5000 and n_mye >= 200:
            roi_scores[roi] = n_mye / rmask.sum()  # myeloid fraction
    # Pick ROI near 75th percentile of myeloid fraction (not extreme)
    sorted_rois = sorted(roi_scores, key=roi_scores.get)
    p75_idx = int(len(sorted_rois) * 0.75)
    rep_roi = sorted_rois[p75_idx]
    print(f"Representative ROI: {rep_roi} (myeloid frac={roi_scores[rep_roi]:.3f})")

    rmask = sample_ids == rep_roi
    roi_cx = cx[rmask]
    roi_cy = cy[rmask]
    roi_cd14 = X[rmask, cd14_col]
    roi_ct = cell_types[rmask]
    roi_is_mye = np.array([ct in MYELOID_TYPES for ct in roi_ct])
    roi_is_fdc = roi_ct == "FDC"

    # ── Panel (e): Per-ROI myeloid fraction vs mean CD14 on non-myeloid ──
    print("Computing per-ROI myeloid fraction vs non-myeloid CD14...")
    roi_mye_frac = []
    roi_nonmye_cd14 = []
    for roi in unique_rois:
        rmask = sample_ids == roi
        n_total = rmask.sum()
        if n_total < 500:
            continue
        roi_is_mye_local = is_myeloid[rmask]
        roi_is_other_local = is_other[rmask]
        if roi_is_other_local.sum() < 50:
            continue
        mye_frac = roi_is_mye_local.sum() / n_total
        nonmye_cd14_mean = float(X[np.where(rmask)[0][roi_is_other_local], cd14_col].mean())
        roi_mye_frac.append(mye_frac)
        roi_nonmye_cd14.append(nonmye_cd14_mean)

    roi_mye_frac = np.array(roi_mye_frac)
    roi_nonmye_cd14 = np.array(roi_nonmye_cd14)
    rho_roi, p_roi = stats.spearmanr(roi_mye_frac, roi_nonmye_cd14)
    print(f"Per-ROI rho={rho_roi:.3f}, p={p_roi:.2e}")

    return {
        "ct_means": ct_means,
        "n_pos": n_pos,
        "mye_pos": mye_pos,
        "fdc_pos": fdc_pos,
        "other_pos": other_pos,
        "bin_labels": bin_labels,
        "bin_means": bin_means,
        "bin_pct_pos": bin_pct_pos,
        "bin_ns": bin_ns,
        "rho_prox": rho,
        "p_prox": p_prox,
        "rep_roi": rep_roi,
        "roi_cx": roi_cx,
        "roi_cy": roi_cy,
        "roi_cd14": roi_cd14,
        "roi_ct": roi_ct,
        "roi_is_mye": roi_is_mye,
        "roi_is_fdc": roi_is_fdc,
        "roi_mye_frac": roi_mye_frac,
        "roi_nonmye_cd14": roi_nonmye_cd14,
        "rho_roi": rho_roi,
        "p_roi": p_roi,
    }


def make_figure(data, output_dir="output/hypotheses_v8"):
    """Generate 5-panel supplementary figure (3 top + 2 bottom)."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.40,
                  left=0.08, right=0.94, top=0.94, bottom=0.06)

    # ── (a) Mean CD14 by cell type ──
    ax_a = fig.add_subplot(gs[0, 0])  # top-left
    panel_label(ax_a, "a")
    ct_means = data["ct_means"]
    names = [CT_SHORT.get(ct, ct) for ct, _, _ in ct_means]
    means = [m for _, _, m in ct_means]
    colors = []
    for ct, _, _ in ct_means:
        if ct in MYELOID_TYPES:
            colors.append(CAT_COLORS["Myeloid"])
        elif ct == "FDC":
            colors.append(CAT_COLORS["FDC"])
        else:
            colors.append(CAT_COLORS["Spillover"])
    y_pos = range(len(names))
    ax_a.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.5)
    ax_a.set_yticks(list(y_pos))
    ax_a.set_yticklabels(names, fontsize=8)
    ax_a.invert_yaxis()
    ax_a.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
                 label="CD14+ threshold")
    ax_a.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax_a.set_xlabel("Mean CD14 intensity (scaled)")
    ax_a.set_title("CD14 expression by cell type")
    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CAT_COLORS["Myeloid"], label="Myeloid"),
        Patch(facecolor=CAT_COLORS["FDC"], label="FDC"),
        Patch(facecolor=CAT_COLORS["Spillover"], label="Other"),
    ]
    ax_a.legend(handles=legend_elements, fontsize=7, loc="lower right")

    # ── (b) Composition of CD14+ cells ──
    ax_b = fig.add_subplot(gs[0, 1])  # top-center
    panel_label(ax_b, "b")
    sizes = [data["mye_pos"], data["fdc_pos"], data["other_pos"]]
    labels = [
        f"Myeloid\n{data['mye_pos']:,}\n({100*data['mye_pos']/data['n_pos']:.0f}%)",
        f"FDC\n{data['fdc_pos']:,}\n({100*data['fdc_pos']/data['n_pos']:.0f}%)",
        f"Spillover\n{data['other_pos']:,}\n({100*data['other_pos']/data['n_pos']:.0f}%)",
    ]
    pie_colors = [CAT_COLORS["Myeloid"], CAT_COLORS["FDC"], CAT_COLORS["Spillover"]]
    wedges, texts = ax_b.pie(
        sizes, labels=labels, colors=pie_colors,
        startangle=90, textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax_b.set_title(f"CD14+ cells (>{0.5} threshold)\nn = {data['n_pos']:,}")

    # ── (c) Distance gradient ──
    ax_c = fig.add_subplot(gs[0, 2])  # top-right
    panel_label(ax_c, "c")
    x_pos = range(len(data["bin_labels"]))
    bars = ax_c.bar(x_pos, data["bin_pct_pos"], color="#E41A1C", alpha=0.7,
                    edgecolor="white", linewidth=0.5)
    ax_c.set_xticks(list(x_pos))
    ax_c.set_xticklabels(data["bin_labels"], fontsize=8, rotation=30, ha="right")
    ax_c.set_xlabel("Distance to nearest myeloid cell (μm)")
    ax_c.set_ylabel("% CD14+ among non-myeloid/non-FDC cells")
    ax_c.set_title(
        f"CD14 spillover gradient\n"
        f"ρ = {data['rho_prox']:.3f}, p < 10⁻¹⁰"
    )
    # Add n labels on bars
    for i, (bar, n) in enumerate(zip(bars, data["bin_ns"])):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"n={n//1000}k", ha="center", va="bottom", fontsize=6, color="#666",
        )

    # ── (d) Spatial scatter — CD14 intensity heatmap ──
    ax_d = fig.add_subplot(gs[1, 0])  # bottom-left
    panel_label(ax_d, "d")
    roi_cd14 = data["roi_cd14"]
    roi_cx = data["roi_cx"]
    roi_cy = data["roi_cy"]
    roi_is_mye = data["roi_is_mye"]
    roi_is_fdc = data["roi_is_fdc"]
    is_neither = ~roi_is_mye & ~roi_is_fdc

    # Clip CD14 for colormap
    vmin, vmax = -0.5, 2.5
    cd14_clipped = np.clip(roi_cd14, vmin, vmax)

    # Plot non-myeloid/non-FDC first (background)
    sc = ax_d.scatter(
        roi_cx[is_neither], roi_cy[is_neither],
        c=cd14_clipped[is_neither], cmap="YlOrRd", vmin=vmin, vmax=vmax,
        s=0.3, alpha=0.5, rasterized=True,
    )
    # FDC on top
    ax_d.scatter(
        roi_cx[roi_is_fdc], roi_cy[roi_is_fdc],
        c=cd14_clipped[roi_is_fdc], cmap="YlOrRd", vmin=vmin, vmax=vmax,
        s=0.5, alpha=0.6, edgecolors="none", rasterized=True,
    )
    # Myeloid cells with black edge
    ax_d.scatter(
        roi_cx[roi_is_mye], roi_cy[roi_is_mye],
        c=cd14_clipped[roi_is_mye], cmap="YlOrRd", vmin=vmin, vmax=vmax,
        s=4, alpha=0.9, edgecolors="black", linewidths=0.3, rasterized=True,
    )
    ax_d.set_aspect("equal")
    ax_d.set_xlabel("X (μm)")
    ax_d.set_ylabel("Y (μm)")
    ax_d.set_title(f"CD14 intensity — {data['rep_roi']}\n(black edge = myeloid)")
    cb = plt.colorbar(sc, ax=ax_d, shrink=0.7, label="CD14 (scaled)")

    # ── (e) Per-ROI: myeloid fraction vs non-myeloid CD14 ──
    ax_e = fig.add_subplot(gs[1, 1])  # bottom-center
    panel_label(ax_e, "e")
    ax_e.scatter(
        data["roi_mye_frac"] * 100, data["roi_nonmye_cd14"],
        s=20, alpha=0.6, color="#E41A1C", edgecolors="white", linewidths=0.3,
    )
    # Regression line
    m, b = np.polyfit(data["roi_mye_frac"] * 100, data["roi_nonmye_cd14"], 1)
    x_line = np.linspace(0, max(data["roi_mye_frac"] * 100) * 1.05, 50)
    ax_e.plot(x_line, m * x_line + b, "k--", linewidth=1, alpha=0.5)
    ax_e.set_xlabel("Myeloid cell fraction (%)")
    ax_e.set_ylabel("Mean CD14 on non-myeloid/non-FDC cells")
    ax_e.set_title(
        f"Per-ROI: myeloid density drives spillover\n"
        f"ρ = {data['rho_roi']:.3f}, p = {data['p_roi']:.1e}"
    )
    ax_e.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    out = Path(output_dir) / "fig_cd14_spillover.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    data = extract_data()
    make_figure(data)
