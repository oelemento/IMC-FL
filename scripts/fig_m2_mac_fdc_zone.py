"""
Figure S10: M2 Macrophages in the FDC network zone.

Representative ROI (C1_FL12) showing M2 Mac spatial distribution within the
FDC network zone. Left: cell types colored by identity (FDC zone highlighted,
non-FDC zone gray), M2 Mac as red stars, with zoom inset on densest M2 niche.
Right: CD21 signal validates FDC network zone boundaries.
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22


# ---------------------------------------------------------------------------
# Shared ROI plot constants — must match fig_macrophage_biology.py panel (e)
# ---------------------------------------------------------------------------

CELL_COLORS = {
    "FDC": "#2ecc71", "B cells (BCL2+)": "#85c1e9",
    "B cells (PAX5+)": "#5dade2", "CD8 T cells": "#8e44ad",
    "CD4 T cells": "#3498db", "M1 Macrophages": "#e67e22",
    "M2 Macrophages": "#e74c3c", "Myeloid (S100A9+)": "#f39c12",
    "Macrophages": "#d35400", "Dendritic cells": "#1abc9c",
}
HIGHLIGHT_TYPES = list(CELL_COLORS.keys())

# Standard cell sizes for ROI scatter plots (points²).
# These values produce consistent dot sizes across Fig 6e and S10.
SZ_ROI = 3          # base size for non-myeloid typed cells
SZ_MYELOID = 7      # non-M2 myeloid (M1, S100A9+, Mac)
SZ_M2_STAR = 18     # M2 Mac stars
SZ_GRAY = 1.5       # untyped / non-FDC-zone background
SZ_INS = 30         # base size in zoom inset
SZ_INS_MYELOID = 75
SZ_INS_M2 = 150


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


def is_tumor(s):
    sl = s.lower()
    for t in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if t in sl:
            return False
    if "_ton_" in sl or "_adr_" in sl:
        return False
    if s.startswith("Biomax"):
        return False
    return True


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"$\\bf{{{letter}}}$", transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE, va="bottom", ha="left")


def scatter_roi_celltypes(ax, rx, ry, rct, fdc_zone, sz_roi, sz_myeloid,
                          sz_m2, sz_gray):
    """Plot cell types in an ROI with FDC zone highlighted."""
    # Non-FDC zone: light gray
    non_fdc = ~fdc_zone
    ax.scatter(rx[non_fdc], ry[non_fdc], c="#D3D3D3", s=sz_gray,
               alpha=0.2, zorder=0, rasterized=True)
    # FDC zone untyped
    fdc_other = fdc_zone & ~np.isin(rct, HIGHLIGHT_TYPES)
    ax.scatter(rx[fdc_other], ry[fdc_other], c="#D3D3D3", s=sz_gray,
               alpha=0.3, zorder=0, rasterized=True)
    # Non-myeloid typed cells in FDC zone
    for ctype in ["FDC", "B cells (BCL2+)", "B cells (PAX5+)",
                  "CD8 T cells", "CD4 T cells"]:
        mask = fdc_zone & (rct == ctype)
        if mask.any():
            ax.scatter(rx[mask], ry[mask], c=CELL_COLORS[ctype],
                       s=sz_roi, alpha=0.5, zorder=1, rasterized=True)
    # Myeloid (non-M2) in FDC zone
    for ctype in ["Macrophages", "Myeloid (S100A9+)", "M1 Macrophages"]:
        mask = fdc_zone & (rct == ctype)
        if mask.any():
            ax.scatter(rx[mask], ry[mask], c=CELL_COLORS[ctype],
                       s=sz_myeloid, alpha=0.7, zorder=2, rasterized=True)
    # M2 Mac stars (FDC zone)
    m2_fdc = fdc_zone & (rct == "M2 Macrophages")
    ax.scatter(rx[m2_fdc], ry[m2_fdc], c="#e74c3c", s=sz_m2,
               alpha=0.9, zorder=3, marker="*", edgecolors="black",
               linewidths=0.3, rasterized=True)
    # M2 Mac outside FDC zone (dimmer)
    m2_other = ~fdc_zone & (rct == "M2 Macrophages")
    if m2_other.any():
        ax.scatter(rx[m2_other], ry[m2_other], c="#e74c3c", s=sz_m2 * 0.5,
                   alpha=0.3, zorder=1, marker="*", rasterized=True)
    return int(m2_fdc.sum())


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_data(s_panel_path, s_utag_path):
    """Load ROI data for C1_FL12."""
    print("Loading UTAG compartment data...")
    f = h5py.File(s_utag_path, "r")
    ct = load_array(f, "cell_type")
    comp = load_array(f, "compartment_name")
    sids = load_array(f, "sample_id")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()

    print("Loading S-panel expression data...")
    f2 = h5py.File(s_panel_path, "r")
    X = f2["X"][:]
    markers = [v.decode() for v in f2["var"]["_index"][:]]
    f2.close()
    marker_idx = {m: i for i, m in enumerate(markers)}

    # Filter to tumor cores
    tumor = np.array([is_tumor(s) and s not in EXCLUDE_ROIS for s in sids])
    ct = ct[tumor]; comp = comp[tumor]; sids = sids[tumor]
    cx = cx[tumor]; cy = cy[tumor]; X = X[tumor]
    print(f"  {len(ct):,} tumor cells")

    # C1_FL12: representative ROI (46 M2 in FDC zone, 28% FDC)
    best_roi = "C1_FL12"
    print(f"  Using ROI: {best_roi}")
    rmask = sids == best_roi
    cd21_idx = marker_idx["CD21"]
    n_m2_fdc = int(((ct[rmask] == "M2 Macrophages") &
                     (comp[rmask] == "FDC network zone")).sum())
    print(f"    Total cells: {rmask.sum():,}, "
          f"FDC zone: {(comp[rmask] == 'FDC network zone').sum():,}, "
          f"M2 in FDC zone: {n_m2_fdc}")

    return {
        "name": best_roi,
        "cx": cx[rmask], "cy": cy[rmask],
        "ct": ct[rmask], "comp": comp[rmask],
        "cd21": X[rmask, cd21_idx],
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(data, output_dir):
    """Single-row figure: cell types (left) + CD21 signal (right)."""
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, figure=fig, wspace=0.15,
                  left=0.05, right=0.97, top=0.93, bottom=0.05)

    rx, ry, rct, rcomp = data["cx"], data["cy"], data["ct"], data["comp"]
    fdc_zone = rcomp == "FDC network zone"

    # ── Left: cell types in FDC zone ──
    ax_a1 = fig.add_subplot(gs[0, 0])
    panel_label(ax_a1, "a")

    n_m2_fdc = scatter_roi_celltypes(ax_a1, rx, ry, rct, fdc_zone,
                                      SZ_ROI, SZ_MYELOID, SZ_M2_STAR, SZ_GRAY)

    ax_a1.set_aspect("equal")
    ax_a1.invert_yaxis()
    ax_a1.set_title(f"Cell types in FDC zone — {data['name']}", fontsize=TITLE_SIZE,
                    fontweight="medium")
    ax_a1.set_xlabel("x (\u00b5m)", fontsize=LABEL_SIZE)
    ax_a1.set_ylabel("y (\u00b5m)", fontsize=LABEL_SIZE)
    ax_a1.tick_params(labelsize=TICK_SIZE)

    # Legend
    leg_items = [("FDC", "o"), ("BCL2+ B", "o"), ("CD8 T", "o"),
                 ("CD4 T", "o"), ("M1 Mac", "o"),
                 (f"M2 Mac (n={n_m2_fdc})", "*")]
    leg_colors = ["#2ecc71", "#85c1e9", "#8e44ad", "#3498db", "#e67e22",
                  "#e74c3c"]
    leg_handles = []
    for (lab, mk), col in zip(leg_items, leg_colors):
        ec = "black" if mk == "*" else "none"
        leg_handles.append(Line2D([0], [0], marker=mk, color="w",
                                   markerfacecolor=col, markersize=8,
                                   markeredgecolor=ec, label=lab))
    leg_handles.append(Patch(facecolor="#D3D3D3", alpha=0.5, label="Non-FDC zone"))
    ax_a1.legend(handles=leg_handles, loc="upper right", fontsize=LEGEND_SIZE,
                 framealpha=0.9)

    # Zoom inset on M2 niche
    zoom_cx, zoom_cy, zoom_half = 699, 320, 100
    rect = Rectangle((zoom_cx - zoom_half, zoom_cy - zoom_half),
                      2 * zoom_half, 2 * zoom_half,
                      linewidth=2, edgecolor="white", facecolor="none", zorder=10)
    ax_a1.add_patch(rect)
    rect2 = Rectangle((zoom_cx - zoom_half, zoom_cy - zoom_half),
                       2 * zoom_half, 2 * zoom_half,
                       linewidth=1.5, edgecolor="black", facecolor="none",
                       linestyle="--", zorder=11)
    ax_a1.add_patch(rect2)

    # Inset axes (bottom-left)
    ax_ins = ax_a1.inset_axes([0.0, 0.02, 0.44, 0.44])
    in_zoom = ((rx >= zoom_cx - zoom_half) & (rx <= zoom_cx + zoom_half) &
               (ry >= zoom_cy - zoom_half) & (ry <= zoom_cy + zoom_half))
    zx, zy, zct = rx[in_zoom], ry[in_zoom], rct[in_zoom]
    zcomp = rcomp[in_zoom]
    z_fdc = zcomp == "FDC network zone"

    scatter_roi_celltypes(ax_ins, zx, zy, zct, z_fdc,
                          SZ_INS, SZ_INS_MYELOID, SZ_INS_M2, SZ_INS * 0.5)

    ax_ins.set_xlim(zoom_cx - zoom_half, zoom_cx + zoom_half)
    ax_ins.set_ylim(zoom_cy + zoom_half, zoom_cy - zoom_half)
    ax_ins.set_aspect("equal")
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    for sp in ax_ins.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)

    # Connection lines
    con1 = ConnectionPatch(
        xyA=(zoom_cx - zoom_half, zoom_cy + zoom_half), coordsA=ax_a1.transData,
        xyB=(1, 0), coordsB=ax_ins.transAxes,
        color="black", linewidth=1, linestyle="--", alpha=0.5)
    fig.add_artist(con1)
    con2 = ConnectionPatch(
        xyA=(zoom_cx - zoom_half, zoom_cy - zoom_half), coordsA=ax_a1.transData,
        xyB=(1, 1), coordsB=ax_ins.transAxes,
        color="black", linewidth=1, linestyle="--", alpha=0.5)
    fig.add_artist(con2)

    # ── Right: CD21 signal ──
    ax_a2 = fig.add_subplot(gs[0, 1])
    panel_label(ax_a2, "b")

    cd21 = data["cd21"]
    vmin, vmax = np.percentile(cd21, 2), np.percentile(cd21, 98)
    sc = ax_a2.scatter(rx, ry, c=cd21, cmap="inferno", s=SZ_ROI, alpha=0.7,
                       vmin=vmin, vmax=vmax, rasterized=True, zorder=0)
    cb = fig.colorbar(sc, ax=ax_a2, fraction=0.03, pad=0.02, shrink=0.7)
    cb.set_label("CD21", fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    ax_a2.set_aspect("equal")
    ax_a2.invert_yaxis()
    ax_a2.set_title(f"CD21 signal — {data['name']}", fontsize=TITLE_SIZE,
                    fontweight="medium")
    ax_a2.set_xlabel("x (\u00b5m)", fontsize=LABEL_SIZE)
    ax_a2.set_ylabel("")
    ax_a2.tick_params(axis="y", labelleft=False)
    ax_a2.tick_params(labelsize=TICK_SIZE)

    # Save
    out_path = Path(output_dir) / "fig_m2_mac_fdc_zone.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {out_path} + PDF")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Figure S10: M2 Macrophages in the FDC network zone"
    )
    parser.add_argument("--s-panel", required=True,
                        help="S-panel expression h5ad")
    parser.add_argument("--s-utag", required=True,
                        help="S-panel UTAG compartment h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8",
                        help="Output directory for figure")
    args = parser.parse_args()

    data = extract_data(args.s_panel, args.s_utag)
    out_path = make_figure(data, args.output_dir)

    import subprocess
    subprocess.run(["open", "-a", "Preview", out_path])


if __name__ == "__main__":
    main()
