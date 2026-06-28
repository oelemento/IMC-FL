#!/usr/bin/env python3
"""Full-core cell type scatter of B1_FL8 with composite IMC inset.

Left panel:  Full ROI cell scatter (FDCs by CD14 level, Mac, CD8, B cells)
             with a rectangle marking the zoom window.
Right panel: Raw IMC composite (CD21=green, CD14=red, CD68=magenta, CD8=cyan; CD21+CD14 overlap appears yellow)
             cropped to the zoom window.
Lines connect the rectangle to the inset.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

import h5py

sys.path.insert(0, str(Path(__file__).parent))
from fig_fdc_intrafollicular import load_array

BASE = Path(__file__).parent.parent
RAW_DIR = BASE / "data" / "raw" / "TMA_B1_S"

CHANNELS = {
    "CD21": "CD21(Er170Di)",
    "CD14": "CD14(Nd148Di)",
    "CD68": "CD68(Tb159Di)",
    "CD8":  "CD8a(Dy162Di)",
}

COMPOSITE_COLORS = {
    "CD21": np.array([0, 1, 0]),       # green
    "CD14": np.array([1, 0.3, 0]),     # orange
    "CD68": np.array([1, 0, 1]),       # magenta
    "CD8":  np.array([0, 0.8, 1]),     # cyan
}


def load_raw_image(txt_path, marker_col):
    df = pd.read_csv(txt_path, sep="\t", usecols=["X", "Y", marker_col])
    x = df["X"].values.astype(int)
    y = df["Y"].values.astype(int)
    vals = df[marker_col].values
    img = np.zeros((y.max() + 1, x.max() + 1), dtype=np.float32)
    img[y, x] = vals
    return img


def find_raw_file(roi_id, raw_dir):
    import re
    fl_part = roi_id.replace("B1_", "")
    pattern = re.compile(rf"_{re.escape(fl_part)}_")
    for fname in os.listdir(raw_dir):
        if pattern.search(fname) and fname.endswith(".txt"):
            return raw_dir / fname
    return None


def build_composite(raw_file, x0, y0, x1, y1):
    """Build RGB composite from raw IMC channels, cropped to window."""
    imgs = {}
    for name, col in CHANNELS.items():
        img = load_raw_image(str(raw_file), col)
        crop = img[max(0, y0):min(img.shape[0], y1),
                   max(0, x0):min(img.shape[1], x1)]
        imgs[name] = np.arcsinh(crop / 5)

    h, w = imgs["CD21"].shape
    composite = np.zeros((h, w, 3), dtype=np.float32)
    for name, color in COMPOSITE_COLORS.items():
        img = imgs[name]
        vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1
        norm = np.clip(img / max(vmax, 1e-6), 0, 1)
        composite += norm[:, :, np.newaxis] * color[np.newaxis, np.newaxis, :]
    return np.clip(composite, 0, 1)


def load_cells(h5ad_path, roi_id):
    with h5py.File(h5ad_path, "r") as f:
        sid = load_array(f, "sample_id")
        ct = load_array(f, "cell_type")
        cx = f["obs"]["centroid_x"][:]
        cy = f["obs"]["centroid_y"][:]
        var_names = [v.decode() if isinstance(v, bytes) else v
                     for v in f["var"]["_index"][:]]
        cd14_idx = var_names.index("CD14")
        cd14 = np.array(f["X"][:, cd14_idx]).flatten()
    mask = sid == roi_id
    return ct[mask], cx[mask], cy[mask], cd14[mask]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    parser.add_argument("--roi", default="B1_FL8")
    parser.add_argument("--cx", type=float, default=1115)
    parser.add_argument("--cy", type=float, default=927)
    parser.add_argument("--window", type=int, default=250)
    args = parser.parse_args()

    roi = args.roi
    win_cx, win_cy = args.cx, args.cy
    win_sz = args.window
    half = win_sz / 2

    # --- Load data ---
    raw_file = find_raw_file(roi, RAW_DIR)
    if not raw_file:
        print(f"ERROR: No raw file for {roi}")
        sys.exit(1)
    print(f"Raw file: {raw_file.name}")

    print("Loading cell data...")
    ct, cx, cy, cd14 = load_cells(args.s_utag, roi)
    print(f"  {len(ct):,} cells")

    # Global CD14 threshold (across all FDCs in the dataset)
    with h5py.File(args.s_utag, "r") as f:
        all_ct = load_array(f, "cell_type")
        var_names = [v.decode() if isinstance(v, bytes) else v
                     for v in f["var"]["_index"][:]]
        cd14_idx = var_names.index("CD14")
        all_cd14 = np.array(f["X"][:, cd14_idx]).flatten()
    cd14_q75 = np.percentile(all_cd14[all_ct == "FDC"], 75)

    print("Loading raw IMC channels for composite...")
    x0, y0 = int(win_cx - half), int(win_cy - half)
    x1, y1 = int(win_cx + half), int(win_cy + half)
    composite = build_composite(raw_file, x0, y0, x1, y1)

    # --- Build figure ---
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1],
                           wspace=0.08, left=0.03, right=0.97,
                           top=0.92, bottom=0.03)

    # === Left panel: full ROI cell scatter ===
    ax_roi = fig.add_subplot(gs[0, 0])

    is_fdc = ct == "FDC"
    fdc_hi = is_fdc & (cd14 >= cd14_q75)
    fdc_lo = is_fdc & ~fdc_hi
    mac = np.isin(ct, ["M1 Macrophages", "M2 Macrophages", "Macrophages"])
    cd8 = ct == "CD8 T cells"
    bcell = np.isin(ct, ["B cells (BCL2+)", "B cells (PAX5+)", "B cells"])
    other = ~(is_fdc | mac | cd8 | bcell)

    # Plot layers
    ax_roi.scatter(cx[other], cy[other], c="#E0E0E0", s=0.5, alpha=0.2,
                   rasterized=True, zorder=1)
    ax_roi.scatter(cx[bcell], cy[bcell], c="#4393C3", s=2, alpha=0.4,
                   label=f"B cells ({bcell.sum():,})", rasterized=True, zorder=2)
    ax_roi.scatter(cx[fdc_lo], cy[fdc_lo], c="#FDDBC7", s=3, alpha=0.4,
                   edgecolors="gray", linewidth=0.1,
                   label=f"FDC CD14-low ({fdc_lo.sum():,})", rasterized=True, zorder=3)
    ax_roi.scatter(cx[cd8], cy[cd8], c="#00BCD4", s=5, alpha=0.6,
                   edgecolors="black", linewidth=0.1,
                   label=f"CD8 T ({cd8.sum():,})", rasterized=True, zorder=4)
    ax_roi.scatter(cx[mac], cy[mac], c="#E41A1C", s=5, alpha=0.6,
                   edgecolors="black", linewidth=0.1,
                   label=f"Macrophages ({mac.sum():,})", rasterized=True, zorder=4)
    ax_roi.scatter(cx[fdc_hi], cy[fdc_hi], c="#FFD700", s=10, alpha=0.85,
                   edgecolors="black", linewidth=0.3,
                   label=f"FDC CD14-high ({fdc_hi.sum():,})", rasterized=True, zorder=5)

    # Draw zoom rectangle
    rect = Rectangle((x0, y0), win_sz, win_sz,
                      linewidth=2.5, edgecolor="white", facecolor="none",
                      linestyle="-", zorder=10)
    ax_roi.add_patch(rect)
    # Second rectangle for contrast on light backgrounds
    rect2 = Rectangle((x0, y0), win_sz, win_sz,
                       linewidth=1.5, edgecolor="black", facecolor="none",
                       linestyle="--", zorder=10)
    ax_roi.add_patch(rect2)

    ax_roi.set_aspect("equal")
    ax_roi.invert_yaxis()
    ax_roi.set_title(f"{roi} — segmented cell types", fontsize=13, fontweight="bold")
    ax_roi.legend(fontsize=7, loc="upper left", markerscale=2.5, framealpha=0.9)
    ax_roi.set_xlabel("x (µm)", fontsize=10)
    ax_roi.set_ylabel("y (µm)", fontsize=10)
    ax_roi.text(-0.02, 1.02, "(a)", transform=ax_roi.transAxes,
                fontsize=14, fontweight="bold", va="bottom")

    # === Right panel: composite IMC inset ===
    ax_comp = fig.add_subplot(gs[0, 1])
    ax_comp.imshow(composite, origin="upper", extent=[x0, x1, y1, y0])
    ax_comp.set_title(f"Raw IMC composite ({win_sz}×{win_sz} µm)",
                      fontsize=13, fontweight="bold")
    ax_comp.set_xlabel("x (µm)", fontsize=10)
    ax_comp.set_xticks([]); ax_comp.set_yticks([])

    # Legend for composite colors
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#00FF00", label="CD21 (FDC network)"),
        Patch(facecolor="#FF8C00", label="CD14"),
        Patch(facecolor="#FF00FF", label="CD68 (macrophages)"),
        Patch(facecolor="#00DDFF", label="CD8 (T cells)"),
    ]
    ax_comp.legend(handles=legend_items, fontsize=8, loc="upper left",
                   framealpha=0.9, facecolor="black", labelcolor="white",
                   edgecolor="white")
    ax_comp.text(-0.02, 1.02, "(b)", transform=ax_comp.transAxes,
                 fontsize=14, fontweight="bold", va="bottom")

    # --- Draw connector lines from rectangle corners to inset edges ---
    from matplotlib.patches import ConnectionPatch

    for y_corner in [y0, y1]:
        con = ConnectionPatch(
            xyA=(x1, y_corner), coordsA=ax_roi.transData,
            xyB=(x0, y_corner), coordsB=ax_comp.transData,
            color="black", linewidth=1.0, linestyle=":", alpha=0.6, zorder=10)
        fig.add_artist(con)

    out = Path(args.output_dir) / f"fig_fdc_composite_inset_{roi}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
