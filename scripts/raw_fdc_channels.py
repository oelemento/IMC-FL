#!/usr/bin/env python3
"""Show raw IMC channel images for CD14-high FDC neighborhoods.

Displays CD21, CD14, CD68, CD8a (and optionally VISTA) as individual channels
plus a composite overlay, cropped to a specific window within an ROI.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import h5py

sys.path.insert(0, str(Path(__file__).parent))
from fig_fdc_intrafollicular import is_tumor_core, load_array
from src.clinical_linkage import EXCLUDE_ROIS

BASE = Path(__file__).parent.parent
RAW_DIR = BASE / "data" / "raw" / "TMA_B1_S"

# Channel names in the raw TXT file header
CHANNELS = {
    "CD21": "CD21(Er170Di)",
    "CD14": "CD14(Nd148Di)",
    "CD68": "CD68(Tb159Di)",
    "CD8":  "CD8a(Dy162Di)",
    "VISTA": "VISTA(Gd160Di)",
    "CD20": "CD20(Dy161Di)",
    "DNA":  "DNA1(Ir191Di)",
}

# Colors for composite: CD21=green, CD14=red, CD68=magenta, CD8=cyan
COMPOSITE_COLORS = {
    "CD21": np.array([0, 1, 0]),       # green
    "CD14": np.array([1, 0.3, 0]),     # orange-red
    "CD68": np.array([1, 0, 1]),       # magenta
    "CD8":  np.array([0, 0.8, 1]),     # cyan
}

# Single-channel colormaps: black→color (fluorescence style)
def _make_cmap(color):
    """Black-to-color linear colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("", ["black", color])

CHANNEL_CMAPS = {
    "CD21": _make_cmap("#00FF00"),    # green
    "CD14": _make_cmap("#FF8C00"),    # orange
    "CD68": _make_cmap("#FF00FF"),    # magenta
    "CD8":  _make_cmap("#00DDFF"),    # cyan
    "VISTA": _make_cmap("#FF3333"),   # red
    "CD20": _make_cmap("#6699FF"),    # blue
}


def load_raw_image(txt_path, marker_col):
    """Load raw TXT and reconstruct pixel image for a marker."""
    df = pd.read_csv(txt_path, sep="\t", usecols=["X", "Y", marker_col])
    x = df["X"].values.astype(int)
    y = df["Y"].values.astype(int)
    vals = df[marker_col].values
    img = np.zeros((y.max() + 1, x.max() + 1), dtype=np.float32)
    img[y, x] = vals
    return img


def find_raw_file(roi_id, raw_dir):
    """Find raw TXT file for a B1 ROI."""
    import re
    fl_part = roi_id.replace("B1_", "")
    pattern = re.compile(rf"_{re.escape(fl_part)}_")
    for fname in os.listdir(raw_dir):
        if pattern.search(fname) and fname.endswith(".txt"):
            return raw_dir / fname
    return None


def load_cell_data(h5ad_path, roi_id):
    """Load cell centroids and types for an ROI."""
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
    return {
        "ct": ct[mask], "cx": cx[mask], "cy": cy[mask], "cd14": cd14[mask],
    }


def plot_channels_and_cells(raw_file, cells, center_x, center_y,
                            window_size, output_path):
    """Multi-panel: individual channels + composite + cell overlay."""

    half = window_size / 2
    # Pixel coordinates for crop (assuming 1 pixel = 1 µm for Hyperion)
    x0, x1 = int(center_x - half), int(center_x + half)
    y0, y1 = int(center_y - half), int(center_y + half)

    # Load channels
    print("  Loading raw channels...")
    channels_to_show = ["CD21", "CD14", "CD68", "CD8"]
    imgs = {}
    for name in channels_to_show + ["DNA"]:
        col = CHANNELS[name]
        img = load_raw_image(str(raw_file), col)
        # Crop
        img_crop = img[max(0, y0):min(img.shape[0], y1),
                       max(0, x0):min(img.shape[1], x1)]
        # Arcsinh transform for display
        imgs[name] = np.arcsinh(img_crop / 5)

    # Also load CD20 for context
    cd20_full = load_raw_image(str(raw_file), CHANNELS["CD20"])
    imgs["CD20"] = np.arcsinh(cd20_full[max(0, y0):min(cd20_full.shape[0], y1),
                                        max(0, x0):min(cd20_full.shape[1], x1)] / 5)

    # Cell data in window
    cx, cy, ct, cd14 = cells["cx"], cells["cy"], cells["ct"], cells["cd14"]
    in_win = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
    wcx = cx[in_win] - x0  # relative to crop
    wcy = cy[in_win] - y0
    wct = ct[in_win]
    wcd14 = cd14[in_win]

    # --- Build figure ---
    # Row 1: CD21, CD14, CD68, CD8 (individual channels)
    # Row 2: CD20 (B cells), composite, cell type overlay
    fig, axes = plt.subplots(2, 4, figsize=(24, 13))

    for i, name in enumerate(channels_to_show):
        ax = axes[0, i]
        img = imgs[name]
        vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1
        ax.imshow(img, cmap=CHANNEL_CMAPS[name], vmin=0, vmax=vmax, origin="upper")
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Row 2, col 0: CD20
    ax = axes[1, 0]
    vmax = np.percentile(imgs["CD20"][imgs["CD20"] > 0], 99) if np.any(imgs["CD20"] > 0) else 1
    ax.imshow(imgs["CD20"], cmap="Blues", vmin=0, vmax=vmax, origin="upper")
    ax.set_title("CD20 (B cells)", fontsize=14, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2, col 1: DNA
    ax = axes[1, 1]
    vmax = np.percentile(imgs["DNA"][imgs["DNA"] > 0], 99) if np.any(imgs["DNA"] > 0) else 1
    ax.imshow(imgs["DNA"], cmap="gray", vmin=0, vmax=vmax, origin="upper")
    ax.set_title("DNA (tissue)", fontsize=14, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2, col 2: Composite overlay
    ax = axes[1, 2]
    h, w = imgs["CD21"].shape
    composite = np.zeros((h, w, 3), dtype=np.float32)
    for name, color in COMPOSITE_COLORS.items():
        img = imgs[name]
        vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1
        norm = np.clip(img / max(vmax, 1e-6), 0, 1)
        composite += norm[:, :, np.newaxis] * color[np.newaxis, np.newaxis, :]
    composite = np.clip(composite, 0, 1)
    ax.imshow(composite, origin="upper")
    ax.set_title("Composite\n(CD21=grn, CD14=org, CD68=mag, CD8=cyn)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2, col 3: Cell type overlay on DNA
    ax = axes[1, 3]
    ax.imshow(imgs["DNA"], cmap="gray", vmin=0,
              vmax=np.percentile(imgs["DNA"][imgs["DNA"] > 0], 99) if np.any(imgs["DNA"] > 0) else 1,
              alpha=0.3, origin="upper")

    # Plot cells by type
    is_fdc = wct == "FDC"
    cd14_q75 = np.percentile(cd14[ct == "FDC"], 75) if (ct == "FDC").sum() > 0 else 1.0
    fdc_hi = is_fdc & (wcd14 >= cd14_q75)
    fdc_lo = is_fdc & (wcd14 < cd14_q75)
    mac = np.isin(wct, ["M1 Macrophages", "M2 Macrophages", "Macrophages"])
    cd8 = wct == "CD8 T cells"
    bcell = np.isin(wct, ["B cells (BCL2+)", "B cells (PAX5+)", "B cells"])
    other = ~(is_fdc | mac | cd8 | bcell)

    ax.scatter(wcx[other], wcy[other], c="#D3D3D3", s=8, alpha=0.3, zorder=1)
    if bcell.sum() > 0:
        ax.scatter(wcx[bcell], wcy[bcell], c="#4393C3", s=15, alpha=0.5,
                   label=f"B cells ({bcell.sum()})", zorder=2)
    if cd8.sum() > 0:
        ax.scatter(wcx[cd8], wcy[cd8], c="#00BCD4", s=30, alpha=0.8,
                   edgecolors="black", linewidth=0.3, marker="^",
                   label=f"CD8 T ({cd8.sum()})", zorder=4)
    if mac.sum() > 0:
        ax.scatter(wcx[mac], wcy[mac], c="#E41A1C", s=30, alpha=0.8,
                   edgecolors="black", linewidth=0.3, marker="s",
                   label=f"Mac ({mac.sum()})", zorder=4)
    if fdc_lo.sum() > 0:
        ax.scatter(wcx[fdc_lo], wcy[fdc_lo], c="#FDDBC7", s=20, alpha=0.5,
                   edgecolors="gray", linewidth=0.2,
                   label=f"FDC lo ({fdc_lo.sum()})", zorder=3)
    if fdc_hi.sum() > 0:
        ax.scatter(wcx[fdc_hi], wcy[fdc_hi], c="#FFD700", s=50, alpha=0.95,
                   edgecolors="black", linewidth=0.5, marker="*",
                   label=f"FDC hi ({fdc_hi.sum()})", zorder=5)

    ax.set_title("Cell types", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", markerscale=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, w); ax.set_ylim(h, 0)

    fig.suptitle(f"B1_FL8 — raw IMC channels ({window_size}×{window_size} µm window)\n"
                 f"center=({center_x:.0f}, {center_y:.0f})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    # Best windows from zoom script
    parser.add_argument("--cx", type=float, default=1012, help="Window center x")
    parser.add_argument("--cy", type=float, default=993, help="Window center y")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--roi", default="B1_FL8")
    args = parser.parse_args()

    raw_file = find_raw_file(args.roi, RAW_DIR)
    if not raw_file:
        print(f"ERROR: No raw file found for {args.roi} in {RAW_DIR}")
        sys.exit(1)
    print(f"Raw file: {raw_file.name}")

    print("Loading cell data...")
    cells = load_cell_data(args.s_utag, args.roi)
    print(f"  {len(cells['ct']):,} cells in {args.roi}")

    out = Path(args.output_dir) / f"raw_channels_{args.roi}_w{args.window}.png"
    plot_channels_and_cells(raw_file, cells, args.cx, args.cy,
                            args.window, out)

    # Also do window 3 (more mixed)
    print("\n--- Window 3 (mixed) ---")
    out2 = Path(args.output_dir) / f"raw_channels_{args.roi}_w3_{args.window}.png"
    plot_channels_and_cells(raw_file, cells, 1115, 927,
                            args.window, out2)

    print("\nDone.")
