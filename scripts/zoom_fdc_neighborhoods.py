#!/usr/bin/env python3
"""Zoom into CD14-high FDC neighborhoods showing myeloid + CD8 co-localization.

For each candidate ROI, finds 200×200 µm windows centered on CD14-high FDC
clusters that have macrophages and CD8 T cells within the window.
"""

import sys
import numpy as np
from pathlib import Path

import h5py
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from fig_fdc_intrafollicular import is_tumor_core, load_array
from src.clinical_linkage import EXCLUDE_ROIS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_s_panel(path):
    with h5py.File(path, "r") as f:
        sid = load_array(f, "sample_id")
        ct = load_array(f, "cell_type")
        cx = f["obs"]["centroid_x"][:]
        cy = f["obs"]["centroid_y"][:]
        var_names = [v.decode() if isinstance(v, bytes) else v
                     for v in f["var"]["_index"][:]]
        cd14_idx = var_names.index("CD14")
        cd14 = np.array(f["X"][:, cd14_idx]).flatten()
    mask = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS for s in sid])
    return {
        "sid": sid[mask], "ct": ct[mask],
        "cx": cx[mask], "cy": cy[mask], "cd14": cd14[mask],
    }


def find_best_windows(data, roi, window_size=200, top_n=3):
    """Find windows with highest density of CD14-high FDC + mac + CD8 trios."""
    rm = data["sid"] == roi
    cx, cy, ct, cd14 = data["cx"][rm], data["cy"][rm], data["ct"][rm], data["cd14"][rm]

    is_fdc = ct == "FDC"
    cd14_q75 = np.percentile(data["cd14"][data["ct"] == "FDC"], 75)
    is_cd14hi = is_fdc & (cd14 >= cd14_q75)
    is_mac = np.isin(ct, ["M1 Macrophages", "M2 Macrophages", "Macrophages"])
    is_cd8 = ct == "CD8 T cells"
    is_bcell = np.isin(ct, ["B cells (BCL2+)", "B cells (PAX5+)", "B cells"])

    # For each CD14-high FDC, count mac + CD8 within window_size/2
    hi_idx = np.where(is_cd14hi)[0]
    half = window_size / 2
    scores = []

    for i in hi_idx:
        xc, yc = cx[i], cy[i]
        in_win = ((cx >= xc - half) & (cx <= xc + half) &
                  (cy >= yc - half) & (cy <= yc + half))
        n_hi = (in_win & is_cd14hi).sum()
        n_mac = (in_win & is_mac).sum()
        n_cd8 = (in_win & is_cd8).sum()
        n_bcell = (in_win & is_bcell).sum()
        # Want: multiple CD14-high FDCs, some mac, some CD8, not too many B cells
        if n_hi >= 5 and n_mac >= 3 and n_cd8 >= 3:
            score = n_hi + n_mac + n_cd8 - 0.5 * n_bcell
            scores.append((score, xc, yc, n_hi, n_mac, n_cd8, n_bcell))

    scores.sort(key=lambda x: -x[0])

    # Deduplicate: skip windows that overlap >50% with a better one
    selected = []
    for s in scores:
        _, xc, yc = s[0], s[1], s[2]
        overlap = False
        for prev in selected:
            if abs(xc - prev[1]) < half and abs(yc - prev[2]) < half:
                overlap = True
                break
        if not overlap:
            selected.append(s)
        if len(selected) >= top_n:
            break

    return selected


def plot_zoom(data, roi, center_x, center_y, window_size, output_path, win_idx):
    """Plot zoomed spatial scatter for one window."""
    rm = data["sid"] == roi
    cx, cy, ct, cd14 = data["cx"][rm], data["cy"][rm], data["ct"][rm], data["cd14"][rm]

    half = window_size / 2
    in_win = ((cx >= center_x - half) & (cx <= center_x + half) &
              (cy >= center_y - half) & (cy <= center_y + half))

    wx, wy, wct, wcd14 = cx[in_win], cy[in_win], ct[in_win], cd14[in_win]

    is_fdc = wct == "FDC"
    cd14_q75 = np.percentile(data["cd14"][data["ct"] == "FDC"], 75)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Background: all other cells
    other = ~(is_fdc | np.isin(wct, [
        "M1 Macrophages", "M2 Macrophages", "Macrophages",
        "CD8 T cells", "B cells (BCL2+)", "B cells (PAX5+)", "B cells"]))
    ax.scatter(wx[other], wy[other], c="#E0E0E0", s=15, alpha=0.4,
               rasterized=True, zorder=1)

    # B cells
    b_mask = np.isin(wct, ["B cells (BCL2+)", "B cells (PAX5+)", "B cells"])
    if b_mask.sum() > 0:
        ax.scatter(wx[b_mask], wy[b_mask], c="#4393C3", s=30, alpha=0.6,
                   edgecolors="white", linewidth=0.3,
                   label=f"B cells ({b_mask.sum()})", rasterized=True, zorder=2)

    # CD8 T cells
    cd8_mask = wct == "CD8 T cells"
    if cd8_mask.sum() > 0:
        ax.scatter(wx[cd8_mask], wy[cd8_mask], c="#00BCD4", s=50, alpha=0.8,
                   edgecolors="black", linewidth=0.5, marker="^",
                   label=f"CD8 T ({cd8_mask.sum()})", zorder=5)

    # Macrophages
    mac_mask = np.isin(wct, ["M1 Macrophages", "M2 Macrophages", "Macrophages"])
    if mac_mask.sum() > 0:
        ax.scatter(wx[mac_mask], wy[mac_mask], c="#E41A1C", s=50, alpha=0.8,
                   edgecolors="black", linewidth=0.5, marker="s",
                   label=f"Macrophages ({mac_mask.sum()})", zorder=5)

    # FDC CD14-low
    fdc_lo = is_fdc & (wcd14 < cd14_q75)
    if fdc_lo.sum() > 0:
        ax.scatter(wx[fdc_lo], wy[fdc_lo], c="#FDDBC7", s=40, alpha=0.6,
                   edgecolors="gray", linewidth=0.3,
                   label=f"FDC CD14-low ({fdc_lo.sum()})", rasterized=True, zorder=3)

    # FDC CD14-high — prominent
    fdc_hi = is_fdc & (wcd14 >= cd14_q75)
    if fdc_hi.sum() > 0:
        ax.scatter(wx[fdc_hi], wy[fdc_hi], c="#FFD700", s=80, alpha=0.95,
                   edgecolors="black", linewidth=0.8, marker="*",
                   label=f"FDC CD14-high ({fdc_hi.sum()})", zorder=6)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(center_y + half, center_y - half)
    ax.set_title(f"{roi} — window {win_idx} ({window_size}×{window_size} µm)\n"
                 f"center=({center_x:.0f}, {center_y:.0f})", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", markerscale=1.2)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    parser.add_argument("--rois", default="B1_FL8,C1_FL19,C1_FL10")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of windows per ROI")
    args = parser.parse_args()

    print("Loading S-panel UTAG...")
    data = load_s_panel(args.s_utag)

    rois = args.rois.split(",")
    out_dir = Path(args.output_dir)

    for roi in rois:
        print(f"\n--- {roi} ---")
        windows = find_best_windows(data, roi, args.window, args.top_n)
        if not windows:
            print("  No qualifying windows found")
            continue

        for i, (score, xc, yc, n_hi, n_mac, n_cd8, n_b) in enumerate(windows):
            print(f"  Window {i+1}: center=({xc:.0f},{yc:.0f}) "
                  f"CD14hi={n_hi} Mac={n_mac} CD8={n_cd8} B={n_b} score={score:.0f}")
            out = out_dir / f"zoom_{roi}_w{i+1}.png"
            plot_zoom(data, roi, xc, yc, args.window, out, i+1)
            print(f"    → {out}")

    print("\nDone.")
