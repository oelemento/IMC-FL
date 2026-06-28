"""Visualize FDC segmentation spillover on a representative ROI.

Shows that cells annotated as FDC sit in CD20+ B cell zones, and that their
'B-marker positivity' comes from segmentation picking up signal from neighbors.

Output: output/fdc_validation/fdc_spillover_visual_B1_FL8.png
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.patches import Rectangle

ROOT = Path("/Users/ole2001/PROGRAMS/IMC-FL")
TXT = ROOT / "data/raw/TMA_B1_S/20210518_CT14_09_B1_Stromalpanel_2_FL8_L_1.txt"
OUT = ROOT / "output/fdc_validation/fdc_spillover_visual_B1_FL8.png"
SAMPLE_ID_GUESSES = ["B1_FL8", "FL8", "B1_Stromalpanel_FL8"]


def reconstruct_channels(txt_path: Path, channels: list[str]) -> dict:
    """Load pixel-level Hyperion data and return 2D arrays for given channels."""
    df = pd.read_csv(txt_path, sep="\t")
    # Find matching columns
    col_map = {}
    for target in channels:
        for c in df.columns:
            if c.startswith(target + "(") or c == target:
                col_map[target] = c
                break
    X = df["X"].values.astype(int)
    Y = df["Y"].values.astype(int)
    W = X.max() + 1
    H = Y.max() + 1
    out = {"width": W, "height": H}
    for target, col in col_map.items():
        img = np.zeros((H, W), dtype=np.float32)
        img[Y, X] = df[col].values
        out[target] = img
    return out


def find_sample_in_h5ad(a, guesses):
    sids = a.obs["sample_id"].astype(str).unique()
    for g in guesses:
        matches = [s for s in sids if g in s]
        if matches:
            return matches[0]
    # Fallback: print all B1 samples
    b1 = [s for s in sids if "B1" in s and ("FL8" in s or "FL08" in s)]
    if b1:
        return b1[0]
    return None


def main():
    print("Loading pixel data...")
    imgs = reconstruct_channels(TXT, ["CD21", "CD20", "PAX5", "BCL_2"])
    W, H = imgs["width"], imgs["height"]
    print(f"  ROI shape: {H}x{W}")

    print("Loading h5ad for cell centroids...")
    a = ad.read_h5ad(str(ROOT / "output/all_TMA_S_global_v8.h5ad"))
    sample = find_sample_in_h5ad(a, SAMPLE_ID_GUESSES)
    print(f"  sample: {sample}")
    if sample is None:
        print("ERROR: could not match sample_id in h5ad. Available:")
        print([s for s in a.obs["sample_id"].unique() if "B1" in s][:20])
        return

    mask = (a.obs["sample_id"].astype(str) == sample).values
    print(f"  cells in sample: {mask.sum()}")
    xy = np.column_stack([a.obs.loc[mask, "centroid_x"].values,
                           a.obs.loc[mask, "centroid_y"].values])
    ct = a.obs.loc[mask, "cell_type"].astype(str).values

    # Get raw intensities for each cell in this sample
    raw = a.raw.X
    raw_vars = list(a.raw.var.index)
    def getraw(name):
        i = raw_vars.index(name)
        col = raw[:, i]
        arr = np.array(col.todense()).flatten() if sp.issparse(col) else np.array(col).flatten()
        return arr[mask]
    cd21_cell = getraw("CD21")
    cd20_cell = getraw("CD20")

    fdc_mask = ct == "FDC"
    is_b = np.array([c.startswith("B cells") for c in ct])

    # Compose figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.15)

    def show_channel(ax, img, name, vmax_percentile=99, cmap="magma"):
        vmax = max(1e-6, np.percentile(img, vmax_percentile))
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, origin="upper")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Row 1: channel maps
    ax1 = fig.add_subplot(gs[0, 0]); show_channel(ax1, imgs["CD21"], "CD21 raw (FDC network)", cmap="Greens")
    ax2 = fig.add_subplot(gs[0, 1]); show_channel(ax2, imgs["CD20"], "CD20 raw (B cells)", cmap="Blues")
    ax3 = fig.add_subplot(gs[0, 2])
    # RGB merge: CD20 blue, CD21 green, PAX5 red
    def norm(img, p=99):
        v = max(1e-6, np.percentile(img, p)); return np.clip(img / v, 0, 1)
    rgb = np.stack([norm(imgs["PAX5"]), norm(imgs["CD21"]), norm(imgs["CD20"])], axis=-1)
    ax3.imshow(rgb, origin="upper")
    ax3.set_title("Merge: CD21 (G) | CD20 (B) | PAX5 (R)", fontsize=12, fontweight="bold")
    ax3.set_xticks([]); ax3.set_yticks([])

    # Overlay cell centroids on ax2 (CD20)
    ax2.scatter(xy[fdc_mask, 0], xy[fdc_mask, 1], s=8, facecolors="none",
                edgecolors="#00ff99", linewidths=0.6, label=f"FDC (n={fdc_mask.sum()})")

    # Find a nice zoom region: max-density FDC patch
    if fdc_mask.sum() > 0:
        fx, fy = xy[fdc_mask].T
        # Use median of highest-CD20 10% FDCs (proof of spillover region)
        top_spill = np.argsort(-cd20_cell[fdc_mask])[: max(1, fdc_mask.sum() // 20)]
        cx = int(np.median(fx[top_spill]))
        cy = int(np.median(fy[top_spill]))
        half = 150
        x0, x1 = max(0, cx - half), min(W, cx + half)
        y0, y1 = max(0, cy - half), min(H, cy + half)

        # Draw zoom rectangle on ax3
        ax3.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                fill=False, edgecolor="yellow", linewidth=2))

        # Row 2: zoom-ins
        def zoom(img, cmap):
            return img[y0:y1, x0:x1]

        ax4 = fig.add_subplot(gs[1, 0])
        show_channel(ax4, zoom(imgs["CD21"], "Greens"), "ZOOM CD21", cmap="Greens")
        # Overlay FDCs + B cells
        m_in = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
        xy_z = xy[m_in].copy(); xy_z[:, 0] -= x0; xy_z[:, 1] -= y0
        fdc_z = fdc_mask[m_in]
        b_z = is_b[m_in]
        ax4.scatter(xy_z[fdc_z, 0], xy_z[fdc_z, 1], s=30, facecolors="none",
                    edgecolors="#ffff00", linewidths=1.2)
        ax4.scatter(xy_z[b_z, 0], xy_z[b_z, 1], s=8, c="#66ccff", alpha=0.5)

        ax5 = fig.add_subplot(gs[1, 1])
        show_channel(ax5, zoom(imgs["CD20"], "Blues"), "ZOOM CD20", cmap="Blues")
        ax5.scatter(xy_z[fdc_z, 0], xy_z[fdc_z, 1], s=30, facecolors="none",
                    edgecolors="#ffff00", linewidths=1.2, label="FDC centroid")
        ax5.scatter(xy_z[b_z, 0], xy_z[b_z, 1], s=8, c="#66ccff", alpha=0.5, label="B cell centroid")
        ax5.legend(loc="upper right", fontsize=9, frameon=True, facecolor="black",
                   labelcolor="white")

        ax6 = fig.add_subplot(gs[1, 2])
        show_channel(ax6, zoom(imgs["PAX5"], "Reds"), "ZOOM PAX5 (B cell nuclei)", cmap="Reds")
        ax6.scatter(xy_z[fdc_z, 0], xy_z[fdc_z, 1], s=30, facecolors="none",
                    edgecolors="#ffff00", linewidths=1.2)

    fig.suptitle(
        f"FDC segmentation spillover — {sample} (B1 S-panel)\n"
        "Yellow circles = FDC centroids; note how they sit in CD20+/PAX5+ dense zones. "
        "Segmentation captures neighboring B cell cytoplasm, producing apparent PAX5/BCL2 positivity.",
        fontsize=12, fontweight="bold", y=1.00,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
