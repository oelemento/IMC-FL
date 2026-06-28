#!/usr/bin/env python3
"""High-zoom inspection of one H&E ↔ IMC pair for sanity-checking concordance.

Renders:
- H&E centered crop (full core)
- IMC composite (CD21=green, CD14=orange, CD68=magenta, CD8=cyan)
- IMC cell-type scatter (centroids colored by cell_type)
- IMC CD21 channel only (so we can see exactly where the FDC network is)

Usage:
    .venv/bin/python scripts/he_imc_zoom.py --slide-id B1_FL45
"""
import argparse
import os
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

CHANNELS = {
    "CD21": "CD21(Er170Di)",
    "CD14": "CD14(Nd148Di)",
    "CD68": "CD68(Tb159Di)",
    "CD8":  "CD8a(Dy162Di)",
}
COLORS = {
    "CD21": np.array([0, 1, 0]),
    "CD14": np.array([1, 0.3, 0]),
    "CD68": np.array([1, 0, 1]),
    "CD8":  np.array([0, 0.8, 1]),
}
MPP_FALLBACK = 0.25
S_UTAG = "output/all_TMA_S_utag_ct_merged.h5ad"

CT_COLORS = {
    "CD14+ FDC": "#FF1493", "CD14- FDC": "#FFB6C1",
    "PAX5+ B": "#1f77b4", "BCL2+ B": "#17becf",
    "M1 Macrophages": "#d62728", "M2 Macrophages": "#9467bd",
    "Myeloid (S100A9+)": "#ff7f0e", "S100A9+": "#ff7f0e",
    "DC (CD11c+HLA-DR+)": "#bcbd22", "pDC": "#e377c2",
    "Endothelial": "#8c564b", "Fibroblast": "#7f7f7f",
    "PDPN+ FRC": "#aec7e8", "Vasc-perivasc": "#c5b0d5",
    "Other stromal": "#c7c7c7", "Unassigned": "#D3D3D3",
}


def load_he(svs_path, fov_um=1500.0, target_long=900):
    s = openslide.OpenSlide(str(svs_path))
    w, h = s.dimensions
    mpp = float(s.properties.get("openslide.mpp-x", "nan"))
    if not np.isfinite(mpp):
        mpp = MPP_FALLBACK
    fov_px = min(int(round(fov_um / mpp)), w, h)
    x0 = (w - fov_px) // 2
    y0 = (h - fov_px) // 2
    factor = fov_px / target_long
    level = s.get_best_level_for_downsample(factor)
    ds = s.level_downsamples[level]
    out_size = (int(fov_px / ds), int(fov_px / ds))
    img = s.read_region((x0, y0), level, out_size).convert("RGB")
    return np.array(img), fov_px * mpp, mpp


def load_raw_channel(txt_path, marker_col):
    df = pd.read_csv(txt_path, sep="\t", usecols=["X", "Y", marker_col])
    x = df["X"].astype(int).values
    y = df["Y"].astype(int).values
    img = np.zeros((y.max() + 1, x.max() + 1), dtype=np.float32)
    img[y, x] = df[marker_col].values
    return img


def find_raw(roi_id, raw_dir):
    fl = roi_id.split("_", 1)[1]
    pat = re.compile(rf"_{re.escape(fl)}_")
    for f in os.listdir(raw_dir):
        if pat.search(f) and f.endswith(".txt"):
            return Path(raw_dir) / f
    return None


def build_imc(raw_file):
    imgs = {n: np.arcsinh(load_raw_channel(str(raw_file), c) / 5)
            for n, c in CHANNELS.items()}
    h, w = imgs["CD21"].shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for n, color in COLORS.items():
        img = imgs[n]
        vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1
        norm = np.clip(img / max(vmax, 1e-6), 0, 1)
        rgb += norm[:, :, np.newaxis] * color
    return np.clip(rgb, 0, 1), imgs["CD21"], (w, h)


def load_cells_for_roi(slide_id, h5ad=S_UTAG):
    with h5py.File(h5ad, "r") as f:
        sid = np.array([s.decode() if isinstance(s, bytes) else s
                        for s in f["obs/sample_id/categories"][:]
                        ])[f["obs/sample_id/codes"][:]]
        ct_cats = [c.decode() if isinstance(c, bytes) else c
                   for c in f["obs/cell_type/categories"][:]]
        ct = np.array(ct_cats)[f["obs/cell_type/codes"][:]]
        cx = f["obs/centroid_x"][:]
        cy = f["obs/centroid_y"][:]
    mask = sid == slide_id
    return ct[mask], cx[mask], cy[mask]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slide-id", default="B1_FL45")
    p.add_argument("--he-dir", default="data/he_samples")
    p.add_argument("--raw-dir", default="data/raw/TMA_B1_S")
    p.add_argument("--fov-um", type=float, default=1500.0)
    p.add_argument("--out", default="output/he_imc_sidebyside")
    args = p.parse_args()

    slide = args.slide_id
    cl = pd.read_csv("data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    sample_id = cl[cl["slide_ID"] == slide]["Sample_ID"].iloc[0]

    # Find H&E svs by core number
    core_n = int(slide.split("FL")[1])
    tma = "B" if slide.startswith("B") else "A"
    pre = "CT14-01_B1_10" if tma == "B" else "CT14-01_A1_11"
    he_path = Path(args.he_dir) / f"{pre}.{core_n}.svs"
    print(f"[H&E]  {he_path}")
    he, he_um, he_mpp = load_he(he_path, fov_um=args.fov_um)
    print(f"  shape={he.shape}, native mpp={he_mpp:.3f}, FOV={he_um:.0f} µm")

    raw = find_raw(slide, args.raw_dir)
    print(f"[IMC]  raw={raw}")
    imc_rgb, cd21_img, (iw, ih) = build_imc(raw)
    print(f"  shape=({ih}, {iw})")

    print(f"[Cells] {S_UTAG}")
    ct, cx, cy = load_cells_for_roi(slide)
    print(f"  {len(ct)} cells in {slide}")
    counts = pd.Series(ct).value_counts()
    print("  top cell types:")
    for k, v in counts.head(8).items():
        print(f"    {k:25s} {v}")

    # Render
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))

    axes[0, 0].imshow(he)
    axes[0, 0].set_title(f"H&E — {slide} ({sample_id})\n{he.shape[1]}×{he.shape[0]} px @ ~{he_mpp:.2f} µm/px,"
                         f" FOV={he_um:.0f} µm",
                         fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(imc_rgb)
    axes[0, 1].set_title(f"IMC composite — {slide}\n{iw}×{ih} µm  |  CD21=g, CD14=o, CD68=m, CD8=cyan",
                         fontsize=11)
    axes[0, 1].axis("off")

    # CD21 only (so we KNOW where the FDC network actually is)
    cd21_norm = np.clip(np.arcsinh(load_raw_channel(str(raw), CHANNELS["CD21"]) / 5)
                        / max(np.percentile(cd21_img[cd21_img > 0], 99), 1e-6), 0, 1)
    axes[1, 0].imshow(cd21_norm, cmap="Greens")
    axes[1, 0].set_title(f"CD21 channel only (FDC network) — {slide}", fontsize=11)
    axes[1, 0].axis("off")

    # Cell-type scatter
    ax = axes[1, 1]
    # Plot Unassigned first as background
    is_un = ct == "Unassigned"
    ax.scatter(cx[is_un], cy[is_un], c="#D3D3D3", s=0.5, alpha=0.4, zorder=1)
    for ctype in counts.index:
        if ctype == "Unassigned":
            continue
        m = ct == ctype
        if not m.any():
            continue
        col = CT_COLORS.get(ctype, "#444444")
        ax.scatter(cx[m], cy[m], c=col, s=2.5, alpha=0.85, zorder=2,
                   label=f"{ctype} (n={m.sum()})")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"Cell-type scatter — {slide}", fontsize=11)
    ax.legend(loc="upper right", fontsize=7, frameon=True, markerscale=2)
    ax.set_xlabel("centroid_x (µm)")
    ax.set_ylabel("centroid_y (µm)")

    plt.tight_layout()
    out = Path(args.out) / f"{slide}_he_imc_zoom.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
