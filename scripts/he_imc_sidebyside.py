#!/usr/bin/env python3
"""Side-by-side H&E + raw IMC composite for the same TMA core (feasibility check).

Usage:
    .venv/bin/python scripts/he_imc_sidebyside.py \
        --he data/he_samples/CT14-01_B1_10.8.svs \
        --roi B1_FL8 \
        --raw-dir data/raw/TMA_B1_S
"""
import argparse
import os
import re
from pathlib import Path

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


def load_raw_channel(txt_path, marker_col):
    df = pd.read_csv(txt_path, sep="\t", usecols=["X", "Y", marker_col])
    x = df["X"].astype(int).values
    y = df["Y"].astype(int).values
    img = np.zeros((y.max() + 1, x.max() + 1), dtype=np.float32)
    img[y, x] = df[marker_col].values
    return img


def find_raw_file(roi_id, raw_dir):
    fl_part = roi_id.split("_", 1)[1]
    pat = re.compile(rf"_{re.escape(fl_part)}_")
    for fname in os.listdir(raw_dir):
        if pat.search(fname) and fname.endswith(".txt"):
            return Path(raw_dir) / fname
    return None


def build_imc_composite(raw_file):
    imgs = {n: np.arcsinh(load_raw_channel(str(raw_file), c) / 5) for n, c in CHANNELS.items()}
    h, w = imgs["CD21"].shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for n, color in COLORS.items():
        img = imgs[n]
        vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1
        norm = np.clip(img / max(vmax, 1e-6), 0, 1)
        rgb += norm[:, :, np.newaxis] * color
    return np.clip(rgb, 0, 1)


def load_he_thumbnail(svs_path, target_long_edge=1500, crop_um=2000.0, mpp_fallback=0.25):
    """Read SVS, crop a centered window of `crop_um` µm × `crop_um` µm, and downsample.

    The CT14 SVS files lack mpp-x metadata; we assume 0.5 µm/pixel (20x scan) by default.
    """
    slide = openslide.OpenSlide(str(svs_path))
    w, h = slide.dimensions
    mpp = float(slide.properties.get("openslide.mpp-x", "nan"))
    if not np.isfinite(mpp):
        mpp = mpp_fallback
    crop_px = int(round(crop_um / mpp))
    crop_px = min(crop_px, w, h)
    x0 = (w - crop_px) // 2
    y0 = (h - crop_px) // 2
    factor = crop_px / target_long_edge
    level = slide.get_best_level_for_downsample(factor)
    ds = slide.level_downsamples[level]
    out_size = (int(crop_px / ds), int(crop_px / ds))
    img = slide.read_region((x0, y0), level, out_size).convert("RGB")
    return np.array(img), mpp, ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--he", required=True, help="Path to H&E SVS")
    p.add_argument("--roi", required=True, help="e.g. B1_FL8")
    p.add_argument("--raw-dir", required=True, help="Directory of raw IMC .txt files")
    p.add_argument("--out", default="output/he_imc_sidebyside")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading H&E thumbnail from {args.he} ...")
    he, he_mpp, he_ds = load_he_thumbnail(args.he, target_long_edge=1500)
    print(f"  H&E shape={he.shape}, mpp={he_mpp:.3f}, downsample={he_ds:.1f}")

    print(f"Finding raw IMC file for {args.roi} in {args.raw_dir} ...")
    raw = find_raw_file(args.roi, args.raw_dir)
    if raw is None:
        raise SystemExit(f"No raw IMC file for {args.roi} found in {args.raw_dir}")
    print(f"  -> {raw.name}")
    imc = build_imc_composite(raw)
    print(f"  IMC composite shape={imc.shape} (1 px = 1 µm)")

    # Side-by-side: matched display sizes
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(he)
    axes[0].set_title(f"H&E ({Path(args.he).name})\n{he.shape[1]}×{he.shape[0]} px @ ~{he_mpp*he_ds:.2f} µm/px",
                      fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(imc)
    axes[1].set_title(f"IMC composite ({args.roi})\n{imc.shape[1]}×{imc.shape[0]} µm "
                      "(CD21=green, CD14=orange, CD68=magenta, CD8=cyan)",
                      fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    out = out_dir / f"{args.roi}_he_imc_sidebyside.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
