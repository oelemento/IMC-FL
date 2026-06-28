#!/usr/bin/env python3
"""Multi-pair H&E ↔ IMC composite side-by-side grid.

For every core listed in PAIRS, render an H&E thumbnail (centered crop) next
to the raw IMC composite at matched physical scale. Rows = cores; left = H&E,
right = IMC.

Usage:
    .venv/bin/python scripts/he_imc_grid.py
"""
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

# Local helpers (re-implemented to avoid argparse-at-import-time in he_imc_register.py)
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
HE_FOV_UM = 1500.0  # match approximate IMC core size


def load_he(svs_path):
    s = openslide.OpenSlide(str(svs_path))
    w, h = s.dimensions
    mpp = float(s.properties.get("openslide.mpp-x", "nan"))
    if not np.isfinite(mpp):
        mpp = MPP_FALLBACK
    fov_px = min(int(round(HE_FOV_UM / mpp)), w, h)
    x0 = (w - fov_px) // 2
    y0 = (h - fov_px) // 2
    target = 900
    factor = fov_px / target
    level = s.get_best_level_for_downsample(factor)
    ds = s.level_downsamples[level]
    out_size = (int(fov_px / ds), int(fov_px / ds))
    img = s.read_region((x0, y0), level, out_size).convert("RGB")
    return np.array(img), fov_px * mpp  # actual µm


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
    return np.clip(rgb, 0, 1), (w, h)


PAIRS = [
    {"slide_id": "B1_FL8",  "he": "data/he_samples/CT14-01_B1_10.8.svs",  "raw_dir": "data/raw/TMA_B1_S"},
    {"slide_id": "B1_FL33", "he": "data/he_samples/CT14-01_B1_10.33.svs", "raw_dir": "data/raw/TMA_B1_S"},
    {"slide_id": "B1_FL34", "he": "data/he_samples/CT14-01_B1_10.34.svs", "raw_dir": "data/raw/TMA_B1_S"},
    {"slide_id": "B1_FL45", "he": "data/he_samples/CT14-01_B1_10.45.svs", "raw_dir": "data/raw/TMA_B1_S"},
]


def main():
    cl = pd.read_csv("data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    n = len(PAIRS)
    fig, axes = plt.subplots(n, 2, figsize=(11, 5 * n))
    for row, p in enumerate(PAIRS):
        slide = p["slide_id"]
        sample = cl[cl["slide_ID"] == slide]["Sample_ID"]
        sample_id = sample.iloc[0] if len(sample) else "?"
        print(f"[{slide}] sample={sample_id}")

        he_img, he_um = load_he(p["he"])
        raw = find_raw(slide, p["raw_dir"])
        imc_rgb, (iw, ih) = build_imc(raw) if raw else (None, (0, 0))

        ax_he = axes[row, 0]
        ax_he.imshow(he_img)
        ax_he.set_title(f"H&E — {slide} (sample {sample_id})\n"
                        f"~{he_um:.0f} µm centered crop  |  {Path(p['he']).name}",
                        fontsize=10)
        ax_he.axis("off")

        ax_imc = axes[row, 1]
        if imc_rgb is not None:
            ax_imc.imshow(imc_rgb)
            ax_imc.set_title(f"IMC composite — {slide}\n"
                             f"{iw}×{ih} µm  |  CD21=g, CD14=o, CD68=m, CD8=cyan",
                             fontsize=10)
        else:
            ax_imc.text(0.5, 0.5, "(no raw IMC found)",
                        ha="center", va="center", transform=ax_imc.transAxes)
        ax_imc.axis("off")

    plt.tight_layout()
    out = Path("output/he_imc_sidebyside/he_imc_pairs_grid.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
