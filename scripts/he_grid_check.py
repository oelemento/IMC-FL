#!/usr/bin/env python3
"""Render all 6 pulled H&E cores in a grid with TMA registry labels.

If `.8`, `.33`, `.34`, `.45` etc are truly different cores, the 6 cores
should show visually distinct lymphoid architecture (different follicle
counts, sizes, positions). If they all look identical, the filename
mapping is wrong.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

MPP_FALLBACK = 0.25
CROP_UM = 2000.0


def load_centered_thumbnail(svs_path, target_long_edge=900):
    s = openslide.OpenSlide(str(svs_path))
    w, h = s.dimensions
    mpp = float(s.properties.get("openslide.mpp-x", "nan"))
    if not np.isfinite(mpp):
        mpp = MPP_FALLBACK
    crop_px = min(int(round(CROP_UM / mpp)), w, h)
    x0 = (w - crop_px) // 2
    y0 = (h - crop_px) // 2
    factor = crop_px / target_long_edge
    level = s.get_best_level_for_downsample(factor)
    ds = s.level_downsamples[level]
    out_size = (int(crop_px / ds), int(crop_px / ds))
    return np.array(s.read_region((x0, y0), level, out_size).convert("RGB"))


def main():
    samples_dir = Path("data/he_samples")
    files = sorted([f for f in os.listdir(samples_dir) if f.endswith(".svs")])

    # Build registry mapping
    reg = pd.read_excel("data/he_samples/metadata/BCCA_xtrFL_TMA_ID.xlsx")
    cl = pd.read_csv("data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, fname in zip(axes.flat, files):
        # Parse: CT14-01_<TMA>_<sec>.<core>.svs
        # e.g. CT14-01_B1_10.8.svs -> TMA=B, core=8
        stem = fname.replace(".svs", "")
        tma = "B" if "_B1_" in fname else "A"
        core_n = int(stem.rsplit(".", 1)[1])
        # Lookup registry
        reg_match = reg[(reg["Slides"] == tma) & (reg["Core #"] == core_n)]
        sample_id = reg_match.iloc[0]["Other ID"] if len(reg_match) else "?"
        # Lookup clinical for transformation status
        cl_match = cl[cl["Sample_ID"] == sample_id]
        trans = cl_match.iloc[0]["Transformation"] if len(cl_match) else None
        trans_str = " (transformed!)" if trans == "Yes" else ""
        slide_id = cl_match.iloc[0]["slide_ID"] if len(cl_match) else "?"

        img = load_centered_thumbnail(samples_dir / fname)
        ax.imshow(img)
        ax.set_title(f"{slide_id} = {sample_id}{trans_str}\n{fname}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out = Path("output/he_imc_sidebyside/he_grid_check.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
