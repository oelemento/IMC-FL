#!/usr/bin/env python3
"""Co-register H&E (Aperio SVS) to IMC composite via mutual information.

Pipeline:
1. Load H&E from SVS at low magnification (~2 µm/px), crop a window centered on
   the core. Convert to grayscale (intensity inverted so tissue is bright).
2. Build an IMC composite (CD21+CD14+CD68+CD8 sum) at 1 µm/px from raw .txt.
   Make grayscale by summing channels.
3. Resample H&E to the same physical resolution and field-of-view as the IMC
   image (using SimpleITK at the IMC resolution = 1 µm/px).
4. Run rigid (Euler2D) registration with Mattes mutual information.
5. Apply the transform to the H&E thumbnail (in RGB) and save side-by-side
   plus a checkerboard overlay.

Usage:
    .venv/bin/python scripts/he_imc_register.py \
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
import SimpleITK as sitk

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
MPP_FALLBACK = 0.25  # CT14 SVS lacks mpp metadata; assume 40x scan

# ────────── IMC ──────────


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
    imgs = {n: np.arcsinh(load_raw_channel(str(raw_file), c) / 5)
            for n, c in CHANNELS.items()}
    h, w = imgs["CD21"].shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    gray = np.zeros((h, w), dtype=np.float32)
    for n, color in COLORS.items():
        img = imgs[n]
        vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1
        norm = np.clip(img / max(vmax, 1e-6), 0, 1)
        rgb += norm[:, :, np.newaxis] * color
        gray += norm
    return np.clip(rgb, 0, 1), np.clip(gray / 4.0, 0, 1)

# ────────── H&E ──────────


def load_he_at_resolution(svs_path, target_um_per_px=2.0, fov_um=2000.0):
    """Read SVS, crop centered window of `fov_um` µm, resample to ~target_um_per_px."""
    s = openslide.OpenSlide(str(svs_path))
    w, h = s.dimensions
    mpp = float(s.properties.get("openslide.mpp-x", "nan"))
    if not np.isfinite(mpp):
        mpp = MPP_FALLBACK
    fov_px = min(int(round(fov_um / mpp)), w, h)
    x0 = (w - fov_px) // 2
    y0 = (h - fov_px) // 2
    factor = target_um_per_px / mpp
    level = s.get_best_level_for_downsample(factor)
    ds = s.level_downsamples[level]
    out_size = (int(fov_px / ds), int(fov_px / ds))
    rgb = np.array(s.read_region((x0, y0), level, out_size).convert("RGB"))
    actual_mpp = mpp * ds
    return rgb, actual_mpp


def he_to_grayscale_signal(rgb):
    """Tissue signal: inverted brightness, so darker tissue → higher signal."""
    g = rgb.mean(axis=2).astype(np.float32) / 255.0
    sig = 1.0 - g
    sig[sig < 0.05] = 0  # suppress slide background noise
    return sig

# ────────── Registration ──────────


def _build_registration_method():
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.20)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R


def register_he_to_imc(he_gray, he_mpp, imc_gray, imc_mpp=1.0):
    """Run rigid 2D registration via Mattes MI with multi-rotation initialization.

    Tries 8 starting angles (0, 45, …, 315°) and keeps the one with the best
    final metric.
    """
    fixed = sitk.GetImageFromArray(imc_gray.astype(np.float32))
    fixed.SetSpacing((float(imc_mpp), float(imc_mpp)))
    moving = sitk.GetImageFromArray(he_gray.astype(np.float32))
    moving.SetSpacing((float(he_mpp), float(he_mpp)))

    best = (float("inf"), None, 0)  # (metric, transform, niter)
    for deg in (0, 45, 90, 135, 180, 225, 270, 315):
        # Initialize with rotation `deg` and centered translation
        center_init = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        # Cast back to Euler2DTransform so we can set the angle
        init_tx = sitk.Euler2DTransform()
        init_tx.SetCenter(center_init.GetParameters()[1:3] if center_init.GetNumberOfParameters() >= 3
                          else center_init.GetFixedParameters())
        init_tx.SetFixedParameters(center_init.GetFixedParameters())
        init_tx.SetTranslation(center_init.GetParameters()[1:3]
                               if center_init.GetNumberOfParameters() == 3 else (0.0, 0.0))
        init_tx.SetAngle(deg * np.pi / 180.0)

        R = _build_registration_method()
        R.SetInitialTransform(init_tx, inPlace=False)
        try:
            tx = R.Execute(fixed, moving)
            metric = R.GetMetricValue()
            niter = R.GetOptimizerIteration()
        except RuntimeError as e:
            print(f"  init={deg:3d}°: FAILED ({e})")
            continue
        print(f"  init={deg:3d}°: MI={metric:.4f}, iters={niter}")
        if metric < best[0]:
            best = (metric, tx, niter)

    final_tx = best[1]
    if final_tx is None:
        raise RuntimeError("All registration attempts failed")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_tx)
    out = resampler.Execute(moving)
    return final_tx, sitk.GetArrayFromImage(out), best[0], best[2]


def apply_transform_to_rgb(he_rgb, he_mpp, fixed_shape, fixed_spacing, transform):
    """Resample each H&E RGB channel onto the IMC grid using the same transform."""
    fixed_arr = np.zeros(fixed_shape, dtype=np.float32)
    fixed_im = sitk.GetImageFromArray(fixed_arr)
    fixed_im.SetSpacing((float(fixed_spacing), float(fixed_spacing)))

    out_rgb = np.zeros((fixed_shape[0], fixed_shape[1], 3), dtype=np.uint8)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_im)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(255)  # background = white
    for c in range(3):
        ch = sitk.GetImageFromArray(he_rgb[:, :, c].astype(np.float32))
        ch.SetSpacing((float(he_mpp), float(he_mpp)))
        warped = sitk.GetArrayFromImage(resampler.Execute(ch))
        out_rgb[:, :, c] = np.clip(warped, 0, 255).astype(np.uint8)
    return out_rgb

# ────────── Main ──────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--he", required=True)
    p.add_argument("--roi", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out", default="output/he_imc_register")
    p.add_argument("--he-target-mpp", type=float, default=2.0,
                   help="Resampled H&E mpp before registration")
    p.add_argument("--fov-um", type=float, default=2000.0,
                   help="H&E centered-crop window in µm")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[H&E] {args.he}  fov={args.fov_um} µm  target={args.he_target_mpp} µm/px")
    he_rgb, he_mpp = load_he_at_resolution(args.he, args.he_target_mpp, args.fov_um)
    he_gray = he_to_grayscale_signal(he_rgb)
    print(f"  shape={he_rgb.shape}, actual mpp={he_mpp:.3f}")

    print(f"[IMC] {args.roi}  raw_dir={args.raw_dir}")
    raw = find_raw_file(args.roi, args.raw_dir)
    if raw is None:
        raise SystemExit(f"No raw IMC file for {args.roi} in {args.raw_dir}")
    imc_rgb, imc_gray = build_imc_composite(raw)
    imc_mpp = 1.0
    print(f"  shape={imc_rgb.shape}, mpp={imc_mpp:.1f}")

    print("[Registration] Mattes MI, Euler2D, multi-resolution …")
    tx, he_gray_warped, metric, niter = register_he_to_imc(
        he_gray, he_mpp, imc_gray, imc_mpp
    )
    if hasattr(tx, "GetAngle"):
        angle = tx.GetAngle() * 180.0 / np.pi
        tx_xy = tx.GetTranslation()
    else:
        # Composite transform — extract Euler2D component if present
        try:
            inner = tx.GetNthTransform(0)
            angle = inner.GetAngle() * 180.0 / np.pi if hasattr(inner, "GetAngle") else float("nan")
            tx_xy = inner.GetTranslation() if hasattr(inner, "GetTranslation") else (float("nan"), float("nan"))
        except Exception:
            angle = float("nan")
            tx_xy = (float("nan"), float("nan"))
    print(f"  iters={niter}, MI metric={metric:.4f}, angle={angle:.2f}°, "
          f"translation=({tx_xy[0]:.1f}, {tx_xy[1]:.1f}) µm")

    # Apply transform to RGB H&E
    he_rgb_aligned = apply_transform_to_rgb(
        he_rgb, he_mpp, imc_gray.shape, imc_mpp, tx
    )

    # Build figure: original IMC, original H&E (cropped), aligned H&E, overlay
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, wspace=0.05, hspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(imc_rgb)
    ax.set_title(f"IMC composite ({args.roi})\n{imc_rgb.shape[1]}×{imc_rgb.shape[0]} µm",
                 fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(he_rgb)
    ax.set_title(f"H&E pre-registration\n{he_rgb.shape[1]}×{he_rgb.shape[0]} px @ {he_mpp:.2f} µm/px",
                 fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(he_rgb_aligned)
    ax.set_title(f"H&E registered to IMC\nMI={metric:.3f}, angle={angle:.1f}°",
                 fontsize=11)
    ax.axis("off")

    # Overlay: IMC composite on top of grayscale H&E
    ax = fig.add_subplot(gs[1, 0])
    he_gray_disp = (255 - he_rgb_aligned.mean(axis=2)).astype(np.uint8)
    ax.imshow(he_gray_disp, cmap="gray")
    ax.set_title("Overlay: H&E grayscale + IMC composite",
                 fontsize=11)
    ax.imshow(imc_rgb, alpha=0.45)
    ax.axis("off")

    # Checkerboard
    ax = fig.add_subplot(gs[1, 1])
    chk_size = 64
    h, w = imc_gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy // chk_size) + (xx // chk_size)) % 2 == 0
    chk = np.where(mask[..., None], imc_rgb, he_rgb_aligned.astype(np.float32) / 255.0)
    ax.imshow(np.clip(chk, 0, 1))
    ax.set_title("Checkerboard (IMC | H&E)", fontsize=11)
    ax.axis("off")

    # Difference / signal overlap
    ax = fig.add_subplot(gs[1, 2])
    he_norm = he_gray_warped / max(he_gray_warped.max(), 1e-6)
    diff = np.zeros((h, w, 3), dtype=np.float32)
    diff[..., 0] = he_norm
    diff[..., 1] = imc_gray
    ax.imshow(np.clip(diff, 0, 1))
    ax.set_title("Tissue overlap (H&E=red, IMC=green)\nyellow = both",
                 fontsize=11)
    ax.axis("off")

    out = out_dir / f"{args.roi}_he_imc_registered.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
