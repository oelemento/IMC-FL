#!/usr/bin/env python3
"""Compare VISTA expression in FL vs normal tonsil controls (S-panel).

Cell types: M2 Mac, S100A9+, M1 Mac, CD14+ FDC, CD14- FDC.
Reports per-cell mean +/- SEM (arcsinh, raw.X), per-cell VISTA+ fraction
(>0.5 scaled, .X), per-ROI mean +/- SEM, and one-sided Mann-Whitney p-values
at both the per-cell and per-ROI levels (FL > tonsil).

Per-ROI is the rigorous test (ROIs are the independent units); per-cell is
inflated by pseudoreplication. The companion figure script can use either,
but per-ROI stars are recommended for significance annotation.

Usage:
    .venv/bin/python scripts/vista_fl_vs_tonsil.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


CONTROL_TONSIL_TERMS = ("tonsil", "_ton_")
EXCLUDE_OTHER_CONTROLS = ("prostate", "kidney", "spleen", "adrenal", "_adr_")

CELL_TYPES_FOR_COMPARISON = [
    "M2 Macrophages",
    "Myeloid (S100A9+)",
    "M1 Macrophages",
]
MIN_CELLS_PER_ROI = 5  # require this many cells of a type in an ROI to count
MIN_ROIS_PER_GROUP = 3  # require this many ROIs in each group to test


def is_tonsil(sample_id: str) -> bool:
    s = sample_id.lower()
    return any(term in s for term in CONTROL_TONSIL_TERMS)


def is_other_control(sample_id: str) -> bool:
    s = sample_id.lower()
    return any(term in s for term in EXCLUDE_OTHER_CONTROLS)


def is_tumor(sample_id: str) -> bool:
    return not is_tonsil(sample_id) and not is_other_control(sample_id)


def load_panel(path: str):
    a = ad.read_h5ad(path)
    if a.raw is None:
        raise ValueError(
            f"{path} has no .raw — cannot extract arcsinh-transformed values. "
            "This script assumes .raw.X = arcsinh, .X = scaled."
        )
    sid = a.obs["sample_id"].astype(str).values
    ct = a.obs["cell_type"].astype(str).values

    raw_vn = list(a.raw.var_names)
    for m in ("VISTA", "CD14"):
        if m not in raw_vn:
            raise ValueError(f"{m} not found in raw markers of {path}")
    Xraw = a.raw.X.toarray() if hasattr(a.raw.X, "toarray") else np.asarray(a.raw.X)
    vista_arcsinh = Xraw[:, raw_vn.index("VISTA")]
    cd14_arcsinh = Xraw[:, raw_vn.index("CD14")]

    # Scaled VISTA from .X — used for the canonical >0.5 VISTA+ gate
    sc_vn = list(a.var_names)
    if "VISTA" not in sc_vn:
        raise ValueError(f"VISTA not in scaled (.X) var of {path}")
    Xsc = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    vista_scaled = Xsc[:, sc_vn.index("VISTA")]

    return {
        "sid": sid, "ct": ct,
        "vista_arcsinh": vista_arcsinh, "vista_scaled": vista_scaled,
        "cd14_arcsinh": cd14_arcsinh,
    }


def split_fdc_by_cd14(ct: np.ndarray, cd14: np.ndarray, threshold: float):
    """Return labels with FDCs split into CD14+ / CD14- by raw arcsinh CD14."""
    out = ct.copy()
    fdc = ct == "FDC"
    out[fdc & (cd14 >= threshold)] = "CD14+ FDC"
    out[fdc & (cd14 < threshold)] = "CD14- FDC"
    return out


def per_roi_means(sid_subset, vista_subset, ct_subset, cell_type, *, fdc_split=None):
    """Mean VISTA per ROI for cells of `cell_type` (skip ROIs with <MIN_CELLS_PER_ROI)."""
    mask = (ct_subset == cell_type) if fdc_split is None else (fdc_split == cell_type)
    if mask.sum() == 0:
        return np.array([])
    df = pd.DataFrame({"sid": sid_subset[mask], "v": vista_subset[mask]})
    counts = df.groupby("sid").size()
    keep_rois = counts[counts >= MIN_CELLS_PER_ROI].index
    if len(keep_rois) == 0:
        return np.array([])
    return df[df["sid"].isin(keep_rois)].groupby("sid")["v"].mean().values


def compare_per_roi(fl_means: np.ndarray, ton_means: np.ndarray) -> dict:
    """Mann-Whitney across ROIs, FL > tonsil hypothesis. Returns mean +/- SEM per group."""
    def _sem(a):
        if len(a) <= 1:
            return float("nan")
        return float(np.std(a, ddof=1) / np.sqrt(len(a)))

    if len(fl_means) < MIN_ROIS_PER_GROUP or len(ton_means) < MIN_ROIS_PER_GROUP:
        return {
            "n_fl_rois": len(fl_means),
            "n_ton_rois": len(ton_means),
            "fl_roi_mean": float(np.mean(fl_means)) if len(fl_means) else np.nan,
            "ton_roi_mean": float(np.mean(ton_means)) if len(ton_means) else np.nan,
            "fl_roi_sem": _sem(fl_means),
            "ton_roi_sem": _sem(ton_means),
            "p_greater_per_roi": np.nan,
            "skipped": True,
        }
    res = mannwhitneyu(fl_means, ton_means, alternative="greater")
    return {
        "n_fl_rois": int(len(fl_means)),
        "n_ton_rois": int(len(ton_means)),
        "fl_roi_mean": float(np.mean(fl_means)),
        "ton_roi_mean": float(np.mean(ton_means)),
        "fl_roi_sem": _sem(fl_means),
        "ton_roi_sem": _sem(ton_means),
        "p_greater_per_roi": float(res.pvalue),
        "skipped": False,
    }


def per_cell_descriptive(d, mask, ct_split, cell_type):
    """Pooled per-cell summary (mean +/- SEM across cells)."""
    cells = mask & (ct_split == cell_type)
    n = int(cells.sum())
    if n == 0:
        return {
            "n_cells": 0,
            "arcsinh_mean": np.nan,
            "arcsinh_sem": np.nan,
            "scaled_pos_pct": np.nan,
        }
    vals = d["vista_arcsinh"][cells]
    sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n_cells": n,
        "arcsinh_mean": float(vals.mean()),
        "arcsinh_sem": sem,
        "scaled_pos_pct": float(100 * (d["vista_scaled"][cells] > 0.5).mean()),
    }


def per_cell_test(d, fl_mask, ton_mask, ct_split, cell_type):
    """Mann-Whitney across cells, FL > tonsil (one-sided)."""
    fl_vals = d["vista_arcsinh"][fl_mask & (ct_split == cell_type)]
    ton_vals = d["vista_arcsinh"][ton_mask & (ct_split == cell_type)]
    if len(fl_vals) < 10 or len(ton_vals) < 10:
        return float("nan")
    return float(mannwhitneyu(fl_vals, ton_vals, alternative="greater").pvalue)


def derive_fdc_cd14_threshold(d, fl_mask, percentile: float = 75) -> float:
    """Derive CD14 threshold from FL FDCs only (the cohort distribution we
    use throughout the project) so that tonsil FDCs are stratified against
    the same reference."""
    fdc_in_fl = fl_mask & (d["ct"] == "FDC")
    if fdc_in_fl.sum() < 100:
        raise ValueError(f"Only {fdc_in_fl.sum()} FL FDCs — cannot derive CD14 p{percentile}")
    return float(np.percentile(d["cd14_arcsinh"][fdc_in_fl], percentile))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-panel", default="output/all_TMA_S_global_v8.h5ad")
    parser.add_argument("--out", default="output/vista_fl_vs_tonsil.csv")
    args = parser.parse_args()

    print(f"Loading {args.s_panel} ...")
    d = load_panel(args.s_panel)

    fl_mask = np.array([is_tumor(s) for s in d["sid"]])
    ton_mask = np.array([is_tonsil(s) for s in d["sid"]])
    other_mask = ~fl_mask & ~ton_mask
    n_fl_rois = len(set(d["sid"][fl_mask]))
    n_ton_rois = len(set(d["sid"][ton_mask]))
    print(f"  FL: {fl_mask.sum():,} cells across {n_fl_rois} ROIs")
    print(f"  Tonsil: {ton_mask.sum():,} cells across {n_ton_rois} ROIs")
    print(f"  Excluded other controls (prostate/kidney/spleen/adrenal): "
          f"{other_mask.sum():,} cells")

    cd14_thresh = derive_fdc_cd14_threshold(d, fl_mask, percentile=75)
    print(f"  FDC split threshold: CD14 (raw arcsinh) p75 within FL FDCs = {cd14_thresh:.3f}")

    ct_split = split_fdc_by_cd14(d["ct"], d["cd14_arcsinh"], threshold=cd14_thresh)

    cell_types = ["CD14+ FDC", "CD14- FDC", *CELL_TYPES_FOR_COMPARISON]

    rows = []
    print(
        f"\n{'Cell type':22s}  {'FL n':>7s}  {'Ton n':>6s}  "
        f"{'FL arcsinh':>10s}  {'Ton arcsinh':>11s}  "
        f"{'FL %scaled+':>11s}  {'Ton %scaled+':>12s}  "
        f"{'FL ROIs':>8s}  {'Ton ROIs':>9s}  {'p (per-ROI, 1-sided)':>22s}"
    )
    print("-" * 140)
    for ct in cell_types:
        # Pooled descriptive per-cell stats
        fl_d = per_cell_descriptive(d, fl_mask, ct_split, ct)
        ton_d = per_cell_descriptive(d, ton_mask, ct_split, ct)
        p_per_cell = per_cell_test(d, fl_mask, ton_mask, ct_split, ct)

        # Per-ROI means (the primary statistical test)
        is_fdc_split = ct in ("CD14+ FDC", "CD14- FDC")
        fl_roi = per_roi_means(
            d["sid"][fl_mask], d["vista_arcsinh"][fl_mask],
            d["ct"][fl_mask],
            ct,
            fdc_split=ct_split[fl_mask] if is_fdc_split else None,
        )
        ton_roi = per_roi_means(
            d["sid"][ton_mask], d["vista_arcsinh"][ton_mask],
            d["ct"][ton_mask],
            ct,
            fdc_split=ct_split[ton_mask] if is_fdc_split else None,
        )
        roi_test = compare_per_roi(fl_roi, ton_roi)

        rows.append({
            "cell_type": ct,
            "n_fl_cells": fl_d["n_cells"],
            "n_ton_cells": ton_d["n_cells"],
            "fl_arcsinh_mean": fl_d["arcsinh_mean"],
            "fl_arcsinh_sem": fl_d["arcsinh_sem"],
            "ton_arcsinh_mean": ton_d["arcsinh_mean"],
            "ton_arcsinh_sem": ton_d["arcsinh_sem"],
            "fl_scaled_pos_pct": fl_d["scaled_pos_pct"],
            "ton_scaled_pos_pct": ton_d["scaled_pos_pct"],
            "p_greater_per_cell": p_per_cell,
            **roi_test,
        })

        def fmt(v, kind="f"):
            if isinstance(v, float) and np.isnan(v):
                return "n/a"
            if kind == "%":
                return f"{v:.1f}%"
            if kind == "e":
                return f"{v:.2e}"
            return f"{v:.3f}"

        p_disp = (
            f"[skipped: {roi_test['n_fl_rois']}/{roi_test['n_ton_rois']} ROIs]"
            if roi_test["skipped"]
            else fmt(roi_test["p_greater_per_roi"], "e")
        )

        print(
            f"{ct:22s}  {fl_d['n_cells']:>7d}  {ton_d['n_cells']:>6d}  "
            f"{fmt(fl_d['arcsinh_mean']):>10s}  {fmt(ton_d['arcsinh_mean']):>11s}  "
            f"{fmt(fl_d['scaled_pos_pct'], '%'):>11s}  {fmt(ton_d['scaled_pos_pct'], '%'):>12s}  "
            f"{roi_test['n_fl_rois']:>8d}  {roi_test['n_ton_rois']:>9d}  {p_disp:>22s}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(
        "Note: per-ROI p-values are the rigorous test (FL > tonsil, one-sided "
        "Mann-Whitney, ROIs as independent observations). Per-cell columns "
        "(p_greater_per_cell) are also written but are inflated by sample "
        "size — use them only as a sensitivity check."
    )


if __name__ == "__main__":
    main()
