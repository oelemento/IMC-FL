#!/usr/bin/env python3
"""Diagnose why Treg-enriched T zone fraction vs grade is NS.

Shows three views of the same metric on T-panel mixed cores (n=23 patients):
  (a) per-grade box+strip with all data points visible
  (b) low (FOLL1+2) vs high (FOLL3A) box+strip
  (c) the underlying ROI-level distribution (zero-inflation diagnostic)
  (d) cumulative distribution function per grade

Reads the patient-level CSV produced by grade_arch_mixed_cores.py and the
ROI-level CSV from the same script.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}
COL = "frac_Treg-enriched T zone"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-patient", default="output/grade_arch/mixed_cores/grade_mixed_per_patient_t.csv")
    ap.add_argument("--per-roi", default="output/grade_arch/mixed_cores/grade_mixed_per_roi_t.csv")
    ap.add_argument("--out", default="output/grade_arch/treg_diagnostic")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pt = pd.read_csv(args.per_patient)
    roi = pd.read_csv(args.per_roi)
    # Reorder roi to mixed only — re-derive from per-roi flag
    roi = roi[roi["is_mixed"] == True].copy()
    roi = roi[roi["grade"].isin(GRADE_ORDER)].copy()

    print(f"Patients per grade (mixed cores): {pt.grade.value_counts().to_dict()}")
    print(f"ROIs per grade   (mixed cores): {roi.grade.value_counts().to_dict()}")
    print()

    # KW + MW
    grp = [pt.loc[pt.grade == g, COL].dropna().values for g in GRADE_ORDER]
    _, p_kw = kruskal(*grp)
    a = pt.loc[pt.grade.isin(["FOLL1", "FOLL2"]), COL].dropna().values
    b = pt.loc[pt.grade == "FOLL3A", COL].dropna().values
    _, p_mw = mannwhitneyu(a, b, alternative="two-sided")
    print(f"KW (3 grades): p={p_kw:.4f}")
    print(f"MW (low vs high): p={p_mw:.4f}")
    print()
    # Per-grade summary
    print("Per-grade summary (patient-level):")
    for g in GRADE_ORDER:
        v = pt.loc[pt.grade == g, COL].dropna().values
        n_zero = int((v < 0.005).sum())
        print(f"  {g:7s} n={len(v):2d}  median={np.median(v):.4f}  mean={np.mean(v):.4f}  "
              f"sd={np.std(v):.4f}  range=[{v.min():.4f}, {v.max():.4f}]  "
              f"n_near_zero(<0.005)={n_zero}")
    print()
    print("Low (FOLL1+2) vs High (FOLL3A) at patient level:")
    print(f"  Low (n={len(a):2d})  median={np.median(a):.4f}  mean={np.mean(a):.4f}  "
          f"sd={np.std(a):.4f}  range=[{a.min():.4f}, {a.max():.4f}]")
    print(f"  High(n={len(b):2d})  median={np.median(b):.4f}  mean={np.mean(b):.4f}  "
          f"sd={np.std(b):.4f}  range=[{b.min():.4f}, {b.max():.4f}]")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    rng = np.random.default_rng(0)

    # (a) per-grade patient box+strip
    ax = axes[0]
    data = grp
    bp = ax.boxplot(data, tick_labels=GRADE_ORDER, patch_artist=True, widths=0.55,
                    showfliers=False)
    for patch, g in zip(bp["boxes"], GRADE_ORDER):
        patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.5)
    for i, (g, vals) in enumerate(zip(GRADE_ORDER, data)):
        xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.22
        ax.scatter(xs, vals, color=GRADE_COLORS[g], s=40, alpha=0.85,
                   edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_title(f"(a) Patient-level by grade\nKW p={p_kw:.3f}", fontsize=11)
    ax.set_ylabel("Treg-enriched T zone fraction")
    ax.set_xlabel("Grade")
    ax.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (b) low vs high
    ax = axes[1]
    data2 = [a, b]
    labels2 = [f"Low (FOLL1+2)\nn={len(a)}", f"High (FOLL3A)\nn={len(b)}"]
    bp = ax.boxplot(data2, tick_labels=labels2, patch_artist=True, widths=0.55,
                    showfliers=False)
    bp["boxes"][0].set_facecolor("#1f77b4"); bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("#d62728"); bp["boxes"][1].set_alpha(0.5)
    for i, vals in enumerate(data2):
        xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.22
        col = "#1f77b4" if i == 0 else "#d62728"
        ax.scatter(xs, vals, color=col, s=40, alpha=0.85, edgecolor="black",
                   linewidth=0.5, zorder=3)
    ax.set_title(f"(b) Patient-level — low vs high\nMW p={p_mw:.3f}", fontsize=11)
    ax.set_ylabel("Treg-enriched T zone fraction")
    ax.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (c) ROI-level distribution showing zero-inflation
    ax = axes[2]
    bins = np.linspace(0, max(roi[COL].max(), 0.5), 25)
    for g in GRADE_ORDER:
        v = roi.loc[roi.grade == g, COL].dropna().values
        ax.hist(v, bins=bins, color=GRADE_COLORS[g], alpha=0.45,
                label=f"{g} (n={len(v)} ROIs)", edgecolor="white", linewidth=0.6)
    ax.set_title("(c) Per-ROI distribution\n(zero-inflation diagnostic)", fontsize=11)
    ax.set_xlabel("Treg-enriched T zone fraction (per ROI)")
    ax.set_ylabel("ROI count")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (d) ECDF per grade
    ax = axes[3]
    for g in GRADE_ORDER:
        v = np.sort(pt.loc[pt.grade == g, COL].dropna().values)
        if len(v) == 0: continue
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where="post", color=GRADE_COLORS[g], lw=2.3,
                label=f"{g} (n={len(v)})")
        # Markers at each data point so the small sample is visible
        ax.scatter(v, y, color=GRADE_COLORS[g], s=22, zorder=3,
                   edgecolor="white", linewidth=0.5)
    ax.set_title("(d) Patient-level ECDF\n(curves apart = different distributions)",
                 fontsize=11)
    ax.set_xlabel("Treg-enriched T zone fraction (patient mean)")
    ax.set_ylabel("Cumulative probability")
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("Why Treg-enriched T zone fraction is NS vs grade — T-panel mixed cores",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_treg_zone_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
