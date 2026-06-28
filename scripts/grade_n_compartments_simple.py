#!/usr/bin/env python3
"""Single-panel: number of compartments per ROI by FL grade.

Reads `output/grade_arch/grade_compartment_biomarkers_per_patient.csv` and
plots the n_compartments_present at three thresholds side-by-side, plus a
no-threshold variant (count any compartment with ≥1 cell — usually all 11).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}


def kw_p(df, metric):
    groups = [df.loc[df.grade == g, metric].dropna().values for g in GRADE_ORDER]
    if any(len(x) < 3 for x in groups):
        return np.nan
    if len(np.unique(np.concatenate(groups))) < 2:
        return np.nan
    try:
        _, p = kruskal(*groups)
        return float(p)
    except ValueError:
        return np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-patient",
                   default="output/grade_arch/grade_compartment_biomarkers_per_patient.csv")
    p.add_argument("--out", default="output/grade_arch/fig_grade_n_compartments_simple.png")
    args = p.parse_args()

    df = pd.read_csv(args.per_patient)
    df = df[df.grade.isin(GRADE_ORDER)].copy()
    print(f"Patients: {len(df)}  ({df.grade.value_counts().to_dict()})")

    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=False)
    for ax, thr in zip(axes, (0.02, 0.05, 0.10)):
        col = f"n_compartments_present_p{int(thr*100):02d}"
        data = [df.loc[df.grade == g, col].dropna().values for g in GRADE_ORDER]
        bp = ax.boxplot(data, tick_labels=GRADE_ORDER, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GRADE_ORDER):
            patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.55)
        for i, (g, vals) in enumerate(zip(GRADE_ORDER, data)):
            xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(xs, vals, color=GRADE_COLORS[g], s=18, alpha=0.75,
                       edgecolor="white", linewidth=0.4, zorder=3)
        meds = [int(np.median(d)) if len(d) else "" for d in data]
        pv = kw_p(df, col)
        ax.set_title(f"≥{int(thr*100)}% threshold\np = {pv:.3g}",
                     fontsize=12)
        ax.set_xlabel("Grade"); ax.set_ylabel("Compartments present")
        ax.tick_params(labelsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # Annotate medians above boxes
        for i, m in enumerate(meds, 1):
            ax.text(i, ax.get_ylim()[1] * 0.97, f"med={m}",
                    ha="center", va="top", fontsize=10, color="#333")

    fig.suptitle(f"Number of compartments present per ROI by FL grade "
                 f"(S-panel, n={len(df)} patients)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
