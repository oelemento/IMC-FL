#!/usr/bin/env python3
"""Plot within-compartment cell-type fraction shifts by grade.

Reads `output/grade_arch/grade_followups_s_per_patient.csv` and renders the
two within-compartment shifts that flagged in the analysis:
  - FDC fraction within FDC network zone
  - B cells (PAX5+) fraction within FDC network zone

Plus a third panel: stacked-bar of FDC-network-zone composition by grade
(median per grade across the four cell types we tracked there).
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
CT_COLORS = {
    "FDC": "#d62728", "M2 Macrophages": "#9467bd",
    "Myeloid (S100A9+)": "#ff7f0e", "B cells (PAX5+)": "#1f77b4",
    "B cells (BCL2+)": "#17becf",
}


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


def boxplot(ax, df, metric, label, q=None):
    data = [df.loc[df.grade == g, metric].dropna().values for g in GRADE_ORDER]
    bp = ax.boxplot(data, tick_labels=GRADE_ORDER, patch_artist=True,
                    widths=0.55, showfliers=False)
    for patch, g in zip(bp["boxes"], GRADE_ORDER):
        patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(zip(GRADE_ORDER, data)):
        xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(xs, vals, color=GRADE_COLORS[g], s=12, alpha=0.7,
                   edgecolor="white", linewidth=0.4, zorder=3)
    p = kw_p(df, metric)
    title = f"{label}\np={p:.3g}"
    if q is not None and not np.isnan(q):
        flag = " *" if q < 0.05 else ""
        title += f", q={q:.3g}{flag}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Grade")
    ax.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-patient", default="output/grade_arch/grade_followups_s_per_patient.csv")
    p.add_argument("--shift-csv", default="output/grade_arch/grade_followups_compartment_shift.csv")
    p.add_argument("--out", default="output/grade_arch/fig_grade_within_compartment.png")
    args = p.parse_args()

    df = pd.read_csv(args.per_patient)
    df = df[df.grade.isin(GRADE_ORDER)]
    print(f"Patients: {len(df)} (FOLL1={sum(df.grade=='FOLL1')}, "
          f"FOLL2={sum(df.grade=='FOLL2')}, FOLL3A={sum(df.grade=='FOLL3A')})")

    # Pull q-values from the shift summary
    qmap = {}
    if Path(args.shift_csv).exists():
        sh = pd.read_csv(args.shift_csv)
        qmap = dict(zip(sh["metric"], sh["q_BH"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel 1: FDC fraction within FDC network zone
    m = "in_FDC_network_zone_FDC_frac"
    boxplot(axes[0], df, m, "FDC fraction within\nFDC network zone",
            q=qmap.get(m))

    # Panel 2: B cells (PAX5+) within FDC network zone (the dramatic one)
    m = "in_FDC_network_zone_B_cells_PAX5+_frac"
    boxplot(axes[1], df, m, "B cells (PAX5+) fraction\nwithin FDC network zone",
            q=qmap.get(m))

    # Panel 3: stacked bar of median composition by grade in FDC network zone
    cell_types = ("FDC", "M2 Macrophages", "Myeloid (S100A9+)",
                  "B cells (PAX5+)", "B cells (BCL2+)")
    cols = []
    for ct in cell_types:
        col = f"in_FDC_network_zone_{ct}_frac".replace(" ", "_") \
            .replace("(", "").replace(")", "")
        if col in df.columns:
            cols.append((ct, col))

    medians = pd.DataFrame({
        ct: [df.loc[df.grade == g, col].median() for g in GRADE_ORDER]
        for ct, col in cols
    }, index=GRADE_ORDER)
    # renormalize so the bar sums to the actual sum of these 5 cell types
    # (don't force to 1.0 — there are other cell types not shown)

    bottom = np.zeros(len(GRADE_ORDER))
    for ct, _col in cols:
        vals = medians[ct].values
        axes[2].bar(GRADE_ORDER, vals, bottom=bottom, color=CT_COLORS.get(ct, "#888"),
                    label=ct, edgecolor="white", linewidth=0.5)
        bottom += vals
    axes[2].set_ylabel("Median fraction of cells\n(within FDC network zone)")
    axes[2].set_xlabel("Grade")
    axes[2].set_title("FDC network zone\ncomposition shift", fontsize=11)
    axes[2].legend(fontsize=9, loc="upper right")
    for sp in ("top", "right"):
        axes[2].spines[sp].set_visible(False)

    fig.suptitle("Within-compartment cell-type composition by FL grade\n"
                 f"(S-panel, n={len(df)} patients)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
