#!/usr/bin/env python3
"""
EZH2-wt vs EZH2-mut marker expression in malignant B cells.
Horizontal paired bar chart (like VISTA+/- format), sorted by p-value.

Two panels stacked: T-panel markers, S-panel markers.

Usage:
    PYTHONPATH=. python3.11 scripts/ezh2_barplot_figure.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import mannwhitneyu
from src.clinical_linkage import normalize_sample_id

OUTPUT_DIR = "output/cd14_validation"
C_WT = "#b0b0b0"
C_MUT = "#d62728"


def get_bcell_mask(adata):
    ct = adata.obs["cell_type"].astype(str)
    return ct.str.contains("B cell", case=False, na=False) | ct.str.contains("GC B", case=False, na=False)


def compute_panel_data(path, panel_name, ezh2_map):
    """Load panel, compute per-patient B cell marker means, return results."""
    print(f"Loading {panel_name}-panel...")
    adata = ad.read_h5ad(path)
    adata.obs["slide_ID"] = [normalize_sample_id(s) for s in adata.obs["sample_id"]]
    mask = get_bcell_mask(adata)
    bcells = adata[mask]
    print(f"  {mask.sum():,} B cells")

    per_patient = bcells.obs.groupby("slide_ID").apply(
        lambda g: pd.Series(
            np.asarray(adata[g.index].X.mean(axis=0)).flatten(),
            index=adata.var_names
        )
    )
    per_patient["EZH2"] = per_patient.index.map(ezh2_map)
    per_patient = per_patient[per_patient["EZH2"].isin(["wt", "mut"])]

    results = []
    markers = [m for m in adata.var_names if not m.startswith("DNA")]
    for marker in markers:
        wt = per_patient.loc[per_patient["EZH2"] == "wt", marker].dropna()
        mut = per_patient.loc[per_patient["EZH2"] == "mut", marker].dropna()
        if len(wt) >= 5 and len(mut) >= 5:
            _, p = mannwhitneyu(wt, mut, alternative="two-sided")
            results.append({
                "marker": marker,
                "wt_mean": wt.mean(),
                "mut_mean": mut.mean(),
                "pval": p,
                "n_wt": len(wt),
                "n_mut": len(mut),
            })

    res = pd.DataFrame(results).sort_values("pval")
    # BH correction within panel
    m = len(res)
    res["rank"] = range(1, m + 1)
    res["q"] = (res["pval"] * m / res["rank"]).clip(upper=1.0)
    res["q"] = res["q"][::-1].cummin()[::-1]
    return res


def draw_panel(ax, res, panel_name, n_wt, n_mut):
    """Draw horizontal paired bar chart for one panel."""
    # Sort by p-value (most significant at top)
    res = res.sort_values("pval", ascending=False).reset_index(drop=True)

    n = len(res)
    y = np.arange(n)
    h = 0.35

    ax.barh(y + h/2, res["mut_mean"], h, label=f"EZH2-mut (n={n_mut})",
            color=C_MUT, edgecolor="white", linewidth=0.3)
    ax.barh(y - h/2, res["wt_mean"], h, label=f"EZH2-wt (n={n_wt})",
            color=C_WT, edgecolor="white", linewidth=0.3)

    # Significance stars
    for i, (_, row) in enumerate(res.iterrows()):
        max_val = max(row["wt_mean"], row["mut_mean"])
        if row["q"] < 0.001:
            stars = "***"
        elif row["q"] < 0.01:
            stars = "**"
        elif row["q"] < 0.05:
            stars = "*"
        elif row["pval"] < 0.05:
            stars = "."  # nominal
        else:
            stars = ""

        if stars:
            color = C_MUT if row["q"] < 0.05 else "#ff7f0e"
            fontweight = "bold" if row["q"] < 0.05 else "normal"
            ax.text(max_val + 0.02, i, stars, va="center", fontsize=10,
                    color=color, fontweight=fontweight)

    # Marker labels
    labels = [m.replace("_", "-") for m in res["marker"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean expression in B cells (z-scored)", fontsize=10)
    ax.set_title(f"{panel_name}-panel markers\n(n={n_wt} wt vs {n_mut} mut)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_clinical_ezh2.csv"))
    ezh2_map = dict(zip(
        master.loc[master["EZH2"].isin(["wt", "mut"]), "slide_ID"],
        master.loc[master["EZH2"].isin(["wt", "mut"]), "EZH2"],
    ))

    t_res = compute_panel_data("output/all_TMA_T_global_v8.h5ad", "T", ezh2_map)
    s_res = compute_panel_data("output/all_TMA_S_global_v8.h5ad", "S", ezh2_map)

    n_wt_t = t_res["n_wt"].iloc[0]
    n_mut_t = t_res["n_mut"].iloc[0]
    n_wt_s = s_res["n_wt"].iloc[0]
    n_mut_s = s_res["n_mut"].iloc[0]

    # Figure: two panels side by side
    n_t = len(t_res)
    n_s = len(s_res)
    row_height = 0.38
    fig_h = max(n_t, n_s) * row_height + 2

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(16, fig_h))
    fig.suptitle("EZH2-wt vs EZH2-mut: Marker Expression in Malignant B Cells",
                 fontsize=15, fontweight="bold", y=1.01)

    # Footnote
    fig.text(0.5, -0.01,
             "Sorted by p-value (most significant at top).  "
             "*** q<0.001  ** q<0.01  * q<0.05 (FDR)  . p<0.05 (nominal only)",
             ha="center", fontsize=9, color="#555555")

    draw_panel(ax_t, t_res, "T", n_wt_t, n_mut_t)
    draw_panel(ax_s, s_res, "S", n_wt_s, n_mut_s)

    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    out_path = os.path.join(OUTPUT_DIR, "ezh2_bcell_barplot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
