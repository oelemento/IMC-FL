#!/usr/bin/env python3
"""
EZH2-wt vs EZH2-mut in malignant B cells — horizontal layout with error bars.

Three panels stacked vertically:
  (a) T-panel markers
  (b) S-panel markers
  (c) Tissue compartment fractions (T-panel UTAG)

Usage:
    PYTHONPATH=. python3.11 scripts/ezh2_barplot_horizontal.py
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


def compute_marker_data(path, panel_name, ezh2_map):
    """Per-patient B cell marker means + SEM."""
    print(f"Loading {panel_name}-panel...")
    adata = ad.read_h5ad(path)
    adata.obs["slide_ID"] = [normalize_sample_id(s) for s in adata.obs["sample_id"]]
    mask = get_bcell_mask(adata)
    print(f"  {mask.sum():,} B cells")

    per_patient = adata[mask].obs.groupby("slide_ID").apply(
        lambda g: pd.Series(
            np.asarray(adata[g.index].X.mean(axis=0)).flatten(),
            index=adata.var_names
        )
    )
    per_patient["EZH2"] = per_patient.index.map(ezh2_map)
    per_patient = per_patient[per_patient["EZH2"].isin(["wt", "mut"])]

    markers = [m for m in adata.var_names if not m.startswith("DNA")]
    results = []
    for marker in markers:
        wt = per_patient.loc[per_patient["EZH2"] == "wt", marker].dropna()
        mut = per_patient.loc[per_patient["EZH2"] == "mut", marker].dropna()
        if len(wt) >= 5 and len(mut) >= 5:
            _, p = mannwhitneyu(wt, mut, alternative="two-sided")
            results.append({
                "marker": marker,
                "wt_mean": wt.mean(), "wt_sem": wt.sem(),
                "mut_mean": mut.mean(), "mut_sem": mut.sem(),
                "pval": p, "n_wt": len(wt), "n_mut": len(mut),
            })

    res = pd.DataFrame(results).sort_values("pval")
    m = len(res)
    res["rank"] = range(1, m + 1)
    res["q"] = (res["pval"] * m / res["rank"]).clip(upper=1.0)
    res["q"] = res["q"][::-1].cummin()[::-1]
    return res


def compute_compartment_data(utag_path, ezh2_map):
    """Per-patient compartment fractions."""
    print("Loading UTAG compartments...")
    adata = ad.read_h5ad(utag_path)
    adata.obs["slide_ID"] = [normalize_sample_id(s) for s in adata.obs["sample_id"]]

    comp_col = "compartment_name" if "compartment_name" in adata.obs.columns else "tissue_compartment"
    comp_counts = adata.obs.groupby(["slide_ID", comp_col]).size().unstack(fill_value=0)
    comp_frac = comp_counts.div(comp_counts.sum(axis=1), axis=0)
    comp_frac["EZH2"] = comp_frac.index.map(ezh2_map)
    comp_frac = comp_frac[comp_frac["EZH2"].isin(["wt", "mut"])]

    compartments = [c for c in comp_frac.columns if c != "EZH2"]
    results = []
    for comp in compartments:
        wt = comp_frac.loc[comp_frac["EZH2"] == "wt", comp].dropna()
        mut = comp_frac.loc[comp_frac["EZH2"] == "mut", comp].dropna()
        if len(wt) >= 5 and len(mut) >= 5:
            _, p = mannwhitneyu(wt, mut, alternative="two-sided")
            results.append({
                "marker": comp,
                "wt_mean": wt.mean(), "wt_sem": wt.sem(),
                "mut_mean": mut.mean(), "mut_sem": mut.sem(),
                "pval": p, "n_wt": len(wt), "n_mut": len(mut),
            })

    res = pd.DataFrame(results).sort_values("pval")
    m = len(res)
    res["rank"] = range(1, m + 1)
    res["q"] = (res["pval"] * m / res["rank"]).clip(upper=1.0)
    res["q"] = res["q"][::-1].cummin()[::-1]
    return res


def draw_horizontal_bars(ax, res, title, xlabel, bar_height=0.35):
    """Horizontal paired bars: markers on y-axis, values on x-axis, with error bars."""
    # Sort so most significant is at top (= last in plot, since y-axis is inverted)
    res = res.sort_values("pval", ascending=True).reset_index(drop=True)

    n = len(res)
    x = np.arange(n)
    h = bar_height

    n_wt = res["n_wt"].iloc[0]
    n_mut = res["n_mut"].iloc[0]

    ax.barh(x - h/2, res["wt_mean"], h,
            xerr=res["wt_sem"], error_kw=dict(lw=0.7, capsize=2, capthick=0.7),
            label=f"EZH2-wt (n={n_wt})", color=C_WT, edgecolor="white", linewidth=0.3)
    ax.barh(x + h/2, res["mut_mean"], h,
            xerr=res["mut_sem"], error_kw=dict(lw=0.7, capsize=2, capthick=0.7),
            label=f"EZH2-mut (n={n_mut})", color=C_MUT, edgecolor="white", linewidth=0.3)

    # Significance annotations
    for i, (_, row) in enumerate(res.iterrows()):
        max_val = max(row["wt_mean"] + row["wt_sem"], row["mut_mean"] + row["mut_sem"])
        if row["q"] < 0.001:
            stars = "***"
        elif row["q"] < 0.01:
            stars = "**"
        elif row["q"] < 0.05:
            stars = "*"
        elif row["pval"] < 0.05:
            stars = "\u2020"  # dagger for nominal
        else:
            stars = ""
        if stars:
            color = C_MUT if row["q"] < 0.05 else "#ff7f0e"
            fw = "bold" if row["q"] < 0.05 else "normal"
            ax.text(max_val + 0.02, i, stars, va="center", fontsize=10, color=color, fontweight=fw)

    labels = [m.replace("_", "-") for m in res["marker"]]
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_clinical_ezh2.csv"))
    ezh2_map = dict(zip(
        master.loc[master["EZH2"].isin(["wt", "mut"]), "slide_ID"],
        master.loc[master["EZH2"].isin(["wt", "mut"]), "EZH2"],
    ))

    t_res = compute_marker_data("output/all_TMA_T_global_v8.h5ad", "T", ezh2_map)
    s_res = compute_marker_data("output/all_TMA_S_global_v8.h5ad", "S", ezh2_map)
    comp_res = compute_compartment_data("output/all_TMA_T_utag_ct_merged.h5ad", ezh2_map)

    # Figure: 3 panels stacked vertically
    n_t = len(t_res)
    n_s = len(s_res)
    n_c = len(comp_res)
    row_h = 0.32
    h_t = max(n_t * row_h, 4)
    h_s = max(n_s * row_h, 4)
    h_c = max(n_c * row_h, 3)
    total_h = h_t + h_s + h_c + 3  # extra for titles/spacing

    fig, (ax_t, ax_s, ax_c) = plt.subplots(3, 1, figsize=(12, total_h),
                                             gridspec_kw={"height_ratios": [n_t, n_s, n_c]})
    fig.suptitle("EZH2-wt vs EZH2-mut in Malignant B Cells",
                 fontsize=16, fontweight="bold", y=1.0)

    draw_horizontal_bars(ax_t, t_res,
                         f"(a) T-panel markers (n={n_t})",
                         "Mean expression in B cells (z-scored ± SEM)")

    draw_horizontal_bars(ax_s, s_res,
                         f"(b) S-panel markers (n={n_s})",
                         "Mean expression in B cells (z-scored ± SEM)")

    draw_horizontal_bars(ax_c, comp_res,
                         f"(c) Tissue compartment fractions (T-panel UTAG, n={n_c})",
                         "Mean fraction per patient (± SEM)",
                         bar_height=0.35)

    fig.text(0.5, -0.005,
             "Sorted by p-value (most significant at top).  "
             "*** q<0.001   ** q<0.01   * q<0.05 (FDR-corrected)   "
             "\u2020 p<0.05 (nominal only, not FDR-corrected)",
             ha="center", fontsize=9, color="#555555")

    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    out_path = os.path.join(OUTPUT_DIR, "ezh2_bcell_barplot_horizontal.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
