#!/usr/bin/env python3
"""
EZH2 figure — vertical grouped bars, B cells + all cells + compartments.

Layout:
  Row 1: (a) T-panel B cells  |  (b) S-panel B cells
  Row 2: (c) T-panel all cells  |  (d) S-panel all cells
  Row 3: (e) Tissue compartment fractions (wide)

Usage:
    PYTHONPATH=. python3.11 scripts/ezh2_barplot_v2.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def compute_marker_stats(adata, ezh2_map, cell_mask=None):
    """Compute per-patient marker mean ± SEM, split by EZH2."""
    subset = adata[cell_mask] if cell_mask is not None else adata

    per_patient = subset.obs.groupby("slide_ID").apply(
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


def draw_vertical_bars(ax, res, title, ylabel, bar_width=0.35):
    """Vertical grouped bars: markers along x-axis, values on y-axis."""
    res = res.sort_values("pval", ascending=True).reset_index(drop=True)
    n = len(res)
    x = np.arange(n)
    w = bar_width

    n_wt = res["n_wt"].iloc[0]
    n_mut = res["n_mut"].iloc[0]

    ax.bar(x - w/2, res["wt_mean"], w,
           yerr=res["wt_sem"], error_kw=dict(lw=0.6, capsize=1.5, capthick=0.6),
           label=f"EZH2-wt (n={n_wt})", color=C_WT, edgecolor="white", linewidth=0.3)
    ax.bar(x + w/2, res["mut_mean"], w,
           yerr=res["mut_sem"], error_kw=dict(lw=0.6, capsize=1.5, capthick=0.6),
           label=f"EZH2-mut (n={n_mut})", color=C_MUT, edgecolor="white", linewidth=0.3)

    # Significance stars above bars
    for i, (_, row) in enumerate(res.iterrows()):
        top = max(row["wt_mean"] + row["wt_sem"], row["mut_mean"] + row["mut_sem"])
        if row["q"] < 0.001:
            stars = "***"
        elif row["q"] < 0.01:
            stars = "**"
        elif row["q"] < 0.05:
            stars = "*"
        elif row["pval"] < 0.05:
            stars = "\u2020"
        else:
            continue
        color = C_MUT if row["q"] < 0.05 else "#ff7f0e"
        fw = "bold" if row["q"] < 0.05 else "normal"
        ax.text(i, top + 0.03, stars, ha="center", va="bottom",
                fontsize=9, color=color, fontweight=fw)

    labels = [m.replace("_", "-") for m in res["marker"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=55, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.set_xlim(-0.6, n - 0.4)


def draw_compartment_bars(ax, res, title):
    """Vertical grouped bars for compartment fractions."""
    res = res.sort_values("pval", ascending=True).reset_index(drop=True)
    n = len(res)
    x = np.arange(n)
    w = 0.35

    n_wt = res["n_wt"].iloc[0]
    n_mut = res["n_mut"].iloc[0]

    ax.bar(x - w/2, res["wt_mean"], w,
           yerr=res["wt_sem"], error_kw=dict(lw=0.6, capsize=2, capthick=0.6),
           label=f"EZH2-wt (n={n_wt})", color=C_WT, edgecolor="white", linewidth=0.3)
    ax.bar(x + w/2, res["mut_mean"], w,
           yerr=res["mut_sem"], error_kw=dict(lw=0.6, capsize=2, capthick=0.6),
           label=f"EZH2-mut (n={n_mut})", color=C_MUT, edgecolor="white", linewidth=0.3)

    for i, (_, row) in enumerate(res.iterrows()):
        top = max(row["wt_mean"] + row["wt_sem"], row["mut_mean"] + row["mut_sem"])
        if row["q"] < 0.05:
            stars = "*"
        elif row["pval"] < 0.05:
            stars = "\u2020"
        else:
            continue
        color = C_MUT if row["q"] < 0.05 else "#ff7f0e"
        ax.text(i, top + 0.003, stars, ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold" if row["q"] < 0.05 else "normal")

    labels = [m.replace("_", "-").replace("LQ / ", "") for m in res["marker"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel("Fraction per patient (± SEM)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(-0.6, n - 0.4)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_clinical_ezh2.csv"))
    ezh2_map = dict(zip(
        master.loc[master["EZH2"].isin(["wt", "mut"]), "slide_ID"],
        master.loc[master["EZH2"].isin(["wt", "mut"]), "EZH2"],
    ))

    # Load panels
    print("Loading T-panel...")
    adata_t = ad.read_h5ad("output/all_TMA_T_global_v8.h5ad")
    adata_t.obs["slide_ID"] = [normalize_sample_id(s) for s in adata_t.obs["sample_id"]]

    print("Loading S-panel...")
    adata_s = ad.read_h5ad("output/all_TMA_S_global_v8.h5ad")
    adata_s.obs["slide_ID"] = [normalize_sample_id(s) for s in adata_s.obs["sample_id"]]

    # B cell masks
    b_mask_t = get_bcell_mask(adata_t)
    b_mask_s = get_bcell_mask(adata_s)
    print(f"  T B cells: {b_mask_t.sum():,}; S B cells: {b_mask_s.sum():,}")

    # Compute stats
    print("Computing B cell markers...")
    t_bcell = compute_marker_stats(adata_t, ezh2_map, cell_mask=b_mask_t)
    s_bcell = compute_marker_stats(adata_s, ezh2_map, cell_mask=b_mask_s)

    print("Computing all-cell markers...")
    t_all = compute_marker_stats(adata_t, ezh2_map, cell_mask=None)
    s_all = compute_marker_stats(adata_s, ezh2_map, cell_mask=None)

    # Compartments
    print("Loading UTAG compartments...")
    utag_t = ad.read_h5ad("output/all_TMA_T_utag_ct_merged.h5ad")
    utag_t.obs["slide_ID"] = [normalize_sample_id(s) for s in utag_t.obs["sample_id"]]
    comp_col = "compartment_name" if "compartment_name" in utag_t.obs.columns else "tissue_compartment"
    comp_counts = utag_t.obs.groupby(["slide_ID", comp_col]).size().unstack(fill_value=0)
    comp_frac = comp_counts.div(comp_counts.sum(axis=1), axis=0)
    comp_frac["EZH2"] = comp_frac.index.map(ezh2_map)
    comp_frac = comp_frac[comp_frac["EZH2"].isin(["wt", "mut"])]

    compartments = [c for c in comp_frac.columns if c != "EZH2"]
    comp_results = []
    for comp in compartments:
        wt = comp_frac.loc[comp_frac["EZH2"] == "wt", comp].dropna()
        mut = comp_frac.loc[comp_frac["EZH2"] == "mut", comp].dropna()
        if len(wt) >= 5 and len(mut) >= 5:
            _, p = mannwhitneyu(wt, mut, alternative="two-sided")
            comp_results.append({
                "marker": comp, "wt_mean": wt.mean(), "wt_sem": wt.sem(),
                "mut_mean": mut.mean(), "mut_sem": mut.sem(),
                "pval": p, "n_wt": len(wt), "n_mut": len(mut),
            })
    comp_res = pd.DataFrame(comp_results).sort_values("pval")
    mc = len(comp_res)
    comp_res["rank"] = range(1, mc + 1)
    comp_res["q"] = (comp_res["pval"] * mc / comp_res["rank"]).clip(upper=1.0)
    comp_res["q"] = comp_res["q"][::-1].cummin()[::-1]

    # Print summary
    for name, df in [("T B-cell", t_bcell), ("S B-cell", s_bcell),
                     ("T all-cell", t_all), ("S all-cell", s_all),
                     ("Compartments", comp_res)]:
        n_nom = (df["pval"] < 0.05).sum()
        n_fdr = (df["q"] < 0.05).sum()
        print(f"  {name}: {len(df)} features, {n_nom} nominal, {n_fdr} FDR")

    # === FIGURE ===
    print("Creating figure...")
    fig = plt.figure(figsize=(28, 22))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.2,
                           height_ratios=[1, 1, 0.55])

    fig.suptitle("EZH2 Mutation: Effect on B Cell Markers and Tumor Microenvironment",
                 fontsize=17, fontweight="bold")

    # Row 1: B cell markers
    ax_t_b = fig.add_subplot(gs[0, 0])
    n_fdr_t = (t_bcell["q"] < 0.05).sum()
    draw_vertical_bars(ax_t_b, t_bcell,
                       f"(a) T-panel — Malignant B Cells ({n_fdr_t} FDR-significant)",
                       "Mean expression (z-scored ± SEM)")

    ax_s_b = fig.add_subplot(gs[0, 1])
    n_fdr_s = (s_bcell["q"] < 0.05).sum()
    draw_vertical_bars(ax_s_b, s_bcell,
                       f"(b) S-panel — Malignant B Cells ({n_fdr_s} FDR-significant)",
                       "Mean expression (z-scored ± SEM)")

    # Row 2: All-cell markers (TME effect)
    ax_t_a = fig.add_subplot(gs[1, 0])
    n_fdr_ta = (t_all["q"] < 0.05).sum()
    draw_vertical_bars(ax_t_a, t_all,
                       f"(c) T-panel — All Cells / TME ({n_fdr_ta} FDR-significant)",
                       "Mean expression (z-scored ± SEM)")

    ax_s_a = fig.add_subplot(gs[1, 1])
    n_fdr_sa = (s_all["q"] < 0.05).sum()
    draw_vertical_bars(ax_s_a, s_all,
                       f"(d) S-panel — All Cells / TME ({n_fdr_sa} FDR-significant)",
                       "Mean expression (z-scored ± SEM)")

    # Row 3: Compartments (spanning both columns)
    ax_comp = fig.add_subplot(gs[2, :])
    n_fdr_c = (comp_res["q"] < 0.05).sum()
    draw_compartment_bars(ax_comp, comp_res,
                          f"(e) Tissue Compartment Fractions — T-panel UTAG ({n_fdr_c} FDR-significant)")

    # Legend footnote
    fig.text(0.5, 0.005,
             "Sorted by p-value (most significant at left).   "
             "*** q<0.001   ** q<0.01   * q<0.05 (FDR)   "
             "\u2020 p<0.05 (nominal only)",
             ha="center", fontsize=10, color="#555555")

    out_path = os.path.join(OUTPUT_DIR, "ezh2_barplot_final.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
