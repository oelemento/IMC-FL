#!/usr/bin/env python3
"""
EZH2 mutation vs marker expression in malignant B cells only.

EZH2 mutations are IN the tumor B cells, so epigenetic effects
should be most visible in B cell marker expression — not diluted
by all cell types.

Usage:
    PYTHONPATH=. python3.11 scripts/ezh2_bcell_markers.py
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


def get_bcell_mask(adata):
    """Return boolean mask for all B cell subtypes (malignant B cells)."""
    ct = adata.obs["cell_type"].astype(str)
    b_types = ct.str.contains("B cell", case=False, na=False)
    gc_b = ct.str.contains("GC B", case=False, na=False)
    return b_types | gc_b


def extract_bcell_marker_means(adata, panel_name):
    """Compute mean marker expression per patient, restricted to B cells."""
    adata.obs["slide_ID"] = [normalize_sample_id(s) for s in adata.obs["sample_id"]]

    mask = get_bcell_mask(adata)
    print(f"  {panel_name}: {mask.sum():,} B cells out of {adata.n_obs:,} total ({100*mask.mean():.1f}%)")

    # Print B cell subtypes
    b_subtypes = adata.obs.loc[mask, "cell_type"].value_counts()
    for ct, n in b_subtypes.items():
        print(f"    {ct}: {n:,}")

    bcells = adata[mask]

    # Per-patient mean marker expression in B cells
    marker_means = bcells.obs.groupby("slide_ID").apply(
        lambda g: pd.Series(
            np.asarray(adata[g.index].X.mean(axis=0)).flatten(),
            index=adata.var_names
        )
    )

    # Also get per-patient B cell count and B cell fraction
    b_counts = bcells.obs.groupby("slide_ID").size()
    total_counts = adata.obs.groupby("slide_ID").size()
    b_frac = b_counts / total_counts

    marker_means.columns = [f"{panel_name}_{c}" for c in marker_means.columns]
    marker_means[f"{panel_name}_B_cell_count"] = b_counts
    marker_means[f"{panel_name}_B_cell_fraction"] = b_frac

    return marker_means


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load EZH2 data
    master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_clinical_ezh2.csv"))
    ezh2_map = master[master["EZH2"].isin(["wt", "mut"])][["slide_ID", "EZH2"]].drop_duplicates("slide_ID")
    print(f"EZH2 data: {len(ezh2_map)} patients ({(ezh2_map['EZH2']=='mut').sum()} mut, {(ezh2_map['EZH2']=='wt').sum()} wt)")

    # Load panels
    print("\nLoading T-panel...")
    adata_t = ad.read_h5ad("output/all_TMA_T_global_v8.h5ad")
    t_means = extract_bcell_marker_means(adata_t, "T")

    print("\nLoading S-panel...")
    adata_s = ad.read_h5ad("output/all_TMA_S_global_v8.h5ad")
    s_means = extract_bcell_marker_means(adata_s, "S")

    # Merge
    combined = ezh2_map.set_index("slide_ID")
    combined = combined.join(t_means, how="left").join(s_means, how="left")
    print(f"\nCombined: {len(combined)} patients × {combined.shape[1]} columns")

    # Feature columns (markers only, not count/fraction)
    marker_cols = [c for c in combined.columns
                   if c not in ["EZH2"] and "B_cell_count" not in c and "B_cell_fraction" not in c]

    # Test EZH2-wt vs mut
    wt = combined[combined["EZH2"] == "wt"]
    mut = combined[combined["EZH2"] == "mut"]

    results = []
    for col in marker_cols:
        wt_vals = wt[col].dropna()
        mut_vals = mut[col].dropna()
        if len(wt_vals) >= 5 and len(mut_vals) >= 5:
            stat, pval = mannwhitneyu(wt_vals, mut_vals, alternative="two-sided")
            results.append({
                "marker": col,
                "panel": col.split("_")[0],
                "marker_name": "_".join(col.split("_")[1:]),
                "wt_median": wt_vals.median(),
                "mut_median": mut_vals.median(),
                "diff": mut_vals.median() - wt_vals.median(),
                "wt_mean": wt_vals.mean(),
                "mut_mean": mut_vals.mean(),
                "pval": pval,
                "n_wt": len(wt_vals),
                "n_mut": len(mut_vals),
            })

    results_df = pd.DataFrame(results).sort_values("pval")

    # BH correction
    m = len(results_df)
    results_df["rank"] = range(1, m + 1)
    results_df["q_value"] = (results_df["pval"] * m / results_df["rank"]).clip(upper=1.0)
    results_df["q_value"] = results_df["q_value"][::-1].cummin()[::-1]

    # Print results
    print(f"\n{'='*80}")
    print("EZH2-wt vs EZH2-mut: Marker expression in MALIGNANT B CELLS")
    print(f"{'='*80}")
    print(f"\nAll {m} markers tested:")
    print("-" * 100)
    for _, row in results_df.iterrows():
        sig = " ***" if row["q_value"] < 0.001 else (" **" if row["q_value"] < 0.01 else (" *" if row["q_value"] < 0.05 else (" ." if row["pval"] < 0.05 else "")))
        print(f"  {row['marker_name']:25s} ({row['panel']})  p={row['pval']:.4f}  q={row['q_value']:.4f}  "
              f"wt={row['wt_median']:+.3f}  mut={row['mut_median']:+.3f}  Δ={row['diff']:+.3f}{sig}")

    n_sig = (results_df["pval"] < 0.05).sum()
    n_fdr = (results_df["q_value"] < 0.05).sum()
    print(f"\nNominal p < 0.05: {n_sig}/{m}")
    print(f"FDR q < 0.05: {n_fdr}/{m}")

    results_df.to_csv(os.path.join(OUTPUT_DIR, "ezh2_bcell_marker_tests.csv"), index=False)

    # Figure
    make_figure(combined, results_df, wt, mut, OUTPUT_DIR)


def make_figure(combined, results_df, wt, mut, output_dir):
    """Create figure showing EZH2 vs B cell marker expression."""

    # Separate T-panel and S-panel results
    t_results = results_df[results_df["panel"] == "T"].copy()
    s_results = results_df[results_df["panel"] == "S"].copy()

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle("EZH2 Mutation vs Marker Expression in Malignant B Cells",
                 fontsize=16, fontweight="bold", y=0.98)

    # Layout: 3 rows
    # Row 1: (a) T-panel heatmap-style barplot, (b) S-panel heatmap-style barplot
    # Row 2: (c) Top significant boxplots, (d) Volcano
    # Row 3: (e) Key markers detailed boxplots

    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    # (a) T-panel markers ranked by p-value
    ax = fig.add_subplot(gs[0, 0:2])
    if len(t_results) > 0:
        t_sorted = t_results.sort_values("pval")
        labels = [r["marker_name"] for _, r in t_sorted.iterrows()]
        diffs = t_sorted["diff"].values
        pvals = t_sorted["pval"].values
        colors = ["#d62728" if p < 0.05 else "#7f7f7f" for p in pvals]
        y = range(len(labels))
        ax.barh(y, diffs, color=colors, edgecolor="black", linewidth=0.3, height=0.7)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="gray", linestyle="-", alpha=0.5)
        for i, p in enumerate(pvals):
            if p < 0.05:
                ax.text(diffs[i] + (0.02 if diffs[i] >= 0 else -0.02), i,
                        f"p={p:.3f}", fontsize=6, va="center",
                        ha="left" if diffs[i] >= 0 else "right", color="#d62728")
        ax.set_xlabel("Median difference in B cells (mut − wt)", fontsize=10)
        ax.invert_yaxis()
    ax.set_title(f"(a) T-panel markers ({len(t_results)} markers)", fontsize=12)

    # (b) S-panel markers ranked by p-value
    ax = fig.add_subplot(gs[0, 2:4])
    if len(s_results) > 0:
        s_sorted = s_results.sort_values("pval")
        labels = [r["marker_name"] for _, r in s_sorted.iterrows()]
        diffs = s_sorted["diff"].values
        pvals = s_sorted["pval"].values
        colors = ["#d62728" if p < 0.05 else "#7f7f7f" for p in pvals]
        y = range(len(labels))
        ax.barh(y, diffs, color=colors, edgecolor="black", linewidth=0.3, height=0.7)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="gray", linestyle="-", alpha=0.5)
        for i, p in enumerate(pvals):
            if p < 0.05:
                ax.text(diffs[i] + (0.02 if diffs[i] >= 0 else -0.02), i,
                        f"p={p:.3f}", fontsize=6, va="center",
                        ha="left" if diffs[i] >= 0 else "right", color="#d62728")
        ax.set_xlabel("Median difference in B cells (mut − wt)", fontsize=10)
        ax.invert_yaxis()
    ax.set_title(f"(b) S-panel markers ({len(s_results)} markers)", fontsize=12)

    # (c) Volcano plot — both panels combined
    ax = fig.add_subplot(gs[1, 0:2])
    df = results_df.copy()
    df["-log10p"] = -np.log10(df["pval"].clip(lower=1e-10))
    panel_colors = {"T": "#1f77b4", "S": "#ff7f0e"}
    for panel in ["T", "S"]:
        sub = df[df["panel"] == panel]
        sig = sub[sub["pval"] < 0.05]
        ns = sub[sub["pval"] >= 0.05]
        ax.scatter(ns["diff"], ns["-log10p"], c=panel_colors[panel], alpha=0.3, s=20,
                   edgecolors="none", label=f"{panel}-panel (NS)")
        ax.scatter(sig["diff"], sig["-log10p"], c=panel_colors[panel], alpha=0.9, s=40,
                   edgecolors="black", linewidths=0.5, label=f"{panel}-panel (p<0.05)")
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.3)
    # Label significant hits
    sig_df = df[df["pval"] < 0.05]
    for _, row in sig_df.iterrows():
        ax.annotate(row["marker_name"], (row["diff"], row["-log10p"]),
                    fontsize=7, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("Median difference (mut − wt)", fontsize=10)
    ax.set_ylabel("-log10(p-value)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    n_fdr = (df["q_value"] < 0.05).sum()
    ax.set_title(f"(c) Volcano plot — B cell markers\n{len(df)} markers, {(df['pval']<0.05).sum()} nominal, {n_fdr} FDR", fontsize=11)

    # (d-h) Boxplots of top significant markers
    top_markers = results_df[results_df["pval"] < 0.1].head(6)
    for i, (_, row) in enumerate(top_markers.iterrows()):
        r = i // 3 + 1
        c = i % 3 + (1 if i < 3 else -2)  # positions in grid
        if i < 3:
            ax = fig.add_subplot(gs[1, 2 + i % 1]) if i == 0 else (
                fig.add_subplot(gs[1, 3]) if i == 1 else fig.add_subplot(gs[2, 0]))
        # Simpler: use last two rows of grid
        if i < 2:
            ax = fig.add_subplot(gs[1, 2 + i])
        else:
            ax = fig.add_subplot(gs[2, i - 2])

        feat = row["marker"]
        wt_vals = wt[feat].dropna()
        mut_vals = mut[feat].dropna()

        bp = ax.boxplot([wt_vals, mut_vals],
                        tick_labels=["EZH2-wt", "EZH2-mut"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4DBEEE")
        bp["boxes"][1].set_facecolor("#D95319")
        # Jitter points
        for j, (vals, xpos) in enumerate([(wt_vals, 1), (mut_vals, 2)]):
            ax.scatter(np.random.normal(xpos, 0.04, len(vals)), vals,
                       alpha=0.3, s=10, color=bp["boxes"][j].get_facecolor(), zorder=2)

        sig = "*" if row["pval"] < 0.05 else ""
        fdr_note = f" (q={row['q_value']:.3f})" if row["q_value"] < 0.1 else ""
        ax.set_title(f"{row['marker_name']} ({row['panel']}-panel)\n"
                     f"p={row['pval']:.4f}{sig}{fdr_note}", fontsize=10)
        ax.set_ylabel("Expression (z-scored)", fontsize=9)
        ax.text(0.5, 0.02, f"wt: {wt_vals.median():.3f} | mut: {mut_vals.median():.3f}",
                transform=ax.transAxes, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, "ezh2_bcell_markers.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()
