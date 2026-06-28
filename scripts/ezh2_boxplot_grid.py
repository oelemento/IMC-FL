#!/usr/bin/env python3
"""
EZH2 vs all markers in malignant B cells — simple boxplot grid.

All markers side by side, sorted by p-value, FDR-significant highlighted.

Usage:
    PYTHONPATH=. python3.11 scripts/ezh2_boxplot_grid.py
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
C_WT = "#4DBEEE"
C_MUT = "#D95319"


def get_bcell_mask(adata):
    ct = adata.obs["cell_type"].astype(str)
    return ct.str.contains("B cell", case=False, na=False) | ct.str.contains("GC B", case=False, na=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load EZH2 mapping
    master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_clinical_ezh2.csv"))
    ezh2_map = dict(zip(
        master.loc[master["EZH2"].isin(["wt", "mut"]), "slide_ID"],
        master.loc[master["EZH2"].isin(["wt", "mut"]), "EZH2"],
    ))

    # Load and compute B cell marker means per patient
    all_markers = {}  # {(marker_name, panel): {"wt": array, "mut": array}}

    for panel_name, path in [("T", "output/all_TMA_T_global_v8.h5ad"),
                              ("S", "output/all_TMA_S_global_v8.h5ad")]:
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

        for marker in adata.var_names:
            wt_vals = per_patient.loc[per_patient["EZH2"] == "wt", marker].dropna().values
            mut_vals = per_patient.loc[per_patient["EZH2"] == "mut", marker].dropna().values
            # Skip DNA channels
            if marker.startswith("DNA"):
                continue
            all_markers[(marker, panel_name)] = {"wt": wt_vals, "mut": mut_vals}

    # Compute stats and sort
    results = []
    for (marker, panel), data in all_markers.items():
        if len(data["wt"]) >= 5 and len(data["mut"]) >= 5:
            _, p = mannwhitneyu(data["wt"], data["mut"], alternative="two-sided")
            results.append({
                "marker": marker, "panel": panel, "pval": p,
                "wt_med": np.median(data["wt"]), "mut_med": np.median(data["mut"]),
                "diff": np.median(data["mut"]) - np.median(data["wt"]),
            })
    res_df = pd.DataFrame(results).sort_values("pval")
    m = len(res_df)
    res_df["rank"] = range(1, m + 1)
    res_df["q"] = (res_df["pval"] * m / res_df["rank"]).clip(upper=1.0)
    res_df["q"] = res_df["q"][::-1].cummin()[::-1]

    print(f"\n{m} markers total, {(res_df['pval']<0.05).sum()} nominal, {(res_df['q']<0.05).sum()} FDR")

    # Grid layout
    ncols = 10
    nrows = int(np.ceil(m / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.6))
    fig.suptitle("EZH2-wt vs EZH2-mut: All Markers in Malignant B Cells (sorted by p-value)",
                 fontsize=16, fontweight="bold", y=1.0)

    for idx, (_, row) in enumerate(res_df.iterrows()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        key = (row["marker"], row["panel"])
        wt = all_markers[key]["wt"]
        mut = all_markers[key]["mut"]

        bp = ax.boxplot([wt, mut], tick_labels=["wt", "mut"],
                        patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.2))
        bp["boxes"][0].set_facecolor(C_WT)
        bp["boxes"][1].set_facecolor(C_MUT)
        for b in bp["boxes"]:
            b.set_linewidth(0.6)
        for w in bp["whiskers"] + bp["caps"]:
            w.set_linewidth(0.5)

        # Jitter
        for j, (vals, xpos) in enumerate([(wt, 1), (mut, 2)]):
            jitter = np.random.normal(xpos, 0.06, len(vals))
            ax.scatter(jitter, vals, alpha=0.3, s=6,
                       color=bp["boxes"][j].get_facecolor(), edgecolors="none", zorder=2)

        # Title with marker name and p-value
        display = row["marker"].replace("_", "-")
        panel_tag = f" [{row['panel']}]"

        if row["q"] < 0.05:
            # FDR-significant: red border + bold
            for spine in ax.spines.values():
                spine.set_edgecolor("#d62728")
                spine.set_linewidth(2.5)
            ax.set_title(f"{display}{panel_tag}\np={row['pval']:.1e} q={row['q']:.3f}",
                         fontsize=7.5, fontweight="bold", color="#d62728")
        elif row["pval"] < 0.05:
            # Nominal: orange border
            for spine in ax.spines.values():
                spine.set_edgecolor("#ff7f0e")
                spine.set_linewidth(1.8)
            ax.set_title(f"{display}{panel_tag}\np={row['pval']:.3f}",
                         fontsize=7.5, fontweight="bold", color="#ff7f0e")
        else:
            ax.set_title(f"{display}{panel_tag}\np={row['pval']:.2f}",
                         fontsize=7.5, color="#888888")

        ax.tick_params(labelsize=7)
        ax.set_ylabel("")

    # Hide unused axes
    for idx in range(m, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(OUTPUT_DIR, "ezh2_bcell_boxplot_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
