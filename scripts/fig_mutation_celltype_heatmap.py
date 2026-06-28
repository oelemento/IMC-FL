"""Supplementary figure: Mutation × cell type association heatmap.

Shows log2 fold change (mut/wt) for top mutated genes vs cell type fractions.
Nothing survives FDR correction — this is a negative result demonstrating
that mutations do not have large effects on cell composition.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22



def main():
    df = pd.read_csv("output/cd14_validation/mutation_celltype_associations.csv")
    print(f"Loaded {len(df)} associations")
    print(f"FDR < 0.05: {(df['q'] < 0.05).sum()}, nominal p < 0.05: {(df['p'] < 0.05).sum()}")

    # Consolidate cell types across panels into aggregate groups
    # Use S-panel myeloid subtypes (more detailed) and T-panel lymphocytes
    keep_ct = {
        # T-panel
        "CD8 T cells", "CD4 T cells", "Treg", "CD8 T exhausted",
        "GC B cells", "B cells", "Macrophages",
        "Macrophages (GzmB+)", "CD8 T pre-exhausted (TOX+)",
        # S-panel
        "M1 Macrophages", "M2 Macrophages", "Myeloid (S100A9+)",
        "FDC", "Dendritic cells", "B cells (BCL2+)", "B cells (PAX5+)",
    }
    ct_short = {
        "CD8 T cells": "CD8 T", "CD4 T cells": "CD4 T", "Treg": "Treg",
        "CD8 T exhausted": "CD8 T exh", "GC B cells": "GC B",
        "B cells": "B cells (T)", "Macrophages": "Mac (T)",
        "Macrophages (GzmB+)": "GzmB+ CD8",
        "CD8 T pre-exhausted (TOX+)": "CD8 pre-exh",
        "M1 Macrophages": "M1 Mac", "M2 Macrophages": "M2 Mac",
        "Myeloid (S100A9+)": "S100A9+", "FDC": "FDC",
        "Dendritic cells": "DC", "B cells (BCL2+)": "BCL2+ B",
        "B cells (PAX5+)": "PAX5+ B",
    }

    sub = df[df["cell_type"].isin(keep_ct)].copy()
    sub["ct_short"] = sub["cell_type"].map(ct_short)
    sub["log2fc"] = np.log2(sub["fold_change"].clip(lower=0.1, upper=10))

    # Order genes by mutation frequency (most common first)
    gene_order = ["CREBBP", "KMT2D", "BCL2", "TNFRSF14", "EZH2", "FOXO1",
                  "MEF2B", "STAT6", "CARD11", "SOCS1", "IRF8", "BCL7A",
                  "ARID1A", "FAT4", "HVCN1"]
    gene_order = [g for g in gene_order if g in sub["gene"].unique()]

    # Order cell types: T-panel lymphocytes, then S-panel myeloid/stromal
    ct_order = ["CD8 T", "CD4 T", "Treg", "CD8 T exh", "CD8 pre-exh",
                "GzmB+ CD8", "GC B", "B cells (T)",
                "M1 Mac", "M2 Mac", "Mac (T)", "S100A9+", "DC", "FDC",
                "BCL2+ B", "PAX5+ B"]
    ct_order = [c for c in ct_order if c in sub["ct_short"].unique()]

    # Build matrices
    fc_mat = np.full((len(gene_order), len(ct_order)), np.nan)
    p_mat = np.ones((len(gene_order), len(ct_order)))

    for _, row in sub.iterrows():
        g = row["gene"]
        c = row["ct_short"]
        if g in gene_order and c in ct_order:
            gi = gene_order.index(g)
            ci = ct_order.index(c)
            fc_mat[gi, ci] = row["log2fc"]
            p_mat[gi, ci] = row["p"]

    # Figure
    fig, ax = plt.subplots(figsize=(14, 8))

    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    im = ax.imshow(fc_mat, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="nearest")

    # Overlay significance dots
    for i in range(len(gene_order)):
        for j in range(len(ct_order)):
            p = p_mat[i, j]
            if p < 0.001:
                ax.text(j, i, "***", ha="center", va="center", fontsize=7,
                        fontweight="bold", color="black")
            elif p < 0.01:
                ax.text(j, i, "**", ha="center", va="center", fontsize=7,
                        fontweight="bold", color="black")
            elif p < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=7,
                        color="black")

    ax.set_xticks(range(len(ct_order)))
    ax.set_xticklabels(ct_order, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_yticks(range(len(gene_order)))
    ax.set_yticklabels(gene_order, fontsize=TICK_SIZE)

    # Divider between T-panel and S-panel cell types
    t_panel_end = ct_order.index("B cells (T)") + 0.5
    ax.axvline(t_panel_end, color="black", linewidth=1.5, linestyle="--")
    ax.text(t_panel_end / 2, -1.5, "T-panel", ha="center", fontsize=9,
            fontstyle="italic", color="#555")
    ax.text((t_panel_end + len(ct_order)) / 2, -1.5, "S-panel", ha="center",
            fontsize=9, fontstyle="italic", color="#555")

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="both")
    cb.set_label("log₂(fold change: mutant / wild-type)", fontsize=10)

    ax.set_title("(Mann-Whitney U, nominal *p<0.05 **p<0.01 ***p<0.001)",
                 fontsize=ANNOT_SIZE)

    plt.tight_layout()
    out = Path("output/hypotheses_v8/fig_mutation_celltype_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
