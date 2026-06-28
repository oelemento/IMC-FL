#!/usr/bin/env python
"""Plot UTAG domains for B1_FL11 at multiple resolutions, side-by-side with cell types."""
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "<PROJECT_ROOT>/output"
UTAG_FILE = f"{OUTDIR}/all_TMA_T_utag.h5ad"
ROI = "B1_FL11"

print(f"Loading UTAG results...")
adata = ad.read_h5ad(UTAG_FILE)
roi = adata[adata.obs["sample_id"] == ROI].copy()
print(f"{ROI}: {roi.shape[0]} cells")

# Get spatial coords
x = roi.obs["centroid_x"].values if "centroid_x" in roi.obs else roi.obsm["spatial"][:, 0]
y = roi.obs["centroid_y"].values if "centroid_y" in roi.obs else roi.obsm["spatial"][:, 1]

# Cell type colors
ct_colors = {
    "B cells": "#FFB347", "B cells (CXCR5hi)": "#4A90D9", "B cells (CD20hi)": "#87CEEB",
    "B (weak CD20)": "#DEB887", "GC B cells": "#FF8C00", "Activated B": "#FF6347",
    "B (TOXhi)": "#CD853F",
    "CD4 T cells": "#9370DB", "CD8 T cells": "#90EE90", "CD8 T exhausted": "#FFB6C1",
    "CD8 T pre-exhausted (TOX+)": "#FF69B4",
    "Treg": "#DC143C", "Macrophages": "#228B22", "LQ": "#D3D3D3",
    "Mixed": "#A9A9A9", "Other": "#C0C0C0", "Cytotoxic": "#006400",
}

fig, axes = plt.subplots(1, 4, figsize=(28, 7))

# Panel 1: Cell types
ax = axes[0]
for ct in roi.obs["cell_type"].unique():
    mask = roi.obs["cell_type"] == ct
    color = ct_colors.get(ct, "#808080")
    ax.scatter(x[mask], y[mask], c=color, s=1, alpha=0.6, label=ct, rasterized=True)
ax.set_title(f"{ROI} — Cell Types (v8)", fontweight="bold", fontsize=11)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xticks([]); ax.set_yticks([])

# Panels 2-4: UTAG domains at 3 resolutions
for i, res in enumerate(["0.05", "0.1", "0.2"]):
    ax = axes[i + 1]
    col = f"UTAG Label_leiden_{res}"
    if col not in roi.obs.columns:
        ax.set_title(f"Missing: {col}")
        continue
    domains = roi.obs[col].astype(str)
    unique_domains = sorted(domains.unique(), key=lambda x: int(x))
    cmap = plt.cm.get_cmap("tab20", len(unique_domains))
    colors = {d: cmap(j) for j, d in enumerate(unique_domains)}
    for d in unique_domains:
        mask = domains == d
        ax.scatter(x[mask], y[mask], c=[colors[d]], s=1, alpha=0.6, label=d, rasterized=True)
    n_domains = len(unique_domains)
    ax.set_title(f"{ROI} — UTAG res={res} ({n_domains} domains here)", fontweight="bold", fontsize=11)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])

# Add legend for cell types
handles, labels = axes[0].get_legend_handles_labels()
# Sort by frequency
ct_counts = roi.obs["cell_type"].value_counts()
order = [labels.index(ct) for ct in ct_counts.index if ct in labels]
axes[0].legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper left", fontsize=6, markerscale=3, framealpha=0.8)

plt.suptitle(f"UTAG Tissue Domains — {ROI} (T-panel v8)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
outpath = f"{OUTDIR}/FL11_utag_domains.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {outpath}")
