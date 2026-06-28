#!/usr/bin/env python
"""Sweep max_dist for UTAG on B1 T-panel, visualize FL11."""
import time
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "<PROJECT_ROOT>/output"
INPUT = f"{OUTDIR}/all_TMA_T_global_v8.h5ad"

print("=== Loading T-panel v8 (B1 only) ===")
t0 = time.time()
adata = ad.read_h5ad(INPUT)
b1 = adata[adata.obs["tma"] == "B1"].copy()
del adata
print(f"B1: {b1.shape[0]} cells, {b1.obs['sample_id'].nunique()} ROIs, loaded in {time.time()-t0:.0f}s")

from utag import utag as run_utag

MAX_DISTS = [20, 50, 100, 150]
RES = 0.5

results = {}
for md in MAX_DISTS:
    t1 = time.time()
    print(f"\n=== max_dist={md} ===")
    result = run_utag(
        b1.copy(),
        slide_key="sample_id",
        max_dist=md,
        normalization_mode="l2_norm",
        apply_clustering=True,
        clustering_method="leiden",
        resolutions=[RES],
        leiden_kwargs=dict(flavor="igraph", n_iterations=2, directed=False),
    )
    col = f"UTAG Label_leiden_{RES}"
    n_domains = result.obs[col].nunique()
    elapsed = time.time() - t1
    print(f"  {n_domains} domains in {elapsed:.0f}s")
    results[md] = result

# ---- Plot FL11 for each max_dist ----
print("\n=== Plotting FL11 ===")

ct_colors = {
    "B cells": "#FFB347", "B cells (CXCR5hi)": "#4A90D9", "B cells (CD20hi)": "#87CEEB",
    "B (weak CD20)": "#DEB887", "GC B cells": "#FF8C00", "Activated B": "#FF6347",
    "B (TOXhi)": "#CD853F", "CD4 T cells": "#9370DB", "CD8 T cells": "#90EE90",
    "CD8 T exhausted": "#FFB6C1", "CD8 T pre-exhausted (TOX+)": "#FF69B4",
    "Treg": "#DC143C", "Macrophages": "#228B22", "LQ": "#D3D3D3",
    "Mixed": "#A9A9A9", "Other": "#C0C0C0", "Cytotoxic": "#006400",
}

n_cols = 1 + len(MAX_DISTS)
fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))

# Panel 0: Cell types
roi_mask = b1.obs["sample_id"] == "B1_FL11"
roi = b1[roi_mask]
x = roi.obsm["spatial"][:, 0] if "spatial" in roi.obsm else roi.obs["centroid_x"].values
y = roi.obsm["spatial"][:, 1] if "spatial" in roi.obsm else roi.obs["centroid_y"].values

ax = axes[0]
for ct in roi.obs["cell_type"].unique():
    m = roi.obs["cell_type"] == ct
    ax.scatter(x[m], y[m], c=ct_colors.get(ct, "#808080"), s=2, alpha=0.7, label=ct, rasterized=True)
ax.set_title("B1_FL11 — Cell Types (v8)", fontweight="bold", fontsize=11)
ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
handles, labels = ax.get_legend_handles_labels()
ct_counts = roi.obs["cell_type"].value_counts()
order = [labels.index(ct) for ct in ct_counts.index if ct in labels]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          loc="upper left", fontsize=5, markerscale=2, framealpha=0.8)

# Panels 1-4: UTAG domains at each max_dist
for i, md in enumerate(MAX_DISTS):
    ax = axes[i + 1]
    result = results[md]
    col = f"UTAG Label_leiden_{RES}"

    roi_result = result[result.obs["sample_id"] == "B1_FL11"]
    domains = roi_result.obs[col].astype(str)
    unique_d = sorted(domains.unique(), key=lambda d: int(d))
    n_d = len(unique_d)

    # Use a colormap with enough colors
    if n_d <= 20:
        cmap = plt.colormaps.get_cmap("tab20")
    else:
        cmap = plt.colormaps.get_cmap("gist_ncar")
    colors = {d: cmap(j / max(n_d - 1, 1)) for j, d in enumerate(unique_d)}

    rx = roi_result.obsm["spatial"][:, 0] if "spatial" in roi_result.obsm else roi_result.obs["centroid_x"].values
    ry = roi_result.obsm["spatial"][:, 1] if "spatial" in roi_result.obsm else roi_result.obs["centroid_y"].values

    for d in unique_d:
        m = domains == d
        ax.scatter(rx[m.values], ry[m.values], c=[colors[d]], s=2, alpha=0.7, rasterized=True)
    ax.set_title(f"max_dist={md}µm — {n_d} domains", fontweight="bold", fontsize=11)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(f"UTAG max_dist Sweep — B1_FL11 (T-panel v8, leiden res={RES})",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
outpath = f"{OUTDIR}/utag_maxdist_sweep_FL11.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {outpath}")

# Print domain counts for each max_dist
print("\n=== Summary ===")
for md in MAX_DISTS:
    col = f"UTAG Label_leiden_{RES}"
    n_global = results[md].obs[col].nunique()
    roi_result = results[md][results[md].obs["sample_id"] == "B1_FL11"]
    n_local = roi_result.obs[col].nunique()
    print(f"  max_dist={md:>3d}: {n_global} global domains, {n_local} in FL11")

total = time.time() - t0
print(f"\n=== DONE (total: {total:.0f}s / {total/60:.1f} min) ===")
