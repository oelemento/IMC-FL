#!/usr/bin/env python
"""UTAG on cell-type indicator features (one-hot encoded).

Instead of raw marker expression, use one-hot cell type vectors as input.
After UTAG message passing (AX), each cell's feature vector becomes the
neighborhood cell-type composition — exactly what defines tissue compartments.

This should resolve follicular structure in FL tissue where raw markers
fail because ~90% B cells homogenize during message passing.
"""
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

# ---- Create one-hot cell type features ----
print("\n=== Creating one-hot cell type features ===")
ct_dummies = pd.get_dummies(b1.obs["cell_type"]).astype(np.float32)
print(f"Cell types: {list(ct_dummies.columns)}")
print(f"Feature matrix shape: {ct_dummies.shape}")

# Create a new AnnData with cell-type indicators as .X
b1_ct = ad.AnnData(
    X=ct_dummies.values,
    obs=b1.obs.copy(),
    obsm={"spatial": b1.obsm["spatial"]} if "spatial" in b1.obsm else {},
    var=pd.DataFrame(index=ct_dummies.columns),
)

# Copy spatial coordinates if in obs
if "centroid_x" in b1.obs.columns:
    b1_ct.obs["centroid_x"] = b1.obs["centroid_x"].values
    b1_ct.obs["centroid_y"] = b1.obs["centroid_y"].values
if "spatial" not in b1_ct.obsm and "centroid_x" in b1.obs.columns:
    b1_ct.obsm["spatial"] = np.column_stack([
        b1.obs["centroid_x"].values,
        b1.obs["centroid_y"].values,
    ])

del b1  # free memory

from utag import utag as run_utag

# Test multiple resolutions
RESOLUTIONS = [0.3, 0.5, 1.0, 2.0]

print(f"\n=== Running UTAG on cell-type features (max_dist=20, res={RESOLUTIONS}) ===")
t1 = time.time()
result = run_utag(
    b1_ct,
    slide_key="sample_id",
    max_dist=20,
    normalization_mode="l2_norm",
    apply_clustering=True,
    clustering_method="leiden",
    resolutions=RESOLUTIONS,
    leiden_kwargs=dict(flavor="igraph", n_iterations=2, directed=False),
)
elapsed = time.time() - t1
print(f"UTAG done in {elapsed:.0f}s")

for res in RESOLUTIONS:
    col = f"UTAG Label_leiden_{res}"
    if col in result.obs.columns:
        n = result.obs[col].nunique()
        print(f"  res={res}: {n} domains")

# ---- Plot FL11 ----
print("\n=== Plotting FL11 ===")

ct_colors = {
    "B cells": "#FFB347", "B cells (CXCR5hi)": "#4A90D9", "B cells (CD20hi)": "#87CEEB",
    "B (weak CD20)": "#DEB887", "GC B cells": "#FF8C00", "Activated B": "#FF6347",
    "B (TOXhi)": "#CD853F", "CD4 T cells": "#9370DB", "CD8 T cells": "#90EE90",
    "CD8 T exhausted": "#FFB6C1", "CD8 T pre-exhausted (TOX+)": "#FF69B4",
    "Treg": "#DC143C", "Macrophages": "#228B22", "LQ": "#D3D3D3",
    "Mixed": "#A9A9A9", "Other": "#C0C0C0", "Cytotoxic": "#006400",
}

n_res = len(RESOLUTIONS)
fig, axes = plt.subplots(1, 1 + n_res, figsize=(7 * (1 + n_res), 7))

# Panel 0: Cell types
roi_mask = result.obs["sample_id"] == "B1_FL11"
roi = result[roi_mask]
x = roi.obsm["spatial"][:, 0]
y = roi.obsm["spatial"][:, 1]

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

# Panels 1+: UTAG domains at each resolution
for i, res in enumerate(RESOLUTIONS):
    ax = axes[i + 1]
    col = f"UTAG Label_leiden_{res}"
    if col not in result.obs.columns:
        ax.set_title(f"res={res} — N/A")
        continue

    domains = roi.obs[col].astype(str)
    unique_d = sorted(domains.unique(), key=lambda d: int(d))
    n_d = len(unique_d)

    if n_d <= 20:
        cmap = plt.colormaps.get_cmap("tab20")
    else:
        cmap = plt.colormaps.get_cmap("gist_ncar")
    colors = {d: cmap(j / max(n_d - 1, 1)) for j, d in enumerate(unique_d)}

    for d in unique_d:
        m = domains == d
        ax.scatter(x[m.values], y[m.values], c=[colors[d]], s=2, alpha=0.7, rasterized=True)
    ax.set_title(f"res={res} — {n_d} domains", fontweight="bold", fontsize=11)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("UTAG on Cell-Type Features — B1_FL11 (T-panel v8, max_dist=20)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
outpath = f"{OUTDIR}/utag_celltype_features_FL11.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {outpath}")

# ---- Also try max_dist=50 to see if larger neighborhoods help with cell-type features ----
print("\n=== Testing max_dist=50 with cell-type features ===")
t2 = time.time()

# Reload b1_ct (was consumed by UTAG)
b1_reload = ad.read_h5ad(INPUT)
b1_reload = b1_reload[b1_reload.obs["tma"] == "B1"].copy()
ct_dummies2 = pd.get_dummies(b1_reload.obs["cell_type"]).astype(np.float32)
b1_ct2 = ad.AnnData(
    X=ct_dummies2.values,
    obs=b1_reload.obs.copy(),
    var=pd.DataFrame(index=ct_dummies2.columns),
)
if "spatial" in b1_reload.obsm:
    b1_ct2.obsm["spatial"] = b1_reload.obsm["spatial"].copy()
elif "centroid_x" in b1_reload.obs.columns:
    b1_ct2.obsm["spatial"] = np.column_stack([
        b1_reload.obs["centroid_x"].values,
        b1_reload.obs["centroid_y"].values,
    ])
del b1_reload

result50 = run_utag(
    b1_ct2,
    slide_key="sample_id",
    max_dist=50,
    normalization_mode="l2_norm",
    apply_clustering=True,
    clustering_method="leiden",
    resolutions=RESOLUTIONS,
    leiden_kwargs=dict(flavor="igraph", n_iterations=2, directed=False),
)
print(f"max_dist=50 done in {time.time()-t2:.0f}s")

for res in RESOLUTIONS:
    col = f"UTAG Label_leiden_{res}"
    if col in result50.obs.columns:
        n = result50.obs[col].nunique()
        print(f"  res={res}: {n} domains")

# Plot FL11 for max_dist=50
fig2, axes2 = plt.subplots(1, 1 + n_res, figsize=(7 * (1 + n_res), 7))

roi50 = result50[result50.obs["sample_id"] == "B1_FL11"]
x50 = roi50.obsm["spatial"][:, 0]
y50 = roi50.obsm["spatial"][:, 1]

ax = axes2[0]
for ct in roi50.obs["cell_type"].unique():
    m = roi50.obs["cell_type"] == ct
    ax.scatter(x50[m], y50[m], c=ct_colors.get(ct, "#808080"), s=2, alpha=0.7, label=ct, rasterized=True)
ax.set_title("B1_FL11 — Cell Types (v8)", fontweight="bold", fontsize=11)
ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
handles, labels = ax.get_legend_handles_labels()
ct_counts = roi50.obs["cell_type"].value_counts()
order = [labels.index(ct) for ct in ct_counts.index if ct in labels]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          loc="upper left", fontsize=5, markerscale=2, framealpha=0.8)

for i, res in enumerate(RESOLUTIONS):
    ax = axes2[i + 1]
    col = f"UTAG Label_leiden_{res}"
    if col not in result50.obs.columns:
        ax.set_title(f"res={res} — N/A")
        continue

    domains = roi50.obs[col].astype(str)
    unique_d = sorted(domains.unique(), key=lambda d: int(d))
    n_d = len(unique_d)

    if n_d <= 20:
        cmap = plt.colormaps.get_cmap("tab20")
    else:
        cmap = plt.colormaps.get_cmap("gist_ncar")
    colors = {d: cmap(j / max(n_d - 1, 1)) for j, d in enumerate(unique_d)}

    for d in unique_d:
        m = domains == d
        ax.scatter(x50[m.values], y50[m.values], c=[colors[d]], s=2, alpha=0.7, rasterized=True)
    ax.set_title(f"res={res} — {n_d} domains", fontweight="bold", fontsize=11)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("UTAG on Cell-Type Features — B1_FL11 (T-panel v8, max_dist=50)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
outpath2 = f"{OUTDIR}/utag_celltype_features_md50_FL11.png"
fig2.savefig(outpath2, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Saved to {outpath2}")

# Summary
print("\n=== Summary ===")
print("max_dist=20:")
for res in RESOLUTIONS:
    col = f"UTAG Label_leiden_{res}"
    if col in result.obs.columns:
        n_global = result.obs[col].nunique()
        n_fl11 = result[result.obs["sample_id"] == "B1_FL11"].obs[col].nunique()
        print(f"  res={res}: {n_global} global, {n_fl11} in FL11")

print("max_dist=50:")
for res in RESOLUTIONS:
    col = f"UTAG Label_leiden_{res}"
    if col in result50.obs.columns:
        n_global = result50.obs[col].nunique()
        n_fl11 = result50[result50.obs["sample_id"] == "B1_FL11"].obs[col].nunique()
        print(f"  res={res}: {n_global} global, {n_fl11} in FL11")

total = time.time() - t0
print(f"\n=== DONE (total: {total:.0f}s / {total/60:.1f} min) ===")
