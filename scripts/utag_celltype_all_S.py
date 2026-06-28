#!/usr/bin/env python
"""UTAG on cell-type indicator features — all TMAs, S-panel v8.

Uses one-hot encoded cell types as input features with max_dist=50.
This resolves follicular structure that raw markers cannot detect.
"""
import time
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTDIR = "<PROJECT_ROOT>/output"
INPUT = f"{OUTDIR}/all_TMA_S_global_v8.h5ad"
PREFIX = "all_TMA_S_utag_ct"
RESOLUTIONS = [0.3, 0.5, 1.0]
MAX_DIST = 50

print("=== Loading S-panel v8 ===")
t0 = time.time()
adata = ad.read_h5ad(INPUT)
print(f"Loaded {adata.shape} in {time.time()-t0:.0f}s")
print(f"TMAs: {sorted(adata.obs['tma'].unique())}")
print(f"Cell types: {sorted(adata.obs['cell_type'].unique())}")

# ---- Create one-hot cell type features ----
print("\n=== Creating one-hot cell type features ===")
ct_dummies = pd.get_dummies(adata.obs["cell_type"]).astype(np.float32)
print(f"Feature matrix: {ct_dummies.shape} ({list(ct_dummies.columns)})")

adata_ct = ad.AnnData(
    X=ct_dummies.values,
    obs=adata.obs.copy(),
    var=pd.DataFrame(index=ct_dummies.columns),
)
if "spatial" in adata.obsm:
    adata_ct.obsm["spatial"] = adata.obsm["spatial"].copy()
elif "centroid_x" in adata.obs.columns:
    adata_ct.obsm["spatial"] = np.column_stack([
        adata.obs["centroid_x"].values,
        adata.obs["centroid_y"].values,
    ])

del ct_dummies

# ---- Run UTAG ----
print(f"\n=== Running UTAG (max_dist={MAX_DIST}, res={RESOLUTIONS}) ===")
from utag import utag as run_utag

t1 = time.time()
result = run_utag(
    adata_ct,
    slide_key="sample_id",
    max_dist=MAX_DIST,
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

# ---- Transfer UTAG labels back to original adata ----
print("\n=== Transferring labels ===")
for res in RESOLUTIONS:
    col = f"UTAG Label_leiden_{res}"
    if col in result.obs.columns:
        adata.obs[f"utag_ct_{res}"] = result.obs[col].values

# Save
adata.write_h5ad(f"{OUTDIR}/{PREFIX}.h5ad")
print(f"Saved {OUTDIR}/{PREFIX}.h5ad")

# ---- Composition heatmap for each resolution ----
print("\n=== Generating plots ===")

for res in RESOLUTIONS:
    col = f"utag_ct_{res}"
    if col not in adata.obs.columns:
        continue

    comp = pd.crosstab(adata.obs[col], adata.obs["cell_type"], normalize="index")
    n_domains = comp.shape[0]

    fig, ax = plt.subplots(figsize=(14, max(6, n_domains * 0.3)))
    sns.heatmap(comp, annot=n_domains <= 30, fmt=".2f" if n_domains <= 30 else "",
                cmap="YlOrRd", ax=ax, linewidths=0.5,
                cbar_kws={"label": "Fraction"}, annot_kws={"size": 7})
    ax.set_title(f"S-panel v8: UTAG Cell-Type Features (max_dist={MAX_DIST}, res={res})\n"
                 f"{n_domains} domains, {adata.shape[0]:,} cells",
                 fontweight="bold")
    ax.set_ylabel("UTAG Domain")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{PREFIX}_composition_res{res}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PREFIX}_composition_res{res}.png")

    # Per-TMA domain distribution
    tma_comp = pd.crosstab(adata.obs["tma"], adata.obs[col], normalize="index")
    fig, ax = plt.subplots(figsize=(max(10, n_domains * 0.4), 5))
    sns.heatmap(tma_comp, cmap="YlOrRd", ax=ax, linewidths=0.5,
                annot=n_domains <= 30, fmt=".2f" if n_domains <= 30 else "",
                cbar_kws={"label": "Fraction"})
    ax.set_title(f"S-panel v8: Domain Distribution per TMA (res={res})", fontweight="bold")
    ax.set_ylabel("TMA")
    ax.set_xlabel("UTAG Domain")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{PREFIX}_tma_dist_res{res}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PREFIX}_tma_dist_res{res}.png")

# ---- FL11 spatial plot ----
print("\n=== FL11 spatial plots ===")

ct_colors = {
    "B cells": "#FFB347", "FDC": "#8B4513", "FDC (CD21+)": "#A0522D",
    "Macrophages": "#228B22", "M2 macrophages": "#006400",
    "CD4 T cells": "#9370DB", "CD8 T cells": "#90EE90",
    "Treg": "#DC143C", "Plasma cells": "#FF1493",
    "Endothelial": "#4169E1", "Fibroblasts": "#DDA0DD",
    "Smooth muscle": "#B8860B", "LQ": "#D3D3D3",
    "Mixed": "#A9A9A9", "Other": "#C0C0C0",
}

roi = adata[adata.obs["sample_id"] == "B1_FL11"].copy()
if len(roi) > 0:
    x = roi.obsm["spatial"][:, 0] if "spatial" in roi.obsm else roi.obs["centroid_x"].values
    y = roi.obsm["spatial"][:, 1] if "spatial" in roi.obsm else roi.obs["centroid_y"].values

    plot_res = [r for r in RESOLUTIONS if f"utag_ct_{r}" in roi.obs.columns]
    n_panels = 1 + len(plot_res)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))

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

    for i, res in enumerate(plot_res):
        ax = axes[i + 1]
        col = f"utag_ct_{res}"
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

    plt.suptitle(f"UTAG Cell-Type Features — B1_FL11 (S-panel v8, max_dist={MAX_DIST})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{PREFIX}_FL11_spatial.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PREFIX}_FL11_spatial.png")
else:
    print("  B1_FL11 not found in S-panel")

# ---- Summary ----
print("\n=== Summary ===")
for res in RESOLUTIONS:
    col = f"utag_ct_{res}"
    if col in adata.obs.columns:
        n_global = adata.obs[col].nunique()
        roi_obs = adata[adata.obs["sample_id"] == "B1_FL11"].obs
        n_fl11 = roi_obs[col].nunique() if len(roi_obs) > 0 else 0
        print(f"  res={res}: {n_global} global domains, {n_fl11} in FL11")

if "utag_ct_0.3" in adata.obs.columns:
    print("\nDomain sizes (res=0.3):")
    sizes = adata.obs["utag_ct_0.3"].value_counts().sort_index()
    for d, n in sizes.items():
        print(f"  Domain {d}: {n:,} cells ({n/len(adata)*100:.1f}%)")

total = time.time() - t0
print(f"\n=== DONE (total: {total:.0f}s / {total/60:.1f} min) ===")
