#!/usr/bin/env python3
"""Cluster cells from hybrid segmentation and identify major cell types."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white')

OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')
sc.settings.figdir = str(OUTPUT_DIR) + '/'

# Load segmented data
print("Loading data...")
adata = sc.read_h5ad(OUTPUT_DIR / 'FL01_hybrid.h5ad')
print(f"AnnData: {adata.n_obs} cells x {adata.n_vars} markers")
print(f"Markers: {list(adata.var_names)}")

# Exclude non-biological channels
exclude = ['80ArAr', '129Xe', '190BCKG', '197Au', 'Pb204']
biological = [m for m in adata.var_names if m not in exclude]
print(f"\nBiological markers ({len(biological)}): {biological}")

adata = adata[:, biological].copy()

# Also exclude structural markers from clustering (keep for reference)
structural = ['DNA1', 'DNA2', 'HistoneH3', 'p_H3s28', 'H3K27me3']
cluster_markers = [m for m in biological if m not in structural]
print(f"Markers for clustering ({len(cluster_markers)}): {cluster_markers}")

# Save raw
adata.raw = adata

# Preprocessing
print("\nPreprocessing...")

# arcsinh transform (standard for mass cytometry, cofactor=5)
cofactor = 5
adata.X = np.arcsinh(adata.X / cofactor)
print(f"Applied arcsinh transform (cofactor={cofactor})")

# Scale
sc.pp.scale(adata, max_value=10)

# Subset to clustering markers for PCA/neighbors
adata_cluster = adata[:, cluster_markers].copy()

# PCA
print("\nRunning PCA...")
sc.tl.pca(adata_cluster, svd_solver='arpack', n_comps=min(20, len(cluster_markers) - 1))

# Store PCA in main adata
adata.obsm['X_pca'] = adata_cluster.obsm['X_pca']

# Neighbors
print("Computing neighbors...")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=15, use_rep='X_pca')

# UMAP
print("Running UMAP...")
sc.tl.umap(adata)

# Clustering at multiple resolutions
print("\nClustering...")
for res in [0.3, 0.5, 0.8, 1.0]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
    n_clusters = adata.obs[f'leiden_{res}'].nunique()
    print(f"  Resolution {res}: {n_clusters} clusters")

# Use resolution 0.5 as default
adata.obs['leiden'] = adata.obs['leiden_0.5']
n_clusters = adata.obs['leiden'].nunique()
print(f"\nUsing resolution 0.5: {n_clusters} clusters")

# Find marker genes for each cluster
print("\nFinding marker genes per cluster...")
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', use_raw=True)

# Print top markers per cluster
print("\n=== Top markers per cluster ===")
for group in sorted(adata.obs['leiden'].unique(), key=int):
    markers_df = sc.get.rank_genes_groups_df(adata, group=group)
    top = markers_df.head(5)
    marker_str = ', '.join([f"{r['names']}({r['logfoldchanges']:.1f})" for _, r in top.iterrows()])
    n_cells = (adata.obs['leiden'] == group).sum()
    print(f"  Cluster {group} (n={n_cells}): {marker_str}")

# Cell type annotation based on known markers
print("\n=== Cell Type Annotation ===")

# Use raw (untransformed) expression for annotation
adata_raw = adata.raw.to_adata()

# Compute mean expression per cluster for key markers
key_markers = ['CD3', 'CD4', 'CD8a', 'CD20', 'FoxP3', 'CD68',
               'PD_1', 'GranzymeB', 'CD45RO', 'CXCR5', 'CD38']
key_markers = [m for m in key_markers if m in adata_raw.var_names]

mean_expr = pd.DataFrame(index=sorted(adata.obs['leiden'].unique(), key=int))
for marker in key_markers:
    idx = list(adata_raw.var_names).index(marker)
    for cluster in mean_expr.index:
        mask = adata.obs['leiden'] == cluster
        mean_expr.loc[cluster, marker] = adata_raw[mask, idx].X.mean()

print("\nMean expression of key markers per cluster:")
print(mean_expr.round(2).to_string())

# Simple rule-based annotation
annotations = {}
for cluster in mean_expr.index:
    cd3 = mean_expr.loc[cluster, 'CD3'] if 'CD3' in mean_expr.columns else 0
    cd4 = mean_expr.loc[cluster, 'CD4'] if 'CD4' in mean_expr.columns else 0
    cd8 = mean_expr.loc[cluster, 'CD8a'] if 'CD8a' in mean_expr.columns else 0
    cd20 = mean_expr.loc[cluster, 'CD20'] if 'CD20' in mean_expr.columns else 0
    foxp3 = mean_expr.loc[cluster, 'FoxP3'] if 'FoxP3' in mean_expr.columns else 0
    cd68 = mean_expr.loc[cluster, 'CD68'] if 'CD68' in mean_expr.columns else 0
    pd1 = mean_expr.loc[cluster, 'PD_1'] if 'PD_1' in mean_expr.columns else 0
    gzmb = mean_expr.loc[cluster, 'GranzymeB'] if 'GranzymeB' in mean_expr.columns else 0
    cxcr5 = mean_expr.loc[cluster, 'CXCR5'] if 'CXCR5' in mean_expr.columns else 0

    # Annotation logic for FL TME
    if cd20 > cd3 and cd20 > cd68:
        annotations[cluster] = 'B cells'
    elif cd3 > cd20 and cd4 > cd8 and foxp3 > mean_expr['FoxP3'].median() * 1.5:
        annotations[cluster] = 'Treg'
    elif cd3 > cd20 and cd4 > cd8 and cxcr5 > mean_expr['CXCR5'].median() * 1.5 and pd1 > mean_expr['PD_1'].median():
        annotations[cluster] = 'Tfh'
    elif cd3 > cd20 and cd4 > cd8:
        annotations[cluster] = 'CD4 T cells'
    elif cd3 > cd20 and cd8 > cd4:
        annotations[cluster] = 'CD8 T cells'
    elif cd3 > cd20:
        annotations[cluster] = 'T cells'
    elif cd68 > cd3 and cd68 > cd20:
        annotations[cluster] = 'Macrophages'
    else:
        annotations[cluster] = f'Cluster {cluster}'

adata.obs['cell_type'] = adata.obs['leiden'].map(annotations)

print("\nAnnotations:")
for cluster, ctype in sorted(annotations.items(), key=lambda x: int(x[0])):
    n = (adata.obs['leiden'] == cluster).sum()
    print(f"  Cluster {cluster}: {ctype} (n={n})")

# Save
print("\nSaving...")
adata.write(OUTPUT_DIR / 'FL01_clustered.h5ad')

# ---- Plotting ----
print("\nGenerating plots...")
import matplotlib.pyplot as plt

# 1. UMAP by cluster and cell type
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc.pl.umap(adata, color='leiden', legend_loc='on data', legend_fontsize=10,
           legend_fontoutline=2, frameon=False, ax=axes[0], show=False,
           title='Leiden clusters (res=0.5)')

sc.pl.umap(adata, color='cell_type', legend_loc='right margin', legend_fontsize=9,
           frameon=False, ax=axes[1], show=False,
           title='Cell type annotation')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'FL01_umap_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. UMAP colored by key markers
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
plot_markers = ['CD3', 'CD4', 'CD8a', 'CD20', 'FoxP3', 'PD_1', 'GranzymeB', 'CD45RO']
plot_markers = [m for m in plot_markers if m in adata.raw.var_names][:8]

for i, marker in enumerate(plot_markers):
    ax = axes2[i // 4, i % 4]
    sc.pl.umap(adata, color=marker, use_raw=True, ax=ax, show=False,
               frameon=False, title=marker, color_map='viridis', vmax='p99')

plt.suptitle('FL01: Key Markers on UMAP', fontsize=14)
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / 'FL01_umap_markers.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Dot plot
fig3, ax3 = plt.subplots(figsize=(10, 5))
sc.pl.dotplot(adata, var_names=key_markers, groupby='cell_type', use_raw=True,
              ax=ax3, show=False)
plt.tight_layout()
fig3.savefig(OUTPUT_DIR / 'FL01_dotplot.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Heatmap of top markers per cluster
sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, use_raw=True,
                                 show=False, save='_FL01_markers.png')

# 5. Spatial plot colored by cell type
fig5, ax5 = plt.subplots(figsize=(8, 8))
spatial = adata.obsm['spatial']
cell_types = adata.obs['cell_type'].values
unique_types = sorted(adata.obs['cell_type'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))

for ct, color in zip(unique_types, colors):
    mask = cell_types == ct
    ax5.scatter(spatial[mask, 0], spatial[mask, 1], c=[color], s=1, alpha=0.5, label=ct)

ax5.set_aspect('equal')
ax5.invert_yaxis()
ax5.legend(markerscale=5, fontsize=9, loc='upper right')
ax5.set_title(f'FL01: Spatial Cell Type Map ({adata.n_obs} cells)')
ax5.axis('off')

fig5.savefig(OUTPUT_DIR / 'FL01_spatial_celltypes.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: FL01_umap_clusters.png")
print(f"Saved: FL01_umap_markers.png")
print(f"Saved: FL01_dotplot.png")
print(f"Saved: FL01_spatial_celltypes.png")
print(f"Saved: FL01_clustered.h5ad")

import subprocess
subprocess.run(['open', str(OUTPUT_DIR / 'FL01_umap_clusters.png')])
subprocess.run(['open', str(OUTPUT_DIR / 'FL01_spatial_celltypes.png')])

print("\n=== Summary ===")
print(f"Cells: {adata.n_obs}")
print(f"Clusters: {n_clusters}")
print(f"\nCell type composition:")
print(adata.obs['cell_type'].value_counts().to_string())
