#!/usr/bin/env python3
"""Re-annotate clusters with better cell type labels based on marker profiles."""

import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('/Users/ole2001/PROGRAMS/IMC-FL/output')

# Load clustered data
adata = sc.read_h5ad(OUTPUT_DIR / 'FL01_clustered.h5ad')
print(f"Loaded: {adata.n_obs} cells, {adata.obs['leiden'].nunique()} clusters")

# Print the marker profile table again for reference
adata_raw = adata.raw.to_adata()
key_markers = ['CD3', 'CD4', 'CD8a', 'CD20', 'FoxP3', 'CD68',
               'PD_1', 'GranzymeB', 'CD45RO', 'CXCR5', 'CD38',
               'TOX', 'CD27', 'CD31', 'IRF4', 'pCREB', 'CD57',
               'TIM3', 'CD127', 'CD39', 'ICOS', 'T_Bet']
key_markers = [m for m in key_markers if m in adata_raw.var_names]

mean_expr = pd.DataFrame(index=sorted(adata.obs['leiden'].unique(), key=int))
for marker in key_markers:
    idx = list(adata_raw.var_names).index(marker)
    for cluster in mean_expr.index:
        mask = adata.obs['leiden'] == cluster
        mean_expr.loc[cluster, marker] = adata_raw[mask, idx].X.mean()

print("\nMean expression per cluster:")
print(mean_expr.round(2).to_string())

# Manual annotation based on differential expression results:
# Cluster 0 (n=4031): TOX+, CD45RO+, CD3+, PD_1+, CD8a high -> Exhausted CD8 T cells
# Cluster 1 (n=3997): CD20++, pS6+, CXCR5+ -> B cells (FL tumor cells)
# Cluster 2 (n=2507): CD4+, CD3+, FoxP3+, GATA3+ -> CD4 T cells / Tregs
# Cluster 3 (n=2128): CD68++, CD39+, CD31+, pSTAT3+ -> Macrophages / myeloid
# Cluster 4 (n=543):  Low markers, high DNA -> Low quality / debris
# Cluster 5 (n=300):  CD38+++, IRF4+++ -> Plasma cells
# Cluster 6 (n=269):  pCREB+, CD127+, GATA3+ -> CD4 T naive/resting
# Cluster 7 (n=265):  GranzymeB+++, CD68+ -> Cytotoxic (NK/CTL)
# Cluster 8 (n=108):  TIM3+++, CXCR5+++, CD57+ -> Tfh-like / exhausted
# Cluster 9 (n=37):   CD57+, CD27+, CD3+, CD127+ -> NK-T / effector memory

annotations = {
    '0': 'CD8 T exhausted',
    '1': 'B cells (FL)',
    '2': 'CD4 T / Treg',
    '3': 'Macrophages',
    '4': 'Low quality',
    '5': 'Plasma cells',
    '6': 'T naive/resting',
    '7': 'Cytotoxic (NK/CTL)',
    '8': 'Tfh-like',
    '9': 'NK-T cells',
}

adata.obs['cell_type'] = adata.obs['leiden'].map(annotations)

print("\n=== Cell Type Composition ===")
comp = adata.obs['cell_type'].value_counts()
for ct, n in comp.items():
    pct = n / adata.n_obs * 100
    print(f"  {ct}: {n} ({pct:.1f}%)")

# Save
adata.write(OUTPUT_DIR / 'FL01_clustered.h5ad')
print(f"\nSaved updated annotations to FL01_clustered.h5ad")

# Re-generate plots with corrected annotations
print("\nGenerating updated plots...")

sc.settings.set_figure_params(dpi=100, facecolor='white')

# 1. UMAP by cell type
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc.pl.umap(adata, color='leiden', legend_loc='on data', legend_fontsize=10,
           legend_fontoutline=2, frameon=False, ax=axes[0], show=False,
           title='Leiden clusters')

sc.pl.umap(adata, color='cell_type', legend_loc='right margin', legend_fontsize=9,
           frameon=False, ax=axes[1], show=False,
           title='Cell type annotation',
           palette='tab10')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'FL01_umap_celltypes.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Dotplot with corrected types
fig2, ax2 = plt.subplots(figsize=(12, 6))
plot_markers = ['CD3', 'CD4', 'CD8a', 'CD20', 'FoxP3', 'CD68',
                'PD_1', 'GranzymeB', 'CD45RO', 'CXCR5', 'CD38', 'IRF4',
                'TOX', 'TIM3', 'CD57', 'CD127']
plot_markers = [m for m in plot_markers if m in adata.raw.var_names]

sc.pl.dotplot(adata, var_names=plot_markers, groupby='cell_type', use_raw=True,
              show=False, save=None)
plt.savefig(OUTPUT_DIR / 'FL01_dotplot_celltypes.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Spatial map by cell type
fig3, ax3 = plt.subplots(figsize=(8, 8))
spatial = adata.obsm['spatial']

# Filter out low quality cells for spatial plot
mask = adata.obs['cell_type'] != 'Low quality'
adata_filt = adata[mask]
spatial_filt = adata_filt.obsm['spatial']

unique_types = sorted(adata_filt.obs['cell_type'].unique())
cmap = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))

for ct, color in zip(unique_types, cmap):
    ct_mask = adata_filt.obs['cell_type'].values == ct
    ax3.scatter(spatial_filt[ct_mask, 0], spatial_filt[ct_mask, 1],
                c=[color], s=1, alpha=0.5, label=ct)

ax3.set_aspect('equal')
ax3.invert_yaxis()
ax3.legend(markerscale=5, fontsize=8, loc='upper right')
ax3.set_title(f'FL01: Spatial Cell Type Map ({adata_filt.n_obs} cells)')
ax3.axis('off')

fig3.savefig(OUTPUT_DIR / 'FL01_spatial_celltypes_v2.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Composition bar
fig4, ax4 = plt.subplots(figsize=(8, 5))
comp_filt = adata_filt.obs['cell_type'].value_counts()
colors = plt.cm.tab10(np.linspace(0, 1, len(comp_filt)))
ax4.barh(range(len(comp_filt)), comp_filt.values, color=colors)
ax4.set_yticks(range(len(comp_filt)))
ax4.set_yticklabels(comp_filt.index)
ax4.set_xlabel('Number of cells')
ax4.set_title('FL01: Cell Type Composition')
for i, (n, pct) in enumerate(zip(comp_filt.values, comp_filt.values / adata_filt.n_obs * 100)):
    ax4.text(n + 50, i, f'{n} ({pct:.1f}%)', va='center', fontsize=9)
ax4.invert_yaxis()
plt.tight_layout()
fig4.savefig(OUTPUT_DIR / 'FL01_composition.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: FL01_umap_celltypes.png")
print(f"Saved: FL01_dotplot_celltypes.png")
print(f"Saved: FL01_spatial_celltypes_v2.png")
print(f"Saved: FL01_composition.png")

import subprocess
subprocess.run(['open', str(OUTPUT_DIR / 'FL01_umap_celltypes.png')])
subprocess.run(['open', str(OUTPUT_DIR / 'FL01_spatial_celltypes_v2.png')])
subprocess.run(['open', str(OUTPUT_DIR / 'FL01_composition.png')])

print("\n=== Done ===")
