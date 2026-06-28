#!/usr/bin/env python3
"""Heatmap of mean marker expression per cluster for annotation review."""

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

# Load
adata = sc.read_h5ad(OUTPUT_DIR / 'FL01_clustered.h5ad')
adata_raw = adata.raw.to_adata()

# All biological markers (exclude non-biological channels)
exclude = ['80ArAr', '129Xe', '190BCKG', '197Au', 'Pb204']
bio_markers = [m for m in adata_raw.var_names if m not in exclude]

# Compute mean raw expression per cluster
clusters = sorted(adata.obs['leiden'].unique(), key=int)
cell_types = [adata.obs.loc[adata.obs['leiden'] == c, 'cell_type'].iloc[0] for c in clusters]
n_cells = [int((adata.obs['leiden'] == c).sum()) for c in clusters]
row_labels = [f"{ct}\n(c{c}, n={n})" for c, ct, n in zip(clusters, cell_types, n_cells)]

mean_expr = pd.DataFrame(index=clusters, columns=bio_markers, dtype=float)
for marker in bio_markers:
    idx = list(adata_raw.var_names).index(marker)
    for cluster in clusters:
        mask = adata.obs['leiden'] == cluster
        mean_expr.loc[cluster, marker] = adata_raw[mask, idx].X.mean()

# Z-score per marker (column) for better visualization
mean_z = mean_expr.copy()
for col in mean_z.columns:
    vals = mean_z[col].astype(float)
    mean_z[col] = (vals - vals.mean()) / (vals.std() + 1e-6)

# Plot 1: Heatmap with z-scored expression
fig, ax = plt.subplots(figsize=(18, 8))
im = ax.imshow(mean_z.values.astype(float), aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)

ax.set_xticks(range(len(bio_markers)))
ax.set_xticklabels(bio_markers, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(clusters)))
ax.set_yticklabels(row_labels, fontsize=9)

# Add raw values as text
for i in range(len(clusters)):
    for j in range(len(bio_markers)):
        val = float(mean_expr.iloc[i, j])
        if val > 1.0:
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6, fontweight='bold')
        elif val > 0.5:
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6)

plt.colorbar(im, ax=ax, label='Z-score', shrink=0.6)
ax.set_title('FL01: Mean Marker Expression per Cluster (z-scored, raw values shown)', fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'FL01_cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Key markers only, with raw values
key_markers = ['CD3', 'CD4', 'CD8a', 'CD20', 'FoxP3', 'CD68', 'PD_1',
               'GranzymeB', 'CD45RO', 'CXCR5', 'CD38', 'IRF4', 'TOX',
               'TIM3', 'CD57', 'CD127', 'CD27', 'CD31', 'CD39', 'ICOS',
               'T_Bet', 'GATA3', 'pSTAT3', 'CD86', 'CTLA4', 'LAG3']
key_markers = [m for m in key_markers if m in bio_markers]

mean_key = mean_expr[key_markers]
mean_key_z = mean_z[key_markers]

fig2, ax2 = plt.subplots(figsize=(14, 8))
im2 = ax2.imshow(mean_key_z.values.astype(float), aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)

ax2.set_xticks(range(len(key_markers)))
ax2.set_xticklabels(key_markers, rotation=45, ha='right', fontsize=10)
ax2.set_yticks(range(len(clusters)))
ax2.set_yticklabels(row_labels, fontsize=10)

for i in range(len(clusters)):
    for j in range(len(key_markers)):
        val = float(mean_key.iloc[i, j])
        ax2.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7,
                 fontweight='bold' if val > 1.0 else 'normal')

plt.colorbar(im2, ax=ax2, label='Z-score', shrink=0.6)
ax2.set_title('FL01: Key Marker Expression per Cluster', fontsize=13)
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / 'FL01_cluster_heatmap_key.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {OUTPUT_DIR}/FL01_cluster_heatmap.png (all markers)")
print(f"Saved: {OUTPUT_DIR}/FL01_cluster_heatmap_key.png (key markers)")

# Also print table
print("\n=== Raw Mean Expression (key markers) ===\n")
print(mean_key.round(2).to_string())

import subprocess
subprocess.run(['open', str(OUTPUT_DIR / 'FL01_cluster_heatmap_key.png')])
subprocess.run(['open', str(OUTPUT_DIR / 'FL01_cluster_heatmap.png')])
