#!/usr/bin/env python3
"""Investigate S-panel v4 'Other' clusters."""
import anndata as ad
import pandas as pd
import numpy as np

adata = ad.read_h5ad('<PROJECT_ROOT>/output/all_TMA_S_global_v4.h5ad')
adata_raw = adata.raw.to_adata()

# All bio markers (exclude DNA, histone)
skip = ['DNA1', 'DNA2', 'HistoneH3', 'p_H3s28']
markers = [m for m in adata_raw.var_names if m not in skip]

# Focus on Other clusters
other_clusters = []
for c in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
    mask = adata.obs['leiden'] == c
    ct = adata.obs.loc[mask, 'cell_type'].mode()[0]
    if ct == 'Other':
        other_clusters.append(c)

print(f"=== 'Other' clusters: {len(other_clusters)} clusters ===\n")

# Print marker profiles for Other clusters
key_markers = ['CD20', 'PAX5', 'BCL_2', 'BCL_6', 'CD21', 'CXCL13',
               'CD68', 'CD163', 'CD206', 'CD14', 'CD11c', 'HLA_DR',
               'Vimentin', 'PDPN', 'Fibronectin', 'SOX9', 'CD146',
               'CD31', 'CD34', 'CD4', 'CD8a', 'CD209', 'CXCL12',
               'CCL21', 'Ki-67', 'CD123', 'CD44', 'VISTA', 'PD_L1',
               'IDO', 'CD11b', 'CD49a', 'CD1a', 'S100A9', 'HLA_Class_I']
key_markers = [m for m in key_markers if m in adata_raw.var_names]

header = f"{'Cl':>3} {'n':>7}"
for m in key_markers:
    header += f' {m[:5]:>5}'
print(header)
print('-' * len(header))

for c in other_clusters:
    mask = adata.obs['leiden'] == c
    n = mask.sum()
    row = f'{c:>3} {n:>7}'
    for m in key_markers:
        idx = list(adata_raw.var_names).index(m)
        val = float(adata_raw[mask, idx].X.mean())
        row += f' {val:>5.1f}'
    print(row)

# Per-TMA breakdown of Other clusters
print(f"\n=== Per-TMA breakdown of 'Other' clusters ===")
for c in other_clusters:
    mask = adata.obs['leiden'] == c
    n = mask.sum()
    print(f"\nCluster {c} ({n:,} cells):")
    tma_counts = adata.obs.loc[mask, 'tma'].value_counts()
    for tma, cnt in tma_counts.items():
        pct = cnt / n * 100
        print(f"  {tma}: {cnt:,} ({pct:.1f}%)")

# Summary: what's the dominant signal in each Other cluster?
print(f"\n=== Suggested reclassification ===")
for c in other_clusters:
    mask = adata.obs['leiden'] == c
    n = mask.sum()
    vals = {}
    for m in key_markers:
        idx = list(adata_raw.var_names).index(m)
        vals[m] = float(adata_raw[mask, idx].X.mean())

    # Find top 3 markers
    top = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:5]
    top_str = ', '.join(f'{m}={v:.1f}' for m, v in top)
    print(f"  Cluster {c} ({n:,}): top markers = {top_str}")
