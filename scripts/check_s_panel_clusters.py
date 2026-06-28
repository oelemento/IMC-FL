#!/usr/bin/env python3
"""Check S-panel v4 'Other' clusters in detail."""
import anndata as ad
import pandas as pd
import numpy as np

adata = ad.read_h5ad('<PROJECT_ROOT>/output/all_TMA_S_global_v4.h5ad')
adata_raw = adata.raw.to_adata()

# Key markers for FDC, CAF, stromal
markers = [
    'CD21', 'CXCL13', 'CXCL12', 'CCL21', 'CD209',  # FDC / stromal chemokines
    'Vimentin', 'PDPN', 'Fibronectin', 'SOX9', 'CD146',  # CAF / stromal
    'CD20', 'PAX5', 'BCL_2', 'CD68', 'CD4', 'CD8a',  # lineage
    'CD31', 'CD34', 'HLA_DR', 'CD11c',  # endothelial / DC
]
available = [m for m in markers if m in adata_raw.var_names]

# Current annotation
ct = adata.obs.get('cell_type', pd.Series('N/A', index=adata.obs.index))

print('S-panel cluster profiles (res 2.0, 54 clusters)')
print(f'Focus: FDC, CAF, stromal markers\n')

header = f"{'Cl':>3} {'n':>7} {'cell_type':<25}"
for m in available:
    header += f' {m[:6]:>6}'
print(header)
print('-' * len(header))

for c in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
    mask = adata.obs['leiden'] == c
    n = mask.sum()
    # Get cell type for this cluster
    ct_vals = ct[mask]
    ct_label = ct_vals.mode()[0] if len(ct_vals) > 0 else 'N/A'
    row = f'{c:>3} {n:>7} {ct_label:<25}'
    for m in available:
        idx = list(adata_raw.var_names).index(m)
        val = float(adata_raw[mask, idx].X.mean())
        row += f' {val:>6.1f}'
    print(row)

# Highlight FDC candidates (CD21 > 2)
print('\n=== FDC candidates (CD21 > 2.0) ===')
cd21_idx = list(adata_raw.var_names).index('CD21')
cxcl13_idx = list(adata_raw.var_names).index('CXCL13')
for c in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
    mask = adata.obs['leiden'] == c
    cd21_val = float(adata_raw[mask, cd21_idx].X.mean())
    if cd21_val > 2.0:
        cxcl13_val = float(adata_raw[mask, cxcl13_idx].X.mean())
        n = mask.sum()
        ct_label = ct[mask].mode()[0] if len(ct[mask]) > 0 else 'N/A'
        print(f'  Cluster {c}: CD21={cd21_val:.1f}, CXCL13={cxcl13_val:.1f}, n={n:,}, labeled={ct_label}')

# Highlight CAF/stromal candidates (Vimentin > 1.5 or PDPN > 1.5)
print('\n=== CAF / Stromal candidates (Vimentin > 1.5 or PDPN > 1.5) ===')
vim_idx = list(adata_raw.var_names).index('Vimentin')
pdpn_idx = list(adata_raw.var_names).index('PDPN')
fn_idx = list(adata_raw.var_names).index('Fibronectin')
for c in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
    mask = adata.obs['leiden'] == c
    vim_val = float(adata_raw[mask, vim_idx].X.mean())
    pdpn_val = float(adata_raw[mask, pdpn_idx].X.mean())
    fn_val = float(adata_raw[mask, fn_idx].X.mean())
    if vim_val > 1.5 or pdpn_val > 1.5:
        n = mask.sum()
        ct_label = ct[mask].mode()[0] if len(ct[mask]) > 0 else 'N/A'
        cd20_val = float(adata_raw[mask, list(adata_raw.var_names).index('CD20')].X.mean())
        cd68_val = float(adata_raw[mask, list(adata_raw.var_names).index('CD68')].X.mean())
        print(f'  Cluster {c}: Vim={vim_val:.1f}, PDPN={pdpn_val:.1f}, FN={fn_val:.1f}, CD20={cd20_val:.1f}, CD68={cd68_val:.1f}, n={n:,}, labeled={ct_label}')
