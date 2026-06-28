#!/usr/bin/env python3
"""Cross-panel annotation transfer: use confident annotations from one panel
to learn what those cells look like in the other panel's marker space.

Uses FL32 (97% tissue overlap, best concordance) as pilot.
Works at the patch level since cell-level correspondence isn't possible
between serial sections.
"""
import sys
sys.path.insert(0, '/Users/ole2001/PROGRAMS/IMC-FL')

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.sparse import issparse
from src.data_loader import load_roi_txt
from pathlib import Path

# --- Config ---
ROI = 'B1_FL32'
PATCH_SIZE = 50  # µm (50x50 patches for spatial transfer)
MIN_CELLS_PER_PATCH = 5

DATA_T = Path('data/raw/TMA_B1_T')
DATA_S = Path('data/raw/TMA_B1_S')
TXT_T = '20220118_CT14_09_B1_Tcellpanel_4_FL32_R_5.txt'
TXT_S = '20210518_CT14_09_B1_Stromalpanel_5_FL32_R_2.txt'

print("Loading v5 h5ad files...")
s_adata = ad.read_h5ad('output/all_TMA_S_global_v5.h5ad')
t_adata = ad.read_h5ad('output/all_TMA_T_global_v5.h5ad')

# Build sample_id
for adata in [s_adata, t_adata]:
    if 'sample_id' not in adata.obs.columns:
        adata.obs['sample_id'] = adata.obs['tma'].astype(str) + '_' + adata.obs['roi'].astype(str)

# Extract FL32 cells
s_fl32 = s_adata[s_adata.obs['sample_id'] == ROI].copy()
t_fl32 = t_adata[t_adata.obs['sample_id'] == ROI].copy()
print(f"FL32 cells: S={s_fl32.n_obs:,}, T={t_fl32.n_obs:,}")

# Get raw expression
s_raw = s_fl32.raw.to_adata()
t_raw = t_fl32.raw.to_adata()


def get_coords(adata):
    if 'spatial' in adata.obsm:
        return adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1]
    elif 'X_spatial' in adata.obsm:
        return adata.obsm['X_spatial'][:, 0], adata.obsm['X_spatial'][:, 1]
    return adata.obs['centroid_x'].values, adata.obs['centroid_y'].values


def raw_expr(raw_adata, marker):
    if marker not in raw_adata.var_names:
        return None
    idx = list(raw_adata.var_names).index(marker)
    x = raw_adata.X[:, idx]
    if issparse(x):
        return x.toarray().ravel()
    return np.asarray(x).ravel()


# Cell coordinates
sx, sy = get_coords(s_fl32)
tx, ty = get_coords(t_fl32)

# --- Registration: compute shift from DNA images ---
print("\nRegistering via DNA center-of-mass...")
img_s, markers_s, _ = load_roi_txt(DATA_S / TXT_S)
img_t, markers_t, _ = load_roi_txt(DATA_T / TXT_T)

def dna_composite(img, markers):
    chs = []
    for m in ['DNA1', 'DNA2']:
        if m in markers:
            chs.append(img[:, :, markers.index(m)])
    return sum(chs)

def tissue_mask(dna, sigma=5, q=0.15):
    sm = gaussian_filter(dna.astype(float), sigma=sigma)
    thresh = np.quantile(sm[sm > 0], q) if (sm > 0).any() else 0
    return sm > thresh

dna_s = dna_composite(img_s, markers_s)
dna_t = dna_composite(img_t, markers_t)
mask_s = tissue_mask(dna_s)
mask_t = tissue_mask(dna_t)
com_s = np.array(center_of_mass(mask_s.astype(float)))
com_t = np.array(center_of_mass(mask_t.astype(float)))
shift_yx = com_s - com_t
print(f"  Shift: dy={shift_yx[0]:.1f}, dx={shift_yx[1]:.1f}")

# Apply shift to T-panel cell coordinates
tx_reg = tx + shift_yx[1]
ty_reg = ty + shift_yx[0]

# --- Assign cells to patches ---
all_x = np.concatenate([sx, tx_reg])
all_y = np.concatenate([sy, ty_reg])
x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()

s_patch_x = ((sx - x_min) / PATCH_SIZE).astype(int)
s_patch_y = ((sy - y_min) / PATCH_SIZE).astype(int)
s_patch_id = s_patch_y * 1000 + s_patch_x

t_patch_x = ((tx_reg - x_min) / PATCH_SIZE).astype(int)
t_patch_y = ((ty_reg - y_min) / PATCH_SIZE).astype(int)
t_patch_id = t_patch_y * 1000 + t_patch_x

# Patches with cells in BOTH panels
s_patches = set(s_patch_id)
t_patches = set(t_patch_id)
shared_patches = s_patches & t_patches
print(f"\nPatches: S={len(s_patches)}, T={len(t_patches)}, shared={len(shared_patches)}")

# --- Cross-tabulation: patch-level dominant cell types ---
print("\n=== Patch-level cross-tabulation ===")

s_types = s_fl32.obs['cell_type'].values
t_types = t_fl32.obs['cell_type'].values

# For each shared patch, get dominant cell type from each panel
rows = []
for pid in shared_patches:
    s_mask = s_patch_id == pid
    t_mask = t_patch_id == pid
    n_s = s_mask.sum()
    n_t = t_mask.sum()
    if n_s < MIN_CELLS_PER_PATCH or n_t < MIN_CELLS_PER_PATCH:
        continue

    s_dom = pd.Series(s_types[s_mask]).mode()[0]
    t_dom = pd.Series(t_types[t_mask]).mode()[0]
    rows.append({'patch': pid, 's_type': s_dom, 't_type': t_dom, 'n_s': n_s, 'n_t': n_t})

df_patches = pd.DataFrame(rows)
print(f"Shared patches with ≥{MIN_CELLS_PER_PATCH} cells in both panels: {len(df_patches)}")

# Cross-tab
xtab = pd.crosstab(df_patches['t_type'], df_patches['s_type'], margins=True)
print(f"\nPatch-level cross-tabulation (rows=T-panel, cols=S-panel):")
print(xtab.to_string())


# ================================================================
# KEY ANALYSIS: What do confident T-panel cell types look like
# in S-panel marker space?
# ================================================================
print("\n\n" + "=" * 70)
print("CROSS-PANEL MARKER PROFILES")
print("=" * 70)

# S-panel markers we want to examine
s_key_markers = ['CD20', 'PAX5', 'BCL_2', 'BCL_6', 'CD21', 'CXCL13',
                 'CD4', 'CD8a', 'CD68', 'CD163', 'CD206', 'CD14',
                 'Vimentin', 'PDPN', 'CD34', 'CD31', 'HLA_DR',
                 'S100A9', 'CD44', 'CD11c', 'VISTA', 'PD_L1', 'Ki-67']
s_key_markers = [m for m in s_key_markers if m in s_raw.var_names]

# T-panel markers
t_key_markers = ['CD20', 'CD3', 'CD4', 'CD8a', 'CD68', 'FoxP3',
                 'GranzymeB', 'PD_1', 'TIM3', 'LAG3', 'ICOS',
                 'CXCR5', 'CD38', 'CD57', 'TOX', 'CD31',
                 'Ki-67', 'pSTAT3', 'CD39', 'CTLA4', 'CD86']
t_key_markers = [m for m in t_key_markers if m in t_raw.var_names]

# For each T-panel cell type, find matching patches and get S-panel marker profile
print("\n--- T-panel confident types → S-panel marker profile ---")
print("(What do T-panel's T cells look like in S-panel marker space?)\n")

t_cell_types = t_fl32.obs['cell_type'].value_counts()
print(f"T-panel cell types in FL32: {dict(t_cell_types)}\n")

for ct in t_cell_types.index:
    if t_cell_types[ct] < 50:
        continue

    # Find patches dominated by this T-panel cell type
    ct_patches = df_patches[df_patches['t_type'] == ct]['patch'].values

    # Get S-panel cells in those patches
    s_in_patches = np.isin(s_patch_id, ct_patches)
    n_s_cells = s_in_patches.sum()
    if n_s_cells < 20:
        continue

    print(f"T-panel '{ct}' ({t_cell_types[ct]} cells) → {len(ct_patches)} patches → {n_s_cells} S-panel cells")

    # S-panel cell type distribution in those patches
    s_ct_dist = pd.Series(s_types[s_in_patches]).value_counts()
    for sct, cnt in s_ct_dist.head(5).items():
        pct = cnt / n_s_cells * 100
        print(f"    S-panel calls: {sct} = {cnt} ({pct:.1f}%)")

    # S-panel mean raw marker expression for those cells
    print(f"    S-panel marker profile (mean raw):")
    vals = {}
    for m in s_key_markers:
        expr = raw_expr(s_raw, m)
        if expr is not None:
            vals[m] = float(expr[s_in_patches].mean())
    top = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"      Top: {', '.join(f'{m}={v:.1f}' for m, v in top)}")
    print()

# Reverse: S-panel confident types → T-panel marker profile
print("\n--- S-panel confident types → T-panel marker profile ---")
print("(What do S-panel's FDC/stromal/myeloid look like in T-panel marker space?)\n")

s_cell_types = s_fl32.obs['cell_type'].value_counts()
print(f"S-panel cell types in FL32: {dict(s_cell_types)}\n")

for ct in s_cell_types.index:
    if s_cell_types[ct] < 50:
        continue

    ct_patches = df_patches[df_patches['s_type'] == ct]['patch'].values
    t_in_patches = np.isin(t_patch_id, ct_patches)
    n_t_cells = t_in_patches.sum()
    if n_t_cells < 20:
        continue

    print(f"S-panel '{ct}' ({s_cell_types[ct]} cells) → {len(ct_patches)} patches → {n_t_cells} T-panel cells")

    t_ct_dist = pd.Series(t_types[t_in_patches]).value_counts()
    for tct, cnt in t_ct_dist.head(5).items():
        pct = cnt / n_t_cells * 100
        print(f"    T-panel calls: {tct} = {cnt} ({pct:.1f}%)")

    vals = {}
    for m in t_key_markers:
        expr = raw_expr(t_raw, m)
        if expr is not None:
            vals[m] = float(expr[t_in_patches].mean())
    top = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"    T-panel marker profile (mean raw):")
    print(f"      Top: {', '.join(f'{m}={v:.1f}' for m, v in top)}")
    print()


# ================================================================
# FIGURE: Cross-panel transfer heatmaps
# ================================================================
print("\n=== Generating figures ===")

# Figure 1: Patch-level cross-tab heatmap
fig, ax = plt.subplots(figsize=(14, 8))
# Remove 'All' margin for heatmap
xtab_core = xtab.drop('All', axis=0).drop('All', axis=1)
# Normalize rows (what % of T-panel type X patches are called Y in S-panel)
xtab_norm = xtab_core.div(xtab_core.sum(axis=1), axis=0) * 100
im = ax.imshow(xtab_norm.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(xtab_norm.columns)))
ax.set_xticklabels(xtab_norm.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(xtab_norm.index)))
ax.set_yticklabels(xtab_norm.index, fontsize=8)
ax.set_xlabel('S-panel dominant type', fontsize=10)
ax.set_ylabel('T-panel dominant type', fontsize=10)
# Add text annotations
for i in range(len(xtab_norm.index)):
    for j in range(len(xtab_norm.columns)):
        val = xtab_norm.values[i, j]
        count = xtab_core.values[i, j]
        if count > 0:
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%\n({count})', ha='center', va='center',
                   fontsize=7, color=color)
plt.colorbar(im, ax=ax, label='% of T-panel patches', shrink=0.8)
ax.set_title(f'FL32 patch-level cross-tabulation ({PATCH_SIZE}×{PATCH_SIZE}µm patches)\n'
             f'Rows=T-panel dominant type, Cols=S-panel dominant type',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig('output/cross_panel_transfer_xtab.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: output/cross_panel_transfer_xtab.png")


# Figure 2: S-panel marker profiles for T-panel cell types
# Build matrix: rows = T-panel cell types, cols = S-panel markers
ct_list = [ct for ct in t_cell_types.index if t_cell_types[ct] >= 50]
marker_matrix = []
ct_labels = []
for ct in ct_list:
    ct_patches_ids = df_patches[df_patches['t_type'] == ct]['patch'].values
    s_in_patches = np.isin(s_patch_id, ct_patches_ids)
    if s_in_patches.sum() < 20:
        continue
    row = []
    for m in s_key_markers:
        expr = raw_expr(s_raw, m)
        if expr is not None:
            row.append(float(expr[s_in_patches].mean()))
        else:
            row.append(0)
    marker_matrix.append(row)
    ct_labels.append(f"{ct} (T)")

# Also add S-panel cell types → S-panel markers (reference)
for ct in s_cell_types.index:
    if s_cell_types[ct] < 50:
        continue
    mask = s_types == ct
    row = []
    for m in s_key_markers:
        expr = raw_expr(s_raw, m)
        if expr is not None:
            row.append(float(expr[mask].mean()))
        else:
            row.append(0)
    marker_matrix.append(row)
    ct_labels.append(f"{ct} (S-ref)")

mat = np.array(marker_matrix)
# Z-score per marker
mat_z = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-8)

fig, ax = plt.subplots(figsize=(16, max(8, len(ct_labels) * 0.5)))
im = ax.imshow(mat_z, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax.set_xticks(range(len(s_key_markers)))
ax.set_xticklabels(s_key_markers, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(ct_labels)))
ax.set_yticklabels(ct_labels, fontsize=8)
# Separator line between T-panel types and S-panel reference
n_t_types = len([l for l in ct_labels if l.endswith('(T)')])
ax.axhline(n_t_types - 0.5, color='black', linewidth=2, linestyle='--')
ax.text(len(s_key_markers) + 0.5, n_t_types / 2, '← T-panel types\nin S-panel space',
        va='center', fontsize=9, fontstyle='italic')
ax.text(len(s_key_markers) + 0.5, n_t_types + (len(ct_labels) - n_t_types) / 2,
        '← S-panel reference',
        va='center', fontsize=9, fontstyle='italic')
plt.colorbar(im, ax=ax, label='Z-score', shrink=0.6)
ax.set_title(f'FL32: T-panel cell types profiled in S-panel marker space\n'
             f'(top: T-panel types mapped to S-panel markers; bottom: S-panel native reference)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig('output/cross_panel_transfer_profiles.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: output/cross_panel_transfer_profiles.png")


# Figure 3: Reverse — T-panel marker profiles for S-panel cell types
ct_list_s = [ct for ct in s_cell_types.index if s_cell_types[ct] >= 50]
marker_matrix2 = []
ct_labels2 = []
for ct in ct_list_s:
    ct_patches_ids = df_patches[df_patches['s_type'] == ct]['patch'].values
    t_in_patches = np.isin(t_patch_id, ct_patches_ids)
    if t_in_patches.sum() < 20:
        continue
    row = []
    for m in t_key_markers:
        expr = raw_expr(t_raw, m)
        if expr is not None:
            row.append(float(expr[t_in_patches].mean()))
        else:
            row.append(0)
    marker_matrix2.append(row)
    ct_labels2.append(f"{ct} (S)")

# T-panel reference
for ct in t_cell_types.index:
    if t_cell_types[ct] < 50:
        continue
    mask = t_types == ct
    row = []
    for m in t_key_markers:
        expr = raw_expr(t_raw, m)
        if expr is not None:
            row.append(float(expr[mask].mean()))
        else:
            row.append(0)
    marker_matrix2.append(row)
    ct_labels2.append(f"{ct} (T-ref)")

mat2 = np.array(marker_matrix2)
mat2_z = (mat2 - mat2.mean(axis=0)) / (mat2.std(axis=0) + 1e-8)

fig, ax = plt.subplots(figsize=(16, max(8, len(ct_labels2) * 0.5)))
im = ax.imshow(mat2_z, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax.set_xticks(range(len(t_key_markers)))
ax.set_xticklabels(t_key_markers, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(ct_labels2)))
ax.set_yticklabels(ct_labels2, fontsize=8)
n_s_types = len([l for l in ct_labels2 if l.endswith('(S)')])
ax.axhline(n_s_types - 0.5, color='black', linewidth=2, linestyle='--')
ax.text(len(t_key_markers) + 0.5, n_s_types / 2, '← S-panel types\nin T-panel space',
        va='center', fontsize=9, fontstyle='italic')
ax.text(len(t_key_markers) + 0.5, n_s_types + (len(ct_labels2) - n_s_types) / 2,
        '← T-panel reference',
        va='center', fontsize=9, fontstyle='italic')
plt.colorbar(im, ax=ax, label='Z-score', shrink=0.6)
ax.set_title(f'FL32: S-panel cell types profiled in T-panel marker space\n'
             f'(top: S-panel types mapped to T-panel markers; bottom: T-panel native reference)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig('output/cross_panel_transfer_profiles_reverse.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: output/cross_panel_transfer_profiles_reverse.png")

print("\nDone!")
