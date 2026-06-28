#!/usr/bin/env python3
"""Plot raw marker expression spatially for paired ROIs across T and S panels."""
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import issparse

# --- Config ---
ROIs = ['C1_FL44', 'B1_FL32', 'B1_FL18', 'B1_FL10', 'B1_FL26', 'A1_FL5']
ROI_LABELS = [
    'C1_FL44 (concordant)', 'B1_FL32 (concordant)',
    'B1_FL18 (moderate)', 'B1_FL10 (moderate)',
    'B1_FL26 (discordant)', 'A1_FL5 (discordant)',
]

# Key markers: shared first, then panel-specific
SHARED_MARKERS = ['CD20', 'CD4', 'CD8a', 'CD68']
S_SPECIFIC = ['PAX5', 'BCL_2']
T_SPECIFIC = ['CD3', 'FoxP3']

print("Loading S-panel...")
s_adata = ad.read_h5ad('output/all_TMA_S_global_v5.h5ad')
print("Loading T-panel...")
t_adata = ad.read_h5ad('output/all_TMA_T_global_v5.h5ad')

# Build sample_id from obs columns
for adata, label in [(s_adata, 'S'), (t_adata, 'T')]:
    if 'sample_id' not in adata.obs.columns:
        adata.obs['sample_id'] = adata.obs['tma'].astype(str) + '_' + adata.obs['roi'].astype(str)

# Get raw expression matrices
s_raw = s_adata.raw.to_adata()
t_raw = t_adata.raw.to_adata()


def get_raw_expr(raw_adata, marker):
    """Get raw expression vector for a marker."""
    idx = list(raw_adata.var_names).index(marker)
    x = raw_adata.X[:, idx]
    if issparse(x):
        x = x.toarray().ravel()
    else:
        x = np.asarray(x).ravel()
    return x


def get_spatial(adata):
    """Get x, y spatial coords."""
    if 'spatial' in adata.obsm:
        return adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1]
    elif 'X_spatial' in adata.obsm:
        return adata.obsm['X_spatial'][:, 0], adata.obsm['X_spatial'][:, 1]
    else:
        return adata.obs['centroid_x'].values, adata.obs['centroid_y'].values


# For each panel, determine which markers to show
s_markers = SHARED_MARKERS + S_SPECIFIC  # CD20, CD4, CD8a, CD68, PAX5, BCL_2
t_markers = SHARED_MARKERS + T_SPECIFIC  # CD20, CD4, CD8a, CD68, CD3, FoxP3

n_markers = len(s_markers)  # 6
n_rois = len(ROIs)

# Layout: rows = ROIs, columns = markers * 2 panels (S then T side by side)
# Better: for each marker, show S and T side by side
# Let's do: rows = markers (6), columns = ROIs (6), with S on top half, T on bottom half
# Actually clearest: 2 figures - one per panel, or one big figure

# Best layout: For each ROI pair, show a column. For rows: S-panel markers, then T-panel markers
# But that's complex. Let's do 2 separate figures.

# Figure 1: Shared markers across both panels for all 6 ROIs
# Layout: rows = shared markers (4), cols = ROIs (6), each cell split into S (left) and T (right)
# Too complex. Simpler: one figure per marker.

# Simplest clear layout: one figure with rows = markers, cols = ROIs
# Two figures: one for S-panel, one for T-panel

for panel_label, adata, raw_adata, markers in [
    ('S-panel', s_adata, s_raw, s_markers),
    ('T-panel', t_adata, t_raw, t_markers),
]:
    print(f"\nPlotting {panel_label}...")
    sx, sy = get_spatial(adata)
    sample_ids = adata.obs['sample_id'].values

    fig, axes = plt.subplots(n_markers, n_rois, figsize=(n_rois * 3, n_markers * 3))

    for j, (roi, roi_label) in enumerate(zip(ROIs, ROI_LABELS)):
        mask = sample_ids == roi
        if mask.sum() == 0:
            # Try alternative naming
            for alt in [roi.replace('_', '-'), roi]:
                mask = sample_ids == alt
                if mask.sum() > 0:
                    break

        n_cells = mask.sum()
        x_roi = sx[mask]
        y_roi = sy[mask]

        for i, marker in enumerate(markers):
            ax = axes[i, j]

            if n_cells == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(roi_label, fontsize=9, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(marker, fontsize=10, fontweight='bold')
                continue

            expr = get_raw_expr(raw_adata, marker)[mask]

            # Clip to 99th percentile for better visualization
            vmax = np.percentile(expr, 99)
            if vmax <= 0:
                vmax = 1.0

            sc = ax.scatter(x_roi, y_roi, c=expr, s=0.3, cmap='magma',
                          vmin=0, vmax=vmax, rasterized=True)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()

            if i == 0:
                ax.set_title(roi_label, fontsize=8, fontweight='bold')
            if j == 0:
                ax.set_ylabel(marker, fontsize=10, fontweight='bold')

            # Add colorbar for rightmost column
            if j == n_rois - 1:
                cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=6)

    fig.suptitle(f'{panel_label} — Raw marker expression (6 paired ROIs)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    outpath = f'output/paired_roi_raw_markers_{panel_label.replace("-", "").replace(" ", "_").lower()}.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")

# Figure 3: Side-by-side comparison of shared markers only
# For each shared marker, show S-panel and T-panel rows for the 6 ROIs
print("\nPlotting side-by-side shared markers...")
n_shared = len(SHARED_MARKERS)
fig, axes = plt.subplots(n_shared * 2, n_rois, figsize=(n_rois * 3, n_shared * 2 * 2.5))

s_sx, s_sy = get_spatial(s_adata)
t_sx, t_sy = get_spatial(t_adata)
s_ids = s_adata.obs['sample_id'].values
t_ids = t_adata.obs['sample_id'].values

for j, (roi, roi_label) in enumerate(zip(ROIs, ROI_LABELS)):
    s_mask = s_ids == roi
    t_mask = t_ids == roi

    for mi, marker in enumerate(SHARED_MARKERS):
        row_s = mi * 2       # S-panel row
        row_t = mi * 2 + 1   # T-panel row

        for row_idx, panel_name, mask, raw_ad, px, py in [
            (row_s, 'S', s_mask, s_raw, s_sx, s_sy),
            (row_t, 'T', t_mask, t_raw, t_sx, t_sy),
        ]:
            ax = axes[row_idx, j]
            n_cells = mask.sum()

            if n_cells == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                expr = get_raw_expr(raw_ad, marker)[mask]
                vmax = np.percentile(expr, 99)
                if vmax <= 0:
                    vmax = 1.0

                sc = ax.scatter(px[mask], py[mask], c=expr, s=0.3,
                              cmap='magma', vmin=0, vmax=vmax, rasterized=True)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.invert_yaxis()

                if j == n_rois - 1:
                    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    cb.ax.tick_params(labelsize=6)

            if j == 0:
                ax.set_ylabel(f'{marker}\n({panel_name})', fontsize=9, fontweight='bold')

            if row_idx == 0:
                ax.set_title(roi_label, fontsize=8, fontweight='bold')

fig.suptitle('Shared markers — S-panel vs T-panel raw expression',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('output/paired_roi_raw_markers_shared.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: output/paired_roi_raw_markers_shared.png")

print("\nDone!")
