#!/usr/bin/env python3
"""Publication-quality QC figures for segmentation, annotation, and marker QC.

Primary figures (pooled cohort, no per-TMA breakdown):
  1. fig_marker_qc.png        — marker dynamic range and QC
  2. fig_segmentation_qc.png  — segmentation quality
  3. fig_annotation_qc.png    — cell type annotation validation

Supplementary figures (per-TMA breakdowns):
  S1. fig_marker_qc_supp.png        — dynamic range heatmap, CD163, unassigned by TMA
  S2. fig_segmentation_qc_supp.png  — area and cell count by TMA
  S3. fig_annotation_qc_supp.png    — composition by TMA
"""

import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from collections import Counter

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TMA_COLORS = {'A1': '#4e79a7', 'B1': '#f28e2b', 'C1': '#e15759', 'Biomax': '#76b7b2'}
TMA_ORDER = ['A1', 'B1', 'C1', 'Biomax']

# Consolidation: merge only tiny residual categories
CONSOLIDATE_MAP = {
    'T cells':            'Other',       # 4 cells, residual category
}

CELL_TYPE_PALETTE = {
    'B cells':            '#1f77b4',
    'B cells (CD20hi)':   '#4393c3',
    'B cells (CXCR5hi)':  '#2166ac',
    'B cells (weak CD20)':'#92c5de',
    'B cells (TOXhi)':    '#053061',
    'B cells (BCL2+)':    '#4393c3',
    'B cells (PAX5+)':    '#2166ac',
    'GC B cells':         '#aec7e8',
    'Activated B / Plasmablast': '#dbdb8d',
    'CD4 T cells':        '#ff7f0e',
    'CD8 T cells':        '#d62728',
    'CD8 T pre-exhausted (TOX+)': '#e377c2',
    'CD8 T exhausted':    '#bcbd22',
    'Treg':               '#8c564b',
    'Macrophages (GzmB+)':  '#9467bd',
    'Macrophages':        '#7f7f7f',
    'FDC':                '#98df8a',
    'M1 Macrophages':     '#636363',
    'M2 Macrophages':     '#969696',
    'Dendritic cells':    '#e377c2',
    'Stromal / CAF':      '#8c564b',
    'Endothelial':        '#9467bd',
    'Myeloid (S100A9+)':  '#bcbd22',
    'FRC (PDPN+)':        '#dbdb8d',
    'Histiocytes (CD44hi)': '#c49c94',
    'Mixed / Border cells':'#c7c7c7',
    'pDC':                '#ffbb78',
    'Other':              '#c49c94',
    'Low quality / Unassigned': '#D3D3D3',
}

HEATMAP_MARKERS_T = [
    'CD20', 'CD3', 'CD4', 'CD8a', 'CD68', 'FoxP3',
    'GranzymeB', 'TOX', 'PD_1', 'CXCR5', 'CD38', 'IRF4',
]
HEATMAP_MARKERS_S = [
    'CD20', 'CD4', 'CD8a', 'CD68', 'CD21', 'PDPN',
    'Vimentin', 'CD163', 'CD206', 'S100A9', 'CD11c', 'PAX5', 'BCL_2',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_array(f, key):
    ds = f['obs'][key]
    if isinstance(ds, h5py.Group) and 'categories' in ds:
        cats = ds['categories'][:]
        codes = ds['codes'][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


def load_numeric(f, key):
    return f['obs'][key][:]


def get_marker_idx(f):
    key = '_index' if '_index' in f['var'] else 'index'
    names = f['var'][key][:]
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return {n: i for i, n in enumerate(names)}


def is_tumor_core(sample_id):
    s_lower = sample_id.lower()
    if '_ton_' in s_lower or '_adr_' in s_lower:
        return False
    for tissue in ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal']:
        if tissue in s_lower:
            return False
    if sample_id == 'Biomax_ROI_006':
        return False
    return True


def get_tumor_mask(sample_ids):
    return np.array([is_tumor_core(s) for s in sample_ids])


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f'$\\bf{{{letter}}}$', transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')


LQ = 'Low quality / Unassigned'


def consolidate_cell_types(cell_types):
    """Merge tiny residual categories only. B cell subtypes are preserved."""
    return np.array([CONSOLIDATE_MAP.get(ct, ct) for ct in cell_types])


# ===========================================================================
# PRIMARY FIGURES
# ===========================================================================

# ---------------------------------------------------------------------------
# Figure 1: Segmentation Quality (2x2)
# ---------------------------------------------------------------------------

def figure_segmentation(f_t, outdir):
    print('  Figure 2: Segmentation Quality ...')

    sample_ids = load_array(f_t, 'sample_id')
    cell_types = consolidate_cell_types(load_array(f_t, 'cell_type'))
    area = load_numeric(f_t, 'area')
    cx = load_numeric(f_t, 'centroid_x')
    cy = load_numeric(f_t, 'centroid_y')
    tumor = get_tumor_mask(sample_ids)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Segmentation example — embed existing PNG
    ax = axes[0, 0]
    seg_path = 'output/FL01_hybrid_segmentation.png'
    if os.path.exists(seg_path):
        img = mpimg.imread(seg_path)
        ax.imshow(img)
        ax.set_title('Cellpose segmentation (B1_FL01)', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Segmentation image\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='gray')
    ax.axis('off')
    panel_label(ax, 'a', x=-0.02)

    # (b) Cell area distribution
    ax = axes[0, 1]
    a_tumor = area[tumor]
    p99 = np.percentile(a_tumor, 99)
    a_clip = a_tumor[a_tumor <= p99]
    bin_edges = np.arange(0, int(p99) + 4, 2)  # 2-px wide, integer-aligned
    ax.hist(a_clip, bins=bin_edges, color='#4e79a7', edgecolor='none', alpha=0.8)
    med = np.median(a_tumor)
    ax.axvline(med, color='#d62728', ls='--', lw=1.5, label=f'Median = {med:.0f} px')
    ax.set_xlabel('Cell area (pixels)')
    ax.set_ylabel('Count')
    ax.set_title(f'Cell area distribution (n={len(a_tumor):,} tumor cells)', fontsize=10)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    panel_label(ax, 'b')

    # (c) Cell area by cell type — violin, ordered by median area (large→small)
    ax = axes[1, 0]
    ct_tumor = cell_types[tumor]
    a_tumor_all = area[tumor]
    ct_counts = Counter(ct_tumor)
    # Include all cell types with enough cells (including Unidentified)
    candidates = [t for t, c in ct_counts.most_common() if c >= 100]
    # Compute median area for each, sort descending
    median_areas = {}
    for t in candidates:
        median_areas[t] = float(np.median(a_tumor_all[ct_tumor == t]))
    top_by_area = sorted(candidates, key=lambda t: median_areas[t], reverse=True)[:9]
    data_by_type = []
    labels = []
    for t in top_by_area:
        vals = a_tumor_all[ct_tumor == t]
        p99t = np.percentile(vals, 99)
        data_by_type.append(vals[vals <= p99t])
        display = 'Unidentified' if t == LQ else t
        labels.append(display.replace(' cells', '').replace(' (', '\n('))
    parts = ax.violinplot(data_by_type, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        color = CELL_TYPE_PALETTE.get(top_by_area[i], '#888888')
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=TICK_SIZE, rotation=45, ha='right')
    ax.set_ylabel('Cell area (pixels)')
    ax.set_title('Cell area by cell type (T-panel)', fontsize=10)
    panel_label(ax, 'c')

    # (d) Spatial density: dense vs sparse core
    ax = axes[1, 1]
    non_lq = tumor & (cell_types != LQ)
    roi_nlq = Counter(sample_ids[non_lq])
    rois_sorted = sorted(roi_nlq.keys(), key=lambda s: roi_nlq[s])
    idx_sparse = len(rois_sorted) // 4
    idx_dense = 3 * len(rois_sorted) // 4
    roi_sparse = rois_sorted[idx_sparse]
    roi_dense = rois_sorted[idx_dense]

    ax.set_visible(False)
    pos = ax.get_position()
    for i, (roi, label) in enumerate([(roi_dense, 'Dense'), (roi_sparse, 'Sparse')]):
        sub_ax = fig.add_axes([
            pos.x0 + i * pos.width * 0.52, pos.y0,
            pos.width * 0.48, pos.height
        ])
        m = (sample_ids == roi) & tumor
        x, y, ct = cx[m], cy[m], cell_types[m]
        typed = ct != LQ
        if np.any(~typed):
            sub_ax.scatter(x[~typed], y[~typed], c='#D3D3D3', s=0.3, alpha=0.4,
                          edgecolors='none', rasterized=True, zorder=1)
        if np.any(typed):
            colors = [CELL_TYPE_PALETTE.get(t, '#888888') for t in ct[typed]]
            sub_ax.scatter(x[typed], y[typed], c=colors, s=0.5,
                          edgecolors='none', rasterized=True, zorder=2)
        sub_ax.set_aspect('equal')
        sub_ax.invert_yaxis()
        sub_ax.set_title(f'{label}: {roi}\n({roi_nlq[roi]:,} typed cells)', fontsize=8)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        for sp in sub_ax.spines.values():
            sp.set_visible(False)
        if i == 0:
            panel_label(sub_ax, 'd', x=-0.05)

    fig.suptitle('Figure 2: Segmentation Quality', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(outdir, 'fig_segmentation_qc.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f'    Saved: {out}')


# ---------------------------------------------------------------------------
# Figure 2: Cell Type Annotation (3x2)
# ---------------------------------------------------------------------------

def figure_annotation(f_t, f_s, X_t, X_s, outdir):
    print('  Figure 3: Cell Type Annotation ...')

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    for panel_idx, (f, X_mem, panel_name, palette, heatmap_markers) in enumerate([
        (f_t, X_t, 'T-panel', CELL_TYPE_PALETTE, HEATMAP_MARKERS_T),
        (f_s, X_s, 'S-panel', CELL_TYPE_PALETTE, HEATMAP_MARKERS_S),
    ]):
        sample_ids = load_array(f, 'sample_id')
        cell_types = consolidate_cell_types(load_array(f, 'cell_type'))
        tumor = get_tumor_mask(sample_ids)

        # --- (a)/(b) UMAP ---
        ax = axes[0, panel_idx]
        umap = f['obsm']['X_umap'][:]
        n_total = len(cell_types)
        n_sub = min(100_000, n_total)
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, n_sub, replace=False)
        rng.shuffle(idx)
        u = umap[idx]
        ct_sub = cell_types[idx]
        is_lq = ct_sub == LQ
        if np.any(is_lq):
            ax.scatter(u[is_lq, 0], u[is_lq, 1], c='#D3D3D3', s=0.3, alpha=0.3,
                      edgecolors='none', rasterized=True, zorder=1)
        typed = ~is_lq
        if np.any(typed):
            colors = [palette.get(t, '#888888') for t in ct_sub[typed]]
            ax.scatter(u[typed, 0], u[typed, 1], c=colors, s=0.3,
                      edgecolors='none', rasterized=True, zorder=2)
        ax.set_xlabel('UMAP 1', fontsize=LABEL_SIZE)
        ax.set_ylabel('UMAP 2', fontsize=LABEL_SIZE)
        ax.set_title(f'UMAP by cell type ({panel_name})', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        panel_label(ax, chr(ord('a') + panel_idx))

        # --- (c)/(d) Marker heatmap ---
        ax = axes[1, panel_idx]
        marker_idx = get_marker_idx(f)
        ct_counts = Counter(cell_types[tumor])
        ct_order = [t for t, _ in ct_counts.most_common() if t != LQ and t != 'T cells']
        ct_order.append(LQ)  # Unidentified as last row to show low/no expression
        marker_cols = []
        marker_names_valid = []
        for mname in heatmap_markers:
            if mname in marker_idx:
                marker_cols.append(marker_idx[mname])
                marker_names_valid.append(mname)
        heatmap_data = np.zeros((len(ct_order), len(marker_names_valid)))
        for i, ct_name in enumerate(ct_order):
            mask = tumor & (cell_types == ct_name)
            if not np.any(mask):
                continue
            for j, col_idx in enumerate(marker_cols):
                heatmap_data[i, j] = float(np.mean(X_mem[mask, col_idx]))

        # Z-score per marker (column)
        for j in range(heatmap_data.shape[1]):
            col_data = heatmap_data[:, j]
            std = np.std(col_data)
            if std > 0:
                heatmap_data[:, j] = (col_data - np.mean(col_data)) / std

        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set_xticks(range(len(marker_names_valid)))
        ax.set_xticklabels(marker_names_valid, rotation=45, ha='right', fontsize=TICK_SIZE)
        ct_labels = ['Unidentified' if t == LQ else t[:25] for t in ct_order]
        ax.set_yticks(range(len(ct_labels)))
        ax.set_yticklabels(ct_labels, fontsize=TICK_SIZE)
        ax.set_title(f'Marker expression by cell type ({panel_name})', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.6, label='Z-score')
        panel_label(ax, chr(ord('c') + panel_idx))

        # --- (e)/(f) Cell type composition (tumor cores, pooled) ---
        ax = axes[2, panel_idx]
        ct_tumor = cell_types[tumor]
        n_tumor = len(ct_tumor)
        ct_counts_tumor = Counter(ct_tumor)
        # Include LQ as "Unidentified" at the end
        ct_sorted = [t for t, _ in ct_counts_tumor.most_common() if t != LQ]
        ct_sorted.append(LQ)
        fracs = [ct_counts_tumor.get(t, 0) / n_tumor * 100 for t in ct_sorted]
        counts = [ct_counts_tumor.get(t, 0) for t in ct_sorted]
        display_names = [t if t != LQ else 'Unidentified' for t in ct_sorted]
        bar_colors = [palette.get(t, '#888888') for t in ct_sorted]

        y_pos = np.arange(len(ct_sorted))
        ax.barh(y_pos, fracs, color=bar_colors, edgecolor='none')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=TICK_SIZE)
        ax.invert_yaxis()
        ax.set_xlabel('% of tumor cells')
        ax.set_title(f'Cell type composition ({panel_name}, n={n_tumor:,})', fontsize=10)
        # Add count annotations
        for i, (frac, count) in enumerate(zip(fracs, counts)):
            if frac > 1.5:
                ax.text(frac + 0.3, i, f'{frac:.1f}%', va='center', fontsize=6)
        panel_label(ax, chr(ord('e') + panel_idx))

    # UMAP legends
    for pidx, f_h in enumerate([f_t, f_s]):
        ct_counts = Counter(consolidate_cell_types(load_array(f_h, 'cell_type')))
        top = [t for t, _ in ct_counts.most_common() if t != LQ][:10]
        handles = [Patch(facecolor=CELL_TYPE_PALETTE.get(t, '#888'), label=t[:20]) for t in top]
        handles.append(Patch(facecolor='#D3D3D3', alpha=0.5, label='Unidentified'))
        axes[0, pidx].legend(handles=handles, fontsize=LEGEND_SIZE, loc='lower right', ncol=2,
                             framealpha=0.8)

    fig.suptitle('Figure 3: Cell Type Annotation', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(outdir, 'fig_annotation_qc.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f'    Saved: {out}')


# ---------------------------------------------------------------------------
# Figure 3: Marker QC (1x3)
# ---------------------------------------------------------------------------

def figure_marker_qc(qc_stats, outdir):
    """Figure 3: Marker QC using arcsinh-transformed (not z-scored) expression."""
    print('  Figure 1: Marker QC ...')

    from matplotlib.lines import Line2D

    markers = list(qc_stats['t_markers'])
    pooled_p99 = qc_stats['t_pooled_p99']
    tmas = list(qc_stats['t_tmas'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Pooled marker dynamic range — horizontal bar of p99 per marker
    ax = axes[0]
    skip = {'DNA1', 'DNA2', 'HistoneH3', 'p_H3s28', 'H3K27me3'}
    bio_idx = [i for i, m in enumerate(markers) if m not in skip]
    bio_markers = [markers[i] for i in bio_idx]
    bio_p99 = [float(pooled_p99[i]) for i in bio_idx]
    order = np.argsort(bio_p99)
    markers_sorted = [bio_markers[i] for i in order]
    p99_sorted = [bio_p99[i] for i in order]
    colors = ['#d62728' if v < 0.3 else '#ff7f0e' if v < 1.0 else '#2ca02c' for v in p99_sorted]
    ax.barh(range(len(markers_sorted)), p99_sorted, color=colors, edgecolor='none')
    ax.set_yticks(range(len(markers_sorted)))
    ax.set_yticklabels(markers_sorted, fontsize=TICK_SIZE)
    ax.set_xlabel('p99 expression (arcsinh, pooled tumor)')
    ax.set_title('Marker dynamic range (T-panel)', fontsize=10)
    ax.axvline(0.3, color='gray', ls=':', lw=1, alpha=0.5)
    ax.axvline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
    leg = [Line2D([0], [0], color='#d62728', lw=6, label='Dead (p99 < 0.3)'),
           Line2D([0], [0], color='#ff7f0e', lw=6, label='Weak (0.3–1.0)'),
           Line2D([0], [0], color='#2ca02c', lw=6, label='Good (> 1.0)')]
    ax.legend(handles=leg, fontsize=LEGEND_SIZE, loc='lower right')
    panel_label(ax, 'a')

    # (b) Good vs dead marker: CD20 vs LAG3 — pooled across TMAs
    ax = axes[1]
    cd20_all = np.concatenate([qc_stats['t_sub_CD20_%s' % t] for t in tmas])
    lag3_all = np.concatenate([qc_stats['t_sub_LAG3_%s' % t] for t in tmas])
    parts1 = ax.violinplot([cd20_all], positions=[1], showmedians=True, showextrema=False)
    for pc in parts1['bodies']:
        pc.set_facecolor('#2ca02c')
        pc.set_alpha(0.7)
    parts1['cmedians'].set_color('black')
    parts2 = ax.violinplot([lag3_all], positions=[2], showmedians=True, showextrema=False)
    for pc in parts2['bodies']:
        pc.set_facecolor('#d62728')
        pc.set_alpha(0.7)
    parts2['cmedians'].set_color('black')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['CD20\n(functional)', 'LAG3\n(dead)'], fontsize=10)
    ax.set_ylabel('Expression (arcsinh)')
    ax.set_title('Good vs dead marker', fontsize=TITLE_SIZE)
    panel_label(ax, 'b')

    # (c) CXCR5 distribution
    ax = axes[2]
    cxcr5_all = np.concatenate([qc_stats['t_sub_CXCR5_%s' % t] for t in tmas])
    ax.hist(cxcr5_all, bins=100, color='#4e79a7', edgecolor='none', alpha=0.8)
    ax.axvline(2.0, color='#d62728', ls='--', lw=2, label='Threshold = 2.0 (p90)')
    ax.axvline(0.5, color='#ff7f0e', ls=':', lw=1.5, label='Standard threshold = 0.5')
    ax.set_xlabel('CXCR5 expression (arcsinh)')
    ax.set_ylabel('Count')
    ax.set_title('CXCR5: continuous, not bimodal', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    panel_label(ax, 'c')

    fig.suptitle('Figure 1: Marker QC', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(outdir, 'fig_marker_qc.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f'    Saved: {out}')


# ===========================================================================
# COMBINED QC FIGURE (Supplementary S1)
# ===========================================================================

def figure_qc_combined(f_t, qc_stats, outdir):
    """Supplementary S1: Marker QC + segmentation + composition.

    Layout (3 rows):
      Row 1 (2 cols, wide):
        (a) T-panel marker dynamic range
        (b) S-panel marker dynamic range
      Row 2 (3 equal cols):
        (c) Segmentation illustration — zoomed ROI, cells drawn as circles
            sized by area, colored by cell type
        (d) Cell area distribution histogram
        (e) Cell area by cell type — violin
      Row 3 (1 col, right-aligned):
        (f) FL vs tonsil composition stacked bar
    """
    from matplotlib.lines import Line2D
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle
    from matplotlib.gridspec import GridSpec

    print('  Fig S1: Combined QC figure ...')

    # Load obs-only fields (fast — no X)
    sample_ids = load_array(f_t, 'sample_id')
    cell_types  = consolidate_cell_types(load_array(f_t, 'cell_type'))
    area        = load_numeric(f_t, 'area')
    cx          = load_numeric(f_t, 'centroid_x')
    cy          = load_numeric(f_t, 'centroid_y')
    tumor       = get_tumor_mask(sample_ids)
    a_tumor     = area[tumor]
    ct_tumor    = cell_types[tumor]

    # Segmentation illustration: hardcoded to B1_FL21 for figure stability across
    # annotation revisions. Falls back to the 75th-percentile typed-count ROI if
    # the fixed ROI is absent for any reason.
    SEG_ROI_FIXED = 'B1_FL21'
    if SEG_ROI_FIXED in set(sample_ids):
        roi_dense = SEG_ROI_FIXED
    else:
        non_lq      = tumor & (cell_types != LQ)
        roi_nlq     = Counter(sample_ids[non_lq])
        rois_sorted = sorted(roi_nlq.keys(), key=lambda s: roi_nlq[s])
        roi_dense   = rois_sorted[3 * len(rois_sorted) // 4]

    fig = plt.figure(figsize=(20, 24))
    # Row 1: marker QC (2 wide panels) — tallest row (many markers)
    gs_top = GridSpec(1, 2, figure=fig, top=0.97, bottom=0.60,
                      left=0.06, right=0.98, wspace=0.30)
    # Row 2: segmentation panels (3 equal cols) — height matches square panel c
    gs_mid = GridSpec(1, 3, figure=fig, top=0.55, bottom=0.33,
                      left=0.04, right=0.98, wspace=0.30)
    # Row 3: composition (1 panel in column 0, same 3-col grid as Row 2 so the
    # panel width matches panel c above — partial row aligned, not stretched/centered)
    gs_bot = GridSpec(1, 3, figure=fig, top=0.27, bottom=0.04,
                      left=0.04, right=0.98, wspace=0.30)

    # ── Helper: marker dynamic range bar chart ───────────────────────────────
    def _plot_dynamic_range(ax, panel_prefix, panel_name, panel_letter):
        skip = {'DNA1', 'DNA2', 'HistoneH3', 'p_H3s28', 'H3K27me3'}
        markers_all = list(qc_stats[f'{panel_prefix}_markers'])
        pooled_p99  = qc_stats[f'{panel_prefix}_pooled_p99']
        bio_idx     = [i for i, m in enumerate(markers_all) if m not in skip]
        bio_markers = [markers_all[i] for i in bio_idx]
        bio_p99     = [float(pooled_p99[i]) for i in bio_idx]
        order          = np.argsort(bio_p99)
        markers_sorted = [bio_markers[i] for i in order]
        p99_sorted     = [bio_p99[i]     for i in order]
        colors = ['#d62728' if v < 0.3 else '#ff7f0e' if v < 1.0 else '#2ca02c'
                  for v in p99_sorted]
        ax.barh(range(len(markers_sorted)), p99_sorted, color=colors,
                edgecolor='none', height=0.7)
        ax.set_yticks(range(len(markers_sorted)))
        ax.set_yticklabels(markers_sorted, fontsize=TICK_SIZE)
        ax.set_xlabel('p99 expression (arcsinh, pooled tumor)', fontsize=15)
        ax.set_title(f'Marker dynamic range ({panel_name})', fontsize=16,
                     fontweight='medium')
        ax.axvline(0.3, color='gray', ls=':', lw=1.5, alpha=0.6)
        ax.axvline(1.0, color='gray', ls=':', lw=1.5, alpha=0.6)
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        leg = [Line2D([0], [0], color='#d62728', lw=10, label='Low (p99 < 0.3)'),
               Line2D([0], [0], color='#ff7f0e', lw=10, label='Average (0.3–1.0)'),
               Line2D([0], [0], color='#2ca02c', lw=10, label='Good (> 1.0)')]
        ax.legend(handles=leg, fontsize=LEGEND_SIZE, loc='lower right')
        panel_label(ax, panel_letter)

    # ── (a) T-panel marker dynamic range ─────────────────────────────────────
    ax_a = fig.add_subplot(gs_top[0, 0])
    _plot_dynamic_range(ax_a, 't', 'T-panel', 'a')

    # ── (b) S-panel marker dynamic range ─────────────────────────────────────
    ax_b = fig.add_subplot(gs_top[0, 1])
    _plot_dynamic_range(ax_b, 's', 'S-panel', 'b')

    # ── (c) Segmentation illustration: zoomed ROI, cells as area-scaled circles ──
    ax_c = fig.add_subplot(gs_mid[0, 0])
    m_roi = (sample_ids == roi_dense) & tumor
    x_r, y_r, a_r, ct_r = cx[m_roi], cy[m_roi], area[m_roi], cell_types[m_roi]
    sub = 350
    xc, yc = (x_r.max() + x_r.min()) / 2, (y_r.max() + y_r.min()) / 2
    in_sub = ((x_r >= xc - sub/2) & (x_r <= xc + sub/2) &
              (y_r >= yc - sub/2) & (y_r <= yc + sub/2))
    xs, ys, as_, cts = x_r[in_sub], y_r[in_sub], a_r[in_sub], ct_r[in_sub]
    radii = np.sqrt(as_ / np.pi)
    patches = [Circle((xi, yi), ri) for xi, yi, ri in zip(xs, ys, radii)]
    face_colors = [CELL_TYPE_PALETTE.get(t, '#888888') for t in cts]
    coll = PatchCollection(patches, facecolors=face_colors,
                           edgecolors='white', linewidths=0.25, alpha=0.9)
    ax_c.add_collection(coll)
    ax_c.set_xlim(xc - sub/2, xc + sub/2)
    ax_c.set_ylim(yc + sub/2, yc - sub/2)   # inverted y
    ax_c.set_aspect('equal')
    ax_c.set_title(f'Segmentation: {roi_dense}\n(350×350 µm crop)', fontsize=16,
                   fontweight='medium')
    ax_c.set_xticks([]); ax_c.set_yticks([])
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    panel_label(ax_c, 'c', x=-0.04)

    # ── (d) Cell area distribution ───────────────────────────────────────────
    ax_d = fig.add_subplot(gs_mid[0, 1])
    p99 = np.percentile(a_tumor, 99)
    a_clip = a_tumor[a_tumor <= p99]
    ax_d.hist(a_clip, bins=np.arange(0, int(p99) + 4, 2),
              color='#4e79a7', edgecolor='none', alpha=0.8)
    med = np.median(a_tumor)
    ax_d.axvline(med, color='#d62728', ls='--', lw=2,
                 label=f'Median = {med:.0f} µm²')
    ax_d.set_xlabel('Cell area (µm²)', fontsize=15)
    ax_d.set_ylabel('Count', fontsize=LABEL_SIZE)
    ax_d.set_title(f'Cell area distribution\n(n={len(a_tumor):,} tumor cells)',
                   fontsize=16, fontweight='medium')
    ax_d.legend(fontsize=LEGEND_SIZE)
    ax_d.tick_params(labelsize=TICK_SIZE)
    ax_d.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    panel_label(ax_d, 'd')

    # ── (e) Cell area by broad cell type — violin ─────────────────────────────
    ax_e = fig.add_subplot(gs_mid[0, 2])
    # Group into broad categories
    broad_groups = {
        'B cells': ['GC B cells', 'B cells (CD20hi)', 'B cells (CXCR5hi)',
                     'Other B cells', 'B cells (TOXhi)', 'Activated B / Plasmablast',
                     'B cells (weak CD20)', 'B cells'],
        'CD4 T': ['CD4 T cells'],
        'CD8 T': ['CD8 T cells', 'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)',
                  'Macrophages (GzmB+)'],
        'Treg': ['Treg'],
        'Macrophages': ['Macrophages'],
        'Unidentified': [LQ],
    }
    broad_colors = {
        'B cells': '#1f77b4', 'CD4 T': '#ff7f0e', 'CD8 T': '#d62728',
        'Treg': '#8c564b', 'Macrophages': '#7f7f7f', 'Unidentified': '#D3D3D3',
    }
    broad_order = ['Macrophages', 'B cells', 'Unidentified', 'Treg', 'CD8 T', 'CD4 T']
    data_v, labels_v, colors_v = [], [], []
    for grp in broad_order:
        mask = np.isin(ct_tumor, broad_groups[grp])
        if mask.sum() < 100:
            continue
        vals = a_tumor[mask]
        p99t = np.percentile(vals, 99)
        data_v.append(vals[vals <= p99t])
        labels_v.append(grp)
        colors_v.append(broad_colors[grp])
    parts = ax_e.violinplot(data_v, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_v[i])
        pc.set_alpha(0.75)
    parts['cmedians'].set_color('black')
    ax_e.set_xticks(range(1, len(labels_v) + 1))
    ax_e.set_xticklabels(labels_v, fontsize=TICK_SIZE, rotation=40, ha='right')
    ax_e.set_ylabel('Cell area (µm²)', fontsize=15)
    ax_e.set_title('Cell area by cell type (T-panel)', fontsize=16,
                   fontweight='medium')
    ax_e.tick_params(axis='y', labelsize=TICK_SIZE)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    panel_label(ax_e, 'e')

    # ── (f) FL vs tonsil composition — stacked bar (column 0, aligned with row 2) ──
    ax_f = fig.add_subplot(gs_bot[0, 0])
    # Build tonsil comparison data from f_t
    from fig_tonsil_comparison import plot_tonsil_composition, is_tonsil, is_tumor
    _ct = cell_types  # already loaded above (full array, not just tumor)
    _sid = sample_ids
    _tonsil_mask = np.array([is_tonsil(s) for s in _sid])
    _tumor_mask = np.array([is_tumor(s) for s in _sid])
    _typed_mask = ~np.isin(_ct, ['Unassigned', 'Low quality / Unassigned'])
    tonsil_data = {
        'cell_types': _ct,
        'tumor_mask': _tumor_mask,
        'tonsil_mask': _tonsil_mask,
        'typed_mask': _typed_mask,
    }
    plot_tonsil_composition(ax_f, tonsil_data)
    panel_label(ax_f, 'f')

    out = os.path.join(outdir, 'fig_qc_combined.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f'    Saved: {out}')


# ===========================================================================
# SUPPLEMENTARY FIGURES
# ===========================================================================

# ---------------------------------------------------------------------------
# Fig S1: Segmentation per-TMA
# ---------------------------------------------------------------------------

def figure_segmentation_supp(f_t, outdir):
    print('  Fig S2: Segmentation per-TMA ...')

    sample_ids = load_array(f_t, 'sample_id')
    tmas = load_array(f_t, 'tma')
    area = load_numeric(f_t, 'area')
    tumor = get_tumor_mask(sample_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Build ROI-level data
    roi_median_area = {}
    roi_tma = {}
    roi_counts = Counter(sample_ids[tumor])
    for sid in sorted(set(sample_ids[tumor])):
        m = (sample_ids == sid) & tumor
        roi_median_area[sid] = np.median(area[m])
        roi_tma[sid] = tmas[m][0]

    # (a) Cell area by TMA
    ax = axes[0]
    box_data, box_colors = [], []
    for tma in TMA_ORDER:
        vals = [v for s, v in roi_median_area.items() if roi_tma[s] == tma]
        box_data.append(vals)
        box_colors.append(TMA_COLORS[tma])
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(TMA_ORDER)
    ax.set_ylabel('Median cell area per ROI (px)')
    ax.set_title('Segmentation consistency across TMAs', fontsize=TITLE_SIZE)
    panel_label(ax, 'a')

    # (b) Cells per ROI
    ax = axes[1]
    for tma in TMA_ORDER:
        vals = [roi_counts[s] for s in roi_counts if roi_tma.get(s) == tma]
        x_jit = np.random.normal(TMA_ORDER.index(tma) + 1, 0.08, len(vals))
        ax.scatter(x_jit, vals, c=TMA_COLORS[tma], s=20, alpha=0.6,
                   edgecolors='white', linewidths=0.3, zorder=3)
    counts_by_tma = [[roi_counts[s] for s in roi_counts if roi_tma.get(s) == tma]
                     for tma in TMA_ORDER]
    ax.boxplot(counts_by_tma, widths=0.4, zorder=2,
               boxprops=dict(color='black'), whiskerprops=dict(color='black'),
               medianprops=dict(color='red', lw=2))
    ax.set_xticklabels(TMA_ORDER)
    ax.set_ylabel('Cells per ROI')
    ax.set_title('Data volume per ROI by TMA', fontsize=TITLE_SIZE)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    panel_label(ax, 'b')

    fig.suptitle('Figure S2: Segmentation — per-TMA', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(outdir, 'fig_segmentation_qc_supp.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f'    Saved: {out}')


# ---------------------------------------------------------------------------
# Fig S2: Annotation per-TMA
# ---------------------------------------------------------------------------

def figure_annotation_supp(f_t, f_s, outdir):
    print('  Fig S3: Annotation per-TMA ...')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for panel_idx, (f, panel_name, palette) in enumerate([
        (f_t, 'T-panel', CELL_TYPE_PALETTE),
        (f_s, 'S-panel', CELL_TYPE_PALETTE),
    ]):
        ax = axes[panel_idx]
        sample_ids = load_array(f, 'sample_id')
        cell_types = consolidate_cell_types(load_array(f, 'cell_type'))
        tmas = load_array(f, 'tma')
        tumor = get_tumor_mask(sample_ids)
        typed_tumor = tumor & (cell_types != LQ)

        ct_counts = Counter(cell_types[typed_tumor])
        top_types = [t for t, _ in ct_counts.most_common(12)]
        fracs = np.zeros((len(TMA_ORDER), len(top_types)))
        for i, tma in enumerate(TMA_ORDER):
            m = typed_tumor & (tmas == tma)
            n = np.sum(m)
            if n == 0:
                continue
            ct_here = Counter(cell_types[m])
            for j, t in enumerate(top_types):
                fracs[i, j] = ct_here.get(t, 0) / n

        bottom = np.zeros(len(TMA_ORDER))
        x_pos = np.arange(len(TMA_ORDER))
        for j, t in enumerate(top_types):
            color = palette.get(t, '#888888')
            ax.bar(x_pos, fracs[:, j], bottom=bottom, color=color, label=t[:25],
                   width=0.7, edgecolor='white', linewidth=0.3)
            bottom += fracs[:, j]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(TMA_ORDER)
        ax.set_ylabel('Fraction')
        ax.set_title(f'Cell type composition by TMA ({panel_name})', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=LEGEND_SIZE, loc='upper right', ncol=2, bbox_to_anchor=(1.0, 1.0))
        panel_label(ax, chr(ord('a') + panel_idx))

    fig.suptitle('Figure S3: Cell Type Composition — per-TMA', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(outdir, 'fig_annotation_qc_supp.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f'    Saved: {out}')


# ---------------------------------------------------------------------------
# Fig S3: Marker QC per-TMA
# ---------------------------------------------------------------------------

def figure_marker_qc_supp(f_t, f_s, qc_stats, outdir):
    """Fig S3: Marker QC per-TMA using arcsinh-transformed data from npz."""
    print('  Fig S1: Marker QC per-TMA ...')

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a)/(b) Dynamic range heatmap by TMA — from precomputed arcsinh p99
    for panel_idx, (panel_prefix, panel_name) in enumerate([('t', 'T-panel'), ('s', 'S-panel')]):
        ax = axes[0, panel_idx]
        markers_all = list(qc_stats['%s_markers' % panel_prefix])

        skip = {'DNA1', 'DNA2', 'HistoneH3', 'p_H3s28', 'H3K27me3'}
        bio_idx = [i for i, m in enumerate(markers_all) if m not in skip]
        markers = [markers_all[i] for i in bio_idx]

        p99_matrix = np.zeros((len(markers), len(TMA_ORDER)))
        for j, tma in enumerate(TMA_ORDER):
            p99_all = qc_stats['%s_p99_%s' % (panel_prefix, tma)]
            for i, midx in enumerate(bio_idx):
                p99_matrix[i, j] = float(p99_all[midx])

        im = ax.imshow(p99_matrix, aspect='auto', cmap='RdYlGn', vmin=0,
                       vmax=max(np.percentile(p99_matrix[p99_matrix > 0], 90), 0.5))
        ax.set_xticks(range(len(TMA_ORDER)))
        ax.set_xticklabels(TMA_ORDER, fontsize=TICK_SIZE)
        ax.set_yticks(range(len(markers)))
        ax.set_yticklabels(markers, fontsize=TICK_SIZE)
        ax.set_title(f'p99 arcsinh expression by TMA ({panel_name})', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.6, label='p99 (arcsinh)')
        panel_label(ax, chr(ord('a') + panel_idx))

    # (c) CD163 by TMA (S-panel) — from precomputed subsamples
    ax = axes[1, 0]
    violin_data = []
    for tma in TMA_ORDER:
        key = 's_sub_CD163_%s' % tma
        if key in qc_stats:
            violin_data.append(qc_stats[key])
        else:
            violin_data.append(np.array([0.0]))

    parts = ax.violinplot(violin_data, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(TMA_COLORS[TMA_ORDER[i]])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax.set_xticks(range(1, len(TMA_ORDER) + 1))
    ax.set_xticklabels(TMA_ORDER)
    ax.set_ylabel('CD163 expression (arcsinh)')
    ax.set_title('CD163: dead in A1/B1, functional in Biomax', fontsize=TITLE_SIZE)
    panel_label(ax, 'c')

    # (d) Unassigned cell fraction by ROI
    ax = axes[1, 1]
    sample_ids_t = load_array(f_t, 'sample_id')
    cell_types_t = consolidate_cell_types(load_array(f_t, 'cell_type'))
    tmas_t = load_array(f_t, 'tma')
    tumor_t = get_tumor_mask(sample_ids_t)

    roi_lq_frac = {}
    roi_tma = {}
    for sid in sorted(set(sample_ids_t[tumor_t])):
        m = (sample_ids_t == sid) & tumor_t
        n = np.sum(m)
        n_lq = np.sum(cell_types_t[m] == LQ)
        roi_lq_frac[sid] = n_lq / n * 100
        roi_tma[sid] = tmas_t[m][0]

    for tma in TMA_ORDER:
        rois = [s for s in roi_lq_frac if roi_tma[s] == tma]
        vals = [roi_lq_frac[s] for s in rois]
        x_jit = np.random.normal(TMA_ORDER.index(tma) + 1, 0.08, len(vals))
        ax.scatter(x_jit, vals, c=TMA_COLORS[tma], s=20, alpha=0.6,
                   edgecolors='white', linewidths=0.3, zorder=3)
    box_data = [[roi_lq_frac[s] for s in roi_lq_frac if roi_tma[s] == tma]
                for tma in TMA_ORDER]
    ax.boxplot(box_data, widths=0.4, zorder=2,
               boxprops=dict(color='black'), whiskerprops=dict(color='black'),
               medianprops=dict(color='red', lw=2))
    ax.set_xticklabels(TMA_ORDER)
    ax.set_ylabel('% Unidentified cells per ROI')
    ax.set_title('Unidentified cell fraction by ROI', fontsize=TITLE_SIZE)
    panel_label(ax, 'd')

    fig.suptitle('Figure S1: Marker QC — per-TMA', fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(outdir, 'fig_marker_qc_supp.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f'    Saved: {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate QC figures for IMC-FL paper')
    parser.add_argument('--t-panel', required=True, help='T-panel h5ad file')
    parser.add_argument('--s-panel', required=True, help='S-panel h5ad file')
    parser.add_argument('--qc-stats', default='output/marker_qc_stats.npz',
                        help='Marker QC stats npz (arcsinh p99/subsamples)')
    parser.add_argument('--output-dir', default='output/qc', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading data ...')
    f_t = h5py.File(args.t_panel, 'r')
    f_s = h5py.File(args.s_panel, 'r')

    print('  Loading T-panel X into memory ...')
    X_t = f_t['X'][:]
    print(f'    {X_t.shape}, {X_t.dtype}')
    print('  Loading S-panel X into memory ...')
    X_s = f_s['X'][:]
    print(f'    {X_s.shape}, {X_s.dtype}')

    print('  Loading marker QC stats ...')
    qc_stats = dict(np.load(args.qc_stats, allow_pickle=True))
    print(f'    {len(qc_stats)} arrays loaded')

    # Primary figures
    figure_marker_qc(qc_stats, args.output_dir)
    figure_segmentation(f_t, args.output_dir)
    figure_annotation(f_t, f_s, X_t, X_s, args.output_dir)

    # Supplementary figures
    figure_qc_combined(f_t, qc_stats, args.output_dir)   # S1: combined seg + marker QC
    figure_marker_qc_supp(f_t, f_s, qc_stats, args.output_dir)
    figure_segmentation_supp(f_t, args.output_dir)
    figure_annotation_supp(f_t, f_s, args.output_dir)

    f_t.close()
    f_s.close()
    print('Done — 7 figures generated (3 primary + 4 supplementary).')


if __name__ == '__main__':
    main()
