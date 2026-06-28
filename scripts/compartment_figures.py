#!/usr/bin/env python3
"""Compartment validation figures: combined T+S panel overview.

Shows spatial maps, composition heatmaps, marker expression, cross-TMA
reproducibility, and compartment frequency to validate UTAG tissue compartments.
"""

import argparse, os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from collections import Counter
from scipy.stats import zscore

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22


# ---------------------------------------------------------------------------
# Helpers (reused from run_hypotheses_v2.py)
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


def get_marker_names(f):
    key = '_index' if '_index' in f['var'] else 'index'
    names = f['var'][key][:]
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def get_tumor_mask(sample_ids):
    control_tags = ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal',
                    '_ton_', '_adr_']
    mask = np.array([not any(t in s.lower() for t in control_tags)
                     for s in sample_ids])
    return mask


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Display renames: remove "LQ" from labels (cells are Unassigned, not Low quality)
# Applied after loading from h5ad. The h5ad data retains original names.
# ---------------------------------------------------------------------------

DISPLAY_RENAME = {
    # Cell types
    'Low quality / Unassigned': 'Unassigned',
    'B cells': 'Other B cells',
    # Compartments
    'LQ / B transitional': 'B / Unassigned transitional',
    'Cytotoxic / LQ niche': 'Cytotoxic niche',
    'Weak CD20 / LQ border': 'Weak CD20 border',
}

def rename_labels(arr):
    """Apply display renames to a string array."""
    return np.array([DISPLAY_RENAME.get(v, v) for v in arr])


T_COMPARTMENT_COLORS = {
    # Follicular (warm)
    'GC core': '#B22222',
    'Follicle core (GC/CD20hi/CXCR5hi)': '#DC143C',
    'Follicle mantle (CXCR5hi)': '#E8734A',
    'Activated B / CXCR5hi zone': '#FF8C00',
    'B cell follicle (CD20hi/CXCR5hi)': '#E06060',
    'B cell zone': '#DAA520',
    # Interfollicular (cool)
    'T cell zone (CD4/CD8)': '#4169E1',
    'Treg-enriched T zone': '#20B2AA',
    'Macrophage-rich zone': '#191970',
    'Follicle-T zone interface': '#6495ED',
    'Cytotoxic niche': '#5F9EA0',
    # Excluded (gray)
    'B / Unassigned transitional': '#A9A9A9',
    'Unidentified zone': '#C0C0C0',
    'Weak CD20 border': '#BEBEBE',
}

S_COMPARTMENT_COLORS = {
    # Follicular (warm)
    'B cell zone (BCL2+)': '#B22222',
    'B cell zone (PAX5+)': '#DC143C',
    'FDC network zone': '#E8734A',
    'FDC / myeloid zone': '#FF8C00',
    # Interfollicular (cool)
    'T cell zone': '#4169E1',
    'Stromal / CAF zone': '#20B2AA',
    # Excluded (gray)
    'B/T mixed zone': '#A9A9A9',
    'Mixed (B cells (PAX 27%)': '#BEBEBE',
    'Mixed (M2 Macrophag 26%)': '#C0C0C0',
    'Other / myeloid zone': '#D3D3D3',
    'Unidentified zone': '#DCDCDC',
}

T_FOLL = ['GC core', 'Follicle core (GC/CD20hi/CXCR5hi)',
          'Follicle mantle (CXCR5hi)', 'Activated B / CXCR5hi zone',
          'B cell follicle (CD20hi/CXCR5hi)', 'B cell zone']
T_INTER = ['T cell zone (CD4/CD8)', 'Treg-enriched T zone',
           'Macrophage-rich zone', 'Follicle-T zone interface',
           'Cytotoxic niche']
T_EXCL = ['B / Unassigned transitional', 'Unidentified zone', 'Weak CD20 border']

S_FOLL = ['B cell zone (BCL2+)', 'B cell zone (PAX5+)',
          'FDC network zone', 'FDC / myeloid zone']
S_INTER = ['T cell zone', 'Stromal / CAF zone']
S_EXCL = ['B/T mixed zone', 'Mixed (B cells (PAX 27%)',
          'Mixed (M2 Macrophag 26%)', 'Other / myeloid zone',
          'Unidentified zone']

T_MARKERS = ['CD20', 'CD3', 'CD4', 'CD8a', 'CXCR5', 'FoxP3', 'GranzymeB',
             'TOX', 'CD68', 'PD_1', 'CD57', 'CD45RO', 'CD86', 'CD39', 'ICOS']
S_MARKERS = ['CD20', 'PAX5', 'BCL_2', 'CD21', 'CD4', 'CD8a', 'PDPN',
             'Vimentin', 'CD68', 'CD163', 'CD206', 'S100A9', 'CD31',
             'CXCL13', 'CD11c']

# ---------------------------------------------------------------------------
# ROI selection
# ---------------------------------------------------------------------------

def select_representative_rois(sample_ids, tma_arr, compartments, tumor_mask,
                               foll_list, inter_list, n_rois=2):
    """Select 2 ROIs from different TMAs with best overall compartment diversity
    and good balance between follicular and interfollicular zones."""
    unique_rois = sorted(set(sample_ids[tumor_mask]))
    roi_info = []
    for roi in unique_rois:
        m = (sample_ids == roi) & tumor_mask
        n = np.sum(m)
        if n < 5000:
            continue
        comp = compartments[m]
        tma = tma_arr[m][0]
        unique_comp = set(comp)
        n_foll_types = len([c for c in unique_comp if c in foll_list])
        n_inter_types = len([c for c in unique_comp if c in inter_list])
        n_foll_cells = np.sum(np.isin(comp, foll_list))
        n_inter_cells = np.sum(np.isin(comp, inter_list))
        # Balance score: want both foll and inter well represented
        total_typed = n_foll_cells + n_inter_cells
        if total_typed < 1000:
            continue
        balance = min(n_foll_cells, n_inter_cells) / max(n_foll_cells, n_inter_cells)
        diversity = n_foll_types + n_inter_types
        roi_info.append({
            'roi': roi, 'tma': tma, 'n': n,
            'n_foll_types': n_foll_types, 'n_inter_types': n_inter_types,
            'diversity': diversity, 'balance': balance,
            'score': diversity * 0.4 + balance * 0.6,
        })

    # Sort by combined score (diversity + balance)
    roi_info.sort(key=lambda x: x['score'], reverse=True)
    roi1 = roi_info[0]

    # ROI 2: best score from a different TMA
    candidates = [r for r in roi_info if r['tma'] != roi1['tma']]
    roi2 = candidates[0] if candidates else roi_info[1]

    print(f"  ROI 1: {roi1['roi']} ({roi1['tma']}) — {roi1['diversity']} compartment types, balance={roi1['balance']:.2f}")
    print(f"  ROI 2: {roi2['roi']} ({roi2['tma']}) — {roi2['diversity']} compartment types, balance={roi2['balance']:.2f}")
    return roi1['roi'], roi2['roi']


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_spatial_map(ax, cx, cy, compartments, color_map, roi_name, tma,
                     s=3, show_legend=False, legend_fontsize=None,
                     legend_comps=None):
    """Scatter cells colored by compartment."""
    unique_comp = sorted(set(compartments))
    for comp in unique_comp:
        m = compartments == comp
        c = color_map.get(comp, '#808080')
        ax.scatter(cx[m], cy[m], c=c, s=s, alpha=0.8,
                   edgecolors='none', rasterized=True)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(f'{roi_name} ({tma})', fontsize=TITLE_SIZE)
    if show_legend:
        from matplotlib.patches import Patch
        show_comps = legend_comps if legend_comps is not None else unique_comp
        handles = [Patch(facecolor=color_map.get(c, '#808080'), label=c)
                   for c in show_comps if c in color_map]
        # If some cells belong to compartments not shown in the legend, append
        # a gray "Other / unidentified" swatch so readers know what gray means.
        # The actual rendered cells use several light gray shades (#A9A9A9–#DCDCDC)
        # from the excluded compartments; #C0C0C0 is picked as a representative
        # middle shade so the swatch visually matches the scatter.
        has_other = any(comp not in show_comps for comp in unique_comp)
        if has_other:
            handles.append(Patch(facecolor='#C0C0C0',
                                 label='Other / unidentified'))
        fs = legend_fontsize or LEGEND_SIZE
        ax.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=fs,
                  frameon=False, handletextpad=0.3, columnspacing=0.8)


def plot_composition_heatmap(ax, compartments, cell_types, tumor_mask,
                             comp_order, color_sidebar, panel_label,
                             drop_types=None):
    """Heatmap: compartment (rows) × cell type (columns)."""
    if drop_types is None:
        drop_types = {'Unassigned'}

    # Compute composition
    comp_ct = {}
    for comp in comp_order:
        m = (compartments == comp) & tumor_mask
        ct = cell_types[m]
        ct_clean = ct[~np.isin(ct, list(drop_types))]
        counts = Counter(ct_clean)
        total = sum(counts.values())
        comp_ct[comp] = {k: v / total if total > 0 else 0 for k, v in counts.items()}

    # Get all cell types sorted by overall abundance
    all_ct = Counter()
    for comp in comp_order:
        m = (compartments == comp) & tumor_mask
        ct = cell_types[m]
        all_ct.update(ct[~np.isin(ct, list(drop_types))])
    ct_order = [k for k, _ in all_ct.most_common()]

    # Build matrix
    mat = np.zeros((len(comp_order), len(ct_order)))
    for i, comp in enumerate(comp_order):
        for j, ct in enumerate(ct_order):
            mat[i, j] = comp_ct[comp].get(ct, 0)

    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.7)

    # Row labels colored by group
    short_names = [c[:35] for c in comp_order]
    ax.set_yticks(range(len(comp_order)))
    ax.set_yticklabels(short_names, fontsize=TICK_SIZE)
    for i, (label, color) in enumerate(zip(ax.get_yticklabels(), color_sidebar)):
        label.set_color(color)
        label.set_fontweight('bold')

    ax.set_xticks(range(len(ct_order)))
    ax.set_xticklabels(ct_order, fontsize=TICK_SIZE, rotation=45, ha='right')

    # Group separators
    n_foll = sum(1 for c in color_sidebar if c == '#e74c3c')
    n_inter = sum(1 for c in color_sidebar if c == '#3498db')
    if n_foll > 0:
        ax.axhline(y=n_foll - 0.5, color='black', linewidth=1)
    if n_foll + n_inter < len(comp_order):
        ax.axhline(y=n_foll + n_inter - 0.5, color='black', linewidth=1)

    ax.set_title(f'{panel_label}: Cell Type Composition', fontsize=TITLE_SIZE, fontweight='medium')
    return im


def plot_marker_heatmap(ax, f_v8, compartments, tumor_mask, comp_order,
                        marker_subset, color_sidebar, panel_label):
    """Heatmap: compartment (rows) × markers (columns), z-scored median expression."""
    marker_names = get_marker_names(f_v8)
    marker_idx = {n: i for i, n in enumerate(marker_names)}

    # Load all needed marker columns at once
    print(f"  Loading {len(marker_subset)} markers for {panel_label}...")
    X = f_v8['X']
    n_cells = X.shape[0]

    mat = np.zeros((len(comp_order), len(marker_subset)))
    for j, mk in enumerate(marker_subset):
        if mk not in marker_idx:
            print(f"    WARNING: marker {mk} not found")
            continue
        idx = marker_idx[mk]
        # Read full column — h5py dense dataset
        col = X[:, idx]
        if hasattr(col, 'toarray'):
            col = col.toarray().flatten()
        else:
            col = np.asarray(col).flatten()
        for i, comp in enumerate(comp_order):
            m = (compartments == comp) & tumor_mask
            vals = col[m]
            if len(vals) > 0:
                mat[i, j] = float(np.median(vals))

    # Z-score across compartments (column-wise)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        mu, sd = col.mean(), col.std()
        if sd > 0:
            mat[:, j] = (col - mu) / sd

    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)

    short_names = [c[:35] for c in comp_order]
    ax.set_yticks(range(len(comp_order)))
    ax.set_yticklabels(short_names, fontsize=TICK_SIZE)
    for i, (label, color) in enumerate(zip(ax.get_yticklabels(), color_sidebar)):
        label.set_color(color)
        label.set_fontweight('bold')

    display_markers = [m.replace('_', '-') for m in marker_subset]
    ax.set_xticks(range(len(marker_subset)))
    ax.set_xticklabels(display_markers, fontsize=TICK_SIZE, rotation=45, ha='right')

    # Group separators
    n_foll = sum(1 for c in color_sidebar if c == '#e74c3c')
    n_inter = sum(1 for c in color_sidebar if c == '#3498db')
    if n_foll > 0:
        ax.axhline(y=n_foll - 0.5, color='black', linewidth=1)
    if n_foll + n_inter < len(comp_order):
        ax.axhline(y=n_foll + n_inter - 0.5, color='black', linewidth=1)

    ax.set_title(f'{panel_label}: Marker Expression (z-score)', fontsize=TITLE_SIZE, fontweight='bold')
    return im


def plot_cross_tma(ax, sample_ids, tma_arr, compartments, tumor_mask,
                   comp_order, color_map, panel_label):
    """Stacked bar: compartment fractions per TMA."""
    tmas = sorted(set(tma_arr[tumor_mask]))
    tma_fracs = {}
    for tma in tmas:
        m = (tma_arr == tma) & tumor_mask
        comp = compartments[m]
        total = len(comp)
        fracs = []
        for c in comp_order:
            fracs.append(np.sum(comp == c) / total)
        tma_fracs[tma] = fracs

    x = np.arange(len(tmas))
    bottoms = np.zeros(len(tmas))
    for i, comp in enumerate(comp_order):
        vals = [tma_fracs[t][i] for t in tmas]
        color = color_map.get(comp, '#808080')
        ax.bar(x, vals, bottom=bottoms, color=color, width=0.7, edgecolor='white', linewidth=0.3)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(tmas, fontsize=TICK_SIZE)
    ax.set_ylabel('Fraction of cells', fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.set_title(f'{panel_label}: Cross-TMA', fontsize=TITLE_SIZE, fontweight='bold')


def plot_compartment_frequency(ax, sample_ids, compartments, tumor_mask,
                               comp_order, color_map, panel_label, min_cells=50):
    """Bar chart: % of tumor ROIs containing each compartment."""
    unique_rois = sorted(set(sample_ids[tumor_mask]))
    n_rois = len(unique_rois)

    freqs = []
    for comp in comp_order:
        count = 0
        for roi in unique_rois:
            m = (sample_ids == roi) & (compartments == comp) & tumor_mask
            if np.sum(m) >= min_cells:
                count += 1
        freqs.append(100 * count / n_rois)

    colors = [color_map.get(c, '#808080') for c in comp_order]
    y = np.arange(len(comp_order))
    ax.barh(y, freqs, color=colors, edgecolor='white', linewidth=0.3)
    short_names = [c[:30] for c in comp_order]
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=TICK_SIZE)
    ax.set_xlabel('% of tumor ROIs', fontsize=LABEL_SIZE)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.set_title(f'{panel_label}: Prevalence', fontsize=TITLE_SIZE, fontweight='bold')


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def generate_panel_figure(panel_label, f_v8, f_utag, output_dir,
                          foll_list, inter_list, excl_list,
                          color_map, marker_subset,
                          shared_rois=None):
    """Generate one compartment figure for a single panel (T or S)."""

    sid = load_array(f_utag, 'sample_id')
    tma = load_array(f_utag, 'tma')
    comp = rename_labels(load_array(f_utag, 'compartment_name'))
    ct = rename_labels(load_array(f_v8, 'cell_type'))
    cx = f_utag['obs']['centroid_x'][:]
    cy = f_utag['obs']['centroid_y'][:]
    tumor = get_tumor_mask(sid)

    n_tumor = np.sum(tumor)
    n_rois = len(set(sid[tumor]))
    print(f"\n{panel_label}-panel: {n_tumor:,} tumor cells, {n_rois} ROIs")

    # ROI selection (restrict to shared if provided)
    restrict = tumor
    if shared_rois is not None:
        restrict = tumor & np.isin(sid, shared_rois)
    print(f"  ROI selection:")
    roi1, roi2 = select_representative_rois(sid, tma, comp, restrict,
                                             foll_list, inter_list)

    # Compartment ordering
    comp_set = set(comp)
    comp_order = [c for c in foll_list if c in comp_set] + \
                 [c for c in inter_list if c in comp_set] + \
                 [c for c in excl_list if c in comp_set]
    sidebar = ['#e74c3c'] * len([c for c in foll_list if c in comp_set]) + \
              ['#3498db'] * len([c for c in inter_list if c in comp_set]) + \
              ['#95a5a6'] * len([c for c in excl_list if c in comp_set])

    legend_items = [Patch(facecolor=color_map.get(c, '#808080'),
                          label=c) for c in comp_order]

    # --- Figure: 3 rows (spatial | composition | markers), full width ---
    fig = plt.figure(figsize=(16, 24))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.05,
                           height_ratios=[1, 1.5, 1.5],
                           left=0.22, right=0.92, top=0.95, bottom=0.06)

    # Row 1: 2 spatial maps
    for col, roi in enumerate([roi1, roi2]):
        ax = fig.add_subplot(gs[0, col])
        m = (sid == roi) & tumor
        tma_val = tma[m][0] if np.any(m) else '?'
        plot_spatial_map(ax, cx[m], cy[m], comp[m], color_map, roi, tma_val)
        label = chr(ord('a') + col)
        ax.text(-0.02, 1.02, rf'$\bf{{{label}}}$', transform=ax.transAxes,
                fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # Legend below spatial maps — position dynamically
    # Get bottom of row 0 and top of row 1
    row0_bottom = gs[0, 0].get_position(fig).y0
    row1_top = gs[1, 0].get_position(fig).y1
    leg_y = (row0_bottom + row1_top) / 2 - 0.01
    ax_leg = fig.add_axes([0.05, leg_y, 0.90, 0.04])
    ax_leg.axis('off')
    ax_leg.legend(handles=legend_items, loc='center', ncol=3, fontsize=LEGEND_SIZE,
                   frameon=False, handletextpad=0.4, columnspacing=1.0)

    # Row 2: Composition heatmap (full width)
    ax_comp = fig.add_subplot(gs[1, :])
    im_comp = plot_composition_heatmap(ax_comp, comp, ct, tumor,
                                        comp_order, sidebar, panel_label)
    ax_comp.text(-0.02, 1.02, r'$\bf{c}$', transform=ax_comp.transAxes,
                  fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # Row 3: Marker expression heatmap (full width)
    ax_mk = fig.add_subplot(gs[2, :])
    im_mk = plot_marker_heatmap(ax_mk, f_v8, comp, tumor,
                                 comp_order, marker_subset, sidebar, panel_label)
    ax_mk.text(-0.02, 1.02, r'$\bf{d}$', transform=ax_mk.transAxes,
                fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # Colorbar for marker heatmap
    cbar_ax = fig.add_axes([0.94, 0.08, 0.015, 0.12])
    cb = plt.colorbar(im_mk, cax=cbar_ax)
    cb.set_label('z-score', fontsize=LABEL_SIZE)

    fig.suptitle(f'{panel_label}-panel: Tissue Compartments',
                 fontsize=TITLE_SIZE, fontweight='bold', y=0.99)

    fig_path = os.path.join(output_dir, f'fig_compartments_{panel_label}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Main figure saved: {fig_path}")

    return sid, tma, comp, tumor, comp_order, sidebar, color_map


def _find_best_roi_for_compartment(sid, tma, comp, tumor, target_comp,
                                    excl_list, min_target=200, min_total=2000):
    """Find the best ROI to showcase a specific compartment.

    Prefers ROIs where the target is clearly visible (15-50% of cells)
    with good diversity of other compartments for context.
    """
    candidates = []
    for roi in sorted(set(sid[tumor])):
        m = (sid == roi) & tumor
        n_total = int(np.sum(m))
        if n_total < min_total:
            continue
        c_roi = comp[m]
        n_target = int(np.sum(c_roi == target_comp))
        if n_target < min_target:
            continue
        frac = n_target / n_total
        # Ideal fraction: 15-50%. Penalize >60% (whole-core dominated).
        if frac > 0.70:
            frac_score = 0.1
        elif frac > 0.50:
            frac_score = 0.5
        else:
            frac_score = 1.0
        # Diversity: how many non-excluded compartments present?
        non_excl = [c for c in set(c_roi) if c not in excl_list]
        n_diverse = sum(1 for c in non_excl if np.sum(c_roi == c) >= 50)
        tma_bonus = 1.0 if 'Biomax' not in tma[m][0] else 0.3
        score = n_target * frac_score * (n_diverse / 5.0) * tma_bonus
        candidates.append((score, n_target, roi, tma[m][0]))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][2], candidates[0][3]


def _save_panel_cache(fig_single, cache_path, dpi=200):
    """Save a single-panel figure to cache."""
    fig_single.savefig(cache_path, dpi=dpi, bbox_inches='tight',
                        pad_inches=0.05)
    plt.close(fig_single)


def _generate_example_pair(sid, tma, comp, ct, cx, cy, tumor,
                            target, cache_dir,
                            excl_list, comp_colors, ct_colors):
    """Generate and cache a compartment example pair (compartment + cell types).

    Returns (comp_path, ct_path, best_roi, best_tma, n_target) or Nones.
    """
    comp_cache = os.path.join(cache_dir, f'ex_{target[:20].replace("/","_")}_comp.png')
    ct_cache = os.path.join(cache_dir, f'ex_{target[:20].replace("/","_")}_ct.png')

    # Always compute best_roi so the label can be shown even when cache exists
    best_roi, best_tma = _find_best_roi_for_compartment(
        sid, tma, comp, tumor, target, excl_list)

    if os.path.exists(comp_cache) and os.path.exists(ct_cache):
        # Compute n_target for the cached ROI
        if best_roi is not None:
            m_roi = (sid == best_roi) & tumor
            n_target = int(((comp[m_roi]) == target).sum())
        else:
            n_target = None
        return comp_cache, ct_cache, best_roi, best_tma, n_target

    if best_roi is None:
        return None, None, None, None, 0

    m_roi = (sid == best_roi) & tumor
    x, y = cx[m_roi], cy[m_roi]
    c_roi = comp[m_roi]
    ct_roi = ct[m_roi]
    fg = c_roi == target
    bg = ~fg
    n_target = int(np.sum(fg))

    # --- Compartment panel ---
    fig_c, ax_c = plt.subplots(figsize=(8, 8))
    ax_c.scatter(x[bg], y[bg], c='#E0E0E0', s=2, alpha=0.3,
                 edgecolors='none', rasterized=True)
    color = comp_colors.get(target, '#808080')
    ax_c.scatter(x[fg], y[fg], c=color, s=4, alpha=0.9,
                 edgecolors='none', rasterized=True)
    ax_c.set_aspect('equal')
    ax_c.invert_yaxis()
    ax_c.axis('off')
    _save_panel_cache(fig_c, comp_cache)

    # --- Cell type panel (only cells WITHIN compartment colored) ---
    fig_ct, ax_ct = plt.subplots(figsize=(8, 8))
    ax_ct.scatter(x[bg], y[bg], c='#F0F0F0', s=1, alpha=0.15,
                  edgecolors='none', rasterized=True)
    ct_target = ct_roi[fg]
    x_fg, y_fg = x[fg], y[fg]
    for ctype in sorted(set(ct_target)):
        m_ct = ct_target == ctype
        frac = np.sum(m_ct) / len(ct_target)
        if frac < 0.01:
            ax_ct.scatter(x_fg[m_ct], y_fg[m_ct], c='#C0C0C0', s=3,
                          alpha=0.5, edgecolors='none', rasterized=True)
        else:
            c_ct = ct_colors.get(ctype, '#808080')
            ax_ct.scatter(x_fg[m_ct], y_fg[m_ct], c=c_ct, s=4, alpha=0.9,
                          edgecolors='none', rasterized=True)
    ax_ct.set_aspect('equal')
    ax_ct.invert_yaxis()
    ax_ct.axis('off')
    _save_panel_cache(fig_ct, ct_cache)

    return comp_cache, ct_cache, best_roi, best_tma, n_target


def generate_combined_figure(f_v8_t, f_utag_t, f_v8_s, f_utag_s, output_dir,
                              shared_rois=None):
    """Generate T-panel compartment figure with heatmap, ROIs, and 3x3 examples.

    Layout:
      Top row: T-panel composition heatmap (a, left) + 2 ROIs stacked (b, c, right)
      Bottom: 3x3 grid of compartment examples (d-l), each showing
              compartment highlight (left) + cell types within compartment (right)
      One shared cell-type legend at bottom.

    Uses panel caching: individual panels are saved as PNGs in a cache dir.
    Re-runs skip data-heavy scatter plot generation if cache exists.
    Delete the cache dir to force full regeneration.
    """
    from matplotlib.patches import Patch
    from matplotlib.image import imread

    # --- Cache directory ---
    cache_dir = os.path.join(output_dir, '_cache_compartments')
    os.makedirs(cache_dir, exist_ok=True)

    # --- Load T-panel ---
    sid_t = load_array(f_utag_t, 'sample_id')
    tma_t = load_array(f_utag_t, 'tma')
    comp_t = rename_labels(load_array(f_utag_t, 'compartment_name'))
    ct_t = rename_labels(load_array(f_v8_t, 'cell_type'))
    cx_t = f_utag_t['obs']['centroid_x'][:]
    cy_t = f_utag_t['obs']['centroid_y'][:]
    tumor_t = get_tumor_mask(sid_t)

    # --- ROI selection ---
    restrict_t = tumor_t & np.isin(sid_t, shared_rois) if shared_rois else tumor_t
    print("\nT-panel ROI selection:")
    roi_t1, roi_t2 = select_representative_rois(
        sid_t, tma_t, comp_t, restrict_t, T_FOLL, T_INTER)

    # --- Compartment ordering ---
    def build_order(comp_arr, foll, inter, excl):
        comp_set = set(comp_arr)
        order = [c for c in foll if c in comp_set] + \
                [c for c in inter if c in comp_set] + \
                [c for c in excl if c in comp_set]
        sidebar = ['#e74c3c'] * len([c for c in foll if c in comp_set]) + \
                  ['#3498db'] * len([c for c in inter if c in comp_set]) + \
                  ['#95a5a6'] * len([c for c in excl if c in comp_set])
        return order, sidebar

    t_order, t_sidebar = build_order(comp_t, T_FOLL, T_INTER, T_EXCL)

    # --- 9 key compartments ---
    KEY_COMPS = [
        'GC core',
        'Follicle core (GC/CD20hi/CXCR5hi)',
        'Follicle mantle (CXCR5hi)',
        'B cell follicle (CD20hi/CXCR5hi)',
        'B cell zone',
        'T cell zone (CD4/CD8)',
        'Treg-enriched T zone',
        'Macrophage-rich zone',
        'Follicle-T zone interface',
    ]
    COMP_SHORT = {
        'GC core': 'GC core',
        'Follicle core (GC/CD20hi/CXCR5hi)': 'Follicle core',
        'Follicle mantle (CXCR5hi)': 'Follicle mantle',
        'B cell follicle (CD20hi/CXCR5hi)': 'B cell follicle',
        'B cell zone': 'B cell zone',
        'T cell zone (CD4/CD8)': 'T cell zone',
        'Treg-enriched T zone': 'Treg-enriched zone',
        'Macrophage-rich zone': 'Macrophage-rich zone',
        'Follicle-T zone interface': 'Follicle-T zone interface',
    }

    # --- Cache ROI panels ---
    roi_cache_b = os.path.join(cache_dir, f'roi_{roi_t1}.png')
    roi_cache_c = os.path.join(cache_dir, f'roi_{roi_t2}.png')
    for roi_id, cache_path in [(roi_t1, roi_cache_b), (roi_t2, roi_cache_c)]:
        if not os.path.exists(cache_path):
            m = (sid_t == roi_id) & tumor_t
            fig_r, ax_r = plt.subplots(figsize=(8, 8))
            plot_spatial_map(ax_r, cx_t[m], cy_t[m], comp_t[m],
                             T_COMPARTMENT_COLORS, roi_id,
                             tma_t[m][0] if np.any(m) else '?')
            ax_r.set_title('')  # title added during assembly
            _save_panel_cache(fig_r, cache_path)
            print(f"  Cached ROI panel: {cache_path}")

    # --- Cache example pairs ---
    example_data = []
    for target in KEY_COMPS:
        comp_path, ct_path, roi, tma_label, n = _generate_example_pair(
            sid_t, tma_t, comp_t, ct_t, cx_t, cy_t, tumor_t,
            target, cache_dir,
            excl_list=T_EXCL, comp_colors=T_COMPARTMENT_COLORS,
            ct_colors=T_CELLTYPE_COLORS)
        example_data.append((target, comp_path, ct_path, roi, tma_label, n))
        if roi is not None:
            short = COMP_SHORT.get(target, target)
            print(f"  Cached example: {short} → {roi} ({tma_label}), n={n:,}")

    # ===== Assembly from cached panels =====
    print("  Assembling combined figure from cached panels...")

    fig = plt.figure(figsize=(20, 26))

    # Top: heatmap (left, 2 rows) + 2 ROIs stacked (right)
    gs_top = gridspec.GridSpec(2, 2, figure=fig,
                                width_ratios=[1.6, 1],
                                hspace=0.10, wspace=0.12,
                                left=0.14, right=0.96, top=0.97, bottom=0.58)

    # Bottom: 3×3 grid, each entry = 2 subplots
    gs_bot = gridspec.GridSpec(3, 6, figure=fig,
                                hspace=0.18, wspace=0.04,
                                left=-0.02, right=0.94, top=0.46, bottom=0.06)

    # ---- Panel a: heatmap (live, not cached — fast to render) ----
    ax_a = fig.add_subplot(gs_top[:, 0])
    plot_composition_heatmap(ax_a, comp_t, ct_t, tumor_t,
                             t_order, t_sidebar, 'T')
    ax_a.set_title('T-panel: Compartment × Cell Type', fontsize=TITLE_SIZE,
                    fontweight='medium')
    ax_a.set_yticklabels([c[:35] for c in t_order], fontsize=TICK_SIZE)
    for tick in ax_a.get_xticklabels():
        tick.set_fontsize(TICK_SIZE)
    ax_a.text(-0.02, 1.02, r'$\bf{a}$', transform=ax_a.transAxes,
              fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # ---- Panels b, c: ROIs from cache ----
    for panel_idx, (roi_id, cache_path, gs_pos) in enumerate([
        (roi_t1, roi_cache_b, gs_top[0, 1]),
        (roi_t2, roi_cache_c, gs_top[1, 1]),
    ]):
        ax = fig.add_subplot(gs_pos)
        img = imread(cache_path)
        ax.imshow(img)
        ax.axis('off')
        m = (sid_t == roi_id) & tumor_t
        tma_label = tma_t[m][0] if np.any(m) else '?'
        ax.set_title(f'{roi_id} ({tma_label})', fontsize=TITLE_SIZE, fontweight='medium')
        label = chr(ord('b') + panel_idx)
        ax.text(-0.02, 1.02, rf'$\bf{{{label}}}$', transform=ax.transAxes,
                fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # (Compartment legend removed — colors are shown in ROI panels b/c and heatmap sidebar)

    # ---- Panels d-l: 3×3 examples from cache ----
    for idx, (target, comp_path, ct_path, roi, tma_label, n) in enumerate(example_data):
        row = idx // 3
        col_pair = (idx % 3) * 2

        ax_comp = fig.add_subplot(gs_bot[row, col_pair])
        ax_ct = fig.add_subplot(gs_bot[row, col_pair + 1])

        if comp_path is None:
            ax_comp.set_visible(False)
            ax_ct.set_visible(False)
            continue

        # Load cached images
        ax_comp.imshow(imread(comp_path))
        ax_comp.axis('off')
        short_name = COMP_SHORT.get(target, target)
        n_label = f'(n={n:,})' if n is not None else ''
        ax_comp.set_title(f'{short_name}\n{n_label}',
                          fontsize=TITLE_SIZE, fontweight='bold', pad=18)
        label = chr(ord('d') + idx)
        ax_comp.text(-0.02, 1.02, rf'$\bf{{{label}}}$', transform=ax_comp.transAxes,
                     fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

        ax_ct.imshow(imread(ct_path))
        ax_ct.axis('off')
        roi_label = roi if roi else ''
        ax_ct.set_title(f'Cell types\n{roi_label}', fontsize=TITLE_SIZE, pad=12)

    # ---- Shared cell-type legend at bottom ----
    ct_items = [Patch(facecolor=c, label=name)
                for name, c in T_CELLTYPE_COLORS.items()
                if name != 'Unassigned']
    ax_ct_leg = fig.add_axes([0.02, 0.01, 0.96, 0.04])
    ax_ct_leg.axis('off')
    ax_ct_leg.legend(handles=ct_items, loc='center', ncol=6, fontsize=LEGEND_SIZE,
                      frameon=False, handletextpad=0.4, columnspacing=1.0,
                      title='Cell types',
                      title_fontproperties={'weight': 'bold', 'size': LABEL_SIZE})

    # (no suptitle — PDF assembly adds figure title)

    fig_path = os.path.join(output_dir, 'fig_compartments_combined.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Combined figure saved: {fig_path}")

    # Return T+S data for downstream (supplementary uses ret_t/ret_s)
    sid_s = load_array(f_utag_s, 'sample_id')
    tma_s = load_array(f_utag_s, 'tma')
    comp_s = rename_labels(load_array(f_utag_s, 'compartment_name'))
    tumor_s = get_tumor_mask(sid_s)
    s_order, s_sidebar = build_order(comp_s, S_FOLL, S_INTER, S_EXCL)

    return (sid_t, tma_t, comp_t, tumor_t, t_order, t_sidebar, T_COMPARTMENT_COLORS,
            sid_s, tma_s, comp_s, tumor_s, s_order, s_sidebar, S_COMPARTMENT_COLORS)


def generate_combined_figure_s(f_v8_s, f_utag_s, output_dir, shared_rois=None):
    """Generate S-panel compartment figure — same layout as T-panel (Figure 2).

    Layout:
      Top row: S-panel composition heatmap (a, left) + 2 ROIs stacked (b, c, right)
      Bottom: 2x3 grid of compartment examples (d-i), each showing
              compartment highlight (left) + cell types within compartment (right)
      One shared cell-type legend at bottom.

    Uses panel caching in _cache_compartments_S/.
    """
    from matplotlib.patches import Patch
    from matplotlib.image import imread

    cache_dir = os.path.join(output_dir, '_cache_compartments_S')
    os.makedirs(cache_dir, exist_ok=True)

    # --- Load S-panel ---
    sid = load_array(f_utag_s, 'sample_id')
    tma = load_array(f_utag_s, 'tma')
    comp = rename_labels(load_array(f_utag_s, 'compartment_name'))
    ct = rename_labels(load_array(f_v8_s, 'cell_type'))
    cx = f_utag_s['obs']['centroid_x'][:]
    cy = f_utag_s['obs']['centroid_y'][:]
    tumor = get_tumor_mask(sid)

    # --- ROI selection ---
    restrict = tumor & np.isin(sid, shared_rois) if shared_rois else tumor
    print("\nS-panel ROI selection:")
    roi1, roi2 = select_representative_rois(
        sid, tma, comp, restrict, S_FOLL, S_INTER)

    # --- Compartment ordering (exclude unidentified/gray compartments) ---
    comp_set = set(comp)
    s_order = [c for c in S_FOLL if c in comp_set] + \
              [c for c in S_INTER if c in comp_set]
    s_sidebar = ['#e74c3c'] * len([c for c in S_FOLL if c in comp_set]) + \
                ['#3498db'] * len([c for c in S_INTER if c in comp_set])

    # --- 6 key compartments (all non-excluded) ---
    KEY_COMPS_S = [
        'B cell zone (BCL2+)',
        'B cell zone (PAX5+)',
        'FDC network zone',
        'FDC / myeloid zone',
        'T cell zone',
        'Stromal / CAF zone',
    ]
    COMP_SHORT_S = {
        'B cell zone (BCL2+)': 'B cell zone (BCL2+)',
        'B cell zone (PAX5+)': 'B cell zone (PAX5+)',
        'FDC network zone': 'FDC network zone',
        'FDC / myeloid zone': 'FDC / myeloid zone',
        'T cell zone': 'T cell zone',
        'Stromal / CAF zone': 'Stromal / CAF zone',
    }

    # Filter to compartments present in data
    KEY_COMPS_S = [c for c in KEY_COMPS_S if c in comp_set]
    n_key = len(KEY_COMPS_S)
    n_rows_bot = (n_key + 1) // 2  # 2 compartments per row → ceil(n/2) rows

    # --- Cache ROI panel (single ROI: roi2 = panel b) ---
    roi_cache_b = os.path.join(cache_dir, f'roi_{roi2}.png')
    if not os.path.exists(roi_cache_b):
        m = (sid == roi2) & tumor
        # Compact size so the panel fits the heatmap's natural top-row height
        # without stretching panel a vertically.
        fig_r, ax_r = plt.subplots(figsize=(10, 7))
        plot_spatial_map(ax_r, cx[m], cy[m], comp[m],
                         S_COMPARTMENT_COLORS, roi2,
                         tma[m][0] if np.any(m) else '?',
                         s=12, show_legend=True,
                         legend_fontsize=18,
                         legend_comps=S_FOLL + S_INTER)
        ax_r.set_title('')
        _save_panel_cache(fig_r, roi_cache_b)
        print(f"  Cached ROI panel: {roi_cache_b}")

    # --- Cache example pairs ---
    example_data = []
    for target in KEY_COMPS_S:
        comp_path, ct_path, roi, tma_label, n = _generate_example_pair(
            sid, tma, comp, ct, cx, cy, tumor,
            target, cache_dir,
            excl_list=S_EXCL, comp_colors=S_COMPARTMENT_COLORS,
            ct_colors=S_CELLTYPE_COLORS)
        example_data.append((target, comp_path, ct_path, roi, tma_label, n))
        if roi is not None:
            short = COMP_SHORT_S.get(target, target)
            print(f"  Cached example: {short} → {roi} ({tma_label}), n={n:,}")

    # ---- Cache panel a (heatmap) using Fig 2's figsize convention ----
    # Fig 2 caches at (18, 10) for 9 compartment rows — aspect ~1.8:1.
    # S5 uses proportional height for same per-row size, with width tuned so
    # the heatmap is slightly narrower (similar aspect to Fig 2).
    n_comp_rows = len(s_order)
    heatmap_cache = os.path.join(cache_dir, f'heatmap_a_{n_comp_rows}rows.png')
    if not os.path.exists(heatmap_cache):
        fig_h, ax_h = plt.subplots(figsize=(14, 10 * n_comp_rows / 9))
        plot_composition_heatmap(ax_h, comp, ct, tumor,
                                  s_order, s_sidebar, 'S')
        ax_h.set_title('S-panel: Compartment × Cell Type',
                       fontsize=TITLE_SIZE, fontweight='medium')
        ax_h.set_yticklabels([c[:35] for c in s_order], fontsize=TICK_SIZE)
        for tick in ax_h.get_xticklabels():
            tick.set_fontsize(TICK_SIZE)
        _save_panel_cache(fig_h, heatmap_cache)

    # ===== Assembly from cached panels =====
    print("  Assembling S-panel combined figure from cached panels...")

    fig = plt.figure(figsize=(20, 24))

    # Top: heatmap (left) + 1 ROI (right) — use pixel-based width_ratios like
    # Fig 2. Row height = frac1_fig2 * (n_comp_s5 / 9) so cells match Fig 2.
    # Fig 2: frac1 ≈ 0.317 for 9 rows → scale proportionally.
    img_a = imread(heatmap_cache)
    img_b_top = imread(roi_cache_b)
    ha, wa = img_a.shape[:2]
    hb_raw, wb_raw = img_b_top.shape[:2]
    # Pad panel b with white to match panel a height (like Fig 2 does)
    if hb_raw < ha:
        pad_total = ha - hb_raw
        pad_top = pad_total // 2
        pad_bot = pad_total - pad_top
        n_ch = img_b_top.shape[2] if img_b_top.ndim == 3 else 1
        white = np.ones((pad_top, wb_raw, n_ch), dtype=img_b_top.dtype)
        white_bot = np.ones((pad_bot, wb_raw, n_ch), dtype=img_b_top.dtype)
        img_b_top = np.concatenate([white, img_b_top, white_bot], axis=0)
    hb, wb = img_b_top.shape[:2]

    # Match Fig 2: top row height = 0.92 * h1 / total_h where h1=10. Use the
    # same per-row scale: n_rows * (0.317 / 9).
    top_frac = 0.317 * n_comp_rows / 9 + 0.03  # small header margin
    top_bottom = 0.97 - top_frac
    gs_top = gridspec.GridSpec(1, 2, figure=fig,
                                width_ratios=[wa, wb],
                                wspace=0.02,
                                left=0.02, right=0.98, top=0.97, bottom=top_bottom)

    # Bottom: n_rows_bot × 4 (2 compartments per row, each pair = 2 cols)
    gs_bot = gridspec.GridSpec(n_rows_bot, 4, figure=fig,
                                hspace=0.18, wspace=0.04,
                                left=-0.02, right=0.94, top=top_bottom - 0.02, bottom=0.06)

    # ---- Panel a: heatmap from cache ----
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.imshow(img_a)
    ax_a.axis('off')
    ax_a.text(-0.02, 1.02, r'$\bf{a}$', transform=ax_a.transAxes,
              fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # ---- Panel b: single ROI from cache ----
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_b.imshow(img_b_top)
    ax_b.axis('off')
    m = (sid == roi2) & tumor
    tma_label = tma[m][0] if np.any(m) else '?'
    ax_b.set_title(f'{roi2} ({tma_label})', fontsize=TITLE_SIZE, fontweight='medium')
    ax_b.text(-0.02, 1.02, r'$\bf{b}$', transform=ax_b.transAxes,
              fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # ---- Panels c-h: examples from cache (3 rows × 2 compartments) ----
    for idx, (target, comp_path, ct_path, roi, tma_label, n) in enumerate(example_data):
        row = idx // 2
        col_pair = (idx % 2) * 2

        ax_comp = fig.add_subplot(gs_bot[row, col_pair])
        ax_ct = fig.add_subplot(gs_bot[row, col_pair + 1])

        if comp_path is None:
            ax_comp.set_visible(False)
            ax_ct.set_visible(False)
            continue

        ax_comp.imshow(imread(comp_path))
        ax_comp.axis('off')
        short_name = COMP_SHORT_S.get(target, target)
        ax_comp.set_title(short_name,
                          fontsize=TITLE_SIZE, fontweight='bold', pad=18)
        label = chr(ord('c') + idx)
        ax_comp.text(-0.02, 1.02, rf'$\bf{{{label}}}$', transform=ax_comp.transAxes,
                     fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

        ax_ct.imshow(imread(ct_path))
        ax_ct.axis('off')
        roi_label = roi if roi else ''
        ax_ct.set_title(f'Cell types\n{roi_label}', fontsize=TITLE_SIZE, pad=12)

    # ---- Shared cell-type legend at bottom ----
    ct_items = [Patch(facecolor=c, label=name)
                for name, c in S_CELLTYPE_COLORS.items()
                if name != 'Unassigned']
    ax_ct_leg = fig.add_axes([0.02, 0.01, 0.96, 0.04])
    ax_ct_leg.axis('off')
    ax_ct_leg.legend(handles=ct_items, loc='center', ncol=6, fontsize=LEGEND_SIZE,
                      frameon=False, handletextpad=0.4, columnspacing=1.0,
                      title='Cell types',
                      title_fontproperties={'weight': 'bold', 'size': LABEL_SIZE})

    fig_path = os.path.join(output_dir, 'fig_compartments_combined_S.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  S-panel combined figure saved: {fig_path}")


def generate_supplementary(sid_t, tma_t, comp_t, tumor_t, t_order, t_sidebar, t_cmap,
                           sid_s, tma_s, comp_s, tumor_s, s_order, s_sidebar, s_cmap,
                           output_dir):
    """Supplementary: cross-TMA reproducibility + compartment prevalence."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    plot_cross_tma(axes[0, 0], sid_t, tma_t, comp_t, tumor_t,
                   t_order, t_cmap, 'T-panel')
    axes[0, 0].text(-0.02, 1.02, r'$\bf{a}$', transform=axes[0, 0].transAxes,
                     fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    plot_compartment_frequency(axes[0, 1], sid_t, comp_t, tumor_t,
                               t_order, t_cmap, 'T-panel')
    axes[0, 1].text(-0.02, 1.02, r'$\bf{b}$', transform=axes[0, 1].transAxes,
                     fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    plot_cross_tma(axes[1, 0], sid_s, tma_s, comp_s, tumor_s,
                   s_order, s_cmap, 'S-panel')
    axes[1, 0].text(-0.02, 1.02, r'$\bf{c}$', transform=axes[1, 0].transAxes,
                     fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    plot_compartment_frequency(axes[1, 1], sid_s, comp_s, tumor_s,
                               s_order, s_cmap, 'S-panel')
    axes[1, 1].text(-0.02, 1.02, r'$\bf{d}$', transform=axes[1, 1].transAxes,
                     fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    fig.suptitle('Supplementary: Compartment Reproducibility Across TMAs',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    supp_path = os.path.join(output_dir, 'fig_compartments_supp.png')
    fig.savefig(supp_path, dpi=150, bbox_inches='tight')
    fig.savefig(supp_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSupplementary saved: {supp_path}")


def print_compartment_stats(panel_label, f_v8, f_utag,
                            foll_list, inter_list, excl_list,
                            comp_order, marker_subset):
    """Print detailed compartment statistics as markdown tables."""
    sid = load_array(f_utag, 'sample_id')
    tma = load_array(f_utag, 'tma')
    comp = rename_labels(load_array(f_utag, 'compartment_name'))
    ct = rename_labels(load_array(f_v8, 'cell_type'))
    tumor = get_tumor_mask(sid)

    n_tumor = int(np.sum(tumor))
    drop_types = {'Unassigned'}

    # --- 1. Global cell counts per compartment ---
    print(f"\n#### {panel_label}-panel: Cell counts per compartment\n")
    print(f"| Compartment | Group | Cells | % of tumor |")
    print(f"|-------------|-------|------:|----------:|")
    for c in comp_order:
        m = (comp == c) & tumor
        n = int(np.sum(m))
        pct = 100 * n / n_tumor
        grp = 'Follicular' if c in foll_list else ('Interfollicular' if c in inter_list else 'Excluded')
        print(f"| {c} | {grp} | {n:,} | {pct:.1f}% |")
    print(f"| **Total tumor** | | **{n_tumor:,}** | **100%** |")

    # --- 2. Cell type composition per compartment ---
    print(f"\n#### {panel_label}-panel: Cell type composition (fraction)\n")
    # Gather all cell types
    all_ct_counter = Counter()
    for c in comp_order:
        m = (comp == c) & tumor
        ct_vals = ct[m]
        ct_clean = ct_vals[~np.isin(ct_vals, list(drop_types))]
        all_ct_counter.update(ct_clean)
    ct_order = [k for k, _ in all_ct_counter.most_common(12)]

    header = f"| Compartment | " + " | ".join(ct_order) + " |"
    sep = f"|-------------|" + "|".join(["-----:" for _ in ct_order]) + "|"
    print(header)
    print(sep)
    for c in comp_order:
        m = (comp == c) & tumor
        ct_vals = ct[m]
        ct_clean = ct_vals[~np.isin(ct_vals, list(drop_types))]
        total = len(ct_clean)
        counts = Counter(ct_clean)
        vals = []
        for t in ct_order:
            frac = counts.get(t, 0) / total if total > 0 else 0
            vals.append(f"{frac:.2f}" if frac >= 0.01 else "—")
        print(f"| {c[:35]} | " + " | ".join(vals) + " |")

    # --- 3. Median marker z-scores per compartment ---
    print(f"\n#### {panel_label}-panel: Median marker expression (z-score)\n")
    marker_names = get_marker_names(f_v8)
    marker_idx = {n: i for i, n in enumerate(marker_names)}
    X = f_v8['X']

    # Load marker columns
    mk_data = {}
    for mk in marker_subset:
        if mk not in marker_idx:
            continue
        idx = marker_idx[mk]
        col = X[:, idx]
        if hasattr(col, 'toarray'):
            col = col.toarray().flatten()
        else:
            col = np.asarray(col).flatten()
        mk_data[mk] = col

    # Compute medians, then z-score across compartments
    raw_medians = np.zeros((len(comp_order), len(marker_subset)))
    for i, c in enumerate(comp_order):
        m = (comp == c) & tumor
        for j, mk in enumerate(marker_subset):
            if mk in mk_data:
                vals = mk_data[mk][m]
                if len(vals) > 0:
                    raw_medians[i, j] = float(np.median(vals))

    z_medians = raw_medians.copy()
    for j in range(z_medians.shape[1]):
        col = z_medians[:, j]
        mu, sd = col.mean(), col.std()
        if sd > 0:
            z_medians[:, j] = (col - mu) / sd

    display_markers = [m.replace('_', '-') for m in marker_subset]
    header = f"| Compartment | " + " | ".join(display_markers) + " |"
    sep = f"|-------------|" + "|".join(["-----:" for _ in display_markers]) + "|"
    print(header)
    print(sep)
    for i, c in enumerate(comp_order):
        vals = [f"{z_medians[i, j]:+.1f}" for j in range(len(marker_subset))]
        print(f"| {c[:35]} | " + " | ".join(vals) + " |")

    # --- 4. Per-TMA compartment fractions ---
    print(f"\n#### {panel_label}-panel: Per-TMA compartment fractions\n")
    tmas = sorted(set(tma[tumor]))
    header = f"| Compartment | " + " | ".join(tmas) + " |"
    sep = f"|-------------|" + "|".join(["-----:" for _ in tmas]) + "|"
    print(header)
    print(sep)
    for c in comp_order:
        vals = []
        for t in tmas:
            m_tma = (tma == t) & tumor
            n_tma = int(np.sum(m_tma))
            n_comp = int(np.sum((comp == c) & m_tma))
            frac = n_comp / n_tma if n_tma > 0 else 0
            vals.append(f"{frac:.3f}")
        print(f"| {c[:35]} | " + " | ".join(vals) + " |")

    # --- 5. Prevalence ---
    print(f"\n#### {panel_label}-panel: Compartment prevalence (% of tumor ROIs with ≥50 cells)\n")
    unique_rois = sorted(set(sid[tumor]))
    n_rois = len(unique_rois)
    print(f"| Compartment | ROIs present | % of {n_rois} tumor ROIs |")
    print(f"|-------------|------------:|---------------------:|")
    for c in comp_order:
        count = 0
        for roi in unique_rois:
            m = (sid == roi) & (comp == c) & tumor
            if np.sum(m) >= 50:
                count += 1
        print(f"| {c[:35]} | {count} | {100*count/n_rois:.1f}% |")


T_CELLTYPE_COLORS = {
    'GC B cells': '#B22222',
    'B cells (CD20hi)': '#DC143C',
    'B cells (CXCR5hi)': '#E8734A',
    'Other B cells': '#FF8C00',
    'B cells (weak CD20)': '#F4A460',
    'CD4 T cells': '#4169E1',
    'CD8 T cells': '#1E90FF',
    'CD8 T pre-exhausted (TOX+)': '#6A5ACD',
    'Treg': '#20B2AA',
    'Macrophages': '#2E8B57',
    'Mixed / Border cells': '#DAA520',
    'Unassigned': '#D3D3D3',
    'Other': '#A9A9A9',
}

S_CELLTYPE_COLORS = {
    'B cells (BCL2+)': '#B22222',
    'B cells (PAX5+)': '#DC143C',
    'B cells': '#FF8C00',
    'Other B cells': '#F4A460',
    'FDC': '#E8734A',
    'CD4 T cells': '#4169E1',
    'CD8 T cells': '#1E90FF',
    'M1 Macrophages': '#3CB371',
    'M2 Macrophages': '#2E8B57',
    'Macrophages': '#228B22',
    'Dendritic cells': '#9370DB',
    'Myeloid (S100A9+)': '#DA70D6',
    'Stromal / CAF': '#20B2AA',
    'FRC (PDPN+)': '#5F9EA0',
    'Endothelial': '#FF69B4',
    'Histiocytes (CD44hi)': '#8B4513',
    'pDC': '#DDA0DD',
    'Mixed / Border cells': '#DAA520',
    'Unassigned': '#D3D3D3',
    'Other': '#A9A9A9',
}


def generate_compartment_examples(f_utag, f_v8, output_dir, panel_label,
                                  foll_list, inter_list, excl_list,
                                  comp_colors, ct_colors):
    """For each key compartment, show paired spatial maps:
    left = compartment highlighted (rest gray), right = cell types within it.

    Validates that compartments match expected biology.
    """
    sid = load_array(f_utag, 'sample_id')
    tma = load_array(f_utag, 'tma')
    comp = rename_labels(load_array(f_utag, 'compartment_name'))
    ct = rename_labels(load_array(f_v8, 'cell_type'))
    cx = f_utag['obs']['centroid_x'][:]
    cy = f_utag['obs']['centroid_y'][:]
    tumor = get_tumor_mask(sid)

    # Key compartments (skip rare ones with <5 ROIs having ≥50 cells)
    key_comps = []
    for c in foll_list + inter_list:
        if c not in set(comp):
            continue
        n_rois_present = 0
        for roi in set(sid[tumor]):
            m = (sid == roi) & tumor & (comp == c)
            if np.sum(m) >= 50:
                n_rois_present += 1
        if n_rois_present >= 5:
            key_comps.append(c)

    n_comp = len(key_comps)
    print(f"\n{panel_label}-panel compartment examples: {n_comp} compartments")

    # Layout: n_comp rows × 3 columns (compartment map | cell type map | legend)
    fig = plt.figure(figsize=(14, 4.2 * n_comp))
    gs = gridspec.GridSpec(n_comp, 3, figure=fig, width_ratios=[1, 1, 0.35],
                           hspace=0.25, wspace=0.05,
                           left=0.02, right=0.98, top=0.98, bottom=0.01)
    axes = np.empty((n_comp, 3), dtype=object)
    for r in range(n_comp):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])

    for i, target_comp in enumerate(key_comps):
        # Find best ROI: target compartment clearly present but NOT dominant.
        # Want 15-50% of cells in target (visible zone, not whole-core).
        # Score = n_target * diversity_penalty * fraction_penalty
        candidates = []
        for roi in sorted(set(sid[tumor])):
            m = (sid == roi) & tumor
            n_total = int(np.sum(m))
            if n_total < 2000:
                continue
            c_roi_tmp = comp[m]
            n_target = int(np.sum(c_roi_tmp == target_comp))
            if n_target < 200:
                continue
            frac = n_target / n_total
            # Ideal fraction: 15-50%. Penalize >60% heavily (whole-core).
            if frac > 0.70:
                frac_score = 0.1
            elif frac > 0.50:
                frac_score = 0.5
            else:
                frac_score = 1.0
            # Diversity: how many non-excluded compartments present (≥50 cells)?
            non_excl = [c for c in set(c_roi_tmp) if c not in excl_list]
            n_diverse = sum(1 for c in non_excl if np.sum(c_roi_tmp == c) >= 50)
            # Prefer FL TMAs over Biomax (different architecture)
            tma_bonus = 1.0 if 'Biomax' not in tma[m][0] else 0.3
            score = n_target * frac_score * (n_diverse / 5.0) * tma_bonus
            candidates.append((score, n_target, roi, tma[m][0]))

        if not candidates:
            axes[i, 0].set_visible(False)
            axes[i, 1].set_visible(False)
            axes[i, 2].set_visible(False)
            continue

        candidates.sort(key=lambda x: -x[0])
        _, best_n, best_roi, best_tma = candidates[0]

        m_roi = (sid == best_roi) & tumor
        x, y = cx[m_roi], cy[m_roi]
        c_roi = comp[m_roi]
        ct_roi = ct[m_roi]

        # --- Left: compartment map (target highlighted, rest gray) ---
        ax_l = axes[i, 0]
        # Background cells (gray, small, transparent)
        bg = c_roi != target_comp
        ax_l.scatter(x[bg], y[bg], c='#E0E0E0', s=1.5, alpha=0.3,
                     edgecolors='none', rasterized=True)
        # Target compartment cells (colored)
        fg = c_roi == target_comp
        color = comp_colors.get(target_comp, '#808080')
        ax_l.scatter(x[fg], y[fg], c=color, s=4, alpha=0.9,
                     edgecolors='none', rasterized=True)
        ax_l.set_aspect('equal')
        ax_l.invert_yaxis()
        ax_l.axis('off')
        short_name = target_comp[:35]
        ax_l.set_title(f'{short_name}\n{best_roi} ({best_tma}), n={best_n:,}',
                        fontsize=10, fontweight='bold')

        # --- Right: cell types within this compartment ---
        ax_r = axes[i, 1]
        # Background cells (very faint gray)
        ax_r.scatter(x[bg], y[bg], c='#F0F0F0', s=1, alpha=0.15,
                     edgecolors='none', rasterized=True)
        # Target compartment cells colored by cell type
        ct_target = ct_roi[fg]
        x_fg, y_fg = x[fg], y[fg]
        unique_ct = sorted(set(ct_target))
        legend_handles = []
        for ctype in unique_ct:
            m_ct = ct_target == ctype
            n_ct = int(np.sum(m_ct))
            frac = n_ct / len(ct_target)
            if frac < 0.01:
                # Lump very rare types into gray
                ax_r.scatter(x_fg[m_ct], y_fg[m_ct], c='#C0C0C0', s=3,
                             alpha=0.5, edgecolors='none', rasterized=True)
                continue
            c_ct = ct_colors.get(ctype, '#808080')
            ax_r.scatter(x_fg[m_ct], y_fg[m_ct], c=c_ct, s=4, alpha=0.9,
                         edgecolors='none', rasterized=True)
            legend_handles.append(Patch(facecolor=c_ct,
                                        label=f'{ctype} ({frac:.0%})'))
        ax_r.set_aspect('equal')
        ax_r.invert_yaxis()
        ax_r.axis('off')
        ax_r.set_title('Cell types within compartment', fontsize=TITLE_SIZE)

        # Legend in third column
        ax_leg = axes[i, 2]
        ax_leg.axis('off')
        if legend_handles:
            ax_leg.legend(handles=legend_handles, fontsize=LEGEND_SIZE,
                          loc='center left', frameon=False,
                          handlelength=1.2, handletextpad=0.4,
                          borderpad=0, labelspacing=0.3)
    fig_path = os.path.join(output_dir, f'fig_compartment_examples_{panel_label}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Compartment examples saved: {fig_path}")


def generate_roi_archetypes(f_utag, f_v8, output_dir, panel_label,
                            foll_list, inter_list, excl_list,
                            color_map, min_cells=500):
    """Cluster tumor ROIs by compartment composition and generate figure.

    Clusters only on follicular + interfollicular compartments (not excluded).
    ROIs with <30% non-excluded cells are dropped. Per-TMA distribution goes
    to a supplementary figure, not the main figure.
    """
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import pdist
    from sklearn.metrics import silhouette_score

    sid = load_array(f_utag, 'sample_id')
    tma = load_array(f_utag, 'tma')
    comp = rename_labels(load_array(f_utag, 'compartment_name'))
    cx = f_utag['obs']['centroid_x'][:]
    cy = f_utag['obs']['centroid_y'][:]
    tumor = get_tumor_mask(sid)

    # Non-excluded compartments only (for clustering features)
    comp_set = set(comp)
    clust_comps = [c for c in foll_list if c in comp_set] + \
                  [c for c in inter_list if c in comp_set]
    all_comps = clust_comps + [c for c in excl_list if c in comp_set]

    # --- Step 1: Build ROI × compartment fraction matrix ---
    unique_rois = sorted(set(sid[tumor]))
    roi_data = []
    for roi in unique_rois:
        m = (sid == roi) & tumor
        n = int(np.sum(m))
        if n < min_cells:
            continue
        comp_vals = comp[m]
        # Fractions for clustering compartments (renormalized)
        clust_counts = [int(np.sum(comp_vals == c)) for c in clust_comps]
        clust_total = sum(clust_counts)
        if clust_total / n < 0.30:
            continue  # skip ROIs with <30% assigned to real compartments
        clust_fracs = np.array([x / clust_total for x in clust_counts])
        # Full fractions (for display heatmap)
        full_fracs = np.array([np.sum(comp_vals == c) / n for c in all_comps])
        roi_data.append({
            'roi': roi, 'tma': tma[m][0], 'n_cells': n,
            'clust_fracs': clust_fracs, 'full_fracs': full_fracs,
        })

    n_rois = len(roi_data)
    roi_names = [r['roi'] for r in roi_data]
    roi_tmas = [r['tma'] for r in roi_data]
    clust_mat = np.array([r['clust_fracs'] for r in roi_data])
    full_mat = np.array([r['full_fracs'] for r in roi_data])

    n_total_tumor = len(sorted(set(sid[tumor])))
    n_dropped = n_total_tumor - n_rois
    print(f"\n{panel_label}-panel ROI archetypes: {n_rois}/{n_total_tumor} tumor ROIs "
          f"({n_dropped} dropped: <{min_cells} cells or <30% in non-excluded compartments)")
    print(f"  Clustering on {len(clust_comps)} compartments (excl. {len(excl_list)} excluded types)")

    # --- Step 2: Hierarchical clustering on non-excluded fractions ---
    dist = pdist(clust_mat, metric='euclidean')
    Z = linkage(dist, method='ward')

    sil_scores = {}
    for k in range(2, min(9, n_rois)):
        labels = fcluster(Z, t=k, criterion='maxclust')
        sil = silhouette_score(clust_mat, labels, metric='euclidean')
        sil_scores[k] = sil
        print(f"  k={k}: silhouette={sil:.3f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"  Best k={best_k} (silhouette={sil_scores[best_k]:.3f})")
    labels = fcluster(Z, t=best_k, criterion='maxclust')

    # --- Step 3: Main figure (no per-TMA panel) ---
    n_spatial = best_k  # show ALL archetypes
    n_sp_cols = min(n_spatial, 4)
    n_sp_rows = (n_spatial + n_sp_cols - 1) // n_sp_cols

    fig = plt.figure(figsize=(18, 10 + 4.5 * n_sp_rows))
    gs = gridspec.GridSpec(1 + n_sp_rows, 2, figure=fig, hspace=0.35, wspace=0.25,
                           height_ratios=[2.5] + [1.2] * n_sp_rows,
                           left=0.10, right=0.95, top=0.95, bottom=0.04)

    # (a) Clustermap — non-excluded compartments
    ax_hm = fig.add_subplot(gs[0, 0])
    leaf_order = leaves_list(Z)
    mat_ordered = clust_mat[leaf_order]
    labels_ordered = labels[leaf_order]
    tmas_ordered = [roi_tmas[i] for i in leaf_order]

    im = ax_hm.imshow(mat_ordered, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    ax_hm.set_xticks(range(len(clust_comps)))
    short_names = [c[:25] for c in clust_comps]
    ax_hm.set_xticklabels(short_names, fontsize=TICK_SIZE, rotation=45, ha='right')
    ax_hm.set_ylabel('ROIs (clustered)', fontsize=11)
    ax_hm.set_yticks([])

    # Foll/inter separator
    n_foll = len([c for c in foll_list if c in comp_set])
    if n_foll > 0 and n_foll < len(clust_comps):
        ax_hm.axvline(x=n_foll - 0.5, color='black', linewidth=1.5)

    # TMA sidebar
    tma_colors = {'A1': '#e74c3c', 'B1': '#3498db', 'Biomax': '#2ecc71', 'C1': '#9b59b6'}
    for i, t in enumerate(tmas_ordered):
        ax_hm.add_patch(plt.Rectangle((-1.5, i - 0.5), 0.8, 1,
                                       color=tma_colors.get(t, '#888'), clip_on=False))

    # Cluster boundaries
    for i in range(1, len(labels_ordered)):
        if labels_ordered[i] != labels_ordered[i - 1]:
            ax_hm.axhline(y=i - 0.5, color='white', linewidth=2)

    # Cluster labels
    for cl in range(1, best_k + 1):
        idx = np.where(labels_ordered == cl)[0]
        if len(idx) > 0:
            mid = (idx[0] + idx[-1]) / 2
            ax_hm.text(len(clust_comps) + 0.3, mid, f'C{cl}\n(n={len(idx)})',
                       fontsize=10, va='center', fontweight='bold')

    cb = plt.colorbar(im, ax=ax_hm, shrink=0.5, pad=0.12)
    cb.set_label('Fraction', fontsize=LABEL_SIZE)
    ax_hm.set_title(f'{panel_label}: ROI Archetypes (k={best_k}, sil={sil_scores[best_k]:.2f})',
                     fontsize=12, fontweight='bold')
    ax_hm.text(-0.03, 1.02, 'a', transform=ax_hm.transAxes,
               fontsize=14, fontweight='bold', va='top')

    # (b) Cluster profiles — stacked bar (non-excluded fractions)
    ax_prof = fig.add_subplot(gs[0, 1])
    x = np.arange(best_k)
    bottoms = np.zeros(best_k)
    for j, c in enumerate(clust_comps):
        vals = [clust_mat[labels == cl + 1, j].mean() for cl in range(best_k)]
        color = color_map.get(c, '#808080')
        ax_prof.bar(x, vals, bottom=bottoms, color=color, width=0.7,
                    edgecolor='white', linewidth=0.3)
        bottoms += vals
    ax_prof.set_xticks(x)
    ax_prof.set_xticklabels([f'C{i+1}\n(n={np.sum(labels==i+1)})' for i in range(best_k)],
                             fontsize=10)
    ax_prof.set_ylabel('Mean fraction (non-excluded)', fontsize=10)
    ax_prof.set_ylim(0, 1.05)
    ax_prof.set_title('Cluster Profiles', fontsize=TITLE_SIZE, fontweight='bold')
    ax_prof.text(-0.08, 1.02, 'b', transform=ax_prof.transAxes,
                 fontsize=14, fontweight='bold', va='top')

    # (c-...) Spatial examples — one per cluster, prefer dense ROIs
    for cl_idx in range(n_spatial):
        cl = cl_idx + 1
        row = 1 + cl_idx // n_sp_cols
        col = cl_idx % n_sp_cols
        # Use full-width subplots: n_sp_cols per row across both columns
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, n_sp_cols, subplot_spec=gs[row, :], wspace=0.15)
        ax_sp = fig.add_subplot(gs_inner[0, col])

        idx_cl = np.where(labels == cl)[0]
        centroid = clust_mat[idx_cl].mean(axis=0)
        dists = np.linalg.norm(clust_mat[idx_cl] - centroid, axis=1)
        # Pick top-5 closest to centroid, then prefer the one with most cells
        top_n = min(5, len(idx_cl))
        top_idx = idx_cl[np.argsort(dists)[:top_n]]
        top_cells = [roi_data[i]['n_cells'] for i in top_idx]
        rep_idx = top_idx[np.argmax(top_cells)]
        rep_roi = roi_names[rep_idx]
        rep_tma = roi_tmas[rep_idx]

        m = (sid == rep_roi) & tumor
        plot_spatial_map(ax_sp, cx[m], cy[m], comp[m], color_map, rep_roi, rep_tma)
        ax_sp.set_title(f'C{cl}: {rep_roi} ({rep_tma})', fontsize=9)
        label = chr(ord('c') + cl_idx)
        ax_sp.text(-0.02, 1.05, label, transform=ax_sp.transAxes,
                   fontsize=14, fontweight='bold', va='top')

    fig_path = os.path.join(output_dir, f'fig_roi_archetypes_{panel_label}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Main archetype figure saved: {fig_path}")

    # --- Step 4: Supplementary — per-TMA distribution ---
    fig_s, ax_tma = plt.subplots(figsize=(8, 5))
    all_tmas = sorted(tma_colors.keys())
    bar_width = 0.8 / best_k
    for cl_idx in range(best_k):
        cl = cl_idx + 1
        counts = [sum(1 for i in range(n_rois) if labels[i] == cl and roi_tmas[i] == t)
                  for t in all_tmas]
        positions = np.arange(len(all_tmas)) + cl_idx * bar_width
        ax_tma.bar(positions, counts, width=bar_width,
                   color=plt.cm.Set2(cl_idx), edgecolor='white', label=f'C{cl}')
    ax_tma.set_xticks(np.arange(len(all_tmas)) + bar_width * (best_k - 1) / 2)
    ax_tma.set_xticklabels(all_tmas, fontsize=TICK_SIZE)
    ax_tma.set_ylabel('Number of ROIs', fontsize=LABEL_SIZE)
    ax_tma.legend(fontsize=LEGEND_SIZE, ncol=2)
    ax_tma.set_title(f'{panel_label}: ROI Archetype TMA Distribution', fontsize=TITLE_SIZE, fontweight='bold')
    supp_path = os.path.join(output_dir, f'fig_roi_archetypes_{panel_label}_supp.png')
    fig_s.savefig(supp_path, dpi=150, bbox_inches='tight')
    plt.close(fig_s)
    print(f"  Supplementary (per-TMA) saved: {supp_path}")

    # Return results for logging
    cluster_results = {}
    for cl in range(1, best_k + 1):
        idx_cl = np.where(labels == cl)[0]
        cl_rois = [roi_names[i] for i in idx_cl]
        cl_tmas = [roi_tmas[i] for i in idx_cl]
        mean_fracs = clust_mat[idx_cl].mean(axis=0)
        cluster_results[cl] = {
            'n_rois': len(idx_cl),
            'rois': cl_rois,
            'tma_counts': Counter(cl_tmas),
            'mean_fracs': {clust_comps[j]: float(mean_fracs[j]) for j in range(len(clust_comps))},
        }
    return best_k, sil_scores[best_k], sil_scores, cluster_results, clust_comps


def generate_figure(f_v8_t, f_utag_t, f_v8_s, f_utag_s, output_dir):
    """Generate all compartment figures."""

    # Find shared ROIs for matched selection
    sid_t = load_array(f_utag_t, 'sample_id')
    sid_s = load_array(f_utag_s, 'sample_id')
    tumor_t_tmp = get_tumor_mask(sid_t)
    tumor_s_tmp = get_tumor_mask(sid_s)
    shared = sorted(set(sid_t[tumor_t_tmp]) & set(sid_s[tumor_s_tmp]))
    print(f"Shared tumor ROIs: {len(shared)}")

    # T-panel main figure (standalone — kept for reference)
    ret_t = generate_panel_figure('T', f_v8_t, f_utag_t, output_dir,
                                   T_FOLL, T_INTER, T_EXCL,
                                   T_COMPARTMENT_COLORS, T_MARKERS,
                                   shared_rois=shared)

    # S-panel main figure (standalone — kept for reference)
    ret_s = generate_panel_figure('S', f_v8_s, f_utag_s, output_dir,
                                   S_FOLL, S_INTER, S_EXCL,
                                   S_COMPARTMENT_COLORS, S_MARKERS,
                                   shared_rois=shared)

    # Combined T+S figure (no marker heatmaps, bigger fonts)
    combined_ret = generate_combined_figure(
        f_v8_t, f_utag_t, f_v8_s, f_utag_s, output_dir,
        shared_rois=shared)

    # Supplementary (cross-TMA + prevalence)
    sid_t2, tma_t2, comp_t2, tumor_t2, t_order, t_sidebar, t_cmap = ret_t
    sid_s2, tma_s2, comp_s2, tumor_s2, s_order, s_sidebar, s_cmap = ret_s
    generate_supplementary(sid_t2, tma_t2, comp_t2, tumor_t2, t_order, t_sidebar, t_cmap,
                           sid_s2, tma_s2, comp_s2, tumor_s2, s_order, s_sidebar, s_cmap,
                           output_dir)

    # Detailed stats for analysis log
    print("\n" + "=" * 80)
    print("DETAILED COMPARTMENT STATISTICS (for analysis log)")
    print("=" * 80)
    print_compartment_stats('T', f_v8_t, f_utag_t, T_FOLL, T_INTER, T_EXCL,
                            t_order, T_MARKERS)
    print_compartment_stats('S', f_v8_s, f_utag_s, S_FOLL, S_INTER, S_EXCL,
                            s_order, S_MARKERS)

    # Compartment examples (paired compartment vs cell type maps)
    print("\n" + "=" * 80)
    print("COMPARTMENT EXAMPLES")
    print("=" * 80)
    generate_compartment_examples(
        f_utag_t, f_v8_t, output_dir, 'T',
        T_FOLL, T_INTER, T_EXCL, T_COMPARTMENT_COLORS, T_CELLTYPE_COLORS)

    # ROI archetype clustering (T-panel)
    print("\n" + "=" * 80)
    print("ROI ARCHETYPE CLUSTERING")
    print("=" * 80)
    archetype_results = generate_roi_archetypes(
        f_utag_t, f_v8_t, output_dir, 'T',
        T_FOLL, T_INTER, T_EXCL, T_COMPARTMENT_COLORS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t-panel', required=True)
    parser.add_argument('--s-panel', required=True)
    parser.add_argument('--t-utag', required=True)
    parser.add_argument('--s-utag', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--only', choices=['combined', 's-combined', 'all'],
                        default='all',
                        help='Generate only one figure: combined (T, Fig 2), '
                             's-combined (S, Fig S6), or all')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Delete cached panels before regenerating')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.clear_cache:
        import shutil
        for subdir in ['_cache_compartments', '_cache_compartments_S']:
            cache_dir = os.path.join(args.output_dir, subdir)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"Cleared cache: {cache_dir}")

    f_v8_t = h5py.File(args.t_panel, 'r')
    f_utag_t = h5py.File(args.t_utag, 'r')
    f_v8_s = h5py.File(args.s_panel, 'r')
    f_utag_s = h5py.File(args.s_utag, 'r')

    # Shared ROIs (needed by combined figures)
    def _get_shared_rois():
        sid_t = load_array(f_utag_t, 'sample_id')
        sid_s = load_array(f_utag_s, 'sample_id')
        tumor_t = get_tumor_mask(sid_t)
        tumor_s = get_tumor_mask(sid_s)
        shared = sorted(set(sid_t[tumor_t]) & set(sid_s[tumor_s]))
        print(f"Shared tumor ROIs: {len(shared)}")
        return shared

    if args.only == 'combined':
        shared = _get_shared_rois()
        generate_combined_figure(f_v8_t, f_utag_t, f_v8_s, f_utag_s,
                                  args.output_dir, shared_rois=shared)
    elif args.only == 's-combined':
        shared = _get_shared_rois()
        generate_combined_figure_s(f_v8_s, f_utag_s,
                                    args.output_dir, shared_rois=shared)
    else:
        generate_figure(f_v8_t, f_utag_t, f_v8_s, f_utag_s, args.output_dir)

    f_v8_t.close()
    f_utag_t.close()
    f_v8_s.close()
    f_utag_s.close()


if __name__ == '__main__':
    main()
