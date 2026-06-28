#!/usr/bin/env python3
"""Hypothesis testing v2: pooled cohort, control exclusion, leave-one-TMA-out.

Implements analysis principles:
  1. Pool all tumor ROIs as one cohort — pooled result shown first
  2. Exclude Biomax control cores (_Ton_, _Adr_)
  3. Interpretation / driver panels (cell type correlations with metric)
  4. Leave-one-TMA-out sensitivity check — shown last
  5. Publication figures with statistical annotations

Figure layout per hypothesis:
  concept → pooled result → interpretation/drivers → per-TMA sensitivity

Tests: H6a (entropy), H2a (Tfh spatial), H2e (cytotoxic barrier), H3a (macrophage niches), H6c (domain entropy)
"""

import argparse, os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, pearsonr, spearmanr
from collections import Counter

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


def get_marker_idx(f):
    key = '_index' if '_index' in f['var'] else 'index'
    names = f['var'][key][:]
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return {n: i for i, n in enumerate(names)}


def load_marker(f, name, idx):
    if name in idx:
        col = idx[name]
    else:
        alt = name.replace('-', '_')
        if alt in idx:
            col = idx[alt]
        else:
            return None
    X = f['X']
    if len(X.shape) == 1:
        return None
    return X[:, col]


def is_tumor_core(sample_id):
    """Returns True if the ROI is a tumor (FL) core, False if control."""
    s = sample_id
    s_lower = s.lower()
    # Biomax-style short codes
    if '_ton_' in s_lower or '_adr_' in s_lower:
        return False
    # In-house TMA control tissue names (A1/B1/C1 Prostate, Kidney, Spleen, Tonsil)
    for tissue in ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal']:
        if tissue in s_lower:
            return False
    if s == 'Biomax_ROI_006':
        return False  # unknown, tiny
    return True


def get_tumor_mask(sample_ids):
    """Boolean mask: True for tumor cores, False for controls."""
    return np.array([is_tumor_core(s) for s in sample_ids])


def classify_domains(f_utag, cell_types, panel):
    """Classify cells as follicular or interfollicular via tissue compartment names.

    Reads 'compartment_name' from the merged UTAG h5ad (produced by
    utag_name_compartments.py) and classifies each compartment by keyword
    matching into follicular, interfollicular, or excluded.

    Returns (compartments, foll, inter) where compartments is the per-cell
    array of compartment names, and foll/inter are lists of name strings.
    """
    compartments = load_array(f_utag, 'compartment_name')

    # Follicular compartments (B cell-dominated zones)
    foll_kw = ['GC B', 'Follicle core', 'Follicle mantle', 'Activated B',
               'B cell follicle', 'B cell zone']
    if panel == 'S':
        foll_kw.extend(['FDC', 'BCL2+', 'PAX5+'])

    # Interfollicular compartments
    inter_kw = ['T cell zone', 'Treg', 'Macrophage', 'Cytotoxic', 'interface']
    if panel == 'S':
        inter_kw.extend(['Stromal', 'Endothelial', 'Dendritic', 'pDC',
                         'Histiocyte', 'Myeloid'])

    unique_names = sorted(set(compartments))
    foll, inter, excluded = [], [], []
    for name in unique_names:
        is_f = any(k in name for k in foll_kw)
        is_i = any(k in name for k in inter_kw)
        if is_f and not is_i:
            foll.append(name)
        elif is_i:
            inter.append(name)
        else:
            excluded.append(name)

    print(f"  Follicular ({len(foll)}): {foll}")
    print(f"  Interfollicular ({len(inter)}): {inter}")
    if excluded:
        print(f"  Excluded ({len(excluded)}): {excluded}")

    return compartments, foll, inter


def leave_one_out_check(values_by_tma, test_func, label):
    """Run leave-one-TMA-out sensitivity. Returns list of (excluded, p, note)."""
    tmas = sorted(values_by_tma.keys())
    results = []
    for exclude in tmas:
        remaining = {t: v for t, v in values_by_tma.items() if t != exclude}
        p = test_func(remaining)
        results.append((exclude, p))
    return results


def add_cartoon(ax, cartoon_path):
    """Load and display cartoon in the given axes."""
    if os.path.exists(cartoon_path):
        img = mpimg.imread(cartoon_path)
        ax.imshow(img)
    ax.set_title('Concept', fontsize=11)
    ax.axis('off')


def label_panel(ax, letter):
    """Add a bold lowercase letter label (a, b, c, …) at top-left of axes."""
    ax.text(-0.08, 1.08, letter, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')


def plot_representative_core_spatial(ax, cx, cy, sample_ids, cell_types, tumor_mask,
                                      roi_metric, label_lo='Low', label_hi='High',
                                      metric_name='Metric', top_n_types=10,
                                      domains=None, foll_domains=None,
                                      highlight_mask=None, highlight_label=None,
                                      highlight_color='#FFD700', min_cells=8000,
                                      highlight_size=20, min_highlight_hi=5,
                                      domain_focus=False):
    """Show spatial scatter of cells for representative low and high ROIs.

    Plots cells at (centroid_x, centroid_y) colored by cell type.
    When domains/foll_domains provided, adds background tint for follicular zones.
    highlight_size: marker size for highlighted cells (default 20).
    min_highlight_hi: minimum highlighted cells required in the high ROI (default 5).
    domain_focus: if True, suppress cell type colors and show only domain overlay +
                  highlighted cells. All non-highlighted cells rendered as uniform gray.
    When highlight_mask provided, overlays those cells as larger colored markers.
    Filters out ROIs with fewer than min_cells to avoid sparse/fragmented cores.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Filter ROIs by minimum *non-LQ* cell count to avoid sparse/damaged cores
    lq_types = {'Low quality / Unassigned'}
    non_lq = tumor_mask & ~np.isin(cell_types, list(lq_types))
    roi_cell_counts = Counter(sample_ids[non_lq])
    rois = sorted(r for r in roi_metric.keys()
                  if roi_cell_counts.get(r, 0) >= min_cells)
    vals = np.array([roi_metric[r] for r in rois])
    if len(vals) < 4:
        ax.text(0.5, 0.5, 'Too few ROIs', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return

    p15 = np.percentile(vals, 15)
    p85 = np.percentile(vals, 85)
    idx_lo = np.argmin(np.abs(vals - p15))
    idx_hi = np.argmin(np.abs(vals - p85))

    # If highlight_mask provided, ensure high ROI has enough highlighted cells.
    # Search from highest metric downward until we find one with enough.
    if highlight_mask is not None and min_highlight_hi > 0:
        sorted_desc = np.argsort(-vals)
        for idx_candidate in sorted_desc:
            roi_c = rois[idx_candidate]
            m_c = sample_ids == roi_c
            n_hl = np.sum(highlight_mask[m_c])
            if n_hl >= min_highlight_hi:
                idx_hi = idx_candidate
                break

    roi_lo, roi_hi = rois[idx_lo], rois[idx_hi]
    val_lo, val_hi = vals[idx_lo], vals[idx_hi]

    ax.axis('off')
    ax_lo = ax.inset_axes([0.0, 0.0, 0.47, 0.88])
    ax_hi = ax.inset_axes([0.53, 0.0, 0.47, 0.88])

    # Build unified color palette from both ROIs
    combined_counts = Counter()
    for roi in [roi_lo, roi_hi]:
        m = sample_ids == roi
        ct = cell_types[m]
        combined_counts.update(ct[~np.isin(ct, list(lq_types))])
    sorted_types = [k for k, _ in combined_counts.most_common(top_n_types)]

    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    type_to_rgba = {}
    for i, t in enumerate(sorted_types):
        type_to_rgba[t] = tab20[i % 20]

    for sub_ax, roi, val, label in [(ax_lo, roi_lo, val_lo, label_lo),
                                     (ax_hi, roi_hi, val_hi, label_hi)]:
        m = sample_ids == roi
        x_all, y_all, ct_all = cx[m], cy[m], cell_types[m]
        is_lq = np.isin(ct_all, list(lq_types))
        typed = ~is_lq

        # Domain background tint (all cells, including LQ)
        if domains is not None and foll_domains is not None:
            dom = domains[m]
            is_foll = np.isin(dom, foll_domains)
            if np.any(is_foll):
                sub_ax.scatter(x_all[is_foll], y_all[is_foll], c='#FFDDDD', s=5,
                              alpha=0.3, edgecolors='none', rasterized=True, zorder=0)
            if np.any(~is_foll):
                sub_ax.scatter(x_all[~is_foll], y_all[~is_foll], c='#DDDDFF', s=5,
                              alpha=0.3, edgecolors='none', rasterized=True, zorder=0)

        # All non-highlighted cells as light gray (domain_focus) or colored by type
        if domain_focus:
            # Domain-focus mode: all cells as uniform gray, only highlight pops
            sub_ax.scatter(x_all, y_all, c='#D3D3D3', s=0.3,
                          alpha=0.4, edgecolors='none', rasterized=True, zorder=1)
        else:
            # Full cell type mode
            if np.any(is_lq):
                sub_ax.scatter(x_all[is_lq], y_all[is_lq], c='#D3D3D3', s=0.3,
                              alpha=0.4, edgecolors='none', rasterized=True, zorder=1)
            x, y, ct = x_all[typed], y_all[typed], ct_all[typed]
            n = len(ct)
            rgba = np.full((n, 4), 0.0)
            rgba[:, :3] = 0.82
            rgba[:, 3] = 0.5
            for t, c in type_to_rgba.items():
                mask_t = ct == t
                rgba[mask_t] = c
            sub_ax.scatter(x, y, c=rgba, s=0.5, edgecolors='none', rasterized=True, zorder=2)

        # Highlight specific cells (e.g., Tfh, cytotoxic)
        if highlight_mask is not None:
            hl = highlight_mask[m][typed]
            if np.any(hl):
                x_typed, y_typed = x_all[typed], y_all[typed]
                sub_ax.scatter(x_typed[hl], y_typed[hl], c=highlight_color, s=highlight_size,
                              edgecolors='black', linewidths=0.5, zorder=3, rasterized=True)

        sub_ax.set_aspect('equal')
        sub_ax.invert_yaxis()
        sub_ax.axis('off')

        short = roi if len(roi) <= 18 else roi[:16] + '..'
        sub_ax.set_title(f'{label}\n{short}\n{metric_name}={val:.2f}', fontsize=7)

    # Legend
    legend_items = []
    if not domain_focus:
        legend_items = [Patch(facecolor=type_to_rgba[t], label=t[:25]) for t in sorted_types]
        legend_items.append(Patch(facecolor='#D3D3D3', alpha=0.5, label='Unidentified'))
    if highlight_mask is not None and highlight_label:
        legend_items.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=highlight_color, markeredgecolor='black',
                                    markersize=7, label=highlight_label))
    if domains is not None:
        legend_items.append(Patch(facecolor='#FFDDDD', alpha=0.5, label='Follicular zone'))
        legend_items.append(Patch(facecolor='#DDDDFF', alpha=0.5, label='Interfollicular'))
    ax.legend(handles=legend_items, loc='lower center', ncol=3, fontsize=5,
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.set_title('Representative Cores', fontsize=10)


def plot_single_core_spatial(ax, cx, cy, sample_ids, cell_types, tumor_mask,
                             domains, foll_domains, inter_domains,
                             highlight_mask=None, highlight_label=None,
                             highlight_color='#FFD700', highlight_size=20,
                             roi_name=None, metric_value=None, metric_name='Tfh%'):
    """Show ONE representative core with follicular/interfollicular domain overlay + highlighted cells.

    If roi_name is None, auto-selects the ROI with the most balanced follicular/interfollicular split
    that also has enough highlighted cells.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    lq_types = {'Low quality / Unassigned'}

    # --- Auto-select best ROI if not specified ---
    if roi_name is None:
        unique_rois = sorted(set(sample_ids[tumor_mask]))
        best_roi, best_score = None, -1
        for roi in unique_rois:
            m = (sample_ids == roi) & tumor_mask
            n_total = np.sum(m)
            if n_total < 3000:
                continue
            dom = domains[m]
            n_foll = np.sum(np.isin(dom, foll_domains))
            n_inter = np.sum(np.isin(dom, inter_domains))
            if n_foll < 100 or n_inter < 100:
                continue
            balance = min(n_foll, n_inter) / max(n_foll, n_inter)  # 0-1
            n_hl = np.sum(highlight_mask[m]) if highlight_mask is not None else 0
            score = balance * 0.6 + min(n_hl / 50, 1.0) * 0.4
            if score > best_score:
                best_score = score
                best_roi = roi
        roi_name = best_roi

    if roi_name is None:
        ax.text(0.5, 0.5, 'No suitable ROI', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return

    m = (sample_ids == roi_name) & tumor_mask
    x_all, y_all = cx[m], cy[m]
    dom = domains[m]
    ct_all = cell_types[m]
    is_lq = np.isin(ct_all, list(lq_types))

    # Domain background tint
    is_foll = np.isin(dom, foll_domains)
    is_inter = np.isin(dom, inter_domains)
    is_excl = ~is_foll & ~is_inter
    if np.any(is_foll):
        ax.scatter(x_all[is_foll], y_all[is_foll], c='#FFCCCC', s=8,
                   alpha=0.5, edgecolors='none', rasterized=True, zorder=0)
    if np.any(is_inter):
        ax.scatter(x_all[is_inter], y_all[is_inter], c='#CCCCFF', s=8,
                   alpha=0.5, edgecolors='none', rasterized=True, zorder=0)
    if np.any(is_excl):
        ax.scatter(x_all[is_excl], y_all[is_excl], c='#E8E8E8', s=8,
                   alpha=0.3, edgecolors='none', rasterized=True, zorder=0)

    # Non-highlighted cells as gray
    ax.scatter(x_all, y_all, c='#D3D3D3', s=0.5,
               alpha=0.4, edgecolors='none', rasterized=True, zorder=1)

    # Highlight cells
    if highlight_mask is not None:
        hl = highlight_mask[m] & ~is_lq
        if np.any(hl):
            ax.scatter(x_all[hl], y_all[hl], c=highlight_color, s=highlight_size,
                       edgecolors='black', linewidths=0.5, zorder=3, rasterized=True)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    short = roi_name if len(roi_name) <= 20 else roi_name[:18] + '..'
    title = f'Representative Core\n{short}'
    if metric_value is not None:
        title += f'\n{metric_name}={metric_value:.1f}'
    ax.set_title(title, fontsize=9)

    # Legend
    legend_items = []
    if highlight_label:
        legend_items.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=highlight_color, markeredgecolor='black',
                                    markersize=7, label=highlight_label))
    legend_items.append(Patch(facecolor='#FFCCCC', alpha=0.7, label='Follicular'))
    legend_items.append(Patch(facecolor='#CCCCFF', alpha=0.7, label='Interfollicular'))
    ax.legend(handles=legend_items, loc='lower center', ncol=3, fontsize=6,
              bbox_to_anchor=(0.5, -0.05), frameon=False)


def compute_roi_celltype_fractions(sample_ids, cell_types, tumor_mask, lq_types=None):
    """Compute per-ROI cell type fractions. Returns dict of {roi: {ct: frac}}."""
    if lq_types is None:
        lq_types = {'Low quality / Unassigned'}

    unique_rois = sorted(set(sample_ids[tumor_mask]))
    roi_fracs = {}
    for roi in unique_rois:
        roi_mask = (sample_ids == roi) & tumor_mask
        ct = cell_types[roi_mask]
        ct_clean = ct[~np.isin(ct, list(lq_types))]
        if len(ct_clean) < 50:
            continue
        counts = Counter(ct_clean)
        total = sum(counts.values())
        roi_fracs[roi] = {k: v / total for k, v in counts.items()}
    return roi_fracs


def correlate_celltype_with_metric(roi_fracs, roi_metric, min_rois=10):
    """Correlate per-ROI cell type fractions with a metric.

    Args:
        roi_fracs: dict {roi: {ct: frac}}
        roi_metric: dict {roi: value}
        min_rois: skip cell types present in fewer ROIs

    Returns: list of (cell_type, rho, p) sorted by |rho| descending.
    """
    common_rois = sorted(set(roi_fracs.keys()) & set(roi_metric.keys()))
    if len(common_rois) < 5:
        return []

    metric_vals = np.array([roi_metric[r] for r in common_rois])
    all_types = set()
    for fracs in roi_fracs.values():
        all_types.update(fracs.keys())

    results = []
    for ct in sorted(all_types):
        ct_vals = np.array([roi_fracs[r].get(ct, 0.0) for r in common_rois])
        # Skip if cell type present in too few ROIs
        if np.sum(ct_vals > 0) < min_rois:
            continue
        # Skip if no variance
        if np.std(ct_vals) < 1e-8 or np.std(metric_vals) < 1e-8:
            continue
        rho, p = spearmanr(ct_vals, metric_vals)
        if not np.isnan(rho):
            results.append((ct, rho, p))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# H6a: Per-ROI Phenotypic Entropy (pooled cohort)
# ---------------------------------------------------------------------------

def run_h6a(f, panel, sample_ids, tma_arr, tumor_mask, output_dir, cartoon_dir, cx, cy):
    print(f"\n{'='*70}")
    print(f"H6a: Per-ROI Phenotypic Entropy — {panel}-panel (pooled tumor cohort)")
    print(f"{'='*70}")

    cell_types = load_array(f, 'cell_type')
    lq_types = {'Low quality / Unassigned'}

    unique_rois = sorted(set(sample_ids[tumor_mask]))
    roi_data = []  # (sample_id, tma, H, H_clean, n_cells)

    for roi in unique_rois:
        roi_mask = (sample_ids == roi) & tumor_mask
        n = np.sum(roi_mask)
        if n < 50:
            continue

        ct = cell_types[roi_mask]
        tma_val = tma_arr[roi_mask][0]

        # Full entropy
        counts = Counter(ct)
        total = sum(counts.values())
        props = np.array([c / total for c in counts.values()])
        props = props[props > 0]
        H = -np.sum(props * np.log2(props))

        # Clean entropy (exclude LQ)
        clean_ct = ct[~np.isin(ct, list(lq_types))]
        if len(clean_ct) > 20:
            counts_c = Counter(clean_ct)
            total_c = sum(counts_c.values())
            props_c = np.array([c / total_c for c in counts_c.values()])
            props_c = props_c[props_c > 0]
            H_clean = -np.sum(props_c * np.log2(props_c))
        else:
            H_clean = np.nan

        roi_data.append((roi, tma_val, H, H_clean, n))

    # Pooled stats
    all_H = np.array([r[2] for r in roi_data])
    all_H_clean = np.array([r[3] for r in roi_data if not np.isnan(r[3])])
    print(f"\nPooled tumor ROIs: {len(roi_data)}")
    print(f"Entropy: mean={np.mean(all_H):.2f}, std={np.std(all_H):.2f}, "
          f"range=[{np.min(all_H):.2f}, {np.max(all_H):.2f}]")
    print(f"Clean entropy: mean={np.mean(all_H_clean):.2f}, std={np.std(all_H_clean):.2f}")

    # Group by TMA for Kruskal-Wallis
    tma_groups = {}
    tma_groups_clean = {}
    for roi, tma_val, H, H_clean, n in roi_data:
        tma_groups.setdefault(tma_val, []).append(H)
        if not np.isnan(H_clean):
            tma_groups_clean.setdefault(tma_val, []).append(H_clean)

    tmas = sorted(tma_groups.keys())
    print(f"\n{'TMA':>8} {'n_ROIs':>7} {'mean_H':>8} {'std_H':>7} {'mean_Hclean':>12}")
    print("-" * 50)
    for t in tmas:
        vals = tma_groups[t]
        vals_c = tma_groups_clean.get(t, [])
        print(f"{t:>8} {len(vals):>7} {np.mean(vals):>8.2f} {np.std(vals):>7.2f} "
              f"{np.mean(vals_c):>12.2f}" if vals_c else f"{t:>8} {len(vals):>7} {np.mean(vals):>8.2f} {np.std(vals):>7.2f}")

    # Kruskal-Wallis on pooled
    groups = [tma_groups[t] for t in tmas]
    H_stat, p_val = kruskal(*groups)
    print(f"\nKruskal-Wallis (full): H={H_stat:.2f}, p={p_val:.4f}")

    p_c = 1.0
    groups_c = [tma_groups_clean[t] for t in tmas if t in tma_groups_clean]
    if len(groups_c) >= 2:
        H_c, p_c = kruskal(*groups_c)
        print(f"Kruskal-Wallis (clean): H={H_c:.2f}, p={p_c:.4f}")

    # --- Entropy drivers: correlate cell type fractions with clean entropy ---
    roi_fracs = compute_roi_celltype_fractions(sample_ids, cell_types, tumor_mask, lq_types)
    roi_entropy = {r[0]: r[3] for r in roi_data if not np.isnan(r[3])}
    drivers = correlate_celltype_with_metric(roi_fracs, roi_entropy)

    print(f"\nEntropy drivers (Spearman ρ with clean entropy):")
    for ct, rho, p in drivers[:15]:
        print(f"  {ct:40s}  ρ={rho:+.3f}  p={p:.2e}")

    # Leave-one-TMA-out
    print(f"\nLeave-one-TMA-out sensitivity (clean entropy):")
    loo_pvals = []
    for exclude in tmas:
        remaining = [tma_groups_clean[t] for t in tmas if t != exclude and t in tma_groups_clean]
        if len(remaining) >= 2:
            _, p_loo = kruskal(*remaining)
            loo_pvals.append((exclude, p_loo))
            print(f"  Exclude {exclude}: p={p_loo:.4f}")

    # --- Figure: 2 rows × 3 cols ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Cartoon
    add_cartoon(axes[0, 0], os.path.join(cartoon_dir, 'h6a_entropy_concept.png'))
    label_panel(axes[0, 0], 'a')

    # (b) Pooled distribution (main result)
    ax = axes[0, 1]
    ax.hist(all_H_clean, bins=25, color='#2c3e50', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(all_H_clean), color='red', linestyle='--', lw=1.5,
               label=f'Mean = {np.mean(all_H_clean):.2f}')
    ax.axvline(np.median(all_H_clean), color='orange', linestyle=':', lw=1.5,
               label=f'Median = {np.median(all_H_clean):.2f}')
    ax.set_xlabel('Shannon Entropy (bits)')
    ax.set_ylabel('Number of ROIs')
    ax.set_title(f'Pooled Distribution\n(n={len(all_H_clean)} tumor ROIs, KW p={p_c:.4f})')
    ax.legend(fontsize=8)
    label_panel(ax, 'b')

    # (c) Entropy drivers (top correlates)
    ax = axes[0, 2]
    top_n = min(12, len(drivers))
    if top_n > 0:
        top = drivers[:top_n]
        names = [d[0] for d in reversed(top)]
        rhos = [d[1] for d in reversed(top)]
        colors_bar = ['#e74c3c' if r < 0 else '#2ecc71' for r in rhos]
        ax.barh(range(len(names)), rhos, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Spearman ρ')
        ax.set_title('Entropy Drivers\n(cell type ρ with H)')
        ax.axvline(0, color='gray', lw=0.5)
    else:
        ax.text(0.5, 0.5, 'No drivers\nfound', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Entropy Drivers')
    label_panel(ax, 'c')

    # (d) Per-TMA boxplot
    ax = axes[1, 0]
    colors_box = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    box_data = [tma_groups_clean.get(t, []) for t in tmas]
    bp = ax.boxplot(box_data, tick_labels=tmas, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors_box[:len(tmas)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('Per-TMA Breakdown')
    ax.set_xlabel('TMA')
    label_panel(ax, 'd')

    # (e) Leave-one-out
    ax = axes[1, 1]
    if loo_pvals:
        excl_names = [x[0] for x in loo_pvals]
        pvals = [x[1] for x in loo_pvals]
        ax.bar(excl_names, [-np.log10(p) for p in pvals],
               color=colors_box[:len(excl_names)], alpha=0.7)
        ax.axhline(-np.log10(0.05), color='gray', linestyle='--', label='p=0.05')
        ax.set_ylabel('-log10(p-value)')
        ax.set_xlabel('Excluded TMA')
        ax.set_title('Leave-One-TMA-Out')
        ax.legend(fontsize=8)
    label_panel(ax, 'e')

    # (f) Representative cores: low-entropy vs high-entropy
    ax = axes[1, 2]
    plot_representative_core_spatial(ax, cx, cy, sample_ids, cell_types, tumor_mask,
                                      roi_entropy, label_lo='Low H', label_hi='High H',
                                      metric_name='H')
    label_panel(ax, 'f')

    plt.suptitle(f'H6a: Phenotypic Entropy — {panel}-panel (tumor only)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(output_dir, f'fig_h6a_{panel}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    return tma_groups_clean


# ---------------------------------------------------------------------------
# H2a + H2e: Tfh follicular enrichment + cytotoxic barrier
# ---------------------------------------------------------------------------

def run_h2a_h2e(f_v8, f_utag, marker_idx, panel, sample_ids, tma_arr, tumor_mask,
                cell_types, output_dir, cartoon_dir, cx, cy):
    print(f"\n{'='*70}")
    print(f"H2a + H2e: Tfh Enrichment & Cytotoxic Barrier — {panel}-panel")
    print(f"{'='*70}")

    domains, foll, inter = classify_domains(f_utag, cell_types, panel)
    foll_mask = np.isin(domains, foll) & tumor_mask
    inter_mask = np.isin(domains, inter) & tumor_mask

    n_foll = np.sum(foll_mask)
    n_inter = np.sum(inter_mask)
    print(f"Tumor follicular: {n_foll:,} cells | Interfollicular: {n_inter:,} cells")

    # Load markers
    cd3 = load_marker(f_v8, 'CD3', marker_idx)
    cd4 = load_marker(f_v8, 'CD4', marker_idx)
    cd8 = load_marker(f_v8, 'CD8a', marker_idx)
    cd20 = load_marker(f_v8, 'CD20', marker_idx)
    cxcr5 = load_marker(f_v8, 'CXCR5', marker_idx)
    pd1 = load_marker(f_v8, 'PD-1', marker_idx)
    granzb = load_marker(f_v8, 'GranzymeB', marker_idx)

    # Gates
    cd4t = (cd3 > 0.5) & (cd4 > 0.5) & (cd8 < 0.5) & (cd20 < 0.5) & tumor_mask
    tfh = cd4t & (cxcr5 > 2.0)
    tfh_pd1hi = tfh & (pd1 > 0.5)
    cd8t = (cd3 > 0.5) & (cd8 > 0.5) & tumor_mask
    cytotox = cd8t & (granzb > 0.5)

    # --- H2a: Tfh enrichment ---
    tfh_foll = np.sum(tfh & foll_mask)
    tfh_inter = np.sum(tfh & inter_mask)
    cd4t_foll = np.sum(cd4t & foll_mask)
    cd4t_inter = np.sum(cd4t & inter_mask)

    pct_f = 100 * tfh_foll / cd4t_foll if cd4t_foll > 0 else 0
    pct_i = 100 * tfh_inter / cd4t_inter if cd4t_inter > 0 else 0
    enrichment = pct_f / pct_i if pct_i > 0 else float('inf')

    ct_h2a = np.array([[tfh_foll, cd4t_foll - tfh_foll],
                        [tfh_inter, cd4t_inter - tfh_inter]])
    chi2_h2a, p_h2a, _, _ = chi2_contingency(ct_h2a)

    print(f"\nH2a — Tfh (CXCR5>2.0) among CD4 T:")
    print(f"  Follicular: {tfh_foll:,}/{cd4t_foll:,} = {pct_f:.2f}%")
    print(f"  Interfollicular: {tfh_inter:,}/{cd4t_inter:,} = {pct_i:.2f}%")
    print(f"  Enrichment: {enrichment:.2f}x (χ²={chi2_h2a:.1f}, p={p_h2a:.2e})")

    # Per-TMA H2a
    tmas = sorted(set(tma_arr[tumor_mask]))
    print(f"\n  {'TMA':>8} {'Foll%':>8} {'Inter%':>8} {'Enrich':>8} {'p':>12}")
    tma_h2a = {}
    for t in tmas:
        tm = tma_arr == t
        tf = np.sum(tfh & foll_mask & tm)
        ti = np.sum(tfh & inter_mask & tm)
        cf = np.sum(cd4t & foll_mask & tm)
        ci = np.sum(cd4t & inter_mask & tm)
        pf = 100 * tf / cf if cf > 0 else 0
        pi = 100 * ti / ci if ci > 0 else 0
        en = pf / pi if pi > 0 else float('inf')
        try:
            _, pv, _, _ = chi2_contingency(np.array([[tf, max(cf-tf, 0)], [ti, max(ci-ti, 0)]]))
        except:
            pv = 1.0
        print(f"  {t:>8} {pf:>7.2f}% {pi:>7.2f}% {en:>7.2f}x {pv:>12.2e}")
        tma_h2a[t] = {'foll': pf, 'inter': pi, 'enrichment': en}

    # Leave-one-out H2a
    print(f"\n  Leave-one-out H2a:")
    for exclude in tmas:
        tm_keep = np.array([t != exclude for t in tma_arr]) & tumor_mask
        tf_f = np.sum(tfh & foll_mask & tm_keep)
        tf_i = np.sum(tfh & inter_mask & tm_keep)
        c_f = np.sum(cd4t & foll_mask & tm_keep)
        c_i = np.sum(cd4t & inter_mask & tm_keep)
        pf2 = 100 * tf_f / c_f if c_f > 0 else 0
        pi2 = 100 * tf_i / c_i if c_i > 0 else 0
        en2 = pf2 / pi2 if pi2 > 0 else float('inf')
        print(f"    Excl {exclude}: foll={pf2:.2f}%, inter={pi2:.2f}%, enrich={en2:.2f}x")

    # --- H2e: Cytotoxic barrier ---
    cyt_foll = np.sum(cytotox & foll_mask)
    cyt_inter = np.sum(cytotox & inter_mask)
    cd8_foll = np.sum(cd8t & foll_mask)
    cd8_inter = np.sum(cd8t & inter_mask)

    pct_cyt_f = 100 * cyt_foll / cd8_foll if cd8_foll > 0 else 0
    pct_cyt_i = 100 * cyt_inter / cd8_inter if cd8_inter > 0 else 0

    ct_h2e = np.array([[cyt_foll, cd8_foll - cyt_foll],
                        [cyt_inter, cd8_inter - cyt_inter]])
    chi2_h2e, p_h2e, _, _ = chi2_contingency(ct_h2e)

    print(f"\nH2e — CD8+GzmB+ cytotoxic:")
    print(f"  Follicular: {cyt_foll:,}/{cd8_foll:,} = {pct_cyt_f:.2f}%")
    print(f"  Interfollicular: {cyt_inter:,}/{cd8_inter:,} = {pct_cyt_i:.2f}%")
    print(f"  χ²={chi2_h2e:.1f}, p={p_h2e:.2e}")

    # --- Interpretation: per-ROI Tfh% correlates ---
    # Compute per-ROI Tfh% of CD4 T, then correlate with cell type fractions
    rois_tumor = sorted(set(sample_ids[tumor_mask]))
    roi_tfh_pct = {}
    for roi in rois_tumor:
        rm = (sample_ids == roi) & tumor_mask
        n_cd4 = np.sum(cd4t & rm)
        if n_cd4 < 20:
            continue
        n_tfh = np.sum(tfh & rm)
        roi_tfh_pct[roi] = 100.0 * n_tfh / n_cd4

    roi_fracs = compute_roi_celltype_fractions(sample_ids, cell_types, tumor_mask)
    tfh_drivers = correlate_celltype_with_metric(roi_fracs, roi_tfh_pct)

    print(f"\nTfh% drivers (Spearman ρ):")
    for ct_name, rho, p_d in tfh_drivers[:10]:
        print(f"  {ct_name:40s}  ρ={rho:+.3f}  p={p_d:.2e}")

    # --- Figure: 2 rows × 3 cols ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Cartoon
    add_cartoon(axes[0, 0], os.path.join(cartoon_dir, 'h2a_tfh_follicular.png'))
    label_panel(axes[0, 0], 'a')

    # (b) Pooled H2a bar
    ax = axes[0, 1]
    ax.bar(['Follicular', 'Interfollicular'], [pct_f, pct_i],
           color=['#e74c3c', '#3498db'], alpha=0.8, edgecolor='white')
    ax.set_ylabel('Tfh % of CD4 T')
    ax.set_title(f'H2a: Tfh Enrichment (pooled)\n{enrichment:.1f}x, p={p_h2a:.1e}')
    max_y = max(pct_f, pct_i)
    ax.plot([0, 0, 1, 1], [max_y*1.05, max_y*1.1, max_y*1.1, max_y*1.05], 'k-', lw=1)
    ax.text(0.5, max_y*1.12, f'p={p_h2a:.1e}', ha='center', fontsize=9)
    label_panel(ax, 'b')

    # (c) Pooled H2e bar
    ax = axes[0, 2]
    ax.bar(['Follicular', 'Interfollicular'], [pct_cyt_f, pct_cyt_i],
           color=['#e74c3c', '#3498db'], alpha=0.8, edgecolor='white')
    ax.set_ylabel('GzmB+ % of CD8 T')
    ax.set_title(f'H2e: Cytotoxic CD8 Barrier\n(p={p_h2e:.1e})')
    max_y2 = max(pct_cyt_f, pct_cyt_i)
    ax.plot([0, 0, 1, 1], [max_y2*1.05, max_y2*1.1, max_y2*1.1, max_y2*1.05], 'k-', lw=1)
    ax.text(0.5, max_y2*1.12, f'p={p_h2e:.1e}', ha='center', fontsize=9)
    label_panel(ax, 'c')

    # (d) Tfh drivers
    ax = axes[1, 0]
    top_n = min(10, len(tfh_drivers))
    if top_n > 0:
        top = tfh_drivers[:top_n]
        names = [d[0] for d in reversed(top)]
        rhos = [d[1] for d in reversed(top)]
        colors_bar = ['#e74c3c' if r < 0 else '#2ecc71' for r in rhos]
        ax.barh(range(len(names)), rhos, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Spearman ρ')
        ax.set_title('Tfh% Drivers\n(per-ROI correlates)')
        ax.axvline(0, color='gray', lw=0.5)
    label_panel(ax, 'd')

    # (e) Per-TMA enrichment bars
    ax = axes[1, 1]
    x = np.arange(len(tmas))
    w = 0.35
    ax.bar(x - w/2, [tma_h2a[t]['foll'] for t in tmas], w,
           label='Follicular', color='#e74c3c', alpha=0.8)
    ax.bar(x + w/2, [tma_h2a[t]['inter'] for t in tmas], w,
           label='Interfollicular', color='#3498db', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tmas)
    ax.set_ylabel('Tfh % of CD4 T')
    ax.set_title('Per-TMA Sensitivity')
    ax.legend(fontsize=8)
    label_panel(ax, 'e')

    # (f) Single representative core with follicular/interfollicular overlay + Tfh highlight
    ax = axes[1, 2]
    # Auto-select ROI with both zones and most Tfh cells
    best_roi = None
    best_tfh_val = None
    best_score = -1
    for roi_candidate in sorted(roi_tfh_pct.keys()):
        rm = (sample_ids == roi_candidate) & tumor_mask
        n_total = np.sum(rm)
        if n_total < 3000:
            continue
        dom_c = domains[rm]
        n_f = np.sum(np.isin(dom_c, foll))
        n_i = np.sum(np.isin(dom_c, inter))
        n_hl = np.sum(tfh[rm])
        if n_f < 500 or n_i < 500 or n_hl < 5:
            continue
        balance = min(n_f, n_i) / max(n_f, n_i)
        score = balance * 0.3 + min(n_hl / 100, 1.0) * 0.7  # prioritize Tfh count
        if score > best_score:
            best_score = score
            best_roi = roi_candidate
            best_tfh_val = roi_tfh_pct[roi_candidate]
    plot_single_core_spatial(ax, cx, cy, sample_ids, cell_types, tumor_mask,
                             domains, foll, inter,
                             highlight_mask=tfh, highlight_label='Tfh (CD4+CXCR5hi)',
                             highlight_color='#FFD700', highlight_size=20,
                             roi_name=best_roi, metric_value=best_tfh_val,
                             metric_name='Tfh%')
    label_panel(ax, 'f')

    plt.suptitle('H2a + H2e: T Cell Spatial Organization (tumor only)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(output_dir, f'fig_h2a_h2e_{panel}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")


# ---------------------------------------------------------------------------
# H2b: CD57+ Tfh spatial niche (T-panel)
# ---------------------------------------------------------------------------

def run_h2b(f_v8, f_utag, marker_idx, panel, sample_ids, tma_arr, tumor_mask,
            cell_types, output_dir, cartoon_dir, cx, cy):
    """H2b: Do CD57+ Tfh occupy a distinct spatial niche vs CD57- Tfh?

    CD57+ Tfh are a late-differentiation subset linked to poor prognosis
    (Mayo Clinic CyTOF study). Tests whether CD57+ Tfh are more
    follicle-restricted than CD57- Tfh.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    print(f"\n{'='*70}")
    print(f"H2b: CD57+ Tfh Spatial Niche — {panel}-panel")
    print(f"{'='*70}")

    domains, foll, inter = classify_domains(f_utag, cell_types, panel)
    foll_mask = np.isin(domains, foll) & tumor_mask
    inter_mask = np.isin(domains, inter) & tumor_mask

    n_foll = np.sum(foll_mask)
    n_inter = np.sum(inter_mask)
    print(f"Tumor follicular: {n_foll:,} cells | Interfollicular: {n_inter:,} cells")

    # Load markers
    cd3 = load_marker(f_v8, 'CD3', marker_idx)
    cd4 = load_marker(f_v8, 'CD4', marker_idx)
    cd8 = load_marker(f_v8, 'CD8a', marker_idx)
    cd20 = load_marker(f_v8, 'CD20', marker_idx)
    cxcr5 = load_marker(f_v8, 'CXCR5', marker_idx)
    cd57 = load_marker(f_v8, 'CD57', marker_idx)

    if cd57 is None:
        print("ERROR: CD57 marker not found in panel — skipping H2b")
        return

    # Gates (reuse H2a Tfh gate exactly, then split by CD57)
    cd4t = (cd3 > 0.5) & (cd4 > 0.5) & (cd8 < 0.5) & (cd20 < 0.5) & tumor_mask
    tfh = cd4t & (cxcr5 > 2.0)
    tfh_cd57pos = tfh & (cd57 > 0.5)
    tfh_cd57neg = tfh & (cd57 <= 0.5)

    n_tfh = np.sum(tfh)
    n_cd57pos = np.sum(tfh_cd57pos)
    n_cd57neg = np.sum(tfh_cd57neg)
    print(f"\nTotal Tfh (CD4+CXCR5>2.0): {n_tfh:,}")
    print(f"  CD57+ Tfh: {n_cd57pos:,} ({100*n_cd57pos/n_tfh:.1f}%)" if n_tfh > 0 else "  CD57+ Tfh: 0")
    print(f"  CD57- Tfh: {n_cd57neg:,} ({100*n_cd57neg/n_tfh:.1f}%)" if n_tfh > 0 else "  CD57- Tfh: 0")

    # --- Test 1: CD57+ Tfh enrichment in follicular vs interfollicular ---
    cd57pos_foll = np.sum(tfh_cd57pos & foll_mask)
    cd57pos_inter = np.sum(tfh_cd57pos & inter_mask)
    cd57neg_foll = np.sum(tfh_cd57neg & foll_mask)
    cd57neg_inter = np.sum(tfh_cd57neg & inter_mask)

    # Fraction of each Tfh subset that is follicular
    frac_pos_foll = cd57pos_foll / (cd57pos_foll + cd57pos_inter) if (cd57pos_foll + cd57pos_inter) > 0 else 0
    frac_neg_foll = cd57neg_foll / (cd57neg_foll + cd57neg_inter) if (cd57neg_foll + cd57neg_inter) > 0 else 0

    print(f"\nFollicular fraction:")
    print(f"  CD57+ Tfh: {cd57pos_foll:,}/{cd57pos_foll+cd57pos_inter:,} = {100*frac_pos_foll:.1f}% follicular")
    print(f"  CD57- Tfh: {cd57neg_foll:,}/{cd57neg_foll+cd57neg_inter:,} = {100*frac_neg_foll:.1f}% follicular")

    # Chi-squared: CD57+ Tfh vs CD57- Tfh × follicular vs interfollicular
    ct_table = np.array([[cd57pos_foll, cd57pos_inter],
                          [cd57neg_foll, cd57neg_inter]])
    try:
        chi2, p_diff, _, _ = chi2_contingency(ct_table)
        print(f"  χ²={chi2:.1f}, p={p_diff:.2e}")
    except:
        chi2, p_diff = 0, 1.0
        print(f"  Could not compute chi-squared")

    # Also: CD57+ Tfh as % of CD4T in follicular vs interfollicular
    cd4t_foll = np.sum(cd4t & foll_mask)
    cd4t_inter = np.sum(cd4t & inter_mask)
    pct_pos_foll = 100 * cd57pos_foll / cd4t_foll if cd4t_foll > 0 else 0
    pct_pos_inter = 100 * cd57pos_inter / cd4t_inter if cd4t_inter > 0 else 0
    enrich_pos = pct_pos_foll / pct_pos_inter if pct_pos_inter > 0 else float('inf')

    pct_neg_foll = 100 * cd57neg_foll / cd4t_foll if cd4t_foll > 0 else 0
    pct_neg_inter = 100 * cd57neg_inter / cd4t_inter if cd4t_inter > 0 else 0
    enrich_neg = pct_neg_foll / pct_neg_inter if pct_neg_inter > 0 else float('inf')

    print(f"\n  CD57+ Tfh % of CD4T: foll={pct_pos_foll:.2f}%, inter={pct_pos_inter:.2f}%, enrich={enrich_pos:.2f}x")
    print(f"  CD57- Tfh % of CD4T: foll={pct_neg_foll:.2f}%, inter={pct_neg_inter:.2f}%, enrich={enrich_neg:.2f}x")

    # --- Test 2: Per-ROI differential niche (paired Wilcoxon) ---
    rois_tumor = sorted(set(sample_ids[tumor_mask]))
    roi_frac_pos_foll = {}
    roi_frac_neg_foll = {}
    roi_cd57_frac = {}  # fraction of Tfh that is CD57+

    for roi in rois_tumor:
        rm = (sample_ids == roi) & tumor_mask
        n_tfh_roi = np.sum(tfh & rm)
        if n_tfh_roi < 5:
            continue

        # CD57+ and CD57- Tfh counts in follicular vs total domain-assigned
        pos_f = np.sum(tfh_cd57pos & foll_mask & rm)
        pos_i = np.sum(tfh_cd57pos & inter_mask & rm)
        neg_f = np.sum(tfh_cd57neg & foll_mask & rm)
        neg_i = np.sum(tfh_cd57neg & inter_mask & rm)

        pos_total = pos_f + pos_i
        neg_total = neg_f + neg_i

        if pos_total >= 3:
            roi_frac_pos_foll[roi] = pos_f / pos_total
        if neg_total >= 3:
            roi_frac_neg_foll[roi] = neg_f / neg_total
        roi_cd57_frac[roi] = np.sum(tfh_cd57pos & rm) / n_tfh_roi

    # Paired test on ROIs that have both subsets
    common = sorted(set(roi_frac_pos_foll.keys()) & set(roi_frac_neg_foll.keys()))
    if len(common) >= 5:
        pos_vals = np.array([roi_frac_pos_foll[r] for r in common])
        neg_vals = np.array([roi_frac_neg_foll[r] for r in common])
        from scipy.stats import wilcoxon
        stat_w, p_paired = wilcoxon(pos_vals, neg_vals)
        mean_diff = np.mean(pos_vals - neg_vals)
        print(f"\nPaired Wilcoxon (CD57+ vs CD57- Tfh follicular fraction):")
        print(f"  n={len(common)} ROIs, mean diff={mean_diff:+.3f}")
        print(f"  W={stat_w:.1f}, p={p_paired:.2e}")
    else:
        p_paired = np.nan
        mean_diff = np.nan
        print(f"\nInsufficient ROIs for paired test (n={len(common)})")

    # --- Per-TMA breakdown ---
    tmas = sorted(set(tma_arr[tumor_mask]))
    print(f"\n  {'TMA':>8} {'CD57+ foll%':>12} {'CD57- foll%':>12} {'CD57+ enrich':>13}")
    tma_data = {}
    for t in tmas:
        tm = tma_arr == t
        pf = np.sum(tfh_cd57pos & foll_mask & tm)
        pi = np.sum(tfh_cd57pos & inter_mask & tm)
        nf = np.sum(tfh_cd57neg & foll_mask & tm)
        ni = np.sum(tfh_cd57neg & inter_mask & tm)
        frac_p = pf / (pf + pi) if (pf + pi) > 0 else 0
        frac_n = nf / (nf + ni) if (nf + ni) > 0 else 0
        # CD57+ Tfh as % of CD4T in foll vs inter
        cf = np.sum(cd4t & foll_mask & tm)
        ci = np.sum(cd4t & inter_mask & tm)
        ep_f = 100 * pf / cf if cf > 0 else 0
        ep_i = 100 * pi / ci if ci > 0 else 0
        en_p = ep_f / ep_i if ep_i > 0 else float('inf')
        print(f"  {t:>8} {100*frac_p:>11.1f}% {100*frac_n:>11.1f}% {en_p:>12.1f}x")
        tma_data[t] = {'pos_foll_frac': frac_p, 'neg_foll_frac': frac_n,
                       'pos_pct_foll': ep_f, 'pos_pct_inter': ep_i, 'enrich': en_p}

    # --- Leave-one-TMA-out ---
    loo_results = []
    print(f"\n  Leave-one-out H2b (CD57+ vs CD57- follicular fraction chi-squared):")
    for exclude in tmas:
        tm_keep = np.array([t != exclude for t in tma_arr]) & tumor_mask
        pf2 = np.sum(tfh_cd57pos & foll_mask & tm_keep)
        pi2 = np.sum(tfh_cd57pos & inter_mask & tm_keep)
        nf2 = np.sum(tfh_cd57neg & foll_mask & tm_keep)
        ni2 = np.sum(tfh_cd57neg & inter_mask & tm_keep)
        try:
            _, p_loo, _, _ = chi2_contingency(np.array([[pf2, pi2], [nf2, ni2]]))
        except:
            p_loo = 1.0
        fp2 = pf2 / (pf2 + pi2) if (pf2 + pi2) > 0 else 0
        fn2 = nf2 / (nf2 + ni2) if (nf2 + ni2) > 0 else 0
        loo_results.append((exclude, p_loo))
        print(f"    Excl {exclude}: CD57+ foll={100*fp2:.1f}%, CD57- foll={100*fn2:.1f}%, p={p_loo:.2e}")

    # --- Driver analysis: what cell types correlate with CD57+ Tfh fraction ---
    roi_fracs = compute_roi_celltype_fractions(sample_ids, cell_types, tumor_mask)
    drivers = correlate_celltype_with_metric(roi_fracs, roi_cd57_frac)

    print(f"\nCD57+ Tfh fraction drivers (Spearman ρ):")
    for ct_name, rho, p_d in drivers[:10]:
        print(f"  {ct_name:40s}  ρ={rho:+.3f}  p={p_d:.2e}")

    # ===================================================================
    # Helper: plot a representative core with domain overlay + CD57+/- Tfh
    # ===================================================================
    def _plot_representative_core(ax, roi, metric_val):
        m = (sample_ids == roi) & tumor_mask
        x_all, y_all = cx[m], cy[m]
        dom = domains[m]
        ct_roi = cell_types[m]
        is_lq = np.isin(ct_roi, ['Low quality / Unassigned'])

        is_foll_r = np.isin(dom, foll)
        is_inter_r = np.isin(dom, inter)
        is_excl_r = ~is_foll_r & ~is_inter_r
        if np.any(is_foll_r):
            ax.scatter(x_all[is_foll_r], y_all[is_foll_r], c='#FFCCCC', s=8,
                       alpha=0.5, edgecolors='none', rasterized=True, zorder=0)
        if np.any(is_inter_r):
            ax.scatter(x_all[is_inter_r], y_all[is_inter_r], c='#CCCCFF', s=8,
                       alpha=0.5, edgecolors='none', rasterized=True, zorder=0)
        if np.any(is_excl_r):
            ax.scatter(x_all[is_excl_r], y_all[is_excl_r], c='#E8E8E8', s=8,
                       alpha=0.3, edgecolors='none', rasterized=True, zorder=0)
        ax.scatter(x_all, y_all, c='#D3D3D3', s=0.3,
                   alpha=0.4, edgecolors='none', rasterized=True, zorder=1)
        hl_neg = tfh_cd57neg[m] & ~is_lq
        if np.any(hl_neg):
            ax.scatter(x_all[hl_neg], y_all[hl_neg], c='#FFD700', s=12,
                       edgecolors='black', linewidths=0.3, zorder=2, rasterized=True)
        hl_pos = tfh_cd57pos[m] & ~is_lq
        if np.any(hl_pos):
            ax.scatter(x_all[hl_pos], y_all[hl_pos], c='#FF1493', s=20,
                       edgecolors='black', linewidths=0.5, zorder=3, rasterized=True)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        short = roi if len(roi) <= 20 else roi[:18] + '..'
        ax.set_title(f'{short}\nCD57+ frac={100*metric_val:.0f}%', fontsize=9)
        ax.axis('off')

    # Build candidate ROI list for representative cores
    candidate_rois = []
    for roi in sorted(roi_cd57_frac.keys()):
        rm = (sample_ids == roi) & tumor_mask
        n_total = np.sum(rm)
        if n_total < 3000:
            continue
        dom_c = domains[rm]
        n_f = np.sum(np.isin(dom_c, foll))
        n_i = np.sum(np.isin(dom_c, inter))
        n_pos = np.sum(tfh_cd57pos[rm])
        if n_f < 500 or n_i < 500:
            continue
        candidate_rois.append((roi, roi_cd57_frac[roi], n_pos))

    # Select high and low CD57+ fraction ROIs
    hi_roi, lo_roi = None, None
    hi_metric, lo_metric = 0, 0
    if candidate_rois:
        by_metric = sorted(candidate_rois, key=lambda x: x[1])
        # Low: ~15th percentile, must have >=3 CD57+ Tfh
        idx_lo = max(0, int(0.15 * len(by_metric)))
        for i in range(idx_lo, len(by_metric)):
            if by_metric[i][2] >= 3:
                lo_roi, lo_metric = by_metric[i][0], by_metric[i][1]
                break
        # High: ~85th percentile, must have >=5 CD57+ Tfh
        idx_hi = min(len(by_metric) - 1, int(0.85 * len(by_metric)))
        for i in range(idx_hi, -1, -1):
            if by_metric[i][2] >= 5:
                hi_roi, hi_metric = by_metric[i][0], by_metric[i][1]
                break

    # ===================================================================
    # MAIN FIGURE: 2 rows × 3 cols (all pooled cross-TMA)
    # ===================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Cartoon
    add_cartoon(axes[0, 0], os.path.join(cartoon_dir, 'h2b_cd57_tfh_niche.png'))
    label_panel(axes[0, 0], 'a')

    # (b) CD57+ vs CD57- Tfh follicular enrichment (grouped bars, pooled)
    ax = axes[0, 1]
    x_b = np.arange(2)
    w_b = 0.35
    bars_pos = [pct_pos_foll, pct_pos_inter]
    bars_neg = [pct_neg_foll, pct_neg_inter]
    ax.bar(x_b - w_b/2, bars_pos, w_b, label='CD57+ Tfh', color='#FF1493', alpha=0.85)
    ax.bar(x_b + w_b/2, bars_neg, w_b, label='CD57- Tfh', color='#FFD700', alpha=0.85)
    ax.set_xticks(x_b)
    ax.set_xticklabels(['Follicular', 'Interfollicular'])
    ax.set_ylabel('Tfh subset % of CD4 T')
    ax.set_title(f'CD57+/- Tfh Enrichment (pooled)\nχ²={chi2:.0f}, p={p_diff:.1e}')
    ax.legend(fontsize=9, loc='upper right')
    label_panel(ax, 'b')

    # (c) Pooled per-ROI CD57+ Tfh fraction distribution (violin + strip, no TMA split)
    ax = axes[0, 2]
    all_fracs = [100 * v for v in roi_cd57_frac.values()]
    parts = ax.violinplot(all_fracs, positions=[0], showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#9b59b6')
        pc.set_alpha(0.3)
    parts['cmedians'].set_color('black')
    jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(all_fracs))
    ax.scatter(jitter, all_fracs, c='#9b59b6', alpha=0.5, s=15, edgecolors='white', linewidths=0.3)
    ax.set_xticks([0])
    ax.set_xticklabels(['All tumor ROIs'])
    ax.set_ylabel('CD57+ % of all Tfh')
    med_val = float(np.median(all_fracs))
    ax.set_title(f'CD57+ Tfh Fraction per ROI\n(n={len(all_fracs)}, median={med_val:.0f}%)')
    label_panel(ax, 'c')

    # (d) CD57+ Tfh fraction drivers (Spearman rho)
    ax = axes[1, 0]
    top_n = min(10, len(drivers))
    if top_n > 0:
        top = drivers[:top_n]
        names = [d[0] for d in reversed(top)]
        rhos = [d[1] for d in reversed(top)]
        colors_bar = ['#e74c3c' if r < 0 else '#2ecc71' for r in rhos]
        ax.barh(range(len(names)), rhos, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Spearman ρ')
        ax.set_title('CD57+ Tfh Fraction Drivers\n(per-ROI correlates)')
        ax.axvline(0, color='gray', lw=0.5)
    label_panel(ax, 'd')

    # (e) Representative core — low CD57+ Tfh fraction
    ax = axes[1, 1]
    if lo_roi:
        _plot_representative_core(ax, lo_roi, lo_metric)
        ax.set_title(f'Low CD57+ core\n' + ax.get_title(), fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No suitable ROI', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    label_panel(ax, 'e')

    # (f) Representative core — high CD57+ Tfh fraction
    ax = axes[1, 2]
    if hi_roi:
        _plot_representative_core(ax, hi_roi, hi_metric)
        ax.set_title(f'High CD57+ core\n' + ax.get_title(), fontsize=9)
        # Shared legend on the high-metric panel
        legend_items = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF1493',
                   markeredgecolor='black', markersize=7, label='CD57+ Tfh'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700',
                   markeredgecolor='black', markersize=7, label='CD57- Tfh'),
            Patch(facecolor='#D3D3D3', alpha=0.5, label='Other cells'),
            Patch(facecolor='#FFCCCC', alpha=0.7, label='Follicular'),
            Patch(facecolor='#CCCCFF', alpha=0.7, label='Interfollicular'),
        ]
        ax.legend(handles=legend_items, loc='lower center', ncol=3, fontsize=5,
                  bbox_to_anchor=(0.5, -0.08), frameon=False)
    else:
        ax.text(0.5, 0.5, 'No suitable ROI', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    label_panel(ax, 'f')

    plt.suptitle(f'H2b: CD57+ Tfh Spatial Niche ({panel}-panel, tumor only)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(output_dir, f'fig_h2b_{panel}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # ===================================================================
    # SUPPLEMENTARY FIGURE: per-TMA breakdown (1 row × 3 cols)
    # ===================================================================
    fig_s, axes_s = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Per-TMA strip plot of CD57+ Tfh fraction
    ax = axes_s[0]
    tma_for_roi = {roi: tma_arr[(sample_ids == roi) & tumor_mask][0] for roi in roi_cd57_frac}
    colors_tma = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for i, t in enumerate(tmas):
        vals_t = [roi_cd57_frac[r] for r in roi_cd57_frac if tma_for_roi.get(r) == t]
        jitter = np.random.RandomState(42 + i).uniform(-0.15, 0.15, len(vals_t))
        ax.scatter(np.full(len(vals_t), i) + jitter, [100*v for v in vals_t],
                   c=colors_tma[i % len(colors_tma)], alpha=0.6, s=18, edgecolors='white', linewidths=0.3)
    ax.boxplot([[100*roi_cd57_frac[r] for r in roi_cd57_frac if tma_for_roi.get(r) == t]
                for t in tmas],
               positions=range(len(tmas)), widths=0.5, showfliers=False,
               medianprops=dict(color='black', linewidth=1.5),
               boxprops=dict(facecolor='white', alpha=0.5), patch_artist=True)
    ax.set_xticks(range(len(tmas)))
    ax.set_xticklabels(tmas)
    ax.set_ylabel('CD57+ % of all Tfh')
    ax.set_title(f'CD57+ Tfh Fraction per ROI by TMA\n(n={len(roi_cd57_frac)} ROIs)')
    label_panel(ax, 'a')

    # (b) Per-TMA sensitivity: CD57+ vs CD57- Tfh follicular fraction
    ax = axes_s[1]
    x_e = np.arange(len(tmas))
    w_e = 0.35
    ax.bar(x_e - w_e/2, [100*tma_data[t]['pos_foll_frac'] for t in tmas], w_e,
           label='CD57+ Tfh', color='#FF1493', alpha=0.85)
    ax.bar(x_e + w_e/2, [100*tma_data[t]['neg_foll_frac'] for t in tmas], w_e,
           label='CD57- Tfh', color='#FFD700', alpha=0.85)
    ax.set_xticks(x_e)
    ax.set_xticklabels(tmas)
    ax.set_ylabel('% follicular')
    ax.set_title('Per-TMA Sensitivity\n(follicular fraction by CD57 status)')
    ax.legend(fontsize=8)
    label_panel(ax, 'b')

    # (c) Leave-one-TMA-out chi-squared p-values
    ax = axes_s[2]
    loo_names = [x[0] for x in loo_results]
    loo_pvals = [x[1] for x in loo_results]
    ax.bar(loo_names, [-np.log10(p) if p > 0 else 10 for p in loo_pvals],
           color=colors_tma[:len(loo_names)], alpha=0.7)
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', label='p=0.05')
    ax.set_ylabel('-log10(p-value)')
    ax.set_xlabel('Excluded TMA')
    ax.set_title('Leave-One-TMA-Out\n(CD57+ vs CD57- follicular fraction)')
    ax.legend(fontsize=8)
    label_panel(ax, 'c')

    plt.suptitle(f'H2b Supplementary: Per-TMA Breakdown ({panel}-panel)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    supp_path = os.path.join(output_dir, f'fig_h2b_{panel}_supp.png')
    plt.savefig(supp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Supplementary saved: {supp_path}")


# ---------------------------------------------------------------------------
# H3a: Macrophage niche separation (S-panel, Biomax tumor only)
# ---------------------------------------------------------------------------

def run_h3a(f_v8, f_utag, marker_idx, sample_ids, tma_arr, tumor_mask,
            cell_types, output_dir, cartoon_dir, cx, cy):
    print(f"\n{'='*70}")
    print(f"H3a: Macrophage Niche Separation — S-panel (Biomax tumor cores)")
    print(f"{'='*70}")

    domains, foll, inter = classify_domains(f_utag, cell_types, 'S')

    # Biomax tumor only (CD163/CD206 only reliable there)
    biomax_tumor = (tma_arr == 'Biomax') & tumor_mask
    n_bt = np.sum(biomax_tumor)
    print(f"Biomax tumor cells: {n_bt:,}")

    cd68 = load_marker(f_v8, 'CD68', marker_idx)
    cd163 = load_marker(f_v8, 'CD163', marker_idx)
    cd206 = load_marker(f_v8, 'CD206', marker_idx)
    cd14 = load_marker(f_v8, 'CD14', marker_idx)
    s100a9 = load_marker(f_v8, 'S100A9', marker_idx)

    mac = (cd68 > 0.5) & biomax_tumor
    m2 = mac & (cd163 > 0.5) & (cd206 > 0.5)
    inflam = mac & (cd14 > 0.5) & (s100a9 > 0.5)

    n_mac = np.sum(mac)
    n_m2 = np.sum(m2)
    n_inf = np.sum(inflam)
    print(f"CD68+ macrophages: {n_mac:,}")
    print(f"M2-like: {n_m2:,} ({100*n_m2/n_mac:.1f}%)")
    print(f"Inflammatory: {n_inf:,} ({100*n_inf/n_mac:.1f}%)")

    # Per UTAG domain
    foll_mask = np.isin(domains, foll) & biomax_tumor
    inter_mask = np.isin(domains, inter) & biomax_tumor

    unique_d = sorted(set(domains[biomax_tumor]))
    domain_data = []
    print(f"\n{'Compartment':>45} {'n_mac':>8} {'M2%':>7} {'Inflam%':>8} {'Type'}")
    print("-" * 75)
    for d in unique_d:
        dm = (domains == d) & biomax_tumor
        n_d_mac = np.sum(mac & dm)
        if n_d_mac < 10:
            continue
        n_d_m2 = np.sum(m2 & dm)
        n_d_inf = np.sum(inflam & dm)
        pct_m2 = 100 * n_d_m2 / n_d_mac
        pct_inf = 100 * n_d_inf / n_d_mac
        dtype = "FOLL" if d in foll else "INTER"
        print(f"{d:>45} {n_d_mac:>8,} {pct_m2:>6.1f}% {pct_inf:>7.1f}%  {dtype}")
        domain_data.append((d, n_d_mac, pct_m2, pct_inf, dtype))

    # Per-ROI analysis within Biomax tumor
    rois = sorted(set(sample_ids[biomax_tumor]))
    roi_m2 = []
    roi_inf = []
    roi_m2_pct_dict = {}
    for roi in rois:
        rm = (sample_ids == roi) & biomax_tumor
        n_rm = np.sum(mac & rm)
        if n_rm < 10:
            continue
        m2_pct = 100 * np.sum(m2 & rm) / n_rm
        inf_pct = 100 * np.sum(inflam & rm) / n_rm
        roi_m2.append(m2_pct)
        roi_inf.append(inf_pct)
        roi_m2_pct_dict[roi] = m2_pct

    r, p_corr = np.nan, 1.0
    if len(roi_m2) >= 3:
        r, p_corr = pearsonr(roi_m2, roi_inf)
        print(f"\nPer-ROI M2% vs Inflam% (Biomax tumor): r={r:.3f}, p={p_corr:.4f}")

    roi_fracs_bio = compute_roi_celltype_fractions(sample_ids, cell_types, biomax_tumor)
    m2_drivers = correlate_celltype_with_metric(roi_fracs_bio, roi_m2_pct_dict, min_rois=5)

    print(f"\nM2% drivers (Spearman ρ, Biomax tumor ROIs):")
    for ct_name, rho, p_d in m2_drivers[:10]:
        print(f"  {ct_name:40s}  ρ={rho:+.3f}  p={p_d:.2e}")

    # --- Figure: 2 rows × 3 cols ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Cartoon
    add_cartoon(axes[0, 0], os.path.join(cartoon_dir, 'h3a_macrophage_niches.png'))
    label_panel(axes[0, 0], 'a')

    # (b) Per-ROI scatter (main result)
    ax = axes[0, 1]
    if len(roi_m2) >= 3:
        ax.scatter(roi_m2, roi_inf, c='#2c3e50', alpha=0.7, s=60, edgecolors='white', lw=0.5)
        z = np.polyfit(roi_m2, roi_inf, 1)
        xline = np.linspace(min(roi_m2), max(roi_m2), 100)
        ax.plot(xline, np.polyval(z, xline), 'r--', lw=1.5, alpha=0.7)
        ax.set_xlabel('M2-like % of CD68+')
        ax.set_ylabel('Inflammatory % of CD68+')
        ax.set_title(f'Per-ROI (Biomax tumor)\nr={r:.2f}, p={p_corr:.3f}')
    else:
        ax.text(0.5, 0.5, 'Too few ROIs', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Per-ROI Scatter')
    label_panel(ax, 'b')

    # (c) M2% drivers
    ax = axes[0, 2]
    top_n = min(10, len(m2_drivers))
    if top_n > 0:
        top = m2_drivers[:top_n]
        names = [d[0] for d in reversed(top)]
        rhos = [d[1] for d in reversed(top)]
        colors_bar = ['#9b59b6' if r > 0 else '#e67e22' for r in rhos]
        ax.barh(range(len(names)), rhos, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Spearman ρ')
        ax.set_title('M2% Drivers\n(per-ROI correlates)')
        ax.axvline(0, color='gray', lw=0.5)
    else:
        ax.text(0.5, 0.5, 'No drivers', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('M2% Drivers')
    label_panel(ax, 'c')

    # (d) Per-domain bars
    ax = axes[1, 0]
    if domain_data:
        ds = [d[0] for d in domain_data]
        m2_vals = [d[2] for d in domain_data]
        inf_vals = [d[3] for d in domain_data]
        x = np.arange(len(ds))
        w = 0.35
        ax.bar(x - w/2, m2_vals, w, label='M2-like', color='#9b59b6', alpha=0.8)
        ax.bar(x + w/2, inf_vals, w, label='Inflammatory', color='#e67e22', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d[0][:20]}\n({d[4]})" for d in domain_data],
                           fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('% of CD68+ macrophages')
        ax.set_title('Per-Compartment Breakdown')
        ax.legend(fontsize=8)
    label_panel(ax, 'd')

    # (e) Polarization pie
    ax = axes[1, 1]
    other_mac = max(n_mac - n_m2 - n_inf, 0)
    ax.pie([n_m2, n_inf, other_mac],
           labels=['M2-like', 'Inflammatory', 'Other CD68+'],
           colors=['#9b59b6', '#e67e22', '#95a5a6'],
           autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Biomax Tumor Macrophages\n(n={n_mac:,})')
    label_panel(ax, 'e')

    # (f) Representative cores: low-M2 vs high-M2
    ax = axes[1, 2]
    plot_representative_core_spatial(ax, cx, cy, sample_ids, cell_types, biomax_tumor,
                                      roi_m2_pct_dict, label_lo='Low M2', label_hi='High M2',
                                      metric_name='M2%', min_cells=5000)
    label_panel(ax, 'f')

    plt.suptitle('H3a: Macrophage Niche Separation (Biomax tumor cores)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(output_dir, 'fig_h3a_S.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")


# ---------------------------------------------------------------------------
# H6c: Per-compartment entropy
# ---------------------------------------------------------------------------

def run_h6c(f_utag, cell_types, tma_arr, tumor_mask, panel, output_dir, cartoon_dir,
            cx, cy, sample_ids):
    print(f"\n{'='*70}")
    print(f"H6c: Per-Compartment Entropy — {panel}-panel (tumor only)")
    print(f"{'='*70}")

    domains, foll, inter = classify_domains(f_utag, cell_types, panel)

    # Exclude LQ
    lq_types = {'Low quality / Unassigned'}

    # --- Global per-compartment entropy ---
    unique_d = sorted(set(domains[tumor_mask]))
    global_H = {}  # {compartment_name: (H, type_str, n)}

    print(f"\n{'Compartment':>45} {'N tumor':>10} {'H (bits)':>9} {'Type':>6}")
    print("-" * 75)

    for d in unique_d:
        dm = (domains == d) & tumor_mask
        ct = cell_types[dm]
        ct_clean = ct[~np.isin(ct, list(lq_types))]
        n = len(ct_clean)
        if n < 100:
            continue

        counts = Counter(ct_clean)
        total = sum(counts.values())
        props = np.array([c / total for c in counts.values()])
        props = props[props > 0]
        H = -np.sum(props * np.log2(props))

        dtype = "FOLL" if d in foll else ("INTER" if d in inter else "EXCL")
        print(f"{d:>45} {n:>10,} {H:>8.2f}  {dtype}")
        global_H[d] = (H, dtype, n)

    # --- Per-ROI per-compartment entropy (for statistical test) ---
    unique_rois = sorted(set(sample_ids[tumor_mask]))
    foll_H = []
    inter_H = []

    for roi in unique_rois:
        for comp in unique_d:
            if comp not in foll and comp not in inter:
                continue
            m = (sample_ids == roi) & (domains == comp) & tumor_mask
            ct = cell_types[m]
            ct_clean = ct[~np.isin(ct, list(lq_types))]
            n = len(ct_clean)
            if n < 50:
                continue
            counts = Counter(ct_clean)
            total = sum(counts.values())
            props = np.array([c / total for c in counts.values()])
            props = props[props > 0]
            H = -np.sum(props * np.log2(props))
            if comp in foll:
                foll_H.append(H)
            else:
                inter_H.append(H)

    p_mw = 1.0
    if foll_H and inter_H:
        print(f"\nPer-ROI per-compartment entropy:")
        print(f"Follicular: {np.mean(foll_H):.2f} ± {np.std(foll_H):.2f} (n={len(foll_H)})")
        print(f"Interfollicular: {np.mean(inter_H):.2f} ± {np.std(inter_H):.2f} (n={len(inter_H)})")
        if len(foll_H) >= 2 and len(inter_H) >= 2:
            U, p_mw = mannwhitneyu(foll_H, inter_H, alternative='two-sided')
            print(f"Mann-Whitney U={U:.0f}, p={p_mw:.4f}")

    # --- Interpretation: average cell type composition by domain type ---
    foll_mask_all = np.isin(domains, foll) & tumor_mask
    inter_mask_all = np.isin(domains, inter) & tumor_mask
    lq_types = {'Low quality / Unassigned'}

    ct_foll = cell_types[foll_mask_all]
    ct_foll_clean = ct_foll[~np.isin(ct_foll, list(lq_types))]
    ct_inter = cell_types[inter_mask_all]
    ct_inter_clean = ct_inter[~np.isin(ct_inter, list(lq_types))]

    foll_counts = Counter(ct_foll_clean)
    inter_counts = Counter(ct_inter_clean)
    all_types = sorted(set(list(foll_counts.keys()) + list(inter_counts.keys())))

    foll_total = sum(foll_counts.values())
    inter_total = sum(inter_counts.values())
    comp_data = []
    for ct_name in all_types:
        f_pct = 100 * foll_counts.get(ct_name, 0) / foll_total if foll_total > 0 else 0
        i_pct = 100 * inter_counts.get(ct_name, 0) / inter_total if inter_total > 0 else 0
        diff = f_pct - i_pct
        comp_data.append((ct_name, f_pct, i_pct, diff))

    # Sort by absolute difference
    comp_data.sort(key=lambda x: abs(x[3]), reverse=True)

    print(f"\nCell type composition (foll vs inter compartments, top differences):")
    for ct_name, f_pct, i_pct, diff in comp_data[:10]:
        print(f"  {ct_name:40s}  foll={f_pct:5.1f}%  inter={i_pct:5.1f}%  Δ={diff:+5.1f}%")

    # Extract per-compartment entropy from global_H for plotting
    domain_entropy = {d: global_H[d][0] for d in global_H}

    # --- Figure: 2 rows × 3 cols ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Cartoon
    add_cartoon(axes[0, 0], os.path.join(cartoon_dir, 'h6c_domain_entropy.png'))
    label_panel(axes[0, 0], 'a')

    # (b) Pooled boxplot (main result)
    ax = axes[0, 1]
    if foll_H and inter_H:
        bp = ax.boxplot([foll_H, inter_H], tick_labels=['Follicular', 'Interfollicular'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#3498db')
        bp['boxes'][1].set_alpha(0.6)
        ax.set_ylabel('Shannon Entropy (bits)')
        ax.set_title(f'Compartment Entropy (per-ROI)\n(MW p={p_mw:.4f}, n_f={len(foll_H)}, n_i={len(inter_H)})')
        y_max = max(max(foll_H), max(inter_H))
        ax.plot([1, 1, 2, 2], [y_max*1.02, y_max*1.05, y_max*1.05, y_max*1.02], 'k-', lw=1)
        ax.text(1.5, y_max*1.06, f'p={p_mw:.4f}', ha='center', fontsize=9)
    label_panel(ax, 'b')

    # (c) Representative core with domain overlay
    ax = axes[0, 2]
    # Compute per-ROI average compartment entropy for representative core selection
    roi_avg_comp_H = {}
    for roi in unique_rois:
        roi_m = (sample_ids == roi) & tumor_mask
        comps_in_roi = set(domains[roi_m])
        h_list = [domain_entropy[d] for d in comps_in_roi if d in domain_entropy]
        if h_list:
            roi_avg_comp_H[roi] = np.mean(h_list)
    if roi_avg_comp_H:
        plot_representative_core_spatial(ax, cx, cy, sample_ids, cell_types, tumor_mask,
                                          roi_avg_comp_H, label_lo='Homogeneous',
                                          label_hi='Diverse', metric_name='avg H',
                                          domains=domains, foll_domains=foll)
    label_panel(ax, 'c')

    # (d) Composition differences (interpretation)
    ax = axes[1, 0]
    top_comp = comp_data[:10]
    if top_comp:
        names = [d[0] for d in reversed(top_comp)]
        diffs = [d[3] for d in reversed(top_comp)]
        colors_bar = ['#e74c3c' if d > 0 else '#3498db' for d in diffs]
        ax.barh(range(len(names)), diffs, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Δ percentage points (foll − inter)')
        ax.set_title('Composition Differences\n(red = enriched in follicular)')
        ax.axvline(0, color='gray', lw=0.5)
    label_panel(ax, 'd')

    # (e) Sorted domain bar chart
    ax = axes[1, 1]
    from matplotlib.patches import Patch
    sorted_comps = sorted(global_H.items(), key=lambda x: x[1][0])
    comp_labels = [x[0][:30] for x in sorted_comps]
    comp_H_vals = [x[1][0] for x in sorted_comps]
    comp_colors = ['#e74c3c' if x[1][1] == 'FOLL' else '#3498db' if x[1][1] == 'INTER'
                   else '#95a5a6' for x in sorted_comps]
    ax.barh(range(len(comp_labels)), comp_H_vals, color=comp_colors, alpha=0.7)
    ax.set_yticks(range(len(comp_labels)))
    ax.set_yticklabels(comp_labels, fontsize=6)
    ax.set_xlabel('Shannon Entropy (bits)')
    ax.set_title(f'{panel}-panel: Per-Compartment (n={len(sorted_comps)})')
    ax.legend(handles=[Patch(color='#e74c3c', alpha=0.7, label='Follicular'),
                       Patch(color='#3498db', alpha=0.7, label='Interfollicular'),
                       Patch(color='#95a5a6', alpha=0.7, label='Excluded')],
              fontsize=7)
    label_panel(ax, 'e')

    # (f) hide unused
    axes[1, 2].axis('off')

    plt.suptitle(f'H6c: Compartment-Level Entropy — {panel}-panel (tumor only)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(output_dir, f'fig_h6c_{panel}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")


# ---------------------------------------------------------------------------
# H4a: FRC domain attenuation in follicle-effaced ROIs (S-panel)
# ---------------------------------------------------------------------------

def run_h4a(f_v8, f_utag, marker_idx, sample_ids, tma_arr, tumor_mask, cell_types,
            output_dir, cartoon_dir, cx, cy):
    """H4a: FRC (PDPN+) T-zone domains compressed in ROIs with expanded follicles.

    FL reprograms FRCs via TNF/TGF-beta (Mourcin 2021). As follicles expand,
    the T-zone (and its FRC scaffold) gets compressed. We test whether per-ROI
    follicularity anti-correlates with FRC/stromal abundance.

    PDPN is WEAK but FRC cell type (PDPN>1.5) is already annotated in v8.
    CCL21 is DEAD — cannot gate on it.
    """
    print(f"\n{'='*70}")
    print(f"H4a: FRC Domain Attenuation — S-panel")
    print(f"{'='*70}")

    # --- Classify domains ---
    domains, foll, inter = classify_domains(f_utag, cell_types, 'S')
    foll_mask = np.isin(domains, foll) & tumor_mask
    inter_mask = np.isin(domains, inter) & tumor_mask

    n_foll = np.sum(foll_mask)
    n_inter = np.sum(inter_mask)
    print(f"Follicular: {n_foll:,} | Interfollicular: {n_inter:,}")

    # --- Cell type sets ---
    frc_types = {'FRC (PDPN+)'}
    stromal_types = {'Stromal / CAF', 'FRC (PDPN+)'}
    lq_types = {'Low quality / Unassigned'}

    is_frc = np.isin(cell_types, list(frc_types)) & tumor_mask
    is_stromal = np.isin(cell_types, list(stromal_types)) & tumor_mask
    non_lq = tumor_mask & ~np.isin(cell_types, list(lq_types))

    n_frc = np.sum(is_frc)
    n_stromal = np.sum(is_stromal)
    print(f"FRC (PDPN+): {n_frc:,} | Stromal+FRC: {n_stromal:,}")

    # --- Per-ROI analysis ---
    roi_cell_counts = Counter(sample_ids[non_lq])
    unique_rois = sorted(set(sample_ids[tumor_mask]))
    min_cells = 5000  # S-panel Biomax uses lower threshold

    roi_follicularity = {}   # fraction of cells in follicular domains
    roi_frc_frac = {}        # FRC fraction of typed cells
    roi_stromal_frac = {}    # Stromal+FRC fraction
    roi_tma = {}

    for roi in unique_rois:
        if roi_cell_counts.get(roi, 0) < min_cells:
            continue

        rm = (sample_ids == roi) & tumor_mask
        n_typed = np.sum(non_lq & (sample_ids == roi))
        if n_typed < min_cells:
            continue

        # Follicularity
        n_foll_roi = np.sum(foll_mask & rm)
        n_inter_roi = np.sum(inter_mask & rm)
        n_domain = n_foll_roi + n_inter_roi
        if n_domain < 100:
            continue
        roi_follicularity[roi] = n_foll_roi / n_domain

        # FRC fraction
        n_frc_roi = np.sum(is_frc & rm)
        roi_frc_frac[roi] = n_frc_roi / n_typed

        # Stromal fraction (broader)
        n_strom_roi = np.sum(is_stromal & rm)
        roi_stromal_frac[roi] = n_strom_roi / n_typed

        roi_tma[roi] = tma_arr[rm][0]

    n_rois = len(roi_follicularity)
    print(f"\nROIs passing filters: {n_rois}")

    # --- Correlations ---
    common_rois = sorted(roi_follicularity.keys())
    foll_vals = np.array([roi_follicularity[r] for r in common_rois])
    frc_vals = np.array([roi_frc_frac[r] for r in common_rois])
    strom_vals = np.array([roi_stromal_frac[r] for r in common_rois])

    rho_frc, p_frc = spearmanr(foll_vals, frc_vals)
    rho_strom, p_strom = spearmanr(foll_vals, strom_vals)

    print(f"\n--- Pooled correlations ---")
    print(f"Follicularity vs FRC frac:     rho={rho_frc:+.3f}, p={p_frc:.2e}")
    print(f"Follicularity vs Stromal frac: rho={rho_strom:+.3f}, p={p_strom:.2e}")

    # --- FRC density by domain type (cell-level) ---
    # What fraction of cells are FRC in follicular vs interfollicular domains?
    frc_in_foll = np.sum(is_frc & foll_mask)
    frc_in_inter = np.sum(is_frc & inter_mask)
    typed_in_foll = np.sum(non_lq & foll_mask)
    typed_in_inter = np.sum(non_lq & inter_mask)

    pct_frc_foll = 100 * frc_in_foll / typed_in_foll if typed_in_foll > 0 else 0
    pct_frc_inter = 100 * frc_in_inter / typed_in_inter if typed_in_inter > 0 else 0

    from scipy.stats import chi2_contingency as chi2c
    ct_domain = np.array([[frc_in_foll, typed_in_foll - frc_in_foll],
                           [frc_in_inter, typed_in_inter - frc_in_inter]])
    chi2_val, p_domain, _, _ = chi2c(ct_domain)

    print(f"\n--- FRC by domain ---")
    print(f"Follicular:     {frc_in_foll:,}/{typed_in_foll:,} = {pct_frc_foll:.3f}%")
    print(f"Interfollicular: {frc_in_inter:,}/{typed_in_inter:,} = {pct_frc_inter:.3f}%")
    print(f"chi2={chi2_val:.1f}, p={p_domain:.2e}")

    # Per-ROI FRC fraction in follicular vs interfollicular (paired comparison)
    roi_frc_foll = {}
    roi_frc_inter = {}
    for roi in common_rois:
        rm = (sample_ids == roi) & tumor_mask
        n_frc_f = np.sum(is_frc & foll_mask & rm)
        n_frc_i = np.sum(is_frc & inter_mask & rm)
        n_typed_f = np.sum(non_lq & foll_mask & rm)
        n_typed_i = np.sum(non_lq & inter_mask & rm)
        if n_typed_f > 50:
            roi_frc_foll[roi] = 100 * n_frc_f / n_typed_f
        if n_typed_i > 50:
            roi_frc_inter[roi] = 100 * n_frc_i / n_typed_i

    paired_rois = sorted(set(roi_frc_foll.keys()) & set(roi_frc_inter.keys()))
    if len(paired_rois) > 5:
        foll_arr = np.array([roi_frc_foll[r] for r in paired_rois])
        inter_arr = np.array([roi_frc_inter[r] for r in paired_rois])
        from scipy.stats import wilcoxon
        try:
            W, p_wilc = wilcoxon(inter_arr, foll_arr, alternative='greater')
            print(f"\nPaired Wilcoxon (inter > foll): W={W:.0f}, p={p_wilc:.2e} (n={len(paired_rois)} ROIs)")
            print(f"  FRC% in foll: median={np.median(foll_arr):.4f}%, mean={np.mean(foll_arr):.4f}%")
            print(f"  FRC% in inter: median={np.median(inter_arr):.4f}%, mean={np.mean(inter_arr):.4f}%")
        except Exception as e:
            p_wilc = 1.0
            print(f"Wilcoxon failed: {e}")
    else:
        p_wilc = 1.0
        foll_arr = np.array([])
        inter_arr = np.array([])

    # --- Leave-one-TMA-out ---
    print(f"\n--- Leave-one-TMA-out ---")
    tmas = sorted(set(roi_tma.values()))
    loo_results = []
    for excl in tmas:
        rois_keep = [r for r in common_rois if roi_tma[r] != excl]
        if len(rois_keep) < 5:
            continue
        foll_k = np.array([roi_follicularity[r] for r in rois_keep])
        frc_k = np.array([roi_frc_frac[r] for r in rois_keep])
        strom_k = np.array([roi_stromal_frac[r] for r in rois_keep])
        rho_f, p_f = spearmanr(foll_k, frc_k)
        rho_s, p_s = spearmanr(foll_k, strom_k)
        print(f"  Excl {excl} (n={len(rois_keep)}): "
              f"Foll-FRC rho={rho_f:+.3f} p={p_f:.2e} | "
              f"Foll-Strom rho={rho_s:+.3f} p={p_s:.2e}")
        loo_results.append((excl, rho_f, p_f, rho_s, p_s))

    # --- Drivers: cell type correlations with follicularity ---
    roi_fracs = compute_roi_celltype_fractions(sample_ids, cell_types, tumor_mask, lq_types)
    drivers = correlate_celltype_with_metric(roi_fracs, roi_follicularity)
    print(f"\nFollicularity drivers (Spearman ρ with follicular fraction):")
    for ct_name, rho_d, p_d in drivers[:12]:
        print(f"  {ct_name:40s}  ρ={rho_d:+.3f}  p={p_d:.2e}")

    # ===================================================================
    # FIGURE: 2 rows × 3 cols
    # ===================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    from matplotlib.lines import Line2D

    # (a) Concept cartoon
    add_cartoon(axes[0, 0], os.path.join(cartoon_dir, 'h4a_frc_attenuation.png'))
    label_panel(axes[0, 0], 'a')

    # (b) Per-ROI scatter: follicularity vs FRC fraction
    ax = axes[0, 1]
    tma_colors = {'A1': '#3498db', 'B1': '#e74c3c', 'C1': '#2ecc71', 'Biomax': '#f39c12'}
    for roi in common_rois:
        c = tma_colors.get(roi_tma[roi], '#999999')
        ax.scatter(roi_follicularity[roi] * 100, roi_frc_frac[roi] * 100, c=c, s=30,
                   alpha=0.6, edgecolors='white', linewidths=0.3)
    z = np.polyfit(foll_vals * 100, frc_vals * 100, 1)
    x_line = np.linspace((foll_vals * 100).min(), (foll_vals * 100).max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Follicularity (% cells in follicular domains)')
    ax.set_ylabel('FRC (PDPN+) % of typed cells')
    sig_frc = '***' if p_frc < 0.001 else '**' if p_frc < 0.01 else '*' if p_frc < 0.05 else 'n.s.'
    ax.set_title(f'Follicularity vs FRC Fraction\nρ={rho_frc:+.3f}, p={p_frc:.2e} {sig_frc}')
    tma_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=7, label=t) for t, c in tma_colors.items()]
    ax.legend(handles=tma_handles, fontsize=7, loc='best')
    label_panel(ax, 'b')

    # (c) Per-ROI scatter: follicularity vs Stromal+FRC fraction
    ax = axes[0, 2]
    for roi in common_rois:
        c = tma_colors.get(roi_tma[roi], '#999999')
        ax.scatter(roi_follicularity[roi] * 100, roi_stromal_frac[roi] * 100, c=c, s=30,
                   alpha=0.6, edgecolors='white', linewidths=0.3)
    z2 = np.polyfit(foll_vals * 100, strom_vals * 100, 1)
    ax.plot(x_line, np.polyval(z2, x_line), 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Follicularity (% cells in follicular domains)')
    ax.set_ylabel('Stromal (CAF+FRC) % of typed cells')
    sig_strom = '***' if p_strom < 0.001 else '**' if p_strom < 0.01 else '*' if p_strom < 0.05 else 'n.s.'
    ax.set_title(f'Follicularity vs Stromal Fraction\nρ={rho_strom:+.3f}, p={p_strom:.2e} {sig_strom}')
    label_panel(ax, 'c')

    # (d) Representative spatial plots (low vs high follicularity, FRCs highlighted)
    ax = axes[1, 0]
    plot_representative_core_spatial(
        ax, cx, cy, sample_ids, cell_types, tumor_mask,
        roi_follicularity, label_lo='Low Foll.', label_hi='High Foll.',
        metric_name='Foll%', top_n_types=10,
        domains=domains, foll_domains=foll,
        highlight_mask=is_frc, highlight_label='FRC (PDPN+)',
        highlight_color='#FF69B4', min_cells=min_cells,
        highlight_size=25, min_highlight_hi=3)
    label_panel(ax, 'd')

    # (e) FRC % in follicular vs interfollicular domains (paired per-ROI boxplot)
    ax = axes[1, 1]
    if len(paired_rois) > 5:
        bp = ax.boxplot([foll_arr, inter_arr],
                        tick_labels=['Follicular', 'Interfollicular'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#FFDDDD')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#DDDDFF')
        bp['boxes'][1].set_alpha(0.7)
        ax.set_ylabel('FRC (PDPN+) % of typed cells')
        sig_wilc = '***' if p_wilc < 0.001 else '**' if p_wilc < 0.01 else '*' if p_wilc < 0.05 else 'n.s.'
        ax.set_title(f'FRC Density by Domain\nWilcoxon p={p_wilc:.2e} {sig_wilc}\n(n={len(paired_rois)} paired ROIs)')
        # Significance bracket
        ymax = max(np.percentile(foll_arr, 95), np.percentile(inter_arr, 95))
        ax.plot([1, 1, 2, 2], [ymax*1.1, ymax*1.2, ymax*1.2, ymax*1.1], 'k-', lw=1)
        ax.text(1.5, ymax*1.25, sig_wilc, ha='center', fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Insufficient paired ROIs', ha='center', va='center',
                transform=ax.transAxes)
    label_panel(ax, 'e')

    # (f) Driver analysis
    ax = axes[1, 2]
    top_n = min(12, len(drivers))
    if top_n > 0:
        top = drivers[:top_n]
        names = [d[0] for d in reversed(top)]
        rhos = [d[1] for d in reversed(top)]
        colors_bar = ['#e74c3c' if r < 0 else '#2ecc71' for r in rhos]
        ax.barh(range(len(names)), rhos, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Spearman ρ')
        ax.set_title('Follicularity Drivers\n(cell type ρ with follicular fraction)')
        ax.axvline(0, color='gray', lw=0.5)
    label_panel(ax, 'f')

    plt.suptitle('H4a: FRC Domain Attenuation in Follicle-Effaced Tissue — S-panel\n'
                 '(all tumor ROIs)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = os.path.join(output_dir, 'fig_h4a_S.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # --- Supplementary: Leave-one-TMA-out ---
    if loo_results:
        fig_s, axes_s = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes_s[0]
        excl_names = [r[0] for r in loo_results]
        rhos_loo = [r[1] for r in loo_results]
        colors_loo = [tma_colors.get(t, '#999') for t in excl_names]
        ax.bar(excl_names, rhos_loo, color=colors_loo, alpha=0.7)
        ax.axhline(rho_frc, color='black', linestyle='--', lw=1, label=f'Pooled ρ={rho_frc:+.3f}')
        ax.set_ylabel('Spearman ρ (Foll. vs FRC frac)')
        ax.set_xlabel('Excluded TMA')
        ax.set_title('Leave-One-Out: Follicularity vs FRC')
        ax.legend(fontsize=8)

        ax = axes_s[1]
        rhos_loo_s = [r[3] for r in loo_results]
        ax.bar(excl_names, rhos_loo_s, color=colors_loo, alpha=0.7)
        ax.axhline(rho_strom, color='black', linestyle='--', lw=1, label=f'Pooled ρ={rho_strom:+.3f}')
        ax.set_ylabel('Spearman ρ (Foll. vs Stromal frac)')
        ax.set_xlabel('Excluded TMA')
        ax.set_title('Leave-One-Out: Follicularity vs Stromal')
        ax.legend(fontsize=8)

        plt.suptitle('H4a Supplementary: Leave-One-TMA-Out Sensitivity', fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig_s_path = os.path.join(output_dir, 'fig_h4a_S_supp.png')
        plt.savefig(fig_s_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Supplementary saved: {fig_s_path}")

    return {
        'rho_frc': rho_frc, 'p_frc': p_frc,
        'rho_strom': rho_strom, 'p_strom': p_strom,
        'p_wilcoxon': p_wilc, 'n_rois': n_rois,
        'pct_frc_foll': pct_frc_foll, 'pct_frc_inter': pct_frc_inter,
    }


# ---------------------------------------------------------------------------
# H5b: CD47-high tumor B cells vs macrophage proximity
# ---------------------------------------------------------------------------

def run_h5b(f_v8, f_utag, marker_idx, sample_ids, tma_arr, tumor_mask, cell_types,
            output_dir, cartoon_dir, cx, cy):
    """H5b: CD47-high tumor regions have reduced CD68+ macrophage proximity.

    CD47 is the 'don't eat me' signal — tumor B cells upregulate it to evade
    phagocytosis by macrophages. We test whether per-ROI CD47 expression on B
    cells anti-correlates with macrophage density and proximity.

    CD47 is WEAK in A1/B1/C1 (p99 0.27-0.45) and DEAD in Biomax → exclude Biomax.
    """
    from scipy.spatial import KDTree

    print(f"\n{'='*70}")
    print(f"H5b: CD47 'Don't Eat Me' vs Macrophage Proximity — T-panel")
    print(f"{'='*70}")

    # --- Exclude Biomax (CD47 dead there) ---
    biomax_mask = np.array([t == 'Biomax' for t in tma_arr])
    h5b_mask = tumor_mask & ~biomax_mask
    n_excluded_biomax = np.sum(tumor_mask & biomax_mask)
    print(f"Excluding Biomax ({n_excluded_biomax:,} cells) — CD47 dead in Biomax")
    print(f"Working set: {np.sum(h5b_mask):,} tumor cells from A1/B1/C1")

    # --- Identify B cells and macrophages by cell type ---
    b_types = {'B cells', 'B cells (CD20hi)', 'B cells (CXCR5hi)', 'B cells (TOXhi)',
               'B cells (weak CD20)', 'GC B cells', 'Activated B / Plasmablast'}
    mac_types = {'Macrophages'}

    is_b = np.isin(cell_types, list(b_types)) & h5b_mask
    is_mac = np.isin(cell_types, list(mac_types)) & h5b_mask

    n_b = np.sum(is_b)
    n_mac = np.sum(is_mac)
    print(f"B cells: {n_b:,} | Macrophages: {n_mac:,}")

    # --- Load CD47 marker ---
    cd47 = load_marker(f_v8, 'CD47', marker_idx)
    if cd47 is None:
        print("ERROR: CD47 marker not found!")
        return
    print(f"CD47 on B cells: mean={np.mean(cd47[is_b]):.3f}, "
          f"std={np.std(cd47[is_b]):.3f}, "
          f"p50={np.median(cd47[is_b]):.3f}, "
          f"p90={np.percentile(cd47[is_b], 90):.3f}")

    # --- Classify domains ---
    domains, foll, inter = classify_domains(f_utag, cell_types, 'T')
    foll_mask = np.isin(domains, foll) & h5b_mask
    inter_mask = np.isin(domains, inter) & h5b_mask

    # --- Per-ROI analysis ---
    lq_types = {'Low quality / Unassigned'}
    non_lq = h5b_mask & ~np.isin(cell_types, list(lq_types))
    roi_cell_counts = Counter(sample_ids[non_lq])

    unique_rois = sorted(set(sample_ids[h5b_mask]))
    min_cells_default = 8000

    roi_cd47_mean = {}      # mean CD47 on B cells
    roi_mac_frac = {}       # macrophage fraction among typed cells
    roi_mean_b2mac = {}     # mean nearest-neighbor distance B→Mac
    roi_median_b2mac = {}   # median nearest-neighbor distance B→Mac
    roi_tma = {}

    for roi in unique_rois:
        # Min cell filter
        if roi_cell_counts.get(roi, 0) < min_cells_default:
            continue

        rm = (sample_ids == roi) & h5b_mask
        b_roi = is_b & rm
        mac_roi = is_mac & rm
        n_b_roi = np.sum(b_roi)
        n_mac_roi = np.sum(mac_roi)

        if n_b_roi < 20 or n_mac_roi < 3:
            continue

        # Mean CD47 on B cells
        roi_cd47_mean[roi] = float(np.mean(cd47[b_roi]))

        # Macrophage fraction
        typed_roi = non_lq & rm
        n_typed = np.sum(typed_roi)
        roi_mac_frac[roi] = n_mac_roi / n_typed if n_typed > 0 else 0.0

        # Nearest-neighbor distance B→Mac using KDTree
        mac_coords = np.column_stack([cx[mac_roi], cy[mac_roi]])
        b_coords = np.column_stack([cx[b_roi], cy[b_roi]])
        tree = KDTree(mac_coords)
        dists, _ = tree.query(b_coords, k=1)
        roi_mean_b2mac[roi] = float(np.mean(dists))
        roi_median_b2mac[roi] = float(np.median(dists))

        roi_tma[roi] = tma_arr[rm][0]

    n_rois = len(roi_cd47_mean)
    print(f"\nROIs with sufficient B cells + macrophages: {n_rois}")

    # --- Correlations ---
    common_rois = sorted(roi_cd47_mean.keys())
    cd47_vals = np.array([roi_cd47_mean[r] for r in common_rois])
    mac_vals = np.array([roi_mac_frac[r] for r in common_rois])
    dist_vals = np.array([roi_mean_b2mac[r] for r in common_rois])

    rho_mac, p_mac = spearmanr(cd47_vals, mac_vals)
    rho_dist, p_dist = spearmanr(cd47_vals, dist_vals)

    print(f"\n--- Pooled correlations ---")
    print(f"CD47_mean vs Mac fraction:  rho={rho_mac:+.3f}, p={p_mac:.2e}")
    print(f"CD47_mean vs B→Mac dist:    rho={rho_dist:+.3f}, p={p_dist:.2e}")

    # --- Cell-level: CD47-high vs CD47-low B cells ---
    cd47_b = cd47[is_b]
    q25, q75 = np.percentile(cd47_b, [25, 75])
    cd47_lo_mask_b = cd47_b <= q25
    cd47_hi_mask_b = cd47_b >= q75

    # For each B cell, compute nearest Mac distance (across all ROIs at once
    # would mix ROIs — need per-ROI). Do it ROI by ROI.
    b_indices = np.where(is_b)[0]
    mac_indices = np.where(is_mac)[0]
    b_sample = sample_ids[is_b]
    b_nn_dist = np.full(n_b, np.nan)

    # Group B and Mac indices by ROI for efficiency
    from collections import defaultdict
    b_by_roi = defaultdict(list)
    mac_by_roi = defaultdict(list)
    for i, idx in enumerate(b_indices):
        b_by_roi[sample_ids[idx]].append(i)
    for idx in mac_indices:
        mac_by_roi[sample_ids[idx]].append(idx)

    for roi in set(b_sample):
        b_local = b_by_roi.get(roi, [])
        mac_local = mac_by_roi.get(roi, [])
        if len(b_local) == 0 or len(mac_local) < 2:
            continue
        mac_xy = np.column_stack([cx[mac_local], cy[mac_local]])
        tree = KDTree(mac_xy)
        b_local_indices = b_indices[b_local]
        b_xy = np.column_stack([cx[b_local_indices], cy[b_local_indices]])
        d, _ = tree.query(b_xy, k=1)
        for j, bi in enumerate(b_local):
            b_nn_dist[bi] = d[j]

    valid = ~np.isnan(b_nn_dist)
    lo_dists = b_nn_dist[valid & cd47_lo_mask_b[valid] if False else (valid & cd47_lo_mask_b)]
    hi_dists = b_nn_dist[valid & cd47_hi_mask_b]
    lo_dists = b_nn_dist[cd47_lo_mask_b & valid]
    hi_dists = b_nn_dist[cd47_hi_mask_b & valid]

    if len(lo_dists) > 0 and len(hi_dists) > 0:
        U_cell, p_cell = mannwhitneyu(hi_dists, lo_dists, alternative='greater')
        print(f"\n--- Cell-level: nearest Mac distance ---")
        print(f"CD47-low  (Q1, n={len(lo_dists):,}): mean={np.mean(lo_dists):.1f}, "
              f"median={np.median(lo_dists):.1f}")
        print(f"CD47-high (Q4, n={len(hi_dists):,}): mean={np.mean(hi_dists):.1f}, "
              f"median={np.median(hi_dists):.1f}")
        print(f"Mann-Whitney (greater): U={U_cell:.0f}, p={p_cell:.2e}")
    else:
        p_cell = 1.0
        print("WARNING: insufficient cells for cell-level test")

    # --- Domain-stratified analysis ---
    print(f"\n--- Domain-stratified (follicular vs interfollicular) ---")
    for domain_label, domain_mask in [('Follicular', foll_mask), ('Interfoll.', inter_mask)]:
        b_dom = is_b & domain_mask
        mac_dom = is_mac & domain_mask
        n_b_d = np.sum(b_dom)
        n_mac_d = np.sum(mac_dom)
        if n_b_d < 50 or n_mac_d < 10:
            print(f"  {domain_label}: too few cells (B={n_b_d}, Mac={n_mac_d})")
            continue
        # Per-ROI within domain
        dom_rois = sorted(set(sample_ids[b_dom]))
        cd47_d, mac_d = [], []
        for roi in dom_rois:
            rm_b = b_dom & (sample_ids == roi)
            rm_mac = mac_dom & (sample_ids == roi)
            n_b_r = np.sum(rm_b)
            n_mac_r = np.sum(rm_mac)
            if n_b_r < 10:
                continue
            cd47_d.append(float(np.mean(cd47[rm_b])))
            typed_r = non_lq & domain_mask & (sample_ids == roi)
            n_typed_r = np.sum(typed_r)
            mac_d.append(n_mac_r / n_typed_r if n_typed_r > 0 else 0.0)
        if len(cd47_d) > 5:
            rho_d, p_d = spearmanr(cd47_d, mac_d)
            print(f"  {domain_label} (n={len(cd47_d)} ROIs): CD47 vs Mac frac "
                  f"rho={rho_d:+.3f}, p={p_d:.2e}")

    # --- Leave-one-TMA-out ---
    print(f"\n--- Leave-one-TMA-out ---")
    tmas = sorted(set(roi_tma.values()))
    loo_results = []
    for excl in tmas:
        rois_keep = [r for r in common_rois if roi_tma[r] != excl]
        if len(rois_keep) < 5:
            continue
        cd47_k = np.array([roi_cd47_mean[r] for r in rois_keep])
        mac_k = np.array([roi_mac_frac[r] for r in rois_keep])
        dist_k = np.array([roi_mean_b2mac[r] for r in rois_keep])
        rho_m, p_m = spearmanr(cd47_k, mac_k)
        rho_d2, p_d2 = spearmanr(cd47_k, dist_k)
        print(f"  Excl {excl} (n={len(rois_keep)}): "
              f"CD47-Mac rho={rho_m:+.3f} p={p_m:.2e} | "
              f"CD47-Dist rho={rho_d2:+.3f} p={p_d2:.2e}")
        loo_results.append((excl, rho_m, p_m, rho_d2, p_d2))

    # --- Drivers: cell type correlations with CD47 mean ---
    roi_fracs = compute_roi_celltype_fractions(sample_ids, cell_types, h5b_mask, lq_types)
    drivers = correlate_celltype_with_metric(roi_fracs, roi_cd47_mean)
    print(f"\nCD47 drivers (Spearman ρ with per-ROI mean CD47 on B cells):")
    for ct_name, rho_d, p_d in drivers[:12]:
        print(f"  {ct_name:40s}  ρ={rho_d:+.3f}  p={p_d:.2e}")

    # ===================================================================
    # FIGURE: 2 rows × 3 cols
    # ===================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Concept cartoon
    add_cartoon(axes[0, 0], os.path.join(cartoon_dir, 'h5b_cd47_macrophage.png'))
    label_panel(axes[0, 0], 'a')

    # (b) Per-ROI scatter: CD47 vs macrophage fraction
    ax = axes[0, 1]
    tma_colors = {'A1': '#3498db', 'B1': '#e74c3c', 'C1': '#2ecc71'}
    for roi in common_rois:
        c = tma_colors.get(roi_tma[roi], '#999999')
        ax.scatter(roi_cd47_mean[roi], roi_mac_frac[roi] * 100, c=c, s=30,
                   alpha=0.6, edgecolors='white', linewidths=0.3)
    # Regression line
    z = np.polyfit(cd47_vals, mac_vals * 100, 1)
    x_line = np.linspace(cd47_vals.min(), cd47_vals.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Mean CD47 on B cells (z-scored)')
    ax.set_ylabel('Macrophage % of typed cells')
    sig_mac = '***' if p_mac < 0.001 else '**' if p_mac < 0.01 else '*' if p_mac < 0.05 else 'n.s.'
    ax.set_title(f'CD47 vs Macrophage Density\nρ={rho_mac:+.3f}, p={p_mac:.2e} {sig_mac}')
    # TMA legend
    from matplotlib.lines import Line2D
    tma_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=7, label=t) for t, c in tma_colors.items()]
    ax.legend(handles=tma_handles, fontsize=7, loc='best')
    label_panel(ax, 'b')

    # (c) Per-ROI scatter: CD47 vs B→Mac distance
    ax = axes[0, 2]
    for roi in common_rois:
        c = tma_colors.get(roi_tma[roi], '#999999')
        ax.scatter(roi_cd47_mean[roi], roi_mean_b2mac[roi], c=c, s=30,
                   alpha=0.6, edgecolors='white', linewidths=0.3)
    z2 = np.polyfit(cd47_vals, dist_vals, 1)
    ax.plot(x_line, np.polyval(z2, x_line), 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Mean CD47 on B cells (z-scored)')
    ax.set_ylabel('Mean B→nearest Mac distance (px)')
    sig_dist = '***' if p_dist < 0.001 else '**' if p_dist < 0.01 else '*' if p_dist < 0.05 else 'n.s.'
    ax.set_title(f'CD47 vs Mac Proximity\nρ={rho_dist:+.3f}, p={p_dist:.2e} {sig_dist}')
    label_panel(ax, 'c')

    # (d) Representative spatial plots (low vs high CD47 ROI)
    ax = axes[1, 0]
    plot_representative_core_spatial(
        ax, cx, cy, sample_ids, cell_types, h5b_mask,
        roi_cd47_mean, label_lo='Low CD47', label_hi='High CD47',
        metric_name='CD47', top_n_types=10,
        domains=domains, foll_domains=foll,
        highlight_mask=is_mac, highlight_label='Macrophages (CD68+)',
        highlight_color='#2ecc71', min_cells=min_cells_default,
        highlight_size=25, min_highlight_hi=5)
    label_panel(ax, 'd')

    # (e) Cell-level violin: nearest Mac distance for CD47-high vs CD47-low
    ax = axes[1, 1]
    if len(lo_dists) > 0 and len(hi_dists) > 0:
        # Subsample for plotting if very large
        max_plot = 50000
        lo_plot = lo_dists if len(lo_dists) <= max_plot else np.random.choice(lo_dists, max_plot, replace=False)
        hi_plot = hi_dists if len(hi_dists) <= max_plot else np.random.choice(hi_dists, max_plot, replace=False)

        vp = ax.violinplot([lo_plot, hi_plot], positions=[0, 1], showmedians=True, showextrema=False)
        colors_v = ['#3498db', '#e74c3c']
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors_v[i])
            body.set_alpha(0.7)
        vp['cmedians'].set_color('black')

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'CD47-low\n(Q1, n={len(lo_dists):,})',
                            f'CD47-high\n(Q4, n={len(hi_dists):,})'])
        ax.set_ylabel('Distance to nearest macrophage (px)')
        sig_cell = '***' if p_cell < 0.001 else '**' if p_cell < 0.01 else '*' if p_cell < 0.05 else 'n.s.'
        ax.set_title(f'Cell-Level: B→Mac Distance\np={p_cell:.2e} {sig_cell}')

        # Significance bracket
        ymax = max(np.percentile(lo_plot, 95), np.percentile(hi_plot, 95))
        ax.plot([0, 0, 1, 1], [ymax*1.05, ymax*1.1, ymax*1.1, ymax*1.05], 'k-', lw=1)
        ax.text(0.5, ymax*1.12, sig_cell, ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(top=ymax * 1.25)
    label_panel(ax, 'e')

    # (f) Driver analysis
    ax = axes[1, 2]
    top_n = min(12, len(drivers))
    if top_n > 0:
        top = drivers[:top_n]
        names = [d[0] for d in reversed(top)]
        rhos = [d[1] for d in reversed(top)]
        colors_bar = ['#e74c3c' if r < 0 else '#2ecc71' for r in rhos]
        ax.barh(range(len(names)), rhos, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Spearman ρ')
        ax.set_title('CD47 Drivers\n(cell type ρ with mean CD47)')
        ax.axvline(0, color='gray', lw=0.5)
    label_panel(ax, 'f')

    plt.suptitle('H5b: CD47 "Don\'t Eat Me" Signal vs Macrophage Proximity — T-panel\n'
                 '(Biomax excluded, A1/B1/C1 tumor only)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = os.path.join(output_dir, 'fig_h5b_T.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # --- Supplementary: Leave-one-TMA-out ---
    if loo_results:
        fig_s, axes_s = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes_s[0]
        excl_names = [r[0] for r in loo_results]
        rhos_loo = [r[1] for r in loo_results]
        pvals_loo = [r[2] for r in loo_results]
        colors_loo = [tma_colors.get(t, '#999') for t in excl_names]
        ax.bar(excl_names, rhos_loo, color=colors_loo, alpha=0.7)
        ax.axhline(rho_mac, color='black', linestyle='--', lw=1, label=f'Pooled ρ={rho_mac:+.3f}')
        ax.set_ylabel('Spearman ρ (CD47 vs Mac frac)')
        ax.set_xlabel('Excluded TMA')
        ax.set_title('Leave-One-Out: CD47 vs Mac Density')
        ax.legend(fontsize=8)

        ax = axes_s[1]
        rhos_loo_d = [r[3] for r in loo_results]
        ax.bar(excl_names, rhos_loo_d, color=colors_loo, alpha=0.7)
        ax.axhline(rho_dist, color='black', linestyle='--', lw=1, label=f'Pooled ρ={rho_dist:+.3f}')
        ax.set_ylabel('Spearman ρ (CD47 vs B→Mac dist)')
        ax.set_xlabel('Excluded TMA')
        ax.set_title('Leave-One-Out: CD47 vs Mac Distance')
        ax.legend(fontsize=8)

        plt.suptitle('H5b Supplementary: Leave-One-TMA-Out Sensitivity', fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig_s_path = os.path.join(output_dir, 'fig_h5b_T_supp.png')
        plt.savefig(fig_s_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Supplementary saved: {fig_s_path}")

    return {
        'rho_mac': rho_mac, 'p_mac': p_mac,
        'rho_dist': rho_dist, 'p_dist': p_dist,
        'p_cell': p_cell, 'n_rois': n_rois,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t-panel', required=True)
    parser.add_argument('--s-panel', required=True)
    parser.add_argument('--t-utag', required=True)
    parser.add_argument('--s-utag', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cartoon-dir', default='output/hypothesis_cartoons')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ========== T-PANEL ==========
    print("\n" + "#" * 70)
    print("# T-PANEL")
    print("#" * 70)

    f_t = h5py.File(args.t_panel, 'r')
    f_t_utag = h5py.File(args.t_utag, 'r')
    marker_idx_t = get_marker_idx(f_t)

    sample_t = load_array(f_t, 'sample_id')
    tma_t = load_array(f_t, 'tma')
    ct_t = load_array(f_t, 'cell_type')
    tumor_t = get_tumor_mask(sample_t)

    n_total = len(sample_t)
    n_tumor = np.sum(tumor_t)
    n_control = n_total - n_tumor
    print(f"\nT-panel: {n_total:,} total → {n_tumor:,} tumor + {n_control:,} control (excluded)")

    cx_t = f_t['obs']['centroid_x'][:]
    cy_t = f_t['obs']['centroid_y'][:]

    run_h6a(f_t, 'T', sample_t, tma_t, tumor_t, args.output_dir, args.cartoon_dir, cx_t, cy_t)
    run_h2a_h2e(f_t, f_t_utag, marker_idx_t, 'T', sample_t, tma_t, tumor_t, ct_t,
                args.output_dir, args.cartoon_dir, cx_t, cy_t)
    run_h2b(f_t, f_t_utag, marker_idx_t, 'T', sample_t, tma_t, tumor_t, ct_t,
            args.output_dir, args.cartoon_dir, cx_t, cy_t)
    run_h6c(f_t_utag, ct_t, tma_t, tumor_t, 'T', args.output_dir, args.cartoon_dir,
            cx_t, cy_t, sample_t)
    run_h5b(f_t, f_t_utag, marker_idx_t, sample_t, tma_t, tumor_t, ct_t,
            args.output_dir, args.cartoon_dir, cx_t, cy_t)

    f_t.close()
    f_t_utag.close()

    # ========== S-PANEL ==========
    print("\n" + "#" * 70)
    print("# S-PANEL")
    print("#" * 70)

    f_s = h5py.File(args.s_panel, 'r')
    f_s_utag = h5py.File(args.s_utag, 'r')
    marker_idx_s = get_marker_idx(f_s)

    sample_s = load_array(f_s, 'sample_id')
    tma_s = load_array(f_s, 'tma')
    ct_s = load_array(f_s, 'cell_type')
    tumor_s = get_tumor_mask(sample_s)

    n_total_s = len(sample_s)
    n_tumor_s = np.sum(tumor_s)
    n_control_s = n_total_s - n_tumor_s
    print(f"\nS-panel: {n_total_s:,} total → {n_tumor_s:,} tumor + {n_control_s:,} control (excluded)")

    cx_s = f_s['obs']['centroid_x'][:]
    cy_s = f_s['obs']['centroid_y'][:]

    run_h6a(f_s, 'S', sample_s, tma_s, tumor_s, args.output_dir, args.cartoon_dir, cx_s, cy_s)
    run_h3a(f_s, f_s_utag, marker_idx_s, sample_s, tma_s, tumor_s, ct_s,
            args.output_dir, args.cartoon_dir, cx_s, cy_s)
    run_h6c(f_s_utag, ct_s, tma_s, tumor_s, 'S', args.output_dir, args.cartoon_dir,
            cx_s, cy_s, sample_s)
    run_h4a(f_s, f_s_utag, marker_idx_s, sample_s, tma_s, tumor_s, ct_s,
            args.output_dir, args.cartoon_dir, cx_s, cy_s)

    f_s.close()
    f_s_utag.close()

    print(f"\n{'#'*70}")
    print(f"# ALL DONE — results in {args.output_dir}")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
