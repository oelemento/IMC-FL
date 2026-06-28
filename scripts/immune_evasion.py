#!/usr/bin/env python3
"""Immune evasion spatial analysis for follicular lymphoma TME.

Generates 4 multi-panel figures exploring immune evasion mechanisms:
  1. Immune exclusion gradient across follicle→T zone axis (T-panel)
  2. Regulatory barrier & CD39 suppression (T-panel)
  3. Immune pressure landscape — effector:suppressor balance (T-panel)
  4. Myeloid suppression topology (S-panel)

Each figure includes a cartoon/schematic panel explaining the concept.

Usage:
    .venv/bin/python scripts/immune_evasion.py \
        --t-utag output/all_TMA_T_utag_ct_merged.h5ad \
        --s-utag output/all_TMA_S_utag_ct_merged.h5ad \
        --output-dir output/hypotheses_v8
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_array(f, key):
    ds = f['obs'][key]
    if isinstance(ds, h5py.Group) and 'categories' in ds:
        cats = ds['categories'][:]
        codes = ds['codes'][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c)
                             for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v)
                     for v in vals])


def get_marker_names(f):
    key = '_index' if '_index' in f['var'] else 'index'
    names = f['var'][key][:]
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def get_tumor_mask(sample_ids):
    control_tags = ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal',
                    '_ton_', '_adr_']
    return np.array([not any(t in s.lower() for t in control_tags)
                     for s in sample_ids])


DISPLAY_RENAME = {
    'Low quality / Unassigned': 'Unassigned',
    'B cells': 'Other B cells',
    'LQ / B transitional': 'B / Unassigned transitional',
    'Cytotoxic / LQ niche': 'Cytotoxic niche',
    'Weak CD20 / LQ border': 'Weak CD20 border',
}

def rename_labels(arr):
    return np.array([DISPLAY_RENAME.get(v, v) for v in arr])


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# T-panel compartments ordered follicle center → T zone
GRADIENT_ORDER = [
    'GC core',
    'Follicle core (GC/CD20hi/CXCR5hi)',
    'Follicle mantle (CXCR5hi)',
    'B cell follicle (CD20hi/CXCR5hi)',
    'B cell zone',
    'Follicle-T zone interface',
    'Treg-enriched T zone',
    'T cell zone (CD4/CD8)',
    'Macrophage-rich zone',
]

GRADIENT_SHORT = [
    'GC\ncore', 'Follicle\ncore', 'Follicle\nmantle',
    'B cell\nfollicle', 'B cell\nzone', 'Foll-T\ninterface',
    'Treg\nT zone', 'T cell\nzone', 'Mac\nzone',
]

GRADIENT_COLORS = [
    '#B22222', '#DC143C', '#E8734A', '#E06060', '#DAA520',
    '#6495ED', '#20B2AA', '#4169E1', '#191970',
]

# Follicular vs interfollicular boundary index (for divider line)
FOLL_INTER_BOUNDARY = 5  # interface is first interfollicular

# Cell type groups (use original h5ad names, rename_labels applied separately)
B_TYPES = ['GC B cells', 'B cells (CD20hi)', 'B cells (CXCR5hi)',
           'Other B cells', 'B cells (TOXhi)', 'Activated B / Plasmablast',
           'B cells (weak CD20)']
CD8_ALL = ['CD8 T cells', 'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)']
CD8_EXHAUSTED = ['CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)']
CD8_EFFECTOR = ['CD8 T cells', 'Macrophages (GzmB+)']
SUPPRESSOR = ['Treg', 'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)']
EFFECTOR = ['CD8 T cells', 'Macrophages (GzmB+)']

# S-panel compartments
S_GRADIENT_ORDER = [
    'B cell zone (BCL2+)', 'B cell zone (PAX5+)',
    'FDC network zone', 'FDC / myeloid zone',
    'T cell zone', 'Stromal / CAF zone',
]
S_GRADIENT_SHORT = [
    'B zone\n(BCL2+)', 'B zone\n(PAX5+)', 'FDC\nnetwork',
    'FDC/\nmyeloid', 'T cell\nzone', 'Stromal\n/ CAF',
]
S_GRADIENT_COLORS = [
    '#B22222', '#DC143C', '#E8734A', '#FF8C00',
    '#4169E1', '#20B2AA',
]
S_MYELOID_TYPES = ['M1 Macrophages', 'M2 Macrophages', 'Macrophages',
                   'Myeloid (S100A9+)', 'Dendritic cells', 'pDC']


# ═══════════════════════════════════════════════════════════════════════════
# Cartoon drawing functions
# ═══════════════════════════════════════════════════════════════════════════

def _add_bg(ax, xlim=(-1.6, 1.6), ylim=(-1.8, 1.4)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    bg = Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                    fc='#E8F0FE', ec='none', zorder=0)
    ax.add_patch(bg)


def draw_gradient_cartoon(ax):
    """Follicle cross-section showing all 9 UTAG compartments as concentric rings."""
    ax.set_xlim(-1.6, 3.2)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')
    bg = Rectangle((-1.6, -1.8), 4.8, 3.6, fc='#F0F4FF', ec='none', zorder=0)
    ax.add_patch(bg)

    # 9 zones: (outer_radius, color, short_label) — draw outermost first
    zones = [
        (1.35, '#191970', 'Macrophage-rich zone'),
        (1.20, '#4169E1', 'T cell zone'),
        (1.05, '#20B2AA', 'Treg-enriched zone'),
        (0.90, '#6495ED', 'Follicle-T interface'),
        (0.75, '#DAA520', 'B cell zone'),
        (0.60, '#E8734A', 'B cell follicle'),
        (0.45, '#E06060', 'Follicle mantle'),
        (0.30, '#CC2222', 'Follicle core'),
        (0.15, '#8B0000', 'GC core'),
    ]
    step = 0.15
    for r, color, name in zones:
        ax.add_patch(Circle((0, 0), r, fc=color, ec='white', lw=0.6, zorder=1))

    # Fan-out labels on the right: evenly spaced y, lines connecting to ring midpoint
    label_x = 1.65
    n = len(zones)
    y_labels = np.linspace(1.3, -1.3, n)
    for i, (r_out, color, name) in enumerate(zones):
        mid_r = r_out - step / 2
        y_lab = y_labels[i]
        # Direction from origin to (label_x, y_lab) — intersection with ring
        d = np.array([label_x, y_lab])
        d_norm = d / np.linalg.norm(d)
        x_conn, y_conn = mid_r * d_norm
        ax.plot([x_conn, label_x - 0.08], [y_conn, y_lab],
                '-', color=color, lw=0.9, alpha=0.85, zorder=3)
        ax.text(label_x, y_lab, name, fontsize=8.5, va='center',
                color=color, fontweight='bold', zorder=4)

    # Gradient arrow at bottom
    ax.annotate('', xy=(1.3, -1.6), xytext=(-1.3, -1.6),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.5))
    ax.text(0, -1.75, 'Increasing CD8 T cell density', ha='center',
            fontsize=9, style='italic', color='#444')


def draw_treg_cartoon(ax):
    """Treg barrier / regulatory shield cartoon."""
    _add_bg(ax)
    ax.set_title('(a) Concept: Treg barrier', fontsize=11,
                 fontweight='bold', loc='left')

    # Follicle
    ax.add_patch(Circle((0, 0), 0.8, fc='#FFD4B8', ec='#B22222', lw=1.5, zorder=1))
    ax.add_patch(Circle((0, 0), 0.35, fc='#FF8080', ec='#B22222', lw=0.8, zorder=1))

    # Treg barrier ring
    ax.add_patch(Circle((0, 0), 0.93, fc='none', ec='#2E8B57', lw=3.5,
                         ls='--', zorder=2))

    rng = np.random.RandomState(42)
    # B cells
    for _ in range(40):
        r, t = rng.uniform(0, 0.7), rng.uniform(0, 2*np.pi)
        ax.plot(r*np.cos(t), r*np.sin(t), 'o', color='#DC143C',
                ms=2.5, zorder=3, alpha=0.6)
    # Tregs at barrier
    for i in range(16):
        t = i * 2*np.pi/16 + rng.normal(0, 0.1)
        r = 0.90 + rng.normal(0, 0.04)
        ax.plot(r*np.cos(t), r*np.sin(t), '^', color='#2E8B57',
                ms=4.5, zorder=4, alpha=0.9)
    # CD8 T outside
    for _ in range(20):
        r = 1.1 + rng.exponential(0.2)
        t = rng.uniform(0, 2*np.pi)
        if r < 1.5:
            ax.plot(r*np.cos(t), r*np.sin(t), 's', color='#4169E1',
                    ms=2.5, zorder=3, alpha=0.5)
    # Suppression arrows
    for t in [0, np.pi/2, np.pi, 3*np.pi/2]:
        ax.annotate('', xy=(1.02*np.cos(t), 1.02*np.sin(t)),
                    xytext=(1.22*np.cos(t), 1.22*np.sin(t)),
                    arrowprops=dict(arrowstyle='-|>', color='red', lw=1.2))

    ax.text(0, -1.45, 'Treg "shield" at interface', ha='center', fontsize=8,
            fontweight='bold', color='#2E8B57')
    ax.text(0, -1.65, 'suppresses CD8 effector access', ha='center',
            fontsize=7, style='italic', color='#666')

    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC143C',
               ms=5, label='Tumor B'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2E8B57',
               ms=5, label='Treg'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4169E1',
               ms=5, label='CD8 T'),
        Line2D([0], [0], color='red', lw=1.5, label='Suppression'),
    ], loc='upper right', fontsize=6.5, frameon=False, handletextpad=0.2)


def draw_pressure_cartoon(ax):
    """Effector vs suppressor balance cartoon."""
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Concept: immune pressure', fontsize=11,
                 fontweight='bold', loc='left')

    # Balance beam (tilted — suppressors heavier)
    angle = np.radians(-12)
    bx = np.array([-1.3, 1.3])
    by = bx * np.sin(angle) + 0.3
    ax.plot(bx, by, color='#555', lw=3.5, zorder=2)
    # Fulcrum triangle
    ax.fill([0, -0.15, 0.15], [-0.15, 0.3, 0.3], color='#555', zorder=1)
    ax.plot([0, 0], [-0.6, -0.15], color='#555', lw=2.5, zorder=1)

    # Effector side (left, higher)
    ex, ey = -1.1, 0.3 + 1.1*np.sin(-angle) + 0.15
    ax.text(ex, ey + 0.35, 'Effectors', ha='center', fontsize=9,
            fontweight='bold', color='#4169E1')
    for dx in [-0.2, 0.05, 0.3]:
        ax.plot(ex + dx, ey + 0.05, 's', color='#4169E1', ms=9, zorder=3)

    # Suppressor side (right, lower — heavier)
    sx, sy = 1.1, 0.3 + 1.1*np.sin(angle) + 0.15
    ax.text(sx, sy + 0.6, 'Suppressors', ha='center', fontsize=9,
            fontweight='bold', color='#B22222')
    positions = [(-0.25, 0.05), (0.0, 0.05), (0.25, 0.05),
                 (-0.12, 0.25), (0.12, 0.25)]
    for dx, dy in positions:
        ax.plot(sx + dx, sy + dy, '^', color='#2E8B57', ms=7, zorder=3)
    for dx, dy in [(-0.2, 0.38), (0.05, 0.42), (0.2, 0.38)]:
        ax.plot(sx + dx, sy + dy, 'D', color='#6A5ACD', ms=5, zorder=3)

    ax.text(0, -1.2, 'Suppressors outweigh effectors', ha='center',
            fontsize=8, style='italic', color='#666')
    ax.text(0, -1.5, 'in most tissue compartments', ha='center',
            fontsize=8, style='italic', color='#666')

    ax.legend(handles=[
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4169E1',
               ms=6, label='CD8 effector'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2E8B57',
               ms=6, label='Treg'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#6A5ACD',
               ms=5, label='CD8 exhausted'),
    ], loc='upper left', fontsize=6.5, frameon=False, handletextpad=0.2)


def draw_myeloid_cartoon(ax):
    """M2 macrophage barrier cartoon."""
    _add_bg(ax)
    ax.set_title('(a) Concept: myeloid barrier', fontsize=11,
                 fontweight='bold', loc='left')

    ax.add_patch(Circle((0, 0), 0.75, fc='#FFD4B8', ec='#B22222', lw=1.5, zorder=1))
    ax.add_patch(Circle((0, 0), 0.3, fc='#FF8080', ec='#B22222', lw=0.8, zorder=1))

    rng = np.random.RandomState(42)
    for _ in range(35):
        r, t = rng.uniform(0, 0.65), rng.uniform(0, 2*np.pi)
        ax.plot(r*np.cos(t), r*np.sin(t), 'o', color='#DC143C',
                ms=2.5, zorder=3, alpha=0.6)
    # M2 macrophages
    for i in range(10):
        t = i * 2*np.pi/10 + rng.normal(0, 0.12)
        r = 0.90 + rng.normal(0, 0.06)
        ax.plot(r*np.cos(t), r*np.sin(t), 'H', color='#006400',
                ms=10, zorder=4, alpha=0.85)
    # CD8 T blocked
    for _ in range(18):
        r = 1.15 + rng.exponential(0.2)
        t = rng.uniform(0, 2*np.pi)
        if r < 1.5:
            ax.plot(r*np.cos(t), r*np.sin(t), 's', color='#4169E1',
                    ms=2.5, zorder=3, alpha=0.5)
    # Block symbols
    for t in [np.pi/5, 2*np.pi/5, 3*np.pi/5, 4*np.pi/5,
              6*np.pi/5, 7*np.pi/5, 8*np.pi/5, 9*np.pi/5]:
        ax.text(0.98*np.cos(t), 0.98*np.sin(t), '×', ha='center',
                va='center', fontsize=11, color='red', fontweight='bold', zorder=5)

    ax.text(0, -1.45, 'M2 macrophages block', ha='center', fontsize=8,
            fontweight='bold', color='#006400')
    ax.text(0, -1.65, 'T cell access to tumor', ha='center',
            fontsize=7, style='italic', color='#666')

    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC143C',
               ms=5, label='Tumor B'),
        Line2D([0], [0], marker='H', color='w', markerfacecolor='#006400',
               ms=8, label='M2 Mac'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4169E1',
               ms=5, label='CD8 T'),
    ], loc='upper right', fontsize=6.5, frameon=False, handletextpad=0.2)


# ═══════════════════════════════════════════════════════════════════════════
# Spatial ROI helpers
# ═══════════════════════════════════════════════════════════════════════════

def select_example_rois(data, comp_order, n=2, min_cells=5000,
                        min_diverse=5, exclude_tma='Biomax'):
    """Select representative ROIs with diverse compartment structure.

    Returns list of (roi_id, tma) tuples.
    """
    sample_ids = data['sample_ids']
    comps = data['comps']
    tma = data['tma']

    rois = np.unique(sample_ids)
    candidates = []
    for roi in rois:
        mask = (sample_ids == roi)
        n_cells = mask.sum()
        if n_cells < min_cells:
            continue
        t = tma[mask][0]
        if exclude_tma and t == exclude_tma:
            continue
        # Count diverse compartments (>50 cells)
        comp_roi = comps[mask]
        n_diverse = sum(1 for c in comp_order if (comp_roi == c).sum() > 50)
        if n_diverse < min_diverse:
            continue
        # Balance: fraction follicular vs interfollicular
        foll = sum((comp_roi == c).sum() for c in comp_order[:5]) / n_cells
        inter = sum((comp_roi == c).sum()
                     for c in comp_order[5:] if c in comp_order) / n_cells
        balance = 1.0 - abs(foll - inter)
        score = n_diverse * balance
        candidates.append((roi, t, score, n_cells, n_diverse))

    candidates.sort(key=lambda x: x[2], reverse=True)

    # Pick top n from different TMAs if possible
    selected = []
    used_tmas = set()
    for roi, t, score, nc, nd in candidates:
        if len(selected) >= n:
            break
        if t not in used_tmas or len(candidates) < n * 3:
            selected.append((roi, t))
            used_tmas.add(t)
    return selected


def plot_spatial_scatter(ax, cx, cy, colors, title, ms=0.8, legend_handles=None):
    """Plot spatial scatter of cells with given colors."""
    colors = np.array(colors)
    # Plot gray background cells first, then colored cells on top
    gray_mask = (colors == '#D3D3D3') | (colors == '#E8E8E8')
    if gray_mask.any():
        ax.scatter(cx[gray_mask], cy[gray_mask], c=colors[gray_mask],
                   s=ms*0.5, alpha=0.12, rasterized=True)
    if (~gray_mask).any():
        ax.scatter(cx[~gray_mask], cy[~gray_mask], c=colors[~gray_mask],
                   s=ms*2.0, alpha=0.75, rasterized=True)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', loc='left')
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=LEGEND_SIZE,
                  frameon=True, framealpha=0.85, handletextpad=0.2,
                  markerscale=2.5, borderpad=0.3)


def get_roi_mask(data, roi_id):
    """Get mask and data for a specific ROI."""
    mask = (data['sample_ids'] == roi_id)
    return mask


def color_by_compartment(comps_roi, comp_order, comp_colors):
    """Assign colors based on compartment membership."""
    comp_color_map = dict(zip(comp_order, comp_colors))
    return [comp_color_map.get(c, '#E8E8E8') for c in comps_roi]


def plot_paired_roi(fig, gs, gs_row, data, roi_id, roi_tma, panel_letters,
                    comp_order, comp_colors, comp_short_names,
                    celltype_color_fn, celltype_legend, ms=0.6):
    """Plot one ROI twice: left by compartment, right by cell type.

    gs: GridSpec object
    gs_row: int, row index in the GridSpec
    celltype_color_fn: function(ctypes_array) -> list of color strings
    celltype_legend: list of Line2D handles for the cell type panel
    """
    mask = get_roi_mask(data, roi_id)
    cx_roi = data['cx'][mask]
    cy_roi = data['cy'][mask]
    ct_roi = data['ctypes'][mask]
    comp_roi = data['comps'][mask]
    n_roi = int(mask.sum())

    # --- Left: compartment coloring ---
    ax_l = fig.add_subplot(gs[gs_row, 0])
    colors_comp = color_by_compartment(comp_roi, comp_order, comp_colors)
    # Compartment legend (only named ones)
    comp_legend = []
    for c, col, short in zip(comp_order, comp_colors, comp_short_names):
        if (comp_roi == c).sum() > 0:
            comp_legend.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=col,
                       ms=5, label=short.replace('\n', ' ')))
    lbl_l = f'({panel_letters[0]}) {roi_id} [{roi_tma}] — compartments'
    plot_spatial_scatter(ax_l, cx_roi, cy_roi, colors_comp, lbl_l,
                         ms=ms, legend_handles=comp_legend)

    # --- Right: cell type coloring ---
    ax_r = fig.add_subplot(gs[gs_row, 1])
    colors_ct = celltype_color_fn(ct_roi)
    lbl_r = f'({panel_letters[1]}) {roi_id} [{roi_tma}] — cell types ({n_roi:,} cells)'
    plot_spatial_scatter(ax_r, cx_roi, cy_roi, colors_ct, lbl_r,
                         ms=ms, legend_handles=celltype_legend)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_t_panel(path):
    """Load T-panel UTAG merged data. Returns dict of arrays."""
    print(f"Loading T-panel: {path}")
    f = h5py.File(path, 'r')
    markers = get_marker_names(f)
    sample_ids = load_array(f, 'sample_id')
    tma = load_array(f, 'tma')
    comps_raw = load_array(f, 'compartment_name')
    ctypes_raw = load_array(f, 'cell_type')

    # Apply display renames
    comps = rename_labels(comps_raw)
    ctypes = rename_labels(ctypes_raw)

    # Tumor mask
    tumor = get_tumor_mask(sample_ids)
    print(f"  Total cells: {len(sample_ids):,}, Tumor: {tumor.sum():,}")

    # Load expression matrix for immune markers
    marker_idx = {}
    for m in ['TOX', 'PD_1', 'CD39', 'GranzymeB', 'FoxP3', 'CD8a', 'CD4',
              'CD68', 'CD20', 'CD3']:
        if m in markers:
            marker_idx[m] = markers.index(m)

    # Only load rows we need (tumor cells) — but h5py doesn't support fancy indexing
    # Load full X then mask
    X = f['X'][:]
    cx = load_array(f, 'centroid_x') if 'centroid_x' in f['obs'] else None
    cy = load_array(f, 'centroid_y') if 'centroid_y' in f['obs'] else None
    f.close()

    # Apply tumor mask
    data = {
        'comps': comps[tumor],
        'ctypes': ctypes[tumor],
        'sample_ids': sample_ids[tumor],
        'tma': tma[tumor],
        'X': X[tumor],
        'marker_idx': marker_idx,
        'markers': markers,
    }
    if cx is not None:
        data['cx'] = cx[tumor].astype(float)
        data['cy'] = cy[tumor].astype(float)

    print(f"  Markers indexed: {list(marker_idx.keys())}")
    return data


def load_s_panel(path):
    """Load S-panel UTAG merged data."""
    print(f"Loading S-panel: {path}")
    f = h5py.File(path, 'r')
    markers = get_marker_names(f)
    sample_ids = load_array(f, 'sample_id')
    tma = load_array(f, 'tma')
    comps = rename_labels(load_array(f, 'compartment_name'))
    ctypes = rename_labels(load_array(f, 'cell_type'))
    tumor = get_tumor_mask(sample_ids)

    marker_idx = {}
    for m in ['CD163', 'CD206', 'VISTA', 'IDO', 'HLA_DR', 'CD68',
              'S100A9', 'CD11c', 'CD4', 'CD8a', 'CD20']:
        if m in markers:
            marker_idx[m] = markers.index(m)

    X = f['X'][:]
    cx = load_array(f, 'centroid_x') if 'centroid_x' in f['obs'] else None
    cy = load_array(f, 'centroid_y') if 'centroid_y' in f['obs'] else None
    f.close()

    data = {
        'comps': comps[tumor], 'ctypes': ctypes[tumor],
        'sample_ids': sample_ids[tumor], 'tma': tma[tumor],
        'X': X[tumor], 'marker_idx': marker_idx, 'markers': markers,
    }
    if cx is not None:
        data['cx'] = cx[tumor].astype(float)
        data['cy'] = cy[tumor].astype(float)

    print(f"  Total: {len(sample_ids):,}, Tumor: {tumor.sum():,}")
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════════

def compartment_celltype_fractions(comps, ctypes, comp_order, type_groups):
    """Compute fraction of each cell type group per compartment.

    type_groups: dict of group_name -> list of cell type labels
    Returns: dict of group_name -> array of fractions (one per compartment)
    """
    result = {}
    counts_per_comp = []
    for c in comp_order:
        mask = (comps == c)
        n = mask.sum()
        counts_per_comp.append(n)
    result['_counts'] = np.array(counts_per_comp)

    for gname, gtypes in type_groups.items():
        fracs = []
        for c in comp_order:
            mask = (comps == c)
            n = mask.sum()
            if n == 0:
                fracs.append(0.0)
            else:
                in_group = np.isin(ctypes[mask], gtypes).sum()
                fracs.append(float(in_group) / n)
        result[gname] = np.array(fracs)
    return result


def exhaustion_fraction_by_compartment(comps, ctypes, comp_order):
    """Fraction of CD8 T cells that are exhausted, per compartment."""
    fracs = []
    counts = []
    for c in comp_order:
        mask = (comps == c)
        cd8_mask = mask & np.isin(ctypes, CD8_ALL)
        n_cd8 = cd8_mask.sum()
        if n_cd8 < 10:
            fracs.append(np.nan)
            counts.append(n_cd8)
        else:
            n_exh = (mask & np.isin(ctypes, CD8_EXHAUSTED)).sum()
            fracs.append(float(n_exh) / n_cd8)
            counts.append(n_cd8)
    return np.array(fracs), np.array(counts)


def marker_expression_by_compartment(comps, ctypes, X, marker_idx,
                                      comp_order, cell_types, marker_names):
    """Mean marker expression among specific cell types, per compartment."""
    result = {}
    for mname in marker_names:
        if mname not in marker_idx:
            continue
        midx = marker_idx[mname]
        vals = []
        for c in comp_order:
            mask = (comps == c) & np.isin(ctypes, cell_types)
            if mask.sum() < 10:
                vals.append(np.nan)
            else:
                vals.append(float(np.mean(X[mask, midx])))
        result[mname] = np.array(vals)
    return result


def effector_suppressor_ratio(comps, ctypes, comp_order):
    """Effector:Suppressor ratio per compartment."""
    ratios = []
    n_eff_list, n_sup_list = [], []
    for c in comp_order:
        mask = (comps == c)
        n_eff = (mask & np.isin(ctypes, EFFECTOR)).sum()
        n_sup = (mask & np.isin(ctypes, SUPPRESSOR)).sum()
        n_eff_list.append(n_eff)
        n_sup_list.append(n_sup)
        if n_sup == 0:
            ratios.append(np.nan if n_eff == 0 else 10.0)  # cap
        else:
            ratios.append(float(n_eff) / n_sup)
    return np.array(ratios), np.array(n_eff_list), np.array(n_sup_list)


def per_roi_immune_pressure(comps, ctypes, sample_ids):
    """Compute per-ROI immune pressure score (E:S ratio)."""
    rois = np.unique(sample_ids)
    scores = {}
    for roi in rois:
        mask = (sample_ids == roi)
        n_eff = np.isin(ctypes[mask], EFFECTOR).sum()
        n_sup = np.isin(ctypes[mask], SUPPRESSOR).sum()
        if n_sup == 0:
            scores[roi] = 10.0 if n_eff > 0 else np.nan
        else:
            scores[roi] = float(n_eff) / n_sup
    return scores


def cd39_fraction_by_compartment(comps, ctypes, X, marker_idx, comp_order,
                                  threshold=0.5):
    """Fraction of T cells that are CD39+ per compartment."""
    if 'CD39' not in marker_idx:
        return None
    cidx = marker_idx['CD39']
    t_types = ['CD4 T cells', 'CD8 T cells', 'CD8 T exhausted',
               'CD8 T pre-exhausted (TOX+)', 'Treg']
    fracs = []
    counts = []
    for c in comp_order:
        mask = (comps == c) & np.isin(ctypes, t_types)
        n = mask.sum()
        if n < 10:
            fracs.append(np.nan)
        else:
            cd39_pos = (X[mask, cidx] > threshold).sum()
            fracs.append(float(cd39_pos) / n)
        counts.append(n)
    return np.array(fracs), np.array(counts)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Immune Exclusion Gradient
# ═══════════════════════════════════════════════════════════════════════════

def generate_fig_gradient(data, output_dir, tonsil_t_data=None):
    """Generate immune exclusion gradient + Treg barrier figure.

    If tonsil_t_data is provided, adds a third row (g, h, i) with FL vs tonsil
    comparison panels, making it a 3x3 grid. Otherwise 2x3.
    """
    print("\n=== Figure S5: Immune Exclusion Gradient + Treg Barrier ===")
    comps, ctypes = data['comps'], data['ctypes']
    X, midx = data['X'], data['marker_idx']
    n_comp = len(GRADIENT_ORDER)
    x = np.arange(n_comp)

    # --- Compute gradient data ---
    type_groups = {
        'B cells': B_TYPES,
        'CD4 T': ['CD4 T cells'],
        'CD8 T (all)': CD8_ALL,
        'Treg': ['Treg'],
        'Macrophages': ['Macrophages'],
    }
    fracs = compartment_celltype_fractions(comps, ctypes, GRADIENT_ORDER, type_groups)
    exh_frac, exh_n = exhaustion_fraction_by_compartment(comps, ctypes, GRADIENT_ORDER)
    marker_expr = marker_expression_by_compartment(
        comps, ctypes, X, midx, GRADIENT_ORDER, CD8_ALL,
        ['TOX', 'PD_1', 'CD39'])

    # --- Compute Treg barrier data ---
    treg_frac = []
    cd8_eff_frac = []
    for c in GRADIENT_ORDER:
        mask = (comps == c)
        n = mask.sum()
        if n == 0:
            treg_frac.append(0)
            cd8_eff_frac.append(0)
        else:
            treg_frac.append(float((mask & (ctypes == 'Treg')).sum()) / n)
            cd8_eff_frac.append(float((mask & np.isin(ctypes, CD8_EFFECTOR)).sum()) / n)
    treg_frac = np.array(treg_frac)
    cd8_eff_frac = np.array(cd8_eff_frac)

    treg_cd8_ratio = []
    for c in GRADIENT_ORDER:
        mask = (comps == c)
        n_treg = (mask & (ctypes == 'Treg')).sum()
        n_cd8e = (mask & np.isin(ctypes, CD8_EFFECTOR)).sum()
        if n_cd8e == 0:
            treg_cd8_ratio.append(np.nan if n_treg == 0 else 10.0)
        else:
            treg_cd8_ratio.append(float(n_treg) / n_cd8e)
    treg_cd8_ratio = np.array(treg_cd8_ratio)

    cd39_result = cd39_fraction_by_compartment(
        comps, ctypes, X, midx, GRADIENT_ORDER)

    # Print summary
    print("\nCD8 exhaustion fraction:", exh_frac)
    print("\nTreg vs CD8 effector fractions:")
    for i, c in enumerate(GRADIENT_ORDER):
        print(f"  {c:<35} Treg={treg_frac[i]*100:.1f}%  CD8eff={cd8_eff_frac[i]*100:.1f}%  "
              f"ratio={treg_cd8_ratio[i]:.2f}")

    # --- Figure: 2×3 or 3×3 grid ---
    n_rows = 3 if tonsil_t_data is not None else 2
    fig_h = 24 if n_rows == 3 else 16
    fig = plt.figure(figsize=(20, fig_h))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.50, wspace=0.35,
                  left=0.06, right=0.96, top=0.95 if n_rows == 3 else 0.93,
                  bottom=0.05 if n_rows == 3 else 0.08)

    bar_colors = [GRADIENT_COLORS[i] for i in range(n_comp)]

    # Import tonsil functions if needed
    if tonsil_t_data is not None:
        from fig_tonsil_comparison import (plot_tonsil_exclusion,
                                           plot_tonsil_exhaustion,
                                           plot_tonsil_treg)

    # (a) Cell type fractions across gradient [0,0]
    ax_a = fig.add_subplot(gs[0, 0])
    w = 0.15
    colors_grp = {'B cells': '#DC143C', 'CD4 T': '#4169E1',
                  'CD8 T (all)': '#1E90FF', 'Treg': '#2E8B57',
                  'Macrophages': '#8B4513'}
    for j, (gname, color) in enumerate(colors_grp.items()):
        offset = (j - 2) * w
        ax_a.bar(x + offset, fracs[gname], w, color=color, label=gname,
                 alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_a.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_a.text(2, ax_a.get_ylim()[1] * 0.95, 'Follicular', ha='center',
              fontsize=10, color='#B22222', fontstyle='italic')
    ax_a.text(7, ax_a.get_ylim()[1] * 0.95, 'Interfollicular', ha='center',
              fontsize=10, color='#4169E1', fontstyle='italic')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_a.set_ylabel('Fraction of cells', fontsize=LABEL_SIZE)
    ax_a.legend(fontsize=LEGEND_SIZE, loc='upper right', frameon=True,
                edgecolor='#cccccc', fancybox=False)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.text(-0.02, 1.02, '$\\bf{a}$', transform=ax_a.transAxes,
              fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # (b) Tonsil exclusion gradient [0,1]
    if tonsil_t_data is not None:
        ax_b = fig.add_subplot(gs[0, 1])
        plot_tonsil_exclusion(ax_b, tonsil_t_data)
        ax_b.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')  # override to match row
        ax_b.text(-0.02, 1.02, '$\\bf{b}$', transform=ax_b.transAxes,
                  fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')
        next_label = 'c'
        next_pos = gs[0, 2]
    else:
        next_label = 'b'
        next_pos = gs[0, 1]

    # (c/b) CD8 exhaustion fraction
    ax_exh = fig.add_subplot(next_pos)
    ax_exh.bar(x, exh_frac, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax_exh.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_exh.set_xticks(x)
    ax_exh.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_exh.set_ylabel('Exhaustion fraction\n(exhausted / all CD8 T)', fontsize=LABEL_SIZE)
    ax_exh.spines['top'].set_visible(False)
    ax_exh.spines['right'].set_visible(False)
    ax_exh.text(-0.02, 1.02, f'$\\bf{{{next_label}}}$', transform=ax_exh.transAxes,
              fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # (d/c) Marker expression among CD8 T cells
    if tonsil_t_data is not None:
        mk_pos, mk_label = gs[1, 0], 'd'
    else:
        mk_pos, mk_label = gs[0, 2], 'c'
    ax_mk = fig.add_subplot(mk_pos)
    marker_colors = {'TOX': '#6A5ACD', 'PD_1': '#CD5C5C', 'CD39': '#2E8B57'}
    marker_labels = {'TOX': 'TOX', 'PD_1': 'PD-1', 'CD39': 'CD39'}
    for mname, color in marker_colors.items():
        if mname in marker_expr:
            ax_mk.plot(x, marker_expr[mname], 'o-', color=color, ms=7, lw=2,
                      label=marker_labels[mname], alpha=0.85)
    ax_mk.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_mk.set_xticks(x)
    ax_mk.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_mk.set_ylabel('Mean expression (z-scored)', fontsize=LABEL_SIZE)
    ax_mk.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax_mk.spines['top'].set_visible(False)
    ax_mk.spines['right'].set_visible(False)
    ax_mk.text(-0.02, 1.02, f'$\\bf{{{mk_label}}}$', transform=ax_mk.transAxes,
              fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # (e) Tonsil exhaustion [1,1]
    if tonsil_t_data is not None:
        ax_te = fig.add_subplot(gs[1, 1])
        plot_tonsil_exhaustion(ax_te, tonsil_t_data)
        ax_te.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')  # override to match row
        ax_te.text(-0.02, 1.02, '$\\bf{e}$', transform=ax_te.transAxes,
                  fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')
        treg_pos, treg_label = gs[1, 2], 'f'
        ratio_pos, ratio_label = gs[2, 0], 'g'
        ttreg_pos, ttreg_label = gs[2, 1], 'h'
        cd39_pos, cd39_label = gs[2, 2], 'i'
    else:
        treg_pos, treg_label = gs[1, 0], 'd'
        ratio_pos, ratio_label = gs[1, 1], 'e'
        cd39_pos, cd39_label = gs[1, 2], 'f'

    # (f/d) Treg vs CD8 effector fraction across gradient
    ax_treg = fig.add_subplot(treg_pos)
    w2 = 0.35
    ax_treg.bar(x - w2 / 2, treg_frac * 100, w2, color='#2E8B57', label='Treg',
             alpha=0.85, edgecolor='white')
    ax_treg.bar(x + w2 / 2, cd8_eff_frac * 100, w2, color='#4169E1',
             label='CD8 effector', alpha=0.85, edgecolor='white')
    ax_treg.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_treg.set_xticks(x)
    ax_treg.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_treg.set_ylabel('% of cells in compartment', fontsize=LABEL_SIZE)
    ax_treg.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax_treg.spines['top'].set_visible(False)
    ax_treg.spines['right'].set_visible(False)
    ax_treg.text(-0.02, 1.02, f'$\\bf{{{treg_label}}}$', transform=ax_treg.transAxes,
              fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # (g/e) Treg:CD8 effector ratio
    ax_ratio = fig.add_subplot(ratio_pos)
    ax_ratio.bar(x, treg_cd8_ratio, color=bar_colors, edgecolor='white')
    ax_ratio.axhline(1.0, color='red', ls='--', lw=1, alpha=0.7, label='Parity (1:1)')
    ax_ratio.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_ratio.set_ylabel('Treg : CD8 effector ratio', fontsize=LABEL_SIZE)
    ax_ratio.legend(fontsize=LEGEND_SIZE, frameon=False, loc='upper right')
    ax_ratio.spines['top'].set_visible(False)
    ax_ratio.spines['right'].set_visible(False)
    ax_ratio.text(-0.02, 1.02, f'$\\bf{{{ratio_label}}}$', transform=ax_ratio.transAxes,
              fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # (h) Tonsil Treg [2,1]
    if tonsil_t_data is not None:
        ax_tt = fig.add_subplot(ttreg_pos)
        plot_tonsil_treg(ax_tt, tonsil_t_data)
        ax_tt.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')  # override to match row
        ax_tt.text(-0.02, 1.02, f'$\\bf{{{ttreg_label}}}$', transform=ax_tt.transAxes,
                  fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    # (i/f) CD39+ fraction among T cells
    ax_cd39 = fig.add_subplot(cd39_pos)
    if cd39_result is not None:
        cd39_frac_vals, cd39_n = cd39_result
        ax_cd39.bar(x, cd39_frac_vals * 100, color='#8B4513', alpha=0.85,
                 edgecolor='white')
        for i in range(n_comp):
            if not np.isnan(cd39_frac_vals[i]):
                ax_cd39.text(i, cd39_frac_vals[i] * 100 + 0.5, f'n={cd39_n[i]:,}',
                          ha='center', fontsize=7, color='#666')
        ax_cd39.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
        ax_cd39.set_ylabel('% CD39+ among T cells', fontsize=LABEL_SIZE)
    else:
        ax_cd39.text(0.5, 0.5, 'CD39 marker not available', transform=ax_cd39.transAxes,
                  ha='center', va='center', fontsize=12)
    ax_cd39.set_xticks(x)
    ax_cd39.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_cd39.spines['top'].set_visible(False)
    ax_cd39.spines['right'].set_visible(False)
    ax_cd39.text(-0.02, 1.02, f'$\\bf{{{cd39_label}}}$', transform=ax_cd39.transAxes,
              fontsize=PANEL_LABEL_SIZE, va='bottom', ha='left')

    out = os.path.join(output_dir, 'fig_ie_gradient.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return fracs, exh_frac, marker_expr


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Regulatory Barrier & CD39
# ═══════════════════════════════════════════════════════════════════════════

def generate_fig_treg_barrier(data, output_dir):
    """Generate Treg barrier and CD39 suppression figure."""
    print("\n=== Figure 2: Regulatory Barrier & CD39 ===")
    comps, ctypes = data['comps'], data['ctypes']
    X, midx = data['X'], data['marker_idx']
    n_comp = len(GRADIENT_ORDER)
    x_pos = np.arange(n_comp)

    # --- Compute ---
    # Treg and CD8 effector fractions
    treg_frac = []
    cd8_eff_frac = []
    for c in GRADIENT_ORDER:
        mask = (comps == c)
        n = mask.sum()
        if n == 0:
            treg_frac.append(0); cd8_eff_frac.append(0)
        else:
            treg_frac.append(float((mask & (ctypes == 'Treg')).sum()) / n)
            cd8_eff_frac.append(float((mask & np.isin(ctypes, CD8_EFFECTOR)).sum()) / n)
    treg_frac = np.array(treg_frac)
    cd8_eff_frac = np.array(cd8_eff_frac)

    # Treg-to-CD8 effector ratio
    treg_cd8_ratio = []
    for c in GRADIENT_ORDER:
        mask = (comps == c)
        n_treg = (mask & (ctypes == 'Treg')).sum()
        n_cd8e = (mask & np.isin(ctypes, CD8_EFFECTOR)).sum()
        if n_cd8e == 0:
            treg_cd8_ratio.append(np.nan if n_treg == 0 else 10.0)
        else:
            treg_cd8_ratio.append(float(n_treg) / n_cd8e)
    treg_cd8_ratio = np.array(treg_cd8_ratio)

    # CD39+ fraction among T cells
    cd39_result = cd39_fraction_by_compartment(
        comps, ctypes, X, midx, GRADIENT_ORDER)

    print("\nTreg vs CD8 effector fractions:")
    print(f"{'Compartment':<30} {'Treg%':>8} {'CD8eff%':>8} {'Treg:CD8':>10}")
    for i, c in enumerate(GRADIENT_ORDER):
        print(f"{c:<30} {treg_frac[i]*100:>7.1f}% {cd8_eff_frac[i]*100:>7.1f}% "
              f"{treg_cd8_ratio[i]:>10.2f}")

    # --- Figure ---
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1, 1, 0.9],
                  left=0.06, right=0.96, top=0.95, bottom=0.04)

    # (a) Cartoon
    ax_a = fig.add_subplot(gs[0, 0])
    draw_treg_cartoon(ax_a)

    # (b) Treg vs CD8 effector fraction
    ax_b = fig.add_subplot(gs[0, 1])
    w = 0.35
    ax_b.bar(x_pos - w/2, treg_frac * 100, w, color='#2E8B57', label='Treg',
             alpha=0.85, edgecolor='white')
    ax_b.bar(x_pos + w/2, cd8_eff_frac * 100, w, color='#4169E1',
             label='CD8 effector', alpha=0.85, edgecolor='white')
    ax_b.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_b.set_ylabel('% of cells in compartment', fontsize=LABEL_SIZE)
    ax_b.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax_b.set_title('(b) Treg vs CD8 effector across gradient', fontsize=11,
                   fontweight='bold', loc='left')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # (c) Treg:CD8 ratio
    ax_c = fig.add_subplot(gs[1, 0])
    bar_colors = [GRADIENT_COLORS[i] for i in range(n_comp)]
    ax_c.bar(x_pos, treg_cd8_ratio, color=bar_colors, edgecolor='white')
    ax_c.axhline(1.0, color='red', ls='--', lw=1, alpha=0.7, label='Parity (1:1)')
    ax_c.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_c.set_ylabel('Treg : CD8 effector ratio', fontsize=LABEL_SIZE)
    ax_c.legend(fontsize=LEGEND_SIZE, frameon=False, loc='upper right')
    ax_c.set_title('(c) Regulatory pressure (Treg:CD8 ratio)', fontsize=11,
                   fontweight='bold', loc='left')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # (d) CD39+ fraction among T cells
    ax_d = fig.add_subplot(gs[1, 1])
    if cd39_result is not None:
        cd39_frac, cd39_n = cd39_result
        ax_d.bar(x_pos, cd39_frac * 100, color='#8B4513', alpha=0.85,
                 edgecolor='white')
        for i in range(n_comp):
            if not np.isnan(cd39_frac[i]):
                ax_d.text(i, cd39_frac[i]*100 + 0.5, f'n={cd39_n[i]:,}',
                          ha='center', fontsize=6, color='#666')
        ax_d.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
        ax_d.set_ylabel('% CD39+ among T cells', fontsize=LABEL_SIZE)
        ax_d.set_title('(d) CD39+ fraction (adenosine suppression)', fontsize=11,
                       fontweight='bold', loc='left')
    else:
        ax_d.text(0.5, 0.5, 'CD39 marker not available', transform=ax_d.transAxes,
                  ha='center', va='center', fontsize=12)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # (e, f) Spatial ROI: left = compartments, right = Treg vs CD8
    example_rois = select_example_rois(data, GRADIENT_ORDER, n=1)
    b_set = set(B_TYPES + ['B / Unassigned transitional', 'Weak CD20 border'])
    def treg_ct_colors(ct_arr):
        colors = []
        for ct in ct_arr:
            if ct == 'Treg':
                colors.append('#2E8B57')       # green = Treg
            elif ct in ('CD8 T cells', 'Macrophages (GzmB+)'):
                colors.append('#1E90FF')       # blue = CD8 effector
            elif ct in ('CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)'):
                colors.append('#6A5ACD')       # purple = CD8 exhausted
            elif ct in b_set:
                colors.append('#D3D3D3')       # B cells background
            else:
                colors.append('#E8E8E8')
        return colors
    ct_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E8B57',
               ms=6, label='Treg'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E90FF',
               ms=6, label='CD8 effector'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6A5ACD',
               ms=6, label='CD8 exhausted'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D3D3D3',
               ms=6, label='B cells (background)'),
    ]
    if example_rois:
        roi_id, roi_tma = example_rois[0]
        plot_paired_roi(fig, gs, 2, data, roi_id, roi_tma, ('e', 'f'),
                        GRADIENT_ORDER, GRADIENT_COLORS, GRADIENT_SHORT,
                        treg_ct_colors, ct_legend)

    out = os.path.join(output_dir, 'fig_ie_treg_barrier.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return treg_frac, cd8_eff_frac, treg_cd8_ratio


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Immune Pressure Landscape
# ═══════════════════════════════════════════════════════════════════════════

def generate_fig_pressure(data, output_dir):
    """Generate effector:suppressor balance figure."""
    print("\n=== Figure 3: Immune Pressure Landscape ===")
    comps, ctypes = data['comps'], data['ctypes']
    sample_ids = data['sample_ids']
    tma = data['tma']
    n_comp = len(GRADIENT_ORDER)
    x_pos = np.arange(n_comp)

    # --- Compute ---
    es_ratio, n_eff, n_sup = effector_suppressor_ratio(
        comps, ctypes, GRADIENT_ORDER)

    roi_scores = per_roi_immune_pressure(comps, ctypes, sample_ids)

    # Per-TMA scores
    tma_scores = {}
    roi_tma = {}
    for roi in np.unique(sample_ids):
        mask_roi = (sample_ids == roi)
        t = tma[mask_roi][0]
        roi_tma[roi] = t
        if t not in tma_scores:
            tma_scores[t] = []
        if roi in roi_scores and not np.isnan(roi_scores[roi]):
            tma_scores[t].append(roi_scores[roi])

    # ROI archetype clustering (simplified — recompute)
    excl_comps = ['B / Unassigned transitional', 'Unidentified zone',
                  'Weak CD20 border', 'Activated B / CXCR5hi zone',
                  'Cytotoxic niche']
    non_excl = [c for c in GRADIENT_ORDER if c not in excl_comps]
    rois_list = [r for r in np.unique(sample_ids)]
    roi_fracs = []
    valid_rois = []
    for roi in rois_list:
        mask = (sample_ids == roi)
        comp_roi = comps[mask]
        total = len(comp_roi)
        excl_frac = sum((comp_roi == ec).sum() for ec in excl_comps) / total
        if excl_frac > 0.7 or total < 500:
            continue
        row = []
        non_excl_n = sum((comp_roi == c).sum() for c in non_excl)
        if non_excl_n < 100:
            continue
        for c in non_excl:
            row.append((comp_roi == c).sum() / non_excl_n)
        roi_fracs.append(row)
        valid_rois.append(roi)
    roi_frac_mat = np.array(roi_fracs)

    # Cluster
    k = 8
    if len(valid_rois) > k:
        Z = linkage(pdist(roi_frac_mat, 'euclidean'), method='ward')
        labels = fcluster(Z, t=k, criterion='maxclust')
    else:
        labels = np.ones(len(valid_rois), dtype=int)

    roi_cluster = dict(zip(valid_rois, labels))

    # Per-cluster immune pressure
    cluster_scores = {}
    for cl in sorted(set(labels)):
        cl_rois = [r for r, c in roi_cluster.items() if c == cl]
        scores = [roi_scores[r] for r in cl_rois if r in roi_scores
                  and not np.isnan(roi_scores[r])]
        if scores:
            cluster_scores[cl] = scores

    print(f"\nE:S ratio per compartment:")
    for i, c in enumerate(GRADIENT_ORDER):
        print(f"  {c:<30} E:S = {es_ratio[i]:.2f} "
              f"(eff={n_eff[i]:,}, sup={n_sup[i]:,})")

    all_scores = [v for v in roi_scores.values() if not np.isnan(v)]
    print(f"\nPer-ROI immune pressure: median={np.median(all_scores):.2f}, "
          f"range=[{np.min(all_scores):.2f}, {np.max(all_scores):.2f}]")

    # --- Figure ---
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1, 1, 0.9],
                  left=0.06, right=0.96, top=0.95, bottom=0.04)

    # (a) Cartoon
    ax_a = fig.add_subplot(gs[0, 0])
    draw_pressure_cartoon(ax_a)

    # (b) E:S ratio per compartment
    ax_b = fig.add_subplot(gs[0, 1])
    bars = ax_b.bar(x_pos, es_ratio, color=GRADIENT_COLORS, edgecolor='white')
    ax_b.axhline(1.0, color='red', ls='--', lw=1.5, alpha=0.7, label='Parity (1:1)')
    ax_b.axvline(FOLL_INTER_BOUNDARY - 0.5, color='#999', ls='--', lw=1, alpha=0.7)
    # Annotate
    for i in range(n_comp):
        if not np.isnan(es_ratio[i]):
            ax_b.text(i, es_ratio[i] + 0.02, f'{es_ratio[i]:.2f}',
                      ha='center', fontsize=7, fontweight='bold')
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_b.set_ylabel('Effector : Suppressor ratio', fontsize=LABEL_SIZE)
    ax_b.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax_b.set_title('(b) Immune pressure by compartment', fontsize=11,
                   fontweight='bold', loc='left')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # (c) Per-ROI distribution by TMA
    ax_c = fig.add_subplot(gs[1, 0])
    tma_order = sorted(tma_scores.keys())
    tma_colors = {'A1': '#E74C3C', 'B1': '#3498DB', 'C1': '#2ECC71',
                  'Biomax': '#9B59B6'}
    bp_data = [tma_scores.get(t, []) for t in tma_order]
    bp = ax_c.boxplot(bp_data, labels=tma_order, patch_artist=True,
                      widths=0.6, showfliers=True,
                      flierprops={'ms': 3, 'alpha': 0.5})
    for patch, t in zip(bp['boxes'], tma_order):
        patch.set_facecolor(tma_colors.get(t, '#888'))
        patch.set_alpha(0.6)
    # Overlay individual points
    for i, t in enumerate(tma_order):
        scores = tma_scores.get(t, [])
        jitter = np.random.RandomState(42).normal(0, 0.08, len(scores))
        ax_c.scatter(np.full(len(scores), i+1) + jitter, scores,
                     c=tma_colors.get(t, '#888'), s=12, alpha=0.5, zorder=3)
    ax_c.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5)
    ax_c.set_ylabel('E:S ratio per ROI', fontsize=LABEL_SIZE)
    ax_c.set_title('(c) Immune pressure by TMA', fontsize=11,
                   fontweight='bold', loc='left')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # (d) By archetype cluster
    ax_d = fig.add_subplot(gs[1, 1])
    cl_order = sorted(cluster_scores.keys())
    bp_data2 = [cluster_scores[cl] for cl in cl_order]
    bp2 = ax_d.boxplot(bp_data2, labels=[f'C{cl}' for cl in cl_order],
                       patch_artist=True, widths=0.6, showfliers=True,
                       flierprops={'ms': 3, 'alpha': 0.5})
    arch_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    for i, patch in enumerate(bp2['boxes']):
        patch.set_facecolor(arch_colors[i % 8])
        patch.set_alpha(0.6)
    # Overlay points
    for i, cl in enumerate(cl_order):
        scores = cluster_scores[cl]
        jitter = np.random.RandomState(42).normal(0, 0.08, len(scores))
        ax_d.scatter(np.full(len(scores), i+1) + jitter, scores,
                     c=[arch_colors[i % 8]], s=12, alpha=0.5, zorder=3)
    ax_d.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5)
    ax_d.set_ylabel('E:S ratio per ROI', fontsize=LABEL_SIZE)
    ax_d.set_xlabel('Archetype cluster', fontsize=LABEL_SIZE)
    ax_d.set_title('(d) Immune pressure by tissue archetype', fontsize=11,
                   fontweight='bold', loc='left')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # (e, f) Spatial ROI contrast: LOW vs HIGH E:S
    valid_es = [(r, s) for r, s in roi_scores.items()
                if not np.isnan(s) and (sample_ids == r).sum() >= 3000]
    valid_es.sort(key=lambda x: x[1])
    b_set = set(B_TYPES + ['B / Unassigned transitional', 'Weak CD20 border'])

    def es_ct_colors(ct_arr):
        """Effectors = blue, Suppressors = warm red/green, B = gray."""
        colors = []
        for ct in ct_arr:
            if ct in ('CD8 T cells', 'Macrophages (GzmB+)'):
                colors.append('#1E90FF')       # blue = effector
            elif ct in ('CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)'):
                colors.append('#E74C3C')       # red = suppressor (exhausted)
            elif ct == 'Treg':
                colors.append('#E67E22')       # orange = suppressor (Treg)
            elif ct in b_set:
                colors.append('#D3D3D3')
            else:
                colors.append('#E8E8E8')
        return colors

    es_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E90FF',
               ms=6, label='Effector (CD8/GzmB+)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               ms=6, label='CD8 exhausted'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E67E22',
               ms=6, label='Treg'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D3D3D3',
               ms=6, label='B cells (background)'),
    ]

    for pi, (example, label_prefix) in enumerate(
            [(valid_es[0] if valid_es else None, 'LOW E:S'),
             (valid_es[-1] if valid_es else None, 'HIGH E:S')]):
        ax_roi = fig.add_subplot(gs[2, pi])
        if example is None:
            ax_roi.text(0.5, 0.5, 'No suitable ROI', transform=ax_roi.transAxes,
                        ha='center', va='center')
            continue
        roi_id, es_val = example
        mask = get_roi_mask(data, roi_id)
        cx_roi = data['cx'][mask]
        cy_roi = data['cy'][mask]
        ct_roi = data['ctypes'][mask]
        t_roi = tma[mask][0]
        colors_roi = es_ct_colors(ct_roi)
        n_eff_r = np.isin(ct_roi, EFFECTOR).sum()
        n_sup_r = np.isin(ct_roi, SUPPRESSOR).sum()
        lbl = (f'({chr(101+pi)}) {label_prefix}: {roi_id} [{t_roi}] — '
               f'E:S={es_val:.2f} ({n_eff_r} eff, {n_sup_r} sup)')
        plot_spatial_scatter(ax_roi, cx_roi, cy_roi, colors_roi, lbl,
                             ms=0.8, legend_handles=es_legend if pi == 0 else None)

    out = os.path.join(output_dir, 'fig_ie_pressure.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return es_ratio, roi_scores


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Myeloid Suppression (S-panel)
# ═══════════════════════════════════════════════════════════════════════════

def generate_fig_myeloid(data, output_dir):
    """Generate myeloid suppression figure (S-panel)."""
    print("\n=== Figure 4: Myeloid Suppression (S-panel) ===")
    comps, ctypes = data['comps'], data['ctypes']
    X, midx = data['X'], data['marker_idx']
    n_comp = len(S_GRADIENT_ORDER)
    x_pos = np.arange(n_comp)

    # --- Compute myeloid fractions ---
    myeloid_groups = {
        'M2 Mac': ['M2 Macrophages'],
        'M1 Mac': ['M1 Macrophages'],
        'Mac (other)': ['Macrophages'],
        'DC': ['Dendritic cells', 'pDC'],
        'Myeloid (S100A9+)': ['Myeloid (S100A9+)'],
    }
    mye_fracs = compartment_celltype_fractions(
        comps, ctypes, S_GRADIENT_ORDER, myeloid_groups)

    # T cell fractions for context
    t_groups = {
        'CD4 T': ['CD4 T cells'],
        'CD8 T': ['CD8 T cells'],
    }
    t_fracs = compartment_celltype_fractions(
        comps, ctypes, S_GRADIENT_ORDER, t_groups)

    # VISTA, IDO, HLA-DR among myeloid cells per compartment
    s_mye_types = ['M1 Macrophages', 'M2 Macrophages', 'Macrophages',
                   'Myeloid (S100A9+)', 'Dendritic cells']
    checkpoint_expr = marker_expression_by_compartment(
        comps, ctypes, X, midx, S_GRADIENT_ORDER, s_mye_types,
        ['VISTA', 'IDO', 'HLA_DR', 'CD163', 'CD206'])

    print("\nMyeloid fractions per S-panel compartment:")
    print(f"{'Compartment':<25} {'M2':>6} {'M1':>6} {'Mac':>6} {'DC':>6} "
          f"{'S100A9':>6} {'n':>8}")
    for i, c in enumerate(S_GRADIENT_ORDER):
        print(f"{c:<25} {mye_fracs['M2 Mac'][i]*100:>5.1f}% "
              f"{mye_fracs['M1 Mac'][i]*100:>5.1f}% "
              f"{mye_fracs['Mac (other)'][i]*100:>5.1f}% "
              f"{mye_fracs['DC'][i]*100:>5.1f}% "
              f"{mye_fracs['Myeloid (S100A9+)'][i]*100:>5.1f}% "
              f"{mye_fracs['_counts'][i]:>8,}")

    # --- Figure ---
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1, 1, 0.9],
                  left=0.06, right=0.96, top=0.95, bottom=0.04)

    # (a) Cartoon
    ax_a = fig.add_subplot(gs[0, 0])
    draw_myeloid_cartoon(ax_a)

    # (b) Myeloid subtype fractions (stacked bar)
    ax_b = fig.add_subplot(gs[0, 1])
    mye_colors = {'M2 Mac': '#006400', 'M1 Mac': '#228B22',
                  'Mac (other)': '#8FBC8F', 'DC': '#DDA0DD',
                  'Myeloid (S100A9+)': '#DAA520'}
    bottom = np.zeros(n_comp)
    for gname in ['M2 Mac', 'M1 Mac', 'Mac (other)', 'DC', 'Myeloid (S100A9+)']:
        vals = mye_fracs[gname] * 100
        ax_b.bar(x_pos, vals, bottom=bottom, color=mye_colors[gname],
                 label=gname, edgecolor='white', linewidth=0.5)
        bottom += vals
    ax_b.axvline(3.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_b.text(1.5, ax_b.get_ylim()[1] if ax_b.get_ylim()[1] > 0 else 30,
              'Follicular', ha='center', fontsize=8, color='#B22222',
              fontstyle='italic', va='top')
    ax_b.text(4.5, ax_b.get_ylim()[1] if ax_b.get_ylim()[1] > 0 else 30,
              'Interfollicular', ha='center', fontsize=8, color='#4169E1',
              fontstyle='italic', va='top')
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(S_GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_b.set_ylabel('% of cells', fontsize=LABEL_SIZE)
    ax_b.legend(fontsize=LEGEND_SIZE, loc='upper right', frameon=False)
    ax_b.set_title('(b) Myeloid cell types per compartment', fontsize=11,
                   fontweight='bold', loc='left')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # (c) Checkpoint molecule expression on myeloid cells
    ax_c = fig.add_subplot(gs[1, 0])
    cp_colors = {'VISTA': '#8B008B', 'IDO': '#FF6347', 'HLA_DR': '#4682B4',
                 'CD163': '#006400', 'CD206': '#228B22'}
    cp_labels = {'VISTA': 'VISTA', 'IDO': 'IDO', 'HLA_DR': 'HLA-DR',
                 'CD163': 'CD163', 'CD206': 'CD206'}
    for mname, color in cp_colors.items():
        if mname in checkpoint_expr:
            ax_c.plot(x_pos, checkpoint_expr[mname], 'o-', color=color,
                      ms=6, lw=2, label=cp_labels[mname], alpha=0.85)
    ax_c.axvline(3.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(S_GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_c.set_ylabel('Mean expression (z-scored)', fontsize=10)
    ax_c.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax_c.set_title('(c) Checkpoint/suppression markers on myeloid cells',
                   fontsize=11, fontweight='bold', loc='left')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # (d) T cell fractions for context (shows exclusion pattern)
    ax_d = fig.add_subplot(gs[1, 1])
    w = 0.3
    ax_d.bar(x_pos - w/2, t_fracs['CD4 T'] * 100, w, color='#4169E1',
             label='CD4 T', alpha=0.85, edgecolor='white')
    ax_d.bar(x_pos + w/2, t_fracs['CD8 T'] * 100, w, color='#1E90FF',
             label='CD8 T', alpha=0.85, edgecolor='white')
    # Overlay total myeloid as line
    total_mye = sum(mye_fracs[g] for g in mye_colors) * 100
    ax_d.plot(x_pos, total_mye, 'H-', color='#006400', ms=8, lw=2,
              label='Total myeloid', alpha=0.8)
    ax_d.axvline(3.5, color='#999', ls='--', lw=1, alpha=0.7)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(S_GRADIENT_SHORT, fontsize=TICK_SIZE, rotation=40, ha='right', rotation_mode='anchor')
    ax_d.set_ylabel('% of cells', fontsize=LABEL_SIZE)
    ax_d.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax_d.set_title('(d) T cells vs myeloid by compartment', fontsize=11,
                   fontweight='bold', loc='left')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # (e, f) Spatial ROI: left = compartments, right = myeloid subtypes
    example_rois = select_example_rois(data, S_GRADIENT_ORDER, n=1,
                                        min_diverse=3)
    def myeloid_ct_colors(ct_arr):
        colors = []
        for ct in ct_arr:
            if ct in ('M1 Macrophages',):
                colors.append('#228B22')       # bright green = M1
            elif ct in ('M2 Macrophages',):
                colors.append('#006400')       # dark green = M2
            elif ct in ('Macrophages',):
                colors.append('#8FBC8F')       # light green = other mac
            elif ct in ('Myeloid (S100A9+)',):
                colors.append('#DAA520')       # gold = S100A9+ myeloid
            elif ct in ('Dendritic cells', 'pDC'):
                colors.append('#9B59B6')       # purple = DC
            elif ct in ('CD4 T cells', 'CD8 T cells', 'Treg'):
                colors.append('#1E90FF')       # blue = T cells
            else:
                colors.append('#D3D3D3')       # everything else gray
        return colors
    ct_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#228B22',
               ms=6, label='M1 Mac'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#006400',
               ms=6, label='M2 Mac'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6',
               ms=6, label='DC / pDC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E90FF',
               ms=6, label='T cells'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D3D3D3',
               ms=6, label='B / other'),
    ]
    if example_rois:
        roi_id, roi_tma = example_rois[0]
        plot_paired_roi(fig, gs, 2, data, roi_id, roi_tma, ('e', 'f'),
                        S_GRADIENT_ORDER, S_GRADIENT_COLORS, S_GRADIENT_SHORT,
                        myeloid_ct_colors, ct_legend)

    out = os.path.join(output_dir, 'fig_ie_myeloid.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return mye_fracs


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Immune Evasion Archetypes
# ═══════════════════════════════════════════════════════════════════════════

def compute_roi_evasion_metrics(data, min_cells=500):
    """Compute per-ROI immune evasion metrics.

    Returns: dict with 'metric_matrix' (n_rois × n_metrics), 'metric_names',
             'roi_ids', 'tma_labels'.
    """
    comps = data['comps']
    ctypes = data['ctypes']
    sample_ids = data['sample_ids']
    tma = data['tma']
    X = data['X']
    midx = data['marker_idx']

    rois = np.unique(sample_ids)
    b_set = set(B_TYPES + ['B / Unassigned transitional', 'Weak CD20 border'])
    t_types = ['CD4 T cells', 'CD8 T cells', 'CD8 T exhausted',
               'CD8 T pre-exhausted (TOX+)', 'Treg']
    foll_comps = set(GRADIENT_ORDER[:5])  # first 5 = follicular

    metric_names = [
        'CD8 T\ninfiltration',
        'CD8\nexhaustion',
        'Treg\nfraction',
        'Treg:CD8eff\nratio',
        'E:S\nratio',
        'CD39+\nT cells',
        'Follicular\nfraction',
        'Macrophage\nfraction',
    ]

    rows = []
    roi_ids = []
    tma_labels = []

    for roi in rois:
        mask = (sample_ids == roi)
        n = mask.sum()
        if n < min_cells:
            continue

        ct_roi = ctypes[mask]
        comp_roi = comps[mask]
        t_roi = tma[mask][0]

        # 1. CD8 T infiltration (fraction of all cells)
        n_cd8_all = np.isin(ct_roi, CD8_ALL).sum()
        cd8_infilt = n_cd8_all / n

        # 2. CD8 exhaustion fraction
        if n_cd8_all >= 10:
            n_cd8_exh = np.isin(ct_roi, CD8_EXHAUSTED).sum()
            cd8_exh = n_cd8_exh / n_cd8_all
        else:
            cd8_exh = np.nan

        # 3. Treg fraction
        n_treg = (ct_roi == 'Treg').sum()
        treg_frac = n_treg / n

        # 4. Treg:CD8 effector ratio
        n_cd8_eff = np.isin(ct_roi, CD8_EFFECTOR).sum()
        if n_cd8_eff > 0:
            treg_cd8_r = n_treg / n_cd8_eff
        else:
            treg_cd8_r = np.nan

        # 5. E:S ratio
        n_eff = np.isin(ct_roi, EFFECTOR).sum()
        n_sup = np.isin(ct_roi, SUPPRESSOR).sum()
        if n_sup > 0:
            es = n_eff / n_sup
        else:
            es = 10.0 if n_eff > 0 else np.nan

        # 6. CD39+ fraction among T cells
        t_mask = mask & np.isin(ctypes, t_types)
        n_t = t_mask.sum()
        if n_t >= 10 and 'CD39' in midx:
            cd39_pos = (X[t_mask, midx['CD39']] > 0.5).sum() / n_t
        else:
            cd39_pos = np.nan

        # 7. Follicular fraction
        foll_frac = sum(1 for c in comp_roi if c in foll_comps) / n

        # 8. Macrophage fraction
        mac_frac = (ct_roi == 'Macrophages').sum() / n

        rows.append([cd8_infilt, cd8_exh, treg_frac, treg_cd8_r,
                     es, cd39_pos, foll_frac, mac_frac])
        roi_ids.append(roi)
        tma_labels.append(t_roi)

    mat = np.array(rows)
    return {
        'matrix': mat,
        'metric_names': metric_names,
        'roi_ids': roi_ids,
        'tma_labels': tma_labels,
    }


def generate_fig_evasion_archetypes(data, output_dir):
    """Generate immune evasion archetype clustermap."""
    print("\n=== Figure 5: Immune Evasion Archetypes ===")
    from sklearn.metrics import silhouette_score
    import warnings

    metrics = compute_roi_evasion_metrics(data)
    mat = metrics['matrix']
    names = metrics['metric_names']
    roi_ids = metrics['roi_ids']
    tma_labels = np.array(metrics['tma_labels'])
    n_rois, n_metrics = mat.shape
    print(f"  {n_rois} ROIs × {n_metrics} metrics")

    # Drop rows with any NaN — impute with column median first
    for j in range(n_metrics):
        col = mat[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            mat[nan_mask, j] = median_val

    # Log-transform E:S ratio and Treg:CD8 ratio (skewed)
    mat[:, 3] = np.log1p(mat[:, 3])   # Treg:CD8
    mat[:, 4] = np.log1p(mat[:, 4])   # E:S ratio

    # Z-score normalize each metric
    mat_z = np.zeros_like(mat)
    for j in range(n_metrics):
        mu, sd = mat[:, j].mean(), mat[:, j].std()
        mat_z[:, j] = (mat[:, j] - mu) / max(sd, 1e-10)

    # Clip extreme z-scores for visualization
    mat_z = np.clip(mat_z, -3, 3)

    # Hierarchical clustering
    dist = pdist(mat_z, 'euclidean')
    Z = linkage(dist, method='ward')

    # Optimal k by silhouette
    best_k, best_sil = 2, -1
    for k in range(2, 8):
        labels = fcluster(Z, t=k, criterion='maxclust')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sil = silhouette_score(mat_z, labels)
        if sil > best_sil:
            best_k, best_sil = k, sil
    print(f"  Optimal k={best_k}, silhouette={best_sil:.3f}")

    labels = fcluster(Z, t=best_k, criterion='maxclust')

    # Reorder rows by dendrogram
    from scipy.cluster.hierarchy import leaves_list
    leaf_order = leaves_list(Z)
    mat_z_ord = mat_z[leaf_order]
    mat_raw_ord = mat[leaf_order]
    labels_ord = labels[leaf_order]
    tma_ord = tma_labels[leaf_order]
    roi_ord = [roi_ids[i] for i in leaf_order]

    # --- Figure ---
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35,
                  height_ratios=[2.5, 1, 1],
                  left=0.08, right=0.95, top=0.95, bottom=0.04)

    # (a) Heatmap
    ax_heat = fig.add_subplot(gs[0, :])
    im = ax_heat.imshow(mat_z_ord, aspect='auto', cmap='RdBu_r',
                         vmin=-3, vmax=3, interpolation='nearest')

    # Column labels
    ax_heat.set_xticks(range(n_metrics))
    ax_heat.set_xticklabels([n.replace('\n', ' ') for n in names],
                             fontsize=9, rotation=45, ha='right')
    ax_heat.set_ylabel(f'ROIs (n={n_rois})', fontsize=11)

    # TMA color sidebar (left)
    tma_colors_map = {'A1': '#E74C3C', 'B1': '#3498DB', 'C1': '#2ECC71',
                      'Biomax': '#9B59B6'}
    tma_color_arr = [tma_colors_map.get(t, '#888') for t in tma_ord]
    for i, c in enumerate(tma_color_arr):
        ax_heat.plot(-0.8, i, 's', color=c, ms=2.5, clip_on=False)

    # Cluster color sidebar (right)
    cluster_cmap = plt.cm.Set2(np.linspace(0, 1, max(best_k, 3)))
    for i, cl in enumerate(labels_ord):
        ax_heat.plot(n_metrics - 0.2, i, 's',
                     color=cluster_cmap[cl - 1], ms=2.5, clip_on=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.5, pad=0.02)
    cbar.set_label('z-score', fontsize=LABEL_SIZE)

    # Y-axis: hide individual ROI labels (too many)
    ax_heat.set_yticks([])

    # TMA legend + cluster legend
    tma_legend = [Line2D([0], [0], marker='s', color='w',
                         markerfacecolor=tma_colors_map[t], ms=7, label=t)
                  for t in sorted(tma_colors_map.keys())]
    cl_legend = [Line2D([0], [0], marker='s', color='w',
                        markerfacecolor=cluster_cmap[i], ms=7,
                        label=f'Archetype {i+1}')
                 for i in range(best_k)]
    leg1 = ax_heat.legend(handles=tma_legend, loc='upper left',
                          title='TMA', fontsize=LEGEND_SIZE, title_fontsize=8,
                          frameon=True, framealpha=0.9)
    ax_heat.add_artist(leg1)
    ax_heat.legend(handles=cl_legend, loc='lower left',
                   title='Cluster', fontsize=LEGEND_SIZE, title_fontsize=8,
                   frameon=True, framealpha=0.9)

    ax_heat.set_title('(a) Immune evasion archetypes — per-ROI metric heatmap',
                      fontsize=13, fontweight='bold', loc='left')

    # (b) Cluster profiles — mean raw metric per cluster
    ax_prof = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(n_metrics)
    bar_w = 0.8 / best_k
    for ci in range(best_k):
        cl_mask = (labels == ci + 1)
        means = mat[cl_mask].mean(axis=0)
        # Normalize each metric to [0,1] range for comparability
        for j in range(n_metrics):
            col_min, col_max = mat[:, j].min(), mat[:, j].max()
            rng = col_max - col_min
            if rng > 0:
                means[j] = (means[j] - col_min) / rng
        offset = (ci - best_k/2 + 0.5) * bar_w
        ax_prof.bar(x_pos + offset, means, bar_w,
                    color=cluster_cmap[ci], label=f'A{ci+1}',
                    edgecolor='white', linewidth=0.5, alpha=0.85)
    ax_prof.set_xticks(x_pos)
    ax_prof.set_xticklabels([n.replace('\n', ' ') for n in names],
                             fontsize=7, rotation=45, ha='right')
    ax_prof.set_ylabel('Normalized mean', fontsize=LABEL_SIZE)
    ax_prof.legend(fontsize=LEGEND_SIZE, frameon=False, ncol=best_k)
    ax_prof.set_title('(b) Archetype profiles (min-max normalized)',
                      fontsize=11, fontweight='bold', loc='left')
    ax_prof.spines['top'].set_visible(False)
    ax_prof.spines['right'].set_visible(False)

    # (c) TMA distribution per cluster
    ax_tma = fig.add_subplot(gs[1, 1])
    tma_order = sorted(tma_colors_map.keys())
    cluster_tma_counts = np.zeros((best_k, len(tma_order)))
    for ci in range(best_k):
        cl_mask = (labels == ci + 1)
        cl_tmas = tma_labels[cl_mask]
        for ti, t in enumerate(tma_order):
            cluster_tma_counts[ci, ti] = (cl_tmas == t).sum()
    # Stacked bar (clusters as x, TMAs as stacks)
    bottom = np.zeros(best_k)
    for ti, t in enumerate(tma_order):
        ax_tma.bar(range(best_k), cluster_tma_counts[:, ti], bottom=bottom,
                   color=tma_colors_map[t], label=t, edgecolor='white')
        bottom += cluster_tma_counts[:, ti]
    ax_tma.set_xticks(range(best_k))
    ax_tma.set_xticklabels([f'A{ci+1}' for ci in range(best_k)], fontsize=9)
    ax_tma.set_ylabel('Number of ROIs', fontsize=LABEL_SIZE)
    ax_tma.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax_tma.set_title('(c) TMA composition per archetype',
                      fontsize=11, fontweight='bold', loc='left')
    ax_tma.spines['top'].set_visible(False)
    ax_tma.spines['right'].set_visible(False)

    # (d) Spatial ROI examples — one per archetype (closest to centroid)
    ax_examples = fig.add_subplot(gs[2, :])
    ax_examples.axis('off')
    # Create sub-gridspec for archetype examples
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    n_show = min(best_k, 4)  # show up to 4
    gs_sub = GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[2, :],
                                     wspace=0.15)
    ct_color_fn = lambda ct_arr: [
        '#1E90FF' if ct in ('CD8 T cells', 'Macrophages (GzmB+)') else
        '#E74C3C' if ct in ('CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)') else
        '#2E8B57' if ct == 'Treg' else
        '#D3D3D3' if ct in b_set else '#E8E8E8'
        for ct in ct_arr
    ]
    b_set = set(B_TYPES + ['B / Unassigned transitional', 'Weak CD20 border'])
    ex_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E90FF',
               ms=5, label='CD8 effector'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               ms=5, label='CD8 exhausted'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E8B57',
               ms=5, label='Treg'),
    ]

    for ci in range(n_show):
        cl_mask = (labels == ci + 1)
        cl_indices = np.where(cl_mask)[0]
        centroid = mat_z[cl_mask].mean(axis=0)
        dists = np.array([np.linalg.norm(mat_z[i] - centroid)
                          for i in cl_indices])
        best_idx = cl_indices[np.argmin(dists)]
        roi_id = roi_ids[best_idx]
        t_label = tma_labels[best_idx]

        ax_ex = fig.add_subplot(gs_sub[0, ci])
        mask = get_roi_mask(data, roi_id)
        cx_roi = data['cx'][mask]
        cy_roi = data['cy'][mask]
        ct_roi = data['ctypes'][mask]
        colors_roi = ct_color_fn(ct_roi)

        n_eff = np.isin(ct_roi, EFFECTOR).sum()
        n_sup = np.isin(ct_roi, SUPPRESSOR).sum()
        es_val = n_eff / max(n_sup, 1)

        lbl = (f'A{ci+1}: {roi_id}\n[{t_label}] E:S={es_val:.1f} '
               f'({int(cl_mask.sum())} ROIs)')
        plot_spatial_scatter(ax_ex, cx_roi, cy_roi, colors_roi, lbl,
                             ms=0.5,
                             legend_handles=ex_legend if ci == 0 else None)

    # Print cluster summary
    print(f"\nArchetype summary (k={best_k}):")
    raw_names_short = ['CD8inf', 'CD8exh', 'Treg', 'Treg:CD8',
                       'E:S', 'CD39+', 'Foll%', 'Mac%']
    for ci in range(best_k):
        cl_mask = (labels == ci + 1)
        n_cl = cl_mask.sum()
        cl_tmas = tma_labels[cl_mask]
        tma_str = ', '.join(f'{t}:{(cl_tmas==t).sum()}'
                            for t in tma_order if (cl_tmas == t).sum() > 0)
        means = mat[cl_mask].mean(axis=0)
        print(f"\n  Archetype {ci+1} (n={n_cl}): {tma_str}")
        for j, mn in enumerate(raw_names_short):
            print(f"    {mn}: {means[j]:.3f}")

    out = os.path.join(output_dir, 'fig_ie_archetypes.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return metrics, labels, best_k


def main():
    parser = argparse.ArgumentParser(description='Immune evasion analysis')
    parser.add_argument('--t-utag', required=True,
                        help='T-panel UTAG merged h5ad')
    parser.add_argument('--s-utag', required=True,
                        help='S-panel UTAG merged h5ad')
    parser.add_argument('--output-dir', default='output/hypotheses_v8',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    t_data = load_t_panel(args.t_utag)
    s_data = load_s_panel(args.s_utag)

    # Generate all 4 figures
    print("\n" + "="*70)
    print("IMMUNE EVASION SPATIAL ANALYSIS")
    print("="*70)

    # Load tonsil T-panel data for FL vs tonsil comparison row
    from fig_tonsil_comparison import extract_t_panel as extract_tonsil_t
    tonsil_t_data = extract_tonsil_t(args.t_utag)
    print(f"  Tonsil T-panel: {tonsil_t_data['tonsil_mask'].sum():,} tonsil cells, "
          f"{tonsil_t_data['tumor_mask'].sum():,} FL cells")

    r1 = generate_fig_gradient(t_data, args.output_dir,
                                tonsil_t_data=tonsil_t_data)
    r2 = generate_fig_treg_barrier(t_data, args.output_dir)
    r3 = generate_fig_pressure(t_data, args.output_dir)
    r4 = generate_fig_myeloid(s_data, args.output_dir)
    r5 = generate_fig_evasion_archetypes(t_data, args.output_dir)

    print("\n" + "="*70)
    print("All immune evasion figures generated successfully.")
    print("="*70)


if __name__ == '__main__':
    main()
