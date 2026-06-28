#!/usr/bin/env python3
"""Generate Figure 7: dual-compartment immune evasion model.

Programmatic creation for precise control over cell placement and annotations.
4 concentric zones: Activated FDC network → B cell zone → Treg barrier → Interfollicular.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, RegularPolygon
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpl_patches
import numpy as np
from pathlib import Path

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22


OUT = Path("output/hypotheses_v8")

# ── Colors (matching the Gemini version the user liked) ──
C_BG_INTER = "#D6E4F0"       # interfollicular zone (light blue-gray)
C_TREG_RING = "#FFF3B0"      # Treg barrier (light gold)
C_TREG_EDGE = "#D4C060"      # Treg barrier edge
C_BCELL_ZONE = "#C8E6C0"     # B cell zone (light green)
C_BCELL_EDGE = "#6AAF6A"     # B cell zone edge
C_FDC_ZONE = "#F5DEB3"       # Activated FDC network (warm tan)
C_FDC_EDGE = "#C49A60"       # FDC zone edge

C_TUMOR_B = "#2E7D32"        # BCL2+ tumor B (dark green)
C_FDC = "#DAA520"             # CD14+ FDC (goldenrod)
C_FDC_EDGE_C = "#8B6914"     # FDC edge color
C_M2 = "#8B0000"              # M2 Mac (dark red)
C_EXH_CD8 = "#1B2A4A"        # exhausted CD8 T (dark navy)
C_EFF_CD8 = "#4169E1"        # effector CD8 T (royal blue)
C_CD4 = "#B07030"            # CD4 T / Tfh (warm brown)
C_TREG_CELL = "#E8963A"      # Treg cells (amber/orange)
C_MDSC = "#C71585"            # S100A9+ MDSC (medium violet red)

C_ARROW_INHIB = "#C0392B"    # inhibitory arrows (red)
C_ARROW_SURV = "#2D8B46"     # survival signal arrows (green)
C_CXCL13 = "#8B5E3C"         # CXCL13 chemokine (warm brown)


def _make_arm_polygon(cx, cy, angle, length, width, curve, n_pts=30):
    """Create one curved arm as a Shapely Polygon (thin tapered ribbon)."""
    from shapely.geometry import Polygon as ShapelyPolygon
    t = np.linspace(0, 1, n_pts)
    # Curved centerline
    arm_x = cx + t * length * np.cos(angle) + curve * np.sin(np.pi * t) * (-np.sin(angle))
    arm_y = cy + t * length * np.sin(angle) + curve * np.sin(np.pi * t) * np.cos(angle)
    # Width tapers from base to tip
    half_w = width * (1.0 - 0.75 * t)
    # Normal at each point
    dx = np.gradient(arm_x)
    dy = np.gradient(arm_y)
    norm = np.sqrt(dx**2 + dy**2)
    norm[norm == 0] = 1
    nx, ny = -dy / norm, dx / norm
    # Left and right edges
    lx = arm_x + nx * half_w
    ly = arm_y + ny * half_w
    rx = arm_x - nx * half_w
    ry = arm_y - ny * half_w
    xs = np.concatenate([lx, rx[::-1]])
    ys = np.concatenate([ly, ry[::-1]])
    coords = list(zip(xs, ys))
    try:
        poly = ShapelyPolygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return None


def draw_fdc(ax, x, y, size=0.035, color=C_FDC, zorder=10, seed=None):
    """Draw an FDC as one seamless shape: Shapely union of body + curved arms."""
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPathLocal
    rng = np.random.RandomState(seed)

    body_r = size * 0.38

    # Blobby body
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    radii = body_r * (1.0 + 0.1 * np.sin(3 * angles + rng.uniform(0, np.pi))
                      + 0.06 * np.cos(5 * angles + rng.uniform(0, np.pi)))
    body_coords = list(zip(x + radii * np.cos(angles), y + radii * np.sin(angles)))
    body = ShapelyPolygon(body_coords)

    # Curved arms
    n_arms = rng.randint(5, 8)
    arm_angles = np.linspace(0, 2 * np.pi, n_arms, endpoint=False)
    arm_angles += rng.uniform(-0.3, 0.3, n_arms)
    parts = [body]
    for a in arm_angles:
        length = size * rng.uniform(0.8, 1.3)
        width = size * 0.12
        curve = size * rng.uniform(-0.3, 0.3)
        arm = _make_arm_polygon(x, y, a, length, width, curve)
        if arm is not None and arm.area > 0:
            parts.append(arm)

    # Union into one shape, then smooth edges
    unified = unary_union(parts)
    smooth = size * 0.015
    unified = unified.buffer(smooth, resolution=16).buffer(-smooth * 0.5, resolution=16)

    # Convert exterior to matplotlib patch
    ext = np.array(unified.exterior.coords)
    codes = [MplPathLocal.MOVETO] + [MplPathLocal.LINETO] * (len(ext) - 2) + [MplPathLocal.CLOSEPOLY]
    path = MplPathLocal(ext, codes)
    patch = PathPatch(path, facecolor=color, edgecolor=C_FDC_EDGE_C,
                      linewidth=0.6, zorder=zorder, alpha=0.95)
    ax.add_patch(patch)

    # Nucleus — proportionally sized to match other cells (~40% of body)
    nuc_r = body_r * 0.85
    nuc = mpatches.Ellipse((x + body_r * 0.08, y - body_r * 0.05),
                           nuc_r * 1.3, nuc_r,
                           angle=rng.uniform(0, 360),
                           facecolor="#7A5A10", edgecolor="#5A4008",
                           linewidth=0.4, zorder=zorder + 1, alpha=0.85)
    ax.add_patch(nuc)


def draw_m2_mac(ax, x, y, size=0.024, color=C_M2, edgecolor=None, zorder=5, alpha=1.0, seed=None):
    """Draw an M2 macrophage: MDSC-like star vertices smoothed with cubic spline → organic blob."""
    import colorsys
    from scipy.interpolate import CubicSpline
    rng = np.random.RandomState(seed)
    if edgecolor is None:
        r, g, b, _ = plt.matplotlib.colors.to_rgba(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0, l - 0.15)
        edgecolor = colorsys.hls_to_rgb(h, l, s)
    # Same star-point generation as MDSC
    n_spikes = rng.randint(7, 11)
    angles = np.linspace(0, 2 * np.pi, n_spikes * 2, endpoint=False)
    angles += rng.uniform(-0.1, 0.1, n_spikes * 2)
    base_r = size * 1.0
    raw_x, raw_y = [], []
    for i, a in enumerate(angles):
        if i % 2 == 0:
            r_ = base_r * rng.uniform(1.1, 1.45)
        else:
            r_ = base_r * rng.uniform(0.55, 0.7)
        raw_x.append(x + r_ * np.cos(a))
        raw_y.append(y + r_ * np.sin(a))
    # Close the loop for periodic spline
    raw_x.append(raw_x[0])
    raw_y.append(raw_y[0])
    # Parameterize by cumulative arc length
    t = np.zeros(len(raw_x))
    for i in range(1, len(raw_x)):
        t[i] = t[i - 1] + np.sqrt((raw_x[i] - raw_x[i - 1])**2 +
                                    (raw_y[i] - raw_y[i - 1])**2)
    # Periodic cubic spline interpolation
    cs_x = CubicSpline(t, raw_x, bc_type="periodic")
    cs_y = CubicSpline(t, raw_y, bc_type="periodic")
    t_fine = np.linspace(0, t[-1], 200)
    sx = cs_x(t_fine)
    sy = cs_y(t_fine)
    # Draw as polygon
    from matplotlib.patches import Polygon
    poly = Polygon(np.column_stack([sx, sy]), closed=True,
                   facecolor=color, edgecolor=edgecolor,
                   linewidth=0.8, zorder=zorder, alpha=alpha)
    ax.add_patch(poly)
    # Nucleus — darker, eccentric
    r, g, b, _ = plt.matplotlib.colors.to_rgba(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, l - 0.25)
    s = min(1.0, s * 1.1)
    nuc_color = colorsys.hls_to_rgb(h, l, s)
    nuc = mpatches.Ellipse((x + size * 0.15, y + size * 0.05),
                            width=size * 0.85, height=size * 0.65,
                            angle=rng.uniform(-30, 30), facecolor=nuc_color,
                            edgecolor=edgecolor, linewidth=0.4,
                            zorder=zorder + 0.001, alpha=alpha)
    ax.add_patch(nuc)


def draw_mdsc(ax, x, y, size=0.024, color=C_MDSC, edgecolor=None, zorder=5, alpha=1.0, seed=None):
    """Draw an MDSC with spiky/star-like edges."""
    import colorsys
    rng = np.random.RandomState(seed)
    if edgecolor is None:
        r, g, b, _ = plt.matplotlib.colors.to_rgba(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0, l - 0.15)
        edgecolor = colorsys.hls_to_rgb(h, l, s)
    # Star shape: alternate between outer spikes and inner valleys
    n_spikes = rng.randint(7, 11)
    angles = np.linspace(0, 2 * np.pi, n_spikes * 2, endpoint=False)
    angles += rng.uniform(-0.1, 0.1, n_spikes * 2)
    base_r = size * 1.0
    verts = []
    codes = []
    for i, a in enumerate(angles):
        if i % 2 == 0:
            # Spike tip (outer)
            r_ = base_r * rng.uniform(1.1, 1.45)
        else:
            # Valley (inner)
            r_ = base_r * rng.uniform(0.55, 0.7)
        px = x + r_ * np.cos(a)
        py = y + r_ * np.sin(a)
        if i == 0:
            verts.append((px, py))
            codes.append(MplPath.MOVETO)
        else:
            verts.append((px, py))
            codes.append(MplPath.LINETO)
    verts.append(verts[0])
    codes.append(MplPath.CLOSEPOLY)
    path = MplPath(verts, codes)
    patch = mpl_patches.PathPatch(path, facecolor=color, edgecolor=edgecolor,
                                   linewidth=0.8, zorder=zorder, alpha=alpha)
    ax.add_patch(patch)
    # Nucleus — round, eccentric
    r, g, b, _ = plt.matplotlib.colors.to_rgba(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, l - 0.25)
    s = min(1.0, s * 1.1)
    nuc_color = colorsys.hls_to_rgb(h, l, s)
    nuc = mpatches.Ellipse((x + size * 0.12, y + size * 0.05),
                            width=size * 0.7, height=size * 0.8,
                            angle=rng.uniform(-15, 15), facecolor=nuc_color,
                            edgecolor=edgecolor, linewidth=0.4,
                            zorder=zorder + 0.001, alpha=alpha)
    ax.add_patch(nuc)


def draw_cell(ax, x, y, size=0.018, color="gray", edgecolor=None, zorder=5, alpha=1.0):
    """Draw a slightly oval cell with an eccentric oval nucleus offset to the right."""
    import colorsys
    if edgecolor is None:
        r, g, b, _ = plt.matplotlib.colors.to_rgba(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0, l - 0.15)
        edgecolor = colorsys.hls_to_rgb(h, l, s)
    # Cell body — slightly oval (taller than wide)
    c = mpatches.Ellipse((x, y), width=size * 1.9, height=size * 2.1,
                          facecolor=color, edgecolor=edgecolor,
                          linewidth=0.8, zorder=zorder, alpha=alpha)
    ax.add_patch(c)
    # Nucleus — darker oval, eccentric toward right
    r, g, b, _ = plt.matplotlib.colors.to_rgba(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, l - 0.25)
    s = min(1.0, s * 1.1)
    nuc_color = colorsys.hls_to_rgb(h, l, s)
    nuc = mpatches.Ellipse((x + size * 0.25, y + size * 0.05),
                            width=size * 0.75, height=size * 0.9,
                            angle=10, facecolor=nuc_color,
                            edgecolor=edgecolor, linewidth=0.4,
                            zorder=zorder + 0.001, alpha=alpha)
    ax.add_patch(nuc)


def _shorten(x1, y1, x2, y2, margin=0.018):
    """Shorten a line segment by `margin` at both ends (cell-center to cell-edge)."""
    dx, dy = x2 - x1, y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    if L < 2 * margin:
        return x1, y1, x2, y2
    ux, uy = dx / L, dy / L
    return (x1 + ux * margin, y1 + uy * margin,
            x2 - ux * margin, y2 - uy * margin)


def tbar_arrow(ax, x1, y1, x2, y2, color=C_ARROW_INHIB, lw=2.0):
    """Draw a T-bar inhibitory arrow: long line + perpendicular crossbar at target.

    No arrowhead — just a straight line ending with a short perpendicular bar (⊣).
    Coordinates are cell centers; the line is shortened to stop at cell edges.
    """
    sx1, sy1, sx2, sy2 = _shorten(x1, y1, x2, y2)
    # Main line (no arrowhead)
    ax.plot([sx1, sx2], [sy1, sy2], color=color, lw=lw, zorder=15,
            solid_capstyle="round")
    # Perpendicular crossbar at target end
    dx, dy = x2 - x1, y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    if L > 0:
        nx, ny = -dy / L, dx / L  # perpendicular unit vector
        bar_len = 0.014
        ax.plot([sx2 - nx * bar_len, sx2 + nx * bar_len],
                [sy2 - ny * bar_len, sy2 + ny * bar_len],
                color=color, lw=lw + 1.0, zorder=15, solid_capstyle="round")


def green_arrow(ax, x1, y1, x2, y2, color=C_ARROW_SURV, lw=2.0):
    """Draw a support arrow (green, pointed). Cell-center to cell-center."""
    sx1, sy1, sx2, sy2 = _shorten(x1, y1, x2, y2)
    ax.annotate("", xy=(sx2, sy2), xytext=(sx1, sy1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=15),
                zorder=15)


def make_figure():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-0.05, 1.20)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Center of the follicle — shifted left for legend space
    cx, cy = 0.42, 0.50

    # ── Zone 4: Interfollicular background ──
    bg = mpatches.Rectangle((-0.05, -0.05), 1.20, 1.10,
                            facecolor=C_BG_INTER, zorder=0)
    ax.add_patch(bg)

    # ── Zone 3: Treg barrier (thin ring) ──
    treg_outer = Circle((cx, cy), 0.38, facecolor=C_TREG_RING,
                         edgecolor=C_TREG_EDGE, linewidth=1.5, zorder=1,
                         alpha=0.7)
    ax.add_patch(treg_outer)

    # ── Zone 2: B cell zone ──
    bcell_zone = Circle((cx, cy), 0.33, facecolor=C_BCELL_ZONE,
                         edgecolor=C_BCELL_EDGE, linewidth=1.0, zorder=2)
    ax.add_patch(bcell_zone)

    # ── Zone 1: Activated FDC network (innermost) ──
    fdc_zone = Circle((cx, cy), 0.18, facecolor=C_FDC_ZONE,
                       edgecolor=C_FDC_EDGE, linewidth=1.0, zorder=2)
    ax.add_patch(fdc_zone)

    # ── Zone labels (with white background to avoid cell overlap) ──
    _tbbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75,
                  edgecolor="none")
    ax.text(cx, cy + 0.145, "Activated FDC\nnetwork", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#7A5520", zorder=20,
            fontstyle="italic", bbox=_tbbox)
    ax.text(cx + 0.27, cy + 0.20, "B cell zone", ha="center", va="center",
            fontsize=9, fontstyle="italic", color="#3A6B3A", zorder=20,
            bbox=_tbbox)
    ax.text(cx - 0.08, cy + 0.41, "Treg barrier", ha="center", va="bottom",
            fontsize=10, fontstyle="italic", color="#8A7A20", zorder=20,
            bbox=_tbbox)
    ax.text(0.05, 0.97, "Interfollicular zone", ha="left", va="top",
            fontsize=12, fontweight="bold", color="#4A5A6A", zorder=20)

    # ── Treg cells (ring at r ~ 0.355) ──
    n_treg = 28
    for i in range(n_treg):
        angle = 2 * np.pi * i / n_treg + 0.05
        r = 0.355
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        draw_cell(ax, x, y, size=0.020, color=C_TREG_CELL, zorder=4 + i * 0.01)

    # ── Tumor B cells (B cell zone, r ~ 0.19-0.31) ──
    np.random.seed(42)
    n_b = 70
    for i in range(n_b):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.19, 0.31)
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        draw_cell(ax, x, y, size=0.020, color=C_TUMOR_B, zorder=5 + i * 0.01)

    # ── FDC cells with dendritic processes (inner zone) ──
    fdc_positions = [
        (cx - 0.08, cy + 0.02),
        (cx + 0.06, cy + 0.04),
        (cx + 0.00, cy - 0.07),
        (cx + 0.10, cy - 0.04),
        (cx - 0.04, cy - 0.12),
    ]
    for i, (fx, fy) in enumerate(fdc_positions):
        draw_fdc(ax, fx, fy, size=0.045, zorder=10, seed=100 + i)

    # ── CXCL13 molecules: 3 small dots near one FDC, with label ──
    fx_c, fy_c = fdc_positions[0]  # top-left FDC
    cxcl13_dots = [
        (fx_c - 0.035, fy_c + 0.025),
        (fx_c - 0.050, fy_c + 0.010),
        (fx_c - 0.040, fy_c - 0.010),
    ]
    for dx, dy in cxcl13_dots:
        ax.add_patch(mpatches.Circle((dx, dy), 0.006, facecolor=C_CXCL13,
                                     edgecolor="#5C3D28", linewidth=0.5,
                                     zorder=11))
    # Label next to the dots
    ax.text(cxcl13_dots[1][0] - 0.02, cxcl13_dots[1][1],
            "CXCL13", fontsize=7, fontstyle="italic", color=C_CXCL13,
            fontweight="bold", ha="right", va="center", zorder=20,
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                      alpha=0.75, edgecolor="none"))

    # ── M2 Mac (in FDC zone) ──
    m2_positions = [
        (cx - 0.04, cy - 0.04),
        (cx + 0.08, cy + 0.00),
        (cx - 0.02, cy + 0.08),
    ]
    for i, (mx, my) in enumerate(m2_positions):
        draw_m2_mac(ax, mx, my, size=0.024, color=C_M2, zorder=8, seed=200 + i)

    # ── Tumor B cells in FDC zone (interspersed with FDCs, in gaps) ──
    b_fdc_zone = [
        (cx - 0.14, cy - 0.02),   # left gap, below CXCL13 dots
        (cx + 0.06, cy - 0.13),   # bottom gap, between two FDCs
        (cx + 0.15, cy - 0.01),   # right edge gap
    ]
    for bx, by in b_fdc_zone:
        draw_cell(ax, bx, by, size=0.018, color=C_TUMOR_B, zorder=9)

    # ── Exhausted CD8 T cells (in FDC zone, rare) ──
    exh_positions = [
        (cx + 0.13, cy + 0.08),
        (cx - 0.12, cy - 0.06),
    ]
    for ex, ey in exh_positions:
        draw_cell(ax, ex, ey, size=0.023, color=C_EXH_CD8, zorder=9)

    # ── Effector CD8 T cells (interfollicular — all must be outside Treg barrier r=0.38) ──
    eff_positions = [
        (0.02, 0.68), (0.05, 0.28), (0.75, 0.78),
        (0.72, 0.18), (0.12, 0.88), (0.80, 0.45),
        (0.05, 0.12), (0.55, 0.03), (0.35, 0.92),
    ]
    for ex, ey in eff_positions:
        draw_cell(ax, ex, ey, size=0.023, color=C_EFF_CD8, zorder=6)

    # ── CD4 T / Tfh cells (a few in B cell zone + interfollicular) ──
    cd4_positions = [
        (cx + 0.20, cy - 0.20),   # in B cell zone
        (cx - 0.22, cy + 0.18),   # in B cell zone
        (0.10, 0.50),             # interfollicular
        (0.68, 0.85),             # interfollicular
    ]
    for cx4, cy4 in cd4_positions:
        draw_cell(ax, cx4, cy4, size=0.020, color=C_CD4, zorder=6)

    # ── S100A9+ MDSCs (interfollicular) ──
    mdsc_positions = [
        (0.05, 0.78), (0.02, 0.35),
        (0.50, 0.06), (0.72, 0.13),
        (0.82, 0.55), (0.68, 0.92),
    ]
    for i, (mx, my) in enumerate(mdsc_positions):
        draw_mdsc(ax, mx, my, size=0.024, color=C_MDSC, zorder=7, seed=300 + i)

    # ── KEY INTERACTIONS (4 arrows, all cell-center to cell-center) ──

    # Arrow 1: FDC → exhausted CD8 T (VISTA) — inside FDC zone
    fx, fy = fdc_positions[1]
    ex, ey = exh_positions[0]
    tbar_arrow(ax, fx, fy, ex, ey)
    mid_x, mid_y = (fx + ex) / 2, (fy + ey) / 2
    _abbox = dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8,
                  edgecolor="none")
    ax.text(mid_x + 0.04, mid_y + 0.02,
            "VISTA", fontsize=8, color=C_ARROW_INHIB,
            fontweight="bold", ha="center", zorder=20, bbox=_abbox)

    # Arrow 2: M2 Mac → exhausted CD8 T (VISTA) — inside FDC zone
    mx2, my2 = m2_positions[0]
    ex2, ey2 = exh_positions[1]
    tbar_arrow(ax, mx2, my2, ex2, ey2)
    mid_x2, mid_y2 = (mx2 + ex2) / 2, (my2 + ey2) / 2
    ax.text(mid_x2 - 0.04, mid_y2 + 0.01,
            "VISTA", fontsize=8, color=C_ARROW_INHIB,
            fontweight="bold", ha="center", zorder=20, bbox=_abbox)

    # Arrows 3a-c: ONE FDC → 3 nearest tumor B cells (survival signals)
    fx3, fy3 = fdc_positions[3]  # bottom-right FDC (0.52, 0.46)
    b_targets = [
        (cx + 0.18, cy - 0.08),   # close right
        (cx + 0.16, cy - 0.15),   # close lower-right
        (cx + 0.08, cy - 0.19),   # close below
    ]
    for btx, bty in b_targets:
        draw_cell(ax, btx, bty, size=0.020, color=C_TUMOR_B, zorder=5)
        green_arrow(ax, fx3, fy3, btx, bty)
    # Label right next to the arrow source FDC, in the tan FDC zone
    ax.text(fx3 + 0.03, fy3 - 0.04, "Survival\nsignals", fontsize=8,
            color=C_ARROW_SURV, fontweight="bold", ha="left", va="top", zorder=20,
            bbox=_abbox)

    # Arrow 4: MDSC → effector CD8 T (VISTA) — interfollicular
    mdsc_x, mdsc_y = mdsc_positions[0]   # top-left MDSC
    eff_x, eff_y = eff_positions[0]      # nearby effector CD8 T
    tbar_arrow(ax, mdsc_x, mdsc_y, eff_x, eff_y)
    mid_x4, mid_y4 = (mdsc_x + eff_x) / 2, (mdsc_y + eff_y) / 2
    ax.text(mid_x4 - 0.05, mid_y4 + 0.02,
            "VISTA", fontsize=8, color=C_ARROW_INHIB,
            fontweight="bold", ha="center", zorder=20, bbox=_abbox)

    # ── LEGEND ──
    legend_x = 0.92
    legend_top = 0.92
    legend_spacing = 0.075

    ax.text(legend_x + 0.03, legend_top + 0.05, "LEGEND", fontsize=11,
            fontweight="bold", ha="left", va="top", zorder=20)

    legend_items = [
        ("CD14⁺ FDC", C_FDC, "fdc"),
        ("BCL2⁺ tumor B", C_TUMOR_B, "circle"),
        ("M2 Mac", C_M2, "m2"),
        ("Exhausted CD8 T", C_EXH_CD8, "circle"),
        ("Effector CD8 T", C_EFF_CD8, "circle"),
        ("CD4 T / Tfh", C_CD4, "circle"),
        ("Treg", C_TREG_CELL, "circle"),
        ("S100A9⁺ MDSC", C_MDSC, "mdsc"),
    ]

    for i, (label, color, shape) in enumerate(legend_items):
        ly = legend_top - i * legend_spacing
        if shape == "fdc":
            draw_fdc(ax, legend_x + 0.01, ly, size=0.020, color=color, zorder=20, seed=999)
        elif shape == "m2":
            draw_m2_mac(ax, legend_x + 0.01, ly, size=0.018, color=color, zorder=20, seed=998)
        elif shape == "mdsc":
            draw_mdsc(ax, legend_x + 0.01, ly, size=0.018, color=color, zorder=20, seed=997)
        else:
            draw_cell(ax, legend_x + 0.01, ly, size=0.018, color=color, zorder=20)
        ax.text(legend_x + 0.045, ly, label, fontsize=9, va="center",
                ha="left", zorder=20)

    # Arrow legend — T-bar for suppression
    arr_y = legend_top - len(legend_items) * legend_spacing - 0.02
    # Line
    ax.plot([legend_x, legend_x + 0.035], [arr_y, arr_y],
            color=C_ARROW_INHIB, lw=2.0, zorder=20, solid_capstyle="round")
    # Crossbar at end
    bar_len = 0.012
    ax.plot([legend_x + 0.035, legend_x + 0.035], [arr_y - bar_len, arr_y + bar_len],
            color=C_ARROW_INHIB, lw=3.0, zorder=20, solid_capstyle="round")
    ax.text(legend_x + 0.05, arr_y, "Suppression", fontsize=8,
            va="center", color=C_ARROW_INHIB, zorder=20)

    arr_y2 = arr_y - 0.045
    ax.annotate("", xy=(legend_x + 0.04, arr_y2), xytext=(legend_x, arr_y2),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW_SURV, lw=1.5),
                zorder=20)
    ax.text(legend_x + 0.05, arr_y2, "Support", fontsize=8,
            va="center", color=C_ARROW_SURV, zorder=20)

    # ── Save ──
    png_path = OUT / "fig_model_immune_evasion.png"
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(png_path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path} + PDF")


if __name__ == "__main__":
    make_figure()
