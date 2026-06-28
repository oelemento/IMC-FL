"""
Visualization tools for Hyperion IMC data.

Includes:
- Single-channel and multi-channel composite plotting (from 3D image arrays)
- Raw IMC TXT file loading and composite building (pixel-level)
- Scatter + composite inset figure builder (cell-level + pixel-level)

Raw IMC workflow
----------------
1. ``load_raw_channel`` reads one marker from a Hyperion TXT export.
2. ``find_raw_file`` locates the TXT file for a given ROI.
3. ``build_raw_composite`` creates an additive RGB composite from
   multiple raw channels, optionally cropped to a window.
4. ``plot_scatter_composite_inset`` draws a full-core cell scatter
   (left) linked to a raw-IMC composite zoom (right).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional
import matplotlib.patches as mpatches


# Custom colormaps for IMC visualization
CMAPS = {
    'cyan': LinearSegmentedColormap.from_list('cyan', ['black', 'cyan']),
    'magenta': LinearSegmentedColormap.from_list('magenta', ['black', 'magenta']),
    'yellow': LinearSegmentedColormap.from_list('yellow', ['black', 'yellow']),
    'green': LinearSegmentedColormap.from_list('green', ['black', 'lime']),
    'red': LinearSegmentedColormap.from_list('red', ['black', 'red']),
    'blue': LinearSegmentedColormap.from_list('blue', ['black', 'blue']),
    'orange': LinearSegmentedColormap.from_list('orange', ['black', 'orange']),
    'white': LinearSegmentedColormap.from_list('white', ['black', 'white']),
}

# Default colors for common markers
MARKER_COLORS = {
    'DNA1': 'blue',
    'DNA2': 'blue',
    'CD3': 'green',
    'CD4': 'cyan',
    'CD8a': 'magenta',
    'CD20': 'yellow',
    'CD68': 'orange',
    'PD_1': 'red',
    'FoxP3': 'white',
}


def normalize_image(img: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Normalize image to 0-1 range using percentile clipping."""
    vmin = np.percentile(img, 100 - percentile)
    vmax = np.percentile(img, percentile)
    if vmax > vmin:
        img_norm = (img - vmin) / (vmax - vmin)
    else:
        img_norm = np.zeros_like(img)
    return np.clip(img_norm, 0, 1)


def plot_single_channel(
    image: np.ndarray,
    markers: list[str],
    marker: str,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    percentile: float = 99.5,
    title: bool = True,
    colorbar: bool = True,
) -> plt.Axes:
    """Plot a single marker channel."""
    if marker not in markers:
        raise ValueError(f"Marker '{marker}' not found")

    idx = markers.index(marker)
    channel = image[:, :, idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Use custom cmap if marker has default color
    if cmap == 'viridis' and marker in MARKER_COLORS:
        cmap = CMAPS.get(MARKER_COLORS[marker], cmap)
    elif cmap in CMAPS:
        cmap = CMAPS[cmap]

    # Normalize and display
    channel_norm = normalize_image(channel, percentile)
    im = ax.imshow(channel_norm, cmap=cmap)

    if title:
        ax.set_title(marker, fontsize=12)
    ax.axis('off')

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


def plot_marker_grid(
    image: np.ndarray,
    markers: list[str],
    markers_to_plot: Optional[list[str]] = None,
    ncols: int = 4,
    figsize: Optional[tuple] = None,
    percentile: float = 99.5,
) -> plt.Figure:
    """Plot a grid of marker channels."""
    if markers_to_plot is None:
        # Default: skip technical channels
        skip = {'80ArAr', '129Xe', '190BCKG', 'Pb204', 'Start_push', 'End_push'}
        markers_to_plot = [m for m in markers if m not in skip]

    n = len(markers_to_plot)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for i, marker in enumerate(markers_to_plot):
        plot_single_channel(
            image, markers, marker,
            ax=axes[i],
            percentile=percentile,
            colorbar=False,
        )

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def create_composite(
    image: np.ndarray,
    markers: list[str],
    channels: dict[str, str],
    percentile: float = 99.5,
) -> np.ndarray:
    """Create RGB composite from multiple channels.

    Args:
        image: 3D array (H, W, C)
        markers: list of marker names
        channels: dict mapping marker names to colors ('red', 'green', 'blue', 'cyan', 'magenta', 'yellow')

    Returns:
        RGB image as (H, W, 3) array
    """
    h, w = image.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    color_to_rgb = {
        'red': (1, 0, 0),
        'green': (0, 1, 0),
        'blue': (0, 0, 1),
        'cyan': (0, 1, 1),
        'magenta': (1, 0, 1),
        'yellow': (1, 1, 0),
        'white': (1, 1, 1),
        'orange': (1, 0.5, 0),
    }

    for marker, color in channels.items():
        if marker not in markers:
            print(f"Warning: marker '{marker}' not found, skipping")
            continue

        idx = markers.index(marker)
        channel = normalize_image(image[:, :, idx], percentile)

        rgb_vals = color_to_rgb.get(color, (1, 1, 1))
        for i, val in enumerate(rgb_vals):
            rgb[:, :, i] += channel * val

    # Clip to valid range
    rgb = np.clip(rgb, 0, 1)
    return rgb


def plot_composite(
    image: np.ndarray,
    markers: list[str],
    channels: dict[str, str],
    ax: Optional[plt.Axes] = None,
    percentile: float = 99.5,
    title: Optional[str] = None,
    legend: bool = True,
) -> plt.Axes:
    """Plot RGB composite image with legend."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    rgb = create_composite(image, markers, channels, percentile)
    ax.imshow(rgb)
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14)

    if legend:
        color_map = {
            'red': 'red', 'green': 'lime', 'blue': 'blue',
            'cyan': 'cyan', 'magenta': 'magenta', 'yellow': 'yellow',
            'white': 'white', 'orange': 'orange',
        }
        patches = [
            mpatches.Patch(color=color_map.get(c, c), label=m)
            for m, c in channels.items()
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=10)

    return ax


def plot_intensity_histograms(
    image: np.ndarray,
    markers: list[str],
    markers_to_plot: Optional[list[str]] = None,
    ncols: int = 4,
    figsize: Optional[tuple] = None,
    log_scale: bool = True,
) -> plt.Figure:
    """Plot intensity histograms for markers."""
    if markers_to_plot is None:
        skip = {'80ArAr', '129Xe', '190BCKG', 'Pb204'}
        markers_to_plot = [m for m in markers if m not in skip]

    n = len(markers_to_plot)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (3 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for i, marker in enumerate(markers_to_plot):
        idx = markers.index(marker)
        values = image[:, :, idx].flatten()
        values = values[values > 0]  # Remove zeros for log scale

        ax = axes[i]
        ax.hist(values, bins=50, color='steelblue', alpha=0.7)
        ax.set_title(marker, fontsize=10)
        ax.set_xlabel('Intensity')
        if log_scale and len(values) > 0:
            ax.set_yscale('log')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def quick_view(
    image: np.ndarray,
    markers: list[str],
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """Quick overview: DNA, key markers composite, and full grid."""
    fig = plt.figure(figsize=figsize)

    # DNA channel
    ax1 = fig.add_subplot(131)
    dna_marker = 'DNA1' if 'DNA1' in markers else markers[0]
    plot_single_channel(image, markers, dna_marker, ax=ax1, cmap='white', colorbar=False)
    ax1.set_title(f'{dna_marker} (nuclei)', fontsize=12)

    # Composite of key markers
    ax2 = fig.add_subplot(132)

    # Determine which markers are available
    composite_channels = {}
    if 'CD20' in markers:
        composite_channels['CD20'] = 'yellow'  # B cells
    if 'CD3' in markers:
        composite_channels['CD3'] = 'green'  # T cells
    if 'CD68' in markers:
        composite_channels['CD68'] = 'red'  # Macrophages
    if 'CD4' in markers and 'CD3' not in markers:
        composite_channels['CD4'] = 'cyan'
    if 'CD8a' in markers:
        composite_channels['CD8a'] = 'magenta'

    if composite_channels:
        plot_composite(image, markers, composite_channels, ax=ax2)
        ax2.set_title('Key markers', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No key markers found', ha='center', va='center')
        ax2.axis('off')

    # Histone for structure
    ax3 = fig.add_subplot(133)
    histone = 'HistoneH3' if 'HistoneH3' in markers else 'DNA2' if 'DNA2' in markers else markers[1]
    plot_single_channel(image, markers, histone, ax=ax3, cmap='cyan', colorbar=False)
    ax3.set_title(f'{histone}', fontsize=12)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Raw IMC TXT utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_channel(txt_path: str | Path, marker_col: str) -> np.ndarray:
    """Load a single marker channel from a Hyperion IMC TXT export.

    Args:
        txt_path: Path to the tab-separated TXT file.
        marker_col: Column name in the TXT header, e.g. ``"CD21(Er170Di)"``.

    Returns:
        2-D float32 array (H, W) reconstructed from pixel coordinates.
    """
    df = pd.read_csv(txt_path, sep="\t", usecols=["X", "Y", marker_col])
    x = df["X"].values.astype(int)
    y = df["Y"].values.astype(int)
    img = np.zeros((y.max() + 1, x.max() + 1), dtype=np.float32)
    img[y, x] = df[marker_col].values
    return img


def find_raw_file(
    roi_id: str,
    raw_dir: str | Path,
    tma_prefix: str = "B1_",
) -> Optional[Path]:
    """Locate the raw TXT file for a given ROI inside *raw_dir*.

    The ROI id is expected to start with *tma_prefix* (e.g. ``"B1_FL8"``).
    The function strips that prefix and searches for a filename containing
    ``_FL8_`` (or whichever suffix remains).

    Returns:
        Path to the matching file, or ``None`` if not found.
    """
    raw_dir = Path(raw_dir)
    fl_part = roi_id.replace(tma_prefix, "", 1) if roi_id.startswith(tma_prefix) else roi_id
    pattern = re.compile(rf"_{re.escape(fl_part)}_")
    for fname in os.listdir(raw_dir):
        if pattern.search(fname) and fname.endswith(".txt"):
            return raw_dir / fname
    return None


def build_raw_composite(
    raw_file: str | Path,
    channels: dict[str, str],
    colors: dict[str, np.ndarray],
    x0: int = 0,
    y0: int = 0,
    x1: Optional[int] = None,
    y1: Optional[int] = None,
    cofactor: float = 5.0,
    percentile: float = 99.0,
) -> np.ndarray:
    """Build an additive RGB composite from raw IMC channels.

    Args:
        raw_file: Path to the Hyperion TXT export.
        channels: Mapping from short name to TXT column name,
            e.g. ``{"CD21": "CD21(Er170Di)", "CD14": "CD14(Nd148Di)"}``.
        colors: Mapping from the same short names to RGB triplets
            (each an ``np.array([r, g, b])`` with values 0-1).
        x0, y0, x1, y1: Pixel crop window.  Pass ``x1=None, y1=None`` for
            the full image.
        cofactor: Arcsinh cofactor (default 5).
        percentile: Percentile for per-channel normalisation (default 99).

    Returns:
        (H, W, 3) float32 RGB array clipped to [0, 1].
    """
    imgs: dict[str, np.ndarray] = {}
    for name, col in channels.items():
        img = load_raw_channel(str(raw_file), col)
        if x1 is not None and y1 is not None:
            img = img[max(0, y0):min(img.shape[0], y1),
                       max(0, x0):min(img.shape[1], x1)]
        imgs[name] = np.arcsinh(img / cofactor)

    first = next(iter(imgs.values()))
    h, w = first.shape
    composite = np.zeros((h, w, 3), dtype=np.float32)
    for name, color in colors.items():
        img = imgs[name]
        vmax = np.percentile(img[img > 0], percentile) if np.any(img > 0) else 1.0
        norm = np.clip(img / max(vmax, 1e-6), 0, 1)
        composite += norm[:, :, np.newaxis] * color[np.newaxis, np.newaxis, :]
    return np.clip(composite, 0, 1)


def plot_scatter_composite_inset(
    fig: plt.Figure,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_type: np.ndarray,
    highlight_value: np.ndarray,
    highlight_threshold: float,
    composite_rgb: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    roi_label: str = "",
    window_size: Optional[int] = None,
    highlight_name: str = "FDC",
    scatter_layers: Optional[list[dict]] = None,
    composite_legend: Optional[list[dict]] = None,
    panel_letter: str = "",
) -> None:
    """Full-core cell scatter (left) + raw IMC composite zoom (right).

    The left panel shows all cells with optional highlight layers.
    The right panel shows the pre-built *composite_rgb* image with
    connector lines from the zoom rectangle.

    Args:
        fig: Matplotlib Figure to draw on (will create its own GridSpec).
        cell_x, cell_y: Centroid coordinates for all cells in the ROI.
        cell_type: Cell type labels (string array, same length).
        highlight_value: Continuous value used to split the highlight
            population (e.g. CD14 expression for FDCs).
        highlight_threshold: Value above which highlighted cells are
            "high" (e.g. CD14 Q75).
        composite_rgb: Pre-built (H, W, 3) RGB array from
            :func:`build_raw_composite`.
        x0, y0, x1, y1: Pixel coordinates of the zoom window.
        roi_label: Label printed in the scatter title.
        window_size: Size of zoom window (for title); inferred from
            ``x1 - x0`` if not given.
        highlight_name: Name of the highlighted population (default "FDC").
        scatter_layers: Optional list of dicts, each with keys
            ``mask``, ``color``, ``size``, ``alpha``, ``label``, ``zorder``,
            and optionally ``edgecolors``, ``linewidth``, ``marker``.
            If ``None``, a default FDC/Mac/CD8/B cell layer set is used.
        composite_legend: Optional list of dicts with ``color`` and ``label``
            for the composite panel legend. If ``None``, no legend is drawn.
        panel_letter: If non-empty, a bold ``(letter)`` label is placed at
            the top-left of the scatter axes.
    """
    if window_size is None:
        window_size = x1 - x0

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06,
                          left=0.04, right=0.96, top=0.90, bottom=0.04)

    # --- Left: full ROI scatter ---
    ax_roi = fig.add_subplot(gs[0, 0])

    if scatter_layers is None:
        is_hl = cell_type == highlight_name
        hl_hi = is_hl & (highlight_value >= highlight_threshold)
        hl_lo = is_hl & ~hl_hi
        mac = np.isin(cell_type, ["M1 Macrophages", "M2 Macrophages", "Macrophages"])
        cd8 = cell_type == "CD8 T cells"
        bcell = np.isin(cell_type, ["B cells (BCL2+)", "B cells (PAX5+)", "B cells"])
        other = ~(is_hl | mac | cd8 | bcell)
        scatter_layers = [
            dict(mask=other, color="#E0E0E0", size=0.5, alpha=0.2,
                 label=None, zorder=1),
            dict(mask=bcell, color="#4393C3", size=2, alpha=0.4,
                 label=f"B cells ({bcell.sum():,})", zorder=2),
            dict(mask=hl_lo, color="#FDDBC7", size=3, alpha=0.4,
                 edgecolors="gray", linewidth=0.1,
                 label=f"{highlight_name} lo ({hl_lo.sum():,})", zorder=3),
            dict(mask=cd8, color="#00BCD4", size=5, alpha=0.6,
                 edgecolors="black", linewidth=0.1,
                 label=f"CD8 T ({cd8.sum():,})", zorder=4),
            dict(mask=mac, color="#E41A1C", size=5, alpha=0.6,
                 edgecolors="black", linewidth=0.1,
                 label=f"Mac ({mac.sum():,})", zorder=4),
            dict(mask=hl_hi, color="#FFD700", size=10, alpha=0.85,
                 edgecolors="black", linewidth=0.3,
                 label=f"{highlight_name} hi ({hl_hi.sum():,})", zorder=5),
        ]

    for layer in scatter_layers:
        m = layer["mask"]
        kw = dict(c=layer["color"], s=layer["size"], alpha=layer["alpha"],
                  rasterized=True, zorder=layer.get("zorder", 1))
        if layer.get("label"):
            kw["label"] = layer["label"]
        if "edgecolors" in layer:
            kw["edgecolors"] = layer["edgecolors"]
        if "linewidth" in layer:
            kw["linewidth"] = layer["linewidth"]
        if "marker" in layer:
            kw["marker"] = layer["marker"]
        ax_roi.scatter(cell_x[m], cell_y[m], **kw)

    # Zoom rectangle (white + dashed black for contrast)
    rect = Rectangle((x0, y0), window_size, window_size,
                      linewidth=2.5, edgecolor="white", facecolor="none", zorder=10)
    ax_roi.add_patch(rect)
    rect2 = Rectangle((x0, y0), window_size, window_size,
                       linewidth=1.5, edgecolor="black", facecolor="none",
                       linestyle="--", zorder=10)
    ax_roi.add_patch(rect2)

    ax_roi.set_aspect("equal")
    ax_roi.invert_yaxis()
    if roi_label:
        ax_roi.set_title(f"{roi_label} — segmented cells", fontsize=11)
    ax_roi.legend(fontsize=6, loc="upper left", markerscale=2.5, framealpha=0.9)
    ax_roi.set_xlabel("x (µm)", fontsize=9)
    ax_roi.set_ylabel("y (µm)", fontsize=9)
    if panel_letter:
        ax_roi.text(-0.02, 1.02, rf"$\bf{{{panel_letter}}}$", transform=ax_roi.transAxes,
                    fontsize=22, va="bottom", ha="left")

    # --- Right: raw IMC composite ---
    ax_comp = fig.add_subplot(gs[0, 1])
    ax_comp.imshow(composite_rgb, origin="upper",
                   extent=[x0, x1, y1, y0])
    ax_comp.set_title(f"Raw IMC ({window_size}×{window_size} µm)", fontsize=11)
    ax_comp.set_xticks([]); ax_comp.set_yticks([])

    if composite_legend:
        legend_items = [
            Patch(facecolor=item["color"], label=item["label"])
            for item in composite_legend
        ]
        ax_comp.legend(handles=legend_items, fontsize=9, loc="upper left",
                       framealpha=0.9, facecolor="black", labelcolor="white",
                       edgecolor="white", handleheight=1.5, handlelength=3.0)

    # Connector lines from rectangle corners to inset edges
    for y_corner in [y0, y1]:
        con = ConnectionPatch(
            xyA=(x1, y_corner), coordsA=ax_roi.transData,
            xyB=(x0, y_corner), coordsB=ax_comp.transData,
            color="black", linewidth=1.0, linestyle=":", alpha=0.5, zorder=10)
        fig.add_artist(con)
