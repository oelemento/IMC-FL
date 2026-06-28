"""
Figure 1: Dataset overview and cell type annotation.

Portrait-oriented figure for publication. Panels:
  (a) Dataset schematic cartoon (full width)
  (b) T-panel UMAP by cell type | (c) S-panel UMAP by cell type
  (d) T-panel marker heatmap    | (e) S-panel marker heatmap
  (f) T-panel composition bars  | (g) S-panel composition bars

Direct-render into single GridSpec figure (vectorized PDF output).
"""

import argparse
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# Standardized font sizes — match figure_style.py
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSOLIDATE_MAP = {"T cells": "Other"}

CELL_TYPE_PALETTE = {
    "B cells":            "#1f77b4",
    "B cells (CD20hi)":   "#4393c3",
    "B cells (CXCR5hi)":  "#2166ac",
    "B cells (weak CD20)":"#92c5de",
    "B cells (TOXhi)":    "#053061",
    "B cells (BCL2+)":    "#4393c3",
    "B cells (PAX5+)":    "#2166ac",
    "GC B cells":         "#aec7e8",
    "Activated B / Plasmablast": "#dbdb8d",
    "CD4 T cells":        "#ff7f0e",
    "CD8 T cells":        "#d62728",
    "CD8 T pre-exhausted (TOX+)": "#e377c2",
    "CD8 T exhausted":    "#bcbd22",
    "Treg":               "#8c564b",
    "Macrophages (GzmB+)":  "#9467bd",
    "Macrophages":        "#7f7f7f",
    "FDC":                "#98df8a",
    "M1 Macrophages":     "#636363",
    "M2 Macrophages":     "#969696",
    "Dendritic cells":    "#e377c2",
    "Stromal / CAF":      "#8c564b",
    "Endothelial":        "#9467bd",
    "Myeloid (S100A9+)":  "#bcbd22",
    "FRC (PDPN+)":        "#dbdb8d",
    "Histiocytes (CD44hi)": "#c49c94",
    "Mixed / Border cells":"#c7c7c7",
    "pDC":                "#ffbb78",
    "Other":              "#c49c94",
    "Low quality / Unassigned": "#D3D3D3",
}

HEATMAP_MARKERS_T = [
    "CD20", "CD3", "CD4", "CD8a", "CD68", "FoxP3",
    "GranzymeB", "TOX", "PD_1", "CXCR5", "CD38", "IRF4",
]
HEATMAP_MARKERS_S = [
    "CD20", "CD4", "CD8a", "CD68", "CD21", "CXCL13", "PDPN",
    "Vimentin", "CD31", "CD34", "CD163", "CD206", "S100A9", "CD11c", "PAX5", "BCL_2",
]

LQ = "Low quality / Unassigned"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_array(f, key):
    ds = f["obs"][key]
    if isinstance(ds, h5py.Group) and "categories" in ds:
        cats = ds["categories"][:]
        codes = ds["codes"][:]
        cats_str = np.array(
            [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
        )
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


def get_marker_idx(f):
    key = "_index" if "_index" in f["var"] else "index"
    names = f["var"][key][:]
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return {n: i for i, n in enumerate(names)}


def is_tumor_core(sample_id):
    s_lower = sample_id.lower()
    if "_ton_" in s_lower or "_adr_" in s_lower:
        return False
    for tissue in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if tissue in s_lower:
            return False
    if sample_id == "Biomax_ROI_006":
        return False
    return True


def get_tumor_mask(sample_ids):
    return np.array([is_tumor_core(s) for s in sample_ids])


def consolidate_cell_types(cell_types):
    return np.array([CONSOLIDATE_MAP.get(ct, ct) for ct in cell_types])


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, f"$\\bf{{{letter}}}$",
            transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
            va="bottom", ha="left")






# ---------------------------------------------------------------------------
# Individual panel plot functions
# ---------------------------------------------------------------------------

def _plot_umap(ax, umap, cell_types, panel_name, all_ct_counts):
    """UMAP scatter for one panel."""
    n_total = len(cell_types)
    n_sub = min(100_000, n_total)
    rng = np.random.RandomState(42)
    idx = rng.choice(n_total, n_sub, replace=False)
    rng.shuffle(idx)
    u = umap[idx]
    ct_sub = cell_types[idx]

    is_lq = ct_sub == LQ
    if np.any(is_lq):
        ax.scatter(u[is_lq, 0], u[is_lq, 1], c="#D3D3D3", s=0.5,
                   alpha=0.3, edgecolors="none", rasterized=True, zorder=1)
    typed = ~is_lq
    if np.any(typed):
        colors = [CELL_TYPE_PALETTE.get(t, "#888888") for t in ct_sub[typed]]
        ax.scatter(u[typed, 0], u[typed, 1], c=colors, s=0.5,
                   edgecolors="none", rasterized=True, zorder=2)

    # Tight limits around data
    margin = 0.03
    x_range = u[:, 0].max() - u[:, 0].min()
    y_range = u[:, 1].max() - u[:, 1].min()
    ax.set_xlim(u[:, 0].min() - margin * x_range,
                u[:, 0].max() + margin * x_range)
    ax.set_ylim(u[:, 1].min() - margin * y_range,
                u[:, 1].max() + margin * y_range)
    ax.set_xlabel("UMAP 1", fontsize=LABEL_SIZE)
    ax.set_ylabel("UMAP 2", fontsize=LABEL_SIZE)
    ax.set_title(f"Cell types — {panel_name}", fontsize=TITLE_SIZE, fontweight="medium")
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    top = [t for t, _ in all_ct_counts.most_common() if t != LQ][:10]
    handles = [Patch(facecolor=CELL_TYPE_PALETTE.get(t, "#888"), label=t[:20])
               for t in top]
    handles.append(Patch(facecolor="#D3D3D3", alpha=0.5, label="Unidentified"))
    ax.legend(handles=handles, fontsize=LEGEND_SIZE, loc="upper right", ncol=2,
              framealpha=0.9, handlelength=1.0, handletextpad=0.3,
              columnspacing=0.5, borderpad=0.3)


def _plot_heatmap(ax, X_mem, cell_types, tumor, marker_idx, heatmap_markers, panel_name):
    """Marker expression heatmap for one panel."""
    ct_counts_tumor = Counter(cell_types[tumor])
    ct_order = [t for t, _ in ct_counts_tumor.most_common()
                if t != LQ and t != "T cells"]
    ct_order.append(LQ)

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

    # Z-score per marker
    for j in range(heatmap_data.shape[1]):
        col_data = heatmap_data[:, j]
        std = np.std(col_data)
        if std > 0:
            heatmap_data[:, j] = (col_data - np.mean(col_data)) / std

    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(marker_names_valid)))
    ax.set_xticklabels(marker_names_valid, rotation=45, ha="right", fontsize=TICK_SIZE)
    ct_labels = ["Unidentified" if t == LQ else t[:25] for t in ct_order]
    ax.set_yticks(range(len(ct_labels)))
    ax.set_yticklabels(ct_labels, fontsize=TICK_SIZE)
    ax.set_title(f"Marker expression — {panel_name}", fontsize=TITLE_SIZE, fontweight="medium")
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Z-score")
    cbar.set_label("Z-score", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)


def _plot_composition(ax, cell_types, tumor, panel_name):
    """Cell type composition bar chart for one panel."""
    ct_tumor = cell_types[tumor]
    n_tumor = len(ct_tumor)
    ct_counts_t = Counter(ct_tumor)
    ct_sorted = [t for t, _ in ct_counts_t.most_common() if t != LQ]
    ct_sorted.append(LQ)
    fracs = [ct_counts_t.get(t, 0) / n_tumor * 100 for t in ct_sorted]
    display_names = [t if t != LQ else "Unidentified" for t in ct_sorted]
    bar_colors = [CELL_TYPE_PALETTE.get(t, "#888888") for t in ct_sorted]

    y_pos = np.arange(len(ct_sorted))
    ax.barh(y_pos, fracs, color=bar_colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel("% of tumor cells", fontsize=LABEL_SIZE)
    ax.set_title(f"Composition — {panel_name} (n={n_tumor:,})",
                 fontsize=TITLE_SIZE, fontweight="medium")
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    for i, frac in enumerate(fracs):
        if frac > 1.5:
            ax.text(frac + 0.3, i, f"{frac:.1f}%", va="center", fontsize=ANNOT_SIZE)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def make_figure(t_panel_path, s_panel_path, cartoon_path, output_dir):
    """Create portrait-oriented Figure 1 — direct-render into single GridSpec."""
    print("Loading data...")
    f_t = h5py.File(t_panel_path, "r")
    f_s = h5py.File(s_panel_path, "r")
    X_t = f_t["X"][:]
    X_s = f_s["X"][:]

    # Extract data for both panels
    panel_data = []
    for f, X_mem, pname, hm in [
        (f_t, X_t, "T-panel", HEATMAP_MARKERS_T),
        (f_s, X_s, "S-panel", HEATMAP_MARKERS_S),
    ]:
        sample_ids = load_array(f, "sample_id")
        cell_types = consolidate_cell_types(load_array(f, "cell_type"))
        tumor = get_tumor_mask(sample_ids)
        marker_idx = get_marker_idx(f)
        umap = f["obsm"]["X_umap"][:]
        ct_counts = Counter(cell_types)
        panel_data.append({
            "X": X_mem, "cell_types": cell_types, "tumor": tumor,
            "marker_idx": marker_idx, "umap": umap, "ct_counts": ct_counts,
            "panel_name": pname, "heatmap_markers": hm,
        })

    f_t.close()
    f_s.close()

    # ── Load and crop cartoon ──
    cartoon = mpimg.imread(str(cartoon_path))
    if cartoon.ndim == 3:
        gray = np.mean(cartoon[:, :, :3], axis=(1, 2))
    else:
        gray = np.mean(cartoon, axis=1)
    content_rows = np.where(gray < 0.95)[0]
    if len(content_rows) > 0:
        pad = max(20, int(0.06 * cartoon.shape[0]))
        row_start = max(0, content_rows[0] - pad)
        row_end = min(cartoon.shape[0], content_rows[-1] + pad)
        cartoon = cartoon[row_start:row_end]

    # ── Direct-render into single figure ──
    print("Rendering figure...")
    fig = plt.figure(figsize=(20, 28))

    # Row fractions — aim for roughly equal row heights
    gap = 0.045
    h_cartoon = 0.14
    h_umap = 0.20
    h_heat = 0.22
    h_comp = 0.20

    top_a = 0.995
    bot_a = top_a - h_cartoon
    top_bc = bot_a - gap
    bot_bc = top_bc - h_umap
    top_de = bot_bc - gap
    bot_de = top_de - h_heat
    top_fg = bot_de - gap
    bot_fg = top_fg - h_comp

    # Row 1: cartoon (full width, rasterized image)
    gs_a = GridSpec(1, 1, figure=fig, left=0.05, right=0.95,
                    top=top_a, bottom=bot_a)
    ax_a = fig.add_subplot(gs_a[0, 0])
    ax_a.imshow(cartoon)
    ax_a.axis("off")
    panel_label(ax_a, "a")

    # Row 2: UMAPs (b, c)
    gs_bc = GridSpec(1, 2, figure=fig, left=0.05, right=0.95,
                     top=top_bc, bottom=bot_bc, wspace=0.25)
    for i, d in enumerate(panel_data):
        ax = fig.add_subplot(gs_bc[0, i])
        _plot_umap(ax, d["umap"], d["cell_types"], d["panel_name"], d["ct_counts"])
        panel_label(ax, chr(ord("b") + i))

    # Row 3: heatmaps (d, e) — compressed horizontally to fit shorter row
    gs_de = GridSpec(1, 2, figure=fig, left=0.08, right=0.92,
                     top=top_de, bottom=bot_de, wspace=0.45)
    for i, d in enumerate(panel_data):
        ax = fig.add_subplot(gs_de[0, i])
        _plot_heatmap(ax, d["X"], d["cell_types"], d["tumor"], d["marker_idx"],
                      d["heatmap_markers"], d["panel_name"])
        panel_label(ax, chr(ord("d") + i))

    # Row 4: composition bars (f, g)
    gs_fg = GridSpec(1, 2, figure=fig, left=0.05, right=0.95,
                     top=top_fg, bottom=bot_fg, wspace=0.40)
    for i, d in enumerate(panel_data):
        ax = fig.add_subplot(gs_fg[0, i])
        _plot_composition(ax, d["cell_types"], d["tumor"], d["panel_name"])
        panel_label(ax, chr(ord("f") + i))

    # Save
    out_path = Path(output_dir) / "fig_cell_type_annotation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved: {out_path} + PDF")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Figure 1: Dataset overview and cell type annotation"
    )
    parser.add_argument("--t-panel", required=True,
                        help="T-panel h5ad (e.g. all_TMA_T_global_v8.h5ad)")
    parser.add_argument("--s-panel", required=True,
                        help="S-panel h5ad (e.g. all_TMA_S_global_v8.h5ad)")
    parser.add_argument("--cartoon", required=True,
                        help="Dataset schematic cartoon PNG")
    parser.add_argument("--output-dir", default="output/qc",
                        help="Output directory")
    args = parser.parse_args()

    out_path = make_figure(args.t_panel, args.s_panel, args.cartoon,
                           args.output_dir)

    import subprocess
    subprocess.run(["open", "-a", "Preview", out_path])


if __name__ == "__main__":
    main()
