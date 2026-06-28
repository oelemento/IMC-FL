#!/usr/bin/env python
"""Figure: VISTA targeting landscape and checkpoint signaling in FL.

Combines VISTA characterization (panels a-c) with checkpoint/signaling
expression across cell types and compartments (panels d-e).

Uses Han et al. 2022 scRNA-seq and IMC S-panel data.
"""
import sys
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_signaling_architecture import (
    extract_scrna_data as extract_signaling_scrna,
    extract_imc_data as extract_signaling_imc,
    plot_checkpoint_celltypes,
    plot_checkpoint_compartments,
)

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22


def panel_label(ax, letter):
    ax.text(
        -0.08, 1.06, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, va="top",
    )


def get_expr(adata, name2ens, gene_name):
    eid = name2ens.get(gene_name)
    if eid is None:
        return None
    idx = list(adata.var.index).index(eid)
    v = adata.X[:, idx]
    if hasattr(v, "toarray"):
        v = v.toarray().ravel()
    return np.array(v).ravel()


def extract_data(h5ad_path="data/external/steen2022_fl_scrna.h5ad"):
    print("Loading scRNA-seq data...")
    adata = sc.read_h5ad(h5ad_path)
    fl = adata[adata.obs["disease"] == "follicular lymphoma"].copy()
    name2ens = dict(zip(fl.var["feature_name"], fl.var.index))
    gx = lambda g: get_expr(fl, name2ens, g)

    mye_mask = (fl.obs["cell_type"] == "myeloid cell").values
    fdc_mask = (fl.obs["cell_type"] == "follicular dendritic cell").values
    other_mask = ~mye_mask & ~fdc_mask
    n_mye = mye_mask.sum()
    print(f"FL cells: {fl.shape[0]:,}, Myeloid: {n_mye}, FDC: {fdc_mask.sum()}")

    vista = gx("VSIR")
    mye_vista = vista[mye_mask]
    v_pos = mye_vista > 0
    v_neg = ~v_pos

    # ── (a) VISTA+ vs VISTA- myeloid: key gene comparison ──
    compare_genes = [
        ("CD163", "CD163"), ("TGFB1", "TGFβ1"), ("SIRPA", "SIRPα"),
        ("HAVCR2", "TIM-3"), ("LGALS9", "Gal-9"), ("SIGLEC10", "SIGLEC10"),
        ("LILRB4", "LILRB4"), ("CSF1R", "CSF1R"),
        ("IL1B", "IL-1β"), ("CD274", "PD-L1"),
    ]
    panel_a = []
    for gene, label in compare_genes:
        v = gx(gene)
        if v is None:
            continue
        mye_v = v[mye_mask]
        pos_m = float(mye_v[v_pos].mean())
        neg_m = float(mye_v[v_neg].mean())
        _, p = stats.mannwhitneyu(mye_v[v_pos], mye_v[v_neg], alternative="two-sided")
        panel_a.append((label, pos_m, neg_m, p))

    # ── (b) Checkpoint co-expression count on VISTA+ myeloid ──
    cp_genes = ["CD274", "HAVCR2", "IDO1", "SIGLEC10", "LGALS9", "PDCD1LG2"]
    v_pos_idx = np.where(v_pos)[0]
    n_cp = np.zeros(len(v_pos_idx))
    for gene in cp_genes:
        v = gx(gene)
        if v is None:
            continue
        mye_v = v[mye_mask]
        n_cp += (mye_v[v_pos_idx] > 0).astype(int)
    cp_counts = {}
    for k in range(int(n_cp.max()) + 1):
        cp_counts[k] = int((n_cp == k).sum())

    # ── (c) Druggable target heatmap: % positive on myeloid vs FDC vs other ──
    target_genes = [
        ("VSIR", "VISTA"), ("HAVCR2", "TIM-3"), ("LGALS9", "Gal-9"),
        ("SIGLEC10", "SIGLEC10"), ("CD274", "PD-L1"),
        ("SIRPA", "SIRPα"), ("CSF1R", "CSF1R"),
        ("LILRB2", "LILRB2"), ("LILRB4", "LILRB4"),
        ("CD163", "CD163"), ("TREM2", "TREM2"),
        ("CD40", "CD40"), ("CD14", "CD14"),
    ]
    heatmap_rows = []
    for gene, label in target_genes:
        v = gx(gene)
        if v is None:
            continue
        mye_pct = 100 * (v[mye_mask] > 0).mean()
        fdc_pct = 100 * (v[fdc_mask] > 0).mean()
        oth_pct = 100 * (v[other_mask] > 0).mean()
        heatmap_rows.append((label, mye_pct, fdc_pct, oth_pct))

    return {
        "panel_a": panel_a,
        "cp_counts": cp_counts,
        "n_vista_pos": int(v_pos.sum()),
        "heatmap_rows": heatmap_rows,
        "n_mye": n_mye,
    }


def plot_vista_abundance(ax, imc_data):
    """Composition of the follicular VISTA+ cell pool (pooled across cohort).

    Panel (e) shows per-cell VISTA+ fractions (M2 Mac 91%, S100A9+ 88%,
    CD14+ FDC 32%, CD14- FDC 15%). This panel integrates fraction × abundance:
    among VISTA+ source populations in the follicle, CD14+ FDC dominates by
    total count. Single horizontal stacked bar, pooled across all tumor cores.
    """
    per_roi = imc_data.get("vista_per_roi_foll", {})
    cts = ["CD14+ FDC", "CD14- FDC", "M2 Mac", "S100A9+"]
    totals = {ct: int(np.sum(per_roi.get(ct, []))) for ct in cts}
    # Sort by count, descending
    ordered = sorted(cts, key=lambda c: -totals[c])
    counts = [totals[ct] for ct in ordered]

    y = np.arange(len(ordered))
    ax.barh(y, counts, color="#984EA3", alpha=0.85, height=0.65,
            edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(ordered, fontsize=TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel("VISTA+ cells (follicular compartment, pooled)",
                  fontsize=LABEL_SIZE)
    n_rois = len(per_roi.get(cts[0], []))
    ax.set_title(f"Follicular VISTA+ source cell counts\n(IMC, pooled across {n_rois} ROIs)",
                 fontsize=TITLE_SIZE)
    ax.set_xlim(0, max(counts) * 1.05)
    ax.tick_params(labelsize=TICK_SIZE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_vista_fl_vs_tonsil(ax, csv_path="output/vista_fl_vs_tonsil.csv"):
    """VISTA in FL vs tonsil, per cell type.

    Bars = per-cell pooled mean across all cells in the group; error bars =
    SEM across cells. Significance stars come from a per-ROI Mann-Whitney
    (one-sided FL > tonsil) — this is the rigorous test (ROIs as independent
    observations); the per-cell test is dominated by sample size and is
    reported alongside in the CSV for transparency only.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Drop M1 Macrophages to match the cell types shown in the rest of Fig S11
    df = df[df["cell_type"] != "M1 Macrophages"].copy()
    # Match the short label used in panels d/e/f for consistency
    df["cell_type"] = df["cell_type"].replace({"Myeloid (S100A9+)": "S100A9+"})
    # Order: descending by per-cell fold change (matches what the bars show)
    df["fold"] = df["fl_arcsinh_mean"] / df["ton_arcsinh_mean"]
    df = df.sort_values("fold", ascending=False).reset_index(drop=True)

    cts = df["cell_type"].tolist()
    fl_means = df["fl_arcsinh_mean"].values
    ton_means = df["ton_arcsinh_mean"].values
    fl_sems = df["fl_arcsinh_sem"].values
    ton_sems = df["ton_arcsinh_sem"].values
    # Stars: per-ROI test (the rigorous one; matches CD14- FDC NS, M1 Mac **).
    pvals = df["p_greater_per_roi"].values

    x = np.arange(len(cts))
    w = 0.35
    ax.bar(x - w / 2, fl_means, w, color="#D62728", alpha=0.85,
           edgecolor="white", linewidth=0.5, label="FL tumor",
           yerr=fl_sems, capsize=3, ecolor="#333333", error_kw={"linewidth": 0.8})
    ax.bar(x + w / 2, ton_means, w, color="#377EB8", alpha=0.85,
           edgecolor="white", linewidth=0.5, label="Normal tonsil",
           yerr=ton_sems, capsize=3, ecolor="#333333", error_kw={"linewidth": 0.8})

    # Significance stars: per-ROI Mann-Whitney (one-sided)
    for i, p in enumerate(pvals):
        if np.isnan(p):
            continue
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = "ns"
        ymax = max(fl_means[i], ton_means[i])
        ax.text(i, ymax * 1.06, stars, ha="center", va="bottom",
                fontsize=ANNOT_SIZE, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(cts, fontsize=TICK_SIZE, rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("Mean VISTA (arcsinh)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 0.6)
    # Two-line title so the panel-letter ('g') has clearance from the title text
    ax.set_title("VISTA in FL vs normal tonsil\n(IMC, S-panel)",
                 fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")
    ax.tick_params(labelsize=TICK_SIZE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def make_figure(data, signaling_scrna, signaling_imc,
                output_dir="output/hypotheses_v8"):
    """Generate 7-panel figure: VISTA targeting (a-c) + checkpoint signaling (d-g)."""
    fig = plt.figure(figsize=(20, 22))
    # Three rows, uniform height (~0.23) with ~0.10 vertical gap between rows
    # so panel titles in row 2/3 don't crowd the x-tick text of the row above.
    gs_top = GridSpec(1, 3, figure=fig, wspace=0.50,
                      left=0.07, right=0.96, top=0.97, bottom=0.74)
    gs_mid = GridSpec(1, 3, figure=fig, wspace=0.45,
                      left=0.07, right=0.96, top=0.64, bottom=0.41)
    # Panel g sits in the same column-0 footprint as panels a/d above it
    # (figure-conventions rule: a single bottom panel takes one-column width,
    # left-aligned with the upper grid — never stretch across the full row).
    gs_bot = GridSpec(1, 3, figure=fig, wspace=0.45,
                      left=0.07, right=0.96, top=0.31, bottom=0.08)

    # ── (a) VISTA+ vs VISTA- myeloid: gene expression ──
    ax_a = fig.add_subplot(gs_top[0, 0])
    panel_label(ax_a, "a")
    pa = data["panel_a"]
    labels = [x[0] for x in pa]
    pos_vals = [x[1] for x in pa]
    neg_vals = [x[2] for x in pa]
    y = np.arange(len(labels))
    h = 0.35
    ax_a.barh(y - h / 2, pos_vals, h, label="VISTA+", color="#D62728", alpha=0.85)
    ax_a.barh(y + h / 2, neg_vals, h, label="VISTA−", color="#7F7F7F", alpha=0.65)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Mean expression", fontsize=LABEL_SIZE)
    ax_a.set_title(f"VISTA+ vs VISTA− myeloid cells\n(n={data['n_vista_pos']} vs {data['n_mye'] - data['n_vista_pos']})", fontsize=TITLE_SIZE)
    ax_a.legend(fontsize=LEGEND_SIZE, loc="lower right")
    # Stars require BOTH significance (p<0.01) AND meaningful effect size
    # (|log2FC| > 0.25, ratio >1.19x) to avoid flagging genes where large n
    # gives statistical power for biologically trivial differences (e.g. PD-L1).
    for i, (_, pos_m, neg_m, p) in enumerate(pa):
        if pos_m > 0 and neg_m > 0:
            lfc = np.log2((pos_m + 1e-3) / (neg_m + 1e-3))
        else:
            lfc = 0.0
        if abs(lfc) < 0.25:
            continue
        if p < 0.001:
            ax_a.text(max(pos_vals[i], neg_vals[i]) + 0.02, i, "***",
                      va="center", fontsize=7, color="#333")
        elif p < 0.01:
            ax_a.text(max(pos_vals[i], neg_vals[i]) + 0.02, i, "**",
                      va="center", fontsize=7, color="#333")

    # ── (b) Checkpoint stacking on VISTA+ myeloid ──
    ax_b = fig.add_subplot(gs_top[0, 1])
    panel_label(ax_b, "b")
    cp = data["cp_counts"]
    ks = sorted(cp.keys())
    vals = [cp[k] for k in ks]
    colors_b = ["#2CA02C" if k == 0 else "#FF7F0E" if k <= 2 else "#D62728" for k in ks]
    bars = ax_b.bar(ks, vals, color=colors_b, edgecolor="white", linewidth=0.5)
    ax_b.set_xlabel("Additional checkpoints co-expressed", fontsize=LABEL_SIZE)
    ax_b.set_ylabel("Number of VISTA+ myeloid cells", fontsize=LABEL_SIZE)
    ax_b.set_title(f"Checkpoint stacking on VISTA+ myeloid\n(n={data['n_vista_pos']})", fontsize=TITLE_SIZE)
    ax_b.set_xticks(ks)
    ax_b.tick_params(labelsize=TICK_SIZE)
    total = sum(vals)
    for bar, v in zip(bars, vals):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 3,
                  f"{100 * v / total:.0f}%", ha="center", fontsize=8)
    multi = sum(v for k, v in cp.items() if k >= 2)
    ax_b.text(0.95, 0.95, f"{100 * multi / total:.0f}% express\n2+ checkpoints",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEEEE", edgecolor="#D62728"))

    # ── (c) Druggable target heatmap ──
    ax_c = fig.add_subplot(gs_top[0, 2])
    panel_label(ax_c, "c")
    hr = data["heatmap_rows"]
    gene_labels = [x[0] for x in hr]
    mat = np.array([[x[1], x[2], x[3]] for x in hr])
    im = ax_c.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=60)
    ax_c.set_yticks(range(len(gene_labels)))
    ax_c.set_yticklabels(gene_labels, fontsize=TICK_SIZE)
    ax_c.set_xticks([0, 1, 2])
    ax_c.set_xticklabels(["Myeloid", "FDC", "Other"], fontsize=TICK_SIZE)
    ax_c.set_title("% cells expressing target\n(druggable surface molecules)", fontsize=TITLE_SIZE)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v >= 1:
                color = "white" if v > 30 else "black"
                ax_c.text(j, i, f"{v:.0f}", ha="center", va="center",
                          fontsize=7, color=color)
    plt.colorbar(im, ax=ax_c, shrink=0.7, label="% positive")

    # ── (d) VISTA/IDO checkpoint expression across cell types (scRNA) ──
    ax_d = fig.add_subplot(gs_mid[0, 0])
    panel_label(ax_d, "d")
    plot_checkpoint_celltypes(ax_d, signaling_scrna)

    # ── (e) VISTA+/IDO+ fraction by compartment and cell type (IMC) ──
    ax_e = fig.add_subplot(gs_mid[0, 1])
    panel_label(ax_e, "e")
    plot_checkpoint_compartments(ax_e, signaling_imc)

    # ── (f) Absolute VISTA+ cell counts in follicle (abundance view) ──
    ax_f = fig.add_subplot(gs_mid[0, 2])
    panel_label(ax_f, "f")
    plot_vista_abundance(ax_f, signaling_imc)

    # ── (g) FL vs tonsil VISTA per cell type (per-ROI) ──
    ax_g = fig.add_subplot(gs_bot[0, 0])
    # Shift the panel letter slightly left/up so it doesn't sit on the 0.6 y-tick
    ax_g.text(-0.13, 1.10, r"$\bf{g}$", transform=ax_g.transAxes,
              fontsize=PANEL_LABEL_SIZE, va="top")
    plot_vista_fl_vs_tonsil(ax_g)

    out = Path(output_dir) / "fig_vista_targeting.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    data = extract_data()
    signaling_scrna = extract_signaling_scrna("data/external/steen2022_fl_scrna.h5ad")
    signaling_imc = extract_signaling_imc("output/all_TMA_S_global_v8.h5ad",
                                           "output/all_TMA_S_utag_ct_merged.h5ad")
    make_figure(data, signaling_scrna, signaling_imc)
