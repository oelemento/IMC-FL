#!/usr/bin/env python
"""Validation figure: CD14 on FDCs confirmed by external scRNA-seq.

Uses Han et al. 2022 (Blood Cancer Discovery) FL scRNA-seq data
from CZ CELLxGENE (137K cells, 20 FL tumors + 3 controls).
"""
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.spatial.distance import cosine
from pathlib import Path


def panel_label(ax, letter):
    ax.text(
        -0.08, 1.06, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top",
    )


# Gene name mapping
KEY_GENES = {
    "CD14": "ENSG00000170458",
    "CD68": "ENSG00000129226",
    "VSIR": "ENSG00000107738",      # VISTA
    "ITGAM": "ENSG00000169896",     # CD11b
    "CR2": "ENSG00000117322",       # CD21
    "FCER2": "ENSG00000104921",     # CD23
    "CXCL13": "ENSG00000156234",
    "PDPN": "ENSG00000162493",
    "HLA-DRA": "ENSG00000204287",
    "CD274": "ENSG00000120217",     # PD-L1
    "IDO1": "ENSG00000131203",
    "MS4A1": "ENSG00000156738",     # CD20
    "TLR5": "ENSG00000187554",
    "C3AR1": "ENSG00000171860",
    "FCGR1A": "ENSG00000150337",
}

GENE_LABELS = {
    "CD14": "CD14",
    "CD68": "CD68",
    "VSIR": "VISTA",
    "ITGAM": "CD11b",
    "CR2": "CD21",
    "FCER2": "CD23",
    "CXCL13": "CXCL13",
    "PDPN": "PDPN",
    "HLA-DRA": "HLA-DRA",
    "CD274": "PD-L1",
    "IDO1": "IDO1",
    "MS4A1": "CD20",
    "TLR5": "TLR5",
    "C3AR1": "C3AR1",
    "FCGR1A": "FCGR1A",
}

# Cell type short names
CT_SHORT = {
    "follicular dendritic cell": "FDC",
    "myeloid cell": "Myeloid",
    "B cell": "B cell",
    "malignant cell": "Malignant B",
    "T follicular helper cell": "Tfh",
    "regulatory T cell": "Treg",
    "exhausted T cell": "Exhausted T",
    "effector CD8-positive, alpha-beta T cell": "CD8 effector",
    "naive thymus-derived CD4-positive, alpha-beta T cell": "Naive CD4 T",
    "CD4-positive, alpha-beta cytotoxic T cell": "Cytotoxic CD4",
    "mature NK T cell": "NKT",
    "T cell": "T cell",
    "CD4-positive, alpha-beta T cell": "CD4 T",
    "CD8-positive, alpha-beta T cell": "CD8 T",
    "naive thymus-derived CD8-positive, alpha-beta T cell": "Naive CD8 T",
    "plasmacytoid dendritic cell": "pDC",
    "plasma cell": "Plasma",
    "erythrocyte": "Erythrocyte",
}

CT_COLORS = {
    "FDC": "#FF7F00",
    "Myeloid": "#E41A1C",
    "B cell": "#377EB8",
    "Malignant B": "#A6CEE3",
    "Tfh": "#33A02C",
    "Treg": "#6A3D9A",
    "Exhausted T": "#B15928",
    "CD8 effector": "#FB9A99",
}


def get_gene_expr(adata, ensembl_id):
    """Get expression vector for a gene."""
    idx = list(adata.var.index).index(ensembl_id)
    vals = adata.X[:, idx]
    if hasattr(vals, "toarray"):
        vals = vals.toarray().ravel()
    return np.array(vals).ravel()


def extract_data(h5ad_path="data/external/steen2022_fl_scrna.h5ad"):
    """Extract all data for the validation figure."""
    print("Loading scRNA-seq data...")
    adata = sc.read_h5ad(h5ad_path)

    # FL only
    fl_mask = adata.obs["disease"] == "follicular lymphoma"
    adata_fl = adata[fl_mask].copy()
    cell_types = adata_fl.obs["cell_type"].values
    print(f"FL cells: {adata_fl.shape[0]:,}")

    # ── Panel (a): CD14 by cell type ──
    cd14_expr = get_gene_expr(adata_fl, KEY_GENES["CD14"])
    unique_cts = sorted(set(cell_types))
    cd14_by_ct = []
    for ct in unique_cts:
        mask = cell_types == ct
        n = mask.sum()
        if n < 10:
            continue
        vals = cd14_expr[mask]
        cd14_by_ct.append((
            CT_SHORT.get(ct, ct), n,
            float(vals.mean()),
            100 * (vals > 0).mean(),
        ))
    cd14_by_ct.sort(key=lambda x: -x[2])

    # ── Panel (b): FDC marker heatmap ──
    focus_cts = [
        "follicular dendritic cell", "myeloid cell",
        "B cell", "malignant cell",
    ]
    # Myeloid markers + FDC markers
    myeloid_genes = ["CD14", "CD68", "VSIR", "ITGAM", "IDO1", "TLR5",
                     "C3AR1", "FCGR1A", "CD274"]
    fdc_genes = ["CR2", "FCER2", "CXCL13", "PDPN"]
    heatmap_genes = myeloid_genes + fdc_genes

    heatmap_data = {}
    for ct in focus_cts:
        mask = cell_types == ct
        ct_label = CT_SHORT.get(ct, ct)
        heatmap_data[ct_label] = {}
        for gene in heatmap_genes:
            ensid = KEY_GENES.get(gene)
            if ensid:
                vals = get_gene_expr(adata_fl, ensid)
                heatmap_data[ct_label][gene] = float(vals[mask].mean())

    # ── Panel (c): % CD14+ by cell type (detection rate) ──
    # Already computed in cd14_by_ct

    # ── Panel (d): Transcriptional similarity ──
    profiles = {}
    for ct in focus_cts:
        mask = cell_types == ct
        X_sub = adata_fl.X[mask]
        if hasattr(X_sub, "toarray"):
            X_sub = X_sub.toarray()
        profiles[CT_SHORT.get(ct, ct)] = X_sub.mean(axis=0)

    similarities = {}
    for ct1 in ["FDC"]:
        for ct2 in ["Myeloid", "B cell", "Malignant B"]:
            rho, _ = stats.spearmanr(profiles[ct1], profiles[ct2])
            cos_sim = 1 - cosine(profiles[ct1], profiles[ct2])
            similarities[(ct1, ct2)] = {"spearman": rho, "cosine": cos_sim}

    # ── Panel (e): VISTA (VSIR) expression across cell types ──
    vsir_expr = get_gene_expr(adata_fl, KEY_GENES["VSIR"])
    vsir_by_ct = []
    for ct in unique_cts:
        mask = cell_types == ct
        n = mask.sum()
        if n < 30:
            continue
        vals = vsir_expr[mask]
        vsir_by_ct.append((
            CT_SHORT.get(ct, ct), n,
            float(vals.mean()),
            100 * (vals > 0).mean(),
        ))
    vsir_by_ct.sort(key=lambda x: -x[2])

    return {
        "cd14_by_ct": cd14_by_ct,
        "heatmap_data": heatmap_data,
        "heatmap_genes": heatmap_genes,
        "myeloid_genes": myeloid_genes,
        "similarities": similarities,
        "vsir_by_ct": vsir_by_ct,
        "n_fdc": int((cell_types == "follicular dendritic cell").sum()),
        "n_myeloid": int((cell_types == "myeloid cell").sum()),
        "n_total": len(cell_types),
    }


def make_figure(data, output_dir="output/hypotheses_v8"):
    """Generate 5-panel validation figure (3 top + 2 bottom)."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.40,
                  left=0.08, right=0.94, top=0.94, bottom=0.06)

    # ── (a) CD14 mRNA by cell type ──
    ax_a = fig.add_subplot(gs[0, 0])  # top-left
    panel_label(ax_a, "a")
    ct_data = data["cd14_by_ct"]
    names = [x[0] for x in ct_data]
    means = [x[2] for x in ct_data]
    colors = []
    for name, _, _, _ in ct_data:
        if name == "FDC":
            colors.append("#FF7F00")
        elif name == "Myeloid":
            colors.append("#E41A1C")
        else:
            colors.append("#CCCCCC")
    y_pos = range(len(names))
    ax_a.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.5)
    ax_a.set_yticks(list(y_pos))
    ax_a.set_yticklabels(names, fontsize=8)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Mean CD14 mRNA expression")
    ax_a.set_title("CD14 expression by cell type\n(Han et al. 2022, scRNA-seq)")
    # Annotate FDC and Myeloid
    for i, (name, n, mean, pct) in enumerate(ct_data):
        if name in ("FDC", "Myeloid"):
            ax_a.text(
                mean + 0.01, i, f"{pct:.0f}% cells +",
                va="center", fontsize=7, color="#333",
            )

    # ── (b) Marker heatmap: FDC vs Myeloid vs B vs Malignant ──
    ax_b = fig.add_subplot(gs[0, 1])  # top-center
    panel_label(ax_b, "b")
    hm = data["heatmap_data"]
    genes = data["heatmap_genes"]
    cts_order = ["FDC", "Myeloid", "B cell", "Malignant B"]
    mat = np.array([[hm[ct].get(g, 0) for ct in cts_order] for g in genes])
    # Z-score per gene (row)
    mat_z = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        row = mat[i]
        if row.std() > 0:
            mat_z[i] = (row - row.mean()) / row.std()
    im = ax_b.imshow(mat_z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax_b.set_xticks(range(len(cts_order)))
    ax_b.set_xticklabels(cts_order, fontsize=9, rotation=30, ha="right")
    ax_b.set_yticks(range(len(genes)))
    gene_labels = [GENE_LABELS.get(g, g) for g in genes]
    ax_b.set_yticklabels(gene_labels, fontsize=8)
    # Add divider between myeloid and FDC markers
    n_mye = len(data["myeloid_genes"])
    ax_b.axhline(n_mye - 0.5, color="black", linewidth=1.5)
    ax_b.text(-0.6, n_mye / 2 - 0.5, "Myeloid\nmarkers", ha="right",
              va="center", fontsize=7, style="italic")
    ax_b.text(-0.6, n_mye + 1.5, "FDC\nmarkers", ha="right",
              va="center", fontsize=7, style="italic")
    # Annotate raw values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val > 0.05:
                color = "white" if abs(mat_z[i, j]) > 1.2 else "black"
                ax_b.text(j, i, f"{val:.2f}", ha="center", va="center",
                          fontsize=6, color=color)
    ax_b.set_title("Marker expression (z-scored)\nmyeloid vs FDC identity genes")
    plt.colorbar(im, ax=ax_b, shrink=0.6, label="z-score")

    # ── (c) % CD14+ detection rate ──
    ax_c = fig.add_subplot(gs[0, 2])  # top-right
    panel_label(ax_c, "c")
    # Show only cell types with >0.1% CD14+
    ct_data_filt = [(n, nn, m, p) for n, nn, m, p in data["cd14_by_ct"] if p > 0.1]
    names_c = [x[0] for x in ct_data_filt]
    pcts = [x[3] for x in ct_data_filt]
    colors_c = []
    for name, _, _, _ in ct_data_filt:
        if name == "FDC":
            colors_c.append("#FF7F00")
        elif name == "Myeloid":
            colors_c.append("#E41A1C")
        else:
            colors_c.append("#CCCCCC")
    y_pos = range(len(names_c))
    ax_c.barh(y_pos, pcts, color=colors_c, edgecolor="white", linewidth=0.5)
    ax_c.set_yticks(list(y_pos))
    ax_c.set_yticklabels(names_c, fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("% cells with CD14 > 0")
    ax_c.set_title("CD14 detection rate by cell type")
    # Annotate counts
    for i, (name, n, _, pct) in enumerate(ct_data_filt):
        ax_c.text(
            pct + 0.5, i, f"n={n:,}",
            va="center", fontsize=6, color="#666",
        )

    # ── (d) Transcriptional similarity ──
    ax_d = fig.add_subplot(gs[1, 0])  # bottom-left
    panel_label(ax_d, "d")
    sim = data["similarities"]
    pairs = [("FDC", "Myeloid"), ("FDC", "B cell"), ("FDC", "Malignant B")]
    pair_labels = ["FDC vs\nMyeloid", "FDC vs\nB cell", "FDC vs\nMalignant B"]
    spearman_vals = [sim[p]["spearman"] for p in pairs]
    cosine_vals = [sim[p]["cosine"] for p in pairs]
    x = np.arange(len(pairs))
    w = 0.35
    bars1 = ax_d.bar(x - w / 2, spearman_vals, w, label="Spearman ρ",
                     color="#377EB8", alpha=0.8)
    bars2 = ax_d.bar(x + w / 2, cosine_vals, w, label="Cosine sim.",
                     color="#E41A1C", alpha=0.8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(pair_labels, fontsize=9)
    ax_d.set_ylabel("Similarity")
    ax_d.set_title("FDC transcriptional similarity\n(whole-transcriptome)")
    ax_d.legend(fontsize=8)
    ax_d.set_ylim(0.7, 1.0)
    # Annotate values
    for bar, val in zip(bars1, spearman_vals):
        ax_d.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, cosine_vals):
        ax_d.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # ── (e) VISTA (VSIR) by cell type ──
    ax_e = fig.add_subplot(gs[1, 1])  # bottom-center
    panel_label(ax_e, "e")
    vsir_data = data["vsir_by_ct"][:12]  # top 12
    names_e = [x[0] for x in vsir_data]
    means_e = [x[2] for x in vsir_data]
    colors_e = []
    for name, _, _, _ in vsir_data:
        if name == "FDC":
            colors_e.append("#FF7F00")
        elif name == "Myeloid":
            colors_e.append("#E41A1C")
        else:
            colors_e.append("#CCCCCC")
    y_pos = range(len(names_e))
    ax_e.barh(y_pos, means_e, color=colors_e, edgecolor="white", linewidth=0.5)
    ax_e.set_yticks(list(y_pos))
    ax_e.set_yticklabels(names_e, fontsize=8)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Mean VISTA (VSIR) mRNA expression")
    ax_e.set_title("VISTA expression by cell type\n(confirms IMC protein finding)")
    for i, (name, n, mean, pct) in enumerate(vsir_data):
        if name in ("FDC", "Myeloid"):
            ax_e.text(
                mean + 0.01, i, f"{pct:.0f}% +",
                va="center", fontsize=7, color="#333",
            )

    out = Path(output_dir) / "fig_scrna_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    data = extract_data()
    make_figure(data)
