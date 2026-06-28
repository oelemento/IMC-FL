#!/usr/bin/env python
"""scRNA-seq deep analysis: CD14+ vs CD14- FDCs in follicular lymphoma.

Uses Han et al. 2022 (Blood Cancer Discovery) FL scRNA-seq data
from CZ CELLxGENE (137K cells, 20 FL tumors + 3 controls).

Split FDCs by CD14 expression and characterize transcriptomic differences.
"""
import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path


def panel_label(ax, letter):
    ax.text(
        -0.08, 1.06, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top",
    )


# Gene name mapping (Ensembl IDs used in CZ CELLxGENE)
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
    "CXCL12": "ENSG00000107562",
    "CCL21": "ENSG00000137077",
    "CCL19": "ENSG00000090659",
    "HLA-A": "ENSG00000206503",     # HLA class I
    "HLA-B": "ENSG00000234745",
    "HLA-C": "ENSG00000204525",
    "B2M": "ENSG00000166710",
    "TNFRSF13B": "ENSG00000240505", # TACI
    "TNFSF13B": "ENSG00000102524",  # BAFF
    "TNFSF13": "ENSG00000161955",   # APRIL
    "IL6": "ENSG00000136244",
    "IL10": "ENSG00000136634",
    "TGFB1": "ENSG00000105329",
    "S100A9": "ENSG00000163220",
    "CD163": "ENSG00000177575",
    "MRC1": "ENSG00000260314",      # CD206
    "ICAM1": "ENSG00000090339",     # CD54
    "VCAM1": "ENSG00000162692",
    "FN1": "ENSG00000115414",       # Fibronectin
    "CLU": "ENSG00000120885",       # Clusterin (FDC marker)
    "SERPINE1": "ENSG00000106366",  # PAI-1 (FDC marker)
    "FDCSP": "ENSG00000135424",     # FDC secreted protein
    "CD55": "ENSG00000196352",      # DAF (FDC marker)
    "CD44": "ENSG00000026508",
    "PTPRC": "ENSG00000081237",     # CD45
    "PECAM1": "ENSG00000261371",    # CD31
    "ENG": "ENSG00000106991",       # CD105/endoglin
    "MKI67": "ENSG00000148773",     # Ki67
    "BCL2": "ENSG00000171791",
    "BCL6": "ENSG00000113916",
    "IRF4": "ENSG00000137265",
    "AICDA": "ENSG00000111732",     # AID
}

GENE_LABELS = {
    "CD14": "CD14", "CD68": "CD68", "VSIR": "VISTA", "ITGAM": "CD11b",
    "CR2": "CD21", "FCER2": "CD23", "CXCL13": "CXCL13", "PDPN": "PDPN",
    "HLA-DRA": "HLA-DRA", "CD274": "PD-L1", "IDO1": "IDO1",
    "MS4A1": "CD20", "CXCL12": "CXCL12", "CCL21": "CCL21", "CCL19": "CCL19",
    "HLA-A": "HLA-A", "HLA-B": "HLA-B", "HLA-C": "HLA-C", "B2M": "B2M",
    "TNFRSF13B": "TACI", "TNFSF13B": "BAFF", "TNFSF13": "APRIL",
    "IL6": "IL6", "IL10": "IL10", "TGFB1": "TGF-β1",
    "S100A9": "S100A9", "CD163": "CD163", "MRC1": "CD206",
    "ICAM1": "CD54", "VCAM1": "VCAM1", "FN1": "Fibronectin",
    "CLU": "Clusterin", "SERPINE1": "PAI-1", "FDCSP": "FDCSP",
    "CD55": "CD55", "CD44": "CD44", "PTPRC": "CD45",
    "PECAM1": "CD31", "ENG": "CD105", "MKI67": "Ki67",
    "BCL2": "BCL2", "BCL6": "BCL6", "IRF4": "IRF4", "AICDA": "AID",
    "TLR5": "TLR5", "C3AR1": "C3AR1", "FCGR1A": "FCGR1A",
}


def get_gene_expr(adata, ensembl_id):
    """Get expression vector for a gene (handles sparse)."""
    if ensembl_id not in adata.var.index:
        return None
    idx = list(adata.var.index).index(ensembl_id)
    vals = adata.X[:, idx]
    if hasattr(vals, "toarray"):
        vals = vals.toarray().ravel()
    return np.array(vals).ravel()


def run_analysis(h5ad_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and subset ──
    print("Loading scRNA-seq data...")
    adata = sc.read_h5ad(h5ad_path)

    # FL only
    fl_mask = adata.obs["disease"] == "follicular lymphoma"
    adata_fl = adata[fl_mask].copy()
    print(f"FL cells: {adata_fl.shape[0]:,}")

    # Subset to FDCs
    fdc_mask = adata_fl.obs["cell_type"] == "follicular dendritic cell"
    adata_fdc = adata_fl[fdc_mask].copy()
    n_fdc = adata_fdc.shape[0]
    print(f"FDCs: {n_fdc}")

    if n_fdc < 20:
        print("Too few FDCs for meaningful analysis. Exiting.")
        return

    # ── CD14 expression on FDCs ──
    cd14_expr = get_gene_expr(adata_fdc, KEY_GENES["CD14"])
    if cd14_expr is None:
        print("CD14 gene not found. Exiting.")
        return

    cd14_nonzero = cd14_expr[cd14_expr > 0]
    print(f"\nCD14 on FDCs:")
    print(f"  Total FDCs: {n_fdc}")
    print(f"  CD14 > 0: {len(cd14_nonzero)} ({100*len(cd14_nonzero)/n_fdc:.1f}%)")
    print(f"  Mean (all): {cd14_expr.mean():.3f}")
    print(f"  Mean (nonzero): {cd14_nonzero.mean():.3f}" if len(cd14_nonzero) > 0 else "  Mean (nonzero): N/A")
    print(f"  Median: {np.median(cd14_expr):.3f}")

    # Split: CD14+ (any expression) vs CD14- (zero)
    cd14_pos = cd14_expr > 0
    n_pos = cd14_pos.sum()
    n_neg = (~cd14_pos).sum()
    print(f"\n  CD14+ FDCs: {n_pos} ({100*n_pos/n_fdc:.1f}%)")
    print(f"  CD14- FDCs: {n_neg} ({100*n_neg/n_fdc:.1f}%)")

    adata_fdc.obs["cd14_status"] = pd.Categorical(
        ["CD14+" if x else "CD14-" for x in cd14_pos],
        categories=["CD14+", "CD14-"],
    )

    # ── Differential expression ──
    print("\nRunning differential expression (CD14+ vs CD14- FDCs)...")
    sc.tl.rank_genes_groups(
        adata_fdc, groupby="cd14_status", groups=["CD14+"],
        reference="CD14-", method="wilcoxon", n_genes=adata_fdc.shape[1],
    )

    # Extract results
    de_results = sc.get.rank_genes_groups_df(adata_fdc, group="CD14+")
    # Map Ensembl IDs to gene symbols where possible
    ensid_to_name = {v: k for k, v in KEY_GENES.items()}
    de_results["gene_symbol"] = de_results["names"].map(
        lambda x: ensid_to_name.get(x, x)
    )

    # Print top DE genes
    sig = de_results[de_results["pvals_adj"] < 0.05].copy()
    print(f"\nSignificant DE genes (FDR < 0.05): {len(sig)}")
    print(f"  Upregulated in CD14+: {(sig['logfoldchanges'] > 0).sum()}")
    print(f"  Downregulated in CD14+: {(sig['logfoldchanges'] < 0).sum()}")

    print("\nTop 30 upregulated in CD14+ FDCs:")
    up = sig[sig["logfoldchanges"] > 0].sort_values("logfoldchanges", ascending=False)
    for _, row in up.head(30).iterrows():
        sym = row["gene_symbol"]
        if sym == row["names"]:
            # Try to get gene symbol from var if available
            sym = row["names"][:20]  # truncate long IDs
        print(f"  {sym:20s} logFC={row['logfoldchanges']:+.3f}  padj={row['pvals_adj']:.2e}")

    print("\nTop 30 downregulated in CD14+ FDCs:")
    down = sig[sig["logfoldchanges"] < 0].sort_values("logfoldchanges", ascending=True)
    for _, row in down.head(30).iterrows():
        sym = row["gene_symbol"]
        if sym == row["names"]:
            sym = row["names"][:20]
        print(f"  {sym:20s} logFC={row['logfoldchanges']:+.3f}  padj={row['pvals_adj']:.2e}")

    # ── Check our IMC markers specifically ──
    print("\n\n=== IMC-relevant markers (CD14+ vs CD14- FDCs) ===")
    imc_markers = [
        "CD14", "CD68", "VSIR", "ITGAM", "CR2", "FCER2", "CXCL13", "PDPN",
        "HLA-DRA", "IDO1", "CD274", "CXCL12", "CCL21", "CCL19",
        "HLA-A", "HLA-B", "HLA-C", "B2M",
        "TNFSF13B", "TNFSF13", "IL6", "IL10", "TGFB1",
        "S100A9", "CD163", "MRC1",
        "ICAM1", "VCAM1", "FN1", "CLU", "SERPINE1", "FDCSP", "CD55",
        "MKI67", "BCL2", "BCL6",
        "TLR5", "C3AR1", "FCGR1A",
    ]

    imc_results = []
    for gene in imc_markers:
        ensid = KEY_GENES.get(gene)
        if not ensid:
            continue
        row = de_results[de_results["names"] == ensid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        label = GENE_LABELS.get(gene, gene)

        # Also get mean expression in each group
        expr = get_gene_expr(adata_fdc, ensid)
        if expr is None:
            continue
        mean_pos = expr[cd14_pos].mean()
        mean_neg = expr[~cd14_pos].mean()
        pct_pos = 100 * (expr[cd14_pos] > 0).mean()
        pct_neg = 100 * (expr[~cd14_pos] > 0).mean()

        star = ""
        if row["pvals_adj"] < 0.001:
            star = "***"
        elif row["pvals_adj"] < 0.01:
            star = "**"
        elif row["pvals_adj"] < 0.05:
            star = "*"

        imc_results.append({
            "gene": gene, "label": label,
            "logFC": row["logfoldchanges"],
            "padj": row["pvals_adj"],
            "mean_pos": mean_pos, "mean_neg": mean_neg,
            "pct_pos": pct_pos, "pct_neg": pct_neg,
            "star": star,
        })
        direction = "UP" if row["logfoldchanges"] > 0 else "DOWN"
        print(f"  {label:15s}  logFC={row['logfoldchanges']:+.3f}  "
              f"padj={row['pvals_adj']:.2e} {star:4s}  "
              f"mean: {mean_pos:.3f} vs {mean_neg:.3f}  "
              f"det: {pct_pos:.0f}% vs {pct_neg:.0f}%  [{direction}]")

    imc_df = pd.DataFrame(imc_results)

    # ── Check if gene symbols are available ──
    # Try to get gene symbols from var columns
    has_symbols = False
    symbol_col = None
    for col in ["feature_name", "gene_symbol", "symbol", "gene_name"]:
        if col in adata_fdc.var.columns:
            symbol_col = col
            has_symbols = True
            break

    if has_symbols:
        print(f"\nGene symbols available via column: {symbol_col}")
        de_results["symbol"] = adata_fdc.var.loc[de_results["names"], symbol_col].values
    else:
        de_results["symbol"] = de_results["gene_symbol"]

    # ── Gene ontology / functional categories ──
    # Manually curate key functional categories from top DE genes
    print("\n\n=== Functional interpretation ===")

    categories = {
        "Myeloid identity": ["CD14", "CD68", "ITGAM", "S100A9", "CD163", "MRC1",
                             "TLR5", "C3AR1", "FCGR1A"],
        "FDC identity": ["CR2", "FCER2", "CXCL13", "PDPN", "CLU", "SERPINE1",
                         "FDCSP", "CD55"],
        "Chemokines": ["CXCL13", "CXCL12", "CCL21", "CCL19"],
        "Antigen presentation": ["HLA-DRA", "HLA-A", "HLA-B", "HLA-C", "B2M"],
        "Immune checkpoint": ["VSIR", "CD274", "IDO1"],
        "B cell support": ["TNFSF13B", "TNFSF13", "ICAM1", "VCAM1"],
        "Immunosuppressive": ["IL10", "TGFB1", "IDO1"],
    }

    for cat_name, genes in categories.items():
        print(f"\n  {cat_name}:")
        for gene in genes:
            match = imc_df[imc_df["gene"] == gene]
            if len(match) > 0:
                r = match.iloc[0]
                direction = "↑" if r["logFC"] > 0 else "↓"
                print(f"    {r['label']:15s} {direction} logFC={r['logFC']:+.3f}  "
                      f"padj={r['padj']:.2e} {r['star']}")

    # ── Save full DE results ──
    out_csv = output_dir / "de_cd14pos_vs_neg_fdcs.csv"
    de_results.to_csv(out_csv, index=False)
    print(f"\nFull DE results saved: {out_csv}")

    # ── Make figure ──
    make_figure(adata_fdc, cd14_expr, cd14_pos, de_results, imc_df,
                n_pos, n_neg, output_dir, has_symbols, symbol_col)


def make_figure(adata_fdc, cd14_expr, cd14_pos, de_results, imc_df,
                n_pos, n_neg, output_dir, has_symbols, symbol_col):
    """6-panel figure: CD14+ vs CD14- FDC transcriptomic comparison."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.40,
                  left=0.07, right=0.95, top=0.94, bottom=0.05)

    # Colors
    COL_POS = "#FF7F00"   # orange for CD14+
    COL_NEG = "#377EB8"   # blue for CD14-

    # ── (a) CD14 expression distribution on FDCs ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    ax_a.hist(cd14_expr[cd14_expr == 0], bins=1, color=COL_NEG, alpha=0.7,
              label=f"CD14- (n={n_neg})", edgecolor="white")
    if (cd14_expr > 0).any():
        ax_a.hist(cd14_expr[cd14_expr > 0], bins=30, color=COL_POS, alpha=0.7,
                  label=f"CD14+ (n={n_pos})", edgecolor="white")
    ax_a.set_xlabel("CD14 mRNA expression (log-normalized)")
    ax_a.set_ylabel("Number of FDCs")
    ax_a.set_title(f"CD14 expression in FDCs\n(Han et al. 2022 scRNA-seq)")
    ax_a.legend(fontsize=9)
    # Annotate percentage
    pct_pos = 100 * n_pos / (n_pos + n_neg)
    ax_a.text(0.95, 0.85, f"{pct_pos:.0f}% CD14+",
              transform=ax_a.transAxes, ha="right", fontsize=11,
              fontweight="bold", color=COL_POS)

    # ── (b) Volcano plot ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    lfc = de_results["logfoldchanges"].values
    pvals = de_results["pvals_adj"].values
    neg_log_p = -np.log10(np.clip(pvals, 1e-300, 1))

    # Color: significant up (orange), significant down (blue), NS (gray)
    sig_up = (pvals < 0.05) & (lfc > 0.25)
    sig_down = (pvals < 0.05) & (lfc < -0.25)
    ns = ~(sig_up | sig_down)
    ax_b.scatter(lfc[ns], neg_log_p[ns], s=2, c="#CCCCCC", alpha=0.3, rasterized=True)
    ax_b.scatter(lfc[sig_down], neg_log_p[sig_down], s=4, c=COL_NEG, alpha=0.5,
                 label=f"Down (n={(sig_down).sum()})", rasterized=True)
    ax_b.scatter(lfc[sig_up], neg_log_p[sig_up], s=4, c=COL_POS, alpha=0.5,
                 label=f"Up (n={(sig_up).sum()})", rasterized=True)

    # Label top genes
    # Get gene symbols for labeling
    if has_symbols and symbol_col:
        ensid_to_sym = dict(zip(adata_fdc.var.index, adata_fdc.var[symbol_col]))
    else:
        ensid_to_sym = {v: k for k, v in KEY_GENES.items()}

    for direction, mask in [("up", sig_up), ("down", sig_down)]:
        subset = de_results[mask].copy()
        subset["neg_log_p"] = -np.log10(np.clip(subset["pvals_adj"].values, 1e-300, 1))
        # Rank by combined score
        subset["score"] = subset["neg_log_p"] * abs(subset["logfoldchanges"])
        top = subset.nlargest(8, "score")
        for _, row in top.iterrows():
            sym = ensid_to_sym.get(row["names"], row["names"][:10])
            ax_b.annotate(
                sym, (row["logfoldchanges"], row["neg_log_p"]),
                fontsize=6, alpha=0.8,
                xytext=(5, 3), textcoords="offset points",
            )

    ax_b.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.5)
    ax_b.axvline(0.25, color="gray", linestyle="--", linewidth=0.5)
    ax_b.axvline(-0.25, color="gray", linestyle="--", linewidth=0.5)
    ax_b.set_xlabel("Log₂ fold change (CD14+ / CD14-)")
    ax_b.set_ylabel("-log₁₀(adjusted p-value)")
    ax_b.set_title("Differential expression\nCD14+ vs CD14- FDCs")
    ax_b.legend(fontsize=8, loc="upper left")

    # ── (c) IMC-relevant markers: logFC barplot ──
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c")
    imc_plot = imc_df.sort_values("logFC", ascending=True).copy()
    # Only show markers with some expression
    imc_plot = imc_plot[imc_plot["pct_pos"] + imc_plot["pct_neg"] > 1]
    y_pos = range(len(imc_plot))
    colors = [COL_POS if lfc > 0 else COL_NEG for lfc in imc_plot["logFC"]]
    edge_colors = ["black" if p < 0.05 else "none" for p in imc_plot["padj"]]
    ax_c.barh(list(y_pos), imc_plot["logFC"].values, color=colors,
              edgecolor=edge_colors, linewidth=0.8)
    ax_c.set_yticks(list(y_pos))
    ax_c.set_yticklabels(imc_plot["label"].values, fontsize=7)
    ax_c.axvline(0, color="black", linewidth=0.5)
    ax_c.set_xlabel("Log₂ fold change (CD14+ / CD14-)")
    ax_c.set_title("IMC-relevant markers\n(black border = FDR < 0.05)")
    # Annotate significance
    for i, (_, row) in enumerate(imc_plot.iterrows()):
        if row["star"]:
            x_pos = row["logFC"] + (0.02 if row["logFC"] > 0 else -0.02)
            ha = "left" if row["logFC"] > 0 else "right"
            ax_c.text(x_pos, i, row["star"], va="center", ha=ha,
                      fontsize=7, color="#333")

    # ── (d) Detection rate comparison ──
    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, "d")
    # Show key markers: detection rate in CD14+ vs CD14-
    key_markers = ["CD14", "CD68", "VSIR", "ITGAM", "CR2", "CXCL13",
                   "HLA-DRA", "IDO1", "ICAM1", "TGFB1", "IL10",
                   "TNFSF13B", "PDPN", "FCER2", "CCL21"]
    det_data = imc_df[imc_df["gene"].isin(key_markers)].copy()
    det_data = det_data.sort_values("pct_pos", ascending=True)
    y_pos = np.arange(len(det_data))
    bar_h = 0.35
    ax_d.barh(y_pos - bar_h/2, det_data["pct_pos"].values, bar_h,
              color=COL_POS, alpha=0.8, label="CD14+ FDCs")
    ax_d.barh(y_pos + bar_h/2, det_data["pct_neg"].values, bar_h,
              color=COL_NEG, alpha=0.8, label="CD14- FDCs")
    ax_d.set_yticks(list(y_pos))
    ax_d.set_yticklabels(det_data["label"].values, fontsize=8)
    ax_d.set_xlabel("% cells expressing gene")
    ax_d.set_title("Detection rates\n(% cells with expression > 0)")
    ax_d.legend(fontsize=8, loc="lower right")

    # ── (e) Mean expression comparison (dot plot style) ──
    ax_e = fig.add_subplot(gs[1, 1])
    panel_label(ax_e, "e")
    # Functional categories
    cat_genes = {
        "Myeloid": ["CD14", "CD68", "ITGAM", "S100A9", "TLR5", "C3AR1", "FCGR1A"],
        "FDC": ["CR2", "FCER2", "CXCL13", "PDPN", "CLU", "FDCSP", "CD55"],
        "Checkpoint": ["VSIR", "IDO1", "CD274"],
        "Chemokine": ["CXCL12", "CCL21", "CCL19"],
        "Ag. pres.": ["HLA-DRA", "HLA-A", "B2M"],
        "B support": ["TNFSF13B", "TNFSF13", "ICAM1"],
    }
    all_genes_ordered = []
    cat_positions = {}
    pos = 0
    for cat, genes in cat_genes.items():
        cat_positions[cat] = (pos, pos + len(genes) - 1)
        for g in genes:
            all_genes_ordered.append((cat, g))
            pos += 1

    y_vals = []
    x_pos_vals = []
    x_neg_vals = []
    size_pos_vals = []
    size_neg_vals = []
    labels = []
    for i, (cat, gene) in enumerate(all_genes_ordered):
        ensid = KEY_GENES.get(gene)
        if not ensid:
            continue
        expr = get_gene_expr(adata_fdc, ensid)
        if expr is None:
            continue
        mp = float(expr[cd14_pos].mean())
        mn = float(expr[~cd14_pos].mean())
        pp = 100 * (expr[cd14_pos] > 0).mean()
        pn = 100 * (expr[~cd14_pos] > 0).mean()
        y_vals.append(i)
        x_pos_vals.append(mp)
        x_neg_vals.append(mn)
        size_pos_vals.append(pp)
        size_neg_vals.append(pn)
        labels.append(GENE_LABELS.get(gene, gene))

    # Scale dot size
    max_size = 200
    size_pos = [s / 100 * max_size for s in size_pos_vals]
    size_neg = [s / 100 * max_size for s in size_neg_vals]

    ax_e.scatter(x_pos_vals, y_vals, s=size_pos, c=COL_POS, alpha=0.7,
                 edgecolors="black", linewidth=0.5, label="CD14+", zorder=3)
    ax_e.scatter([-0.15 + x for x in x_neg_vals], y_vals, s=size_neg, c=COL_NEG,
                 alpha=0.7, edgecolors="black", linewidth=0.5, label="CD14-", zorder=3)

    ax_e.set_yticks(y_vals)
    ax_e.set_yticklabels(labels, fontsize=7)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Mean expression")
    ax_e.set_title("Dot plot: mean expression × detection\n(size ∝ % expressing)")
    ax_e.legend(fontsize=8, loc="lower right")

    # Add category labels on right
    for cat, (start, end) in cat_positions.items():
        mid = (start + end) / 2
        ax_e.text(ax_e.get_xlim()[1] + 0.05, mid, cat, va="center",
                  fontsize=7, fontweight="bold", fontstyle="italic")

    # ── (f) Concordance with IMC findings ──
    ax_f = fig.add_subplot(gs[1, 2])
    panel_label(ax_f, "f")
    # IMC protein vs scRNA logFC for markers in common
    # IMC data from our NOTEBOOK analysis: CD14-high vs CD14-low FDC marker differences
    imc_protein_delta = {
        "CD21": 2.07, "CXCL13": 1.06, "CXCL12": 0.93, "HLA-I": 1.22,
        "CD11c": 0.89, "VISTA": 0.61, "CD68": 0.74, "CD11b": 0.78,
        "CCL21": 0.54, "IDO": 0.07, "HLA-DR": 0.16, "PDPN": 0.12,
    }
    # Map to scRNA gene names
    imc_to_scrna = {
        "CD21": "CR2", "CXCL13": "CXCL13", "CXCL12": "CXCL12",
        "VISTA": "VSIR", "CD68": "CD68", "CD11b": "ITGAM",
        "CCL21": "CCL21", "IDO": "IDO1", "HLA-DR": "HLA-DRA",
        "PDPN": "PDPN",
    }

    concordance_x = []  # IMC protein delta
    concordance_y = []  # scRNA logFC
    concordance_labels = []
    for imc_name, delta in imc_protein_delta.items():
        scrna_gene = imc_to_scrna.get(imc_name)
        if not scrna_gene:
            continue
        match = imc_df[imc_df["gene"] == scrna_gene]
        if len(match) > 0:
            concordance_x.append(delta)
            concordance_y.append(match.iloc[0]["logFC"])
            concordance_labels.append(imc_name)

    if len(concordance_x) > 2:
        ax_f.scatter(concordance_x, concordance_y, s=60, c="#333", zorder=3)
        for x, y, lab in zip(concordance_x, concordance_y, concordance_labels):
            ax_f.annotate(lab, (x, y), fontsize=8, xytext=(5, 5),
                          textcoords="offset points")

        # Correlation
        rho, p = stats.spearmanr(concordance_x, concordance_y)
        # Fit line
        z = np.polyfit(concordance_x, concordance_y, 1)
        xline = np.linspace(min(concordance_x) - 0.1, max(concordance_x) + 0.1, 50)
        ax_f.plot(xline, np.polyval(z, xline), "r--", alpha=0.5)
        ax_f.text(0.05, 0.92, f"Spearman ρ = {rho:.2f}\np = {p:.3f}",
                  transform=ax_f.transAxes, fontsize=10,
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax_f.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax_f.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    else:
        ax_f.text(0.5, 0.5, "Insufficient overlapping markers",
                  transform=ax_f.transAxes, ha="center")

    ax_f.set_xlabel("IMC protein Δ (CD14-high − CD14-low FDCs)")
    ax_f.set_ylabel("scRNA-seq logFC (CD14+ / CD14-)")
    ax_f.set_title("IMC protein vs scRNA-seq concordance\n(CD14+ vs CD14- FDCs)")

    out_path = output_dir / "fig_scrna_cd14_fdc.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="scRNA-seq analysis of CD14+ vs CD14- FDCs",
    )
    parser.add_argument(
        "--scrna", default="data/external/steen2022_fl_scrna.h5ad",
        help="Path to Han 2022 scRNA-seq h5ad file",
    )
    parser.add_argument(
        "--output-dir", default="output/hypotheses_v8",
        help="Output directory for figures and tables",
    )
    args = parser.parse_args()
    run_analysis(args.scrna, args.output_dir)
