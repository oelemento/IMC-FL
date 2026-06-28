#!/usr/bin/env python
"""scRNA-seq figure: CD14+ vs CD14- FDCs — library-size-corrected analysis.

Han et al. 2022 (Blood Cancer Discovery), 108 FDCs from 20 FL patients.
Key finding: CD14+ FDCs are more transcriptionally active and express a
specific gene program (immune regulation, TGF-β modulation, anti-apoptosis)
independent of global library complexity.
"""
import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from pathlib import Path


def panel_label(ax, letter):
    ax.text(
        -0.08, 1.06, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top",
    )


KEY_GENES = {
    "CD14": "ENSG00000170458",
    "CD68": "ENSG00000129226",
    "VSIR": "ENSG00000107738",
    "ITGAM": "ENSG00000169896",
    "CR2": "ENSG00000117322",
    "FCER2": "ENSG00000104921",
    "CXCL13": "ENSG00000156234",
    "PDPN": "ENSG00000162493",
    "HLA-DRA": "ENSG00000204287",
    "IDO1": "ENSG00000131203",
    "CD274": "ENSG00000120217",
    "CXCL12": "ENSG00000107562",
    "CCL21": "ENSG00000137077",
    "HLA-A": "ENSG00000206503",
    "HLA-B": "ENSG00000234745",
    "HLA-C": "ENSG00000204525",
    "B2M": "ENSG00000166710",
    "TNFSF13B": "ENSG00000102524",
    "TNFSF13": "ENSG00000161955",
    "IL6": "ENSG00000136244",
    "TGFB1": "ENSG00000105329",
    "ICAM1": "ENSG00000090339",
    "VCAM1": "ENSG00000162692",
    "CLU": "ENSG00000120885",
    "FN1": "ENSG00000115414",
}

GENE_LABELS = {
    "CD14": "CD14", "CD68": "CD68", "VSIR": "VISTA", "ITGAM": "CD11b",
    "CR2": "CD21", "FCER2": "CD23", "CXCL13": "CXCL13", "PDPN": "PDPN",
    "HLA-DRA": "HLA-DRA", "CD274": "PD-L1", "IDO1": "IDO1",
    "CXCL12": "CXCL12", "CCL21": "CCL21",
    "HLA-A": "HLA-A", "HLA-B": "HLA-B", "HLA-C": "HLA-C", "B2M": "B2M",
    "TNFSF13B": "BAFF", "TNFSF13": "APRIL",
    "IL6": "IL6", "TGFB1": "TGF-β1",
    "ICAM1": "ICAM1", "VCAM1": "VCAM1", "CLU": "Clusterin", "FN1": "FN1",
}


def get_gene_expr(adata, ensembl_id):
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

    adata = sc.read_h5ad(h5ad_path)
    fl = adata[adata.obs["disease"] == "follicular lymphoma"].copy()
    fdcs = fl[fl.obs["cell_type"] == "follicular dendritic cell"].copy()
    n_fdc = fdcs.shape[0]
    print(f"FDCs: {n_fdc}")

    ensid_to_name = dict(zip(fdcs.var.index, fdcs.var["feature_name"]))

    # CD14 expression
    cd14 = get_gene_expr(fdcs, KEY_GENES["CD14"])
    pos_mask = cd14 > 0
    n_pos = pos_mask.sum()
    n_neg = (~pos_mask).sum()

    # Library complexity
    n_genes_per = np.array((fdcs.X > 0).sum(axis=1)).ravel()
    raw_X = fdcs.raw.X
    if hasattr(raw_X, "toarray"):
        umi_per_cell = np.array(raw_X.sum(axis=1)).ravel()
    else:
        umi_per_cell = raw_X.sum(axis=1)

    # ── Partial correlation (library-size corrected) ──
    cd14_rank = rankdata(cd14)
    ng_rank = rankdata(n_genes_per)
    slope, intercept, _, _, _ = stats.linregress(ng_rank, cd14_rank)
    cd14_resid = cd14_rank - (slope * ng_rank + intercept)

    results = []
    for j in range(fdcs.shape[1]):
        vals = fdcs.X[:, j]
        if hasattr(vals, "toarray"):
            vals = vals.toarray().ravel()
        vals = np.array(vals).ravel()
        if vals.std() == 0:
            continue
        gene_rank = rankdata(vals)
        slope_g, int_g, _, _, _ = stats.linregress(ng_rank, gene_rank)
        gene_resid = gene_rank - (slope_g * ng_rank + int_g)
        rho, p = stats.pearsonr(cd14_resid, gene_resid)
        results.append({
            "ensid": fdcs.var.index[j],
            "symbol": ensid_to_name.get(fdcs.var.index[j], "?"),
            "rho": rho,
            "pval": p,
        })

    pcorr = pd.DataFrame(results)
    _, pcorr["padj"], _, _ = multipletests(pcorr["pval"], method="fdr_bh")

    # Filter
    pcorr_bio = pcorr[
        ~pcorr["symbol"].str.match(r"^(MT-|RP[SL]\d|MRPL|MRPS)", na=False)
        & (pcorr["symbol"] != "CD14")
    ].copy()

    sig = pcorr_bio[pcorr_bio["padj"] < 0.05].sort_values("rho", ascending=False)

    # Also get uncorrected correlations for IMC markers
    uncorr = pd.read_csv(output_dir / "corr_cd14_fdcs.csv")

    # ── Figure ──
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.42,
                  left=0.07, right=0.95, top=0.93, bottom=0.06)

    COL_POS = "#FF7F00"
    COL_NEG = "#377EB8"
    COL_SIG = "#E41A1C"

    # ── (a) CD14 distribution + library size ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    ax_a.scatter(n_genes_per[~pos_mask], cd14[~pos_mask], s=25,
                 c=COL_NEG, alpha=0.6, label=f"CD14- (n={n_neg})",
                 edgecolors="white", linewidth=0.3, zorder=2)
    ax_a.scatter(n_genes_per[pos_mask], cd14[pos_mask], s=40,
                 c=COL_POS, alpha=0.8, label=f"CD14+ (n={n_pos})",
                 edgecolors="black", linewidth=0.5, zorder=3)
    rho_lib, p_lib = stats.spearmanr(cd14, n_genes_per)
    ax_a.set_xlabel("Genes detected per cell")
    ax_a.set_ylabel("CD14 mRNA expression")
    ax_a.set_title("CD14+ FDCs: higher transcriptional complexity\n"
                    f"(ρ = {rho_lib:.2f}, p = {p_lib:.1e})")
    ax_a.legend(fontsize=9)
    # Add median lines
    ax_a.axvline(np.median(n_genes_per[pos_mask]), color=COL_POS,
                 linestyle="--", alpha=0.5, linewidth=1)
    ax_a.axvline(np.median(n_genes_per[~pos_mask]), color=COL_NEG,
                 linestyle="--", alpha=0.5, linewidth=1)

    # ── (b) Top correlated genes (library-corrected) ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    top_genes = sig.head(27)  # all significant
    y_pos = range(len(top_genes))
    bars = ax_b.barh(
        list(y_pos), top_genes["rho"].values,
        color=COL_POS, edgecolor="white", linewidth=0.5,
    )
    # Color the one negative gene differently
    for i, (_, row) in enumerate(top_genes.iterrows()):
        if row["rho"] < 0:
            bars[i].set_color(COL_NEG)
    ax_b.set_yticks(list(y_pos))
    ax_b.set_yticklabels(top_genes["symbol"].values, fontsize=7)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Partial Spearman ρ with CD14\n(library-size corrected)")
    ax_b.set_title(f"Genes correlated with CD14 on FDCs\n"
                    f"({len(sig)} genes, FDR < 0.05, n={n_fdc})")
    # Annotate padj
    for i, (_, row) in enumerate(top_genes.iterrows()):
        ax_b.text(row["rho"] + 0.005, i, f"p={row['padj']:.1e}",
                  va="center", fontsize=5.5, color="#555")

    # ── (c) Functional annotation of top genes ──
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c")
    # Manual functional annotation
    gene_functions = {
        "GATD3": ("Metabolism", "Glutamine amidotransferase"),
        "TCIM": ("Immune reg.", "NF-κB activator"),
        "NPPC": ("Signaling", "C-natriuretic peptide"),
        "PACSIN2": ("Membrane", "Endocytosis/membrane dynamics"),
        "RASA3": ("RAS pathway", "RAS-GAP (suppressor)"),
        "G0S2": ("Anti-apoptotic", "G0/G1 switch, lipid metabolism"),
        "KLF13": ("Transcription", "Krüppel-like factor"),
        "NINJ1": ("Inflammation", "Membrane rupture, pyroptosis"),
        "KMT5B": ("Epigenetic", "Histone methyltransferase"),
        "BAMBI": ("TGF-β mod.", "TGF-β pseudoreceptor inhibitor"),
        "CYP27A1": ("Lipid metab.", "Cholesterol 27-hydroxylase"),
        "LMO4": ("Transcription", "LIM domain transcription factor"),
        "TPST1": ("Post-transl.", "Tyrosylprotein sulfotransferase"),
        "SSTR2": ("Signaling", "Somatostatin receptor 2"),
        "NEDD4L": ("TGF-β mod.", "E3 ubiquitin ligase"),
        "RNASE4": ("Angiogenesis", "Ribonuclease/angiogenin"),
        "SIGIRR": ("Immune reg.", "IL-1R/TLR signaling suppressor"),
    }

    categories = {}
    for gene, (cat, desc) in gene_functions.items():
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((gene, desc))

    # Build table
    y = 0
    y_labels = []
    cat_spans = {}
    for cat, genes in categories.items():
        cat_start = y
        for gene, desc in genes:
            y_labels.append(f"{gene}")
            y += 1
        cat_spans[cat] = (cat_start, y - 1)

    ax_c.set_xlim(0, 3)
    ax_c.set_ylim(-0.5, len(y_labels) - 0.5)
    ax_c.invert_yaxis()
    ax_c.axis("off")
    ax_c.set_title("Functional annotation\nof CD14-correlated genes")

    y = 0
    for cat, genes in categories.items():
        for gene, desc in genes:
            # Category color coding
            cat_colors = {
                "Immune reg.": "#E41A1C",
                "TGF-β mod.": "#984EA3",
                "Anti-apoptotic": "#4DAF4A",
                "Lipid metab.": "#FF7F00",
                "Metabolism": "#FF7F00",
                "Inflammation": "#E41A1C",
                "Signaling": "#377EB8",
                "RAS pathway": "#A65628",
                "Transcription": "#666666",
                "Epigenetic": "#666666",
                "Membrane": "#377EB8",
                "Post-transl.": "#377EB8",
                "Angiogenesis": "#4DAF4A",
            }
            color = cat_colors.get(cat, "#333333")
            ax_c.text(0.05, y, gene, fontsize=8, fontweight="bold",
                      color=color, va="center")
            ax_c.text(1.0, y, desc, fontsize=7, va="center", color="#333")
            y += 1
        # Draw category bracket
        start, end = cat_spans[cat]
        mid = (start + end) / 2
        ax_c.plot([-0.3, -0.3], [start - 0.3, end + 0.3], color=color,
                  linewidth=2, clip_on=False)
        ax_c.text(-0.4, mid, cat, fontsize=7, rotation=90, ha="right",
                  va="center", color=color, fontweight="bold")

    # ── (d) IMC markers: uncorrected vs corrected ──
    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, "d")
    imc_markers_ordered = [
        "CD68", "ICAM1", "CD23", "B2M", "VCAM1", "HLA-B", "PD-L1",
        "VISTA", "CD21", "HLA-A", "CLU", "CXCL13", "HLA-DRA",
        "HLA-C", "IL6", "PDPN", "CCL21",
    ]
    imc_ensids = {v: k for k, v in KEY_GENES.items()}
    y_data = []
    uncorr_vals = []
    corr_vals = []
    for gene in imc_markers_ordered:
        ensid = KEY_GENES.get(gene)
        if not ensid:
            continue
        uc = uncorr[uncorr["ensid"] == ensid]
        pc = pcorr[pcorr["ensid"] == ensid]
        if len(uc) == 0 or len(pc) == 0:
            continue
        label = GENE_LABELS.get(gene, gene)
        y_data.append(label)
        uncorr_vals.append(uc.iloc[0]["rho"])
        corr_vals.append(pc.iloc[0]["rho"])

    y_pos = np.arange(len(y_data))
    bar_h = 0.35
    ax_d.barh(y_pos - bar_h / 2, uncorr_vals, bar_h, color="#AAAAAA",
              alpha=0.7, label="Uncorrected", edgecolor="white")
    ax_d.barh(y_pos + bar_h / 2, corr_vals, bar_h, color=COL_POS,
              alpha=0.8, label="Library-corrected", edgecolor="white")
    ax_d.set_yticks(list(y_pos))
    ax_d.set_yticklabels(y_data, fontsize=8)
    ax_d.axvline(0, color="black", linewidth=0.5)
    ax_d.set_xlabel("Spearman ρ with CD14")
    ax_d.set_title("IMC markers: library-size correction\nattenuates most correlations")
    ax_d.legend(fontsize=8, loc="lower right")
    ax_d.invert_yaxis()

    # ── (e) Per-donor CD14+ FDC frequency ──
    ax_e = fig.add_subplot(gs[1, 1])
    panel_label(ax_e, "e")
    donor_stats = []
    for donor, group in fdcs.obs.groupby("donor_id"):
        n = len(group)
        if n < 1:
            continue
        cd14_sub = cd14[fdcs.obs["donor_id"] == donor]
        n_pos_d = (cd14_sub > 0).sum()
        donor_stats.append({"donor": donor, "n_fdc": n, "n_cd14pos": n_pos_d,
                            "pct_cd14pos": 100 * n_pos_d / n})

    donor_df = pd.DataFrame(donor_stats).sort_values("pct_cd14pos", ascending=True)
    y_pos = range(len(donor_df))
    colors = [COL_POS if p > 0 else COL_NEG for p in donor_df["pct_cd14pos"]]
    ax_e.barh(list(y_pos), donor_df["pct_cd14pos"].values, color=colors,
              edgecolor="white", linewidth=0.5)
    ax_e.set_yticks(list(y_pos))
    ax_e.set_yticklabels(
        [f"{r['donor']} (n={r['n_fdc']})" for _, r in donor_df.iterrows()],
        fontsize=7,
    )
    ax_e.set_xlabel("% FDCs expressing CD14")
    ax_e.set_title(f"CD14+ FDC frequency by patient\n({len(donor_df)} FL patients)")
    # Annotate counts
    for i, (_, row) in enumerate(donor_df.iterrows()):
        if row["n_cd14pos"] > 0:
            ax_e.text(row["pct_cd14pos"] + 1, i,
                      f"{row['n_cd14pos']}/{row['n_fdc']}",
                      va="center", fontsize=6, color="#555")

    # ── (f) Summary model ──
    ax_f = fig.add_subplot(gs[1, 2])
    panel_label(ax_f, "f")
    ax_f.axis("off")
    ax_f.set_xlim(0, 10)
    ax_f.set_ylim(0, 10)
    ax_f.set_title("scRNA-seq model: CD14+ FDC program")

    # Summary text
    summary_lines = [
        ("CD14+ FDCs (19% of FL FDCs) are transcriptionally", 9.5, 11, False),
        ("hyperactive cells with a specific gene program:", 9.0, 11, False),
        ("", 8.5, 11, False),
        ("  Immune regulation", 8.2, 11, True),
        ("    TCIM (NF-κB activator), KLF13, SIGIRR (IL-1R suppressor)", 7.7, 8, False),
        ("  TGF-β pathway modulation", 7.2, 11, True),
        ("    BAMBI (TGF-β inhibitor), NEDD4L (E3 ligase)", 6.7, 8, False),
        ("  Anti-apoptotic / metabolic", 6.2, 11, True),
        ("    G0S2 (anti-apoptotic), CYP27A1 (cholesterol)", 5.7, 8, False),
        ("  RAS pathway suppression", 5.2, 11, True),
        ("    RASA3 ↑ (RAS-GAP), MAP2K2 ↓ (MEK2)", 4.7, 8, False),
        ("  Inflammatory signaling", 4.2, 11, True),
        ("    NINJ1 (membrane rupture), NPPC (natriuretic peptide)", 3.7, 8, False),
        ("", 3.2, 11, False),
        ("Key: IMC protein differences (CD68, VCAM1, HLA) are", 2.7, 11, False),
        ("explained by global transcriptional activity, not", 2.2, 11, False),
        ("gene-specific programs. The 27 CD14-specific genes", 1.7, 11, False),
        ("above represent biology invisible to our IMC panels.", 1.2, 11, False),
    ]
    for text, y, fontsize_val, bold in summary_lines:
        ax_f.text(0.3, y, text, fontsize=8,
                  fontweight="bold" if bold else "normal",
                  color="#E41A1C" if bold else "#333",
                  va="top")

    out_path = output_dir / "fig_scrna_cd14_fdc_v2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrna", default="data/external/steen2022_fl_scrna.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()
    run_analysis(args.scrna, args.output_dir)
