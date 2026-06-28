#!/usr/bin/env python
"""Combined figure: CD14 signal decomposition, scRNA-seq validation, and CD14+ FDC survival.

Panels:
  (a) CD14 expression by cell type (IMC protein) — FDC is #2 after myeloid
  (b) Composition of CD14+ cells — myeloid ~35%, FDC ~25%, spillover ~40%
  (c) Spillover gradient — % CD14+ vs distance to nearest myeloid cell
  (d) CD14 mRNA by cell type (scRNA-seq, Han 2022) — validates protein finding
  (e) Marker heatmap: myeloid vs FDC identity genes (z-scored)
  (f) Forest: All FDCs vs CD14+ FDCs — PFS and OS Cox HRs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figure_style import (TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
                          ANNOT_SIZE, PANEL_LABEL_SIZE)

import h5py
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.spatial import KDTree
from scipy import stats
from collections import Counter
from lifelines import CoxPHFitter

from src.clinical_linkage import normalize_sample_id


# ── Shared helpers ────────────────────────────────────────────────────────────
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


def is_tumor_core(s):
    sl = s.lower()
    if "_ton_" in sl or "_adr_" in sl:
        return False
    for tissue in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if tissue in sl:
            return False
    if s == "Biomax_ROI_006":
        return False
    return True


def panel_label(ax, letter):
    ax.text(
        -0.08, 1.06, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top",
    )


# ── IMC constants ─────────────────────────────────────────────────────────────
MYELOID_TYPES = {
    "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Myeloid (S100A9+)", "Dendritic cells",
}

CAT_COLORS = {
    "Myeloid": "#E41A1C",
    "FDC": "#FF7F00",
    "Spillover": "#999999",
}

CT_SHORT_IMC = {
    "M1 Macrophages": "M1 Mac",
    "M2 Macrophages": "M2 Mac",
    "Macrophages": "Mac (generic)",
    "Myeloid (S100A9+)": "S100A9+ Myeloid",
    "Dendritic cells": "DC",
    "B cells (BCL2+)": "BCL2+ B",
    "B cells (PAX5+)": "PAX5+ B",
    "B cells": "B cells",
    "CD4 T cells": "CD4 T",
    "CD8 T cells": "CD8 T",
    "Endothelial": "Endothelial",
    "Mixed / Border cells": "Mixed/Border",
    "Stromal / CAF": "Stromal/CAF",
    "FRC (PDPN+)": "FRC",
    "Histiocytes (CD44hi)": "Histiocytes",
    "Other": "Other",
    "Low quality / Unassigned": "Unassigned",
    "pDC": "pDC",
    "FDC": "FDC",
}


# ── scRNA constants ───────────────────────────────────────────────────────────
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
    "CD274": "ENSG00000120217",
    "IDO1": "ENSG00000131203",
    "MS4A1": "ENSG00000156738",
    "TLR5": "ENSG00000187554",
    "C3AR1": "ENSG00000171860",
    "FCGR1A": "ENSG00000150337",
}

GENE_LABELS = {
    "CD14": "CD14", "CD68": "CD68", "VSIR": "VISTA", "ITGAM": "CD11b",
    "CR2": "CD21", "FCER2": "CD23", "CXCL13": "CXCL13", "PDPN": "PDPN",
    "HLA-DRA": "HLA-DRA", "CD274": "PD-L1", "IDO1": "IDO1", "MS4A1": "CD20",
    "TLR5": "TLR5", "C3AR1": "C3AR1", "FCGR1A": "FCGR1A",
}

CT_SHORT_SCRNA = {
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


def get_gene_expr(adata, ensembl_id):
    idx = list(adata.var.index).index(ensembl_id)
    vals = adata.X[:, idx]
    if hasattr(vals, "toarray"):
        vals = vals.toarray().ravel()
    return np.array(vals).ravel()


# ── Data extraction ───────────────────────────────────────────────────────────
EXCLUDE_ROIS = {"A1_ROI_003"}


def extract_spillover_data(s_panel_path):
    """Extract IMC CD14 data for panels a/b/c."""
    print("Loading S-panel for CD14 decomposition...")
    f = h5py.File(s_panel_path, "r")
    X = f["X"][:]
    markers = [v.decode() for v in f["var"]["_index"][:]]
    cell_types = load_array(f, "cell_type")
    sample_ids = load_array(f, "sample_id")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()

    marker_idx = {m: i for i, m in enumerate(markers)}
    cd14_col = marker_idx["CD14"]

    # Raw (arcsinh, non-negative) CD14 — used for panel (a) display.
    # Falls back to scaled .X if .raw unavailable.
    try:
        import anndata as _ad
        _a = _ad.read_h5ad(s_panel_path, backed="r")
        if _a.raw is not None:
            raw_cd14_idx = list(_a.raw.var_names).index("CD14")
            X_raw = np.asarray(_a.raw.X[:, raw_cd14_idx]).ravel()
        else:
            X_raw = X[:, cd14_col].copy()
    except Exception:
        X_raw = X[:, cd14_col].copy()

    tumor_mask = np.array(
        [is_tumor_core(s) and not s.startswith("Biomax") and s not in EXCLUDE_ROIS
         for s in sample_ids]
    )
    X = X[tumor_mask]
    X_raw = X_raw[tumor_mask]
    cell_types = cell_types[tumor_mask]
    sample_ids = sample_ids[tumor_mask]
    cx = cx[tumor_mask]
    cy = cy[tumor_mask]
    cd14_vals = X[:, cd14_col]
    print(f"  Tumor cells: {len(cell_types):,}")

    is_myeloid = np.array([ct in MYELOID_TYPES for ct in cell_types])
    is_fdc = cell_types == "FDC"
    is_other = ~is_myeloid & ~is_fdc & (cell_types != "Low quality / Unassigned")

    # Panel (a): Mean CD14 by cell type (raw arcsinh intensity, non-negative)
    unique_cts = sorted(set(cell_types) - {"Low quality / Unassigned"})
    ct_means = []
    for ct in unique_cts:
        mask = cell_types == ct
        ct_means.append((ct, mask.sum(), float(X_raw[mask].mean())))
    ct_means.sort(key=lambda x: -x[2])

    # Panel (b): Composition of CD14+ cells
    cd14_pos = cd14_vals > 0.5
    pos_cts = Counter(cell_types[cd14_pos])
    n_pos = cd14_pos.sum()
    mye_pos = sum(n for ct, n in pos_cts.items() if ct in MYELOID_TYPES)
    fdc_pos = pos_cts.get("FDC", 0)
    other_pos = n_pos - mye_pos - fdc_pos

    # Panel (c): Distance gradient
    print("  Computing proximity gradient...")
    unique_rois = np.unique(sample_ids)
    all_dists = []
    all_cd14 = []
    for roi in unique_rois:
        rmask = sample_ids == roi
        roi_mye = np.where(rmask & is_myeloid)[0]
        roi_target = np.where(rmask & is_other)[0]
        if len(roi_mye) < 50 or len(roi_target) < 100:
            continue
        tree = KDTree(np.column_stack([cx[roi_mye], cy[roi_mye]]))
        target_coords = np.column_stack([cx[roi_target], cy[roi_target]])
        dists, _ = tree.query(target_coords, k=1)
        all_dists.extend(dists)
        all_cd14.extend(X[roi_target, cd14_col])

    all_dists = np.array(all_dists)
    all_cd14 = np.array(all_cd14)
    rho, p_prox = stats.spearmanr(all_dists, all_cd14)
    print(f"  Proximity rho={rho:.4f}, p={p_prox:.2e}")

    bins = [0, 5, 10, 15, 20, 30, 50, 100]
    bin_pct_pos = []
    bin_labels = []
    bin_ns = []
    for i in range(len(bins) - 1):
        mask = (all_dists >= bins[i]) & (all_dists < bins[i + 1])
        if mask.sum() > 0:
            bin_pct_pos.append(100 * (all_cd14[mask] > 0.5).mean())
            bin_labels.append(f"{bins[i]}\u2013{bins[i+1]}")
            bin_ns.append(int(mask.sum()))

    return {
        "ct_means": ct_means,
        "n_pos": n_pos, "mye_pos": mye_pos, "fdc_pos": fdc_pos, "other_pos": other_pos,
        "bin_labels": bin_labels, "bin_pct_pos": bin_pct_pos, "bin_ns": bin_ns,
        "rho_prox": rho, "p_prox": p_prox,
    }


def extract_scrna_data(scrna_path):
    """Extract scRNA-seq data for panels d/e."""
    print("Loading scRNA-seq data...")
    adata = sc.read_h5ad(scrna_path)
    fl_mask = adata.obs["disease"] == "follicular lymphoma"
    adata_fl = adata[fl_mask].copy()
    cell_types = adata_fl.obs["cell_type"].values
    print(f"  FL cells: {adata_fl.shape[0]:,}")

    # Panel (d): CD14 mRNA by cell type
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
            CT_SHORT_SCRNA.get(ct, ct), n,
            float(vals.mean()),
            100 * (vals > 0).mean(),
        ))
    cd14_by_ct.sort(key=lambda x: -x[2])

    # Panel (e): Marker heatmap
    focus_cts = [
        "follicular dendritic cell", "myeloid cell",
        "B cell", "malignant cell",
    ]
    myeloid_genes = ["CD14", "CD68", "VSIR", "ITGAM", "IDO1", "TLR5",
                     "C3AR1", "FCGR1A", "CD274"]
    fdc_genes = ["CR2", "FCER2", "CXCL13", "PDPN"]
    heatmap_genes = myeloid_genes + fdc_genes

    heatmap_data = {}
    for ct in focus_cts:
        mask = cell_types == ct
        ct_label = CT_SHORT_SCRNA.get(ct, ct)
        heatmap_data[ct_label] = {}
        for gene in heatmap_genes:
            ensid = KEY_GENES.get(gene)
            if ensid:
                vals = get_gene_expr(adata_fl, ensid)
                heatmap_data[ct_label][gene] = float(vals[mask].mean())

    return {
        "cd14_by_ct": cd14_by_ct,
        "heatmap_data": heatmap_data,
        "heatmap_genes": heatmap_genes,
        "myeloid_genes": myeloid_genes,
    }


def extract_fdc_survival(s_panel_path):
    """Extract FDC vs CD14+ FDC survival data for panel f."""
    print("Computing FDC survival metrics...")
    with h5py.File(s_panel_path, "r") as f:
        var_names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in f["var"]["_index"][:]]
        sids = load_array(f, "sample_id")
        ctypes = load_array(f, "cell_type")
        X = f["X"]
        cd14_idx = var_names.index("CD14")

        # Global CD14 Q75 on all FDCs
        fdc_indices = np.where(ctypes == "FDC")[0]
        print(f"  Total FDCs: {len(fdc_indices):,}")
        chunk = 10000
        all_cd14 = []
        for start in range(0, len(fdc_indices), chunk):
            batch = np.sort(fdc_indices[start:start + chunk])
            all_cd14.append(X[batch, cd14_idx])
        all_cd14 = np.concatenate(all_cd14)
        q75 = np.percentile(all_cd14, 75)
        print(f"  FDC CD14 Q75 threshold: {q75:.3f}")

        # Per-ROI metrics
        rois = sorted(set(sids))
        rois = [r for r in rois if is_tumor_core(r) and r not in EXCLUDE_ROIS
                and not r.startswith("Biomax")]
        rows = []
        for roi in rois:
            mask = sids == roi
            idx = np.where(mask)[0]
            n = len(idx)
            if n < 200:
                continue
            roi_ct = ctypes[mask]
            n_fdc = int((roi_ct == "FDC").sum())
            fdc_in_roi = np.where(roi_ct == "FDC")[0]
            if len(fdc_in_roi) > 0:
                global_fdc_idx = idx[fdc_in_roi]
                cd14_vals = X[global_fdc_idx, cd14_idx]
                n_cd14pos = int((cd14_vals >= q75).sum())
            else:
                n_cd14pos = 0
            rows.append({
                "sample_id": roi,
                "slide_ID": normalize_sample_id(roi),
                "fdc_frac": n_fdc / n,
                "cd14pos_fdc_frac": n_cd14pos / n,
            })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} tumor ROIs")

    # Merge clinical
    clin = pd.read_csv("data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    clin["slide_ID"] = clin["slide_ID"].apply(normalize_sample_id)
    clin = clin.drop_duplicates("slide_ID")
    merged = df.merge(clin, on="slide_ID", how="inner")
    treated = merged[merged["INITIAL OBSERVATION"] != "Yes"].copy()
    print(f"  {len(treated)} treated patients")

    # Cox regression
    results = []
    metrics = [("fdc_frac", "All FDCs (%)"), ("cd14pos_fdc_frac", "CD14+ FDCs (%)")]
    endpoints = [
        ("Progression free survival (y)", "CODE_PFS", "PFS"),
        ("Overall survival (y)", "CODE_OS", "OS"),
    ]
    for metric, label in metrics:
        for time_col, event_col, ep_label in endpoints:
            sub = treated[[metric, time_col, event_col]].dropna()
            if len(sub) < 20 or sub[event_col].sum() < 5:
                continue
            sub = sub.copy()
            mu, sd = sub[metric].mean(), sub[metric].std()
            if sd < 1e-12:
                continue
            sub[metric] = (sub[metric] - mu) / sd
            cph = CoxPHFitter()
            try:
                cph.fit(sub, duration_col=time_col, event_col=event_col)
                s = cph.summary.iloc[0]
                results.append({
                    "metric": metric, "label": label, "endpoint": ep_label,
                    "HR": s["exp(coef)"],
                    "lo": s["exp(coef) lower 95%"],
                    "hi": s["exp(coef) upper 95%"],
                    "p": s["p"], "n": len(sub),
                })
                sig = "***" if s["p"] < 0.001 else "**" if s["p"] < 0.01 else "*" if s["p"] < 0.05 else ""
                print(f"    {ep_label} {label:20s} HR={s['exp(coef)']:.3f} P={s['p']:.4f} {sig}")
            except Exception:
                pass

    return {"results": results, "q75": q75}


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(spillover, scrna, survival, output_dir="output/hypotheses_v8"):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.40,
                  left=0.08, right=0.94, top=0.94, bottom=0.06)

    # ── (a) Mean CD14 by cell type (IMC) ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    ct_means = spillover["ct_means"]
    names = [CT_SHORT_IMC.get(ct, ct) for ct, _, _ in ct_means]
    means = [m for _, _, m in ct_means]
    colors = []
    for ct, _, _ in ct_means:
        if ct in MYELOID_TYPES:
            colors.append(CAT_COLORS["Myeloid"])
        elif ct == "FDC":
            colors.append(CAT_COLORS["FDC"])
        else:
            colors.append(CAT_COLORS["Spillover"])
    y_pos = range(len(names))
    ax_a.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.5)
    ax_a.set_yticks(list(y_pos))
    ax_a.set_yticklabels(names, fontsize=TICK_SIZE)
    ax_a.invert_yaxis()
    ax_a.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_a.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax_a.set_xlabel("Mean CD14 intensity (scaled)")
    ax_a.set_title("CD14 expression by cell type (IMC)")
    legend_elements = [
        Patch(facecolor=CAT_COLORS["Myeloid"], label="Myeloid"),
        Patch(facecolor=CAT_COLORS["FDC"], label="FDC"),
        Patch(facecolor=CAT_COLORS["Spillover"], label="Other"),
    ]
    ax_a.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="lower right")

    # ── (b) Composition of CD14+ cells ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    sizes = [spillover["mye_pos"], spillover["fdc_pos"], spillover["other_pos"]]
    labels = [
        f"Myeloid\n{spillover['mye_pos']:,}\n({100*spillover['mye_pos']/spillover['n_pos']:.0f}%)",
        f"FDC\n{spillover['fdc_pos']:,}\n({100*spillover['fdc_pos']/spillover['n_pos']:.0f}%)",
        f"Spillover\n{spillover['other_pos']:,}\n({100*spillover['other_pos']/spillover['n_pos']:.0f}%)",
    ]
    pie_colors = [CAT_COLORS["Myeloid"], CAT_COLORS["FDC"], CAT_COLORS["Spillover"]]
    ax_b.pie(
        sizes, labels=labels, colors=pie_colors,
        startangle=90, textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax_b.set_title(f"CD14+ cells (>{0.5} threshold)\nn = {spillover['n_pos']:,}")

    # ── (c) Distance gradient ──
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c")
    x_pos = range(len(spillover["bin_labels"]))
    bars = ax_c.bar(x_pos, spillover["bin_pct_pos"], color="#E41A1C", alpha=0.7,
                    edgecolor="white", linewidth=0.5)
    ax_c.set_xticks(list(x_pos))
    ax_c.set_xticklabels(spillover["bin_labels"], fontsize=TICK_SIZE, rotation=30, ha="right")
    ax_c.set_xlabel("Distance to nearest myeloid cell (\u03bcm)")
    ax_c.set_ylabel("% CD14+ among non-myeloid/non-FDC cells")
    ax_c.set_title(
        f"CD14 spillover gradient\n"
        f"\u03c1 = {spillover['rho_prox']:.3f}, p < 10\u207b\u00b9\u2070"
    )
    for bar, n in zip(bars, spillover["bin_ns"]):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"n={n//1000}k", ha="center", va="bottom", fontsize=6, color="#666",
        )

    # ── (d) CD14 mRNA by cell type (scRNA-seq) ──
    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, "d")
    ct_data = scrna["cd14_by_ct"]
    names_d = [x[0] for x in ct_data]
    means_d = [x[2] for x in ct_data]
    colors_d = []
    for name, _, _, _ in ct_data:
        if name == "FDC":
            colors_d.append("#FF7F00")
        elif name == "Myeloid":
            colors_d.append("#E41A1C")
        else:
            colors_d.append("#CCCCCC")
    y_pos = range(len(names_d))
    ax_d.barh(y_pos, means_d, color=colors_d, edgecolor="white", linewidth=0.5)
    ax_d.set_yticks(list(y_pos))
    ax_d.set_yticklabels(names_d, fontsize=TICK_SIZE)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Mean CD14 mRNA expression")
    ax_d.set_title("CD14 expression by cell type\n(Han et al. 2022, scRNA-seq)")
    for i, (name, n, mean, pct) in enumerate(ct_data):
        if name in ("FDC", "Myeloid"):
            ax_d.text(
                mean + 0.01, i, f"{pct:.0f}% cells +",
                va="center", fontsize=7, color="#333",
            )

    # ── (e) Marker heatmap ──
    ax_e = fig.add_subplot(gs[1, 1])
    panel_label(ax_e, "e")
    hm = scrna["heatmap_data"]
    genes = scrna["heatmap_genes"]
    cts_order = ["FDC", "Myeloid", "B cell", "Malignant B"]
    mat = np.array([[hm[ct].get(g, 0) for ct in cts_order] for g in genes])
    mat_z = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        row = mat[i]
        if row.std() > 0:
            mat_z[i] = (row - row.mean()) / row.std()
    im = ax_e.imshow(mat_z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax_e.set_xticks(range(len(cts_order)))
    ax_e.set_xticklabels(cts_order, fontsize=TICK_SIZE, rotation=30, ha="right")
    ax_e.set_yticks(range(len(genes)))
    gene_labels = [GENE_LABELS.get(g, g) for g in genes]
    ax_e.set_yticklabels(gene_labels, fontsize=TICK_SIZE)
    n_mye = len(scrna["myeloid_genes"])
    ax_e.axhline(n_mye - 0.5, color="black", linewidth=1.5)
    ax_e.text(-0.6, n_mye / 2 - 0.5, "Myeloid\nmarkers", ha="right",
              va="center", fontsize=7, style="italic")
    ax_e.text(-0.6, n_mye + 1.5, "FDC\nmarkers", ha="right",
              va="center", fontsize=7, style="italic")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val > 0.05:
                color = "white" if abs(mat_z[i, j]) > 1.2 else "black"
                ax_e.text(j, i, f"{val:.2f}", ha="center", va="center",
                          fontsize=6, color=color)
    ax_e.set_title("Marker expression (z-scored)\nmyeloid vs FDC identity genes")
    plt.colorbar(im, ax=ax_e, shrink=0.6, label="z-score")

    # ── (f) FDC survival forest ──
    ax_f = fig.add_subplot(gs[1, 2])
    panel_label(ax_f, "f")
    results = survival["results"]

    # Order: PFS All FDC, PFS CD14+ FDC, OS All FDC, OS CD14+ FDC
    order = [
        ("PFS", "fdc_frac"),
        ("PFS", "cd14pos_fdc_frac"),
        ("OS", "fdc_frac"),
        ("OS", "cd14pos_fdc_frac"),
    ]
    y_positions = []
    row_labels = []
    hrs = []
    los = []
    his = []
    ps = []
    row_colors = []

    y = 0
    prev_ep = None
    for ep, metric in order:
        if prev_ep is not None and ep != prev_ep:
            y += 0.5  # gap between endpoints
        match = [r for r in results if r["endpoint"] == ep and r["metric"] == metric]
        if match:
            r = match[0]
            y_positions.append(y)
            row_labels.append(f"{ep}: {r['label']}")
            hrs.append(r["HR"])
            los.append(r["lo"])
            his.append(r["hi"])
            ps.append(r["p"])
            row_colors.append("#FF7F00" if "CD14+" in r["label"] else "#666666")
        y += 1
        prev_ep = ep

    y_positions = np.array(y_positions)
    hrs = np.array(hrs)
    los = np.array(los)
    his = np.array(his)

    for i in range(len(y_positions)):
        ax_f.plot([los[i], his[i]], [y_positions[i], y_positions[i]],
                  color=row_colors[i], linewidth=2, solid_capstyle="round")
        ax_f.plot(hrs[i], y_positions[i], "o", color=row_colors[i],
                  markersize=8, markeredgecolor="white", markeredgewidth=0.5)
        sig = "***" if ps[i] < 0.001 else "**" if ps[i] < 0.01 else "*" if ps[i] < 0.05 else ""
        ax_f.text(
            max(his[i] + 0.02, 1.85), y_positions[i],
            f"HR={hrs[i]:.2f} P={ps[i]:.4f} {sig}",
            va="center", fontsize=7, color=row_colors[i], fontweight="bold",
        )

    ax_f.set_yticks(y_positions)
    ax_f.set_yticklabels(row_labels, fontsize=TICK_SIZE)
    ax_f.invert_yaxis()
    ax_f.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_f.set_xlabel("Hazard Ratio (per SD)")
    ax_f.set_title(f"CD14+ FDCs vs all FDCs\n(Cox, treated patients, n={results[0]['n'] if results else '?'})")
    ax_f.set_xlim(0.8, 2.3)

    # Legend
    legend_f = [
        Patch(facecolor="#666666", label="All FDCs"),
        Patch(facecolor="#FF7F00", label="CD14+ FDCs"),
    ]
    ax_f.legend(handles=legend_f, fontsize=LEGEND_SIZE, loc="lower right")

    out = Path(output_dir) / "fig_cd14_fdc_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {out} + PDF")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-panel", required=True, help="S-panel h5ad")
    parser.add_argument("--scrna", required=True, help="scRNA-seq h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    spillover = extract_spillover_data(args.s_panel)
    scrna = extract_scrna_data(args.scrna)
    survival = extract_fdc_survival(args.s_panel)
    make_figure(spillover, scrna, survival, args.output_dir)


if __name__ == "__main__":
    main()
