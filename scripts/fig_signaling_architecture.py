#!/usr/bin/env python
"""Supplementary figure: Signaling Architecture in the FL Microenvironment.

Bridges cell type/compartment analyses to the ABM by showing which cells
produce which signals and where.

S13 panels (this figure):
  (a) scRNA: VISTA/IDO1 checkpoint expression across cell types
  (b) IMC: VISTA+/IDO+ fractions by compartment and cell type

Panels moved to S9 (fig_fdc_cd14_biology.py supplementary):
  plot_scrna_dotplot       — signaling molecules × cell types (source map)
  plot_fdc_survival_bars   — B cell survival signals by CD14+/- FDCs
  plot_cxcl13_cd21         — CXCL13-CD21 per-ROI concordance (H7a)
  plot_signaling_heatmap   — signaling marker protein expression by cell type
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS

from figure_style import (TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
                          ANNOT_SIZE, PANEL_LABEL_SIZE)


# ── Gene IDs for scRNA-seq (Han 2022, Ensembl-indexed) ──
SCRNA_GENES = {
    "CXCL13": "ENSG00000156234",
    "CXCL12": "ENSG00000107562",
    "IDO1":   "ENSG00000131203",
    "VSIR":   "ENSG00000107738",     # VISTA
    "TNFSF13B": "ENSG00000102524",   # BAFF
    "TNFSF13": "ENSG00000161955",    # APRIL
    "TGFB1":  "ENSG00000105329",
    "IL6":    "ENSG00000136244",
    "IL10":   "ENSG00000136634",
    "CCL21":  "ENSG00000137077",
    "CD274":  "ENSG00000120217",     # PD-L1
}
GENE_DISPLAY = {
    "CXCL13": "CXCL13", "CXCL12": "CXCL12", "IDO1": "IDO1",
    "VSIR": "VISTA", "TNFSF13B": "BAFF", "TNFSF13": "APRIL",
    "TGFB1": "TGF-β1", "IL6": "IL-6", "IL10": "IL-10",
    "CCL21": "CCL21", "CD274": "PD-L1",
}

# scRNA cell type short names
CT_SHORT = {
    "CD14+ FDC": "CD14+ FDC",
    "CD14- FDC": "CD14- FDC",
    "M2-like Mac": "M2 Mac",
    "M1-like Mac": "M1 Mac",
    "S100A9+ monocyte": "S100A9+",
    "DC-like": "DC",
    "B cell": "B cell",
    "malignant cell": "Malig. B",
    "T follicular helper cell": "Tfh",
    "regulatory T cell": "Treg",
    "exhausted T cell": "Exh. T",
    "effector CD8-positive, alpha-beta T cell": "CD8 eff.",
    "CD4-positive, alpha-beta cytotoxic T cell": "Cytotox. CD4",
    "plasma cell": "Plasma",
}
# Cell types to include in dot plot (in display order)
CT_ORDER = [
    "CD14+ FDC", "CD14- FDC", "M2-like Mac", "M1-like Mac",
    "S100A9+ monocyte", "DC-like", "malignant cell",
    "B cell", "T follicular helper cell", "regulatory T cell",
    "exhausted T cell", "effector CD8-positive, alpha-beta T cell",
]

# Myeloid subtyping: Leiden cluster → IMC-matching label
# CL0+CL4 = M2-like (APOE/C1Q/FOLR2), CL2+CL7 = M1-like (CXCL9/CD86/ITGAX),
# CL1 = S100A9+ monocyte, CL3+CL6 = DC-like (IDO1/ITGAX-high)
MYELOID_CLUSTER_MAP = {
    "0": "M2-like Mac", "4": "M2-like Mac",
    "2": "M1-like Mac", "7": "M1-like Mac",
    "1": "S100A9+ monocyte",
    "3": "DC-like", "6": "DC-like",
    "5": None,  # inflammatory monocyte — no clean IMC match, exclude
}

# IMC data helpers
FOLL_COMPARTMENTS = {
    "B cell zone (BCL2+)", "B cell zone (PAX5+)",
    "FDC network zone", "FDC / myeloid zone",
}
INTER_COMPARTMENTS = {
    "T cell zone", "Stromal / CAF zone",
    "Other / myeloid zone", "B/T mixed zone",
}
IMC_CELLTYPES = [
    "FDC", "M2 Macrophages", "M1 Macrophages", "Myeloid (S100A9+)",
    "B cells (BCL2+)", "CD8 T cells", "CD4 T cells", "Dendritic cells",
]
IMC_CT_SHORT = {
    "FDC": "FDC", "M2 Macrophages": "M2 Mac",
    "M1 Macrophages": "M1 Mac", "Myeloid (S100A9+)": "S100A9+",
    "B cells (BCL2+)": "Tumor B", "CD8 T cells": "CD8 T",
    "CD4 T cells": "CD4 T", "Dendritic cells": "DC",
}


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(
        x, y, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
        va="bottom", ha="left",
    )


def clean_axes(ax):
    """Remove top/right spines for clean look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def is_tumor_core(sample_id):
    s = sample_id.lower()
    if "_ton_" in s or "_adr_" in s:
        return False
    for tissue in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if tissue in s:
            return False
    if sample_id == "Biomax_ROI_006":
        return False
    return True


def get_gene_expr(adata_X, var_index, ensembl_id):
    """Get expression vector from matrix."""
    idx = list(var_index).index(ensembl_id)
    vals = adata_X[:, idx]
    if hasattr(vals, "toarray"):
        vals = vals.toarray().ravel()
    return np.array(vals).ravel()


def load_h5_array(f, key):
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


# ──────────────────────────────────────────────────────────────
# scRNA myeloid subtyping + FDC CD14 split
# ──────────────────────────────────────────────────────────────

def refine_celltypes(fl, seed=42):
    """Refine scRNA cell types: split FDCs by CD14, cluster myeloid cells.

    1. FDCs → CD14+ FDC / CD14- FDC (based on CD14 > 0)
    2. Myeloid cells → M2-like Mac / M1-like Mac / S100A9+ monocyte / DC-like
       via top-2000-variance → PCA → Leiden (res=0.5).

    Returns updated cell_type string array.
    """
    import scanpy as sc

    # Convert from Categorical to plain strings
    cell_types = np.array(fl.obs["cell_type"].astype(str).values)
    var_index = list(fl.var.index)

    # ── 1. Split FDCs by CD14 expression ──
    fdc_mask = cell_types == "follicular dendritic cell"
    n_fdc = fdc_mask.sum()
    if n_fdc > 0:
        cd14_ensid = "ENSG00000170458"  # CD14
        cd14_idx = var_index.index(cd14_ensid)
        cd14_vals = fl.X[fdc_mask, cd14_idx]
        if hasattr(cd14_vals, "toarray"):
            cd14_vals = cd14_vals.toarray().ravel()
        cd14_vals = np.array(cd14_vals).ravel()
        cd14_pos = cd14_vals > 0
        fdc_indices = np.where(fdc_mask)[0]
        for i, is_pos in zip(fdc_indices, cd14_pos):
            cell_types[i] = "CD14+ FDC" if is_pos else "CD14- FDC"
        print(f"  FDCs: {n_fdc} total → {cd14_pos.sum()} CD14+, {(~cd14_pos).sum()} CD14-")

    # ── 2. Cluster myeloid cells ──
    mye_mask = cell_types == "myeloid cell"
    n_mye = mye_mask.sum()
    if n_mye < 50:
        print(f"  Only {n_mye} myeloid cells, skipping subtyping")
        return cell_types

    print(f"  Subtyping {n_mye} myeloid cells...")
    mye = fl[mye_mask].copy()

    # Normalize + log
    sc.pp.normalize_total(mye, target_sum=1e4)
    sc.pp.log1p(mye)

    # Top-variance gene selection (avoids HVG module issues)
    if hasattr(mye.X, "toarray"):
        X_dense = mye.X.toarray()
    else:
        X_dense = np.array(mye.X)
    gene_var = X_dense.var(axis=0)
    top_idx = np.argsort(gene_var)[-2000:]
    mye_sub = mye[:, top_idx].copy()

    # Cluster
    sc.pp.scale(mye_sub, max_value=10)
    sc.tl.pca(mye_sub, n_comps=30, random_state=seed)
    sc.pp.neighbors(mye_sub, n_neighbors=15, n_pcs=20, random_state=seed)
    sc.tl.leiden(mye_sub, resolution=0.5, flavor="igraph",
                 n_iterations=2, directed=False, random_state=seed)

    # Map clusters to IMC-matching labels
    clusters = mye_sub.obs["leiden"].values
    subtypes = np.array([MYELOID_CLUSTER_MAP.get(c) for c in clusters])

    mye_indices = np.where(mye_mask)[0]
    for i, subtype in zip(mye_indices, subtypes):
        if subtype is not None:
            cell_types[i] = subtype

    # Report
    counts = Counter(subtypes)
    for label, n in sorted(counts.items(), key=lambda x: -(x[1] or 0)):
        print(f"    {label or 'unassigned':20s}: {n}")

    return cell_types


# ──────────────────────────────────────────────────────────────
# Data extraction (importable by other scripts)
# ──────────────────────────────────────────────────────────────

def extract_scrna_data(scrna_path):
    """Extract signaling molecule expression by cell type from scRNA-seq."""
    import scanpy as sc
    print("Loading scRNA-seq data...")
    adata = sc.read_h5ad(scrna_path)
    fl = adata[adata.obs["disease"] == "follicular lymphoma"].copy()
    cell_types = refine_celltypes(fl)
    var_index = fl.var.index

    # ── Dot plot data ──
    dotplot = {}
    for gene_name, ensid in SCRNA_GENES.items():
        if ensid not in var_index:
            print(f"  WARNING: {gene_name} ({ensid}) not in var")
            continue
        expr = get_gene_expr(fl.X, var_index, ensid)
        dotplot[gene_name] = {}
        for ct in CT_ORDER:
            mask = cell_types == ct
            n = mask.sum()
            if n < 10:
                continue
            vals = expr[mask]
            dotplot[gene_name][ct] = {
                "mean": float(vals.mean()),
                "pct": float(100 * (vals > 0).mean()),
                "n": int(n),
            }

    # ── FDC subset for B cell survival signals ──
    # refine_celltypes already split FDCs → "CD14+ FDC" / "CD14- FDC"
    fdc_mask = (cell_types == "CD14+ FDC") | (cell_types == "CD14- FDC")
    fdc_X = fl.X[fdc_mask]
    cd14_pos = np.array(cell_types[fdc_mask] == "CD14+ FDC")

    survival_genes = ["TNFSF13B", "TNFSF13", "TGFB1", "IL6"]
    fdc_survival = {}
    for gene_name in survival_genes:
        ensid = SCRNA_GENES[gene_name]
        if ensid not in var_index:
            continue
        vals = get_gene_expr(fdc_X, var_index, ensid)
        fdc_survival[gene_name] = {
            "cd14_pos_mean": float(vals[cd14_pos].mean()),
            "cd14_neg_mean": float(vals[~cd14_pos].mean()),
            "cd14_pos_pct": float(100 * (vals[cd14_pos] > 0).mean()),
            "cd14_neg_pct": float(100 * (vals[~cd14_pos] > 0).mean()),
            "cd14_pos_vals": vals[cd14_pos],
            "cd14_neg_vals": vals[~cd14_pos],
        }
        # Wilcoxon test
        if cd14_pos.sum() > 5 and (~cd14_pos).sum() > 5:
            _, p = stats.mannwhitneyu(vals[cd14_pos], vals[~cd14_pos],
                                       alternative="two-sided")
            fdc_survival[gene_name]["pval"] = float(p)

    n_fdc = int(fdc_mask.sum())
    n_pos = int(cd14_pos.sum())
    n_neg = int((~cd14_pos).sum())

    return {
        "dotplot": dotplot,
        "fdc_survival": fdc_survival,
        "n_fdc": n_fdc, "n_cd14pos": n_pos, "n_cd14neg": n_neg,
    }


def extract_imc_data(s_panel_path, s_utag_path):
    """Extract signaling marker data from IMC S-panel."""
    print("Loading IMC S-panel data...")
    f = h5py.File(s_panel_path, "r")
    X = f["X"][:]
    markers = [v.decode() for v in f["var"]["_index"][:]]
    cell_types = load_h5_array(f, "cell_type")
    sample_ids = load_h5_array(f, "sample_id")
    f.close()

    marker_idx = {m: i for i, m in enumerate(markers)}

    # Filter to tumor cores (exclude Biomax for clinical)
    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS and not s.startswith("Biomax")
        for s in sample_ids
    ])
    X = X[tumor_mask]
    cell_types = cell_types[tumor_mask]
    sample_ids = sample_ids[tumor_mask]
    print(f"  {len(X):,} tumor cells (non-Biomax)")

    # ── CXCL13-CD21 per-ROI concordance ──
    print("  Computing CXCL13-CD21 per-ROI concordance...")
    cxcl13_idx = marker_idx.get("CXCL13")
    cd21_idx = marker_idx.get("CD21")
    roi_cxcl13 = []
    roi_cd21 = []
    roi_names = []
    for roi in np.unique(sample_ids):
        rmask = sample_ids == roi
        n = rmask.sum()
        if n < 100:
            continue
        roi_cxcl13.append(float(X[rmask, cxcl13_idx].mean()))
        roi_cd21.append(float(X[rmask, cd21_idx].mean()))
        roi_names.append(roi)

    rho_h7a, p_h7a = stats.spearmanr(roi_cxcl13, roi_cd21)
    print(f"  H7a: CXCL13-CD21 rho={rho_h7a:.3f}, p={p_h7a:.1e}")

    # ── Signaling marker heatmap by cell type ──
    print("  Computing signaling marker profiles by cell type...")
    signal_markers = ["VISTA", "IDO", "CXCL13", "CXCL12", "CCL21",
                      "CD14", "HLA_DR", "HLA_Class_I", "CD11c", "S100A9"]
    signal_markers = [m for m in signal_markers if m in marker_idx]
    profiles = {}
    cd14_marker_idx = marker_idx.get("CD14")
    for ct in IMC_CELLTYPES:
        mask = cell_types == ct
        if mask.sum() < 50:
            continue
        short = IMC_CT_SHORT[ct]
        row = {}
        for m in signal_markers:
            row[m] = float(X[mask, marker_idx[m]].mean())
        profiles[short] = row
        # Split FDC by CD14 expression
        if ct == "FDC" and cd14_marker_idx is not None:
            cd14_vals = X[mask, cd14_marker_idx]
            cd14_hi = cd14_vals > 0.5
            for sublbl, submask in [("CD14+ FDC", cd14_hi),
                                     ("CD14- FDC", ~cd14_hi)]:
                ns = submask.sum()
                if ns < 50:
                    continue
                sub_X = X[mask][submask]
                sub_row = {}
                for m in signal_markers:
                    sub_row[m] = float(sub_X[:, marker_idx[m]].mean())
                profiles[sublbl] = sub_row
                print(f"    {sublbl}: n={ns:,}")

    # ── VISTA+/IDO+ fractions by compartment ──
    print("  Loading compartment data...")
    fu = h5py.File(s_utag_path, "r")
    comp_all = load_h5_array(fu, "compartment_name")
    ct_utag = load_h5_array(fu, "cell_type")
    sid_utag = load_h5_array(fu, "sample_id")
    X_utag = fu["X"][:]
    markers_utag = [v.decode() for v in fu["var"]["_index"][:]]
    fu.close()

    midx_utag = {m: i for i, m in enumerate(markers_utag)}
    tumor_u = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS and not s.startswith("Biomax")
        for s in sid_utag
    ])
    comp_all = comp_all[tumor_u]
    ct_utag = ct_utag[tumor_u]
    X_utag = X_utag[tumor_u]

    foll_mask = np.isin(comp_all, list(FOLL_COMPARTMENTS))
    inter_mask = np.isin(comp_all, list(INTER_COMPARTMENTS))

    vista_idx = midx_utag.get("VISTA")
    ido_idx = midx_utag.get("IDO")

    # VISTA+/IDO+ fraction by compartment (threshold: > 0.5 scaled)
    compartment_fracs = {}
    for label, cmask in [("Follicular", foll_mask), ("Interfollicular", inter_mask)]:
        n = cmask.sum()
        if n < 100:
            continue
        vista_pos = (X_utag[cmask, vista_idx] > 0.5).sum() if vista_idx is not None else 0
        ido_pos = (X_utag[cmask, ido_idx] > 0.5).sum() if ido_idx is not None else 0
        compartment_fracs[label] = {
            "vista_frac": float(100 * vista_pos / n),
            "ido_frac": float(100 * ido_pos / n),
            "n": int(n),
        }
        # Also per cell type within compartment
        cd14_idx = midx_utag.get("CD14")
        for ct in ["FDC", "M2 Macrophages", "Myeloid (S100A9+)", "CD8 T cells"]:
            ct_comp = cmask & (ct_utag == ct)
            nc = ct_comp.sum()
            if nc < 10:
                continue
            vp = (X_utag[ct_comp, vista_idx] > 0.5).sum() if vista_idx is not None else 0
            ip = (X_utag[ct_comp, ido_idx] > 0.5).sum() if ido_idx is not None else 0
            short = IMC_CT_SHORT.get(ct, ct)
            compartment_fracs[f"{short}_{label}"] = {
                "vista_frac": float(100 * vp / nc),
                "ido_frac": float(100 * ip / nc),
                "n": int(nc),
            }
            # Split FDC by CD14 expression
            if ct == "FDC" and cd14_idx is not None:
                cd14_vals = X_utag[ct_comp, cd14_idx]
                cd14_hi = cd14_vals > 0.5
                for sublbl, submask in [("CD14+ FDC", cd14_hi),
                                         ("CD14- FDC", ~cd14_hi)]:
                    ns = submask.sum()
                    if ns < 10:
                        continue
                    # Index back into ct_comp positions
                    sub_idx = np.where(ct_comp)[0][submask]
                    svp = (X_utag[sub_idx, vista_idx] > 0.5).sum() if vista_idx is not None else 0
                    sip = (X_utag[sub_idx, ido_idx] > 0.5).sum() if ido_idx is not None else 0
                    compartment_fracs[f"{sublbl}_{label}"] = {
                        "vista_frac": float(100 * svp / ns),
                        "ido_frac": float(100 * sip / ns),
                        "n": int(ns),
                    }

    # Per-ROI absolute VISTA+ cell counts by cell type (for S11f abundance view).
    # Sources of VISTA only: CD14+ FDC, CD14- FDC, M2 Mac, S100A9+.
    sid_utag_sub = sid_utag[tumor_u]
    vista_per_roi_foll = {ct: [] for ct in
                           ["CD14+ FDC", "CD14- FDC", "M2 Mac", "S100A9+"]}
    sid_foll = sid_utag_sub[foll_mask]
    ct_foll = ct_utag[foll_mask]
    cd14_foll = X_utag[foll_mask, midx_utag.get("CD14")] if midx_utag.get("CD14") is not None else None
    vista_foll = X_utag[foll_mask, vista_idx] if vista_idx is not None else None
    if vista_foll is not None:
        vpos_all = vista_foll > 0.5
        for roi in np.unique(sid_foll):
            rmask = sid_foll == roi
            if rmask.sum() < 100:
                continue
            vpos_roi = rmask & vpos_all
            if vpos_roi.sum() < 5:
                continue
            vp_ct = ct_foll[vpos_roi]
            cd14_vp = cd14_foll[vpos_roi] if cd14_foll is not None else None
            # Absolute counts per ROI
            vista_per_roi_foll["M2 Mac"].append(float((vp_ct == "M2 Macrophages").sum()))
            vista_per_roi_foll["S100A9+"].append(float((vp_ct == "Myeloid (S100A9+)").sum()))
            # FDC split by CD14
            fdc_mask = vp_ct == "FDC"
            if cd14_vp is not None:
                cd14hi = fdc_mask & (cd14_vp > 0.5)
                cd14lo = fdc_mask & (cd14_vp <= 0.5)
                vista_per_roi_foll["CD14+ FDC"].append(float(cd14hi.sum()))
                vista_per_roi_foll["CD14- FDC"].append(float(cd14lo.sum()))

    return {
        "roi_cxcl13": roi_cxcl13, "roi_cd21": roi_cd21,
        "rho_h7a": rho_h7a, "p_h7a": p_h7a,
        "profiles": profiles, "signal_markers": signal_markers,
        "compartment_fracs": compartment_fracs,
        "vista_per_roi_foll": vista_per_roi_foll,
    }


# ──────────────────────────────────────────────────────────────
# Standalone panel plot functions (importable by fig_fdc_cd14_biology.py)
# ──────────────────────────────────────────────────────────────

def plot_scrna_dotplot(ax, scrna_data):
    """scRNA dot plot: signaling molecules × cell types (source map)."""
    dotplot = scrna_data["dotplot"]
    gene_order = ["CXCL13", "CXCL12", "CCL21", "VSIR", "IDO1", "CD274",
                  "TNFSF13B", "TNFSF13", "TGFB1", "IL6", "IL10"]
    gene_order = [g for g in gene_order if g in dotplot]
    ct_labels = [CT_SHORT.get(ct, ct) for ct in CT_ORDER]

    mean_mat = np.zeros((len(gene_order), len(CT_ORDER)))
    pct_mat = np.zeros_like(mean_mat)
    for i, gene in enumerate(gene_order):
        for j, ct in enumerate(CT_ORDER):
            if ct in dotplot[gene]:
                mean_mat[i, j] = dotplot[gene][ct]["mean"]
                pct_mat[i, j] = dotplot[gene][ct]["pct"]

    vmax = max(0.5, mean_mat.max())
    sc = None
    for i in range(len(gene_order)):
        for j in range(len(CT_ORDER)):
            size = pct_mat[i, j] / 100 * 300
            if size < 2:
                continue
            color_val = mean_mat[i, j]
            sc = ax.scatter(j, i, s=size, c=[color_val], cmap="YlOrRd",
                            vmin=0, vmax=vmax,
                            edgecolors="black", linewidth=0.3, zorder=3)
    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.4, pad=0.15,
                            label="Mean expression")
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        cbar.set_label("Mean expression", fontsize=LABEL_SIZE)

    ax.set_xticks(range(len(ct_labels)))
    ax.set_xticklabels(ct_labels, fontsize=TICK_SIZE, rotation=45, ha="right")
    ax.set_yticks(range(len(gene_order)))
    ax.set_yticklabels([GENE_DISPLAY.get(g, g) for g in gene_order], fontsize=LABEL_SIZE)
    ax.set_xlim(-0.5, len(ct_labels) - 0.5)
    ax.set_ylim(-0.5, len(gene_order) - 0.5)
    ax.invert_yaxis()
    ax.set_title("Signaling molecule expression\n(scRNA-seq, Han 2022)",
                  fontsize=TITLE_SIZE, fontweight="medium", pad=12)
    ax.grid(True, alpha=0.15)
    clean_axes(ax)

    # Category backgrounds
    cat_colors = {
        "Chemokines": "#E8F0FE", "Checkpoints": "#FDE8E8",
        "Survival": "#E8F8E8", "Suppressive": "#F8F0E8",
    }
    cats = [("Chemokines", 0, 2), ("Checkpoints", 3, 5),
            ("Survival", 6, 7), ("Suppressive", 8, 10)]
    for cat_name, start, end in cats:
        if end >= len(gene_order):
            end = len(gene_order) - 1
        ax.axhspan(start - 0.45, end + 0.45,
                   color=cat_colors.get(cat_name, "#F5F5F5"), alpha=0.3, zorder=0)

    ax.text(0.02, -0.22, "Dot size = % expressing", fontsize=ANNOT_SIZE,
            transform=ax.transAxes, color="#777", fontstyle="italic")


def plot_fdc_survival_bars(ax, scrna_data):
    """B cell survival signals: CD14+ vs CD14- FDCs (scRNA)."""
    surv = scrna_data["fdc_survival"]
    genes_b = ["TNFSF13B", "TNFSF13", "TGFB1", "IL6"]
    genes_b = [g for g in genes_b if g in surv]
    x_pos = np.arange(len(genes_b))
    bar_w = 0.35

    means_pos = [surv[g]["cd14_pos_mean"] for g in genes_b]
    means_neg = [surv[g]["cd14_neg_mean"] for g in genes_b]

    COL_POS, COL_NEG = "#FF7F00", "#377EB8"
    ax.bar(x_pos - bar_w / 2, means_pos, bar_w, color=COL_POS, alpha=0.85,
           label=f"CD14+ FDC (n={scrna_data['n_cd14pos']})", edgecolor="white")
    ax.bar(x_pos + bar_w / 2, means_neg, bar_w, color=COL_NEG, alpha=0.85,
           label=f"CD14- FDC (n={scrna_data['n_cd14neg']})", edgecolor="white")

    for i, g in enumerate(genes_b):
        p = surv[g].get("pval", 1.0)
        if p < 0.001:
            lbl, color, fs = "***", "#C00", 20
        elif p < 0.01:
            lbl, color, fs = "**", "#C00", 20
        elif p < 0.05:
            lbl, color, fs = "*", "#C00", 20
        else:
            lbl, color, fs = "ns", "#888", 14
        y_max = max(means_pos[i], means_neg[i])
        ax.text(i, y_max + 0.015, lbl, ha="center", fontsize=fs,
                fontweight="bold", color=color)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([GENE_DISPLAY.get(g, g) for g in genes_b], fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean mRNA expression", fontsize=LABEL_SIZE)
    ax.set_ylim(0, max(max(means_pos), max(means_neg)) * 1.25)
    ax.set_title("B cell survival signals\nCD14+ vs CD14- FDCs (scRNA)",
                  fontsize=TITLE_SIZE, fontweight="medium")
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")
    ax.tick_params(labelsize=TICK_SIZE)
    clean_axes(ax)


def plot_checkpoint_celltypes(ax, scrna_data):
    """VISTA + PD-L1 expression across cell types (scRNA).

    Shows that VISTA dominates PD-L1 at the transcript level in myeloid cells,
    supporting the IMC finding that VISTA (not PD-L1) is the operative checkpoint.
    """
    dotplot = scrna_data["dotplot"]
    chk_genes = ["VSIR", "CD274"]
    ct_chk = CT_ORDER
    x_chk = np.arange(len(ct_chk))
    bar_w2 = 0.35

    for gi, gene in enumerate(chk_genes):
        if gene not in dotplot:
            continue
        means = [dotplot[gene].get(ct, {}).get("mean", 0) for ct in ct_chk]
        pcts = [dotplot[gene].get(ct, {}).get("pct", 0) for ct in ct_chk]
        offset = (gi - 0.5) * bar_w2
        color = "#984EA3" if gene == "VSIR" else "#4DAF4A"
        bars = ax.bar(x_chk + offset, means, bar_w2, color=color, alpha=0.8,
                      label=GENE_DISPLAY.get(gene, gene), edgecolor="white")
        for _, (bar, pct) in enumerate(zip(bars, pcts)):
            if pct > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=ANNOT_SIZE,
                        color="#555")

    ax.set_xticks(x_chk)
    ax.set_xticklabels([CT_SHORT.get(ct, ct) for ct in ct_chk],
                       fontsize=TICK_SIZE, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Mean mRNA expression", fontsize=LABEL_SIZE)
    ax.set_title("Immune checkpoint expression\nacross cell types (scRNA)",
                  fontsize=TITLE_SIZE, fontweight="medium")
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    clean_axes(ax)


def plot_cxcl13_cd21(ax, imc_data):
    """CXCL13-CD21 per-ROI concordance (H7a) — IMC S-panel."""
    ax.scatter(imc_data["roi_cxcl13"], imc_data["roi_cd21"],
               s=20, c="#333", alpha=0.5, edgecolors="none", rasterized=True)
    x_arr = np.array(imc_data["roi_cxcl13"])
    y_arr = np.array(imc_data["roi_cd21"])
    z = np.polyfit(x_arr, y_arr, 1)
    xl = np.linspace(x_arr.min(), x_arr.max(), 50)
    ax.plot(xl, np.polyval(z, xl), "r--", alpha=0.6, linewidth=1.5)

    rho = imc_data["rho_h7a"]
    p = imc_data["p_h7a"]
    ax.text(0.05, 0.92, f"Spearman ρ = {rho:.3f}\np = {p:.1e}",
            transform=ax.transAxes, fontsize=LABEL_SIZE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Mean CXCL13 expression (per ROI)", fontsize=TITLE_SIZE)
    ax.set_ylabel("Mean CD21 expression (per ROI)", fontsize=TITLE_SIZE)
    ax.set_title("CXCL13-CD21 spatial concordance (H7a)\nIMC S-panel",
                  fontsize=TITLE_SIZE, fontweight="medium")
    ax.tick_params(labelsize=TICK_SIZE)
    clean_axes(ax)


def plot_signaling_heatmap(ax, imc_data):
    """Signaling marker protein expression by cell type — IMC S-panel (z-scored)."""
    profiles = imc_data["profiles"]
    signal_markers = imc_data["signal_markers"]
    ct_order_e = [ct for ct in ["CD14+ FDC", "CD14- FDC", "M2 Mac", "M1 Mac",
                                "S100A9+", "Tumor B", "CD8 T", "CD4 T", "DC"]
                  if ct in profiles]
    mat = np.array([
        [profiles[ct].get(m, 0) for ct in ct_order_e]
        for m in signal_markers
    ])
    mat_z = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        row = mat[i]
        if row.std() > 0:
            mat_z[i] = (row - row.mean()) / row.std()

    im = ax.imshow(mat_z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(ct_order_e)))
    ax.set_xticklabels(ct_order_e, fontsize=TICK_SIZE, rotation=45, ha="right")
    ax.set_yticks(range(len(signal_markers)))
    marker_display = {"VISTA": "VISTA", "IDO": "IDO", "CXCL13": "CXCL13",
                      "CXCL12": "CXCL12", "CCL21": "CCL21", "CD14": "CD14",
                      "HLA_DR": "HLA-DR", "HLA_Class_I": "HLA-I",
                      "CD11c": "CD11c", "S100A9": "S100A9"}
    ax.set_yticklabels([marker_display.get(m, m) for m in signal_markers],
                       fontsize=LABEL_SIZE)
    ax.set_title("Signaling marker protein expression\nIMC S-panel (z-scored)",
                  fontsize=TITLE_SIZE, fontweight="medium")
    cbar_e = plt.colorbar(im, ax=ax, shrink=0.7, label="z-score")
    cbar_e.ax.tick_params(labelsize=TICK_SIZE)
    cbar_e.set_label("z-score", fontsize=LABEL_SIZE)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val > 0.02:
                color = "white" if abs(mat_z[i, j]) > 1.2 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=ANNOT_SIZE, color=color)


def plot_checkpoint_compartments(ax, imc_data):
    """VISTA+ fraction by compartment and cell type — IMC.

    FDC split into CD14+ and CD14- subgroups.
    """
    comp_fracs = imc_data["compartment_fracs"]
    cts_f = ["CD14+ FDC", "CD14- FDC", "M2 Mac", "S100A9+"]
    x_f = np.arange(len(cts_f))
    bar_w3 = 0.35

    groups = [
        ("Follicular", "vista_frac", "Follicular", "#984EA3", 0.85),
        ("Interfollicular", "vista_frac", "Interfollicular", "#984EA3", 0.45),
    ]

    for gi, (lbl, key, comp, color, alpha) in enumerate(groups):
        vals = []
        for ct in cts_f:
            k = f"{ct}_{comp}"
            vals.append(comp_fracs.get(k, {}).get(key, 0))
        offset = (gi - 0.5) * bar_w3
        hatch = "//" if "nter" in lbl else None
        ax.bar(x_f + offset, vals, bar_w3, color=color, alpha=alpha,
               label=lbl, edgecolor="white" if not hatch else color,
               hatch=hatch, linewidth=0.5)

    ax.set_xticks(x_f)
    ax.set_xticklabels(cts_f, fontsize=TICK_SIZE, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("% VISTA+ cells (>0.5)", fontsize=LABEL_SIZE)
    ax.set_title("VISTA+ fraction by\ncompartment and cell type (IMC)",
                  fontsize=TITLE_SIZE, fontweight="medium")
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")
    ax.tick_params(labelsize=TICK_SIZE)
    clean_axes(ax)


# ──────────────────────────────────────────────────────────────
# S13 figure: 2 panels (checkpoint celltypes + signaling heatmap)
# ──────────────────────────────────────────────────────────────

def make_figure(scrna_data, imc_data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 7))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.35,
                  left=0.07, right=0.97, top=0.90, bottom=0.12)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    plot_checkpoint_celltypes(ax_a, scrna_data)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    plot_checkpoint_compartments(ax_b, imc_data)

    out_path = output_dir / "fig_signaling_architecture.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight",
                pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Signaling architecture supplementary figure",
    )
    parser.add_argument("--scrna", default="data/external/steen2022_fl_scrna.h5ad",
                        help="Path to Han 2022 scRNA-seq h5ad")
    parser.add_argument("--s-panel", default="output/all_TMA_S_global_v8.h5ad",
                        help="IMC S-panel h5ad")
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad",
                        help="IMC S-panel UTAG compartment h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8",
                        help="Output directory")
    args = parser.parse_args()

    scrna_data = extract_scrna_data(args.scrna)
    imc_data = extract_imc_data(args.s_panel, args.s_utag)
    make_figure(scrna_data, imc_data, args.output_dir)


if __name__ == "__main__":
    main()
