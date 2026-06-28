#!/usr/bin/env python3
"""CD14+ FDC biology: consolidated main + supplementary figure.

Caching:
  --no-cache         Force re-extract data from h5ad (default: load pickle cache)
  --panels a,e,g     Re-render only these main-figure panels (rest from cache)
  --suppl-panels a,d Re-render only these supplementary panels

Storyline:
  1. CD14 signal decomposition implicates FDCs (in addition to myeloid)
  2. Characterize CD14+ FDCs (phenotype, localization)
  3. Understand what they do to help the tumor (intrafollicular biology)

Main figure (6 panels):
  (a) CD14 by cell type — FDC second highest
  (b) Compartment localization (follicular enrichment)
  (c) Compartment-split survival: follicular FDC CD14 predicts PFS/OS, interfollicular does not
  (d) Intrafollicular marker profile
  (e) Intrafollicular neighbor composition
  (f) Raw IMC composite inset: cell scatter + CD21/CD14/CD68/CD8 channels

Supplementary figure (9 panels):
  (a) scRNA-seq validation
  (b) CD14+ cell decomposition + spillover gradient (moved from main)
  (c) UMI counts per cell: CD14+ vs CD14- FDCs (transcriptional activity)
  (d) Ki-67 proliferation: B cells close vs distant to CD14+ FDCs
  (e) HLA expression: B cells close vs distant to CD14+ FDCs
  (f) scRNA signaling dotplot: signaling molecules × cell types (from S13)
  (g) B cell survival signals: CD14+ vs CD14- FDCs (from S13)
  (h) CXCL13-CD21 per-ROI concordance, H7a (from S13)
  (i) IMC signaling marker protein heatmap (from S13)
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.image import imread

# Import extract functions from existing scripts
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from fig_cd14_fdc_combined import extract_spillover_data, extract_scrna_data
from fig_fdc_functional import extract_data as extract_functional_data
from fig_fdc_intrafollicular import extract_s, extract_t, analyze_all
from fig_signaling_architecture import (
    extract_scrna_data as extract_signaling_scrna,
    extract_imc_data as extract_signaling_imc,
    plot_scrna_dotplot, plot_fdc_survival_bars,
    plot_cxcl13_cd21, plot_signaling_heatmap,
)

from src.visualization import (find_raw_file, build_raw_composite,
                                plot_scatter_composite_inset)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

from figure_style import (TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
                          ANNOT_SIZE, PANEL_LABEL_SIZE, apply_style, save_figure)
apply_style()

MYELOID_TYPES = {
    "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Myeloid (S100A9+)", "Dendritic cells",
}

CAT_COLORS = {"Myeloid": "#E41A1C", "FDC": "#FF7F00", "Spillover": "#999999"}

CT_SHORT_IMC = {
    "M1 Macrophages": "M1 Mac", "M2 Macrophages": "M2 Mac",
    "Macrophages": "Mac (generic)", "Myeloid (S100A9+)": "S100A9+ Myeloid",
    "Dendritic cells": "DC", "B cells (BCL2+)": "BCL2+ B",
    "B cells (PAX5+)": "PAX5+ B", "B cells": "B cells",
    "CD4 T cells": "CD4 T", "CD8 T cells": "CD8 T",
    "Endothelial": "Endothelial", "Mixed / Border cells": "Mixed/Border",
    "Stromal / CAF": "Stromal/CAF", "FRC (PDPN+)": "FRC",
    "Histiocytes (CD44hi)": "Histiocytes", "Other": "Other",
    "Low quality / Unassigned": "Unassigned", "pDC": "pDC", "FDC": "FDC",
}

FUNCTIONAL_MARKERS = {
    "Antigen\npresentation": ["HLA_DR", "HLA_Class_I", "CD11c"],
    "Immune\nsuppression":   ["VISTA", "IDO"],
    "FDC\nnetwork":          ["CD21", "PDPN"],
    "Chemokines":            ["CXCL13", "CXCL12", "CCL21"],
    "Myeloid":               ["CD68", "CD11b", "S100A9"],
    "Tumor /\nproliferation": ["BCL_2", "Ki-67", "PAX5"],
    "Stromal":               ["Vimentin", "CD146", "Fibronectin"],
}

MARKER_DISPLAY = {
    "HLA_Class_I": "HLA-I", "HLA_DR": "HLA-DR", "BCL_2": "BCL-2",
    "PD_L1": "PD-L1", "BCL_6": "BCL-6", "Ki-67": "Ki67",
    "CD11b": "CD11b", "S100A9": "S100A9", "p_H3s28": "pH3S28",
}

CAT_COLOR_MAP_INTRAFOLL = {
    "Antigen\npresentation": "#4CAF50", "Immune\nsuppression": "#D32F2F",
    "FDC\nnetwork": "#FF9800", "Chemokines": "#9C27B0",
    "Myeloid": "#795548", "Tumor /\nproliferation": "#607D8B",
    "Stromal": "#00BCD4",
}


# Raw IMC data for composite panel
RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "TMA_B1_S"
RAW_CHANNELS = {
    "CD21": "CD21(Er170Di)",
    "CD14": "CD14(Nd148Di)",
    "CD68": "CD68(Tb159Di)",
    "CD8":  "CD8a(Dy162Di)",
}
COMPOSITE_RGB = {
    "CD21": np.array([0, 1, 0]),       # green
    "CD14": np.array([1, 0, 0]),       # red (green+red → yellow where CD21+CD14 overlap)
    "CD68": np.array([1, 0, 1]),       # magenta
    "CD8":  np.array([0, 0.8, 1]),     # cyan
}
# Default ROI and window for composite panel
COMPOSITE_ROI = "B1_FL8"
COMPOSITE_CENTER = (1115, 927)
COMPOSITE_WINDOW = 250


# Compartment definitions for survival analysis (strict: matches notebook 2026-02-21)
S_FOLL_STRICT = ["B cell zone (BCL2+)", "B cell zone (PAX5+)", "FDC network zone"]
S_IFOLL_STRICT = ["FDC/myeloid zone", "T cell zone", "Stromal/CAF zone",
                   "FDC / myeloid zone", "Stromal / CAF zone"]
MIN_FDC_PER_COMP = 5


def extract_compartment_survival(s_utag_path):
    """Compute per-ROI follicular vs interfollicular FDC CD14, run Cox."""
    import h5py
    import pandas as pd
    from lifelines import CoxPHFitter
    from fig_fdc_intrafollicular import load_array, is_tumor_core, EXCLUDE_ROIS

    with h5py.File(s_utag_path, "r") as f:
        X = f["X"][:]
        markers = [v.decode() if isinstance(v, bytes) else str(v)
                   for v in f["var"]["_index"][:]]
        ct = load_array(f, "cell_type")
        sid = load_array(f, "sample_id")
        comp = load_array(f, "compartment_name")

    cd14_idx = markers.index("CD14")
    cd14 = X[:, cd14_idx]

    # Filter to tumor cores
    mask = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS
                     and not s.startswith("Biomax") for s in sid])
    ct, sid, comp, cd14 = ct[mask], sid[mask], comp[mask], cd14[mask]

    # FDC CD14 Q75 threshold (global)
    fdc_mask = ct == "FDC"
    q75 = float(np.percentile(cd14[fdc_mask], 75))
    print(f"  FDC CD14 Q75 = {q75:.3f} (n={fdc_mask.sum():,} FDCs)")

    # Per-ROI metrics
    rois = sorted(set(sid))
    rows = []
    for roi in rois:
        rmask = sid == roi
        n_total = rmask.sum()
        if n_total < 200:
            continue
        roi_fdc = fdc_mask[rmask]
        roi_comp = comp[rmask]
        roi_cd14 = cd14[rmask]

        fdc_in_foll = roi_fdc & np.isin(roi_comp, S_FOLL_STRICT)
        fdc_in_ifoll = roi_fdc & np.isin(roi_comp, S_IFOLL_STRICT)

        row = {"sample_id": roi}
        # Follicular FDC CD14 mean
        if fdc_in_foll.sum() >= MIN_FDC_PER_COMP:
            row["fdc_cd14_foll"] = float(roi_cd14[fdc_in_foll].mean())
        else:
            row["fdc_cd14_foll"] = np.nan
        # Interfollicular FDC CD14 mean
        if fdc_in_ifoll.sum() >= MIN_FDC_PER_COMP:
            row["fdc_cd14_ifoll"] = float(roi_cd14[fdc_in_ifoll].mean())
        else:
            row["fdc_cd14_ifoll"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # Normalize sample_id for clinical merge
    from src.clinical_linkage import normalize_sample_id, load_clinical
    df["slide_ID"] = df["sample_id"].apply(normalize_sample_id)

    clin = load_clinical()
    clin_t1 = clin.sort_values("T").drop_duplicates(subset="slide_ID", keep="first")
    merged = df.merge(clin_t1, on="slide_ID", how="inner")
    # Dedup by patient
    merged = merged.sort_values("T").drop_duplicates(subset="Patient_ID", keep="first")
    # Treated only
    treated = merged[merged["INITIAL OBSERVATION"] != "Yes"].copy()
    treated = treated.rename(columns={
        "Progression free survival (y)": "pfs_time",
        "CODE_PFS": "pfs_event",
        "Overall survival (y)": "os_time",
        "CODE_OS": "os_event",
    })
    for col in ["pfs_time", "pfs_event", "os_time", "os_event"]:
        treated[col] = pd.to_numeric(treated[col], errors="coerce")

    print(f"  {len(treated)} treated patients for Cox regression")

    # Cox regression per metric per endpoint
    results = []
    metrics = [
        ("fdc_cd14_foll", "Follicular FDC CD14"),
        ("fdc_cd14_ifoll", "Interfollicular FDC CD14"),
    ]
    endpoints = [
        ("pfs_time", "pfs_event", "PFS"),
        ("os_time", "os_event", "OS"),
    ]
    for metric, label in metrics:
        for tc, ec, ep in endpoints:
            sub = treated[[metric, tc, ec]].dropna()
            if len(sub) < 20 or sub[ec].sum() < 5:
                print(f"    {ep} {label}: skipped (n={len(sub)}, events={sub[ec].sum():.0f})")
                continue
            sub = sub.copy()
            mu, sd = sub[metric].mean(), sub[metric].std()
            if sd < 1e-12:
                continue
            sub[metric] = (sub[metric] - mu) / sd
            cph = CoxPHFitter()
            try:
                cph.fit(sub, duration_col=tc, event_col=ec)
                s = cph.summary.iloc[0]
                results.append({
                    "metric": metric, "label": label, "endpoint": ep,
                    "HR": s["exp(coef)"], "lo": s["exp(coef) lower 95%"],
                    "hi": s["exp(coef) upper 95%"], "p": s["p"],
                    "n": len(sub),
                })
                sig = "***" if s["p"] < 0.001 else "**" if s["p"] < 0.01 else "*" if s["p"] < 0.05 else ""
                print(f"    {ep} {label:30s} HR={s['exp(coef)']:.3f} "
                      f"[{s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}] "
                      f"P={s['p']:.4f} {sig}  n={len(sub)}")
            except Exception as e:
                print(f"    {ep} {label}: Cox failed — {e}")

    return {"results": results, "q75": q75}


B_CELL_TYPES = {"B cells (BCL2+)", "B cells (PAX5+)", "B cells"}
PROX_THRESHOLD_PX = 30  # pixels (~30 µm at 1 µm/px)
SCRNA_FDC_LABEL = "follicular dendritic cell"


def extract_proliferation_data(s_utag_path):
    """Ki-67 on B cells near CD14-high vs CD14-low FDCs (within 30px)."""
    import h5py
    from scipy.spatial import KDTree
    from fig_fdc_intrafollicular import load_array, is_tumor_core, EXCLUDE_ROIS

    with h5py.File(s_utag_path, "r") as f:
        X = f["X"][:]
        markers = [v.decode() if isinstance(v, bytes) else str(v)
                   for v in f["var"]["_index"][:]]
        ct = load_array(f, "cell_type")
        sid = load_array(f, "sample_id")
        comp = load_array(f, "compartment_name")
        cx = f["obs"]["centroid_x"][:]
        cy = f["obs"]["centroid_y"][:]

    cd14_idx = markers.index("CD14")
    ki67_idx = markers.index("Ki-67")
    cd14 = X[:, cd14_idx]
    ki67 = X[:, ki67_idx]

    # Filter to tumor cores (exclude Biomax — Ki-67 dead there)
    mask = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS
                     and not s.startswith("Biomax") for s in sid])
    ct, sid, comp, cd14, ki67, cx, cy = (
        ct[mask], sid[mask], comp[mask], cd14[mask], ki67[mask], cx[mask], cy[mask])

    fdc_mask = ct == "FDC"
    b_mask = np.isin(ct, list(B_CELL_TYPES))

    # FDC CD14 Q75 threshold
    q75 = float(np.percentile(cd14[fdc_mask], 75))
    fdc_hi = fdc_mask & (cd14 >= q75)
    fdc_lo = fdc_mask & (cd14 < np.percentile(cd14[fdc_mask], 25))

    print(f"  FDC CD14 Q75={q75:.3f}, n_hi={fdc_hi.sum():,}, n_lo={fdc_lo.sum():,}")
    print(f"  B cells: {b_mask.sum():,}")

    # Restrict to follicular compartments
    foll_mask = np.isin(comp, S_FOLL_STRICT)
    fdc_hi_foll = fdc_hi & foll_mask
    fdc_lo_foll = fdc_lo & foll_mask
    b_foll = b_mask & foll_mask

    # Per-ROI: find B cells within 30px of CD14-high vs CD14-low FDCs
    rois = sorted(set(sid))
    ki67_near_hi, ki67_near_lo = [], []
    for roi in rois:
        rm = sid == roi
        b_idx = np.where(rm & b_foll)[0]
        hi_idx = np.where(rm & fdc_hi_foll)[0]
        lo_idx = np.where(rm & fdc_lo_foll)[0]
        if len(b_idx) < 20 or (len(hi_idx) < 3 and len(lo_idx) < 3):
            continue
        b_coords = np.column_stack([cx[b_idx], cy[b_idx]])
        if len(hi_idx) >= 3:
            tree_hi = KDTree(np.column_stack([cx[hi_idx], cy[hi_idx]]))
            d_hi, _ = tree_hi.query(b_coords)
            near = d_hi <= PROX_THRESHOLD_PX
            ki67_near_hi.extend(ki67[b_idx[near]].tolist())
        if len(lo_idx) >= 3:
            tree_lo = KDTree(np.column_stack([cx[lo_idx], cy[lo_idx]]))
            d_lo, _ = tree_lo.query(b_coords)
            near = d_lo <= PROX_THRESHOLD_PX
            ki67_near_lo.extend(ki67[b_idx[near]].tolist())

    ki67_near_hi = np.array(ki67_near_hi)
    ki67_near_lo = np.array(ki67_near_lo)
    from scipy.stats import mannwhitneyu
    _, p = mannwhitneyu(ki67_near_hi, ki67_near_lo, alternative="two-sided")
    print(f"  B cells near CD14-high FDCs: Ki-67 mean={ki67_near_hi.mean():.3f} (n={len(ki67_near_hi):,})")
    print(f"  B cells near CD14-low FDCs:  Ki-67 mean={ki67_near_lo.mean():.3f} (n={len(ki67_near_lo):,})")
    print(f"  Mann-Whitney P={p:.2e}")

    return {
        "ki67_hi": ki67_near_hi, "ki67_lo": ki67_near_lo,
        "mean_hi": float(ki67_near_hi.mean()), "mean_lo": float(ki67_near_lo.mean()),
        "n_hi": len(ki67_near_hi), "n_lo": len(ki67_near_lo),
        "p": float(p),
    }


def extract_proximity_markers(s_utag_path):
    """Ki-67 and HLA on B cells close vs distant to CD14+ FDCs (within 30px).

    'Close' = within PROX_THRESHOLD_PX of any CD14+ FDC.
    'Distant' = farther than PROX_THRESHOLD_PX from any CD14+ FDC.
    """
    import h5py
    from scipy.spatial import KDTree
    from fig_fdc_intrafollicular import load_array, is_tumor_core, EXCLUDE_ROIS

    with h5py.File(s_utag_path, "r") as f:
        X = f["X"][:]
        markers = [v.decode() if isinstance(v, bytes) else str(v)
                   for v in f["var"]["_index"][:]]
        ct = load_array(f, "cell_type")
        sid = load_array(f, "sample_id")
        comp = load_array(f, "compartment_name")
        cx = f["obs"]["centroid_x"][:]
        cy = f["obs"]["centroid_y"][:]

    cd14_idx = markers.index("CD14")
    ki67_idx = markers.index("Ki-67")
    hlai_idx = markers.index("HLA_Class_I")
    hladr_idx = markers.index("HLA_DR")
    cd14 = X[:, cd14_idx]
    ki67 = X[:, ki67_idx]
    hlai = X[:, hlai_idx]
    hladr = X[:, hladr_idx]

    # Filter to tumor cores (exclude Biomax)
    mask = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS
                     and not s.startswith("Biomax") for s in sid])
    ct, sid, comp = ct[mask], sid[mask], comp[mask]
    cd14, ki67, hlai, hladr = cd14[mask], ki67[mask], hlai[mask], hladr[mask]
    cx, cy = cx[mask], cy[mask]

    fdc_mask = ct == "FDC"
    b_mask = np.isin(ct, list(B_CELL_TYPES))

    # CD14+ FDC = top quartile
    q75 = float(np.percentile(cd14[fdc_mask], 75))
    fdc_hi = fdc_mask & (cd14 >= q75)

    # Restrict to follicular compartments
    foll_mask = np.isin(comp, S_FOLL_STRICT)
    fdc_hi_foll = fdc_hi & foll_mask
    b_foll = b_mask & foll_mask

    # Per-ROI: B cells close vs distant to CD14+ FDCs
    rois = sorted(set(sid))
    ki67_close, ki67_dist = [], []
    hlai_close, hlai_dist = [], []
    hladr_close, hladr_dist = [], []
    for roi in rois:
        rm = sid == roi
        b_idx = np.where(rm & b_foll)[0]
        hi_idx = np.where(rm & fdc_hi_foll)[0]
        if len(b_idx) < 20 or len(hi_idx) < 3:
            continue
        b_coords = np.column_stack([cx[b_idx], cy[b_idx]])
        tree_hi = KDTree(np.column_stack([cx[hi_idx], cy[hi_idx]]))
        d_hi, _ = tree_hi.query(b_coords)
        close = d_hi <= PROX_THRESHOLD_PX
        dist = d_hi > PROX_THRESHOLD_PX
        ki67_close.extend(ki67[b_idx[close]].tolist())
        ki67_dist.extend(ki67[b_idx[dist]].tolist())
        hlai_close.extend(hlai[b_idx[close]].tolist())
        hlai_dist.extend(hlai[b_idx[dist]].tolist())
        hladr_close.extend(hladr[b_idx[close]].tolist())
        hladr_dist.extend(hladr[b_idx[dist]].tolist())

    from scipy.stats import mannwhitneyu
    ki67_close, ki67_dist = np.array(ki67_close), np.array(ki67_dist)
    hlai_close, hlai_dist = np.array(hlai_close), np.array(hlai_dist)
    hladr_close, hladr_dist = np.array(hladr_close), np.array(hladr_dist)

    _, p_ki67 = mannwhitneyu(ki67_close, ki67_dist, alternative="two-sided")
    _, p_hlai = mannwhitneyu(hlai_close, hlai_dist, alternative="two-sided")
    _, p_hladr = mannwhitneyu(hladr_close, hladr_dist, alternative="two-sided")

    print(f"  B cells close to CD14+ FDCs: n={len(ki67_close):,}")
    print(f"  B cells distant: n={len(ki67_dist):,}")
    print(f"  Ki-67: close={ki67_close.mean():.3f} vs dist={ki67_dist.mean():.3f}, P={p_ki67:.2e}")
    print(f"  HLA-I: close={hlai_close.mean():.3f} vs dist={hlai_dist.mean():.3f}, P={p_hlai:.2e}")
    print(f"  HLA-DR: close={hladr_close.mean():.3f} vs dist={hladr_dist.mean():.3f}, P={p_hladr:.2e}")

    return {
        "ki67_close": ki67_close, "ki67_dist": ki67_dist,
        "hlai_close": hlai_close, "hlai_dist": hlai_dist,
        "hladr_close": hladr_close, "hladr_dist": hladr_dist,
        "n_close": len(ki67_close), "n_dist": len(ki67_dist),
        "p_ki67": float(p_ki67), "p_hlai": float(p_hlai), "p_hladr": float(p_hladr),
        "mean_ki67_close": float(ki67_close.mean()),
        "mean_ki67_dist": float(ki67_dist.mean()),
        "mean_hlai_close": float(hlai_close.mean()),
        "mean_hlai_dist": float(hlai_dist.mean()),
        "mean_hladr_close": float(hladr_close.mean()),
        "mean_hladr_dist": float(hladr_dist.mean()),
        "sd_ki67_close": float(ki67_close.std()),
        "sd_ki67_dist": float(ki67_dist.std()),
        "sd_hlai_close": float(hlai_close.std()),
        "sd_hlai_dist": float(hlai_dist.std()),
        "sd_hladr_close": float(hladr_close.std()),
        "sd_hladr_dist": float(hladr_dist.std()),
    }


def extract_fdc_umi(scrna_path):
    """UMI counts per cell for CD14+ vs CD14- FDCs from scRNA-seq."""
    import scanpy as sc
    adata = sc.read_h5ad(scrna_path)
    fl = adata[adata.obs["disease"] == "follicular lymphoma"]
    fdc = fl[fl.obs["cell_type"] == SCRNA_FDC_LABEL].copy()
    print(f"  scRNA FDCs: {fdc.shape[0]}")

    # Total UMI per cell (from raw counts if available)
    if "n_counts" in fdc.obs.columns:
        umi = fdc.obs["n_counts"].values
    elif "total_counts" in fdc.obs.columns:
        umi = fdc.obs["total_counts"].values
    else:
        # Compute from X (assumes raw or log-normalized)
        import scipy.sparse as sp
        X = fdc.X
        if sp.issparse(X):
            umi = np.array(X.sum(axis=1)).flatten()
        else:
            umi = X.sum(axis=1)
    umi = np.asarray(umi, dtype=float)

    # CD14 expression
    from fig_cd14_fdc_combined import get_gene_expr, KEY_GENES
    cd14 = get_gene_expr(fdc, KEY_GENES["CD14"])
    cd14_pos = cd14 > 0

    umi_pos = umi[cd14_pos]
    umi_neg = umi[~cd14_pos]
    ratio = float(np.median(umi_pos) / max(np.median(umi_neg), 1))
    from scipy.stats import mannwhitneyu
    _, p = mannwhitneyu(umi_pos, umi_neg, alternative="two-sided")
    print(f"  CD14+ FDC UMI: median={np.median(umi_pos):.0f} (n={len(umi_pos)})")
    print(f"  CD14- FDC UMI: median={np.median(umi_neg):.0f} (n={len(umi_neg)})")
    print(f"  Ratio: {ratio:.1f}x, P={p:.2e}")

    return {
        "umi_pos": umi_pos, "umi_neg": umi_neg,
        "med_pos": float(np.median(umi_pos)), "med_neg": float(np.median(umi_neg)),
        "n_pos": len(umi_pos), "n_neg": len(umi_neg),
        "ratio": ratio, "p": float(p),
    }


def panel_label(ax, letter, fontsize=PANEL_LABEL_SIZE):
    ax.text(-0.02, 1.02, f"$\\bf{{{letter}}}$", transform=ax.transAxes,
            fontsize=fontsize, va="bottom")


def _render_panel(panel_id, plot_fn, plot_args, figsize, cache_dir, force=False):
    """Render a single panel as standalone PNG. Returns path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{panel_id}.png"
    if path.exists() and not force:
        print(f"    [{panel_id}] cached")
        return path
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax, *plot_args)
    panel_label(ax, panel_id.split("_")[-1])  # e.g. "main_a" → "a"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"    [{panel_id}] rendered → {path.name}")
    return path


def _paste_panel(ax, png_path):
    """Load a cached panel PNG into a matplotlib axes."""
    img = imread(str(png_path))
    ax.imshow(img)
    ax.axis("off")


def _render_panel_fig(panel_id, plot_fn, plot_args, figsize, cache_dir, force=False):
    """Render a figure-level panel (plot_fn receives fig, not ax). Returns path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{panel_id}.png"
    if path.exists() and not force:
        print(f"    [{panel_id}] cached")
        return path
    fig = plt.figure(figsize=figsize)
    plot_fn(fig, *plot_args)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"    [{panel_id}] rendered → {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Main figure
# ═══════════════════════════════════════════════════════════════════════════

def make_main_figure(data, output_dir, cache_dir, panels_to_refresh=None):
    """6-panel main figure: discovery → characterization → tumor support.

    Args:
        data: dict with keys 'spillover', 'comp_surv', 'intrafoll', 'composite'
        output_dir: where to save final composite
        cache_dir: where to cache individual panels
        panels_to_refresh: set of panel letters to force-regenerate, or None for all
    """
    # Standard single-axes panels
    panels = {
        "a": (_plot_cd14_by_celltype, ["spillover"], (9, 7)),
        "b": (_plot_localization, ["intrafoll"], (9, 7)),
        "c": (_plot_compartment_survival, ["comp_surv"], (9, 7)),
        "d": (_plot_intrafoll_markers, ["intrafoll"], (9, 7)),
        "e": (_plot_intrafoll_neighbors, ["intrafoll"], (9, 7)),
    }

    refresh_all = panels_to_refresh is None

    # Render standard panels (or load from cache)
    panel_paths = {}
    for letter, (plot_fn, data_keys, figsize) in panels.items():
        force = refresh_all or letter in (panels_to_refresh or set())
        plot_args = [data[k] for k in data_keys]
        panel_paths[letter] = _render_panel(
            f"main_{letter}", plot_fn, plot_args, figsize, cache_dir, force=force)

    # Panel (f) is figure-level (two sub-axes: scatter + composite)
    force_f = refresh_all or "f" in (panels_to_refresh or set())
    panel_paths["f"] = _render_panel_fig(
        "main_f", _plot_composite_inset, [data["composite"]],
        (14, 7), cache_dir, force=force_f)

    # Compose: rows 1-2 are 2-col, row 3 has e (narrow) + f (wide)
    fig = plt.figure(figsize=(20, 24))
    gs_top = gridspec.GridSpec(2, 2, figure=fig, hspace=0.02, wspace=0.02,
                               left=0.005, right=0.995, top=0.995, bottom=0.34)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig, wspace=0.02,
                               width_ratios=[1, 1.4],
                               left=0.005, right=0.995, top=0.32, bottom=0.005)

    for letter, gs_spec in [("a", gs_top[0, 0]), ("b", gs_top[0, 1]),
                             ("c", gs_top[1, 0]), ("d", gs_top[1, 1])]:
        ax = fig.add_subplot(gs_spec)
        _paste_panel(ax, panel_paths[letter])

    ax_e = fig.add_subplot(gs_bot[0, 0])
    _paste_panel(ax_e, panel_paths["e"])
    ax_f = fig.add_subplot(gs_bot[0, 1])
    _paste_panel(ax_f, panel_paths["f"])

    out = Path(output_dir) / "fig_fdc_cd14_main.png"
    save_figure(fig, out)
    return out


def _plot_cd14_by_celltype(ax, spillover):
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
    ax.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Mean CD14 intensity (arcsinh)", fontsize=LABEL_SIZE)
    ax.set_title("CD14 expression by cell type (IMC)", fontsize=TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    legend_elements = [
        Patch(facecolor=CAT_COLORS["Myeloid"], label="Myeloid"),
        Patch(facecolor=CAT_COLORS["FDC"], label="FDC"),
        Patch(facecolor=CAT_COLORS["Spillover"], label="Other"),
    ]
    ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="lower right")


def _plot_decomposition_spillover(ax, spillover):
    """Combined panel: pie on left, spillover gradient on right via inset."""
    # Spillover gradient as main plot
    x_pos = range(len(spillover["bin_labels"]))
    bars = ax.bar(x_pos, spillover["bin_pct_pos"], color="#E41A1C", alpha=0.7,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(spillover["bin_labels"], fontsize=TICK_SIZE, rotation=30, ha="right")
    ax.set_xlabel("Distance to nearest myeloid cell (\u00b5m)", fontsize=LABEL_SIZE)
    ax.set_ylabel("% CD14+ among other cells", fontsize=LABEL_SIZE)
    ax.set_title(f"CD14 spillover gradient (\u03c1={spillover['rho_prox']:.3f})", fontsize=TITLE_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    # n= annotations removed (minimalist convention)

    # Inset pie for decomposition
    ax_ins = ax.inset_axes([0.42, 0.45, 0.42, 0.50])
    sizes = [spillover["mye_pos"], spillover["fdc_pos"], spillover["other_pos"]]
    n_pos = spillover["n_pos"]
    labels = [
        f"Myeloid\n{100*spillover['mye_pos']/n_pos:.0f}%",
        f"FDC\n{100*spillover['fdc_pos']/n_pos:.0f}%",
        f"Spillover\n{100*spillover['other_pos']/n_pos:.0f}%",
    ]
    pie_colors = [CAT_COLORS["Myeloid"], CAT_COLORS["FDC"], CAT_COLORS["Spillover"]]
    ax_ins.pie(sizes, labels=labels, colors=pie_colors, startangle=90,
               textprops={"fontsize": 10},
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax_ins.set_title(f"CD14+ cells (n={n_pos:,})", fontsize=TICK_SIZE, pad=2)


def _plot_compartment_survival(ax, comp_surv):
    """Forest plot: follicular vs interfollicular FDC CD14 → PFS/OS."""
    results = comp_surv["results"]
    order = [
        ("PFS", "fdc_cd14_foll"), ("PFS", "fdc_cd14_ifoll"),
        ("OS", "fdc_cd14_foll"), ("OS", "fdc_cd14_ifoll"),
    ]
    y_positions, row_labels, hrs, los, his, ps, row_colors = [], [], [], [], [], [], []
    y = 0
    prev_ep = None
    for ep, metric in order:
        if prev_ep is not None and ep != prev_ep:
            y += 0.5
        match = [r for r in results if r["endpoint"] == ep and r["metric"] == metric]
        if match:
            r = match[0]
            y_positions.append(y)
            row_labels.append(f"{ep}: {r['label']}")
            hrs.append(r["HR"]); los.append(r["lo"]); his.append(r["hi"])
            ps.append(r["p"])
            row_colors.append("#D32F2F" if "Follicular" in r["label"] else "#7F8C8D")
        y += 1
        prev_ep = ep

    if not y_positions:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=ANNOT_SIZE)
        return

    y_positions = np.array(y_positions)
    hrs = np.array(hrs); los = np.array(los); his = np.array(his)
    for i in range(len(y_positions)):
        ax.plot([los[i], his[i]], [y_positions[i]]*2,
                color=row_colors[i], linewidth=2, solid_capstyle="round")
        ax.plot(hrs[i], y_positions[i], "o", color=row_colors[i],
                markersize=8, markeredgecolor="white", markeredgewidth=0.5)
        sig = "***" if ps[i] < 0.001 else "**" if ps[i] < 0.01 else "*" if ps[i] < 0.05 else ""
        ax.text(max(his[i] + 0.02, 2.2), y_positions[i],
                f"HR={hrs[i]:.2f} P={ps[i]:.4f} {sig}",
                va="center", fontsize=ANNOT_SIZE, color=row_colors[i], fontweight="bold")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=TICK_SIZE)
    ax.invert_yaxis()
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Hazard Ratio (per SD)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    # Get n from first result
    n_foll = next((r["n"] for r in results if "Follicular" in r["label"]), "?")
    n_ifoll = next((r["n"] for r in results if "Interfollicular" in r["label"]), "?")
    ax.set_title(f"FDC CD14 by compartment\n(Cox, treated, n={n_foll} foll / {n_ifoll} ifoll)",
                 fontsize=TITLE_SIZE)
    ax.set_xlim(0.2, 3.5)
    # Colors are self-explanatory from row labels — no legend needed


def _plot_localization(ax, intrafoll):
    fl = intrafoll["foll_localization"]
    categories = ["Follicular", "Interfollicular", "Other"]
    hi_vals = [fl["foll_frac_hi"]*100, fl["ifoll_frac_hi"]*100, fl["other_frac_hi"]*100]
    lo_vals = [fl["foll_frac_lo"]*100, fl["ifoll_frac_lo"]*100, fl["other_frac_lo"]*100]
    x_pos = np.arange(len(categories))
    bar_w = 0.35
    ax.bar(x_pos - bar_w/2, hi_vals, bar_w, color="#FFD700",
           edgecolor="black", linewidth=0.5, label=f"CD14-high (n={fl['n_hi']:,})")
    ax.bar(x_pos + bar_w/2, lo_vals, bar_w, color="#1976D2",
           edgecolor="black", linewidth=0.5, label=f"CD14-low (n={fl['n_lo']:,})")
    for xp, hv, lv in zip(x_pos, hi_vals, lo_vals):
        if hv > 3:
            ax.text(xp - bar_w/2, hv + 1, f"{hv:.0f}%", ha="center", fontsize=ANNOT_SIZE,
                    fontweight="bold")
        if lv > 3:
            ax.text(xp + bar_w/2, lv + 1, f"{lv:.0f}%", ha="center", fontsize=ANNOT_SIZE,
                    fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=TICK_SIZE)
    ax.set_ylabel("% of FDCs", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")
    p_str = f"P={fl['chi2_p']:.1e}" if fl["chi2_p"] < 0.001 else f"P={fl['chi2_p']:.3f}"
    ax.set_title(f"FDC compartment localization\n(\u03c7\u00b2={fl['chi2']:.0f}, {p_str})",
                 fontsize=TITLE_SIZE)
    ax.set_ylim(0, max(hi_vals + lo_vals) * 1.15)


def _plot_intrafoll_markers(ax, intrafoll):
    mf = intrafoll["marker_foll"]
    cat_markers, cat_labels, bar_colors = [], [], []
    for cat, mkrs in FUNCTIONAL_MARKERS.items():
        for m in mkrs:
            if m in mf:
                cat_markers.append(m)
                cat_labels.append(cat)
                bar_colors.append(CAT_COLOR_MAP_INTRAFOLL.get(cat, "#999"))
    diffs = [mf[m]["diff"] for m in cat_markers]
    pvals = [mf[m]["p"] for m in cat_markers]
    names = [MARKER_DISPLAY.get(m, m) for m in cat_markers]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, diffs, color=bar_colors, alpha=0.85,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=TICK_SIZE)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Mean difference (CD14-high \u2212 CD14-low)", fontsize=LABEL_SIZE)
    ax.set_title("Intrafollicular FDC marker profile\n(follicular compartments only)",
                 fontsize=TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.invert_yaxis()
    for i, (d, p) in enumerate(zip(diffs, pvals)):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if star:
            x_off = 0.01 if d >= 0 else -0.01
            ha = "left" if d >= 0 else "right"
            ax.text(d + x_off, i, star, va="center", ha=ha, fontsize=ANNOT_SIZE, color="red")
    # Category legend
    seen = {}
    for cat in cat_labels:
        if cat not in seen:
            seen[cat] = CAT_COLOR_MAP_INTRAFOLL.get(cat, "#999")
    legend_h = [Patch(facecolor=c, label=cat.replace("\n", " ")) for cat, c in seen.items()]
    ax.legend(handles=legend_h, fontsize=LEGEND_SIZE, loc="center left",
              bbox_to_anchor=(1.02, 0.5), ncol=1,
              framealpha=0.8, title="Category", title_fontsize=LEGEND_SIZE)


def _plot_intrafoll_neighbors(ax, intrafoll):
    nd = intrafoll["nbr_data"]
    top_nbr = nd[:12]
    n_names = [ct[:30] for ct, _, _, _ in top_nbr]
    n_diffs = [diff * 100 for _, _, _, diff in top_nbr]
    n_colors = ["#D32F2F" if d > 0 else "#1976D2" for d in n_diffs]
    y_pos = np.arange(len(n_names))
    ax.barh(y_pos, n_diffs, color=n_colors, alpha=0.8,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(n_names, fontsize=TICK_SIZE)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("\u0394 neighbor fraction (pp)\n(CD14-high \u2212 CD14-low FDC)", fontsize=LABEL_SIZE)
    ax.set_title(
        f"Intrafollicular FDC neighbors (k=10)\n"
        f"(n_hi={intrafoll['n_hi_nbr']:,}, n_lo={intrafoll['n_lo_nbr']:,})",
        fontsize=TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.invert_yaxis()


def _plot_composite_inset(fig, comp):
    """Full-core cell scatter (left) + raw IMC composite zoom (right).

    Delegates to ``src.visualization.plot_scatter_composite_inset``.
    """
    plot_scatter_composite_inset(
        fig,
        cell_x=comp["cx"],
        cell_y=comp["cy"],
        cell_type=comp["ct"],
        highlight_value=comp["cd14"],
        highlight_threshold=comp["cd14_q75"],
        composite_rgb=comp["composite_rgb"],
        x0=comp["x0"], y0=comp["y0"],
        x1=comp["x1"], y1=comp["y1"],
        roi_label=comp["roi"],
        window_size=comp["window_size"],
        highlight_name="FDC",
        composite_legend=[
            {"color": "#00FF00", "label": "CD21 (FDC)"},
            {"color": "#FF0000", "label": "CD14"},
            {"color": "#FFFF00", "label": "CD21+CD14"},
            {"color": "#FF00FF", "label": "CD68 (Mac)"},
            {"color": "#00DDFF", "label": "CD8 (T cells)"},
        ],
        panel_letter="f",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Supplementary figure
# ═══════════════════════════════════════════════════════════════════════════

def _plot_suppl_scrna(ax, scrna):
    ct_data = scrna["cd14_by_ct"]
    names_d = [x[0] for x in ct_data]
    means_d = [x[2] for x in ct_data]
    colors_d = ["#FF7F00" if n == "FDC" else "#E41A1C" if n == "Myeloid"
                else "#CCCCCC" for n, _, _, _ in ct_data]
    ax.barh(range(len(names_d)), means_d, color=colors_d,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(range(len(names_d))))
    ax.set_yticklabels(names_d, fontsize=TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Mean CD14 mRNA expression", fontsize=LABEL_SIZE)
    ax.set_title("CD14 expression by cell type\n(Han et al. 2022, scRNA-seq)", fontsize=TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    for i, (name, _, mean, pct) in enumerate(ct_data):
        if name in ("FDC", "Myeloid"):
            ax.text(mean + 0.01, i, f"{pct:.0f}% cells +",
                    va="center", fontsize=TICK_SIZE, color="#333")


def _plot_suppl_proximity_ki67(ax, prox):
    """Ki-67 on B cells close vs distant to CD14+ FDCs — box plot."""
    # Subsample for plotting (box plots with 200k points are slow)
    rng = np.random.RandomState(42)
    max_pts = 5000
    ki67_c = prox["ki67_close"]
    ki67_d = prox["ki67_dist"]
    if len(ki67_c) > max_pts:
        ki67_c = rng.choice(ki67_c, max_pts, replace=False)
    if len(ki67_d) > max_pts:
        ki67_d = rng.choice(ki67_d, max_pts, replace=False)

    bp = ax.boxplot([ki67_c, ki67_d], positions=[0, 1], widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    colors = ["#FFD700", "#1976D2"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"Close\n(≤{PROX_THRESHOLD_PX} µm)\nn={prox['n_close']:,}",
        f"Distant\n(>{PROX_THRESHOLD_PX} µm)\nn={prox['n_dist']:,}",
    ], fontsize=LEGEND_SIZE)
    ax.set_ylabel("Ki-67 expression (scaled)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    p = prox["p_ki67"]
    p_str = f"P={p:.1e}" if p < 0.001 else f"P={p:.3f}"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    # Significance bracket above whiskers
    q75_c = np.percentile(ki67_c, 75)
    q75_d = np.percentile(ki67_d, 75)
    iqr_c = q75_c - np.percentile(ki67_c, 25)
    iqr_d = q75_d - np.percentile(ki67_d, 25)
    whisk_top = max(q75_c + 1.5 * iqr_c, q75_d + 1.5 * iqr_d)
    ymax = whisk_top * 1.05
    ax.plot([0, 0, 1, 1], [ymax, ymax * 1.02, ymax * 1.02, ymax],
            color="black", lw=1.2)
    ax.text(0.5, ymax * 1.03, f"{p_str}", ha="center", va="bottom",
            fontsize=ANNOT_SIZE)
    ax.set_title("B cell proliferation near CD14+ FDCs\n"
                 "(follicular B cells, IMC S-panel)",
                 fontsize=TITLE_SIZE, fontweight="medium", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_suppl_proximity_hla(ax, prox):
    """HLA-I and HLA-DR on B cells close vs distant to CD14+ FDCs — box plots."""
    rng = np.random.RandomState(42)
    max_pts = 5000

    markers_data = [
        ("HLA-I", prox["hlai_close"], prox["hlai_dist"], prox["p_hlai"]),
        ("HLA-DR", prox["hladr_close"], prox["hladr_dist"], prox["p_hladr"]),
    ]

    positions_close = [0, 2.5]
    positions_dist = [0.6, 3.1]
    colors_close = "#FFD700"
    colors_dist = "#1976D2"
    box_w = 0.5

    all_data_close, all_data_dist = [], []
    for name, close_arr, dist_arr, p in markers_data:
        c = rng.choice(close_arr, max_pts, replace=False) if len(close_arr) > max_pts else close_arr
        d = rng.choice(dist_arr, max_pts, replace=False) if len(dist_arr) > max_pts else dist_arr
        all_data_close.append(c)
        all_data_dist.append(d)

    bp_c = ax.boxplot(all_data_close, positions=positions_close, widths=box_w,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color="black", linewidth=2),
                      whiskerprops=dict(linewidth=1.2),
                      capprops=dict(linewidth=1.2))
    bp_d = ax.boxplot(all_data_dist, positions=positions_dist, widths=box_w,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color="black", linewidth=2),
                      whiskerprops=dict(linewidth=1.2),
                      capprops=dict(linewidth=1.2))
    for patch in bp_c["boxes"]:
        patch.set_facecolor(colors_close)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
    for patch in bp_d["boxes"]:
        patch.set_facecolor(colors_dist)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")

    # X-axis labels centered between each pair
    ax.set_xticks([0.3, 2.8])
    ax.set_xticklabels(["HLA-I", "HLA-DR"], fontsize=TICK_SIZE)
    ax.set_ylabel("Expression (scaled)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    # Legend
    from matplotlib.patches import Patch as LegPatch
    ax.legend(handles=[
        LegPatch(facecolor=colors_close, edgecolor="black",
                 label=f"Close (≤{PROX_THRESHOLD_PX} µm, n={prox['n_close']:,})"),
        LegPatch(facecolor=colors_dist, edgecolor="black",
                 label=f"Distant (>{PROX_THRESHOLD_PX} µm, n={prox['n_dist']:,})"),
    ], fontsize=LEGEND_SIZE, loc="upper center")

    # Significance brackets
    for i, (name, close_arr, dist_arr, p) in enumerate(markers_data):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        p_str = f"P={p:.1e}" if p < 0.001 else f"P={p:.3f}"
        pc, pd = positions_close[i], positions_dist[i]
        c_arr, d_arr = all_data_close[i], all_data_dist[i]
        q75_c = np.percentile(c_arr, 75)
        q75_d = np.percentile(d_arr, 75)
        iqr_c = q75_c - np.percentile(c_arr, 25)
        iqr_d = q75_d - np.percentile(d_arr, 25)
        whisk_top = max(q75_c + 1.5 * iqr_c, q75_d + 1.5 * iqr_d)
        ymax = whisk_top * 1.05
        ax.plot([pc, pc, pd, pd], [ymax, ymax * 1.02, ymax * 1.02, ymax],
                color="black", lw=1.2)
        ax.text((pc + pd) / 2, ymax * 1.03, f"{p_str}",
                ha="center", va="bottom", fontsize=ANNOT_SIZE)

    ax.set_title("B cell HLA expression near CD14+ FDCs\n"
                 "(follicular B cells, IMC S-panel)",
                 fontsize=TITLE_SIZE, fontweight="medium", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_suppl_umi(ax, fdc_umi):
    """UMI counts per cell: CD14+ vs CD14- FDCs (transcriptional activity)."""
    umi_pos = fdc_umi["umi_pos"]
    umi_neg = fdc_umi["umi_neg"]

    # Log-scale histogram for better visual separation
    bins = np.logspace(np.log10(max(1, min(umi_neg.min(), umi_pos.min()))),
                       np.log10(max(umi_pos.max(), umi_neg.max())), 50)
    ax.hist(umi_pos, bins=bins, alpha=0.6, color="#FFD700", density=True,
            label=f"CD14+ FDC (n={fdc_umi['n_pos']:,})")
    ax.hist(umi_neg, bins=bins, alpha=0.6, color="#1976D2", density=True,
            label=f"CD14\u2212 FDC (n={fdc_umi['n_neg']:,})")
    ax.axvline(fdc_umi["med_pos"], color="#B8860B", ls="--", lw=2,
               label=f"Median={fdc_umi['med_pos']:,.0f}")
    ax.axvline(fdc_umi["med_neg"], color="#0D47A1", ls="--", lw=2,
               label=f"Median={fdc_umi['med_neg']:,.0f}")
    ax.set_xscale("log")
    p = fdc_umi["p"]
    p_str = f"P={p:.1e}" if p < 0.001 else f"P={p:.3f}"
    ax.set_xlabel("UMI counts per cell", fontsize=LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_title(
        f"FDC transcriptional activity ({fdc_umi['ratio']:.1f}x, {p_str})\n"
        f"(Han et al. 2022, scRNA-seq)", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")


def make_suppl_figure(data, output_dir, cache_dir=None, panels_to_refresh=None):
    """10-panel supplementary: validation + tonsil CD14 + proximity + signaling.

    Direct-render into single figure (no PNG cache) for vectorized PDF output.
    Layout: row 1 (3 cols: a, b, c), rows 2-3 (2 cols: d-g), row 4 (3 cols: h-j).
    """
    from fig_tonsil_comparison import plot_tonsil_cd14_fdc

    panels = [
        ("a", _plot_decomposition_spillover, ["spillover"]),
        ("b", _plot_suppl_scrna, ["scrna"]),
        ("c", plot_tonsil_cd14_fdc, ["tonsil_s"]),
        ("d", _plot_suppl_umi, ["fdc_umi"]),
        ("e", _plot_suppl_proximity_ki67, ["proximity"]),
        ("f", _plot_suppl_proximity_hla, ["proximity"]),
        ("g", plot_fdc_survival_bars, ["signaling_scrna"]),
        ("h", plot_scrna_dotplot, ["signaling_scrna"]),
        ("i", plot_cxcl13_cd21, ["signaling_imc"]),
        ("j", plot_signaling_heatmap, ["signaling_imc"]),
    ]

    fig = plt.figure(figsize=(20, 26))
    gs_outer = gridspec.GridSpec(4, 1, figure=fig,
                                 height_ratios=[1, 1, 1, 1.1],
                                 hspace=0.32,
                                 left=0.06, right=0.96, top=0.97, bottom=0.04)
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0],
                                                wspace=0.30)
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[1],
                                                wspace=0.28)
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[2],
                                                wspace=0.28)
    gs_row4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[3],
                                                wspace=0.30,
                                                width_ratios=[1.2, 1, 1.2])

    layout = {
        "a": gs_row1[0], "b": gs_row1[1], "c": gs_row1[2],
        "d": gs_row2[0], "e": gs_row2[1],
        "f": gs_row3[0], "g": gs_row3[1],
        "h": gs_row4[0], "i": gs_row4[1], "j": gs_row4[2],
    }

    for letter, plot_fn, data_keys in panels:
        ax = fig.add_subplot(layout[letter])
        plot_args = [data[k] for k in data_keys]
        plot_fn(ax, *plot_args)
        panel_label(ax, letter)

    out = Path(output_dir) / "fig_fdc_cd14_suppl.png"
    save_figure(fig, out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _load_composite_data(s_utag_path):
    """Load cell data + raw IMC composite for the representative ROI."""
    import h5py
    from fig_fdc_intrafollicular import load_array, is_tumor_core, EXCLUDE_ROIS

    roi = COMPOSITE_ROI
    cx_center, cy_center = COMPOSITE_CENTER
    win_sz = COMPOSITE_WINDOW
    half = win_sz / 2
    x0, y0 = int(cx_center - half), int(cy_center - half)
    x1, y1 = int(cx_center + half), int(cy_center + half)

    # Cell data for this ROI
    with h5py.File(s_utag_path, "r") as f:
        sid = load_array(f, "sample_id")
        ct = load_array(f, "cell_type")
        cx = f["obs"]["centroid_x"][:]
        cy = f["obs"]["centroid_y"][:]
        var_names = [v.decode() if isinstance(v, bytes) else v
                     for v in f["var"]["_index"][:]]
        cd14_idx = var_names.index("CD14")
        cd14_all = np.array(f["X"][:, cd14_idx]).flatten()
    # Filter to tumor cores for Q75 (consistent with other extract functions)
    tumor_mask = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS
                           and not s.startswith("Biomax") for s in sid])
    cd14_q75 = float(np.percentile(cd14_all[(ct == "FDC") & tumor_mask], 75))
    mask = sid == roi
    print(f"  {roi}: {mask.sum():,} cells, CD14 Q75={cd14_q75:.3f}")

    # Raw IMC composite
    raw_file = find_raw_file(roi, RAW_DIR)
    if raw_file is None:
        raise FileNotFoundError(f"No raw file for {roi} in {RAW_DIR}")
    print(f"  Raw file: {raw_file.name}")
    composite_rgb = build_raw_composite(raw_file, RAW_CHANNELS, COMPOSITE_RGB,
                                        x0, y0, x1, y1)

    return {
        "ct": ct[mask], "cx": cx[mask], "cy": cy[mask], "cd14": cd14_all[mask],
        "cd14_q75": cd14_q75, "composite_rgb": composite_rgb,
        "roi": roi, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "window_size": win_sz,
    }


def load_all_data(args):
    """Extract data from all sources. Returns unified dict."""
    print("=== Loading data from h5ad files ===")

    print("\n--- Spillover data ---")
    spillover = extract_spillover_data(args.s_panel)
    print("\n--- scRNA data ---")
    scrna = extract_scrna_data(args.scrna)
    print("\n--- Compartment-split FDC survival ---")
    comp_surv = extract_compartment_survival(args.s_utag)
    print("\n--- Functional data ---")
    functional = extract_functional_data(args.s_panel)
    print("\n--- Intrafollicular data ---")
    s = extract_s(args.s_utag)
    t = extract_t(args.t_utag)
    intrafoll = analyze_all(s, t)
    print("\n--- FDC UMI counts ---")
    fdc_umi = extract_fdc_umi(args.scrna)
    print("\n--- Proliferation proximity ---")
    prolif = extract_proliferation_data(args.s_utag)
    print("\n--- Proximity markers (Ki-67, HLA) ---")
    proximity = extract_proximity_markers(args.s_utag)
    print("\n--- Composite panel data ---")
    composite = _load_composite_data(args.s_utag)
    print("\n--- Signaling architecture data (for suppl panels e-h) ---")
    signaling_scrna = extract_signaling_scrna(args.scrna)
    signaling_imc = extract_signaling_imc(args.s_panel, args.s_utag)

    print("\n--- Tonsil S-panel data (for CD14 FDC comparison) ---")
    from fig_tonsil_comparison import extract_s_panel as extract_tonsil_s
    tonsil_s = extract_tonsil_s(args.s_utag)
    print(f"  Tonsil: {tonsil_s['tonsil_mask'].sum():,} cells, "
          f"FL: {tonsil_s['tumor_mask'].sum():,} cells")

    return {
        "spillover": spillover,
        "scrna": scrna,
        "comp_surv": comp_surv,
        "functional": functional,
        "intrafoll": intrafoll,
        "fdc_umi": fdc_umi,
        "prolif": prolif,
        "proximity": proximity,
        "composite": composite,
        "signaling_scrna": signaling_scrna,
        "signaling_imc": signaling_imc,
        "tonsil_s": tonsil_s,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CD14+ FDC biology figures (main + supplementary)")
    parser.add_argument("--s-panel", default="output/all_TMA_S_global_v8.h5ad")
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--t-utag", default="output/all_TMA_T_utag_ct_merged.h5ad")
    parser.add_argument("--scrna", default="data/external/steen2022_fl_scrna.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    parser.add_argument("--cache-dir", default="output/hypotheses_v8/_panel_cache")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-extract data from h5ad files")
    parser.add_argument("--panels",
                        help="Main panels to re-render (comma-separated, e.g. a,e,g)")
    parser.add_argument("--suppl-panels",
                        help="Suppl panels to re-render (comma-separated, e.g. a,d)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    data_cache = cache_dir / "data.pkl"

    # --- Data loading (with pickle cache) ---
    if not args.no_cache and data_cache.exists():
        print(f"=== Loading cached data from {data_cache} ===")
        with open(data_cache, "rb") as f:
            data = pickle.load(f)
        print(f"  Keys: {sorted(data.keys())}")
    else:
        data = load_all_data(args)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(data_cache, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n  Cached data → {data_cache}")

    # Ensure tonsil_s is available (may be missing from old cache)
    if "tonsil_s" not in data:
        print("\n--- Tonsil S-panel data (supplementing cache) ---")
        from fig_tonsil_comparison import extract_s_panel as extract_tonsil_s
        data["tonsil_s"] = extract_tonsil_s(args.s_utag)
        print(f"  Tonsil: {data['tonsil_s']['tonsil_mask'].sum():,} cells")

    # --- Parse panel refresh args ---
    main_refresh = set(args.panels.split(",")) if args.panels else None
    suppl_refresh = set(args.suppl_panels.split(",")) if args.suppl_panels else None

    # --- Build figures ---
    print("\n=== Building main figure ===")
    out_main = make_main_figure(data, args.output_dir, cache_dir,
                                panels_to_refresh=main_refresh)

    print("\n=== Building supplementary figure ===")
    out_suppl = make_suppl_figure(data, args.output_dir, cache_dir,
                                  panels_to_refresh=suppl_refresh)

    print(f"\nDone.\n  Main:  {out_main}\n  Suppl: {out_suppl}")
