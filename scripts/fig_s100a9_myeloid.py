#!/usr/bin/env python3
"""
Figure: S100A9+ MDSC-like myeloid cells in follicular lymphoma (H3d).

Characterizes S100A9+ myeloid cells as calprotectin-expressing MDSC-like
inflammatory myeloid cells concentrated in interfollicular zones, co-occurring
with active immunity (cytotoxic T, M1 Mac), and showing dramatic compartment-
dependent phenotype switching.

Panels:
  (a) Neighborhood enrichment: K=10 nearest-neighbor cell type fractions
  (b) Spatial scatter: representative ROI with S100A9+ highlighted
  (c) Co-occurrence ecology: per-ROI correlation with other cell types
  (d) Compartment-specific phenotype: marker expression by UTAG compartment
  (e) scRNA-seq validation: DE genes in S100A9-high vs S100A9-low myeloid

Usage:
    .venv/bin/python scripts/fig_s100a9_myeloid.py \
        --s-panel output/all_TMA_S_global_v8.h5ad \
        --s-utag output/all_TMA_S_utag_ct_merged.h5ad \
        --t-panel output/all_TMA_T_global_v8.h5ad \
        --scrna data/external/steen2022_fl_scrna.h5ad \
        --output-dir output/hypotheses_v8
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22



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


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(
        x, y, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, va="bottom", ha="left",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOLL_COMPARTMENTS = {
    "B cell zone (BCL2+)", "B cell zone (PAX5+)",
    "FDC network zone", "FDC / myeloid zone",
}

# scRNA-seq genes (Ensembl IDs for Han 2022)
SCRNA_GENES = {
    "S100A9": "ENSG00000163220",
    "S100A8": "ENSG00000143546",
    "S100A12": "ENSG00000163221",
    "FCN1": "ENSG00000085265",
    "VCAN": "ENSG00000038427",
    "SERPINA1": "ENSG00000197249",
    "TYROBP": "ENSG00000011600",
    "LILRB2": "ENSG00000131042",
    "CD14": "ENSG00000170458",
    "ITGAM": "ENSG00000169896",
    "VSIR": "ENSG00000107738",
    "HLA-DRA": "ENSG00000204287",
}
SCRNA_LABELS = {
    "S100A9": "S100A9", "S100A8": "S100A8", "S100A12": "S100A12",
    "FCN1": "FCN1", "VCAN": "VCAN", "SERPINA1": "SERPINA1",
    "TYROBP": "TYROBP", "LILRB2": "LILRB2",
    "CD14": "CD14", "ITGAM": "CD11b", "VSIR": "VISTA",
    "HLA-DRA": "HLA-DRA",
}
SCRNA_ORDER = [
    "S100A9", "S100A8", "S100A12", "FCN1", "VCAN", "SERPINA1",
    "TYROBP", "LILRB2", "CD14", "ITGAM", "VSIR", "HLA-DRA",
]

# Compartment groupings for Q4
COMP_GROUPS = {
    "T cell zone": {"T cell zone"},
    "B/T mixed": {"B/T mixed zone", "Other / myeloid zone"},
    "Follicular": {"B cell zone (BCL2+)", "B cell zone (PAX5+)",
                   "FDC network zone", "FDC / myeloid zone"},
}
COMP_COLORS = {
    "T cell zone": "#4292C6",
    "B/T mixed": "#9ECAE1",
    "Follicular": "#D73027",
}

# Q4 markers to show
Q4_MARKERS = ["VISTA", "S100A9", "IDO", "CD11b", "CD68", "CD11c", "CD34",
              "CD14", "HLA_DR", "CD163"]
Q4_LABELS = {
    "VISTA": "VISTA", "S100A9": "S100A9", "IDO": "IDO", "CD11b": "CD11b",
    "CD68": "CD68", "CD11c": "CD11c", "CD34": "CD34", "CD14": "CD14",
    "HLA_DR": "HLA-DR", "CD163": "CD163",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_panel_data(path, panel_name):
    """Extract expression, cell types, coordinates from a panel h5ad."""
    print(f"Loading {panel_name} data...")
    f = h5py.File(path, "r")
    X = f["X"][:]
    markers = [v.decode() for v in f["var"]["_index"][:]]
    cell_types = load_array(f, "cell_type")
    sample_ids = load_array(f, "sample_id")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()

    marker_idx = {m: i for i, m in enumerate(markers)}
    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS and not s.startswith("Biomax")
        for s in sample_ids
    ])
    print(f"  {tumor_mask.sum():,} tumor cells")

    return {
        "X": X[tumor_mask], "markers": markers, "marker_idx": marker_idx,
        "cell_types": cell_types[tumor_mask],
        "sample_ids": sample_ids[tumor_mask],
        "cx": cx[tumor_mask], "cy": cy[tumor_mask],
    }


def extract_compartment_data(s_utag_path):
    """Extract UTAG compartment labels."""
    print("Loading S-UTAG data...")
    f = h5py.File(s_utag_path, "r")
    comp_names = load_array(f, "compartment_name")
    sample_ids = load_array(f, "sample_id")
    cell_types = load_array(f, "cell_type")
    f.close()

    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS and not s.startswith("Biomax")
        for s in sample_ids
    ])
    return {
        "comps": comp_names[tumor_mask],
        "cell_types": cell_types[tumor_mask],
        "sample_ids": sample_ids[tumor_mask],
    }


def extract_scrna_data(scrna_path):
    """Extract S100A9-high vs S100A9-low myeloid DE from Han 2022."""
    import scanpy as sc
    from scipy.stats import mannwhitneyu

    print("Loading scRNA-seq data...")
    adata = sc.read_h5ad(scrna_path)
    fl = adata[adata.obs["disease"] == "follicular lymphoma"]
    myeloid = fl[fl.obs["cell_type"] == "myeloid cell"].copy()
    print(f"  FL myeloid cells: {myeloid.n_obs}")

    var_ids = list(myeloid.var.index)
    s100a9_idx = var_ids.index(SCRNA_GENES["S100A9"])
    s100a9_vals = myeloid.X[:, s100a9_idx]
    if hasattr(s100a9_vals, "toarray"):
        s100a9_vals = s100a9_vals.toarray().ravel()
    s100a9_vals = np.array(s100a9_vals).ravel()

    thresh = np.percentile(s100a9_vals[s100a9_vals > 0], 50)
    is_hi = s100a9_vals > thresh
    n_hi, n_lo = int(is_hi.sum()), int((~is_hi).sum())
    print(f"  S100A9-high: {n_hi}, S100A9-low: {n_lo}")

    results = {}
    for gene_name in SCRNA_ORDER:
        ens_id = SCRNA_GENES[gene_name]
        if ens_id not in var_ids:
            continue
        idx = var_ids.index(ens_id)
        vals = myeloid.X[:, idx]
        if hasattr(vals, "toarray"):
            vals = vals.toarray().ravel()
        vals = np.array(vals).ravel()

        hi_mean = float(vals[is_hi].mean())
        lo_mean = float(vals[~is_hi].mean())
        fc = hi_mean / lo_mean if lo_mean > 0.01 else (hi_mean / 0.01 if hi_mean > 0 else 1.0)
        log2fc = np.log2(fc) if fc > 0 else 0.0

        _, pval = mannwhitneyu(vals[is_hi], vals[~is_hi], alternative="two-sided")
        results[gene_name] = {"log2fc": log2fc, "pval": pval}

    return {"de_results": results, "n_hi": n_hi, "n_lo": n_lo}


# ---------------------------------------------------------------------------
# Q3: Co-occurrence ecology
# ---------------------------------------------------------------------------

def compute_cooccurrence(s_data, t_data):
    """Per-ROI Spearman correlation of S100A9+ fraction vs all other cell types."""
    s_ct = s_data["cell_types"]
    s_sid = s_data["sample_ids"]
    t_ct = t_data["cell_types"]
    t_sid = t_data["sample_ids"]

    # S-panel per-ROI fractions
    s_rois = sorted(set(s_sid))
    s_all_cts = sorted(set(s_ct))
    s_roi_data = {}
    for roi in s_rois:
        m = s_sid == roi
        n = m.sum()
        if n < 200:
            continue
        ct_counts = Counter(s_ct[m])
        s_roi_data[roi] = {ct: ct_counts.get(ct, 0) / n for ct in s_all_cts}

    # T-panel per-ROI fractions
    t_all_cts = sorted(set(t_ct))
    t_roi_data = {}
    for roi in sorted(set(t_sid)):
        m = t_sid == roi
        n = m.sum()
        if n < 200:
            continue
        ct_counts = Counter(t_ct[m])
        t_roi_data[roi] = {ct: ct_counts.get(ct, 0) / n for ct in t_all_cts}

    s100_per_roi = {roi: d.get("Myeloid (S100A9+)", 0) for roi, d in s_roi_data.items()}
    s100_arr = np.array([s100_per_roi[roi] for roi in s_roi_data])

    # S-panel correlations
    s_results = []
    skip_s = {"Myeloid (S100A9+)", "Unassigned", "Low quality / Unassigned"}
    for ct in s_all_cts:
        if ct in skip_s:
            continue
        vals = np.array([s_roi_data[roi].get(ct, 0) for roi in s_roi_data])
        rho, p = stats.spearmanr(s100_arr, vals)
        if not np.isnan(rho):
            s_results.append((ct, rho, p, "S"))

    # T-panel cross-correlations
    shared = sorted(set(s_roi_data.keys()) & set(t_roi_data.keys()))
    s100_shared = np.array([s100_per_roi[roi] for roi in shared])
    skip_t = {"Unassigned", "Low quality / Unassigned", "Mixed / Border cells"}
    t_results = []
    for ct in t_all_cts:
        if ct in skip_t:
            continue
        vals = np.array([t_roi_data[roi].get(ct, 0) for roi in shared])
        rho, p = stats.spearmanr(s100_shared, vals)
        if not np.isnan(rho):
            t_results.append((ct, rho, p, "T"))

    return s_results, t_results


# ---------------------------------------------------------------------------
# Q4: Compartment phenotype
# ---------------------------------------------------------------------------

def compute_compartment_phenotype(s_data, utag_data):
    """Mean marker expression in S100A9+ cells by compartment group."""
    s_ct = s_data["cell_types"]
    s_X = s_data["X"]
    markers = s_data["markers"]
    u_comps = utag_data["comps"]

    # Verify alignment
    if len(s_ct) != len(u_comps):
        print(f"  WARNING: length mismatch S-panel {len(s_ct)} vs UTAG {len(u_comps)}")
        return None

    s100_mask = s_ct == "Myeloid (S100A9+)"
    s100_comps = u_comps[s100_mask]
    s100_X = s_X[s100_mask]

    result = {}  # marker -> {group: (mean, n)}
    for mk_name in Q4_MARKERS:
        if mk_name not in markers:
            continue
        mk_idx = markers.index(mk_name)
        result[mk_name] = {}
        for group, comp_set in COMP_GROUPS.items():
            mask = np.isin(s100_comps, list(comp_set))
            if mask.sum() > 5:
                result[mk_name][group] = (float(s100_X[mask, mk_idx].mean()), int(mask.sum()))
            else:
                result[mk_name][group] = (np.nan, 0)

    return result


# ---------------------------------------------------------------------------
# Display group consolidation for neighbor bar chart
# ---------------------------------------------------------------------------

DISPLAY_GROUPS = {
    "FDC": {"FDC", "FDC (CD14+)", "FDC (CXCL13+)", "FDC (CD21+)"},
    "BCL2+ B": {"B cells (BCL2+)"},
    "PAX5+ B": {"B cells (PAX5+)"},
    "CD4 T": {"CD4 T cells"},
    "CD8 T": {"CD8 T cells"},
    "M1 Mac": {"M1 Macrophages"},
    "M2 Mac": {"M2 Macrophages"},
    "S100A9+": {"Myeloid (S100A9+)"},
    "DC": {"Dendritic cells", "pDC"},
    "Endothelial": {"Endothelial"},
    "Stromal/CAF": {"Stromal / CAF", "FRC (PDPN+)"},
    "Other": {"Histiocytes", "Other", "Unassigned", "Low quality / Unassigned"},
}

DISPLAY_GROUP_COLORS = {
    "FDC": "#984EA3",
    "BCL2+ B": "#1F78B4",
    "PAX5+ B": "#A6CEE3",
    "CD4 T": "#33A02C",
    "CD8 T": "#FFD700",
    "M1 Mac": "#E41A1C",
    "M2 Mac": "#FF7F00",
    "S100A9+": "#A65628",
    "DC": "#F781BF",
    "Endothelial": "#66C2A5",
    "Stromal/CAF": "#B2DF8A",
    "Other": "#CCCCCC",
}

# Reverse lookup: cell_type -> display group
_CT_TO_GROUP = {}
for grp, cts in DISPLAY_GROUPS.items():
    for ct in cts:
        _CT_TO_GROUP[ct] = grp


def compute_neighborhood_enrichment(s_data, K=10, n_perm=200):
    """Permutation-based neighborhood enrichment z-scores for S100A9+ cells.

    For each display group, computes z = (observed - null_mean) / null_std
    where null is generated by shuffling cell type labels within each ROI.
    Returns dict {display_group: z_score} and n_query count.
    """
    print(f"  Computing S100A9+ neighborhood enrichment (K={K}, {n_perm} perms)...")
    ct = s_data["cell_types"]
    sid = s_data["sample_ids"]
    cx = s_data["cx"]
    cy = s_data["cy"]
    rng = np.random.default_rng(42)

    rois = sorted(set(sid))
    groups = list(DISPLAY_GROUPS.keys())
    n_groups = len(groups)

    # Map all cells to display groups
    ct_mapped = np.array([_CT_TO_GROUP.get(c, "Other") for c in ct])

    obs_counts = np.zeros(n_groups)
    null_counts = np.zeros((n_perm, n_groups))
    n_query = 0

    for roi in rois:
        rmask = sid == roi
        n_cells = rmask.sum()
        if n_cells < 200:
            continue
        roi_idx = np.where(rmask)[0]
        roi_ct_mapped = ct_mapped[roi_idx]
        roi_cx = cx[roi_idx]
        roi_cy = cy[roi_idx]

        s100_local = roi_ct_mapped == "S100A9+"
        n_s100 = s100_local.sum()
        if n_s100 < 3:
            continue

        coords = np.column_stack([roi_cx, roi_cy])
        tree = KDTree(coords)
        k_q = min(K + 1, n_cells)
        _, indices = tree.query(coords[s100_local], k=k_q)
        neigh_idx = indices[:, 1:]  # exclude self

        # Observed
        neigh_labels = roi_ct_mapped[neigh_idx.ravel()]
        for gi, g in enumerate(groups):
            obs_counts[gi] += (neigh_labels == g).sum()

        # Permutations
        for p in range(n_perm):
            perm_ct = rng.permutation(roi_ct_mapped)
            perm_s100 = perm_ct == "S100A9+"
            if perm_s100.sum() < 3:
                continue
            # Use same tree, query permuted S100A9+ positions
            _, p_indices = tree.query(coords[perm_s100], k=k_q)
            p_neigh = p_indices[:, 1:]
            p_labels = perm_ct[p_neigh.ravel()]
            for gi, g in enumerate(groups):
                null_counts[p, gi] += (p_labels == g).sum()

        n_query += n_s100

    # Z-scores
    null_mean = null_counts.mean(axis=0)
    null_std = null_counts.std(axis=0)
    z_scores = np.where(null_std > 0, (obs_counts - null_mean) / null_std, 0)

    result = {groups[i]: float(z_scores[i]) for i in range(n_groups)}
    print(f"    {n_query:,} S100A9+ cells queried")
    for g in groups:
        if abs(result[g]) > 2:
            print(f"    {g}: z={result[g]:+.1f}")
    return result, n_query


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(imc, utag, t_data, scrna, output_dir):
    """Generate 5-panel S100A9+ characterization figure."""

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(3, 2, figure=fig, wspace=0.32, hspace=0.30,
                  left=0.08, right=0.95, top=0.94, bottom=0.04,
                  height_ratios=[1, 1, 1])

    # No suptitle — PDF assembly adds the figure title

    cell_types = imc["cell_types"]
    sample_ids = imc["sample_ids"]

    # ── (a) Neighborhood enrichment: z-scores + raw fractions side by side ──
    print("  Panel (a): neighborhood enrichment z-scores + fractions...")
    # Use two side-by-side subplots within the gs[0,0] cell
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.4)
    ax_a1 = fig.add_subplot(gs_a[0])
    ax_a2 = fig.add_subplot(gs_a[1])
    panel_label(ax_a1, "a", x=-0.35)

    z_data, n_query = compute_neighborhood_enrichment(imc, K=10, n_perm=200)

    # Also compute raw neighbor fractions
    from scipy.spatial import KDTree as _KDTree
    ct_all = imc["cell_types"]
    sid_all = imc["sample_ids"]
    cx_all = imc["cx"]
    cy_all = imc["cy"]
    ct_mapped_all = np.array([_CT_TO_GROUP.get(c, "Other") for c in ct_all])
    raw_counts = Counter()
    n_total_nbr = 0
    for roi in sorted(set(sid_all)):
        rmask = sid_all == roi
        if rmask.sum() < 200:
            continue
        roi_idx = np.where(rmask)[0]
        roi_ct = ct_mapped_all[roi_idx]
        s100_local = roi_ct == "S100A9+"
        if s100_local.sum() < 3:
            continue
        coords = np.column_stack([cx_all[roi_idx], cy_all[roi_idx]])
        tree = _KDTree(coords)
        k_q = min(11, len(coords))
        _, indices = tree.query(coords[s100_local], k=k_q)
        neigh_idx = indices[:, 1:]
        neigh_labels = roi_ct[neigh_idx.ravel()]
        raw_counts.update(neigh_labels)
        n_total_nbr += len(neigh_labels)
    raw_fracs = {g: raw_counts.get(g, 0) / n_total_nbr * 100 if n_total_nbr > 0 else 0
                 for g in DISPLAY_GROUPS}

    # Sort by z-score, exclude S100A9+ self and Other
    grp_order = sorted([g for g in z_data if g not in ("S100A9+", "Other")],
                       key=lambda g: -z_data[g])

    y_pos = np.arange(len(grp_order))

    # Left panel: z-scores
    z_vals = [z_data[g] for g in grp_order]
    colors_z = ["#D73027" if z > 0 else "#4292C6" for z in z_vals]
    ax_a1.barh(y_pos, z_vals, color=colors_z, alpha=0.85,
               edgecolor="white", linewidth=0.3)
    ax_a1.set_yticks(y_pos)
    ax_a1.set_yticklabels(grp_order, fontsize=TICK_SIZE)
    ax_a1.invert_yaxis()
    ax_a1.axvline(0, color="black", linewidth=0.8)
    ax_a1.set_xlabel("Enrichment z-score", fontsize=LABEL_SIZE)
    ax_a1.set_title(f"Enrichment (200 perms)", fontsize=TITLE_SIZE, fontweight="bold")
    for i, z in enumerate(z_vals):
        ha = "left" if z >= 0 else "right"
        offset = 0.5 if z >= 0 else -0.5
        ax_a1.text(z + offset, i, f"{z:+.0f}", va="center", ha=ha, fontsize=ANNOT_SIZE)
    ax_a1.spines["top"].set_visible(False)
    ax_a1.spines["right"].set_visible(False)
    ax_a1.set_xlim(-250, max(z_vals) + 0.25 * max(abs(min(z_vals)), abs(max(z_vals))))

    # Right panel: raw neighbor fractions
    frac_vals = [raw_fracs[g] for g in grp_order]
    ax_a2.barh(y_pos, frac_vals, color="#555555", alpha=0.85,
               edgecolor="white", linewidth=0.3)
    ax_a2.set_yticks(y_pos)
    ax_a2.set_yticklabels([""] * len(grp_order))  # labels on left panel only
    ax_a2.invert_yaxis()
    ax_a2.set_xlabel("Neighbor fraction (%)", fontsize=LABEL_SIZE)
    ax_a2.set_title(f"Raw frequency (K=10)", fontsize=TITLE_SIZE, fontweight="bold")
    for i, f in enumerate(frac_vals):
        ax_a2.text(f + 0.3, i, f"{f:.1f}%", va="center", ha="left", fontsize=ANNOT_SIZE)
    ax_a2.spines["top"].set_visible(False)
    ax_a2.spines["right"].set_visible(False)

    # ── (b) Spatial scatter ─────────────────────────────────────────────
    print("  Panel (b): spatial scatter...")
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")

    s100_per_roi = Counter()
    for i, ct in enumerate(cell_types):
        if ct == "Myeloid (S100A9+)":
            s100_per_roi[sample_ids[i]] += 1

    best_roi = None
    best_score = 0
    for roi, n_s100 in s100_per_roi.most_common(30):
        rmask = sample_ids == roi
        ct_c = Counter(cell_types[rmask])
        n_m1 = ct_c.get("M1 Macrophages", 0)
        n_cd8 = ct_c.get("CD8 T cells", 0)
        if n_s100 >= 50 and n_m1 >= 30 and n_cd8 >= 30:
            score = (n_s100 * n_m1 * n_cd8) ** (1/3)
            if score > best_score:
                best_score = score
                best_roi = roi

    if best_roi is None:
        best_roi = s100_per_roi.most_common(1)[0][0]

    print(f"    Representative ROI: {best_roi}")
    rmask = sample_ids == best_roi
    rx, ry = imc["cx"][rmask], imc["cy"][rmask]
    rct = cell_types[rmask]

    other_mask = ~np.isin(rct, ["Myeloid (S100A9+)", "M1 Macrophages", "CD8 T cells"])
    ax_b.scatter(rx[other_mask], ry[other_mask], c="#E0E0E0", s=0.5,
                 alpha=0.2, edgecolors="none", rasterized=True, zorder=1)
    cd8_m = rct == "CD8 T cells"
    ax_b.scatter(rx[cd8_m], ry[cd8_m], c="#1E90FF", s=2.5,
                 alpha=0.5, edgecolors="none", rasterized=True, zorder=2,
                 label=f"CD8 T ({cd8_m.sum()})")
    m1_m = rct == "M1 Macrophages"
    ax_b.scatter(rx[m1_m], ry[m1_m], c="#3CB371", s=5,
                 alpha=0.6, edgecolors="black", linewidth=0.2,
                 rasterized=True, zorder=3,
                 label=f"M1 Mac ({m1_m.sum()})")
    s100_m = rct == "Myeloid (S100A9+)"
    ax_b.scatter(rx[s100_m], ry[s100_m], c="#DA70D6", s=10,
                 alpha=0.8, edgecolors="black", linewidth=0.3,
                 rasterized=True, zorder=4,
                 label=f"S100A9+ ({s100_m.sum()})")

    ax_b.set_aspect("equal")
    ax_b.invert_yaxis()
    ax_b.axis("off")
    ax_b.set_title(f"Representative ROI ({best_roi})", fontsize=TITLE_SIZE, fontweight="bold")
    ax_b.legend(fontsize=LEGEND_SIZE, loc="lower left", framealpha=0.8,
                markerscale=1.5, handletextpad=0.3)

    # ── (c) Co-occurrence ecology ──────────────────────────────────────
    print("  Panel (c): co-occurrence ecology...")
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")

    s_results, t_results = compute_cooccurrence(imc, t_data)

    # Select top cell types by |rho| from each panel
    # S-panel: top 8 by |rho|
    s_results.sort(key=lambda x: -abs(x[1]))
    s_top = s_results[:8]
    # T-panel: top 6 by |rho|, excluding "Other" and types already similar to S-panel
    t_results.sort(key=lambda x: -abs(x[1]))
    t_top = t_results[:6]

    # Merge: T-panel on top, then separator, then S-panel
    all_entries = []
    for ct, rho, p, panel in reversed(t_top):
        short = ct.replace(" cells", "").replace("Macrophages", "Mac")
        all_entries.append((f"{short} (T)", rho, p, panel))
    all_entries.append(("__SEP__", 0, 1, ""))  # separator
    for ct, rho, p, panel in reversed(s_top):
        short = ct.replace(" cells", "").replace("Macrophages", "Mac")
        all_entries.append((f"{short} (S)", rho, p, panel))

    y_pos = []
    labels = []
    rhos = []
    pvals_b = []
    bar_colors = []
    y = 0
    for name, rho, p, panel in all_entries:
        if name == "__SEP__":
            y += 0.5  # gap for separator
            continue
        y_pos.append(y)
        labels.append(name)
        rhos.append(rho)
        pvals_b.append(p)
        if p < 0.05:
            bar_colors.append("#A65628" if rho > 0 else "#377EB8")
        else:
            bar_colors.append("#CCCCCC")
        y += 1

    ax_c.barh(y_pos, rhos, color=bar_colors, edgecolor="white",
              linewidth=0.5, alpha=0.85, height=0.75)

    for i, (yp, rho, p) in enumerate(zip(y_pos, rhos, pvals_b)):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if stars:
            offset = 0.02 if rho > 0 else -0.02
            ax_c.text(rho + offset, yp, stars,
                      ha="left" if rho > 0 else "right",
                      va="center", fontsize=ANNOT_SIZE, color="#333333")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax_c.axvline(0, color="black", linewidth=0.8)
    ax_c.set_xlabel("Spearman ρ with S100A9+ fraction", fontsize=LABEL_SIZE)
    ax_c.set_title("Co-occurrence ecology (per ROI)", fontsize=TITLE_SIZE, fontweight="bold")

    # Separator line between S and T panel results
    sep_y = y_pos[len(t_top) - 1] + 0.75
    ax_c.axhline(sep_y, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_c.text(0.95, sep_y + 0.3, "cross-panel (T)", fontsize=ANNOT_SIZE,
              color="gray", style="italic", ha="right", transform=ax_c.get_yaxis_transform())
    ax_c.text(0.95, sep_y - 0.5, "same panel (S)", fontsize=ANNOT_SIZE,
              color="gray", style="italic", ha="right", transform=ax_c.get_yaxis_transform())

    ax_c.set_xlim(-0.3, 0.65)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ── (d) Compartment-specific phenotype ─────────────────────────────
    print("  Panel (d): compartment phenotype...")
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")

    comp_data = compute_compartment_phenotype(imc, utag)
    if comp_data is not None:
        groups = list(COMP_GROUPS.keys())
        n_markers = len(Q4_MARKERS)
        n_groups = len(groups)
        bar_width = 0.25
        y_positions = np.arange(n_markers)

        for gi, group in enumerate(groups):
            vals = []
            for mk in Q4_MARKERS:
                if mk in comp_data and group in comp_data[mk]:
                    vals.append(comp_data[mk][group][0])
                else:
                    vals.append(0)
            offset = (gi - (n_groups - 1) / 2) * bar_width
            ax_d.barh(y_positions + offset, vals, height=bar_width * 0.9,
                      color=COMP_COLORS[group], alpha=0.85,
                      edgecolor="white", linewidth=0.3,
                      label=group)

        # Add significance markers (T zone vs Follicular, Mann-Whitney U)
        from scipy.stats import mannwhitneyu
        s_ct_all = imc["cell_types"]
        s_X_all = imc["X"]
        mk_list = imc["markers"]
        u_comps_all = utag["comps"]
        s100_mask_all = s_ct_all == "Myeloid (S100A9+)"
        s100_comps_all = u_comps_all[s100_mask_all]
        s100_X_all = s_X_all[s100_mask_all]
        foll_set = set(COMP_GROUPS["Follicular"])
        tzone_set = set(COMP_GROUPS["T cell zone"])
        for mi, mk in enumerate(Q4_MARKERS):
            if mk not in mk_list or mk not in comp_data:
                continue
            mk_idx = mk_list.index(mk)
            foll_vals = s100_X_all[np.isin(s100_comps_all, list(foll_set)), mk_idx]
            tzone_vals = s100_X_all[np.isin(s100_comps_all, list(tzone_set)), mk_idx]
            if len(foll_vals) > 5 and len(tzone_vals) > 5:
                _, pval = mannwhitneyu(tzone_vals, foll_vals, alternative="two-sided")
                if pval < 0.001:
                    max_val = max(comp_data[mk].get("T cell zone", (0,))[0],
                                  comp_data[mk].get("Follicular", (0,))[0])
                    ax_d.text(max_val + 0.15, mi, "**",
                              fontsize=ANNOT_SIZE, va="center", color="#333333",
                              fontweight="bold")

        ax_d.set_yticks(y_positions)
        ax_d.set_yticklabels([Q4_LABELS.get(m, m) for m in Q4_MARKERS], fontsize=TICK_SIZE)
        ax_d.invert_yaxis()
        ax_d.set_xlabel("Mean expression (z-scored)", fontsize=LABEL_SIZE)
        ax_d.set_title("Compartment-specific phenotype", fontsize=TITLE_SIZE, fontweight="bold")
        # Add significance note to legend
        from matplotlib.lines import Line2D
        leg_handles = [plt.Rectangle((0, 0), 1, 1, fc=COMP_COLORS[g], alpha=0.85)
                       for g in COMP_GROUPS]
        leg_labels = list(COMP_GROUPS.keys())
        leg_handles.append(Line2D([], [], marker='None', linestyle='None',
                                  label='** P < 0.001'))
        leg_labels.append('** P < 0.001 (Mann-Whitney)')
        ax_d.legend(leg_handles, leg_labels, fontsize=LEGEND_SIZE, loc="lower right",
                    framealpha=0.8)
        ax_d.axvline(0, color="black", linewidth=0.5, alpha=0.3)
        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)

    # ── (e) scRNA-seq validation ──────────────────────────────────────
    print("  Panel (e): scRNA-seq validation...")
    ax_e = fig.add_subplot(gs[2, 0])
    panel_label(ax_e, "e")

    de = scrna["de_results"]
    genes_to_plot = [g for g in SCRNA_ORDER if g in de]
    y_pos_d = np.arange(len(genes_to_plot))
    log2fcs = [de[g]["log2fc"] for g in genes_to_plot]
    pvals_d = [de[g]["pval"] for g in genes_to_plot]

    colors_e = ["#A65628" if lfc > 0 else "#377EB8" for lfc in log2fcs]
    ax_e.barh(y_pos_d, log2fcs, color=colors_e, edgecolor="white",
              linewidth=0.5, alpha=0.8)

    for i, (g, lfc, pv) in enumerate(zip(genes_to_plot, log2fcs, pvals_d)):
        stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        offset = 0.3 if lfc > 0 else -0.3
        ax_e.text(lfc + offset, i, stars, ha="left" if lfc > 0 else "right",
                  va="center", fontsize=ANNOT_SIZE, color="#555555")

    ax_e.set_yticks(y_pos_d)
    ax_e.set_yticklabels([SCRNA_LABELS.get(g, g) for g in genes_to_plot], fontsize=TICK_SIZE)
    ax_e.axvline(0, color="black", linewidth=0.8)
    ax_e.set_xlabel("log$_2$ fold-change (S100A9-high / S100A9-low)", fontsize=LABEL_SIZE)
    ax_e.set_title(f"scRNA-seq validation\n"
                   f"(Han 2022, n={scrna['n_hi']} vs {scrna['n_lo']})",
                   fontsize=TITLE_SIZE, fontweight="bold")
    ax_e.invert_yaxis()

    # Separator between calprotectin and IMC-concordance genes
    ax_e.axhline(y=7.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_e.text(max(log2fcs) * 0.8, 5.5, "Calprotectin\nprogram",
              fontsize=ANNOT_SIZE, color="gray", ha="center", style="italic")
    ax_e.text(max(log2fcs) * 0.8, 9.5, "IMC panel\nmarkers",
              fontsize=ANNOT_SIZE, color="gray", ha="center", style="italic")

    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    # Save
    out_path = Path(output_dir) / "fig_s100a9_myeloid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s-panel", required=True, help="S-panel v8 h5ad")
    parser.add_argument("--s-utag", required=True, help="S-panel UTAG merged h5ad")
    parser.add_argument("--t-panel", required=True, help="T-panel v8 h5ad")
    parser.add_argument("--scrna", required=True, help="Han 2022 FL scRNA-seq h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    imc = extract_panel_data(args.s_panel, "S-panel")
    utag = extract_compartment_data(args.s_utag)
    t_data = extract_panel_data(args.t_panel, "T-panel")
    scrna = extract_scrna_data(args.scrna)

    out = make_figure(imc, utag, t_data, scrna, args.output_dir)
    return out


if __name__ == "__main__":
    out = main()
    import subprocess
    subprocess.run(["open", "-a", "Preview", out])
