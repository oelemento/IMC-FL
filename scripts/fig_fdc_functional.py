"""
Figure: CD14-high FDCs as a functionally distinct, myeloid-hybrid state.

Panels:
  (a) Marker profile heatmap: CD14-high vs CD14-low FDCs by functional category
  (b) Spatial neighborhood composition differences
  (c) Per-ROI FDC CD14 vs cell type composition (driver analysis)
  (d) Representative spatial scatter: CD14-high vs CD14-low FDCs in tissue
  (e) Perivascular niche
  (f) FDC CD14 tracks with myeloid inflammation
  (g) FDC CD14 tracks with CD8 T cell infiltration
  (h) Concept cartoon (summary model)
"""

import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from PIL import Image as PILImage
from scipy import stats
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS


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


def panel_label(ax, letter, x=-0.08, y=1.05):
    ax.text(
        x, y, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top", ha="left",
    )


MYELOID_TYPES = [
    "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Myeloid (S100A9+)", "Dendritic cells", "pDC",
]
SKIP_MARKERS = {"DNA1", "DNA2", "HistoneH3"}

# Functional categories for marker grouping
MARKER_CATEGORIES = {
    "Chemokines": ["CXCL13", "CXCL12", "CCL21"],
    "Antigen\npresentation": ["HLA_Class_I", "CD11c", "IDO", "HLA_DR"],
    "Immune\ncheckpoint": ["VISTA"],
    "FDC network": ["CD21", "PDPN"],
    "Anti-apoptotic": ["BCL_2"],
    "Myeloid\nmarkers": ["CD68", "CD11b", "S100A9"],
    "Stromal": ["Vimentin", "CD146", "Fibronectin"],
}

# Nice display names for markers
MARKER_DISPLAY = {
    "HLA_Class_I": "HLA-I", "HLA_DR": "HLA-DR", "BCL_2": "BCL-2",
    "PD_L1": "PD-L1", "BCL_6": "BCL-6", "p_H3s28": "pH3S28",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_data(s_panel_path):
    """Extract all data needed for the figure."""
    print("Loading S-panel h5ad...")
    f = h5py.File(s_panel_path, "r")
    X = f["X"][:]
    markers = [v.decode() if isinstance(v, bytes) else str(v)
               for v in f["var"]["_index"][:]]
    cell_types = load_array(f, "cell_type")
    sample_ids = load_array(f, "sample_id")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()

    marker_idx = {m: i for i, m in enumerate(markers)}

    # Filter to tumor cores (excl Biomax)
    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS
        and not s.startswith("Biomax")
        for s in sample_ids
    ])
    X = X[tumor_mask]
    cell_types = cell_types[tumor_mask]
    sample_ids = sample_ids[tumor_mask]
    cx = cx[tumor_mask]
    cy = cy[tumor_mask]
    print(f"  {len(X):,} tumor cells")

    fdc_mask = cell_types == "FDC"
    cd14_col = marker_idx["CD14"]
    print(f"  {fdc_mask.sum():,} FDC cells")

    # ── Analysis 1: CD14-high vs CD14-low marker profiles ──
    print("  Computing marker profiles...")
    fdc_cd14 = X[fdc_mask, cd14_col]
    q25, q75 = np.percentile(fdc_cd14, [25, 75])
    hi_mask = fdc_cd14 >= q75
    lo_mask = fdc_cd14 <= q25

    fdc_X = X[fdc_mask]
    marker_diffs = {}
    for i, m in enumerate(markers):
        if m in SKIP_MARKERS:
            continue
        hi_vals = fdc_X[hi_mask, i]
        lo_vals = fdc_X[lo_mask, i]
        diff = float(hi_vals.mean() - lo_vals.mean())
        marker_diffs[m] = {
            "diff": diff,
            "hi_mean": float(hi_vals.mean()),
            "lo_mean": float(lo_vals.mean()),
        }

    # ── Analysis 2: Per-ROI FDC CD14 vs cell type composition ──
    print("  Computing per-ROI correlations...")
    ct_list = sorted(set(cell_types) - {"Low quality / Unassigned"})
    roi_fdc_cd14 = {}
    roi_ct_frac = {}
    for roi in np.unique(sample_ids):
        rmask = sample_ids == roi
        fdc_in_roi = rmask & fdc_mask
        if fdc_in_roi.sum() < 10:
            continue
        roi_fdc_cd14[roi] = float(X[fdc_in_roi, cd14_col].mean())
        cts = cell_types[rmask]
        ct_counts = Counter(cts)
        n_total = rmask.sum()
        for ct in ct_list:
            roi_ct_frac.setdefault(ct, {})[roi] = ct_counts.get(ct, 0) / n_total

    common_rois = sorted(roi_fdc_cd14.keys())
    fdc_cd14_arr = np.array([roi_fdc_cd14[r] for r in common_rois])

    ct_correlations = []
    for ct in ct_list:
        fracs = np.array([roi_ct_frac[ct].get(r, 0) for r in common_rois])
        rho, p = stats.spearmanr(fdc_cd14_arr, fracs)
        ct_correlations.append((ct, rho, p))
    ct_correlations.sort(key=lambda x: -abs(x[1]))

    # ── Analysis 3: Spatial neighborhoods ──
    print("  Computing spatial neighborhoods...")
    fdc_counts_per_roi = Counter(sample_ids[fdc_mask])
    candidate_rois = [(r, c) for r, c in fdc_counts_per_roi.items() if c >= 500]
    candidate_rois.sort(key=lambda x: -x[1])
    sel_rois = [r for r, c in candidate_rois[:8]]

    neighbor_types_hi = Counter()
    neighbor_types_lo = Counter()
    n_hi_total = 0
    n_lo_total = 0
    for roi in sel_rois:
        rmask = sample_ids == roi
        roi_idx = np.where(rmask)[0]
        roi_cx = cx[roi_idx]
        roi_cy = cy[roi_idx]
        roi_ct = cell_types[roi_idx]
        roi_is_fdc = roi_ct == "FDC"
        roi_cd14 = X[roi_idx, cd14_col]
        fdc_local = np.where(roi_is_fdc)[0]
        non_fdc_local = np.where(~roi_is_fdc)[0]
        if len(non_fdc_local) < 20:
            continue
        tree = KDTree(np.column_stack([roi_cx[non_fdc_local], roi_cy[non_fdc_local]]))
        fdc_coords = np.column_stack([roi_cx[fdc_local], roi_cy[fdc_local]])
        _, idxs = tree.query(fdc_coords, k=10)
        fdc_cd14_vals = roi_cd14[fdc_local]
        q25_r, q75_r = np.percentile(fdc_cd14_vals, [25, 75])
        for j in range(len(fdc_local)):
            nbr_cts = roi_ct[non_fdc_local[idxs[j]]]
            if fdc_cd14_vals[j] >= q75_r:
                neighbor_types_hi.update(nbr_cts)
                n_hi_total += 10
            elif fdc_cd14_vals[j] <= q25_r:
                neighbor_types_lo.update(nbr_cts)
                n_lo_total += 10

    all_nbr_types = sorted(set(neighbor_types_hi.keys()) | set(neighbor_types_lo.keys()))
    nbr_results = []
    for ct in all_nbr_types:
        hi_n = neighbor_types_hi.get(ct, 0)
        lo_n = neighbor_types_lo.get(ct, 0)
        hi_frac = hi_n / n_hi_total if n_hi_total > 0 else 0
        lo_frac = lo_n / n_lo_total if n_lo_total > 0 else 0
        n1, n2 = n_hi_total, n_lo_total
        p_pool = (hi_n + lo_n) / (n1 + n2) if (n1 + n2) > 0 else 0
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if 0 < p_pool < 1 else 1
        z = (hi_frac - lo_frac) / se if se > 0 else 0
        pval = 2 * stats.norm.sf(abs(z))
        nbr_results.append((ct, hi_frac, lo_frac, hi_frac - lo_frac, pval))
    nbr_results.sort(key=lambda x: x[3])  # sort by difference (negative first)

    # ── Analysis 4: Per-ROI FDC CD14 vs survival markers ──
    surv_markers = ["CD68", "S100A9", "CD8a"]
    surv_corr = {}
    for m in surv_markers:
        vals = np.array([float(X[sample_ids == r, marker_idx[m]].mean())
                         for r in common_rois])
        rho, p = stats.spearmanr(fdc_cd14_arr, vals)
        surv_corr[m] = {"rho": rho, "p": p, "vals": vals}

    # ── Representative ROI for spatial scatter ──
    # Pick ROI with both good FDC count AND good BCL2+ B cell count
    # to show the perivascular niche properly
    bcl2_counts = Counter(cell_types[cell_types == "B cells (BCL2+)"])
    roi_scores = {}
    for roi in np.unique(sample_ids):
        rmask = sample_ids == roi
        ct_roi = cell_types[rmask]
        ct_counts = Counter(ct_roi)
        n_fdc = ct_counts.get("FDC", 0)
        n_bcl2 = ct_counts.get("B cells (BCL2+)", 0)
        if n_fdc >= 200 and n_bcl2 >= 100:
            # Score: geometric mean of FDC and BCL2+ counts
            roi_scores[roi] = (n_fdc * n_bcl2) ** 0.5
    if roi_scores:
        rep_roi = max(roi_scores, key=roi_scores.get)
    else:
        rep_roi = sel_rois[0]  # fallback to most FDCs
    print(f"  Representative ROI: {rep_roi}")
    rmask = sample_ids == rep_roi
    rep_data = {
        "x": cx[rmask],
        "y": cy[rmask],
        "ct": cell_types[rmask],
        "cd14": X[rmask, cd14_col],
        "is_fdc": cell_types[rmask] == "FDC",
    }

    return {
        "marker_diffs": marker_diffs,
        "ct_correlations": ct_correlations,
        "nbr_results": nbr_results,
        "surv_corr": surv_corr,
        "fdc_cd14_arr": fdc_cd14_arr,
        "common_rois": common_rois,
        "rep_data": rep_data,
        "rep_roi": rep_roi,
        "q25": q25,
        "q75": q75,
        "n_hi": int(hi_mask.sum()),
        "n_lo": int(lo_mask.sum()),
        "markers": markers,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(data, output_dir, cartoon_path):
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2, figure=fig, hspace=0.38, wspace=0.32,
                  left=0.08, right=0.95, top=0.97, bottom=0.03,
                  height_ratios=[1.0, 1.2, 1.0, 1.0])

    # ── (a) Marker profile by functional category ──
    ax_b = fig.add_subplot(gs[0, 0])
    panel_label(ax_b, "a")
    md = data["marker_diffs"]

    # Build grouped data
    cat_names = []
    cat_diffs = []
    cat_markers_display = []
    for cat, marker_list in MARKER_CATEGORIES.items():
        for m in marker_list:
            if m in md:
                cat_names.append(cat)
                cat_diffs.append(md[m]["diff"])
                display = MARKER_DISPLAY.get(m, m)
                cat_markers_display.append(display)

    y_pos = np.arange(len(cat_markers_display))
    colors = []
    cat_color_map = {
        "Chemokines": "#E41A1C",
        "Antigen\npresentation": "#377EB8",
        "Immune\ncheckpoint": "#984EA3",
        "FDC network": "#FF7F00",
        "Anti-apoptotic": "#4DAF4A",
        "Myeloid\nmarkers": "#A65628",
        "Stromal": "#F781BF",
    }
    for cn in cat_names:
        colors.append(cat_color_map.get(cn, "#999999"))

    ax_b.barh(y_pos, cat_diffs, color=colors, edgecolor="white", height=0.65)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(cat_markers_display, fontsize=9)
    ax_b.axvline(0, color="black", linewidth=0.5)
    ax_b.set_xlabel("Mean difference (CD14-high − CD14-low FDC)")
    ax_b.set_title(
        f"Marker profile shift\n(Q75 vs Q25, n={data['n_hi']:,} per group)",
        fontsize=11,
    )
    ax_b.invert_yaxis()

    # Add category labels on right side
    prev_cat = None
    cat_start = 0
    for i, cn in enumerate(cat_names + [None]):
        if cn != prev_cat and prev_cat is not None:
            mid = (cat_start + i - 1) / 2
            ax_b.text(
                1.02, mid / len(cat_markers_display),
                prev_cat.replace("\n", " "),
                transform=ax_b.get_yaxis_transform(),
                fontsize=7, va="center", ha="left",
                color=cat_color_map.get(prev_cat, "#999999"),
                fontweight="bold",
            )
            cat_start = i
        prev_cat = cn

    # ── (b) Spatial neighborhood composition ──
    ax_c = fig.add_subplot(gs[0, 1])
    panel_label(ax_c, "b")
    nbr = data["nbr_results"]

    # Filter to significant and meaningful differences
    nbr_sig = [(ct, hf, lf, d, p) for ct, hf, lf, d, p in nbr
               if p < 0.05 and abs(d) > 0.001
               and ct != "Low quality / Unassigned"]
    if len(nbr_sig) > 12:
        nbr_sig = nbr_sig[:6] + nbr_sig[-6:]  # top/bottom 6

    ct_names_c = [x[0] for x in nbr_sig]
    diffs_c = [x[3] * 100 for x in nbr_sig]  # convert to percentage points
    pvals_c = [x[4] for x in nbr_sig]
    colors_c = ["#E41A1C" if d > 0 else "#377EB8" for d in diffs_c]

    short_names_c = []
    for n in ct_names_c:
        n = n.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+ myeloid")
        n = n.replace("Dendritic cells", "DCs").replace("B cells (PAX5+)", "B (PAX5+)")
        n = n.replace("B cells (BCL2+)", "B (BCL2+)").replace("B cells", "B")
        n = n.replace("Stromal / CAF", "Stromal/CAF").replace("CD4 T cells", "CD4 T")
        n = n.replace("CD8 T cells", "CD8 T").replace("Mixed / Border cells", "Mixed/Border")
        n = n.replace("FRC (PDPN+)", "FRC").replace("Histiocytes (CD44hi)", "Histiocytes")
        short_names_c.append(n)

    y_c = np.arange(len(short_names_c))
    ax_c.barh(y_c, diffs_c, color=colors_c, edgecolor="white", height=0.6)
    ax_c.axvline(0, color="black", linewidth=0.8)
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(short_names_c, fontsize=9)
    ax_c.set_xlabel("Difference in neighbor fraction (percentage points)")
    ax_c.set_title("Spatial neighborhood:\nCD14-high vs CD14-low FDCs (k=10 nearest)", fontsize=11)

    # Add significance stars
    for i, (d, p) in enumerate(zip(diffs_c, pvals_c)):
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
        offset = 0.05 if d > 0 else -0.05
        ha = "left" if d > 0 else "right"
        ax_c.text(d + offset, i, stars, va="center", ha=ha, fontsize=8, color="#666666")

    ax_c.legend(
        handles=[
            Patch(color="#E41A1C", label="Enriched near CD14-hi FDC"),
            Patch(color="#377EB8", label="Depleted near CD14-hi FDC"),
        ],
        fontsize=8, loc="lower right",
    )

    # ── (c) Per-ROI FDC CD14 vs cell type composition (driver analysis) ──
    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, "c")
    ct_corr = data["ct_correlations"]

    # Show top correlations (significant only)
    ct_sig = [(ct, rho, p) for ct, rho, p in ct_corr if p < 0.05]
    if len(ct_sig) > 12:
        ct_sig = ct_sig[:12]

    ct_names_d = [x[0] for x in ct_sig]
    rhos_d = [x[1] for x in ct_sig]
    pvals_d = [x[2] for x in ct_sig]

    short_names_d = []
    for n in ct_names_d:
        n = n.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+ myeloid")
        n = n.replace("Dendritic cells", "DCs").replace("B cells (PAX5+)", "B (PAX5+)")
        n = n.replace("B cells (BCL2+)", "B (BCL2+)").replace("B cells", "B")
        n = n.replace("Stromal / CAF", "Stromal/CAF").replace("CD4 T cells", "CD4 T")
        n = n.replace("CD8 T cells", "CD8 T").replace("Mixed / Border cells", "Mixed/Border")
        n = n.replace("FRC (PDPN+)", "FRC").replace("Histiocytes (CD44hi)", "Histiocytes")
        short_names_d.append(n)

    colors_d = ["#E41A1C" if r > 0 else "#377EB8" for r in rhos_d]
    y_d = np.arange(len(short_names_d))
    ax_d.barh(y_d, rhos_d, color=colors_d, edgecolor="white", height=0.6)
    ax_d.axvline(0, color="black", linewidth=0.8)
    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(short_names_d, fontsize=9)
    ax_d.set_xlabel("Spearman ρ with per-ROI FDC CD14")
    ax_d.set_title(
        f"Cell type composition drivers\n({len(data['common_rois'])} ROIs with ≥10 FDCs)",
        fontsize=11,
    )

    for i, (r, p) in enumerate(zip(rhos_d, pvals_d)):
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
        offset = 0.02 if r > 0 else -0.02
        ha = "left" if r > 0 else "right"
        ax_d.text(r + offset, i, stars, va="center", ha=ha, fontsize=8, color="#666666")

    # ── (d) Representative spatial scatter ──
    ax_e = fig.add_subplot(gs[1, 1])
    panel_label(ax_e, "d")
    rd = data["rep_data"]
    x, y = rd["x"], rd["y"]
    ct = rd["ct"]
    cd14 = rd["cd14"]
    is_fdc = rd["is_fdc"]

    # All non-FDC cells in light gray
    non_fdc = ~is_fdc
    ax_e.scatter(x[non_fdc], y[non_fdc], c="#D3D3D3", s=0.3, alpha=0.3,
                 rasterized=True, zorder=1)

    # FDC cells colored by CD14 level
    fdc_x = x[is_fdc]
    fdc_y = y[is_fdc]
    fdc_cd14 = cd14[is_fdc]
    q25_local = np.percentile(fdc_cd14, 25)
    q75_local = np.percentile(fdc_cd14, 75)

    # CD14-low FDCs
    lo = fdc_cd14 <= q25_local
    ax_e.scatter(fdc_x[lo], fdc_y[lo], c="#4393C3", s=4, alpha=0.7,
                 edgecolors="black", linewidth=0.2, label="CD14-low FDC",
                 rasterized=True, zorder=3)
    # CD14-mid FDCs
    mid = (~lo) & (fdc_cd14 < q75_local)
    ax_e.scatter(fdc_x[mid], fdc_y[mid], c="#FDDBC7", s=3, alpha=0.5,
                 edgecolors="black", linewidth=0.1, label="CD14-mid FDC",
                 rasterized=True, zorder=2)
    # CD14-high FDCs
    hi = fdc_cd14 >= q75_local
    ax_e.scatter(fdc_x[hi], fdc_y[hi], c="#FFD700", s=8, alpha=0.9,
                 edgecolors="black", linewidth=0.3, label="CD14-high FDC",
                 rasterized=True, zorder=4)

    ax_e.set_aspect("equal")
    ax_e.invert_yaxis()
    ax_e.set_title(f"FDC CD14 in tissue — {data['rep_roi']}", fontsize=11)
    ax_e.set_xlabel("x (μm)")
    ax_e.set_ylabel("y (μm)")
    ax_e.legend(fontsize=8, loc="upper right", markerscale=2)

    n_fdc_roi = is_fdc.sum()
    n_hi_roi = hi.sum()
    ax_e.text(0.02, 0.02,
              f"{n_fdc_roi:,} FDCs ({n_hi_roi:,} CD14-high)",
              transform=ax_e.transAxes, fontsize=8,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── (e) Perivascular niche ──
    ax_f = fig.add_subplot(gs[2, 0])
    panel_label(ax_f, "e")

    # Show same ROI but highlight the endothelial + BCL2+ B niche
    endo_mask = ct == "Endothelial"
    bcl2_mask = ct == "B cells (BCL2+)"
    other_mask = ~is_fdc & ~endo_mask & ~bcl2_mask

    ax_f.scatter(x[other_mask], y[other_mask], c="#D3D3D3", s=0.3, alpha=0.2,
                 rasterized=True, zorder=1)
    ax_f.scatter(x[bcl2_mask], y[bcl2_mask], c="#4393C3", s=1.5, alpha=0.5,
                 label="BCL2+ B cells", rasterized=True, zorder=2)
    ax_f.scatter(x[endo_mask], y[endo_mask], c="#E41A1C", s=4, alpha=0.8,
                 edgecolors="black", linewidth=0.2,
                 label="Endothelial", rasterized=True, zorder=3)
    # CD14-high FDCs on top
    ax_f.scatter(fdc_x[hi], fdc_y[hi], c="#FFD700", s=10, alpha=0.9,
                 edgecolors="black", linewidth=0.4,
                 label="CD14-high FDC", rasterized=True, zorder=5)

    ax_f.set_aspect("equal")
    ax_f.invert_yaxis()
    ax_f.set_title(f"Perivascular niche — {data['rep_roi']}", fontsize=11)
    ax_f.set_xlabel("x (μm)")
    ax_f.set_ylabel("y (μm)")
    ax_f.legend(fontsize=8, loc="upper right", markerscale=2)

    # ── (f) Per-ROI FDC CD14 vs CD68 (myeloid inflammation) ──
    ax_g = fig.add_subplot(gs[2, 1])
    panel_label(ax_g, "f")

    sc = data["surv_corr"]
    fdc_arr = data["fdc_cd14_arr"]
    ax_g.scatter(fdc_arr, sc["CD68"]["vals"], c="#A65628", alpha=0.5, s=25,
                 edgecolors="white", linewidth=0.3)
    m_fit, b_fit = np.polyfit(fdc_arr, sc["CD68"]["vals"], 1)
    x_range = np.linspace(fdc_arr.min(), fdc_arr.max(), 50)
    ax_g.plot(x_range, m_fit * x_range + b_fit, "r--", linewidth=1.5, alpha=0.7)
    ax_g.set_xlabel("Per-ROI mean FDC CD14")
    ax_g.set_ylabel("Per-ROI mean CD68 (all cells)")
    ax_g.set_title("FDC CD14 tracks with myeloid inflammation", fontsize=11)
    ax_g.text(0.05, 0.95,
              f"Spearman ρ={sc['CD68']['rho']:.2f}\np={sc['CD68']['p']:.1e}",
              transform=ax_g.transAxes, va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── (g) Per-ROI FDC CD14 vs CD8a ──
    ax_h = fig.add_subplot(gs[3, 0])
    panel_label(ax_h, "g")

    ax_h.scatter(fdc_arr, sc["CD8a"]["vals"], c="#984EA3", alpha=0.5, s=25,
                 edgecolors="white", linewidth=0.3)
    m_fit2, b_fit2 = np.polyfit(fdc_arr, sc["CD8a"]["vals"], 1)
    ax_h.plot(x_range, m_fit2 * x_range + b_fit2, "r--", linewidth=1.5, alpha=0.7)
    ax_h.set_xlabel("Per-ROI mean FDC CD14")
    ax_h.set_ylabel("Per-ROI mean CD8a (all cells)")
    ax_h.set_title("FDC CD14 tracks with CD8 T cell infiltration", fontsize=11)
    ax_h.text(0.05, 0.95,
              f"Spearman ρ={sc['CD8a']['rho']:.2f}\np={sc['CD8a']['p']:.1e}",
              transform=ax_h.transAxes, va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── (h) Concept cartoon ──
    ax_cartoon = fig.add_subplot(gs[3, 1])
    panel_label(ax_cartoon, "h")
    if Path(cartoon_path).exists():
        img = PILImage.open(cartoon_path)
        ax_cartoon.imshow(img)
    ax_cartoon.set_title("CD14+ FDC myeloid-hybrid model", fontsize=11)
    ax_cartoon.axis("off")

    out = Path(output_dir) / "fig_fdc_functional.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-panel", default="output/all_TMA_S_global_v8.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    parser.add_argument("--cartoon",
                        default="output/hypothesis_cartoons/fdc_cd14_myeloid_hybrid.png")
    args = parser.parse_args()

    data = extract_data(args.s_panel)
    make_figure(data, args.output_dir, args.cartoon)


if __name__ == "__main__":
    main()
