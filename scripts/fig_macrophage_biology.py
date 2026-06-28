"""
Figure: Myeloid ecosystem in follicular lymphoma.

Spatial organization, compartmentalization, and functional specialization
of myeloid subtypes across follicular architecture.

Panels (8):
  (a) Marker heatmap: functional markers across M1, M2, S100A9+
  (b) Compartment distribution: follicular vs interfollicular
  (c) Detailed compartment localization: 6 compartments stacked bars
  (d) Neighborhood enrichment heatmap: permutation z-scores in FDC zone (M1/M2 vs neighbors incl. CD14+/- FDC)
  (e) Spatial example: M2 Mac pocket in FDC zone (C1_FL41, inset zoom)
  (f) VISTA checkpoint expression by myeloid subtype
  (g) EP300 mutation → VISTA+ fraction by myeloid subtype
  (h) Driver bar chart: markers correlated with myeloid fraction
"""

import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
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
        transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
        va="bottom", ha="left",
    )


MAC_SUBTYPES = ["M1 Macrophages", "M2 Macrophages", "Myeloid (S100A9+)"]
MAC_SHORT = {"M1 Macrophages": "M1 Mac", "M2 Macrophages": "M2 Mac",
             "Myeloid (S100A9+)": "S100A9+"}
MAC_COLORS = {"M1 Mac": "#E41A1C", "M2 Mac": "#984EA3",
              "S100A9+": "#A65628"}
SKIP = {"DNA1", "DNA2", "HistoneH3"}

FOLL_COMPARTMENTS = {
    "B cell zone (BCL2+)", "B cell zone (PAX5+)",
    "FDC network zone", "FDC / myeloid zone",
}
INTER_COMPARTMENTS = {
    "T cell zone", "Stromal / CAF zone",
    "Other / myeloid zone", "B/T mixed zone",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_data(s_panel_path):
    print("Loading S-panel data...")
    f = h5py.File(s_panel_path, "r")
    X = f["X"][:]
    markers = [v.decode() for v in f["var"]["_index"][:]]
    cell_types = load_array(f, "cell_type")
    sample_ids = load_array(f, "sample_id")
    cx_arr = f["obs"]["centroid_x"][:]
    cy_arr = f["obs"]["centroid_y"][:]
    f.close()

    marker_idx = {m: i for i, m in enumerate(markers)}
    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS and not s.startswith("Biomax")
        for s in sample_ids
    ])
    X = X[tumor_mask]
    cell_types = cell_types[tumor_mask]
    sample_ids = sample_ids[tumor_mask]
    cx_arr = cx_arr[tumor_mask]
    cy_arr = cy_arr[tumor_mask]
    print(f"  {len(X):,} tumor cells")

    data = {"markers": markers, "marker_idx": marker_idx}

    # ── (a) Marker profiles ──
    print("  Computing marker profiles...")
    func_markers = ["VISTA", "IDO", "CCL21", "CXCL12", "CD14", "CD68",
                    "HLA_DR", "HLA_Class_I", "CD11c", "CXCL13",
                    "Ki-67", "BCL_2", "CD11b", "S100A9", "CD163", "CD206"]
    profiles = {}
    for mt in MAC_SUBTYPES:
        mask = cell_types == mt
        short = MAC_SHORT[mt]
        row = {}
        for m in func_markers:
            if m in marker_idx:
                row[m] = float(X[mask, marker_idx[m]].mean())
        profiles[short] = row
    data["profiles"] = profiles
    data["func_markers"] = func_markers

    # ── (b, d) Spatial distances ──
    print("  Computing spatial distances...")
    mac_any = np.isin(cell_types, MAC_SUBTYPES)
    mac_per_roi = Counter(sample_ids[mac_any])
    top_rois = [r for r, _ in mac_per_roi.most_common(15)]

    # M1-M2 distances
    m1m2_dists = []
    m1m1_dists = []
    m2m2_dists = []
    for roi in top_rois:
        rmask = sample_ids == roi
        roi_ct = cell_types[rmask]
        roi_cx = cx_arr[rmask]
        roi_cy = cy_arr[rmask]
        m1_idx = np.where(roi_ct == "M1 Macrophages")[0]
        m2_idx = np.where(roi_ct == "M2 Macrophages")[0]
        if len(m1_idx) < 10 or len(m2_idx) < 10:
            continue
        m1_c = np.column_stack([roi_cx[m1_idx], roi_cy[m1_idx]])
        m2_c = np.column_stack([roi_cx[m2_idx], roi_cy[m2_idx]])
        m2_tree = KDTree(m2_c)
        d, _ = m2_tree.query(m1_c, k=1)
        m1m2_dists.extend(d.tolist())
        m1_tree = KDTree(m1_c)
        if len(m1_idx) > 1:
            d, _ = m1_tree.query(m1_c, k=2)
            m1m1_dists.extend(d[:, 1].tolist())
        if len(m2_idx) > 1:
            m2_tree2 = KDTree(m2_c)
            d, _ = m2_tree2.query(m2_c, k=2)
            m2m2_dists.extend(d[:, 1].tolist())

    data["m1m2_dists"] = np.array(m1m2_dists)
    data["m1m1_dists"] = np.array(m1m1_dists)
    data["m2m2_dists"] = np.array(m2m2_dists)

    # ── (d) Mac-lymphocyte distances ──
    print("  Computing mac-lymphocyte distances...")
    lymph_types = ["CD8 T cells", "CD4 T cells", "B cells (BCL2+)", "FDC"]
    mac_lymph_dists = {}
    for mt in MAC_SUBTYPES:
        short = MAC_SHORT[mt]
        mac_lymph_dists[short] = {}
        for lt in lymph_types:
            dists = []
            for roi in top_rois:
                rmask = sample_ids == roi
                roi_ct = cell_types[rmask]
                roi_cx = cx_arr[rmask]
                roi_cy = cy_arr[rmask]
                mt_local = np.where(roi_ct == mt)[0]
                lt_local = np.where(roi_ct == lt)[0]
                if len(mt_local) < 10 or len(lt_local) < 5:
                    continue
                lt_tree = KDTree(np.column_stack([roi_cx[lt_local], roi_cy[lt_local]]))
                mt_coords = np.column_stack([roi_cx[mt_local], roi_cy[mt_local]])
                d, _ = lt_tree.query(mt_coords, k=1)
                dists.extend(d.tolist())
            mac_lymph_dists[short][lt] = np.array(dists) if dists else np.array([])
    data["mac_lymph_dists"] = mac_lymph_dists

    # ── (e) EP300 mutation → VISTA on myeloid subtypes ──
    print("  Computing EP300 → VISTA by myeloid subtype...")
    from src.clinical_linkage import normalize_sample_id
    import pandas as pd
    clinical = pd.read_csv("output/cd14_validation/master_clinical_ezh2.csv")
    clinical["FL_base"] = clinical["Sample_ID"].str.extract(r"(FL\d+)")[0]
    slide_to_flbase = dict(zip(clinical["slide_ID"], clinical["FL_base"]))

    mut_df = pd.read_csv("data/fl_genetics/FL_mutations_basics.csv")
    mut_df["FL_base"] = mut_df["Tumor_Sample_Barcode"].str.extract(r"(FL\d+)")[0]
    ep300_patients = set(mut_df[mut_df["Hugo_Symbol"] == "EP300"]["FL_base"])

    vista_idx = marker_idx.get("VISTA")
    ep300_vista = {}  # {subtype: {"wt": [...], "mut": [...]}}
    myeloid_plus = list(MAC_SUBTYPES)
    for mt in myeloid_plus:
        ep300_vista[mt] = {"wt": [], "mut": []}
    for roi in np.unique(sample_ids):
        sid = normalize_sample_id(roi)
        fb = slide_to_flbase.get(sid)
        if fb is None:
            continue
        ep300_status = "mut" if fb in ep300_patients else "wt"
        rmask = sample_ids == roi
        roi_ct = cell_types[rmask]
        roi_vista = X[rmask, vista_idx] if vista_idx is not None else None
        if roi_vista is None:
            continue
        for mt in myeloid_plus:
            mt_mask = roi_ct == mt
            if mt_mask.sum() >= 3:
                frac_pos = float((roi_vista[mt_mask] > 0.5).mean())
                ep300_vista[mt][ep300_status].append(frac_pos)
    data["ep300_vista"] = ep300_vista

    # ── (g) Per-ROI macrophage fraction vs BCL2+ B ──
    print("  Computing per-ROI fractions...")
    roi_mac_frac = {}
    roi_bcl2_frac = {}
    roi_cd8_frac = {}
    roi_marker_corr = {}
    for roi in np.unique(sample_ids):
        rmask = sample_ids == roi
        n = rmask.sum()
        if n < 100:
            continue
        roi_ct = cell_types[rmask]
        roi_mac_frac[roi] = np.isin(roi_ct, MAC_SUBTYPES).mean()
        roi_bcl2_frac[roi] = (roi_ct == "B cells (BCL2+)").mean()
        roi_cd8_frac[roi] = (roi_ct == "CD8 T cells").mean()

    common_r = sorted(roi_mac_frac.keys())
    mac_arr = np.array([roi_mac_frac[r] for r in common_r])
    bcl2_arr = np.array([roi_bcl2_frac[r] for r in common_r])
    cd8_arr = np.array([roi_cd8_frac[r] for r in common_r])
    data["mac_arr"] = mac_arr
    data["bcl2_arr"] = bcl2_arr
    data["cd8_arr"] = cd8_arr

    # (Compartment data extracted separately via extract_compartment_data)

    # ── (g) VISTA expression per subtype (pooled across all cells) ──
    print("  Computing VISTA expression per subtype...")
    vista_idx = marker_idx.get("VISTA")
    vista_by_subtype = {}
    if vista_idx is not None:
        for mt in MAC_SUBTYPES:
            short = MAC_SHORT[mt]
            mask = cell_types == mt
            if mask.sum() > 0:
                vista_by_subtype[short] = X[mask, vista_idx]
    data["vista_by_subtype"] = vista_by_subtype

    # ── (h) Marker correlations with macrophage fraction ──
    all_func = [m for m in markers if m not in SKIP]
    corrs = []
    for m in all_func:
        vals = np.array([float(X[sample_ids == r, marker_idx[m]].mean()) for r in common_r])
        rho, p = stats.spearmanr(mac_arr, vals)
        corrs.append((m, rho, p))
    corrs.sort(key=lambda x: -abs(x[1]))
    data["marker_corrs"] = corrs[:12]

    return data


def extract_compartment_data(s_utag_path):
    """Load UTAG compartment data for myeloid compartmentalization."""
    print("Loading UTAG compartment data...")
    f = h5py.File(s_utag_path, "r")
    ct = load_array(f, "cell_type")
    comp = load_array(f, "compartment_name")
    sids = load_array(f, "sample_id")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()

    tumor = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS and not s.startswith("Biomax")
        for s in sids
    ])
    ct = ct[tumor]
    comp = comp[tumor]
    sids = sids[tumor]
    cx = cx[tumor]
    cy = cy[tumor]
    print(f"  {len(ct):,} tumor cells with compartment labels")

    data = {}

    # ── (b) Per-subtype compartment distribution ──
    print("  Computing compartment distribution...")
    types_for_comp = MAC_SUBTYPES
    type_shorts = dict(MAC_SHORT)
    comp_dist = {}
    for mt in types_for_comp:
        mask = ct == mt
        n = int(mask.sum())
        if n == 0:
            continue
        comps = comp[mask]
        n_foll = int(np.sum(np.isin(comps, list(FOLL_COMPARTMENTS))))
        n_inter = int(np.sum(np.isin(comps, list(INTER_COMPARTMENTS))))
        n_mixed = n - n_foll - n_inter
        short = type_shorts.get(mt, mt)
        comp_dist[short] = {
            "n": n,
            "pct_foll": 100 * n_foll / n,
            "pct_inter": 100 * n_inter / n,
            "pct_mixed": 100 * n_mixed / n,
        }
        print(f"    {short:20s} n={n:>8,d}  "
              f"foll={100*n_foll/n:.1f}%  inter={100*n_inter/n:.1f}%")
    data["comp_dist"] = comp_dist

    # ── (c) Detailed per-compartment distribution ──
    print("  Computing detailed compartment distribution...")
    KEY_COMPS = ["FDC network zone", "B cell zone (BCL2+)", "T cell zone",
                 "Other / myeloid zone", "B/T mixed zone", "Stromal / CAF zone"]
    comp_detail = {}
    for mt in MAC_SUBTYPES:
        n = int((ct == mt).sum())
        if n == 0:
            continue
        short = MAC_SHORT[mt]
        pcts = {}
        for c_name in KEY_COMPS:
            n_c = int(((ct == mt) & (comp == c_name)).sum())
            pcts[c_name] = 100 * n_c / n
        pcts["Other"] = 100 - sum(pcts.values())
        comp_detail[short] = pcts
        print(f"    {short:15s}  FDC zone={pcts['FDC network zone']:.1f}%  "
              f"T zone={pcts['T cell zone']:.1f}%")
    data["comp_detail"] = comp_detail
    data["KEY_COMPS"] = KEY_COMPS

    # ── (d) Permutation enrichment in FDC network zone ──
    print("  Computing permutation enrichment in FDC zone...")
    fdc_zone_mask = comp == "FDC network zone"

    # Subdivide FDCs by CD14
    # Need CD14 from the s-panel h5ad — load it
    s_f = h5py.File(s_utag_path, "r")
    s_markers = [v.decode() for v in s_f["var"]["_index"][:]]
    cd14_col = s_markers.index("CD14")
    cd14_vals = s_f["X"][:, cd14_col]
    s_f.close()
    cd14_vals = cd14_vals[tumor]

    fdc_ct_mask = ct == "FDC"
    cd14_p75 = float(np.percentile(cd14_vals[fdc_ct_mask & fdc_zone_mask], 75))
    print(f"    CD14 p75 threshold for FDC split: {cd14_p75:.3f}")

    ct_sub = ct.copy()
    ct_sub[(ct == "FDC") & (cd14_vals >= cd14_p75)] = "FDC (CD14+)"
    ct_sub[(ct == "FDC") & (cd14_vals < cd14_p75)] = "FDC (CD14-)"

    query_types = ["M1 Macrophages", "M2 Macrophages"]
    neighbor_types = ["FDC (CD14+)", "FDC (CD14-)", "B cells (BCL2+)",
                      "B cells (PAX5+)", "CD4 T cells", "CD8 T cells",
                      "M1 Macrophages", "M2 Macrophages",
                      "Myeloid (S100A9+)", "Macrophages"]
    K_PERM = 10
    N_PERM = 500

    perm_results = {}
    for qt in query_types:
        from collections import defaultdict
        obs_counts = defaultdict(int)
        total_neighbors = 0
        null_counts = defaultdict(list)

        for roi in np.unique(sids):
            roi_fdc = (sids == roi) & fdc_zone_mask
            n_cells = roi_fdc.sum()
            if n_cells < 50:
                continue
            ct_roi = ct_sub[roi_fdc]
            query_mask = ct_roi == qt
            n_query = int(query_mask.sum())
            if n_query < 3:
                continue

            coords = np.column_stack([cx[roi_fdc], cy[roi_fdc]])
            tree = KDTree(coords)
            query_idx = np.where(query_mask)[0]
            _, neighbors = tree.query(coords[query_idx], k=K_PERM + 1)
            neigh_idx = neighbors[:, 1:]
            neigh_ct = ct_roi[neigh_idx.flatten()].reshape(neigh_idx.shape)

            for nt in neighbor_types:
                obs_counts[nt] += int((neigh_ct == nt).sum())
            total_neighbors += neigh_ct.size

            for p in range(N_PERM):
                perm_ct = np.random.permutation(ct_roi)
                perm_neigh = perm_ct[neigh_idx.flatten()].reshape(neigh_idx.shape)
                for nt in neighbor_types:
                    if p >= len(null_counts[nt]):
                        null_counts[nt].append(0)
                    null_counts[nt][p] += int((perm_neigh == nt).sum())

        qt_short = MAC_SHORT.get(qt, qt)
        res = {}
        for nt in neighbor_types:
            obs_frac = obs_counts[nt] / total_neighbors if total_neighbors else 0
            null_arr = np.array(null_counts[nt]) / total_neighbors if total_neighbors else np.zeros(1)
            null_mean = float(null_arr.mean())
            null_std = float(null_arr.std())
            z = (obs_frac - null_mean) / null_std if null_std > 0 else 0.0
            res[nt] = {"obs": obs_frac, "null": null_mean, "z": float(z)}
        perm_results[qt_short] = res
        # Print self-enrichment z-score
        self_key = qt  # e.g. "M1 Macrophages"
        self_z = res.get(self_key, {}).get("z", 0)
        print(f"    {qt_short}: {total_neighbors:,} neighbors, self z={self_z:+.1f}")

    data["perm_results"] = perm_results
    data["perm_neighbor_types"] = neighbor_types

    # ── (f) Compartment-specific myeloid–lymphocyte distances ──
    print("  Computing compartment-specific distances...")
    target_types = ["B cells (BCL2+)", "CD8 T cells"]
    target_short = {"B cells (BCL2+)": "BCL2+ B", "CD8 T cells": "CD8 T"}
    unique_rois = np.unique(sids)

    comp_interactions = {}
    for mt in MAC_SUBTYPES:
        short = MAC_SHORT[mt]
        comp_interactions[short] = {}
        for zone_name, zone_comps in [("Follicular", FOLL_COMPARTMENTS),
                                       ("Interfollicular", INTER_COMPARTMENTS)]:
            zone_mask = np.isin(comp, list(zone_comps))
            comp_interactions[short][zone_name] = {}
            for target in target_types:
                dists = []
                for roi in unique_rois:
                    rmask = (sids == roi) & zone_mask
                    if rmask.sum() < 20:
                        continue
                    roi_ct = ct[rmask]
                    roi_cx = cx[rmask]
                    roi_cy = cy[rmask]
                    mt_idx = np.where(roi_ct == mt)[0]
                    tgt_idx = np.where(roi_ct == target)[0]
                    if len(mt_idx) < 5 or len(tgt_idx) < 3:
                        continue
                    tgt_tree = KDTree(
                        np.column_stack([roi_cx[tgt_idx], roi_cy[tgt_idx]])
                    )
                    mt_coords = np.column_stack([roi_cx[mt_idx], roi_cy[mt_idx]])
                    d, _ = tgt_tree.query(mt_coords, k=1)
                    dists.extend(d.tolist())
                ts = target_short[target]
                arr = np.array(dists) if dists else np.array([])
                comp_interactions[short][zone_name][ts] = arr
                if len(arr) > 0:
                    print(f"    {short:15s} {zone_name:16s} → {ts:8s}  "
                          f"n={len(arr):>6,d}  med={np.median(arr):.1f}μm")
    data["comp_interactions"] = comp_interactions

    # ── (e) Spatial example: M2 pocket in FDC zone (C1_FL41) ──
    print("  Extracting spatial example (C1_FL41 FDC zone)...")
    example_roi = "C1_FL41"
    roi_mask = (sids == example_roi) & fdc_zone_mask
    data["example_roi"] = example_roi
    data["example_x"] = cx[roi_mask]
    data["example_y"] = cy[roi_mask]
    data["example_ct"] = ct[roi_mask]
    data["example_zoom_center"] = (394, 727)
    data["example_zoom_half"] = 100
    print(f"    {roi_mask.sum()} FDC zone cells in {example_roi}")

    return data


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(data, output_dir):
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35,
                  left=0.10, right=0.95, top=0.97, bottom=0.03,
                  height_ratios=[1.0, 1.0, 1.0, 1.0])

    subtypes_plot = ["S100A9+", "M1 Mac", "M2 Mac"]

    # ── (a) Marker heatmap ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    profiles = data["profiles"]
    fm = data["func_markers"]
    mat = np.array([[profiles[st].get(m, 0) for m in fm] for st in subtypes_plot])

    display_fm = [m.replace("HLA_Class_I", "HLA-I").replace("HLA_DR", "HLA-DR")
                  .replace("BCL_2", "BCL-2").replace("Ki-67", "Ki-67") for m in fm]
    im = ax_a.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=3.0)
    ax_a.set_xticks(range(len(fm)))
    ax_a.set_xticklabels(display_fm, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax_a.set_yticks(range(len(subtypes_plot)))
    ax_a.set_yticklabels(subtypes_plot, fontsize=TICK_SIZE)
    ax_a.set_title("Functional marker expression by myeloid subtype", fontsize=TITLE_SIZE)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cb.set_label("Mean scaled intensity", fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)

    # Annotate values
    for i in range(len(subtypes_plot)):
        for j in range(len(fm)):
            v = mat[i, j]
            color = "white" if abs(v) > 1.5 else "black"
            ax_a.text(j, i, f"{v:.1f}", ha="center", va="center",
                      fontsize=10, color=color)

    # ── (b) Compartment distribution by myeloid subtype ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    comp_dist = data.get("comp_dist", {})
    if comp_dist:
        sub_order_b = [s for s in subtypes_plot if s in comp_dist]
        y_b = np.arange(len(sub_order_b))
        bar_h = 0.25

        foll_pcts = [comp_dist[s]["pct_foll"] for s in sub_order_b]
        inter_pcts = [comp_dist[s]["pct_inter"] for s in sub_order_b]
        mixed_pcts = [comp_dist[s]["pct_mixed"] for s in sub_order_b]

        ax_b.barh(y_b - bar_h, foll_pcts, bar_h, label="Follicular",
                  color="#4DAF4A", edgecolor="white", alpha=0.8)
        ax_b.barh(y_b, inter_pcts, bar_h, label="Interfollicular",
                  color="#377EB8", edgecolor="white", alpha=0.8)
        ax_b.barh(y_b + bar_h, mixed_pcts, bar_h, label="Mixed/other",
                  color="#D3D3D3", edgecolor="white", alpha=0.6)

        for i, s in enumerate(sub_order_b):
            n = comp_dist[s]["n"]
            max_pct = max(foll_pcts[i], inter_pcts[i], mixed_pcts[i])
            ax_b.text(max_pct + 2, i, f"n={n:,}", va="center",
                      fontsize=10, color="#555555")

        ax_b.set_yticks(y_b)
        ax_b.set_yticklabels(sub_order_b, fontsize=TICK_SIZE)
        ax_b.set_xlabel("% of cells", fontsize=LABEL_SIZE)
        ax_b.tick_params(axis="x", labelsize=TICK_SIZE)
        ax_b.set_title("Compartment distribution by myeloid subtype", fontsize=TITLE_SIZE)
        ax_b.legend(fontsize=LEGEND_SIZE, loc="lower right")
        # Extend x-axis so n= labels don't collide with right edge
        cur_xmax = ax_b.get_xlim()[1]
        ax_b.set_xlim(0, cur_xmax * 1.18)

    # ── (c) Detailed compartment distribution (6 specific compartments) ──
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")

    comp_detail = data.get("comp_detail", {})
    key_comps = data.get("KEY_COMPS", [])
    COMP_COLORS = {
        "FDC network zone": "#FF7F00", "B cell zone (BCL2+)": "#4DAF4A",
        "T cell zone": "#E41A1C", "Other / myeloid zone": "#984EA3",
        "B/T mixed zone": "#377EB8", "Stromal / CAF zone": "#A65628",
        "Other": "#D3D3D3",
    }
    if comp_detail:
        y_c = np.arange(len(subtypes_plot))
        all_comps = key_comps + ["Other"]
        for st_i, st in enumerate(subtypes_plot):
            if st not in comp_detail:
                continue
            left = 0
            for c_name in all_comps:
                pct = comp_detail[st].get(c_name, 0)
                ax_c.barh(st_i, pct, left=left, height=0.45,
                          color=COMP_COLORS.get(c_name, "#D3D3D3"),
                          edgecolor="white", linewidth=0.3)
                if pct > 8:
                    ax_c.text(left + pct / 2, st_i, f"{pct:.0f}%",
                              va="center", ha="center", fontsize=10,
                              color="white" if pct > 15 else "black")
                left += pct
        ax_c.set_yticks(y_c)
        ax_c.set_yticklabels(subtypes_plot, fontsize=TICK_SIZE)
        ax_c.set_xlabel("% of cells", fontsize=LABEL_SIZE)
        ax_c.tick_params(axis="x", labelsize=TICK_SIZE)
        ax_c.set_title("Specific compartment localization", fontsize=TITLE_SIZE)
        ax_c.set_xlim(0, 100)
        # Expand y-limits to compress bars vertically; legend goes below x-axis
        ax_c.set_ylim(len(subtypes_plot) - 0.3, -1.0)
        ax_c.legend(
            handles=[Patch(color=COMP_COLORS[c],
                           label=c.replace(" zone", "").replace("B cell (BCL2+)", "BCL2+ B"))
                     for c in all_comps],
            fontsize=LEGEND_SIZE, loc="upper center",
            bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False,
        )
    for sp in ["top", "right"]:
        ax_c.spines[sp].set_visible(False)

    # ── (d) Permutation enrichment heatmap (FDC network zone) ──
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")

    perm_results = data["perm_results"]
    perm_neighbor_types = data["perm_neighbor_types"]
    query_order = ["M1 Mac", "M2 Mac"]
    nbr_short = {
        "FDC (CD14+)": "FDC\n(CD14+)", "FDC (CD14-)": "FDC\n(CD14−)",
        "B cells (BCL2+)": "BCL2+\nB", "B cells (PAX5+)": "PAX5+\nB",
        "CD4 T cells": "CD4 T", "CD8 T cells": "CD8 T",
        "M1 Macrophages": "M1\nMac", "M2 Macrophages": "M2\nMac",
        "Myeloid (S100A9+)": "S100A9+", "Macrophages": "Mac\n(generic)",
    }

    # Build z-score matrix
    z_matrix = np.zeros((len(query_order), len(perm_neighbor_types)))
    for qi, qt in enumerate(query_order):
        for ni, nt in enumerate(perm_neighbor_types):
            z_matrix[qi, ni] = perm_results[qt][nt]["z"]

    # Plot heatmap — colorbar capped at ±50 to keep off-diagonal values readable
    # (self-enrichment z-scores reach +83/+122 and would flatten the rest)
    vmax = 50
    im = ax_d.imshow(z_matrix, cmap="RdBu_r", aspect="auto",
                     vmin=-vmax, vmax=vmax)

    # Annotate cells with z-scores
    for qi in range(len(query_order)):
        for ni in range(len(perm_neighbor_types)):
            z = z_matrix[qi, ni]
            txt = f"{z:+.0f}" if abs(z) >= 1 else f"{z:+.1f}"
            color = "white" if abs(z) > vmax * 0.6 else "black"
            ax_d.text(ni, qi, txt, ha="center", va="center",
                      fontsize=11, fontweight="bold" if abs(z) > 10 else "normal",
                      color=color)

    ax_d.set_xticks(range(len(perm_neighbor_types)))
    ax_d.set_xticklabels([nbr_short.get(nt, nt) for nt in perm_neighbor_types],
                         fontsize=TICK_SIZE, rotation=45, ha="right", rotation_mode="anchor")
    ax_d.set_yticks(range(len(query_order)))
    ax_d.set_yticklabels(query_order, fontsize=TICK_SIZE)
    ax_d.set_title("Neighborhood enrichment in FDC zone\n(permutation z-scores, K=10)",
                   fontsize=TITLE_SIZE)

    cbar_d = plt.colorbar(im, ax=ax_d, shrink=0.8, pad=0.02)
    cbar_d.set_label("Z-score", fontsize=LABEL_SIZE)
    cbar_d.ax.tick_params(labelsize=TICK_SIZE)

    # ── (e) M2 Mac pocket spatial example (C1_FL41, FDC zone only) ──
    ax_e = fig.add_subplot(gs[2, 0])
    panel_label(ax_e, "e")

    ex_x = data["example_x"]
    ex_y = data["example_y"]
    ex_ct = data["example_ct"]
    ex_cx, ex_cy = data["example_zoom_center"]
    ex_half = data["example_zoom_half"]
    ex_roi = data["example_roi"]

    # Cell colors — shared with fig_m2_mac_fdc_zone.py (S10)
    # S100A9+ omitted: Fig 5e focuses on M2 Mac vs FDC spatial relationship
    CELL_COLORS = {
        "FDC": "#2ecc71", "B cells (BCL2+)": "#85c1e9",
        "B cells (PAX5+)": "#5dade2", "CD8 T cells": "#8e44ad",
        "CD4 T cells": "#3498db", "M1 Macrophages": "#e67e22",
        "M2 Macrophages": "#e74c3c",
        "Macrophages": "#d35400", "Dendritic cells": "#1abc9c",
    }
    highlight = list(CELL_COLORS.keys())
    # Standard sizes — match fig_m2_mac_fdc_zone.py
    sz_roi = 3
    sz_myeloid = 7
    sz_m2 = 18
    sz_gray = 1.5

    # Background: non-highlighted
    other = ~np.isin(ex_ct, highlight)
    ax_e.scatter(ex_x[other], ex_y[other], c="#D3D3D3", s=sz_gray,
                 alpha=0.3, zorder=0, rasterized=True)
    # Non-myeloid
    for ctype in ["FDC", "B cells (BCL2+)", "B cells (PAX5+)",
                  "CD8 T cells", "CD4 T cells"]:
        mask = ex_ct == ctype
        if mask.sum() == 0:
            continue
        ax_e.scatter(ex_x[mask], ex_y[mask], c=CELL_COLORS[ctype],
                     s=sz_roi, alpha=0.5, zorder=1, rasterized=True)
    # Myeloid (non-M2)
    for ctype in ["Macrophages", "M1 Macrophages"]:
        mask = ex_ct == ctype
        if mask.sum() == 0:
            continue
        ax_e.scatter(ex_x[mask], ex_y[mask], c=CELL_COLORS[ctype],
                     s=sz_myeloid, alpha=0.7, zorder=2, rasterized=True)
    # M2 Mac stars
    m2 = ex_ct == "M2 Macrophages"
    ax_e.scatter(ex_x[m2], ex_y[m2], c="#e74c3c", s=sz_m2,
                 alpha=0.9, zorder=3, marker="*", edgecolors="black",
                 linewidths=0.3, rasterized=True)

    ax_e.set_xlim(ex_x.min() - 5, ex_x.max() + 5)
    ax_e.set_ylim(ex_y.min() - 5, ex_y.max() + 5)
    ax_e.invert_yaxis()
    ax_e.set_aspect("equal")
    ax_e.set_title(f"{ex_roi} — FDC network zone", fontsize=TITLE_SIZE)
    ax_e.set_xlabel("x (\u00b5m)", fontsize=LABEL_SIZE)
    ax_e.set_ylabel("y (\u00b5m)", fontsize=LABEL_SIZE)
    ax_e.tick_params(labelsize=TICK_SIZE)
    for sp in ["top", "right"]:
        ax_e.spines[sp].set_visible(False)

    # Zoom rectangle
    from matplotlib.patches import Rectangle, ConnectionPatch
    rect = Rectangle((ex_cx - ex_half, ex_cy - ex_half), 2 * ex_half, 2 * ex_half,
                      linewidth=2, edgecolor="white", facecolor="none", zorder=10)
    ax_e.add_patch(rect)
    rect2 = Rectangle((ex_cx - ex_half, ex_cy - ex_half), 2 * ex_half, 2 * ex_half,
                       linewidth=1.5, edgecolor="black", facecolor="none",
                       linestyle="--", zorder=11)
    ax_e.add_patch(rect2)

    # Inset
    ax_ins = ax_e.inset_axes([0.55, 0.02, 0.44, 0.44])
    in_zoom = ((ex_x >= ex_cx - ex_half) & (ex_x <= ex_cx + ex_half) &
               (ex_y >= ex_cy - ex_half) & (ex_y <= ex_cy + ex_half))
    zx, zy, zct = ex_x[in_zoom], ex_y[in_zoom], ex_ct[in_zoom]
    # Standard inset sizes — match fig_m2_mac_fdc_zone.py
    sz_ins = 30
    sz_ins_myeloid = 75
    sz_ins_m2 = 150

    other_z = ~np.isin(zct, highlight)
    ax_ins.scatter(zx[other_z], zy[other_z], c="#D3D3D3", s=sz_ins * 0.5,
                   alpha=0.3, zorder=0, rasterized=True)
    for ctype in ["FDC", "B cells (BCL2+)", "B cells (PAX5+)",
                  "CD8 T cells", "CD4 T cells"]:
        mask = zct == ctype
        if mask.sum() > 0:
            ax_ins.scatter(zx[mask], zy[mask], c=CELL_COLORS[ctype],
                           s=sz_ins, alpha=0.5, zorder=1, rasterized=True)
    for ctype in ["Macrophages", "M1 Macrophages"]:
        mask = zct == ctype
        if mask.sum() > 0:
            ax_ins.scatter(zx[mask], zy[mask], c=CELL_COLORS[ctype],
                           s=sz_ins_myeloid, alpha=0.7, zorder=2, rasterized=True)
    m2_z = zct == "M2 Macrophages"
    ax_ins.scatter(zx[m2_z], zy[m2_z], c="#e74c3c", s=sz_ins_m2,
                   alpha=0.9, zorder=3, marker="*", edgecolors="black",
                   linewidths=0.5, rasterized=True)

    ax_ins.set_xlim(ex_cx - ex_half, ex_cx + ex_half)
    ax_ins.set_ylim(ex_cy + ex_half, ex_cy - ex_half)
    ax_ins.set_aspect("equal")
    ax_ins.tick_params(labelsize=TICK_SIZE)
    ax_ins.set_xlabel("x (\u00b5m)", fontsize=9)
    ax_ins.set_ylabel("y (\u00b5m)", fontsize=9)
    for sp in ax_ins.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)

    # Connection lines
    con1 = ConnectionPatch(
        xyA=(ex_cx + ex_half, ex_cy + ex_half), coordsA=ax_e.transData,
        xyB=(0, 0), coordsB=ax_ins.transAxes,
        color="black", linewidth=1, linestyle="--", alpha=0.5)
    fig.add_artist(con1)
    con2 = ConnectionPatch(
        xyA=(ex_cx + ex_half, ex_cy - ex_half), coordsA=ax_e.transData,
        xyB=(0, 1), coordsB=ax_ins.transAxes,
        color="black", linewidth=1, linestyle="--", alpha=0.5)
    fig.add_artist(con2)

    # Legend (S100A9+ omitted — panel focuses on M2 Mac vs FDC)
    leg_items = [("FDC", "o"), ("CD8 T", "o"), ("CD4 T", "o"),
                 ("M1 Mac", "o"), ("M2 Mac", "*")]
    leg_colors = ["#2ecc71", "#8e44ad", "#3498db", "#e67e22", "#e74c3c"]
    leg_handles = []
    for (lab, mk), col in zip(leg_items, leg_colors):
        ec = "black" if mk == "*" else "none"
        leg_handles.append(plt.Line2D([0], [0], marker=mk, color="w",
                                       markerfacecolor=col, markersize=8,
                                       markeredgecolor=ec, label=lab))
    leg_handles.append(Patch(facecolor="#D3D3D3", alpha=0.5, label="Other"))
    ax_e.legend(handles=leg_handles, loc="upper right", fontsize=LEGEND_SIZE, framealpha=0.9)

    # ── (f) VISTA checkpoint expression by myeloid subtype ──
    ax_f = fig.add_subplot(gs[2, 1])
    panel_label(ax_f, "f")
    vista_data = data.get("vista_by_subtype", {})
    if vista_data:
        vista_order = ["S100A9+", "M1 Mac", "M2 Mac"]
        vista_vals = [vista_data.get(st, np.array([])) for st in vista_order]
        positions = list(range(len(vista_order)))

        bp = ax_f.boxplot(
            vista_vals, positions=positions, widths=0.5,
            patch_artist=True, showfliers=False,
        )
        for patch, label in zip(bp["boxes"], vista_order):
            patch.set_facecolor(MAC_COLORS.get(label, "#999999"))
            patch.set_alpha(0.7)

        for i, (st, vals) in enumerate(zip(vista_order, vista_vals)):
            n = len(vals)
            med = float(np.median(vals)) if len(vals) > 0 else 0
            ax_f.text(i, med + 0.05, f"n={n:,}", ha="center", fontsize=10,
                      color="#555555")

        ax_f.set_xticks(positions)
        ax_f.set_xticklabels(vista_order, fontsize=TICK_SIZE)
        ax_f.set_ylabel("VISTA expression (scaled)", fontsize=LABEL_SIZE)
        ax_f.tick_params(axis="y", labelsize=TICK_SIZE)
        ax_f.set_title("VISTA checkpoint by myeloid subtype", fontsize=TITLE_SIZE)
        ax_f.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # ── (g) EP300 mutation → VISTA+ fraction by myeloid subtype ──
    ax_g = fig.add_subplot(gs[3, 0])
    panel_label(ax_g, "g")

    ep300_vista = data["ep300_vista"]
    plot_types = ["M2 Macrophages", "M1 Macrophages", "Myeloid (S100A9+)"]
    short_labels = {"M2 Macrophages": "M2 Mac", "M1 Macrophages": "M1 Mac",
                    "Myeloid (S100A9+)": "S100A9+"}
    x_pos = np.arange(len(plot_types))
    width = 0.35

    wt_means, mut_means, wt_sems, mut_sems = [], [], [], []
    pvals = []
    for mt in plot_types:
        wt_vals = np.array(ep300_vista[mt]["wt"])
        mut_vals = np.array(ep300_vista[mt]["mut"])
        wt_means.append(np.mean(wt_vals) * 100 if len(wt_vals) > 0 else 0)
        mut_means.append(np.mean(mut_vals) * 100 if len(mut_vals) > 0 else 0)
        wt_sems.append(stats.sem(wt_vals) * 100 if len(wt_vals) > 1 else 0)
        mut_sems.append(stats.sem(mut_vals) * 100 if len(mut_vals) > 1 else 0)
        if len(wt_vals) >= 3 and len(mut_vals) >= 3:
            _, p = stats.mannwhitneyu(wt_vals, mut_vals, alternative="two-sided")
            pvals.append(p)
        else:
            pvals.append(1.0)

    ax_g.bar(x_pos - width/2, wt_means, width, yerr=wt_sems,
             color="#4DBEEE", edgecolor="black", linewidth=0.5,
             capsize=3, label=f"EP300-wt (n={len(ep300_vista[plot_types[0]]['wt'])})")
    ax_g.bar(x_pos + width/2, mut_means, width, yerr=mut_sems,
             color="#D95319", edgecolor="black", linewidth=0.5,
             capsize=3, label=f"EP300-mut (n={len(ep300_vista[plot_types[0]]['mut'])})")

    # Significance brackets
    for i, p in enumerate(pvals):
        if p < 0.1:
            y_max = max(wt_means[i] + wt_sems[i], mut_means[i] + mut_sems[i]) + 3
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "~"
            ax_g.plot([i - width/2, i - width/2, i + width/2, i + width/2],
                      [y_max, y_max + 2, y_max + 2, y_max], "k-", linewidth=0.8)
            ax_g.text(i, y_max + 2.5, f"{sig}\nP={p:.3f}", ha="center",
                      va="bottom", fontsize=10)

    ax_g.set_xticks(x_pos)
    ax_g.set_xticklabels([short_labels[mt] for mt in plot_types], fontsize=TICK_SIZE)
    ax_g.set_ylabel("VISTA+ fraction (%)", fontsize=LABEL_SIZE)
    ax_g.tick_params(axis="y", labelsize=TICK_SIZE)
    ax_g.set_title("EP300 mutation \u2192 VISTA on myeloid", fontsize=TITLE_SIZE,
                    fontweight="bold")
    ax_g.legend(fontsize=LEGEND_SIZE, loc="upper right")
    ax_g.spines["top"].set_visible(False)
    ax_g.spines["right"].set_visible(False)

    # ── (h) Marker correlates with macrophage fraction ──
    ax_h = fig.add_subplot(gs[3, 1])
    panel_label(ax_h, "h")
    mc = data["marker_corrs"]
    m_names = [x[0].replace("HLA_Class_I", "HLA-I").replace("HLA_DR", "HLA-DR") for x in mc]
    rhos = [x[1] for x in mc]
    colors_h = ["#E41A1C" if r > 0 else "#377EB8" for r in rhos]
    y_h = np.arange(len(m_names))
    ax_h.barh(y_h, rhos, color=colors_h, edgecolor="white", height=0.6)
    ax_h.axvline(0, color="black", linewidth=0.8)
    ax_h.set_yticks(y_h)
    ax_h.set_yticklabels(m_names, fontsize=TICK_SIZE)
    ax_h.set_xlabel("Spearman \u03c1 with myeloid fraction", fontsize=LABEL_SIZE)
    ax_h.tick_params(axis="x", labelsize=TICK_SIZE)
    ax_h.set_title("Markers driving myeloid-rich microenvironment", fontsize=TITLE_SIZE)
    for i, (r, p) in enumerate(zip(rhos, [x[2] for x in mc])):
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        offset = 0.02 if r > 0 else -0.02
        ha = "left" if r > 0 else "right"
        ax_h.text(r + offset, i, stars, va="center", ha=ha, fontsize=ANNOT_SIZE, color="#666666")

    # Footnote
    fig.text(
        0.50, 0.005,
        "* p < 0.05, ** p < 0.01, *** p < 0.001 (nominal, Mann-Whitney or Spearman)",
        ha="center", fontsize=10, color="#555555", style="italic",
    )

    out = Path(output_dir) / "fig_macrophage_biology.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-panel", default="output/all_TMA_S_global_v8.h5ad")
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    data = extract_data(args.s_panel)

    # Compartment data for panels (b) and (f)
    comp_data = extract_compartment_data(args.s_utag)
    data.update(comp_data)

    make_figure(data, args.output_dir)


if __name__ == "__main__":
    main()
