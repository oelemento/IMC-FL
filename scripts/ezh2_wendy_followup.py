#!/usr/bin/env python3
"""
EZH2 follow-up analyses per Wendy's feedback (2026-02-22).

Two analyses:
  1) Low-grade (G1+G2) EZH2 analysis: repeat TME comparison restricted to
     grades 1-2 — cell type fractions, B cell markers, compartment fractions,
     FDC features (CD14+ FDCs, CXCL13+ FDCs).
  2) Cell-cell distance by EZH2 status: measure intermeshing/proximity between
     malignant B cells and macrophages, T cells, FDCs within the FDC network
     zone. Motivated by Xi/Sean observation that macrophage distance to B cells
     is shorter in mutEZH2 DLBCL.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/ezh2_wendy_followup.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu
from scipy.spatial import cKDTree
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.clinical_linkage import normalize_sample_id

# ── Paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "output/cd14_validation"
T_PANEL = "output/all_TMA_T_global_v8.h5ad"
S_PANEL = "output/all_TMA_S_global_v8.h5ad"
S_UTAG = "output/all_TMA_S_utag_ct_merged.h5ad"
T_UTAG = "output/all_TMA_T_utag_ct_merged.h5ad"
MASTER_CSV = os.path.join(OUTPUT_DIR, "master_clinical_ezh2.csv")

C_WT = "#0072BD"
C_MUT = "#D95319"


def load_ezh2_data():
    """Load master clinical+EZH2 table with grade annotation."""
    df = pd.read_csv(MASTER_CSV)
    df["has_ezh2"] = df["EZH2"].isin(["wt", "mut"])
    df["grade_group"] = df["DIAG"].map(
        lambda x: "G1" if "FOLL1" in str(x) else
                  ("G2" if "FOLL2" in str(x) else
                   ("G3" if "FOLL3" in str(x) else "?")))
    return df


def get_bcell_mask(adata):
    ct = adata.obs["cell_type"].astype(str)
    return (ct.str.contains("B cell", case=False, na=False) |
            ct.str.contains("GC B", case=False, na=False))


def boxplot_ezh2(ax, wt_vals, mut_vals, ylabel, title):
    """Standard EZH2-wt vs mut boxplot with jittered points and p-value."""
    wt_vals = np.array(wt_vals, dtype=float)
    mut_vals = np.array(mut_vals, dtype=float)
    wt_vals = wt_vals[~np.isnan(wt_vals)]
    mut_vals = mut_vals[~np.isnan(mut_vals)]

    bp = ax.boxplot([wt_vals, mut_vals], tick_labels=["EZH2-wt", "EZH2-mut"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(C_WT); bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor(C_MUT); bp["boxes"][1].set_alpha(0.4)

    for j, vals in enumerate([wt_vals, mut_vals]):
        jitter = np.random.normal(j + 1, 0.06, len(vals))
        ax.scatter(jitter, vals, alpha=0.35, s=10,
                   color=bp["boxes"][j].get_facecolor(), edgecolors="none", zorder=2)

    if len(wt_vals) >= 3 and len(mut_vals) >= 3:
        _, pval = mannwhitneyu(wt_vals, mut_vals, alternative="two-sided")
    else:
        pval = np.nan
    star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
    ax.set_title(f"{title}\np={pval:.3g} {star}", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)
    return pval


def compute_distances_in_zone(utag, adata, ezh2_map, zone_name, target_types,
                              b_types=None):
    """
    For each ROI, within a specified UTAG zone, compute nearest-neighbor
    distances from B cells to each target cell type.

    Returns DataFrame: slide_ID, EZH2, target_type, median_distance, mean_distance, n_b, n_target
    """
    if b_types is None:
        b_types = ["B cells", "B cells (BCL2+)", "B cells (PAX5+)"]

    comp_col = "compartment_name"
    results = []

    for sid in utag.obs["slide_ID"].unique():
        ezh2 = ezh2_map.get(sid)
        if ezh2 not in ("wt", "mut"):
            continue

        # Cells in this ROI and zone
        mask = (utag.obs["slide_ID"] == sid) & (utag.obs[comp_col] == zone_name)
        if mask.sum() < 20:
            continue

        zone_obs = utag.obs[mask]
        zone_ct = zone_obs["cell_type"].values
        zone_x = zone_obs["centroid_x"].values
        zone_y = zone_obs["centroid_y"].values

        # B cells in zone
        b_mask = np.isin(zone_ct, b_types)
        if b_mask.sum() < 5:
            continue
        b_xy = np.column_stack([zone_x[b_mask], zone_y[b_mask]])

        for ttype in target_types:
            t_mask = zone_ct == ttype
            if t_mask.sum() < 3:
                continue
            t_xy = np.column_stack([zone_x[t_mask], zone_y[t_mask]])

            tree = cKDTree(t_xy)
            dists, _ = tree.query(b_xy, k=1)

            results.append({
                "slide_ID": sid,
                "EZH2": ezh2,
                "target_type": ttype,
                "median_distance": float(np.median(dists)),
                "mean_distance": float(np.mean(dists)),
                "n_b": int(b_mask.sum()),
                "n_target": int(t_mask.sum()),
            })

    return pd.DataFrame(results)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master = load_ezh2_data()
    ezh2_sub = master[master["has_ezh2"]].copy()

    # Low-grade filter
    low_grade_ids = set(
        ezh2_sub[ezh2_sub["grade_group"].isin(["G1", "G2"])]["slide_ID"])
    ezh2_map_all = dict(zip(ezh2_sub["slide_ID"], ezh2_sub["EZH2"]))
    ezh2_map_low = {k: v for k, v in ezh2_map_all.items() if k in low_grade_ids}

    n_wt_low = sum(1 for v in ezh2_map_low.values() if v == "wt")
    n_mut_low = sum(1 for v in ezh2_map_low.values() if v == "mut")
    print(f"Low grade (G1+G2): {n_wt_low} wt, {n_mut_low} mut")

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading T-panel...")
    adata_t = ad.read_h5ad(T_PANEL)
    adata_t.obs["slide_ID"] = [normalize_sample_id(s) for s in adata_t.obs["sample_id"]]

    print("Loading S-panel...")
    adata_s = ad.read_h5ad(S_PANEL)
    adata_s.obs["slide_ID"] = [normalize_sample_id(s) for s in adata_s.obs["sample_id"]]

    print("Loading S-panel UTAG...")
    utag_s = ad.read_h5ad(S_UTAG)
    utag_s.obs["slide_ID"] = [normalize_sample_id(s) for s in utag_s.obs["sample_id"]]

    print("Loading T-panel UTAG...")
    utag_t = ad.read_h5ad(T_UTAG)
    utag_t.obs["slide_ID"] = [normalize_sample_id(s) for s in utag_t.obs["sample_id"]]

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Low-grade EZH2 — TME features
    # ══════════════════════════════════════════════════════════════════
    print("\n=== ANALYSIS 1: Low-grade (G1+G2) EZH2 comparison ===")

    # 1a. Cell type fractions (T-panel, low grade only)
    t_low = adata_t[adata_t.obs["slide_ID"].isin(low_grade_ids)]
    ct_counts = t_low.obs.groupby(["slide_ID", "cell_type"]).size().unstack(fill_value=0)
    ct_frac = ct_counts.div(ct_counts.sum(axis=1), axis=0)
    ct_frac["EZH2"] = ct_frac.index.map(ezh2_map_low)
    ct_frac = ct_frac[ct_frac["EZH2"].isin(["wt", "mut"])]

    # Test each cell type
    ct_results = []
    for col in ct_frac.columns:
        if col == "EZH2":
            continue
        wt = ct_frac.loc[ct_frac["EZH2"] == "wt", col].dropna()
        mt = ct_frac.loc[ct_frac["EZH2"] == "mut", col].dropna()
        if len(wt) >= 3 and len(mt) >= 3:
            _, p = mannwhitneyu(wt, mt, alternative="two-sided")
            ct_results.append({
                "feature": col,
                "wt_mean": float(wt.mean()),
                "mut_mean": float(mt.mean()),
                "diff": float(mt.mean() - wt.mean()),
                "pval": p,
            })
    ct_results = pd.DataFrame(ct_results).sort_values("pval")
    print("\nCell type fractions (low grade, T-panel):")
    print(ct_results.to_string(index=False))

    # 1b. B cell markers (T-panel, low grade only)
    b_mask_t = get_bcell_mask(t_low)
    bcells_t_low = t_low[b_mask_t]
    bc_marker = bcells_t_low.obs.groupby("slide_ID").apply(
        lambda g: pd.Series(
            np.asarray(t_low[g.index].X.mean(axis=0)).flatten(),
            index=t_low.var_names))
    bc_marker["EZH2"] = bc_marker.index.map(ezh2_map_low)
    bc_marker = bc_marker[bc_marker["EZH2"].isin(["wt", "mut"])]

    bc_results = []
    for col in bc_marker.columns:
        if col == "EZH2":
            continue
        wt = bc_marker.loc[bc_marker["EZH2"] == "wt", col].dropna()
        mt = bc_marker.loc[bc_marker["EZH2"] == "mut", col].dropna()
        if len(wt) >= 3 and len(mt) >= 3:
            _, p = mannwhitneyu(wt, mt, alternative="two-sided")
            bc_results.append({
                "marker": col,
                "wt_mean": float(wt.mean()),
                "mut_mean": float(mt.mean()),
                "diff": float(mt.mean() - wt.mean()),
                "pval": p,
            })
    bc_results = pd.DataFrame(bc_results).sort_values("pval")
    print("\nB cell markers (low grade, T-panel):")
    print(bc_results.head(15).to_string(index=False))

    # 1c. B cell markers (S-panel, low grade only)
    s_low = adata_s[adata_s.obs["slide_ID"].isin(low_grade_ids)]
    b_mask_s = get_bcell_mask(s_low)
    bcells_s_low = s_low[b_mask_s]
    bc_marker_s = bcells_s_low.obs.groupby("slide_ID").apply(
        lambda g: pd.Series(
            np.asarray(s_low[g.index].X.mean(axis=0)).flatten(),
            index=s_low.var_names))
    bc_marker_s["EZH2"] = bc_marker_s.index.map(ezh2_map_low)
    bc_marker_s = bc_marker_s[bc_marker_s["EZH2"].isin(["wt", "mut"])]

    bc_results_s = []
    for col in bc_marker_s.columns:
        if col == "EZH2":
            continue
        wt = bc_marker_s.loc[bc_marker_s["EZH2"] == "wt", col].dropna()
        mt = bc_marker_s.loc[bc_marker_s["EZH2"] == "mut", col].dropna()
        if len(wt) >= 3 and len(mt) >= 3:
            _, p = mannwhitneyu(wt, mt, alternative="two-sided")
            bc_results_s.append({
                "marker": col,
                "wt_mean": float(wt.mean()),
                "mut_mean": float(mt.mean()),
                "diff": float(mt.mean() - wt.mean()),
                "pval": p,
            })
    bc_results_s = pd.DataFrame(bc_results_s).sort_values("pval")
    print("\nB cell markers (low grade, S-panel):")
    print(bc_results_s.head(15).to_string(index=False))

    # 1d. FDC features (S-panel UTAG, low grade only)
    utag_s_low = utag_s[utag_s.obs["slide_ID"].isin(low_grade_ids)]
    fdc_features = []
    for sid in utag_s_low.obs["slide_ID"].unique():
        ezh2 = ezh2_map_low.get(sid)
        if ezh2 not in ("wt", "mut"):
            continue
        roi = utag_s_low[utag_s_low.obs["slide_ID"] == sid]
        ct = roi.obs["cell_type"].value_counts()
        total = len(roi)
        n_fdc = ct.get("FDC", 0)

        # CD14 on FDCs
        fdc_mask = roi.obs["cell_type"] == "FDC"
        cd14_idx = list(utag_s_low.var_names).index("CD14") if "CD14" in utag_s_low.var_names else None
        cxcl13_idx = list(utag_s_low.var_names).index("CXCL13") if "CXCL13" in utag_s_low.var_names else None
        cd21_idx = list(utag_s_low.var_names).index("CD21") if "CD21" in utag_s_low.var_names else None

        if fdc_mask.sum() > 0:
            fdc_cells = roi[fdc_mask]
            cd14_mean = float(np.asarray(fdc_cells.X[:, cd14_idx].mean())) if cd14_idx is not None else np.nan
            cxcl13_mean = float(np.asarray(fdc_cells.X[:, cxcl13_idx].mean())) if cxcl13_idx is not None else np.nan
            cd21_mean = float(np.asarray(fdc_cells.X[:, cd21_idx].mean())) if cd21_idx is not None else np.nan
            # CD14-high FDCs (Q75 = 1.084 from previous analysis)
            if cd14_idx is not None:
                cd14_vals = np.asarray(fdc_cells.X[:, cd14_idx]).flatten()
                cd14_hi_frac = float((cd14_vals > 1.084).sum() / len(cd14_vals))
            else:
                cd14_hi_frac = np.nan
        else:
            cd14_mean = cxcl13_mean = cd21_mean = cd14_hi_frac = np.nan

        # FDC zone fraction
        comp = roi.obs.get("compartment_name", pd.Series(dtype=str))
        fdc_zone_frac = float((comp == "FDC network zone").sum() / total) if len(comp) > 0 else 0

        fdc_features.append({
            "slide_ID": sid,
            "EZH2": ezh2,
            "FDC_fraction": n_fdc / total,
            "FDC_CD14_mean": cd14_mean,
            "FDC_CXCL13_mean": cxcl13_mean,
            "FDC_CD21_mean": cd21_mean,
            "FDC_CD14hi_fraction": cd14_hi_frac,
            "FDC_zone_fraction": fdc_zone_frac,
        })

    fdc_df = pd.DataFrame(fdc_features)
    print("\nFDC features (low grade, S-panel):")
    for col in ["FDC_fraction", "FDC_CD14_mean", "FDC_CXCL13_mean", "FDC_CD21_mean",
                "FDC_CD14hi_fraction", "FDC_zone_fraction"]:
        wt = fdc_df.loc[fdc_df["EZH2"] == "wt", col].dropna()
        mt = fdc_df.loc[fdc_df["EZH2"] == "mut", col].dropna()
        if len(wt) >= 3 and len(mt) >= 3:
            _, p = mannwhitneyu(wt, mt, alternative="two-sided")
            print(f"  {col}: wt={float(wt.mean()):.4f}, mut={float(mt.mean()):.4f}, p={p:.4f}")

    # 1e. Compartment fractions (S-panel UTAG, low grade)
    comp_counts_s = utag_s_low.obs.groupby(["slide_ID", "compartment_name"]).size().unstack(fill_value=0)
    comp_frac_s = comp_counts_s.div(comp_counts_s.sum(axis=1), axis=0)
    comp_frac_s["EZH2"] = comp_frac_s.index.map(ezh2_map_low)
    comp_frac_s = comp_frac_s[comp_frac_s["EZH2"].isin(["wt", "mut"])]

    comp_results_s = []
    for col in comp_frac_s.columns:
        if col == "EZH2":
            continue
        wt = comp_frac_s.loc[comp_frac_s["EZH2"] == "wt", col].dropna()
        mt = comp_frac_s.loc[comp_frac_s["EZH2"] == "mut", col].dropna()
        if len(wt) >= 3 and len(mt) >= 3:
            _, p = mannwhitneyu(wt, mt, alternative="two-sided")
            comp_results_s.append({"compartment": col, "wt_mean": float(wt.mean()),
                                   "mut_mean": float(mt.mean()), "pval": p})
    comp_results_s = pd.DataFrame(comp_results_s).sort_values("pval")
    print("\nCompartment fractions (low grade, S-panel):")
    print(comp_results_s.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Cell-cell distances by EZH2
    # ══════════════════════════════════════════════════════════════════
    print("\n=== ANALYSIS 2: Cell-cell distances by EZH2 ===")

    # S-panel: B cell to macrophage/FDC distances in FDC network zone
    s_b_types = ["B cells", "B cells (BCL2+)", "B cells (PAX5+)"]
    s_target_types = ["FDC", "M1 Macrophages", "Macrophages", "Dendritic cells",
                      "CD4 T cells", "CD8 T cells", "Stromal / CAF"]

    print("\nS-panel: B cell → target distances in FDC network zone")
    dist_s_fdc = compute_distances_in_zone(
        utag_s, adata_s, ezh2_map_all, "FDC network zone", s_target_types, s_b_types)

    if len(dist_s_fdc) > 0:
        for ttype in dist_s_fdc["target_type"].unique():
            sub = dist_s_fdc[dist_s_fdc["target_type"] == ttype]
            wt = sub.loc[sub["EZH2"] == "wt", "median_distance"]
            mt = sub.loc[sub["EZH2"] == "mut", "median_distance"]
            if len(wt) >= 3 and len(mt) >= 3:
                _, p = mannwhitneyu(wt, mt, alternative="two-sided")
                print(f"  B→{ttype}: wt={float(wt.mean()):.1f}µm, "
                      f"mut={float(mt.mean()):.1f}µm, p={p:.4f} (n_wt={len(wt)}, n_mut={len(mt)})")

    # Also compute in B cell zone (BCL2+) for comparison
    print("\nS-panel: B cell → target distances in B cell zone (BCL2+)")
    dist_s_bcl2 = compute_distances_in_zone(
        utag_s, adata_s, ezh2_map_all, "B cell zone (BCL2+)", s_target_types, s_b_types)

    if len(dist_s_bcl2) > 0:
        for ttype in dist_s_bcl2["target_type"].unique():
            sub = dist_s_bcl2[dist_s_bcl2["target_type"] == ttype]
            wt = sub.loc[sub["EZH2"] == "wt", "median_distance"]
            mt = sub.loc[sub["EZH2"] == "mut", "median_distance"]
            if len(wt) >= 3 and len(mt) >= 3:
                _, p = mannwhitneyu(wt, mt, alternative="two-sided")
                print(f"  B→{ttype}: wt={float(wt.mean()):.1f}µm, "
                      f"mut={float(mt.mean()):.1f}µm, p={p:.4f} (n_wt={len(wt)}, n_mut={len(mt)})")

    # T-panel: B cell to T cell/macrophage distances in follicular domains
    t_b_types = ["B cells", "B cells (CD20hi)", "B cells (CXCR5hi)"]
    t_target_types = ["CD4 T cells", "CD8 T cells", "CD8 T exhausted",
                      "Treg", "Macrophages"]
    t_follicular = ["B cell zone", "B cell zone (CD20hi)", "B cell zone (CXCR5hi)",
                    "GC-like zone"]

    # Find actual follicular compartment names in T-panel
    t_comps = utag_t.obs["compartment_name"].unique()
    t_b_comps = [c for c in t_comps if "B cell" in c or "GC" in c]
    print(f"\nT-panel follicular compartments: {t_b_comps}")

    # Use all follicular compartments
    if t_b_comps:
        print("\nT-panel: B cell → target distances in follicular zones")
        for comp in t_b_comps[:3]:  # limit to top 3
            dist_t = compute_distances_in_zone(
                utag_t, adata_t, ezh2_map_all, comp, t_target_types, t_b_types)
            if len(dist_t) > 0:
                print(f"\n  In '{comp}':")
                for ttype in dist_t["target_type"].unique():
                    sub = dist_t[dist_t["target_type"] == ttype]
                    wt = sub.loc[sub["EZH2"] == "wt", "median_distance"]
                    mt = sub.loc[sub["EZH2"] == "mut", "median_distance"]
                    if len(wt) >= 3 and len(mt) >= 3:
                        _, p = mannwhitneyu(wt, mt, alternative="two-sided")
                        print(f"    B→{ttype}: wt={float(wt.mean()):.1f}µm, "
                              f"mut={float(mt.mean()):.1f}µm, p={p:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE: Combined results
    # ══════════════════════════════════════════════════════════════════
    print("\n=== Creating figure ===")

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle("EZH2 Follow-up: Low-Grade FL & Cell-Cell Distances",
                 fontsize=16, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.96, bottom=0.04, left=0.06, right=0.97)

    # ── Row 1: Low-grade cell type fractions (top hits) ───────────────
    fig.text(0.02, 0.97, "Low-Grade (G1+G2) Cell Type Fractions by EZH2",
             fontsize=12, fontweight="bold", color="#333", va="top")

    top_ct = ct_results.head(4)
    for i, (_, row) in enumerate(top_ct.iterrows()):
        ax = fig.add_subplot(gs[0, i])
        wt = ct_frac.loc[ct_frac["EZH2"] == "wt", row["feature"]].dropna().values
        mt = ct_frac.loc[ct_frac["EZH2"] == "mut", row["feature"]].dropna().values
        boxplot_ezh2(ax, wt, mt, "Fraction", row["feature"])

    # ── Row 2: Low-grade B cell markers (top hits) ────────────────────
    fig.text(0.02, 0.655, "Low-Grade B Cell Markers by EZH2",
             fontsize=12, fontweight="bold", color="#333", va="top")

    top_bc = bc_results.head(3)
    for i, (_, row) in enumerate(top_bc.iterrows()):
        ax = fig.add_subplot(gs[1, i])
        wt = bc_marker.loc[bc_marker["EZH2"] == "wt", row["marker"]].dropna().values
        mt = bc_marker.loc[bc_marker["EZH2"] == "mut", row["marker"]].dropna().values
        boxplot_ezh2(ax, wt, mt, "Mean expression", f"B cells: {row['marker']}")

    # FDC CD14 in low grade
    ax = fig.add_subplot(gs[1, 3])
    wt_fdc = fdc_df.loc[fdc_df["EZH2"] == "wt", "FDC_CD14_mean"].dropna().values
    mt_fdc = fdc_df.loc[fdc_df["EZH2"] == "mut", "FDC_CD14_mean"].dropna().values
    boxplot_ezh2(ax, wt_fdc, mt_fdc, "Mean CD14", "FDC CD14 (low grade)")

    # ── Row 3: Cell-cell distances ────────────────────────────────────
    fig.text(0.02, 0.34, "Cell-Cell Distances in FDC Network Zone by EZH2",
             fontsize=12, fontweight="bold", color="#333", va="top")

    # Pick top distance comparisons from S-panel FDC network zone
    if len(dist_s_fdc) > 0:
        dist_pvals = []
        for ttype in dist_s_fdc["target_type"].unique():
            sub = dist_s_fdc[dist_s_fdc["target_type"] == ttype]
            wt = sub.loc[sub["EZH2"] == "wt", "median_distance"]
            mt = sub.loc[sub["EZH2"] == "mut", "median_distance"]
            if len(wt) >= 3 and len(mt) >= 3:
                _, p = mannwhitneyu(wt, mt, alternative="two-sided")
                dist_pvals.append((ttype, p, wt, mt))
        dist_pvals.sort(key=lambda x: x[1])

        for i, (ttype, p, wt, mt) in enumerate(dist_pvals[:4]):
            ax = fig.add_subplot(gs[2, i])
            boxplot_ezh2(ax, wt.values, mt.values,
                        "Median distance (µm)", f"B→{ttype}")

    out_path = os.path.join(OUTPUT_DIR, "ezh2_wendy_followup.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Save results tables
    ct_results.to_csv(os.path.join(OUTPUT_DIR, "ezh2_lowgrade_celltype_tests.csv"), index=False)
    bc_results.to_csv(os.path.join(OUTPUT_DIR, "ezh2_lowgrade_bcell_markers_T.csv"), index=False)
    bc_results_s.to_csv(os.path.join(OUTPUT_DIR, "ezh2_lowgrade_bcell_markers_S.csv"), index=False)
    if len(dist_s_fdc) > 0:
        dist_s_fdc.to_csv(os.path.join(OUTPUT_DIR, "ezh2_distances_fdc_zone.csv"), index=False)
    if len(dist_s_bcl2) > 0:
        dist_s_bcl2.to_csv(os.path.join(OUTPUT_DIR, "ezh2_distances_bcl2_zone.csv"), index=False)
    print("Result tables saved.")


if __name__ == "__main__":
    main()
