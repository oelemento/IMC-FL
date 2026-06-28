#!/usr/bin/env python3
"""Wendy EZH2 mut vs WT — slide 3: neighbor fraction in BCL2 B cell zone.

For each B cells (BCL2+) cell within the 'B cell zone (BCL2+)' compartment,
find its k=10 nearest neighbors, count neighbor cell types. Aggregate per ROI,
then per patient, then per group (Mut vs WT). Δ(Mut - WT) neighbor fraction
per cell type as a horizontal bar chart, plus mean fraction bars for each group.

S-panel (has FDC + CD14 cell types). LQ/Unassigned are excluded from neighbor
fraction denominator.

Outputs:
  output/ezh2/bcl2_neighbors/fig_ezh2_bcl2_neighbors.png
  output/ezh2/bcl2_neighbors/bcl2_neighbor_fractions_per_patient.csv
  output/ezh2/bcl2_neighbors/bcl2_neighbor_summary.csv
"""
import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

MIN_CELLS_PER_ROI = 8000
MIN_BCL2_CELLS_PER_ROI = 50
K_NN = 10
UNASSIGNED_CT = ["Unassigned", "Low quality / Unassigned"]
BCL2_COMPARTMENT = "B cell zone (BCL2+)"
SOURCE_CELL_TYPE = "B cells (BCL2+)"

# Display order for the bar chart (top = strongest Mut-WT positive enrichment)
DISPLAY_CELL_TYPES = [
    "B cells (BCL2+)", "B cells (PAX5+)", "B cells",
    "FDC", "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "CD4 T cells", "CD8 T cells",
    "Dendritic cells", "pDC", "Myeloid (S100A9+)",
    "Histiocytes (CD44hi)",
    "FRC (PDPN+)", "Stromal / CAF",
    "Endothelial",
    "Mixed / Border cells", "Other",
]


def is_tumor_core(sid):
    s = str(sid).lower()
    if any(t in s for t in ("tonsil", "prostate", "kidney", "spleen", "adrenal")):
        return False
    if any(t in s for t in ("_ton_", "_adr_", "_lym_", "_lym ")):
        return False
    if s.startswith("biomax"):
        return False
    if sid in EXCLUDE_ROIS:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s-panel", default="output/all_TMA_S_utag_ct_merged.h5ad")
    ap.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    ap.add_argument("--ezh2", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    ap.add_argument("--out", default="output/ezh2/bcl2_neighbors")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.s_panel} ...")
    with h5py.File(args.s_panel, "r") as f:
        sid_codes = f["obs/sample_id/codes"][:]
        sid_cats = np.array([c.decode() for c in f["obs/sample_id/categories"][:]])
        sample_id = sid_cats[sid_codes]
        ct_codes = f["obs/cell_type/codes"][:]
        ct_cats = np.array([c.decode() for c in f["obs/cell_type/categories"][:]])
        cell_type = ct_cats[ct_codes]
        comp_codes = f["obs/compartment_name/codes"][:]
        comp_cats = np.array([c.decode() for c in f["obs/compartment_name/categories"][:]])
        compartment = comp_cats[comp_codes]
        cx = f["obs/centroid_x"][:]
        cy = f["obs/centroid_y"][:]

    df = pd.DataFrame({"sample_id": sample_id, "cell_type": cell_type,
                       "compartment": compartment, "cx": cx, "cy": cy})
    df = df[df.sample_id.apply(is_tumor_core)].copy()
    df["sid_norm"] = df.sample_id.apply(normalize_sample_id)

    # QC: min typed cells per ROI — compute BEFORE EZH2 filter so the threshold
    # reflects the full ROI, not just cells with a successful EZH2 mapping.
    typed_per_roi = df[~df.cell_type.isin(UNASSIGNED_CT)].groupby("sid_norm").size()
    keep_rois = set(typed_per_roi[typed_per_roi >= MIN_CELLS_PER_ROI].index)
    df = df[df.sid_norm.isin(keep_rois)].copy()

    # EZH2 status
    clin = pd.read_csv(args.clinical)[["slide_ID", "Sample_ID", "Patient_ID"]]
    ezh = pd.read_excel(args.ezh2).rename(columns={"FL ID": "Sample_ID"})[["Sample_ID", "EZH2"]]
    mapping = (clin.merge(ezh, on="Sample_ID", how="inner")
                [["slide_ID", "Patient_ID", "EZH2"]]
                .drop_duplicates())
    if not mapping.slide_ID.is_unique:
        # Resolve duplicates: prefer "mut" over "wt" (any mut sample => patient is mut)
        mapping = (mapping.assign(rank=mapping.EZH2.map({"mut": 0, "wt": 1}).fillna(2))
                   .sort_values("rank").drop_duplicates("slide_ID", keep="first")
                   .drop(columns="rank"))
        assert mapping.slide_ID.is_unique
    df = df.merge(mapping, left_on="sid_norm", right_on="slide_ID", how="left")
    df = df[df.EZH2.isin(["wt", "mut"])].copy()

    print(f"  Total cells (post QC + EZH2 mapping): {len(df):,}")
    print(f"  ROIs: WT={df[df.EZH2=='wt'].sid_norm.nunique()}, Mut={df[df.EZH2=='mut'].sid_norm.nunique()}")
    print(f"  Patients: WT={df[df.EZH2=='wt'].Patient_ID.nunique()}, Mut={df[df.EZH2=='mut'].Patient_ID.nunique()}")

    # For each ROI, restrict to BCL2 B cell zone, compute neighbor fractions
    # around each B cells (BCL2+) source cell.
    print(f"\nComputing k={K_NN} neighbor fractions around '{SOURCE_CELL_TYPE}' in '{BCL2_COMPARTMENT}' ...")

    rows = []
    for sid, sub in df.groupby("sid_norm"):
        # Subset to the BCL2 compartment cells (use ALL cells in compartment as
        # the k-NN search space, not just B cells)
        compart_sub = sub[sub.compartment == BCL2_COMPARTMENT]
        if len(compart_sub) < MIN_BCL2_CELLS_PER_ROI:
            continue

        coords = np.column_stack([compart_sub.cx.values, compart_sub.cy.values])
        types = compart_sub.cell_type.values
        # Source mask: only BCL2+ B cells
        src_mask = types == SOURCE_CELL_TYPE
        if src_mask.sum() < 5:
            continue
        if len(coords) < K_NN + 1:
            continue

        tree = cKDTree(coords)
        # For each source cell, query k+1 nearest (drop self)
        src_coords = coords[src_mask]
        _, idx = tree.query(src_coords, k=K_NN + 1)
        nb_idx = idx[:, 1:]  # exclude self
        nb_types = types[nb_idx]  # (n_src, k)

        # Aggregate neighbor types into counts
        flat = nb_types.flatten()
        # Drop LQ/Unassigned from neighbor count
        flat = flat[~np.isin(flat, UNASSIGNED_CT)]
        if len(flat) == 0:
            continue

        ct_counts = pd.Series(flat).value_counts(normalize=True)

        row = {"sid": sid, "EZH2": sub.EZH2.iloc[0],
               "Patient_ID": sub.Patient_ID.iloc[0],
               "n_src": int(src_mask.sum()), "n_neighbors_total": len(flat)}
        for c in DISPLAY_CELL_TYPES:
            row[c] = float(ct_counts.get(c, 0.0))
        rows.append(row)

    roi_df = pd.DataFrame(rows)
    if len(roi_df) == 0:
        raise RuntimeError("No ROIs survived BCL2 source-cell filter")

    print(f"  ROIs with neighbor data: {len(roi_df)} ({roi_df.EZH2.value_counts().to_dict()})")

    # Patient-level
    pt_df = (roi_df.groupby(["Patient_ID", "EZH2"])[DISPLAY_CELL_TYPES]
             .mean().reset_index())
    pt_df.to_csv(out_dir / "bcl2_neighbor_fractions_per_patient.csv", index=False)
    print(f"  Patients: WT={(pt_df.EZH2=='wt').sum()}, Mut={(pt_df.EZH2=='mut').sum()}")

    # Group-level mean + MW
    summary_rows = []
    for c in DISPLAY_CELL_TYPES:
        a = pt_df.loc[pt_df.EZH2 == "wt", c].values
        b = pt_df.loc[pt_df.EZH2 == "mut", c].values
        wt_mean = float(np.mean(a)) if len(a) else np.nan
        mut_mean = float(np.mean(b)) if len(b) else np.nan
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            p_val = float(p)
        except (ValueError, IndexError):
            p_val = np.nan
        summary_rows.append({"cell_type": c, "WT_mean": wt_mean,
                              "Mut_mean": mut_mean,
                              "delta_mut_minus_wt_pp": (mut_mean - wt_mean) * 100,
                              "MW_p": p_val})
    summary = pd.DataFrame(summary_rows).sort_values("delta_mut_minus_wt_pp", key=lambda x: -np.abs(x))
    summary.to_csv(out_dir / "bcl2_neighbor_summary.csv", index=False)
    print("\nTop differences (|Δ| sorted):")
    print(summary.head(10).to_string(index=False))

    # ===================================================================
    # Figure: two panels
    #   left: Mut and WT mean fractions (horizontal grouped bars)
    #   right: Δ(Mut - WT) percentage points (signed bar chart)
    # ===================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # For plotting: show cell types in fixed display order, sorted by |Δ| descending
    plot_order = summary.sort_values("delta_mut_minus_wt_pp", key=lambda x: -np.abs(x)).cell_type.tolist()
    y = np.arange(len(plot_order))

    # Panel a: WT vs Mut means
    ax = axes[0]
    bar_h = 0.38
    wt_vals = [summary.loc[summary.cell_type == c, "WT_mean"].iloc[0] * 100 for c in plot_order]
    mut_vals = [summary.loc[summary.cell_type == c, "Mut_mean"].iloc[0] * 100 for c in plot_order]
    ax.barh(y - bar_h/2, wt_vals, height=bar_h, color="#1f77b4", label="EZH2 WT", alpha=0.85)
    ax.barh(y + bar_h/2, mut_vals, height=bar_h, color="#d62728", label="EZH2 Mut", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Neighbor fraction (%)")
    n_wt = int((pt_df.EZH2 == "wt").sum()); n_mut = int((pt_df.EZH2 == "mut").sum())
    ax.set_title(f"(a) Mean neighbor fraction\naround {SOURCE_CELL_TYPE} cells\nin {BCL2_COMPARTMENT}\n(n={n_wt} WT vs {n_mut} Mut patients)",
                 fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel b: Δ(Mut - WT) percentage points
    ax = axes[1]
    deltas = [summary.loc[summary.cell_type == c, "delta_mut_minus_wt_pp"].iloc[0] for c in plot_order]
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in deltas]
    ax.barh(y, deltas, color=colors, alpha=0.85)
    # Significance stars
    for i, c in enumerate(plot_order):
        p = summary.loc[summary.cell_type == c, "MW_p"].iloc[0]
        if not np.isnan(p) and p < 0.05:
            ax.text(deltas[i] + (0.3 if deltas[i] >= 0 else -0.3), i,
                    "*", ha="center", va="center",
                    fontsize=14, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(plot_order, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("Δ neighbor fraction (pp)\n(Mut − WT)")
    ax.set_title(f"(b) Δ Mut − WT  (*: MW p<0.05)", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle(f"Neighbor fraction around BCL2+ B cells in BCL2 B cell zone — EZH2 Mut vs WT",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_ezh2_bcl2_neighbors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
