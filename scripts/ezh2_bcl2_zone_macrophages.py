#!/usr/bin/env python3
"""Wendy EZH2 mut vs WT — slide 4: BCL2 B cell zone composition + CD68 intensity.

Tests Wendy's DLBCL prior that EZH2 mutant patients show increased macrophage
infiltration in the tumor B cell zone. Two analyses:

(a) Per-ROI cell type composition WITHIN 'B cell zone (BCL2+)' compartment,
    aggregated to patient level, compared between EZH2 WT and Mut groups.
    Mann-Whitney per cell type. S-panel.

(b) Per-cell CD68 intensity (mean per ROI) within the BCL2 B cell zone, plus
    a per-ROI BCL2 cell density (BCL2+ B cells / total compartment cells).
    Boxplots of CD68 mean intensity (Mut vs WT) and BCL2 density (Mut vs WT).

Outputs:
  output/ezh2/bcl2_zone/fig_ezh2_bcl2_zone_macrophages.png
  output/ezh2/bcl2_zone/bcl2_zone_composition.csv
  output/ezh2/bcl2_zone/bcl2_zone_cd68_intensity.csv
"""
import argparse, sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

MIN_CELLS_PER_ROI = 8000
MIN_BCL2_COMPARTMENT_CELLS = 100
UNASSIGNED_CT = ["Unassigned", "Low quality / Unassigned"]
BCL2_COMPARTMENT = "B cell zone (BCL2+)"

DISPLAY_CELL_TYPES = [
    "B cells (BCL2+)", "B cells (PAX5+)", "B cells",
    "FDC", "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Histiocytes (CD44hi)",
    "CD4 T cells", "CD8 T cells",
    "Dendritic cells", "pDC", "Myeloid (S100A9+)",
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
    ap.add_argument("--out", default="output/ezh2/bcl2_zone")
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
        # Load CD68 intensity from scaled .X
        var_names = [v.decode() for v in f["var/_index"][:]]
        cd68_idx = var_names.index("CD68")
        bcl2_idx = var_names.index("BCL_2")
        X = f["X"]
        n_obs = len(sample_id)
        cd68 = X[:, cd68_idx] if X.shape == (n_obs, len(var_names)) else np.array(X[:, cd68_idx])
        bcl2 = X[:, bcl2_idx] if X.shape == (n_obs, len(var_names)) else np.array(X[:, bcl2_idx])

    df = pd.DataFrame({"sample_id": sample_id, "cell_type": cell_type,
                       "compartment": compartment, "CD68": cd68, "BCL_2": bcl2})
    df = df[df.sample_id.apply(is_tumor_core)].copy()
    df["sid_norm"] = df.sample_id.apply(normalize_sample_id)

    # QC BEFORE EZH2 filter (reflect full-ROI cell count)
    typed_per_roi = df[~df.cell_type.isin(UNASSIGNED_CT)].groupby("sid_norm").size()
    keep_rois = set(typed_per_roi[typed_per_roi >= MIN_CELLS_PER_ROI].index)
    df = df[df.sid_norm.isin(keep_rois)].copy()

    clin = pd.read_csv(args.clinical)[["slide_ID", "Sample_ID", "Patient_ID"]]
    ezh = pd.read_excel(args.ezh2).rename(columns={"FL ID": "Sample_ID"})[["Sample_ID", "EZH2"]]
    mapping = (clin.merge(ezh, on="Sample_ID", how="inner")
                [["slide_ID", "Patient_ID", "EZH2"]]
                .drop_duplicates())
    if not mapping.slide_ID.is_unique:
        mapping = (mapping.assign(rank=mapping.EZH2.map({"mut": 0, "wt": 1}).fillna(2))
                   .sort_values("rank").drop_duplicates("slide_ID", keep="first")
                   .drop(columns="rank"))
        assert mapping.slide_ID.is_unique
    df = df.merge(mapping, left_on="sid_norm", right_on="slide_ID", how="left")
    df = df[df.EZH2.isin(["wt", "mut"])].copy()
    print(f"  Patients post QC: WT={df[df.EZH2=='wt'].Patient_ID.nunique()}, Mut={df[df.EZH2=='mut'].Patient_ID.nunique()}")

    # Restrict to BCL2 compartment
    bcl2_df = df[df.compartment == BCL2_COMPARTMENT].copy()
    # Drop LQ for composition denominator; keep CD68 only for typed cells
    bcl2_typed = bcl2_df[~bcl2_df.cell_type.isin(UNASSIGNED_CT)].copy()

    # Per-ROI metrics
    rows = []
    for sid, sub in bcl2_typed.groupby("sid_norm"):
        if len(sub) < MIN_BCL2_COMPARTMENT_CELLS:
            continue
        ct_frac = sub.cell_type.value_counts(normalize=True)
        # BCL2 marker mean (across all cells in compartment) and CD68 mean (across all cells)
        bcl2_mean = float(sub.BCL_2.mean())
        cd68_mean = float(sub.CD68.mean())
        # BCL2 cell density = fraction of BCL2+ B cells among compartment cells
        bcl2_density = float((sub.cell_type == "B cells (BCL2+)").mean())
        row = {"sid": sid, "EZH2": sub.EZH2.iloc[0],
               "Patient_ID": sub.Patient_ID.iloc[0],
               "n_cells_in_compartment": int(len(sub)),
               "BCL2_mean_intensity": bcl2_mean,
               "CD68_mean_intensity": cd68_mean,
               "BCL2_cell_density": bcl2_density}
        for c in DISPLAY_CELL_TYPES:
            row[f"frac_{c}"] = float(ct_frac.get(c, 0.0))
        rows.append(row)
    roi_df = pd.DataFrame(rows)
    print(f"  ROIs with sufficient compartment cells: {len(roi_df)} ({roi_df.EZH2.value_counts().to_dict()})")

    # Patient-level
    metric_cols = ["BCL2_mean_intensity", "CD68_mean_intensity", "BCL2_cell_density"] + \
                  [f"frac_{c}" for c in DISPLAY_CELL_TYPES]
    pt_df = roi_df.groupby(["Patient_ID", "EZH2"])[metric_cols].mean().reset_index()
    pt_df.to_csv(out_dir / "bcl2_zone_composition.csv", index=False)
    print(f"  Patients: WT={(pt_df.EZH2=='wt').sum()}, Mut={(pt_df.EZH2=='mut').sum()}")

    # MW per cell type fraction + intensity metrics
    summary_rows = []
    for c in DISPLAY_CELL_TYPES:
        col = f"frac_{c}"
        a = pt_df.loc[pt_df.EZH2 == "wt", col].values
        b = pt_df.loc[pt_df.EZH2 == "mut", col].values
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except (ValueError, IndexError):
            p = np.nan
        summary_rows.append({"cell_type": c, "WT_mean": float(np.mean(a)) if len(a) else np.nan,
                              "Mut_mean": float(np.mean(b)) if len(b) else np.nan,
                              "delta_mut_minus_wt_pp": (np.mean(b) - np.mean(a)) * 100 if len(a) and len(b) else np.nan,
                              "MW_p": float(p) if not np.isnan(p) else np.nan})
    for m in ["BCL2_mean_intensity", "CD68_mean_intensity", "BCL2_cell_density"]:
        a = pt_df.loc[pt_df.EZH2 == "wt", m].values
        b = pt_df.loc[pt_df.EZH2 == "mut", m].values
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except (ValueError, IndexError):
            p = np.nan
        summary_rows.append({"cell_type": m, "WT_mean": float(np.mean(a)),
                              "Mut_mean": float(np.mean(b)),
                              "delta_mut_minus_wt_pp": np.nan,
                              "MW_p": float(p) if not np.isnan(p) else np.nan})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "bcl2_zone_cd68_intensity.csv", index=False)
    print("\nKey results:")
    print(summary.sort_values("MW_p").head(8).to_string(index=False))

    # ===================================================================
    # Figure
    # ===================================================================
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.6, 0.6], wspace=0.4)

    # (a) Cell type composition in BCL2 zone — grouped horizontal bars
    ax = fig.add_subplot(gs[0])
    summary_frac = summary[summary.cell_type.isin(DISPLAY_CELL_TYPES)].copy()
    summary_frac = summary_frac.sort_values("delta_mut_minus_wt_pp", key=lambda x: -np.abs(x))
    y = np.arange(len(summary_frac))
    bar_h = 0.38
    ax.barh(y - bar_h/2, summary_frac.WT_mean * 100, height=bar_h,
            color="#1f77b4", label="EZH2 WT", alpha=0.85)
    ax.barh(y + bar_h/2, summary_frac.Mut_mean * 100, height=bar_h,
            color="#d62728", label="EZH2 Mut", alpha=0.85)
    for i, (idx, row) in enumerate(summary_frac.iterrows()):
        if not np.isnan(row.MW_p) and row.MW_p < 0.05:
            ax.text(max(row.WT_mean, row.Mut_mean) * 100 + 1, i, "*",
                    ha="left", va="center", fontsize=14, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(summary_frac.cell_type, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of cells in BCL2 B cell zone (%)", fontsize=10)
    n_wt = int((pt_df.EZH2 == "wt").sum()); n_mut = int((pt_df.EZH2 == "mut").sum())
    ax.set_title(f"(a) Cell types in BCL2 B cell zone\n(EZH2 WT n={n_wt} vs Mut n={n_mut} patients; *: MW p<0.05)",
                 fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (b) BCL2 density boxplot
    def boxplot(ax, metric, ylabel, title):
        a = pt_df.loc[pt_df.EZH2 == "wt", metric].values
        b = pt_df.loc[pt_df.EZH2 == "mut", metric].values
        data = [a, b]
        bp = ax.boxplot(data, tick_labels=["WT", "Mut"], patch_artist=True,
                        widths=0.55, showfliers=False)
        bp["boxes"][0].set_facecolor("#1f77b4"); bp["boxes"][0].set_alpha(0.55)
        bp["boxes"][1].set_facecolor("#d62728"); bp["boxes"][1].set_alpha(0.55)
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data):
            xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
            col = "#1f77b4" if i == 0 else "#d62728"
            ax.scatter(xs, vals, color=col, s=22, alpha=0.8,
                       edgecolor="white", linewidth=0.4, zorder=3)
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            star = " *" if p < 0.05 else ""
            ax.set_title(f"{title}\nMW p={p:.3g}{star}", fontsize=10)
        except ValueError:
            ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    boxplot(fig.add_subplot(gs[1]), "BCL2_cell_density",
            "BCL2+ B cell fraction (of compartment cells)",
            "(b) BCL2+ B cell density")
    boxplot(fig.add_subplot(gs[2]), "CD68_mean_intensity",
            "CD68 mean intensity (scaled)",
            "(c) CD68 intensity in BCL2 zone")

    fig.suptitle("EZH2 Mut vs WT — BCL2 B cell zone composition and macrophage signal",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_ezh2_bcl2_zone_macrophages.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
