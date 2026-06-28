#!/usr/bin/env python3
"""Wendy EZH2 mut vs WT — slide 5: compartment zone patterns + representative ROIs.

(a) Per-ROI compartment fractions, patient-level then group-level mean. Stacked
    bar comparing EZH2 WT vs Mut. MW per compartment.
(b) 2x3 grid of representative ROIs: row 1 = 3 EZH2 Mut ROIs, row 2 = 3 EZH2 WT
    ROIs. Cell types colored using the standard project palette. Picked at the
    50th, 70th, 30th percentile of compartment Shannon entropy for each group
    (i.e. give Wendy a 'typical' and two off-modal exemplars).

S-panel.

Outputs:
  output/ezh2/zone_patterns/fig_ezh2_zone_patterns.png
  output/ezh2/zone_patterns/zone_fractions_per_patient.csv
  output/ezh2/zone_patterns/zone_mw_summary.csv
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
UNASSIGNED_CT = ["Unassigned", "Low quality / Unassigned"]

# S-panel compartments and color palette (matches existing project figures)
S_COMPART_ORDER = [
    "B cell zone (BCL2+)", "B cell zone (PAX5+)", "B/T mixed zone",
    "FDC / myeloid zone", "FDC network zone",
    "Mixed (B cells (PAX 27%)", "Mixed (M2 Macrophag 26%)",
    "Other / myeloid zone", "Stromal / CAF zone", "T cell zone",
    "Unidentified zone",
]
S_COMPART_COLORS = {
    "B cell zone (BCL2+)":      "#1f77b4",
    "B cell zone (PAX5+)":      "#aec7e8",
    "B/T mixed zone":           "#9467bd",
    "FDC / myeloid zone":       "#ff7f0e",
    "FDC network zone":         "#d62728",
    "Mixed (B cells (PAX 27%)": "#bcbd22",
    "Mixed (M2 Macrophag 26%)": "#8c564b",
    "Other / myeloid zone":     "#7f7f7f",
    "Stromal / CAF zone":       "#e377c2",
    "T cell zone":              "#2ca02c",
    "Unidentified zone":        "#cccccc",
}
S_CELL_TYPE_COLORS = {
    "B cells (BCL2+)":     "#1f77b4",
    "B cells (PAX5+)":     "#aec7e8",
    "B cells":             "#5b9bd5",
    "FDC":                 "#d62728",
    "M1 Macrophages":      "#8c564b",
    "M2 Macrophages":      "#a0522d",
    "Macrophages":         "#7f4f24",
    "Histiocytes (CD44hi)":"#bcbd22",
    "CD4 T cells":         "#2ca02c",
    "CD8 T cells":         "#17becf",
    "Dendritic cells":     "#ff7f0e",
    "pDC":                 "#ffbb78",
    "Myeloid (S100A9+)":   "#9467bd",
    "FRC (PDPN+)":         "#c49c94",
    "Stromal / CAF":       "#e377c2",
    "Endothelial":         "#1abc9c",
    "Mixed / Border cells":"#cccccc",
    "Other":               "#999999",
}


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
    ap.add_argument("--out", default="output/ezh2/zone_patterns")
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

    # QC pre-merge
    typed_per_roi = df[~df.cell_type.isin(UNASSIGNED_CT)].groupby("sid_norm").size()
    keep_rois = set(typed_per_roi[typed_per_roi >= MIN_CELLS_PER_ROI].index)
    df = df[df.sid_norm.isin(keep_rois)].copy()

    clin = pd.read_csv(args.clinical)[["slide_ID", "Sample_ID", "Patient_ID"]]
    ezh = pd.read_excel(args.ezh2).rename(columns={"FL ID": "Sample_ID"})[["Sample_ID", "EZH2"]]
    mapping = (clin.merge(ezh, on="Sample_ID", how="inner")
                [["slide_ID", "Patient_ID", "EZH2"]].drop_duplicates())
    if not mapping.slide_ID.is_unique:
        mapping = (mapping.assign(rank=mapping.EZH2.map({"mut": 0, "wt": 1}).fillna(2))
                   .sort_values("rank").drop_duplicates("slide_ID", keep="first")
                   .drop(columns="rank"))
    df = df.merge(mapping, left_on="sid_norm", right_on="slide_ID", how="left")
    df = df[df.EZH2.isin(["wt", "mut"])].copy()
    print(f"  Patients post QC: WT={df[df.EZH2=='wt'].Patient_ID.nunique()}, "
          f"Mut={df[df.EZH2=='mut'].Patient_ID.nunique()}")

    # Per-ROI compartment fractions + entropy
    rows = []
    for sid, sub in df.groupby("sid_norm"):
        cf = sub.compartment.value_counts(normalize=True)
        p_ = cf.values
        shannon = float(-(p_ * np.log2(p_ + 1e-12)).sum())
        row = {"sid": sid, "EZH2": sub.EZH2.iloc[0],
               "Patient_ID": sub.Patient_ID.iloc[0],
               "shannon": shannon, "n_cells": int(len(sub))}
        for c in S_COMPART_ORDER:
            row[f"frac_{c}"] = float(cf.get(c, 0.0))
        rows.append(row)
    roi_df = pd.DataFrame(rows)
    print(f"  ROIs: {len(roi_df)} ({roi_df.EZH2.value_counts().to_dict()})")

    metric_cols = ["shannon"] + [f"frac_{c}" for c in S_COMPART_ORDER]
    pt_df = roi_df.groupby(["Patient_ID", "EZH2"])[metric_cols].mean().reset_index()
    pt_df.to_csv(out_dir / "zone_fractions_per_patient.csv", index=False)

    # MW per compartment
    summary = []
    for c in S_COMPART_ORDER:
        col = f"frac_{c}"
        a = pt_df.loc[pt_df.EZH2 == "wt", col].values
        b = pt_df.loc[pt_df.EZH2 == "mut", col].values
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except (ValueError, IndexError):
            p = np.nan
        summary.append({"compartment": c, "WT_mean": float(np.mean(a)),
                         "Mut_mean": float(np.mean(b)),
                         "delta_mut_minus_wt_pp": (np.mean(b) - np.mean(a)) * 100,
                         "MW_p": float(p) if not np.isnan(p) else np.nan})
    # Add Shannon as a separate row
    a = pt_df.loc[pt_df.EZH2 == "wt", "shannon"].values
    b = pt_df.loc[pt_df.EZH2 == "mut", "shannon"].values
    try:
        _, p_sh = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        p_sh = np.nan
    summary.append({"compartment": "shannon (entropy)",
                     "WT_mean": float(np.mean(a)), "Mut_mean": float(np.mean(b)),
                     "delta_mut_minus_wt_pp": np.nan,
                     "MW_p": float(p_sh) if not np.isnan(p_sh) else np.nan})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "zone_mw_summary.csv", index=False)
    print("\nKey results:")
    print(summary_df.sort_values("MW_p").to_string(index=False))

    # ===================================================================
    # Figure
    # ===================================================================
    n_wt = int((pt_df.EZH2 == "wt").sum()); n_mut = int((pt_df.EZH2 == "mut").sum())

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.1, 1, 1], hspace=0.35, wspace=0.15)

    # (a) Stacked bar comparison (left), MW bar (right)
    ax_a = fig.add_subplot(gs[0, :2])
    grade_means = pt_df.groupby("EZH2")[[f"frac_{c}" for c in S_COMPART_ORDER]].mean().reindex(["wt", "mut"])
    grade_means = grade_means.div(grade_means.sum(axis=1), axis=0)
    labels = [f"WT (n={n_wt})", f"Mut (n={n_mut})"]
    bottoms = np.zeros(2)
    for c in S_COMPART_ORDER:
        vals = grade_means[f"frac_{c}"].values * 100
        ax_a.bar(labels, vals, bottom=bottoms, color=S_COMPART_COLORS[c],
                 label=c, edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax_a.set_ylabel("Mean compartment fraction (%)")
    ax_a.set_title("(a) Compartment composition by EZH2 status", fontsize=11)
    ax_a.set_ylim(0, 100)
    ax_a.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for sp in ("top", "right"):
        ax_a.spines[sp].set_visible(False)

    # (b) MW Δ bar chart
    ax_b = fig.add_subplot(gs[0, 3:])
    sub_summary = summary_df[summary_df.compartment.isin(S_COMPART_ORDER)].copy()
    sub_summary = sub_summary.sort_values("delta_mut_minus_wt_pp", key=lambda x: -np.abs(x))
    y = np.arange(len(sub_summary))
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in sub_summary.delta_mut_minus_wt_pp]
    ax_b.barh(y, sub_summary.delta_mut_minus_wt_pp, color=colors, alpha=0.85)
    for i, (_, r) in enumerate(sub_summary.iterrows()):
        if not np.isnan(r.MW_p) and r.MW_p < 0.05:
            d = r.delta_mut_minus_wt_pp
            ax_b.text(d + (0.5 if d >= 0 else -0.5), i, "*",
                      ha="center", va="center", fontsize=14, fontweight="bold")
    ax_b.set_yticks(y); ax_b.set_yticklabels(sub_summary.compartment, fontsize=9)
    ax_b.invert_yaxis()
    ax_b.axvline(0, color="gray", lw=0.7)
    ax_b.set_xlabel("Δ compartment fraction (pp)\n(Mut − WT)")
    ax_b.set_title(f"(b) Per-compartment Δ Mut − WT (*: MW p<0.05)", fontsize=11)
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(False)

    # (c) Representative ROIs: 3 Mut on row 1, 3 WT on row 2
    # Pick ROIs at 50th, 30th, 70th percentile of shannon for each group
    def pick_rois(group_label, ezh2_val, target_quantiles=(0.30, 0.50, 0.70)):
        sub = roi_df[roi_df.EZH2 == ezh2_val].sort_values("shannon")
        if len(sub) == 0:
            return []
        out = []
        for q in target_quantiles:
            idx = int(round(q * (len(sub) - 1)))
            out.append(sub.iloc[idx].sid)
        return out

    mut_rois = pick_rois("Mut", "mut")
    wt_rois = pick_rois("WT", "wt")

    def plot_roi(ax, sid, group_label):
        sub = df[df.sid_norm == sid]
        if len(sub) == 0:
            ax.text(0.5, 0.5, f"{sid}\nno data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return
        # Unassigned in light gray (background), typed cells colored
        un_mask = sub.cell_type.isin(UNASSIGNED_CT)
        ax.scatter(sub.loc[un_mask, "cx"], sub.loc[un_mask, "cy"],
                   c="#D3D3D3", s=1, alpha=0.4, edgecolors="none", zorder=1)
        for ct, sub_ct in sub[~un_mask].groupby("cell_type"):
            color = S_CELL_TYPE_COLORS.get(ct, "#888888")
            ax.scatter(sub_ct.cx, sub_ct.cy, c=color, s=2.5, alpha=0.85,
                       edgecolors="none", zorder=2)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{group_label}: {sid}", fontsize=9)

    for col_i, sid in enumerate(mut_rois):
        ax = fig.add_subplot(gs[1, col_i*2:col_i*2+2])
        plot_roi(ax, sid, "Mut")
    for col_i, sid in enumerate(wt_rois):
        ax = fig.add_subplot(gs[2, col_i*2:col_i*2+2])
        plot_roi(ax, sid, "WT")

    # Cell-type legend at bottom right
    from matplotlib.patches import Patch
    handles = [Patch(color=col, label=name)
               for name, col in S_CELL_TYPE_COLORS.items()]
    fig.legend(handles=handles, loc="lower right", ncol=3, fontsize=7,
               bbox_to_anchor=(0.99, 0.0), title="Cell types")

    fig.suptitle("EZH2 Mut vs WT — compartment zone patterns + representative cores",
                 fontsize=13, y=0.995)
    out = out_dir / "fig_ezh2_zone_patterns.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
