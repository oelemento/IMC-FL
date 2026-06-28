#!/usr/bin/env python3
"""Wendy EZH2 mut vs WT — slide 6 fix: T cells SURROUNDING the macrophage zone.

Her actual question was spatial proximity ("Are Treg or exhausted T cells
SURROUNDING the macrophage zone in Mut EZH2 patients?"), not per-compartment
composition. For each Treg and exhausted-CD8 cell, compute distance to the
nearest cell that sits in the 'Macrophage-rich zone' compartment. Aggregate
per ROI, then per patient, then by EZH2 status.

Metrics per ROI:
  - median distance from Treg cells to nearest Mac-zone cell
  - median distance from exh-CD8 cells to nearest Mac-zone cell
  - fraction of Treg cells within 30 µm of any Mac-zone cell ("surrounding")
  - fraction of exh-CD8 cells within 30 µm of any Mac-zone cell

T-panel.

Outputs:
  output/ezh2/tcell_near_mac/fig_ezh2_tcell_near_mac.png
  output/ezh2/tcell_near_mac/tcell_proximity_per_patient.csv
  output/ezh2/tcell_near_mac/tcell_proximity_mw_summary.csv
"""
import argparse, sys
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
MIN_MAC_ZONE_CELLS = 100
MIN_TREG_CELLS_PER_ROI = 5
MIN_EXH_CELLS_PER_ROI = 5
PROXIMITY_THRESHOLD_UM = 30.0
UNASSIGNED_CT = ["Unassigned", "Low quality / Unassigned"]
MAC_ZONE = "Macrophage-rich zone"
EXH_CD8_TYPES = ["CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]
TREG_TYPES = ["Treg"]


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
    ap.add_argument("--t-panel", default="output/all_TMA_T_utag_ct_merged.h5ad")
    ap.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    ap.add_argument("--ezh2", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    ap.add_argument("--out", default="output/ezh2/tcell_near_mac")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.t_panel} ...")
    with h5py.File(args.t_panel, "r") as f:
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

    rows = []
    for sid, sub in df.groupby("sid_norm"):
        mac_zone = sub[sub.compartment == MAC_ZONE]
        if len(mac_zone) < MIN_MAC_ZONE_CELLS:
            continue
        # Exclude query cells that are themselves in the Mac-rich zone — "Treg
        # surrounding the Mac zone" means Tregs OUTSIDE the zone, queried for
        # distance to it. Otherwise cKDTree returns distance 0 for self.
        treg = sub[sub.cell_type.isin(TREG_TYPES) & (sub.compartment != MAC_ZONE)]
        exh = sub[sub.cell_type.isin(EXH_CD8_TYPES) & (sub.compartment != MAC_ZONE)]
        if len(treg) < MIN_TREG_CELLS_PER_ROI and len(exh) < MIN_EXH_CELLS_PER_ROI:
            continue
        mac_coords = np.column_stack([mac_zone.cx.values, mac_zone.cy.values])
        tree = cKDTree(mac_coords)

        row = {"sid": sid, "EZH2": sub.EZH2.iloc[0],
               "Patient_ID": sub.Patient_ID.iloc[0],
               "n_mac_zone_cells": int(len(mac_zone)),
               "n_treg": int(len(treg)), "n_exh": int(len(exh))}
        if len(treg) >= MIN_TREG_CELLS_PER_ROI:
            tc = np.column_stack([treg.cx.values, treg.cy.values])
            dists, _ = tree.query(tc, k=1)
            row["treg_median_dist_to_mac_zone"] = float(np.median(dists))
            row["treg_pct_within_30um"] = float((dists <= PROXIMITY_THRESHOLD_UM).mean())
        else:
            row["treg_median_dist_to_mac_zone"] = np.nan
            row["treg_pct_within_30um"] = np.nan
        if len(exh) >= MIN_EXH_CELLS_PER_ROI:
            ec = np.column_stack([exh.cx.values, exh.cy.values])
            dists, _ = tree.query(ec, k=1)
            row["exh_median_dist_to_mac_zone"] = float(np.median(dists))
            row["exh_pct_within_30um"] = float((dists <= PROXIMITY_THRESHOLD_UM).mean())
        else:
            row["exh_median_dist_to_mac_zone"] = np.nan
            row["exh_pct_within_30um"] = np.nan
        rows.append(row)

    roi_df = pd.DataFrame(rows)
    print(f"  ROIs with Mac zone + T cells: {len(roi_df)} ({roi_df.EZH2.value_counts().to_dict()})")
    if len(roi_df) == 0:
        raise RuntimeError("No ROIs survived QC")

    metric_cols = ["treg_median_dist_to_mac_zone", "exh_median_dist_to_mac_zone",
                   "treg_pct_within_30um", "exh_pct_within_30um"]
    pt_df = roi_df.groupby(["Patient_ID", "EZH2"])[metric_cols].mean().reset_index()
    pt_df.to_csv(out_dir / "tcell_proximity_per_patient.csv", index=False)
    print(f"  Patients with proximity data: WT={(pt_df.EZH2=='wt').sum()}, "
          f"Mut={(pt_df.EZH2=='mut').sum()}")

    rows = []
    for m in metric_cols:
        a = pt_df.loc[pt_df.EZH2 == "wt", m].dropna().values
        b = pt_df.loc[pt_df.EZH2 == "mut", m].dropna().values
        if len(a) >= 3 and len(b) >= 3:
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                p = np.nan
        else:
            p = np.nan
        rows.append({"metric": m, "n_WT": len(a), "n_Mut": len(b),
                     "WT_mean": float(np.mean(a)) if len(a) else np.nan,
                     "Mut_mean": float(np.mean(b)) if len(b) else np.nan,
                     "MW_p": float(p) if not np.isnan(p) else np.nan})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "tcell_proximity_mw_summary.csv", index=False)
    print("\nResults:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    rng = np.random.default_rng(0)

    def boxplot(ax, metric, ylabel, title):
        a = pt_df.loc[pt_df.EZH2 == "wt", metric].dropna().values
        b = pt_df.loc[pt_df.EZH2 == "mut", metric].dropna().values
        data = [a, b]
        labels = [f"WT (n={len(a)})", f"Mut (n={len(b)})"]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        widths=0.55, showfliers=False)
        bp["boxes"][0].set_facecolor("#1f77b4"); bp["boxes"][0].set_alpha(0.55)
        bp["boxes"][1].set_facecolor("#d62728"); bp["boxes"][1].set_alpha(0.55)
        for i, vals in enumerate(data):
            xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.2
            col = "#1f77b4" if i == 0 else "#d62728"
            ax.scatter(xs, vals, color=col, s=28, alpha=0.85,
                       edgecolor="white", linewidth=0.5, zorder=3)
        if len(a) >= 3 and len(b) >= 3:
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
                star = " *" if p < 0.05 else ""
                ax.set_title(f"{title}\nMW p={p:.3g}{star}", fontsize=10)
            except (ValueError, IndexError):
                ax.set_title(title, fontsize=10)
        else:
            ax.set_title(f"{title}\n(n<3 in one arm — MW not run)", fontsize=10)
        ax.set_ylabel(ylabel)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    boxplot(axes[0], "treg_median_dist_to_mac_zone",
            "Median distance (µm)",
            "(a) Treg → Mac zone\nmedian distance per ROI")
    boxplot(axes[1], "treg_pct_within_30um",
            f"Fraction within {int(PROXIMITY_THRESHOLD_UM)} µm",
            f"(b) Treg surrounding Mac zone\n(% within {int(PROXIMITY_THRESHOLD_UM)} µm)")
    boxplot(axes[2], "exh_median_dist_to_mac_zone",
            "Median distance (µm)",
            "(c) Exh CD8 → Mac zone\nmedian distance per ROI")
    boxplot(axes[3], "exh_pct_within_30um",
            f"Fraction within {int(PROXIMITY_THRESHOLD_UM)} µm",
            f"(d) Exh CD8 surrounding Mac zone\n(% within {int(PROXIMITY_THRESHOLD_UM)} µm)")

    fig.suptitle("EZH2 Mut vs WT — Treg and exhausted CD8 proximity to the Macrophage-rich zone",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    out = out_dir / "fig_ezh2_tcell_near_mac.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
