#!/usr/bin/env python3
"""Wendy EZH2 mut vs WT — slide 6: CD8 exhaustion + Treg distribution by compartment.

For each T-panel compartment in the paper-9 scheme, compute:
  - Fraction of CD8 T cells that are exhausted (TOX+ / pre-exh) within that compartment
  - Fraction of Treg cells (% of typed) within that compartment

Stratify by EZH2 status (WT vs Mut). MW per compartment.

Outputs:
  output/ezh2/tcell_status/fig_ezh2_tcell_status.png
  output/ezh2/tcell_status/tcell_status_per_patient.csv
  output/ezh2/tcell_status/tcell_status_mw_summary.csv
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
MIN_COMP_CELLS = 100
UNASSIGNED_CT = ["Unassigned", "Low quality / Unassigned"]

CD8_LIKE = ["CD8 T cells", "CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]
CD8_EXH = ["CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]
TREG = ["Treg"]

PAPER_9 = [
    "GC core", "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)", "B cell follicle (CD20hi/CXCR5hi)",
    "B cell zone", "Follicle-T zone interface",
    "Treg-enriched T zone", "T cell zone (CD4/CD8)", "Macrophage-rich zone",
]
PAPER_9_SHORT = ["GC core", "Follicle core", "Follicle mantle",
                  "B cell follicle", "B cell zone", "Foll-T interface",
                  "Treg zone", "T cell zone", "Mac zone"]


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
    ap.add_argument("--out", default="output/ezh2/tcell_status")
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

    df = pd.DataFrame({"sample_id": sample_id, "cell_type": cell_type,
                       "compartment": compartment})
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

    # Per-ROI per-compartment metrics
    rows = []
    for sid, sub in df.groupby("sid_norm"):
        typed = sub[~sub.cell_type.isin(UNASSIGNED_CT)]
        for comp in PAPER_9:
            comp_sub = typed[typed.compartment == comp]
            n = len(comp_sub)
            if n < MIN_COMP_CELLS:
                continue
            n_cd8 = comp_sub.cell_type.isin(CD8_LIKE).sum()
            n_exh = comp_sub.cell_type.isin(CD8_EXH).sum()
            n_treg = comp_sub.cell_type.isin(TREG).sum()
            rows.append({
                "sid": sid, "EZH2": sub.EZH2.iloc[0],
                "Patient_ID": sub.Patient_ID.iloc[0],
                "compartment": comp,
                "n": int(n),
                "cd8_exh_frac": float(n_exh / max(n_cd8, 1)) if n_cd8 > 0 else np.nan,
                "treg_frac": float(n_treg / n),
            })
    roi_df = pd.DataFrame(rows)
    print(f"  (ROI, compartment) rows: {len(roi_df)}")

    # Patient-level
    pt = (roi_df.groupby(["Patient_ID", "EZH2", "compartment"])
          [["cd8_exh_frac", "treg_frac"]].mean().reset_index())
    pt.to_csv(out_dir / "tcell_status_per_patient.csv", index=False)

    # MW per compartment
    rows = []
    for comp in PAPER_9:
        sub = pt[pt.compartment == comp]
        for metric in ["cd8_exh_frac", "treg_frac"]:
            a = sub.loc[sub.EZH2 == "wt", metric].dropna().values
            b = sub.loc[sub.EZH2 == "mut", metric].dropna().values
            if len(a) >= 3 and len(b) >= 3:
                try:
                    _, p = mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    p = np.nan
            else:
                p = np.nan
            rows.append({"compartment": comp, "metric": metric,
                         "WT_mean": float(np.mean(a)) if len(a) else np.nan,
                         "Mut_mean": float(np.mean(b)) if len(b) else np.nan,
                         "n_WT": int(len(a)), "n_Mut": int(len(b)),
                         "MW_p": float(p) if not np.isnan(p) else np.nan})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "tcell_status_mw_summary.csv", index=False)
    print("\nKey results (sorted by p):")
    print(summary.sort_values("MW_p").head(10).to_string(index=False))

    # ===================================================================
    # Figure: two grouped bar charts (exhaustion + Treg) by compartment
    # ===================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    def plot_grouped(ax, metric, ylabel, title):
        comp_means_wt = []; comp_means_mut = []; p_vals = []
        for comp in PAPER_9:
            sub = summary[(summary.compartment == comp) & (summary.metric == metric)]
            if len(sub) == 0:
                comp_means_wt.append(np.nan); comp_means_mut.append(np.nan); p_vals.append(np.nan)
                continue
            comp_means_wt.append(sub.WT_mean.iloc[0])
            comp_means_mut.append(sub.Mut_mean.iloc[0])
            p_vals.append(sub.MW_p.iloc[0])
        x = np.arange(len(PAPER_9))
        bw = 0.38
        ax.bar(x - bw/2, np.array(comp_means_wt) * 100, width=bw,
               color="#1f77b4", label="EZH2 WT", alpha=0.85)
        ax.bar(x + bw/2, np.array(comp_means_mut) * 100, width=bw,
               color="#d62728", label="EZH2 Mut", alpha=0.85)
        for i, p in enumerate(p_vals):
            if p is not None and not np.isnan(p) and p < 0.05:
                top = max(comp_means_wt[i] or 0, comp_means_mut[i] or 0) * 100
                ax.text(i, top + 2, "*", ha="center", va="bottom",
                        fontsize=14, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(PAPER_9_SHORT, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    plot_grouped(axes[0], "cd8_exh_frac",
                 "% TOX+/PD-1+ of CD8 T cells",
                 "(a) CD8 exhaustion by compartment\n(EZH2 WT vs Mut; *: MW p<0.05)")
    plot_grouped(axes[1], "treg_frac",
                 "Treg (% of typed cells in compartment)",
                 "(b) Treg distribution by compartment")

    fig.suptitle("EZH2 Mut vs WT — CD8 exhaustion and Treg distribution by compartment",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_ezh2_tcell_status.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
