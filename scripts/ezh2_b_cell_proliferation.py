#!/usr/bin/env python3
"""Wendy EZH2 mut vs WT — slide 7: B cell proliferation near CD14+ FDCs.

Reuses the paper's distance-to-CD14+-FDC analysis (Fig 5 / Fig S9), stratifies
by EZH2 mut vs WT.

For each follicular B cell (B cells (BCL2+) or B cells (PAX5+)) within the
same ROI, compute its distance to the nearest CD14+ FDC. Define:
  - 'close'  : <= 30 micrometres
  - 'distant': > 30 micrometres

Compare mean Ki-67 expression in close vs distant B cells, separately for
EZH2 WT and EZH2 Mut patients. Test whether Mut FDC-proximal B cells have
higher proliferation than WT proximal B cells (Wendy's question).

CD14+ FDCs are defined as cells with cell_type == "FDC" AND CD14 expression
above the patient-pooled p75 threshold (consistent with fig_fdc_cd14_biology).

The scRNA panel in Wendy's slide (BAFF/APRIL/TGF-beta1/IL-6) is NOT
stratifiable by EZH2 because Han 2022 scRNA dataset lacks EZH2 status per
donor — explicitly skipped here.

Outputs:
  output/ezh2/b_proliferation/fig_ezh2_b_proliferation.png
  output/ezh2/b_proliferation/ki67_close_vs_distant_per_patient.csv
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
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

MIN_CELLS_PER_ROI = 8000
UNASSIGNED_CT = ["Unassigned", "Low quality / Unassigned"]
FOLLICULAR_B = ["B cells (BCL2+)", "B cells (PAX5+)"]
DIST_THRESHOLD_UM = 30.0
CD14_PERCENTILE = 75  # threshold for CD14+ FDC
MIN_CD14_FDC_PER_ROI = 3


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
    ap.add_argument("--out", default="output/ezh2/b_proliferation")
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
        var_names = [v.decode() for v in f["var/_index"][:]]
        ki67_idx = var_names.index("Ki-67")
        cd14_idx = var_names.index("CD14")
        X = f["X"]
        n_obs = len(sample_id)
        ki67 = X[:, ki67_idx] if X.shape == (n_obs, len(var_names)) else np.array(X[:, ki67_idx])
        cd14 = X[:, cd14_idx] if X.shape == (n_obs, len(var_names)) else np.array(X[:, cd14_idx])
        cx = f["obs/centroid_x"][:]
        cy = f["obs/centroid_y"][:]

    df = pd.DataFrame({"sample_id": sample_id, "cell_type": cell_type,
                       "cx": cx, "cy": cy, "Ki67": ki67, "CD14": cd14})
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

    # CD14+ FDC threshold: p75 of CD14 across all FDC cells (cohort-wide)
    fdc_cells = df[df.cell_type == "FDC"]
    cd14_thr = float(np.quantile(fdc_cells.CD14.values, CD14_PERCENTILE / 100))
    print(f"  CD14+ FDC threshold (p{CD14_PERCENTILE} of FDC CD14): {cd14_thr:.3f}")
    print(f"  Total FDC cells: {len(fdc_cells):,}, CD14+ FDC: {(fdc_cells.CD14 > cd14_thr).sum():,}")

    # Per-ROI: distance from each follicular B cell to nearest CD14+ FDC
    print(f"\nComputing nearest CD14+ FDC distance per follicular B cell ...")
    rows = []
    for sid, sub in df.groupby("sid_norm"):
        fdc_sub = sub[(sub.cell_type == "FDC") & (sub.CD14 > cd14_thr)]
        if len(fdc_sub) < MIN_CD14_FDC_PER_ROI:
            continue
        b_sub = sub[sub.cell_type.isin(FOLLICULAR_B)]
        if len(b_sub) < 50:
            continue
        fdc_coords = np.column_stack([fdc_sub.cx.values, fdc_sub.cy.values])
        b_coords = np.column_stack([b_sub.cx.values, b_sub.cy.values])
        tree = cKDTree(fdc_coords)
        dists, _ = tree.query(b_coords, k=1)
        close_mask = dists <= DIST_THRESHOLD_UM
        if close_mask.sum() < 20 or (~close_mask).sum() < 20:
            continue
        ki67_close = float(b_sub.Ki67.values[close_mask].mean())
        ki67_distant = float(b_sub.Ki67.values[~close_mask].mean())
        rows.append({
            "sid": sid, "EZH2": sub.EZH2.iloc[0],
            "Patient_ID": sub.Patient_ID.iloc[0],
            "n_close": int(close_mask.sum()),
            "n_distant": int((~close_mask).sum()),
            "ki67_close_mean": ki67_close,
            "ki67_distant_mean": ki67_distant,
            "delta_close_minus_distant": ki67_close - ki67_distant,
        })
    roi_df = pd.DataFrame(rows)
    print(f"  ROIs with both close+distant B cells: {len(roi_df)} ({roi_df.EZH2.value_counts().to_dict()})")

    pt_df = (roi_df.groupby(["Patient_ID", "EZH2"])
             [["ki67_close_mean", "ki67_distant_mean", "delta_close_minus_distant",
                "n_close", "n_distant"]]
             .mean().reset_index())
    pt_df.to_csv(out_dir / "ki67_close_vs_distant_per_patient.csv", index=False)
    n_wt = (pt_df.EZH2 == "wt").sum(); n_mut = (pt_df.EZH2 == "mut").sum()
    print(f"  Patients: WT={n_wt}, Mut={n_mut}")

    # MW tests
    def mw_test(a, b):
        if len(a) < 3 or len(b) < 3:
            return np.nan
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            return float(p)
        except ValueError:
            return np.nan

    # 1) Within each EZH2 group: close vs distant — PAIRED by patient.
    # Use Wilcoxon signed-rank on the per-patient delta (close - distant) vs 0.
    def wilcoxon_test(deltas):
        d = np.asarray(deltas)
        d = d[~np.isnan(d)]
        if len(d) < 3:
            return np.nan
        # Wilcoxon requires non-zero differences; if everything is zero, return 1.
        if not np.any(d != 0):
            return 1.0
        try:
            _, p = wilcoxon(d, alternative="two-sided")
            return float(p)
        except ValueError:
            return np.nan
    p_wt_close_vs_dist = wilcoxon_test(pt_df.loc[pt_df.EZH2 == "wt", "delta_close_minus_distant"].values)
    p_mut_close_vs_dist = wilcoxon_test(pt_df.loc[pt_df.EZH2 == "mut", "delta_close_minus_distant"].values)

    # 2) Within each distance: WT vs Mut
    p_close_wt_vs_mut = mw_test(pt_df.loc[pt_df.EZH2 == "wt", "ki67_close_mean"].values,
                                 pt_df.loc[pt_df.EZH2 == "mut", "ki67_close_mean"].values)
    p_distant_wt_vs_mut = mw_test(pt_df.loc[pt_df.EZH2 == "wt", "ki67_distant_mean"].values,
                                   pt_df.loc[pt_df.EZH2 == "mut", "ki67_distant_mean"].values)

    # 3) Delta close-distant: WT vs Mut
    p_delta = mw_test(pt_df.loc[pt_df.EZH2 == "wt", "delta_close_minus_distant"].values,
                       pt_df.loc[pt_df.EZH2 == "mut", "delta_close_minus_distant"].values)

    print(f"\n  Wilcoxon close-vs-distant within WT (paired): p={p_wt_close_vs_dist:.4g}")
    print(f"  Wilcoxon close-vs-distant within Mut (paired): p={p_mut_close_vs_dist:.4g}")
    print(f"  MW WT-vs-Mut on close cells:    p={p_close_wt_vs_mut:.4g}")
    print(f"  MW WT-vs-Mut on distant cells:  p={p_distant_wt_vs_mut:.4g}")
    print(f"  MW WT-vs-Mut on (close-distant) delta: p={p_delta:.4g}")

    means = pt_df.groupby("EZH2")[["ki67_close_mean", "ki67_distant_mean",
                                    "delta_close_minus_distant"]].mean()
    print("\nMeans:")
    print(means.to_string())

    # ===================================================================
    # Figure: 1x2 — (a) Ki-67 close/distant by EZH2 group;
    #               (b) Delta close-distant by EZH2.
    # The scRNA part of Wendy's slide 7 is not stratifiable by EZH2; that
    # caveat goes in the email/caption, not in a text-only panel.
    # ===================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    rng = np.random.default_rng(0)

    # (a) Grouped boxplot: 4 conditions (WT close, WT distant, Mut close, Mut distant)
    ax = axes[0]
    groups = [
        ("WT close", pt_df.loc[pt_df.EZH2 == "wt", "ki67_close_mean"].values, "#1f77b4"),
        ("WT distant", pt_df.loc[pt_df.EZH2 == "wt", "ki67_distant_mean"].values, "#aec7e8"),
        ("Mut close", pt_df.loc[pt_df.EZH2 == "mut", "ki67_close_mean"].values, "#d62728"),
        ("Mut distant", pt_df.loc[pt_df.EZH2 == "mut", "ki67_distant_mean"].values, "#f7b6b1"),
    ]
    data = [g[1] for g in groups]
    labels = [f"{g[0]}\n(n={len(g[1])})" for g in groups]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55, showfliers=False)
    for patch, (_, _, col) in zip(bp["boxes"], groups):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    for i, (_, vals, col) in enumerate(groups):
        xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.2
        ax.scatter(xs, vals, color=col, s=22, alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3)
    # Annotate within-group close-vs-distant tests
    ax.text(1.5, ax.get_ylim()[1] * 0.92,
            f"close vs distant\nWilcoxon (paired)\n(WT) p={p_wt_close_vs_dist:.3g}",
            ha="center", fontsize=9, color="#1f77b4")
    ax.text(3.5, ax.get_ylim()[1] * 0.92,
            f"close vs distant\nWilcoxon (paired)\n(Mut) p={p_mut_close_vs_dist:.3g}",
            ha="center", fontsize=9, color="#d62728")
    ax.set_ylabel("Mean Ki-67 (scaled, per patient)")
    ax.set_title(
        f"(a) B cell Ki-67 near vs distant from CD14+ FDCs by EZH2\n"
        f"≤{int(DIST_THRESHOLD_UM)}µm = close; CD14+ FDC threshold = p{CD14_PERCENTILE} of FDC CD14",
        fontsize=10,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (b) Delta close - distant: WT vs Mut
    ax = axes[1]
    a = pt_df.loc[pt_df.EZH2 == "wt", "delta_close_minus_distant"].values
    b = pt_df.loc[pt_df.EZH2 == "mut", "delta_close_minus_distant"].values
    data2 = [a, b]
    labels2 = [f"WT (n={len(a)})", f"Mut (n={len(b)})"]
    bp = ax.boxplot(data2, tick_labels=labels2, patch_artist=True, widths=0.55, showfliers=False)
    bp["boxes"][0].set_facecolor("#1f77b4"); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#d62728"); bp["boxes"][1].set_alpha(0.7)
    for i, vals in enumerate(data2):
        xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.2
        col = "#1f77b4" if i == 0 else "#d62728"
        ax.scatter(xs, vals, color=col, s=28, alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(0, color="gray", lw=0.7, linestyle="--")
    star = " *" if (not np.isnan(p_delta)) and p_delta < 0.05 else ""
    ax.set_title(f"(b) Δ Ki-67 (close − distant) by EZH2\nMW p={p_delta:.3g}{star}", fontsize=10)
    ax.set_ylabel("Δ Ki-67 (close − distant), per patient")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("EZH2 Mut vs WT — B cell proliferation near CD14+ FDCs (S-panel IMC)",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    out = out_dir / "fig_ezh2_b_proliferation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
