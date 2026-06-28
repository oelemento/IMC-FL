#!/usr/bin/env python3
"""Grade follow-up analyses (six tests in one script).

Patient-level aggregation, BH within-family + combined, FOLL1/FOLL2/FOLL3A only.

  1. Ki-67 by grade (S-panel)        — sanity check vs histological grade
  2. CD21+ FDC network connectivity   — direct test of FDC scaffold breakdown
  3. CD8 T exhaustion by grade (T-panel) — links to paper's exhaustion topography
  4. CD14+ FDC fraction by grade     — links to paper's CD14+ FDC biology
  5. Compartment composition shift   — cell-type fractions WITHIN each compartment by grade
  6. EZH2 × grade interaction        — does EZH2 mutation modify the grade effect?

Cohort filtering follows CLAUDE.md guardrail #3 + EXCLUDE_ROIS + Biomax.
Same patient-level aggregation as prelim_grade_architecture.py.

Usage:
    .venv/bin/python scripts/grade_followups.py
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
from scipy.stats import kruskal, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

DEFAULT_MIN_CELLS_PER_ROI = 8000
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}
EZH2_COLORS = {"wt": "#7f7f7f", "mut": "#d62728"}


def is_tumor_core(sid: str) -> bool:
    s = str(sid).lower()
    if any(t in s for t in ("tonsil", "prostate", "kidney", "spleen", "adrenal")):
        return False
    if "_ton_" in s or "_adr_" in s or "_lym_" in s or "_lym " in s:
        return False
    if s.startswith("biomax"):
        return False
    if sid in EXCLUDE_ROIS:
        return False
    return True


def load_panel(h5ad_path: Path, biomarkers):
    with h5py.File(h5ad_path, "r") as f:
        sid_codes = f["obs/sample_id/codes"][:]
        sid_cats = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in f["obs/sample_id/categories"][:]])
        sample_id = sid_cats[sid_codes]
        ct_codes = f["obs/cell_type/codes"][:]
        ct_cats = np.array([c.decode() if isinstance(c, bytes) else c
                            for c in f["obs/cell_type/categories"][:]])
        cell_type = ct_cats[ct_codes]
        comp_codes = f["obs/compartment_name/codes"][:]
        comp_cats = np.array([c.decode() if isinstance(c, bytes) else c
                              for c in f["obs/compartment_name/categories"][:]])
        compartment = comp_cats[comp_codes]
        cx = f["obs/centroid_x"][:]
        cy = f["obs/centroid_y"][:]

        var_names = [v.decode() if isinstance(v, bytes) else v
                     for v in f["var/_index"][:]]
        biomarker_data = {}
        for b in biomarkers:
            if b in var_names:
                biomarker_data[b] = f["X"][:, var_names.index(b)]
            else:
                print(f"  WARN: marker '{b}' not in var names, skipping")

    return pd.DataFrame({
        "sample_id": sample_id, "cell_type": cell_type, "compartment": compartment,
        "x": cx, "y": cy, **biomarker_data,
    })


def join_grade(metrics_df, clinical_csv, grade_xlsx):
    clin = pd.read_csv(clinical_csv)[["slide_ID", "Sample_ID", "Patient_ID"]]
    grade = pd.read_excel(grade_xlsx).rename(
        columns={"FL ID": "Sample_ID", "DIAG": "grade"})[["Sample_ID", "grade", "EZH2"]]
    out = metrics_df.merge(clin, left_on="sample_id", right_on="slide_ID", how="left")
    out = out.merge(grade, on="Sample_ID", how="left")
    return out


def kw(df, metric):
    groups = [df.loc[df["grade"] == g, metric].dropna().values for g in GRADE_ORDER]
    if any(len(x) < 3 for x in groups):
        return np.nan, {g: np.nan for g in GRADE_ORDER}
    medians = {g: float(np.median(grp)) for g, grp in zip(GRADE_ORDER, groups)}
    # Guard against all-identical values (kruskal raises ValueError)
    if len(np.unique(np.concatenate(groups))) < 2:
        return np.nan, medians
    try:
        _, p = kruskal(*groups)
    except ValueError:
        return np.nan, medians
    return float(p), medians


def bh_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    out = np.full_like(pvals, np.nan)
    if not valid.any():
        return out
    p_valid = pvals[valid]
    n = len(p_valid)
    order = np.argsort(p_valid)
    q_ranked = p_valid[order] * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    out[valid] = q
    return out


def connected_component_sizes(xy, neighbor_dist):
    n = len(xy)
    if n == 0:
        return np.array([], dtype=int)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(neighbor_dist, output_type="ndarray")
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return pd.Series([find(i) for i in range(n)]).value_counts().values


# ────────── Per-ROI metric extractors ──────────

def s_panel_metrics(roi_df, *, ki67_thresh, cd14_thresh_fdc, min_cells):
    # Per CLAUDE.md guardrail #10 + #9: "Unassigned" is the S-panel non-typed label.
    # Some merges encode it as "Low quality / Unassigned"; filter both for safety.
    not_unassigned = ~roi_df["cell_type"].isin(["Unassigned", "Low quality / Unassigned"])
    n_typed = int(not_unassigned.sum())
    if n_typed < min_cells:
        return None

    out = {"n_typed": n_typed}

    # 1. Ki-67 by grade
    if "Ki-67" in roi_df.columns:
        out["ki67_pct_pos"] = float((roi_df["Ki-67"] > ki67_thresh).mean())
        out["ki67_mean"] = float(roi_df["Ki-67"].mean())

    # 2. CD21+ FDC network connectivity
    is_cd21_pos = roi_df["cell_type"] == "FDC"
    n_cd21 = int(is_cd21_pos.sum())
    out["n_cd21_cells"] = n_cd21
    if n_cd21 >= 10:
        sizes = connected_component_sizes(
            roi_df.loc[is_cd21_pos, ["x", "y"]].to_numpy(), neighbor_dist=30.0
        )
        big = sizes[sizes >= 10]
        out["cd21_n_components"] = int(len(big))
        out["cd21_largest_frac"] = float(big.max() / n_cd21) if len(big) else np.nan
        out["cd21_mean_component_size"] = float(big.mean()) if len(big) else np.nan
    else:
        out.update({"cd21_n_components": np.nan,
                    "cd21_largest_frac": np.nan,
                    "cd21_mean_component_size": np.nan})

    # 4. CD14+ FDC fraction (using p75 of CD14 within FDCs as threshold)
    if "CD14" in roi_df.columns and is_cd21_pos.any():
        fdc_cd14 = roi_df.loc[is_cd21_pos, "CD14"].values
        if len(fdc_cd14) >= 5:
            out["cd14pos_fdc_frac"] = float((fdc_cd14 > cd14_thresh_fdc).mean())
        else:
            out["cd14pos_fdc_frac"] = np.nan
    else:
        out["cd14pos_fdc_frac"] = np.nan

    # 5. Cell-type fraction within FDC network zone (and within other key compartments)
    for compart in ("FDC network zone", "B cell zone (BCL2+)", "T cell zone"):
        sub = roi_df[roi_df["compartment"] == compart]
        if len(sub) < 50:
            continue
        ct_frac = sub["cell_type"].value_counts(normalize=True)
        # Cell-type labels verified against S-panel categories:
        #   FDC, M2 Macrophages, Myeloid (S100A9+), B cells (PAX5+), B cells (BCL2+)
        for ct in ("FDC", "M2 Macrophages", "Myeloid (S100A9+)",
                   "B cells (PAX5+)", "B cells (BCL2+)"):
            key = f"in_{compart}_{ct}_frac".replace(" ", "_").replace("(", "").replace(")", "")
            out[key] = float(ct_frac.get(ct, 0.0))

    return out


def t_panel_metrics(roi_df, *, min_cells):
    n_typed = int((roi_df["cell_type"] != "Low quality / Unassigned").sum())
    if n_typed < min_cells:
        return None
    is_cd8 = roi_df["cell_type"].str.startswith("CD8 T")
    n_cd8 = int(is_cd8.sum())
    if n_cd8 < 20:
        return None
    is_exh = roi_df["cell_type"].isin(["CD8 T exhausted",
                                        "CD8 T pre-exhausted (TOX+)"])
    return {
        "n_typed": n_typed,
        "n_cd8": n_cd8,
        "cd8_exh_frac": float(is_exh.sum() / n_cd8),
    }


def per_roi_loop(df, fn, **kwargs):
    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = fn(sub, **kwargs)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    return pd.DataFrame(rows)


def patient_aggregate(metrics_df):
    metric_cols = [c for c in metrics_df.columns
                   if c not in {"sample_id", "slide_ID", "Sample_ID",
                                 "Patient_ID", "grade", "EZH2"}]
    return (metrics_df.groupby(["Patient_ID", "grade", "EZH2"], dropna=False)[metric_cols]
            .mean().reset_index())


# ────────── Main ──────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s-panel", default="output/all_TMA_S_utag_ct_merged.h5ad")
    p.add_argument("--t-panel", default="output/all_TMA_T_utag_ct_merged.h5ad")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--out", default="output/grade_arch")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # ===== S-panel =====
    print("\n========== S-PANEL ==========")
    s_df = load_panel(Path(args.s_panel), ["Ki-67", "CD14", "CD21"])
    s_df = s_df[s_df["sample_id"].apply(is_tumor_core)].copy()
    s_df["sample_id"] = s_df["sample_id"].apply(normalize_sample_id)
    print(f"  cells={len(s_df):,}, ROIs={s_df['sample_id'].nunique()}")

    # Cohort-wide thresholds
    ki67_thresh = float(np.quantile(s_df["Ki-67"].values, 0.90)) if "Ki-67" in s_df else np.nan
    fdc_cd14 = s_df.loc[s_df["cell_type"] == "FDC", "CD14"].values
    cd14_thresh_fdc = float(np.quantile(fdc_cd14, 0.75)) if len(fdc_cd14) else np.nan
    print(f"  Ki-67 p90 (scaled) = {ki67_thresh:.3f}")
    print(f"  CD14 p75 within FDC (scaled) = {cd14_thresh_fdc:.3f}")

    s_metrics = per_roi_loop(s_df, s_panel_metrics,
                             ki67_thresh=ki67_thresh,
                             cd14_thresh_fdc=cd14_thresh_fdc,
                             min_cells=args.min_cells)
    s_metrics = join_grade(s_metrics, Path(args.clinical), Path(args.grade))
    s_metrics = s_metrics[s_metrics["grade"].isin(GRADE_ORDER)].copy()
    print(f"  S-panel ROIs with grade: {len(s_metrics)}")

    # ===== T-panel =====
    print("\n========== T-PANEL ==========")
    t_df = load_panel(Path(args.t_panel), ["TOX", "PD_1"])
    t_df = t_df[t_df["sample_id"].apply(is_tumor_core)].copy()
    t_df["sample_id"] = t_df["sample_id"].apply(normalize_sample_id)
    print(f"  cells={len(t_df):,}, ROIs={t_df['sample_id'].nunique()}")

    t_metrics = per_roi_loop(t_df, t_panel_metrics, min_cells=args.min_cells)
    t_metrics = join_grade(t_metrics, Path(args.clinical), Path(args.grade))
    t_metrics = t_metrics[t_metrics["grade"].isin(GRADE_ORDER)].copy()
    print(f"  T-panel ROIs with grade: {len(t_metrics)}")

    # ===== Patient-level aggregation, separate per panel =====
    s_pt = patient_aggregate(s_metrics)
    t_pt = patient_aggregate(t_metrics)
    print(f"\n  S-panel patients: {len(s_pt)} | grade counts: {s_pt['grade'].value_counts().to_dict()}")
    print(f"  T-panel patients: {len(t_pt)} | grade counts: {t_pt['grade'].value_counts().to_dict()}")

    s_pt.to_csv(out_dir / "grade_followups_s_per_patient.csv", index=False)
    t_pt.to_csv(out_dir / "grade_followups_t_per_patient.csv", index=False)

    # ===== Family-wise BH for the 4 headline grade tests =====
    headline_metrics = [
        ("ki67_pct_pos",          "Ki-67+ fraction (S)",            "S"),
        ("cd21_n_components",     "CD21 FDC components per ROI (S)", "S"),
        ("cd21_largest_frac",     "Largest CD21 component / total (S)", "S"),
        ("cd14pos_fdc_frac",      "CD14+ fraction of FDCs (S)",     "S"),
        ("cd8_exh_frac",          "CD8 T exhausted fraction (T)",   "T"),
    ]
    print("\n=== Headline grade tests (KW + within-family BH, n_metrics=5) ===")
    pvals, medians = [], []
    for k, _, panel in headline_metrics:
        df_pt = s_pt if panel == "S" else t_pt
        p_v, med = kw(df_pt, k)
        pvals.append(p_v); medians.append(med)
    qvals = bh_correct(pvals)
    headline_summary = []
    print(f"{'metric':45s} {'p':>10s} {'q (BH)':>10s}  medians (FOLL1/2/3A)")
    for (k, label, panel), p_v, q_v, med in zip(headline_metrics, pvals, qvals, medians):
        med_str = " / ".join(f"{med[g]:.3g}" for g in GRADE_ORDER)
        flag = " *" if q_v < 0.05 else ""
        print(f"  {label:45s} {p_v:10.4g} {q_v:10.4g}  {med_str}{flag}")
        headline_summary.append({"metric": k, "label": label, "panel": panel,
                                  "p_KW": p_v, "q_BH": q_v,
                                  **{f"med_{g}": med[g] for g in GRADE_ORDER}})
    pd.DataFrame(headline_summary).to_csv(out_dir / "grade_followups_headline_summary.csv",
                                           index=False)

    # ===== Compartment composition shift (per-compartment cell-type fractions) =====
    print("\n=== Cell-type fraction shifts WITHIN compartments (S-panel) ===")
    in_cols = [c for c in s_pt.columns if c.startswith("in_")]
    pvals_in, medians_in = [], []
    for c in in_cols:
        p_v, med = kw(s_pt, c)
        pvals_in.append(p_v); medians_in.append(med)
    qvals_in = bh_correct(pvals_in)
    print(f"{'metric':60s} {'p':>10s} {'q (BH)':>10s}  medians (FOLL1/2/3A)")
    comp_shift_summary = []
    for k, p_v, q_v, med in zip(in_cols, pvals_in, qvals_in, medians_in):
        med_str = " / ".join(f"{med[g]:.3g}" for g in GRADE_ORDER)
        flag = " *" if q_v < 0.05 else ""
        if p_v < 0.05 or q_v < 0.20:
            print(f"  {k:60s} {p_v:10.4g} {q_v:10.4g}  {med_str}{flag}")
        comp_shift_summary.append({"metric": k, "p_KW": p_v, "q_BH": q_v,
                                    **{f"med_{g}": med[g] for g in GRADE_ORDER}})
    pd.DataFrame(comp_shift_summary).to_csv(
        out_dir / "grade_followups_compartment_shift.csv", index=False)

    # ===== EZH2 × grade interaction =====
    print("\n=== EZH2 × grade interaction (compartment Shannon entropy as response) ===")
    # Recompute compartment shannon from S-panel patient table
    # (Use the existing prelim_grade output if available)
    cb_path = out_dir / "grade_compartment_biomarkers_per_patient.csv"
    if cb_path.exists():
        cb = pd.read_csv(cb_path)
        # Patient_ID is in cb; merge EZH2 from grade file. A patient may appear
        # in BOTH T1 and T2 with different EZH2 calls — collapse to one row per
        # Patient_ID to prevent duplicate rows downstream.
        grade_xlsx = pd.read_excel(args.grade).rename(columns={"FL ID": "Sample_ID"})
        clin = pd.read_csv(args.clinical)[["Sample_ID", "Patient_ID"]]
        ezh2_per_sample = grade_xlsx.merge(clin, on="Sample_ID")[
            ["Patient_ID", "EZH2"]].dropna()
        # Mark patients with conflicting EZH2 across timepoints as 'mixed'
        def collapse(group):
            vals = set(group)
            if len(vals) == 1:
                return next(iter(vals))
            return "mixed"
        ezh2 = (ezh2_per_sample.groupby("Patient_ID")["EZH2"]
                .agg(collapse).reset_index())
        n_mixed = int((ezh2["EZH2"] == "mixed").sum())
        if n_mixed:
            print(f"  NOTE: {n_mixed} patient(s) had conflicting EZH2 across "
                  f"timepoints — labelled 'mixed' and excluded from interaction test")
        cb_ezh2 = cb.merge(ezh2, on="Patient_ID", how="left")
        cb_ezh2 = cb_ezh2[cb_ezh2["grade"].isin(GRADE_ORDER)
                           & cb_ezh2["EZH2"].isin(["wt", "mut"])]
        ct = pd.crosstab(cb_ezh2["grade"], cb_ezh2["EZH2"])
        print(f"  EZH2 × grade table:")
        print(ct.to_string())
        print()
        # Per cell, KW within EZH2 stratum
        print(f"  Compartment Shannon entropy by EZH2 × grade:")
        ezh2_results = []
        for ez in ("wt", "mut"):
            sub = cb_ezh2[cb_ezh2["EZH2"] == ez]
            p_v, med = kw(sub, "shannon_compartment")
            n_str = " / ".join(str(int((sub["grade"] == g).sum())) for g in GRADE_ORDER)
            med_str = " / ".join(f"{med[g]:.3g}" for g in GRADE_ORDER)
            print(f"    EZH2={ez:3s} (n={n_str}): KW p = {p_v:.4g}, "
                  f"medians = {med_str}")
            ezh2_results.append({"ezh2": ez, "p_KW": p_v,
                                  **{f"med_{g}": med[g] for g in GRADE_ORDER},
                                  **{f"n_{g}": int((sub["grade"] == g).sum())
                                     for g in GRADE_ORDER}})
        pd.DataFrame(ezh2_results).to_csv(out_dir / "grade_followups_ezh2_interaction.csv",
                                            index=False)
    else:
        print(f"  SKIP: {cb_path} not found — run grade_compartment_biomarkers.py first")

    # ===== Figure: 6-panel summary =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    rng = np.random.default_rng(0)

    def boxplot(ax, df_pt, metric, label, q=None):
        data = [df_pt.loc[df_pt.grade == g, metric].dropna().values for g in GRADE_ORDER]
        bp = ax.boxplot(data, tick_labels=GRADE_ORDER, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GRADE_ORDER):
            patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.55)
        for i, (g, vals) in enumerate(zip(GRADE_ORDER, data)):
            xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(xs, vals, color=GRADE_COLORS[g], s=10, alpha=0.7,
                       edgecolor="white", linewidth=0.4, zorder=3)
        if all(len(x) >= 3 for x in data):
            _, pv = kruskal(*data)
            tag = f", q={q:.3g}" + (" *" if q is not None and q < 0.05 else "") if q is not None else ""
            ax.set_title(f"{label}\np={pv:.3g}{tag}", fontsize=11)
        else:
            ax.set_title(label, fontsize=11)
        ax.set_xlabel("Grade")
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Headline 5
    panel_df = {"S": s_pt, "T": t_pt}
    for ax, ((k, label, panel), q) in zip(axes.flat[:5], zip(headline_metrics, qvals)):
        boxplot(ax, panel_df[panel], k, label, q=q)

    # Sixth panel: EZH2 stratification of compartment Shannon (if available)
    ax = axes.flat[5]
    if cb_path.exists():
        cb_ezh2_use = cb_ezh2[cb_ezh2["EZH2"].isin(["wt", "mut"])].copy()
        # Plot each EZH2 stratum side-by-side per grade
        rng = np.random.default_rng(1)
        for ez_idx, ez in enumerate(("wt", "mut")):
            for gi, g in enumerate(GRADE_ORDER):
                sub = cb_ezh2_use[(cb_ezh2_use.EZH2 == ez) & (cb_ezh2_use.grade == g)]
                xc = gi * 3 + ez_idx + 1
                if len(sub):
                    ax.boxplot([sub["shannon_compartment"].values], positions=[xc],
                               widths=0.6, patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor=EZH2_COLORS[ez], alpha=0.55))
                    xs = xc + (rng.random(len(sub)) - 0.5) * 0.3
                    ax.scatter(xs, sub["shannon_compartment"], color=EZH2_COLORS[ez],
                               s=10, alpha=0.7, edgecolor="white", linewidth=0.4, zorder=3)
        # x-tick labels: grade-EZH2
        positions = []
        labels = []
        for gi, g in enumerate(GRADE_ORDER):
            for ez_idx, ez in enumerate(("wt", "mut")):
                positions.append(gi * 3 + ez_idx + 1)
                labels.append(f"{g}\n{ez}")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title("Compartment Shannon entropy: grade × EZH2", fontsize=11)
        ax.set_ylabel("Shannon entropy")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    else:
        ax.text(0.5, 0.5, "(EZH2 panel skipped)", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")

    fig.suptitle("FL grade follow-ups: Ki-67, CD21 connectivity, exhaustion, "
                 "CD14+ FDC, EZH2 × grade", fontsize=13, y=1.0)
    plt.tight_layout()
    out = out_dir / "fig_grade_followups.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
