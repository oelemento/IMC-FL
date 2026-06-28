#!/usr/bin/env python3
"""IDO compartment-specific expression and survival analysis.

Hypothesis: IDO expression on myeloid cells is compartment-specific and
independently predicts clinical outcome in FL, beyond VISTA alone.

Sub-questions:
  Q1. IDO+ fraction by compartment and cell type
  Q2. Per-ROI IDO intensity on myeloid → PFS/OS survival
  Q3. Intrafollicular vs interfollicular IDO → survival
  Q4. IDO vs VISTA: independent or redundant prognostic value?
  Q5. IDO+VISTA+ dual vs single-positive → additive effect?
"""

import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import load_clinical, normalize_sample_id

CONTROL_PATTERNS = ["tonsil", "prostate", "kidney", "spleen", "adrenal",
                    "_Ton_", "_Adr_"]

def is_control(sid):
    s = str(sid).lower()
    for pat in CONTROL_PATTERNS:
        if pat.lower() in s:
            return True
    return False


def load_data(s_panel_path, s_utag_path):
    """Load S-panel data with UTAG compartments."""
    print("Loading S-panel global...")
    sg = ad.read_h5ad(s_panel_path, backed="r")

    print("Loading S-panel UTAG...")
    su = ad.read_h5ad(s_utag_path, backed="r")

    # Extract marker data for relevant markers
    markers = list(sg.var_names)
    ido_idx = markers.index("IDO")
    vista_idx = markers.index("VISTA")

    print(f"  IDO index: {ido_idx}, VISTA index: {vista_idx}")

    # Build obs DataFrame
    obs = sg.obs[["sample_id", "cell_type"]].copy()
    obs["IDO"] = np.asarray(sg.X[:, ido_idx]).flatten()
    obs["VISTA"] = np.asarray(sg.X[:, vista_idx]).flatten()

    # Add UTAG compartments
    utag_obs = su.obs[["compartment_name"]].copy()
    utag_obs.rename(columns={"compartment_name": "compartment_merged"}, inplace=True)
    obs = obs.join(utag_obs, how="left")

    # Filter controls
    obs["is_control"] = obs["sample_id"].apply(is_control)
    obs = obs[~obs["is_control"]].copy()
    print(f"  After control filter: {len(obs):,} cells")

    # Classify compartments
    foll_comps = ["GC core", "B cell zone (BCL2+)", "FDC network zone",
                  "B cell zone (mixed)", "Mantle zone"]
    obs["is_follicular"] = obs["compartment_merged"].isin(foll_comps)

    # Classify myeloid (use actual S-panel v8 cell type names)
    myeloid_types = ["FDC", "M1 Macrophages", "M2 Macrophages", "Macrophages",
                     "Dendritic cells", "Myeloid (S100A9+)", "pDC",
                     "Histiocytes (CD44hi)"]
    obs["is_myeloid"] = obs["cell_type"].isin(myeloid_types)

    return obs, markers


def compute_roi_stats(obs):
    """Compute per-ROI IDO and VISTA statistics on myeloid cells."""
    myeloid = obs[obs["is_myeloid"]].copy()

    # Thresholds (z > 0.5 on scaled data, matching H3b)
    myeloid["IDO_pos"] = myeloid["IDO"] > 0.5
    myeloid["VISTA_pos"] = myeloid["VISTA"] > 0.5
    myeloid["dual_pos"] = myeloid["IDO_pos"] & myeloid["VISTA_pos"]
    myeloid["IDO_only"] = myeloid["IDO_pos"] & ~myeloid["VISTA_pos"]
    myeloid["VISTA_only"] = ~myeloid["IDO_pos"] & myeloid["VISTA_pos"]

    # Per-ROI stats
    roi = myeloid.groupby("sample_id").agg(
        n_myeloid=("IDO", "size"),
        ido_mean=("IDO", "mean"),
        vista_mean=("VISTA", "mean"),
        ido_frac=("IDO_pos", "mean"),
        vista_frac=("VISTA_pos", "mean"),
        dual_frac=("dual_pos", "mean"),
        ido_only_frac=("IDO_only", "mean"),
        vista_only_frac=("VISTA_only", "mean"),
    ).reset_index()

    # Follicular-specific stats
    myeloid_foll = myeloid[myeloid["is_follicular"]]
    myeloid_inter = myeloid[~myeloid["is_follicular"]]

    foll_stats = myeloid_foll.groupby("sample_id").agg(
        ido_mean_foll=("IDO", "mean"),
        vista_mean_foll=("VISTA", "mean"),
        ido_frac_foll=("IDO_pos", "mean"),
        vista_frac_foll=("VISTA_pos", "mean"),
        dual_frac_foll=("dual_pos", "mean"),
        n_myeloid_foll=("IDO", "size"),
    ).reset_index()

    inter_stats = myeloid_inter.groupby("sample_id").agg(
        ido_mean_inter=("IDO", "mean"),
        vista_mean_inter=("VISTA", "mean"),
        ido_frac_inter=("IDO_pos", "mean"),
        vista_frac_inter=("VISTA_pos", "mean"),
        dual_frac_inter=("dual_pos", "mean"),
        n_myeloid_inter=("IDO", "size"),
    ).reset_index()

    roi = roi.merge(foll_stats, on="sample_id", how="left")
    roi = roi.merge(inter_stats, on="sample_id", how="left")

    # Min myeloid filter
    roi = roi[roi["n_myeloid"] >= 20].copy()
    print(f"  ROIs with ≥20 myeloid: {len(roi)}")

    return roi


def merge_clinical(roi):
    """Merge ROI stats with clinical data."""
    clin = load_clinical()

    # Normalize sample_id for matching
    roi["slide_ID"] = roi["sample_id"].apply(normalize_sample_id)

    # Exclude Biomax (no clinical data)
    roi = roi[~roi["sample_id"].str.contains("Biomax", case=False)].copy()

    # Merge at ROI level first
    clin_cols = ["slide_ID", "Patient_ID", "Overall survival (y)", "CODE_OS",
                 "Progression free survival (y)", "CODE_PFS", "FLIPI",
                 "Transformation"]
    clin_sub = clin[[c for c in clin_cols if c in clin.columns]].drop_duplicates(
        subset="slide_ID", keep="first")
    roi = roi.merge(clin_sub, on="slide_ID", how="inner")

    # Rename for convenience
    rename = {
        "Overall survival (y)": "OS_years",
        "CODE_OS": "OS_event",
        "Progression free survival (y)": "PFS_years",
        "CODE_PFS": "PFS_event",
    }
    roi.rename(columns={k: v for k, v in rename.items() if k in roi.columns},
               inplace=True)

    # Average across ROIs per patient
    num_cols = ["ido_mean", "vista_mean", "ido_frac", "vista_frac",
                "dual_frac", "ido_only_frac", "vista_only_frac",
                "ido_mean_foll", "vista_mean_foll", "ido_frac_foll",
                "vista_frac_foll", "dual_frac_foll",
                "ido_mean_inter", "vista_mean_inter", "ido_frac_inter",
                "vista_frac_inter", "dual_frac_inter"]
    existing_num = [c for c in num_cols if c in roi.columns]

    agg_dict = {c: "mean" for c in existing_num}
    agg_dict["sample_id"] = "size"

    patient = roi.groupby("Patient_ID").agg(agg_dict).reset_index()
    patient.rename(columns={"sample_id": "n_rois"}, inplace=True)

    # Re-merge clinical (one row per patient)
    clin_patient = roi.groupby("Patient_ID").first()[
        [c for c in ["OS_years", "OS_event", "PFS_years", "PFS_event",
                      "FLIPI", "Transformation"] if c in roi.columns]
    ].reset_index()
    merged = patient.merge(clin_patient, on="Patient_ID", how="inner")

    print(f"  Patients with clinical data: {len(merged)}")
    return merged


def run_survival_correlations(merged):
    """Spearman correlations of IDO/VISTA metrics with PFS and OS."""
    outcomes = []
    for surv_col, event_col, label in [
        ("PFS_years", "PFS_event", "PFS"),
        ("OS_years", "OS_event", "OS"),
    ]:
        if surv_col not in merged.columns:
            continue
        valid = merged.dropna(subset=[surv_col, event_col])
        if len(valid) < 20:
            continue

        metrics = [
            ("IDO mean (all myeloid)", "ido_mean"),
            ("VISTA mean (all myeloid)", "vista_mean"),
            ("IDO+ fraction", "ido_frac"),
            ("VISTA+ fraction", "vista_frac"),
            ("IDO+VISTA+ fraction", "dual_frac"),
            ("IDO-only fraction", "ido_only_frac"),
            ("VISTA-only fraction", "vista_only_frac"),
            ("IDO mean (follicular)", "ido_mean_foll"),
            ("IDO mean (interfollicular)", "ido_mean_inter"),
            ("VISTA mean (follicular)", "vista_mean_foll"),
            ("VISTA mean (interfollicular)", "vista_mean_inter"),
            ("IDO+ frac (follicular)", "ido_frac_foll"),
            ("IDO+ frac (interfollicular)", "ido_frac_inter"),
            ("Dual+ frac (follicular)", "dual_frac_foll"),
            ("Dual+ frac (interfollicular)", "dual_frac_inter"),
        ]

        for name, col in metrics:
            v = valid.dropna(subset=[col])
            if len(v) < 20:
                outcomes.append((label, name, len(v), np.nan, np.nan))
                continue
            rho, p = stats.spearmanr(v[col], v[surv_col])
            outcomes.append((label, name, len(v), rho, p))

    df = pd.DataFrame(outcomes, columns=["Outcome", "Metric", "N", "rho", "p"])
    return df


def run_km_analysis(merged):
    """Kaplan-Meier + log-rank for IDO high/low."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    results = []
    for surv_col, event_col, label in [
        ("PFS_years", "PFS_event", "PFS"),
        ("OS_years", "OS_event", "OS"),
    ]:
        if surv_col not in merged.columns:
            continue
        valid = merged.dropna(subset=[surv_col, event_col])

        for metric, metric_label in [
            ("ido_frac", "IDO+ frac (all)"),
            ("ido_frac_foll", "IDO+ frac (follicular)"),
            ("ido_frac_inter", "IDO+ frac (interfollicular)"),
            ("dual_frac", "IDO+VISTA+ frac"),
            ("vista_frac", "VISTA+ frac (all)"),
        ]:
            v = valid.dropna(subset=[metric])
            if len(v) < 20:
                continue
            med = v[metric].median()
            hi = v[v[metric] >= med]
            lo = v[v[metric] < med]

            lr = logrank_test(hi[surv_col], lo[surv_col],
                              hi[event_col], lo[event_col])
            results.append({
                "outcome": label,
                "metric": metric_label,
                "n": len(v),
                "median_cutoff": med,
                "logrank_p": lr.p_value,
            })

    return pd.DataFrame(results)


def run_cox_multivariate(merged):
    """Cox PH: IDO + VISTA + FLIPI → PFS."""
    from lifelines import CoxPHFitter

    results = []
    for surv_col, event_col, label in [
        ("PFS_years", "PFS_event", "PFS"),
        ("OS_years", "OS_event", "OS"),
    ]:
        if surv_col not in merged.columns:
            continue

        # Model 1: IDO alone
        for metric, mlabel in [
            ("ido_frac", "IDO+ frac"),
            ("vista_frac", "VISTA+ frac"),
            ("dual_frac", "IDO+VISTA+ frac"),
            ("ido_frac_foll", "IDO+ frac (foll)"),
            ("ido_frac_inter", "IDO+ frac (inter)"),
        ]:
            cols = [surv_col, event_col, metric]
            if "FLIPI" in merged.columns:
                cols.append("FLIPI")
            v = merged.dropna(subset=cols)
            if len(v) < 20:
                continue
            try:
                cph = CoxPHFitter()
                covariates = [metric]
                if "FLIPI" in v.columns:
                    covariates.append("FLIPI")
                cph.fit(v[covariates + [surv_col, event_col]],
                        duration_col=surv_col, event_col=event_col)
                s = cph.summary
                hr = s.loc[metric, "exp(coef)"]
                p = s.loc[metric, "p"]
                ci_lo = s.loc[metric, "exp(coef) lower 95%"]
                ci_hi = s.loc[metric, "exp(coef) upper 95%"]
                results.append({
                    "outcome": label,
                    "metric": mlabel,
                    "model": "univariate" if "FLIPI" not in covariates else "+FLIPI",
                    "HR": hr, "CI_lo": ci_lo, "CI_hi": ci_hi,
                    "p": p, "n": len(v),
                })
            except Exception as e:
                print(f"  Cox failed for {mlabel}/{label}: {e}")

        # Model 2: IDO + VISTA together
        cols = [surv_col, event_col, "ido_frac", "vista_frac"]
        if "FLIPI" in merged.columns:
            cols.append("FLIPI")
        v = merged.dropna(subset=cols)
        if len(v) >= 20:
            try:
                cph = CoxPHFitter()
                covariates = ["ido_frac", "vista_frac"]
                if "FLIPI" in v.columns:
                    covariates.append("FLIPI")
                cph.fit(v[covariates + [surv_col, event_col]],
                        duration_col=surv_col, event_col=event_col)
                for m in ["ido_frac", "vista_frac"]:
                    s = cph.summary
                    results.append({
                        "outcome": label,
                        "metric": f"{m} (joint model)",
                        "model": "IDO+VISTA" + ("+FLIPI" if "FLIPI" in covariates else ""),
                        "HR": s.loc[m, "exp(coef)"],
                        "CI_lo": s.loc[m, "exp(coef) lower 95%"],
                        "CI_hi": s.loc[m, "exp(coef) upper 95%"],
                        "p": s.loc[m, "p"],
                        "n": len(v),
                    })
            except Exception as e:
                print(f"  Joint Cox failed: {e}")

    return pd.DataFrame(results)


def q1_compartment_celltype(obs):
    """Q1: IDO+ fraction by compartment and cell type."""
    print("\n═══ Q1: IDO+ fraction by compartment and cell type ═══")
    myeloid = obs[obs["is_myeloid"]].copy()
    myeloid["IDO_pos"] = myeloid["IDO"] > 0.5
    myeloid["VISTA_pos"] = myeloid["VISTA"] > 0.5

    # Overall
    print(f"\nOverall myeloid: {len(myeloid):,} cells")
    print(f"  IDO+: {myeloid['IDO_pos'].mean():.1%}")
    print(f"  VISTA+: {myeloid['VISTA_pos'].mean():.1%}")
    print(f"  IDO+VISTA+: {(myeloid['IDO_pos'] & myeloid['VISTA_pos']).mean():.1%}")

    # By cell type
    print("\nBy cell type:")
    for ct in sorted(myeloid["cell_type"].unique()):
        sub = myeloid[myeloid["cell_type"] == ct]
        if len(sub) < 50:
            continue
        print(f"  {ct} (n={len(sub):,}): IDO+ {sub['IDO_pos'].mean():.1%}, "
              f"VISTA+ {sub['VISTA_pos'].mean():.1%}, "
              f"IDO mean={sub['IDO'].mean():.3f}")

    # By compartment
    print("\nBy compartment (follicular vs interfollicular):")
    for comp, label in [(True, "Follicular"), (False, "Interfollicular")]:
        sub = myeloid[myeloid["is_follicular"] == comp]
        if len(sub) < 50:
            continue
        print(f"  {label} (n={len(sub):,}): IDO+ {sub['IDO_pos'].mean():.1%}, "
              f"VISTA+ {sub['VISTA_pos'].mean():.1%}, "
              f"IDO mean={sub['IDO'].mean():.3f}, VISTA mean={sub['VISTA'].mean():.3f}")

    # Compartment enrichment (Wilcoxon per-ROI)
    foll = myeloid[myeloid["is_follicular"]]
    inter = myeloid[~myeloid["is_follicular"]]
    foll_roi = foll.groupby("sample_id")["IDO"].mean()
    inter_roi = inter.groupby("sample_id")["IDO"].mean()
    common = foll_roi.index.intersection(inter_roi.index)
    if len(common) > 10:
        stat, p = stats.wilcoxon(foll_roi[common], inter_roi[common])
        ratio = foll_roi[common].mean() / max(inter_roi[common].mean(), 1e-6)
        print(f"\n  IDO follicular/interfollicular ratio: {ratio:.2f}x "
              f"(Wilcoxon p={p:.2e}, n={len(common)} paired ROIs)")

        stat_v, p_v = stats.wilcoxon(
            foll.groupby("sample_id")["VISTA"].mean()[common],
            inter.groupby("sample_id")["VISTA"].mean()[common])
        ratio_v = (foll.groupby("sample_id")["VISTA"].mean()[common].mean() /
                   max(inter.groupby("sample_id")["VISTA"].mean()[common].mean(), 1e-6))
        print(f"  VISTA follicular/interfollicular ratio: {ratio_v:.2f}x "
              f"(Wilcoxon p={p_v:.2e})")

    # By cell type × compartment
    print("\nBy cell type × compartment:")
    for ct in ["FDC", "M1 Macrophages", "M2 Macrophages", "Macrophages",
               "Myeloid (S100A9+)", "Dendritic cells"]:
        sub_f = myeloid[(myeloid["cell_type"] == ct) & myeloid["is_follicular"]]
        sub_i = myeloid[(myeloid["cell_type"] == ct) & ~myeloid["is_follicular"]]
        if len(sub_f) < 20 or len(sub_i) < 20:
            continue
        print(f"  {ct}: foll IDO={sub_f['IDO'].mean():.3f} "
              f"({sub_f['IDO_pos'].mean():.1%}), "
              f"inter IDO={sub_i['IDO'].mean():.3f} "
              f"({sub_i['IDO_pos'].mean():.1%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s-panel", default="output/all_TMA_S_global_v8.h5ad")
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    args = parser.parse_args()

    # Load data
    obs, _ = load_data(args.s_panel, args.s_utag)

    # Q1: Compartment × cell type
    q1_compartment_celltype(obs)

    # Compute per-ROI stats
    roi = compute_roi_stats(obs)

    # Merge clinical
    merged = merge_clinical(roi)

    # Q2: Survival correlations
    print("\n═══ Q2: Spearman correlations with survival ═══")
    corr_df = run_survival_correlations(merged)
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Q3: KM analysis
    print("\n═══ Q3: Kaplan-Meier log-rank tests ═══")
    km_df = run_km_analysis(merged)
    print(km_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Q4-Q5: Cox multivariate
    print("\n═══ Q4-Q5: Cox proportional hazards ═══")
    cox_df = run_cox_multivariate(merged)
    print(cox_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Summary
    print("\n═══ SUMMARY ═══")
    sig_corr = corr_df[corr_df["p"] < 0.05]
    if len(sig_corr) > 0:
        print("\nSignificant correlations (p<0.05):")
        print(sig_corr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("\nNo significant correlations at p<0.05")

    sig_km = km_df[km_df["logrank_p"] < 0.05] if len(km_df) > 0 else pd.DataFrame()
    if len(sig_km) > 0:
        print("\nSignificant KM splits (p<0.05):")
        print(sig_km.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    sig_cox = cox_df[cox_df["p"] < 0.05] if len(cox_df) > 0 else pd.DataFrame()
    if len(sig_cox) > 0:
        print("\nSignificant Cox results (p<0.05):")
        print(sig_cox.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
