"""
Test whether follicular CD14+ FDC expression predicts transformation.

Replicates Smeltzer et al. (CCR 2014) finding that follicular-localized
CD14+ cells (identified as FDCs) predict shorter time to transformation
(HR=3.0, P=0.004). Uses IMC spatial data with UTAG compartment labels
for true single-cell spatial resolution.

Metrics tested:
  1. Per-patient mean FDC CD14 (all FDCs)
  2. Per-patient mean FDC CD14 (follicular FDCs only)
  3. Per-patient mean FDC CD14 (interfollicular FDCs only)
  4. Per-patient follicular FDC fraction
  5. Per-patient FDC density (fraction of all cells)

Output: fig_fdc_cd14_transformation.png
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.gridspec import GridSpec
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, load_clinical, normalize_sample_id


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


def panel_label(ax, letter, x=-0.08, y=1.05):
    ax.text(
        x, y, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top", ha="left",
    )


# T-panel follicular compartments
T_FOLLICULAR_COMPARTMENTS = [
    "Activated B / CXCR5hi zone",
    "B cell follicle (CD20hi/CXCR5hi)",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)",
    "Follicle-T zone interface",
    "GC core",
]

# S-panel follicular compartments (B-cell/FDC-dominated zones)
S_FOLLICULAR_COMPARTMENTS = [
    "B cell zone (BCL2+)",
    "B cell zone (PAX5+)",
    "FDC network zone",
]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_fdc_metrics(s_panel_path, s_utag_path):
    """Extract per-ROI FDC CD14 metrics with compartment resolution."""
    print("Loading S-panel h5ad...")
    with h5py.File(s_panel_path, "r") as f:
        markers = [v.decode() if isinstance(v, bytes) else str(v)
                   for v in f["var"]["_index"][:]]
        marker_idx = {m: i for i, m in enumerate(markers)}
        cd14_idx = marker_idx["CD14"]

        cell_types = load_array(f, "cell_type")
        sample_ids = load_array(f, "sample_id")
        X_cd14 = f["X"][:, cd14_idx]

    print("Loading S-panel UTAG h5ad for compartments...")
    with h5py.File(s_utag_path, "r") as f:
        comps_utag = load_array(f, "compartment_name")
        sids_utag = load_array(f, "sample_id")
        ct_utag = load_array(f, "cell_type")

    # Build ROI-level lookup for compartment of each cell
    # S-panel main and UTAG should have same cells, but verify by ROI
    print("Building compartment lookup...")
    # Index UTAG cells by (sample_id, position) for matching
    # Simpler approach: since both files have same ROI order, match by index
    # But safer: match by ROI-level, using cell_type to verify alignment
    utag_comp_by_roi = {}
    for roi in np.unique(sids_utag):
        mask = sids_utag == roi
        utag_comp_by_roi[roi] = comps_utag[mask]

    # Filter to tumor cores
    tumor_rois = set()
    for roi in np.unique(sample_ids):
        if is_tumor_core(roi) and roi not in EXCLUDE_ROIS and not roi.startswith("Biomax"):
            tumor_rois.add(roi)

    print(f"  {len(tumor_rois)} tumor ROIs")

    rows = []
    for roi in sorted(tumor_rois):
        # Main h5ad
        mask = sample_ids == roi
        roi_ct = cell_types[mask]
        roi_cd14 = X_cd14[mask]

        fdc_mask = roi_ct == "FDC"
        n_fdc = fdc_mask.sum()
        n_total = mask.sum()
        if n_fdc < 5:
            continue

        # Get compartments from UTAG
        if roi not in utag_comp_by_roi:
            continue
        roi_comps = utag_comp_by_roi[roi]
        if len(roi_comps) != mask.sum():
            # Length mismatch — skip
            continue

        is_foll = np.isin(roi_comps, S_FOLLICULAR_COMPARTMENTS)

        # FDC metrics
        fdc_cd14_all = roi_cd14[fdc_mask]
        fdc_foll = fdc_mask & is_foll
        fdc_ifoll = fdc_mask & ~is_foll

        row = {
            "sample_id": roi,
            "slide_ID": normalize_sample_id(roi),
            "tma": roi.split("_")[0],
            "n_cells": n_total,
            "n_fdc": n_fdc,
            "fdc_frac": n_fdc / n_total,
            "fdc_cd14_all": float(fdc_cd14_all.mean()),
            "n_fdc_foll": int(fdc_foll.sum()),
            "n_fdc_ifoll": int(fdc_ifoll.sum()),
            "fdc_foll_frac": float(fdc_foll.sum()) / n_fdc,
        }

        if fdc_foll.sum() >= 3:
            row["fdc_cd14_foll"] = float(roi_cd14[fdc_foll].mean())
        else:
            row["fdc_cd14_foll"] = np.nan

        if fdc_ifoll.sum() >= 3:
            row["fdc_cd14_ifoll"] = float(roi_cd14[fdc_ifoll].mean())
        else:
            row["fdc_cd14_ifoll"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {len(df)} ROIs with ≥5 FDCs and compartment data")
    return df


def build_patient_table(roi_df):
    """Merge with clinical data and aggregate to patient level."""
    clin = load_clinical()
    clin_t1 = clin.sort_values("T").drop_duplicates(subset="slide_ID", keep="first")

    # Exclude Biomax
    roi_df = roi_df[roi_df["tma"] != "Biomax"].copy()

    # Merge
    merged = roi_df.merge(clin_t1, on="slide_ID", how="inner")
    print(f"\n  Merged: {len(merged)} ROIs with clinical data")

    # For patients with multiple ROIs, average metrics
    numeric_cols = [
        "fdc_cd14_all", "fdc_cd14_foll", "fdc_cd14_ifoll",
        "fdc_foll_frac", "fdc_frac", "n_fdc",
    ]
    agg_dict = {c: "mean" for c in numeric_cols if c in merged.columns}
    agg_dict["sample_id"] = "first"  # keep one ROI ID for reference
    agg_dict["tma"] = "first"

    # Group by Patient_ID, keeping clinical columns
    clin_cols = [
        "Patient_ID", "AGE", "SEX",
        "Overall survival (y)", "CODE_OS",
        "Progression free survival (y)", "CODE_PFS",
        "FLIPI", "FLIPI.1", "Transformation", "T",
    ]
    clin_cols = [c for c in clin_cols if c in merged.columns]

    # Take T1 biopsy for clinical, mean for spatial
    pat_df = merged.sort_values("T").drop_duplicates(
        subset="Patient_ID", keep="first"
    )

    # Rename for convenience
    pat_df = pat_df.rename(columns={
        "Overall survival (y)": "os_time",
        "CODE_OS": "os_event",
        "Progression free survival (y)": "pfs_time",
        "CODE_PFS": "pfs_event",
    })

    for col in ["os_time", "os_event", "pfs_time", "pfs_event", "AGE", "FLIPI"]:
        pat_df[col] = pd.to_numeric(pat_df[col], errors="coerce")

    pat_df["transformed"] = (
        pat_df["Transformation"].str.strip().str.lower() == "yes"
    ).fillna(False).astype(int)

    pat_df["flipi_high"] = (pat_df["FLIPI.1"] == "HIGH").astype(int)

    n_tfm = pat_df["transformed"].sum()
    n_notfm = (pat_df["transformed"] == 0).sum()
    print(f"  {len(pat_df)} unique patients")
    print(f"  Transformed: {n_tfm}, Not transformed: {n_notfm}")

    return pat_df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(pat_df, output_dir):
    fig = plt.figure(figsize=(18, 5.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35,
                  left=0.06, right=0.97, top=0.88, bottom=0.12)

    # ── (a) Boxplot: follicular FDC CD14 by transformation ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")

    metric = "fdc_cd14_foll"
    sub = pat_df[[metric, "transformed"]].dropna()
    grp0 = sub[sub["transformed"] == 0][metric]
    grp1 = sub[sub["transformed"] == 1][metric]

    bp = ax_a.boxplot(
        [grp0, grp1],
        tick_labels=["No transformation", "Transformed"],
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    bp["boxes"][0].set_facecolor("#377EB8")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#E41A1C")
    bp["boxes"][1].set_alpha(0.6)

    for j, (grp, xpos) in enumerate([(grp0, 1), (grp1, 2)]):
        jitter = np.random.normal(0, 0.05, len(grp))
        ax_a.scatter(
            xpos + jitter, grp,
            c=["#377EB8", "#E41A1C"][j],
            alpha=0.4, s=20, edgecolors="white", linewidth=0.3,
            zorder=3,
        )

    _, p_val = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
    ax_a.text(0.5, 0.95,
              f"P={p_val:.3f}\nn={len(grp0)} vs {len(grp1)}",
              transform=ax_a.transAxes, ha="center", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax_a.set_ylabel("Mean FDC CD14 (follicular FDCs)")
    ax_a.set_title("Follicular FDC CD14 by transformation status", fontsize=11)

    # ── (b) KM: PFS by follicular FDC CD14 ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")

    sub = pat_df[[metric, "pfs_time", "pfs_event"]].dropna()
    median_val = sub[metric].median()
    hi = sub[sub[metric] >= median_val]
    lo = sub[sub[metric] < median_val]

    kmf_hi = KaplanMeierFitter()
    kmf_lo = KaplanMeierFitter()
    kmf_hi.fit(hi["pfs_time"], hi["pfs_event"], label=f"CD14-high (n={len(hi)})")
    kmf_lo.fit(lo["pfs_time"], lo["pfs_event"], label=f"CD14-low (n={len(lo)})")

    kmf_lo.plot_survival_function(ax=ax_b, color="#377EB8", ci_show=False)
    kmf_hi.plot_survival_function(ax=ax_b, color="#E41A1C", ci_show=False)

    lr = logrank_test(
        hi["pfs_time"], lo["pfs_time"],
        hi["pfs_event"], lo["pfs_event"],
    )
    ax_b.set_title(f"PFS by follicular FDC CD14 (P={lr.p_value:.3f})", fontsize=11)
    ax_b.set_xlabel("Time (years)")
    ax_b.set_ylabel("Progression-free survival")
    ax_b.legend(fontsize=9, loc="lower left")

    # ── (c) Cox forest: follicular vs interfollicular vs all ──
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c")

    forest_metrics = [
        ("fdc_cd14_foll", "Follicular FDC CD14"),
        ("fdc_cd14_all", "All FDC CD14"),
        ("fdc_cd14_ifoll", "Interfollicular FDC CD14"),
    ]

    cox_results = []
    for metric_name, label in forest_metrics:
        sub = pat_df[[metric_name, "pfs_time", "pfs_event"]].dropna()
        if len(sub) < 20 or sub["pfs_event"].sum() < 5:
            continue
        cph = CoxPHFitter()
        try:
            sub_std = sub.copy()
            std = sub_std[metric_name].std()
            if std > 0:
                sub_std[metric_name] = (
                    sub_std[metric_name] - sub_std[metric_name].mean()
                ) / std
            cph.fit(sub_std, duration_col="pfs_time", event_col="pfs_event")
            s = cph.summary.iloc[0]
            cox_results.append({
                "metric": label,
                "HR": s["exp(coef)"],
                "HR_lo": s["exp(coef) lower 95%"],
                "HR_hi": s["exp(coef) upper 95%"],
                "p": s["p"],
                "n": len(sub),
            })
        except Exception:
            pass

    if cox_results:
        cox_df = pd.DataFrame(cox_results)
        y_pos = np.arange(len(cox_df))

        for i, (_, r) in enumerate(cox_df.iterrows()):
            color = "#E41A1C" if r["p"] < 0.05 else "#999999"
            ax_c.errorbar(
                r["HR"], i,
                xerr=[[r["HR"] - r["HR_lo"]], [r["HR_hi"] - r["HR"]]],
                fmt="o", color=color, capsize=5, markersize=8,
                markeredgecolor="white", markeredgewidth=0.5,
            )

        ax_c.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(
            [f"{r['metric']}\n(P={r['p']:.3f}, n={r['n']})"
             for _, r in cox_df.iterrows()],
            fontsize=9,
        )
        ax_c.set_xlabel("Hazard ratio per SD (PFS)")
        ax_c.set_title("Univariate Cox: compartment matters", fontsize=11)

    out = Path(output_dir) / "fig_fdc_cd14_transformation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")

    return cox_results


# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------

def print_summary(pat_df):
    """Print key statistics for logging."""
    print("\n" + "=" * 70)
    print("SUMMARY: FDC CD14 × Transformation Analysis")
    print("=" * 70)

    metrics = ["fdc_cd14_all", "fdc_cd14_foll", "fdc_cd14_ifoll",
               "fdc_foll_frac", "fdc_frac"]

    for m in metrics:
        sub = pat_df[[m, "transformed"]].dropna()
        grp0 = sub[sub["transformed"] == 0][m]
        grp1 = sub[sub["transformed"] == 1][m]
        if len(grp0) >= 5 and len(grp1) >= 5:
            u, p = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
            print(f"\n{m}:")
            print(f"  Not transformed: mean={grp0.mean():.4f}, median={grp0.median():.4f} (n={len(grp0)})")
            print(f"  Transformed:     mean={grp1.mean():.4f}, median={grp1.median():.4f} (n={len(grp1)})")
            print(f"  Mann-Whitney U P={p:.4f}")

    # Cox PH for PFS
    print("\nUnivariate Cox (PFS):")
    for m in metrics:
        sub = pat_df[[m, "pfs_time", "pfs_event"]].dropna()
        if len(sub) < 20 or sub["pfs_event"].sum() < 5:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col="pfs_time", event_col="pfs_event")
            s = cph.summary.iloc[0]
            print(f"  {m}: HR={s['exp(coef)']:.3f} "
                  f"[{s['exp(coef) lower 95%']:.3f}-{s['exp(coef) upper 95%']:.3f}], "
                  f"P={s['p']:.4f}")
        except Exception as e:
            print(f"  {m}: FAILED ({e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-panel", default="output/all_TMA_S_global_v8.h5ad")
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    roi_df = extract_fdc_metrics(args.s_panel, args.s_utag)
    pat_df = build_patient_table(roi_df)
    print_summary(pat_df)
    make_figure(pat_df, args.output_dir)


if __name__ == "__main__":
    main()
