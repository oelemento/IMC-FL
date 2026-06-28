#!/usr/bin/env python3
"""Binary low (FOLL1+FOLL2) vs high (FOLL3A) grade comparison.

Reuses the loaders, metric definitions, and compartment classifications from
`grade_arch_mixed_cores.py`. For each panel (S, T) and each cohort filter
(full, mixed-cores), runs a Mann-Whitney U test on patient-level metrics:

  - shannon_compartment, simpson_compartment, n_compartments_present_p05
  - frac_follicular, frac_interfollicular
  - per-biomarker per-ROI mean

Produces one 4-row figure: rows = (S full, S mixed, T full, T mixed),
cols = the headline metrics. Each cell is a low-vs-high boxplot with the MW
p-value annotated.

Outputs:
  output/grade_arch/low_vs_high/grade_low_vs_high_summary.csv
  output/grade_arch/low_vs_high/fig_grade_low_vs_high.png

Usage:
    .venv/bin/python scripts/grade_arch_low_vs_high.py
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
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

# Reuse implementation from grade_arch_mixed_cores
from scripts.grade_arch_mixed_cores import (
    panel_config, load_data, per_roi_metrics,
    join_clinical_and_grade, is_tumor_core,
    DEFAULT_MIN_CELLS_PER_ROI, DEFAULT_MIN_MIXED_FRAC,
)

GROUP_LOW = "Low (FOLL1+2)"
GROUP_HIGH = "High (FOLL3A)"
GROUP_COLORS = {GROUP_LOW: "#1f77b4", GROUP_HIGH: "#d62728"}


def assign_group(grade):
    if grade in ("FOLL1", "FOLL2"):
        return GROUP_LOW
    if grade == "FOLL3A":
        return GROUP_HIGH
    return None


def mw(df, metric):
    a = df.loc[df["group"] == GROUP_LOW, metric].dropna().values
    b = df.loc[df["group"] == GROUP_HIGH, metric].dropna().values
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan
    if len(np.unique(np.concatenate([a, b]))) < 2:
        return np.nan, float(np.median(a)), float(np.median(b))
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p), float(np.median(a)), float(np.median(b))
    except ValueError:
        return np.nan, float(np.median(a)), float(np.median(b))


def bh_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    out = np.full_like(pvals, np.nan)
    if not valid.any():
        return out
    p_valid = pvals[valid]
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    out[valid] = q
    return out


def compute_panel(panel: str, args, mixed_only: bool):
    """Return patient-level dataframe with `group` column for one panel +
    cohort filter, plus the list of biomarkers actually loaded."""
    cfg = panel_config(panel)
    df, bm_present = load_data(Path(cfg["h5ad"]), cfg["biomarkers"])
    df = df[df["sample_id"].apply(is_tumor_core)].copy()
    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)

    # Defensive: same checks as the main script
    obs_comps = set(df["compartment"].unique())
    missing = obs_comps - set(cfg["colors"])
    if missing:
        raise RuntimeError(f"{panel}-panel: palette missing keys for {missing}")

    bm_p90 = {b: float(np.quantile(df[b].dropna().values, 0.90)) for b in bm_present}

    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_metrics(sub, bm_present, args.min_cells, bm_p90,
                            list(cfg["colors"].keys()))
        if m is None:
            continue
        f_frac = sum(m[f"frac_{c}"] for c in cfg["follicular"])
        i_frac = sum(m[f"frac_{c}"] for c in cfg["interfollicular"])
        m["frac_follicular"] = f_frac
        m["frac_interfollicular"] = i_frac
        m["is_mixed"] = bool(f_frac >= args.min_mixed_frac
                             and i_frac >= args.min_mixed_frac)
        rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows)
    metrics_df = join_clinical_and_grade(metrics_df, Path(args.clinical),
                                          Path(args.grade))
    if mixed_only:
        metrics_df = metrics_df[metrics_df.is_mixed].copy()

    metric_cols = [c for c in metrics_df.columns
                   if c not in {"sample_id", "slide_ID", "Sample_ID",
                                "Patient_ID", "grade", "is_mixed"}]
    pt = (metrics_df.groupby(["Patient_ID", "grade"])[metric_cols]
          .mean().reset_index())
    pt["group"] = pt["grade"].apply(assign_group)
    pt = pt[pt["group"].notna()].copy()
    return pt, bm_present


def plot_box(ax, df, metric, label, p_val):
    data = [df.loc[df["group"] == g, metric].dropna().values
            for g in (GROUP_LOW, GROUP_HIGH)]
    bp = ax.boxplot(data, tick_labels=[GROUP_LOW, GROUP_HIGH], patch_artist=True,
                    widths=0.55, showfliers=False)
    rng = np.random.default_rng(0)
    for patch, g in zip(bp["boxes"], (GROUP_LOW, GROUP_HIGH)):
        patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.55)
    for i, (g, vals) in enumerate(zip((GROUP_LOW, GROUP_HIGH), data)):
        xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(xs, vals, color=GROUP_COLORS[g], s=12, alpha=0.7,
                   edgecolor="white", linewidth=0.4, zorder=3)
    n_low = len(data[0]); n_high = len(data[1])
    star = " *" if (not np.isnan(p_val) and p_val < 0.05) else ""
    ax.set_title(f"{label}\nMW p={p_val:.3g}{star}  (n={n_low} vs {n_high})",
                 fontsize=10)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--out", default="output/grade_arch/low_vs_high")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
    p.add_argument("--min-mixed-frac", type=float, default=DEFAULT_MIN_MIXED_FRAC)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Binary FOLL1+2 vs FOLL3A grade comparison ===\n")

    rows = [
        ("S-panel — full cohort",  "s", False),
        ("S-panel — mixed cores",  "s", True),
        ("T-panel — full cohort",  "t", False),
        ("T-panel — mixed cores",  "t", True),
    ]
    cohorts = {}
    biomarkers = {}
    for label, panel, mixed in rows:
        pt, bm = compute_panel(panel, args, mixed_only=mixed)
        n_low = int((pt.group == GROUP_LOW).sum())
        n_high = int((pt.group == GROUP_HIGH).sum())
        print(f"  {label:30s}: n={n_low} low, n={n_high} high")
        cohorts[label] = pt
        biomarkers[label] = bm

    # Headline metric set per panel
    SHARED_METRICS = ["shannon_compartment", "n_compartments_present_p05",
                      "frac_follicular", "frac_interfollicular"]

    # KW + BH per (label, metric_family)
    summary_rows = []
    fig, axes = plt.subplots(len(rows), 5, figsize=(24, 4.6 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, -1)

    for r, (label, panel, mixed) in enumerate(rows):
        pt = cohorts[label]
        bm = biomarkers[label]

        # Build family of metrics to test for this cohort
        compart_metrics = SHARED_METRICS + ["simpson_compartment"]
        biomarker_metrics = [f"{b}_mean" for b in bm]
        test_metrics = compart_metrics + biomarker_metrics

        pvals = []
        meds_lo = []
        meds_hi = []
        for m in test_metrics:
            p_v, ml, mh = mw(pt, m)
            pvals.append(p_v); meds_lo.append(ml); meds_hi.append(mh)
        pvals = np.array(pvals)
        q_compart = bh_correct(pvals[:len(compart_metrics)])
        q_biomark = bh_correct(pvals[len(compart_metrics):])
        q_within = np.concatenate([q_compart, q_biomark])
        q_combined = bh_correct(pvals)

        for m, p_v, q_w, q_c, ml, mh in zip(test_metrics, pvals, q_within,
                                              q_combined, meds_lo, meds_hi):
            summary_rows.append({"cohort": label, "metric": m,
                                  "p_MW": p_v, "q_BH_within_family": q_w,
                                  "q_BH_combined": q_c,
                                  "median_low": ml, "median_high": mh})

        # Pick the best biomarker by raw p (smallest)
        bm_pmap = {b: pvals[test_metrics.index(f"{b}_mean")] for b in bm}
        bm_pmap = {b: pv for b, pv in bm_pmap.items() if not np.isnan(pv)}
        best_b = min(bm_pmap, key=bm_pmap.get) if bm_pmap else None

        # Print summary line
        print(f"\n  --- {label} ---")
        print(f"    {'metric':32s} {'p_MW':>10s} {'q_within':>10s}  med low / med high")
        for m, p_v, q_w, ml, mh in zip(test_metrics, pvals, q_within, meds_lo, meds_hi):
            star = " *" if (not np.isnan(q_w) and q_w < 0.05) else ""
            ml_s = f"{ml:.3g}" if ml is not None and not np.isnan(ml) else "NA"
            mh_s = f"{mh:.3g}" if mh is not None and not np.isnan(mh) else "NA"
            print(f"    {m:32s} {p_v:10.4g} {q_w:10.4g}  {ml_s} / {mh_s}{star}")

        # Plot the 5 headline metric columns:
        #   Shannon, n_compartments_p05, frac_follicular, frac_interfollicular,
        #   best biomarker (or hide if none).
        plot_metrics = [
            ("shannon_compartment", "Compartment Shannon"),
            ("n_compartments_present_p05", "# Compartments (≥5%)"),
            ("frac_follicular", "Follicular fraction"),
            ("frac_interfollicular", "Interfollicular fraction"),
        ]
        if best_b is not None:
            plot_metrics.append((f"{best_b}_mean", f"{best_b} per-ROI mean"))

        for c, (metric, mlabel) in enumerate(plot_metrics):
            ax = axes[r, c]
            idx = test_metrics.index(metric)
            plot_box(ax, pt, metric, mlabel, pvals[idx])
            if c == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
        # Hide unused axes
        for c in range(len(plot_metrics), 5):
            axes[r, c].axis("off")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "grade_low_vs_high_summary.csv", index=False)
    print(f"\nSaved summary: {out_dir / 'grade_low_vs_high_summary.csv'}")

    fig.suptitle(f"FL grade — Low (FOLL1+2) vs High (FOLL3A) — patient-level "
                 f"Mann-Whitney  [mixed cores: ≥{int(args.min_mixed_frac*100)}% on each side]",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    out = out_dir / "fig_grade_low_vs_high.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
