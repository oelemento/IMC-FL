#!/usr/bin/env python3
"""Cell-type fraction survival analysis in follicular lymphoma.

Tests which cell types, by abundance (fraction of all cells), predict
PFS, OS, and transformation.  Motivated by the finding that S100A9+
myeloid cells are 3x more abundant in transformers (p<0.001) and M1
macrophages ~2x (p<0.001), while M2 macrophages are not enriched.

Metrics computed per ROI (denominator = all cells):
  - s100a9_frac:   Myeloid (S100A9+) fraction — strongest transformation signal
  - m1_frac:       M1 Macrophages fraction
  - m2_frac:       M2 Macrophages fraction
  - mac_frac:      Macrophages (generic) fraction
  - fdc_frac:      FDC fraction
  - myeloid_frac:  All myeloid combined
  - cd8_frac:      CD8 T cells fraction — immune surveillance context
  - bcl2b_frac:    B cells (BCL2+) fraction — tumor burden proxy

Figure panels:
  (a) Forest plot: Cox HR for PFS across cell-type fractions
  (b) Forest plot: Cox HR for OS
  (c) Transformation: fractions in transformers vs non-transformers
  (d) KM curve: best PFS predictor split at median
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, load_clinical, normalize_sample_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cell types to track as individual fractions
CELL_TYPES = {
    "s100a9_frac": "Myeloid (S100A9+)",
    "m1_frac": "M1 Macrophages",
    "m2_frac": "M2 Macrophages",
    "mac_frac": "Macrophages",
    "fdc_frac": "FDC",
    "cd8_frac": "CD8 T cells",
    "bcl2b_frac": "B cells (BCL2+)",
}

# Myeloid cell types combined for aggregate metric
MYELOID_TYPES = [
    "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Myeloid (S100A9+)", "Dendritic cells",
]

MIN_CELLS = 200  # minimum total cells per ROI


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


# ---------------------------------------------------------------------------
# Step 1: Extract per-ROI cell-type fractions
# ---------------------------------------------------------------------------

def extract_celltype_fractions(s_panel_path):
    """Compute per-ROI cell-type fractions (denominator = all cells)."""
    print("Loading S-panel h5ad...")
    with h5py.File(s_panel_path, "r") as f:
        ctypes = load_array(f, "cell_type")
        sids = load_array(f, "sample_id")

    # Filter to tumor cores
    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS and not s.startswith("Biomax")
        for s in sids
    ])
    ctypes = ctypes[tumor_mask]
    sids = sids[tumor_mask]
    print(f"  {len(ctypes):,} tumor cells")

    # Per-ROI fractions
    rois = sorted(set(sids))
    rows = []
    for roi in rois:
        mask = sids == roi
        roi_ct = ctypes[mask]
        n_total = len(roi_ct)
        if n_total < MIN_CELLS:
            continue

        row = {
            "sample_id": roi,
            "slide_ID": normalize_sample_id(roi),
            "n_total": int(n_total),
        }

        # Individual cell-type fractions
        for metric, ct_name in CELL_TYPES.items():
            n_ct = (roi_ct == ct_name).sum()
            row[metric] = float(n_ct / n_total)

        # Aggregate myeloid fraction
        n_myeloid = np.isin(roi_ct, MYELOID_TYPES).sum()
        row["myeloid_frac"] = float(n_myeloid / n_total)

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {len(df)} ROIs with >= {MIN_CELLS} cells")

    # Summary stats
    for metric in METRIC_ORDER_ALL:
        if metric in df.columns:
            vals = df[metric]
            print(f"    {metric:20s}  median={vals.median()*100:.2f}%  "
                  f"mean={vals.mean()*100:.2f}%  range=[{vals.min()*100:.2f}%, {vals.max()*100:.2f}%]")

    return df


# ---------------------------------------------------------------------------
# Step 2: Clinical merge
# ---------------------------------------------------------------------------

def merge_clinical(df):
    """Merge with clinical data, one row per patient."""
    clin = load_clinical()
    clin_t1 = clin.sort_values("T").drop_duplicates(subset="slide_ID", keep="first")

    merged = df.merge(clin_t1, on="slide_ID", how="inner")
    print(f"\n  Merged: {len(merged)} ROIs with clinical data")

    # Deduplicate by patient (keep T1)
    merged = merged.sort_values("T").drop_duplicates(
        subset="Patient_ID", keep="first"
    )
    print(f"  After T1-only dedup: {len(merged)} unique patients")

    # Rename clinical columns
    merged = merged.rename(columns={
        "Overall survival (y)": "os_time",
        "CODE_OS": "os_event",
        "Progression free survival (y)": "pfs_time",
        "CODE_PFS": "pfs_event",
    })

    for col in ["os_time", "os_event", "pfs_time", "pfs_event", "AGE", "FLIPI"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["flipi_high"] = (merged["FLIPI.1"] == "HIGH").astype(int)

    merged["transformed"] = (
        merged["Transformation"].str.strip().str.lower() == "yes"
    ).astype(int)

    # POD24
    tx_col = 'INITIAL "TREATMENT"'
    if tx_col in merged.columns:
        treated = merged[tx_col].str.strip() != "OBSE"
    else:
        treated = pd.Series(True, index=merged.index)
    merged["treated"] = treated.astype(int)
    merged["pod24"] = np.where(
        ~treated, np.nan,
        np.where(
            (merged["pfs_event"] == 1) & (merged["pfs_time"] <= 2.0), 1,
            np.where(merged["pfs_time"] > 2.0, 0, np.nan),
        ),
    )

    n_os = int(merged["os_event"].sum())
    n_pfs = int(merged["pfs_event"].sum())
    n_trans = int(merged["transformed"].sum())
    n_pod24 = int(merged["pod24"].dropna().sum())
    print(f"  OS events: {n_os}/{len(merged)}, PFS events: {n_pfs}/{len(merged)}")
    print(f"  Transformed: {n_trans}, POD24: {n_pod24}")

    return merged


# ---------------------------------------------------------------------------
# Step 3: Survival models
# ---------------------------------------------------------------------------

def univariate_cox(df, metric, time_col, event_col):
    """Univariate Cox PH for one metric (z-scored → HR per SD)."""
    sub = df[[metric, time_col, event_col]].dropna()
    if len(sub) < 20 or sub[event_col].sum() < 5:
        return None
    # Z-score so HR is interpretable as "per 1 SD increase"
    sub = sub.copy()
    mu = sub[metric].mean()
    sd = sub[metric].std()
    if sd < 1e-12:
        return None
    sub[metric] = (sub[metric] - mu) / sd
    cph = CoxPHFitter()
    try:
        cph.fit(sub, duration_col=time_col, event_col=event_col)
        s = cph.summary.iloc[0]
        return {
            "metric": metric,
            "HR": s["exp(coef)"],
            "HR_lo": s["exp(coef) lower 95%"],
            "HR_hi": s["exp(coef) upper 95%"],
            "p": s["p"],
            "concordance": cph.concordance_index_,
            "n": len(sub),
            "events": int(sub[event_col].sum()),
        }
    except Exception as e:
        print(f"  Cox failed for {metric}: {e}")
        return None


# Metric ordering — figure display
METRIC_ORDER = [
    "s100a9_frac",
    "m1_frac",
    "m2_frac",
    "mac_frac",
    "fdc_frac",
    "myeloid_frac",
    "cd8_frac",
    "bcl2b_frac",
]

# All metrics (same for this analysis)
METRIC_ORDER_ALL = METRIC_ORDER[:]

METRIC_LABELS = {
    "s100a9_frac": "S100A9+ Myeloid (%)",
    "m1_frac": "M1 Macrophages (%)",
    "m2_frac": "M2 Macrophages (%)",
    "mac_frac": "Macrophages, generic (%)",
    "fdc_frac": "FDC (%)",
    "myeloid_frac": "All myeloid (%)",
    "cd8_frac": "CD8 T cells (%)",
    "bcl2b_frac": "B cells BCL2+ (%)",
}


# ---------------------------------------------------------------------------
# Step 4: Figure
# ---------------------------------------------------------------------------

def plot_forest(ax, results_df, title):
    """Forest plot of Cox HRs."""
    ordered = [m for m in METRIC_ORDER if m in results_df["metric"].values]
    sub = results_df[results_df["metric"].isin(ordered)].copy()
    sub["rank"] = sub["metric"].apply(lambda m: ordered.index(m))
    sub = sub.sort_values("rank", ascending=True)

    for i, (_, row) in enumerate(sub.iterrows()):
        color = "#C0392B" if row["p"] < 0.05 else "#7F8C8D"
        ax.plot(
            [row["HR_lo"], row["HR_hi"]], [i, i],
            color=color, linewidth=2, solid_capstyle="round",
        )
        ax.plot(row["HR"], i, "o", color=color, markersize=7, zorder=5)

        label = METRIC_LABELS.get(row["metric"], row["metric"])
        sig = "*" if row["p"] < 0.05 else ""
        sig = "**" if row["p"] < 0.01 else sig
        sig = "***" if row["p"] < 0.001 else sig
        ax.text(
            -0.02, i,
            f"{label}{sig}",
            va="center", ha="right", fontsize=8,
            transform=ax.get_yaxis_transform(),
        )

        ax.annotate(
            f"{row['HR']:.1f}",
            xy=(row["HR"], i), xytext=(4, -10),
            textcoords="offset points", fontsize=7, color=color,
        )

    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks([])
    ax.set_xlabel("Hazard Ratio (per SD)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.5, len(sub) - 0.5)

    all_lo = sub["HR_lo"].min()
    all_hi = sub["HR_hi"].max()
    margin = (all_hi - all_lo) * 0.1
    ax.set_xlim(max(0.1, all_lo - margin), all_hi + margin)


def plot_km(ax, df, metric, time_col, event_col, label):
    """KM curve split at median."""
    sub = df[[metric, time_col, event_col]].dropna()
    median_val = sub[metric].median()
    hi = sub[sub[metric] >= median_val]
    lo = sub[sub[metric] < median_val]

    kmf = KaplanMeierFitter()
    kmf.fit(hi[time_col], hi[event_col], label=f"High (n={len(hi)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#C0392B")

    kmf.fit(lo[time_col], lo[event_col], label=f"Low (n={len(lo)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2980B9")

    lr = logrank_test(
        hi[time_col], lo[time_col],
        event_observed_A=hi[event_col], event_observed_B=lo[event_col],
    )
    ax.text(
        0.95, 0.95, f"log-rank p = {lr.p_value:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)}\n{label}", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)


def plot_transformation(ax, df, metrics):
    """Box plots of fractions in transformers vs non-transformers."""
    trans = df[df["transformed"] == 1]
    notrans = df[df["transformed"] == 0]

    x_pos = np.arange(len(metrics))
    width = 0.35

    vals_trans = []
    vals_notrans = []
    pvals = []
    for m in metrics:
        # Convert to percentage for display
        t_vals = trans[m].dropna() * 100
        nt_vals = notrans[m].dropna() * 100
        vals_trans.append(t_vals)
        vals_notrans.append(nt_vals)
        if len(t_vals) >= 3 and len(nt_vals) >= 3:
            _, p = stats.mannwhitneyu(t_vals, nt_vals, alternative="two-sided")
            pvals.append(p)
        else:
            pvals.append(np.nan)

    bp1 = ax.boxplot(
        vals_notrans, positions=x_pos - width / 2, widths=width * 0.8,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="#AED6F1", edgecolor="#2980B9"),
        medianprops=dict(color="#2980B9", linewidth=2),
    )
    bp2 = ax.boxplot(
        vals_trans, positions=x_pos + width / 2, widths=width * 0.8,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="#F5B7B1", edgecolor="#C0392B"),
        medianprops=dict(color="#C0392B", linewidth=2),
    )

    for i, p in enumerate(pvals):
        if np.isnan(p):
            continue
        whisker_caps = []
        for vals in [vals_notrans[i], vals_trans[i]]:
            if len(vals) == 0:
                continue
            q75 = np.percentile(vals, 75)
            iqr = q75 - np.percentile(vals, 25)
            cap = q75 + 1.5 * iqr
            whisker_caps.append(min(cap, vals.max()))
        y_max = max(whisker_caps) if whisker_caps else 0

        sig = "n.s."
        color = "#666666"
        if p < 0.001:
            sig = "***"
            color = "#C0392B"
        elif p < 0.01:
            sig = "**"
            color = "#C0392B"
        elif p < 0.05:
            sig = "*"
            color = "#E67E22"
        ax.text(i, y_max + 0.3, sig, ha="center", fontsize=8,
                fontweight="bold", color=color)

    labels = [METRIC_LABELS.get(m, m) for m in metrics]
    labels = [l.replace(" (", "\n(") for l in labels]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Fraction of all cells (%)")
    ax.set_title("Transformation", fontsize=11, fontweight="bold")
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        [f"No transform (n={len(notrans)})", f"Transformed (n={len(trans)})"],
        fontsize=8, loc="upper right",
    )


def make_figure(df, pfs_results, os_results, output_dir):
    """Create 4-panel figure: forest PFS, forest OS, transformation, KM."""
    fig_metrics = set(METRIC_ORDER)
    pfs_fig = pfs_results[pfs_results["metric"].isin(fig_metrics)].copy()
    os_fig = os_results[os_results["metric"].isin(fig_metrics)].copy()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.45,
                          left=0.20, right=0.92, top=0.95, bottom=0.10)

    # (a) Forest plot: PFS
    ax_a = fig.add_subplot(gs[0, 0])
    plot_forest(ax_a, pfs_fig, "Progression-free survival")
    panel_label(ax_a, "a", x=-0.50)

    # (b) Forest plot: OS
    ax_b = fig.add_subplot(gs[0, 1])
    plot_forest(ax_b, os_fig, "Overall survival")
    panel_label(ax_b, "b", x=-0.50)

    # (c) Transformation comparison
    ax_c = fig.add_subplot(gs[1, 0])
    trans_metrics = [m for m in METRIC_ORDER
                     if m in df.columns and df[m].notna().sum() >= 20]
    plot_transformation(ax_c, df, trans_metrics)
    panel_label(ax_c, "c", x=-0.50)

    # (d) KM curve for best PFS predictor
    ax_d = fig.add_subplot(gs[1, 1])
    # Find best PFS predictor (lowest p-value)
    if len(pfs_fig) > 0:
        best = pfs_fig.loc[pfs_fig["p"].idxmin()]
        best_metric = best["metric"]
        plot_km(ax_d, df, best_metric, "pfs_time", "pfs_event",
                "Progression-free survival")
    panel_label(ax_d, "d")

    # Footnote
    fig.text(
        0.50, 0.01,
        "* p < 0.05, ** p < 0.01, *** p < 0.001 (nominal, univariate Cox or Mann-Whitney)",
        ha="center", fontsize=7, color="#555555", style="italic",
    )

    out = Path(output_dir) / "fig_fdc_survival.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-panel", required=True,
                        help="S-panel global h5ad (v8 annotations)")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Extract per-ROI cell-type fractions
    print("=" * 60)
    print("STEP 1: Extract cell-type fractions per ROI")
    print("=" * 60)
    roi_df = extract_celltype_fractions(args.s_panel)

    # Step 2: Merge with clinical data
    print("\n" + "=" * 60)
    print("STEP 2: Merge with clinical data")
    print("=" * 60)
    df = merge_clinical(roi_df)

    # Step 3: Univariate Cox for all metrics
    print("\n" + "=" * 60)
    print("STEP 3: Univariate Cox regression")
    print("=" * 60)

    metrics_all = [m for m in METRIC_ORDER_ALL if m in df.columns]
    endpoints = {
        "PFS": ("pfs_time", "pfs_event"),
        "OS": ("os_time", "os_event"),
    }

    all_results = []
    for ep_name, (tc, ec) in endpoints.items():
        print(f"\n  --- {ep_name} ---")
        for m in metrics_all:
            res = univariate_cox(df, m, tc, ec)
            if res:
                res["endpoint"] = ep_name
                all_results.append(res)
                sig = "***" if res["p"] < 0.001 else (
                    "**" if res["p"] < 0.01 else (
                        "*" if res["p"] < 0.05 else ""))
                print(f"    {METRIC_LABELS.get(m, m):30s}  "
                      f"HR={res['HR']:.3f} [{res['HR_lo']:.2f}-{res['HR_hi']:.2f}]  "
                      f"p={res['p']:.4f} {sig}  n={res['n']}")

    results_df = pd.DataFrame(all_results)
    pfs_results = results_df[results_df["endpoint"] == "PFS"].copy()
    os_results = results_df[results_df["endpoint"] == "OS"].copy()

    # Step 4: Transformation analysis
    print("\n" + "=" * 60)
    print("STEP 4: Transformation analysis")
    print("=" * 60)
    trans = df[df["transformed"] == 1]
    notrans = df[df["transformed"] == 0]
    print(f"  Transformed: {len(trans)}, Non-transformed: {len(notrans)}")

    for m in metrics_all:
        t_vals = trans[m].dropna()
        nt_vals = notrans[m].dropna()
        if len(t_vals) >= 3 and len(nt_vals) >= 3:
            _, p = stats.mannwhitneyu(t_vals, nt_vals, alternative="two-sided")
            sig = "***" if p < 0.001 else (
                "**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"    {METRIC_LABELS.get(m, m):30s}  "
                  f"trans={t_vals.median()*100:.2f}% vs non={nt_vals.median()*100:.2f}%  "
                  f"p={p:.4f} {sig}")

    # Step 5: POD24 analysis
    print("\n" + "=" * 60)
    print("STEP 5: POD24 analysis")
    print("=" * 60)
    pod24_valid = df[df["pod24"].notna()].copy()
    pod24_pos = pod24_valid[pod24_valid["pod24"] == 1]
    pod24_neg = pod24_valid[pod24_valid["pod24"] == 0]
    print(f"  POD24+: {len(pod24_pos)}, POD24-: {len(pod24_neg)}")

    for m in metrics_all:
        p_vals = pod24_pos[m].dropna()
        n_vals = pod24_neg[m].dropna()
        if len(p_vals) >= 3 and len(n_vals) >= 3:
            u_stat, p = stats.mannwhitneyu(p_vals, n_vals, alternative="two-sided")
            sig = "***" if p < 0.001 else (
                "**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"    {METRIC_LABELS.get(m, m):30s}  "
                  f"POD24+={p_vals.median()*100:.2f}% vs POD24-={n_vals.median()*100:.2f}%  "
                  f"p={p:.4f} {sig}")

    # Step 6: Figure
    print("\n" + "=" * 60)
    print("STEP 6: Generate figure")
    print("=" * 60)
    make_figure(df, pfs_results, os_results, args.output_dir)

    # Save CSV
    csv_out = Path(args.output_dir) / "fdc_survival_metrics.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved metrics CSV: {csv_out}")


if __name__ == "__main__":
    main()
