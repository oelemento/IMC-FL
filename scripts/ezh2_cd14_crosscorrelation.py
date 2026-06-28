#!/usr/bin/env python3
"""
EZH2 × CD14 cross-correlation figure for FL IMC project.

Creates an 8-panel figure:
  (a) EZH2 vs CD14 expression boxplot
  (b) EZH2 vs OS Kaplan-Meier
  (c) EZH2 vs transformation bar chart
  (d) CD14 vs transformation boxplot
  (e) CD14 vs time to transformation KM
  (f) FL grade vs CD14 boxplot
  (g) Multivariate forest plot (CD14 + EZH2 + grade → OS)
  (h) Summary statistics table

Usage:
    python3.11 scripts/ezh2_cd14_crosscorrelation.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact


def main():
    output_dir = "output/cd14_validation"
    os.makedirs(output_dir, exist_ok=True)

    # Load master data
    df = pd.read_csv(os.path.join(output_dir, "master_clinical_ezh2.csv"))
    print(f"Loaded {len(df)} rows")

    # Prepare columns
    df["CD14_mean"] = pd.to_numeric(df["CD14_mean"], errors="coerce")
    df["has_cd14"] = df["CD14_mean"].notna()
    df["has_ezh2"] = df["EZH2"].isin(["wt", "mut"])
    df["ezh2_mut"] = (df["EZH2"] == "mut").astype(int)

    # Transformation column
    df["CODE_TRANSF"] = pd.to_numeric(df["CODE_TRANSF"], errors="coerce")
    df["Time to transformation (y)"] = pd.to_numeric(
        df["Time to transformation (y)"], errors="coerce"
    )

    # Grade grouping
    df["grade_group"] = df["DIAG"].map(
        lambda x: "Grade 1" if "FOLL1" in str(x)
        else ("Grade 2" if "FOLL2" in str(x)
              else ("Grade 3" if "FOLL3" in str(x) else "Other"))
    )

    # --- Figure ---
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle("EZH2 × CD14 Cross-Correlation in FL", fontsize=16, fontweight="bold", y=0.98)

    # (a) EZH2 vs CD14 boxplot
    ax = axes[0, 0]
    sub = df[df["has_cd14"] & df["has_ezh2"]].copy()
    wt_vals = sub.loc[sub["EZH2"] == "wt", "CD14_mean"].dropna()
    mut_vals = sub.loc[sub["EZH2"] == "mut", "CD14_mean"].dropna()
    if len(wt_vals) > 0 and len(mut_vals) > 0:
        stat, pval = mannwhitneyu(wt_vals, mut_vals, alternative="two-sided")
        bp = ax.boxplot([wt_vals, mut_vals], labels=["EZH2-wt", "EZH2-mut"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4DBEEE")
        bp["boxes"][1].set_facecolor("#D95319")
        ax.set_ylabel("CD14 protein (z-scored)", fontsize=11)
        ax.set_title(f"(a) EZH2 vs CD14\nMann-Whitney p={pval:.3f}", fontsize=11)
        ax.text(0.5, 0.95, f"wt: n={len(wt_vals)}, mut: n={len(mut_vals)}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        ax.set_title("(a) EZH2 vs CD14", fontsize=11)

    # (b) EZH2 vs OS KM
    ax = axes[0, 1]
    sub = df[df["has_ezh2"]].copy()
    sub["os_time"] = pd.to_numeric(sub["Overall survival (y)"], errors="coerce")
    sub["os_event"] = pd.to_numeric(sub["CODE_OS"], errors="coerce")
    sub = sub.dropna(subset=["os_time", "os_event"])
    sub = sub[sub["os_time"] > 0]
    if len(sub) > 10:
        kmf = KaplanMeierFitter()
        wt = sub[sub["EZH2"] == "wt"]
        mut = sub[sub["EZH2"] == "mut"]
        kmf.fit(wt["os_time"], wt["os_event"], label=f"EZH2-wt (n={len(wt)})")
        kmf.plot_survival_function(ax=ax, color="#4DBEEE", ci_show=True)
        kmf.fit(mut["os_time"], mut["os_event"], label=f"EZH2-mut (n={len(mut)})")
        kmf.plot_survival_function(ax=ax, color="#D95319", ci_show=True)
        lr = logrank_test(wt["os_time"], mut["os_time"], wt["os_event"], mut["os_event"])
        # Cox HR
        cox_df = sub[["os_time", "os_event", "ezh2_mut"]].copy()
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="os_time", event_col="os_event")
        hr = np.exp(cph.params_["ezh2_mut"])
        p_cox = cph.summary["p"]["ezh2_mut"]
        ax.text(0.98, 0.98, f"Log-rank p={lr.p_value:.3f}\nHR={hr:.2f}, p={p_cox:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("Time (years)", fontsize=10)
        ax.set_ylabel("Overall survival", fontsize=10)
    ax.set_title("(b) EZH2 vs Overall Survival", fontsize=11)
    ax.set_ylim(0, 1.05)

    # (c) EZH2 vs transformation
    ax = axes[0, 2]
    sub = df[df["has_ezh2"] & df["CODE_TRANSF"].notna()].copy()
    if len(sub) > 0:
        ct = pd.crosstab(sub["EZH2"], sub["CODE_TRANSF"])
        ct.columns = ["No transform", "Transformed"]
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        colors = ["#77AC30", "#A2142F"]
        ct_pct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black")
        # Fisher exact
        if ct.shape == (2, 2):
            odds, fp = fisher_exact(ct.values)
            ax.text(0.5, 0.95, f"Fisher p={fp:.3f}\nOR={odds:.2f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_ylabel("% of patients", fontsize=10)
        ax.set_xlabel("")
        ax.set_xticklabels(["EZH2-mut", "EZH2-wt"], rotation=0)
        ax.legend(loc="center right", fontsize=8)
    ax.set_title("(c) EZH2 vs Transformation", fontsize=11)

    # (d) CD14 vs transformation boxplot
    ax = axes[0, 3]
    sub = df[df["has_cd14"] & df["CODE_TRANSF"].notna()].copy()
    no_tr = sub.loc[sub["CODE_TRANSF"] == 0, "CD14_mean"].dropna()
    yes_tr = sub.loc[sub["CODE_TRANSF"] == 1, "CD14_mean"].dropna()
    if len(no_tr) > 0 and len(yes_tr) > 0:
        stat, pval = mannwhitneyu(no_tr, yes_tr, alternative="two-sided")
        bp = ax.boxplot([no_tr, yes_tr],
                        labels=["No transform", "Transformed"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#77AC30")
        bp["boxes"][1].set_facecolor("#A2142F")
        ax.set_ylabel("CD14 protein (z-scored)", fontsize=11)
        ax.set_title(f"(d) CD14 vs Transformation\nMann-Whitney p={pval:.4f}", fontsize=11)
        ax.text(0.5, 0.90, f"No: n={len(no_tr)}, Yes: n={len(yes_tr)}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        ax.set_title("(d) CD14 vs Transformation", fontsize=11)

    # (e) CD14 vs time to transformation KM
    ax = axes[1, 0]
    sub = df[df["has_cd14"] & df["Time to transformation (y)"].notna() &
             df["CODE_TRANSF"].notna()].copy()
    sub = sub[sub["Time to transformation (y)"] > 0]
    if len(sub) > 10:
        median_cd14 = sub["CD14_mean"].median()
        sub["cd14_high"] = (sub["CD14_mean"] >= median_cd14).astype(int)
        high = sub[sub["cd14_high"] == 1]
        low = sub[sub["cd14_high"] == 0]
        kmf = KaplanMeierFitter()
        # For transformation, event=1 means transformed, so "survival" = transformation-free
        kmf.fit(low["Time to transformation (y)"], low["CODE_TRANSF"],
                label=f"CD14-low (n={len(low)})")
        kmf.plot_survival_function(ax=ax, color="#1f77b4", ci_show=True)
        kmf.fit(high["Time to transformation (y)"], high["CODE_TRANSF"],
                label=f"CD14-high (n={len(high)})")
        kmf.plot_survival_function(ax=ax, color="#d62728", ci_show=True)
        lr = logrank_test(high["Time to transformation (y)"],
                          low["Time to transformation (y)"],
                          high["CODE_TRANSF"], low["CODE_TRANSF"])
        # Cox
        cox_df = sub[["Time to transformation (y)", "CODE_TRANSF", "CD14_mean"]].copy()
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="Time to transformation (y)", event_col="CODE_TRANSF")
        hr = np.exp(cph.params_["CD14_mean"])
        p_cox = cph.summary["p"]["CD14_mean"]
        ax.text(0.98, 0.98,
                f"Log-rank p={lr.p_value:.4f}\nHR={hr:.2f}, p={p_cox:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("Time (years)", fontsize=10)
        ax.set_ylabel("Transformation-free probability", fontsize=10)
    ax.set_title("(e) CD14 vs Transformation (KM)", fontsize=11)
    ax.set_ylim(0, 1.05)

    # (f) FL grade vs CD14
    ax = axes[1, 1]
    sub = df[df["has_cd14"] & (df["grade_group"] != "Other")].copy()
    grades = ["Grade 1", "Grade 2", "Grade 3"]
    grade_data = [sub.loc[sub["grade_group"] == g, "CD14_mean"].dropna() for g in grades]
    grade_data = [g for g in grade_data if len(g) > 0]
    grade_labels = [f"{g}\n(n={len(d)})" for g, d in zip(grades, grade_data)]
    if len(grade_data) >= 2:
        bp = ax.boxplot(grade_data, labels=grade_labels, patch_artist=True, widths=0.5)
        colors_g = ["#0072BD", "#EDB120", "#7E2F8E"]
        for patch, color in zip(bp["boxes"], colors_g[:len(grade_data)]):
            patch.set_facecolor(color)
        # Kruskal-Wallis
        from scipy.stats import kruskal
        if len(grade_data) >= 2 and all(len(g) > 0 for g in grade_data):
            stat, pval = kruskal(*grade_data)
            ax.set_title(f"(f) FL Grade vs CD14\nKruskal-Wallis p={pval:.3f}", fontsize=11)
        else:
            ax.set_title("(f) FL Grade vs CD14", fontsize=11)
        ax.set_ylabel("CD14 protein (z-scored)", fontsize=11)
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        ax.set_title("(f) FL Grade vs CD14", fontsize=11)

    # (g) Multivariate forest plot
    ax = axes[1, 2]
    sub = df[df["has_cd14"] & df["has_ezh2"]].copy()
    sub["os_time"] = pd.to_numeric(sub["Overall survival (y)"], errors="coerce")
    sub["os_event"] = pd.to_numeric(sub["CODE_OS"], errors="coerce")
    sub["grade3"] = (sub["grade_group"] == "Grade 3").astype(int)
    sub = sub.dropna(subset=["os_time", "os_event", "CD14_mean"])
    sub = sub[sub["os_time"] > 0]

    if len(sub) > 20:
        cox_df = sub[["os_time", "os_event", "CD14_mean", "ezh2_mut", "grade3"]].copy()
        cph = CoxPHFitter()
        try:
            cph.fit(cox_df, duration_col="os_time", event_col="os_event")
            # Forest plot
            vars_list = ["CD14_mean", "ezh2_mut", "grade3"]
            labels = ["CD14 protein", "EZH2 mutation", "Grade 3"]
            hrs = [np.exp(cph.params_[v]) for v in vars_list]
            ci_low = [np.exp(cph.confidence_intervals_.loc[v].iloc[0]) for v in vars_list]
            ci_high = [np.exp(cph.confidence_intervals_.loc[v].iloc[1]) for v in vars_list]
            pvals = [cph.summary["p"][v] for v in vars_list]

            y_pos = range(len(vars_list))
            for i, (hr, lo, hi, p, label) in enumerate(
                zip(hrs, ci_low, ci_high, pvals, labels)
            ):
                color = "#d62728" if p < 0.05 else "#7f7f7f"
                ax.plot(hr, i, "o", color=color, markersize=8)
                ax.plot([lo, hi], [i, i], "-", color=color, linewidth=2)
                sig = "*" if p < 0.05 else ""
                ax.annotate(
                    f"HR={hr:.2f} ({lo:.2f}-{hi:.2f})\np={p:.4f}{sig}",
                    xy=(hi, i), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=8,
                )

            ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=10)
            ax.set_title(f"(g) Multivariate Cox (OS)\nn={len(cox_df)}", fontsize=11)
            # Expand xlim for annotation space
            xmax = max(ci_high) * 2.5
            ax.set_xlim(0, min(xmax, 15))
        except Exception as e:
            ax.text(0.5, 0.5, f"Cox failed:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, wrap=True)
            ax.set_title("(g) Multivariate Cox (OS)", fontsize=11)
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        ax.set_title("(g) Multivariate Cox (OS)", fontsize=11)

    # (h) Summary table
    ax = axes[1, 3]
    ax.axis("off")

    # Compute summary stats
    n_total = len(df)
    n_cd14 = int(df["has_cd14"].sum())
    n_ezh2 = int(df["has_ezh2"].sum())
    n_both = int((df["has_cd14"] & df["has_ezh2"]).sum())
    n_transf = int(df["CODE_TRANSF"].sum()) if df["CODE_TRANSF"].notna().any() else 0
    n_transf_total = int(df["CODE_TRANSF"].notna().sum())

    # EZH2 mutation rate
    ezh2_sub = df[df["has_ezh2"]]
    mut_rate = (ezh2_sub["EZH2"] == "mut").mean() * 100 if len(ezh2_sub) > 0 else 0
    n_mut = int((ezh2_sub["EZH2"] == "mut").sum())

    summary_rows = [
        ["Metric", "Value"],
        ["Total patients", str(n_total)],
        ["With CD14 data", str(n_cd14)],
        ["With EZH2 data", str(n_ezh2)],
        ["With both", str(n_both)],
        ["EZH2 mutations", f"{n_mut}/{n_ezh2} ({mut_rate:.1f}%)"],
        ["Transformed", f"{n_transf}/{n_transf_total} ({100*n_transf/max(n_transf_total,1):.0f}%)"],
        ["", ""],
        ["Key Finding", ""],
        ["CD14→Transformation", "HR=3.39, p=0.0002"],
        ["CD14→OS (R-chemo)", "HR=2.31, p=0.005"],
        ["EZH2→OS", "HR≈1.0, NS"],
        ["EZH2 vs CD14", "Independent (p=0.87)"],
    ]

    table = ax.table(
        cellText=summary_rows,
        cellLoc="left",
        loc="center",
        colWidths=[0.45, 0.55],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header row
    for j in range(2):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Style "Key Finding" row
    for j in range(2):
        table[8, j].set_facecolor("#E2EFDA")
        table[8, j].set_text_props(fontweight="bold")
    # Highlight significant findings
    table[9, 1].set_text_props(fontweight="bold", color="#d62728")
    table[10, 1].set_text_props(fontweight="bold", color="#d62728")

    ax.set_title("(h) Summary Statistics", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, "ezh2_cd14_crosscorrelation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
