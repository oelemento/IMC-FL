#!/usr/bin/env python
"""Survival figure v2: cell type and marker screens → transformation → POD24.

Reads the pre-computed survival_covariates.csv and produces a focused
publication figure:
  (a) Cell type fraction forest (PFS, treated) — all annotation types
  (b) Cell type fraction forest (transformation) — all annotation types
  (c) S-panel marker intensity forest (PFS, treated)
  (d) KM: CD14 vs PFS
  (e) KM: CD14 vs PFS (Kaplan-Meier)
  (f) Multivariate Cox: CD14 progressive adjustment
  (g) Full multivariate Cox: all covariates
  (h) POD24 ROC curves (FLIPI, CD14, CD14+FLIPI)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu
from pathlib import Path

# Standardized font sizes (direct-render, no PNG scaling)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22



def panel_label(ax, letter):
    ax.text(-0.18, 1.02, f"$\\bf{{{letter}}}$",
            transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, va="bottom", ha="left")


# ── Shorter display names ────────────────────────────────────────────────────

CT_LABELS = {
    "Activated B / Plasmablast": "Act. B / Plasmablast",
    "B cells": "B cells",
    "B cells (CD20hi)": "B (CD20hi)",
    "B cells (CXCR5hi)": "B (CXCR5hi)",
    "B cells (TOXhi)": "B (TOXhi)",
    "B cells (weak CD20)": "B (weak CD20)",
    "CD4 T cells": "CD4 T cells",
    "CD8 T cells": "CD8 T cells",
    "CD8 T exhausted": "CD8 T exhausted",
    "CD8 T pre-exhausted (TOX+)": "CD8 T pre-exh",
    "Macrophages (GzmB+)": "Mac (GzmB+)",
    "GC B cells": "GC B cells",
    "Macrophages": "Macrophages (T)",
    "T cells": "T cells (generic)",
    "Treg": "Treg",
    # S-panel cell type fractions
    "s_M1_frac": "M1 Mac (S)",
    "s_M2_frac": "M2 Mac (S)",
    "s_mac_frac": "Mac generic (S)",
    "s_myeloid_S100A9_frac": "S100A9+ myeloid (S)",
    "s_DC_frac": "DC (S)",
    "s_FDC_frac": "FDC (S)",
}

MARKER_LABELS = {
    "s_CD14": "CD14",
    "s_CD68": "CD68",
    "s_S100A9": "S100A9",
    "s_VISTA": "VISTA",
    "s_CD163": "CD163",
    "s_CD206": "CD206",
    "s_CD11b": "CD11b",
    "s_CD11c": "CD11c",
    "s_HLA_DR": "HLA-DR",
    "s_HLA_Class_I": "HLA-I",
    "s_CD20": "CD20",
    "s_CD4": "CD4",
    "s_CD8a": "CD8a",
    "s_BCL_2": "BCL-2",
    "s_BCL_6": "BCL-6",
    "s_Ki-67": "Ki-67",
    "s_CD44": "CD44",
    "s_Vimentin": "Vimentin",
    "s_PDPN": "PDPN",
    "s_CD31": "CD31",
    "s_CD146": "CD146",
    "s_CD34": "CD34",
    "s_Fibronectin": "Fibronectin",
    "s_CD21": "CD21",
    "s_CXCL13": "CXCL13",
    "s_CXCL12": "CXCL12",
    "s_CCL21": "CCL21",
    "s_IDO": "IDO",
    "s_PD_L1": "PD-L1",
    "s_CD123": "CD123",
    "s_CD209": "CD209",
    "s_CD1a": "CD1a",
    "s_CD49a": "CD49a",
    "s_SOX9": "SOX9",
    "s_PAX5": "PAX5",
    "s_p_H3s28": "p-H3s28",
}


# ── Univariate Cox helper ────────────────────────────────────────────────────

def unicox(df, metric, time_col="pfs_time", event_col="pfs_event"):
    sub = df[[metric, time_col, event_col]].dropna()
    if len(sub) < 20 or sub[event_col].sum() < 5:
        return None
    sub = sub.copy()
    mu = sub[metric].mean()
    sd = sub[metric].std()
    if sd < 1e-12:
        return None
    sub[metric] = (sub[metric] - mu) / sd  # z-score → HR per SD
    cph = CoxPHFitter()
    try:
        cph.fit(sub, duration_col=time_col, event_col=event_col)
        s = cph.summary.iloc[0]
        return {
            "metric": metric,
            "HR": s["exp(coef)"],
            "lo": s["exp(coef) lower 95%"],
            "hi": s["exp(coef) upper 95%"],
            "p": s["p"],
            "n": len(sub),
        }
    except Exception:
        return None


# ── Forest plot helper ───────────────────────────────────────────────────────

def plot_forest(ax, rows, title, highlight=None, label_map=None):
    """rows = list of dicts with metric, HR, lo, hi, p. Sorted by p ascending
    (best at bottom). highlight = metric name to color red."""
    # Sort so lowest p is at bottom (last drawn = most prominent)
    rows = sorted(rows, key=lambda r: -r["p"])

    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        is_hl = highlight and r["metric"] == highlight
        is_sig = r["p"] < 0.05
        if is_hl:
            color = "#E41A1C"
            lw, ms = 2.5, 9
        elif is_sig:
            color = "#FF8C00"
            lw, ms = 2, 7
        else:
            color = "#999999"
            lw, ms = 1.5, 6

        ax.plot([r["lo"], r["hi"]], [i, i], color=color, linewidth=lw)
        ax.plot(r["HR"], i, "o", color=color, markersize=ms,
                zorder=5 if is_hl else 3)

    ax.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_yticks(y)
    labs = []
    for r in rows:
        m = r["metric"]
        lab = (label_map or {}).get(m, m)
        labs.append(lab)
    ax.set_yticklabels(labs, fontsize=TICK_SIZE)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xscale("log")
    ax.tick_params(axis="x", labelsize=TICK_SIZE)

    # Extend xlim right to create padding for annotations inside axes
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 4)

    # Annotate inside axes (right-aligned at 99% width)
    for i, r in enumerate(rows):
        p = r["p"]
        p_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
        is_hl = highlight and r["metric"] == highlight
        fw = "bold" if is_hl else "normal"
        ax.annotate(
            f"HR={r['HR']:.2f}  {p_str}",
            xy=(0.99, i), xycoords=("axes fraction", "data"),
            ha="right", va="center", fontsize=9, color="#333", fontweight=fw,
        )


# ── Transformation forest helper ─────────────────────────────────────────────

def transformation_test(df, metrics, label_map=None):
    """Mann-Whitney U test: transformers vs non-transformers for each metric."""
    trans = df[df["transformed"] == 1]
    non = df[df["transformed"] == 0]
    rows = []
    for m in metrics:
        t_vals = trans[m].dropna()
        n_vals = non[m].dropna()
        if len(t_vals) < 5 or len(n_vals) < 5:
            continue
        u, p = mannwhitneyu(t_vals, n_vals, alternative="two-sided")
        # Effect size: fold change of medians
        med_t = t_vals.median()
        med_n = n_vals.median()
        rows.append({
            "metric": m,
            "med_trans": med_t,
            "med_non": med_n,
            "fold": med_t / med_n if med_n > 0 else float("inf"),
            "p": p,
        })
    return sorted(rows, key=lambda r: r["p"])


def plot_transformation_forest(ax, rows, label_map=None):
    """Forest-style plot: log2 fold change (transformer / non-transformer)."""
    # Sort so lowest p at bottom
    rows = sorted(rows, key=lambda r: -r["p"])

    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        is_sig = r["p"] < 0.05
        fc = r["fold"]
        log2fc = np.log2(fc) if fc > 0 and np.isfinite(fc) else 0

        if is_sig:
            color = "#E41A1C" if log2fc > 0 else "#377EB8"
            lw, ms = 2, 8
        else:
            color = "#999999"
            lw, ms = 1.5, 6

        ax.plot(log2fc, i, "o", color=color, markersize=ms, zorder=5 if is_sig else 3)

    ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_yticks(y)
    labs = [(label_map or {}).get(r["metric"], r["metric"]) for r in rows]
    ax.set_yticklabels(labs, fontsize=TICK_SIZE)
    ax.set_xlabel("log₂(fold change)", fontsize=LABEL_SIZE)
    ax.set_title("Cell type fractions → Transformation\n(Mann-Whitney U)", fontsize=TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)

    # Extend xlim right for annotation padding
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.5)

    # Annotate inside axes (right-aligned)
    for i, r in enumerate(rows):
        p = r["p"]
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = ""
        p_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
        ax.annotate(
            f"{p_str} {stars}",
            xy=(0.99, i), xycoords=("axes fraction", "data"),
            ha="right", va="center", fontsize=9, color="#333",
            fontweight="bold" if stars else "normal",
        )


# ── KM plot helper ───────────────────────────────────────────────────────────

def plot_km(ax, df, metric, time_col, event_col, ep_label,
            metric_label=None, show_treated_split=False):
    """KM split at median. If show_treated_split, overlay observed patients."""
    sub = df[[metric, time_col, event_col, "treated"]].dropna()
    treated = sub[sub["treated"] == 1]
    median_val = treated[metric].median()

    hi = treated[treated[metric] >= median_val]
    lo = treated[treated[metric] < median_val]

    kmf_hi = KaplanMeierFitter()
    kmf_lo = KaplanMeierFitter()

    label = metric_label or metric
    kmf_hi.fit(hi[time_col], hi[event_col], label=f"High {label} (n={len(hi)})")
    kmf_lo.fit(lo[time_col], lo[event_col], label=f"Low {label} (n={len(lo)})")

    kmf_lo.plot_survival_function(ax=ax, ci_show=True, color="#377EB8")
    kmf_hi.plot_survival_function(ax=ax, ci_show=True, color="#E41A1C")

    lr = logrank_test(
        lo[time_col], hi[time_col],
        event_observed_A=lo[event_col], event_observed_B=hi[event_col],
    )

    # Overlay observed (watch-and-wait) patients
    observed = sub[sub["treated"] == 0]
    if len(observed) >= 3 and show_treated_split:
        kmf_obs = KaplanMeierFitter()
        kmf_obs.fit(observed[time_col], observed[event_col],
                     label=f"Observed (n={len(observed)})")
        kmf_obs.plot_survival_function(ax=ax, ci_show=False, color="#999999",
                                        linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time (years)", fontsize=LABEL_SIZE)
    ax.set_ylabel(f"{ep_label} probability", fontsize=LABEL_SIZE)
    ax.set_title(f"{label} — {ep_label}", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    p_str = f"p={lr.p_value:.4f}" if lr.p_value >= 0.0001 else f"p={lr.p_value:.1e}"
    ax.text(
        0.95, 0.95, f"Log-rank {p_str}",
        transform=ax.transAxes, ha="right", va="top", fontsize=ANNOT_SIZE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.legend(fontsize=LEGEND_SIZE, loc="lower left")
    ax.set_ylim(-0.05, 1.05)


# ── POD24 ROC plot ───────────────────────────────────────────────────────────

def plot_pod24_roc(ax, df):
    """ROC curves: FLIPI-only vs CD14+FLIPI."""
    pod = df[df["pod24"].notna()].copy()
    pod["pod24"] = pod["pod24"].astype(int)

    cd14 = "s_CD14"
    models = []

    # FLIPI only
    sub = pod[["flipi_high", "pod24"]].dropna()
    lr = LogisticRegression(max_iter=1000)
    lr.fit(sub[["flipi_high"]], sub["pod24"])
    prob = lr.predict_proba(sub[["flipi_high"]])[:, 1]
    auc = roc_auc_score(sub["pod24"], prob)
    fpr, tpr, _ = roc_curve(sub["pod24"], prob)
    models.append(("FLIPI only", fpr, tpr, auc, "#999999", 1.5, "--"))

    # CD14 only
    sub = pod[[cd14, "pod24"]].dropna()
    if len(sub) >= 30:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(sub[[cd14]], sub["pod24"])
        prob = lr.predict_proba(sub[[cd14]])[:, 1]
        auc = roc_auc_score(sub["pod24"], prob)
        fpr, tpr, _ = roc_curve(sub["pod24"], prob)
        models.append(("CD14 only", fpr, tpr, auc, "#FF8C00", 2, "-."))

    # CD14 + FLIPI
    sub = pod[[cd14, "flipi_high", "pod24"]].dropna()
    if len(sub) >= 30:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(sub[[cd14, "flipi_high"]], sub["pod24"])
        prob = lr.predict_proba(sub[[cd14, "flipi_high"]])[:, 1]
        auc = roc_auc_score(sub["pod24"], prob)
        fpr, tpr, _ = roc_curve(sub["pod24"], prob)
        models.append(("CD14 + FLIPI", fpr, tpr, auc, "#E41A1C", 2.5, "-"))

    # Plot
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
    for name, fpr, tpr, auc, color, lw, ls in models:
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=LABEL_SIZE)
    ax.set_ylabel("True Positive Rate", fontsize=LABEL_SIZE)
    ax.set_title("POD24 prediction (treated patients)", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    csv = Path("output/hypotheses_v8/survival_covariates.csv")
    df = pd.read_csv(csv)
    print(f"Loaded {len(df)} patients from {csv}")
    print(f"  Treated: {(df['treated']==1).sum()}, Observed: {(df['treated']==0).sum()}")
    print(f"  PFS events: {df['pfs_event'].sum():.0f}, OS events: {df['os_event'].sum():.0f}")

    # ── (a) Cell type fraction screen (T-panel + S-panel annotation types) ──
    ct_cols = [
        "Activated B / Plasmablast", "B cells", "B cells (CD20hi)",
        "B cells (CXCR5hi)", "B cells (TOXhi)", "B cells (weak CD20)",
        "CD4 T cells", "CD8 T cells", "CD8 T exhausted",
        "CD8 T pre-exhausted (TOX+)", "Macrophages (GzmB+)", "GC B cells",
        "Macrophages", "Treg",
    ]
    # S-panel individual cell type fractions (no aggregates)
    s_frac_cols = [
        "s_M1_frac", "s_M2_frac", "s_mac_frac",
        "s_myeloid_S100A9_frac", "s_DC_frac", "s_FDC_frac",
    ]
    all_ct_cols = [c for c in ct_cols + s_frac_cols if c in df.columns]

    # Run on treated patients only for PFS
    treated_df = df[df["treated"] == 1].copy()

    ct_rows = []
    for c in all_ct_cols:
        r = unicox(treated_df, c)
        if r:
            ct_rows.append(r)

    print(f"\n(a) Cell type forest: {len(ct_rows)} testable")
    for r in sorted(ct_rows, key=lambda x: x["p"]):
        lab = CT_LABELS.get(r["metric"], r["metric"])
        sig = "*" if r["p"] < 0.05 else ""
        print(f"  {lab:30s} HR={r['HR']:.3f} [{r['lo']:.3f}–{r['hi']:.3f}] p={r['p']:.4f} {sig}")

    # ── (b) Cell type fraction screen — OS ──────────────────────────────
    os_rows = []
    for c in all_ct_cols:
        r = unicox(treated_df, c, time_col="os_time", event_col="os_event")
        if r:
            os_rows.append(r)

    print(f"\n(b) Cell type forest (OS): {len(os_rows)} testable")
    for r in sorted(os_rows, key=lambda x: x["p"]):
        lab = CT_LABELS.get(r["metric"], r["metric"])
        sig = "*" if r["p"] < 0.05 else ""
        print(f"  {lab:30s} HR={r['HR']:.3f} [{r['lo']:.3f}–{r['hi']:.3f}] p={r['p']:.4f} {sig}")

    # ── (c) Transformation screen ──────────────────────────────────────
    trans_rows = transformation_test(df, all_ct_cols, label_map=CT_LABELS)
    print(f"\n(c) Transformation forest: {len(trans_rows)} testable")
    for r in trans_rows:
        lab = CT_LABELS.get(r["metric"], r["metric"])
        sig = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else (
            "*" if r["p"] < 0.05 else ""))
        print(f"  {lab:30s} fold={r['fold']:.2f}  p={r['p']:.4f} {sig}")

    # ── (e) S-panel marker intensity screen — PFS ─────────────────────
    marker_cols = [c for c in df.columns
                   if c.startswith("s_") and not c.endswith("_frac")
                   and df[c].notna().sum() >= 20]

    marker_rows = []
    for c in marker_cols:
        r = unicox(treated_df, c)
        if r:
            marker_rows.append(r)

    print(f"\n(e) Marker forest: {len(marker_rows)} testable")
    for r in sorted(marker_rows, key=lambda x: x["p"]):
        lab = MARKER_LABELS.get(r["metric"], r["metric"])
        sig = "*" if r["p"] < 0.05 else ""
        print(f"  {lab:20s} HR={r['HR']:.3f} [{r['lo']:.3f}–{r['hi']:.3f}] p={r['p']:.4f} {sig}")

    # ── (g) Multivariate Cox: CD14 adjusted for clinical covariates ──
    # Grade + stage now sourced from the DWS-annotated clinical file (native
    # GRADE column added in the May 2026 BCCA re-annotation). Stage: Ann Arbor
    # 3-4 -> ADV, 1-2 -> LIM (standard FL convention; replaces the prior
    # STAGEGRP-based mapping from master_clinical_ezh2.csv).
    from src.clinical_linkage import load_clinical
    clin = load_clinical()
    grade_map = dict(zip(clin["slide_ID"], clin["GRADE"]))
    stage_map = dict(zip(clin["slide_ID"],
                         pd.to_numeric(clin["ANN ARBOR STAGE"], errors="coerce")))
    treated_df["grade_num"] = treated_df["slide_ID"].map(grade_map).map(
        {"FOLL1": 1, "FOLL2": 2, "FOLL3A": 3}
    )
    treated_df["stage_adv"] = treated_df["slide_ID"].map(stage_map).apply(
        lambda s: 1.0 if (pd.notna(s) and s >= 3) else (0.0 if pd.notna(s) else float("nan"))
    )

    # Multivariate models: progressively adjusted
    mv_results = []
    cd14_col = "s_CD14"

    # Model 1: CD14 univariate (reference)
    sub1 = treated_df[[cd14_col, "pfs_time", "pfs_event"]].dropna().copy()
    mu1, sd1 = sub1[cd14_col].mean(), sub1[cd14_col].std()
    sub1[cd14_col] = (sub1[cd14_col] - mu1) / sd1
    cph1 = CoxPHFitter()
    cph1.fit(sub1, duration_col="pfs_time", event_col="pfs_event")
    s1 = cph1.summary.iloc[0]
    mv_results.append({"label": "CD14 (univariate)", "HR": s1["exp(coef)"],
                        "lo": s1["exp(coef) lower 95%"], "hi": s1["exp(coef) upper 95%"],
                        "p": s1["p"], "n": len(sub1)})

    # Model 2: CD14 + FLIPI
    sub2 = treated_df[[cd14_col, "FLIPI", "pfs_time", "pfs_event"]].dropna()
    sub2 = sub2[sub2["FLIPI"] >= 0].copy()
    sub2[cd14_col] = (sub2[cd14_col] - sub2[cd14_col].mean()) / sub2[cd14_col].std()
    cph2 = CoxPHFitter()
    cph2.fit(sub2, duration_col="pfs_time", event_col="pfs_event")
    s2 = cph2.summary.loc[cd14_col]
    mv_results.append({"label": "CD14 + FLIPI", "HR": s2["exp(coef)"],
                        "lo": s2["exp(coef) lower 95%"], "hi": s2["exp(coef) upper 95%"],
                        "p": s2["p"], "n": len(sub2)})

    # Model 3: CD14 + FLIPI + grade
    sub3 = treated_df[[cd14_col, "FLIPI", "grade_num", "pfs_time", "pfs_event"]].dropna()
    sub3 = sub3[sub3["FLIPI"] >= 0].copy()
    sub3[cd14_col] = (sub3[cd14_col] - sub3[cd14_col].mean()) / sub3[cd14_col].std()
    cph3 = CoxPHFitter()
    cph3.fit(sub3, duration_col="pfs_time", event_col="pfs_event")
    s3 = cph3.summary.loc[cd14_col]
    mv_results.append({"label": "CD14 + FLIPI + grade", "HR": s3["exp(coef)"],
                        "lo": s3["exp(coef) lower 95%"], "hi": s3["exp(coef) upper 95%"],
                        "p": s3["p"], "n": len(sub3)})

    # Model 4: CD14 + FLIPI + grade + stage + age (full model)
    sub4 = treated_df[[cd14_col, "FLIPI", "grade_num", "stage_adv", "AGE",
                        "pfs_time", "pfs_event"]].dropna()
    sub4 = sub4[sub4["FLIPI"] >= 0].copy()
    sub4[cd14_col] = (sub4[cd14_col] - sub4[cd14_col].mean()) / sub4[cd14_col].std()
    sub4["AGE"] = (sub4["AGE"] - sub4["AGE"].mean()) / sub4["AGE"].std()
    cph4 = CoxPHFitter()
    cph4.fit(sub4, duration_col="pfs_time", event_col="pfs_event")
    s4 = cph4.summary.loc[cd14_col]
    mv_results.append({"label": "CD14 + FLIPI + grade\n+ stage + age",
                        "HR": s4["exp(coef)"],
                        "lo": s4["exp(coef) lower 95%"], "hi": s4["exp(coef) upper 95%"],
                        "p": s4["p"], "n": len(sub4)})

    print(f"\n(g) Multivariate Cox — CD14 HR for PFS:")
    for r in mv_results:
        sig = "*" if r["p"] < 0.05 else ""
        print(f"  {r['label']:35s} HR={r['HR']:.3f} [{r['lo']:.3f}–{r['hi']:.3f}] p={r['p']:.4f} n={r['n']} {sig}")

    # Full model summary (all covariates)
    print(f"\n  Full model summary:")
    print(cph4.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_string())

    # ── Build figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24))
    # Top 2 rows: 3-column grid for panels a–f
    gs_top = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.60,
                      left=0.07, right=0.88, top=0.97, bottom=0.30,
                      width_ratios=[1.2, 1.2, 1])
    # Bottom row: separate 2-column grid for panels g–h
    gs_bot = GridSpec(1, 2, figure=fig, wspace=0.55,
                      left=0.07, right=0.88, top=0.24, bottom=0.04,
                      width_ratios=[1, 1.2])

    # (a) Cell type forest — PFS, treated only
    ax_a = fig.add_subplot(gs_top[0, 0])
    panel_label(ax_a, "a")
    best_pfs = min(ct_rows, key=lambda r: r["p"])["metric"]
    plot_forest(ax_a, ct_rows,
                "Cell type fractions → PFS\n(treated patients, univariate Cox)",
                highlight=best_pfs, label_map=CT_LABELS)

    # (b) Cell type forest — OS, treated only
    ax_b = fig.add_subplot(gs_top[0, 1])
    panel_label(ax_b, "b")
    best_os = min(os_rows, key=lambda r: r["p"])["metric"]
    plot_forest(ax_b, os_rows,
                "Cell type fractions → OS\n(treated patients, univariate Cox)",
                highlight=best_os, label_map=CT_LABELS)

    # (c) Transformation forest — all patients
    ax_c = fig.add_subplot(gs_top[0, 2])
    ax_c.text(-0.18, 1.02, r"$\bf{c}$",
              transform=ax_c.transAxes, fontsize=PANEL_LABEL_SIZE, va="bottom", ha="left")
    plot_transformation_forest(ax_c, trans_rows, label_map=CT_LABELS)

    # (d) Marker forest — PFS, treated only
    ax_d = fig.add_subplot(gs_top[1, 0])
    panel_label(ax_d, "d")
    best_marker = min(marker_rows, key=lambda r: r["p"])["metric"]
    plot_forest(ax_d, marker_rows,
                "S-panel markers → PFS\n(treated patients, univariate Cox)",
                highlight=best_marker, label_map=MARKER_LABELS)

    # (e) KM: CD14 vs PFS
    ax_e = fig.add_subplot(gs_top[1, 1])
    panel_label(ax_e, "e")
    plot_km(ax_e, df, "s_CD14", "pfs_time", "pfs_event", "PFS",
            metric_label="CD14 intensity (S)", show_treated_split=True)

    # (f) Multivariate Cox — CD14 HR across progressive adjustment
    ax_f = fig.add_subplot(gs_top[1, 2])
    panel_label(ax_f, "f")

    y_mv = np.arange(len(mv_results))
    for i, r in enumerate(mv_results):
        is_sig = r["p"] < 0.05
        color = "#E41A1C" if is_sig else "#999999"
        lw = 2.5 if i == len(mv_results) - 1 else 2  # emphasize full model
        ms = 9 if i == len(mv_results) - 1 else 7
        ax_f.plot([r["lo"], r["hi"]], [i, i], color=color, linewidth=lw)
        ax_f.plot(r["HR"], i, "o", color=color, markersize=ms, zorder=5)

    ax_f.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax_f.set_yticks(y_mv)
    ax_f.set_yticklabels([r["label"] for r in mv_results], fontsize=TICK_SIZE)
    ax_f.set_xlabel("Hazard Ratio (95% CI) for CD14", fontsize=LABEL_SIZE)
    ax_f.set_title("CD14 → PFS: multivariate Cox\n(progressive adjustment, treated)", fontsize=TITLE_SIZE)
    ax_f.tick_params(axis="x", labelsize=TICK_SIZE)
    ax_f.set_xscale("log")

    # Extend xlim and annotate inside axes
    xmin_f, xmax_f = ax_f.get_xlim()
    ax_f.set_xlim(xmin_f, xmax_f * 4)
    for i, r in enumerate(mv_results):
        p = r["p"]
        p_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
        ax_f.annotate(
            f"HR={r['HR']:.2f}  {p_str}  n={r['n']}",
            xy=(0.99, i), xycoords=("axes fraction", "data"),
            ha="right", va="center", fontsize=9, color="#333",
            fontweight="bold" if i == len(mv_results) - 1 else "normal",
        )

    # (g) Full model: all covariates forest
    ax_g = fig.add_subplot(gs_bot[0, 0])
    panel_label(ax_g, "g")

    full_labels = {cd14_col: "CD14 (per SD)", "FLIPI": "FLIPI score",
                   "grade_num": "Grade (1→3A)", "stage_adv": "Stage (adv vs lim)",
                   "AGE": "Age (per SD)"}
    full_rows = []
    for var in [cd14_col, "FLIPI", "grade_num", "stage_adv", "AGE"]:
        s = cph4.summary.loc[var]
        full_rows.append({
            "metric": var,
            "HR": s["exp(coef)"],
            "lo": s["exp(coef) lower 95%"],
            "hi": s["exp(coef) upper 95%"],
            "p": s["p"],
        })

    y_full = np.arange(len(full_rows))
    for i, r in enumerate(full_rows):
        is_cd14 = r["metric"] == cd14_col
        is_sig = r["p"] < 0.05
        if is_cd14:
            color = "#E41A1C"
            lw, ms = 2.5, 9
        elif is_sig:
            color = "#FF8C00"
            lw, ms = 2, 7
        else:
            color = "#999999"
            lw, ms = 1.5, 6
        ax_g.plot([r["lo"], r["hi"]], [i, i], color=color, linewidth=lw)
        ax_g.plot(r["HR"], i, "o", color=color, markersize=ms, zorder=5)

    ax_g.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax_g.set_yticks(y_full)
    ax_g.set_yticklabels([full_labels.get(r["metric"], r["metric"]) for r in full_rows], fontsize=TICK_SIZE)
    ax_g.set_xlabel("Hazard Ratio (95% CI)", fontsize=LABEL_SIZE)
    ax_g.set_title(f"Full multivariate Cox → PFS\n(n={len(sub4)}, treated patients)", fontsize=TITLE_SIZE)
    ax_g.set_xscale("log")
    ax_g.tick_params(axis="x", labelsize=TICK_SIZE)

    # Extend xlim and annotate inside axes
    xmin_g, xmax_g = ax_g.get_xlim()
    ax_g.set_xlim(xmin_g, xmax_g * 4)
    for i, r in enumerate(full_rows):
        is_cd14 = r["metric"] == cd14_col
        p = r["p"]
        p_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
        ax_g.annotate(
            f"HR={r['HR']:.2f} [{r['lo']:.2f}–{r['hi']:.2f}]  {p_str}",
            xy=(0.99, i), xycoords=("axes fraction", "data"),
            ha="right", va="center", fontsize=9, color="#333",
            fontweight="bold" if is_cd14 else "normal",
        )

    # (h) POD24 ROC: FLIPI vs CD14 vs CD14+FLIPI
    ax_h = fig.add_subplot(gs_bot[0, 1])
    panel_label(ax_h, "h")
    plot_pod24_roc(ax_h, treated_df)

    out = Path("output/hypotheses_v8/fig_survival_v2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
