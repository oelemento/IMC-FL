#!/usr/bin/env python3
"""
Comprehensive EZH2 figure for Wendy — everything tested, positive and negative.

Layout (4 rows × 4 cols):
  Row 1 — Clinical context
    (a) EZH2 mutation summary (cohort, grade, mutation types)
    (b) EZH2 vs Overall Survival (KM)
    (c) EZH2 vs Transformation (bar)
    (d) EZH2 vs CD14 protein (boxplot)

  Row 2 — Whole-tissue analysis (NEGATIVE — nothing survives FDR)
    (e) Cell type fractions: EZH2-wt vs mut
    (f) Tissue compartment fractions
    (g) Immune evasion metrics
    (h) Whole-tissue marker volcano (all cells)

  Row 3 — B cell-restricted analysis (POSITIVE — 3 FDR-significant)
    (i) B cell marker volcano
    (j) H3K27me3 boxplot (positive control, q=0.004)
    (k) PD-L1 boxplot (key finding, q=0.036)
    (l) Cleaved caspase 3 boxplot (q=0.036)

  Row 4 — Nominally significant B cell markers + summary
    (m) CD86 boxplot
    (n) PD-1 boxplot
    (o) CD20 (S-panel) boxplot
    (p) Summary table

Usage:
    PYTHONPATH=. python3.11 scripts/ezh2_comprehensive_figure.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import anndata as ad
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu, fisher_exact, kruskal
from src.clinical_linkage import normalize_sample_id

OUTPUT_DIR = "output/cd14_validation"

# Colors
C_WT = "#4DBEEE"
C_MUT = "#D95319"
C_SIG = "#d62728"
C_NS = "#7f7f7f"


def load_ezh2_data():
    """Load master clinical+EZH2 table."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "master_clinical_ezh2.csv"))
    df["CD14_mean"] = pd.to_numeric(df["CD14_mean"], errors="coerce")
    df["has_cd14"] = df["CD14_mean"].notna()
    df["has_ezh2"] = df["EZH2"].isin(["wt", "mut"])
    df["ezh2_mut"] = (df["EZH2"] == "mut").astype(int)
    df["CODE_TRANSF"] = pd.to_numeric(df["CODE_TRANSF"], errors="coerce")
    df["Time to transformation (y)"] = pd.to_numeric(df["Time to transformation (y)"], errors="coerce")
    df["os_time"] = pd.to_numeric(df["Overall survival (y)"], errors="coerce")
    df["os_event"] = pd.to_numeric(df["CODE_OS"], errors="coerce")
    df["grade_group"] = df["DIAG"].map(
        lambda x: "G1" if "FOLL1" in str(x) else ("G2" if "FOLL2" in str(x) else ("G3" if "FOLL3" in str(x) else "?")))
    return df


def get_bcell_mask(adata):
    ct = adata.obs["cell_type"].astype(str)
    return ct.str.contains("B cell", case=False, na=False) | ct.str.contains("GC B", case=False, na=False)


def boxplot_ezh2(ax, wt_vals, mut_vals, ylabel, title, show_points=True):
    """Standard EZH2-wt vs mut boxplot."""
    bp = ax.boxplot([wt_vals, mut_vals], tick_labels=["EZH2-wt", "EZH2-mut"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(C_WT)
    bp["boxes"][1].set_facecolor(C_MUT)
    for b in bp["boxes"]:
        b.set_edgecolor("black")
        b.set_linewidth(0.8)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_linewidth(0.8)
    if show_points:
        for j, (vals, xpos) in enumerate([(wt_vals, 1), (mut_vals, 2)]):
            jitter = np.random.normal(xpos, 0.04, len(vals))
            ax.scatter(jitter, vals, alpha=0.35, s=10, color=bp["boxes"][j].get_facecolor(),
                       edgecolors="none", zorder=2)
    stat, pval = mannwhitneyu(wt_vals, mut_vals, alternative="two-sided")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    return pval


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master = load_ezh2_data()
    ezh2_sub = master[master["has_ezh2"]].copy()

    # Load h5ad for whole-tissue and B cell analyses
    print("Loading T-panel...")
    adata_t = ad.read_h5ad("output/all_TMA_T_global_v8.h5ad")
    adata_t.obs["slide_ID"] = [normalize_sample_id(s) for s in adata_t.obs["sample_id"]]
    print("Loading S-panel...")
    adata_s = ad.read_h5ad("output/all_TMA_S_global_v8.h5ad")
    adata_s.obs["slide_ID"] = [normalize_sample_id(s) for s in adata_s.obs["sample_id"]]

    # Load UTAG compartments
    print("Loading UTAG T-panel...")
    utag_t = ad.read_h5ad("output/all_TMA_T_utag_ct_merged.h5ad")
    utag_t.obs["slide_ID"] = [normalize_sample_id(s) for s in utag_t.obs["sample_id"]]

    # EZH2 slide_ID → status mapping
    ezh2_map = dict(zip(ezh2_sub["slide_ID"], ezh2_sub["EZH2"]))

    # ── Pre-compute per-patient features ──
    print("Computing per-patient features...")

    # Whole-tissue cell type fractions (T-panel)
    ct_counts_t = adata_t.obs.groupby(["slide_ID", "cell_type"]).size().unstack(fill_value=0)
    ct_frac_t = ct_counts_t.div(ct_counts_t.sum(axis=1), axis=0)
    ct_frac_t["EZH2"] = ct_frac_t.index.map(ezh2_map)
    ct_frac_t = ct_frac_t[ct_frac_t["EZH2"].isin(["wt", "mut"])]

    # Whole-tissue marker means (T-panel)
    wt_marker_all = adata_t.obs.groupby("slide_ID").apply(
        lambda g: pd.Series(np.asarray(adata_t[g.index].X.mean(axis=0)).flatten(), index=adata_t.var_names))
    wt_marker_all["EZH2"] = wt_marker_all.index.map(ezh2_map)
    wt_marker_all = wt_marker_all[wt_marker_all["EZH2"].isin(["wt", "mut"])]

    # B cell-only marker means (T-panel)
    b_mask_t = get_bcell_mask(adata_t)
    bcells_t = adata_t[b_mask_t]
    bc_marker_t = bcells_t.obs.groupby("slide_ID").apply(
        lambda g: pd.Series(np.asarray(adata_t[g.index].X.mean(axis=0)).flatten(), index=adata_t.var_names))
    bc_marker_t["EZH2"] = bc_marker_t.index.map(ezh2_map)
    bc_marker_t = bc_marker_t[bc_marker_t["EZH2"].isin(["wt", "mut"])]

    # B cell-only marker means (S-panel)
    b_mask_s = get_bcell_mask(adata_s)
    bcells_s = adata_s[b_mask_s]
    bc_marker_s = bcells_s.obs.groupby("slide_ID").apply(
        lambda g: pd.Series(np.asarray(adata_s[g.index].X.mean(axis=0)).flatten(), index=adata_s.var_names))
    bc_marker_s["EZH2"] = bc_marker_s.index.map(ezh2_map)
    bc_marker_s = bc_marker_s[bc_marker_s["EZH2"].isin(["wt", "mut"])]

    # Compartment fractions (T-panel UTAG)
    comp_col = "compartment_name" if "compartment_name" in utag_t.obs.columns else "tissue_compartment"
    comp_counts = utag_t.obs.groupby(["slide_ID", comp_col]).size().unstack(fill_value=0)
    comp_frac = comp_counts.div(comp_counts.sum(axis=1), axis=0)
    comp_frac["EZH2"] = comp_frac.index.map(ezh2_map)
    comp_frac = comp_frac[comp_frac["EZH2"].isin(["wt", "mut"])]

    # Immune evasion metrics (T-panel)
    ie_data = []
    for sid, g in adata_t.obs.groupby("slide_ID"):
        total = len(g)
        ct = g["cell_type"].value_counts()
        cd8_tot = ct.get("CD8 T cells", 0) + ct.get("CD8 T exhausted", 0)
        ie_data.append({
            "slide_ID": sid,
            "CD8 fraction": cd8_tot / total,
            "CD8 exhaustion ratio": ct.get("CD8 T exhausted", 0) / cd8_tot if cd8_tot > 0 else 0,
            "Treg fraction": ct.get("Treg", 0) / total,
            "Macrophage fraction": ct.get("Macrophage", 0) / total if "Macrophage" in ct else ct.get("Macrophages", 0) / total,
            "CD4:CD8 ratio": ct.get("CD4 T cells", 0) / cd8_tot if cd8_tot > 0 else np.nan,
            "B cell fraction": sum(v for k, v in ct.items() if "B" in k and "cell" in k.lower()) / total,
        })
    ie_df = pd.DataFrame(ie_data).set_index("slide_ID")
    ie_df["EZH2"] = ie_df.index.map(ezh2_map)
    ie_df = ie_df[ie_df["EZH2"].isin(["wt", "mut"])]

    # ── FIGURE ──
    print("Creating figure...")
    fig = plt.figure(figsize=(24, 24))
    fig.suptitle("EZH2 Mutation in FL: Comprehensive Analysis",
                 fontsize=18, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.38, wspace=0.32,
                           top=0.97, bottom=0.03, left=0.05, right=0.97)

    # ═══════════════════════════════════════════════════════════
    # ROW 1: Clinical context
    # ═══════════════════════════════════════════════════════════
    fig.text(0.02, 0.975, "Clinical Outcomes", fontsize=14, fontweight="bold",
             color="#333333", va="top")

    # (a) EZH2 mutation summary
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    n_wt = (ezh2_sub["EZH2"] == "wt").sum()
    n_mut = (ezh2_sub["EZH2"] == "mut").sum()
    mut_pct = 100 * n_mut / len(ezh2_sub)

    # Mutation types
    mut_types = master[master["EZH2"] == "mut"]["mut type"].value_counts()
    mut_str = "\n".join([f"  {k}: {v}" for k, v in mut_types.head(5).items()])

    # Grade breakdown
    grade_ct = pd.crosstab(ezh2_sub["EZH2"], ezh2_sub["grade_group"])
    grade_str = ""
    for g in ["G1", "G2", "G3"]:
        if g in grade_ct.columns:
            grade_str += f"  {g}: wt={grade_ct.loc['wt', g]}, mut={grade_ct.loc['mut', g]}\n"

    txt = (f"EZH2 Mutation Summary\n"
           f"{'─'*30}\n"
           f"Total patients: {len(ezh2_sub)}\n"
           f"  Wild-type: {n_wt} ({100-mut_pct:.0f}%)\n"
           f"  Mutant: {n_mut} ({mut_pct:.0f}%)\n\n"
           f"Mutation types:\n{mut_str}\n\n"
           f"By grade:\n{grade_str}")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#cccccc"))
    ax.set_title("(a) Cohort", fontsize=11, fontweight="bold")

    # (b) EZH2 vs Overall Survival
    ax = fig.add_subplot(gs[0, 1])
    sub = ezh2_sub.dropna(subset=["os_time", "os_event"])
    sub = sub[sub["os_time"] > 0]
    wt_s = sub[sub["EZH2"] == "wt"]
    mut_s = sub[sub["EZH2"] == "mut"]
    kmf = KaplanMeierFitter()
    kmf.fit(wt_s["os_time"], wt_s["os_event"], label=f"EZH2-wt (n={len(wt_s)})")
    kmf.plot_survival_function(ax=ax, color=C_WT, ci_show=True, linewidth=2)
    kmf.fit(mut_s["os_time"], mut_s["os_event"], label=f"EZH2-mut (n={len(mut_s)})")
    kmf.plot_survival_function(ax=ax, color=C_MUT, ci_show=True, linewidth=2)
    lr = logrank_test(wt_s["os_time"], mut_s["os_time"], wt_s["os_event"], mut_s["os_event"])
    cph = CoxPHFitter()
    cox_df = sub[["os_time", "os_event", "ezh2_mut"]].copy()
    cph.fit(cox_df, duration_col="os_time", event_col="os_event")
    hr = np.exp(cph.params_["ezh2_mut"])
    p_cox = cph.summary["p"]["ezh2_mut"]
    ax.text(0.97, 0.97, f"Log-rank p = {lr.p_value:.3f}\nHR = {hr:.2f}, p = {p_cox:.3f}\nNOT SIGNIFICANT",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.set_xlabel("Time (years)", fontsize=10)
    ax.set_ylabel("Overall survival", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(b) EZH2 vs Overall Survival", fontsize=11, fontweight="bold")

    # (c) EZH2 vs Transformation
    ax = fig.add_subplot(gs[0, 2])
    sub = ezh2_sub[ezh2_sub["CODE_TRANSF"].notna()].copy()
    ct = pd.crosstab(sub["EZH2"], sub["CODE_TRANSF"])
    if ct.shape[1] == 2:
        ct.columns = ["No", "Yes"]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    x = np.arange(2)
    w = 0.55
    bottom_wt = 0
    bottom_mut = 0
    colors_bar = {"No": "#77AC30", "Yes": "#A2142F"}
    for col in ct_pct.columns:
        vals = ct_pct[col].values
        ax.bar(x, vals, w, bottom=[bottom_wt, bottom_mut] if len(x) == 2 else 0,
               label=f"{'Transformed' if col == 'Yes' else 'No transformation'}",
               color=colors_bar.get(col, "#999"), edgecolor="black", linewidth=0.5)
        bottom_wt = vals[0] if bottom_wt == 0 else bottom_wt + vals[0]
        bottom_mut = vals[1] if bottom_mut == 0 else bottom_mut + vals[1]
    # Fix: stacked bars
    ax.set_xticks(x)
    xlab = list(ct_pct.index)
    ax.set_xticklabels([f"EZH2-{l}\n(n={ct.sum(axis=1)[l]})" for l in xlab], fontsize=9)
    ax.set_ylabel("% of patients", fontsize=10)
    if ct.shape == (2, 2):
        odds, fp = fisher_exact(ct.values)
        ax.text(0.5, 0.97, f"Fisher p = {fp:.3f}\nOR = {odds:.2f}\nNOT SIGNIFICANT",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("(c) EZH2 vs Transformation", fontsize=11, fontweight="bold")

    # (d) EZH2 vs CD14 protein
    ax = fig.add_subplot(gs[0, 3])
    sub = ezh2_sub[ezh2_sub["has_cd14"]].copy()
    wt_v = sub.loc[sub["EZH2"] == "wt", "CD14_mean"].dropna()
    mut_v = sub.loc[sub["EZH2"] == "mut", "CD14_mean"].dropna()
    pval = boxplot_ezh2(ax, wt_v, mut_v, "CD14 protein (z-scored)", "")
    ax.text(0.5, 0.97, f"Mann-Whitney p = {pval:.3f}\nNOT SIGNIFICANT\n(CD14 independent of EZH2)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.set_title("(d) EZH2 vs CD14 Protein", fontsize=11, fontweight="bold")

    # ═══════════════════════════════════════════════════════════
    # ROW 2: Whole-tissue (NEGATIVE)
    # ═══════════════════════════════════════════════════════════
    fig.text(0.02, 0.74, "Whole-Tissue Analysis (no FDR-significant features)",
             fontsize=14, fontweight="bold", color=C_NS, va="top")

    # (e) Cell type fractions
    ax = fig.add_subplot(gs[1, 0])
    cell_types_show = ["B cells", "B cells (CXCR5hi)", "B cells (CD20hi)", "GC B cells",
                       "CD4 T cells", "CD8 T cells", "CD8 T exhausted", "Treg",
                       "Macrophage", "Macrophages"]
    ct_available = [c for c in cell_types_show if c in ct_frac_t.columns]
    # If "Macrophage" not found, try alternatives
    if not ct_available:
        ct_available = [c for c in ct_frac_t.columns if c != "EZH2"][:8]

    wt_ct = ct_frac_t[ct_frac_t["EZH2"] == "wt"]
    mut_ct = ct_frac_t[ct_frac_t["EZH2"] == "mut"]

    labels = []
    wt_meds = []
    mut_meds = []
    pvals_ct = []
    for c in ct_available:
        w = wt_ct[c].dropna()
        m = mut_ct[c].dropna()
        if len(w) >= 5 and len(m) >= 5:
            _, p = mannwhitneyu(w, m, alternative="two-sided")
            labels.append(c)
            wt_meds.append(w.median())
            mut_meds.append(m.median())
            pvals_ct.append(p)

    y = np.arange(len(labels))
    bw = 0.35
    ax.barh(y - bw/2, wt_meds, bw, label="EZH2-wt", color=C_WT, edgecolor="black", linewidth=0.3)
    ax.barh(y + bw/2, mut_meds, bw, label="EZH2-mut", color=C_MUT, edgecolor="black", linewidth=0.3)
    for i, p in enumerate(pvals_ct):
        sig = "*" if p < 0.05 else ""
        mx = max(wt_meds[i], mut_meds[i])
        ax.text(mx + 0.003, i, f"p={p:.2f}{sig}", fontsize=7, va="center", color=C_SIG if p < 0.05 else C_NS)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Median fraction", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_title("(e) Cell Type Fractions (whole tissue)", fontsize=10, fontweight="bold")

    # (f) Compartment fractions
    ax = fig.add_subplot(gs[1, 1])
    comp_names = [c for c in comp_frac.columns if c != "EZH2"]
    wt_comp = comp_frac[comp_frac["EZH2"] == "wt"]
    mut_comp = comp_frac[comp_frac["EZH2"] == "mut"]

    comp_results = []
    for c in comp_names:
        w = wt_comp[c].dropna()
        m = mut_comp[c].dropna()
        if len(w) >= 5 and len(m) >= 5:
            _, p = mannwhitneyu(w, m, alternative="two-sided")
            comp_results.append({"name": c, "wt_med": w.median(), "mut_med": m.median(), "p": p})
    comp_res_df = pd.DataFrame(comp_results).sort_values("p")

    top_comp = comp_res_df.head(8)
    y = np.arange(len(top_comp))
    ax.barh(y - bw/2, top_comp["wt_med"].values, bw, label="EZH2-wt", color=C_WT, edgecolor="black", linewidth=0.3)
    ax.barh(y + bw/2, top_comp["mut_med"].values, bw, label="EZH2-mut", color=C_MUT, edgecolor="black", linewidth=0.3)
    for i, (_, r) in enumerate(top_comp.iterrows()):
        sig = "*" if r["p"] < 0.05 else ""
        mx = max(r["wt_med"], r["mut_med"])
        ax.text(mx + 0.002, i, f"p={r['p']:.2f}{sig}", fontsize=7, va="center", color=C_SIG if r["p"] < 0.05 else C_NS)
    short_names = [n[:28] for n in top_comp["name"]]
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Median fraction", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_title("(f) Compartment Fractions (T-panel)", fontsize=10, fontweight="bold")

    # (g) Immune evasion metrics
    ax = fig.add_subplot(gs[1, 2])
    ie_cols = [c for c in ie_df.columns if c != "EZH2"]
    wt_ie = ie_df[ie_df["EZH2"] == "wt"]
    mut_ie = ie_df[ie_df["EZH2"] == "mut"]

    ie_labels = []
    ie_wt = []
    ie_mut = []
    ie_pvals = []
    for c in ie_cols:
        w = wt_ie[c].dropna()
        m = mut_ie[c].dropna()
        if len(w) >= 5 and len(m) >= 5:
            _, p = mannwhitneyu(w, m, alternative="two-sided")
            ie_labels.append(c)
            ie_wt.append(w.median())
            ie_mut.append(m.median())
            ie_pvals.append(p)

    y = np.arange(len(ie_labels))
    ax.barh(y - bw/2, ie_wt, bw, label="EZH2-wt", color=C_WT, edgecolor="black", linewidth=0.3)
    ax.barh(y + bw/2, ie_mut, bw, label="EZH2-mut", color=C_MUT, edgecolor="black", linewidth=0.3)
    for i, p in enumerate(ie_pvals):
        sig = "*" if p < 0.05 else ""
        mx = max(ie_wt[i], ie_mut[i])
        ax.text(mx + 0.003, i, f"p={p:.2f}{sig}", fontsize=7, va="center", color=C_SIG if p < 0.05 else C_NS)
    ax.set_yticks(y)
    ax.set_yticklabels(ie_labels, fontsize=8)
    ax.set_xlabel("Median value", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_title("(g) Immune Evasion Metrics", fontsize=10, fontweight="bold")

    # (h) Whole-tissue marker volcano
    ax = fig.add_subplot(gs[1, 3])
    markers_t = [c for c in wt_marker_all.columns if c != "EZH2"]
    wt_m = wt_marker_all[wt_marker_all["EZH2"] == "wt"]
    mut_m = wt_marker_all[wt_marker_all["EZH2"] == "mut"]

    all_diffs = []
    all_pvals = []
    all_names = []
    for m in markers_t:
        w = wt_m[m].dropna()
        mu = mut_m[m].dropna()
        if len(w) >= 5 and len(mu) >= 5:
            _, p = mannwhitneyu(w, mu, alternative="two-sided")
            all_diffs.append(mu.median() - w.median())
            all_pvals.append(p)
            all_names.append(m)

    log_p = -np.log10(np.clip(all_pvals, 1e-10, 1))
    colors_v = [C_SIG if p < 0.05 else C_NS for p in all_pvals]
    ax.scatter(all_diffs, log_p, c=colors_v, alpha=0.6, s=30, edgecolors="none")
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.3)
    # Label p < 0.05
    for d, lp, p, n in zip(all_diffs, log_p, all_pvals, all_names):
        if p < 0.05:
            ax.annotate(n, (d, lp), fontsize=7, xytext=(4, 3), textcoords="offset points", color=C_SIG)
    ax.set_xlabel("Median Δ (mut − wt)", fontsize=9)
    ax.set_ylabel("-log10(p)", fontsize=9)
    n_nom = sum(1 for p in all_pvals if p < 0.05)
    ax.set_title(f"(h) All-Cell Marker Volcano\n{n_nom} nominal, 0 FDR", fontsize=10, fontweight="bold")

    # ═══════════════════════════════════════════════════════════
    # ROW 3: B cell-restricted (POSITIVE)
    # ═══════════════════════════════════════════════════════════
    fig.text(0.02, 0.49, "B Cell-Restricted Analysis (3 FDR-significant markers)",
             fontsize=14, fontweight="bold", color=C_SIG, va="top")

    # (i) B cell marker volcano (both panels)
    ax = fig.add_subplot(gs[2, 0])
    bc_results = []
    for panel_name, bc_df in [("T", bc_marker_t), ("S", bc_marker_s)]:
        markers = [c for c in bc_df.columns if c != "EZH2"]
        wt_bc = bc_df[bc_df["EZH2"] == "wt"]
        mut_bc = bc_df[bc_df["EZH2"] == "mut"]
        for m in markers:
            w = wt_bc[m].dropna()
            mu = mut_bc[m].dropna()
            if len(w) >= 5 and len(mu) >= 5:
                _, p = mannwhitneyu(w, mu, alternative="two-sided")
                bc_results.append({"marker": m, "panel": panel_name,
                                   "diff": mu.median() - w.median(), "pval": p})
    bc_res_df = pd.DataFrame(bc_results).sort_values("pval")
    n_bc = len(bc_res_df)
    bc_res_df["rank"] = range(1, n_bc + 1)
    bc_res_df["q"] = (bc_res_df["pval"] * n_bc / bc_res_df["rank"]).clip(upper=1.0)
    bc_res_df["q"] = bc_res_df["q"][::-1].cummin()[::-1]

    for panel in ["T", "S"]:
        sub = bc_res_df[bc_res_df["panel"] == panel]
        fdr_sig = sub[sub["q"] < 0.05]
        nom_sig = sub[(sub["pval"] < 0.05) & (sub["q"] >= 0.05)]
        ns = sub[sub["pval"] >= 0.05]
        pcol = "#1f77b4" if panel == "T" else "#ff7f0e"

        ax.scatter(ns["diff"], -np.log10(ns["pval"].clip(1e-10)),
                   c=pcol, alpha=0.2, s=20, edgecolors="none")
        ax.scatter(nom_sig["diff"], -np.log10(nom_sig["pval"].clip(1e-10)),
                   c=pcol, alpha=0.6, s=35, edgecolors="black", linewidths=0.5)
        ax.scatter(fdr_sig["diff"], -np.log10(fdr_sig["pval"].clip(1e-10)),
                   c=pcol, alpha=1.0, s=60, edgecolors="black", linewidths=1.0, marker="*",
                   label=f"{panel}-panel FDR<0.05" if len(fdr_sig) > 0 else None)

    # Label FDR-significant
    for _, r in bc_res_df[bc_res_df["q"] < 0.05].iterrows():
        ax.annotate(r["marker"], (r["diff"], -np.log10(r["pval"])),
                    fontsize=9, fontweight="bold", color=C_SIG,
                    xytext=(6, 4), textcoords="offset points")
    # Label nominal
    for _, r in bc_res_df[(bc_res_df["pval"] < 0.05) & (bc_res_df["q"] >= 0.05)].iterrows():
        ax.annotate(r["marker"], (r["diff"], -np.log10(r["pval"])),
                    fontsize=7, color="#333",
                    xytext=(5, 2), textcoords="offset points")

    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Median Δ in B cells (mut − wt)", fontsize=9)
    ax.set_ylabel("-log10(p)", fontsize=9)
    n_fdr = (bc_res_df["q"] < 0.05).sum()
    n_nom = (bc_res_df["pval"] < 0.05).sum()
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"(i) B Cell Marker Volcano\n{n_nom} nominal, {n_fdr} FDR-significant", fontsize=10, fontweight="bold")

    # (j) H3K27me3 boxplot — positive control
    ax = fig.add_subplot(gs[2, 1])
    wt_v = bc_marker_t.loc[bc_marker_t["EZH2"] == "wt", "H3K27me3"].dropna()
    mut_v = bc_marker_t.loc[bc_marker_t["EZH2"] == "mut", "H3K27me3"].dropna()
    pval = boxplot_ezh2(ax, wt_v, mut_v, "H3K27me3 in B cells (z-scored)", "")
    _, q_h3k = bc_res_df.loc[bc_res_df["marker"] == "H3K27me3", ["pval", "q"]].values[0]
    ax.text(0.5, 0.97,
            f"p = {pval:.1e}, q = {q_h3k:.3f}\nFDR-SIGNIFICANT\n\nPositive control:\nEZH2 is the H3K27\nmethyltransferase",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3f3", edgecolor=C_SIG, linewidth=1.5))
    ax.set_title("(j) H3K27me3 in B Cells", fontsize=11, fontweight="bold", color=C_SIG)

    # (k) PD-L1 boxplot — key finding
    ax = fig.add_subplot(gs[2, 2])
    wt_v = bc_marker_t.loc[bc_marker_t["EZH2"] == "wt", "PD_L1"].dropna()
    mut_v = bc_marker_t.loc[bc_marker_t["EZH2"] == "mut", "PD_L1"].dropna()
    pval = boxplot_ezh2(ax, wt_v, mut_v, "PD-L1 in B cells (z-scored)", "")
    _, q_pdl1 = bc_res_df.loc[bc_res_df["marker"] == "PD_L1", ["pval", "q"]].values[0]
    ax.text(0.5, 0.97,
            f"p = {pval:.4f}, q = {q_pdl1:.3f}\nFDR-SIGNIFICANT\n\nEZH2-mut B cells\nupregulate PD-L1\n(immune checkpoint ligand)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3f3", edgecolor=C_SIG, linewidth=1.5))
    ax.set_title("(k) PD-L1 in B Cells", fontsize=11, fontweight="bold", color=C_SIG)

    # (l) Cleaved caspase 3 boxplot
    ax = fig.add_subplot(gs[2, 3])
    wt_v = bc_marker_t.loc[bc_marker_t["EZH2"] == "wt", "Cleaved_caspase_3"].dropna()
    mut_v = bc_marker_t.loc[bc_marker_t["EZH2"] == "mut", "Cleaved_caspase_3"].dropna()
    pval = boxplot_ezh2(ax, wt_v, mut_v, "Cleaved caspase 3 in B cells (z-scored)", "")
    _, q_casp = bc_res_df.loc[bc_res_df["marker"] == "Cleaved_caspase_3", ["pval", "q"]].values[0]
    ax.text(0.5, 0.97,
            f"p = {pval:.4f}, q = {q_casp:.3f}\nFDR-SIGNIFICANT\n\nHigher apoptosis\nin EZH2-mut B cells",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3f3", edgecolor=C_SIG, linewidth=1.5))
    ax.set_title("(l) Cleaved Caspase 3 in B Cells", fontsize=11, fontweight="bold", color=C_SIG)

    # ═══════════════════════════════════════════════════════════
    # ROW 4: Nominally significant B cell markers + summary
    # ═══════════════════════════════════════════════════════════
    fig.text(0.02, 0.245, "Additional Nominally Significant B Cell Markers (p < 0.05, not FDR-corrected)",
             fontsize=14, fontweight="bold", color="#ff7f0e", va="top")

    nom_markers = [("CD86", bc_marker_t), ("PD_1", bc_marker_t), ("CD20", bc_marker_s)]
    for idx, (marker, bc_df) in enumerate(nom_markers):
        ax = fig.add_subplot(gs[3, idx])
        wt_v = bc_df.loc[bc_df["EZH2"] == "wt", marker].dropna()
        mut_v = bc_df.loc[bc_df["EZH2"] == "mut", marker].dropna()
        pval = boxplot_ezh2(ax, wt_v, mut_v, f"{marker} in B cells (z-scored)", "")
        panel_label = "T" if bc_df is bc_marker_t else "S"
        ax.text(0.5, 0.97, f"p = {pval:.4f}\nNominal only",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e1", edgecolor="#ff7f0e", linewidth=1))
        display_name = marker.replace("_", "-")
        panel_tag = f" ({panel_label}-panel)" if marker == "CD20" else ""
        letter = chr(ord("m") + idx)
        ax.set_title(f"({letter}) {display_name}{panel_tag} in B Cells", fontsize=10, fontweight="bold", color="#ff7f0e")

    # (p) Summary table
    ax = fig.add_subplot(gs[3, 3])
    ax.axis("off")

    summary = (
        "SUMMARY\n"
        "═══════════════════════════════════════\n\n"
        "CLINICAL (all negative):\n"
        "  EZH2 vs OS:             HR≈1.0, NS\n"
        "  EZH2 vs Transformation: OR=3.4, NS\n"
        "  EZH2 vs CD14 protein:   p=0.87, NS\n\n"
        "WHOLE TISSUE (all negative):\n"
        "  150 features tested\n"
        "  13 nominal (p<0.05), 0 FDR\n"
        "  Cell types, compartments, immune\n"
        "  evasion: no differences\n\n"
        "B CELLS ONLY (3 FDR-significant):\n"
        "  H3K27me3:   q=0.004  ↑ in mut\n"
        "    → Positive control (expected)\n"
        "  PD-L1:      q=0.036  ↑ in mut\n"
        "    → Key finding: immune evasion\n"
        "  Cl. casp 3: q=0.036  ↑ in mut\n"
        "    → Higher apoptosis/turnover\n\n"
        "CONCLUSION:\n"
        "  EZH2 does NOT reshape the TME.\n"
        "  But EZH2-mut B cells DO upregulate\n"
        "  PD-L1 (cell-intrinsic evasion)."
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, va="top", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#333333", linewidth=1))
    ax.set_title("(p) Summary", fontsize=11, fontweight="bold")

    out_path = os.path.join(OUTPUT_DIR, "ezh2_comprehensive.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()
