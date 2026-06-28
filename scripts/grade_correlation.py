#!/usr/bin/env python
"""Exploratory: correlate histological grade with key spatial findings.

Ari suggested checking whether grade (G1/G2/G3a) correlates with
our main findings: CD14 biomarker, immune evasion, exhaustion, etc.
"""
import sys, argparse
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path

EXCLUDE_PAT = r"(?i)(tonsil|prostate|kidney|spleen|adrenal|_Ton_|_Adr_)"


def load_data(t_panel, s_panel, clinical_csv):
    """Load h5ad + clinical, compute per-ROI metrics, merge with grade."""
    clin = pd.read_csv(clinical_csv)
    clin["grade"] = clin["DIAG"].map(
        lambda x: "G1" if "FOLL1" in str(x) else
                  ("G2" if "FOLL2" in str(x) else
                   ("G3a" if "FOLL3" in str(x) else None)))
    clin = clin.dropna(subset=["grade"])
    grade_map = dict(zip(clin["slide_ID"], clin["grade"]))

    # --- T-panel metrics ---
    print("Loading T-panel...")
    ad_t = sc.read_h5ad(t_panel)
    ad_t = ad_t[~ad_t.obs["sample_id"].str.contains(EXCLUDE_PAT, na=False)]
    obs_t = ad_t.obs.copy()

    t_metrics = []
    for sid, grp in obs_t.groupby("sample_id"):
        grade = grade_map.get(sid)
        if grade is None:
            continue
        typed = grp[grp["cell_type"] != "Unassigned"]
        n_typed = len(typed)
        if n_typed < 8000:
            continue
        ct_counts = typed["cell_type"].value_counts()
        total = ct_counts.sum()

        cd8_frac = sum(ct_counts.get(c, 0) for c in ct_counts.index
                       if c.startswith("CD8 T")) / total
        treg_frac = ct_counts.get("Treg", 0) / total
        tfh_frac = ct_counts.get("Tfh", 0) / total
        bcell_frac = sum(ct_counts.get(c, 0) for c in ct_counts.index
                         if "B cell" in c or c == "B cell") / total

        # Exhaustion: TOX+PD-1+ among CD8 T cells (markers in .X)
        cd8_mask = typed["cell_type"].str.startswith("CD8 T")
        n_cd8 = cd8_mask.sum()
        if n_cd8 > 10:
            marker_names = list(ad_t.var_names)
            tox_col = "TOX" if "TOX" in marker_names else None
            pd1_col = next((m for m in marker_names
                            if m in ("PD-1", "PD_1", "PD1", "PDCD1")), None)
            if tox_col and pd1_col:
                cd8_idx = typed[cd8_mask].index
                cd8_ad = ad_t[cd8_idx]
                tox_vals = cd8_ad.X[:, marker_names.index(tox_col)]
                pd1_vals = cd8_ad.X[:, marker_names.index(pd1_col)]
                if hasattr(tox_vals, "toarray"):
                    tox_vals = tox_vals.toarray().ravel()
                if hasattr(pd1_vals, "toarray"):
                    pd1_vals = pd1_vals.toarray().ravel()
                tox_vals = np.array(tox_vals).ravel()
                pd1_vals = np.array(pd1_vals).ravel()
                exh = float(((tox_vals > 0.8) & (pd1_vals > 0.5)).mean())
            else:
                exh = np.nan
        else:
            exh = np.nan

        # Shannon entropy (clean)
        ct_frac = ct_counts / total
        h = -np.sum(ct_frac * np.log2(ct_frac + 1e-10))

        t_metrics.append({
            "sample_id": sid, "grade": grade,
            "cd8_frac": cd8_frac, "treg_frac": treg_frac,
            "tfh_frac": tfh_frac, "bcell_frac": bcell_frac,
            "exhaustion_frac": exh, "entropy": h,
            "n_typed": n_typed,
        })
    df_t = pd.DataFrame(t_metrics)
    print(f"  T-panel: {len(df_t)} ROIs with grade")

    # --- S-panel metrics ---
    print("Loading S-panel...")
    ad_s = sc.read_h5ad(s_panel)
    ad_s = ad_s[~ad_s.obs["sample_id"].str.contains(EXCLUDE_PAT, na=False)]
    obs_s = ad_s.obs.copy()

    # Get marker columns
    marker_cols = list(ad_s.var_names)

    s_metrics = []
    for sid, grp in obs_s.groupby("sample_id"):
        grade = grade_map.get(sid)
        if grade is None:
            continue
        typed = grp[grp["cell_type"] != "Unassigned"]
        n_typed = len(typed)
        if n_typed < 5000:
            continue
        ct_counts = typed["cell_type"].value_counts()
        total = ct_counts.sum()

        fdc_frac = ct_counts.get("FDC", 0) / total
        mac_frac = sum(ct_counts.get(c, 0) for c in ct_counts.index
                       if "macrophage" in c.lower() or "M2" in c) / total
        myeloid_frac = sum(ct_counts.get(c, 0) for c in ct_counts.index
                          if any(k in c.lower() for k in ["myeloid", "macrophage", "monocyte", "dc", "mdsc"])) / total

        # CD14 mean on FDCs
        fdc_mask = typed["cell_type"] == "FDC"
        n_fdc = fdc_mask.sum()
        if n_fdc > 5 and "CD14" in marker_cols:
            idx = list(ad_s.var_names).index("CD14")
            fdc_cells = ad_s[typed[fdc_mask].index]
            x = fdc_cells.X[:, idx]
            if hasattr(x, "toarray"):
                x = x.toarray().ravel()
            cd14_fdc_mean = float(np.mean(x))
        else:
            cd14_fdc_mean = np.nan

        # Overall CD14 mean
        if "CD14" in marker_cols:
            idx = list(ad_s.var_names).index("CD14")
            x_all = ad_s[grp.index].X[:, idx]
            if hasattr(x_all, "toarray"):
                x_all = x_all.toarray().ravel()
            cd14_overall = float(np.mean(x_all))
        else:
            cd14_overall = np.nan

        # VISTA mean on myeloid
        mye_types = [c for c in ct_counts.index
                     if any(k in c.lower() for k in ["macrophage", "monocyte", "myeloid"])]
        mye_mask = typed["cell_type"].isin(mye_types)
        if mye_mask.sum() > 10 and "VISTA" in marker_cols:
            idx_v = list(ad_s.var_names).index("VISTA")
            mye_cells = ad_s[typed[mye_mask].index]
            x_v = mye_cells.X[:, idx_v]
            if hasattr(x_v, "toarray"):
                x_v = x_v.toarray().ravel()
            vista_mye = float(np.mean(x_v))
        else:
            vista_mye = np.nan

        s_metrics.append({
            "sample_id": sid, "grade": grade,
            "fdc_frac": fdc_frac, "mac_frac": mac_frac,
            "myeloid_frac": myeloid_frac,
            "cd14_fdc_mean": cd14_fdc_mean,
            "cd14_overall": cd14_overall,
            "vista_mye_mean": vista_mye,
            "n_fdc": n_fdc,
        })
    df_s = pd.DataFrame(s_metrics)
    print(f"  S-panel: {len(df_s)} ROIs with grade")

    return df_t, df_s


def make_figure(df_t, df_s, output_dir="output/hypotheses_v8"):
    """6-panel figure: grade vs key metrics."""
    fig = plt.figure(figsize=(12, 18))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.10, right=0.96, top=0.95, bottom=0.05)

    grade_order = ["G1", "G2", "G3a"]
    colors = {"G1": "#4DAF4A", "G2": "#FF7F00", "G3a": "#E41A1C"}

    def boxplot_panel(ax, df, metric, title, ylabel, letter):
        data = [df[df["grade"] == g][metric].dropna() for g in grade_order]
        bp = ax.boxplot(data, labels=grade_order, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, g in zip(bp["boxes"], grade_order):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Histological grade")
        ax.text(-0.08, 1.06, f"$\\bf{{{letter}}}$",
                transform=ax.transAxes, fontsize=14, va="top")

        # Kruskal-Wallis test
        valid = [d.values for d in data if len(d) > 2]
        if len(valid) >= 2:
            try:
                _, p_kw = stats.kruskal(*valid)
                ax.text(0.95, 0.95, f"KW p={p_kw:.3g}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEE", edgecolor="#999"))
            except ValueError:
                pass

        # Pairwise Mann-Whitney: G1 vs G3a
        g1 = df[df["grade"] == "G1"][metric].dropna()
        g3 = df[df["grade"] == "G3a"][metric].dropna()
        if len(g1) > 2 and len(g3) > 2:
            _, p_mw = stats.mannwhitneyu(g1, g3, alternative="two-sided")
            ax.text(0.95, 0.85, f"G1 vs G3a p={p_mw:.3g}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color="#555")

        # Add n per group
        for i, g in enumerate(grade_order):
            n = len(df[df["grade"] == g][metric].dropna())
            ax.text(i + 1, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom",
                    fontsize=7, color="#666")

    # (a) CD14 mean per ROI by grade
    ax_a = fig.add_subplot(gs[0, 0])
    boxplot_panel(ax_a, df_s, "cd14_overall", "Per-ROI CD14 mean intensity",
                  "CD14 mean (scaled)", "a")

    # (b) CD14 on FDCs by grade
    ax_b = fig.add_subplot(gs[0, 1])
    boxplot_panel(ax_b, df_s, "cd14_fdc_mean", "FDC CD14 expression by grade",
                  "CD14 mean on FDCs (scaled)", "b")

    # (c) FDC fraction by grade
    ax_c = fig.add_subplot(gs[1, 0])
    boxplot_panel(ax_c, df_s, "fdc_frac", "FDC fraction by grade",
                  "FDC fraction", "c")

    # (d) CD8 T fraction by grade
    ax_d = fig.add_subplot(gs[1, 1])
    boxplot_panel(ax_d, df_t, "cd8_frac", "CD8 T cell fraction by grade",
                  "CD8 T fraction", "d")

    # (e) Exhaustion fraction by grade
    ax_e = fig.add_subplot(gs[2, 0])
    boxplot_panel(ax_e, df_t, "exhaustion_frac",
                  "CD8 T exhaustion (TOX+PD-1+) by grade",
                  "Exhaustion fraction", "e")

    # (f) VISTA on myeloid by grade
    ax_f = fig.add_subplot(gs[2, 1])
    boxplot_panel(ax_f, df_s, "vista_mye_mean",
                  "VISTA expression on myeloid cells by grade",
                  "VISTA mean (scaled)", "f")

    fig.suptitle("Histological Grade vs Key Spatial Findings", fontsize=14, fontweight="bold")

    out = Path(output_dir) / "explore_grade_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-panel", default="output/all_TMA_T_global_v8.h5ad")
    ap.add_argument("--s-panel", default="output/all_TMA_S_global_v8.h5ad")
    ap.add_argument("--clinical", default="output/cd14_validation/master_clinical_ezh2.csv")
    args = ap.parse_args()

    df_t, df_s = load_data(args.t_panel, args.s_panel, args.clinical)
    out = make_figure(df_t, df_s)
