#!/usr/bin/env python3
"""Immune evasion archetype survival + architecture-transformation analysis.

Two questions:
  1. Do immune evasion archetypes (H9h) predict PFS, OS, and transformation?
  2. Does loss of follicular architecture predict transformation? (H8b)

Reuses compute_roi_evasion_metrics() from immune_evasion.py to get the same
8-metric archetype clustering used in fig_ie_archetypes.png.

Figure panels:
  (a) KM: PFS by evasion archetype
  (b) KM: OS by evasion archetype
  (c) Archetype vs transformation (bar chart + Fisher test)
  (d) Architecture metrics in transformers vs non-transformers (H8b)
  (e) KM: PFS by follicularity (median split)
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.clinical_linkage import EXCLUDE_ROIS, load_clinical, normalize_sample_id
from immune_evasion import (
    load_t_panel,
    compute_roi_evasion_metrics,
)


def panel_label(ax, letter, x=-0.08, y=1.05):
    ax.text(x, y, f"$\\bf{{{letter}}}$",
            transform=ax.transAxes, fontsize=14, va="top", ha="left")


# ---------------------------------------------------------------------------
# Step 1: Compute archetype labels
# ---------------------------------------------------------------------------

def compute_archetypes(t_data):
    """Replicate immune evasion archetype clustering from immune_evasion.py."""
    from sklearn.metrics import silhouette_score
    import warnings

    metrics = compute_roi_evasion_metrics(t_data)
    mat = metrics['matrix'].copy()
    roi_ids = metrics['roi_ids']
    tma_labels = metrics['tma_labels']
    n_rois, n_metrics = mat.shape
    print(f"  {n_rois} ROIs × {n_metrics} metrics")

    # Impute NaN with column median
    for j in range(n_metrics):
        col = mat[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            mat[nan_mask, j] = np.nanmedian(col)

    # Log-transform skewed metrics (same as immune_evasion.py)
    mat[:, 3] = np.log1p(mat[:, 3])  # Treg:CD8
    mat[:, 4] = np.log1p(mat[:, 4])  # E:S ratio

    # Z-score
    mat_z = np.zeros_like(mat)
    for j in range(n_metrics):
        mu, sd = mat[:, j].mean(), mat[:, j].std()
        mat_z[:, j] = (mat[:, j] - mu) / max(sd, 1e-10)
    mat_z = np.clip(mat_z, -3, 3)

    # Ward linkage + optimal k
    dist = pdist(mat_z, 'euclidean')
    Z = linkage(dist, method='ward')

    best_k, best_sil = 2, -1
    for k in range(2, 8):
        labels = fcluster(Z, t=k, criterion='maxclust')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sil = silhouette_score(mat_z, labels)
        if sil > best_sil:
            best_k, best_sil = k, sil

    labels = fcluster(Z, t=best_k, criterion='maxclust')
    print(f"  Optimal k={best_k}, silhouette={best_sil:.3f}")

    # Identify which cluster is "evasion" (higher exhaustion fraction)
    cluster_exh = {}
    for ci in range(1, best_k + 1):
        cl_mask = labels == ci
        # Column 1 = CD8 exhaustion fraction
        cluster_exh[ci] = mat[cl_mask, 1].mean()
    evasion_cluster = max(cluster_exh, key=cluster_exh.get)
    print(f"  Evasion cluster: {evasion_cluster} "
          f"(mean exh = {cluster_exh[evasion_cluster]:.3f})")

    # Build per-ROI DataFrame
    rows = []
    for i, roi in enumerate(roi_ids):
        rows.append({
            "sample_id": roi,
            "slide_ID": normalize_sample_id(roi),
            "archetype": int(labels[i]),
            "is_evasion": int(labels[i] == evasion_cluster),
            "tma": tma_labels[i],
            # Raw metrics for reference
            "cd8_infilt": mat[i, 0],
            "cd8_exh_frac": mat[i, 1],
            "treg_frac": mat[i, 2],
            "log_treg_cd8": mat[i, 3],
            "log_es": mat[i, 4],
            "cd39_frac": mat[i, 5],
            "follicularity": mat[i, 6],
            "mac_frac": mat[i, 7],
        })
    df = pd.DataFrame(rows)

    for ci in range(1, best_k + 1):
        n = (labels == ci).sum()
        tag = " (evasion)" if ci == evasion_cluster else " (immune-active)"
        print(f"    Cluster {ci}{tag}: n={n}")

    return df, best_k


# ---------------------------------------------------------------------------
# Step 2: Clinical merge
# ---------------------------------------------------------------------------

def merge_clinical(df):
    """Merge with clinical data, one row per patient."""
    clin = load_clinical()
    clin_t1 = clin.sort_values("T").drop_duplicates(subset="slide_ID", keep="first")

    # Exclude Biomax
    df = df[df["tma"] != "Biomax"].copy()

    merged = df.merge(clin_t1, on="slide_ID", how="inner")
    print(f"\n  Clinical merge: {len(merged)} ROIs")

    merged = merged.sort_values("T").drop_duplicates(
        subset="Patient_ID", keep="first"
    )
    print(f"  After T1-only dedup: {len(merged)} unique patients")

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

    n_ev = int(merged["is_evasion"].sum())
    n_act = len(merged) - n_ev
    n_trans = int(merged["transformed"].sum())
    print(f"  Evasion: {n_ev}, Immune-active: {n_act}")
    print(f"  Transformed: {n_trans}, POD24: {int(merged['pod24'].dropna().sum())}")

    return merged


# ---------------------------------------------------------------------------
# Step 3: Statistical tests
# ---------------------------------------------------------------------------

def univariate_cox(df, metric, time_col, event_col):
    sub = df[[metric, time_col, event_col]].dropna()
    if len(sub) < 20 or sub[event_col].sum() < 5:
        return None
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
            "n": len(sub),
            "events": int(sub[event_col].sum()),
        }
    except Exception as e:
        print(f"  Cox failed for {metric}: {e}")
        return None


def run_archetype_tests(df):
    """Test archetype vs all clinical endpoints."""
    print("\n--- Archetype (binary: evasion vs immune-active) ---")
    results = {}

    # Cox PH for is_evasion
    for ep, (tc, ec) in [("PFS", ("pfs_time", "pfs_event")),
                          ("OS", ("os_time", "os_event"))]:
        res = univariate_cox(df, "is_evasion", tc, ec)
        if res:
            sig = "***" if res["p"] < 0.001 else (
                "**" if res["p"] < 0.01 else ("*" if res["p"] < 0.05 else ""))
            print(f"  {ep}: HR={res['HR']:.2f} [{res['HR_lo']:.2f}-{res['HR_hi']:.2f}] "
                  f"p={res['p']:.4f}{sig} n={res['n']}")
            results[ep] = res

    # Log-rank
    for ep, (tc, ec) in [("PFS", ("pfs_time", "pfs_event")),
                          ("OS", ("os_time", "os_event"))]:
        ev = df[df["is_evasion"] == 1]
        act = df[df["is_evasion"] == 0]
        sub_ev = ev[[tc, ec]].dropna()
        sub_act = act[[tc, ec]].dropna()
        if len(sub_ev) > 5 and len(sub_act) > 5:
            lr = logrank_test(sub_ev[tc], sub_act[tc],
                              event_observed_A=sub_ev[ec],
                              event_observed_B=sub_act[ec])
            print(f"  {ep} log-rank: p={lr.p_value:.4f}")
            results[f"{ep}_logrank"] = lr.p_value

    # Transformation (Fisher exact)
    ev = df[df["is_evasion"] == 1]
    act = df[df["is_evasion"] == 0]
    a = int(ev["transformed"].sum())
    b = int(len(ev) - a)
    c = int(act["transformed"].sum())
    d = int(len(act) - c)
    _, fisher_p = stats.fisher_exact([[a, b], [c, d]])
    pct_ev = 100 * a / (a + b) if (a + b) > 0 else 0
    pct_act = 100 * c / (c + d) if (c + d) > 0 else 0
    print(f"  Transformation: evasion {a}/{a+b} ({pct_ev:.1f}%) vs "
          f"active {c}/{c+d} ({pct_act:.1f}%) Fisher p={fisher_p:.4f}")
    results["transform_fisher_p"] = fisher_p
    results["transform_ev_pct"] = pct_ev
    results["transform_act_pct"] = pct_act

    # POD24 (Fisher exact)
    pod = df[df["pod24"].notna()].copy()
    ev_pod = pod[pod["is_evasion"] == 1]
    act_pod = pod[pod["is_evasion"] == 0]
    a2 = int(ev_pod["pod24"].sum())
    b2 = int(len(ev_pod) - a2)
    c2 = int(act_pod["pod24"].sum())
    d2 = int(len(act_pod) - c2)
    _, fisher_p2 = stats.fisher_exact([[a2, b2], [c2, d2]])
    print(f"  POD24: evasion {a2}/{a2+b2} vs active {c2}/{c2+d2} "
          f"Fisher p={fisher_p2:.4f}")
    results["pod24_fisher_p"] = fisher_p2

    return results


def run_architecture_tests(df):
    """Test architecture loss vs transformation (H8b)."""
    print("\n--- Architecture vs transformation (H8b) ---")
    trans = df[df["transformed"] == 1]
    notrans = df[df["transformed"] == 0]
    results = {}

    arch_metrics = ["follicularity", "cd8_exh_frac", "log_es", "mac_frac"]
    arch_labels = {
        "follicularity": "Follicularity score",
        "cd8_exh_frac": "CD8 exhaustion fraction",
        "log_es": "log(E:S ratio)",
        "mac_frac": "Macrophage fraction",
    }

    for m in arch_metrics:
        t_vals = trans[m].dropna()
        nt_vals = notrans[m].dropna()
        if len(t_vals) >= 3 and len(nt_vals) >= 3:
            u, p = stats.mannwhitneyu(t_vals, nt_vals, alternative="two-sided")
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else (
                "*" if p < 0.05 else ""))
            print(f"  {arch_labels.get(m, m):30s}  "
                  f"trans={t_vals.median():.3f} vs non={nt_vals.median():.3f}  "
                  f"p={p:.4f}{sig}")
            results[m] = {"p": p, "trans_med": t_vals.median(),
                          "notrans_med": nt_vals.median()}

    # Cox for follicularity → PFS and OS
    for ep, (tc, ec) in [("PFS", ("pfs_time", "pfs_event")),
                          ("OS", ("os_time", "os_event"))]:
        res = univariate_cox(df, "follicularity", tc, ec)
        if res:
            sig = "***" if res["p"] < 0.001 else (
                "**" if res["p"] < 0.01 else ("*" if res["p"] < 0.05 else ""))
            print(f"  Follicularity → {ep}: HR={res['HR']:.2f} "
                  f"[{res['HR_lo']:.2f}-{res['HR_hi']:.2f}] "
                  f"p={res['p']:.4f}{sig}")
            results[f"foll_{ep}"] = res

    return results


# ---------------------------------------------------------------------------
# Step 4: Figure
# ---------------------------------------------------------------------------

def make_figure(df, arch_results, h8b_results, output_dir):
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                          left=0.06, right=0.96, top=0.93, bottom=0.08)

    # --- (a) KM: PFS by archetype ---
    ax_a = fig.add_subplot(gs[0, 0])
    ev = df[df["is_evasion"] == 1]
    act = df[df["is_evasion"] == 0]

    kmf = KaplanMeierFitter()
    sub_ev = ev[["pfs_time", "pfs_event"]].dropna()
    sub_act = act[["pfs_time", "pfs_event"]].dropna()

    kmf.fit(sub_ev["pfs_time"], sub_ev["pfs_event"],
            label=f"Evasion (n={len(sub_ev)})")
    kmf.plot_survival_function(ax=ax_a, ci_show=True, color="#C0392B")
    kmf.fit(sub_act["pfs_time"], sub_act["pfs_event"],
            label=f"Immune-active (n={len(sub_act)})")
    kmf.plot_survival_function(ax=ax_a, ci_show=True, color="#2980B9")

    lr = logrank_test(sub_ev["pfs_time"], sub_act["pfs_time"],
                      event_observed_A=sub_ev["pfs_event"],
                      event_observed_B=sub_act["pfs_event"])
    hr_str = ""
    if "PFS" in arch_results:
        hr_str = f"  HR={arch_results['PFS']['HR']:.2f}"
    ax_a.text(0.95, 0.95, f"log-rank p={lr.p_value:.4f}{hr_str}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax_a.set_xlabel("Time (years)")
    ax_a.set_ylabel("PFS probability")
    ax_a.set_title("PFS by evasion archetype", fontsize=11, fontweight="bold")
    ax_a.legend(fontsize=8, loc="lower left")
    panel_label(ax_a, "a")

    # --- (b) KM: OS by archetype ---
    ax_b = fig.add_subplot(gs[0, 1])
    sub_ev_os = ev[["os_time", "os_event"]].dropna()
    sub_act_os = act[["os_time", "os_event"]].dropna()

    kmf.fit(sub_ev_os["os_time"], sub_ev_os["os_event"],
            label=f"Evasion (n={len(sub_ev_os)})")
    kmf.plot_survival_function(ax=ax_b, ci_show=True, color="#C0392B")
    kmf.fit(sub_act_os["os_time"], sub_act_os["os_event"],
            label=f"Immune-active (n={len(sub_act_os)})")
    kmf.plot_survival_function(ax=ax_b, ci_show=True, color="#2980B9")

    lr_os = logrank_test(sub_ev_os["os_time"], sub_act_os["os_time"],
                         event_observed_A=sub_ev_os["os_event"],
                         event_observed_B=sub_act_os["os_event"])
    hr_str_os = ""
    if "OS" in arch_results:
        hr_str_os = f"  HR={arch_results['OS']['HR']:.2f}"
    ax_b.text(0.95, 0.95, f"log-rank p={lr_os.p_value:.4f}{hr_str_os}",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax_b.set_xlabel("Time (years)")
    ax_b.set_ylabel("OS probability")
    ax_b.set_title("OS by evasion archetype", fontsize=11, fontweight="bold")
    ax_b.legend(fontsize=8, loc="lower left")
    panel_label(ax_b, "b")

    # --- (c) Archetype vs transformation + POD24 ---
    ax_c = fig.add_subplot(gs[0, 2])
    # Stacked bar: transformation rate by archetype
    ev_trans = int(ev["transformed"].sum())
    ev_notrans = len(ev) - ev_trans
    act_trans = int(act["transformed"].sum())
    act_notrans = len(act) - act_trans

    x = [0, 1]
    ax_c.bar(x, [ev_trans, act_trans], color="#C0392B", alpha=0.7,
             label="Transformed")
    ax_c.bar(x, [ev_notrans, act_notrans], bottom=[ev_trans, act_trans],
             color="#AED6F1", alpha=0.7, label="Non-transformed")

    pct_ev = 100 * ev_trans / len(ev) if len(ev) > 0 else 0
    pct_act = 100 * act_trans / len(act) if len(act) > 0 else 0
    ax_c.text(0, ev_trans + ev_notrans + 1, f"{pct_ev:.0f}%",
              ha="center", fontsize=10, fontweight="bold", color="#C0392B")
    ax_c.text(1, act_trans + act_notrans + 1, f"{pct_act:.0f}%",
              ha="center", fontsize=10, fontweight="bold", color="#2980B9")

    fp = arch_results.get("transform_fisher_p", 1.0)
    sig = "***" if fp < 0.001 else ("**" if fp < 0.01 else (
        "*" if fp < 0.05 else "n.s."))
    ax_c.text(0.5, 0.95, f"Fisher p={fp:.4f} {sig}",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"Evasion\n(n={len(ev)})",
                           f"Immune-active\n(n={len(act)})"])
    ax_c.set_ylabel("Number of patients")
    ax_c.set_title("Transformation by archetype", fontsize=11, fontweight="bold")
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    panel_label(ax_c, "c")

    # --- (d) Architecture metrics: transformers vs non-transformers (H8b) ---
    ax_d = fig.add_subplot(gs[1, :2])
    trans = df[df["transformed"] == 1]
    notrans = df[df["transformed"] == 0]

    arch_metrics = ["follicularity", "cd8_exh_frac", "log_es", "mac_frac"]
    arch_labels_map = {
        "follicularity": "Follicularity\nscore",
        "cd8_exh_frac": "CD8 exhaustion\nfraction",
        "log_es": "log(E:S\nratio)",
        "mac_frac": "Macrophage\nfraction",
    }

    x_pos = np.arange(len(arch_metrics))
    width = 0.35

    vals_notrans = [notrans[m].dropna() for m in arch_metrics]
    vals_trans = [trans[m].dropna() for m in arch_metrics]

    bp1 = ax_d.boxplot(
        vals_notrans, positions=x_pos - width / 2, widths=width * 0.8,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="#AED6F1", edgecolor="#2980B9"),
        medianprops=dict(color="#2980B9", linewidth=2),
    )
    bp2 = ax_d.boxplot(
        vals_trans, positions=x_pos + width / 2, widths=width * 0.8,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="#F5B7B1", edgecolor="#C0392B"),
        medianprops=dict(color="#C0392B", linewidth=2),
    )

    for i, m in enumerate(arch_metrics):
        if m in h8b_results and "p" in h8b_results[m]:
            p = h8b_results[m]["p"]
            sig = "n.s."
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
            y_max = max(
                vals_notrans[i].max() if len(vals_notrans[i]) > 0 else 0,
                vals_trans[i].max() if len(vals_trans[i]) > 0 else 0,
            )
            ax_d.text(i, y_max * 1.05 + 0.02, sig, ha="center", fontsize=10)

    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([arch_labels_map[m] for m in arch_metrics], fontsize=9)
    ax_d.set_ylabel("Value")
    ax_d.set_title("Architecture metrics: transformers vs non-transformers (H8b)",
                    fontsize=11, fontweight="bold")
    ax_d.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        [f"Non-transformed (n={len(notrans)})",
         f"Transformed (n={len(trans)})"],
        fontsize=8, loc="upper right",
    )
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    panel_label(ax_d, "d", x=-0.05)

    # --- (e) KM: PFS by follicularity ---
    ax_e = fig.add_subplot(gs[1, 2])
    sub = df[["follicularity", "pfs_time", "pfs_event"]].dropna()
    med_foll = sub["follicularity"].median()
    hi_foll = sub[sub["follicularity"] >= med_foll]
    lo_foll = sub[sub["follicularity"] < med_foll]

    kmf.fit(hi_foll["pfs_time"], hi_foll["pfs_event"],
            label=f"High foll (n={len(hi_foll)})")
    kmf.plot_survival_function(ax=ax_e, ci_show=True, color="#E67E22")
    kmf.fit(lo_foll["pfs_time"], lo_foll["pfs_event"],
            label=f"Low foll (n={len(lo_foll)})")
    kmf.plot_survival_function(ax=ax_e, ci_show=True, color="#8E44AD")

    lr_foll = logrank_test(
        hi_foll["pfs_time"], lo_foll["pfs_time"],
        event_observed_A=hi_foll["pfs_event"],
        event_observed_B=lo_foll["pfs_event"],
    )
    foll_cox = h8b_results.get("foll_PFS", {})
    hr_foll = f"  HR={foll_cox['HR']:.2f}" if "HR" in foll_cox else ""
    ax_e.text(0.95, 0.95, f"log-rank p={lr_foll.p_value:.4f}{hr_foll}",
              transform=ax_e.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax_e.set_xlabel("Time (years)")
    ax_e.set_ylabel("PFS probability")
    ax_e.set_title("PFS by follicularity score", fontsize=11, fontweight="bold")
    ax_e.legend(fontsize=8, loc="lower left")
    panel_label(ax_e, "e")

    out = Path(output_dir) / "fig_archetype_survival.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-utag", required=True,
                        help="T-panel UTAG merged h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Load T-panel data and compute archetypes
    print("=" * 60)
    print("STEP 1: Compute evasion archetypes")
    print("=" * 60)
    t_data = load_t_panel(args.t_utag)
    arch_df, best_k = compute_archetypes(t_data)

    # Step 2: Clinical merge
    print("\n" + "=" * 60)
    print("STEP 2: Clinical merge")
    print("=" * 60)
    df = merge_clinical(arch_df)

    # Step 3: Archetype survival tests
    print("\n" + "=" * 60)
    print("STEP 3: Archetype survival analysis")
    print("=" * 60)
    arch_results = run_archetype_tests(df)

    # Step 4: Architecture vs transformation (H8b)
    print("\n" + "=" * 60)
    print("STEP 4: Architecture vs transformation (H8b)")
    print("=" * 60)
    h8b_results = run_architecture_tests(df)

    # Step 5: Figure
    print("\n" + "=" * 60)
    print("STEP 5: Generate figure")
    print("=" * 60)
    make_figure(df, arch_results, h8b_results, args.output_dir)


if __name__ == "__main__":
    main()
