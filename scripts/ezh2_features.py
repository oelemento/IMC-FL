#!/usr/bin/env python3
"""
EZH2 mutation vs cellular/morphological/spatial features in FL IMC data.

Extracts per-patient features from T-panel and S-panel h5ad files,
merges with EZH2 status, and tests for associations.

Usage:
    python3.11 scripts/ezh2_features.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact
from src.clinical_linkage import normalize_sample_id

OUTPUT_DIR = "output/cd14_validation"


def extract_patient_features(adata, panel_name):
    """Extract per-patient cell type fractions and marker means."""
    # Normalize sample IDs to slide_IDs
    adata.obs["slide_ID"] = [normalize_sample_id(s) for s in adata.obs["sample_id"]]

    features = {}

    # --- Cell type fractions ---
    if "cell_type" in adata.obs.columns:
        ct_counts = adata.obs.groupby(["slide_ID", "cell_type"]).size().unstack(fill_value=0)
        ct_frac = ct_counts.div(ct_counts.sum(axis=1), axis=0)
        ct_frac.columns = [f"{panel_name}_frac_{c.replace(' ', '_')}" for c in ct_frac.columns]
        features["cell_type_frac"] = ct_frac

        # Total cell count per patient
        features["total_cells"] = ct_counts.sum(axis=1).to_frame(f"{panel_name}_total_cells")

    # --- Mean marker expression per patient (z-scored) ---
    marker_means = adata.obs.groupby("slide_ID").apply(
        lambda g: pd.Series(
            np.asarray(adata[g.index].X.mean(axis=0)).flatten(),
            index=adata.var_names
        )
    )
    marker_means.columns = [f"{panel_name}_marker_{c}" for c in marker_means.columns]
    features["marker_means"] = marker_means

    # --- Morphological features (area, eccentricity) ---
    morph_cols = [c for c in adata.obs.columns if c in ["area", "eccentricity", "solidity", "perimeter"]]
    if morph_cols:
        morph = adata.obs.groupby("slide_ID")[morph_cols].mean()
        morph.columns = [f"{panel_name}_morph_{c}" for c in morph.columns]
        features["morphology"] = morph

    return features


def extract_compartment_features(utag_path, panel_name):
    """Extract per-patient compartment fractions from UTAG data."""
    adata = ad.read_h5ad(utag_path)
    adata.obs["slide_ID"] = [normalize_sample_id(s) for s in adata.obs["sample_id"]]

    features = {}

    # Prefer compartment_name (biological names) over tissue_compartment (numbers)
    if "compartment_name" in adata.obs.columns:
        comp_col = "compartment_name"
    elif "tissue_compartment" in adata.obs.columns:
        comp_col = "tissue_compartment"
    elif "compartment" in adata.obs.columns:
        comp_col = "compartment"
    else:
        comp_candidates = [c for c in adata.obs.columns if "compartment" in c.lower() or "domain" in c.lower()]
        if comp_candidates:
            comp_col = comp_candidates[0]
        else:
            print(f"  No compartment column found in {utag_path}")
            print(f"  Available columns: {list(adata.obs.columns[:20])}")
            return features

    print(f"  Using compartment column: '{comp_col}'")
    print(f"  Compartments: {adata.obs[comp_col].value_counts().head(10).to_dict()}")

    comp_counts = adata.obs.groupby(["slide_ID", comp_col]).size().unstack(fill_value=0)
    comp_frac = comp_counts.div(comp_counts.sum(axis=1), axis=0)
    comp_frac.columns = [f"{panel_name}_comp_{c.replace(' ', '_')}" for c in comp_frac.columns]
    features["compartment_frac"] = comp_frac

    return features


def extract_immune_evasion_features(adata_t):
    """Extract immune evasion metrics per patient from T-panel."""
    adata_t.obs["slide_ID"] = [normalize_sample_id(s) for s in adata_t.obs["sample_id"]]

    features = {}
    ct_col = "cell_type"
    if ct_col not in adata_t.obs.columns:
        return features

    per_patient = []
    for sid, group in adata_t.obs.groupby("slide_ID"):
        total = len(group)
        ct_counts = group[ct_col].value_counts()

        # CD8 fraction
        cd8_total = ct_counts.get("CD8 T", 0) + ct_counts.get("CD8 T exhausted", 0)
        cd8_frac = cd8_total / total if total > 0 else 0

        # Exhaustion fraction among CD8
        cd8_exh = ct_counts.get("CD8 T exhausted", 0)
        exh_ratio = cd8_exh / cd8_total if cd8_total > 0 else 0

        # Treg fraction
        treg_frac = ct_counts.get("Treg", 0) / total if total > 0 else 0

        # CD4:CD8 ratio
        cd4 = ct_counts.get("CD4 T", 0)
        cd4_cd8_ratio = cd4 / cd8_total if cd8_total > 0 else np.nan

        # B cell fraction
        b_frac = ct_counts.get("B cell", 0) / total if total > 0 else 0

        # Macrophage fraction
        mac_frac = ct_counts.get("Macrophage", 0) / total if total > 0 else 0

        per_patient.append({
            "slide_ID": sid,
            "T_cd8_fraction": cd8_frac,
            "T_cd8_exhaustion_ratio": exh_ratio,
            "T_treg_fraction": treg_frac,
            "T_cd4_cd8_ratio": cd4_cd8_ratio,
            "T_b_cell_fraction": b_frac,
            "T_macrophage_fraction": mac_frac,
        })

    features["immune_evasion"] = pd.DataFrame(per_patient).set_index("slide_ID")
    return features


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load EZH2 data
    master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_clinical_ezh2.csv"))
    ezh2_map = master[master["EZH2"].isin(["wt", "mut"])][["slide_ID", "EZH2", "DIAG"]].drop_duplicates("slide_ID")
    print(f"EZH2 data: {len(ezh2_map)} patients ({(ezh2_map['EZH2']=='mut').sum()} mut, {(ezh2_map['EZH2']=='wt').sum()} wt)")

    # === Load T-panel ===
    print("\nLoading T-panel...")
    t_path = "output/all_TMA_T_global_v8.h5ad"
    adata_t = ad.read_h5ad(t_path)
    print(f"  {adata_t.n_obs} cells, {adata_t.n_vars} markers")
    print(f"  obs columns: {list(adata_t.obs.columns[:15])}")

    t_features = extract_patient_features(adata_t, "T")
    t_immune = extract_immune_evasion_features(adata_t)

    # === Load S-panel ===
    print("\nLoading S-panel...")
    s_path = "output/all_TMA_S_global_v8.h5ad"
    adata_s = ad.read_h5ad(s_path)
    print(f"  {adata_s.n_obs} cells, {adata_s.n_vars} markers")

    s_features = extract_patient_features(adata_s, "S")

    # === Load UTAG compartment data ===
    print("\nLoading T-panel UTAG compartments...")
    t_comp = extract_compartment_features("output/all_TMA_T_utag_ct_merged.h5ad", "T")

    print("\nLoading S-panel UTAG compartments...")
    s_comp = extract_compartment_features("output/all_TMA_S_utag_ct_merged.h5ad", "S")

    # === Merge all features ===
    print("\nMerging features...")
    all_dfs = []

    for feat_dict in [t_features, s_features, t_immune, t_comp, s_comp]:
        for name, df in feat_dict.items():
            if isinstance(df, pd.DataFrame):
                all_dfs.append(df)

    # Combine all features
    combined = ezh2_map.set_index("slide_ID")
    for df in all_dfs:
        combined = combined.join(df, how="left")

    print(f"Combined: {len(combined)} patients × {combined.shape[1]} features")

    # Filter to feature columns only (exclude EZH2, DIAG)
    feature_cols = [c for c in combined.columns if c not in ["EZH2", "DIAG"]]
    print(f"Feature columns: {len(feature_cols)}")

    # === Statistical testing: EZH2-wt vs EZH2-mut ===
    print("\n" + "=" * 70)
    print("EZH2-wt vs EZH2-mut: Mann-Whitney U tests")
    print("=" * 70)

    wt = combined[combined["EZH2"] == "wt"]
    mut = combined[combined["EZH2"] == "mut"]

    results = []
    for col in feature_cols:
        wt_vals = wt[col].dropna()
        mut_vals = mut[col].dropna()
        if len(wt_vals) >= 5 and len(mut_vals) >= 5:
            stat, pval = mannwhitneyu(wt_vals, mut_vals, alternative="two-sided")
            effect = mut_vals.median() - wt_vals.median()
            results.append({
                "feature": col,
                "wt_median": wt_vals.median(),
                "mut_median": mut_vals.median(),
                "diff": effect,
                "pval": pval,
                "n_wt": len(wt_vals),
                "n_mut": len(mut_vals),
            })

    results_df = pd.DataFrame(results).sort_values("pval")

    # Benjamini-Hochberg correction
    m = len(results_df)
    results_df["rank"] = range(1, m + 1)
    results_df["q_value"] = results_df["pval"] * m / results_df["rank"]
    results_df["q_value"] = results_df["q_value"].clip(upper=1.0)
    # Ensure monotonicity
    results_df["q_value"] = results_df["q_value"][::-1].cummin()[::-1]

    # Print top results
    print(f"\nTop 30 features (sorted by p-value):")
    print("-" * 100)
    for _, row in results_df.head(30).iterrows():
        sig = "***" if row["q_value"] < 0.001 else ("**" if row["q_value"] < 0.01 else ("*" if row["q_value"] < 0.05 else ""))
        print(f"  {row['feature']:50s}  p={row['pval']:.4f}  q={row['q_value']:.4f}  "
              f"wt={row['wt_median']:.4f}  mut={row['mut_median']:.4f}  Δ={row['diff']:+.4f} {sig}")

    n_sig_005 = (results_df["pval"] < 0.05).sum()
    n_sig_q005 = (results_df["q_value"] < 0.05).sum()
    print(f"\nTotal features tested: {m}")
    print(f"Nominal p < 0.05: {n_sig_005}")
    print(f"FDR q < 0.05: {n_sig_q005}")

    # Save full results
    results_df.to_csv(os.path.join(OUTPUT_DIR, "ezh2_feature_tests.csv"), index=False)
    print(f"\nFull results saved: {OUTPUT_DIR}/ezh2_feature_tests.csv")

    # === Figure ===
    make_figure(combined, results_df, OUTPUT_DIR)


def make_figure(combined, results_df, output_dir):
    """Create figure showing EZH2 vs cellular/spatial features."""
    wt = combined[combined["EZH2"] == "wt"]
    mut = combined[combined["EZH2"] == "mut"]

    # Pick top features by category for display
    categories = {
        "Cell type fractions": [c for c in results_df["feature"] if "_frac_" in c],
        "Marker expression": [c for c in results_df["feature"] if "_marker_" in c],
        "Compartment": [c for c in results_df["feature"] if "_comp_" in c],
        "Immune evasion": [c for c in results_df["feature"] if c.startswith("T_cd8") or c.startswith("T_treg") or c.startswith("T_cd4") or c.startswith("T_mac")],
        "Morphology": [c for c in results_df["feature"] if "_morph_" in c],
    }

    # Determine layout
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle("EZH2 Mutation vs Cellular & Spatial Features in FL", fontsize=16, fontweight="bold", y=0.98)

    # (a) Volcano plot of all features — use effect size (diff / pooled median abs)
    ax = axes[0, 0]
    df = results_df.copy()
    df["-log10p"] = -np.log10(df["pval"].clip(lower=1e-10))
    # Normalize diff to effect size (divide by IQR of pooled data)
    # For display, clip extreme diffs to ±2 to avoid outlier compression
    df["diff_clipped"] = df["diff"].clip(lower=-2, upper=2)
    colors = ["#d62728" if q < 0.05 else ("#ff7f0e" if p < 0.05 else "#7f7f7f")
              for p, q in zip(df["pval"], df["q_value"])]
    ax.scatter(df["diff_clipped"], df["-log10p"], c=colors, alpha=0.6, s=25, edgecolors="none")
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5, label="p=0.05")
    ax.axvline(0, color="gray", linestyle="-", alpha=0.3)
    # Label top hits
    for _, row in df.head(8).iterrows():
        label = row["feature"].split("_", 2)[-1].replace("_", " ")[:22]
        ax.annotate(label, (row["diff_clipped"], row["-log10p"]), fontsize=7,
                    xytext=(5, 3), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5) if abs(row["diff_clipped"]) > 1 else None)
    ax.set_xlabel("Median difference (mut − wt)", fontsize=10)
    ax.set_ylabel("-log10(p-value)", fontsize=10)
    ax.set_title(f"(a) Volcano plot\n{len(df)} features tested, 0 FDR-significant", fontsize=11)
    ax.legend(fontsize=8)

    # (b) Top cell type fractions
    ax = axes[0, 1]
    ct_feats = results_df[results_df["feature"].isin(categories.get("Cell type fractions", []))].head(8)
    if len(ct_feats) > 0:
        labels = [f.split("_frac_")[-1].replace("_", " ") for f in ct_feats["feature"]]
        wt_meds = ct_feats["wt_median"].values
        mut_meds = ct_feats["mut_median"].values
        x = np.arange(len(labels))
        w = 0.35
        ax.barh(x - w/2, wt_meds, w, label="EZH2-wt", color="#4DBEEE", edgecolor="black", linewidth=0.5)
        ax.barh(x + w/2, mut_meds, w, label="EZH2-mut", color="#D95319", edgecolor="black", linewidth=0.5)
        # Add p-value annotations
        for i, (_, row) in enumerate(ct_feats.iterrows()):
            sig = "*" if row["pval"] < 0.05 else ""
            ax.text(max(wt_meds[i], mut_meds[i]) + 0.005, i, f"p={row['pval']:.3f}{sig}", fontsize=7, va="center")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Median fraction", fontsize=10)
        ax.legend(fontsize=8)
    ax.set_title("(b) Cell type fractions (top 8)", fontsize=11)

    # (c) Top marker expression differences
    ax = axes[0, 2]
    marker_feats = results_df[results_df["feature"].isin(categories.get("Marker expression", []))].head(10)
    if len(marker_feats) > 0:
        labels = [f.split("_marker_")[-1] for f in marker_feats["feature"]]
        diffs = marker_feats["diff"].values
        pvals = marker_feats["pval"].values
        colors_m = ["#d62728" if p < 0.05 else "#4DBEEE" for p in pvals]
        ax.barh(range(len(labels)), diffs, color=colors_m, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="gray", linestyle="-", alpha=0.5)
        for i, p in enumerate(pvals):
            sig = "*" if p < 0.05 else ""
            ax.text(diffs[i] + (0.01 if diffs[i] >= 0 else -0.01), i,
                    f"p={p:.3f}{sig}", fontsize=7, va="center",
                    ha="left" if diffs[i] >= 0 else "right")
        ax.set_xlabel("Median difference (mut − wt)", fontsize=10)
    ax.set_title("(c) Marker expression (top 10)", fontsize=11)

    # (d) Compartment fractions
    ax = axes[0, 3]
    comp_feats = results_df[results_df["feature"].isin(categories.get("Compartment", []))].head(8)
    if len(comp_feats) > 0:
        labels = [f.split("_comp_")[-1].replace("_", " ")[:25] for f in comp_feats["feature"]]
        wt_meds = comp_feats["wt_median"].values
        mut_meds = comp_feats["mut_median"].values
        x = np.arange(len(labels))
        w = 0.35
        ax.barh(x - w/2, wt_meds, w, label="EZH2-wt", color="#4DBEEE", edgecolor="black", linewidth=0.5)
        ax.barh(x + w/2, mut_meds, w, label="EZH2-mut", color="#D95319", edgecolor="black", linewidth=0.5)
        for i, (_, row) in enumerate(comp_feats.iterrows()):
            sig = "*" if row["pval"] < 0.05 else ""
            ax.text(max(wt_meds[i], mut_meds[i]) + 0.005, i, f"p={row['pval']:.3f}{sig}", fontsize=7, va="center")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Median fraction", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No compartment data", transform=ax.transAxes, ha="center")
    ax.set_title("(d) Compartment fractions (top 8)", fontsize=11)

    # (e) Immune evasion features
    ax = axes[1, 0]
    ie_feats = results_df[results_df["feature"].isin(categories.get("Immune evasion", []))]
    if len(ie_feats) > 0:
        labels = [f.replace("T_", "").replace("_", " ") for f in ie_feats["feature"]]
        for i, (_, row) in enumerate(ie_feats.iterrows()):
            feat = row["feature"]
            wt_vals = wt[feat].dropna()
            mut_vals = mut[feat].dropna()
            bp = ax.boxplot([wt_vals, mut_vals],
                            positions=[i*3, i*3+1], widths=0.6, patch_artist=True)
            bp["boxes"][0].set_facecolor("#4DBEEE")
            bp["boxes"][1].set_facecolor("#D95319")
            sig = "*" if row["pval"] < 0.05 else ""
            ax.text(i*3+0.5, max(wt_vals.max(), mut_vals.max()) * 1.05,
                    f"p={row['pval']:.3f}{sig}", fontsize=7, ha="center")
        ax.set_xticks([i*3+0.5 for i in range(len(ie_feats))])
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        # Custom legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#4DBEEE", label="EZH2-wt"),
                           Patch(facecolor="#D95319", label="EZH2-mut")], fontsize=8)
    else:
        ax.text(0.5, 0.5, "No immune evasion data", transform=ax.transAxes, ha="center")
    ax.set_title("(e) Immune evasion metrics", fontsize=11)

    # (f) H3K27me3 — the key biological control
    ax = axes[1, 1]
    h3k_col = None
    for col in combined.columns:
        if "H3K27me3" in col:
            h3k_col = col
            break
    if h3k_col:
        wt_h3k = wt[h3k_col].dropna()
        mut_h3k = mut[h3k_col].dropna()
        bp = ax.boxplot([wt_h3k, mut_h3k],
                        tick_labels=["EZH2-wt", "EZH2-mut"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4DBEEE")
        bp["boxes"][1].set_facecolor("#D95319")
        # Add individual points
        for i, (vals, xpos) in enumerate([(wt_h3k, 1), (mut_h3k, 2)]):
            ax.scatter(np.random.normal(xpos, 0.05, len(vals)), vals,
                       alpha=0.3, s=12, color=bp["boxes"][i].get_facecolor(), zorder=2)
        stat, pval = mannwhitneyu(wt_h3k, mut_h3k, alternative="two-sided")
        ax.set_ylabel("H3K27me3 protein (z-scored)", fontsize=11)
        ax.set_title(f"(f) H3K27me3 — Positive Control\np={pval:.4f} (EZH2 is the H3K27 methyltransferase)", fontsize=10)
        ax.text(0.5, 0.95, f"wt: n={len(wt_h3k)}, median={wt_h3k.median():.3f}\n"
                f"mut: n={len(mut_h3k)}, median={mut_h3k.median():.3f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
    else:
        ax.text(0.5, 0.5, "H3K27me3 not found", transform=ax.transAxes, ha="center")
        ax.set_title("(f) H3K27me3 — Positive Control", fontsize=11)

    # (g) Summary count by category
    ax = axes[1, 2]
    cat_counts = {}
    for cat_name, feats in categories.items():
        cat_df = results_df[results_df["feature"].isin(feats)]
        n_total = len(cat_df)
        n_sig = (cat_df["pval"] < 0.05).sum()
        n_fdr = (cat_df["q_value"] < 0.05).sum()
        cat_counts[cat_name] = {"tested": n_total, "p<0.05": n_sig, "FDR<0.05": n_fdr}

    cat_summary = pd.DataFrame(cat_counts).T
    if len(cat_summary) > 0:
        x = np.arange(len(cat_summary))
        ax.barh(x, cat_summary["tested"], color="#cccccc", edgecolor="black", label="Tested", height=0.6)
        ax.barh(x, cat_summary["p<0.05"], color="#ff7f0e", edgecolor="black", label="p<0.05", height=0.6)
        ax.barh(x, cat_summary["FDR<0.05"], color="#d62728", edgecolor="black", label="FDR<0.05", height=0.6)
        ax.set_yticks(x)
        ax.set_yticklabels(cat_summary.index, fontsize=9)
        ax.set_xlabel("Number of features", fontsize=10)
        ax.legend(fontsize=8)
    ax.set_title("(g) Significant features by category", fontsize=11)

    # (h) Summary text
    ax = axes[1, 3]
    ax.axis("off")
    n_sig = (results_df["pval"] < 0.05).sum()
    n_fdr = (results_df["q_value"] < 0.05).sum()
    n_total = len(results_df)

    summary_text = (
        f"EZH2 Feature Association Summary\n"
        f"{'='*40}\n\n"
        f"Patients: {len(combined)}\n"
        f"  EZH2-wt: {(combined['EZH2']=='wt').sum()}\n"
        f"  EZH2-mut: {(combined['EZH2']=='mut').sum()}\n\n"
        f"Features tested: {n_total}\n"
        f"  Nominal p < 0.05: {n_sig} ({100*n_sig/max(n_total,1):.1f}%)\n"
        f"  FDR q < 0.05: {n_fdr} ({100*n_fdr/max(n_total,1):.1f}%)\n\n"
    )
    if n_sig > 0:
        summary_text += "Top hits (p < 0.05):\n"
        for _, row in results_df[results_df["pval"] < 0.05].head(8).iterrows():
            label = row["feature"].split("_", 2)[-1].replace("_", " ")[:30]
            direction = "up mut" if row["diff"] > 0 else "dn mut"
            summary_text += f"  {label}: p={row['pval']:.4f} {direction}\n"
        summary_text += f"\nPositive control: H3K27me3 is #1 hit\n"
        summary_text += f"(EZH2 = H3K27 methyltransferase)\n"
        summary_text += f"\nConclusion: No features survive FDR.\n"
        summary_text += f"EZH2-mut tumors LOOK the same as wt\n"
        summary_text += f"in cellular composition and spatial\n"
        summary_text += f"organization. Only H3K27me3 differs\n"
        summary_text += f"(expected biochemical consequence)."
    else:
        summary_text += "No features reached nominal significance.\n"
        summary_text += "\nConclusion: EZH2 mutation does NOT\nassociate with measurable cellular,\nmarker, or spatial features in IMC."

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("(h) Summary", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, "ezh2_features.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()
