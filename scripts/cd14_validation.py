#!/usr/bin/env python3
"""
CD14 survival validation in independent FL cohorts.

Tests whether CD14 gene expression predicts survival in:
  1. GSE119214 — BC Cancer, 137 FL, Illumina DASL, R-chemo
  2. GSE93261  — PRIMA trial, 149 FL, Affymetrix U133+2, R-chemo

Usage:
    python3.11 scripts/cd14_validation.py --output-dir output/cd14_validation
"""

import argparse
import gzip
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


def parse_series_matrix(path):
    """Parse GEO series matrix file into phenotype df + expression df."""
    pheno_lines = []
    expr_lines = []
    in_table = False

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_"):
                pheno_lines.append(line)
            elif line.startswith('"ID_REF"'):
                in_table = True
                expr_lines.append(line)
            elif in_table and not line.startswith("!"):
                if line.strip():
                    expr_lines.append(line)

    # Parse phenotype
    pheno_dict = {}
    for line in pheno_lines:
        parts = line.split("\t")
        key = parts[0]
        vals = [v.strip('"') for v in parts[1:]]

        if key == "!Sample_geo_accession":
            pheno_dict["sample_id"] = vals
        elif key == "!Sample_characteristics_ch1":
            # Extract field name from first value
            if ":" in vals[0]:
                field_name = vals[0].split(":")[0].strip()
                field_vals = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
                pheno_dict[field_name] = field_vals

    pheno = pd.DataFrame(pheno_dict)

    # Parse expression
    from io import StringIO
    expr_text = "\n".join(expr_lines)
    expr = pd.read_csv(StringIO(expr_text), sep="\t", index_col=0)
    expr.columns = [c.strip('"') for c in expr.columns]

    return pheno, expr


def find_cd14_probes(expr, platform):
    """Find CD14 probe IDs based on platform."""
    if platform == "illumina_dasl":
        # Illumina HumanHT-12 DASL: probe IDs like ILMN_XXXXXXX
        # CD14 probes: ILMN_1702539, ILMN_2396444
        cd14_probes = [p for p in expr.index if "ILMN" in str(p)]
        # We'll need to look up probe-gene mapping
        return None  # Will use annotation
    elif platform == "affy_u133plus2":
        # Affymetrix U133+2: CD14 probes
        # Known CD14 probes: 201743_at, 201744_s_at
        known = ["201743_at", "201744_s_at"]
        found = [p for p in known if p in expr.index]
        return found
    return None


def analyze_cohort(name, matrix_path, platform, output_dir):
    """Run CD14 survival analysis on one cohort."""
    print(f"\n{'='*60}")
    print(f"Analyzing {name}")
    print(f"{'='*60}")

    pheno, expr = parse_series_matrix(matrix_path)
    print(f"  Samples: {len(pheno)}")
    print(f"  Probes: {len(expr)}")
    print(f"  Phenotype fields: {list(pheno.columns)}")

    # Get CD14 expression
    # Known CD14 probe IDs per platform
    cd14_probe_map = {
        "affy_u133plus2": ["201743_at", "201744_s_at"],
        "illumina_dasl": ["ILMN_2396444", "ILMN_1702539"],
    }
    candidates = cd14_probe_map.get(platform, [])
    found = [p for p in candidates if p in expr.index]
    if not found:
        print(f"  ERROR: No CD14 probes found. Tried: {candidates}")
        return None
    print(f"  CD14 probes found: {found}")
    cd14_expr = expr.loc[found].mean(axis=0) if len(found) > 1 else expr.loc[found[0]]

    # Merge with phenotype
    sample_ids = pheno["sample_id"].values
    cd14_values = cd14_expr[sample_ids].values if hasattr(cd14_expr, "__getitem__") else cd14_expr.values
    pheno["CD14_expr"] = cd14_values.astype(float)

    # Print basic stats
    print(f"\n  CD14 expression: mean={pheno['CD14_expr'].mean():.3f}, "
          f"median={pheno['CD14_expr'].median():.3f}, "
          f"std={pheno['CD14_expr'].std():.3f}")

    return pheno


def get_illumina_cd14(expr):
    """Get CD14 expression from Illumina array using GPL annotation."""
    import urllib.request

    # GPL21185 = Illumina HumanHT-12 WG-DASL V4.0
    # Download annotation table
    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL21nnn/GPL21185/annot/GPL21185.annot.gz"
    try:
        print(f"  Fetching annotation from {url}...")
        response = urllib.request.urlopen(url, timeout=30)
        data = gzip.decompress(response.read()).decode("utf-8")

        # Parse annotation to find CD14 probes
        cd14_probes = []
        for line in data.split("\n"):
            if line.startswith("#") or line.startswith("!") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) > 3:
                probe_id = parts[0].strip('"')
                gene_symbol = parts[3].strip('"') if len(parts) > 3 else ""
                if gene_symbol == "CD14":
                    cd14_probes.append(probe_id)

        if not cd14_probes:
            # Try alternate column positions
            for line in data.split("\n"):
                if "CD14" in line and not line.startswith("#"):
                    parts = line.split("\t")
                    probe_id = parts[0].strip('"')
                    # Check if any field is exactly CD14
                    for p in parts:
                        if p.strip('"').strip() == "CD14":
                            cd14_probes.append(probe_id)
                            break

        found = [p for p in cd14_probes if p in expr.index]
        if found:
            print(f"  Found CD14 probes: {found}")
            return expr.loc[found].mean(axis=0)
        else:
            print(f"  CD14 probe IDs from annotation: {cd14_probes[:5]}")
            print(f"  But none found in expression matrix index (first 5): {list(expr.index[:5])}")
            return None
    except Exception as e:
        print(f"  Annotation download failed: {e}")
        # Fallback: search expression index for anything CD14-like
        return None


def run_survival(pheno, time_col, event_col, name, gene="CD14", output_dir="."):
    """Run Cox PH and KM analysis for CD14."""
    df = pheno[["CD14_expr", time_col, event_col]].dropna().copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna()

    # Ensure positive times
    df = df[df[time_col] > 0]

    n = len(df)
    n_events = int(df[event_col].sum())
    print(f"\n  {name}: n={n}, events={n_events}")

    if n < 20:
        print(f"  Too few samples for survival analysis")
        return None

    # Continuous Cox PH
    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col=time_col, event_col=event_col)
        hr = np.exp(cph.params_["CD14_expr"])
        ci = cph.confidence_intervals_.iloc[0]
        p = cph.summary["p"]["CD14_expr"]
        print(f"  Cox PH (continuous): HR={hr:.3f} "
              f"(95% CI: {np.exp(ci.iloc[0]):.3f}-{np.exp(ci.iloc[1]):.3f}), p={p:.4f}")
    except Exception as e:
        print(f"  Cox PH failed: {e}")
        hr, p = None, None

    # Median split KM
    median_cd14 = df["CD14_expr"].median()
    df["CD14_high"] = (df["CD14_expr"] >= median_cd14).astype(int)

    high = df[df["CD14_high"] == 1]
    low = df[df["CD14_high"] == 0]
    lr = logrank_test(high[time_col], low[time_col], high[event_col], low[event_col])
    print(f"  Log-rank (median split): p={lr.p_value:.4f}")
    print(f"    CD14-high: n={len(high)}, events={int(high[event_col].sum())}")
    print(f"    CD14-low:  n={len(low)}, events={int(low[event_col].sum())}")

    # KM plot
    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()

    kmf.fit(high[time_col], high[event_col], label=f"CD14-high (n={len(high)})")
    kmf.plot_survival_function(ax=ax, color="#d62728", ci_show=True)

    kmf.fit(low[time_col], low[event_col], label=f"CD14-low (n={len(low)})")
    kmf.plot_survival_function(ax=ax, color="#1f77b4", ci_show=True)

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Survival probability", fontsize=12)
    endpoint_label = name.split(" - ")[-1] if " - " in name else name
    ax.set_title(f"CD14 expression and {endpoint_label}", fontsize=13)

    # Add stats
    stats_text = f"Log-rank p = {lr.p_value:.4f}"
    if hr is not None:
        stats_text += f"\nHR = {hr:.2f} (p = {p:.4f})"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.legend(loc="lower left", fontsize=10)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    fig_path = os.path.join(output_dir, f"cd14_{safe_name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return {"name": name, "n": n, "n_events": n_events, "hr": hr, "p_cox": p,
            "p_logrank": lr.p_value}


def main():
    parser = argparse.ArgumentParser(description="CD14 survival validation")
    parser.add_argument("--output-dir", default="output/cd14_validation")
    parser.add_argument("--gse119214", default="data/GSE119214/GSE119214_series_matrix.txt.gz")
    parser.add_argument("--gse93261", default="data/GSE93261/GSE93261_series_matrix.txt.gz")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    # === GSE119214 (BC Cancer, Illumina DASL) ===
    if os.path.exists(args.gse119214):
        pheno = analyze_cohort("GSE119214 (BC Cancer, n=137)",
                               args.gse119214, "illumina_dasl", args.output_dir)
        if pheno is not None:
            # FFS (failure-free survival)
            r = run_survival(pheno, "ffs", "codeffs",
                             "GSE119214 - Failure-Free Survival",
                             output_dir=args.output_dir)
            if r:
                results.append(r)

            # OS
            r = run_survival(pheno, "os", "codeos",
                             "GSE119214 - Overall Survival",
                             output_dir=args.output_dir)
            if r:
                results.append(r)
    else:
        print(f"Skipping GSE119214: {args.gse119214} not found")

    # === GSE93261 (PRIMA trial, Affymetrix U133+2) ===
    if os.path.exists(args.gse93261):
        pheno = analyze_cohort("GSE93261 (PRIMA trial, n=149)",
                               args.gse93261, "affy_u133plus2", args.output_dir)
        if pheno is not None:
            # Check what survival fields are available
            surv_fields = [c for c in pheno.columns
                           if any(s in c.lower() for s in ["pfs", "os", "efs", "surv",
                                                            "event", "time", "status"])]
            print(f"\n  Potential survival fields: {surv_fields}")

            # PRIMA data may have PFS in supplementary — check all fields
            print(f"  All fields: {list(pheno.columns)}")

            # If no explicit survival fields, still report CD14 expression
            if not surv_fields:
                print("  No survival fields found in GSE93261 phenotype data.")
                print("  CD14 expression summary:")
                print(f"    Mean: {pheno['CD14_expr'].mean():.3f}")
                print(f"    Std:  {pheno['CD14_expr'].std():.3f}")

                # Check if FLIPI is available for stratification
                if "flipi_score" in pheno.columns:
                    for score in sorted(pheno["flipi_score"].unique()):
                        subset = pheno[pheno["flipi_score"] == score]
                        print(f"    FLIPI={score}: n={len(subset)}, "
                              f"CD14 mean={subset['CD14_expr'].mean():.3f}")
    else:
        print(f"Skipping GSE93261: {args.gse93261} not found")

    # === Summary ===
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['name']}")
            print(f"    n={r['n']}, events={r['n_events']}")
            if r["hr"] is not None:
                print(f"    Cox HR={r['hr']:.3f}, p={r['p_cox']:.4f}")
            print(f"    Log-rank p={r['p_logrank']:.4f}")
            print()

    # === Combined figure ===
    if len(results) >= 2:
        make_combined_figure(results, args.output_dir)


def make_combined_figure(results, output_dir):
    """Create summary forest plot of CD14 HRs across cohorts."""
    fig, ax = plt.subplots(figsize=(7, 3))

    valid = [r for r in results if r["hr"] is not None]
    for i, r in enumerate(valid):
        ax.plot(r["hr"], i, "ko", markersize=8)
        # Simple CI approximation from p-value
        ax.annotate(f'{r["name"]}\nHR={r["hr"]:.2f}, p={r["p_cox"]:.4f}',
                     xy=(r["hr"], i), xytext=(15, 0),
                     textcoords="offset points", va="center", fontsize=9)

    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels([r["name"].split(" - ")[0] for r in valid])
    ax.set_xlabel("Hazard Ratio (CD14 expression)", fontsize=11)
    ax.set_title("CD14 as survival predictor in FL", fontsize=12)
    plt.tight_layout()

    path = os.path.join(output_dir, "cd14_forest_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Forest plot saved: {path}")


if __name__ == "__main__":
    main()
