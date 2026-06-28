#!/usr/bin/env python3
"""Prelim: does FL architectural structure break down with grade?

Per-ROI metrics (S-panel UTAG h5ad), then aggregated to patient level for
inference. Stratified by FL grade (FOLL1 / FOLL2 / FOLL3A).

Metrics:
  - frac_follicular         : cells in B cell zones (PAX5+/BCL2+) + FDC network zone
  - frac_follicular_loose   : same + Mixed-B-cells + B/T-mixed (sensitivity variant)
  - frac_interfoll          : cells in T cell zone + Stromal + Other myeloid
  - n_follicles             : connected components of follicular cells (size >= MIN_FOLL)
  - mean_follicle_size      : mean cells per follicle
  - shannon_ct              : Shannon entropy of cell type composition (excluding Unassigned)

Stats:
  - Per-ROI Kruskal-Wallis (NOT trustworthy when same patient contributes multiple ROIs)
  - Per-patient Kruskal-Wallis (mean of patient's ROIs as the unit of analysis) ← primary
  - Sensitivity sweep on follicle-count thresholds (NEIGHBOR_DIST_UM, MIN_FOLLICLE_SIZE)

Cohort filtering follows CLAUDE.md guardrail #3 + clinical_linkage.EXCLUDE_ROIS:
  - exclude tonsil/prostate/kidney/spleen/adrenal control cores
  - exclude Biomax samples (no clinical, includes reactive lymph nodes)
  - exclude EXCLUDE_ROIS = {A1_ROI_005}
  - normalize sample_ids via normalize_sample_id (A1_ROI_X → A1_FLX, B1_FL0X → B1_FLX)

Usage:
    .venv/bin/python scripts/prelim_grade_architecture.py \\
        --s-panel output/all_TMA_S_utag_ct_merged.h5ad \\
        --clinical data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv \\
        --grade data/clinicaldata/BCCA_tFL_clinical.xlsx \\
        --out output/grade_arch
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
from scipy.spatial import cKDTree
from scipy.stats import kruskal

# Reuse project cohort-filter helpers
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

FOLLICULAR = {"B cell zone (BCL2+)", "B cell zone (PAX5+)", "FDC network zone"}
FOLLICULAR_LOOSE = FOLLICULAR | {"Mixed (B cells (PAX 27%)", "B/T mixed zone"}
INTERFOLL = {"T cell zone", "Stromal / CAF zone", "Other / myeloid zone"}

# Defaults — overridable from CLI
DEFAULT_MIN_CELLS_PER_ROI = 8000   # CLAUDE.md guardrail #10
DEFAULT_MIN_FOLLICLE_SIZE = 15
DEFAULT_NEIGHBOR_DIST_UM = 30.0
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}


def is_tumor_core(sid: str) -> bool:
    """CLAUDE.md guardrail #3 + Biomax exclusion (no clinical, includes reactive LNs)."""
    s = str(sid).lower()
    if any(t in s for t in ("tonsil", "prostate", "kidney", "spleen", "adrenal")):
        return False
    if "_ton_" in s or "_adr_" in s or "_lym_" in s or "_lym " in s:
        return False
    if s.startswith("biomax"):
        return False
    if sid in EXCLUDE_ROIS:
        return False
    return True


def load_data(h5ad_path: Path) -> pd.DataFrame:
    with h5py.File(h5ad_path, "r") as f:
        sid_codes = f["obs/sample_id/codes"][:]
        sid_cats = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in f["obs/sample_id/categories"][:]])
        sample_id = sid_cats[sid_codes]

        ct_codes = f["obs/cell_type/codes"][:]
        ct_cats = np.array([c.decode() if isinstance(c, bytes) else c
                            for c in f["obs/cell_type/categories"][:]])
        cell_type = ct_cats[ct_codes]

        comp_codes = f["obs/compartment_name/codes"][:]
        comp_cats = np.array([c.decode() if isinstance(c, bytes) else c
                              for c in f["obs/compartment_name/categories"][:]])
        compartment = comp_cats[comp_codes]

        cx = f["obs/centroid_x"][:]
        cy = f["obs/centroid_y"][:]
    return pd.DataFrame({
        "sample_id": sample_id, "cell_type": cell_type,
        "compartment": compartment, "x": cx, "y": cy,
    })


def connected_component_sizes(xy: np.ndarray, neighbor_dist: float) -> np.ndarray:
    """Union-Find connected components on a 2D point set with a distance edge cutoff."""
    n = len(xy)
    if n == 0:
        return np.array([], dtype=int)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(neighbor_dist, output_type="ndarray")
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    return pd.Series(roots).value_counts().values


def per_roi_metrics(roi_df: pd.DataFrame, *, min_cells: int,
                    neighbor_dist: float, min_foll_size: int) -> dict | None:
    n_cells = len(roi_df)
    not_unassigned = ~roi_df["cell_type"].isin(["Unassigned", "Low quality / Unassigned"])
    n_typed = int(not_unassigned.sum())
    if n_typed < min_cells:
        return None

    is_foll = roi_df["compartment"].isin(FOLLICULAR)
    is_foll_loose = roi_df["compartment"].isin(FOLLICULAR_LOOSE)
    is_inter = roi_df["compartment"].isin(INTERFOLL)
    frac_foll = is_foll.sum() / n_cells
    frac_foll_loose = is_foll_loose.sum() / n_cells
    frac_inter = is_inter.sum() / n_cells

    n_follicles, mean_size = 0, np.nan
    if is_foll.sum() >= min_foll_size:
        sizes = connected_component_sizes(roi_df.loc[is_foll, ["x", "y"]].to_numpy(),
                                          neighbor_dist)
        big = sizes[sizes >= min_foll_size]
        n_follicles = int(len(big))
        mean_size = float(big.mean()) if len(big) else 0.0

    typed = roi_df.loc[not_unassigned, "cell_type"]
    if len(typed) > 0:
        p = typed.value_counts(normalize=True).values
        shannon = float(-(p * np.log2(p + 1e-12)).sum())
    else:
        shannon = np.nan

    return {
        "n_cells": int(n_cells), "n_typed": n_typed,
        "frac_follicular": float(frac_foll),
        "frac_follicular_loose": float(frac_foll_loose),
        "frac_interfollicular": float(frac_inter),
        "n_follicles": n_follicles,
        "mean_follicle_size": float(mean_size) if not np.isnan(mean_size) else np.nan,
        "shannon_ct": shannon,
    }


def compute_metrics_for_thresholds(df: pd.DataFrame, min_cells: int,
                                   neighbor_dist: float, min_foll_size: int) -> pd.DataFrame:
    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_metrics(sub, min_cells=min_cells,
                            neighbor_dist=neighbor_dist, min_foll_size=min_foll_size)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    return pd.DataFrame(rows)


def join_clinical_and_grade(metrics_df: pd.DataFrame, clinical_csv: Path,
                             grade_xlsx: Path) -> pd.DataFrame:
    clin = pd.read_csv(clinical_csv)[["slide_ID", "Sample_ID", "Patient_ID"]]
    grade = pd.read_excel(grade_xlsx).rename(columns={"FL ID": "Sample_ID", "DIAG": "grade"})
    grade = grade[["Sample_ID", "grade"]]
    out = metrics_df.merge(clin, left_on="sample_id", right_on="slide_ID", how="left")
    out = out.merge(grade, on="Sample_ID", how="left")
    out = out[out["grade"].isin(GRADE_ORDER)].copy()
    return out


def kw_report(df: pd.DataFrame, metric: str) -> tuple[float, float, dict]:
    """KW across grades on `metric`. Returns (H, p, group_medians)."""
    groups = [df.loc[df["grade"] == g, metric].dropna().values for g in GRADE_ORDER]
    if any(len(x) < 3 for x in groups):
        return np.nan, np.nan, {g: np.nan for g in GRADE_ORDER}
    H, p = kruskal(*groups)
    medians = {g: float(np.median(grp)) for g, grp in zip(GRADE_ORDER, groups)}
    return float(H), float(p), medians


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s-panel", default="output/all_TMA_S_utag_ct_merged.h5ad",
                   help="S-panel UTAG h5ad with compartment_name, cell_type, sample_id")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv",
                   help="BCCA clinical CSV (slide_ID ↔ Sample_ID ↔ Patient_ID)")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx",
                   help="BCCA tFL xlsx with Sample_ID and DIAG=grade")
    p.add_argument("--out", default="output/grade_arch")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
    p.add_argument("--min-foll-size", type=int, default=DEFAULT_MIN_FOLLICLE_SIZE)
    p.add_argument("--neighbor-dist", type=float, default=DEFAULT_NEIGHBOR_DIST_UM)
    p.add_argument("--sensitivity-sweep", action="store_true",
                   help="Sweep follicle-count thresholds and report grade trend stability")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.s_panel} ...")
    df = load_data(Path(args.s_panel))
    print(f"  cells={len(df):,}, ROIs={df['sample_id'].nunique()}")

    # Cohort filter (do BEFORE per-ROI metrics so denominators are honest)
    df = df[df["sample_id"].apply(is_tumor_core)]
    # Apply normalize_sample_id so we match clinical slide_ID format
    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)
    print(f"  after exclusion + normalization: cells={len(df):,}, ROIs={df['sample_id'].nunique()}")

    metrics_df = compute_metrics_for_thresholds(
        df, min_cells=args.min_cells,
        neighbor_dist=args.neighbor_dist, min_foll_size=args.min_foll_size,
    )
    print(f"  ROIs with >={args.min_cells} typed cells: {len(metrics_df)}")

    metrics_df = join_clinical_and_grade(metrics_df, Path(args.clinical), Path(args.grade))
    print(f"  ROIs with grade: {len(metrics_df)}")
    print(f"  Patients with grade: {metrics_df['Patient_ID'].nunique()}")
    print(metrics_df["grade"].value_counts().to_string())
    print()

    out_csv = out_dir / "grade_arch_per_roi.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"  Saved per-ROI metrics: {out_csv}")

    # Per-patient aggregation: mean of each metric across the patient's ROIs
    metric_cols = ["frac_follicular", "frac_follicular_loose", "frac_interfollicular",
                   "n_follicles", "mean_follicle_size", "shannon_ct"]
    agg = (metrics_df.groupby(["Patient_ID", "grade"])[metric_cols]
           .mean().reset_index())
    print(f"\n  Patient-level n: {len(agg)}; per-grade:")
    print(agg["grade"].value_counts().to_string())
    out_csv_pt = out_dir / "grade_arch_per_patient.csv"
    agg.to_csv(out_csv_pt, index=False)
    print(f"  Saved per-patient metrics: {out_csv_pt}")

    # ── Stats: ROI-level vs patient-level ──
    metrics_to_test = [
        ("frac_follicular",        "Follicular fraction"),
        ("frac_follicular_loose",  "Follicular fraction (incl Mixed-B)"),
        ("frac_interfollicular",   "Interfollicular fraction"),
        ("n_follicles",            "Follicle count per ROI"),
        ("mean_follicle_size",     "Mean follicle size (cells)"),
        ("shannon_ct",             "Shannon entropy (cell types)"),
    ]
    print("\n=== KW: ROI-level vs patient-level ===")
    print(f"{'metric':35s} {'ROI p':>10s} {'patient p':>12s}  patient medians (FOLL1/2/3A)")
    summary = []
    for k, label in metrics_to_test:
        _, p_roi, _ = kw_report(metrics_df, k)
        _, p_pat, m_pat = kw_report(agg, k)
        med_str = " / ".join(f"{m_pat[g]:.3g}" for g in GRADE_ORDER)
        print(f"  {label:35s} {p_roi:10.4g} {p_pat:12.4g}  {med_str}")
        summary.append({"metric": k, "label": label,
                        "p_roi": p_roi, "p_patient": p_pat,
                        **{f"med_{g}": m_pat[g] for g in GRADE_ORDER}})
    pd.DataFrame(summary).to_csv(out_dir / "grade_arch_kw_summary.csv", index=False)

    # ── Sensitivity sweep ──
    if args.sensitivity_sweep:
        print("\n=== Sensitivity sweep: follicle-count thresholds ===")
        sweep_rows = []
        for d in (20.0, 30.0, 40.0, 50.0):
            for s in (10, 15, 20, 30):
                m_df = compute_metrics_for_thresholds(
                    df, min_cells=args.min_cells, neighbor_dist=d, min_foll_size=s,
                )
                m_df = join_clinical_and_grade(m_df, Path(args.clinical), Path(args.grade))
                ag = (m_df.groupby(["Patient_ID", "grade"])[["n_follicles"]]
                      .mean().reset_index())
                _, p_pat, m_pat = kw_report(ag, "n_follicles")
                medians = " / ".join(f"{m_pat[g]:.2f}" for g in GRADE_ORDER)
                monotonic = (m_pat[GRADE_ORDER[0]] <= m_pat[GRADE_ORDER[1]] <= m_pat[GRADE_ORDER[2]]
                             or m_pat[GRADE_ORDER[0]] >= m_pat[GRADE_ORDER[1]] >= m_pat[GRADE_ORDER[2]])
                tag = "MONO" if monotonic else "    "
                print(f"  d={d:>4.0f} µm, min_size={s:>3d}: patient KW p={p_pat:.4g}  "
                      f"medians={medians}  [{tag}]")
                sweep_rows.append({"neighbor_dist": d, "min_foll_size": s,
                                   "p_patient": p_pat,
                                   **{f"med_{g}": m_pat[g] for g in GRADE_ORDER},
                                   "monotonic": bool(monotonic)})
        pd.DataFrame(sweep_rows).to_csv(out_dir / "grade_arch_sensitivity.csv", index=False)

    # ── Figure (patient-level box plots) ──
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    plot_metrics = [m for m in metrics_to_test if m[0] != "frac_follicular_loose"]
    rng = np.random.default_rng(0)
    for ax, (k, label) in zip(axes, plot_metrics):
        data = [agg.loc[agg["grade"] == g, k].dropna().values for g in GRADE_ORDER]
        bp = ax.boxplot(data, tick_labels=GRADE_ORDER, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GRADE_ORDER):
            patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.55)
        for i, (g, vals) in enumerate(zip(GRADE_ORDER, data)):
            xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(xs, vals, color=GRADE_COLORS[g], s=10, alpha=0.7,
                       edgecolor="white", linewidth=0.4, zorder=3)
        if all(len(x) >= 3 for x in data):
            _, p = kruskal(*data)
            ax.set_title(f"{label}\nKW p = {p:.3g} (per-patient)", fontsize=12)
        else:
            ax.set_title(label, fontsize=12)
        ax.set_xlabel("Grade")
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle(f"FL grade vs spatial architecture — patient-level (n={len(agg)} patients)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out_png = out_dir / "fig_grade_architecture.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out_png}")


if __name__ == "__main__":
    main()
