#!/usr/bin/env python3
"""Compartment-compartment spatial interactions by FL grade.

Two analyses:

A. Per-ROI 'compartments present' counts at 2%, 5%, 10% (visualize).
B. Compartment-compartment spatial adjacency: for each cell, find its K
   nearest neighbors, count how many sit in a DIFFERENT compartment from the
   center cell. Aggregate per ROI as `inter_compartment_neighbor_frac` — the
   per-cell mean fraction of neighbors that are in a different compartment.
   Interpretation: 0 = perfect compartmentalization (every cell surrounded by
   its own kind); → 1 = fully mixed.
   Then test by grade.

Cohort filter and patient-level aggregation match `grade_followups.py`.

Usage:
    .venv/bin/python scripts/grade_compartment_interactions.py
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

DEFAULT_MIN_CELLS_PER_ROI = 8000
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}
K_NEIGHBORS = 10


def is_tumor_core(sid: str) -> bool:
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


def kw(df, metric):
    groups = [df.loc[df.grade == g, metric].dropna().values for g in GRADE_ORDER]
    if any(len(x) < 3 for x in groups):
        return np.nan, {g: np.nan for g in GRADE_ORDER}
    medians = {g: float(np.median(grp)) for g, grp in zip(GRADE_ORDER, groups)}
    if len(np.unique(np.concatenate(groups))) < 2:
        return np.nan, medians
    try:
        _, p = kruskal(*groups)
        return float(p), medians
    except ValueError:
        return np.nan, medians


def per_roi_metrics(roi_df, k=K_NEIGHBORS):
    """Per-ROI inter-compartment-neighbor fraction.

    All cells (typed + Unassigned) participate as candidate NEIGHBORS so the
    spatial structure is preserved, but only TYPED cells are used as CENTERS
    when averaging — otherwise typing failures (Unassigned center cells) would
    contribute their own pseudo-mixing and bias the metric.
    """
    not_unassigned = ~roi_df["cell_type"].isin(["Unassigned", "Low quality / Unassigned"])
    n_typed = int(not_unassigned.sum())
    out = {"n_typed": n_typed}

    coords = roi_df[["x", "y"]].to_numpy()
    n = len(coords)
    if n < k + 1 or n_typed < k:
        out["inter_compartment_neighbor_frac"] = np.nan
        return out

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    nbrs = idx[:, 1:]
    comp_arr = roi_df["compartment"].to_numpy()
    same_comp = comp_arr[nbrs] == comp_arr[:, None]
    inter_per_cell = 1.0 - same_comp.mean(axis=1)
    # Center-cell filter: TYPED cells only
    centers_mask = not_unassigned.to_numpy()
    out["inter_compartment_neighbor_frac"] = float(inter_per_cell[centers_mask].mean())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s-panel", default="output/all_TMA_S_utag_ct_merged.h5ad")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--per-patient-input",
                   default="output/grade_arch/grade_compartment_biomarkers_per_patient.csv",
                   help="CSV with shannon/n_compartments_present_p* + grade + Patient_ID")
    p.add_argument("--out", default="output/grade_arch")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
    p.add_argument("--k", type=int, default=K_NEIGHBORS)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # --- A. Reuse the per-patient CSV from compartment-biomarkers for the threshold sweep panel
    cb = pd.read_csv(args.per_patient_input)
    cb = cb[cb.grade.isin(GRADE_ORDER)].copy()
    print(f"Patients (compartment-presence): {len(cb)} ({cb.grade.value_counts().to_dict()})")

    # --- B. Compute inter-compartment neighbor fraction directly from S-panel
    print("\nComputing inter-compartment neighbor fractions ...")
    with h5py.File(args.s_panel, "r") as f:
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

    df = pd.DataFrame({"sample_id": sample_id, "cell_type": cell_type,
                       "compartment": compartment, "x": cx, "y": cy})
    df = df[df.sample_id.apply(is_tumor_core)].copy()
    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)
    print(f"  After filter: cells={len(df):,}, ROIs={df.sample_id.nunique()}")
    n_compart = df["compartment"].nunique()
    print(f"  Compartments observed: {n_compart} unique values "
          f"(expect 11 from merged scheme; >15 would indicate raw Leiden)")
    if n_compart > 20:
        print("  WARNING: too many compartments — verify h5ad uses merged scheme.")

    # Filter ROIs that pass min_cells (typed)
    typed_mask = ~df["cell_type"].isin(["Unassigned", "Low quality / Unassigned"])
    typed_per_roi = (df.assign(typed=typed_mask)
                     .groupby("sample_id")["typed"].sum())
    keep = typed_per_roi[typed_per_roi >= args.min_cells].index
    df = df[df.sample_id.isin(keep)]
    print(f"  After min_cells={args.min_cells}: ROIs={df.sample_id.nunique()}")

    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_metrics(sub, k=args.k)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows)

    # Join grade + Patient_ID
    clin = pd.read_csv(args.clinical)[["slide_ID", "Sample_ID", "Patient_ID"]]
    # Grade sourced from DWS clinical (native GRADE col); legacy --grade
    # xlsx arg accepted but ignored.
    import warnings as _warn
    from src.clinical_linkage import load_clinical as _load_clinical
    with _warn.catch_warnings():
        _warn.simplefilter("ignore")
        _dws = _load_clinical()
    grade_df = _dws[["Sample_ID", "GRADE"]].rename(columns={"GRADE": "grade"})
    metrics_df = metrics_df.merge(clin, left_on="sample_id", right_on="slide_ID", how="left")
    metrics_df = metrics_df.merge(grade_df, on="Sample_ID", how="left")
    metrics_df = metrics_df[metrics_df.grade.isin(GRADE_ORDER)].copy()

    # Patient aggregation
    pt = (metrics_df.groupby(["Patient_ID", "grade"])
          [["inter_compartment_neighbor_frac"]]
          .mean().reset_index())
    pt.to_csv(out_dir / "grade_compartment_interactions_per_patient.csv", index=False)
    print(f"  Patient-level n: {len(pt)} ({pt.grade.value_counts().to_dict()})")

    # KW test (single metric — modularity proxy was 1 - inter, redundant)
    print("\n=== KW test (patient-level) ===")
    p_val, med = kw(pt, "inter_compartment_neighbor_frac")
    med_str = " / ".join(f"{med[g]:.3g}" for g in GRADE_ORDER)
    print(f"  Inter-compartment neighbor fraction "
          f"(typed cells as centers, K={args.k}):  p={p_val:.4g}  "
          f"medians={med_str}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    rng = np.random.default_rng(0)

    def boxplot(ax, df_pt, metric, label):
        data = [df_pt.loc[df_pt.grade == g, metric].dropna().values for g in GRADE_ORDER]
        bp = ax.boxplot(data, tick_labels=GRADE_ORDER, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GRADE_ORDER):
            patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.55)
        for i, (g, vals) in enumerate(zip(GRADE_ORDER, data)):
            xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(xs, vals, color=GRADE_COLORS[g], s=10, alpha=0.7,
                       edgecolor="white", linewidth=0.4, zorder=3)
        if all(len(x) >= 3 for x in data):
            _, pv = kruskal(*data)
            ax.set_title(f"{label}\np={pv:.3g}", fontsize=11)
        else:
            ax.set_title(label, fontsize=11)
        ax.set_xlabel("Grade")
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Three threshold panels
    for ax, thr in zip(axes[:3], (0.02, 0.05, 0.10)):
        col = f"n_compartments_present_p{int(thr*100):02d}"
        boxplot(ax, cb, col, f"Compartments present (≥{int(thr*100)}%)")

    # Inter-compartment neighbor fraction
    boxplot(axes[3], pt, "inter_compartment_neighbor_frac",
            f"Inter-compartment neighbor frac\n(K={args.k}, per-cell mean)")

    fig.suptitle("Compartment 'presence' thresholds + spatial mixing by grade "
                 f"(S-panel, n={len(pt)} patients)", fontsize=12, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_grade_compartment_interactions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
