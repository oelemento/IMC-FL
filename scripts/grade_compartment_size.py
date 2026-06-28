#!/usr/bin/env python3
"""Per-ROI compartment size by FL grade.

Three size metrics per ROI:
  - mean_compartment_size_cells     : mean cell count across compartments PRESENT (≥5%)
  - median_compartment_size_cells   : median cell count across compartments PRESENT (≥5%)
  - mean_fragment_size_cells        : mean cell count of connected fragments (per
                                      compartment), neighbor_dist=30 µm, min 10 cells

If grade adds more compartments to a fixed ROI, mean compartment size should drop.
If compartments also FRAGMENT (split into disconnected pieces) with grade, the
fragment-level metric will drop even more sharply.

Cohort filter and patient-level aggregation match `grade_followups.py`.
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
PRESENCE_THRESHOLD = 0.05      # compartment counts only if ≥5% of cells in ROI
NEIGHBOR_DIST = 30.0           # connected-component edge cutoff (µm)
MIN_FRAGMENT_CELLS = 10


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


def connected_component_sizes(xy, neighbor_dist):
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
    return pd.Series([find(i) for i in range(n)]).value_counts().values


def per_roi_metrics(roi_df, min_cells, presence_threshold=PRESENCE_THRESHOLD,
                    neighbor_dist=NEIGHBOR_DIST, min_fragment=MIN_FRAGMENT_CELLS):
    not_unassigned = ~roi_df["cell_type"].isin(["Unassigned", "Low quality / Unassigned"])
    n_typed = int(not_unassigned.sum())
    if n_typed < min_cells:
        return None
    n_total = len(roi_df)

    # Compartment-level sizes (only compartments PRESENT at ≥threshold)
    comp_counts = roi_df["compartment"].value_counts()
    present = comp_counts[comp_counts / n_total >= presence_threshold]
    if len(present) == 0:
        return None
    out = {
        "n_typed": n_typed,
        "n_total": n_total,
        "n_compartments_present": int(len(present)),
        "mean_compartment_size_cells": float(present.mean()),
        "median_compartment_size_cells": float(present.median()),
        "mean_compartment_size_frac": float(present.mean() / n_total),
    }

    # Fragment-level: per compartment, find connected components ≥min_fragment
    fragment_sizes = []
    for comp_name in present.index:
        sub = roi_df[roi_df["compartment"] == comp_name][["x", "y"]].to_numpy()
        sizes = connected_component_sizes(sub, neighbor_dist)
        big = sizes[sizes >= min_fragment]
        fragment_sizes.extend(big.tolist())
    if len(fragment_sizes) > 0:
        out["mean_fragment_size_cells"] = float(np.mean(fragment_sizes))
        out["median_fragment_size_cells"] = float(np.median(fragment_sizes))
        out["n_fragments"] = int(len(fragment_sizes))
    else:
        out["mean_fragment_size_cells"] = np.nan
        out["median_fragment_size_cells"] = np.nan
        out["n_fragments"] = 0

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s-panel", default="output/all_TMA_S_utag_ct_merged.h5ad")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--out", default="output/grade_arch")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.s_panel} ...")
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
    print(f"  cells={len(df):,}, ROIs={df.sample_id.nunique()}")

    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_metrics(sub, args.min_cells)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows)
    print(f"  ROIs after min_cells={args.min_cells}: {len(metrics_df)}")

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
    print(f"  with grade: {len(metrics_df)}")

    metric_cols = ["n_compartments_present",
                    "mean_compartment_size_cells", "median_compartment_size_cells",
                    "mean_compartment_size_frac",
                    "mean_fragment_size_cells", "median_fragment_size_cells",
                    "n_fragments", "n_typed", "n_total"]
    pt = (metrics_df.groupby(["Patient_ID", "grade"])[metric_cols]
          .mean().reset_index())
    pt.to_csv(out_dir / "grade_compartment_size_per_patient.csv", index=False)
    print(f"  Patient-level n: {len(pt)} ({pt.grade.value_counts().to_dict()})")

    print("\n=== KW tests (patient-level) ===")
    test_metrics = [
        ("n_compartments_present",        "Compartments present (≥5%)"),
        ("mean_compartment_size_cells",   "Mean compartment size (cells)"),
        ("median_compartment_size_cells", "Median compartment size (cells)"),
        ("mean_compartment_size_frac",    "Mean compartment size (fraction of ROI)"),
        ("mean_fragment_size_cells",      "Mean fragment size (cells)"),
        ("n_fragments",                    "Total fragments per ROI"),
    ]
    for m, label in test_metrics:
        p_val, med = kw(pt, m)
        med_str = " / ".join(f"{med[g]:.4g}" for g in GRADE_ORDER)
        flag = " *" if p_val < 0.05 else ""
        print(f"  {label:42s} p={p_val:.4g}  medians={med_str}{flag}")

    # ── Figure ──
    rng = np.random.default_rng(0)

    def boxplot(ax, metric, label):
        data = [pt.loc[pt.grade == g, metric].dropna().values for g in GRADE_ORDER]
        bp = ax.boxplot(data, tick_labels=GRADE_ORDER, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GRADE_ORDER):
            patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.55)
        for i, (g, vals) in enumerate(zip(GRADE_ORDER, data)):
            xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(xs, vals, color=GRADE_COLORS[g], s=12, alpha=0.7,
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

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    boxplot(axes[0], "mean_compartment_size_cells", "Mean compartment size\n(cells)")
    boxplot(axes[1], "mean_compartment_size_frac",  "Mean compartment size\n(fraction of ROI)")
    boxplot(axes[2], "mean_fragment_size_cells",     "Mean fragment size\n(cells, ≥10 connected)")
    boxplot(axes[3], "n_fragments",                  "Total fragments per ROI")

    fig.suptitle(f"Compartment size shrinks as grade rises? "
                 f"(S-panel, n={len(pt)} patients)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_grade_compartment_size.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
