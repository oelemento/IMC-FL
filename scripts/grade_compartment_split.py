#!/usr/bin/env python3
"""Compartment count + size by FL grade, split by follicular vs interfollicular.

For each ROI, compute (separately for follicular and interfollicular sub-
compartments):
  - n_compartments_present_p05  : count of sub-compartments with ≥5% of ROI cells
  - mean_compartment_size_cells : mean cell count of those sub-compartments
  - mean_compartment_size_frac  : mean cell fraction (of ROI cells) of those sub-compartments

Stratification: FOLL1 / FOLL2 / FOLL3A. Patient-level KW.

Macro groups (matches CLAUDE.md guardrail #7 + Mixed-B/Mixed-M2 placement):
  Follicular     : B cell zone (BCL2+), B cell zone (PAX5+), FDC network zone,
                    FDC / myeloid zone, Mixed (B cells (PAX 27%)
  Interfollicular: T cell zone, Stromal / CAF zone, Other / myeloid zone,
                    B/T mixed zone, Mixed (M2 Macrophag 26%)
  Excluded       : Unidentified zone

Cohort filter and patient-level aggregation match `grade_compartment_size.py`.
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
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

DEFAULT_MIN_CELLS_PER_ROI = 8000
PRESENCE_THRESHOLD = 0.05
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}

FOLLICULAR = {
    "B cell zone (BCL2+)",
    "B cell zone (PAX5+)",
    "FDC network zone",
    "FDC / myeloid zone",
    "Mixed (B cells (PAX 27%)",
}
INTERFOLLICULAR = {
    "T cell zone",
    "Stromal / CAF zone",
    "Other / myeloid zone",
    "B/T mixed zone",
    "Mixed (M2 Macrophag 26%)",
}
# "Unidentified zone" excluded from both


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


def per_roi_metrics(roi_df, min_cells, presence_thresh=PRESENCE_THRESHOLD):
    not_unassigned = ~roi_df["cell_type"].isin(["Unassigned", "Low quality / Unassigned"])
    n_typed = int(not_unassigned.sum())
    if n_typed < min_cells:
        return None
    n_total = len(roi_df)
    comp_counts = roi_df["compartment"].value_counts()
    fracs = comp_counts / n_total

    out = {"n_typed": n_typed, "n_total": n_total}
    for label, group in (("foll", FOLLICULAR), ("inter", INTERFOLLICULAR)):
        present = fracs[(fracs.index.isin(group)) & (fracs >= presence_thresh)]
        present_counts = comp_counts.loc[present.index]
        out[f"n_{label}_present"] = int(len(present))
        out[f"mean_{label}_size_cells"] = float(present_counts.mean()) if len(present) else np.nan
        out[f"mean_{label}_size_frac"] = float(present.mean()) if len(present) else np.nan
        # Total fraction of the ROI in this macro group (regardless of threshold)
        out[f"frac_{label}_total"] = float(fracs[fracs.index.isin(group)].sum())
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
    df = pd.DataFrame({"sample_id": sample_id, "cell_type": cell_type,
                       "compartment": compartment})
    df = df[df.sample_id.apply(is_tumor_core)].copy()
    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)
    print(f"  cells={len(df):,}, ROIs={df.sample_id.nunique()}")

    obs_compart = set(df.compartment.unique())
    unmapped = obs_compart - FOLLICULAR - INTERFOLLICULAR - {"Unidentified zone"}
    if unmapped:
        raise RuntimeError(f"Unmapped compartments: {unmapped}")
    print(f"  Compartment groups: foll={len(FOLLICULAR)}, "
          f"inter={len(INTERFOLLICULAR)}, excl=Unidentified")

    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_metrics(sub, args.min_cells)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows)

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

    metric_cols = [c for c in metrics_df.columns
                   if c not in {"sample_id", "slide_ID", "Sample_ID",
                                "Patient_ID", "grade"}]
    pt = (metrics_df.groupby(["Patient_ID", "grade"])[metric_cols]
          .mean().reset_index())
    pt.to_csv(out_dir / "grade_compartment_split_per_patient.csv", index=False)
    print(f"  Patient-level n: {len(pt)}  ({pt.grade.value_counts().to_dict()})")

    print("\n=== KW (patient-level) ===")
    test_metrics = [
        ("frac_foll_total",            "Total follicular fraction"),
        ("frac_inter_total",           "Total interfollicular fraction"),
        ("n_foll_present",             "# Follicular sub-compartments (≥5%)"),
        ("n_inter_present",            "# Interfollicular sub-compartments (≥5%)"),
        ("mean_foll_size_frac",        "Mean follicular sub-compartment size (frac)"),
        ("mean_inter_size_frac",       "Mean interfollicular sub-compartment size (frac)"),
        ("mean_foll_size_cells",       "Mean follicular sub-compartment size (cells)"),
        ("mean_inter_size_cells",      "Mean interfollicular sub-compartment size (cells)"),
    ]
    for m, label in test_metrics:
        p_v, med = kw(pt, m)
        med_str = " / ".join(f"{med[g]:.4g}" for g in GRADE_ORDER)
        flag = " *" if p_v < 0.05 else ""
        print(f"  {label:50s} p={p_v:.4g}  medians={med_str}{flag}")

    # ── Figure: 2x4 ──
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
            ax.set_title(f"{label}\np={pv:.3g}", fontsize=10)
        else:
            ax.set_title(label, fontsize=10)
        ax.set_xlabel("Grade")
        ax.tick_params(labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    boxplot(axes[0, 0], "frac_foll_total",       "Total FOLLICULAR\nfraction of ROI")
    boxplot(axes[0, 1], "n_foll_present",        "# FOLLICULAR\nsub-compartments (≥5%)")
    boxplot(axes[0, 2], "mean_foll_size_frac",   "Mean FOLL sub-compartment\nsize (frac of ROI)")
    boxplot(axes[0, 3], "mean_foll_size_cells",  "Mean FOLL sub-compartment\nsize (cells)")

    boxplot(axes[1, 0], "frac_inter_total",      "Total INTERFOLLICULAR\nfraction of ROI")
    boxplot(axes[1, 1], "n_inter_present",       "# INTERFOLLICULAR\nsub-compartments (≥5%)")
    boxplot(axes[1, 2], "mean_inter_size_frac",  "Mean INTER sub-compartment\nsize (frac of ROI)")
    boxplot(axes[1, 3], "mean_inter_size_cells", "Mean INTER sub-compartment\nsize (cells)")

    fig.suptitle(f"Compartment count + size by grade — follicular (top) vs "
                 f"interfollicular (bottom) (S-panel, n={len(pt)} patients)",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    out = out_dir / "fig_grade_compartment_split.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
