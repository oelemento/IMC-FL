#!/usr/bin/env python3
"""T-panel grade-architecture analysis using PAPER's 9-compartment scheme.

Differs from `grade_compartments_t.py`:
  1. Restricts to the 9 compartments used in Fig 2 / immune_evasion.py:
     GC core, Follicle core, Follicle mantle, B cell follicle, B cell zone,
     Follicle-T zone interface, Treg-enriched T zone, T cell zone, Macrophage-rich zone
  2. Computes per-ROI metrics over ONLY these 9 compartments (cells in
     excluded compartments — Unidentified, LQ variants, Activated B — are
     dropped before normalization).
  3. Filters ROIs with elevated low-quality cell fraction (>40% Unassigned),
     which the diagnostic flagged as confounding outliers (e.g., B1_FL36).

Same patient-level KW + BH within family, same cohort filter as prior work.
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
MAX_UNASSIGNED_FRAC = 0.40         # drop ROIs with >40% Unassigned
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}
T_UNASSIGNED = ["Unassigned", "Low quality / Unassigned"]

# Paper's 9-compartment scheme (from immune_evasion.GRADIENT_ORDER)
PAPER_9 = {
    "GC core",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)",
    "B cell follicle (CD20hi/CXCR5hi)",
    "B cell zone",
    "Follicle-T zone interface",
    "Treg-enriched T zone",
    "T cell zone (CD4/CD8)",
    "Macrophage-rich zone",
}


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


def bh_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    out = np.full_like(pvals, np.nan)
    if not valid.any():
        return out
    p_valid = pvals[valid]
    n = len(p_valid)
    order = np.argsort(p_valid)
    q_ranked = p_valid[order] * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    out[valid] = q
    return out


def per_roi_metrics(roi_df, min_cells, max_unassigned, paper_9):
    """Per-ROI compartment count + size restricted to the paper's 9 compartments.

    Cells with cell_type in T_UNASSIGNED are dropped from the denominator.
    Cells in compartments outside `paper_9` are also dropped.
    """
    typed_df = roi_df[~roi_df["cell_type"].isin(T_UNASSIGNED)]
    n_typed_total = len(typed_df)
    if n_typed_total < min_cells:
        return None
    # QC filter: too many Unassigned suggests bad core
    unassigned_frac = 1.0 - n_typed_total / len(roi_df)
    if unassigned_frac > max_unassigned:
        return None
    # Restrict to paper-9 compartments
    in9 = typed_df[typed_df["compartment"].isin(paper_9)]
    n_in_9 = len(in9)
    if n_in_9 < 100:  # need some signal
        return None
    comp_counts = in9["compartment"].value_counts()
    fracs = comp_counts / n_in_9

    out = {"n_typed_total": n_typed_total, "n_in_9": n_in_9,
           "unassigned_frac": float(unassigned_frac)}
    for thr in (0.02, 0.05, 0.10):
        out[f"n_compartments_present_p{int(thr*100):02d}"] = int((fracs >= thr).sum())
    present = comp_counts[fracs >= 0.05]
    out["mean_compartment_size_cells"] = float(present.mean()) if len(present) else np.nan
    out["mean_compartment_size_frac"] = float((present / n_in_9).mean()) if len(present) else np.nan
    out["median_compartment_size_cells"] = float(present.median()) if len(present) else np.nan
    p = fracs.values
    out["shannon_compartment"] = float(-(p * np.log2(p + 1e-12)).sum())

    for c in paper_9:
        out[f"frac_{c}"] = float(fracs.get(c, 0.0))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t-panel", default="output/all_TMA_T_utag_ct_merged.h5ad")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--out", default="output/grade_arch/t_panel_paper9")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
    p.add_argument("--max-unassigned-frac", type=float, default=MAX_UNASSIGNED_FRAC)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.t_panel} ...")
    with h5py.File(args.t_panel, "r") as f:
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

    # Sanity: paper-9 set must exist in this h5ad
    obs_compart = set(df["compartment"].unique())
    missing = PAPER_9 - obs_compart
    if missing:
        raise RuntimeError(f"PAPER_9 compartments missing from h5ad: {missing}")
    n_excluded_compart = len(obs_compart - PAPER_9)
    print(f"  paper-9 verified; {n_excluded_compart} compartments excluded "
          f"(LQ-like / Activated B / Unidentified)")

    rows = []
    n_passing_min_cells = 0
    n_dropped_unassigned = 0
    for sid, sub in df.groupby("sample_id"):
        # Track cohort flow for sanity reporting
        not_un = ~sub["cell_type"].isin(T_UNASSIGNED)
        if int(not_un.sum()) >= args.min_cells:
            n_passing_min_cells += 1
            unfrac = 1.0 - int(not_un.sum()) / len(sub)
            if unfrac > args.max_unassigned_frac:
                n_dropped_unassigned += 1
        m = per_roi_metrics(sub, args.min_cells, args.max_unassigned_frac, PAPER_9)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows)
    print(f"  ROIs passing min_cells={args.min_cells}: {n_passing_min_cells}")
    print(f"  ROIs additionally dropped by Unassigned >{args.max_unassigned_frac}: "
          f"{n_dropped_unassigned}")
    print(f"  ROIs in final analysis (paper-9 ≥100 cells): {len(metrics_df)}")

    # Grade sourced from DWS clinical (native GRADE col). Legacy --grade xlsx
    # arg accepted for back-compat but ignored.
    import warnings as _warn
    from src.clinical_linkage import load_clinical
    with _warn.catch_warnings():
        _warn.simplefilter("ignore")
        clin = load_clinical()
    clin = clin[["slide_ID", "Sample_ID", "Patient_ID", "GRADE"]].rename(
        columns={"GRADE": "grade"}
    )
    metrics_df = metrics_df.merge(clin, left_on="sample_id", right_on="slide_ID", how="left")
    metrics_df = metrics_df[metrics_df.grade.isin(GRADE_ORDER)].copy()

    metric_cols = [c for c in metrics_df.columns
                   if c not in {"sample_id", "slide_ID", "Sample_ID",
                                "Patient_ID", "grade"}]
    pt = (metrics_df.groupby(["Patient_ID", "grade"])[metric_cols]
          .mean().reset_index())
    pt.to_csv(out_dir / "grade_compartments_t_paper9_per_patient.csv", index=False)
    print(f"  Patient-level n: {len(pt)} ({pt.grade.value_counts().to_dict()})")

    family_a = [
        ("n_compartments_present_p02",   "Compartments present (≥2%)"),
        ("n_compartments_present_p05",   "Compartments present (≥5%)"),
        ("n_compartments_present_p10",   "Compartments present (≥10%)"),
        ("shannon_compartment",          "Compartment Shannon entropy"),
        ("mean_compartment_size_cells",  "Mean compartment size (cells)"),
        ("mean_compartment_size_frac",   "Mean compartment size (frac)"),
        ("median_compartment_size_cells", "Median compartment size (cells)"),
    ]
    print("\n=== Family A (paper-9 scheme, BH within family) ===")
    pvals_a, medians_a = [], []
    for k, _ in family_a:
        p_v, med = kw(pt, k)
        pvals_a.append(p_v); medians_a.append(med)
    qvals_a = bh_correct(pvals_a)
    print(f"{'metric':40s} {'p':>10s} {'q (BH)':>10s}  medians (FOLL1/2/3A)")
    fam_a_rows = []
    for (k, label), p_v, q_v, med in zip(family_a, pvals_a, qvals_a, medians_a):
        med_str = " / ".join(f"{med[g]:.4g}" for g in GRADE_ORDER)
        flag = " *" if q_v < 0.05 else ""
        print(f"  {label:40s} {p_v:10.4g} {q_v:10.4g}  {med_str}{flag}")
        fam_a_rows.append({"metric": k, "label": label, "p_KW": p_v, "q_BH": q_v,
                           **{f"med_{g}": med[g] for g in GRADE_ORDER}})
    pd.DataFrame(fam_a_rows).to_csv(out_dir / "t_paper9_family_a.csv", index=False)

    comp_cols = [c for c in pt.columns if c.startswith("frac_")]
    print("\n=== Family B: per-compartment fraction (paper-9, BH within family) ===")
    pvals_b, medians_b = [], []
    for c in comp_cols:
        p_v, med = kw(pt, c)
        pvals_b.append(p_v); medians_b.append(med)
    qvals_b = bh_correct(pvals_b)
    print(f"{'compartment':45s} {'p':>10s} {'q (BH)':>10s}  medians (FOLL1/2/3A)")
    fam_b_rows = []
    for c, p_v, q_v, med in zip(comp_cols, pvals_b, qvals_b, medians_b):
        name = c.replace("frac_", "")
        med_str = " / ".join(f"{med[g]:.4g}" for g in GRADE_ORDER)
        flag = " *" if q_v < 0.05 else ""
        print(f"  {name:45s} {p_v:10.4g} {q_v:10.4g}  {med_str}{flag}")
        fam_b_rows.append({"compartment": name, "p_KW": p_v, "q_BH": q_v,
                           **{f"med_{g}": med[g] for g in GRADE_ORDER}})
    pd.DataFrame(fam_b_rows).to_csv(out_dir / "t_paper9_per_compartment.csv", index=False)

    # ── Figure ──
    rng = np.random.default_rng(0)

    def boxplot(ax, metric, label, q=None):
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
            tag = ""
            if q is not None and not np.isnan(q):
                tag = f", q={q:.3g}" + (" *" if q < 0.05 else "")
            ax.set_title(f"{label}\np={pv:.3g}{tag}", fontsize=11)
        else:
            ax.set_title(label, fontsize=11)
        ax.set_xlabel("Grade"); ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    boxplot(axes[0, 0], "n_compartments_present_p02",
            "Compartments present (≥2%)", q=qvals_a[0])
    boxplot(axes[0, 1], "n_compartments_present_p05",
            "Compartments present (≥5%)", q=qvals_a[1])
    boxplot(axes[0, 2], "n_compartments_present_p10",
            "Compartments present (≥10%)", q=qvals_a[2])
    boxplot(axes[0, 3], "shannon_compartment",
            "Compartment Shannon entropy", q=qvals_a[3])
    boxplot(axes[1, 0], "mean_compartment_size_cells",
            "Mean compartment size (cells)", q=qvals_a[4])
    boxplot(axes[1, 1], "mean_compartment_size_frac",
            "Mean compartment size (frac)", q=qvals_a[5])
    boxplot(axes[1, 2], "median_compartment_size_cells",
            "Median compartment size (cells)", q=qvals_a[6])
    # Stacked bar
    ax_stack = axes[1, 3]
    medians_per_comp = (pt.groupby("grade")[comp_cols]
                        .median().reindex(GRADE_ORDER))
    medians_per_comp = medians_per_comp.rename(
        columns={c: c.replace("frac_", "") for c in comp_cols})
    cmap = plt.cm.tab10
    color_list = [cmap(i) for i in range(len(medians_per_comp.columns))]
    bottom = np.zeros(len(GRADE_ORDER))
    for col, color in zip(medians_per_comp.columns, color_list):
        vals = medians_per_comp[col].values
        ax_stack.bar(GRADE_ORDER, vals, bottom=bottom, color=color,
                     label=col, edgecolor="white", linewidth=0.4)
        bottom += vals
    ax_stack.set_ylabel("Median fraction (per grade)")
    ax_stack.set_xlabel("Grade")
    ax_stack.set_title("Composition (paper-9)", fontsize=11)
    ax_stack.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for sp in ("top", "right"):
        ax_stack.spines[sp].set_visible(False)

    fig.suptitle(f"T-panel paper-9 compartment analysis vs grade "
                 f"(n={len(pt)} patients; FOLL1={sum(pt.grade=='FOLL1')}, "
                 f"FOLL2={sum(pt.grade=='FOLL2')}, FOLL3A={sum(pt.grade=='FOLL3A')})",
                 fontsize=12, y=1.0)
    plt.tight_layout()
    out = out_dir / "fig_grade_compartments_t_paper9.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
