#!/usr/bin/env python3
"""Within-T-panel-compartment cell-type composition shifts by FL grade.

Mirrors the S-panel approach that found "B cells (PAX5+) flooding the FDC
network zone with grade" but on the T-panel UTAG h5ad. For each (compartment,
cell_type) pair, compute per-ROI fraction of that cell type WITHIN that
compartment, then patient-aggregate and KW by grade.

Filtering:
  - Only retain pairs present in ≥10 patients with ≥10 cells in compartment
    AND ≥1 cell of that cell-type in compartment.
  - BH correction across the surviving pair family.

Output: heatmap of monotonic effect sizes + box plots of top hits.
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
MIN_COMPARTMENT_CELLS = 100      # need at least this many cells in compartment per ROI
MIN_CELL_COUNT_TYPE = 1          # at least 1 cell of the type in the compartment
MIN_PATIENTS_PER_PAIR = 10       # need at least this many patients with the metric
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}
T_UNASSIGNED = ["Unassigned", "Low quality / Unassigned"]


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


def per_roi_within_compartment(roi_df, min_cells, min_compart_cells):
    """For each compartment present in this ROI, return per-typed-cell fractions.

    Unassigned cells are excluded from BOTH numerator and denominator (CLAUDE.md
    guardrail #8). Compartments need ≥`min_compart_cells` typed cells.
    """
    typed_df = roi_df[~roi_df["cell_type"].isin(T_UNASSIGNED)]
    n_typed = len(typed_df)
    if n_typed < min_cells:
        return None
    out = {"n_typed": n_typed}
    for comp, sub in typed_df.groupby("compartment"):
        if len(sub) < min_compart_cells:
            continue
        ct_frac = sub["cell_type"].value_counts(normalize=True)
        for ct, frac in ct_frac.items():
            key = f"in___{comp}___{ct}"  # triple-underscore separator
            out[key] = float(frac)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t-panel", default="output/all_TMA_T_utag_ct_merged.h5ad")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--out", default="output/grade_arch/t_panel")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
    p.add_argument("--min-compart-cells", type=int, default=MIN_COMPARTMENT_CELLS)
    p.add_argument("--min-patients", type=int, default=MIN_PATIENTS_PER_PAIR)
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

    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_within_compartment(sub, args.min_cells, args.min_compart_cells)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows).fillna(0.0)  # absent (compartment, ct) → 0
    print(f"  ROIs after min_cells={args.min_cells}: {len(metrics_df)}")

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

    metric_cols = [c for c in metrics_df.columns if c.startswith("in___")]
    pt = (metrics_df.groupby(["Patient_ID", "grade"])[metric_cols]
          .mean().reset_index())
    print(f"  Patient-level n: {len(pt)}  ({pt.grade.value_counts().to_dict()})")

    # Filter pairs: ≥min_patients non-zero AND each grade has ≥1 patient with the
    # COMPARTMENT present (not necessarily this cell-type — otherwise grade-specific
    # appearances are excluded). Compartment presence: any pt[<...same compartment...>] > 0.
    pair_compartment = {c: c.split("___")[1] for c in metric_cols}
    pt_present_per_grade = {}
    for c in metric_cols:
        comp = pair_compartment[c]
        comp_cols = [m for m in metric_cols if pair_compartment[m] == comp]
        # patient has compartment present if any cell-type in that compartment > 0
        comp_present = (pt[comp_cols] > 0).any(axis=1)
        pt_present_per_grade[c] = all(((pt.grade == g) & comp_present).any()
                                       for g in GRADE_ORDER)
    keep_cols = []
    for c in metric_cols:
        n_nonzero = int((pt[c] > 0).sum())
        if n_nonzero >= args.min_patients and pt_present_per_grade[c]:
            keep_cols.append(c)
    print(f"  Pairs surviving filter: {len(keep_cols)} / {len(metric_cols)}")

    # KW + BH
    pvals, medians = [], []
    for c in keep_cols:
        p_v, med = kw(pt, c)
        pvals.append(p_v); medians.append(med)
    qvals = bh_correct(pvals)

    # Sort by p
    order = np.argsort(pvals)
    print(f"\n=== Top within-compartment shifts (T-panel, n={len(pt)} patients) ===")
    print(f"{'(compartment) (cell type)':75s} {'p':>10s} {'q (BH)':>10s}  medians (FOLL1/2/3A)")
    rows_summary = []
    for i in order:
        c = keep_cols[i]
        p_v = pvals[i]; q_v = qvals[i]
        compartment, ct = c.replace("in___", "").split("___", 1)
        med = medians[i]
        med_str = " / ".join(f"{med[g]:.3g}" for g in GRADE_ORDER)
        flag = " *" if q_v < 0.05 else ""
        if p_v < 0.10 or q_v < 0.20:
            print(f"  ({compartment:30s}) ({ct:25s}) {p_v:10.4g} {q_v:10.4g}  {med_str}{flag}")
        rows_summary.append({"compartment": compartment, "cell_type": ct,
                              "p_KW": p_v, "q_BH": q_v,
                              **{f"med_{g}": med[g] for g in GRADE_ORDER}})
    summary_df = pd.DataFrame(rows_summary).sort_values("p_KW")
    summary_df.to_csv(out_dir / "t_grade_within_compartment.csv", index=False)

    # Plot top hits
    top_hits = summary_df[summary_df.p_KW < 0.05].head(8)
    if len(top_hits) == 0:
        top_hits = summary_df.head(6)
    n_panels = max(len(top_hits), 1)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.0 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    rng = np.random.default_rng(0)

    for ax, (_, row) in zip(axes, top_hits.iterrows()):
        col = f"in___{row.compartment}___{row.cell_type}"
        data = [pt.loc[pt.grade == g, col].dropna().values for g in GRADE_ORDER]
        bp = ax.boxplot(data, tick_labels=GRADE_ORDER, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GRADE_ORDER):
            patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.55)
        for i, (g, vals) in enumerate(zip(GRADE_ORDER, data)):
            xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(xs, vals, color=GRADE_COLORS[g], s=12, alpha=0.7,
                       edgecolor="white", linewidth=0.4, zorder=3)
        flag = " *" if row.q_BH < 0.05 else ""
        ax.set_title(f"{row.cell_type}\nin {row.compartment}\n"
                     f"p={row.p_KW:.3g}, q={row.q_BH:.3g}{flag}",
                     fontsize=9)
        ax.set_xlabel("Grade"); ax.set_ylabel("Fraction in compartment")
        ax.tick_params(labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[len(top_hits):]:
        ax.axis("off")

    n_sig = int((summary_df.q_BH < 0.05).sum())
    title = (f"T-panel within-compartment composition shifts vs FL grade "
             f"(n={len(pt)} patients)")
    if n_sig == 0:
        title += "  —  EXPLORATORY: no q<0.05 hits, top 6 by raw p shown"
    fig.suptitle(title, fontsize=12, y=1.0)
    plt.tight_layout()
    out = out_dir / "fig_grade_within_compartment_t.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
