#!/usr/bin/env python3
"""T-panel compartment count + size + per-compartment fraction by FL grade.

Mirrors the S-panel analyses (`grade_compartment_biomarkers.py`,
`grade_compartment_size.py`, per-compartment scan) on the T-panel UTAG h5ad
which uses a different compartment vocabulary (14 compartments, finer
follicular biology).

Outputs to `output/grade_arch/t_panel/`.
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
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}
T_PANEL_UNASSIGNED_LABELS = ["Unassigned", "Low quality / Unassigned"]


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


def per_roi_metrics(roi_df, min_cells, present_thresholds=(0.02, 0.05, 0.10)):
    not_unassigned = ~roi_df["cell_type"].isin(T_PANEL_UNASSIGNED_LABELS)
    n_typed = int(not_unassigned.sum())
    if n_typed < min_cells:
        return None
    n_total = len(roi_df)
    comp_counts = roi_df["compartment"].value_counts()
    fracs = comp_counts / n_total

    out = {"n_typed": n_typed, "n_total": n_total}
    for thr in present_thresholds:
        present_mask = fracs >= thr
        out[f"n_compartments_present_p{int(thr*100):02d}"] = int(present_mask.sum())
    # Compartment-size metrics at the 5% threshold (canonical)
    present = comp_counts[fracs >= 0.05]
    if len(present) > 0:
        out["mean_compartment_size_cells"] = float(present.mean())
        out["mean_compartment_size_frac"] = float((present / n_total).mean())
        out["median_compartment_size_cells"] = float(present.median())
    else:
        out["mean_compartment_size_cells"] = np.nan
        out["mean_compartment_size_frac"] = np.nan
        out["median_compartment_size_cells"] = np.nan
    # Compartment Shannon
    p = fracs.values
    out["shannon_compartment"] = float(-(p * np.log2(p + 1e-12)).sum())
    # Per-compartment fractions
    for c in fracs.index:
        out[f"frac_{c}"] = float(fracs[c])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t-panel", default="output/all_TMA_T_utag_ct_merged.h5ad")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--out", default="output/grade_arch/t_panel")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
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
    n_compart = df.compartment.nunique()
    print(f"  Compartments observed: {n_compart} unique values")

    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_metrics(sub, args.min_cells)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows)
    # Only fill NaN on per-compartment frac columns (NaN = compartment absent in
    # that ROI). Leave size metrics as NaN so kw()'s dropna excludes ROIs
    # where no compartment passed the 5% gate.
    frac_cols = [c for c in metrics_df.columns if c.startswith("frac_")]
    metrics_df[frac_cols] = metrics_df[frac_cols].fillna(0.0)
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

    metric_cols = [c for c in metrics_df.columns
                   if c not in {"sample_id", "slide_ID", "Sample_ID",
                                "Patient_ID", "grade"}]
    pt = (metrics_df.groupby(["Patient_ID", "grade"])[metric_cols]
          .mean().reset_index())
    pt.to_csv(out_dir / "grade_compartments_t_per_patient.csv", index=False)
    print(f"  Patient-level n: {len(pt)}  ({pt.grade.value_counts().to_dict()})")

    # ===== Family A: compartment count + size + diversity =====
    family_a = [
        ("n_compartments_present_p02",  "Compartments present (≥2%)"),
        ("n_compartments_present_p05",  "Compartments present (≥5%)"),
        ("n_compartments_present_p10",  "Compartments present (≥10%)"),
        ("shannon_compartment",         "Compartment Shannon entropy"),
        ("mean_compartment_size_cells", "Mean compartment size (cells)"),
        ("mean_compartment_size_frac",  "Mean compartment size (frac of ROI)"),
        ("median_compartment_size_cells", "Median compartment size (cells)"),
    ]
    print("\n=== Family A: count/size/diversity (BH within family) ===")
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
    pd.DataFrame(fam_a_rows).to_csv(out_dir / "t_grade_family_a.csv", index=False)

    # ===== Family B: per-compartment fraction =====
    comp_cols = [c for c in pt.columns if c.startswith("frac_")]
    print("\n=== Family B: per-compartment fraction (BH within family) ===")
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
        if p_v < 0.10 or q_v < 0.20:
            print(f"  {name:45s} {p_v:10.4g} {q_v:10.4g}  {med_str}{flag}")
        fam_b_rows.append({"compartment": name, "p_KW": p_v, "q_BH": q_v,
                           **{f"med_{g}": med[g] for g in GRADE_ORDER}})
    pd.DataFrame(fam_b_rows).to_csv(out_dir / "t_grade_per_compartment.csv", index=False)

    # ===== Figure: 4-panel headline + per-compartment heatmap =====
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
            tag = f", q={q:.3g}" + (" *" if q is not None and q < 0.05 else "") \
                  if q is not None and not np.isnan(q) else ""
            ax.set_title(f"{label}\np={pv:.3g}{tag}", fontsize=10)
        else:
            ax.set_title(label, fontsize=10)
        ax.set_xlabel("Grade")
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

    boxplot(fig.add_subplot(gs[0, 0]), "n_compartments_present_p02",
            "Compartments present (≥2%)", q=qvals_a[0])
    boxplot(fig.add_subplot(gs[0, 1]), "n_compartments_present_p05",
            "Compartments present (≥5%)", q=qvals_a[1])
    boxplot(fig.add_subplot(gs[0, 2]), "n_compartments_present_p10",
            "Compartments present (≥10%)", q=qvals_a[2])
    boxplot(fig.add_subplot(gs[0, 3]), "shannon_compartment",
            "Compartment Shannon entropy", q=qvals_a[3])

    boxplot(fig.add_subplot(gs[1, 0]), "mean_compartment_size_cells",
            "Mean compartment size (cells)", q=qvals_a[4])
    boxplot(fig.add_subplot(gs[1, 1]), "mean_compartment_size_frac",
            "Mean compartment size (frac)", q=qvals_a[5])

    # Stacked bar of median per-compartment fractions by grade
    ax_stack = fig.add_subplot(gs[1, 2:])
    medians_per_comp = (pt.groupby("grade")[comp_cols]
                        .median().reindex(GRADE_ORDER))
    medians_per_comp = medians_per_comp.rename(
        columns={c: c.replace("frac_", "") for c in comp_cols})
    cmap = plt.cm.tab20
    color_list = [cmap(i % 20) for i in range(len(medians_per_comp.columns))]
    bottom = np.zeros(len(GRADE_ORDER))
    for col, color in zip(medians_per_comp.columns, color_list):
        vals = medians_per_comp[col].values
        ax_stack.bar(GRADE_ORDER, vals, bottom=bottom, color=color,
                     label=col, edgecolor="white", linewidth=0.4)
        bottom += vals
    ax_stack.set_ylabel("Median fraction (per grade)")
    ax_stack.set_xlabel("Grade")
    ax_stack.set_title("T-panel compartment composition by grade", fontsize=10)
    ax_stack.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for sp in ("top", "right"):
        ax_stack.spines[sp].set_visible(False)

    fig.suptitle(f"T-panel: compartment count + size + composition vs FL grade "
                 f"(n={len(pt)} patients, FOLL1={sum(pt.grade=='FOLL1')}, "
                 f"FOLL2={sum(pt.grade=='FOLL2')}, FOLL3A={sum(pt.grade=='FOLL3A')})",
                 fontsize=12, y=1.0)
    out = out_dir / "fig_grade_compartments_t.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
