#!/usr/bin/env python3
"""Is the immune exclusion + exhaustion gradient grade-dependent?

For each ROI, compute per-compartment metrics restricted to the paper's
9-compartment scheme:
  - cd8_density        : CD8 T (any) as fraction of typed cells in compartment
  - cd8_exh_fraction   : exhausted CD8 / total CD8 in compartment

ROI-level gradient metrics (using paper's "follicular core" and "T zone"
endpoints):
  - exclusion_gradient   = cd8_density(T cell zone) - cd8_density(GC core)
                            -- larger = stronger exclusion
  - exhaustion_gradient  = cd8_exh_fraction(GC core) - cd8_exh_fraction(T cell zone)
                            -- larger = stronger exhaustion topography

Stratify by grade. If the gradient shrinks with grade, "follicle as immune
sanctuary" weakens.

Aggregation: patient-level mean across each patient's ROIs. KW + within-
compartment BH for the per-compartment scan; raw p for the gradient metrics
(only 2 tests, no need for correction).
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
MAX_UNASSIGNED_FRAC = 0.40
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}
T_UNASSIGNED = ["Unassigned", "Low quality / Unassigned"]

# Paper's 9-compartment gradient order (immune_evasion.GRADIENT_ORDER)
GRADIENT_ORDER = [
    "GC core",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)",
    "B cell follicle (CD20hi/CXCR5hi)",
    "B cell zone",
    "Follicle-T zone interface",
    "Treg-enriched T zone",
    "T cell zone (CD4/CD8)",
    "Macrophage-rich zone",
]
GRADIENT_SHORT = ["GC", "Foll core", "Mantle", "B foll", "B zone",
                  "Foll-T", "Treg T", "T zone", "Mac zone"]

CD8_ALL = ["CD8 T cells", "CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]
CD8_EXHAUSTED = ["CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]
MIN_COMPARTMENT_TYPED = 100        # need ≥100 typed cells in compartment
MIN_CD8_FOR_EXH_FRAC = 10           # need ≥10 CD8 cells to compute exhaustion fraction


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


def per_roi_metrics(roi_df, min_cells, max_unassigned):
    typed_df = roi_df[~roi_df["cell_type"].isin(T_UNASSIGNED)]
    n_typed_total = len(typed_df)
    if n_typed_total < min_cells:
        return None
    if (1.0 - n_typed_total / len(roi_df)) > max_unassigned:
        return None

    out = {"n_typed_total": n_typed_total}
    for comp in GRADIENT_ORDER:
        sub = typed_df[typed_df["compartment"] == comp]
        if len(sub) < MIN_COMPARTMENT_TYPED:
            out[f"cd8_density_{comp}"] = np.nan
            out[f"cd8_exh_frac_{comp}"] = np.nan
            continue
        n_cd8 = int(sub["cell_type"].isin(CD8_ALL).sum())
        n_exh = int(sub["cell_type"].isin(CD8_EXHAUSTED).sum())
        out[f"cd8_density_{comp}"] = float(n_cd8 / len(sub))
        out[f"cd8_exh_frac_{comp}"] = (float(n_exh / n_cd8)
                                       if n_cd8 >= MIN_CD8_FOR_EXH_FRAC else np.nan)

    # Gradient endpoints — computed at the per-ROI level. NOTE: patients with
    # ROIs lacking the endpoint compartments will have NaN in their per-ROI
    # gradient and contribute NaN; but the patient-level reconstruction below
    # ALSO computes the gradient from patient-mean per-compartment densities
    # (more robust to per-ROI dropout). We keep both for transparency.
    gc_dens = out.get("cd8_density_GC core", np.nan)
    tz_dens = out.get("cd8_density_T cell zone (CD4/CD8)", np.nan)
    out["exclusion_gradient_per_roi"] = (
        float(tz_dens - gc_dens) if not (np.isnan(gc_dens) or np.isnan(tz_dens)) else np.nan
    )
    gc_exh = out.get("cd8_exh_frac_GC core", np.nan)
    tz_exh = out.get("cd8_exh_frac_T cell zone (CD4/CD8)", np.nan)
    out["exhaustion_gradient_per_roi"] = (
        float(gc_exh - tz_exh) if not (np.isnan(gc_exh) or np.isnan(tz_exh)) else np.nan
    )
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

    # Sanity: gradient compartments must exist
    obs_compart = set(df["compartment"].unique())
    missing = set(GRADIENT_ORDER) - obs_compart
    if missing:
        raise RuntimeError(f"Paper-9 compartments missing from h5ad: {missing}")

    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_metrics(sub, args.min_cells, args.max_unassigned_frac)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows)
    print(f"  ROIs in analysis: {len(metrics_df)}")

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

    # Reconstruct gradient at PATIENT level: works even if no single ROI had
    # both endpoints, as long as some ROIs had GC core and some had T cell zone
    # (their patient-level means are then both non-NaN).
    gc_d = pt["cd8_density_GC core"]
    tz_d = pt["cd8_density_T cell zone (CD4/CD8)"]
    gc_e = pt["cd8_exh_frac_GC core"]
    tz_e = pt["cd8_exh_frac_T cell zone (CD4/CD8)"]
    pt["exclusion_gradient"] = tz_d - gc_d
    pt["exhaustion_gradient"] = gc_e - tz_e

    pt.to_csv(out_dir / "grade_exclusion_exhaustion_per_patient.csv", index=False)
    print(f"  Patient-level n: {len(pt)} ({pt.grade.value_counts().to_dict()})")
    print(f"  Patients with non-NaN exclusion_gradient: {pt.exclusion_gradient.notna().sum()}")
    print(f"  Patients with non-NaN exhaustion_gradient: {pt.exhaustion_gradient.notna().sum()}")

    # ── Gradient metrics (the headline test) ──
    print("\n=== Gradient metrics by grade (KW) ===")
    summary_rows = []
    for m, label in [
        ("exclusion_gradient",
         "CD8 density gradient (T zone − GC core; larger = stronger exclusion)"),
        ("exhaustion_gradient",
         "Exhaustion gradient (GC core − T zone; larger = stronger topography)"),
    ]:
        p_v, med = kw(pt, m)
        med_str = " / ".join(f"{med[g]:.4f}" for g in GRADE_ORDER)
        print(f"  {label:75s} p={p_v:.4g}  medians={med_str}")
        summary_rows.append({"metric": m, "label": label, "p_KW": p_v,
                              **{f"med_{g}": med[g] for g in GRADE_ORDER}})

    # ── Per-compartment metric scan (BH within metric type) ──
    cd8_density_cols = [f"cd8_density_{c}" for c in GRADIENT_ORDER]
    cd8_exh_cols = [f"cd8_exh_frac_{c}" for c in GRADIENT_ORDER]

    def run_scan(cols, label_prefix):
        ps, meds = [], []
        for c in cols:
            p_v, med = kw(pt, c)
            ps.append(p_v); meds.append(med)
        qs = bh_correct(ps)
        print(f"\n=== {label_prefix} per compartment (BH within {len(cols)} metrics) ===")
        print(f"{'compartment':40s} {'p':>10s} {'q (BH)':>10s}  medians (FOLL1/2/3A)")
        rows_local = []
        for c, p_v, q_v, med in zip(cols, ps, qs, meds):
            comp = c.replace(f"{label_prefix.lower().replace(' ','_')}_", "")
            comp_short = c.replace(f"cd8_density_", "").replace(f"cd8_exh_frac_", "")
            med_str = " / ".join(f"{med[g]:.4f}" for g in GRADE_ORDER)
            flag = " *" if q_v < 0.05 else ""
            print(f"  {comp_short:40s} {p_v:10.4g} {q_v:10.4g}  {med_str}{flag}")
            rows_local.append({"compartment": comp_short, "metric": label_prefix,
                                "p_KW": p_v, "q_BH": q_v,
                                **{f"med_{g}": med[g] for g in GRADE_ORDER}})
        return rows_local

    dens_rows = run_scan(cd8_density_cols, "cd8_density")
    exh_rows = run_scan(cd8_exh_cols, "cd8_exh_frac")
    pd.DataFrame(summary_rows + dens_rows + exh_rows).to_csv(
        out_dir / "grade_exclusion_exhaustion_summary.csv", index=False)

    # ── Figure: 4 panels ──
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
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Top row: gradient metrics
    boxplot(axes[0, 0], "exclusion_gradient",
            "CD8 density gradient\n(T zone − GC core; bigger = stronger exclusion)")
    boxplot(axes[0, 1], "exhaustion_gradient",
            "Exhaustion gradient\n(GC core − T zone; bigger = stronger topography)")

    # Bottom row: line plots — per-grade gradient curves across compartments
    for ax, cols, ylabel, title in [
        (axes[1, 0], cd8_density_cols, "CD8 density (frac of compartment)",
         "CD8 density along compartment gradient"),
        (axes[1, 1], cd8_exh_cols, "Exhaustion frac (exh CD8 / total CD8)",
         "Exhaustion frac along compartment gradient"),
    ]:
        for g in GRADE_ORDER:
            sub = pt[pt.grade == g][cols]
            med = sub.median(axis=0).values
            ax.plot(GRADIENT_SHORT, med, "-o", color=GRADE_COLORS[g],
                    label=f"{g} (n={len(sub)})", lw=2, ms=6)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Compartment (follicle → T zone)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, loc="best")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle(f"Is the immune exclusion + exhaustion gradient grade-dependent? "
                 f"(T-panel paper-9, n={len(pt)} patients; "
                 f"FOLL1={sum(pt.grade=='FOLL1')}, FOLL2={sum(pt.grade=='FOLL2')}, "
                 f"FOLL3A={sum(pt.grade=='FOLL3A')})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_grade_exclusion_exhaustion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
