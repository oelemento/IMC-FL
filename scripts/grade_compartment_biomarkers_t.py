#!/usr/bin/env python3
"""T-panel analog of grade_compartment_biomarkers.py.

Same 4-panel figure for the T-panel UTAG h5ad:
  Col 1: stacked bar of compartment composition by grade (T-panel compartments)
  Col 2: Compartment Shannon entropy
  Col 3: # Compartments present (>=5% of cells)
  Col 4: best biomarker per-ROI mean (lowest q among the BIOMARKERS list)

Cohort filtering, patient-level aggregation, KW + BH (within-family + combined)
are identical to the S-panel script. Compartment palette and biomarker set are
T-panel specific. CD21 is not in the T-panel (intentional — user request).

Usage:
    .venv/bin/python scripts/grade_compartment_biomarkers_t.py \\
        --t-panel output/all_TMA_T_utag_ct_merged.h5ad \\
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
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

DEFAULT_MIN_CELLS_PER_ROI = 8000
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}

# T-panel biomarkers — no CD21 (per user request) and no FDC markers in T-panel.
# Choose informative T-panel cell-state markers.
BIOMARKERS = ["TOX", "PD_1", "CD8a", "FoxP3", "GranzymeB", "CD68"]

# Stable color palette for T-panel compartments. Group by zone family:
#   B-cell / follicular family -> blues
#   GC family -> oranges/red
#   T-cell family -> greens
#   Macrophage / interface -> browns / purples
#   LQ / unidentified -> grays
COMPART_COLORS = {
    "B cell follicle (CD20hi/CXCR5hi)":  "#aec7e8",
    "Activated B / CXCR5hi zone":        "#1f77b4",
    "Follicle mantle (CXCR5hi)":         "#9edae5",
    "Follicle core (GC/CD20hi/CXCR5hi)": "#ff7f0e",
    "GC core":                           "#d62728",
    "B cell zone":                       "#5b9bd5",
    "Follicle-T zone interface":         "#9467bd",
    "Treg-enriched T zone":              "#bcbd22",
    "T cell zone (CD4/CD8)":             "#2ca02c",
    "Macrophage-rich zone":              "#8c564b",
    "Cytotoxic / LQ niche":              "#c49c94",
    "LQ / B transitional":               "#bbbbbb",
    "Weak CD20 / LQ border":             "#999999",
    "Unidentified zone":                 "#cccccc",
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


def load_data(h5ad_path: Path, biomarkers):
    """Load obs + biomarker columns from `.X` (scaled, z-scored)."""
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

        var_names = [v.decode() if isinstance(v, bytes) else v
                     for v in f["var/_index"][:]]
        biomarker_idx = {b: var_names.index(b) for b in biomarkers if b in var_names}
        missing = [b for b in biomarkers if b not in var_names]
        if missing:
            print(f"  WARN: biomarkers missing from var: {missing}")

        n_obs = len(sample_id)
        X = f["X"]
        biomarker_data = {}
        for b, i in biomarker_idx.items():
            biomarker_data[b] = X[:, i] if X.shape == (n_obs, len(var_names)) else np.array(X[:, i])

    df = pd.DataFrame({
        "sample_id": sample_id, "cell_type": cell_type, "compartment": compartment,
        **biomarker_data,
    })
    return df, list(biomarker_idx.keys())


T_UNASSIGNED = ["Unassigned", "Low quality / Unassigned"]


def per_roi_metrics(roi_df, biomarkers, min_cells, biomarker_p90_thresh=None):
    not_unassigned = ~roi_df["cell_type"].isin(T_UNASSIGNED)
    n_typed = int(not_unassigned.sum())
    if n_typed < min_cells:
        return None

    comp_counts = roi_df["compartment"].value_counts(normalize=True)
    p = comp_counts.values
    shannon = float(-(p * np.log2(p + 1e-12)).sum())
    simpson = float(1.0 - (p**2).sum())

    out = {
        "n_typed": n_typed,
        "shannon_compartment": shannon,
        "simpson_compartment": simpson,
    }
    for thr in (0.02, 0.05, 0.10):
        out[f"n_compartments_present_p{int(thr*100):02d}"] = int((comp_counts >= thr).sum())

    for c in COMPART_COLORS:
        out[f"frac_{c}"] = float(comp_counts.get(c, 0.0))
    for b in biomarkers:
        if b in roi_df.columns:
            out[f"{b}_mean"] = float(roi_df[b].mean())
            if biomarker_p90_thresh is not None and b in biomarker_p90_thresh:
                thr = biomarker_p90_thresh[b]
                out[f"{b}_pct_pos"] = float((roi_df[b] > thr).mean())
    return out


def join_clinical_and_grade(metrics_df, clinical_csv, grade_xlsx):
    clin = pd.read_csv(clinical_csv)[["slide_ID", "Sample_ID", "Patient_ID"]]
    # Grade sourced from DWS clinical (native GRADE col); legacy --grade
    # xlsx arg accepted but ignored.
    import warnings as _warn
    from src.clinical_linkage import load_clinical as _load_clinical
    with _warn.catch_warnings():
        _warn.simplefilter("ignore")
        _dws = _load_clinical()
    grade = _dws[["Sample_ID", "GRADE"]].rename(columns={"GRADE": "grade"})
    out = metrics_df.merge(clin, left_on="sample_id", right_on="slide_ID", how="left")
    out = out.merge(grade, on="Sample_ID", how="left")
    out = out[out["grade"].isin(GRADE_ORDER)].copy()
    return out


def kw(df, metric):
    groups = [df.loc[df["grade"] == g, metric].dropna().values for g in GRADE_ORDER]
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
    ranked = p_valid[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    out[valid] = q
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t-panel", default="output/all_TMA_T_utag_ct_merged.h5ad")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--out", default="output/grade_arch")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.t_panel} ...")
    df, bm_present = load_data(Path(args.t_panel), BIOMARKERS)
    print(f"  cells={len(df):,}, ROIs={df['sample_id'].nunique()}")
    print(f"  Biomarkers loaded: {bm_present}")

    df = df[df["sample_id"].apply(is_tumor_core)]
    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)
    print(f"  after exclusion + normalization: cells={len(df):,}, ROIs={df['sample_id'].nunique()}")

    obs_comps = set(df["compartment"].unique())
    missing_compart = obs_comps - set(COMPART_COLORS)
    if missing_compart:
        raise RuntimeError(
            f"COMPART_COLORS missing keys for {missing_compart}; the stacked-bar "
            "renormalization would silently drop these. Add them before plotting."
        )

    bm_p90 = {b: float(np.quantile(df[b].dropna().values, 0.90)) for b in bm_present}
    print(f"  biomarker p90 thresholds (cohort-wide): "
          + ", ".join(f"{b}={t:.2f}" for b, t in bm_p90.items()))

    rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_metrics(sub, bm_present, args.min_cells, biomarker_p90_thresh=bm_p90)
        if m is not None:
            rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows)

    metrics_df = join_clinical_and_grade(metrics_df, Path(args.clinical), Path(args.grade))
    print(f"  ROIs with grade: {len(metrics_df)}")
    print(metrics_df["grade"].value_counts().to_string())

    metric_cols = [c for c in metrics_df.columns
                   if c not in {"sample_id", "slide_ID", "Sample_ID", "Patient_ID", "grade"}]
    agg = (metrics_df.groupby(["Patient_ID", "grade"])[metric_cols]
           .mean().reset_index())
    print(f"  Patient-level n: {len(agg)} (FOLL1={sum(agg.grade=='FOLL1')}, "
          f"FOLL2={sum(agg.grade=='FOLL2')}, FOLL3A={sum(agg.grade=='FOLL3A')})")

    agg.to_csv(out_dir / "grade_compartment_biomarkers_t_per_patient.csv", index=False)

    compart_metrics = (
        ["shannon_compartment", "simpson_compartment"]
        + [f"n_compartments_present_p{int(t*100):02d}" for t in (0.02, 0.05, 0.10)]
    )
    biomarker_metrics = ([f"{b}_mean" for b in bm_present]
                         + [f"{b}_pct_pos" for b in bm_present])
    test_metrics = compart_metrics + biomarker_metrics
    pvals, medians = [], []
    for m in test_metrics:
        p_val, med = kw(agg, m)
        pvals.append(p_val); medians.append(med)
    pvals = np.array(pvals)

    q_compart = bh_correct(pvals[:len(compart_metrics)])
    q_biomark = bh_correct(pvals[len(compart_metrics):])
    q_within = np.concatenate([q_compart, q_biomark])
    q_combined = bh_correct(pvals)

    print("\n=== KW + BH(q) — patient-level (n={})".format(len(agg)))
    print(f"{'metric':38s} {'p':>10s} {'q_within':>10s} {'q_combined':>11s}  medians (FOLL1/2/3A)")
    summary = []
    for m, p_val, q_w, q_c, med in zip(test_metrics, pvals, q_within, q_combined, medians):
        med_str = " / ".join(f"{med[g]:.3g}" for g in GRADE_ORDER)
        flag = " *" if (not np.isnan(q_w) and q_w < 0.05) else ""
        print(f"  {m:38s} {p_val:10.4g} {q_w:10.4g} {q_c:11.4g}  {med_str}{flag}")
        summary.append({"metric": m, "p_KW": p_val,
                        "q_BH_within_family": q_w, "q_BH_combined": q_c,
                        **{f"med_{g}": med[g] for g in GRADE_ORDER}})
    pd.DataFrame(summary).to_csv(out_dir / "grade_compartment_biomarkers_t_kw_summary.csv",
                                  index=False)
    qvals = q_combined

    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 1, 1, 1], wspace=0.4)

    ax = fig.add_subplot(gs[0, 0])
    comp_cols = [c for c in agg.columns if c.startswith("frac_")]
    grade_means = agg.groupby("grade")[comp_cols].mean().reindex(GRADE_ORDER)
    grade_means = grade_means / grade_means.sum(axis=1).values[:, None]
    bottom = np.zeros(len(GRADE_ORDER))
    for c in comp_cols:
        name = c.replace("frac_", "")
        color = COMPART_COLORS.get(name, "#888888")
        vals = grade_means[c].values
        ax.bar(GRADE_ORDER, vals, bottom=bottom, color=color, label=name,
               edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_ylabel("Mean compartment fraction")
    ax.set_title("Compartment composition by grade (T-panel)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    rng = np.random.default_rng(0)

    def boxplot(ax, metric, label):
        data = [agg.loc[agg.grade == g, metric].dropna().values for g in GRADE_ORDER]
        bp = ax.boxplot(data, tick_labels=GRADE_ORDER, patch_artist=True,
                        widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GRADE_ORDER):
            patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.55)
        for i, (g, vals) in enumerate(zip(GRADE_ORDER, data)):
            xs = i + 1 + (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(xs, vals, color=GRADE_COLORS[g], s=10, alpha=0.7,
                       edgecolor="white", linewidth=0.4, zorder=3)
        if all(len(x) >= 3 for x in data):
            try:
                _, pv = kruskal(*data)
            except ValueError:
                pv = np.nan
            qv = qvals[test_metrics.index(metric)] if metric in test_metrics else np.nan
            tag = " *" if (not np.isnan(qv) and qv < 0.05) else ""
            ax.set_title(f"{label}\np = {pv:.3g}, q = {qv:.3g}{tag}", fontsize=11)
        else:
            ax.set_title(label, fontsize=11)
        ax.set_xlabel("Grade")
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    boxplot(fig.add_subplot(gs[0, 1]), "shannon_compartment", "Compartment Shannon entropy")
    boxplot(fig.add_subplot(gs[0, 2]), "n_compartments_present_p05",
            "Compartments present (≥5% of cells)")

    bm_qs = [(b, qvals[test_metrics.index(f"{b}_mean")]) for b in bm_present]
    bm_qs = [(b, q_) for b, q_ in bm_qs if not np.isnan(q_)]
    if bm_qs:
        best_b = sorted(bm_qs, key=lambda x: x[1])[0][0]
        boxplot(fig.add_subplot(gs[0, 3]), f"{best_b}_mean",
                f"{best_b} per-ROI mean (scaled)")
    else:
        ax = fig.add_subplot(gs[0, 3]); ax.axis("off")

    fig.suptitle(f"FL grade — T-panel compartment diversity & biomarkers (patient-level n={len(agg)})",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    out = out_dir / "fig_grade_compartment_biomarkers_t.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
