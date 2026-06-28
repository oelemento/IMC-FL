#!/usr/bin/env python3
"""Restrict grade-architecture analyses to cores that capture BOTH a
follicular zone and an interfollicular zone.

Motivation: ~0.6 mm TMA cores sometimes punch out a single architectural
region (pure follicle, pure mantle, pure T zone, pure interface). These
"single-zone captures" depress per-ROI compartment diversity regardless of
grade, masking a true grade-architecture signal. Filtering to ROIs with at
least MIN_FRAC of cells in BOTH a follicular AND an interfollicular zone
removes that geometric artifact.

Compartment categorization
--------------------------
S-panel (11 compartments):
  follicular     : B cell zone (BCL2+/PAX5+), FDC network zone, FDC/myeloid zone,
                   Mixed (B cells (PAX 27%)
  interfollicular: T cell zone, Other / myeloid zone, Stromal / CAF zone,
                   Mixed (M2 Macrophag 26%)
  boundary       : B/T mixed zone
  drop           : Unidentified zone

T-panel (14 compartments):
  follicular     : B cell follicle (CD20hi/CXCR5hi), Activated B / CXCR5hi zone,
                   Follicle mantle, Follicle core, GC core, B cell zone
  interfollicular: T cell zone (CD4/CD8), Treg-enriched T zone, Macrophage-rich
                   zone, Cytotoxic / LQ niche
  boundary       : Follicle-T zone interface
  drop           : LQ / B transitional, Weak CD20 / LQ border, Unidentified zone

The filter requires fraction(follicular) ≥ MIN_FRAC AND
fraction(interfollicular) ≥ MIN_FRAC, computed over ALL cells in the ROI.
Boundary compartments do NOT count toward either side. Drop compartments are
excluded from the denominator (so a core that is 50% Unidentified can still
be retained if its remaining 50% includes both sides).

Outputs (per panel):
  CSVs    : per-patient metrics, KW summary, retention table
  Figure  : same 4-panel layout as grade_compartment_biomarkers_t.py /
            grade_compartment_biomarkers.py — stacked composition,
            Shannon, # compartments present (≥5%), best biomarker.

Usage:
    .venv/bin/python scripts/grade_arch_mixed_cores.py --panel s
    .venv/bin/python scripts/grade_arch_mixed_cores.py --panel t
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
DEFAULT_MIN_MIXED_FRAC = 0.10  # require >=10% on both sides
GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}

# Cell-type-level Unassigned sentinels are panel-shared
UNASSIGNED_CT = ["Unassigned", "Low quality / Unassigned"]

# === S-panel ===
S_BIOMARKERS = ["CD14", "S100A9", "VISTA", "IDO", "CD68", "CD21"]
S_COMPART_COLORS = {
    "B cell zone (BCL2+)":      "#1f77b4",
    "B cell zone (PAX5+)":      "#aec7e8",
    "B/T mixed zone":           "#9467bd",
    "FDC / myeloid zone":       "#ff7f0e",
    "FDC network zone":         "#d62728",
    "Mixed (B cells (PAX 27%)": "#bcbd22",
    "Mixed (M2 Macrophag 26%)": "#8c564b",
    "Other / myeloid zone":     "#7f7f7f",
    "Stromal / CAF zone":       "#e377c2",
    "T cell zone":              "#2ca02c",
    "Unidentified zone":        "#cccccc",
}
S_FOLLICULAR = {
    "B cell zone (BCL2+)",
    "B cell zone (PAX5+)",
    "FDC network zone",
    "FDC / myeloid zone",
    "Mixed (B cells (PAX 27%)",
}
S_INTERFOLLICULAR = {
    "T cell zone",
    "Other / myeloid zone",
    "Stromal / CAF zone",
    "Mixed (M2 Macrophag 26%)",
}
S_BOUNDARY = {"B/T mixed zone"}
S_DROP = {"Unidentified zone"}

# === T-panel ===
T_BIOMARKERS = ["TOX", "PD_1", "CD8a", "FoxP3", "GranzymeB", "CD68"]
T_COMPART_COLORS = {
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
T_FOLLICULAR = {
    "B cell follicle (CD20hi/CXCR5hi)",
    "Activated B / CXCR5hi zone",
    "Follicle mantle (CXCR5hi)",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "GC core",
    "B cell zone",
}
T_INTERFOLLICULAR = {
    "T cell zone (CD4/CD8)",
    "Treg-enriched T zone",
    "Macrophage-rich zone",
    "Cytotoxic / LQ niche",
}
T_BOUNDARY = {"Follicle-T zone interface"}
T_DROP = {"LQ / B transitional", "Weak CD20 / LQ border", "Unidentified zone"}


def panel_config(panel: str):
    if panel.lower() == "s":
        return dict(
            h5ad="output/all_TMA_S_utag_ct_merged.h5ad",
            biomarkers=S_BIOMARKERS,
            colors=S_COMPART_COLORS,
            follicular=S_FOLLICULAR,
            interfollicular=S_INTERFOLLICULAR,
            boundary=S_BOUNDARY,
            drop=S_DROP,
        )
    if panel.lower() == "t":
        return dict(
            h5ad="output/all_TMA_T_utag_ct_merged.h5ad",
            biomarkers=T_BIOMARKERS,
            colors=T_COMPART_COLORS,
            follicular=T_FOLLICULAR,
            interfollicular=T_INTERFOLLICULAR,
            boundary=T_BOUNDARY,
            drop=T_DROP,
        )
    raise ValueError(f"unknown panel: {panel}")


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


def per_roi_metrics(roi_df, biomarkers, min_cells, biomarker_p90_thresh, all_compartments):
    not_unassigned = ~roi_df["cell_type"].isin(UNASSIGNED_CT)
    n_typed = int(not_unassigned.sum())
    if n_typed < min_cells:
        return None

    comp_counts = roi_df["compartment"].value_counts(normalize=True)
    p = comp_counts.values
    shannon = float(-(p * np.log2(p + 1e-12)).sum())
    simpson = float(1.0 - (p**2).sum())

    out = {
        "n_cells": int(len(roi_df)),
        "n_typed": n_typed,
        "shannon_compartment": shannon,
        "simpson_compartment": simpson,
    }
    for thr in (0.02, 0.05, 0.10):
        out[f"n_compartments_present_p{int(thr*100):02d}"] = int((comp_counts >= thr).sum())
    for c in all_compartments:
        out[f"frac_{c}"] = float(comp_counts.get(c, 0.0))
    for b in biomarkers:
        if b in roi_df.columns:
            out[f"{b}_mean"] = float(roi_df[b].mean())
            if b in biomarker_p90_thresh:
                out[f"{b}_pct_pos"] = float((roi_df[b] > biomarker_p90_thresh[b]).mean())
    return out


def join_clinical_and_grade(metrics_df, clinical_csv, grade_xlsx):
    # Grade sourced from the DWS-annotated clinical file (native GRADE column,
    # 136/136 patients covered as of the 2026-05-18 BCCA re-annotation).
    # The legacy xlsx path argument is accepted for back-compat but ignored.
    import warnings
    from src.clinical_linkage import load_clinical
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clin = load_clinical()
    clin = clin[["slide_ID", "Sample_ID", "Patient_ID", "GRADE"]].rename(
        columns={"GRADE": "grade"}
    )
    out = metrics_df.merge(clin, left_on="sample_id", right_on="slide_ID", how="left")
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
    p.add_argument("--panel", choices=["s", "t"], required=True)
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--out", default="output/grade_arch/mixed_cores")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS_PER_ROI)
    p.add_argument("--min-mixed-frac", type=float, default=DEFAULT_MIN_MIXED_FRAC,
                   help="ROI must have >=THIS fraction of cells on BOTH the "
                        "follicular and interfollicular sides (default 0.10)")
    args = p.parse_args()

    cfg = panel_config(args.panel)
    panel_label = args.panel.upper()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Mixed-core grade-architecture analysis ({panel_label}-panel) ===")
    print(f"Loading {cfg['h5ad']} ...")
    df, bm_present = load_data(Path(cfg["h5ad"]), cfg["biomarkers"])
    print(f"  cells={len(df):,}, ROIs={df['sample_id'].nunique()}")
    print(f"  biomarkers loaded: {bm_present}")

    df = df[df["sample_id"].apply(is_tumor_core)].copy()
    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)
    print(f"  after exclusion+normalize: cells={len(df):,}, ROIs={df['sample_id'].nunique()}")

    obs_comps = set(df["compartment"].unique())
    missing_compart = obs_comps - set(cfg["colors"])
    if missing_compart:
        raise RuntimeError(f"compartment palette missing keys for {missing_compart}")
    # Defensive: ensure each compartment is in exactly one of the four sets
    union = cfg["follicular"] | cfg["interfollicular"] | cfg["boundary"] | cfg["drop"]
    missing_classification = set(cfg["colors"]) - union
    if missing_classification:
        raise RuntimeError(
            f"compartments not classified into follicular/interfollicular/boundary/drop: "
            f"{missing_classification}"
        )
    overlaps = []
    for a_name, a in (("follicular", cfg["follicular"]), ("interfollicular", cfg["interfollicular"]),
                      ("boundary", cfg["boundary"]), ("drop", cfg["drop"])):
        for b_name, b in (("follicular", cfg["follicular"]), ("interfollicular", cfg["interfollicular"]),
                          ("boundary", cfg["boundary"]), ("drop", cfg["drop"])):
            if a_name < b_name and a & b:
                overlaps.append((a_name, b_name, a & b))
    if overlaps:
        raise RuntimeError(f"compartment classification overlaps: {overlaps}")

    bm_p90 = {b: float(np.quantile(df[b].dropna().values, 0.90)) for b in bm_present}

    rows = []
    retain_rows = []
    for sid, sub in df.groupby("sample_id"):
        m = per_roi_metrics(sub, bm_present, args.min_cells, bm_p90, list(cfg["colors"].keys()))
        if m is None:
            continue
        # Mixed-core gate: require >=min_mixed_frac on each side, computed
        # over the FULL ROI denominator (so an ROI that is 50% drop+50%
        # follicular fails the gate, since interfollicular fraction = 0).
        f_frac = sum(m[f"frac_{c}"] for c in cfg["follicular"])
        i_frac = sum(m[f"frac_{c}"] for c in cfg["interfollicular"])
        b_frac = sum(m[f"frac_{c}"] for c in cfg["boundary"])
        d_frac = sum(m[f"frac_{c}"] for c in cfg["drop"])
        m["frac_follicular"] = f_frac
        m["frac_interfollicular"] = i_frac
        m["frac_boundary"] = b_frac
        m["frac_drop"] = d_frac
        m["is_mixed"] = bool(f_frac >= args.min_mixed_frac and i_frac >= args.min_mixed_frac)
        retain_rows.append({"sample_id": sid, "n_typed": m["n_typed"],
                            "frac_follicular": f_frac, "frac_interfollicular": i_frac,
                            "frac_boundary": b_frac, "frac_drop": d_frac,
                            "is_mixed": m["is_mixed"]})
        rows.append({"sample_id": sid, **m})
    metrics_df = pd.DataFrame(rows)
    retain_df = pd.DataFrame(retain_rows)
    metrics_df = join_clinical_and_grade(metrics_df, Path(args.clinical), Path(args.grade))
    print(f"\n  ROIs with grade (post min-cells): {len(metrics_df)}")
    print(f"  retained (mixed): {int(metrics_df.is_mixed.sum())} of {len(metrics_df)} "
          f"({100 * metrics_df.is_mixed.mean():.1f}%)")
    print(f"  per-grade ROI counts (mixed only):")
    print(metrics_df[metrics_df.is_mixed].grade.value_counts().to_string())

    metrics_df.to_csv(out_dir / f"grade_mixed_per_roi_{args.panel}.csv", index=False)
    retain_df.to_csv(out_dir / f"grade_mixed_retention_{args.panel}.csv", index=False)

    # Patient-level aggregation: AVERAGE only the mixed ROIs. Patients with no
    # mixed ROIs are dropped.
    mixed = metrics_df[metrics_df.is_mixed].copy()
    metric_cols = [c for c in mixed.columns
                   if c not in {"sample_id", "slide_ID", "Sample_ID", "Patient_ID",
                                "grade", "is_mixed"}]
    agg = (mixed.groupby(["Patient_ID", "grade"])[metric_cols]
           .mean().reset_index())
    print(f"\n  Patient-level n (mixed-only): {len(agg)} "
          f"(FOLL1={sum(agg.grade=='FOLL1')}, FOLL2={sum(agg.grade=='FOLL2')}, "
          f"FOLL3A={sum(agg.grade=='FOLL3A')})")

    agg.to_csv(out_dir / f"grade_mixed_per_patient_{args.panel}.csv", index=False)

    if any(sum(agg.grade == g) < 3 for g in GRADE_ORDER):
        print("WARNING: one or more grade groups has <3 patients after mixed filter; KW p-values "
              "will be NaN for those metrics.")

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

    print(f"\n=== KW + BH(q) — patient-level mixed-only (n={len(agg)})")
    print(f"{'metric':38s} {'p':>10s} {'q_within':>10s} {'q_combined':>11s}  medians (FOLL1/2/3A)")
    summary = []
    for m, p_val, q_w, q_c, med in zip(test_metrics, pvals, q_within, q_combined, medians):
        med_str = " / ".join(f"{med[g]:.3g}" if not np.isnan(med[g]) else "NA"
                              for g in GRADE_ORDER)
        flag = " *" if (not np.isnan(q_w) and q_w < 0.05) else ""
        print(f"  {m:38s} {p_val:10.4g} {q_w:10.4g} {q_c:11.4g}  {med_str}{flag}")
        summary.append({"metric": m, "p_KW": p_val,
                        "q_BH_within_family": q_w, "q_BH_combined": q_c,
                        **{f"med_{g}": med[g] for g in GRADE_ORDER}})
    pd.DataFrame(summary).to_csv(out_dir / f"grade_mixed_kw_summary_{args.panel}.csv",
                                  index=False)
    qvals = q_combined

    # ---- Figure: same 4-panel layout ----
    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 1, 1, 1], wspace=0.4)

    ax = fig.add_subplot(gs[0, 0])
    comp_cols = [c for c in agg.columns if c.startswith("frac_")
                 and c.replace("frac_", "") in cfg["colors"]]
    grade_means = agg.groupby("grade")[comp_cols].mean().reindex(GRADE_ORDER)
    grade_means = grade_means / grade_means.sum(axis=1).values[:, None]
    bottom = np.zeros(len(GRADE_ORDER))
    for c in comp_cols:
        name = c.replace("frac_", "")
        color = cfg["colors"].get(name, "#888888")
        vals = grade_means[c].values
        ax.bar(GRADE_ORDER, vals, bottom=bottom, color=color, label=name,
               edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_ylabel("Mean compartment fraction")
    ax.set_title(f"Compartment composition by grade ({panel_label}-panel, mixed cores only)",
                 fontsize=11)
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

    fig.suptitle(
        f"FL grade — {panel_label}-panel architecture (mixed cores only, "
        f"≥{int(args.min_mixed_frac*100)}% follicular AND ≥{int(args.min_mixed_frac*100)}% "
        f"interfollicular, patient-level n={len(agg)})",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()
    out = out_dir / f"fig_grade_mixed_{args.panel}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
