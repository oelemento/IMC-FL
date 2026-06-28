#!/usr/bin/env python3
"""Diagnostic: why does the grade-architecture signal appear on S-panel but not T-panel?

1. Cohort overlap (S vs T patients with grade and ≥8000 typed cells)
2. Restricted analyses: rerun on shared patients only — does signal change?
3. Outlier detection: per-patient residuals vs grade median for shannon_compartment
4. Per-TMA breakdown of patient cohort
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

GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}
MIN_CELLS = 8000


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


def panel_summary(h5ad_path):
    """Per-ROI shannon_compartment + n_typed for a panel."""
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

    df = pd.DataFrame({"sample_id": sample_id, "cell_type": cell_type,
                       "compartment": compartment})
    df = df[df.sample_id.apply(is_tumor_core)].copy()
    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)

    rows = []
    for sid, sub in df.groupby("sample_id"):
        not_un = ~sub["cell_type"].isin(["Unassigned", "Low quality / Unassigned"])
        n_typed = int(not_un.sum())
        if n_typed < MIN_CELLS:
            continue
        comp_fracs = sub["compartment"].value_counts(normalize=True)
        p_ = comp_fracs.values
        shannon = float(-(p_ * np.log2(p_ + 1e-12)).sum())
        n_present_p05 = int((comp_fracs >= 0.05).sum())
        rows.append({"sample_id": sid, "n_typed": n_typed,
                     "n_compart_present_p05": n_present_p05,
                     "shannon_compartment": shannon,
                     "n_compartments_observed": int(comp_fracs.size)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s-panel", default="output/all_TMA_S_utag_ct_merged.h5ad")
    p.add_argument("--t-panel", default="output/all_TMA_T_utag_ct_merged.h5ad")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--out", default="output/grade_arch/diagnostic")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing per-ROI summaries on each panel ...")
    s = panel_summary(args.s_panel); s["panel"] = "S"
    t = panel_summary(args.t_panel); t["panel"] = "T"
    print(f"  S-panel ROIs passing min_cells={MIN_CELLS}: {len(s)}")
    print(f"  T-panel ROIs passing min_cells={MIN_CELLS}: {len(t)}")

    # Join clinical
    clin = pd.read_csv(args.clinical)[["slide_ID", "Sample_ID", "Patient_ID"]]
    grade_df = pd.read_excel(args.grade).rename(columns={"FL ID": "Sample_ID", "DIAG": "grade"})
    grade_df = grade_df[["Sample_ID", "grade"]]

    def join(df):
        x = df.merge(clin, left_on="sample_id", right_on="slide_ID", how="left")
        x = x.merge(grade_df, on="Sample_ID", how="left")
        x = x[x.grade.isin(GRADE_ORDER)].copy()
        # TMA from sample_id prefix
        x["TMA"] = x["sample_id"].str.split("_").str[0]
        return x

    s = join(s); t = join(t)
    print(f"\n  S-panel ROIs with grade: {len(s)} ({s.grade.value_counts().to_dict()})")
    print(f"  T-panel ROIs with grade: {len(t)} ({t.grade.value_counts().to_dict()})")

    # ===== 1. Cohort overlap =====
    s_pts = set(s["Patient_ID"].dropna())
    t_pts = set(t["Patient_ID"].dropna())
    only_s = s_pts - t_pts
    only_t = t_pts - s_pts
    both = s_pts & t_pts
    print(f"\n=== Cohort overlap (patient-level) ===")
    print(f"  Patients in S only:    {len(only_s)}")
    print(f"  Patients in T only:    {len(only_t)}")
    print(f"  Patients in BOTH:      {len(both)}")

    # ===== 2. Per-patient analysis =====
    s_pt = (s.groupby(["Patient_ID", "grade", "TMA"])
            [["shannon_compartment", "n_compart_present_p05", "n_typed"]]
            .mean().reset_index())
    t_pt = (t.groupby(["Patient_ID", "grade", "TMA"])
            [["shannon_compartment", "n_compart_present_p05", "n_typed"]]
            .mean().reset_index())

    print(f"\n=== Full-cohort KW: shannon_compartment ===")
    p_s, med_s = kw(s_pt, "shannon_compartment")
    p_t, med_t = kw(t_pt, "shannon_compartment")
    print(f"  S-panel n={len(s_pt)}: p={p_s:.4g}, medians={[f'{med_s[g]:.3f}' for g in GRADE_ORDER]}")
    print(f"  T-panel n={len(t_pt)}: p={p_t:.4g}, medians={[f'{med_t[g]:.3f}' for g in GRADE_ORDER]}")

    print(f"\n=== Restricted to SHARED patients (n={len(both)}) ===")
    s_shared = s_pt[s_pt["Patient_ID"].isin(both)]
    t_shared = t_pt[t_pt["Patient_ID"].isin(both)]
    p_ss, med_ss = kw(s_shared, "shannon_compartment")
    p_ts, med_ts = kw(t_shared, "shannon_compartment")
    print(f"  S-panel: p={p_ss:.4g}, medians={[f'{med_ss[g]:.3f}' for g in GRADE_ORDER]}, "
          f"per grade: {s_shared.grade.value_counts().to_dict()}")
    print(f"  T-panel: p={p_ts:.4g}, medians={[f'{med_ts[g]:.3f}' for g in GRADE_ORDER]}, "
          f"per grade: {t_shared.grade.value_counts().to_dict()}")

    # ===== 3. Per-TMA breakdown =====
    print(f"\n=== Per-TMA breakdown (full cohort) ===")
    for panel_name, df_pt in (("S", s_pt), ("T", t_pt)):
        print(f"  {panel_name}-panel:")
        for tma in sorted(df_pt["TMA"].unique()):
            sub = df_pt[df_pt["TMA"] == tma]
            print(f"    TMA {tma}: n={len(sub)} ({sub.grade.value_counts().to_dict()})")

    # Leave-one-TMA-out for T-panel shannon
    print(f"\n=== Leave-one-TMA-out KW: T-panel shannon_compartment ===")
    for tma in sorted(t_pt["TMA"].unique()):
        sub = t_pt[t_pt["TMA"] != tma]
        p_l, med_l = kw(sub, "shannon_compartment")
        med_str = " / ".join(f"{med_l[g]:.3f}" for g in GRADE_ORDER)
        print(f"  Excluding TMA {tma}: n={len(sub)}, p={p_l:.4g}, medians={med_str}")

    # ===== 4. n_typed distribution per panel =====
    print(f"\n=== Per-patient n_typed (cell count) by panel ===")
    for panel_name, df_pt in (("S", s_pt), ("T", t_pt)):
        n = df_pt["n_typed"]
        print(f"  {panel_name}-panel: median={n.median():.0f}, "
              f"p25={n.quantile(0.25):.0f}, p75={n.quantile(0.75):.0f}, "
              f"min={n.min():.0f}, max={n.max():.0f}")

    # ===== 5. Look for outlier patients in T-panel =====
    print(f"\n=== T-panel outliers: patients with extreme shannon_compartment ===")
    for g in GRADE_ORDER:
        sub = t_pt[t_pt.grade == g].copy()
        med = sub["shannon_compartment"].median()
        sub["dev"] = sub["shannon_compartment"] - med
        outliers = sub.reindex(sub["dev"].abs().sort_values(ascending=False).index).head(3)
        print(f"  {g} median shannon = {med:.3f}; top-3 outliers:")
        for _, r in outliers.iterrows():
            print(f"    {r['Patient_ID']:12s} ({r['TMA']:3s}): shannon={r['shannon_compartment']:.3f}  "
                  f"(dev={r['dev']:+.3f})  n_typed={r['n_typed']:.0f}")

    # ===== Figure: side-by-side full vs shared cohort + per-TMA =====
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    rng = np.random.default_rng(0)

    def boxplot(ax, df_pt, metric, label):
        data = [df_pt.loc[df_pt.grade == g, metric].dropna().values for g in GRADE_ORDER]
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
            ax.set_title(f"{label}\np={pv:.3g}, n={len(df_pt)}", fontsize=10)
        else:
            ax.set_title(label, fontsize=10)
        ax.set_xlabel("Grade")
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Row 1: full cohort
    boxplot(axes[0, 0], s_pt, "shannon_compartment", "S-panel (full)\nshannon")
    boxplot(axes[0, 1], t_pt, "shannon_compartment", "T-panel (full)\nshannon")
    boxplot(axes[0, 2], t_pt, "n_typed", "T-panel (full)\nn_typed cells")

    # Row 2: shared patients only
    boxplot(axes[1, 0], s_shared, "shannon_compartment", "S-panel (shared)\nshannon")
    boxplot(axes[1, 1], t_shared, "shannon_compartment", "T-panel (shared)\nshannon")
    boxplot(axes[1, 2], t_pt, "n_compart_present_p05", "T-panel (full)\n# compartments present (≥5%)")

    fig.suptitle("Diagnostic: S-panel vs T-panel architecture-grade signal",
                 fontsize=12, y=1.0)
    plt.tight_layout()
    out = out_dir / "fig_panel_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
