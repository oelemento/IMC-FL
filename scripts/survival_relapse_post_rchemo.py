#!/usr/bin/env python3
"""Survival analysis using Relapse_post_rchemo_code as the event indicator.

The standard PFS column in BCCA data has mixed semantics: for observation-only
patients PFS=1 means "needed treatment" (NOT relapse-after-therapy). The DWS
update adds a tumor-biology-pure endpoint, `Relapse_post_rchemo_code` (only
filled for r-chemo-treated patients with first systemic therapy = r-chemo,
n=106), measuring true relapse after first FL-directed therapy.

This script reuses the patient-level survival_covariates.csv produced by
survival_analysis.py, merges in the Relapse_post_rchemo_* columns from the
DWS clinical, restricts to r-chemo-treated patients, and runs univariate
Cox for the headline S-panel markers + CD14 multivariate adjusted for FLIPI
+ grade + age. Outputs a tight 2-panel comparison figure (KM curve + forest
plot vs the standard PFS endpoint).

No modification to survival_analysis.py or fig_survival_v2.py — this is a
parallel endpoint analysis layered on top of their outputs.
"""
import argparse, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import load_clinical


def univariate_cox(df, metric, duration_col, event_col):
    sub = df[[metric, duration_col, event_col]].dropna().copy()
    n = len(sub); n_ev = int(sub[event_col].sum())
    if n < 20 or n_ev < 5:
        return None
    mu, sd = sub[metric].mean(), sub[metric].std()
    if not sd or np.isnan(sd):
        return None
    sub[metric] = (sub[metric] - mu) / sd
    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col=duration_col, event_col=event_col)
        s = cph.summary.iloc[0]
        return {"metric": metric, "HR": s["exp(coef)"],
                "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"],
                "p": s["p"], "n": n, "n_events": n_ev}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--covariates",
                    default="output/hypotheses_v8/survival_covariates.csv")
    ap.add_argument("--out", default="output/hypotheses_v8/relapse_post_rchemo")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-patient covariates: {args.covariates}")
    cov = pd.read_csv(args.covariates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the cohort-diff audit is informational here
        clin = load_clinical()
    rcols = ["slide_ID", "Relapse_post_rchemo_time", "Relapse_post_rchemo_code",
             "First_RCHEMO_for_FL", "GRADE"]
    rcols = [c for c in rcols if c in clin.columns]
    df = cov.merge(clin[rcols], on="slide_ID", how="left")

    # r-chemo treated subset. David's column 'First_RCHEMO_for_FL' is filled
    # with the actual first-systemic regimen for FL (any class) — so we have
    # to additionally filter on rituximab-containing regimens to match his
    # documented endpoint definition ("Relapse_post_rchemo only filled if
    # first systemic was r-chemo"). 4 non-RITU rows (CYCL-CLAD, CHOP9-RAD,
    # CHLORAMBUCIL, UNKNOWN) are excluded here even though they're non-null.
    raw_rchemo = df[df["First_RCHEMO_for_FL"].notna()].copy()
    print(f"  n_total: {len(df)}, n_with_first_systemic: {len(raw_rchemo)}")
    non_rituximab = raw_rchemo[~raw_rchemo["First_RCHEMO_for_FL"].str.contains("RITU", na=False)]
    if len(non_rituximab) > 0:
        print(f"  Excluding {len(non_rituximab)} non-rituximab regimens:")
        print(non_rituximab[["Patient_ID", "slide_ID", "First_RCHEMO_for_FL"]].to_string(index=False))
    rchemo = raw_rchemo[raw_rchemo["First_RCHEMO_for_FL"].str.contains("RITU", na=False)].copy()
    print(f"  n_r-chemo (RITU-containing): {len(rchemo)}")
    n_event_coded = int(rchemo.Relapse_post_rchemo_code.notna().sum())
    n_missing = int(rchemo.Relapse_post_rchemo_code.isna().sum())
    print(f"  Relapse event coding: coded={n_event_coded}, missing={n_missing}")
    # Verify NaN code rows also have NaN time (i.e., truly missing follow-up,
    # not 'still in remission with documented time'). David's convention: code=0
    # means still in remission with non-null time.
    nan_code_with_time = rchemo[
        rchemo.Relapse_post_rchemo_code.isna()
        & rchemo.Relapse_post_rchemo_time.notna()
    ]
    if len(nan_code_with_time) > 0:
        raise ValueError(
            f"Found {len(nan_code_with_time)} r-chemo rows with NaN event code but non-null "
            f"Relapse_post_rchemo_time. These could be silently mis-classified — "
            f"check whether David intended code=0 (censored alive)."
        )
    print(f"  Relapse_post_rchemo events: {int(rchemo.Relapse_post_rchemo_code.sum())} of "
          f"{n_event_coded}")

    rchemo["rel_time"] = pd.to_numeric(rchemo["Relapse_post_rchemo_time"], errors="coerce")
    rchemo["rel_event"] = pd.to_numeric(rchemo["Relapse_post_rchemo_code"], errors="coerce")

    # Headline S-panel markers from CLAUDE.md / paper
    headline_markers = [
        "s_CD14", "s_CD68", "s_S100A9", "s_VISTA", "s_IDO", "s_CD21",
        "s_M1_frac", "s_M2_frac", "s_mac_frac", "s_FDC_frac",
        "s_DC_frac", "s_myeloid_S100A9_frac",
    ]
    avail = [m for m in headline_markers if m in rchemo.columns]

    print(f"\nUnivariate Cox vs Relapse_post_rchemo (n_r-chemo subset):")
    rows = []
    for m in avail:
        r_new = univariate_cox(rchemo, m, "rel_time", "rel_event")
        r_old = univariate_cox(rchemo, m, "pfs_time", "pfs_event")
        if r_new is None or r_old is None:
            continue
        rows.append({
            "marker": m,
            "HR_relapse": r_new["HR"], "lo_relapse": r_new["lo"], "hi_relapse": r_new["hi"],
            "p_relapse": r_new["p"], "n_relapse": r_new["n"], "ev_relapse": r_new["n_events"],
            "HR_pfs": r_old["HR"], "lo_pfs": r_old["lo"], "hi_pfs": r_old["hi"],
            "p_pfs": r_old["p"], "n_pfs": r_old["n"], "ev_pfs": r_old["n_events"],
        })
    summary = pd.DataFrame(rows).sort_values("p_relapse")
    summary.to_csv(out_dir / "univariate_relapse_post_rchemo.csv", index=False)
    print(summary[["marker", "HR_relapse", "p_relapse", "ev_relapse",
                    "HR_pfs", "p_pfs", "ev_pfs"]].to_string(index=False))

    # Multivariate Cox: CD14 + FLIPI + grade, both endpoints
    cd14 = "s_CD14"
    mv_rows = []
    for endpoint_name, t, e in [("PFS (treated)", "pfs_time", "pfs_event"),
                                 ("Relapse_post_rchemo", "rel_time", "rel_event")]:
        # Univariate CD14
        for label, cols in [
            ("CD14 (univariate)", [cd14, t, e]),
            ("CD14 + FLIPI", [cd14, "FLIPI", t, e]),
            ("CD14 + FLIPI + grade", [cd14, "FLIPI", "grade_num", t, e]),
        ]:
            tmp = rchemo.copy()
            tmp["grade_num"] = tmp["GRADE"].map({"FOLL1": 1, "FOLL2": 2, "FOLL3A": 3})
            sub = tmp[cols].dropna()
            sub = sub[(sub[e] == 0) | (sub[e] == 1)].copy()
            if len(sub) < 20:
                continue
            sub[cd14] = (sub[cd14] - sub[cd14].mean()) / sub[cd14].std()
            try:
                cph = CoxPHFitter()
                cph.fit(sub, duration_col=t, event_col=e)
                row = cph.summary.loc[cd14]
                mv_rows.append({"endpoint": endpoint_name, "label": label,
                                "HR": row["exp(coef)"], "lo": row["exp(coef) lower 95%"],
                                "hi": row["exp(coef) upper 95%"], "p": row["p"],
                                "n": len(sub)})
            except Exception as ex:
                print(f"  {endpoint_name} / {label}: FIT FAILED ({ex})")
    mv = pd.DataFrame(mv_rows)
    mv.to_csv(out_dir / "multivariate_cd14.csv", index=False)
    print("\nMultivariate Cox — CD14 HR by endpoint:")
    print(mv.to_string(index=False))

    # ─── LOTO sensitivity for headline CD14 result (CLAUDE.md guardrail #2) ───
    if "tma" in rchemo.columns:
        loto_rows = []
        print(f"\nLOTO sensitivity — CD14 vs Relapse_post_rchemo (univariate Cox):")
        for tma in sorted(rchemo["tma"].dropna().unique()):
            sub_loto = rchemo[rchemo["tma"] != tma]
            r = univariate_cox(sub_loto, cd14, "rel_time", "rel_event")
            if r is None:
                continue
            loto_rows.append({"excluded_tma": tma, "n": r["n"], "events": r["n_events"],
                              "HR": r["HR"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]})
            print(f"  excl. {tma}: HR={r['HR']:.3f} [{r['lo']:.3f}–{r['hi']:.3f}] "
                  f"p={r['p']:.4f}  n={r['n']}, events={r['n_events']}")
        pd.DataFrame(loto_rows).to_csv(out_dir / "loto_cd14_relapse_post_rchemo.csv", index=False)
    else:
        print("\nLOTO sensitivity skipped — no 'tma' column in covariates")

    # ─── Figure: 2-panel ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # (a) KM by CD14 tertile, Relapse_post_rchemo endpoint
    ax = axes[0]
    rs = rchemo[[cd14, "rel_time", "rel_event"]].dropna()
    if len(rs) > 0:
        q33, q66 = rs[cd14].quantile([1/3, 2/3])
        def bucket(v):
            if v <= q33: return "Low"
            if v <= q66: return "Mid"
            return "High"
        rs["grp"] = rs[cd14].apply(bucket)
        kmf = KaplanMeierFitter()
        for grp, color in [("Low", "#1f77b4"), ("Mid", "#7f7f7f"), ("High", "#d62728")]:
            sub = rs[rs.grp == grp]
            kmf.fit(sub["rel_time"], sub["rel_event"], label=f"{grp} CD14 (n={len(sub)})")
            kmf.plot_survival_function(ax=ax, color=color, ci_show=False)
        # logrank low vs high
        lo = rs[rs.grp == "Low"]; hi = rs[rs.grp == "High"]
        lr = logrank_test(lo["rel_time"], hi["rel_time"], lo["rel_event"], hi["rel_event"])
        ax.text(0.05, 0.10, f"log-rank low vs high\np={lr.p_value:.3g}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    ax.set_xlabel("Time from first FL-directed r-chemo (years)")
    ax.set_ylabel("Relapse-free probability")
    ax.set_title("(a) CD14 vs Relapse_post_rchemo\n(r-chemo treated only)",
                 fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (b) Forest: HR per SD for headline markers — both endpoints
    ax = axes[1]
    sub = summary.dropna(subset=["HR_relapse", "HR_pfs"]).head(12).reset_index(drop=True)
    y = np.arange(len(sub))
    # Plot relapse HRs in red, PFS HRs in blue, offset
    for i, row in sub.iterrows():
        ax.errorbar(np.log(row["HR_relapse"]), i + 0.18,
                    xerr=[[np.log(row["HR_relapse"]) - np.log(row["lo_relapse"])],
                          [np.log(row["hi_relapse"]) - np.log(row["HR_relapse"])]],
                    fmt="o", color="#d62728", capsize=3, ms=6,
                    label="Relapse_post_rchemo" if i == 0 else None)
        ax.errorbar(np.log(row["HR_pfs"]), i - 0.18,
                    xerr=[[np.log(row["HR_pfs"]) - np.log(row["lo_pfs"])],
                          [np.log(row["hi_pfs"]) - np.log(row["HR_pfs"])]],
                    fmt="s", color="#1f77b4", capsize=3, ms=6,
                    label="Standard PFS (treated)" if i == 0 else None)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(sub.marker.tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("log(HR) per SD")
    ax.set_title(f"(b) Univariate Cox per-SD HR\nr-chemo subset (n≈{int(sub['n_relapse'].iloc[0])})",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("Relapse_post_rchemo as a tumor-biology-pure endpoint vs standard PFS",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_relapse_post_rchemo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
