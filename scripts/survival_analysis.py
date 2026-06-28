"""
Survival analysis: spatial IMC metrics vs clinical outcomes.

Links per-ROI spatial metrics (entropy, E:S ratio, exhaustion, Tfh,
follicularity), cell type fractions, and marker intensities to
patient-level OS and PFS. Uses both T-panel and S-panel data.

Key findings:
  - S-panel myeloid markers (CD14, CD68, S100A9) predict PFS and OS
  - CD8 T cell fraction (T-panel) independently predicts PFS
  - CD14 mean intensity is the strongest single predictor (PFS p=0.0003)
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import (
    EXCLUDE_ROIS,
    load_clinical,
    normalize_sample_id,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LQ = "Low quality / Unassigned"

EFFECTOR = ["CD8 T cells", "Macrophages (GzmB+)"]
SUPPRESSOR = ["Treg", "CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]

FOLLICULAR_COMPARTMENTS = [
    "Activated B / CXCR5hi zone",
    "B cell follicle (CD20hi/CXCR5hi)",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)",
    "Follicle-T zone interface",
    "GC core",
]

TMA_COLORS = {"A1": "#E41A1C", "B1": "#377EB8", "C1": "#4DAF4A", "Biomax": "#984EA3"}

# S-panel myeloid cell types (split into M1/M2/generic in S-panel annotations)
S_MYELOID_TYPES = [
    "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Myeloid (S100A9+)", "Dendritic cells",
]

# S-panel markers to skip in survival screen (housekeeping / DNA)
S_SKIP_MARKERS = {"DNA1", "DNA2", "HistoneH3"}

# Cell type fraction columns from entropy CSV
CT_FRACTION_COLS = [
    "Activated B / Plasmablast", "B cells", "B cells (CD20hi)",
    "B cells (CXCR5hi)", "B cells (TOXhi)", "B cells (weak CD20)",
    "CD4 T cells", "CD8 T cells", "CD8 T exhausted",
    "CD8 T pre-exhausted (TOX+)", "Macrophages (GzmB+)", "GC B cells",
    "Low quality / Unassigned", "Macrophages", "Mixed / Border cells",
    "Other", "T cells", "Treg",
]

METRIC_LABELS = {
    "entropy": "Shannon entropy",
    "exhausted_cd8_frac": "Exhausted CD8 fraction",
    "log_es_ratio": "log₂(E:S ratio)",
    "tfh_frac": "Tfh fraction",
    "follicularity": "Follicularity score",
    "cd8_foll_ratio": "CD8 follicular ratio",
    "cd8_foll_density": "CD8 density (follicular)",
    "cd8_ifoll_density": "CD8 density (interfollicular)",
}

# Shorter labels for cell type forest plots
CT_SHORT_LABELS = {
    "Activated B / Plasmablast": "Act. B / Plasmablast",
    "B cells (CD20hi)": "B (CD20hi)",
    "B cells (CXCR5hi)": "B (CXCR5hi)",
    "B cells (TOXhi)": "B (TOXhi)",
    "B cells (weak CD20)": "B (weak CD20)",
    "CD8 T pre-exhausted (TOX+)": "CD8 T pre-exh (TOX+)",
    "Macrophages (GzmB+)": "Macrophages (GzmB+)",
    "Low quality / Unassigned": "Low qual / Unassigned",
    "Mixed / Border cells": "Mixed / Border",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_array(f, key):
    ds = f["obs"][key]
    if isinstance(ds, h5py.Group) and "categories" in ds:
        cats = ds["categories"][:]
        codes = ds["codes"][:]
        cats_str = np.array(
            [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
        )
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


def get_marker_idx(f):
    key = "_index" if "_index" in f["var"] else "index"
    names = f["var"][key][:]
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return {n: i for i, n in enumerate(names)}


def is_tumor_core(sample_id):
    s = sample_id.lower()
    if "_ton_" in s or "_adr_" in s:
        return False
    for tissue in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if tissue in s:
            return False
    if sample_id == "Biomax_ROI_006":
        return False
    return True


def panel_label(ax, letter, x=-0.08, y=1.05):
    ax.text(
        x, y, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top", ha="left",
    )


# ---------------------------------------------------------------------------
# Step 1: Extract per-ROI metrics
# ---------------------------------------------------------------------------

def extract_metrics(t_panel_path, t_utag_path, entropy_csv_path):
    """Extract per-ROI spatial metrics from T-panel data."""
    print("Loading entropy CSV...")
    ent_df = pd.read_csv(entropy_csv_path)
    ent_df = ent_df.rename(columns={"roi": "sample_id"})

    # Filter: tumor cores only, exclude duplicates
    ent_df = ent_df[
        ent_df["sample_id"].apply(is_tumor_core)
        & ~ent_df["sample_id"].isin(EXCLUDE_ROIS)
    ].copy()

    # Normalize sample_ids
    ent_df["slide_ID"] = ent_df["sample_id"].apply(normalize_sample_id)

    # Extract E:S ratio and exhausted CD8 fraction from cell type fractions
    # These are already in the entropy CSV as fraction columns
    eff_cols = [c for c in ent_df.columns if c in EFFECTOR]
    sup_cols = [c for c in ent_df.columns if c in SUPPRESSOR]

    ent_df["n_eff"] = ent_df[eff_cols].sum(axis=1) * ent_df["n_cells"]
    ent_df["n_sup"] = ent_df[sup_cols].sum(axis=1) * ent_df["n_cells"]
    ent_df["es_ratio"] = np.where(
        ent_df["n_sup"] > 0,
        ent_df["n_eff"] / ent_df["n_sup"],
        np.where(ent_df["n_eff"] > 0, 10.0, np.nan),
    )
    ent_df["log_es_ratio"] = np.log2(ent_df["es_ratio"].clip(lower=0.01))

    # Exhausted CD8 fraction (of typed cells, not total)
    lq_col = LQ if LQ in ent_df.columns else None
    if lq_col:
        ent_df["typed_frac"] = 1.0 - ent_df[lq_col]
    else:
        ent_df["typed_frac"] = 1.0
    exh_col = "CD8 T exhausted"
    ent_df["exhausted_cd8_frac"] = (
        ent_df[exh_col] / ent_df["typed_frac"].clip(lower=0.01)
        if exh_col in ent_df.columns
        else 0.0
    )

    print(f"  {len(ent_df)} tumor ROIs from entropy CSV")

    # --- Tfh fraction from h5ad (needs raw CXCR5 gating) ---
    print("Loading T-panel h5ad for Tfh gating...")
    tfh_fracs = {}
    with h5py.File(t_panel_path, "r") as f:
        sids = load_array(f, "sample_id")
        ctypes = load_array(f, "cell_type")
        midx = get_marker_idx(f)

        if "CXCR5" in midx:
            cxcr5_idx = midx["CXCR5"]
            X = f["X"]  # lazy
            for roi in ent_df["sample_id"].unique():
                mask = sids == roi
                roi_ct = ctypes[mask]
                cd4_mask = roi_ct == "CD4 T cells"
                n_cd4 = cd4_mask.sum()
                if n_cd4 < 10:
                    tfh_fracs[roi] = 0.0
                    continue
                # Read CXCR5 for CD4 T cells in this ROI
                roi_indices = np.where(mask)[0]
                cd4_indices = roi_indices[cd4_mask]
                cxcr5_vals = X[cd4_indices, cxcr5_idx]
                n_tfh = (cxcr5_vals > 2.0).sum()
                n_typed = (roi_ct != LQ).sum()
                tfh_fracs[roi] = float(n_tfh) / max(n_typed, 1)
        else:
            print("  WARNING: CXCR5 not found in markers, Tfh fraction = 0")

    ent_df["tfh_frac"] = ent_df["sample_id"].map(tfh_fracs).fillna(0.0)
    print(f"  Tfh fraction computed for {len(tfh_fracs)} ROIs")

    # --- Follicularity + compartment-resolved cell type densities ---
    print("Loading UTAG h5ad for compartment-resolved densities...")
    foll_scores = {}
    # Per-ROI dicts for compartment-resolved metrics
    comp_density_foll = {ct: {} for ct in CT_FRACTION_COLS if ct != LQ}
    comp_density_ifoll = {ct: {} for ct in CT_FRACTION_COLS if ct != LQ}
    comp_ratio = {ct: {} for ct in CT_FRACTION_COLS if ct != LQ}  # foll/ifoll ratio

    with h5py.File(t_utag_path, "r") as f:
        sids_u = load_array(f, "sample_id")
        comps = load_array(f, "compartment_name")
        ctypes_u = load_array(f, "cell_type")
        for roi in ent_df["sample_id"].unique():
            mask = sids_u == roi
            if mask.sum() == 0:
                continue
            roi_comps = comps[mask]
            roi_ct = ctypes_u[mask]

            is_foll = np.isin(roi_comps, FOLLICULAR_COMPARTMENTS)
            n_foll = is_foll.sum()
            n_ifoll = (~is_foll).sum()
            foll_scores[roi] = float(n_foll) / mask.sum()

            # Compartment-resolved density for each cell type
            for ct in comp_density_foll:
                is_ct = roi_ct == ct
                ct_in_foll = (is_ct & is_foll).sum()
                ct_in_ifoll = (is_ct & ~is_foll).sum()

                d_foll = float(ct_in_foll) / n_foll if n_foll > 0 else 0.0
                d_ifoll = float(ct_in_ifoll) / n_ifoll if n_ifoll > 0 else 0.0

                comp_density_foll[ct][roi] = d_foll
                comp_density_ifoll[ct][roi] = d_ifoll

                # Ratio: foll density / ifoll density (log2 for symmetry)
                if d_ifoll > 0 and d_foll > 0:
                    comp_ratio[ct][roi] = np.log2(d_foll / d_ifoll)
                elif d_foll > 0:
                    comp_ratio[ct][roi] = 4.0  # cap
                elif d_ifoll > 0:
                    comp_ratio[ct][roi] = -4.0  # cap
                else:
                    comp_ratio[ct][roi] = np.nan

    ent_df["follicularity"] = ent_df["sample_id"].map(foll_scores)

    # Add compartment-resolved columns
    comp_cols = []
    for ct in comp_density_foll:
        safe = ct.replace(" ", "_").replace("/", "_").replace("+", "p").replace("(", "").replace(")", "")
        col_f = f"foll_{safe}"
        col_i = f"ifoll_{safe}"
        col_r = f"ratio_{safe}"
        ent_df[col_f] = ent_df["sample_id"].map(comp_density_foll[ct])
        ent_df[col_i] = ent_df["sample_id"].map(comp_density_ifoll[ct])
        ent_df[col_r] = ent_df["sample_id"].map(comp_ratio[ct])
        comp_cols.extend([col_f, col_i, col_r])

    # Keep backward-compatible CD8 columns
    ent_df["cd8_foll_ratio"] = ent_df["sample_id"].map(
        {roi: comp_density_foll["CD8 T cells"].get(roi, np.nan)
         / max(comp_density_foll["CD8 T cells"].get(roi, 0)
               + comp_density_ifoll["CD8 T cells"].get(roi, 0), 1e-10)
         if "CD8 T cells" in comp_density_foll else np.nan
         for roi in ent_df["sample_id"]}
    )
    ent_df["cd8_foll_density"] = ent_df["sample_id"].map(
        comp_density_foll.get("CD8 T cells", {})
    )
    ent_df["cd8_ifoll_density"] = ent_df["sample_id"].map(
        comp_density_ifoll.get("CD8 T cells", {})
    )

    print(f"  Follicularity computed for {len(foll_scores)} ROIs")
    print(f"  Compartment-resolved densities: {len(comp_cols)} columns "
          f"({len(comp_density_foll)} cell types × 3 metrics)")

    # Keep relevant columns (spatial metrics + cell type fractions + compartment)
    metric_cols = [
        "sample_id", "slide_ID", "tma", "n_cells",
        "entropy", "exhausted_cd8_frac", "es_ratio", "log_es_ratio",
        "tfh_frac", "follicularity",
        "cd8_foll_ratio", "cd8_foll_density", "cd8_ifoll_density",
    ]
    # Add cell type fraction columns present in the CSV
    ct_cols = [c for c in CT_FRACTION_COLS if c in ent_df.columns]
    metric_cols.extend(ct_cols)
    # Add compartment-resolved columns
    metric_cols.extend([c for c in comp_cols if c in ent_df.columns])
    return ent_df[metric_cols].copy()


# ---------------------------------------------------------------------------
# Step 1b: Extract S-panel per-ROI marker intensities + myeloid fractions
# ---------------------------------------------------------------------------

def extract_s_panel_metrics(s_panel_path):
    """Extract per-ROI mean marker intensities and myeloid fractions from S-panel."""
    print("Loading S-panel h5ad for marker intensities...")
    with h5py.File(s_panel_path, "r") as f:
        var_names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in f["var"]["_index"][:]]
        sids = load_array(f, "sample_id")
        ctypes = load_array(f, "cell_type")
        X = f["X"]

        rois = sorted(set(sids))
        rois = [r for r in rois
                if is_tumor_core(r) and r not in EXCLUDE_ROIS
                and not r.startswith("Biomax")]

        rows = []
        for roi in rois:
            mask = sids == roi
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            vals = X[idx[0]:idx[-1] + 1, :]
            n = len(idx)
            roi_ct = ctypes[mask]

            row = {"sample_id": roi, "slide_ID": normalize_sample_id(roi)}

            # Mean marker intensities (transformed/scaled .X)
            means = vals.mean(axis=0)
            for i, m in enumerate(var_names):
                if m not in S_SKIP_MARKERS:
                    row[f"s_{m}"] = float(means[i])

            # Myeloid cell type fractions
            unique_ct, counts = np.unique(roi_ct, return_counts=True)
            ct_fracs = dict(zip(unique_ct, counts / n))
            row["s_M1_frac"] = ct_fracs.get("M1 Macrophages", 0)
            row["s_M2_frac"] = ct_fracs.get("M2 Macrophages", 0)
            row["s_mac_frac"] = ct_fracs.get("Macrophages", 0)
            row["s_myeloid_S100A9_frac"] = ct_fracs.get("Myeloid (S100A9+)", 0)
            row["s_DC_frac"] = ct_fracs.get("Dendritic cells", 0)
            row["s_FDC_frac"] = ct_fracs.get("FDC", 0)
            row["s_all_mac_frac"] = (
                row["s_M1_frac"] + row["s_M2_frac"] + row["s_mac_frac"]
            )
            row["s_all_myeloid_frac"] = row["s_all_mac_frac"] + (
                row["s_myeloid_S100A9_frac"] + row["s_DC_frac"]
            )
            rows.append(row)

    s_df = pd.DataFrame(rows)
    print(f"  {len(s_df)} tumor ROIs from S-panel")
    marker_cols = [c for c in s_df.columns if c.startswith("s_")]
    print(f"  {len(marker_cols)} S-panel columns "
          f"({sum(1 for c in marker_cols if not c.endswith('_frac'))} markers, "
          f"{sum(1 for c in marker_cols if c.endswith('_frac'))} myeloid fractions)")
    return s_df


# ---------------------------------------------------------------------------
# Step 2: Patient-level aggregation + clinical merge
# ---------------------------------------------------------------------------

def build_patient_table(roi_df, s_panel_df=None):
    """Merge ROI metrics with clinical data, one row per patient (T1 only)."""
    clin = load_clinical()

    # For serial biopsies, keep T1 only
    clin_t1 = clin.sort_values("T").drop_duplicates(subset="slide_ID", keep="first")

    # Exclude Biomax (no clinical)
    roi_df = roi_df[roi_df["tma"] != "Biomax"].copy()

    # Merge
    merged = roi_df.merge(clin_t1, on="slide_ID", how="inner")
    print(f"\n  Merged: {len(merged)} ROIs with clinical data")

    # For patients with multiple ROIs (serial biopsies), keep T1
    merged = merged.sort_values("T").drop_duplicates(
        subset="Patient_ID", keep="first"
    )
    print(f"  After T1-only dedup: {len(merged)} unique patients")

    # Rename clinical columns for convenience
    merged = merged.rename(columns={
        "Overall survival (y)": "os_time",
        "CODE_OS": "os_event",
        "Progression free survival (y)": "pfs_time",
        "CODE_PFS": "pfs_event",
    })

    # Ensure numeric
    for col in ["os_time", "os_event", "pfs_time", "pfs_event", "AGE", "FLIPI"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # FLIPI category: HIGH vs LOW/INTERMEDIATE
    merged["flipi_high"] = (merged["FLIPI.1"] == "HIGH").astype(int)

    # Transformation binary
    merged["transformed"] = (
        merged["Transformation"].str.strip().str.lower() == "yes"
    ).astype(int)

    # POD24: progression within 24 months of treatment
    # Exclude observation-only patients (INITIAL "TREATMENT" == "OBSE")
    tx_col = 'INITIAL "TREATMENT"'
    if tx_col in merged.columns:
        treated = merged[tx_col].str.strip() != "OBSE"
    else:
        treated = pd.Series(True, index=merged.index)

    merged["treated"] = treated.astype(int)

    # Derived POD24 (legacy rule): PFS event within 24 months of treatment.
    # Kept as 'pod24_derived' for traceability and regression checks.
    merged["pod24_derived"] = np.where(
        ~treated, np.nan,  # observation-only → exclude
        np.where(
            (merged["pfs_event"] == 1) & (merged["pfs_time"] <= 2.0), 1,
            np.where(merged["pfs_time"] > 2.0, 0, np.nan)
        )
    )

    # BCCA-annotated POD24 (DWS file): only filled for r-chemo-treated patients
    # with first systemic = r-chemo. This is the clinically rigorous definition
    # and is the new default for downstream POD24 analyses. The derived rule
    # remains available as 'pod24_derived' for backward comparison.
    if "POD24_event" in merged.columns:
        merged["pod24_bcca"] = merged["POD24_event"].map(
            {"YES": 1.0, "NO": 0.0}
        )
        merged["pod24"] = merged["pod24_bcca"]
        print(f"  POD24 source: BCCA-annotated POD24_event column "
              f"({int(merged['pod24'].sum())} YES, "
              f"{int((merged['pod24'] == 0).sum())} NO, "
              f"{int(merged['pod24'].isna().sum())} not evaluable)")
        if "pod24_derived" in merged.columns:
            both = merged[["pod24_bcca", "pod24_derived"]].dropna()
            agree = (both["pod24_bcca"] == both["pod24_derived"]).sum()
            print(f"  POD24 BCCA vs derived agreement: {agree}/{len(both)} "
                  f"({100*agree/max(len(both),1):.1f}%)")
    else:
        merged["pod24"] = merged["pod24_derived"]
        print(f"  POD24 source: derived from PFS time/event "
              "(BCCA POD24_event column not found)")

    # Merge S-panel data if provided
    if s_panel_df is not None:
        n_before = len(merged)
        s_cols = [c for c in s_panel_df.columns if c != "sample_id"]
        merged = merged.merge(
            s_panel_df[s_cols], on="slide_ID", how="left",
        )
        n_with_s = merged[[c for c in s_panel_df.columns
                           if c.startswith("s_")]].notna().any(axis=1).sum()
        print(f"  S-panel data merged: {n_with_s}/{n_before} patients have S-panel")

    n_os = merged["os_event"].sum()
    n_pfs = merged["pfs_event"].sum()
    n_pod24 = int(merged["pod24"].sum())
    n_nopod24 = int((merged["pod24"] == 0).sum())
    n_obs = int((~treated).sum())
    print(f"  OS events: {int(n_os)}/{len(merged)}, "
          f"PFS events: {int(n_pfs)}/{len(merged)}")
    print(f"  POD24: {n_pod24} vs non-POD24: {n_nopod24} "
          f"(excluded {n_obs} observation-only)")

    return merged


# ---------------------------------------------------------------------------
# Step 3: Survival models
# ---------------------------------------------------------------------------

def univariate_cox(df, metric, time_col, event_col):
    """Run univariate Cox PH for one metric."""
    sub = df[[metric, time_col, event_col]].dropna()
    if len(sub) < 20 or sub[event_col].sum() < 5:
        return None
    cph = CoxPHFitter()
    try:
        cph.fit(sub, duration_col=time_col, event_col=event_col)
        s = cph.summary.iloc[0]
        return {
            "metric": metric,
            "HR": s["exp(coef)"],
            "HR_lo": s["exp(coef) lower 95%"],
            "HR_hi": s["exp(coef) upper 95%"],
            "p": s["p"],
            "concordance": cph.concordance_index_,
            "n": len(sub),
            "events": int(sub[event_col].sum()),
        }
    except Exception as e:
        print(f"  Cox failed for {metric}: {e}")
        return None


def run_univariate_screen(df, metrics, endpoints):
    """Screen all metrics against all endpoints."""
    results = []
    for endpoint_name, (time_col, event_col) in endpoints.items():
        for m in metrics:
            res = univariate_cox(df, m, time_col, event_col)
            if res:
                res["endpoint"] = endpoint_name
                results.append(res)
    return pd.DataFrame(results)


def run_multivariate(df, spatial_metrics, time_col, event_col):
    """Multivariate Cox with spatial + clinical covariates."""
    covariates = spatial_metrics + ["AGE", "flipi_high"]
    sub = df[covariates + [time_col, event_col]].dropna()
    if len(sub) < 30:
        print(f"  Multivariate: too few observations ({len(sub)})")
        return None
    cph = CoxPHFitter()
    try:
        cph.fit(sub, duration_col=time_col, event_col=event_col)
        return cph
    except Exception as e:
        print(f"  Multivariate Cox failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 4: Figure
# ---------------------------------------------------------------------------

def plot_km(ax, df, metric, time_col, event_col, endpoint_label):
    """KM plot split at median of metric."""
    sub = df[[metric, time_col, event_col]].dropna()
    median_val = sub[metric].median()
    hi = sub[sub[metric] >= median_val]
    lo = sub[sub[metric] < median_val]

    kmf_hi = KaplanMeierFitter()
    kmf_lo = KaplanMeierFitter()

    label_name = METRIC_LABELS.get(metric, metric)

    kmf_hi.fit(hi[time_col], hi[event_col], label=f"High {label_name}")
    kmf_lo.fit(lo[time_col], lo[event_col], label=f"Low {label_name}")

    kmf_lo.plot_survival_function(ax=ax, ci_show=True, color="#377EB8")
    kmf_hi.plot_survival_function(ax=ax, ci_show=True, color="#E41A1C")

    # Log-rank test
    lr = logrank_test(
        lo[time_col], hi[time_col],
        event_observed_A=lo[event_col], event_observed_B=hi[event_col],
    )

    ax.set_xlabel("Time (years)")
    ax.set_ylabel(f"{endpoint_label} probability")
    ax.set_title(f"{label_name} — {endpoint_label}")
    ax.text(
        0.95, 0.05,
        f"log-rank p={lr.p_value:.3f}\nn={len(sub)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(-0.05, 1.05)


def plot_forest(ax, results_df, endpoint_label):
    """Forest plot of univariate Cox HRs."""
    sub = results_df[results_df["endpoint"] == endpoint_label].copy()
    if sub.empty:
        ax.set_visible(False)
        return
    sub = sub.sort_values("p")
    y_pos = range(len(sub))

    for i, (_, row) in enumerate(sub.iterrows()):
        color = "#E41A1C" if row["p"] < 0.05 else "#666666"
        ax.plot(
            [row["HR_lo"], row["HR_hi"]], [i, i],
            color=color, linewidth=2,
        )
        ax.plot(row["HR"], i, "o", color=color, markersize=7)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [METRIC_LABELS.get(r["metric"], r["metric"]) for _, r in sub.iterrows()],
        fontsize=9,
    )
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title(f"Univariate Cox — {endpoint_label}")

    # Annotate p-values
    for i, (_, row) in enumerate(sub.iterrows()):
        p_str = f"p={row['p']:.3f}" if row["p"] >= 0.001 else f"p={row['p']:.1e}"
        ax.text(
            max(row["HR_hi"], 1.0) + 0.05, i,
            f"HR={row['HR']:.2f} {p_str}",
            va="center", fontsize=8,
        )


def plot_multivariate_forest(ax, cph, title):
    """Forest plot from a fitted CoxPHFitter."""
    if cph is None:
        ax.text(0.5, 0.5, "Multivariate model\nnot fitted",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_visible(True)
        return

    summary = cph.summary.copy()
    summary = summary.sort_values("p")

    y_pos = range(len(summary))
    for i, (name, row) in enumerate(summary.iterrows()):
        hr = row["exp(coef)"]
        lo = row["exp(coef) lower 95%"]
        hi = row["exp(coef) upper 95%"]
        p = row["p"]
        color = "#E41A1C" if p < 0.05 else "#666666"
        ax.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax.plot(hr, i, "o", color=color, markersize=7)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    labels = []
    for name in summary.index:
        labels.append(METRIC_LABELS.get(name, name))
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title(title)

    for i, (name, row) in enumerate(summary.iterrows()):
        hr = row["exp(coef)"]
        hi = row["exp(coef) upper 95%"]
        p = row["p"]
        p_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
        ax.text(
            max(hi, 1.0) + 0.05, i,
            f"HR={hr:.2f} {p_str}",
            va="center", fontsize=8,
        )

    ax.text(
        0.02, 0.98,
        f"C-index={cph.concordance_index_:.3f}\nn={cph.summary['n'].iloc[0] if 'n' in cph.summary.columns else '?'}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )


def plot_transformation(ax, df, metric):
    """Boxplot: spatial metric by transformation status."""
    sub = df[[metric, "transformed"]].dropna()
    yes = sub[sub["transformed"] == 1][metric]
    no = sub[sub["transformed"] == 0][metric]

    label = METRIC_LABELS.get(metric, metric)

    bp = ax.boxplot(
        [no, yes], tick_labels=["No", "Yes"],
        patch_artist=True, widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#377EB8")
    bp["boxes"][1].set_facecolor("#E41A1C")
    for b in bp["boxes"]:
        b.set_alpha(0.6)

    # Mann-Whitney
    if len(yes) >= 3 and len(no) >= 3:
        _, u_p = stats.mannwhitneyu(no, yes, alternative="two-sided")
        ax.text(
            0.95, 0.95,
            f"Mann-Whitney p={u_p:.3f}\nn={len(no)}+{len(yes)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Transformation")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs Transformation")


def plot_ct_forest(ax, results_df, endpoint_label, max_show=12):
    """Forest plot of cell type fraction Cox HRs. Show top N by p-value."""
    sub = results_df[results_df["endpoint"] == endpoint_label].copy()
    if sub.empty:
        ax.set_visible(False)
        return

    # Filter to plottable HRs (exclude extreme/unstable)
    sub = sub[sub["HR_hi"] < 1e6].copy()
    sub = sub.nsmallest(max_show, "p")
    sub = sub.sort_values("p", ascending=False)  # bottom = lowest p

    y_pos = range(len(sub))
    for i, (_, row) in enumerate(sub.iterrows()):
        color = "#E41A1C" if row["p"] < 0.05 else "#FF8C00" if row["p"] < 0.1 else "#666666"
        ax.plot(
            [row["HR_lo"], row["HR_hi"]], [i, i],
            color=color, linewidth=2,
        )
        ax.plot(row["HR"], i, "o", color=color, markersize=7)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [CT_SHORT_LABELS.get(r["metric"], r["metric"]) for _, r in sub.iterrows()],
        fontsize=8,
    )
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title(f"Cell type fractions — {endpoint_label}")
    ax.set_xscale("log")

    # Annotate p-values on right
    for i, (_, row) in enumerate(sub.iterrows()):
        p_str = f"p={row['p']:.3f}" if row["p"] >= 0.001 else f"p={row['p']:.1e}"
        ax.annotate(
            p_str, xy=(1.0, i), xycoords=("axes fraction", "data"),
            xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=7, color="#333",
        )

    # Bonferroni line annotation
    n_tests = len(results_df[results_df["endpoint"] == endpoint_label])
    bonf = 0.05 / max(n_tests, 1)
    ax.text(
        0.02, 0.98,
        f"Bonferroni α={bonf:.4f}\n({n_tests} tests × 2 endpoints)",
        transform=ax.transAxes, va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )


def plot_loto(ax, df, metric, time_col, event_col, endpoint_label):
    """Leave-one-TMA-out sensitivity bar chart."""
    results = []
    for tma in sorted(df["tma"].unique()):
        sub = df[df["tma"] != tma]
        res = univariate_cox(sub, metric, time_col, event_col)
        if res:
            results.append({"excl_tma": tma, **res})

    if not results:
        ax.text(0.5, 0.5, "LOTO: insufficient data",
                transform=ax.transAxes, ha="center", va="center")
        return

    loto_df = pd.DataFrame(results)
    y_pos = range(len(loto_df))
    for i, (_, row) in enumerate(loto_df.iterrows()):
        color = TMA_COLORS.get(row["excl_tma"], "#666666")
        ax.plot(
            [row["HR_lo"], row["HR_hi"]], [i, i],
            color=color, linewidth=2.5,
        )
        ax.plot(row["HR"], i, "o", color=color, markersize=8)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"excl. {row['excl_tma']}" for _, row in loto_df.iterrows()],
        fontsize=9,
    )
    ax.set_xlabel("Hazard Ratio (95% CI)")

    label = CT_SHORT_LABELS.get(metric, metric)
    ax.set_title(f"LOTO sensitivity — {label} vs {endpoint_label}")
    ax.set_xscale("log")

    for i, (_, row) in enumerate(loto_df.iterrows()):
        p_str = f"p={row['p']:.3f}" if row["p"] >= 0.001 else f"p={row['p']:.1e}"
        ax.annotate(
            f"HR={row['HR']:.1f} {p_str} (n={row['n']})",
            xy=(1.0, i), xycoords=("axes fraction", "data"),
            xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=8,
        )


def plot_s_panel_forest(ax, s_results, endpoint_label, max_show=15):
    """Forest plot of S-panel marker/fraction Cox HRs."""
    sub = s_results[s_results["endpoint"] == endpoint_label].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No S-panel results",
                transform=ax.transAxes, ha="center", va="center")
        return

    sub = sub[sub["HR_hi"] < 1e6].copy()
    sub = sub.nsmallest(max_show, "p")
    sub = sub.sort_values("p", ascending=False)

    y_pos = range(len(sub))
    for i, (_, row) in enumerate(sub.iterrows()):
        color = "#E41A1C" if row["p"] < 0.05 else "#FF8C00" if row["p"] < 0.1 else "#666666"
        ax.plot([row["HR_lo"], row["HR_hi"]], [i, i], color=color, linewidth=2)
        ax.plot(row["HR"], i, "o", color=color, markersize=7)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(list(y_pos))
    labels = []
    for _, r in sub.iterrows():
        lbl = r["metric"].replace("s_", "").replace("_frac", " (frac)")
        labels.append(lbl)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title(f"S-panel markers — {endpoint_label}")
    ax.set_xscale("log")

    for i, (_, row) in enumerate(sub.iterrows()):
        p_str = f"p={row['p']:.3f}" if row["p"] >= 0.001 else f"p={row['p']:.1e}"
        ax.annotate(
            p_str, xy=(1.0, i), xycoords=("axes fraction", "data"),
            xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=7, color="#333",
        )

    n_tests = len(s_results[s_results["endpoint"] == endpoint_label])
    bonf = 0.05 / max(n_tests, 1)
    ax.text(
        0.02, 0.98,
        f"Bonferroni α={bonf:.4f}\n({n_tests} tests)",
        transform=ax.transAxes, va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )


def plot_cd8_scatter(ax, df):
    """Scatter: CD8 total fraction vs CD8 follicular ratio, colored by TMA."""
    sub = df[["CD8 T cells", "cd8_foll_ratio", "tma"]].dropna()
    for tma in sorted(sub["tma"].unique()):
        t = sub[sub["tma"] == tma]
        ax.scatter(
            t["CD8 T cells"], t["cd8_foll_ratio"],
            c=TMA_COLORS.get(tma, "#666"), label=tma,
            alpha=0.6, s=30, edgecolors="white", linewidth=0.3,
        )
    # Spearman annotation
    rho, p = stats.spearmanr(sub["CD8 T cells"], sub["cd8_foll_ratio"])
    ax.text(
        0.95, 0.95,
        f"Spearman ρ={rho:.2f}\np={p:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("CD8 T cell fraction (total)")
    ax.set_ylabel("CD8 follicular ratio")
    ax.set_title("CD8 quantity vs spatial pattern")
    ax.legend(fontsize=8, loc="lower left")


def plot_pod24_boxplot(ax, df, metric, pod24_res_df):
    """Boxplot: metric by POD24 status."""
    sub = df[[metric, "pod24"]].dropna()
    pod = sub[sub["pod24"] == 1][metric]
    nopod = sub[sub["pod24"] == 0][metric]

    label = CT_SHORT_LABELS.get(metric, METRIC_LABELS.get(metric, metric))
    bp = ax.boxplot(
        [nopod, pod], tick_labels=["non-POD24", "POD24"],
        patch_artist=True, widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#377EB8")
    bp["boxes"][1].set_facecolor("#E41A1C")
    for b in bp["boxes"]:
        b.set_alpha(0.6)

    # Add individual points (jittered)
    for i, (data, color) in enumerate(
        [(nopod, "#377EB8"), (pod, "#E41A1C")], start=1
    ):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
        ax.scatter(
            np.full(len(data), i) + jitter, data,
            c=color, alpha=0.3, s=12, zorder=3,
        )

    # p-value
    row = pod24_res_df[pod24_res_df["metric"] == metric]
    if len(row) > 0:
        p = row.iloc[0]["mw_p"]
        auc = row.iloc[0]["auc"]
        ax.text(
            0.95, 0.95,
            f"Mann-Whitney p={p:.4f}\nAUC={auc:.3f}\n"
            f"n={len(nopod)}+{len(pod)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_ylabel(label)
    ax.set_title(f"{label} vs POD24")


def plot_roc(ax, fpr_mv, tpr_mv, auc_mv, fpr_flipi, tpr_flipi, auc_flipi,
             fpr_cd8, tpr_cd8, auc_cd8, roc_extra=None):
    """ROC curves comparing POD24 prediction models."""
    if fpr_mv is None:
        ax.text(0.5, 0.5, "Insufficient data for ROC",
                transform=ax.transAxes, ha="center", va="center")
        return

    ax.plot(fpr_cd8, tpr_cd8, color="#AAAAAA", linewidth=1.5,
            label=f"CD8 alone (AUC={auc_cd8:.3f})")
    ax.plot(fpr_flipi, tpr_flipi, color="#377EB8", linewidth=1.5,
            label=f"FLIPI only (AUC={auc_flipi:.3f})")
    ax.plot(fpr_mv, tpr_mv, color="#FF8C00", linewidth=1.5, linestyle="--",
            label=f"CD8 + FLIPI (AUC={auc_mv:.3f})")
    # Additional models (CD14-based)
    if roc_extra:
        for name, (fpr, tpr, auc_val, color, lw) in roc_extra.items():
            ax.plot(fpr, tpr, color=color, linewidth=lw, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("POD24 prediction — ROC")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)


def generate_figure(df, ct_results, mv_cph_pfs, cd8_sp_results,
                    mv_cd8_sp, pod24_res_df,
                    fpr_mv, tpr_mv, auc_mv,
                    fpr_flipi, tpr_flipi, auc_flipi,
                    fpr_cd8, tpr_cd8, auc_cd8,
                    output_dir,
                    s_panel_results=None, s_top_marker=None, mv_s_pfs=None,
                    roc_extra=None):
    """Generate 6×2 publication figure: CD8 + S-panel survival."""
    n_rows = 6 if (s_panel_results is not None and len(s_panel_results) > 0) else 5
    fig = plt.figure(figsize=(18, 6 * n_rows))
    gs = GridSpec(
        n_rows, 2, figure=fig, hspace=0.32, wspace=0.45,
        left=0.08, right=0.88, top=0.97, bottom=0.03,
    )

    cd8_col = "CD8 T cells"

    # Row 1: KM plots
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    plot_km(ax_a, df, cd8_col, "pfs_time", "pfs_event", "PFS")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    plot_km(ax_b, df, "cd8_foll_density", "pfs_time", "pfs_event", "PFS")

    # Row 2: Forest plots
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")
    plot_ct_forest(ax_c, ct_results, "PFS")

    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")
    if cd8_sp_results is not None and len(cd8_sp_results) > 0:
        sub = cd8_sp_results.copy()
        sub = sub[sub["HR_hi"] < 1e6]
        sub["label"] = sub.apply(
            lambda r: f"{METRIC_LABELS.get(r['metric'], r['metric'])} ({r['endpoint']})",
            axis=1,
        )
        sub = sub.sort_values("p", ascending=False)
        y_pos = range(len(sub))
        for i, (_, row) in enumerate(sub.iterrows()):
            color = "#E41A1C" if row["p"] < 0.05 else "#FF8C00" if row["p"] < 0.1 else "#666666"
            ax_d.plot([row["HR_lo"], row["HR_hi"]], [i, i], color=color, linewidth=2)
            ax_d.plot(row["HR"], i, "o", color=color, markersize=7)
        ax_d.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_d.set_yticks(list(y_pos))
        ax_d.set_yticklabels([r["label"] for _, r in sub.iterrows()], fontsize=8)
        ax_d.set_xlabel("Hazard Ratio (95% CI)")
        ax_d.set_title("CD8 spatial pattern — Cox PH")
        ax_d.set_xscale("log")
        for i, (_, row) in enumerate(sub.iterrows()):
            p_str = f"p={row['p']:.3f}" if row["p"] >= 0.001 else f"p={row['p']:.1e}"
            ax_d.annotate(
                p_str, xy=(1.0, i), xycoords=("axes fraction", "data"),
                xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=7, color="#333",
            )

    # Row 3: Multivariate models
    ax_e = fig.add_subplot(gs[2, 0])
    panel_label(ax_e, "e")
    plot_multivariate_forest(ax_e, mv_cph_pfs, "Multivariate: CD8 + FLIPI → PFS")

    ax_f = fig.add_subplot(gs[2, 1])
    panel_label(ax_f, "f")
    plot_loto(ax_f, df, cd8_col, "pfs_time", "pfs_event", "PFS")

    # Row 4: POD24
    pod24_df = df[df["pod24"].notna()].copy()

    ax_g = fig.add_subplot(gs[3, 0])
    panel_label(ax_g, "g")
    plot_pod24_boxplot(ax_g, pod24_df, cd8_col, pod24_res_df)

    ax_h = fig.add_subplot(gs[3, 1])
    panel_label(ax_h, "h")
    plot_roc(ax_h, fpr_mv, tpr_mv, auc_mv,
             fpr_flipi, tpr_flipi, auc_flipi,
             fpr_cd8, tpr_cd8, auc_cd8,
             roc_extra=roc_extra)

    # Row 5: CD8 spatial pattern + scatter
    ax_i = fig.add_subplot(gs[4, 0])
    panel_label(ax_i, "i")
    plot_pod24_boxplot(ax_i, pod24_df, "cd8_foll_density", pod24_res_df)

    ax_j = fig.add_subplot(gs[4, 1])
    panel_label(ax_j, "j")
    plot_cd8_scatter(ax_j, df)

    # Row 6: S-panel (if available)
    if n_rows == 6 and s_panel_results is not None:
        ax_k = fig.add_subplot(gs[5, 0])
        panel_label(ax_k, "k")
        plot_s_panel_forest(ax_k, s_panel_results, "PFS")

        ax_l = fig.add_subplot(gs[5, 1])
        panel_label(ax_l, "l")
        if s_top_marker and s_top_marker in df.columns:
            # Add label for S-panel marker
            clean_label = s_top_marker.replace("s_", "S: ").replace("_frac", " (frac)")
            METRIC_LABELS[s_top_marker] = clean_label
            plot_km(ax_l, df, s_top_marker, "pfs_time", "pfs_event", "PFS")
        elif mv_s_pfs is not None:
            plot_multivariate_forest(ax_l, mv_s_pfs,
                                     "Multivariate: S-panel + CD8 → PFS")
        else:
            plot_s_panel_forest(ax_l, s_panel_results, "OS")

    out = Path(output_dir) / "fig_survival.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# ---------------------------------------------------------------------------
# Step 5: LOTO sensitivity
# ---------------------------------------------------------------------------

def loto_sensitivity(df, metric, time_col, event_col):
    """Leave-one-TMA-out sensitivity for a single metric."""
    results = []
    for tma in sorted(df["tma"].unique()):
        sub = df[df["tma"] != tma]
        res = univariate_cox(sub, metric, time_col, event_col)
        if res:
            results.append({"excl_tma": tma, **res})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-panel", required=True, help="T-panel v8 h5ad")
    parser.add_argument("--t-utag", required=True, help="T-panel UTAG h5ad")
    parser.add_argument("--entropy-csv", required=True, help="h6a entropy CSV")
    parser.add_argument("--s-panel", default=None, help="S-panel v8 h5ad (optional)")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    # Step 1: Extract metrics
    print("=" * 60)
    print("STEP 1: Extracting per-ROI spatial metrics")
    print("=" * 60)
    roi_df = extract_metrics(args.t_panel, args.t_utag, args.entropy_csv)

    # Step 1b: S-panel metrics (if provided)
    s_panel_df = None
    if args.s_panel:
        print(f"\n{'=' * 60}")
        print("STEP 1b: Extracting S-panel marker intensities + myeloid fractions")
        print("=" * 60)
        s_panel_df = extract_s_panel_metrics(args.s_panel)

    # Step 2: Patient-level
    print("\n" + "=" * 60)
    print("STEP 2: Building patient-level table")
    print("=" * 60)
    patient_df = build_patient_table(roi_df, s_panel_df=s_panel_df)

    # Save covariate matrix
    out_csv = Path(args.output_dir) / "survival_covariates.csv"
    save_cols = [
        "Patient_ID", "slide_ID", "tma", "n_cells",
        "entropy", "exhausted_cd8_frac", "log_es_ratio", "tfh_frac",
        "follicularity", "cd8_foll_ratio", "cd8_foll_density", "cd8_ifoll_density",
    ]
    # Add cell type fractions
    ct_cols = [c for c in CT_FRACTION_COLS if c in patient_df.columns]
    save_cols.extend(ct_cols)
    # Add S-panel columns if present
    s_cols_in_df = [c for c in patient_df.columns if c.startswith("s_")]
    save_cols.extend(s_cols_in_df)
    save_cols.extend([
        "os_time", "os_event", "pfs_time", "pfs_event",
        "AGE", "SEX", "FLIPI", "FLIPI.1", "flipi_high",
        "ANN ARBOR STAGE", "transformed", "treated", "pod24",
    ])
    patient_df[[c for c in save_cols if c in patient_df.columns]].to_csv(
        out_csv, index=False
    )
    print(f"\n  Covariates saved: {out_csv}")

    # Step 3: Survival models
    print("\n" + "=" * 60)
    print("STEP 3a: Univariate Cox — Tier 1 spatial metrics")
    print("=" * 60)

    tier1_metrics = ["entropy", "exhausted_cd8_frac", "log_es_ratio",
                     "tfh_frac", "follicularity"]
    endpoints = {
        "OS": ("os_time", "os_event"),
        "PFS": ("pfs_time", "pfs_event"),
    }

    tier1_results = run_univariate_screen(patient_df, tier1_metrics, endpoints)
    print("\nTier 1 spatial metrics (univariate):")
    if len(tier1_results) > 0:
        print(tier1_results[["endpoint", "metric", "HR", "p", "concordance", "n"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ))
    else:
        print("  No valid results")

    # Step 3b: Cell type fraction screen
    print(f"\n{'=' * 60}")
    print("STEP 3b: Univariate Cox — cell type fractions")
    print("=" * 60)

    ct_cols = [c for c in CT_FRACTION_COLS if c in patient_df.columns and c != LQ]
    ct_results = run_univariate_screen(patient_df, ct_cols, endpoints)
    print("\nCell type fractions (univariate):")
    if len(ct_results) > 0:
        print(ct_results[["endpoint", "metric", "HR", "p", "concordance", "n"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ))

    # Step 3c: Multivariate Cox with best cell type metric + clinical
    # Find best cell type predictor for PFS (Bonferroni threshold: 0.05/36 = 0.0014)
    ct_pfs = ct_results[ct_results["endpoint"] == "PFS"].copy()
    sig_ct_pfs = ct_pfs[ct_pfs["p"] < 0.05].nsmallest(3, "p")["metric"].tolist()
    if not sig_ct_pfs:
        sig_ct_pfs = ct_pfs.nsmallest(1, "p")["metric"].tolist()

    print(f"\n{'=' * 60}")
    print(f"STEP 3c: Multivariate Cox (PFS) — {sig_ct_pfs} + age + FLIPI")
    print("=" * 60)
    mv_cph_pfs = run_multivariate(patient_df, sig_ct_pfs, "pfs_time", "pfs_event")
    if mv_cph_pfs is not None:
        print(mv_cph_pfs.summary[["exp(coef)", "exp(coef) lower 95%",
                                   "exp(coef) upper 95%", "p"]].to_string(
            float_format=lambda x: f"{x:.4f}"
        ))
        print(f"  C-index: {mv_cph_pfs.concordance_index_:.3f}")

    # Also run multivariate for OS with best cell type metric
    ct_os = ct_results[ct_results["endpoint"] == "OS"].copy()
    sig_ct_os = ct_os[ct_os["p"] < 0.1].nsmallest(2, "p")["metric"].tolist()
    if not sig_ct_os:
        sig_ct_os = ct_os.nsmallest(1, "p")["metric"].tolist()

    print(f"\n{'=' * 60}")
    print(f"STEP 3c': Multivariate Cox (OS) — {sig_ct_os} + age + FLIPI")
    print("=" * 60)
    mv_cph_os = run_multivariate(patient_df, sig_ct_os, "os_time", "os_event")
    if mv_cph_os is not None:
        print(mv_cph_os.summary[["exp(coef)", "exp(coef) lower 95%",
                                  "exp(coef) upper 95%", "p"]].to_string(
            float_format=lambda x: f"{x:.4f}"
        ))
        print(f"  C-index: {mv_cph_os.concordance_index_:.3f}")

    # Transformation — cell type fractions
    print(f"\n{'=' * 60}")
    print("STEP 3d: Transformation analysis — cell type fractions")
    print("=" * 60)
    for m in ct_cols:
        sub = patient_df[[m, "transformed"]].dropna()
        yes = sub[sub["transformed"] == 1][m]
        no = sub[sub["transformed"] == 0][m]
        if len(yes) >= 3 and len(no) >= 3:
            _, p = stats.mannwhitneyu(no, yes, alternative="two-sided")
            if p < 0.1:
                direction = "higher" if yes.median() > no.median() else "lower"
                print(f"  {m:35s}: transformed {direction} "
                      f"(median {yes.median():.4f} vs {no.median():.4f}), p={p:.4f}")

    # LOTO sensitivity for CD8 T cells vs PFS
    cd8_col = "CD8 T cells"
    if cd8_col in patient_df.columns:
        print(f"\n{'=' * 60}")
        print(f"STEP 3e: LOTO sensitivity — {cd8_col} vs PFS")
        print("=" * 60)
        loto_pfs = loto_sensitivity(patient_df, cd8_col, "pfs_time", "pfs_event")
        if len(loto_pfs) > 0:
            print(loto_pfs[["excl_tma", "HR", "p", "n"]].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            ))

        print(f"\n  LOTO — {cd8_col} vs OS")
        loto_os = loto_sensitivity(patient_df, cd8_col, "os_time", "os_event")
        if len(loto_os) > 0:
            print(loto_os[["excl_tma", "HR", "p", "n"]].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            ))

    # Step 3f: CD8 spatial pattern (follicular vs diffuse)
    print(f"\n{'=' * 60}")
    print("STEP 3f: CD8 spatial pattern — follicular vs diffuse infiltration")
    print("=" * 60)

    cd8_spatial_metrics = ["cd8_foll_ratio", "cd8_foll_density", "cd8_ifoll_density"]
    cd8_spatial_labels = {
        "cd8_foll_ratio": "CD8 follicular ratio",
        "cd8_foll_density": "CD8 density (follicular)",
        "cd8_ifoll_density": "CD8 density (interfollicular)",
    }

    # Descriptive stats
    for m in cd8_spatial_metrics:
        if m in patient_df.columns:
            vals = patient_df[m].dropna()
            print(f"  {cd8_spatial_labels[m]:35s}: "
                  f"n={len(vals)}, mean={vals.mean():.4f}, "
                  f"median={vals.median():.4f}, std={vals.std():.4f}")

    # Univariate Cox
    cd8_sp_results = run_univariate_screen(
        patient_df, cd8_spatial_metrics, endpoints
    )
    if len(cd8_sp_results) > 0:
        print("\nCD8 spatial pattern (univariate Cox):")
        print(cd8_sp_results[
            ["endpoint", "metric", "HR", "p", "concordance", "n"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Multivariate: CD8 total + CD8 follicular density + FLIPI → PFS
    mv_cd8_sp = None
    cd8_sp_sig = cd8_sp_results[cd8_sp_results["p"] < 0.2]["metric"].unique().tolist()
    if cd8_sp_sig and cd8_col in patient_df.columns:
        mv_cols = [cd8_col] + [m for m in cd8_sp_sig if m != cd8_col]
        print(f"\n  Multivariate: {mv_cols} + FLIPI → PFS")
        mv_cd8_sp = run_multivariate(patient_df, mv_cols, "pfs_time", "pfs_event")
        if mv_cd8_sp is not None:
            print(mv_cd8_sp.summary[["exp(coef)", "exp(coef) lower 95%",
                                      "exp(coef) upper 95%", "p"]].to_string(
                float_format=lambda x: f"{x:.4f}"
            ))
            print(f"  C-index: {mv_cd8_sp.concordance_index_:.3f}")

    # Correlation between CD8 total and CD8 follicular ratio
    if cd8_col in patient_df.columns and "cd8_foll_ratio" in patient_df.columns:
        sub = patient_df[[cd8_col, "cd8_foll_ratio"]].dropna()
        rho, p = stats.spearmanr(sub[cd8_col], sub["cd8_foll_ratio"])
        print(f"\n  Spearman: CD8 total vs CD8 foll ratio: rho={rho:.3f}, p={p:.4f}")

    # LOTO for best CD8 spatial metric
    best_cd8_sp = cd8_sp_results.nsmallest(1, "p")
    if len(best_cd8_sp) > 0:
        bm = best_cd8_sp.iloc[0]["metric"]
        be = best_cd8_sp.iloc[0]["endpoint"]
        tc, ec = endpoints[be]
        print(f"\n  LOTO — {cd8_spatial_labels.get(bm, bm)} vs {be}")
        loto_sp = loto_sensitivity(patient_df, bm, tc, ec)
        if len(loto_sp) > 0:
            print(loto_sp[["excl_tma", "HR", "p", "n"]].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            ))

    # Step 3g: Compartment-resolved cell type screen
    print(f"\n{'=' * 60}")
    print("STEP 3g: Compartment-resolved cell type screen")
    print("=" * 60)

    # Identify compartment columns in patient_df
    comp_screen_cols = [c for c in patient_df.columns
                        if c.startswith(("foll_", "ifoll_", "ratio_"))]
    print(f"  Screening {len(comp_screen_cols)} compartment metrics × 2 endpoints")

    comp_results = run_univariate_screen(patient_df, comp_screen_cols, endpoints)

    if len(comp_results) > 0:
        # Bonferroni threshold
        n_comp_tests = len(comp_results)
        bonf_comp = 0.05 / max(n_comp_tests, 1)
        print(f"  Bonferroni threshold: {bonf_comp:.5f} ({n_comp_tests} tests)")

        # Show significant + trending results
        sig_comp = comp_results[comp_results["p"] < 0.1].sort_values("p")
        if len(sig_comp) > 0:
            print(f"\n  Compartment metrics with p < 0.1 ({len(sig_comp)} hits):")
            for _, row in sig_comp.iterrows():
                sig = ("***" if row["p"] < 0.001 else "**" if row["p"] < 0.01
                       else "*" if row["p"] < 0.05 else "")
                bonf_flag = " [Bonferroni]" if row["p"] < bonf_comp else ""
                print(f"    {row['endpoint']:3s} | {row['metric']:40s} | "
                      f"HR={row['HR']:.3f} [{row['HR_lo']:.3f}–{row['HR_hi']:.3f}] | "
                      f"p={row['p']:.5f} {sig}{bonf_flag}")
        else:
            print("\n  No compartment metric reached p < 0.1")

        # LOTO for top hit if significant
        top_comp = comp_results.nsmallest(1, "p")
        if len(top_comp) > 0 and top_comp.iloc[0]["p"] < 0.05:
            tc_name = top_comp.iloc[0]["metric"]
            tc_ep = top_comp.iloc[0]["endpoint"]
            tc_time, tc_event = endpoints[tc_ep]
            print(f"\n  LOTO — {tc_name} vs {tc_ep}")
            loto_comp = loto_sensitivity(patient_df, tc_name, tc_time, tc_event)
            if len(loto_comp) > 0:
                print(loto_comp[["excl_tma", "HR", "p", "n"]].to_string(
                    index=False, float_format=lambda x: f"{x:.4f}"
                ))

        # Multivariate: top compartment hit + FLIPI → PFS
        top_pfs_comp = comp_results[
            (comp_results["endpoint"] == "PFS") & (comp_results["p"] < 0.05)
        ]
        if len(top_pfs_comp) > 0:
            top_m = top_pfs_comp.nsmallest(1, "p").iloc[0]["metric"]
            print(f"\n  Multivariate: {top_m} + CD8 T cells + FLIPI → PFS")
            mv_comp = run_multivariate(
                patient_df, [top_m, cd8_col], "pfs_time", "pfs_event"
            )
            if mv_comp is not None:
                print(mv_comp.summary[["exp(coef)", "exp(coef) lower 95%",
                                        "exp(coef) upper 95%", "p"]].to_string(
                    float_format=lambda x: f"{x:.4f}"
                ))
                print(f"  C-index: {mv_comp.concordance_index_:.3f}")
    else:
        comp_results = pd.DataFrame()

    # Step 3i: S-panel marker intensity + myeloid fraction screen
    s_panel_results = pd.DataFrame()
    s_top_marker = None
    mv_s_pfs = None
    if s_panel_df is not None:
        print(f"\n{'=' * 60}")
        print("STEP 3i: S-panel marker intensity + myeloid fraction screen")
        print("=" * 60)

        s_marker_cols = [c for c in patient_df.columns
                         if c.startswith("s_") and patient_df[c].notna().sum() >= 20]
        print(f"  Screening {len(s_marker_cols)} S-panel columns × 2 endpoints")

        s_panel_results = run_univariate_screen(patient_df, s_marker_cols, endpoints)

        if len(s_panel_results) > 0:
            n_s_tests = len(s_panel_results)
            bonf_s = 0.05 / max(n_s_tests, 1)
            print(f"  Bonferroni threshold: {bonf_s:.5f} ({n_s_tests} tests)")

            sig_s = s_panel_results[s_panel_results["p"] < 0.1].sort_values("p")
            print(f"\n  S-panel hits with p < 0.1 ({len(sig_s)}):")
            for _, row in sig_s.iterrows():
                sig = ("***" if row["p"] < 0.001 else "**" if row["p"] < 0.01
                       else "*" if row["p"] < 0.05 else "")
                bonf_flag = " [Bonferroni]" if row["p"] < bonf_s else ""
                label = row["metric"].replace("s_", "").replace("_frac", " (frac)")
                print(f"    {row['endpoint']:3s} | {label:30s} | "
                      f"HR={row['HR']:.3f} [{row['HR_lo']:.3f}–{row['HR_hi']:.3f}] | "
                      f"p={row['p']:.5f} {sig}{bonf_flag}")

            # Identify best S-panel predictor for PFS
            s_pfs = s_panel_results[s_panel_results["endpoint"] == "PFS"]
            if len(s_pfs) > 0:
                s_top_marker = s_pfs.nsmallest(1, "p").iloc[0]["metric"]
                s_top_p = s_pfs.nsmallest(1, "p").iloc[0]["p"]
                print(f"\n  Best S-panel predictor (PFS): {s_top_marker} (p={s_top_p:.5f})")

                # LOTO for best S-panel hit
                if s_top_p < 0.05:
                    print(f"\n  LOTO — {s_top_marker} vs PFS")
                    loto_s = loto_sensitivity(patient_df, s_top_marker,
                                              "pfs_time", "pfs_event")
                    if len(loto_s) > 0:
                        print(loto_s[["excl_tma", "HR", "p", "n"]].to_string(
                            index=False, float_format=lambda x: f"{x:.4f}"
                        ))

                # Multivariate: best S-panel + CD8 + FLIPI → PFS
                if s_top_p < 0.1:
                    mv_s_cols = [s_top_marker, cd8_col]
                    print(f"\n  Multivariate: {s_top_marker} + {cd8_col} + FLIPI → PFS")
                    mv_s_pfs = run_multivariate(
                        patient_df, mv_s_cols, "pfs_time", "pfs_event"
                    )
                    if mv_s_pfs is not None:
                        print(mv_s_pfs.summary[
                            ["exp(coef)", "exp(coef) lower 95%",
                             "exp(coef) upper 95%", "p"]
                        ].to_string(float_format=lambda x: f"{x:.4f}"))
                        print(f"  C-index: {mv_s_pfs.concordance_index_:.3f}")

    # Step 3h: POD24 analysis
    print(f"\n{'=' * 60}")
    print("STEP 3h: POD24 (progression within 24 months)")
    print("=" * 60)

    pod24_df = patient_df[patient_df["pod24"].notna()].copy()
    pod24_df["pod24"] = pod24_df["pod24"].astype(int)
    n_pod = int(pod24_df["pod24"].sum())
    n_nopod = int((pod24_df["pod24"] == 0).sum())
    print(f"  Evaluable patients: {len(pod24_df)} (POD24={n_pod}, non-POD24={n_nopod})")

    # Mann-Whitney for all metrics vs POD24
    pod24_test_metrics = (
        [cd8_col] + cd8_spatial_metrics
        + ["entropy", "follicularity"]
        + [c for c in ct_cols if c != cd8_col]
    )
    pod24_results = []
    for m in pod24_test_metrics:
        if m not in pod24_df.columns:
            continue
        sub = pod24_df[[m, "pod24"]].dropna()
        yes = sub[sub["pod24"] == 1][m]
        no = sub[sub["pod24"] == 0][m]
        if len(yes) < 5 or len(no) < 5:
            continue
        _, mw_p = stats.mannwhitneyu(no, yes, alternative="two-sided")
        # AUC
        try:
            auc = roc_auc_score(sub["pod24"], sub[m])
        except Exception:
            auc = np.nan
        pod24_results.append({
            "metric": m,
            "pod24_median": float(yes.median()),
            "nopod24_median": float(no.median()),
            "mw_p": mw_p,
            "auc": auc,
            "direction": "higher" if yes.median() > no.median() else "lower",
        })

    pod24_res_df = pd.DataFrame(pod24_results).sort_values("mw_p")
    print("\nPOD24 vs metrics (Mann-Whitney, sorted by p):")
    sig_pod = pod24_res_df[pod24_res_df["mw_p"] < 0.2]
    for _, row in sig_pod.iterrows():
        label = CT_SHORT_LABELS.get(row["metric"],
                METRIC_LABELS.get(row["metric"], row["metric"]))
        print(f"  {label:35s}: POD24 {row['direction']} "
              f"({row['pod24_median']:.4f} vs {row['nopod24_median']:.4f}), "
              f"p={row['mw_p']:.4f}, AUC={row['auc']:.3f}")

    # Multivariate logistic regression: CD8 + FLIPI → POD24
    print(f"\n  Logistic regression: {cd8_col} + FLIPI → POD24")
    lr_df = pod24_df[[cd8_col, "flipi_high", "AGE", "pod24"]].dropna()
    roc_extra = None
    if len(lr_df) >= 30:
        from sklearn.preprocessing import StandardScaler
        X = lr_df[[cd8_col, "flipi_high", "AGE"]].values
        y = lr_df["pod24"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(X_scaled, y)
        pred_proba = lr.predict_proba(X_scaled)[:, 1]
        auc_mv = roc_auc_score(y, pred_proba)
        coef_names = [cd8_col, "flipi_high", "AGE"]
        print(f"  AUC (CD8 + FLIPI + age): {auc_mv:.3f}")
        for name, coef in zip(coef_names, lr.coef_[0]):
            or_val = np.exp(coef)
            print(f"    {name:25s}: coef={coef:.3f}, OR={or_val:.2f}")

        # AUC for FLIPI-only model
        lr_flipi = LogisticRegression(penalty=None, max_iter=1000)
        X_flipi = lr_df[["flipi_high"]].values
        lr_flipi.fit(X_flipi, y)
        auc_flipi = roc_auc_score(y, lr_flipi.predict_proba(X_flipi)[:, 1])
        print(f"  AUC (FLIPI only): {auc_flipi:.3f}")

        # ROC curves for figure
        fpr_mv, tpr_mv, _ = roc_curve(y, pred_proba)
        fpr_flipi, tpr_flipi, _ = roc_curve(y, lr_flipi.predict_proba(X_flipi)[:, 1])
        # CD8-only AUC
        auc_cd8 = roc_auc_score(y, lr_df[cd8_col])
        fpr_cd8, tpr_cd8, _ = roc_curve(y, lr_df[cd8_col])

        # CD14-based models (requires S-panel CD14)
        roc_extra = {}
        s_cd14_col = "s_CD14"
        if s_cd14_col in pod24_df.columns:
            lr3_df = pod24_df[
                [cd8_col, s_cd14_col, "flipi_high", "pod24"]
            ].dropna()
            if len(lr3_df) >= 30:
                y3 = lr3_df["pod24"].values
                print(f"\n  CD14-based logistic models (n={len(lr3_df)}):")

                # CD14 + FLIPI
                X_cf = lr3_df[[s_cd14_col, "flipi_high"]].values
                lr_cf = LogisticRegression(max_iter=1000)
                lr_cf.fit(X_cf, y3)
                prob_cf = lr_cf.predict_proba(X_cf)[:, 1]
                auc_cf = roc_auc_score(y3, prob_cf)
                fpr_cf, tpr_cf, _ = roc_curve(y3, prob_cf)
                print(f"    CD14 + FLIPI:           AUC={auc_cf:.3f}")
                roc_extra["CD14 + FLIPI"] = (
                    fpr_cf, tpr_cf, auc_cf, "#984EA3", 2,
                )

                # CD14 + CD8 + FLIPI (triple)
                X_triple = lr3_df[
                    [cd8_col, s_cd14_col, "flipi_high"]
                ].values
                lr_triple = LogisticRegression(max_iter=1000)
                lr_triple.fit(X_triple, y3)
                prob_triple = lr_triple.predict_proba(X_triple)[:, 1]
                auc_triple = roc_auc_score(y3, prob_triple)
                fpr_triple, tpr_triple, _ = roc_curve(y3, prob_triple)
                print(f"    CD14 + CD8 + FLIPI:     AUC={auc_triple:.3f}")
                roc_extra["CD14 + CD8 + FLIPI"] = (
                    fpr_triple, tpr_triple, auc_triple, "#E41A1C", 2.5,
                )

                # Odds ratios for triple model
                for name, coef in zip(
                    [cd8_col, s_cd14_col, "flipi_high"],
                    lr_triple.coef_[0],
                ):
                    print(f"      {name:25s}: OR={np.exp(coef):.2f}")

                # 5-fold CV AUC for triple
                from sklearn.model_selection import (
                    StratifiedKFold, cross_val_score,
                )
                cv = StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=42,
                )
                cv_aucs = cross_val_score(
                    LogisticRegression(max_iter=1000),
                    X_triple, y3, cv=cv, scoring="roc_auc",
                )
                print(f"    CD14+CD8+FLIPI 5-fold CV AUC: "
                      f"{cv_aucs.mean():.3f} +/- {cv_aucs.std():.3f}")
    else:
        fpr_mv = tpr_mv = fpr_flipi = tpr_flipi = fpr_cd8 = tpr_cd8 = None
        auc_mv = auc_flipi = auc_cd8 = np.nan

    # LOTO for POD24
    print(f"\n  LOTO — {cd8_col} vs POD24 (Mann-Whitney)")
    for tma in sorted(pod24_df["tma"].unique()):
        sub = pod24_df[pod24_df["tma"] != tma]
        yes = sub[sub["pod24"] == 1][cd8_col]
        no = sub[sub["pod24"] == 0][cd8_col]
        if len(yes) >= 3 and len(no) >= 3:
            _, p = stats.mannwhitneyu(no, yes, alternative="two-sided")
            print(f"    excl. {tma}: p={p:.4f} (n={len(sub)}, "
                  f"POD24 median={yes.median():.4f} vs {no.median():.4f})")

    # Step 4: Figure
    print(f"\n{'=' * 60}")
    print("STEP 4: Generating figure")
    print("=" * 60)
    generate_figure(patient_df, ct_results, mv_cph_pfs, cd8_sp_results,
                    mv_cd8_sp, pod24_res_df,
                    fpr_mv, tpr_mv, auc_mv,
                    fpr_flipi, tpr_flipi, auc_flipi,
                    fpr_cd8, tpr_cd8, auc_cd8,
                    args.output_dir,
                    s_panel_results=s_panel_results,
                    s_top_marker=s_top_marker,
                    mv_s_pfs=mv_s_pfs,
                    roc_extra=roc_extra)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY — Tier 1 spatial metrics")
    print("=" * 60)
    for _, row in tier1_results.iterrows():
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"  {row['endpoint']:3s} | {METRIC_LABELS.get(row['metric'], row['metric']):30s} | "
              f"HR={row['HR']:.3f} [{row['HR_lo']:.2f}–{row['HR_hi']:.2f}] | "
              f"p={row['p']:.4f} {sig}")

    print(f"\n{'=' * 60}")
    print("SUMMARY — Cell type fractions (p < 0.1 only)")
    print("=" * 60)
    sig_ct = ct_results[ct_results["p"] < 0.1].sort_values("p")
    for _, row in sig_ct.iterrows():
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        label = CT_SHORT_LABELS.get(row["metric"], row["metric"])
        print(f"  {row['endpoint']:3s} | {label:30s} | "
              f"HR={row['HR']:.3f} [{row['HR_lo']:.2f}–{row['HR_hi']:.2f}] | "
              f"p={row['p']:.4f} {sig}")

    if len(s_panel_results) > 0:
        print(f"\n{'=' * 60}")
        print("SUMMARY — S-panel markers (p < 0.05 only)")
        print("=" * 60)
        sig_s_final = s_panel_results[s_panel_results["p"] < 0.05].sort_values("p")
        for _, row in sig_s_final.iterrows():
            sig = ("***" if row["p"] < 0.001 else "**" if row["p"] < 0.01
                   else "*" if row["p"] < 0.05 else "")
            label = row["metric"].replace("s_", "").replace("_frac", " (frac)")
            print(f"  {row['endpoint']:3s} | {label:30s} | "
                  f"HR={row['HR']:.3f} [{row['HR_lo']:.2f}–{row['HR_hi']:.2f}] | "
                  f"p={row['p']:.4f} {sig}")


if __name__ == "__main__":
    main()
