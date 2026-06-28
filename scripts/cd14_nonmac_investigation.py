"""Investigate CD14 survival signal on non-macrophage cells in S-panel."""

import h5py
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S_H5AD = "output/all_TMA_S_global_v8.h5ad"
COVAR_CSV = "output/hypotheses_v8/survival_covariates.csv"

MYELOID_TYPES = {
    "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Myeloid (S100A9+)", "Dendritic cells", "pDC",
}
LQ = "Low quality / Unassigned"

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

def load_array(f, key):
    ds = f["obs"][key]
    if isinstance(ds, h5py.Group) and "categories" in ds:
        cats = ds["categories"][:]
        codes = ds["codes"][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading S-panel h5ad...")
f = h5py.File(S_H5AD, "r")

var_names = [v.decode() if isinstance(v, bytes) else str(v) for v in f["var"]["_index"][:]]
cd14_idx = var_names.index("CD14")

sids = load_array(f, "sample_id")
ctypes = load_array(f, "cell_type")
X = f["X"]  # dense float32, (2.06M, 39)

# Identify tumor-only, non-Biomax ROIs
unique_rois = sorted(set(sids))
tumor_rois = [r for r in unique_rois if is_tumor_core(r) and not r.startswith("Biomax")]
print(f"  {len(tumor_rois)} tumor ROIs (excl Biomax + controls)")

# Build masks
tumor_mask = np.isin(sids, tumor_rois)
tumor_ctypes = ctypes[tumor_mask]
tumor_sids = sids[tumor_mask]
tumor_indices = np.where(tumor_mask)[0]
n_tumor = tumor_mask.sum()
print(f"  {n_tumor:,} tumor cells")

# Read CD14 for all tumor cells (column slice is fast on dense)
print("  Reading CD14 column...")
cd14_all = X[tumor_indices, cd14_idx]
print(f"  CD14 range: [{cd14_all.min():.3f}, {cd14_all.max():.3f}], mean={cd14_all.mean():.4f}")

# ---------------------------------------------------------------------------
# 1. CD14 expression by cell type
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. CD14 EXPRESSION BY CELL TYPE")
print("=" * 70)

unique_ct = sorted(set(tumor_ctypes))
rows = []
for ct in unique_ct:
    mask_ct = tumor_ctypes == ct
    vals = cd14_all[mask_ct]
    n = len(vals)
    rows.append({
        "cell_type": ct,
        "n_cells": n,
        "frac_of_total": n / n_tumor,
        "mean_CD14": float(vals.mean()),
        "median_CD14": float(np.median(vals)),
        "p90_CD14": float(np.percentile(vals, 90)),
        "frac_gt1": float((vals > 1.0).mean()),
        "is_myeloid": ct in MYELOID_TYPES,
    })

ct_df = pd.DataFrame(rows).sort_values("mean_CD14", ascending=False)
print(f"\n{'Cell type':<30s} {'N':>8s} {'%total':>7s} {'mean':>7s} {'med':>7s} {'p90':>7s} {'%>1.0':>7s} {'myeloid':>7s}")
print("-" * 90)
for _, r in ct_df.iterrows():
    tag = " <<<" if r["is_myeloid"] else ""
    print(f"{r['cell_type']:<30s} {r['n_cells']:>8,d} {r['frac_of_total']:>7.1%} "
          f"{r['mean_CD14']:>7.3f} {r['median_CD14']:>7.3f} {r['p90_CD14']:>7.3f} "
          f"{r['frac_gt1']:>7.1%} {str(r['is_myeloid']):>7s}{tag}")

# ---------------------------------------------------------------------------
# 2. CD14 distribution on top 5 non-myeloid cell types
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. CD14 DISTRIBUTION ON TOP NON-MYELOID CELL TYPES")
print("=" * 70)

non_myel = ct_df[~ct_df["is_myeloid"] & (ct_df["cell_type"] != LQ)].head(5)
for _, r in non_myel.iterrows():
    ct = r["cell_type"]
    vals = cd14_all[tumor_ctypes == ct]
    sk = float(stats.skew(vals))
    ku = float(stats.kurtosis(vals))
    p10, p25, p50, p75, p90, p99 = np.percentile(vals, [10, 25, 50, 75, 90, 99])
    frac_zero = float((vals <= 0.05).mean())
    print(f"\n  {ct} (n={len(vals):,d})")
    print(f"    Percentiles: p10={p10:.3f} p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} p90={p90:.3f} p99={p99:.3f}")
    print(f"    Skewness={sk:.3f}, Kurtosis={ku:.3f}")
    print(f"    Fraction near-zero (<=0.05): {frac_zero:.1%}")
    print(f"    Fraction >1.0: {r['frac_gt1']:.1%}")
    # Bimodality: simple test - if median << mean and high skew, it's right-tailed (not bimodal)
    if frac_zero > 0.4 and r["frac_gt1"] > 0.05:
        print(f"    -> Possibly bimodal: {frac_zero:.0%} near-zero + {r['frac_gt1']:.0%} above 1.0")
    elif sk > 1.0:
        print(f"    -> Right-skewed (long positive tail), not clearly bimodal")
    else:
        print(f"    -> Approximately unimodal/symmetric shift")

# ---------------------------------------------------------------------------
# 3. Per-ROI CD14 decomposition
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. PER-ROI CD14 DECOMPOSITION (macrophage vs non-macrophage)")
print("=" * 70)

is_myeloid = np.array([ct in MYELOID_TYPES for ct in tumor_ctypes])
roi_rows = []
for roi in tumor_rois:
    roi_mask = tumor_sids == roi
    n_roi = roi_mask.sum()
    if n_roi < 50:
        continue
    roi_cd14 = cd14_all[roi_mask]
    roi_myel = is_myeloid[roi_mask]

    n_mac = roi_myel.sum()
    n_nonmac = n_roi - n_mac

    mean_all = float(roi_cd14.mean())
    mean_mac = float(roi_cd14[roi_myel].mean()) if n_mac > 0 else 0.0
    mean_nonmac = float(roi_cd14[~roi_myel].mean()) if n_nonmac > 0 else 0.0

    # Weighted contribution: total CD14 = (n_mac/n)*mean_mac + (n_nonmac/n)*mean_nonmac
    contrib_mac = (n_mac / n_roi) * mean_mac
    contrib_nonmac = (n_nonmac / n_roi) * mean_nonmac
    total_signal = contrib_mac + contrib_nonmac
    frac_mac_signal = contrib_mac / total_signal if total_signal > 0 else 0

    roi_rows.append({
        "roi": roi,
        "n_cells": n_roi,
        "n_mac": n_mac,
        "n_nonmac": n_nonmac,
        "mac_frac": n_mac / n_roi,
        "mean_all": mean_all,
        "mean_mac": mean_mac,
        "mean_nonmac": mean_nonmac,
        "contrib_mac": contrib_mac,
        "contrib_nonmac": contrib_nonmac,
        "frac_mac_signal": frac_mac_signal,
    })

roi_df = pd.DataFrame(roi_rows)
print(f"\n  {len(roi_df)} ROIs with >=50 cells")
print(f"\n  Macrophage fraction per ROI: mean={roi_df['mac_frac'].mean():.3f}, median={roi_df['mac_frac'].median():.3f}")
print(f"  Mean CD14 (all): mean={roi_df['mean_all'].mean():.4f}")
print(f"  Mean CD14 (mac only): mean={roi_df['mean_mac'].mean():.4f}")
print(f"  Mean CD14 (non-mac only): mean={roi_df['mean_nonmac'].mean():.4f}")
print(f"\n  Fraction of per-ROI CD14 signal from macrophages:")
print(f"    mean={roi_df['frac_mac_signal'].mean():.3f}, median={roi_df['frac_mac_signal'].median():.3f}")
print(f"    min={roi_df['frac_mac_signal'].min():.3f}, max={roi_df['frac_mac_signal'].max():.3f}")
print(f"  -> {1 - roi_df['frac_mac_signal'].mean():.1%} of per-ROI CD14 signal comes from NON-macrophage cells")

# ---------------------------------------------------------------------------
# 4. Spillover check: non-mac CD14 vs macrophage fraction
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. SPILLOVER CHECK: non-mac CD14 vs macrophage fraction per ROI")
print("=" * 70)

rho, p = stats.spearmanr(roi_df["mac_frac"], roi_df["mean_nonmac"])
print(f"\n  Spearman: rho={rho:.4f}, p={p:.4g}")
if p < 0.05 and rho > 0.3:
    print(f"  -> SIGNIFICANT positive correlation: higher macrophage ROIs have higher non-mac CD14")
    print(f"     This is consistent with segmentation spillover OR shared microenvironment signals")
elif p < 0.05 and rho < -0.3:
    print(f"  -> SIGNIFICANT negative correlation: anti-spillover pattern")
else:
    print(f"  -> Weak/no correlation: non-mac CD14 is largely independent of macrophage density")
    print(f"     This argues AGAINST segmentation spillover as the primary explanation")

# Also check: is non-mac CD14 correlated with mac CD14?
rho2, p2 = stats.spearmanr(roi_df["mean_mac"], roi_df["mean_nonmac"])
print(f"\n  Spearman (mac CD14 vs non-mac CD14): rho={rho2:.4f}, p={p2:.4g}")

# Partial correlation: non-mac CD14 controlling for mac fraction
from numpy.linalg import lstsq
def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    # Residualize x and y on z
    A = np.column_stack([rz, np.ones(len(rz))])
    rx_res = rx - A @ lstsq(A, rx, rcond=None)[0]
    ry_res = ry - A @ lstsq(A, ry, rcond=None)[0]
    return stats.pearsonr(rx_res, ry_res)

rho_partial, p_partial = partial_spearman(
    roi_df["mean_nonmac"].values, roi_df["mean_all"].values, roi_df["mac_frac"].values
)
print(f"\n  Partial Spearman (non-mac CD14 ~ overall CD14 | mac fraction):")
print(f"    rho={rho_partial:.4f}, p={p_partial:.4g}")

# ---------------------------------------------------------------------------
# 5. Which non-mac cell type drives the survival-associated CD14?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. WHICH NON-MAC CELL TYPE DRIVES THE SURVIVAL-ASSOCIATED CD14?")
print("=" * 70)

top3_nonmac = non_myel.head(3)["cell_type"].tolist()
print(f"\n  Top 3 non-myeloid cell types by CD14: {top3_nonmac}")

# Compute per-ROI mean CD14 restricted to each cell type
for ct in top3_nonmac:
    ct_mean_per_roi = {}
    ct_n_per_roi = {}
    for roi in tumor_rois:
        roi_mask = tumor_sids == roi
        ct_mask = (tumor_ctypes == ct) & roi_mask
        n = ct_mask.sum()
        if n >= 5:
            ct_mean_per_roi[roi] = float(cd14_all[ct_mask].mean())
            ct_n_per_roi[roi] = n

    # Correlate with overall ROI mean CD14
    common = set(ct_mean_per_roi.keys()) & set(roi_df["roi"])
    if len(common) < 10:
        print(f"\n  {ct}: only {len(common)} ROIs with >=5 cells — skipping")
        continue

    sub_roi = roi_df[roi_df["roi"].isin(common)].set_index("roi")
    ct_vals = [ct_mean_per_roi[r] for r in sub_roi.index]
    overall_vals = sub_roi["mean_all"].values

    rho_ct, p_ct = stats.spearmanr(ct_vals, overall_vals)
    rho_nm, p_nm = stats.spearmanr(ct_vals, sub_roi["mean_nonmac"].values)

    # Also check: what fraction of non-mac cells does this type represent?
    type_frac = sum(ct_n_per_roi[r] for r in common) / sum(
        sub_roi.loc[r, "n_nonmac"] for r in common
    )

    print(f"\n  {ct} ({len(common)} ROIs, {type_frac:.1%} of non-mac cells):")
    print(f"    Spearman with overall mean CD14:  rho={rho_ct:.4f}, p={p_ct:.4g}")
    print(f"    Spearman with non-mac mean CD14:  rho={rho_nm:.4f}, p={p_nm:.4g}")

# Also show the contribution decomposition by cell type
print("\n\n  CD14 signal decomposition by cell type (across all tumor cells):")
total_cd14 = float(cd14_all.sum())
print(f"  {'Cell type':<30s} {'% cells':>8s} {'% CD14':>8s} {'ratio':>8s}")
print(f"  {'-'*60}")
for _, r in ct_df.iterrows():
    ct = r["cell_type"]
    mask_ct = tumor_ctypes == ct
    ct_cd14_sum = float(cd14_all[mask_ct].sum())
    pct_cells = r["frac_of_total"]
    pct_cd14 = ct_cd14_sum / total_cd14
    ratio = pct_cd14 / pct_cells if pct_cells > 0 else 0
    tag = " <<<" if r["is_myeloid"] else ""
    print(f"  {ct:<30s} {pct_cells:>8.1%} {pct_cd14:>8.1%} {ratio:>8.2f}{tag}")

f.close()
print("\nDone.")
