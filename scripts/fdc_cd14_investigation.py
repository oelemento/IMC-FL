#!/usr/bin/env python
"""Investigate CD14+ FDCs in follicular lymphoma S-panel data."""
import h5py
import numpy as np
from scipy import stats
from scipy.spatial import KDTree
from collections import Counter

def load_array(f, key):
    ds = f["obs"][key]
    if isinstance(ds, h5py.Group) and "categories" in ds:
        cats = ds["categories"][:]
        codes = ds["codes"][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])

def is_tumor_core(sample_id):
    s = sample_id.lower()
    if "_ton_" in s or "_adr_" in s: return False
    for tissue in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if tissue in s: return False
    if sample_id == "Biomax_ROI_006": return False
    return True

print("Loading data...")
f = h5py.File("output/all_TMA_S_global_v8.h5ad", "r")
X = f["X"][:]  # (N, 39) float32
markers = [v.decode() for v in f["var"]["_index"][:]]
cell_types = load_array(f, "cell_type")
sample_ids = load_array(f, "sample_id")
cx = f["obs"]["centroid_x"][:]
cy = f["obs"]["centroid_y"][:]
f.close()

marker_idx = {m: i for i, m in enumerate(markers)}
skip = {"DNA1", "DNA2", "HistoneH3"}

# Filter: tumor cores, exclude Biomax
tumor_mask = np.array([is_tumor_core(s) and not s.startswith("Biomax") for s in sample_ids])
print(f"Total cells: {len(cell_types):,}, Tumor (no Biomax): {tumor_mask.sum():,}")

X = X[tumor_mask]; cell_types = cell_types[tumor_mask]
sample_ids = sample_ids[tumor_mask]; cx = cx[tumor_mask]; cy = cy[tumor_mask]

fdc_mask = cell_types == "FDC"
cd14_col = marker_idx["CD14"]
print(f"FDC cells: {fdc_mask.sum():,}")

# ── Analysis 1: CD14-high vs CD14-low FDC marker profiles ──
print("\n" + "="*70)
print("ANALYSIS 1: CD14-high vs CD14-low FDC marker profiles")
print("="*70)
fdc_cd14 = X[fdc_mask, cd14_col]
q25, q75 = np.percentile(fdc_cd14, [25, 75])
hi_mask = fdc_cd14 >= q75
lo_mask = fdc_cd14 <= q25
print(f"CD14 Q25={q25:.3f}, Q75={q75:.3f}, Hi={hi_mask.sum():,}, Lo={lo_mask.sum():,}")

fdc_X = X[fdc_mask]
highlight = {"CXCL13","CXCL12","HLA_Class_I","CD21","VISTA","BCL_2","Ki-67",
             "CD47","CD68","S100A9","CD163","CD206","CD11c","PDPN","Vimentin",
             "CD34","CD31","CD14","HLA_DR","CD20","CD44","Fibronectin","CD11b",
             "IDO","CD4","CD8a","PAX5","CD146","CD209","CCL21","CD123","CD49a",
             "SOX9","CD1a","p_H3s28","PD_L1","BCL_6"}
results = []
for i, m in enumerate(markers):
    if m in skip: continue
    hi_vals = fdc_X[hi_mask, i]
    lo_vals = fdc_X[lo_mask, i]
    diff = hi_vals.mean() - lo_vals.mean()
    u_stat, pval = stats.mannwhitneyu(hi_vals, lo_vals, alternative="two-sided")
    results.append((m, hi_vals.mean(), lo_vals.mean(), diff, pval))

results.sort(key=lambda x: -abs(x[3]))
print(f"\n{'Marker':<16} {'CD14-hi':>8} {'CD14-lo':>8} {'Diff':>8} {'p-value':>12} {'Note':>6}")
print("-"*62)
for m, hi, lo, d, p in results:
    tag = " ***" if m in highlight else ""
    print(f"{m:<16} {hi:8.3f} {lo:8.3f} {d:+8.3f} {p:12.2e}{tag}")

# ── Analysis 2: Per-ROI FDC CD14 vs cell type composition ──
print("\n" + "="*70)
print("ANALYSIS 2: Per-ROI FDC CD14 level vs cell type composition")
print("="*70)
unique_rois = np.unique(sample_ids)
ct_list = sorted(set(cell_types) - {"Low quality / Unassigned"})
roi_fdc_cd14 = {}
roi_ct_frac = {}
for roi in unique_rois:
    rmask = sample_ids == roi
    n_total = rmask.sum()
    fdc_in_roi = rmask & fdc_mask
    if fdc_in_roi.sum() < 10: continue
    roi_fdc_cd14[roi] = X[fdc_in_roi, cd14_col].mean()
    cts = cell_types[rmask]
    ct_counts = Counter(cts)
    for ct in ct_list:
        roi_ct_frac.setdefault(ct, {})[roi] = ct_counts.get(ct, 0) / n_total

common_rois = sorted(roi_fdc_cd14.keys())
fdc_cd14_arr = np.array([roi_fdc_cd14[r] for r in common_rois])
print(f"ROIs with >=10 FDCs: {len(common_rois)}")
print(f"\n{'Cell type':<28} {'rho':>6} {'p-value':>12} {'n':>4}")
print("-"*54)
corr_results = []
for ct in ct_list:
    fracs = np.array([roi_ct_frac[ct].get(r, 0) for r in common_rois])
    rho, p = stats.spearmanr(fdc_cd14_arr, fracs)
    corr_results.append((ct, rho, p, len(common_rois)))
corr_results.sort(key=lambda x: -abs(x[1]))
for ct, rho, p, n in corr_results:
    sig = " *" if p < 0.05 else "  "
    sig = " **" if p < 0.01 else sig
    sig = "***" if p < 0.001 else sig
    print(f"{ct:<28} {rho:+6.3f} {p:12.4e} {n:4d} {sig}")

# ── Analysis 3: Spatial neighborhood of CD14+ FDCs ──
print("\n" + "="*70)
print("ANALYSIS 3: Spatial neighborhood of CD14+ vs CD14- FDCs")
print("="*70)
fdc_counts_per_roi = Counter(sample_ids[fdc_mask])
candidate_rois = [(r, c) for r, c in fdc_counts_per_roi.items() if c >= 500]
candidate_rois.sort(key=lambda x: -x[1])
sel_rois = [r for r, c in candidate_rois[:8]]
print(f"Selected ROIs ({len(sel_rois)}): {sel_rois}")

neighbor_types_hi = Counter()
neighbor_types_lo = Counter()
n_hi_total = 0; n_lo_total = 0
for roi in sel_rois:
    rmask = sample_ids == roi
    roi_idx = np.where(rmask)[0]
    roi_cx = cx[roi_idx]; roi_cy = cy[roi_idx]
    roi_ct = cell_types[roi_idx]
    roi_is_fdc = roi_ct == "FDC"
    roi_cd14 = X[roi_idx, cd14_col]
    fdc_local = np.where(roi_is_fdc)[0]
    non_fdc_local = np.where(~roi_is_fdc)[0]
    if len(non_fdc_local) < 20: continue
    tree = KDTree(np.column_stack([roi_cx[non_fdc_local], roi_cy[non_fdc_local]]))
    fdc_coords = np.column_stack([roi_cx[fdc_local], roi_cy[fdc_local]])
    dists, idxs = tree.query(fdc_coords, k=10)
    fdc_cd14_vals = roi_cd14[fdc_local]
    q25_r, q75_r = np.percentile(fdc_cd14_vals, [25, 75])
    for j, fi in enumerate(fdc_local):
        nbr_cts = roi_ct[non_fdc_local[idxs[j]]]
        if fdc_cd14_vals[j] >= q75_r:
            neighbor_types_hi.update(nbr_cts); n_hi_total += 10
        elif fdc_cd14_vals[j] <= q25_r:
            neighbor_types_lo.update(nbr_cts); n_lo_total += 10

all_nbr_types = sorted(set(neighbor_types_hi.keys()) | set(neighbor_types_lo.keys()))
print(f"\nCD14-hi FDC neighbors: {n_hi_total:,} total, CD14-lo: {n_lo_total:,}")
print(f"\n{'Cell type':<28} {'Hi-frac':>8} {'Lo-frac':>8} {'Diff':>8} {'p-value':>12}")
print("-"*68)
nbr_results = []
for ct in all_nbr_types:
    hi_n = neighbor_types_hi.get(ct, 0)
    lo_n = neighbor_types_lo.get(ct, 0)
    hi_frac = hi_n / n_hi_total if n_hi_total > 0 else 0
    lo_frac = lo_n / n_lo_total if n_lo_total > 0 else 0
    # Proportion z-test
    n1, n2 = n_hi_total, n_lo_total
    p_pool = (hi_n + lo_n) / (n1 + n2) if (n1+n2) > 0 else 0
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if p_pool > 0 and p_pool < 1 else 1
    z = (hi_frac - lo_frac) / se if se > 0 else 0
    pval = 2 * stats.norm.sf(abs(z))
    nbr_results.append((ct, hi_frac, lo_frac, hi_frac - lo_frac, pval))
nbr_results.sort(key=lambda x: -abs(x[3]))
for ct, hf, lf, d, p in nbr_results:
    sig = "***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else "   "))
    print(f"{ct:<28} {hf:8.4f} {lf:8.4f} {d:+8.4f} {p:12.2e} {sig}")

# ── Analysis 4: Per-ROI FDC CD14 vs survival markers ──
print("\n" + "="*70)
print("ANALYSIS 4: Per-ROI FDC CD14 vs survival-associated markers")
print("="*70)
surv_markers = ["CD68", "S100A9", "CD8a"]
roi_marker_means = {m: {} for m in surv_markers}
for roi in common_rois:
    rmask = sample_ids == roi
    for m in surv_markers:
        roi_marker_means[m][roi] = X[rmask, marker_idx[m]].mean()

print(f"\n{'Marker':<12} {'rho':>8} {'p-value':>12} {'n':>4}")
print("-"*40)
for m in surv_markers:
    vals = np.array([roi_marker_means[m][r] for r in common_rois])
    rho, p = stats.spearmanr(fdc_cd14_arr, vals)
    sig = "***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else "   "))
    print(f"{m:<12} {rho:+8.3f} {p:12.4e} {len(common_rois):4d} {sig}")

# Also correlate with FDC fraction itself
roi_fdc_frac = {}
for roi in common_rois:
    rmask = sample_ids == roi
    roi_fdc_frac[roi] = (cell_types[rmask] == "FDC").sum() / rmask.sum()
fdc_frac_arr = np.array([roi_fdc_frac[r] for r in common_rois])
rho, p = stats.spearmanr(fdc_cd14_arr, fdc_frac_arr)
print(f"{'FDC frac':<12} {rho:+8.3f} {p:12.4e} {len(common_rois):4d}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Done. See results above.")
