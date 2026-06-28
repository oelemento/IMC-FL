#!/usr/bin/env python3
"""Test Radtke et al. 2024 findings in our IMC data.

Three claims:
1. Reduced follicle size in high-risk patients
2. Loss of FDC networks in high-risk patients
3. Expanded stromal communities in high-risk patients
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clinical_linkage import load_clinical, normalize_sample_id


EXCLUDE_PAT = r'(?i)tonsil|prostate|kidney|spleen|adrenal|_ton_|_adr_'

# S-panel compartment names (from compartment_name column)
S_FOLL = ['B cell zone (BCL2+)', 'B cell zone (PAX5+)', 'FDC network zone']
S_FDC = ['FDC network zone', 'FDC / myeloid zone']
S_STROMAL = ['Stromal / CAF zone']

# S-panel cell types
S_B_CELLS = ['B cells', 'B cells (BCL2+)', 'B cells (PAX5+)']
S_FDC_CT = ['FDC']
S_STROMAL_CT = ['Stromal / CAF', 'FRC (PDPN+)']
S_ENDOTHELIAL_CT = ['Endothelial']
S_MYELOID_CT = ['M1 Macrophages', 'M2 Macrophages', 'Macrophages', 'Myeloid (S100A9+)']


def get_comp_fracs(utag, comp_names):
    """Per-ROI fraction of cells in specified compartments."""
    roi_col = 'sample_id'
    comp_col = 'compartment_name'
    mask = ~utag.obs[roi_col].str.contains(EXCLUDE_PAT, na=False)
    obs = utag.obs[mask]
    total = obs.groupby(roi_col, observed=True).size()
    in_comps = obs[obs[comp_col].isin(comp_names)].groupby(roi_col, observed=True).size()
    return (in_comps / total).fillna(0)


def get_celltype_frac(adata, celltypes):
    """Per-ROI fraction of typed cells in specified cell types."""
    roi_col = 'sample_id'
    ct_col = 'cell_type'
    mask = ~adata.obs[roi_col].str.contains(EXCLUDE_PAT, na=False)
    obs = adata.obs[mask]
    typed = obs[~obs[ct_col].isin(['Low quality / Unassigned', 'Other'])]
    total = typed.groupby(roi_col, observed=True).size()
    matching = typed[typed[ct_col].isin(celltypes)].groupby(roi_col, observed=True).size()
    return (matching / total).fillna(0)


def link_clinical(metric_series, clinical_df, metric_name='metric'):
    """Link per-ROI metric to clinical data."""
    df = metric_series.to_frame(metric_name)
    df['slide_ID'] = [normalize_sample_id(s) for s in df.index]
    clin_map = {
        'Overall survival (y)': 'os_years', 'CODE_OS': 'os_event',
        'Progression free survival (y)': 'pfs_years', 'CODE_PFS': 'pfs_event',
        'Transformation': 'transformation',
    }
    keep = [c for c in clin_map if c in clinical_df.columns]
    clin = clinical_df[['slide_ID'] + keep].drop_duplicates('slide_ID', keep='first')
    merged = df.merge(clin, on='slide_ID', how='inner').rename(columns=clin_map)
    # Transformation: "Yes" → 1, NaN → 0
    if 'transformation' in merged.columns:
        merged['transformation'] = (merged['transformation'] == 'Yes').astype(int)
    return merged


def cox_test(merged, metric_name, time_col, event_col):
    sub = merged[[metric_name, time_col, event_col]].dropna()
    if len(sub) < 20 or sub[event_col].sum() < 5:
        return None
    sub = sub.copy()
    sub[metric_name] = (sub[metric_name] - sub[metric_name].mean()) / sub[metric_name].std()
    cph = CoxPHFitter()
    try:
        cph.fit(sub, duration_col=time_col, event_col=event_col)
        hr = float(np.exp(cph.params_[metric_name]))
        p = float(cph.summary['p'][metric_name])
        ci = np.exp(cph.confidence_intervals_.iloc[0].values)
        return {'HR': hr, 'CI': f"{ci[0]:.2f}-{ci[1]:.2f}", 'p': p, 'n': len(sub)}
    except Exception:
        return None


def mw_test(merged, metric_name):
    sub = merged[[metric_name, 'transformation']].dropna()
    g1 = sub[sub['transformation'] == 1][metric_name]
    g0 = sub[sub['transformation'] == 0][metric_name]
    if len(g1) < 5 or len(g0) < 10:
        return None
    _, p = stats.mannwhitneyu(g1, g0, alternative='two-sided')
    return {
        'trans': float(g1.mean()), 'nontrans': float(g0.mean()),
        'fc': float(g1.mean() / g0.mean()) if g0.mean() != 0 else np.inf,
        'p': p, 'n_t': len(g1), 'n_nt': len(g0)
    }


def run_tests(label, series, clinical, name='metric'):
    merged = link_clinical(series, clinical, name)
    print(f"\n  {label}")
    print(f"    Median={series.median():.4f}, Mean={series.mean():.4f}, "
          f"Range=[{series.min():.4f}, {series.max():.4f}], n={len(series)}")

    for endpoint, tcol, ecol in [('PFS', 'pfs_years', 'pfs_event'),
                                  ('OS', 'os_years', 'os_event')]:
        r = cox_test(merged, name, tcol, ecol)
        if r:
            sig = " ***" if r['p'] < 0.001 else " **" if r['p'] < 0.01 else " *" if r['p'] < 0.05 else ""
            print(f"    {endpoint}: HR={r['HR']:.2f} ({r['CI']}), p={r['p']:.4f}, n={r['n']}{sig}")
        else:
            print(f"    {endpoint}: insufficient data")

    r = mw_test(merged, name)
    if r:
        direction = "HIGHER" if r['trans'] > r['nontrans'] else "LOWER"
        sig = " ***" if r['p'] < 0.001 else " **" if r['p'] < 0.01 else " *" if r['p'] < 0.05 else ""
        print(f"    Transform: {direction} ({r['trans']:.4f} vs {r['nontrans']:.4f}), "
              f"fc={r['fc']:.2f}, p={r['p']:.4f}, n={r['n_t']}/{r['n_nt']}{sig}")
    else:
        print(f"    Transform: insufficient data")


def count_follicles_dbscan(t_panel_path, t_utag_path):
    """DBSCAN-based follicle detection. Returns per-ROI: n_follicles, mean_radius, min_dist."""
    from sklearn.cluster import DBSCAN

    t_utag = ad.read_h5ad(t_utag_path)

    # Exclude controls
    mask = ~t_utag.obs['sample_id'].str.contains(EXCLUDE_PAT, na=False)
    obs = t_utag.obs[mask]

    comp_col = 'compartment_name'
    FOLLICLE_CENTER = ['GC core', 'Follicle core (GC/CD20hi/CXCR5hi)']

    roi_stats = []
    for roi in obs['sample_id'].unique():
        rm = obs['sample_id'] == roi
        roi_obs = obs[rm]
        roi_comps = roi_obs[comp_col].values
        roi_cx = roi_obs['centroid_x'].values.astype(float)
        roi_cy = roi_obs['centroid_y'].values.astype(float)

        center_mask = np.isin(roi_comps, FOLLICLE_CENTER)
        if center_mask.sum() < 50:
            roi_stats.append({'roi': roi, 'n_follicles': 0, 'mean_radius': np.nan,
                              'min_dist': np.nan, 'n_cells': len(roi_obs)})
            continue

        coords = np.column_stack([roi_cx[center_mask], roi_cy[center_mask]])
        db = DBSCAN(eps=120, min_samples=20).fit(coords)
        labels = db.labels_

        centroids = []
        radii = []
        for lbl in np.unique(labels):
            if lbl == -1:
                continue
            fl = labels == lbl
            if fl.sum() < 40:
                continue
            cx = coords[fl, 0].mean()
            cy = coords[fl, 1].mean()
            r = np.sqrt(np.var(coords[fl, 0]) + np.var(coords[fl, 1]))
            centroids.append((cx, cy))
            radii.append(r)

        n_foll = len(centroids)
        mean_r = np.mean(radii) if radii else np.nan

        # Min distance between follicle centers
        min_dist = np.nan
        if n_foll >= 2:
            ca = np.array(centroids)
            from scipy.spatial.distance import pdist
            min_dist = float(pdist(ca).min())

        roi_stats.append({'roi': roi, 'n_follicles': n_foll, 'mean_radius': mean_r,
                          'min_dist': min_dist, 'n_cells': len(roi_obs)})

    df = pd.DataFrame(roi_stats).set_index('roi')
    print(f"\n  DBSCAN follicle detection: {(df['n_follicles'] > 0).sum()}/{len(df)} ROIs with follicles")
    print(f"  Follicle counts: {df['n_follicles'].describe().to_dict()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-panel", required=True)
    parser.add_argument("--s-utag", required=True)
    parser.add_argument("--t-utag", required=True)
    parser.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    args = parser.parse_args()

    print("=== Testing Radtke et al. 2024 findings ===\n")

    s_panel = ad.read_h5ad(args.s_panel)
    s_utag = ad.read_h5ad(args.s_utag)
    clinical = load_clinical(args.clinical)

    # DBSCAN follicle detection from T-panel
    print("Running DBSCAN follicle detection...")
    foll_df = count_follicles_dbscan(None, args.t_utag)

    # ============================================================
    print("="*60)
    print("FINDING 1: Reduced follicle size in high-risk patients")
    print("  Proxy: follicular compartment fraction per ROI")
    print("  Radtke prediction: HR < 1 (less follicular = worse)")
    print("="*60)

    run_tests("Follicular compartment fraction (UTAG)",
              get_comp_fracs(s_utag, S_FOLL), clinical)
    run_tests("B cell fraction (cell type)",
              get_celltype_frac(s_panel, S_B_CELLS), clinical)

    # ============================================================
    print("\n" + "="*60)
    print("FINDING 2: Loss of FDC networks in high-risk patients")
    print("  Proxy: FDC compartment fraction + FDC cell type fraction")
    print("  Radtke prediction: HR < 1 (less FDC = worse)")
    print("="*60)

    run_tests("FDC compartment fraction (UTAG)",
              get_comp_fracs(s_utag, S_FDC), clinical)
    run_tests("FDC cell type fraction",
              get_celltype_frac(s_panel, S_FDC_CT), clinical)

    # ============================================================
    print("\n" + "="*60)
    print("FINDING 3: Expanded stromal communities in high-risk patients")
    print("  Proxy: stromal compartment + cell type fraction")
    print("  Radtke prediction: HR > 1 (more stromal = worse)")
    print("="*60)

    run_tests("Stromal compartment fraction (UTAG)",
              get_comp_fracs(s_utag, S_STROMAL), clinical)
    run_tests("Stromal/CAF/FRC cell type fraction",
              get_celltype_frac(s_panel, S_STROMAL_CT), clinical)
    run_tests("Endothelial cell type fraction",
              get_celltype_frac(s_panel, S_ENDOTHELIAL_CT), clinical)

    # ============================================================
    print("\n" + "="*60)
    print("FINDING 1b: Follicle count and size (DBSCAN)")
    print("  Direct metrics: n_follicles, mean_radius, min_dist")
    print("  Radtke: fewer/smaller follicles in high-risk")
    print("="*60)

    # n_follicles
    foll_with_data = foll_df[foll_df['n_follicles'] > 0]
    run_tests("Number of follicles per ROI (all ROIs)",
              foll_df['n_follicles'], clinical)
    run_tests("Number of follicles (ROIs with >=1 follicle)",
              foll_with_data['n_follicles'], clinical)

    # mean radius
    radius_series = foll_df['mean_radius'].dropna()
    if len(radius_series) > 20:
        run_tests("Mean follicle radius (px)",
                  radius_series, clinical)

    # min distance between follicles (back-to-back)
    dist_series = foll_df['min_dist'].dropna()
    if len(dist_series) > 20:
        run_tests("Min inter-follicle distance (px)",
                  dist_series, clinical)

    # ============================================================
    # Bonus: myeloid fraction (Radtke also found myeloid expansion)
    print("\n" + "="*60)
    print("BONUS: Myeloid expansion in high-risk patients")
    print("="*60)

    run_tests("Myeloid cell type fraction",
              get_celltype_frac(s_panel, S_MYELOID_CT), clinical)

    # ============================================================
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
Radtke et al. 2024 (Cancer Cell 42:444-463):
  N=10 discovery (IBEX 40-plex), N=29 validation (Cell DIVE)
  Found in early relapsers: reduced follicle size, FDC loss, stromal expansion

Our data: N=137 (IMC 2x39-plex), ~170 ROIs per panel

  Finding 1 (follicle size):
    HR < 1 = CONFIRMED (less follicular = worse prognosis)
    HR > 1 = OPPOSITE (more follicular = worse)

  Finding 2 (FDC networks):
    HR < 1 = CONFIRMED (FDC loss = worse)
    HR > 1 = OPPOSITE (more FDC = worse, CD14+ FDC remodeling)

  Finding 3 (stromal expansion):
    HR > 1 = CONFIRMED (more stromal = worse)
    HR < 1 = OPPOSITE
""")


if __name__ == "__main__":
    main()
