#!/usr/bin/env python3
"""Test whether follicular sub-architecture complexity predicts outcomes.

Metrics:
1. n_foll_zones: number of follicular sub-zones (0-6) per ROI
2. has_interface: presence of Follicle-T zone interface (binary)
3. has_treg_zone: presence of Treg-enriched T zone (binary)
4. follicular_entropy: Shannon entropy within follicular compartments only
5. exhaustion_gradient: difference in exhaustion between GC core and T zone
6. zone_diversity: Shannon entropy of zone composition (how evenly distributed)
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clinical_linkage import load_clinical, normalize_sample_id

EXCLUDE_PAT = r'(?i)tonsil|prostate|kidney|spleen|adrenal|_ton_|_adr_'

FOLLICULAR_ZONES = [
    "GC core", "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)", "Activated B / CXCR5hi zone",
    "B cell follicle (CD20hi/CXCR5hi)", "B cell zone",
]
INTERFACE = "Follicle-T zone interface"
TREG_ZONE = "Treg-enriched T zone"


def compute_metrics(t_utag, t_panel):
    """Compute per-ROI follicular complexity metrics."""
    comp_col = 'compartment_name'
    roi_col = 'sample_id'
    ct_col = 'cell_type'

    # Exclude controls
    mask_utag = ~t_utag.obs[roi_col].str.contains(EXCLUDE_PAT, na=False)
    obs_utag = t_utag.obs[mask_utag]

    mask_panel = ~t_panel.obs[roi_col].str.contains(EXCLUDE_PAT, na=False)
    obs_panel = t_panel.obs[mask_panel]

    rois = obs_utag[roi_col].unique()
    results = []

    for roi in rois:
        rm = obs_utag[roi_col] == roi
        roi_obs = obs_utag[rm]
        comps = roi_obs[comp_col].values

        # 1. Count follicular sub-zones (>=30 cells)
        foll_zones = set()
        zone_sizes = {}
        for z in FOLLICULAR_ZONES:
            n = (comps == z).sum()
            if n >= 30:
                foll_zones.add(z)
                zone_sizes[z] = n
        n_foll_zones = len(foll_zones)

        # 2. Has interface / Treg zone
        has_interface = int((comps == INTERFACE).sum() >= 20)
        has_treg_zone = int((comps == TREG_ZONE).sum() >= 20)

        # 3. Zone diversity (Shannon entropy of follicular zone sizes)
        if zone_sizes:
            sizes = np.array(list(zone_sizes.values()), dtype=float)
            fracs = sizes / sizes.sum()
            zone_entropy = float(-np.sum(fracs * np.log2(fracs + 1e-10)))
        else:
            zone_entropy = 0.0

        # 4. Exhaustion gradient (need panel data)
        rm_panel = obs_panel[roi_col] == roi
        roi_panel = obs_panel[rm_panel]

        exh_gradient = np.nan
        if ct_col in roi_panel.columns:
            cd8 = roi_panel[roi_panel[ct_col].str.contains('CD8', na=False)]
            # Need compartment info on panel cells - join by cell index or position
            # Actually T-panel cells have compartment in UTAG file
            # Let's use UTAG obs which has both compartment and cell type
            if 'cell_type' in roi_obs.columns or ct_col in roi_obs.columns:
                use_col = 'cell_type' if 'cell_type' in roi_obs.columns else ct_col
                cd8_gc = roi_obs[(roi_obs[use_col].str.contains('CD8', na=False)) &
                                 (roi_obs[comp_col] == 'GC core')]
                cd8_tz = roi_obs[(roi_obs[use_col].str.contains('CD8', na=False)) &
                                 (roi_obs[comp_col] == 'T cell zone (CD4/CD8)')]

                # Check for exhausted subtypes
                exh_types = ['CD8 T exhausted', 'CD8 T pre-exhausted']
                if len(cd8_gc) >= 5 and len(cd8_tz) >= 5:
                    exh_gc = cd8_gc[use_col].str.contains('exhaust', case=False, na=False).mean()
                    exh_tz = cd8_tz[use_col].str.contains('exhaust', case=False, na=False).mean()
                    exh_gradient = float(exh_gc - exh_tz)

        # 5. Total follicular fraction
        n_total = len(roi_obs)
        n_foll_cells = sum((comps == z).sum() for z in FOLLICULAR_ZONES)
        foll_frac = n_foll_cells / n_total if n_total > 0 else 0

        results.append({
            'roi': roi,
            'n_foll_zones': n_foll_zones,
            'has_interface': has_interface,
            'has_treg_zone': has_treg_zone,
            'zone_entropy': zone_entropy,
            'exh_gradient': exh_gradient,
            'foll_frac': foll_frac,
            'n_cells': n_total,
        })

    df = pd.DataFrame(results).set_index('roi')
    return df


def link_clinical(df, clinical):
    df = df.copy()
    df['slide_ID'] = [normalize_sample_id(s) for s in df.index]
    clin_map = {
        'Overall survival (y)': 'os_years', 'CODE_OS': 'os_event',
        'Progression free survival (y)': 'pfs_years', 'CODE_PFS': 'pfs_event',
        'Transformation': 'transformation',
    }
    keep = [c for c in clin_map if c in clinical.columns]
    clin = clinical[['slide_ID'] + keep].drop_duplicates('slide_ID', keep='first')
    merged = df.merge(clin, on='slide_ID', how='inner').rename(columns=clin_map)
    if 'transformation' in merged.columns:
        merged['transformation'] = (merged['transformation'] == 'Yes').astype(int)
    return merged


def cox_test(merged, metric, time_col, event_col):
    sub = merged[[metric, time_col, event_col]].dropna()
    if len(sub) < 20 or sub[event_col].sum() < 5:
        return None
    sub = sub.copy()
    std = sub[metric].std()
    if std < 1e-10:
        return None
    sub[metric] = (sub[metric] - sub[metric].mean()) / std
    cph = CoxPHFitter()
    try:
        cph.fit(sub, duration_col=time_col, event_col=event_col)
        hr = float(np.exp(cph.params_[metric]))
        p = float(cph.summary['p'][metric])
        ci = np.exp(cph.confidence_intervals_.iloc[0].values)
        return {'HR': hr, 'CI': f"{ci[0]:.2f}-{ci[1]:.2f}", 'p': p, 'n': len(sub)}
    except Exception:
        return None


def logrank_binary(merged, metric, time_col, event_col):
    """Log-rank test for binary metric."""
    sub = merged[[metric, time_col, event_col]].dropna()
    g1 = sub[sub[metric] == 1]
    g0 = sub[sub[metric] == 0]
    if len(g1) < 10 or len(g0) < 10:
        return None
    try:
        result = logrank_test(g1[time_col], g0[time_col], g1[event_col], g0[event_col])
        return {'p': result.p_value, 'n1': len(g1), 'n0': len(g0)}
    except Exception:
        return None


def mw_test(merged, metric):
    sub = merged[[metric, 'transformation']].dropna()
    g1 = sub[sub['transformation'] == 1][metric]
    g0 = sub[sub['transformation'] == 0][metric]
    if len(g1) < 5 or len(g0) < 10:
        return None
    _, p = stats.mannwhitneyu(g1, g0, alternative='two-sided')
    return {'trans': float(g1.mean()), 'nontrans': float(g0.mean()), 'p': p,
            'n': f"{len(g1)}/{len(g0)}"}


def run_continuous(label, merged, metric):
    print(f"\n  {label}")
    vals = merged[metric].dropna()
    print(f"    Median={vals.median():.3f}, Mean={vals.mean():.3f}, "
          f"Range=[{vals.min():.3f}, {vals.max():.3f}], n={len(vals)}")

    for ep, tc, ec in [('PFS', 'pfs_years', 'pfs_event'), ('OS', 'os_years', 'os_event')]:
        r = cox_test(merged, metric, tc, ec)
        if r:
            sig = " ***" if r['p'] < 0.001 else " **" if r['p'] < 0.01 else " *" if r['p'] < 0.05 else ""
            print(f"    {ep}: HR={r['HR']:.2f} ({r['CI']}), p={r['p']:.4f}, n={r['n']}{sig}")
        else:
            print(f"    {ep}: insufficient data")

    r = mw_test(merged, metric)
    if r:
        d = "HIGHER" if r['trans'] > r['nontrans'] else "LOWER"
        sig = " ***" if r['p'] < 0.001 else " **" if r['p'] < 0.01 else " *" if r['p'] < 0.05 else ""
        print(f"    Transform: {d} ({r['trans']:.3f} vs {r['nontrans']:.3f}), p={r['p']:.4f}, n={r['n']}{sig}")


def run_binary(label, merged, metric):
    print(f"\n  {label}")
    n1 = (merged[metric] == 1).sum()
    n0 = (merged[metric] == 0).sum()
    print(f"    Present: {n1}, Absent: {n0}")

    for ep, tc, ec in [('PFS', 'pfs_years', 'pfs_event'), ('OS', 'os_years', 'os_event')]:
        r = logrank_binary(merged, metric, tc, ec)
        if r:
            sig = " ***" if r['p'] < 0.001 else " **" if r['p'] < 0.01 else " *" if r['p'] < 0.05 else ""
            print(f"    {ep}: log-rank p={r['p']:.4f}, n={r['n1']}/{r['n0']}{sig}")
        else:
            print(f"    {ep}: insufficient data")

    r = mw_test(merged, metric)
    if r:
        print(f"    Transform: {r['trans']:.3f} vs {r['nontrans']:.3f}, p={r['p']:.4f}, n={r['n']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-panel", required=True)
    parser.add_argument("--t-utag", required=True)
    parser.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    args = parser.parse_args()

    print("=== Follicular complexity vs outcomes ===\n")

    t_panel = ad.read_h5ad(args.t_panel)
    t_utag = ad.read_h5ad(args.t_utag)
    clinical = load_clinical(args.clinical)

    print("Computing per-ROI metrics...")
    metrics = compute_metrics(t_utag, t_panel)
    merged = link_clinical(metrics, clinical)

    print(f"\nROIs: {len(metrics)}")
    print(f"Zone distribution: {metrics['n_foll_zones'].value_counts().sort_index().to_dict()}")

    # === Continuous metrics ===
    print("\n" + "="*60)
    print("CONTINUOUS METRICS (Cox PH per SD)")
    print("="*60)

    run_continuous("Number of follicular sub-zones (0-6)", merged, 'n_foll_zones')
    run_continuous("Zone diversity (Shannon entropy of zone sizes)", merged, 'zone_entropy')
    run_continuous("Exhaustion gradient (GC core - T zone)", merged, 'exh_gradient')
    run_continuous("Follicular fraction", merged, 'foll_frac')

    # === Binary metrics ===
    print("\n" + "="*60)
    print("BINARY METRICS (log-rank)")
    print("="*60)

    run_binary("Has follicle-T zone interface", merged, 'has_interface')
    run_binary("Has Treg-enriched T zone", merged, 'has_treg_zone')

    # === Correlation between metrics ===
    print("\n" + "="*60)
    print("INTER-METRIC CORRELATIONS (Spearman)")
    print("="*60)
    for m1, m2 in [('n_foll_zones', 'zone_entropy'),
                    ('n_foll_zones', 'exh_gradient'),
                    ('n_foll_zones', 'foll_frac'),
                    ('zone_entropy', 'exh_gradient'),
                    ('has_interface', 'exh_gradient')]:
        sub = metrics[[m1, m2]].dropna()
        if len(sub) > 20:
            rho, p = stats.spearmanr(sub[m1], sub[m2])
            print(f"  {m1} vs {m2}: rho={rho:.3f}, p={p:.4f}, n={len(sub)}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
If follicular complexity predicts outcomes:
  - More zones / higher entropy = more organized follicle = ?
  - Exhaustion gradient = immunosuppressive architecture = worse?
  - Interface presence = boundary function = more evasion?

Key question: does follicular sub-architecture INDEPENDENTLY predict
outcomes beyond what cell type fractions already capture?
""")


if __name__ == "__main__":
    main()
