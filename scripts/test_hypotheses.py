#!/usr/bin/env python3
"""Test spatial hypotheses from deep research review.

H6a: Per-ROI phenotypic entropy (Shannon) — compare across TMAs
H2a: Tfh (CD4+CXCR5+) enrichment in follicular vs interfollicular UTAG domains
H3a: M2-like vs inflammatory macrophage niche separation

Usage:
    python scripts/test_hypotheses.py \
        --t-panel output/all_TMA_T_global_v8.h5ad \
        --s-panel output/all_TMA_S_global_v8.h5ad \
        --output-dir output/hypotheses

    # If UTAG files available (for H2a spatial enrichment):
    python scripts/test_hypotheses.py \
        --t-panel output/all_TMA_T_global_v8.h5ad \
        --s-panel output/all_TMA_S_global_v8.h5ad \
        --t-utag output/all_TMA_T_utag.h5ad \
        --s-utag output/all_TMA_S_utag.h5ad \
        --output-dir output/hypotheses
"""

import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_obs_column(f, col):
    """Read an obs column from h5py File, handling categorical encoding."""
    g = f['obs'][col]
    if isinstance(g, h5py.Group) and 'categories' in g:
        cats = [x.decode() if isinstance(x, bytes) else x for x in g['categories'][:]]
        codes = g['codes'][:]
        return np.array([cats[c] if c >= 0 else '' for c in codes])
    else:
        arr = g[:]
        if arr.dtype.kind == 'S' or arr.dtype.kind == 'O':
            return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
        return arr


def shannon_entropy(proportions):
    """Shannon entropy from a proportion vector (ignoring zeros)."""
    p = proportions[proportions > 0]
    return -np.sum(p * np.log2(p))


def per_roi_proportions(cell_types, sample_ids, tma_labels):
    """Compute per-ROI cell type proportions and metadata."""
    rois = sorted(set(sample_ids))
    cell_type_names = sorted(set(cell_types))

    rows = []
    for roi in rois:
        mask = sample_ids == roi
        n = mask.sum()
        if n < 50:
            continue
        ct_counts = pd.Series(cell_types[mask]).value_counts()
        props = ct_counts / n
        row = {'roi': roi, 'n_cells': n, 'tma': tma_labels[mask][0]}
        for ct in cell_type_names:
            row[ct] = props.get(ct, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H6a: Phenotypic entropy
# ---------------------------------------------------------------------------

def test_h6a_entropy(t_panel_path, s_panel_path, output_dir):
    """H6a: Per-ROI phenotypic entropy, compare across TMAs."""
    print("\n" + "="*70)
    print("H6a: Per-ROI Phenotypic Entropy")
    print("="*70)

    results = {}

    for panel_name, path in [('T-panel', t_panel_path), ('S-panel', s_panel_path)]:
        print(f"\n--- {panel_name} ---")
        f = h5py.File(path, 'r')
        cell_types = read_obs_column(f, 'cell_type')
        sample_ids = read_obs_column(f, 'sample_id')
        tma_labels = read_obs_column(f, 'tma')
        f.close()

        df = per_roi_proportions(cell_types, sample_ids, tma_labels)

        # Compute entropy per ROI (excluding metadata columns)
        ct_cols = [c for c in df.columns if c not in ('roi', 'n_cells', 'tma')]
        df['entropy'] = df[ct_cols].apply(lambda row: shannon_entropy(row.values), axis=1)

        # Per-TMA summary
        print(f"\n{'TMA':>10s}  {'n_ROIs':>6s}  {'mean_H':>7s}  {'std_H':>6s}  {'min_H':>6s}  {'max_H':>6s}")
        for tma in sorted(df['tma'].unique()):
            sub = df[df['tma'] == tma]
            h = sub['entropy']
            print(f"{tma:>10s}  {len(sub):>6d}  {h.mean():>7.3f}  {h.std():>6.3f}  {h.min():>6.3f}  {h.max():>6.3f}")

        # Global
        h = df['entropy']
        print(f"\n{'Global':>10s}  {len(df):>6d}  {h.mean():>7.3f}  {h.std():>6.3f}  {h.min():>6.3f}  {h.max():>6.3f}")

        # Kruskal-Wallis across TMAs
        groups = [df[df['tma'] == t]['entropy'].values for t in sorted(df['tma'].unique())]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            stat, p = stats.kruskal(*groups)
            print(f"\nKruskal-Wallis across TMAs: H={stat:.2f}, p={p:.4f}")
        else:
            stat, p = np.nan, np.nan

        results[panel_name] = df
        df.to_csv(os.path.join(output_dir, f'h6a_entropy_{panel_name.replace("-","")}.csv'),
                  index=False, float_format='%.4f')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (panel_name, df) in zip(axes, results.items()):
        tmas = sorted(df['tma'].unique())
        data = [df[df['tma'] == t]['entropy'].values for t in tmas]
        bp = ax.boxplot(data, labels=tmas, patch_artist=True)
        colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
        for patch, color in zip(bp['boxes'], colors[:len(tmas)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        # Overlay points
        for i, (t, d) in enumerate(zip(tmas, data)):
            ax.scatter(np.random.normal(i+1, 0.04, len(d)), d,
                      alpha=0.3, s=15, color=colors[i % len(colors)])
        ax.set_ylabel('Shannon Entropy (bits)')
        ax.set_xlabel('TMA')
        ax.set_title(f'{panel_name}: Per-ROI Phenotypic Entropy')

        # Add p-value
        groups_kw = [df[df['tma'] == t]['entropy'].values for t in tmas]
        if all(len(g) > 0 for g in groups_kw):
            _, p = stats.kruskal(*groups_kw)
            ax.text(0.02, 0.98, f'Kruskal-Wallis p={p:.4f}',
                   transform=ax.transAxes, va='top', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'h6a_entropy_by_tma.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {os.path.join(output_dir, 'h6a_entropy_by_tma.png')}")

    return results


# ---------------------------------------------------------------------------
# H2a: Tfh enrichment in follicular domains
# ---------------------------------------------------------------------------

def test_h2a_tfh(t_panel_path, t_utag_path, output_dir):
    """H2a: Tfh (CD4+CXCR5+) enrichment in follicular vs interfollicular UTAG domains."""
    print("\n" + "="*70)
    print("H2a: Tfh Enrichment in Follicular Domains")
    print("="*70)

    f = h5py.File(t_panel_path, 'r')
    cell_types = read_obs_column(f, 'cell_type')
    sample_ids = read_obs_column(f, 'sample_id')
    tma_labels = read_obs_column(f, 'tma')

    # Read raw marker intensities for Tfh gating
    raw_var = [x.decode() if isinstance(x, bytes) else x for x in f['raw']['var']['_index'][:]]
    raw_X = f['raw']['X']

    # Find marker indices (handle underscores/hyphens)
    marker_idx = {m: i for i, m in enumerate(raw_var)}
    needed = ['CD3', 'CD4', 'CD8a', 'CXCR5', 'CD20', 'PD-1', 'CD57']
    # Build name mapping: canonical -> actual name in data
    name_map = {}
    for m in needed:
        if m in marker_idx:
            name_map[m] = m
        else:
            # Try replacing - with _
            alt = m.replace('-', '_')
            if alt in marker_idx:
                name_map[m] = alt
            else:
                # Fuzzy match
                for var_name in raw_var:
                    if m.lower().replace('-','').replace('_','') == var_name.lower().replace('-','').replace('_',''):
                        name_map[m] = var_name
                        break
    missing = [m for m in needed if m not in name_map]
    if missing:
        print(f"WARNING: Missing markers: {missing}")

    print(f"Markers available: {[m for m in needed if m in name_map]}")
    if name_map:
        mapped = {m: name_map[m] for m in needed if m in name_map and name_map[m] != m}
        if mapped:
            print(f"Name mappings: {mapped}")

    # Read needed columns
    n_cells = raw_X.shape[0]
    data = {}
    for m in needed:
        if m in name_map:
            actual = name_map[m]
            data[m] = raw_X[:, marker_idx[actual]].astype(np.float32)
            print(f"  {m}: mean={data[m].mean():.3f}, p99={np.percentile(data[m], 99):.2f}")

    # Gate Tfh: CD3+CD4+CXCR5+ (CD8−, CD20−)
    if all(m in data for m in ['CD3', 'CD4', 'CD8a', 'CXCR5', 'CD20']):
        tfh_mask = ((data['CD3'] > 0.5) & (data['CD4'] > data['CD8a']) &
                    (data['CD4'] > data['CD20']) & (data['CXCR5'] > 0.5))
        n_tfh = tfh_mask.sum()
        print(f"\nTfh (CD3+CD4+CXCR5+): {n_tfh:,} cells ({n_tfh/n_cells*100:.2f}%)")

        # Refine: PD-1hi Tfh
        if 'PD-1' in data:
            tfh_pd1hi = tfh_mask & (data['PD-1'] > 0.5)
            print(f"Tfh PD-1hi: {tfh_pd1hi.sum():,} cells ({tfh_pd1hi.sum()/n_cells*100:.2f}%)")

        # CD57+ Tfh (H2b)
        if 'CD57' in data:
            tfh_cd57 = tfh_mask & (data['CD57'] > 0.5)
            print(f"Tfh CD57+: {tfh_cd57.sum():,} cells ({tfh_cd57.sum()/n_cells*100:.2f}%)")

        # Per-TMA Tfh
        print(f"\n{'TMA':>10s}  {'n_cells':>10s}  {'Tfh%':>6s}  {'TfhPD1hi%':>10s}  {'TfhCD57%':>9s}")
        for tma in sorted(set(tma_labels)):
            tmask = tma_labels == tma
            n_tma = tmask.sum()
            n_tfh_tma = (tfh_mask & tmask).sum()
            n_pd1 = (tfh_pd1hi & tmask).sum() if 'PD-1' in data else 0
            n_cd57 = (tfh_cd57 & tmask).sum() if 'CD57' in data else 0
            print(f"{tma:>10s}  {n_tma:>10,d}  {n_tfh_tma/n_tma*100:>5.2f}%  {n_pd1/n_tma*100:>9.2f}%  {n_cd57/n_tma*100:>8.2f}%")

        # If UTAG domains available, test enrichment
        if t_utag_path:
            print("\n--- Tfh enrichment by UTAG domain ---")
            fu = h5py.File(t_utag_path, 'r')
            # Try different UTAG label columns
            utag_col = None
            for col_name in ['UTAG Label_leiden_0.015', 'UTAG Label_leiden_0.02',
                             'UTAG Label_leiden_0.01']:
                try:
                    utag_labels = read_obs_column(fu, col_name)
                    utag_col = col_name
                    break
                except KeyError:
                    continue

            if utag_col:
                print(f"Using UTAG column: {utag_col}")
                domains = sorted(set(utag_labels))

                # For each domain, compute Tfh fraction and total cell type composition
                rows = []
                for d in domains:
                    dmask = utag_labels == d
                    n_d = dmask.sum()
                    if n_d < 100:
                        continue
                    n_tfh_d = (tfh_mask & dmask).sum()
                    # Also compute B cell fraction to identify follicular domains
                    ct_d = pd.Series(cell_types[dmask]).value_counts(normalize=True)
                    b_frac = sum(ct_d.get(ct, 0) for ct in ct_d.index if 'B cell' in ct or 'GC B' in ct)
                    cd4_frac = sum(ct_d.get(ct, 0) for ct in ct_d.index if 'CD4' in ct or 'Treg' in ct or 'Tfh' in ct)

                    rows.append({
                        'domain': d,
                        'n_cells': n_d,
                        'tfh_frac': n_tfh_d / n_d,
                        'b_cell_frac': b_frac,
                        'cd4_frac': cd4_frac,
                        'top_type': ct_d.index[0] if len(ct_d) > 0 else 'unknown',
                        'top_frac': ct_d.values[0] if len(ct_d) > 0 else 0,
                    })

                ddf = pd.DataFrame(rows)
                ddf = ddf.sort_values('b_cell_frac', ascending=False)

                print(f"\n{'Domain':>10s}  {'n_cells':>10s}  {'B_cell%':>8s}  {'Tfh%':>7s}  {'Top type'}")
                for _, r in ddf.iterrows():
                    print(f"{r['domain']:>10s}  {r['n_cells']:>10,d}  {r['b_cell_frac']*100:>7.1f}%  "
                          f"{r['tfh_frac']*100:>6.2f}%  {r['top_type']}")

                # Correlation: B cell fraction vs Tfh fraction
                r_val, p_val = stats.pearsonr(ddf['b_cell_frac'], ddf['tfh_frac'])
                print(f"\nCorrelation B_cell% vs Tfh%: r={r_val:.3f}, p={p_val:.4f}")

                ddf.to_csv(os.path.join(output_dir, 'h2a_tfh_by_utag_domain.csv'),
                          index=False, float_format='%.4f')

                # Plot
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(ddf['b_cell_frac']*100, ddf['tfh_frac']*100,
                          s=ddf['n_cells']/500, alpha=0.6)
                for _, r in ddf.iterrows():
                    ax.annotate(r['domain'], (r['b_cell_frac']*100, r['tfh_frac']*100),
                               fontsize=7, alpha=0.7)
                ax.set_xlabel('B cell fraction (%)')
                ax.set_ylabel('Tfh fraction (%)')
                ax.set_title(f'H2a: Tfh Enrichment by UTAG Domain\nr={r_val:.3f}, p={p_val:.4f}')
                fig.savefig(os.path.join(output_dir, 'h2a_tfh_vs_bcell_by_domain.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Plot saved: {os.path.join(output_dir, 'h2a_tfh_vs_bcell_by_domain.png')}")

            fu.close()
    else:
        print("Cannot gate Tfh — missing required markers")

    f.close()


# ---------------------------------------------------------------------------
# H3a: M2-like vs inflammatory macrophage niches
# ---------------------------------------------------------------------------

def test_h3a_macrophage_niches(s_panel_path, s_utag_path, output_dir):
    """H3a: M2-like (CD163+CD206+) vs inflammatory (CD14+S100A9+) macrophage spatial separation."""
    print("\n" + "="*70)
    print("H3a: M2-like vs Inflammatory Macrophage Niches")
    print("="*70)

    f = h5py.File(s_panel_path, 'r')
    cell_types = read_obs_column(f, 'cell_type')
    sample_ids = read_obs_column(f, 'sample_id')
    tma_labels = read_obs_column(f, 'tma')

    raw_var = [x.decode() if isinstance(x, bytes) else x for x in f['raw']['var']['_index'][:]]
    raw_X = f['raw']['X']
    n_cells = raw_X.shape[0]

    marker_idx = {m: i for i, m in enumerate(raw_var)}
    needed = ['CD68', 'CD163', 'CD206', 'CD14', 'S100A9', 'CD11c', 'HLA-DR', 'CD11b']

    data = {}
    for m in needed:
        if m in marker_idx:
            data[m] = raw_X[:, marker_idx[m]].astype(np.float32)
            print(f"  {m}: mean={data[m].mean():.3f}, p99={np.percentile(data[m], 99):.2f}")

    # Gate myeloid subsets
    if 'CD68' in data:
        mac = data['CD68'] > 2.0  # macrophage gate
        n_mac = mac.sum()
        print(f"\nCD68+ macrophages: {n_mac:,} ({n_mac/n_cells*100:.1f}%)")

        # M2-like: CD163+CD206+
        if 'CD163' in data and 'CD206' in data:
            m2 = mac & (data['CD163'] > 1.0) & (data['CD206'] > 0.5)
            print(f"M2-like (CD68+CD163+CD206+): {m2.sum():,} ({m2.sum()/n_mac*100:.1f}% of mac)")
        else:
            m2 = None

        # Inflammatory: CD14+S100A9+
        if 'CD14' in data and 'S100A9' in data:
            inflam = mac & (data['CD14'] > 1.0) & (data['S100A9'] > 0.5)
            print(f"Inflammatory (CD68+CD14+S100A9+): {inflam.sum():,} ({inflam.sum()/n_mac*100:.1f}% of mac)")
        else:
            inflam = None

        # DC-like: CD11c+HLA-DR+
        if 'CD11c' in data and 'HLA-DR' in data:
            dc = (data['CD11c'] > 1.0) & (data['HLA-DR'] > 1.0)
            print(f"DC-like (CD11c+HLA-DR+): {dc.sum():,} ({dc.sum()/n_cells*100:.1f}%)")

        # Per-TMA breakdown
        if m2 is not None and inflam is not None:
            print(f"\n{'TMA':>10s}  {'n_mac':>8s}  {'M2%':>6s}  {'Inflam%':>8s}  {'M2/Inflam':>10s}")
            for tma in sorted(set(tma_labels)):
                tmask = tma_labels == tma
                n_mac_t = (mac & tmask).sum()
                n_m2 = (m2 & tmask).sum()
                n_inf = (inflam & tmask).sum()
                ratio = n_m2 / n_inf if n_inf > 0 else float('inf')
                print(f"{tma:>10s}  {n_mac_t:>8,d}  {n_m2/n_mac_t*100 if n_mac_t>0 else 0:>5.1f}%  "
                      f"{n_inf/n_mac_t*100 if n_mac_t>0 else 0:>7.1f}%  {ratio:>10.2f}")

            # Per-ROI M2 vs inflammatory fractions
            rois = sorted(set(sample_ids))
            roi_rows = []
            for roi in rois:
                rmask = sample_ids == roi
                n_mac_r = (mac & rmask).sum()
                if n_mac_r < 20:
                    continue
                roi_rows.append({
                    'roi': roi,
                    'tma': tma_labels[rmask][0],
                    'n_mac': n_mac_r,
                    'm2_frac': (m2 & rmask).sum() / n_mac_r,
                    'inflam_frac': (inflam & rmask).sum() / n_mac_r,
                })
            rdf = pd.DataFrame(roi_rows)

            # Correlation M2 vs inflammatory
            r_val, p_val = stats.pearsonr(rdf['m2_frac'], rdf['inflam_frac'])
            print(f"\nPer-ROI M2% vs Inflammatory%: r={r_val:.3f}, p={p_val:.4f}")
            print(f"(negative r = distinct niches; positive r = co-occurring)")

            rdf.to_csv(os.path.join(output_dir, 'h3a_macrophage_niches.csv'),
                      index=False, float_format='%.4f')

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Scatter: M2 vs inflammatory per ROI
            ax = axes[0]
            tma_colors = {'A1': '#4e79a7', 'B1': '#f28e2b', 'C1': '#e15759', 'Biomax': '#76b7b2'}
            for tma in sorted(rdf['tma'].unique()):
                sub = rdf[rdf['tma'] == tma]
                ax.scatter(sub['m2_frac']*100, sub['inflam_frac']*100,
                          alpha=0.5, s=20, label=tma,
                          color=tma_colors.get(tma, 'gray'))
            ax.set_xlabel('M2-like fraction (% of macrophages)')
            ax.set_ylabel('Inflammatory fraction (% of macrophages)')
            ax.set_title(f'H3a: M2-like vs Inflammatory per ROI\nr={r_val:.3f}')
            ax.legend()

            # Box: M2/inflammatory ratio by TMA
            ax = axes[1]
            rdf['ratio'] = rdf['m2_frac'] / rdf['inflam_frac'].clip(lower=0.001)
            tmas = sorted(rdf['tma'].unique())
            box_data = [rdf[rdf['tma'] == t]['ratio'].values for t in tmas]
            bp = ax.boxplot(box_data, labels=tmas, patch_artist=True)
            for patch, t in zip(bp['boxes'], tmas):
                patch.set_facecolor(tma_colors.get(t, 'gray'))
                patch.set_alpha(0.6)
            ax.set_ylabel('M2 / Inflammatory ratio')
            ax.set_xlabel('TMA')
            ax.set_title('Macrophage Polarization by TMA')
            ax.set_yscale('log')

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'h3a_macrophage_niches.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved: {os.path.join(output_dir, 'h3a_macrophage_niches.png')}")

        # UTAG domain enrichment
        if s_utag_path and m2 is not None and inflam is not None:
            print("\n--- Macrophage niche by UTAG domain ---")
            fu = h5py.File(s_utag_path, 'r')
            utag_col = None
            for col_name in ['UTAG Label_leiden_0.015', 'UTAG Label_leiden_0.02',
                             'UTAG Label_leiden_0.01']:
                try:
                    utag_labels = read_obs_column(fu, col_name)
                    utag_col = col_name
                    break
                except KeyError:
                    continue

            if utag_col:
                domains = sorted(set(utag_labels))
                print(f"\n{'Domain':>10s}  {'n_cells':>10s}  {'Mac%':>6s}  {'M2%':>6s}  {'Inf%':>6s}  {'Top type'}")
                for d in domains:
                    dmask = utag_labels == d
                    n_d = dmask.sum()
                    if n_d < 100:
                        continue
                    n_mac_d = (mac & dmask).sum()
                    n_m2_d = (m2 & dmask).sum()
                    n_inf_d = (inflam & dmask).sum()
                    ct_d = pd.Series(cell_types[dmask]).value_counts()
                    top_ct = ct_d.index[0] if len(ct_d) > 0 else '?'
                    print(f"{d:>10s}  {n_d:>10,d}  {n_mac_d/n_d*100:>5.1f}%  "
                          f"{n_m2_d/n_d*100:>5.1f}%  {n_inf_d/n_d*100:>5.1f}%  {top_ct}")

            fu.close()

    f.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--t-panel', required=True, help='T-panel v8 h5ad')
    parser.add_argument('--s-panel', required=True, help='S-panel v8 h5ad')
    parser.add_argument('--t-utag', default=None, help='T-panel UTAG h5ad (optional)')
    parser.add_argument('--s-utag', default=None, help='S-panel UTAG h5ad (optional)')
    parser.add_argument('--output-dir', default='output/hypotheses', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # H6a: Phenotypic entropy
    test_h6a_entropy(args.t_panel, args.s_panel, args.output_dir)

    # H2a: Tfh enrichment
    test_h2a_tfh(args.t_panel, args.t_utag, args.output_dir)

    # H3a: Macrophage niches
    test_h3a_macrophage_niches(args.s_panel, args.s_utag, args.output_dir)

    print("\n" + "="*70)
    print("Done. All results saved to:", args.output_dir)
    print("="*70)


if __name__ == '__main__':
    main()
