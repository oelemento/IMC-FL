#!/usr/bin/env python3
"""Spatial hypothesis testing using UTAG domains + v8 cell type annotations.

Tests:
  H2a-spatial: Tfh enrichment in follicular vs interfollicular UTAG domains
  H2e: CD8+GranzymeB+ cytotoxic cells enriched at follicle margins
  H3a-spatial: M2 vs inflammatory macrophage distribution across UTAG domains (Biomax only)
  H6c: Entropy by UTAG domain — which domains are most diverse?

Requires: v8 h5ad (cell types + raw markers) and UTAG h5ad (domain labels, same cell order).
"""

import argparse, os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from collections import Counter

def load_array(f, key):
    """Load obs column, handling categorical encoding."""
    ds = f['obs'][key]
    if isinstance(ds, h5py.Group) and 'categories' in ds and 'codes' in ds:
        cats = ds['categories'][:]
        codes = ds['codes'][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        result = cats_str[codes]
        return result
    else:
        vals = ds[:]
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])

def load_marker(f, marker_name, marker_idx):
    """Load a single marker column from X, handling name mapping."""
    name_map = {}
    if marker_name in marker_idx:
        name_map[marker_name] = marker_name
    else:
        alt = marker_name.replace('-', '_')
        if alt in marker_idx:
            name_map[marker_name] = alt
    if marker_name not in name_map:
        return None
    col = marker_idx[name_map[marker_name]]
    X = f['X']
    if len(X.shape) == 1:  # sparse
        return None
    return X[:, col]

def get_marker_idx(f):
    """Build marker name -> column index mapping."""
    if '_index' in f['var']:
        names = f['var']['_index'][:]
    elif 'index' in f['var']:
        names = f['var']['index'][:]
    else:
        names = list(f['var'].keys())
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return {n: i for i, n in enumerate(names)}

def characterize_domains(f_utag, f_v8, panel_name, marker_idx):
    """Determine which UTAG domains are follicular, stromal, etc. using v8 cell types."""
    domains = load_array(f_utag, 'UTAG Label_leiden_0.015')
    unique_domains = sorted(set(domains), key=lambda x: int(x))

    # Get cell types from v8
    cell_types = load_array(f_v8, 'cell_type')
    tma = load_array(f_v8, 'tma')

    # Define B cell types per panel for follicular classification
    if panel_name == 'T':
        b_cell_types = {'B cells', 'B cells (CD20hi)', 'B cells (CXCR5hi)',
                        'B cells (weak CD20)', 'B cells (TOXhi)',
                        'GC B cells', 'Activated B / Plasmablast'}
    else:
        b_cell_types = {'B cells', 'B cells (BCL2+)', 'B cells (PAX5+)',
                        'GC B cells', 'FDC'}

    print(f"\n{'='*70}")
    print(f"UTAG Domain Characterization — {panel_name}-panel (v8 cell types)")
    print(f"{'='*70}")

    domain_info = {}
    print(f"\n{'Domain':>8} {'N cells':>10} {'%total':>7} {'B cell%':>8} {'Top cell types (v8)'}")
    print("-" * 95)

    for d in unique_domains:
        mask = domains == d
        n = np.sum(mask)
        if n < 10:
            continue
        pct = 100 * n / len(domains)

        # B cell fraction from v8 cell types
        ct_counts = Counter(cell_types[mask])
        b_count = sum(ct_counts.get(bt, 0) for bt in b_cell_types)
        b_pct = 100 * b_count / n

        top3 = ct_counts.most_common(3)
        top3_str = ', '.join([f"{ct}({100*c/n:.0f}%)" for ct, c in top3])

        # Classify: >50% B cells = follicular
        is_follicular = b_pct > 50
        domain_info[d] = {
            'n': n, 'b_pct': b_pct, 'follicular': is_follicular,
            'top_types': top3
        }

        tag = " ← FOLL" if is_follicular else ""
        print(f"{d:>8} {n:>10,} {pct:>6.1f}% {b_pct:>7.1f}%  {top3_str}{tag}")

    foll_domains = [d for d, info in domain_info.items() if info['follicular']]
    inter_domains = [d for d, info in domain_info.items() if not info['follicular']]
    n_foll = sum(domain_info[d]['n'] for d in foll_domains)
    n_inter = sum(domain_info[d]['n'] for d in inter_domains)
    print(f"\nFollicular domains (>50% B cells): {[str(d) for d in foll_domains]} ({n_foll:,} cells)")
    print(f"Interfollicular domains: {[str(d) for d in inter_domains]} ({n_inter:,} cells)")

    return domains, domain_info, foll_domains, inter_domains, cell_types, tma


def test_h2a_spatial(f_v8, marker_idx, domains, foll_domains, inter_domains, tma, output_dir):
    """H2a-spatial: Tfh enrichment in follicular vs interfollicular UTAG domains."""
    print(f"\n{'='*70}")
    print("H2a-spatial: Tfh Enrichment in Follicular vs Interfollicular Domains")
    print(f"{'='*70}")

    # Gate Tfh: CD3+CD4+CD8-CD20-CXCR5>2.0
    cd3 = load_marker(f_v8, 'CD3', marker_idx)
    cd4 = load_marker(f_v8, 'CD4', marker_idx)
    cd8 = load_marker(f_v8, 'CD8a', marker_idx)
    cd20 = load_marker(f_v8, 'CD20', marker_idx)
    cxcr5 = load_marker(f_v8, 'CXCR5', marker_idx)
    pd1 = load_marker(f_v8, 'PD-1', marker_idx)
    granzb = load_marker(f_v8, 'GranzymeB', marker_idx)

    cd4t = (cd3 > 0.5) & (cd4 > 0.5) & (cd8 < 0.5) & (cd20 < 0.5)
    tfh = cd4t & (cxcr5 > 2.0)
    tfh_pd1hi = tfh & (pd1 > 0.5)

    # Follicular vs interfollicular mask
    foll_mask = np.isin(domains, foll_domains)
    inter_mask = np.isin(domains, inter_domains)

    n_total = len(domains)
    n_foll = np.sum(foll_mask)
    n_inter = np.sum(inter_mask)

    # Tfh in follicular vs interfollicular
    tfh_foll = np.sum(tfh & foll_mask)
    tfh_inter = np.sum(tfh & inter_mask)
    cd4t_foll = np.sum(cd4t & foll_mask)
    cd4t_inter = np.sum(cd4t & inter_mask)

    tfh_pct_foll = 100 * tfh_foll / cd4t_foll if cd4t_foll > 0 else 0
    tfh_pct_inter = 100 * tfh_inter / cd4t_inter if cd4t_inter > 0 else 0

    print(f"\nFollicular zones: {n_foll:,} cells ({100*n_foll/n_total:.1f}%)")
    print(f"Interfollicular zones: {n_inter:,} cells ({100*n_inter/n_total:.1f}%)")
    print(f"\nCD4 T cells in follicular: {cd4t_foll:,}")
    print(f"CD4 T cells in interfollicular: {cd4t_inter:,}")
    print(f"\nTfh (CXCR5>2.0) in follicular: {tfh_foll:,} ({tfh_pct_foll:.2f}% of CD4 T)")
    print(f"Tfh (CXCR5>2.0) in interfollicular: {tfh_inter:,} ({tfh_pct_inter:.2f}% of CD4 T)")

    # Enrichment ratio
    if tfh_pct_inter > 0:
        enrichment = tfh_pct_foll / tfh_pct_inter
        print(f"Enrichment ratio (foll/inter): {enrichment:.2f}x")

    # Chi-squared test: Tfh vs non-Tfh CD4T × follicular vs interfollicular
    contingency = np.array([
        [tfh_foll, cd4t_foll - tfh_foll],
        [tfh_inter, cd4t_inter - tfh_inter]
    ])
    chi2, pval, dof, expected = chi2_contingency(contingency)
    print(f"Chi-squared test: χ²={chi2:.1f}, p={pval:.2e}")

    # Per-TMA breakdown
    print(f"\n{'TMA':>8} {'Tfh_foll%':>10} {'Tfh_inter%':>11} {'Enrichment':>11} {'p-value':>10}")
    print("-" * 55)

    tma_results = {}
    for t in sorted(set(tma)):
        tmask = tma == t
        cd4t_f = np.sum(cd4t & foll_mask & tmask)
        cd4t_i = np.sum(cd4t & inter_mask & tmask)
        tfh_f = np.sum(tfh & foll_mask & tmask)
        tfh_i = np.sum(tfh & inter_mask & tmask)
        pct_f = 100 * tfh_f / cd4t_f if cd4t_f > 0 else 0
        pct_i = 100 * tfh_i / cd4t_i if cd4t_i > 0 else 0
        enr = pct_f / pct_i if pct_i > 0 else float('inf')

        ct = np.array([[tfh_f, max(cd4t_f - tfh_f, 0)],
                       [tfh_i, max(cd4t_i - tfh_i, 0)]])
        if ct.min() >= 0 and ct.sum() > 0:
            _, pv, _, _ = chi2_contingency(ct) if ct.min() > 5 else (0, 1, 0, 0)
            try:
                _, pv, _, _ = chi2_contingency(ct)
            except:
                pv = 1.0
        else:
            pv = 1.0

        print(f"{t:>8} {pct_f:>9.2f}% {pct_i:>10.2f}% {enr:>10.2f}x {pv:>10.2e}")
        tma_results[t] = {'foll': pct_f, 'inter': pct_i, 'enrichment': enr}

    # Also test: Tfh PD-1hi enrichment
    tfh_pd1_foll = np.sum(tfh_pd1hi & foll_mask)
    tfh_pd1_inter = np.sum(tfh_pd1hi & inter_mask)
    print(f"\nTfh PD-1hi in follicular: {tfh_pd1_foll:,} ({100*tfh_pd1_foll/cd4t_foll:.2f}% of CD4 T)" if cd4t_foll > 0 else "")
    print(f"Tfh PD-1hi in interfollicular: {tfh_pd1_inter:,} ({100*tfh_pd1_inter/cd4t_inter:.2f}% of CD4 T)" if cd4t_inter > 0 else "")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Tfh% in follicular vs interfollicular per TMA
    tmas = sorted(tma_results.keys())
    x = np.arange(len(tmas))
    w = 0.35
    foll_vals = [tma_results[t]['foll'] for t in tmas]
    inter_vals = [tma_results[t]['inter'] for t in tmas]
    axes[0].bar(x - w/2, foll_vals, w, label='Follicular', color='#e74c3c', alpha=0.8)
    axes[0].bar(x + w/2, inter_vals, w, label='Interfollicular', color='#3498db', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tmas)
    axes[0].set_ylabel('Tfh % of CD4 T cells')
    axes[0].set_title('H2a: Tfh Enrichment by Domain')
    axes[0].legend()

    # Panel 2: Enrichment ratios
    enr_vals = [tma_results[t]['enrichment'] for t in tmas]
    colors = ['#e74c3c' if e > 1 else '#3498db' for e in enr_vals]
    axes[1].bar(tmas, enr_vals, color=colors, alpha=0.8)
    axes[1].axhline(y=1, color='gray', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Enrichment Ratio (Foll / Inter)')
    axes[1].set_title('Tfh Follicular Enrichment')

    # Panel 3: CD8+GranzymeB+ by domain type (H2e preview)
    cd8_cytotox = (cd3 > 0.5) & (cd8 > 0.5) & (granzb > 0.5)
    cd8_all = (cd3 > 0.5) & (cd8 > 0.5)

    cytotox_foll = np.sum(cd8_cytotox & foll_mask)
    cytotox_inter = np.sum(cd8_cytotox & inter_mask)
    cd8_foll = np.sum(cd8_all & foll_mask)
    cd8_inter = np.sum(cd8_all & inter_mask)

    pct_cytotox_foll = 100 * cytotox_foll / cd8_foll if cd8_foll > 0 else 0
    pct_cytotox_inter = 100 * cytotox_inter / cd8_inter if cd8_inter > 0 else 0

    axes[2].bar(['Follicular', 'Interfollicular'],
                [pct_cytotox_foll, pct_cytotox_inter],
                color=['#e74c3c', '#3498db'], alpha=0.8)
    axes[2].set_ylabel('GranzymeB+ % of CD8 T')
    axes[2].set_title('H2e: Cytotoxic CD8 by Domain')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'fig_h2a_spatial.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # Print H2e results
    print(f"\n--- H2e: CD8+GranzymeB+ cytotoxic cells ---")
    print(f"Follicular: {cytotox_foll:,} ({pct_cytotox_foll:.2f}% of CD8 T)")
    print(f"Interfollicular: {cytotox_inter:,} ({pct_cytotox_inter:.2f}% of CD8 T)")
    ct2 = np.array([[cytotox_foll, cd8_foll - cytotox_foll],
                     [cytotox_inter, cd8_inter - cytotox_inter]])
    try:
        chi2_2, pval_2, _, _ = chi2_contingency(ct2)
        print(f"Chi-squared: χ²={chi2_2:.1f}, p={pval_2:.2e}")
    except:
        pass

    return tma_results


def test_h3a_spatial(f_v8, marker_idx, domains, domain_info, foll_domains, inter_domains, tma, output_dir):
    """H3a-spatial: M2 vs inflammatory macrophage distribution across UTAG domains."""
    print(f"\n{'='*70}")
    print("H3a-spatial: Macrophage Niche Separation by UTAG Domain (Biomax only)")
    print(f"{'='*70}")

    cd68 = load_marker(f_v8, 'CD68', marker_idx)
    cd163 = load_marker(f_v8, 'CD163', marker_idx)
    cd206 = load_marker(f_v8, 'CD206', marker_idx)
    cd14 = load_marker(f_v8, 'CD14', marker_idx)
    s100a9 = load_marker(f_v8, 'S100A9', marker_idx)

    if any(m is None for m in [cd68, cd163, cd206, cd14, s100a9]):
        print("Missing markers — skipping")
        return

    # Restrict to Biomax where CD163/CD206 work
    biomax_mask = tma == 'Biomax'
    print(f"Biomax cells: {np.sum(biomax_mask):,}")

    mac = (cd68 > 0.5) & biomax_mask
    m2 = mac & (cd163 > 0.5) & (cd206 > 0.5)
    inflam = mac & (cd14 > 0.5) & (s100a9 > 0.5)

    n_mac = np.sum(mac)
    n_m2 = np.sum(m2)
    n_inflam = np.sum(inflam)
    print(f"CD68+ macrophages (Biomax): {n_mac:,}")
    print(f"M2-like: {n_m2:,} ({100*n_m2/n_mac:.1f}%)")
    print(f"Inflammatory: {n_inflam:,} ({100*n_inflam/n_mac:.1f}%)")

    # Per-domain breakdown
    unique_domains = sorted(set(domains[biomax_mask]), key=lambda x: int(x))

    print(f"\n{'Domain':>8} {'n_mac':>8} {'M2%':>8} {'Inflam%':>8} {'B cell%':>8} {'Type'}")
    print("-" * 55)

    domain_m2 = {}
    domain_inflam = {}
    for d in unique_domains:
        dmask = (domains == d) & biomax_mask
        n_d_mac = np.sum(mac & dmask)
        if n_d_mac < 10:
            continue
        n_d_m2 = np.sum(m2 & dmask)
        n_d_inf = np.sum(inflam & dmask)
        pct_m2 = 100 * n_d_m2 / n_d_mac
        pct_inf = 100 * n_d_inf / n_d_mac
        dtype = "FOLL" if d in foll_domains else "INTER"
        b_pct_d = domain_info[d]['b_pct'] if d in domain_info else 0

        print(f"{d:>8} {n_d_mac:>8,} {pct_m2:>7.1f}% {pct_inf:>7.1f}% {b_pct_d:>7.1f}%  {dtype}")
        domain_m2[d] = pct_m2
        domain_inflam[d] = pct_inf

    # Test: is M2% different in follicular vs interfollicular domains?
    m2_foll = np.sum(m2 & np.isin(domains, foll_domains))
    m2_inter = np.sum(m2 & np.isin(domains, inter_domains) & biomax_mask)
    mac_foll = np.sum(mac & np.isin(domains, foll_domains))
    mac_inter = np.sum(mac & np.isin(domains, inter_domains) & biomax_mask)

    if mac_foll > 0 and mac_inter > 0:
        pct_m2_foll = 100 * m2_foll / mac_foll
        pct_m2_inter = 100 * m2_inter / mac_inter
        print(f"\nM2% in follicular domains: {pct_m2_foll:.1f}%")
        print(f"M2% in interfollicular domains: {pct_m2_inter:.1f}%")

        ct = np.array([[m2_foll, mac_foll - m2_foll],
                       [m2_inter, mac_inter - m2_inter]])
        try:
            chi2, pval, _, _ = chi2_contingency(ct)
            print(f"Chi-squared: χ²={chi2:.1f}, p={pval:.2e}")
        except:
            pass

    # Figure
    if len(domain_m2) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ds = sorted(domain_m2.keys(), key=lambda x: int(x))
        m2_vals = [domain_m2[d] for d in ds]
        inf_vals = [domain_inflam[d] for d in ds]
        colors = ['#e74c3c' if d in foll_domains else '#3498db' for d in ds]

        x = np.arange(len(ds))
        w = 0.35
        axes[0].bar(x - w/2, m2_vals, w, label='M2-like', color='#9b59b6', alpha=0.8)
        axes[0].bar(x + w/2, inf_vals, w, label='Inflammatory', color='#e67e22', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"D{d}\n{'F' if d in foll_domains else 'I'}" for d in ds], fontsize=8)
        axes[0].set_ylabel('% of CD68+ macrophages')
        axes[0].set_title('H3a: Macrophage Polarization by UTAG Domain\n(Biomax only)')
        axes[0].legend()

        # Panel 2: M2 vs Inflam scatter colored by domain type
        axes[1].scatter(m2_vals, inf_vals, c=colors, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        for i, d in enumerate(ds):
            axes[1].annotate(f'D{d}', (m2_vals[i], inf_vals[i]), fontsize=7, ha='center', va='bottom')
        axes[1].set_xlabel('M2-like % of macrophages')
        axes[1].set_ylabel('Inflammatory % of macrophages')
        axes[1].set_title('M2 vs Inflammatory by Domain')
        # Legend
        from matplotlib.lines import Line2D
        axes[1].legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Follicular'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Interfollicular'),
        ])

        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'fig_h3a_spatial.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved: {fig_path}")


def test_h6c(f_v8, domains, domain_info, foll_domains, inter_domains, cell_types, tma, panel_name, output_dir):
    """H6c: Entropy by UTAG domain — which tissue compartments are most diverse?"""
    print(f"\n{'='*70}")
    print(f"H6c: Per-Domain Entropy — {panel_name}-panel")
    print(f"{'='*70}")

    unique_domains = sorted(set(domains), key=lambda x: int(x))

    domain_entropy = {}
    print(f"\n{'Domain':>8} {'N cells':>10} {'H (bits)':>9} {'n_types':>8} {'Type':>6} {'Top cell type'}")
    print("-" * 75)

    for d in unique_domains:
        mask = domains == d
        n = np.sum(mask)
        if n < 100:
            continue

        ct_counts = Counter(cell_types[mask])
        total = sum(ct_counts.values())
        props = np.array([c / total for c in ct_counts.values()])
        props = props[props > 0]
        H = -np.sum(props * np.log2(props))
        n_types = len(props)

        dtype = "FOLL" if d in foll_domains else "INTER"
        top_ct = ct_counts.most_common(1)[0]

        domain_entropy[d] = H
        print(f"{d:>8} {n:>10,} {H:>8.2f} {n_types:>8} {dtype:>6}  {top_ct[0]} ({100*top_ct[1]/n:.0f}%)")

    # Compare follicular vs interfollicular entropy
    foll_H = [domain_entropy[d] for d in foll_domains if d in domain_entropy]
    inter_H = [domain_entropy[d] for d in inter_domains if d in domain_entropy]

    if foll_H and inter_H:
        print(f"\nFollicular domain entropy: {np.mean(foll_H):.2f} ± {np.std(foll_H):.2f} (n={len(foll_H)})")
        print(f"Interfollicular domain entropy: {np.mean(inter_H):.2f} ± {np.std(inter_H):.2f} (n={len(inter_H)})")
        if len(foll_H) >= 2 and len(inter_H) >= 2:
            U, p = mannwhitneyu(foll_H, inter_H, alternative='two-sided')
            print(f"Mann-Whitney U: U={U:.0f}, p={p:.3f}")

    return domain_entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t-panel', required=True)
    parser.add_argument('--s-panel', required=True)
    parser.add_argument('--t-utag', required=True)
    parser.add_argument('--s-utag', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # === T-panel ===
    print("\n" + "=" * 70)
    print("T-PANEL SPATIAL ANALYSIS")
    print("=" * 70)

    f_t = h5py.File(args.t_panel, 'r')
    f_t_utag = h5py.File(args.t_utag, 'r')
    marker_idx_t = get_marker_idx(f_t)

    domains_t, domain_info_t, foll_t, inter_t, ct_t, tma_t = characterize_domains(
        f_t_utag, f_t, 'T', marker_idx_t)

    test_h2a_spatial(f_t, marker_idx_t, domains_t, foll_t, inter_t, tma_t, args.output_dir)
    test_h6c(f_t, domains_t, domain_info_t, foll_t, inter_t, ct_t, tma_t, 'T', args.output_dir)

    f_t.close()
    f_t_utag.close()

    # === S-panel ===
    print("\n" + "=" * 70)
    print("S-PANEL SPATIAL ANALYSIS")
    print("=" * 70)

    f_s = h5py.File(args.s_panel, 'r')
    f_s_utag = h5py.File(args.s_utag, 'r')
    marker_idx_s = get_marker_idx(f_s)

    domains_s, domain_info_s, foll_s, inter_s, ct_s, tma_s = characterize_domains(
        f_s_utag, f_s, 'S', marker_idx_s)

    test_h3a_spatial(f_s, marker_idx_s, domains_s, domain_info_s, foll_s, inter_s, tma_s, args.output_dir)
    test_h6c(f_s, domains_s, domain_info_s, foll_s, inter_s, ct_s, tma_s, 'S', args.output_dir)

    f_s.close()
    f_s_utag.close()

    print(f"\n{'='*70}")
    print(f"Done. Results saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
