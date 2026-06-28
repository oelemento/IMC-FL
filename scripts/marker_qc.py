#!/usr/bin/env python3
"""Systematic marker QC for IMC panels.

Computes per-marker statistics (mean, median, percentiles, % positive, dynamic
range) globally and per-TMA. Flags markers with no signal, weak signal, or
diffuse (non-bimodal) distributions.

Usage:
    python scripts/marker_qc.py \
        --input output/all_TMA_T_global_v7.h5ad \
        --panel T \
        --output output/marker_qc_T_panel.csv

    python scripts/marker_qc.py \
        --input output/all_TMA_S_global_v7.h5ad \
        --panel S \
        --output output/marker_qc_S_panel.csv
"""

import argparse
import h5py
import numpy as np
import pandas as pd


SKIP_MARKERS = {'DNA1', 'DNA2', 'HistoneH3', 'p_H3s28'}


def classify_marker(p99, pct_gt1, mean, median, dyn_range):
    """Classify marker signal quality."""
    issues = []
    if p99 < 0.5:
        issues.append('NO SIGNAL')
    elif p99 < 1.0:
        issues.append('VERY WEAK')
    if pct_gt1 < 1.0 and p99 > 0.5:
        issues.append('SPARSE')
    if dyn_range < 0.5:
        issues.append('NO SEPARATION')
    if mean > 0.3 and median > 0.2 and p99 < 2.0:
        issues.append('DIFFUSE')

    if 'NO SIGNAL' in issues:
        return 'dead', issues
    elif 'VERY WEAK' in issues or ('SPARSE' in issues and 'DIFFUSE' in issues):
        return 'marginal', issues
    elif 'DIFFUSE' in issues:
        return 'diffuse', issues
    elif issues:
        return 'caution', issues
    else:
        return 'good', issues


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='h5ad file to QC')
    parser.add_argument('--panel', required=True, choices=['T', 'S'], help='Panel name')
    parser.add_argument('--output', required=True, help='Output CSV path')
    args = parser.parse_args()

    f = h5py.File(args.input, 'r')

    raw_var = [x.decode() for x in f['raw']['var']['_index'][:]]
    raw_X = f['raw']['X']
    n_cells = raw_X.shape[0]

    # Get TMA labels
    tma_cats = [x.decode() for x in f['obs']['tma']['categories'][:]]
    tma_codes = f['obs']['tma']['codes'][:]
    tma = np.array([tma_cats[c] for c in tma_codes])
    tma_names = sorted(set(tma))

    print(f"{args.panel}-panel: {n_cells:,} cells, {len(raw_var)} markers, TMAs: {tma_names}")

    rows = []
    for i, m in enumerate(raw_var):
        if m in SKIP_MARKERS:
            continue
        col = raw_X[:, i].astype(np.float32)

        row = {
            'marker': m,
            'mean': col.mean(),
            'median': np.median(col),
            'p75': np.percentile(col, 75),
            'p90': np.percentile(col, 90),
            'p95': np.percentile(col, 95),
            'p99': np.percentile(col, 99),
            'max': col.max(),
            'pct_gt_0.5': (col > 0.5).sum() / n_cells * 100,
            'pct_gt_1.0': (col > 1.0).sum() / n_cells * 100,
            'pct_gt_2.0': (col > 2.0).sum() / n_cells * 100,
            'dynamic_range': np.percentile(col, 99) - np.percentile(col, 1),
        }

        # Per-TMA p99
        for t in tma_names:
            mask = tma == t
            row[f'p99_{t}'] = np.percentile(col[mask], 99)

        # Classification
        status, issues = classify_marker(
            row['p99'], row['pct_gt_1.0'], row['mean'], row['median'], row['dynamic_range']
        )
        row['status'] = status
        row['issues'] = '; '.join(issues) if issues else ''

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, float_format='%.3f')
    print(f"\nSaved {args.output}")

    # Print summary
    print(f"\n{'Marker':>25s}  {'p99':>5s}  {'>1.0%':>6s}  {'dynR':>5s}  {'Status':>10s}  Issues")
    for _, r in df.iterrows():
        print(f"{r['marker']:>25s}  {r['p99']:>5.2f}  {r['pct_gt_1.0']:>5.1f}%  {r['dynamic_range']:>5.2f}  {r['status']:>10s}  {r['issues']}")

    # Summary counts
    print(f"\n--- Summary ---")
    for status in ['good', 'diffuse', 'caution', 'marginal', 'dead']:
        n = (df['status'] == status).sum()
        if n > 0:
            markers = ', '.join(df[df['status'] == status]['marker'].tolist())
            print(f"  {status:>10s}: {n} — {markers}")

    # Per-TMA flagging
    print(f"\n--- Per-TMA p99 for flagged markers ---")
    flagged = df[df['status'].isin(['dead', 'marginal', 'diffuse'])]
    if len(flagged) > 0:
        print(f"{'Marker':>25s}", end='')
        for t in tma_names:
            print(f"  {t:>10s}", end='')
        print()
        for _, r in flagged.iterrows():
            print(f"{r['marker']:>25s}", end='')
            for t in tma_names:
                print(f"  {r[f'p99_{t}']:>10.2f}", end='')
            print()

    f.close()


if __name__ == '__main__':
    main()
