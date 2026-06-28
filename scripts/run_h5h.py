#!/usr/bin/env python3
"""H5h: HLA downregulation on FL tumor B cells vs normal tonsil B cells.

Tests whether HLA-DR and HLA Class I are downregulated on tumor B cells
compared to normal tonsil B cells. Uses per-TMA comparison to control for
staining batch effects.

Usage:
    python scripts/run_h5h.py \
        --s-panel output/all_TMA_S_global_v8.h5ad \
        --output-dir output/hypotheses_v8
"""

import argparse, os
import numpy as np
import h5py
from scipy.stats import mannwhitneyu, kruskal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_array(f, key):
    ds = f['obs'][key]
    if isinstance(ds, h5py.Group) and 'categories' in ds:
        cats = ds['categories'][:]
        codes = ds['codes'][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


def get_tma(sid):
    sid = str(sid)
    if sid.startswith('A1_'):
        return 'A1'
    elif sid.startswith('B1_'):
        return 'B1'
    elif sid.startswith('C1_'):
        return 'C1'
    elif 'biomax' in sid.lower() or 'Biomax' in sid:
        return 'Biomax'
    return 'Unknown'


def is_tumor_core(sample_id):
    s_lower = sample_id.lower()
    if '_ton_' in s_lower or '_adr_' in s_lower:
        return False
    for tissue in ['tonsil', 'prostate', 'kidney', 'spleen', 'adrenal']:
        if tissue in s_lower:
            return False
    if sample_id == 'Biomax_ROI_006':
        return False
    return True


def is_tonsil_core(sample_id):
    """Identify tonsil control cores specifically."""
    s_lower = sample_id.lower()
    if '_ton_' in s_lower:
        return True
    if 'tonsil' in s_lower:
        return True
    return False


S_B_CELL_TYPES = {'B cells (BCL2+)', 'B cells (PAX5+)', 'B cells'}
S_LQ_TYPES = {'Low quality / Unassigned', 'Unassigned'}

TMA_COLORS = {'A1': '#e41a1c', 'B1': '#377eb8', 'C1': '#4daf4a', 'Biomax': '#ff7f00'}


def panel_label(ax, letter, x=-0.08, y=1.05):
    ax.text(x, y, f'$\\bf{{{letter}}}$', transform=ax.transAxes,
            fontsize=14, va='top', ha='left')


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(args):
    print("Loading S-panel data...")
    fs = h5py.File(args.s_panel, 'r')
    s_sid = load_array(fs, 'sample_id')
    s_ct = load_array(fs, 'cell_type')

    # Marker intensities
    var_key = '_index' if '_index' in fs['var'] else 'index'
    s_markers = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in fs['var'][var_key][:]]
    s_marker_idx = {m: i for i, m in enumerate(s_markers)}

    assert 'HLA_DR' in s_marker_idx, f"HLA_DR not in S-panel: {s_markers}"
    assert 'HLA_Class_I' in s_marker_idx, f"HLA_Class_I not in S-panel: {s_markers}"

    print("  Loading expression matrix (HLA_DR, HLA_Class_I only)...")
    s_X = fs['X'][:]
    s_hladr = s_X[:, s_marker_idx['HLA_DR']].astype(np.float32)
    s_hlai = s_X[:, s_marker_idx['HLA_Class_I']].astype(np.float32)
    del s_X

    # Masks
    s_tma = np.array([get_tma(s) for s in s_sid])
    s_tumor = np.array([is_tumor_core(s) for s in s_sid])
    s_tonsil = np.array([is_tonsil_core(s) for s in s_sid])
    s_is_b = np.array([c in S_B_CELL_TYPES for c in s_ct])

    fs.close()

    # Report counts
    for tma in ['A1', 'B1', 'C1', 'Biomax']:
        tma_mask = s_tma == tma
        n_tumor_b = int(np.sum(tma_mask & s_tumor & s_is_b))
        n_tonsil_b = int(np.sum(tma_mask & s_tonsil & s_is_b))
        n_tumor_rois = len(set(s_sid[tma_mask & s_tumor]))
        n_tonsil_rois = len(set(s_sid[tma_mask & s_tonsil]))
        print(f"  {tma}: {n_tumor_b:,} tumor B cells ({n_tumor_rois} ROIs), "
              f"{n_tonsil_b:,} tonsil B cells ({n_tonsil_rois} ROIs)")

    # ===================================================================
    # Test 1: Per-TMA HLA-DR comparison (tumor vs tonsil B cells)
    # ===================================================================
    print("\nTest 1: Per-TMA HLA-DR on tumor vs tonsil B cells")
    tma_results = {}
    for tma in ['A1', 'B1', 'C1', 'Biomax']:
        tma_mask = s_tma == tma
        tumor_b = s_hladr[tma_mask & s_tumor & s_is_b]
        tonsil_b = s_hladr[tma_mask & s_tonsil & s_is_b]
        if len(tonsil_b) < 50:
            print(f"  {tma}: skipped (only {len(tonsil_b)} tonsil B cells)")
            continue
        U, p = mannwhitneyu(tumor_b, tonsil_b, alternative='two-sided')
        delta = float(np.median(tumor_b) - np.median(tonsil_b))
        tma_results[tma] = {
            'hladr_tumor_median': float(np.median(tumor_b)),
            'hladr_tonsil_median': float(np.median(tonsil_b)),
            'hladr_delta': delta,
            'hladr_p': p,
            'hladr_n_tumor': len(tumor_b),
            'hladr_n_tonsil': len(tonsil_b),
        }
        direction = "DOWN" if delta < 0 else "UP"
        print(f"  {tma}: tumor median={np.median(tumor_b):.3f}, "
              f"tonsil median={np.median(tonsil_b):.3f}, "
              f"delta={delta:+.3f} ({direction}), p={p:.2e}")

    # ===================================================================
    # Test 2: Per-TMA HLA Class I comparison
    # ===================================================================
    print("\nTest 2: Per-TMA HLA Class I on tumor vs tonsil B cells")
    for tma in ['A1', 'B1', 'C1', 'Biomax']:
        tma_mask = s_tma == tma
        tumor_b = s_hlai[tma_mask & s_tumor & s_is_b]
        tonsil_b = s_hlai[tma_mask & s_tonsil & s_is_b]
        if len(tonsil_b) < 50:
            print(f"  {tma}: skipped (only {len(tonsil_b)} tonsil B cells)")
            continue
        U, p = mannwhitneyu(tumor_b, tonsil_b, alternative='two-sided')
        delta = float(np.median(tumor_b) - np.median(tonsil_b))
        if tma in tma_results:
            tma_results[tma].update({
                'hlai_tumor_median': float(np.median(tumor_b)),
                'hlai_tonsil_median': float(np.median(tonsil_b)),
                'hlai_delta': delta,
                'hlai_p': p,
            })
        direction = "DOWN" if delta < 0 else "UP"
        print(f"  {tma}: tumor median={np.median(tumor_b):.3f}, "
              f"tonsil median={np.median(tonsil_b):.3f}, "
              f"delta={delta:+.3f} ({direction}), p={p:.2e}")

    # ===================================================================
    # Test 3: Per-ROI mean HLA-DR, tumor vs tonsil
    # ===================================================================
    print("\nTest 3: Per-ROI mean HLA-DR on B cells (tumor vs tonsil)")
    unique_rois = sorted(set(s_sid))
    roi_hladr_means = {}
    roi_labels = {}  # 'tumor' or 'tonsil'
    roi_tmas = {}
    for roi in unique_rois:
        m = s_sid == roi
        b_mask = m & s_is_b
        n_b = int(np.sum(b_mask))
        if n_b < 50:
            continue
        roi_hladr_means[roi] = float(np.mean(s_hladr[b_mask]))
        if is_tonsil_core(roi):
            roi_labels[roi] = 'tonsil'
        elif is_tumor_core(roi):
            roi_labels[roi] = 'tumor'
        else:
            continue  # other controls (kidney, spleen, etc.)
        roi_tmas[roi] = get_tma(roi)

    tumor_roi_means = [roi_hladr_means[r] for r in roi_hladr_means
                       if roi_labels.get(r) == 'tumor']
    tonsil_roi_means = [roi_hladr_means[r] for r in roi_hladr_means
                        if roi_labels.get(r) == 'tonsil']
    roi_level_p = None
    if len(tonsil_roi_means) >= 3:
        U, roi_level_p = mannwhitneyu(tumor_roi_means, tonsil_roi_means, alternative='two-sided')
        print(f"  {len(tumor_roi_means)} tumor ROIs, {len(tonsil_roi_means)} tonsil ROIs")
        print(f"  Tumor ROI mean HLA-DR: {np.median(tumor_roi_means):.3f} (median of ROI means)")
        print(f"  Tonsil ROI mean HLA-DR: {np.median(tonsil_roi_means):.3f}")
        print(f"  Mann-Whitney p={roi_level_p:.2e}")

    # ===================================================================
    # Test 4: Distribution analysis — bimodality check
    # ===================================================================
    print("\nTest 4: Distribution shape — tumor B cell HLA-DR")
    all_tumor_b_hladr = s_hladr[s_tumor & s_is_b]
    q25, q50, q75 = np.percentile(all_tumor_b_hladr, [25, 50, 75])
    print(f"  Pooled tumor B cells: n={len(all_tumor_b_hladr):,}")
    print(f"  Q25={q25:.3f}, median={q50:.3f}, Q75={q75:.3f}, IQR={q75 - q25:.3f}")
    # Fraction below a "low" threshold (e.g., arcsinh(0.5/5) ≈ 0.1)
    frac_low = float(np.mean(all_tumor_b_hladr < 0.5))
    print(f"  Fraction with HLA-DR < 0.5 (arcsinh): {frac_low:.1%}")

    # ===================================================================
    # Figure
    # ===================================================================
    print("\nGenerating figure...")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check for cartoon
    cartoon_path = os.path.join('output', 'hypothesis_cartoons', 'h5h_hla_downreg.png')

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (a) Concept cartoon
    ax_a = fig.add_subplot(gs[0, 0])
    if os.path.exists(cartoon_path):
        img = mpimg.imread(cartoon_path)
        ax_a.imshow(img)
    else:
        ax_a.text(0.5, 0.5, 'Cartoon\n(generate separately)', ha='center',
                  va='center', fontsize=10, transform=ax_a.transAxes)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    panel_label(ax_a, 'a')

    # (b) Per-TMA HLA-DR comparison: paired bar chart
    ax_b = fig.add_subplot(gs[0, 1])
    tmas_with_data = [t for t in ['A1', 'B1', 'C1', 'Biomax'] if t in tma_results]
    x = np.arange(len(tmas_with_data))
    width = 0.35
    tumor_medians = [tma_results[t]['hladr_tumor_median'] for t in tmas_with_data]
    tonsil_medians = [tma_results[t]['hladr_tonsil_median'] for t in tmas_with_data]
    bars1 = ax_b.bar(x - width / 2, tumor_medians, width, color='#e74c3c',
                     label='FL tumor', alpha=0.8)
    bars2 = ax_b.bar(x + width / 2, tonsil_medians, width, color='#3498db',
                     label='Tonsil (normal)', alpha=0.8)
    # P-value annotations
    for i, t in enumerate(tmas_with_data):
        p = tma_results[t]['hladr_p']
        y_max = max(tumor_medians[i], tonsil_medians[i])
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax_b.text(i, y_max + 0.05, stars, ha='center', fontsize=9)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(tmas_with_data)
    ax_b.set_ylabel('Median HLA-DR\n(arcsinh-transformed)')
    ax_b.set_title('HLA-DR: Tumor vs Tonsil B cells')
    ax_b.legend(fontsize=8)
    panel_label(ax_b, 'b')

    # (c) Per-TMA HLA Class I comparison: paired bar chart
    ax_c = fig.add_subplot(gs[0, 2])
    tmas_hlai = [t for t in tmas_with_data if 'hlai_tumor_median' in tma_results[t]]
    x2 = np.arange(len(tmas_hlai))
    tumor_hlai = [tma_results[t]['hlai_tumor_median'] for t in tmas_hlai]
    tonsil_hlai = [tma_results[t]['hlai_tonsil_median'] for t in tmas_hlai]
    ax_c.bar(x2 - width / 2, tumor_hlai, width, color='#e74c3c',
             label='FL tumor', alpha=0.8)
    ax_c.bar(x2 + width / 2, tonsil_hlai, width, color='#3498db',
             label='Tonsil (normal)', alpha=0.8)
    for i, t in enumerate(tmas_hlai):
        p = tma_results[t]['hlai_p']
        y_max = max(tumor_hlai[i], tonsil_hlai[i])
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax_c.text(i, y_max + 0.05, stars, ha='center', fontsize=9)
    ax_c.set_xticks(x2)
    ax_c.set_xticklabels(tmas_hlai)
    ax_c.set_ylabel('Median HLA Class I\n(arcsinh-transformed)')
    ax_c.set_title('HLA Class I: Tumor vs Tonsil B cells')
    ax_c.legend(fontsize=8)
    panel_label(ax_c, 'c')

    # (d) Distribution overlay: HLA-DR on tumor vs tonsil B cells (pooled)
    ax_d = fig.add_subplot(gs[1, 0])
    all_tonsil_b_hladr = s_hladr[s_tonsil & s_is_b]
    bins = np.linspace(-1, 5, 120)
    ax_d.hist(all_tumor_b_hladr, bins=bins, density=True, alpha=0.6,
              color='#e74c3c', label=f'FL tumor (n={len(all_tumor_b_hladr):,})')
    ax_d.hist(all_tonsil_b_hladr, bins=bins, density=True, alpha=0.6,
              color='#3498db', label=f'Tonsil (n={len(all_tonsil_b_hladr):,})')
    ax_d.set_xlabel('HLA-DR (arcsinh-transformed)')
    ax_d.set_ylabel('Density')
    ax_d.set_title('HLA-DR distribution on B cells')
    ax_d.legend(fontsize=8)
    panel_label(ax_d, 'd')

    # (e) Per-ROI mean HLA-DR boxplot (tumor vs tonsil, colored by TMA)
    ax_e = fig.add_subplot(gs[1, 1])
    tumor_rois_by_tma = {t: [] for t in ['A1', 'B1', 'C1', 'Biomax']}
    tonsil_rois_by_tma = {t: [] for t in ['A1', 'B1', 'C1', 'Biomax']}
    for roi in roi_hladr_means:
        label = roi_labels.get(roi)
        tma = roi_tmas.get(roi)
        if label == 'tumor' and tma:
            tumor_rois_by_tma[tma].append(roi_hladr_means[roi])
        elif label == 'tonsil' and tma:
            tonsil_rois_by_tma[tma].append(roi_hladr_means[roi])

    # Strip plot: tumor on left, tonsil on right, per TMA
    positions_tumor = []
    positions_tonsil = []
    for i, tma in enumerate(['A1', 'B1', 'C1', 'Biomax']):
        color = TMA_COLORS[tma]
        t_vals = tumor_rois_by_tma[tma]
        n_vals = tonsil_rois_by_tma[tma]
        if t_vals:
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(t_vals))
            ax_e.scatter(np.full(len(t_vals), i * 2) + jitter, t_vals,
                         c=color, s=20, alpha=0.7, edgecolors='white', linewidths=0.3)
            ax_e.plot([i * 2 - 0.2, i * 2 + 0.2],
                      [np.median(t_vals)] * 2, c='black', lw=2)
        if n_vals:
            jitter = np.random.default_rng(43).uniform(-0.12, 0.12, len(n_vals))
            ax_e.scatter(np.full(len(n_vals), i * 2 + 0.6) + jitter, n_vals,
                         c=color, s=20, alpha=0.7, edgecolors='white', linewidths=0.3,
                         marker='D')
            ax_e.plot([i * 2 + 0.4, i * 2 + 0.8],
                      [np.median(n_vals)] * 2, c='black', lw=2)

    ax_e.set_xticks([i * 2 + 0.3 for i in range(4)])
    ax_e.set_xticklabels(['A1', 'B1', 'C1', 'Biomax'])
    ax_e.set_ylabel('Mean B-cell HLA-DR per ROI')
    ax_e.set_title('Per-ROI HLA-DR: Tumor (o) vs Tonsil (◇)')
    ax_e.legend([Patch(facecolor='#e74c3c', label='FL tumor'),
                 Patch(facecolor='#3498db', label='Tonsil')],
                ['FL tumor', 'Tonsil'], fontsize=8, loc='upper right')
    panel_label(ax_e, 'e')

    # (f) Distribution overlay: HLA Class I
    ax_f = fig.add_subplot(gs[1, 2])
    all_tumor_b_hlai = s_hlai[s_tumor & s_is_b]
    all_tonsil_b_hlai = s_hlai[s_tonsil & s_is_b]
    bins_i = np.linspace(-1, 5, 120)
    ax_f.hist(all_tumor_b_hlai, bins=bins_i, density=True, alpha=0.6,
              color='#e74c3c', label=f'FL tumor (n={len(all_tumor_b_hlai):,})')
    ax_f.hist(all_tonsil_b_hlai, bins=bins_i, density=True, alpha=0.6,
              color='#3498db', label=f'Tonsil (n={len(all_tonsil_b_hlai):,})')
    ax_f.set_xlabel('HLA Class I (arcsinh-transformed)')
    ax_f.set_ylabel('Density')
    ax_f.set_title('HLA Class I distribution on B cells')
    ax_f.legend(fontsize=8)
    panel_label(ax_f, 'f')

    out_path = os.path.join(output_dir, 'fig_h5h_S.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Per-TMA HLA-DR (median tumor - median tonsil):")
    for t in tmas_with_data:
        r = tma_results[t]
        direction = "DOWN" if r['hladr_delta'] < 0 else "UP"
        print(f"  {t}: delta={r['hladr_delta']:+.3f} ({direction}), p={r['hladr_p']:.2e}")
    print("Per-TMA HLA Class I (median tumor - median tonsil):")
    for t in tmas_hlai:
        r = tma_results[t]
        direction = "DOWN" if r['hlai_delta'] < 0 else "UP"
        print(f"  {t}: delta={r['hlai_delta']:+.3f} ({direction}), p={r['hlai_p']:.2e}")
    if roi_level_p is not None:
        print(f"Per-ROI HLA-DR (tumor vs tonsil): p={roi_level_p:.2e}")
    print(f"Pooled tumor B cell HLA-DR: median={q50:.3f}, "
          f"frac<0.5={frac_low:.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--s-panel', required=True,
                        help='S-panel h5ad file (e.g., output/all_TMA_S_global_v8.h5ad)')
    parser.add_argument('--output-dir', default='output/hypotheses_v8',
                        help='Output directory for figure')
    args = parser.parse_args()
    run_analysis(args)


if __name__ == '__main__':
    main()
