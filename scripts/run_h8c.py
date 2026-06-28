#!/usr/bin/env python3
"""H8c: Exhaustion density as transformation-proximal feature.

Tests whether higher density of exhausted T cells (CD8+TOX+) associates with
distinct spatial and compositional patterns in FL, potentially indicating
more aggressive or transformation-proximal biology.

Metrics:
  - Exhausted CD8 fraction = (CD8 T exhausted + CD8 T pre-exhausted) / total typed
  - Exhausted fraction of CD8 = (exhausted + pre-exhausted) / all CD8 T
  - Correlations with Shannon entropy, follicularity, macrophage fraction
  - Domain localization (follicular vs interfollicular)
  - Spatial clustering (nearest-neighbor distance among exhausted cells)

Output:
  - output/hypotheses_v8/fig_h8c_T.png (main figure)
  - output/hypotheses_v8/fig_h8c_T_supp.png (supplementary)
"""

import os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from scipy.spatial import cKDTree
from collections import Counter

# Import helpers from run_hypotheses_v2
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_hypotheses_v2 import (
    load_array, get_marker_idx, load_marker, is_tumor_core, get_tumor_mask,
    classify_domains, compute_roi_celltype_fractions, correlate_celltype_with_metric,
    add_cartoon, label_panel, plot_representative_core_spatial
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
T_PANEL = 'output/all_TMA_T_global_v8.h5ad'
T_UTAG  = 'output/all_TMA_T_utag_ct_merged.h5ad'
OUTPUT_DIR   = 'output/hypotheses_v8'
CARTOON_DIR  = 'output/hypothesis_cartoons'
CARTOON_PATH = os.path.join(CARTOON_DIR, 'h8c_exhaustion_density.png')

TMA_COLORS = {'A1': '#3498db', 'B1': '#e74c3c', 'C1': '#2ecc71', 'Biomax': '#f39c12'}
LQ_TYPES = {'Low quality / Unassigned'}

# Exhausted cell types
EXHAUSTED_TYPES = {'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)'}
CD8_TYPES = {'CD8 T cells', 'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)'}

# Min typed cells for ROI inclusion
MIN_CELLS_INHOUSE = 8000
MIN_CELLS_BIOMAX  = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CARTOON_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Generate concept cartoon
# ---------------------------------------------------------------------------
def generate_cartoon():
    if os.path.exists(CARTOON_PATH):
        print(f"Cartoon already exists: {CARTOON_PATH}")
        return

    print("Generating concept cartoon via Gemini...")
    try:
        from google import genai
        from google.genai import types
        from PIL import Image as PILImage

        client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

        prompt = (
            "Create a simple scientific concept diagram illustrating the hypothesis that "
            "follicular lymphoma tumors with higher densities of exhausted CD8 T cells "
            "(TOX+ PD-1+ CD8 T cells) may represent more aggressive or transformation-proximal "
            "disease. Show: (1) On the left, a mild FL tumor nodule with few scattered exhausted "
            "T cells (shown in light red), many active T cells (green), and organized follicles. "
            "(2) On the right, an aggressive/transformation-proximal FL nodule with dense clusters "
            "of exhausted T cells (shown in bright red with 'TOX+' labels), fewer active T cells, "
            "disorganized architecture, and higher entropy. Use a simple clean style suitable for "
            "a scientific publication. Include a horizontal arrow between the two states labeled "
            "'Increasing exhaustion burden'. No text watermarks."
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                import io
                img_data = part.inline_data.data
                img = PILImage.open(io.BytesIO(img_data))
                img.save(CARTOON_PATH)
                print(f"Cartoon saved: {CARTOON_PATH}")
                return

        print("WARNING: No image in Gemini response, cartoon not generated.")
    except Exception as e:
        print(f"WARNING: Cartoon generation failed: {e}")


# ---------------------------------------------------------------------------
# Shannon entropy (clean, excluding Unassigned)
# ---------------------------------------------------------------------------
def compute_clean_entropy(cell_types_roi):
    """Compute Shannon entropy excluding LQ/Unassigned cells."""
    clean = cell_types_roi[~np.isin(cell_types_roi, list(LQ_TYPES))]
    if len(clean) < 20:
        return np.nan
    counts = Counter(clean)
    total = sum(counts.values())
    props = np.array([c / total for c in counts.values()])
    props = props[props > 0]
    return -np.sum(props * np.log2(props))


# ---------------------------------------------------------------------------
# Follicularity: fraction of cells in follicular domains
# ---------------------------------------------------------------------------
def compute_follicularity(domains_roi, foll_domains, inter_domains):
    """Fraction of classified cells that are in follicular domains."""
    is_foll = np.isin(domains_roi, foll_domains)
    is_inter = np.isin(domains_roi, inter_domains)
    n_classified = np.sum(is_foll) + np.sum(is_inter)
    if n_classified < 20:
        return np.nan
    return np.sum(is_foll) / n_classified


# ---------------------------------------------------------------------------
# Spatial clustering: mean NND among exhausted cells
# ---------------------------------------------------------------------------
def compute_mean_nnd(x, y):
    """Mean nearest-neighbor distance among a set of points."""
    if len(x) < 3:
        return np.nan
    coords = np.column_stack([x, y])
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)  # k=2: self + nearest neighbor
    nnd = dists[:, 1]  # nearest neighbor (not self)
    return np.mean(nnd)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_h8c():
    print("=" * 70)
    print("H8c: Exhaustion Density as Transformation-Proximal Feature")
    print("=" * 70)

    # --- Generate cartoon ---
    generate_cartoon()

    # --- Load data ---
    print("\nLoading T-panel v8 data...")
    f_v8 = h5py.File(T_PANEL, 'r')
    f_utag = h5py.File(T_UTAG, 'r')

    sample_ids = load_array(f_v8, 'sample_id')
    tma_arr = load_array(f_v8, 'tma')
    cell_types = load_array(f_v8, 'cell_type')
    tumor_mask = get_tumor_mask(sample_ids)

    cx = f_v8['obs']['centroid_x'][:]
    cy = f_v8['obs']['centroid_y'][:]

    n_total = len(sample_ids)
    n_tumor = np.sum(tumor_mask)
    print(f"T-panel: {n_total:,} total -> {n_tumor:,} tumor cells (controls excluded)")

    # Classify UTAG domains
    print("\nClassifying UTAG domains...")
    domains, foll_domains, inter_domains = classify_domains(f_utag, cell_types, 'T')
    foll_mask = np.isin(domains, foll_domains) & tumor_mask
    inter_mask = np.isin(domains, inter_domains) & tumor_mask

    # --- Identify exhausted CD8 cells ---
    exhausted_mask = np.isin(cell_types, list(EXHAUSTED_TYPES)) & tumor_mask
    cd8_mask = np.isin(cell_types, list(CD8_TYPES)) & tumor_mask

    n_exhausted = np.sum(exhausted_mask)
    n_cd8 = np.sum(cd8_mask)
    print(f"\nExhausted CD8 (TOX+): {n_exhausted:,} cells")
    print(f"All CD8 T: {n_cd8:,} cells")
    print(f"Exhausted fraction of CD8: {100*n_exhausted/n_cd8:.1f}%")

    # Per-type breakdown
    for ct in sorted(EXHAUSTED_TYPES | CD8_TYPES):
        n = np.sum((cell_types == ct) & tumor_mask)
        print(f"  {ct:40s}: {n:>8,}")

    # --- Per-ROI metrics ---
    print("\nComputing per-ROI metrics...")
    unique_rois = sorted(set(sample_ids[tumor_mask]))

    roi_data = {}  # {roi: dict of metrics}
    for roi in unique_rois:
        rm = (sample_ids == roi) & tumor_mask
        ct_roi = cell_types[rm]
        tma_val = tma_arr[rm][0]

        # Count typed cells (excluding Unassigned)
        typed = ct_roi[~np.isin(ct_roi, list(LQ_TYPES))]
        n_typed = len(typed)

        # Min cell threshold
        min_cells = MIN_CELLS_BIOMAX if tma_val == 'Biomax' else MIN_CELLS_INHOUSE
        if n_typed < min_cells:
            continue

        # Exhausted counts
        n_exh = np.sum(np.isin(ct_roi, list(EXHAUSTED_TYPES)))
        n_cd8_roi = np.sum(np.isin(ct_roi, list(CD8_TYPES)))

        # Exhausted fraction of total typed
        exh_frac = n_exh / n_typed if n_typed > 0 else 0

        # Exhausted fraction of CD8
        exh_of_cd8 = n_exh / n_cd8_roi if n_cd8_roi > 20 else np.nan

        # Shannon entropy
        entropy = compute_clean_entropy(ct_roi)

        # Follicularity
        dom_roi = domains[rm]
        follicularity = compute_follicularity(dom_roi, foll_domains, inter_domains)

        # Macrophage fraction
        n_mac = np.sum(ct_roi == 'Macrophages')
        mac_frac = n_mac / n_typed if n_typed > 0 else 0

        # Spatial clustering of exhausted cells
        exh_mask_roi = np.isin(ct_roi, list(EXHAUSTED_TYPES))
        cx_roi = cx[rm]
        cy_roi = cy[rm]
        if np.sum(exh_mask_roi) >= 10:
            mean_nnd = compute_mean_nnd(cx_roi[exh_mask_roi], cy_roi[exh_mask_roi])
        else:
            mean_nnd = np.nan

        roi_data[roi] = {
            'tma': tma_val,
            'n_typed': n_typed,
            'n_exh': n_exh,
            'n_cd8': n_cd8_roi,
            'exh_frac': exh_frac,
            'exh_of_cd8': exh_of_cd8,
            'entropy': entropy,
            'follicularity': follicularity,
            'mac_frac': mac_frac,
            'mean_nnd': mean_nnd,
        }

    print(f"ROIs passing filters: {len(roi_data)}")

    # --- Summary statistics ---
    rois = sorted(roi_data.keys())
    exh_fracs = np.array([roi_data[r]['exh_frac'] for r in rois])
    exh_of_cd8s = np.array([roi_data[r]['exh_of_cd8'] for r in rois])
    entropies = np.array([roi_data[r]['entropy'] for r in rois])
    folls = np.array([roi_data[r]['follicularity'] for r in rois])
    mac_fracs = np.array([roi_data[r]['mac_frac'] for r in rois])
    mean_nnds = np.array([roi_data[r]['mean_nnd'] for r in rois])
    tma_vals = np.array([roi_data[r]['tma'] for r in rois])

    print(f"\nExhausted CD8 fraction (of typed):")
    print(f"  Mean: {np.mean(exh_fracs)*100:.2f}%, Median: {np.median(exh_fracs)*100:.2f}%")
    print(f"  Range: [{np.min(exh_fracs)*100:.2f}%, {np.max(exh_fracs)*100:.2f}%]")

    valid_eoc = exh_of_cd8s[~np.isnan(exh_of_cd8s)]
    print(f"\nExhausted fraction of CD8:")
    print(f"  Mean: {np.mean(valid_eoc)*100:.2f}%, Median: {np.median(valid_eoc)*100:.2f}%")
    print(f"  Range: [{np.min(valid_eoc)*100:.2f}%, {np.max(valid_eoc)*100:.2f}%]")

    # --- Per-TMA breakdown ---
    print(f"\nPer-TMA exhausted CD8 fraction:")
    tmas = sorted(set(tma_vals))
    tma_groups = {}
    for t in tmas:
        mask_t = tma_vals == t
        vals_t = exh_fracs[mask_t]
        tma_groups[t] = vals_t
        print(f"  {t:>8}: n={len(vals_t):>3}, mean={np.mean(vals_t)*100:.2f}%, "
              f"median={np.median(vals_t)*100:.2f}%")

    # Kruskal-Wallis across TMAs
    kw_groups = [tma_groups[t] for t in tmas if len(tma_groups[t]) > 0]
    if len(kw_groups) >= 2:
        H_kw, p_kw = kruskal(*kw_groups)
        print(f"\n  Kruskal-Wallis across TMAs: H={H_kw:.2f}, p={p_kw:.4f}")

    # --- Correlations ---
    print("\n--- Correlations with exhausted CD8 fraction ---")

    # 1. Exhaustion vs entropy
    valid = ~np.isnan(entropies)
    rho_ent, p_ent = spearmanr(exh_fracs[valid], entropies[valid])
    print(f"Exhaustion fraction vs Entropy: rho={rho_ent:.3f}, p={p_ent:.2e}")

    # 2. Exhaustion vs follicularity
    valid_f = ~np.isnan(folls)
    rho_foll, p_foll = spearmanr(exh_fracs[valid_f], folls[valid_f])
    print(f"Exhaustion fraction vs Follicularity: rho={rho_foll:.3f}, p={p_foll:.2e}")

    # 3. Exhaustion vs macrophage fraction
    rho_mac, p_mac = spearmanr(exh_fracs, mac_fracs)
    print(f"Exhaustion fraction vs Macrophage fraction: rho={rho_mac:.3f}, p={p_mac:.2e}")

    # 4. Exhaustion vs mean NND (spatial clustering)
    valid_nnd = ~np.isnan(mean_nnds)
    if np.sum(valid_nnd) >= 10:
        rho_nnd, p_nnd = spearmanr(exh_fracs[valid_nnd], mean_nnds[valid_nnd])
        print(f"Exhaustion fraction vs Mean NND: rho={rho_nnd:.3f}, p={p_nnd:.2e}")
        print(f"  (negative rho = more exhausted cells are more spatially clustered)")
    else:
        rho_nnd, p_nnd = np.nan, np.nan
        print("Exhaustion fraction vs Mean NND: insufficient data")

    # --- Domain localization ---
    print("\n--- Domain localization of exhausted CD8 cells ---")
    n_exh_foll = np.sum(exhausted_mask & foll_mask)
    n_exh_inter = np.sum(exhausted_mask & inter_mask)
    n_cd8_foll = np.sum(cd8_mask & foll_mask)
    n_cd8_inter = np.sum(cd8_mask & inter_mask)

    pct_exh_foll = 100 * n_exh_foll / n_cd8_foll if n_cd8_foll > 0 else 0
    pct_exh_inter = 100 * n_exh_inter / n_cd8_inter if n_cd8_inter > 0 else 0

    print(f"  Follicular: {n_exh_foll:,} exhausted / {n_cd8_foll:,} CD8 = {pct_exh_foll:.1f}%")
    print(f"  Interfollicular: {n_exh_inter:,} exhausted / {n_cd8_inter:,} CD8 = {pct_exh_inter:.1f}%")

    # Per-ROI: exhausted density in follicular vs interfollicular
    foll_exh_densities = []
    inter_exh_densities = []
    for roi in rois:
        rm = (sample_ids == roi) & tumor_mask
        ct_roi = cell_types[rm]
        dom_roi = domains[rm]

        is_foll_roi = np.isin(dom_roi, foll_domains)
        is_inter_roi = np.isin(dom_roi, inter_domains)
        is_exh_roi = np.isin(ct_roi, list(EXHAUSTED_TYPES))

        n_foll_cells = np.sum(is_foll_roi)
        n_inter_cells = np.sum(is_inter_roi)

        if n_foll_cells >= 100:
            foll_exh_densities.append(np.sum(is_exh_roi & is_foll_roi) / n_foll_cells)
        if n_inter_cells >= 100:
            inter_exh_densities.append(np.sum(is_exh_roi & is_inter_roi) / n_inter_cells)

    foll_exh_densities = np.array(foll_exh_densities)
    inter_exh_densities = np.array(inter_exh_densities)

    if len(foll_exh_densities) > 0 and len(inter_exh_densities) > 0:
        u_stat, p_domain = mannwhitneyu(foll_exh_densities, inter_exh_densities, alternative='two-sided')
        print(f"\n  Per-ROI exhausted density:")
        print(f"    Follicular: mean={np.mean(foll_exh_densities)*100:.2f}% (n={len(foll_exh_densities)})")
        print(f"    Interfollicular: mean={np.mean(inter_exh_densities)*100:.2f}% (n={len(inter_exh_densities)})")
        print(f"    Mann-Whitney U: p={p_domain:.2e}")
    else:
        p_domain = np.nan
        print("  Insufficient ROIs for domain comparison")

    # --- Spatial clustering summary ---
    print("\n--- Spatial clustering of exhausted CD8 cells ---")
    valid_nnd_arr = mean_nnds[valid_nnd]
    if len(valid_nnd_arr) > 0:
        print(f"  ROIs with >=10 exhausted cells: {len(valid_nnd_arr)}")
        print(f"  Mean NND: {np.mean(valid_nnd_arr):.1f} um, Median: {np.median(valid_nnd_arr):.1f} um")
        print(f"  Range: [{np.min(valid_nnd_arr):.1f}, {np.max(valid_nnd_arr):.1f}] um")

        # Compare NND of exhausted cells vs random cells of same size
        random_nnds = []
        np.random.seed(42)
        for roi in rois:
            rm = (sample_ids == roi) & tumor_mask
            n_exh_r = roi_data[roi]['n_exh']
            if n_exh_r < 10:
                continue
            cx_r = cx[rm]
            cy_r = cy[rm]
            n_total_r = len(cx_r)
            # Random sample same number
            if n_total_r > n_exh_r:
                idx_rand = np.random.choice(n_total_r, size=n_exh_r, replace=False)
                nnd_rand = compute_mean_nnd(cx_r[idx_rand], cy_r[idx_rand])
                random_nnds.append(nnd_rand)

        random_nnds = np.array(random_nnds)
        if len(random_nnds) > 0 and len(valid_nnd_arr) > 0:
            u_nnd, p_nnd_test = mannwhitneyu(valid_nnd_arr, random_nnds, alternative='two-sided')
            print(f"\n  Exhausted NND vs random NND:")
            print(f"    Exhausted: mean={np.mean(valid_nnd_arr):.1f}")
            print(f"    Random (same n): mean={np.mean(random_nnds):.1f}")
            print(f"    Mann-Whitney U: p={p_nnd_test:.2e}")
            if np.mean(valid_nnd_arr) < np.mean(random_nnds):
                print("    -> Exhausted cells are MORE clustered than random")
            else:
                print("    -> Exhausted cells are NOT more clustered than random")

    # --- Driver analysis ---
    print("\n--- Driver analysis: cell type correlations with exhaustion fraction ---")
    roi_fracs = compute_roi_celltype_fractions(sample_ids, cell_types, tumor_mask, LQ_TYPES)
    roi_metric = {r: roi_data[r]['exh_frac'] for r in rois}
    drivers = correlate_celltype_with_metric(roi_fracs, roi_metric)

    print(f"\nTop cell type correlates with exhausted CD8 fraction:")
    for ct_name, rho, p_d in drivers[:15]:
        print(f"  {ct_name:40s}  rho={rho:+.3f}  p={p_d:.2e}")

    # --- Leave-one-TMA-out sensitivity ---
    print("\n--- Leave-one-TMA-out sensitivity ---")
    for exclude in tmas:
        keep = tma_vals != exclude
        valid_keep = keep & ~np.isnan(entropies)
        if np.sum(valid_keep) < 10:
            print(f"  Excl {exclude}: insufficient data")
            continue
        rho_e, p_e = spearmanr(exh_fracs[valid_keep], entropies[valid_keep])
        valid_keep_f = keep & ~np.isnan(folls)
        rho_f, p_f = spearmanr(exh_fracs[valid_keep_f], folls[valid_keep_f])
        print(f"  Excl {exclude}: entropy rho={rho_e:.3f} p={p_e:.2e}, "
              f"follicularity rho={rho_f:.3f} p={p_f:.2e}")

    # --- TOX marker limitation note ---
    print("\n--- TOX marker QC note ---")
    print("  TOX is weak in Biomax (p99=0.18), good in A1/B1/C1 (p99=0.36-0.61)")
    print("  Biomax may undercount exhausted cells. Check Biomax exclusion sensitivity.")

    # Sensitivity excluding Biomax
    keep_nobiomax = tma_vals != 'Biomax'
    valid_nb = keep_nobiomax & ~np.isnan(entropies)
    rho_nb_ent, p_nb_ent = spearmanr(exh_fracs[valid_nb], entropies[valid_nb])
    valid_nb_f = keep_nobiomax & ~np.isnan(folls)
    rho_nb_foll, p_nb_foll = spearmanr(exh_fracs[valid_nb_f], folls[valid_nb_f])
    print(f"  Excluding Biomax: entropy rho={rho_nb_ent:.3f} p={p_nb_ent:.2e}, "
          f"follicularity rho={rho_nb_foll:.3f} p={p_nb_foll:.2e}")

    # ===================================================================
    # MAIN FIGURE (2x3)
    # ===================================================================
    print("\n--- Generating main figure ---")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Concept cartoon
    add_cartoon(axes[0, 0], CARTOON_PATH)
    label_panel(axes[0, 0], 'a')

    # (b) Exhausted CD8 fraction vs Shannon entropy
    ax = axes[0, 1]
    for t in tmas:
        mask_t = tma_vals == t
        valid_t = mask_t & ~np.isnan(entropies)
        ax.scatter(exh_fracs[valid_t] * 100, entropies[valid_t],
                   c=TMA_COLORS.get(t, 'gray'), alpha=0.6, s=25,
                   edgecolors='white', linewidths=0.3, label=t)
    ax.set_xlabel('Exhausted CD8 Fraction (%)', fontsize=10)
    ax.set_ylabel('Shannon Entropy (bits)', fontsize=10)
    p_str = f'{p_ent:.1e}' if p_ent < 0.001 else f'{p_ent:.3f}'
    ax.set_title(f'Exhaustion vs Entropy\n(Spearman rho={rho_ent:.3f}, p={p_str})', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    # Add regression line
    valid_for_line = ~np.isnan(entropies)
    z = np.polyfit(exh_fracs[valid_for_line] * 100, entropies[valid_for_line], 1)
    x_line = np.linspace(0, np.max(exh_fracs[valid_for_line]) * 100, 50)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.4, lw=1)
    label_panel(ax, 'b')

    # (c) Exhausted CD8 fraction vs Follicularity
    ax = axes[0, 2]
    for t in tmas:
        mask_t = tma_vals == t
        valid_t = mask_t & ~np.isnan(folls)
        ax.scatter(exh_fracs[valid_t] * 100, folls[valid_t] * 100,
                   c=TMA_COLORS.get(t, 'gray'), alpha=0.6, s=25,
                   edgecolors='white', linewidths=0.3, label=t)
    ax.set_xlabel('Exhausted CD8 Fraction (%)', fontsize=10)
    ax.set_ylabel('Follicularity (%)', fontsize=10)
    p_str_f = f'{p_foll:.1e}' if p_foll < 0.001 else f'{p_foll:.3f}'
    ax.set_title(f'Exhaustion vs Follicularity\n(Spearman rho={rho_foll:.3f}, p={p_str_f})', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    valid_for_line_f = ~np.isnan(folls)
    z_f = np.polyfit(exh_fracs[valid_for_line_f] * 100, folls[valid_for_line_f] * 100, 1)
    x_line_f = np.linspace(0, np.max(exh_fracs[valid_for_line_f]) * 100, 50)
    ax.plot(x_line_f, np.polyval(z_f, x_line_f), 'k--', alpha=0.4, lw=1)
    label_panel(ax, 'c')

    # (d) Representative spatial: low vs high exhaustion
    ax = axes[1, 0]
    roi_exh_metric = {r: roi_data[r]['exh_frac'] for r in rois}
    plot_representative_core_spatial(
        ax, cx, cy, sample_ids, cell_types, tumor_mask,
        roi_exh_metric, label_lo='Low Exhaustion', label_hi='High Exhaustion',
        metric_name='Exh%', top_n_types=10,
        domains=domains, foll_domains=foll_domains,
        highlight_mask=exhausted_mask,
        highlight_label='Exhausted CD8 (TOX+)',
        highlight_color='#FF4444', highlight_size=20,
        min_highlight_hi=5
    )
    label_panel(ax, 'd')

    # (e) Exhausted CD8 density in follicular vs interfollicular domains
    ax = axes[1, 1]
    box_data_domain = []
    box_labels_domain = []
    if len(foll_exh_densities) > 0:
        box_data_domain.append(foll_exh_densities * 100)
        box_labels_domain.append(f'Follicular\n(n={len(foll_exh_densities)})')
    if len(inter_exh_densities) > 0:
        box_data_domain.append(inter_exh_densities * 100)
        box_labels_domain.append(f'Interfollicular\n(n={len(inter_exh_densities)})')

    if len(box_data_domain) == 2:
        bp = ax.boxplot(box_data_domain, tick_labels=box_labels_domain, patch_artist=True,
                        widths=0.5)
        bp['boxes'][0].set_facecolor('#FFDDDD')
        bp['boxes'][1].set_facecolor('#DDDDFF')
        for b in bp['boxes']:
            b.set_alpha(0.8)

        # Add individual points (jittered)
        for i, data in enumerate(box_data_domain):
            jitter = np.random.RandomState(42).normal(0, 0.04, len(data))
            ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data,
                       c='gray', alpha=0.4, s=10, edgecolors='none', zorder=5)

        p_str_d = f'{p_domain:.1e}' if p_domain < 0.001 else f'{p_domain:.3f}'
        ax.set_title(f'Exhausted CD8 Density by Domain\n(Mann-Whitney p={p_str_d})', fontsize=10)

        # Significance bracket
        max_y_d = max(np.max(box_data_domain[0]), np.max(box_data_domain[1]))
        ax.plot([1, 1, 2, 2], [max_y_d*1.05, max_y_d*1.1, max_y_d*1.1, max_y_d*1.05],
                'k-', lw=1)
        ax.text(1.5, max_y_d * 1.12, f'p={p_str_d}', ha='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Exhausted CD8 Density (%)', fontsize=10)
    label_panel(ax, 'e')

    # (f) Driver analysis
    ax = axes[1, 2]
    top_n = min(12, len(drivers))
    if top_n > 0:
        top = drivers[:top_n]
        names = [d[0] for d in reversed(top)]
        rhos = [d[1] for d in reversed(top)]
        colors_bar = ['#e74c3c' if r < 0 else '#2ecc71' for r in rhos]
        ax.barh(range(len(names)), rhos, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Spearman rho', fontsize=10)
        ax.set_title('Exhaustion Fraction Drivers\n(cell type correlations)', fontsize=10)
        ax.axvline(0, color='gray', lw=0.5)
    label_panel(ax, 'f')

    plt.suptitle('H8c: Exhaustion Density as Transformation-Proximal Feature (T-panel)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    main_fig_path = os.path.join(OUTPUT_DIR, 'fig_h8c_T.png')
    plt.savefig(main_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nMain figure saved: {main_fig_path}")

    # ===================================================================
    # SUPPLEMENTARY FIGURE (2x3)
    # ===================================================================
    print("\n--- Generating supplementary figure ---")
    fig_s, axes_s = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Per-TMA exhaustion fraction boxplot
    ax = axes_s[0, 0]
    box_data_tma = [tma_groups[t] * 100 for t in tmas]
    bp_tma = ax.boxplot(box_data_tma, tick_labels=tmas, patch_artist=True, widths=0.5)
    for patch, t in zip(bp_tma['boxes'], tmas):
        patch.set_facecolor(TMA_COLORS.get(t, 'gray'))
        patch.set_alpha(0.6)
    ax.set_ylabel('Exhausted CD8 Fraction (%)', fontsize=10)
    ax.set_xlabel('TMA', fontsize=10)
    ax.set_title('Per-TMA Exhaustion Fraction', fontsize=10)
    label_panel(ax, 'a')

    # (b) Leave-one-TMA-out sensitivity (entropy correlation)
    ax = axes_s[0, 1]
    loo_rhos_ent = []
    loo_rhos_foll = []
    for exclude in tmas:
        keep = tma_vals != exclude
        valid_keep = keep & ~np.isnan(entropies)
        if np.sum(valid_keep) >= 10:
            r, _ = spearmanr(exh_fracs[valid_keep], entropies[valid_keep])
            loo_rhos_ent.append(r)
        else:
            loo_rhos_ent.append(np.nan)
        valid_keep_f = keep & ~np.isnan(folls)
        if np.sum(valid_keep_f) >= 10:
            r, _ = spearmanr(exh_fracs[valid_keep_f], folls[valid_keep_f])
            loo_rhos_foll.append(r)
        else:
            loo_rhos_foll.append(np.nan)

    x_pos = np.arange(len(tmas))
    w = 0.35
    ax.bar(x_pos - w/2, loo_rhos_ent, width=w, color='#2c3e50', alpha=0.7, label='vs Entropy')
    ax.bar(x_pos + w/2, loo_rhos_foll, width=w, color='#e67e22', alpha=0.7, label='vs Follicularity')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Excl {t}' for t in tmas], fontsize=8)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(rho_ent, color='#2c3e50', linestyle='--', lw=0.8, alpha=0.5)
    ax.axhline(rho_foll, color='#e67e22', linestyle='--', lw=0.8, alpha=0.5)
    ax.set_ylabel('Spearman rho', fontsize=10)
    ax.set_title('Leave-One-TMA-Out Sensitivity', fontsize=10)
    ax.legend(fontsize=8)
    label_panel(ax, 'b')

    # (c) Exhausted fraction of CD8 by TMA
    ax = axes_s[0, 2]
    eoc_by_tma = {}
    for t in tmas:
        mask_t = tma_vals == t
        vals_t = exh_of_cd8s[mask_t]
        vals_t = vals_t[~np.isnan(vals_t)]
        eoc_by_tma[t] = vals_t
    box_data_eoc = [eoc_by_tma[t] * 100 for t in tmas]
    bp_eoc = ax.boxplot(box_data_eoc, tick_labels=tmas, patch_artist=True, widths=0.5)
    for patch, t in zip(bp_eoc['boxes'], tmas):
        patch.set_facecolor(TMA_COLORS.get(t, 'gray'))
        patch.set_alpha(0.6)
    ax.set_ylabel('Exhausted Fraction of CD8 (%)', fontsize=10)
    ax.set_xlabel('TMA', fontsize=10)
    ax.set_title('Exhaustion Among CD8 T by TMA', fontsize=10)
    label_panel(ax, 'c')

    # (d) Spatial clustering: exhausted NND vs random NND
    ax = axes_s[1, 0]
    if len(valid_nnd_arr) > 0 and len(random_nnds) > 0:
        bp_nnd = ax.boxplot([valid_nnd_arr, random_nnds],
                            tick_labels=['Exhausted CD8', 'Random\n(same n)'],
                            patch_artist=True, widths=0.5)
        bp_nnd['boxes'][0].set_facecolor('#FF4444')
        bp_nnd['boxes'][0].set_alpha(0.6)
        bp_nnd['boxes'][1].set_facecolor('#888888')
        bp_nnd['boxes'][1].set_alpha(0.6)
        ax.set_ylabel('Mean NND (pixels)', fontsize=10)
        ax.set_title(f'Spatial Clustering\n(Mann-Whitney p={p_nnd_test:.2e})', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
    label_panel(ax, 'd')

    # (e) Exhaustion fraction vs macrophage fraction
    ax = axes_s[1, 1]
    for t in tmas:
        mask_t = tma_vals == t
        ax.scatter(exh_fracs[mask_t] * 100, mac_fracs[mask_t] * 100,
                   c=TMA_COLORS.get(t, 'gray'), alpha=0.6, s=25,
                   edgecolors='white', linewidths=0.3, label=t)
    ax.set_xlabel('Exhausted CD8 Fraction (%)', fontsize=10)
    ax.set_ylabel('Macrophage Fraction (%)', fontsize=10)
    p_str_m = f'{p_mac:.1e}' if p_mac < 0.001 else f'{p_mac:.3f}'
    ax.set_title(f'Exhaustion vs Macrophages\n(Spearman rho={rho_mac:.3f}, p={p_str_m})', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    label_panel(ax, 'e')

    # (f) Exhaustion fraction histogram with TMA overlay
    ax = axes_s[1, 2]
    for t in tmas:
        mask_t = tma_vals == t
        ax.hist(exh_fracs[mask_t] * 100, bins=15, alpha=0.5,
                color=TMA_COLORS.get(t, 'gray'), label=t, edgecolor='white')
    ax.set_xlabel('Exhausted CD8 Fraction (%)', fontsize=10)
    ax.set_ylabel('Number of ROIs', fontsize=10)
    ax.set_title('Distribution of Exhaustion Fraction', fontsize=10)
    ax.legend(fontsize=8)
    ax.axvline(np.mean(exh_fracs) * 100, color='black', linestyle='--', lw=1,
               label=f'Mean={np.mean(exh_fracs)*100:.1f}%')
    label_panel(ax, 'f')

    plt.suptitle('H8c: Exhaustion Density — Supplementary (T-panel)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    supp_fig_path = os.path.join(OUTPUT_DIR, 'fig_h8c_T_supp.png')
    plt.savefig(supp_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Supplementary figure saved: {supp_fig_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("H8c SUMMARY")
    print("=" * 70)
    print(f"Exhausted CD8 fraction (of typed): mean={np.mean(exh_fracs)*100:.2f}%, "
          f"range=[{np.min(exh_fracs)*100:.2f}%, {np.max(exh_fracs)*100:.2f}%]")
    print(f"Exhaustion vs Entropy:       rho={rho_ent:+.3f}, p={p_ent:.2e}")
    print(f"Exhaustion vs Follicularity: rho={rho_foll:+.3f}, p={p_foll:.2e}")
    print(f"Exhaustion vs Macrophages:   rho={rho_mac:+.3f}, p={p_mac:.2e}")
    print(f"Domain localization: Follicular {pct_exh_foll:.1f}% vs Interfollicular {pct_exh_inter:.1f}%")
    if not np.isnan(p_domain):
        print(f"  Mann-Whitney p={p_domain:.2e}")
    if not np.isnan(rho_nnd):
        print(f"Spatial clustering: exhausted NND rho={rho_nnd:+.3f} with exhaustion fraction")

    # Determine status
    sig_count = sum([
        p_ent < 0.05,
        p_foll < 0.05,
        p_mac < 0.05,
        p_domain < 0.05 if not np.isnan(p_domain) else False,
    ])

    if sig_count >= 2 and (p_ent < 0.05 or p_foll < 0.05):
        status = "CONFIRMED"
        reason = (f"Exhausted CD8 fraction shows significant associations with "
                  f"entropy (rho={rho_ent:.3f}), follicularity (rho={rho_foll:.3f}), "
                  f"and/or macrophage fraction")
    elif sig_count >= 1:
        status = "PARTIALLY CONFIRMED"
        reason = "Some associations significant but pattern not fully consistent"
    else:
        status = "NOT CONFIRMED"
        reason = "No significant associations found"

    print(f"\nStatus: {status}")
    print(f"Reason: {reason}")
    print(f"\nFigures: {main_fig_path}, {supp_fig_path}")

    f_v8.close()
    f_utag.close()

    return status


if __name__ == '__main__':
    run_h8c()
