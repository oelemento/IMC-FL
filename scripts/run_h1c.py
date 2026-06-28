#!/usr/bin/env python3
"""H1c: FDC network integrity is disrupted in FL.

Panel: S-panel
Rationale: Follicular dendritic cells (FDCs) form reticular networks that
present antigen to germinal-centre B cells and produce CXCL13.  In FL the
FDC meshwork is often reprogrammed (Mourcin 2021).  If the network is
disrupted — fewer FDCs, more scattered, or displaced from follicular
domains — that may correlate with disease aggression.

Test: Quantify per-ROI FDC density, spatial clustering (mean nearest-
neighbour distance), graph connectivity (largest connected component at
50 px), and domain localisation (fraction of FDC in follicular vs
interfollicular UTAG domains).  Correlate FDC connectivity with
follicularity, entropy, and per-ROI cell-type fractions.
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
from scipy.spatial import KDTree
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_hypotheses_v2 import (
    load_array, get_marker_idx, load_marker, is_tumor_core, get_tumor_mask,
    classify_domains, compute_roi_celltype_fractions, correlate_celltype_with_metric,
    add_cartoon, label_panel, plot_representative_core_spatial,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V8_PATH = os.path.join(BASE, 'output', 'all_TMA_S_global_v8.h5ad')
UTAG_PATH = os.path.join(BASE, 'output', 'all_TMA_S_utag_ct_merged.h5ad')
OUTPUT_DIR = os.path.join(BASE, 'output', 'hypotheses_v8')
CARTOON_DIR = os.path.join(BASE, 'output', 'hypothesis_cartoons')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CARTOON_DIR, exist_ok=True)

PANEL = 'S'
MIN_CELLS_DEFAULT = 8000   # in-house TMAs
MIN_CELLS_BIOMAX = 5000    # Biomax TMA
CONNECT_DIST = 50          # px: edge threshold for FDC connectivity graph
MIN_FDC = 10               # minimum FDC cells per ROI to compute metrics

TMA_COLORS = {'A1': '#3498db', 'B1': '#e74c3c', 'C1': '#2ecc71', 'Biomax': '#f39c12'}

LQ_TYPES = {'Low quality / Unassigned'}


# ---------------------------------------------------------------------------
# Generate concept cartoon
# ---------------------------------------------------------------------------
def generate_cartoon():
    cartoon_path = os.path.join(CARTOON_DIR, 'h1c_fdc_network.png')
    if os.path.exists(cartoon_path):
        print(f"Cartoon already exists: {cartoon_path}")
        return cartoon_path

    print("Generating concept cartoon via Gemini...")
    try:
        from google import genai
        from google.genai import types
        from PIL import Image as PILImage
        import io

        client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

        prompt = (
            "Create a simple, clean scientific concept cartoon for a research figure. "
            "Show two panels side by side: LEFT labeled 'Intact FDC network' showing "
            "a circular follicle (pale pink background) with a connected meshwork of "
            "gold/yellow star-shaped cells (FDCs) forming a reticular net inside the "
            "follicle, surrounded by small blue dots (B cells) woven into the mesh. "
            "The FDC cells are touching each other, forming one big connected web. "
            "RIGHT labeled 'Disrupted FDC network' showing a similar follicle but the "
            "gold star-shaped FDC cells are scattered, isolated, and fragmented — no "
            "longer forming a connected network. Some FDC cells have drifted outside "
            "the follicle boundary into the interfollicular zone. The B cells (blue dots) "
            "are more disorganized. "
            "Add a small annotation: left says 'high connectivity', right says 'low connectivity'. "
            "Use a white background, minimal labels, publication-quality style. "
            "No text other than the labels mentioned. Simple, diagrammatic, not photographic."
        )

        response = client.models.generate_images(
            model='imagen-4.0-fast-generate-001',
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )

        if response.generated_images:
            img_bytes = response.generated_images[0].image.image_bytes
            img = PILImage.open(io.BytesIO(img_bytes))
            jpg_path = cartoon_path.replace('.png', '.jpg')
            img.save(jpg_path, 'JPEG', quality=95)
            img_png = PILImage.open(jpg_path)
            img_png.save(cartoon_path, 'PNG')
            os.remove(jpg_path)
            print(f"Cartoon saved: {cartoon_path}")
        else:
            print("Warning: No image generated by Gemini")
    except Exception as e:
        print(f"Warning: Cartoon generation failed: {e}")

    return cartoon_path


# ---------------------------------------------------------------------------
# FDC network metrics
# ---------------------------------------------------------------------------
def compute_fdc_metrics(cx_roi, cy_roi, cell_types_roi, domains_roi,
                        foll_domains, connect_dist=CONNECT_DIST):
    """Compute FDC network metrics for one ROI.

    Returns dict with:
      fdc_fraction:    FDC count / total typed cells
      fdc_mean_nnd:    mean nearest-neighbour distance among FDC cells
      fdc_connectivity: largest connected component / total FDC (0-1)
      fdc_foll_frac:   fraction of FDC in follicular domains
      fdc_b_dist:      mean distance from FDC to nearest B cell
      n_fdc:           number of FDC cells
    or None if fewer than MIN_FDC FDC cells.
    """
    fdc_mask = cell_types_roi == 'FDC'
    n_fdc = int(np.sum(fdc_mask))
    if n_fdc < MIN_FDC:
        return None

    # Typed cells (exclude LQ)
    typed_mask = ~np.isin(cell_types_roi, list(LQ_TYPES))
    n_typed = int(np.sum(typed_mask))
    if n_typed < 100:
        return None

    fdc_fraction = n_fdc / n_typed

    # Coordinates of FDC cells
    fdc_coords = np.column_stack([cx_roi[fdc_mask], cy_roi[fdc_mask]])

    # 1) Mean nearest-neighbour distance among FDC cells
    if n_fdc >= 2:
        tree_fdc = KDTree(fdc_coords)
        dists, _ = tree_fdc.query(fdc_coords, k=2)   # k=2: self + nearest
        nnd = dists[:, 1]
        fdc_mean_nnd = float(np.mean(nnd))
    else:
        fdc_mean_nnd = np.nan

    # 2) Connectivity: build graph, find largest connected component
    if n_fdc >= 2:
        # Find all pairs within connect_dist
        pairs = tree_fdc.query_pairs(connect_dist)
        # Union-Find for connected components
        parent = list(range(n_fdc))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in pairs:
            union(a, b)

        # Size of each component
        comp_ids = [find(i) for i in range(n_fdc)]
        comp_sizes = Counter(comp_ids)
        largest_cc = max(comp_sizes.values())
        fdc_connectivity = largest_cc / n_fdc
    else:
        fdc_connectivity = 0.0

    # 3) Domain localisation: fraction of FDC in follicular domains
    fdc_domains = domains_roi[fdc_mask]
    is_foll = np.isin(fdc_domains, foll_domains)
    fdc_foll_frac = float(np.mean(is_foll)) if n_fdc > 0 else np.nan

    # 4) FDC-to-B distance
    b_types = {'B cells', 'B cells (BCL2+)', 'B cells (PAX5+)'}
    b_mask = np.isin(cell_types_roi, list(b_types))
    n_b = int(np.sum(b_mask))
    if n_b >= 1:
        b_coords = np.column_stack([cx_roi[b_mask], cy_roi[b_mask]])
        tree_b = KDTree(b_coords)
        d_fdc_to_b, _ = tree_b.query(fdc_coords, k=1)
        fdc_b_dist = float(np.mean(d_fdc_to_b))
    else:
        fdc_b_dist = np.nan

    return {
        'fdc_fraction': fdc_fraction,
        'fdc_mean_nnd': fdc_mean_nnd,
        'fdc_connectivity': fdc_connectivity,
        'fdc_foll_frac': fdc_foll_frac,
        'fdc_b_dist': fdc_b_dist,
        'n_fdc': n_fdc,
    }


# ---------------------------------------------------------------------------
# Follicularity and entropy helpers
# ---------------------------------------------------------------------------
def compute_follicularity(domains_roi, foll_domains):
    """Fraction of cells in follicular domains."""
    is_foll = np.isin(domains_roi, foll_domains)
    return float(np.mean(is_foll)) if len(domains_roi) > 0 else 0.0


def compute_clean_entropy(cell_types_roi):
    """Shannon entropy excluding LQ / Unassigned cells."""
    clean = cell_types_roi[~np.isin(cell_types_roi, list(LQ_TYPES))]
    if len(clean) < 20:
        return np.nan
    counts = Counter(clean)
    total = sum(counts.values())
    props = np.array([c / total for c in counts.values()])
    props = props[props > 0]
    return -np.sum(props * np.log2(props))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"{'='*70}")
    print("H1c: FDC Network Integrity — S-panel")
    print(f"{'='*70}")

    # Generate concept cartoon
    cartoon_path = generate_cartoon()

    # Load data
    print("\nLoading S-panel v8 data...")
    f_v8 = h5py.File(V8_PATH, 'r')
    f_utag = h5py.File(UTAG_PATH, 'r')

    sample_ids = load_array(f_v8, 'sample_id')
    tma_arr = load_array(f_v8, 'tma')
    cell_types = load_array(f_v8, 'cell_type')
    cx = f_v8['obs']['centroid_x'][:]
    cy = f_v8['obs']['centroid_y'][:]

    tumor_mask = get_tumor_mask(sample_ids)
    print(f"Total cells: {len(sample_ids):,}")
    print(f"Tumor cells: {np.sum(tumor_mask):,}")
    print(f"FDC cells (tumor): {np.sum((cell_types == 'FDC') & tumor_mask):,}")

    # Classify UTAG domains
    print("\nClassifying UTAG domains...")
    domains, foll_domains, inter_domains = classify_domains(f_utag, cell_types, PANEL)

    # -----------------------------------------------------------------------
    # Compute per-ROI FDC metrics
    # -----------------------------------------------------------------------
    print("\nComputing FDC metrics per ROI...")
    unique_rois = sorted(set(sample_ids[tumor_mask]))
    roi_data = []
    skipped_min_cells = 0
    skipped_no_fdc = 0

    for i, roi in enumerate(unique_rois):
        if (i + 1) % 20 == 0:
            print(f"  Processing ROI {i+1}/{len(unique_rois)}...")

        roi_mask = (sample_ids == roi) & tumor_mask
        ct_roi = cell_types[roi_mask]

        # Min cells filter (non-LQ)
        n_typed = int(np.sum(~np.isin(ct_roi, list(LQ_TYPES))))
        tma_val = tma_arr[roi_mask][0]
        min_cells = MIN_CELLS_BIOMAX if 'Biomax' in tma_val else MIN_CELLS_DEFAULT
        if n_typed < min_cells:
            skipped_min_cells += 1
            continue

        cx_roi = cx[roi_mask]
        cy_roi = cy[roi_mask]
        domains_roi = domains[roi_mask]

        metrics = compute_fdc_metrics(cx_roi, cy_roi, ct_roi, domains_roi,
                                      foll_domains)
        if metrics is None:
            skipped_no_fdc += 1
            continue

        follicularity = compute_follicularity(domains_roi, foll_domains)
        entropy = compute_clean_entropy(ct_roi)

        roi_data.append({
            'roi': roi,
            'tma': tma_val,
            'n_typed': n_typed,
            'follicularity': follicularity,
            'entropy': entropy,
            **metrics,
        })

    print(f"\nROIs with FDC metrics: {len(roi_data)}")
    print(f"Skipped (min_cells): {skipped_min_cells}")
    print(f"Skipped (no/few FDC): {skipped_no_fdc}")

    if len(roi_data) < 10:
        print("ERROR: Too few ROIs with FDC data. Aborting.")
        f_v8.close()
        f_utag.close()
        return

    # Extract arrays
    rois = [d['roi'] for d in roi_data]
    tmas = [d['tma'] for d in roi_data]
    fdc_frac = np.array([d['fdc_fraction'] for d in roi_data])
    fdc_nnd = np.array([d['fdc_mean_nnd'] for d in roi_data])
    fdc_conn = np.array([d['fdc_connectivity'] for d in roi_data])
    fdc_foll = np.array([d['fdc_foll_frac'] for d in roi_data])
    fdc_bdist = np.array([d['fdc_b_dist'] for d in roi_data])
    follicularity = np.array([d['follicularity'] for d in roi_data])
    entropy = np.array([d['entropy'] for d in roi_data])

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FDC NETWORK METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"FDC fraction:     mean={np.mean(fdc_frac):.4f}, "
          f"median={np.median(fdc_frac):.4f}, "
          f"range=[{np.min(fdc_frac):.4f}, {np.max(fdc_frac):.4f}]")
    print(f"FDC mean NND:     mean={np.nanmean(fdc_nnd):.1f} px, "
          f"median={np.nanmedian(fdc_nnd):.1f} px")
    print(f"FDC connectivity: mean={np.mean(fdc_conn):.3f}, "
          f"median={np.median(fdc_conn):.3f}, "
          f"range=[{np.min(fdc_conn):.3f}, {np.max(fdc_conn):.3f}]")
    print(f"FDC follicular%:  mean={np.nanmean(fdc_foll):.3f}, "
          f"median={np.nanmedian(fdc_foll):.3f}")
    valid_bdist = ~np.isnan(fdc_bdist)
    print(f"FDC-to-B dist:    mean={np.nanmean(fdc_bdist):.1f} px, "
          f"median={np.nanmedian(fdc_bdist):.1f} px")

    # -----------------------------------------------------------------------
    # Correlations
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CORRELATIONS (Spearman)")
    print(f"{'='*60}")

    # Connectivity vs follicularity
    rho_cf, p_cf = spearmanr(fdc_conn, follicularity)
    print(f"FDC connectivity vs follicularity:  rho={rho_cf:.3f}, p={p_cf:.2e}")

    # Connectivity vs entropy
    valid_ent = ~np.isnan(entropy)
    rho_ce, p_ce = spearmanr(fdc_conn[valid_ent], entropy[valid_ent])
    print(f"FDC connectivity vs entropy:        rho={rho_ce:.3f}, p={p_ce:.2e}")

    # Connectivity vs FDC fraction
    rho_ccf, p_ccf = spearmanr(fdc_conn, fdc_frac)
    print(f"FDC connectivity vs FDC fraction:   rho={rho_ccf:.3f}, p={p_ccf:.2e}")

    # FDC fraction vs follicularity
    rho_ff, p_ff = spearmanr(fdc_frac, follicularity)
    print(f"FDC fraction vs follicularity:      rho={rho_ff:.3f}, p={p_ff:.2e}")

    # FDC NND vs connectivity
    valid_nnd = ~np.isnan(fdc_nnd)
    rho_nc, p_nc = spearmanr(fdc_nnd[valid_nnd], fdc_conn[valid_nnd])
    print(f"FDC mean NND vs connectivity:       rho={rho_nc:.3f}, p={p_nc:.2e}")

    # FDC follicular localisation vs connectivity
    valid_fl = ~np.isnan(fdc_foll)
    rho_flc, p_flc = spearmanr(fdc_foll[valid_fl], fdc_conn[valid_fl])
    print(f"FDC foll-localisation vs connectiv.: rho={rho_flc:.3f}, p={p_flc:.2e}")

    # FDC-B distance vs connectivity
    if np.sum(valid_bdist) > 10:
        rho_bc, p_bc = spearmanr(fdc_bdist[valid_bdist], fdc_conn[valid_bdist])
        print(f"FDC-to-B distance vs connectivity:  rho={rho_bc:.3f}, p={p_bc:.2e}")

    # -----------------------------------------------------------------------
    # Across-TMA comparison (Kruskal-Wallis)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PER-TMA BREAKDOWN")
    print(f"{'='*60}")
    tma_unique = sorted(set(tmas))
    tma_groups_conn = {}
    print(f"{'TMA':>8} {'n':>5} {'mean_frac':>10} {'mean_NND':>10} "
          f"{'mean_conn':>10} {'mean_foll%':>11}")
    for t in tma_unique:
        idx_t = [i for i, tt in enumerate(tmas) if tt == t]
        tma_groups_conn[t] = fdc_conn[idx_t]
        print(f"{t:>8} {len(idx_t):>5} "
              f"{np.mean(fdc_frac[idx_t]):>10.4f} "
              f"{np.nanmean(fdc_nnd[idx_t]):>10.1f} "
              f"{np.mean(fdc_conn[idx_t]):>10.3f} "
              f"{np.nanmean(fdc_foll[idx_t]):>11.3f}")

    if len(tma_unique) >= 3:
        groups = [fdc_conn[[i for i, tt in enumerate(tmas) if tt == t]]
                  for t in tma_unique]
        groups_nz = [g for g in groups if len(g) >= 2]
        if len(groups_nz) >= 3:
            kw_stat, kw_p = kruskal(*groups_nz)
            print(f"\nKruskal-Wallis (connectivity across TMAs): H={kw_stat:.2f}, p={kw_p:.2e}")

    # -----------------------------------------------------------------------
    # Leave-one-TMA-out sensitivity
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("LEAVE-ONE-TMA-OUT SENSITIVITY")
    print(f"{'='*60}")
    loo_results = []
    for exclude in tma_unique:
        idx_keep = [i for i, tt in enumerate(tmas) if tt != exclude]
        conn_keep = fdc_conn[idx_keep]
        foll_keep = follicularity[idx_keep]
        ent_keep = entropy[idx_keep]

        rho_f, p_f = spearmanr(conn_keep, foll_keep)
        valid_e = ~np.isnan(ent_keep)
        if np.sum(valid_e) > 5:
            rho_e, p_e = spearmanr(conn_keep[valid_e], ent_keep[valid_e])
        else:
            rho_e, p_e = np.nan, np.nan

        loo_results.append({
            'excluded': exclude,
            'n': len(idx_keep),
            'rho_foll': rho_f, 'p_foll': p_f,
            'rho_ent': rho_e, 'p_ent': p_e,
        })
        print(f"Exclude {exclude} (n={len(idx_keep)}): "
              f"conn~foll rho={rho_f:.3f} p={p_f:.2e}, "
              f"conn~ent rho={rho_e:.3f} p={p_e:.2e}")

    # -----------------------------------------------------------------------
    # FDC follicular localisation test: follicular vs interfollicular
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FDC DOMAIN LOCALISATION")
    print(f"{'='*60}")
    fdc_all_mask = (cell_types == 'FDC') & tumor_mask
    fdc_domains_all = domains[fdc_all_mask]
    fdc_in_foll = np.sum(np.isin(fdc_domains_all, foll_domains))
    fdc_in_inter = np.sum(np.isin(fdc_domains_all, inter_domains))
    fdc_total = np.sum(fdc_all_mask)
    fdc_other = fdc_total - fdc_in_foll - fdc_in_inter
    print(f"FDC in follicular:       {fdc_in_foll:>7,} ({fdc_in_foll/fdc_total*100:.1f}%)")
    print(f"FDC in interfollicular:  {fdc_in_inter:>7,} ({fdc_in_inter/fdc_total*100:.1f}%)")
    print(f"FDC in other/excluded:   {fdc_other:>7,} ({fdc_other/fdc_total*100:.1f}%)")

    # Compare per-ROI: FDC in foll vs inter (paired Wilcoxon would be ideal
    # but since fractions are bounded we use Mann-Whitney on per-ROI foll_frac)
    print(f"Per-ROI FDC follicular fraction: mean={np.nanmean(fdc_foll):.3f}, "
          f"median={np.nanmedian(fdc_foll):.3f}")
    # Test if FDC foll frac > 0.5 (more follicular than random)
    fdc_foll_clean = fdc_foll[~np.isnan(fdc_foll)]
    u_stat, u_p = mannwhitneyu(fdc_foll_clean, np.full_like(fdc_foll_clean, 0.5),
                                alternative='greater')
    print(f"Mann-Whitney U (foll_frac > 0.5): U={u_stat:.0f}, p={u_p:.2e}")

    # -----------------------------------------------------------------------
    # Driver analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DRIVER ANALYSIS (cell type correlations with FDC connectivity)")
    print(f"{'='*60}")
    roi_metric = {d['roi']: d['fdc_connectivity'] for d in roi_data}
    roi_fracs = compute_roi_celltype_fractions(sample_ids, cell_types, tumor_mask, LQ_TYPES)
    drivers = correlate_celltype_with_metric(roi_fracs, roi_metric)
    for ct, rho, p in drivers[:15]:
        print(f"  {ct:40s}  rho={rho:+.3f}  p={p:.2e}")

    # ===================================================================
    # MAIN FIGURE (2x3)
    # ===================================================================
    print("\nGenerating main figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Concept cartoon
    add_cartoon(axes[0, 0], cartoon_path)
    label_panel(axes[0, 0], 'a')

    # (b) FDC connectivity vs follicularity scatter
    ax = axes[0, 1]
    for t in tma_unique:
        idx_t = [i for i, tt in enumerate(tmas) if tt == t]
        ax.scatter(follicularity[idx_t], fdc_conn[idx_t],
                   c=TMA_COLORS.get(t, 'gray'), alpha=0.6, s=25,
                   edgecolors='white', linewidths=0.3, label=t)
    z = np.polyfit(follicularity, fdc_conn, 1)
    x_line = np.linspace(follicularity.min(), follicularity.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5, lw=1.5)
    ax.set_xlabel('Follicularity (fraction in follicular domains)', fontsize=10)
    ax.set_ylabel('FDC Connectivity (LCC / total FDC)', fontsize=10)
    ax.set_title(f'FDC Connectivity vs Follicularity\n'
                 f'(Spearman rho={rho_cf:.3f}, p={p_cf:.2e})', fontsize=11)
    ax.legend(fontsize=7, loc='best')
    label_panel(ax, 'b')

    # (c) FDC connectivity vs entropy scatter
    ax = axes[0, 2]
    for t in tma_unique:
        idx_t = [i for i, tt in enumerate(tmas) if tt == t and valid_ent[i]]
        if len(idx_t) > 0:
            ax.scatter(entropy[idx_t], fdc_conn[idx_t],
                       c=TMA_COLORS.get(t, 'gray'), alpha=0.6, s=25,
                       edgecolors='white', linewidths=0.3, label=t)
    z2 = np.polyfit(entropy[valid_ent], fdc_conn[valid_ent], 1)
    x_line2 = np.linspace(np.nanmin(entropy), np.nanmax(entropy), 100)
    ax.plot(x_line2, np.polyval(z2, x_line2), 'k--', alpha=0.5, lw=1.5)
    ax.set_xlabel('Shannon Entropy (clean)', fontsize=10)
    ax.set_ylabel('FDC Connectivity (LCC / total FDC)', fontsize=10)
    ax.set_title(f'FDC Connectivity vs Entropy\n'
                 f'(Spearman rho={rho_ce:.3f}, p={p_ce:.2e})', fontsize=11)
    ax.legend(fontsize=7, loc='best')
    label_panel(ax, 'c')

    # (d) Representative cores (low vs high FDC connectivity)
    ax = axes[1, 0]
    fdc_highlight = cell_types == 'FDC'
    roi_metric_conn = {d['roi']: d['fdc_connectivity'] for d in roi_data}
    plot_representative_core_spatial(
        ax, cx, cy, sample_ids, cell_types, tumor_mask,
        roi_metric_conn, label_lo='Low connectivity', label_hi='High connectivity',
        metric_name='Connectivity', top_n_types=10,
        domains=domains, foll_domains=foll_domains,
        highlight_mask=fdc_highlight, highlight_label='FDC',
        highlight_color='#FFD700', min_cells=MIN_CELLS_BIOMAX,
        highlight_size=18, min_highlight_hi=5, domain_focus=True,
    )
    label_panel(ax, 'd')

    # (e) FDC domain localisation box plot (follicular vs interfollicular)
    ax = axes[1, 1]
    # Per-ROI: fraction of FDC in follicular vs fraction in interfollicular
    fdc_inter_frac = []
    for d in roi_data:
        roi_mask_e = (sample_ids == d['roi']) & tumor_mask
        fdc_in_roi = (cell_types[roi_mask_e] == 'FDC')
        dom_in_roi = domains[roi_mask_e]
        n_fdc_roi = np.sum(fdc_in_roi)
        if n_fdc_roi > 0:
            n_inter = np.sum(np.isin(dom_in_roi[fdc_in_roi], inter_domains))
            fdc_inter_frac.append(n_inter / n_fdc_roi)
        else:
            fdc_inter_frac.append(np.nan)
    fdc_inter_frac = np.array(fdc_inter_frac)
    valid_loc = ~np.isnan(fdc_foll) & ~np.isnan(fdc_inter_frac)

    bp_data = [fdc_foll[valid_loc], fdc_inter_frac[valid_loc]]
    bp = ax.boxplot(bp_data, tick_labels=['Follicular', 'Interfollicular'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#FFDDDD')
    bp['boxes'][1].set_facecolor('#DDDDFF')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)
    # Overlay individual points
    for j, (data_j, color) in enumerate(zip(bp_data, ['#cc4444', '#4444cc'])):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(data_j))
        ax.scatter(np.full(len(data_j), j + 1) + jitter, data_j,
                   c=color, alpha=0.3, s=10, edgecolors='none', zorder=3)
    # Stats
    foll_vals = fdc_foll[valid_loc]
    inter_vals = fdc_inter_frac[valid_loc]
    u2, p2 = mannwhitneyu(foll_vals, inter_vals, alternative='greater')
    sig_txt = '***' if p2 < 0.001 else ('**' if p2 < 0.01 else ('*' if p2 < 0.05 else 'n.s.'))
    max_y = max(np.max(foll_vals), np.max(inter_vals))
    ax.plot([1, 1, 2, 2], [max_y + 0.03, max_y + 0.05, max_y + 0.05, max_y + 0.03],
            'k-', lw=1)
    ax.text(1.5, max_y + 0.06, sig_txt, ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fraction of FDC in domain type', fontsize=10)
    ax.set_title(f'FDC Domain Localisation\n'
                 f'(Mann-Whitney p={p2:.2e})', fontsize=11)
    label_panel(ax, 'e')

    # (f) Cell type drivers of FDC connectivity
    ax = axes[1, 2]
    top_n = min(12, len(drivers))
    if top_n > 0:
        top = drivers[:top_n]
        names_d = [d[0] for d in reversed(top)]
        rhos_d = [d[1] for d in reversed(top)]
        colors_bar = ['#e74c3c' if r < 0 else '#2ecc71' for r in rhos_d]
        ax.barh(range(len(names_d)), rhos_d, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names_d)))
        ax.set_yticklabels([n[:30] for n in names_d], fontsize=7)
        ax.set_xlabel('Spearman rho', fontsize=10)
        ax.set_title('Cell Type Drivers\n(correlation with FDC connectivity)', fontsize=11)
        ax.axvline(0, color='gray', lw=0.5)
    label_panel(ax, 'f')

    plt.suptitle('H1c: FDC Network Integrity — S-panel', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(OUTPUT_DIR, 'fig_h1c_S.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Main figure saved: {fig_path}")

    # ===================================================================
    # SUPPLEMENTARY FIGURE (per-TMA sensitivity)
    # ===================================================================
    print("\nGenerating supplementary figure...")
    fig_supp, axes_supp = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Per-TMA boxplot of FDC connectivity
    ax = axes_supp[0, 0]
    box_data_tma = [tma_groups_conn.get(t, np.array([])) for t in tma_unique]
    bp2 = ax.boxplot(box_data_tma, tick_labels=tma_unique, patch_artist=True)
    for patch, t in zip(bp2['boxes'], tma_unique):
        patch.set_facecolor(TMA_COLORS.get(t, 'gray'))
        patch.set_alpha(0.6)
    ax.set_ylabel('FDC Connectivity (LCC / total FDC)')
    ax.set_xlabel('TMA')
    ax.set_title('Per-TMA FDC Connectivity')
    label_panel(ax, 'a')

    # (b) Leave-one-out: connectivity~follicularity
    ax = axes_supp[0, 1]
    excl_names = [r['excluded'] for r in loo_results]
    rhos_foll_loo = [r['rho_foll'] for r in loo_results]
    pvals_foll_loo = [r['p_foll'] for r in loo_results]
    bar_colors = [TMA_COLORS.get(n, 'gray') for n in excl_names]
    ax.bar(excl_names, rhos_foll_loo, color=bar_colors, alpha=0.7)
    ax.axhline(rho_cf, color='black', linestyle='--', lw=1,
               label=f'Full cohort (rho={rho_cf:.3f})')
    for i, (rv, pv) in enumerate(zip(rhos_foll_loo, pvals_foll_loo)):
        sig = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else 'n.s.'))
        ax.text(i, rv + 0.01, sig, ha='center', fontsize=8)
    ax.set_ylabel('Spearman rho (connectivity vs follicularity)')
    ax.set_xlabel('Excluded TMA')
    ax.set_title('Leave-One-Out: Follicularity')
    ax.legend(fontsize=8)
    label_panel(ax, 'b')

    # (c) Leave-one-out: connectivity~entropy
    ax = axes_supp[0, 2]
    rhos_ent_loo = [r['rho_ent'] for r in loo_results]
    pvals_ent_loo = [r['p_ent'] for r in loo_results]
    ax.bar(excl_names, rhos_ent_loo, color=bar_colors, alpha=0.7)
    ax.axhline(rho_ce, color='black', linestyle='--', lw=1,
               label=f'Full cohort (rho={rho_ce:.3f})')
    for i, (rv, pv) in enumerate(zip(rhos_ent_loo, pvals_ent_loo)):
        sig = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else 'n.s.'))
        ax.text(i, rv + 0.01 * np.sign(rv), sig, ha='center', fontsize=8)
    ax.set_ylabel('Spearman rho (connectivity vs entropy)')
    ax.set_xlabel('Excluded TMA')
    ax.set_title('Leave-One-Out: Entropy')
    ax.legend(fontsize=8)
    label_panel(ax, 'c')

    # (d) FDC mean NND distribution per TMA
    ax = axes_supp[1, 0]
    for t in tma_unique:
        idx_t = [i for i, tt in enumerate(tmas) if tt == t and valid_nnd[i]]
        if len(idx_t) > 0:
            ax.hist(fdc_nnd[idx_t], bins=20, alpha=0.5,
                    color=TMA_COLORS.get(t, 'gray'), label=t, edgecolor='white')
    ax.set_xlabel('FDC Mean Nearest-Neighbour Distance (px)')
    ax.set_ylabel('Number of ROIs')
    ax.set_title('FDC Clustering (NND)')
    ax.legend(fontsize=7)
    label_panel(ax, 'd')

    # (e) FDC fraction vs connectivity scatter
    ax = axes_supp[1, 1]
    for t in tma_unique:
        idx_t = [i for i, tt in enumerate(tmas) if tt == t]
        ax.scatter(fdc_frac[idx_t], fdc_conn[idx_t],
                   c=TMA_COLORS.get(t, 'gray'), alpha=0.6, s=25,
                   edgecolors='white', linewidths=0.3, label=t)
    if np.any(fdc_frac > 0):
        z3 = np.polyfit(fdc_frac, fdc_conn, 1)
        x3 = np.linspace(fdc_frac.min(), fdc_frac.max(), 100)
        ax.plot(x3, np.polyval(z3, x3), 'k--', alpha=0.5, lw=1.5)
    ax.set_xlabel('FDC Fraction')
    ax.set_ylabel('FDC Connectivity')
    ax.set_title(f'FDC Fraction vs Connectivity\n(rho={rho_ccf:.3f}, p={p_ccf:.2e})')
    ax.legend(fontsize=7)
    label_panel(ax, 'e')

    # (f) FDC-to-B distance vs connectivity scatter
    ax = axes_supp[1, 2]
    valid_bd = ~np.isnan(fdc_bdist)
    if np.sum(valid_bd) > 10:
        rho_bd_c, p_bd_c = spearmanr(fdc_bdist[valid_bd], fdc_conn[valid_bd])
        for t in tma_unique:
            idx_t = [i for i, tt in enumerate(tmas) if tt == t and valid_bd[i]]
            if len(idx_t) > 0:
                ax.scatter(fdc_bdist[idx_t], fdc_conn[idx_t],
                           c=TMA_COLORS.get(t, 'gray'), alpha=0.6, s=25,
                           edgecolors='white', linewidths=0.3, label=t)
        ax.set_xlabel('FDC-to-B Mean Distance (px)')
        ax.set_ylabel('FDC Connectivity')
        ax.set_title(f'FDC-B Co-localisation vs Connectivity\n'
                     f'(rho={rho_bd_c:.3f}, p={p_bd_c:.2e})')
        ax.legend(fontsize=7)
    label_panel(ax, 'f')

    plt.suptitle('H1c: FDC Network Integrity — S-panel (Supplementary)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    supp_path = os.path.join(OUTPUT_DIR, 'fig_h1c_S_supp.png')
    plt.savefig(supp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Supplementary figure saved: {supp_path}")

    # ===================================================================
    # Final summary
    # ===================================================================
    print(f"\n{'='*70}")
    print("H1c SUMMARY")
    print(f"{'='*70}")
    print(f"ROIs analysed: {len(roi_data)}")
    print(f"FDC connectivity range: [{np.min(fdc_conn):.3f}, {np.max(fdc_conn):.3f}]")
    print(f"FDC connectivity vs follicularity: rho={rho_cf:.3f}, p={p_cf:.2e}")
    print(f"FDC connectivity vs entropy:       rho={rho_ce:.3f}, p={p_ce:.2e}")
    print(f"FDC follicular localisation: {np.nanmean(fdc_foll)*100:.1f}% "
          f"(Mann-Whitney vs 50%: p={u_p:.2e})")

    # Determine status
    all_loo_foll_sig = all(r['p_foll'] < 0.05 for r in loo_results)
    if p_cf < 0.05 and all_loo_foll_sig:
        status = "CONFIRMED"
        note = ("FDC connectivity positively correlates with follicularity "
                f"(rho={rho_cf:.3f}, p={p_cf:.2e}), robust across TMAs.")
    elif p_cf < 0.05:
        status = "CONFIRMED"
        note = ("FDC connectivity correlates with follicularity "
                f"(rho={rho_cf:.3f}, p={p_cf:.2e}); some LOO instability.")
    elif p_cf < 0.10:
        status = "INCONCLUSIVE"
        note = f"Marginal correlation (rho={rho_cf:.3f}, p={p_cf:.2e})."
    else:
        status = "NOT CONFIRMED"
        note = f"No significant correlation (rho={rho_cf:.3f}, p={p_cf:.2e})."

    print(f"\n>>> STATUS: {status}")
    print(f">>> {note}")
    print(f"\nMain figure:  {fig_path}")
    print(f"Supp figure:  {supp_path}")

    f_v8.close()
    f_utag.close()


if __name__ == '__main__':
    main()
