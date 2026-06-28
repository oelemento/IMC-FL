#!/usr/bin/env python3
"""Test compartment-specific biology within follicular sub-zones.

Questions:
1. Do cell-cell neighbor patterns differ across follicular sub-compartments?
2. Do marker expression profiles on B cells differ by compartment?
3. Are there compartment-specific interaction signatures (Tfh-B, Treg-CD8, Mac-B)?
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clinical_linkage import load_clinical, normalize_sample_id

EXCLUDE_PAT = r'(?i)tonsil|prostate|kidney|spleen|adrenal|_ton_|_adr_'

FOLL_COMPARTMENTS = [
    "GC core",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)",
    "Activated B / CXCR5hi zone",
    "B cell follicle (CD20hi/CXCR5hi)",
    "B cell zone",
]

INTERFACE = "Follicle-T zone interface"
TREG_ZONE = "Treg-enriched T zone"
T_ZONE = "T cell zone (CD4/CD8)"

ALL_ZONES = FOLL_COMPARTMENTS + [INTERFACE, TREG_ZONE, T_ZONE]

# Cell types of interest
B_TYPES = ['GC B cells', 'B cells (CD20hi)', 'B cells (CXCR5hi)',
           'Other B cells', 'B cells (TOXhi)', 'Activated B / Plasmablast',
           'B cells (weak CD20)']
TFH = ['Tfh']
TREG = ['Treg']
CD8 = ['CD8 T cells', 'CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)', 'Macrophages (GzmB+)']
CD8_EXH = ['CD8 T exhausted', 'CD8 T pre-exhausted (TOX+)']
CD4 = ['CD4 T cells']
MAC = ['Macrophages']


def compute_neighbors(cx, cy, ctypes, comps, k=10, max_dist=50):
    """For each cell, find k nearest neighbors and record their types."""
    coords = np.column_stack([cx, cy])
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k+1)  # +1 for self

    # For each compartment, compute neighbor type fractions
    results = {}
    for comp in ALL_ZONES:
        comp_mask = comps == comp
        if comp_mask.sum() < 30:
            continue

        # Neighbor composition for cells in this compartment
        neighbor_counts = {}
        n_cells = 0
        for i in np.where(comp_mask)[0]:
            neighs = indices[i, 1:]  # skip self
            neigh_dists = dists[i, 1:]
            close = neighs[neigh_dists < max_dist]
            if len(close) == 0:
                continue
            n_cells += 1
            for ni in close:
                ct = ctypes[ni]
                neighbor_counts[ct] = neighbor_counts.get(ct, 0) + 1

        if n_cells < 20:
            continue

        total = sum(neighbor_counts.values())
        neighbor_fracs = {ct: n/total for ct, n in neighbor_counts.items()}
        results[comp] = {'fracs': neighbor_fracs, 'n_cells': n_cells}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-panel", required=True)
    parser.add_argument("--t-utag", required=True)
    args = parser.parse_args()

    print("=== Follicular sub-compartment biology ===\n")

    t_panel = ad.read_h5ad(args.t_panel)
    t_utag = ad.read_h5ad(args.t_utag)

    # Get compartment and cell type info
    roi_col = 'sample_id'
    comp_col = 'compartment_name'
    ct_col = 'cell_type'

    mask = ~t_utag.obs[roi_col].str.contains(EXCLUDE_PAT, na=False)
    obs = t_utag.obs[mask].copy()

    rois = obs[roi_col].unique()
    print(f"ROIs: {len(rois)}")

    # ================================================================
    # 1. Cell type fractions BY compartment (pooled across all ROIs)
    # ================================================================
    print("\n" + "="*60)
    print("1. CELL TYPE COMPOSITION BY COMPARTMENT")
    print("="*60)

    ct_groups = {
        'B cells': B_TYPES, 'Tfh': TFH, 'Treg': TREG,
        'CD8 (all)': CD8, 'CD8 exhausted': CD8_EXH,
        'CD4 T': CD4, 'Macrophages': MAC,
    }

    comp_fracs = {}
    for comp in ALL_ZONES:
        cm = obs[comp_col] == comp
        n = cm.sum()
        if n < 100:
            continue
        comp_cts = obs[cm][ct_col]
        fracs = {}
        for group_name, group_types in ct_groups.items():
            fracs[group_name] = comp_cts.isin(group_types).mean()
        comp_fracs[comp] = fracs
        comp_fracs[comp]['n'] = n

    if comp_fracs:
        df = pd.DataFrame(comp_fracs).T
        # Reorder
        order = [c for c in ALL_ZONES if c in df.index]
        df = df.loc[order]
        print(f"\n{'Compartment':<45} {'B%':>6} {'Tfh%':>6} {'Treg%':>6} {'CD8%':>6} {'Exh%':>6} {'CD4%':>6} {'Mac%':>6} {'n':>8}")
        print("-" * 110)
        for comp, row in df.iterrows():
            short = comp[:44]
            print(f"{short:<45} {row.get('B cells',0)*100:>5.1f}% {row.get('Tfh',0)*100:>5.1f}% "
                  f"{row.get('Treg',0)*100:>5.1f}% {row.get('CD8 (all)',0)*100:>5.1f}% "
                  f"{row.get('CD8 exhausted',0)*100:>5.1f}% {row.get('CD4 T',0)*100:>5.1f}% "
                  f"{row.get('Macrophages',0)*100:>5.1f}% {int(row.get('n',0)):>8}")

    # ================================================================
    # 2. Neighborhood analysis by compartment (pooled)
    # ================================================================
    print("\n" + "="*60)
    print("2. NEIGHBORHOOD COMPOSITION BY COMPARTMENT (k=10, <50px)")
    print("   What types surround cells in each compartment?")
    print("="*60)

    # Pool across ROIs — sample up to 50 ROIs for speed
    np.random.seed(42)
    sample_rois = np.random.choice(rois, min(50, len(rois)), replace=False)

    pooled_neighbors = {comp: {} for comp in ALL_ZONES}
    pooled_n = {comp: 0 for comp in ALL_ZONES}

    for roi in sample_rois:
        rm = obs[roi_col] == roi
        roi_obs = obs[rm]
        if len(roi_obs) < 200:
            continue

        cx = roi_obs['centroid_x'].values.astype(float)
        cy = roi_obs['centroid_y'].values.astype(float)
        ctypes = roi_obs[ct_col].values
        comps = roi_obs[comp_col].values

        result = compute_neighbors(cx, cy, ctypes, comps)
        for comp, data in result.items():
            pooled_n[comp] += data['n_cells']
            for ct, frac in data['fracs'].items():
                pooled_neighbors[comp][ct] = pooled_neighbors[comp].get(ct, 0) + frac * data['n_cells']

    # Normalize
    print(f"\n{'Compartment':<45} {'B neigh%':>8} {'Tfh%':>6} {'Treg%':>6} {'CD8%':>6} {'Exh%':>6} {'Mac%':>6}")
    print("-" * 90)
    for comp in ALL_ZONES:
        if pooled_n[comp] < 50:
            continue
        total = sum(pooled_neighbors[comp].values())
        if total < 1:
            continue

        b_frac = sum(pooled_neighbors[comp].get(t, 0) for t in B_TYPES) / total
        tfh_frac = sum(pooled_neighbors[comp].get(t, 0) for t in TFH) / total
        treg_frac = sum(pooled_neighbors[comp].get(t, 0) for t in TREG) / total
        cd8_frac = sum(pooled_neighbors[comp].get(t, 0) for t in CD8) / total
        exh_frac = sum(pooled_neighbors[comp].get(t, 0) for t in CD8_EXH) / total
        mac_frac = sum(pooled_neighbors[comp].get(t, 0) for t in MAC) / total

        short = comp[:44]
        print(f"{short:<45} {b_frac*100:>7.1f}% {tfh_frac*100:>5.1f}% {treg_frac*100:>5.1f}% "
              f"{cd8_frac*100:>5.1f}% {exh_frac*100:>5.1f}% {mac_frac*100:>5.1f}%")

    # ================================================================
    # 3. Key interaction ratios by compartment
    # ================================================================
    print("\n" + "="*60)
    print("3. KEY INTERACTION RATIOS BY COMPARTMENT")
    print("="*60)

    print(f"\n{'Compartment':<45} {'Tfh:CD8':>8} {'Treg:CD8':>9} {'Exh/CD8':>8} {'Mac:B':>8}")
    print("-" * 85)
    for comp in ALL_ZONES:
        if comp not in comp_fracs:
            continue
        f = comp_fracs[comp]
        cd8_f = f.get('CD8 (all)', 0)
        tfh_f = f.get('Tfh', 0)
        treg_f = f.get('Treg', 0)
        exh_f = f.get('CD8 exhausted', 0)
        b_f = f.get('B cells', 0)
        mac_f = f.get('Macrophages', 0)

        tfh_cd8 = tfh_f / cd8_f if cd8_f > 0.001 else np.nan
        treg_cd8 = treg_f / cd8_f if cd8_f > 0.001 else np.nan
        exh_ratio = exh_f / cd8_f if cd8_f > 0.001 else np.nan
        mac_b = mac_f / b_f if b_f > 0.001 else np.nan

        short = comp[:44]
        parts = []
        for v in [tfh_cd8, treg_cd8, exh_ratio, mac_b]:
            if np.isnan(v):
                parts.append(f"{'n/a':>8}")
            else:
                parts.append(f"{v:>8.2f}")
        print(f"{short:<45} {''.join(parts)}")

    # ================================================================
    # 4. Statistical tests: are interactions compartment-specific?
    # ================================================================
    print("\n" + "="*60)
    print("4. COMPARTMENT-SPECIFIC INTERACTIONS (per-ROI tests)")
    print("   Kruskal-Wallis across follicular sub-compartments")
    print("="*60)

    # Per-ROI, per-compartment cell type fractions
    for ct_name, ct_list in [('Tfh fraction', TFH), ('Treg fraction', TREG),
                              ('CD8 exhausted fraction', CD8_EXH),
                              ('Macrophage fraction', MAC)]:
        per_comp_values = {comp: [] for comp in FOLL_COMPARTMENTS}

        for roi in rois:
            rm = obs[roi_col] == roi
            roi_obs = obs[rm]
            for comp in FOLL_COMPARTMENTS:
                cm = roi_obs[comp_col] == comp
                n = cm.sum()
                if n < 30:
                    continue
                frac = roi_obs[cm][ct_col].isin(ct_list).mean()
                per_comp_values[comp].append(float(frac))

        # KW test
        groups = [v for v in per_comp_values.values() if len(v) >= 10]
        # Filter out constant groups
        groups = [v for v in groups if np.std(v) > 1e-10]
        if len(groups) >= 3:
            try:
                h, p = stats.kruskal(*groups)
            except ValueError:
                print(f"\n  {ct_name}: all values identical, skipping")
                continue
            sizes = [len(v) for v in per_comp_values.values()]
            means = {comp[:20]: f"{np.mean(v):.4f}" for comp, v in per_comp_values.items() if len(v) >= 10}
            sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            print(f"\n  {ct_name}: KW H={h:.1f}, p={p:.4f}{sig}")
            print(f"    Means: {means}")
        else:
            print(f"\n  {ct_name}: insufficient compartments with data")


if __name__ == "__main__":
    main()
