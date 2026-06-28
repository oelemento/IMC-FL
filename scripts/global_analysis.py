#!/usr/bin/env python3
"""Global re-clustering and annotation of combined ROI data.

Loads a combined h5ad file, runs PCA/neighbors/UMAP/Leiden on all cells
jointly, then applies rule-based cell type annotation.

Usage:
    python scripts/global_analysis.py --panel T --input output/batch/TMA_B1_T_combined.h5ad
    python scripts/global_analysis.py --panel S --input output/batch_S/TMA_B1_S_combined.h5ad
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import time
import argparse

sc.settings.verbosity = 2

parser = argparse.ArgumentParser(description='Global re-clustering of combined IMC data')
parser.add_argument('--panel', required=True, choices=['T', 'S'], help='Panel type')
parser.add_argument('--input', required=True, help='Path to combined h5ad')
parser.add_argument('--output', default=None, help='Output h5ad path (default: same dir, _global suffix)')


def run_global_clustering(adata, panel):
    """Run full clustering pipeline on combined data."""
    COFACTOR = 5
    EXCLUDE_CHANNELS = ['80ArAr', '129Xe', '190BCKG', 'Pb204', 'Pushes_elapsed']
    STRUCTURAL_T = ['DNA1', 'DNA2', 'HistoneH3', 'p_H3s28', 'H3K27me3']
    STRUCTURAL_S = ['DNA1', 'DNA2', 'HistoneH3', 'p_H3s28']
    STRUCTURAL = STRUCTURAL_T if panel == 'T' else STRUCTURAL_S

    # Filter to bio markers
    bio = [m for m in adata.var_names if m not in EXCLUDE_CHANNELS]
    adata = adata[:, bio].copy()

    structural = [m for m in STRUCTURAL if m in adata.var_names]
    cluster_markers = [m for m in bio if m not in structural]

    print(f"Loaded: {adata.n_obs:,} cells x {len(bio)} markers")
    print(f"Raw shape: {adata.X.shape}, dtype: {adata.X.dtype}")
    print(f"Raw range: [{adata.X.min():.2f}, {adata.X.max():.2f}]")
    print(f"Bio markers: {len(bio)}")
    print(f"Clustering markers: {len(cluster_markers)}")
    print(f"  {cluster_markers}")

    # Store raw
    adata.raw = adata

    # arcsinh transform
    print("Arcsinh transform...")
    adata.X = np.arcsinh(adata.X / COFACTOR)

    # Scale
    print("Scaling...")
    sc.pp.scale(adata, max_value=10)

    # PCA
    print("PCA...")
    n_comps = min(30, len(cluster_markers) - 1, adata.n_obs - 1)
    adata_cluster = adata[:, cluster_markers].copy()
    sc.tl.pca(adata_cluster, svd_solver='arpack', n_comps=n_comps)
    adata.obsm['X_pca'] = adata_cluster.obsm['X_pca']
    var_explained = adata_cluster.uns['pca']['variance_ratio'][:10].sum() * 100
    print(f"PCA: {n_comps} components, variance explained: {var_explained:.1f}% (first 10)")

    # Neighbors
    print("Building neighbor graph (this may take a few minutes)...")
    t0 = time.time()
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(15, n_comps), use_rep='X_pca')
    print(f"Neighbors done in {time.time()-t0:.0f}s")

    # UMAP
    print("Computing UMAP...")
    t0 = time.time()
    sc.tl.umap(adata)
    print(f"UMAP done in {time.time()-t0:.0f}s")

    # Leiden at multiple resolutions
    for res in [0.3, 0.5, 0.8, 1.0]:
        print(f"Leiden resolution={res}...")
        t0 = time.time()
        sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
        n_clust = adata.obs[f'leiden_{res}'].nunique()
        print(f"  -> {n_clust} clusters in {time.time()-t0:.0f}s")

    adata.obs['leiden'] = adata.obs['leiden_0.5']

    return adata, cluster_markers


def annotate_T_panel(adata):
    """Rule-based annotation for T-panel."""
    adata_raw = adata.raw.to_adata()

    def mean_marker(marker):
        if marker not in adata_raw.var_names:
            return pd.Series(0, index=adata.obs['leiden'].unique())
        idx = list(adata_raw.var_names).index(marker)
        return adata.obs.groupby('leiden').apply(
            lambda g: float(adata_raw[g.index, idx].X.mean())
        )

    def pos_frac(marker, threshold=1.0):
        if marker not in adata_raw.var_names:
            return pd.Series(0, index=adata.obs['leiden'].unique())
        idx = list(adata_raw.var_names).index(marker)
        return adata.obs.groupby('leiden').apply(
            lambda g: float((adata_raw[g.index, idx].X > threshold).mean())
        )

    # Compute marker profiles
    cd3 = mean_marker('CD3')
    cd4 = mean_marker('CD4')
    cd8 = mean_marker('CD8a')
    cd20 = mean_marker('CD20')
    foxp3 = mean_marker('FoxP3')
    cd68 = mean_marker('CD68')
    gzmb = mean_marker('GranzymeB')
    cd38 = mean_marker('CD38')
    irf4 = mean_marker('IRF4')
    tox = mean_marker('TOX')
    cxcr5 = mean_marker('CXCR5')
    tim3 = mean_marker('TIM3')
    cd57 = mean_marker('CD57')
    cd31 = mean_marker('CD31')
    cd45ro = mean_marker('CD45RO')

    # Positive fractions
    cd20_pos = pos_frac('CD20', 1.0)
    cd3_pos = pos_frac('CD3', 0.5)
    cd68_pos = pos_frac('CD68', 1.0)

    # Print cluster profiles
    print("\n=== Cluster marker profiles ===")
    header = f" {'Cluster':>7}  {'n_cells':>7}   {'CD20':>6} {'CD3':>6} {'CD4':>6} {'CD8a':>6} {'CD68':>6} {'FoxP3':>6} {'TOX':>6} {'GzmB':>6} {'TIM3':>6} {'PD1':>6} {'CXCR5':>6} {'CD38':>6} {'IRF4':>6}"
    print(header)
    pd1 = mean_marker('PD_1')
    for c in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
        n = (adata.obs['leiden'] == c).sum()
        print(f" {c:>7}  {n:>7}   {cd20[c]:>6.1f} {cd3[c]:>6.1f} {cd4[c]:>6.1f} {cd8[c]:>6.1f} {cd68[c]:>6.1f} {foxp3[c]:>6.1f} {tox[c]:>6.1f} {gzmb[c]:>6.1f} {tim3[c]:>6.1f} {pd1[c]:>6.1f} {cxcr5[c]:>6.1f} {cd38[c]:>6.1f} {irf4[c]:>6.1f}")

    # Annotation rules
    annotations = {}
    for c in adata.obs['leiden'].unique():
        # B cell subtypes
        if cd20[c] > cd3[c] and cd20[c] > cd68[c] and cd20[c] > 1.0:
            if cxcr5[c] > 2.0 and cd20[c] > 8.0:
                annotations[c] = 'GC B cells'
            elif cd38[c] > 1.0 and irf4[c] > 0.5:
                annotations[c] = 'Activated B / Plasmablast'
            elif cd20[c] > 6.0:
                annotations[c] = 'B cells (CD20hi)'
            elif cxcr5[c] > 1.5:
                annotations[c] = 'B cells (CXCR5hi)'
            elif tox[c] > 1.5:
                annotations[c] = 'B cells (TOXhi)'
            else:
                annotations[c] = 'B cells'
        # Cytotoxic
        elif cd68[c] > 2.0 and gzmb[c] > 2.0:
            annotations[c] = 'Macrophages (GzmB+)'
        # Macrophages
        elif cd68[c] > 2.0 and cd68[c] > cd3[c]:
            annotations[c] = 'Macrophages'
        # TIM3-high
        elif tim3[c] > 3.0:
            annotations[c] = 'TIM3-high'
        # T cell subsets
        elif cd3[c] > 0.5 and cd8[c] > cd4[c] and tox[c] > 0.8:
            annotations[c] = 'CD8 T exhausted'
        elif cd3[c] > 0.5 and cd8[c] > cd4[c]:
            annotations[c] = 'CD8 T cells'
        elif cd3[c] > 0.5 and cd4[c] > cd8[c] and foxp3[c] > foxp3.median() * 1.5:
            annotations[c] = 'Treg'
        elif cd3[c] > 0.5 and cd4[c] > cd8[c]:
            annotations[c] = 'CD4 T cells'
        elif cd3[c] > 0.5:
            annotations[c] = 'T cells'
        # Mixed (multi-lineage)
        elif cd20_pos[c] > 0.3 and cd3_pos[c] > 0.3 and cd68_pos[c] > 0.3:
            annotations[c] = 'Mixed / Border cells'
        # Low expression
        elif cd20[c] < 1.5 and cd3[c] < 0.8 and cd68[c] < 1.5:
            annotations[c] = 'Low quality / Unassigned'
        else:
            annotations[c] = 'Other'

    adata.obs['cell_type'] = adata.obs['leiden'].map(annotations)

    print("\n=== Cluster -> Cell Type ===")
    for c in sorted(annotations.keys(), key=lambda x: int(x)):
        n = (adata.obs['leiden'] == c).sum()
        print(f"  Cluster {c}: {annotations[c]} ({n:,} cells)")

    return adata


def annotate_S_panel(adata):
    """Rule-based annotation for S-panel."""
    adata_raw = adata.raw.to_adata()

    def mean_marker(marker):
        if marker not in adata_raw.var_names:
            return pd.Series(0, index=adata.obs['leiden'].unique())
        idx = list(adata_raw.var_names).index(marker)
        return adata.obs.groupby('leiden').apply(
            lambda g: float(adata_raw[g.index, idx].X.mean())
        )

    def pos_frac(marker, threshold=1.0):
        if marker not in adata_raw.var_names:
            return pd.Series(0, index=adata.obs['leiden'].unique())
        idx = list(adata_raw.var_names).index(marker)
        return adata.obs.groupby('leiden').apply(
            lambda g: float((adata_raw[g.index, idx].X > threshold).mean())
        )

    cd20 = mean_marker('CD20')
    cd4 = mean_marker('CD4')
    cd8 = mean_marker('CD8a')
    cd68 = mean_marker('CD68')
    cd14 = mean_marker('CD14')
    cd163 = mean_marker('CD163')
    cd206 = mean_marker('CD206')
    cd11c = mean_marker('CD11c')
    cd11b = mean_marker('CD11b')
    cd31 = mean_marker('CD31')
    cd34 = mean_marker('CD34')
    vimentin = mean_marker('Vimentin')
    hla_dr = mean_marker('HLA_DR')
    bcl6 = mean_marker('BCL_6')
    bcl2 = mean_marker('BCL_2')
    pax5 = mean_marker('PAX5')
    cd21 = mean_marker('CD21')
    ki67 = mean_marker('Ki-67')
    pdpn = mean_marker('PDPN')
    fibronectin = mean_marker('Fibronectin')
    cd123 = mean_marker('CD123')
    cxcl13 = mean_marker('CXCL13')
    ido = mean_marker('IDO')
    vista = mean_marker('VISTA')
    cd44 = mean_marker('CD44')

    # Positive fractions for mixed cell detection
    cd20_pos = pos_frac('CD20', 1.0)
    cd68_pos = pos_frac('CD68', 1.0)

    # Print cluster profiles
    print("\n=== Cluster marker profiles ===")
    header = f" {'Clust':>5} {'n':>7}  {'CD20':>5} {'PAX5':>5} {'BCL6':>5} {'BCL2':>5} {'CD21':>5} {'CD68':>5} {'CD163':>5} {'CD206':>5} {'CD14':>5} {'CD11c':>5} {'CD31':>5} {'Vim':>5} {'PDPN':>5} {'FN':>5} {'CD4':>5} {'CD8a':>5} {'HLADR':>5} {'Ki67':>5}"
    print(header)
    for c in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
        n = (adata.obs['leiden'] == c).sum()
        print(f" {c:>5} {n:>7}  {cd20[c]:>5.1f} {pax5[c]:>5.1f} {bcl6[c]:>5.1f} {bcl2[c]:>5.1f} {cd21[c]:>5.1f} {cd68[c]:>5.1f} {cd163[c]:>5.1f} {cd206[c]:>5.1f} {cd14[c]:>5.1f} {cd11c[c]:>5.1f} {cd31[c]:>5.1f} {vimentin[c]:>5.1f} {pdpn[c]:>5.1f} {fibronectin[c]:>5.1f} {cd4[c]:>5.1f} {cd8[c]:>5.1f} {hla_dr[c]:>5.1f} {ki67[c]:>5.1f}")

    annotations = {}
    for c in adata.obs['leiden'].unique():
        # FDC: CD21+ CXCL13+
        if cd21[c] > 2.0 and cxcl13[c] > 1.0:
            annotations[c] = 'FDC'
        # GC B cells: CD20+ BCL6+
        elif cd20[c] > 2.0 and bcl6[c] > 1.0 and cd20[c] > cd68[c]:
            if ki67[c] > 1.0:
                annotations[c] = 'GC B cells (proliferating)'
            else:
                annotations[c] = 'GC B cells'
        # B cells: CD20+ or PAX5+
        elif cd20[c] > cd68[c] and cd20[c] > 1.0:
            if bcl2[c] > 2.0:
                annotations[c] = 'B cells (BCL2+)'
            elif pax5[c] > 1.0:
                annotations[c] = 'B cells (PAX5+)'
            else:
                annotations[c] = 'B cells'
        # M2 macrophages: CD68+ CD163+ or CD206+
        elif cd68[c] > 2.0 and (cd163[c] > 1.5 or cd206[c] > 1.5):
            annotations[c] = 'M2 Macrophages'
        # M1 macrophages: CD68+ CD11c+ or HLA-DR+
        elif cd68[c] > 2.0 and (cd11c[c] > 1.0 or hla_dr[c] > 2.0):
            annotations[c] = 'M1 Macrophages'
        # General macrophages: CD68+ or CD14+
        elif cd68[c] > 2.0 or (cd14[c] > 2.0 and cd14[c] > cd20[c]):
            annotations[c] = 'Macrophages'
        # Dendritic cells: CD11c+ HLA-DR+ CD14-
        elif cd11c[c] > 1.5 and hla_dr[c] > 1.5 and cd14[c] < 1.0:
            annotations[c] = 'Dendritic cells'
        # pDC: CD123+
        elif cd123[c] > 1.5:
            annotations[c] = 'pDC'
        # Endothelial: CD31+ or CD34+
        elif cd31[c] > 2.0 or cd34[c] > 2.0:
            annotations[c] = 'Endothelial'
        # Fibroblastic reticular cells: PDPN+
        elif pdpn[c] > 2.0:
            annotations[c] = 'FRC (PDPN+)'
        # Stromal: Vimentin+ or Fibronectin+
        elif vimentin[c] > 2.0 or fibronectin[c] > 2.0:
            annotations[c] = 'Stromal / Fibroblast'
        # T cells
        elif cd4[c] > cd20[c] and cd4[c] > 0.5:
            annotations[c] = 'CD4 T cells'
        elif cd8[c] > cd20[c] and cd8[c] > 0.5:
            annotations[c] = 'CD8 T cells'
        # Mixed / multi-lineage
        elif cd20_pos[c] > 0.3 and cd68_pos[c] > 0.3:
            annotations[c] = 'Mixed / Border cells'
        # Low expression
        elif cd20[c] < 1.0 and cd68[c] < 1.0 and cd4[c] < 0.5 and vimentin[c] < 1.0:
            annotations[c] = 'Low quality / Unassigned'
        else:
            annotations[c] = 'Other'

    adata.obs['cell_type'] = adata.obs['leiden'].map(annotations)

    print("\n=== Cluster -> Cell Type ===")
    for c in sorted(annotations.keys(), key=lambda x: int(x)):
        n = (adata.obs['leiden'] == c).sum()
        print(f"  Cluster {c}: {annotations[c]} ({n:,} cells)")

    return adata


def print_composition(adata):
    """Print global cell type composition."""
    print("\n=== Global Cell Type Composition ===")
    comp = adata.obs['cell_type'].value_counts()
    for ct, n in comp.items():
        pct = n / adata.n_obs * 100
        print(f"  {ct}: {n:,} ({pct:.1f}%)")


if __name__ == '__main__':
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / input_path.name.replace('_combined.h5ad', '_global.h5ad')

    print(f"Loading data...")
    adata = sc.read_h5ad(input_path)

    t_start = time.time()

    adata, cluster_markers = run_global_clustering(adata, args.panel)

    print("Annotating cell types...")
    if args.panel == 'T':
        adata = annotate_T_panel(adata)
    else:
        adata = annotate_S_panel(adata)

    print_composition(adata)

    adata.write(output_path)
    print(f"\nSaved: {output_path} ({adata.n_obs:,} cells)")
    print(f"Total time: {time.time()-t_start:.0f}s")
