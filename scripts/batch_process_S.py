#!/usr/bin/env python3
"""Batch process S-panel ROIs: segmentation + clustering + annotation."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import time
import argparse

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import segment_roi

sc.settings.verbosity = 1

# ---- Config ----
parser = argparse.ArgumentParser(description='Batch process S-panel IMC ROIs')
parser.add_argument('--start', type=int, default=0, help='Start ROI index')
parser.add_argument('--end', type=int, default=None, help='End ROI index')
parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
parser.add_argument('--gpu', action='store_true', help='Use GPU for Cellpose')

def parse_args():
    return parser.parse_args()

# Segmentation parameters (same as T-panel)
SEG_PARAMS = dict(method='hybrid', flow_threshold=0.8, min_distance=2, sigma=1.0)

# S-panel channel config
COFACTOR = 5
EXCLUDE_CHANNELS = ['80ArAr', '129Xe', '190BCKG', 'Pb204']
STRUCTURAL = ['DNA1', 'DNA2', 'HistoneH3', 'p_H3s28']


def cluster_adata(adata):
    """Run standard clustering pipeline."""
    bio = [m for m in adata.var_names if m not in EXCLUDE_CHANNELS]
    adata = adata[:, bio].copy()

    structural = [m for m in STRUCTURAL if m in adata.var_names]
    cluster_markers = [m for m in bio if m not in structural]

    adata.raw = adata

    # arcsinh transform
    adata.X = np.arcsinh(adata.X / COFACTOR)

    # Scale
    sc.pp.scale(adata, max_value=10)

    # PCA on clustering markers
    adata_cluster = adata[:, cluster_markers].copy()
    n_comps = min(20, len(cluster_markers) - 1, adata.n_obs - 1)
    sc.tl.pca(adata_cluster, svd_solver='arpack', n_comps=n_comps)
    adata.obsm['X_pca'] = adata_cluster.obsm['X_pca']

    # Neighbors + UMAP
    n_neighbors = min(15, adata.n_obs - 1)
    n_pcs = min(15, n_comps)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X_pca')
    sc.tl.umap(adata)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=0.5)

    return adata


def annotate_clusters(adata):
    """Rule-based cell type annotation for S-panel."""
    adata_raw = adata.raw.to_adata()

    def mean_marker(marker):
        if marker not in adata_raw.var_names:
            return pd.Series(0, index=adata.obs['leiden'].unique())
        idx = list(adata_raw.var_names).index(marker)
        return adata.obs.groupby('leiden').apply(
            lambda g: float(adata_raw[g.index, idx].X.mean())
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

    annotations = {}
    for c in adata.obs['leiden'].unique():
        # Follicular dendritic cells: CD21+, CXCL13+
        if cd21[c] > 2.0 and cxcl13[c] > 1.0:
            annotations[c] = 'FDC'
        # GC B cells: CD20+ BCL6+ Ki67+
        elif cd20[c] > 2.0 and bcl6[c] > 1.0:
            annotations[c] = 'GC B cells'
        # B cells: CD20+ or PAX5+
        elif cd20[c] > cd68[c] and cd20[c] > 1.0:
            annotations[c] = 'B cells'
        # M2 macrophages: CD68+ CD163+ or CD206+
        elif cd68[c] > 2.0 and (cd163[c] > 1.0 or cd206[c] > 1.0):
            annotations[c] = 'M2 Macrophages'
        # M1 macrophages: CD68+ CD11c+
        elif cd68[c] > 2.0 and cd11c[c] > 1.0:
            annotations[c] = 'M1 Macrophages'
        # Macrophages: CD68+ or CD14+
        elif cd68[c] > 2.0 or (cd14[c] > 2.0 and cd14[c] > cd20[c]):
            annotations[c] = 'Macrophages'
        # Dendritic cells: CD11c+ HLA-DR+ CD14-
        elif cd11c[c] > 1.5 and hla_dr[c] > 1.5 and cd14[c] < 1.0:
            annotations[c] = 'Dendritic cells'
        # Endothelial: CD31+ or CD34+
        elif cd31[c] > 2.0 or cd34[c] > 2.0:
            annotations[c] = 'Endothelial'
        # Stromal / Fibroblast: Vimentin+ or Fibronectin+ or PDPN+
        elif vimentin[c] > 2.0 or fibronectin[c] > 2.0 or pdpn[c] > 2.0:
            annotations[c] = 'Stromal / Fibroblast'
        # T cells
        elif cd4[c] > cd20[c] and cd4[c] > 0.5:
            annotations[c] = 'CD4 T cells'
        elif cd8[c] > cd20[c] and cd8[c] > 0.5:
            annotations[c] = 'CD8 T cells'
        # pDC: CD123+
        elif cd123[c] > 1.5:
            annotations[c] = 'pDC'
        else:
            annotations[c] = 'Other'

    adata.obs['cell_type'] = adata.obs['leiden'].map(annotations)
    return adata


# ---- Main ----
if __name__ == '__main__':
    args = parse_args()

    SEG_PARAMS['gpu'] = args.gpu

    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list_rois(DATA_DIR)
    if args.end:
        files = files[args.start:args.end]
    else:
        files = files[args.start:]

    print(f"=== Batch Processing S-panel ===")
    print(f"ROIs: {len(files)}")
    print(f"Output: {OUTPUT_DIR}\n")

    all_adatas = []
    results = []
    t_total = time.time()

    for i, filepath in enumerate(files):
        sample_id = extract_sample_id(filepath.name)
        # Use filename stem for non-FL files (Tonsil, Kidney)
        if sample_id == filepath.name:
            sample_id = filepath.stem
        print(f"[{i+1}/{len(files)}] {sample_id}...", end=' ', flush=True)

        t0 = time.time()

        try:
            image, markers, metadata = load_roi_txt(filepath)
            masks, adata = segment_roi(image, markers, sample_id, **SEG_PARAMS)
            n_cells = adata.n_obs

            if n_cells < 50:
                print(f"SKIP ({n_cells} cells, too few)")
                continue

            adata = cluster_adata(adata)
            adata = annotate_clusters(adata)
            adata.write(OUTPUT_DIR / f'{sample_id}.h5ad')
            all_adatas.append(adata)

            t_roi = time.time() - t0
            n_clusters = adata.obs['leiden'].nunique()

            results.append({
                'sample_id': sample_id,
                'n_cells': n_cells,
                'n_clusters': n_clusters,
                'time': t_roi,
                'shape': f"{image.shape[0]}x{image.shape[1]}",
            })
            print(f"{n_cells} cells, {n_clusters} clusters in {t_roi:.1f}s")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({'sample_id': sample_id, 'n_cells': 0, 'n_clusters': 0, 'time': 0, 'shape': ''})

    t_total = time.time() - t_total

    # Concatenate
    if all_adatas:
        print(f"\nConcatenating {len(all_adatas)} ROIs...")
        adata_combined = ad.concat(all_adatas, join='outer', label='sample_id',
                                    keys=[a.obs['sample_id'].iloc[0] for a in all_adatas])
        adata_combined.obs_names_make_unique()
        adata_combined.write(OUTPUT_DIR / 'TMA_B1_S_combined.h5ad')
        print(f"Saved: TMA_B1_S_combined.h5ad ({adata_combined.n_obs} cells)")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total time: {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"ROIs processed: {len([r for r in results if r['n_cells'] > 0])}/{len(files)}")
    print(f"Total cells: {sum(r['n_cells'] for r in results)}")

    df = pd.DataFrame(results)
    if len(df) > 0:
        print(f"\n{df.to_string(index=False)}")
        df.to_csv(OUTPUT_DIR / 'TMA_B1_S_summary.csv', index=False)

    if all_adatas:
        print(f"\n=== Cell Type Composition ===")
        comp = adata_combined.obs['cell_type'].value_counts()
        for ct, n in comp.items():
            pct = n / adata_combined.n_obs * 100
            print(f"  {ct}: {n} ({pct:.1f}%)")
