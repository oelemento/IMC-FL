#!/usr/bin/env python3
"""Batch process all ROIs: segmentation + clustering + annotation."""

import sys
import os

# Add project root to path (works on both local and cluster)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import time
import argparse

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import segment_roi, prepare_nuclear_image

sc.settings.verbosity = 1

# ---- Config ----
parser = argparse.ArgumentParser(description='Batch process IMC ROIs')
parser.add_argument('--method', default='hybrid', choices=['hybrid', 'local_maxima'],
                    help='Segmentation method (default: hybrid)')
parser.add_argument('--panel', default='T', choices=['T', 'S'],
                    help='Panel to process (default: T)')
parser.add_argument('--start', type=int, default=0, help='Start ROI index')
parser.add_argument('--end', type=int, default=None, help='End ROI index')
parser.add_argument('--data-dir', type=str, default=None,
                    help='Data directory (default: auto-detect local or cluster)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Output directory (default: PROJECT_ROOT/output/batch)')
parser.add_argument('--gpu', action='store_true', help='Use GPU for Cellpose')
args = parser.parse_args()

# Auto-detect paths
if args.data_dir:
    DATA_DIR = Path(args.data_dir)
else:
    DATA_DIR = Path(PROJECT_ROOT) / 'data' / 'raw' / f'TMA_B1_{args.panel}'

if args.output_dir:
    OUTPUT_DIR = Path(args.output_dir)
else:
    OUTPUT_DIR = Path(PROJECT_ROOT) / 'output' / 'batch'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Segmentation parameters
SEG_PARAMS = {
    'local_maxima': dict(method='local_maxima', sigma=1.0, min_distance=2, expansion='voronoi'),
    'hybrid': dict(method='hybrid', flow_threshold=0.8, min_distance=2, sigma=1.0, gpu=args.gpu),
}

# Clustering parameters
COFACTOR = 5
EXCLUDE_CHANNELS = ['80ArAr', '129Xe', '190BCKG', '197Au', 'Pb204']
STRUCTURAL = ['DNA1', 'DNA2', 'HistoneH3', 'p_H3s28', 'H3K27me3']


def cluster_adata(adata):
    """Run standard clustering pipeline on a single ROI's AnnData."""
    # Exclude non-biological
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
    """Rule-based cell type annotation for T-panel."""
    adata_raw = adata.raw.to_adata()

    def mean_marker(marker):
        if marker not in adata_raw.var_names:
            return pd.Series(0, index=adata.obs['leiden'].unique())
        idx = list(adata_raw.var_names).index(marker)
        return adata.obs.groupby('leiden').apply(
            lambda g: float(adata_raw[g.index, idx].X.mean())
        )

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

    annotations = {}
    for c in adata.obs['leiden'].unique():
        if cd20[c] > cd3[c] and cd20[c] > cd68[c] and cd20[c] > 1.0:
            annotations[c] = 'B cells'
        elif cd68[c] > 2.0 and gzmb[c] > 2.0:
            annotations[c] = 'Cytotoxic'
        elif cd68[c] > 2.0 and cd68[c] > cd3[c]:
            annotations[c] = 'Macrophages'
        elif cd38[c] > 1.0 and irf4[c] > 0.5:
            annotations[c] = 'Plasma cells'
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
        elif tim3[c] > 3.0 or cxcr5[c] > 3.0:
            annotations[c] = 'Tfh-like'
        else:
            annotations[c] = 'Other'

    adata.obs['cell_type'] = adata.obs['leiden'].map(annotations)
    return adata


# ---- Main ----
files = list_rois(DATA_DIR)
if args.end:
    files = files[args.start:args.end]
else:
    files = files[args.start:]

print(f"=== Batch Processing TMA_B1_{args.panel} ===")
print(f"Method: {args.method}")
print(f"ROIs: {len(files)}")
print(f"Output: {OUTPUT_DIR}\n")

all_adatas = []
results = []
t_total = time.time()

for i, filepath in enumerate(files):
    sample_id = extract_sample_id(filepath.name)
    print(f"[{i+1}/{len(files)}] {sample_id}...", end=' ', flush=True)

    t0 = time.time()

    try:
        # Load
        image, markers, metadata = load_roi_txt(filepath)

        # Segment
        masks, adata = segment_roi(image, markers, sample_id, **SEG_PARAMS[args.method])
        n_cells = adata.n_obs

        if n_cells < 50:
            print(f"SKIP ({n_cells} cells, too few)")
            continue

        # Cluster
        adata = cluster_adata(adata)

        # Annotate
        adata = annotate_clusters(adata)

        # Save individual
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

# Concatenate all ROIs
if all_adatas:
    print(f"\nConcatenating {len(all_adatas)} ROIs...")
    adata_combined = ad.concat(all_adatas, join='outer', label='sample_id',
                                keys=[a.obs['sample_id'].iloc[0] for a in all_adatas])
    adata_combined.obs_names_make_unique()
    adata_combined.write(OUTPUT_DIR / f'TMA_B1_{args.panel}_combined.h5ad')
    print(f"Saved: TMA_B1_{args.panel}_combined.h5ad ({adata_combined.n_obs} cells)")

# Summary table
print(f"\n=== Summary ===")
print(f"Total time: {t_total:.0f}s ({t_total/60:.1f} min)")
print(f"ROIs processed: {len([r for r in results if r['n_cells'] > 0])}/{len(files)}")
print(f"Total cells: {sum(r['n_cells'] for r in results)}")

df = pd.DataFrame(results)
if len(df) > 0:
    print(f"\n{df.to_string(index=False)}")
    df.to_csv(OUTPUT_DIR / f'TMA_B1_{args.panel}_summary.csv', index=False)
    print(f"\nSaved: TMA_B1_{args.panel}_summary.csv")

# Composition across all ROIs
if all_adatas:
    print(f"\n=== Cell Type Composition (all ROIs) ===")
    comp = adata_combined.obs['cell_type'].value_counts()
    for ct, n in comp.items():
        pct = n / adata_combined.n_obs * 100
        print(f"  {ct}: {n} ({pct:.1f}%)")
