#!/usr/bin/env python3
"""Retry failed ROIs from batch processing, one at a time."""

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

from src.data_loader import load_roi_txt, list_rois, extract_sample_id
from src.segmentation import segment_roi

sc.settings.verbosity = 1

# These ROIs failed in the batch run
FAILED_ROIS = [
    'FL01', 'FL06', 'FL08', 'FL12', 'FL16', 'FL20',
    'FL23', 'FL27', 'FL30', 'FL31', 'FL33', 'FL34',
    'FL36', 'FL37', 'FL38', 'FL41', 'FL47', 'Tonsil',
]

COFACTOR = 5
EXCLUDE_CHANNELS = ['80ArAr', '129Xe', '190BCKG', '197Au', 'Pb204']
STRUCTURAL = ['DNA1', 'DNA2', 'HistoneH3', 'p_H3s28', 'H3K27me3']

parser = argparse.ArgumentParser()
parser.add_argument('--task-id', type=int, required=True)
parser.add_argument('--data-dir', required=True)
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()

if args.task_id >= len(FAILED_ROIS):
    print(f"Task {args.task_id} out of range, nothing to do.")
    sys.exit(0)

roi_name = FAILED_ROIS[args.task_id]
data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Find the file for this ROI
files = list_rois(data_dir)
matching = [f for f in files if roi_name.lower() in extract_sample_id(f.name).lower()
            or roi_name.lower() in f.name.lower()]

if not matching:
    print(f"No file found for {roi_name}")
    sys.exit(1)

filepath = matching[0]
sample_id = extract_sample_id(filepath.name)
# Use filename as sample_id for non-FL files (like Tonsil/Kidney)
if sample_id == filepath.name:
    sample_id = filepath.stem

print(f"=== Retrying {roi_name} ({filepath.name}) ===")

t0 = time.time()
image, markers, metadata = load_roi_txt(filepath)
print(f"Image: {image.shape}")

masks, adata = segment_roi(
    image, markers, sample_id,
    method='hybrid', flow_threshold=0.8, min_distance=2, sigma=1.0
)
print(f"Segmented: {adata.n_obs} cells in {time.time()-t0:.0f}s")

if adata.n_obs < 50:
    print(f"Too few cells ({adata.n_obs}), skipping clustering.")
    sys.exit(0)

# Cluster
def cluster_adata(adata):
    bio = [m for m in adata.var_names if m not in EXCLUDE_CHANNELS]
    adata = adata[:, bio].copy()
    structural = [m for m in STRUCTURAL if m in adata.var_names]
    cluster_markers = [m for m in bio if m not in structural]
    adata.raw = adata
    adata.X = np.arcsinh(adata.X / COFACTOR)
    sc.pp.scale(adata, max_value=10)
    adata_cluster = adata[:, cluster_markers].copy()
    n_comps = min(20, len(cluster_markers) - 1, adata.n_obs - 1)
    sc.tl.pca(adata_cluster, svd_solver='arpack', n_comps=n_comps)
    adata.obsm['X_pca'] = adata_cluster.obsm['X_pca']
    n_neighbors = min(15, adata.n_obs - 1)
    n_pcs = min(15, n_comps)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X_pca')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    return adata

def annotate_clusters(adata):
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

adata = cluster_adata(adata)
adata = annotate_clusters(adata)

# Save
outfile = output_dir / f'{sample_id}.h5ad'
adata.write(outfile)
print(f"Saved: {outfile} ({adata.n_obs} cells, {adata.obs['leiden'].nunique()} clusters)")
