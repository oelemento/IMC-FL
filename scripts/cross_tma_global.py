#!/usr/bin/env python3
"""Cross-TMA global analysis: combine all TMAs for one panel and run unified clustering.

Supports step-based execution with checkpoints so each expensive step is saved to disk
and jobs can be split across SLURM submissions.

Steps:
  embed    - Load TMAs, concat, transform, PCA, neighbors, UMAP → save checkpoint
  leiden   - Load checkpoint, run Leiden at specified resolutions → save after each
  annotate - Load checkpoint with Leiden, annotate cell types → save final output

Usage:
    # Full pipeline in 3 steps:
    python scripts/cross_tma_global.py --panel T --step embed --checkpoint output/all_TMA_T_ckpt.h5ad
    python scripts/cross_tma_global.py --panel T --step leiden --checkpoint output/all_TMA_T_ckpt.h5ad --resolutions 0.3,0.5
    python scripts/cross_tma_global.py --panel T --step annotate --checkpoint output/all_TMA_T_ckpt.h5ad --output output/all_TMA_T_global.h5ad
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
import time
import argparse
import subprocess
import json
from datetime import datetime

sc.settings.verbosity = 2


def stamp_provenance(adata, args, step_override=None):
    """Stamp provenance into adata.uns before saving."""
    # Git commit hash
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        git_dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL
        ).decode().strip())
        git_hash += ' (dirty)' if git_dirty else ''
    except Exception:
        git_hash = 'unknown'

    prov = {
        'script': os.path.basename(__file__),
        'git_commit': git_hash,
        'timestamp': datetime.now().isoformat(),
        'panel': args.panel,
        'step': step_override or args.step,
        'checkpoint': str(args.checkpoint),
        'resolutions': args.resolutions,
        'output': str(args.output) if args.output else None,
        'command': ' '.join(sys.argv),
    }

    # Store current step as a flat dict (h5ad-safe: all string values)
    adata.uns['provenance'] = {k: str(v) for k, v in prov.items()}

    # Append to history as JSON string (h5ad can't store list-of-dicts)
    history = json.loads(adata.uns.get('provenance_history', '[]'))
    history.append(prov)
    adata.uns['provenance_history'] = json.dumps(history)

parser = argparse.ArgumentParser(description='Cross-TMA global clustering (step-based)')
parser.add_argument('--panel', required=True, choices=['T', 'S'], help='Panel type')
parser.add_argument('--step', required=True, choices=['embed', 'leiden', 'annotate'],
                    help='Which step to run')
parser.add_argument('--base-dir', default='.', help='Project base directory')
parser.add_argument('--checkpoint', required=True, help='Checkpoint h5ad path (written by embed, read by leiden/annotate)')
parser.add_argument('--output', default=None, help='Final output h5ad path (for annotate step)')
parser.add_argument('--resolutions', default='0.3,0.5',
                    help='Comma-separated Leiden resolutions (for leiden step)')


# Map of TMA -> combined h5ad path (relative to base-dir/output/)
# All files are *_raw_combined.h5ad with true raw ion counts in .X
TMA_FILES_T = {
    'A1': 'A1_T/A1_T_raw_combined.h5ad',
    'B1': 'batch/B1_T_raw_combined.h5ad',
    'C1': 'C1_T/C1_T_raw_combined.h5ad',
    'Biomax': 'Biomax_T/Biomax_T_raw_combined.h5ad',
}

TMA_FILES_S = {
    'A1': 'A1_S/A1_S_raw_combined.h5ad',
    'B1': 'batch_S/B1_S_raw_combined.h5ad',
    'C1': 'C1_S/C1_S_raw_combined.h5ad',
    'Biomax': 'Biomax_S/Biomax_S_raw_combined.h5ad',
}


def load_and_concat(base_dir, panel):
    """Load all TMA combined files and concatenate with TMA labels."""
    tma_files = TMA_FILES_T if panel == 'T' else TMA_FILES_S
    output_dir = Path(base_dir) / 'output'

    adatas = []
    tma_labels = []
    for tma_name, rel_path in tma_files.items():
        fpath = output_dir / rel_path
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping {tma_name}")
            continue
        print(f"Loading {tma_name}: {fpath}")
        adata = sc.read_h5ad(fpath)

        # If .X is already transformed but .raw has true raw counts, use .raw
        if adata.X.min() < 0 and adata.raw is not None and adata.raw.X.min() >= 0:
            print(f"  {tma_name}: .X already transformed, extracting raw counts from .raw")
            raw_adata = adata.raw.to_adata()
            raw_adata.obs = adata.obs.copy()
            adata = raw_adata

        adata.obs['tma'] = tma_name

        # Ensure sample_id includes TMA prefix to avoid collisions
        if 'sample_id' in adata.obs.columns:
            adata.obs['sample_id_original'] = adata.obs['sample_id'].copy()
            adata.obs['sample_id'] = tma_name + '_' + adata.obs['sample_id'].astype(str)

        print(f"  {tma_name}: {adata.n_obs:,} cells, {adata.n_vars} markers, X range [{float(adata.X.min()):.1f}, {float(adata.X.max()):.1f}]")
        adatas.append(adata)
        tma_labels.append(tma_name)

    if not adatas:
        print("ERROR: No files found!")
        sys.exit(1)

    print(f"\nConcatenating {len(adatas)} TMAs...")
    # Use join='inner' to keep only shared markers across TMAs
    adata = ad.concat(adatas, join='inner', label='tma_source', keys=tma_labels)
    adata.obs_names_make_unique()

    print(f"Combined: {adata.n_obs:,} cells x {adata.n_vars} markers")
    print(f"\nCells per TMA:")
    for tma in tma_labels:
        n = (adata.obs['tma'] == tma).sum()
        print(f"  {tma}: {n:,}")

    return adata


def run_embedding(adata, panel):
    """Run transform + PCA + neighbors + UMAP. Returns adata with embeddings."""
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

    print(f"\nFiltered: {adata.n_obs:,} cells x {len(bio)} markers")
    print(f"Clustering markers: {len(cluster_markers)}")
    print(f"  {cluster_markers}")

    # Store raw
    adata.raw = adata

    # arcsinh transform
    print("\nArcsinh transform (cofactor=5)...")
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
    print(f"PCA: {n_comps} components, top-10 variance: {var_explained:.1f}%")

    # Store cluster markers list for reference
    adata.uns['cluster_markers'] = cluster_markers

    # Neighbors
    print("Building neighbor graph...")
    t0 = time.time()
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(15, n_comps), use_rep='X_pca')
    print(f"Neighbors done in {time.time()-t0:.0f}s")

    # UMAP
    print("Computing UMAP...")
    t0 = time.time()
    sc.tl.umap(adata)
    print(f"UMAP done in {time.time()-t0:.0f}s")

    return adata


def run_leiden(adata, resolutions):
    """Run Leiden clustering at specified resolutions using igraph backend.
    Saves checkpoint after each resolution."""
    for res in resolutions:
        print(f"Leiden resolution={res} (igraph)...")
        t0 = time.time()
        sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}',
                     flavor='igraph', n_iterations=2, directed=False)
        n_clust = adata.obs[f'leiden_{res}'].nunique()
        elapsed = time.time() - t0
        print(f"  -> {n_clust} clusters in {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    # Set default leiden column to the lowest resolution >= 0.5, or the highest available
    available = sorted(resolutions)
    default_res = None
    for r in available:
        if r >= 0.5:
            default_res = r
            break
    if default_res is None:
        default_res = available[-1]
    adata.obs['leiden'] = adata.obs[f'leiden_{default_res}']
    print(f"Default leiden column set to resolution {default_res}")

    return adata


def annotate_T_panel(adata):
    """Per-cell rule-based annotation for T-panel (v8).

    Uses raw marker values per cell (not cluster means) for gating.
    Cluster-level annotation fails for T-panel because B cells dominate every
    cluster, making cluster-mean CD20 > CD4/CD8 everywhere and preventing T cell
    identification. Per-cell gating correctly identifies T cells (CD20 ≈ 0 on
    individual T cells).

    HARMONIZED THRESHOLDS (same in both panels):
      CD20 ≥ 1.5  = confident B cell
      CD4/CD8 > 0.5 and > CD20 = T cell
      CD68 > 2.0  = myeloid

    MARKER QC (v8 update):
      Dead (no signal):  LAG3 (p99=0.29), T-bet (p99=0.50), CTLA4 (p99=0.59),
                         p53 (p99=0.54), pSTAT3 (p99=0.60)
      Marginal:          PD-L1 (p99=0.83), ICOS (p99=0.80), GATA3 (p99=0.85)
      Diffuse:           TIM3 (mean=0.52, 54% of CD8 T 'positive' — no bimodal)
      Working:           TOX (p99=2.79), PD-1 (p99=1.41), CD20, CD3, CD4, CD8a,
                         CD68, FoxP3, GranzymeB, CXCR5, CD38, IRF4, CD45RO

    Exhaustion: TOX + PD-1 only (LAG3/CTLA4 dead, TIM3 diffuse/unreliable).
    """
    adata_raw = adata.raw.to_adata()
    raw_X = adata_raw.X
    if hasattr(raw_X, 'toarray'):
        raw_X = raw_X.toarray()
    raw_X = np.asarray(raw_X, dtype=np.float32)

    var_names = list(adata_raw.var_names)

    def get(name):
        if name in var_names:
            return raw_X[:, var_names.index(name)]
        return np.zeros(adata.n_obs, dtype=np.float32)

    # Extract per-cell marker values
    cd3 = get('CD3')
    cd4 = get('CD4')
    cd8 = get('CD8a')
    cd20 = get('CD20')
    foxp3 = get('FoxP3')
    cd68 = get('CD68')
    gzmb = get('GranzymeB')
    cd38 = get('CD38')
    irf4 = get('IRF4')
    tox = get('TOX')
    cxcr5 = get('CXCR5')
    tim3 = get('TIM3')
    pd1 = get('PD_1')

    # --- Print cluster-level profiles for QC (diagnostic only) ---
    print("\n=== Cluster marker profiles (means, for QC only — annotation is per-cell) ===")
    clusters = sorted(adata.obs['leiden'].unique(), key=lambda x: int(x))
    header = f" {'Clust':>5} {'n':>8}  {'CD20':>5} {'CD3':>5} {'CD4':>5} {'CD8a':>5} {'CD68':>5} {'FoxP3':>5} {'TOX':>5} {'GzmB':>5} {'TIM3':>5} {'PD1':>5} {'CXCR5':>5} {'CD38':>5} {'IRF4':>5}"
    print(header)
    leiden = adata.obs['leiden'].values
    for c in clusters:
        mask = leiden == c
        n = mask.sum()
        print(f" {c:>5} {n:>8}  {cd20[mask].mean():>5.1f} {cd3[mask].mean():>5.1f} {cd4[mask].mean():>5.1f} {cd8[mask].mean():>5.1f} {cd68[mask].mean():>5.1f} {foxp3[mask].mean():>5.1f} {tox[mask].mean():>5.1f} {gzmb[mask].mean():>5.1f} {tim3[mask].mean():>5.1f} {pd1[mask].mean():>5.1f} {cxcr5[mask].mean():>5.1f} {cd38[mask].mean():>5.1f} {irf4[mask].mean():>5.1f}")

    # --- Per-cell annotation using vectorized boolean masks ---
    n_cells = adata.n_obs
    annotations = np.full(n_cells, '', dtype=object)
    assigned = np.zeros(n_cells, dtype=bool)

    def assign(mask, label):
        nonlocal assigned
        m = mask & ~assigned
        annotations[m] = label
        assigned |= m
        count = m.sum()
        return count

    # Priority 1: T cells (minority in FL, must check before B cells)
    # At per-cell level, real T cells have CD4/CD8 > CD20 because their own
    # CD20 is near zero. This check fails at cluster level because B cell
    # majority inflates the cluster-mean CD20.
    foxp3_thresh = float(np.median(foxp3[foxp3 > 0]) * 1.5) if (foxp3 > 0).any() else 0.3
    foxp3_thresh = max(foxp3_thresh, 0.3)

    assign((cd3 > 0.5) & (cd8 > cd4) & (cd8 > cd20) & (tox > 0.8) & (pd1 > 0.5), 'CD8 T exhausted')
    assign((cd3 > 0.5) & (cd8 > cd4) & (cd8 > cd20) & (tox > 0.8), 'CD8 T pre-exhausted (TOX+)')
    assign((cd3 > 0.5) & (cd8 > cd4) & (cd8 > cd20), 'CD8 T cells')
    assign((cd3 > 0.5) & (cd4 > cd8) & (cd4 > cd20) & (foxp3 > foxp3_thresh), 'Treg')
    assign((cd3 > 0.5) & (cd4 > cd8) & (cd4 > cd20), 'CD4 T cells')
    assign((cd3 > 0.5) & ((cd4 > cd20) | (cd8 > cd20)), 'T cells')

    # Priority 2: B cells (CD20 ≥ 1.5, harmonized with S-panel)
    b_base = (cd20 > cd3) & (cd20 > cd68) & (cd20 >= 1.5)
    assign(b_base & (cxcr5 > 2.0) & (cd20 > 8.0), 'GC B cells')
    assign(b_base & (cd38 > 1.0) & (irf4 > 0.5), 'Activated B / Plasmablast')
    assign(b_base & (cd20 > 6.0), 'B cells (CD20hi)')
    assign(b_base & (cxcr5 > 1.5), 'B cells (CXCR5hi)')
    assign(b_base & (tox > 1.5), 'B cells (TOXhi)')
    assign(b_base, 'B cells')

    # Priority 3: Myeloid
    assign((cd68 > 2.0) & (gzmb > 2.0), 'Macrophages (GzmB+)')
    assign((cd68 > 2.0) & (cd68 > cd3), 'Macrophages')
    # TIM3-high gate removed in v8: TIM3 is diffuse (mean=0.52, 54% of CD8 T
    # 'positive' at 0.5) with no bimodal separation — not a usable gate.

    # Priority 4: Weak CD20 B cells (0.5–1.5, requires CD20 clearly dominant)
    assign((cd20 >= 0.5) & (cd20 > 2 * cd3) & (cd20 > 2 * cd68), 'B cells (weak CD20)')

    # Priority 5: Mixed / Border
    assign((cd20 > 1.0) & (cd3 > 0.5), 'Mixed / Border cells')

    # Priority 6: Low quality
    assign((cd20 < 1.5) & (cd3 < 0.8) & (cd68 < 1.5), 'Low quality / Unassigned')

    # Remainder
    annotations[~assigned] = 'Other'

    adata.obs['cell_type'] = annotations

    # Print per-cell composition within each cluster for QC
    print("\n=== Per-cell annotation within clusters ===")
    for c in clusters:
        mask = leiden == c
        n = mask.sum()
        ct_counts = pd.Series(annotations[mask]).value_counts()
        top = ', '.join(f"{ct}: {cnt/n*100:.0f}%" for ct, cnt in ct_counts.head(3).items())
        print(f"  Cluster {c} ({n:,}): {top}")

    return adata


def annotate_S_panel(adata):
    """Per-cell rule-based annotation for S-panel (v8).

    Uses raw marker values per cell (not cluster means) for gating.
    Cluster-level annotation missed T cells because B cells dominate every
    cluster. Per-cell gating correctly identifies all cell types.

    HARMONIZED THRESHOLDS (same in both panels):
      CD20 ≥ 1.5  = confident B cell
      CD4/CD8 > 0.5 and > CD20 = T cell
      CD68 > 2.0  = myeloid

    MARKER QC (v8 update):
      Dead (no signal):  BCL6 (p99=0.50), PD-L1 (p99=0.49)
      Marginal:          IDO (p99=0.89), CXCL13 (p99=0.82, C1 only ~1.15),
                         VISTA (p99=0.90), CD123 (p99=0.98, Biomax/C1 only)
      Diffuse:           CD11b (mean=0.43, p99=1.30), CXCL12, CD1a
      Working:           CD20, PAX5, BCL2, CD21, CD68, CD163, CD206, CD14,
                         CD11c, Vimentin, PDPN, CD31, CD34, HLA-DR, S100A9,
                         CD4, CD8a, CD44, Ki-67, Fibronectin, SOX9, CCL21

    BCL6 gates removed (GC B cannot be identified in S-panel).
    CD123/pDC gate flagged as unreliable (only works in Biomax/C1).
    """
    adata_raw = adata.raw.to_adata()
    raw_X = adata_raw.X
    if hasattr(raw_X, 'toarray'):
        raw_X = raw_X.toarray()
    raw_X = np.asarray(raw_X, dtype=np.float32)

    var_names = list(adata_raw.var_names)

    def get(name):
        if name in var_names:
            return raw_X[:, var_names.index(name)]
        return np.zeros(adata.n_obs, dtype=np.float32)

    # Extract per-cell marker values
    cd20 = get('CD20')
    cd4 = get('CD4')
    cd8 = get('CD8a')
    cd68 = get('CD68')
    cd14 = get('CD14')
    cd163 = get('CD163')
    cd206 = get('CD206')
    cd11c = get('CD11c')
    cd31 = get('CD31')
    cd34 = get('CD34')
    vimentin = get('Vimentin')
    hla_dr = get('HLA_DR')
    bcl6 = get('BCL_6')
    bcl2 = get('BCL_2')
    pax5 = get('PAX5')
    cd21 = get('CD21')
    ki67 = get('Ki-67')
    pdpn = get('PDPN')
    fibronectin = get('Fibronectin')
    cd123 = get('CD123')
    cxcl13 = get('CXCL13')
    s100a9 = get('S100A9')
    cd44 = get('CD44')

    # --- Print cluster-level profiles for QC (diagnostic only) ---
    print("\n=== Cluster marker profiles (means, for QC only — annotation is per-cell) ===")
    clusters = sorted(adata.obs['leiden'].unique(), key=lambda x: int(x))
    header = f" {'Clust':>5} {'n':>8}  {'CD20':>5} {'PAX5':>5} {'BCL6':>5} {'BCL2':>5} {'CD21':>5} {'CD68':>5} {'CD163':>5} {'CD206':>5} {'CD14':>5} {'CD11c':>5} {'CD31':>5} {'Vim':>5} {'PDPN':>5} {'FN':>5} {'CD4':>5} {'CD8a':>5} {'HLADR':>5} {'Ki67':>5}"
    print(header)
    leiden = adata.obs['leiden'].values
    for c in clusters:
        mask = leiden == c
        n = mask.sum()
        print(f" {c:>5} {n:>8}  {cd20[mask].mean():>5.1f} {pax5[mask].mean():>5.1f} {bcl6[mask].mean():>5.1f} {bcl2[mask].mean():>5.1f} {cd21[mask].mean():>5.1f} {cd68[mask].mean():>5.1f} {cd163[mask].mean():>5.1f} {cd206[mask].mean():>5.1f} {cd14[mask].mean():>5.1f} {cd11c[mask].mean():>5.1f} {cd31[mask].mean():>5.1f} {vimentin[mask].mean():>5.1f} {pdpn[mask].mean():>5.1f} {fibronectin[mask].mean():>5.1f} {cd4[mask].mean():>5.1f} {cd8[mask].mean():>5.1f} {hla_dr[mask].mean():>5.1f} {ki67[mask].mean():>5.1f}")

    # --- Per-cell annotation using vectorized boolean masks ---
    n_cells = adata.n_obs
    annotations = np.full(n_cells, '', dtype=object)
    assigned = np.zeros(n_cells, dtype=bool)

    def assign(mask, label):
        nonlocal assigned
        m = mask & ~assigned
        annotations[m] = label
        assigned |= m
        return m.sum()

    # Priority 1: FDC (CD21-high, hallmark of follicular dendritic cells)
    assign((cd21 > 5.0), 'FDC')
    assign((cd21 > 2.0) & (cxcl13 > 0.3) & (cd20 < cd21), 'FDC')

    # Priority 2: FRC (PDPN+, fibroblastic reticular cells)
    assign((pdpn > 1.5) & (cd20 < 2.0) & (cd68 < 2.0), 'FRC (PDPN+)')

    # Priority 3: Stromal / CAF (high vimentin, no strong lineage markers)
    strom_base = (vimentin > 3.0) & (cd20 < 1.5) & (cd68 < 2.0) & (cd4 < 1.0) & (cd8 < 1.0)
    assign(strom_base & ((cd31 > 1.5) | (cd34 > 1.5)), 'Endothelial')
    assign(strom_base, 'Stromal / CAF')

    # Priority 4: T cells (minority, check before B cells)
    assign((cd4 > cd20) & (cd4 > cd68) & (cd4 > 0.5), 'CD4 T cells')
    assign((cd8 > cd20) & (cd8 > cd68) & (cd8 > 0.5), 'CD8 T cells')

    # GC B gates removed in v8: BCL6 antibody is dead in this IMC panel
    # (p99=0.50, only 0.0% of cells > 1.0). Cannot identify GC B cells in
    # S-panel. Use T-panel CXCR5/CD20 gate as sole GC B estimate.

    # Priority 5: B cells (strong CD20 ≥ 1.5)
    b_base = (cd20 > 1.5) & (cd20 > cd68)
    assign(b_base & (bcl2 > 2.0), 'B cells (BCL2+)')
    assign(b_base & (pax5 > 1.0), 'B cells (PAX5+)')
    assign(b_base, 'B cells')

    # Priority 7: B cells (weak CD20, definitive B-lineage markers)
    assign((cd20 > 0.5) & (cd20 > cd68) & (bcl2 > 2.0), 'B cells (BCL2+)')
    assign((cd20 > 0.5) & (cd20 > cd68) & (pax5 > 1.5), 'B cells (PAX5+)')

    # Priority 8: Myeloid
    assign(s100a9 > 5.0, 'Myeloid (S100A9+)')
    assign((cd68 > 2.0) & ((cd163 > 1.5) | (cd206 > 1.5)), 'M2 Macrophages')
    assign((cd68 > 2.0) & ((cd11c > 1.0) | (hla_dr > 2.0)), 'M1 Macrophages')
    assign((cd68 > 2.0) | ((cd14 > 2.0) & (cd14 > cd20)), 'Macrophages')
    assign((cd11c > 1.5) & (hla_dr > 1.5) & (cd14 < 1.0), 'Dendritic cells')
    # pDC gate: CD123 is marginal (p99=0.98 globally; only works in Biomax
    # p99=1.69 and C1 p99=1.11). Results unreliable in A1/B1.
    assign(cd123 > 1.5, 'pDC')

    # Priority 9: Endothelial / Stromal
    assign((cd31 > 2.0) | (cd34 > 1.2), 'Endothelial')
    assign(fibronectin > 2.0, 'Stromal / CAF')

    # Priority 10: CD44-high
    assign(cd44 > 5.0, 'Histiocytes (CD44hi)')

    # Priority 11: Mixed / Border
    assign((cd20 > 1.0) & (cd68 > 1.0), 'Mixed / Border cells')

    # Priority 12: Low quality
    assign((cd20 < 1.0) & (cd68 < 1.0) & (cd4 < 0.5) & (vimentin < 1.5) & (pax5 < 1.0), 'Low quality / Unassigned')

    # Remainder
    annotations[~assigned] = 'Other'

    adata.obs['cell_type'] = annotations

    # Print per-cell composition within each cluster for QC
    print("\n=== Per-cell annotation within clusters ===")
    for c in clusters:
        mask = leiden == c
        n = mask.sum()
        ct_counts = pd.Series(annotations[mask]).value_counts()
        top = ', '.join(f"{ct}: {cnt/n*100:.0f}%" for ct, cnt in ct_counts.head(3).items())
        print(f"  Cluster {c} ({n:,}): {top}")

    return adata


def print_composition(adata):
    """Print global cell type composition, overall and per TMA."""
    print("\n=== Global Cell Type Composition ===")
    comp = adata.obs['cell_type'].value_counts()
    for ct, n in comp.items():
        pct = n / adata.n_obs * 100
        print(f"  {ct}: {n:,} ({pct:.1f}%)")

    print("\n=== Cell Type Composition by TMA ===")
    ct_by_tma = pd.crosstab(adata.obs['tma'], adata.obs['cell_type'], normalize='index') * 100
    print(ct_by_tma.round(1).to_string())


if __name__ == '__main__':
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    checkpoint_path = Path(args.checkpoint)
    resolutions = [float(r) for r in args.resolutions.split(',')]

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = base_dir / 'output' / f'all_TMA_{args.panel}_global.h5ad'

    t_start = time.time()

    if args.step == 'embed':
        print("=== STEP: embed (load + transform + PCA + neighbors + UMAP) ===")
        adata = load_and_concat(base_dir, args.panel)
        adata = run_embedding(adata, args.panel)

        print(f"\nSaving checkpoint to {checkpoint_path}...")
        stamp_provenance(adata, args)
        adata.write(checkpoint_path)
        print(f"Saved: {checkpoint_path} ({adata.n_obs:,} cells)")

    elif args.step == 'leiden':
        print(f"=== STEP: leiden (resolutions: {resolutions}) ===")
        print(f"Loading checkpoint: {checkpoint_path}")
        adata = sc.read_h5ad(checkpoint_path)
        print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars} markers")

        for res in resolutions:
            print(f"\nLeiden resolution={res} (igraph)...")
            t0 = time.time()
            sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}',
                         flavor='igraph', n_iterations=2, directed=False)
            n_clust = adata.obs[f'leiden_{res}'].nunique()
            elapsed = time.time() - t0
            print(f"  -> {n_clust} clusters in {elapsed:.0f}s ({elapsed/3600:.1f}h)")

            # Save after each resolution
            print(f"  Saving checkpoint...")
            stamp_provenance(adata, args, step_override=f'leiden_{res}')
            adata.write(checkpoint_path)
            print(f"  Checkpoint saved with leiden_{res}")

        # Set default leiden column
        available = sorted(resolutions)
        default_res = None
        for r in available:
            if r >= 0.5:
                default_res = r
                break
        if default_res is None:
            default_res = available[-1]
        adata.obs['leiden'] = adata.obs[f'leiden_{default_res}']
        print(f"\nDefault leiden column set to resolution {default_res}")

        # Run annotation and save final output
        print("\nAnnotating cell types...")
        if args.panel == 'T':
            adata = annotate_T_panel(adata)
        else:
            adata = annotate_S_panel(adata)

        print_composition(adata)

        print(f"\nSaving final output to {output_path}...")
        stamp_provenance(adata, args)
        adata.write(output_path)
        print(f"Saved: {output_path} ({adata.n_obs:,} cells)")

    elif args.step == 'annotate':
        print("=== STEP: annotate ===")
        print(f"Loading checkpoint: {checkpoint_path}")
        adata = sc.read_h5ad(checkpoint_path)
        print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars} markers")

        # Check for leiden columns
        leiden_cols = [c for c in adata.obs.columns if c.startswith('leiden_')]
        print(f"Found Leiden columns: {leiden_cols}")

        # Use resolution from --resolutions to pick the right leiden column
        target_res = resolutions[0] if resolutions else None
        target_col = f'leiden_{target_res}' if target_res else None

        if target_col and target_col in adata.obs.columns:
            adata.obs['leiden'] = adata.obs[target_col]
            print(f"Using leiden column: {target_col} ({adata.obs['leiden'].nunique()} clusters)")
        elif 'leiden' not in adata.obs.columns:
            if 'leiden_0.5' in adata.obs.columns:
                adata.obs['leiden'] = adata.obs['leiden_0.5']
            elif leiden_cols:
                adata.obs['leiden'] = adata.obs[leiden_cols[0]]
            else:
                print("ERROR: No leiden columns found in checkpoint!")
                sys.exit(1)

        print("\nAnnotating cell types...")
        if args.panel == 'T':
            adata = annotate_T_panel(adata)
        else:
            adata = annotate_S_panel(adata)

        print_composition(adata)

        print(f"\nSaving to {output_path}...")
        stamp_provenance(adata, args)
        adata.write(output_path)
        print(f"Saved: {output_path} ({adata.n_obs:,} cells)")

    print(f"\nStep '{args.step}' completed in {time.time()-t_start:.0f}s ({(time.time()-t_start)/3600:.1f}h)")
