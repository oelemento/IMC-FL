#!/usr/bin/env python3
"""Quick QC of cross-TMA global analysis results.
Prints text summary - no plots, designed to run on cluster and read via SSH.
"""
import argparse
import anndata as ad
import numpy as np
import pandas as pd


def qc_file(path: str) -> None:
    print(f"\n{'='*70}")
    print(f"FILE: {path}")
    print(f"{'='*70}")

    adata = ad.read_h5ad(path)
    print(f"Shape: {adata.shape[0]:,} cells × {adata.shape[1]} features")

    # .X range
    xmin, xmax = float(adata.X.min()), float(adata.X.max())
    print(f".X range: [{xmin:.3f}, {xmax:.3f}]")

    # .raw range
    if adata.raw is not None:
        raw_min = float(adata.raw.X.min())
        raw_max = float(adata.raw.X.max())
        print(f".raw range: [{raw_min:.3f}, {raw_max:.3f}]")
        if raw_min < 0:
            print("  *** WARNING: .raw has negative values - may contain transformed data ***")
    else:
        print(".raw: None")

    # obs columns
    leiden_cols = [c for c in adata.obs.columns if c.startswith("leiden")]
    print(f"\nLeiden columns: {leiden_cols}")

    # Find TMA column (could be 'TMA' or 'tma')
    tma_col = "TMA" if "TMA" in adata.obs.columns else ("tma" if "tma" in adata.obs.columns else None)

    # TMA composition
    if tma_col:
        print(f"\nCells per TMA:")
        tma_counts = adata.obs[tma_col].value_counts()
        for tma, n in tma_counts.items():
            print(f"  {tma}: {n:,} ({100*n/adata.shape[0]:.1f}%)")

    # Cell type annotation
    if "cell_type" in adata.obs.columns:
        print(f"\nCell type composition (overall):")
        ct_counts = adata.obs["cell_type"].value_counts()
        for ct, n in ct_counts.items():
            print(f"  {ct}: {n:,} ({100*n/adata.shape[0]:.1f}%)")

        # Cell type × TMA crosstab
        if tma_col:
            print(f"\nCell type × TMA (%):")
            ct = pd.crosstab(adata.obs["cell_type"], adata.obs[tma_col], normalize="columns") * 100
            # Sort by overall frequency
            ct = ct.loc[ct_counts.index]
            print(ct.round(1).to_string())

    # Per-resolution cluster counts
    for col in leiden_cols:
        n_clust = adata.obs[col].nunique()
        print(f"\n{col}: {n_clust} clusters")
        vc = adata.obs[col].value_counts().head(10)
        for cl, n in vc.items():
            print(f"  cluster {cl}: {n:,} ({100*n/adata.shape[0]:.1f}%)")
        if n_clust > 10:
            print(f"  ... ({n_clust - 10} more clusters)")

    # Check .raw per TMA for negative values (the B1 issue)
    if adata.raw is not None and tma_col:
        print(f"\n.raw min per TMA:")
        for tma in sorted(adata.obs[tma_col].unique()):
            mask = adata.obs[tma_col] == tma
            tma_raw_min = float(adata.raw.X[mask.values].min())
            flag = " *** NEGATIVE ***" if tma_raw_min < 0 else ""
            print(f"  {tma}: {tma_raw_min:.3f}{flag}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="h5ad files to QC")
    args = parser.parse_args()

    for f in args.files:
        try:
            qc_file(f)
        except Exception as e:
            print(f"\nERROR reading {f}: {e}")


if __name__ == "__main__":
    main()
