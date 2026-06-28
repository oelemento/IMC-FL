#!/usr/bin/env python3
"""
Spatial motif analysis: enumerate enriched cell-type triads in FL tissue.

Builds a Delaunay triangulation on cell centroids, extracts all triangles,
labels each triangle by its 3 cell types (sorted → canonical motif ID),
compares observed motif counts to a permutation null model.

Usage:
    .venv/bin/python scripts/spatial_motifs.py --s-panel output/all_TMA_S_global_v8.h5ad

Focuses on FDC network zone by default. Reports z-scores for each triad motif.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


# ── Configuration ──────────────────────────────────────────────────────────
CONTROL_PATTERNS = ["tonsil", "prostate", "kidney", "spleen", "adrenal",
                    "_Ton_", "_Adr_"]
MIN_CELLS = 200  # minimum cells in zone per ROI to include
N_PERM = 200     # permutations for null model
ZONE = "FDC network zone"


def is_control(sid):
    sid_lower = sid.lower()
    for pat in CONTROL_PATTERNS:
        if pat.lower() in sid_lower:
            return True
    return False


def get_triangle_motifs(tri, labels):
    """Extract canonical motif labels for all Delaunay triangles.

    Args:
        tri: Delaunay triangulation object
        labels: array of cell type labels (same length as points)

    Returns:
        list of tuples (sorted cell type triad)
    """
    simplices = tri.simplices  # (n_triangles, 3)
    motifs = []
    for i0, i1, i2 in simplices:
        m = tuple(sorted([labels[i0], labels[i1], labels[i2]]))
        motifs.append(m)
    return motifs


def count_motifs(tri, labels):
    """Count each canonical triad motif."""
    motifs = get_triangle_motifs(tri, labels)
    return Counter(motifs)


def compute_pairwise_edge_probs(tri, labels):
    """Compute pairwise edge type frequencies from Delaunay triangulation.

    Returns dict mapping (type_a, type_b) -> probability (sorted canonical pair).
    """
    edge_counts = Counter()
    for i0, i1, i2 in tri.simplices:
        for a, b in [(i0, i1), (i1, i2), (i0, i2)]:
            pair = tuple(sorted([labels[a], labels[b]]))
            edge_counts[pair] += 1
    total = sum(edge_counts.values())
    return {k: v / total for k, v in edge_counts.items()}


def expected_triad_count_from_pairwise(motif, edge_probs, n_triangles):
    """Compute expected triad count assuming edge independence.

    For triad (a, b, c) with edges (a,b), (a,c), (b,c):
    Expected = n_triangles × p(a,b) × p(a,c) × p(b,c) × combinatorial_factor

    Combinatorial factor accounts for vertex assignment:
    - (A,A,A): 1 way to assign 3 identical types → factor = 1
    - (A,A,B): 3 ways to pick which vertex is B → factor = 3
    - (A,B,C): 6 ways to assign 3 different types → factor = 6
    But canonical pairs also have multiplicity — p(A,B) counts both A→B and B→A.
    """
    a, b, c = motif

    # Edge probabilities (canonical sorted pairs)
    p_ab = edge_probs.get(tuple(sorted([a, b])), 0)
    p_ac = edge_probs.get(tuple(sorted([a, c])), 0)
    p_bc = edge_probs.get(tuple(sorted([b, c])), 0)

    if p_ab == 0 or p_ac == 0 or p_bc == 0:
        return 0.0

    # Combinatorial factor: how many vertex permutations produce this canonical motif
    n_distinct = len(set([a, b, c]))
    if n_distinct == 1:    # (A, A, A)
        comb = 1
    elif n_distinct == 2:  # (A, A, B)
        comb = 3
    else:                  # (A, B, C)
        comb = 6

    return n_triangles * p_ab * p_ac * p_bc * comb


def run_permutation_test(tri, labels, n_perm=200, seed=42):
    """Compare observed motif counts to permuted null.

    Returns DataFrame with columns: motif, obs_count, null_mean, null_std, z_score, obs_frac
    Also computes conditional enrichment (beyond pairwise prediction).
    """
    rng = np.random.default_rng(seed)

    # Observed counts
    obs_counts = count_motifs(tri, labels)
    n_total = sum(obs_counts.values())

    # Pairwise edge probabilities for conditional test
    edge_probs = compute_pairwise_edge_probs(tri, labels)

    # Permutation null
    all_motif_keys = set(obs_counts.keys())
    null_counts = {m: [] for m in all_motif_keys}

    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_counts = count_motifs(tri, perm_labels)
        for m in all_motif_keys:
            null_counts[m].append(perm_counts.get(m, 0))

    # Compute z-scores (both permutation-based and conditional)
    rows = []
    for m in sorted(all_motif_keys, key=lambda x: obs_counts[x], reverse=True):
        null_arr = np.array(null_counts[m])
        null_mean = null_arr.mean()
        null_std = null_arr.std()
        z = (obs_counts[m] - null_mean) / null_std if null_std > 0 else 0.0

        # Conditional enrichment: observed vs pairwise-predicted
        # NOTE: edge-independence is approximate (edges in triangles share
        # vertices), but the ratio identifies genuine higher-order effects.
        expected = expected_triad_count_from_pairwise(m, edge_probs, n_total)
        # Log2 fold-change: positive = more than pairwise predicts
        if expected > 0 and obs_counts[m] > 0:
            cond_log2fc = np.log2(obs_counts[m] / expected)
        elif obs_counts[m] > 0:
            cond_log2fc = 10.0  # cap
        else:
            cond_log2fc = -10.0  # cap
        # Also store raw expected for filtering
        cond_z = cond_log2fc  # repurpose column name for simplicity

        rows.append({
            "motif": " — ".join(m),
            "cell_a": m[0],
            "cell_b": m[1],
            "cell_c": m[2],
            "obs_count": obs_counts[m],
            "obs_frac": obs_counts[m] / n_total,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z,
            "expected_pairwise": expected,
            "cond_log2fc": cond_log2fc,
            "cond_z": cond_z,
        })

    return pd.DataFrame(rows)


def analyze_roi(coords, cell_types, roi_name="", n_perm=200):
    """Run motif analysis on a single ROI's cells within the zone.

    Args:
        coords: (N, 2) array of centroids
        cell_types: (N,) array of cell type labels
        roi_name: string for logging
        n_perm: number of permutations

    Returns:
        DataFrame of motif enrichment results, or None if too few cells
    """
    if len(coords) < MIN_CELLS:
        return None

    # Build Delaunay triangulation
    try:
        tri = Delaunay(coords)
    except Exception as e:
        print(f"  WARNING: Delaunay failed for {roi_name}: {e}")
        return None

    n_triangles = tri.simplices.shape[0]
    print(f"  {roi_name}: {len(coords)} cells, {n_triangles} triangles")

    # Run permutation test
    labels = np.array(cell_types)
    df = run_permutation_test(tri, labels, n_perm=n_perm)
    df["roi"] = roi_name

    return df


def main():
    parser = argparse.ArgumentParser(description="Spatial motif (triad) analysis")
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad",
                        help="S-panel UTAG h5ad (has compartment_name)")
    parser.add_argument("--zone", default=ZONE, help="Compartment to analyze")
    parser.add_argument("--n-perm", type=int, default=N_PERM, help="Permutations")
    parser.add_argument("--all-zones", action="store_true",
                        help="Analyze all zones (ignore --zone)")
    parser.add_argument("--max-rois", type=int, default=0,
                        help="Limit ROIs for quick testing (0=all)")
    parser.add_argument("--output", default="output/spatial_motifs",
                        help="Output directory")
    args = parser.parse_args()

    n_perm = args.n_perm

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data via h5py (compartments stored as categorical) ────────────
    print(f"Loading {args.s_utag}...")
    f = h5py.File(args.s_utag, "r")

    def load_array(key):
        ds = f["obs"][key]
        if isinstance(ds, h5py.Group) and "categories" in ds:
            cats = ds["categories"][:]
            codes = ds["codes"][:]
            cats_str = np.array(
                [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
            )
            return cats_str[codes]
        vals = ds[:]
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])

    ct = load_array("cell_type")
    comps = load_array("compartment_name")
    sids = load_array("sample_id")
    x = f["obs"]["centroid_x"][:]
    y = f["obs"]["centroid_y"][:]

    # CD14 values for FDC splitting (read single column to avoid loading full X)
    var_names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["var"]["_index"][:]]
    cd14_idx = var_names.index("CD14") if "CD14" in var_names else None
    if cd14_idx is not None:
        # Read just the CD14 column — much faster than loading full X matrix
        cd14_vals = f["X"][:, cd14_idx]
    f.close()

    n_total = len(ct)
    print(f"  {n_total} cells")

    # Filter controls (vectorized via unique sample IDs)
    unique_sids = np.unique(sids)
    control_sids = set(s for s in unique_sids if is_control(s))
    keep = np.array([s not in control_sids for s in sids])
    print(f"  After control exclusion: {keep.sum()} cells")

    # Subdivide FDCs by CD14
    if cd14_idx is not None:
        fdc_mask = ct == "FDC"
        if fdc_mask.sum() > 0:
            cd14_p75 = float(np.percentile(cd14_vals[fdc_mask & keep], 75))
            print(f"  CD14 p75 threshold for FDC split: {cd14_p75:.3f}")
            ct[fdc_mask & (cd14_vals >= cd14_p75)] = "FDC (CD14+)"
            ct[fdc_mask & (cd14_vals < cd14_p75)] = "FDC (CD14-)"
    else:
        print("  WARNING: CD14 not found, FDCs not split")

    # Filter to zone
    if args.all_zones:
        zone_mask = np.ones(n_total, dtype=bool)
        zone_label = "all_zones"
    else:
        zone_mask = comps == args.zone
        zone_label = args.zone.replace(" ", "_").replace("/", "_")
        print(f"  Zone '{args.zone}': {zone_mask.sum()} cells")

    if zone_mask.sum() < MIN_CELLS:
        print(f"ERROR: Too few cells in zone ({zone_mask.sum()} < {MIN_CELLS})")
        sys.exit(1)

    # Exclude Unassigned and controls
    typed_mask = ct != "Unassigned"
    combined_mask = zone_mask & typed_mask & keep
    print(f"  After excluding Unassigned + controls: {combined_mask.sum()} cells")

    # ── Per-ROI analysis ───────────────────────────────────────────────────
    rois = sids
    unique_rois = sorted(set(rois[combined_mask]))
    if args.max_rois > 0:
        unique_rois = unique_rois[:args.max_rois]

    print(f"\nAnalyzing {len(unique_rois)} ROIs with {n_perm} permutations...")

    all_results = []
    for i, roi in enumerate(unique_rois):
        roi_mask = combined_mask & (rois == roi)
        n_cells = roi_mask.sum()
        if n_cells < MIN_CELLS:
            continue

        coords = np.column_stack([x[roi_mask], y[roi_mask]])
        labels = ct[roi_mask]

        df = analyze_roi(coords, labels, roi_name=roi, n_perm=n_perm)
        if df is not None:
            all_results.append(df)

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(unique_rois)} ROIs done")

    if not all_results:
        print("No results — all ROIs had too few cells.")
        sys.exit(1)

    results = pd.concat(all_results, ignore_index=True)

    # ── Aggregate across ROIs ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"RESULTS: {len(unique_rois)} ROIs, {len(results)} motif×ROI entries")
    print(f"{'='*80}")

    # Mean z-score across ROIs for each motif
    agg = results.groupby("motif").agg(
        mean_z=("z_score", "mean"),
        median_z=("z_score", "median"),
        mean_cond_log2fc=("cond_log2fc", "mean"),
        median_cond_log2fc=("cond_log2fc", "median"),
        mean_obs_frac=("obs_frac", "mean"),
        mean_obs_count=("obs_count", "mean"),
        n_rois=("roi", "nunique"),
        n_enriched=("z_score", lambda x: (x > 2).sum()),
        n_depleted=("z_score", lambda x: (x < -2).sum()),
        n_cond_enriched=("cond_log2fc", lambda x: (x > 1).sum()),
        n_cond_depleted=("cond_log2fc", lambda x: (x < -1).sum()),
    ).sort_values("mean_z", ascending=False)

    # Save full results
    results.to_csv(outdir / f"motif_results_per_roi_{zone_label}.csv", index=False)
    agg.to_csv(outdir / f"motif_summary_{zone_label}.csv")

    # Print top enriched motifs (permutation z)
    print(f"\nTop 20 ENRICHED motifs (permutation z-score):")
    print(f"{'Motif':<55} {'perm_z':>7} {'log2FC':>7} {'obs%':>7} {'n_ROI':>5}")
    print("-" * 90)
    for motif, row in agg.head(20).iterrows():
        print(f"{motif:<55} {row['mean_z']:>+7.1f} {row['mean_cond_log2fc']:>+7.1f} "
              f"{row['mean_obs_frac']*100:>6.2f}% {int(row['n_rois']):>5}")

    print(f"\nTop 20 DEPLETED motifs (permutation z-score):")
    print(f"{'Motif':<55} {'perm_z':>7} {'log2FC':>7} {'obs%':>7} {'n_ROI':>5}")
    print("-" * 90)
    for motif, row in agg.tail(20).iloc[::-1].iterrows():
        print(f"{motif:<55} {row['mean_z']:>+7.1f} {row['mean_cond_log2fc']:>+7.1f} "
              f"{row['mean_obs_frac']*100:>6.2f}% {int(row['n_rois']):>5}")

    # ── Conditional enrichment: triads beyond pairwise ─────────────────────
    print(f"\n{'='*80}")
    print(f"CONDITIONAL ENRICHMENT (log2 fold-change vs pairwise prediction):")
    print(f"log2FC > 0 = more frequent than pairwise predicts (genuine higher-order).")
    print(f"log2FC < 0 = less frequent than pairwise predicts (pairwise-driven).")
    print(f"Filter: obs_frac >= 0.1% to exclude rare noise.")
    print(f"{'='*80}")
    # Filter to motifs with at least 0.1% frequency for stability
    agg_filt = agg[agg["mean_obs_frac"] >= 0.001]
    cond_sorted = agg_filt.sort_values("mean_cond_log2fc", ascending=False)

    print(f"\nTop 30 CONDITIONALLY ENRICHED (log2FC > 0, obs >= 0.1%):")
    print(f"{'Motif':<55} {'perm_z':>7} {'log2FC':>7} {'obs%':>7} {'n_ROI':>5} {'c_enr':>5} {'c_dep':>5}")
    print("-" * 100)
    for motif, row in cond_sorted.head(30).iterrows():
        print(f"{motif:<55} {row['mean_z']:>+7.1f} {row['mean_cond_log2fc']:>+7.1f} "
              f"{row['mean_obs_frac']*100:>6.2f}% {int(row['n_rois']):>5} "
              f"{int(row['n_cond_enriched']):>5} {int(row['n_cond_depleted']):>5}")

    print(f"\nTop 30 CONDITIONALLY DEPLETED (log2FC < 0, obs >= 0.1%):")
    print(f"{'Motif':<55} {'perm_z':>7} {'log2FC':>7} {'obs%':>7} {'n_ROI':>5} {'c_enr':>5} {'c_dep':>5}")
    print("-" * 100)
    for motif, row in cond_sorted.tail(30).iloc[::-1].iterrows():
        print(f"{motif:<55} {row['mean_z']:>+7.1f} {row['mean_cond_log2fc']:>+7.1f} "
              f"{row['mean_obs_frac']*100:>6.2f}% {int(row['n_rois']):>5} "
              f"{int(row['n_cond_enriched']):>5} {int(row['n_cond_depleted']):>5}")

    # ── FDC-focused conditional ────────────────────────────────────────────
    fdc_motifs = cond_sorted[cond_sorted.index.str.contains("FDC")]
    if len(fdc_motifs) > 0:
        print(f"\n{'='*80}")
        print(f"FDC-containing motifs — CONDITIONAL enrichment ({len(fdc_motifs)} motifs):")
        print(f"{'='*80}")
        print(f"{'Motif':<55} {'perm_z':>7} {'log2FC':>7} {'obs%':>7}")
        print("-" * 80)
        for motif, row in fdc_motifs.iterrows():
            print(f"{motif:<55} {row['mean_z']:>+7.1f} {row['mean_cond_log2fc']:>+7.1f} "
                  f"{row['mean_obs_frac']*100:>6.2f}%")

    # ── Key comparison: perm_z vs log2FC ──────────────────────────────────
    # Find motifs where perm_z and log2FC disagree (the interesting ones)
    disagree = agg_filt[
        ((agg_filt["mean_z"] > 2) & (agg_filt["mean_cond_log2fc"] < -1)) |
        ((agg_filt["mean_z"] < -2) & (agg_filt["mean_cond_log2fc"] > 1))
    ]
    if len(disagree) > 0:
        print(f"\n{'='*80}")
        print(f"SIGN DISAGREEMENT: perm_z and log2FC have opposite signs (obs >= 0.1%):")
        print(f"perm_z>0 + log2FC<0 = enriched by abundance but depleted as a triad")
        print(f"perm_z<0 + log2FC>0 = depleted by abundance but enriched as a triad")
        print(f"{'='*80}")
        print(f"{'Motif':<55} {'perm_z':>7} {'log2FC':>7} {'obs%':>7}")
        print("-" * 80)
        for motif, row in disagree.sort_values("mean_cond_log2fc", ascending=False).iterrows():
            print(f"{motif:<55} {row['mean_z']:>+7.1f} {row['mean_cond_log2fc']:>+7.1f} "
                  f"{row['mean_obs_frac']*100:>6.2f}%")

    print(f"\nResults saved to {outdir}/")
    print("Done.")


if __name__ == "__main__":
    main()
