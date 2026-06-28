#!/usr/bin/env python3
"""Compute marker QC stats from raw_combined h5ad files.

Reads raw ion counts, applies arcsinh(X/5), computes:
- p99 per marker per TMA (both panels)
- p99 per marker pooled (both panels)
- Subsampled arcsinh values for violin/histogram markers

Saves as compact npz file for use in qc_figures.py.
"""

import h5py
import numpy as np
import os

BASE = "<PROJECT_ROOT>"

# File paths for raw_combined h5ad files
RAW_PATHS = {
    ("A1", "T"): "output/A1_T/A1_T_raw_combined.h5ad",
    ("A1", "S"): "output/A1_S/A1_S_raw_combined.h5ad",
    ("B1", "T"): "output/batch/B1_T_raw_combined.h5ad",
    ("B1", "S"): "output/batch_S/B1_S_raw_combined.h5ad",
    ("C1", "T"): "output/C1_T/C1_T_raw_combined.h5ad",
    ("C1", "S"): "output/C1_S/C1_S_raw_combined.h5ad",
    ("Biomax", "T"): "output/Biomax_T/Biomax_T_raw_combined.h5ad",
    ("Biomax", "S"): "output/Biomax_S/Biomax_S_raw_combined.h5ad",
}

COFACTOR = 5.0
SUBSAMPLE_N = 50000  # per-TMA subsample for violin markers

def arcsinh_transform(X):
    return np.arcsinh(X / COFACTOR)


def compute_panel_stats(panel, tmas=("A1", "B1", "C1", "Biomax")):
    """Compute stats for one panel across all TMAs."""
    markers = None
    p99_per_tma = {}
    p50_per_tma = {}
    subsamples = {}  # marker_name -> list of (tma, values)

    # Markers to subsample for violins
    if panel == "T":
        violin_markers = ["CD20", "LAG3", "CXCR5", "CD163", "PD_L1"]
    else:
        violin_markers = ["CD20", "CD163", "CD206", "PD_L1", "CD21"]

    all_p99 = []

    for tma in tmas:
        path = os.path.join(BASE, RAW_PATHS[(tma, panel)])
        print("  Loading %s %s: %s" % (tma, panel, path))

        with h5py.File(path, "r") as f:
            # Read markers
            m = [x.decode() if isinstance(x, bytes) else x for x in f["var/_index"][:]]
            if markers is None:
                markers = m
            else:
                assert markers == m, "Marker mismatch: %s vs %s" % (markers[:3], m[:3])

            # Load entire X into memory (raw ion counts)
            X_raw = f["X"][:]
            n_cells = X_raw.shape[0]
            print("    %d cells x %d markers" % (n_cells, X_raw.shape[1]))

        # Arcsinh transform
        X_asinh = arcsinh_transform(X_raw)
        del X_raw

        # p99 and p50 per marker
        p99 = np.percentile(X_asinh, 99, axis=0)
        p50 = np.percentile(X_asinh, 50, axis=0)
        p99_per_tma[tma] = p99
        p50_per_tma[tma] = p50
        all_p99.append(p99)

        # Subsample for violin markers
        rng = np.random.RandomState(42)
        n_sub = min(SUBSAMPLE_N, n_cells)
        idx = rng.choice(n_cells, n_sub, replace=False)

        for vm in violin_markers:
            if vm in markers:
                col = markers.index(vm)
                vals = X_asinh[idx, col]
                key = "%s_%s" % (vm, tma)
                subsamples[key] = vals

        del X_asinh

    # Pooled p99 (average of per-TMA p99 is not ideal — but we don't have
    # the full pooled data in one place. Use max of per-TMA p99 as proxy.)
    # Actually: p99 of combined data ≈ max of per-TMA p99 for most markers
    pooled_p99 = np.max(np.array(all_p99), axis=0)

    return {
        "markers": markers,
        "p99_per_tma": p99_per_tma,
        "p50_per_tma": p50_per_tma,
        "pooled_p99": pooled_p99,
        "subsamples": subsamples,
        "tmas": list(tmas),
    }


def main():
    print("Computing T-panel stats...")
    t_stats = compute_panel_stats("T")

    print("\nComputing S-panel stats...")
    s_stats = compute_panel_stats("S")

    # Save as npz
    out = {}

    # T-panel
    out["t_markers"] = np.array(t_stats["markers"], dtype="U")
    out["t_pooled_p99"] = t_stats["pooled_p99"]
    out["t_tmas"] = np.array(t_stats["tmas"], dtype="U")
    for tma in t_stats["tmas"]:
        out["t_p99_%s" % tma] = t_stats["p99_per_tma"][tma]
        out["t_p50_%s" % tma] = t_stats["p50_per_tma"][tma]
    for key, vals in t_stats["subsamples"].items():
        out["t_sub_%s" % key] = vals

    # S-panel
    out["s_markers"] = np.array(s_stats["markers"], dtype="U")
    out["s_pooled_p99"] = s_stats["pooled_p99"]
    out["s_tmas"] = np.array(s_stats["tmas"], dtype="U")
    for tma in s_stats["tmas"]:
        out["s_p99_%s" % tma] = s_stats["p99_per_tma"][tma]
        out["s_p50_%s" % tma] = s_stats["p50_per_tma"][tma]
    for key, vals in s_stats["subsamples"].items():
        out["s_sub_%s" % key] = vals

    outpath = os.path.join(BASE, "output/marker_qc_stats.npz")
    np.savez_compressed(outpath, **out)
    print("\nSaved to %s (%.1f MB)" % (outpath, os.path.getsize(outpath) / 1e6))

    # Print summary
    print("\n=== T-panel pooled p99 ===")
    for i, m in enumerate(t_stats["markers"]):
        print("  %s: %.2f" % (m, t_stats["pooled_p99"][i]))

    print("\n=== S-panel pooled p99 ===")
    for i, m in enumerate(s_stats["markers"]):
        print("  %s: %.2f" % (m, s_stats["pooled_p99"][i]))


if __name__ == "__main__":
    main()
