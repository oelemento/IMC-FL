#!/usr/bin/env python3
"""
OSDR-style tissue dynamics reconstruction for follicular lymphoma.

Implements the approach from Somer, Mannor & Alon (Nature 2026):
1. Compute neighborhood composition for each cell (within radius r)
2. Fit logistic regression: P(Ki67 > threshold) ~ neighborhood counts
3. Estimate division rates as function of neighborhood composition
4. Construct phase portraits for cell type pairs

Usage:
    .venv/bin/python scripts/osdr_fl.py --s-utag output/all_TMA_S_utag_ct_merged.h5ad

Reference: Somer et al., Nature 650, 490-499 (2026)
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ── Configuration ──────────────────────────────────────────────────────────
CONTROL_PATTERNS = ["tonsil", "prostate", "kidney", "spleen", "adrenal",
                    "_Ton_", "_Adr_"]
RADIUS = 80.0        # µm, matching Somer et al.
KI67_THRESH = 0.5    # z-scored Ki67 threshold for division
MIN_CELLS = 500      # minimum cells per ROI


def is_control(sid):
    sid_lower = sid.lower()
    return any(p.lower() in sid_lower for p in CONTROL_PATTERNS)


def load_array(f, key):
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


def compute_neighborhood_counts(coords, labels, cell_types_list, radius=80.0):
    """For each cell, count neighbors of each type within radius.

    Returns: (N, n_types) array of neighbor counts.
    """
    tree = cKDTree(coords)
    n_types = len(cell_types_list)
    type_to_idx = {t: i for i, t in enumerate(cell_types_list)}

    # Encode labels as indices
    label_idx = np.array([type_to_idx.get(l, -1) for l in labels])

    # Query all neighbors within radius
    counts = np.zeros((len(coords), n_types), dtype=np.int32)
    neighbors = tree.query_ball_tree(tree, r=radius)

    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if j == i:
                continue  # exclude self
            idx = label_idx[j]
            if idx >= 0:
                counts[i, idx] += 1

    return counts


def fit_division_model(nhood_counts, ki67_vals, ki67_thresh=0.5):
    """Fit logistic regression: P(dividing) ~ neighborhood composition.

    Returns: fitted model, scaler, training accuracy
    """
    y = (ki67_vals > ki67_thresh).astype(int)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(nhood_counts)

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    model.fit(X, y)

    acc = model.score(X, y)
    frac_pos = y.mean()
    print(f"    Division model: acc={acc:.3f}, pos_frac={frac_pos:.3f}")

    return model, scaler, acc


def compute_phase_portrait(model_a, scaler_a, model_b, scaler_b,
                           cell_types_list, type_a, type_b,
                           n_grid=30, max_count=80, mean_other_counts=None,
                           death_rate_a=None, death_rate_b=None):
    """Compute phase portrait for two cell types using separate models.

    model_a predicts division probability for cells of type_a.
    model_b predicts division probability for cells of type_b.

    Returns: grid_a, grid_b, da_dt, db_dt, div_prob_a, div_prob_b
    """
    idx_a = cell_types_list.index(type_a)
    idx_b = cell_types_list.index(type_b)
    n_types = len(cell_types_list)

    # Create grid
    grid_a = np.linspace(0, max_count, n_grid)
    grid_b = np.linspace(0, max_count, n_grid)
    GA, GB = np.meshgrid(grid_a, grid_b)

    div_prob_a = np.zeros_like(GA)
    div_prob_b = np.zeros_like(GA)

    for i in range(n_grid):
        for j in range(n_grid):
            na = GA[i, j]
            nb = GB[i, j]

            # Build neighborhood vector
            nhood = np.zeros(n_types)
            if mean_other_counts is not None:
                nhood[:] = mean_other_counts
            nhood[idx_a] = na
            nhood[idx_b] = nb

            # Predict division probability for type_a cell in this neighborhood
            Xa = scaler_a.transform(nhood.reshape(1, -1))
            div_prob_a[i, j] = model_a.predict_proba(Xa)[0, 1]

            # Predict division probability for type_b cell in this neighborhood
            Xb = scaler_b.transform(nhood.reshape(1, -1))
            div_prob_b[i, j] = model_b.predict_proba(Xb)[0, 1]

    # Death rate: use provided or estimate as mean division rate per type
    if death_rate_a is None:
        death_rate_a = float(div_prob_a.mean())
    if death_rate_b is None:
        death_rate_b = float(div_prob_b.mean())

    # Rate of change: dX/dt = (div_rate - death_rate) * X
    da_dt = (div_prob_a - death_rate_a) * GA
    db_dt = (div_prob_b - death_rate_b) * GB

    return GA, GB, da_dt, db_dt, div_prob_a, div_prob_b


def plot_phase_portrait(GA, GB, da_dt, db_dt, div_prob_a, div_prob_b,
                        type_a, type_b, obs_a=None, obs_b=None,
                        title="", outpath=None):
    """Plot phase portrait with streamlines and observed data."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel a: Division probability heatmap + streamlines
    ax = axes[0]
    # Average division probability
    avg_div = (div_prob_a + div_prob_b) / 2
    im = ax.contourf(GA, GB, avg_div, levels=20, cmap="RdBu_r", alpha=0.6)
    plt.colorbar(im, ax=ax, label="Mean division probability")

    # Streamlines
    speed = np.sqrt(da_dt**2 + db_dt**2)
    lw = 2 * speed / (speed.max() + 1e-10)
    ax.streamplot(GA[0, :], GB[:, 0], da_dt, db_dt,
                  color="black", linewidth=lw, density=1.5, arrowsize=1.2)

    # Overlay observed data
    if obs_a is not None and obs_b is not None:
        ax.scatter(obs_a, obs_b, c="gray", s=3, alpha=0.3, zorder=5)

    ax.set_xlabel(f"{type_a} count (neighborhood)", fontsize=12)
    ax.set_ylabel(f"{type_b} count (neighborhood)", fontsize=12)
    ax.set_title(f"Phase portrait: {type_a} vs {type_b}", fontsize=13)

    # Panel b: Nullclines
    ax2 = axes[1]
    # Nullcline for type_a: where da_dt = 0 → div_prob_a = death_rate
    mean_div = (div_prob_a.mean() + div_prob_b.mean()) / 2
    ax2.contour(GA, GB, div_prob_a - mean_div, levels=[0],
                colors="blue", linewidths=2, linestyles="--")
    ax2.contour(GA, GB, div_prob_b - mean_div, levels=[0],
                colors="red", linewidths=2, linestyles="--")

    # Fixed points: where both nullclines cross
    # (approximate by finding grid cells where both are near zero)
    near_zero_a = np.abs(div_prob_a - mean_div) < 0.005
    near_zero_b = np.abs(div_prob_b - mean_div) < 0.005
    fixed = near_zero_a & near_zero_b
    if fixed.any():
        fp_a = GA[fixed]
        fp_b = GB[fixed]
        ax2.scatter(fp_a, fp_b, c="black", s=100, marker="o", zorder=10,
                    edgecolors="white", linewidths=2, label="Fixed points")

    if obs_a is not None and obs_b is not None:
        ax2.scatter(obs_a, obs_b, c="gray", s=3, alpha=0.3, zorder=5)

    ax2.set_xlabel(f"{type_a} count (neighborhood)", fontsize=12)
    ax2.set_ylabel(f"{type_b} count (neighborhood)", fontsize=12)
    ax2.set_title("Nullclines (blue=A, red=B) + fixed points", fontsize=13)
    ax2.legend(loc="upper right")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {outpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="OSDR tissue dynamics for FL")
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--radius", type=float, default=RADIUS)
    parser.add_argument("--ki67-thresh", type=float, default=KI67_THRESH)
    parser.add_argument("--max-rois", type=int, default=0, help="Limit ROIs (0=all)")
    parser.add_argument("--output", default="output/osdr")
    parser.add_argument("--zone", default="all",
                        choices=["all", "follicular", "interfollicular"],
                        help="Restrict to follicular or interfollicular zone")
    parser.add_argument("--detailed-types", action="store_true",
                        help="Use paper-relevant cell types (CD14+ FDC, M2 Mac, etc.)")
    args = parser.parse_args()

    zone_suffix = f"_{args.zone}" if args.zone != "all" else ""
    type_suffix = "_detailed" if args.detailed_types else ""
    outdir = Path(args.output + zone_suffix + type_suffix)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Zone definitions ───────────────────────────────────────────────────
    FOLLICULAR_COMPS = {
        "B cell zone (BCL2+)", "B cell zone (PAX5+)", "FDC network zone",
        "FDC / myeloid zone",
    }
    INTERFOLLICULAR_COMPS = {
        "T cell zone", "Other / myeloid zone", "Stromal / CAF zone",
        "Mixed (M2 Macrophag 26%)", "B/T mixed zone",
    }

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading {args.s_utag}...")
    f = h5py.File(args.s_utag, "r")
    ct = load_array(f, "cell_type")
    sids = load_array(f, "sample_id")
    comps = load_array(f, "compartment_name")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]

    var_names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["var"]["_index"][:]]
    ki67_idx = var_names.index("Ki-67")
    ki67 = f["X"][:, ki67_idx]

    # Load CD14 for FDC splitting
    cd14_idx = var_names.index("CD14")
    cd14 = f["X"][:, cd14_idx]
    f.close()

    n_total = len(ct)
    print(f"  {n_total} cells")

    # Filter controls
    unique_sids = np.unique(sids)
    ctrl_sids = set(s for s in unique_sids if is_control(s))
    keep = np.array([s not in ctrl_sids for s in sids])

    ct = ct[keep]; sids = sids[keep]; comps = comps[keep]
    cx = cx[keep]; cy = cy[keep]; ki67 = ki67[keep]; cd14 = cd14[keep]
    print(f"  After control exclusion: {len(ct)} cells")

    # Exclude Unassigned and Low quality
    typed = ~np.isin(ct, ["Unassigned", "Low quality / Unassigned"])
    ct = ct[typed]; sids = sids[typed]; comps = comps[typed]
    cx = cx[typed]; cy = cy[typed]; ki67 = ki67[typed]; cd14 = cd14[typed]
    print(f"  After excluding untyped: {len(ct)} cells")

    # ── Zone filtering ─────────────────────────────────────────────────────
    if args.zone == "follicular":
        zone_mask = np.isin(comps, list(FOLLICULAR_COMPS))
        print(f"  Follicular zone filter: {zone_mask.sum()} cells")
    elif args.zone == "interfollicular":
        zone_mask = np.isin(comps, list(INTERFOLLICULAR_COMPS))
        print(f"  Interfollicular zone filter: {zone_mask.sum()} cells")
    else:
        zone_mask = np.ones(len(ct), dtype=bool)

    ct = ct[zone_mask]; sids = sids[zone_mask]; comps = comps[zone_mask]
    cx = cx[zone_mask]; cy = cy[zone_mask]
    ki67 = ki67[zone_mask]; cd14 = cd14[zone_mask]
    print(f"  After zone filter: {len(ct)} cells")

    # ── Cell type assignment ───────────────────────────────────────────────
    if args.detailed_types:
        # Paper-relevant types with CD14+ FDC split
        fdc_mask = ct == "FDC"
        cd14_p75 = float(np.percentile(cd14[fdc_mask], 75)) if fdc_mask.sum() > 0 else 1.0
        print(f"  CD14 p75 for FDC split: {cd14_p75:.3f}")

        CONSOL = {
            "B cells (BCL2+)": "Tumor B (BCL2+)",
            "B cells (PAX5+)": "Tumor B (PAX5+)",
            "B cells": "B cells",
            "CD4 T cells": "CD4 T",
            "CD8 T cells": "CD8 T",
            "FDC": "FDC",  # placeholder, split below
            "M1 Macrophages": "M1 Mac",
            "M2 Macrophages": "M2 Mac",
            "Macrophages": "Macrophages",
            "Myeloid (S100A9+)": "S100A9+ myeloid",
            "Dendritic cells": "DC",
        }
        ct_consol = np.array([CONSOL.get(c, "Other") for c in ct])
        # Split FDCs by CD14
        ct_consol[fdc_mask & (cd14 >= cd14_p75)] = "FDC (CD14+)"
        ct_consol[fdc_mask & (cd14 < cd14_p75)] = "FDC (CD14-)"
    else:
        CONSOL = {
            "B cells (BCL2+)": "B cells",
            "B cells (PAX5+)": "B cells",
            "B cells": "B cells",
            "CD4 T cells": "T cells",
            "CD8 T cells": "T cells",
            "FDC": "FDC",
            "M1 Macrophages": "Macrophages",
            "M2 Macrophages": "Macrophages",
            "Macrophages": "Macrophages",
            "Myeloid (S100A9+)": "Macrophages",
            "Dendritic cells": "Macrophages",
        }
        ct_consol = np.array([CONSOL.get(c, "Other") for c in ct])

    cell_types_list = sorted(set(ct_consol))
    print(f"  Cell types ({len(cell_types_list)}): {cell_types_list}")
    for t in cell_types_list:
        n = (ct_consol == t).sum()
        ki_frac = float((ki67[ct_consol == t] > args.ki67_thresh).mean())
        print(f"    {t:<20s}: {n:>8d} cells, Ki67+ frac = {ki_frac:.3f}")

    # ── Compute neighborhoods per ROI ──────────────────────────────────────
    unique_rois = sorted(set(sids))
    if args.max_rois > 0:
        unique_rois = unique_rois[:args.max_rois]

    print(f"\nComputing neighborhoods for {len(unique_rois)} ROIs (r={args.radius}µm)...")

    all_nhood = []
    all_ki67 = []
    all_ct = []
    all_roi_labels = []

    for i, roi in enumerate(unique_rois):
        mask = sids == roi
        n = mask.sum()
        if n < MIN_CELLS:
            continue

        coords = np.column_stack([cx[mask], cy[mask]])
        labels = ct_consol[mask]
        ki_vals = ki67[mask]

        nhood = compute_neighborhood_counts(coords, labels, cell_types_list,
                                            radius=args.radius)
        all_nhood.append(nhood)
        all_ki67.append(ki_vals)
        all_ct.append(labels)
        all_roi_labels.append(np.full(n, roi))

        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(unique_rois)} ROIs done")

    nhood_all = np.vstack(all_nhood)
    ki67_all = np.concatenate(all_ki67)
    ct_all = np.concatenate(all_ct)
    print(f"  Total: {len(ki67_all)} cells with neighborhoods")

    # ── Fit division models per cell type ──────────────────────────────────
    print("\nFitting division models...")
    models = {}
    for cell_type in cell_types_list:
        if cell_type == "Other":
            continue
        mask = ct_all == cell_type
        n = mask.sum()
        if n < 1000:
            print(f"  {cell_type}: too few cells ({n}), skipping")
            continue

        print(f"  {cell_type} (n={n}):")
        model, scaler, acc = fit_division_model(
            nhood_all[mask], ki67_all[mask], ki67_thresh=args.ki67_thresh
        )
        models[cell_type] = (model, scaler)

        # Print feature importance (coefficients)
        coefs = model.coef_[0]
        coef_names = cell_types_list
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        print("    Top predictors of division:")
        for k in sorted_idx[:5]:
            print(f"      {coef_names[k]:<15s}: coef={coefs[k]:>+.3f}")

    # ── Phase portraits ───────────────────────────────────────────────────
    print("\nConstructing phase portraits...")

    # Mean neighborhood counts (for "other" types held constant)
    mean_nhood = nhood_all.mean(axis=0)

    if args.detailed_types:
        PAIRS = [
            ("FDC (CD14+)", "Tumor B (BCL2+)", "CD14+ FDC vs Tumor B (BCL2+)"),
            ("FDC (CD14-)", "Tumor B (BCL2+)", "CD14- FDC vs Tumor B (BCL2+)"),
            ("FDC (CD14+)", "FDC (CD14-)", "CD14+ FDC vs CD14- FDC"),
            ("FDC (CD14+)", "M2 Mac", "CD14+ FDC vs M2 Mac"),
            ("CD8 T", "Tumor B (BCL2+)", "CD8 T vs Tumor B (BCL2+)"),
            ("CD4 T", "Tumor B (BCL2+)", "CD4 T vs Tumor B (BCL2+)"),
            ("M1 Mac", "Tumor B (BCL2+)", "M1 Mac vs Tumor B"),
            ("M2 Mac", "Tumor B (BCL2+)", "M2 Mac vs Tumor B"),
        ]
    else:
        PAIRS = [
            ("FDC", "B cells", "FDC vs Tumor B cells"),
            ("FDC", "Macrophages", "FDC vs Macrophages"),
            ("T cells", "B cells", "T cells vs B cells"),
            ("Macrophages", "B cells", "Macrophages vs B cells"),
        ]

    for type_a, type_b, title in PAIRS:
        if type_a not in models or type_b not in models:
            print(f"  Skipping {title}: missing model")
            continue

        print(f"  {title}...")
        model_a, scaler_a = models[type_a]
        model_b, scaler_b = models[type_b]

        # Get max counts for grid
        idx_a = cell_types_list.index(type_a)
        idx_b = cell_types_list.index(type_b)
        max_a = float(np.percentile(nhood_all[:, idx_a], 99))
        max_b = float(np.percentile(nhood_all[:, idx_b], 99))
        max_count = max(max_a, max_b, 20)

        # Death rates: mean Ki67+ fraction per cell type (steady-state approx)
        dr_a = float((ki67_all[ct_all == type_a] > args.ki67_thresh).mean())
        dr_b = float((ki67_all[ct_all == type_b] > args.ki67_thresh).mean())

        GA, GB, da_dt, db_dt, div_a, div_b = compute_phase_portrait(
            model_a, scaler_a, model_b, scaler_b,
            cell_types_list, type_a, type_b,
            n_grid=30, max_count=max_count, mean_other_counts=mean_nhood,
            death_rate_a=dr_a, death_rate_b=dr_b
        )

        # Get observed data points for overlay
        mask_a = ct_all == type_a
        obs_a = nhood_all[mask_a, idx_a]
        obs_b = nhood_all[mask_a, idx_b]

        fname = f"phase_{type_a.replace(' ', '_')}_vs_{type_b.replace(' ', '_')}.png"
        plot_phase_portrait(
            GA, GB, da_dt, db_dt, div_a, div_b,
            type_a, type_b,
            obs_a=obs_a, obs_b=obs_b,
            title=f"OSDR Phase Portrait: {title}",
            outpath=outdir / fname
        )

    # ── Ki67 vs neighborhood plots (Fig 3a analog) ────────────────────────
    print("\nPlotting Ki67 vs neighborhood composition...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (cell_type, color) in zip(axes.flat,
        [("FDC", "#FF7F00"), ("B cells", "#4DAF4A"),
         ("T cells", "#377EB8"), ("Macrophages", "#E41A1C")]):

        mask = ct_all == cell_type
        n = mask.sum()
        if n < 100:
            continue

        ki_vals = ki67_all[mask]
        idx = cell_types_list.index(cell_type)
        own_nhood = nhood_all[mask, idx]

        # Bin by neighborhood count and compute mean Ki67
        bins = np.linspace(0, np.percentile(own_nhood, 99), 20)
        bin_centers = []
        bin_ki67 = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (own_nhood >= lo) & (own_nhood < hi)
            if in_bin.sum() > 50:
                bin_centers.append((lo + hi) / 2)
                bin_ki67.append(float(ki_vals[in_bin].mean()))

        ax.scatter(own_nhood, ki_vals, c=color, s=1, alpha=0.05)
        if bin_centers:
            ax.plot(bin_centers, bin_ki67, "k-o", linewidth=2, markersize=6)

        ax.axhline(args.ki67_thresh, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(f"# {cell_type} neighbors (r={args.radius}µm)", fontsize=11)
        ax.set_ylabel("Ki-67 (z-scored)", fontsize=11)
        ax.set_title(f"{cell_type} (n={n:,})", fontsize=12)
        ax.set_ylim(-1, 6)

    plt.suptitle("Ki-67 vs neighborhood composition (OSDR step 1)", fontsize=14)
    plt.tight_layout()
    fig.savefig(outdir / "ki67_vs_neighborhood.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {outdir}/ki67_vs_neighborhood.png")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
