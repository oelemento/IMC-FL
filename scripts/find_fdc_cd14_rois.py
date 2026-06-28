#!/usr/bin/env python3
"""Find ROIs where CD14-high FDCs co-localize with myeloid + CD8 T cells.

Produces spatial scatter plots of candidate ROIs showing the configuration:
CD14-high FDCs (gold) near macrophages (red) and CD8 T (cyan), away from B cells (blue).
"""

import sys
import numpy as np
from pathlib import Path
from collections import Counter

import h5py
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from fig_fdc_intrafollicular import is_tumor_core, load_array
from src.clinical_linkage import EXCLUDE_ROIS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_s_panel(path):
    """Load S-panel UTAG with cell types, coordinates, CD14."""
    with h5py.File(path, "r") as f:
        sid = load_array(f, "sample_id")
        ct = load_array(f, "cell_type")
        cx = f["obs"]["centroid_x"][:]
        cy = f["obs"]["centroid_y"][:]

        # Get CD14 from .X — find its column index
        var_names = [v.decode() if isinstance(v, bytes) else v
                     for v in f["var"]["_index"][:]]
        cd14_idx = var_names.index("CD14")
        cd14 = np.array(f["X"][:, cd14_idx]).flatten()

        # Get compartment if available
        comp = None
        if "compartment" in f["obs"]:
            comp = load_array(f, "compartment")

    mask = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS for s in sid])
    return {
        "sid": sid[mask], "ct": ct[mask],
        "cx": cx[mask], "cy": cy[mask],
        "cd14": cd14[mask],
        "comp": comp[mask] if comp is not None else None,
    }


def score_rois(data):
    """Score ROIs by co-localization of CD14-high FDCs + macrophages + CD8 T."""
    is_fdc = data["ct"] == "FDC"
    cd14_q75 = np.percentile(data["cd14"][is_fdc], 75)

    is_cd14hi_fdc = is_fdc & (data["cd14"] >= cd14_q75)
    is_mac = np.isin(data["ct"], ["M1 Macrophages", "M2 Macrophages", "Macrophages"])
    is_cd8 = data["ct"] == "CD8 T cells"
    is_bcell = np.isin(data["ct"], ["B cells (BCL2+)", "B cells (PAX5+)", "B cells"])

    rois = sorted(set(data["sid"]))
    scores = []

    for roi in rois:
        rm = data["sid"] == roi
        n_cd14hi = is_cd14hi_fdc[rm].sum()
        n_mac = is_mac[rm].sum()
        n_cd8 = is_cd8[rm].sum()
        n_bcell = is_bcell[rm].sum()
        n_fdc = is_fdc[rm].sum()

        if n_cd14hi < 20 or n_mac < 20 or n_cd8 < 10:
            continue

        # Build KDTree for this ROI
        coords = np.column_stack([data["cx"][rm], data["cy"][rm]])
        tree = KDTree(coords)

        # For each CD14-high FDC, find nearest mac, CD8, B cell
        cd14hi_idx = np.where(is_cd14hi_fdc[rm])[0]
        mac_idx = np.where(is_mac[rm])[0]
        cd8_idx = np.where(is_cd8[rm])[0]
        bcell_idx = np.where(is_bcell[rm])[0]

        if len(mac_idx) == 0 or len(cd8_idx) == 0 or len(bcell_idx) == 0:
            continue

        mac_tree = KDTree(coords[mac_idx])
        cd8_tree = KDTree(coords[cd8_idx])
        bcell_tree = KDTree(coords[bcell_idx])

        cd14hi_coords = coords[cd14hi_idx]
        d_mac, _ = mac_tree.query(cd14hi_coords)
        d_cd8, _ = cd8_tree.query(cd14hi_coords)
        d_bcell, _ = bcell_tree.query(cd14hi_coords)

        # We want: FDCs near mac/CD8, far from B cells
        # Score = fraction of CD14-high FDCs with mac<30µm AND cd8<50µm AND bcell>20µm
        near_mac = d_mac < 30
        near_cd8 = d_cd8 < 50
        far_bcell = d_bcell > 20
        config_frac = (near_mac & near_cd8 & far_bcell).mean()

        # Also track: how many CD14-high FDCs have all 3 nearby (within 50µm)
        all_nearby = (d_mac < 50) & (d_cd8 < 50)
        trio_count = all_nearby.sum()

        scores.append({
            "roi": roi,
            "n_cd14hi": n_cd14hi, "n_mac": n_mac, "n_cd8": n_cd8,
            "n_bcell": n_bcell, "n_fdc": n_fdc,
            "med_d_mac": np.median(d_mac),
            "med_d_cd8": np.median(d_cd8),
            "med_d_bcell": np.median(d_bcell),
            "config_frac": config_frac,
            "trio_count": trio_count,
        })

    scores.sort(key=lambda x: -x["trio_count"])
    return scores


def plot_roi(data, roi, output_dir):
    """Spatial scatter for one ROI highlighting the FDC-myeloid-CD8 configuration."""
    rm = data["sid"] == roi
    cx, cy = data["cx"][rm], data["cy"][rm]
    ct = data["ct"][rm]
    cd14 = data["cd14"][rm]
    is_fdc = ct == "FDC"
    cd14_q75 = np.percentile(data["cd14"][data["ct"] == "FDC"], 75)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Background: all cells gray
    ax.scatter(cx, cy, c="#E0E0E0", s=1, alpha=0.3, rasterized=True, zorder=1)

    # B cells: blue
    b_mask = np.isin(ct, ["B cells (BCL2+)", "B cells (PAX5+)", "B cells"])
    ax.scatter(cx[b_mask], cy[b_mask], c="#4393C3", s=3, alpha=0.4,
               label=f"B cells ({b_mask.sum():,})", rasterized=True, zorder=2)

    # CD8 T cells: cyan
    cd8_mask = ct == "CD8 T cells"
    ax.scatter(cx[cd8_mask], cy[cd8_mask], c="#00BCD4", s=8, alpha=0.7,
               edgecolors="black", linewidth=0.2,
               label=f"CD8 T ({cd8_mask.sum():,})", rasterized=True, zorder=4)

    # Macrophages: red
    mac_mask = np.isin(ct, ["M1 Macrophages", "M2 Macrophages", "Macrophages"])
    ax.scatter(cx[mac_mask], cy[mac_mask], c="#E41A1C", s=8, alpha=0.7,
               edgecolors="black", linewidth=0.2,
               label=f"Macrophages ({mac_mask.sum():,})", rasterized=True, zorder=4)

    # FDCs: CD14-low = light orange, CD14-high = gold with black edge
    fdc_lo = is_fdc & (cd14 < cd14_q75)
    fdc_hi = is_fdc & (cd14 >= cd14_q75)
    ax.scatter(cx[fdc_lo], cy[fdc_lo], c="#FDDBC7", s=6, alpha=0.5,
               edgecolors="gray", linewidth=0.1,
               label=f"FDC CD14-low ({fdc_lo.sum():,})", rasterized=True, zorder=3)
    ax.scatter(cx[fdc_hi], cy[fdc_hi], c="#FFD700", s=15, alpha=0.9,
               edgecolors="black", linewidth=0.5,
               label=f"FDC CD14-high ({fdc_hi.sum():,})", rasterized=True, zorder=5)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{roi}\nCD14-high FDCs + Macrophages + CD8 T cells", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", markerscale=1.5)

    out = Path(output_dir) / f"roi_fdc_config_{roi}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    parser.add_argument("--top-n", type=int, default=6)
    args = parser.parse_args()

    print("Loading S-panel UTAG...")
    data = load_s_panel(args.s_utag)
    print(f"  {len(data['sid']):,} tumor cells")

    print("\nScoring ROIs for CD14-high FDC + myeloid + CD8 configuration...")
    scores = score_rois(data)

    print(f"\nTop {args.top_n} ROIs:")
    print(f"  {'ROI':<25} {'CD14hi':>6} {'Mac':>5} {'CD8':>5} {'B':>6} "
          f"{'d_mac':>6} {'d_cd8':>6} {'d_B':>6} {'trio':>5}")
    for s in scores[:args.top_n]:
        print(f"  {s['roi']:<25} {s['n_cd14hi']:>6} {s['n_mac']:>5} {s['n_cd8']:>5} "
              f"{s['n_bcell']:>6} {s['med_d_mac']:>6.1f} {s['med_d_cd8']:>6.1f} "
              f"{s['med_d_bcell']:>6.1f} {s['trio_count']:>5}")

    print(f"\nPlotting top {args.top_n} ROIs...")
    for s in scores[:args.top_n]:
        out = plot_roi(data, s["roi"], args.output_dir)
        print(f"  {out}")

    print("\nDone.")
