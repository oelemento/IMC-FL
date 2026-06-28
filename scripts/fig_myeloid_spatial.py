"""
Figure: Myeloid spatial characterization and CD14 signal decomposition.

6-panel figure:
  (a) Macrophage domain enrichment (follicular vs interfollicular)
  (b) CD14 expression by cell type (bar chart)
  (c) CD14 signal decomposition (stacked bar: myeloid vs FDC vs other)
  (d) Scatter: non-mac CD14 vs mac CD14 per ROI
  (e) Representative spatial scatter: macrophages + FDCs in tissue
  (f) Per-ROI macrophage fraction vs CD14 intensity
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

# ---------------------------------------------------------------------------
# Helpers (from survival_analysis.py)
# ---------------------------------------------------------------------------

MYELOID_TYPES = [
    "M1 Macrophages", "M2 Macrophages", "Macrophages",
    "Myeloid (S100A9+)", "Dendritic cells",
]
SKIP_MARKERS = {"DNA1", "DNA2", "HistoneH3"}


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


def is_tumor_core(sample_id):
    s = sample_id.lower()
    if "_ton_" in s or "_adr_" in s:
        return False
    for tissue in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if tissue in s:
            return False
    if sample_id == "Biomax_ROI_006":
        return False
    return True


def panel_label(ax, letter, x=-0.08, y=1.05):
    ax.text(
        x, y, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top", ha="left",
    )


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_data(s_panel_path, s_utag_path):
    """Extract all data needed for the 6-panel figure."""
    print("Loading S-panel h5ad...")
    with h5py.File(s_panel_path, "r") as f:
        var_key = "_index" if "_index" in f["var"] else "index"
        var_names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in f["var"][var_key][:]]
        sids = load_array(f, "sample_id")
        ctypes = load_array(f, "cell_type")
        X = f["X"]

        cd14_idx = var_names.index("CD14")

        # Filter to tumor cores (excl Biomax)
        tumor_mask = np.array([
            is_tumor_core(s) and s not in EXCLUDE_ROIS
            and not s.startswith("Biomax")
            for s in sids
        ])
        tumor_sids = sids[tumor_mask]
        tumor_ct = ctypes[tumor_mask]
        tumor_indices = np.where(tumor_mask)[0]

        print(f"  {len(tumor_indices)} tumor cells")

        # --- (b) CD14 by cell type ---
        unique_ct = np.unique(tumor_ct)
        cd14_by_ct = {}
        for ct in unique_ct:
            ct_mask = tumor_ct == ct
            ct_indices = tumor_indices[ct_mask]
            if len(ct_indices) < 100:
                continue
            # Sample if too many to read efficiently
            if len(ct_indices) > 50000:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(ct_indices, 50000, replace=False)
                sample_idx.sort()
            else:
                sample_idx = ct_indices
            cd14_vals = X[sample_idx, cd14_idx]
            cd14_by_ct[ct] = {
                "mean": float(np.mean(cd14_vals)),
                "n": int(ct_mask.sum()),
                "frac_pos": float((cd14_vals > 1.0).mean()),
                "is_myeloid": ct in MYELOID_TYPES,
            }
        print(f"  CD14 by cell type: {len(cd14_by_ct)} types")

        # --- (c, d, f) Per-ROI metrics ---
        rois = sorted(set(tumor_sids))
        roi_rows = []
        for roi in rois:
            roi_mask = tumor_sids == roi
            roi_indices_local = tumor_indices[roi_mask]
            if len(roi_indices_local) < 100:
                continue
            roi_ct = tumor_ct[roi_mask]
            cd14_all = X[roi_indices_local[0]:roi_indices_local[-1] + 1, cd14_idx]
            n = len(roi_indices_local)

            is_myeloid = np.isin(roi_ct, MYELOID_TYPES)
            is_fdc = roi_ct == "FDC"

            # CD14 means by category
            cd14_mac = float(cd14_all[is_myeloid].mean()) if is_myeloid.sum() > 0 else 0
            cd14_fdc = float(cd14_all[is_fdc].mean()) if is_fdc.sum() > 0 else 0
            cd14_other = float(cd14_all[~is_myeloid & ~is_fdc].mean()) if (~is_myeloid & ~is_fdc).sum() > 0 else 0
            cd14_nonmac = float(cd14_all[~is_myeloid].mean()) if (~is_myeloid).sum() > 0 else 0

            # Positive CD14 signal decomposition
            pos_mask = cd14_all > 0
            total_pos = float(cd14_all[pos_mask].sum()) if pos_mask.sum() > 0 else 1
            mac_pos = float(cd14_all[is_myeloid & pos_mask].sum()) if (is_myeloid & pos_mask).sum() > 0 else 0
            fdc_pos = float(cd14_all[is_fdc & pos_mask].sum()) if (is_fdc & pos_mask).sum() > 0 else 0
            other_pos = total_pos - mac_pos - fdc_pos

            roi_rows.append({
                "roi": roi,
                "n_cells": n,
                "mac_frac": float(is_myeloid.sum()) / n,
                "fdc_frac": float(is_fdc.sum()) / n,
                "cd14_all": float(cd14_all.mean()),
                "cd14_mac": cd14_mac,
                "cd14_fdc": cd14_fdc,
                "cd14_other": cd14_other,
                "cd14_nonmac": cd14_nonmac,
                "pct_mac": mac_pos / total_pos * 100,
                "pct_fdc": fdc_pos / total_pos * 100,
                "pct_other": other_pos / total_pos * 100,
            })

        roi_df = pd.DataFrame(roi_rows)
        print(f"  {len(roi_df)} tumor ROIs")

    # --- (a) Domain enrichment from UTAG ---
    print("Loading S-panel UTAG for domain enrichment...")
    domain_enrich = {}
    with h5py.File(s_utag_path, "r") as f:
        u_sids = load_array(f, "sample_id")
        u_ct = load_array(f, "cell_type")
        # Check for compartment/domain columns
        obs_keys = list(f["obs"].keys())
        domain_key = None
        for k in ["compartment_name", "utag_domain", "leiden_0.015",
                   "UTAG Label_leiden_0.015"]:
            if k in obs_keys:
                domain_key = k
                break
        if domain_key is None:
            print(f"  WARNING: No domain column found in UTAG. Keys: {obs_keys[:20]}")
            domain_enrich = None
        else:
            domains = load_array(f, domain_key)
            # Filter to tumor
            u_tumor = np.array([
                is_tumor_core(s) and s not in EXCLUDE_ROIS
                and not s.startswith("Biomax")
                for s in u_sids
            ])

            u_ct_t = u_ct[u_tumor]
            u_dom_t = domains[u_tumor]

            # Classify domains as follicular based on B cell fraction
            # Identify B-lineage cell types (exclude FDC — they inflate domain counts)
            all_ct = np.unique(u_ct_t)
            b_lineage = [ct for ct in all_ct
                         if "B cell" in ct or "B " in ct or ct.startswith("B ")
                         or "GC" in ct or "Plasma" in ct]
            print(f"  B-lineage types for domain classification: {b_lineage}")

            unique_doms = np.unique(u_dom_t)
            foll_domains = set()
            for d in unique_doms:
                d_mask = u_dom_t == d
                d_ct = u_ct_t[d_mask]
                b_frac = np.mean(np.isin(d_ct, b_lineage))
                if b_frac > 0.5:
                    foll_domains.add(d)

            is_foll = np.isin(u_dom_t, list(foll_domains))
            n_foll = is_foll.sum()
            n_ifoll = (~is_foll).sum()
            total = n_foll + n_ifoll

            print(f"  Follicular domains: {foll_domains}")
            print(f"  {n_foll} follicular / {n_ifoll} interfollicular cells")

            for mac_type in MYELOID_TYPES:
                is_mac = u_ct_t == mac_type
                n_mac = is_mac.sum()
                if n_mac < 10:
                    continue
                mac_in_foll = (is_mac & is_foll).sum()
                mac_in_ifoll = (is_mac & ~is_foll).sum()
                expected_foll = n_mac * (n_foll / total)
                fold = (mac_in_foll / expected_foll) if expected_foll > 0 else 0
                domain_enrich[mac_type] = {
                    "fold_enrichment": fold,
                    "n_foll": int(mac_in_foll),
                    "n_ifoll": int(mac_in_ifoll),
                    "n_total": int(n_mac),
                }

    # --- (e) Representative ROI for spatial scatter ---
    # Pick a ROI with decent macrophage + FDC content
    roi_df_sorted = roi_df.sort_values("mac_frac", ascending=False)
    rep_roi = roi_df_sorted.iloc[len(roi_df_sorted) // 4]["roi"]  # 75th percentile
    print(f"  Representative ROI: {rep_roi}")

    # Get coordinates for representative ROI
    print(f"  Extracting coordinates for {rep_roi}...")
    rep_data = None
    with h5py.File(s_panel_path, "r") as f:
        sids2 = load_array(f, "sample_id")
        ctypes2 = load_array(f, "cell_type")
        obs_keys = list(f["obs"].keys())
        cx_key = "centroid_x" if "centroid_x" in obs_keys else "X_centroid"
        cy_key = "centroid_y" if "centroid_y" in obs_keys else "Y_centroid"

        roi_mask2 = sids2 == rep_roi
        if roi_mask2.sum() > 0:
            idx = np.where(roi_mask2)[0]
            cx = f["obs"][cx_key][idx[0]:idx[-1] + 1]
            cy = f["obs"][cy_key][idx[0]:idx[-1] + 1]
            ct_roi = ctypes2[roi_mask2]
            rep_data = pd.DataFrame({
                "x": cx, "y": cy, "cell_type": ct_roi,
            })
            print(f"  {len(rep_data)} cells in {rep_roi}")

    return cd14_by_ct, roi_df, domain_enrich, rep_data, rep_roi


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(cd14_by_ct, roi_df, domain_enrich, rep_data, rep_roi, output_dir):
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.35,
                  left=0.08, right=0.95, top=0.96, bottom=0.04)

    # --- (a) Domain enrichment ---
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    if domain_enrich:
        types = list(domain_enrich.keys())
        folds = [domain_enrich[t]["fold_enrichment"] for t in types]
        short_names = [t.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+ myeloid")
                       .replace("Dendritic cells", "DCs") for t in types]
        colors = ["#E41A1C" if f < 1 else "#377EB8" for f in folds]
        y_pos = range(len(types))
        ax_a.barh(list(y_pos), folds, color=colors, edgecolor="white", height=0.6)
        ax_a.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_a.set_yticks(list(y_pos))
        ax_a.set_yticklabels(short_names, fontsize=10)
        ax_a.set_xlabel("Fold enrichment in follicular domains")
        ax_a.set_title("Macrophage domain localization")
        for i, f in enumerate(folds):
            ax_a.text(f + 0.02, i, f"{f:.2f}x", va="center", fontsize=9)
    else:
        ax_a.text(0.5, 0.5, "No domain data", transform=ax_a.transAxes,
                  ha="center", va="center")

    # --- (b) CD14 by cell type ---
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    ct_sorted = sorted(cd14_by_ct.items(), key=lambda x: x[1]["mean"], reverse=True)
    names = [ct for ct, _ in ct_sorted[:15]]
    means = [d["mean"] for _, d in ct_sorted[:15]]
    colors_b = ["#E41A1C" if cd14_by_ct[n]["is_myeloid"] else
                "#FF8C00" if n == "FDC" else "#377EB8" for n in names]
    short = [n.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+ myeloid")
              .replace("Dendritic cells", "DCs").replace("Low quality / Unassigned", "LQ/Unassigned")
              .replace("B cells", "B").replace("CD8 T cells", "CD8 T")
              .replace("Mixed / Border cells", "Mixed/Border")
              .replace("Endothelial", "Endothelial")
             for n in names]
    y_pos_b = range(len(names))
    ax_b.barh(list(y_pos_b), means, color=colors_b, edgecolor="white", height=0.6)
    ax_b.axvline(0, color="black", linewidth=0.5)
    ax_b.set_yticks(list(y_pos_b))
    ax_b.set_yticklabels(short, fontsize=9)
    ax_b.set_xlabel("Mean CD14 intensity (scaled)")
    ax_b.set_title("CD14 expression by cell type")
    ax_b.invert_yaxis()
    # Legend
    from matplotlib.patches import Patch
    ax_b.legend(handles=[
        Patch(color="#E41A1C", label="Myeloid"),
        Patch(color="#FF8C00", label="FDC"),
        Patch(color="#377EB8", label="Other"),
    ], fontsize=8, loc="lower right")

    # --- (c) CD14 signal decomposition ---
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")
    pct_mac = roi_df["pct_mac"].mean()
    pct_fdc = roi_df["pct_fdc"].mean()
    pct_other = roi_df["pct_other"].mean()
    wedges, texts, autotexts = ax_c.pie(
        [pct_mac, pct_fdc, pct_other],
        labels=["Myeloid cells", "FDC", "Other non-myeloid"],
        colors=["#E41A1C", "#FF8C00", "#377EB8"],
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax_c.set_title("CD14 positive signal source\n(mean across ROIs)")

    # --- (d) Non-mac CD14 vs mac CD14 scatter ---
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")
    ax_d.scatter(roi_df["cd14_mac"], roi_df["cd14_nonmac"],
                 c="#666666", alpha=0.5, s=25, edgecolors="white", linewidth=0.3)
    rho, p = stats.spearmanr(roi_df["cd14_mac"], roi_df["cd14_nonmac"])
    # Regression line
    m, b = np.polyfit(roi_df["cd14_mac"], roi_df["cd14_nonmac"], 1)
    x_range = np.linspace(roi_df["cd14_mac"].min(), roi_df["cd14_mac"].max(), 50)
    ax_d.plot(x_range, m * x_range + b, "r--", linewidth=1.5, alpha=0.7)
    ax_d.set_xlabel("Mean CD14 on myeloid cells")
    ax_d.set_ylabel("Mean CD14 on non-myeloid cells")
    ax_d.set_title("Microenvironment-level CD14 signal")
    ax_d.text(0.05, 0.95, f"Spearman ρ={rho:.2f}\np={p:.1e}",
              transform=ax_d.transAxes, va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- (e) Representative spatial scatter ---
    ax_e = fig.add_subplot(gs[2, 0])
    panel_label(ax_e, "e")
    if rep_data is not None:
        # Color scheme: myeloid = red, FDC = orange, B cells = blue, rest = gray
        for _, row in rep_data.iterrows():
            pass  # will use vectorized below

        ct_arr = rep_data["cell_type"].values
        is_myel = np.isin(ct_arr, MYELOID_TYPES)
        is_fdc = ct_arr == "FDC"
        is_b = np.array(["B cell" in c or c == "FDC" or "GC" in c for c in ct_arr])
        is_other = ~is_myel & ~is_fdc

        # Plot layers: other first, then B, then FDC, then myeloid
        ax_e.scatter(rep_data.loc[is_other, "x"], rep_data.loc[is_other, "y"],
                     c="#D3D3D3", s=0.3, alpha=0.3, rasterized=True)
        ax_e.scatter(rep_data.loc[is_fdc, "x"], rep_data.loc[is_fdc, "y"],
                     c="#FF8C00", s=2.0, alpha=0.7, label="FDC", rasterized=True)
        ax_e.scatter(rep_data.loc[is_myel, "x"], rep_data.loc[is_myel, "y"],
                     c="#E41A1C", s=2.5, alpha=0.8, label="Myeloid", rasterized=True)
        ax_e.set_aspect("equal")
        ax_e.invert_yaxis()
        ax_e.set_title(f"Spatial distribution — {rep_roi}")
        ax_e.legend(fontsize=9, loc="upper right", markerscale=4)
        ax_e.set_xlabel("x (μm)")
        ax_e.set_ylabel("y (μm)")
    else:
        ax_e.text(0.5, 0.5, "No spatial data", transform=ax_e.transAxes,
                  ha="center", va="center")

    # --- (f) Macrophage fraction vs CD14 ---
    ax_f = fig.add_subplot(gs[2, 1])
    panel_label(ax_f, "f")
    ax_f.scatter(roi_df["mac_frac"] * 100, roi_df["cd14_all"],
                 c="#E41A1C", alpha=0.5, s=25, edgecolors="white", linewidth=0.3)
    rho2, p2 = stats.spearmanr(roi_df["mac_frac"], roi_df["cd14_all"])
    m2, b2 = np.polyfit(roi_df["mac_frac"] * 100, roi_df["cd14_all"], 1)
    x2 = np.linspace(0, roi_df["mac_frac"].max() * 100, 50)
    ax_f.plot(x2, m2 * x2 + b2, "r--", linewidth=1.5, alpha=0.7)
    ax_f.set_xlabel("Myeloid cell fraction (%)")
    ax_f.set_ylabel("Mean CD14 intensity (all cells)")
    ax_f.set_title("Macrophage density vs CD14")
    ax_f.text(0.05, 0.95, f"Spearman ρ={rho2:.2f}\np={p2:.1e}",
              transform=ax_f.transAxes, va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    out = Path(output_dir) / "fig_myeloid_spatial.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-panel", default="output/all_TMA_S_global_v8.h5ad")
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    cd14_by_ct, roi_df, domain_enrich, rep_data, rep_roi = extract_data(
        args.s_panel, args.s_utag
    )
    make_figure(cd14_by_ct, roi_df, domain_enrich, rep_data, rep_roi,
                args.output_dir)


if __name__ == "__main__":
    main()
