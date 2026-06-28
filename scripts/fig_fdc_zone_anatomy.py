#!/usr/bin/env python3
"""FDC network zone vs follicular B cell domains: cross-panel registration.

Overlays T-panel follicular domains (registered) onto S-panel spatial maps
to show how FDC network zone relates to follicular B cell domains defined
independently on serial sections.

Usage:
    .venv/bin/python scripts/fig_fdc_zone_anatomy.py \
        --s-utag output/all_TMA_S_utag_ct_merged.h5ad \
        --t-utag output/all_TMA_T_utag_ct_merged.h5ad \
        --registration output/registered_concordance.csv \
        --output-dir output/hypotheses_v8
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def panel_label(ax, letter, x=-0.05, y=1.05):
    ax.text(
        x, y, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=14, va="top", ha="left",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# S-panel: FDC network zone
S_FDC_ZONE = "FDC network zone"

# T-panel: follicular domains
T_FOLLICULAR = [
    "GC core",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)",
    "Activated B / CXCR5hi zone",
    "B cell follicle (CD20hi/CXCR5hi)",
    "B cell zone",
]

# S-panel: B cell follicular domains
S_B_ZONES = {"B cell zone (BCL2+)", "B cell zone (PAX5+)"}

# Colors
COLOR_FDC_ZONE = "#E8734A"     # orange — S-panel FDC network zone
COLOR_T_FOLL = "#2166ac"       # blue — T-panel follicular domains (registered)
COLOR_S_FOLL = "#a6cee3"       # light blue — S-panel B cell zones
COLOR_OTHER = "#E8E8E8"        # light gray


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-utag", required=True)
    parser.add_argument("--t-utag", required=True)
    parser.add_argument("--registration", required=True,
                        help="registered_concordance.csv with shift_dy/shift_dx")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load registration shifts
    reg = pd.read_csv(args.registration)
    reg = reg.set_index("sample_id")
    print(f"Registration: {len(reg)} paired ROIs")

    # Load S-panel data
    print("Loading S-panel UTAG...")
    f = h5py.File(args.s_utag, "r")
    s_comps = load_array(f, "compartment_name")
    s_sids = load_array(f, "sample_id")
    s_cx = f["obs"]["centroid_x"][:]
    s_cy = f["obs"]["centroid_y"][:]
    f.close()

    s_tumor = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS
                         for s in s_sids])

    # Load T-panel data
    print("Loading T-panel UTAG...")
    f = h5py.File(args.t_utag, "r")
    t_comps = load_array(f, "compartment_name")
    t_sids = load_array(f, "sample_id")
    t_cx = f["obs"]["centroid_x"][:]
    t_cy = f["obs"]["centroid_y"][:]
    f.close()

    t_tumor = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS
                         for s in t_sids])

    # Find paired ROIs with good FDC zone + follicular representation
    paired_rois = set(s_sids[s_tumor]) & set(t_sids[t_tumor]) & set(reg.index)
    print(f"  Paired tumor ROIs: {len(paired_rois)}")

    roi_scores = []
    for roi in paired_rois:
        if reg.loc[roi, "pct_overlap"] < 90:
            continue
        # Skip extreme shifts (likely bad registration)
        dy, dx = reg.loc[roi, "shift_dy"], reg.loc[roi, "shift_dx"]
        if abs(dy) > 100 or abs(dx) > 100:
            continue
        sm = s_tumor & (s_sids == roi)
        tm = t_tumor & (t_sids == roi)
        n_fdc = (s_comps[sm] == S_FDC_ZONE).sum()
        n_t_foll = np.isin(t_comps[tm], T_FOLLICULAR).sum()
        n_s_foll = np.isin(s_comps[sm], list(S_B_ZONES)).sum()
        if n_fdc < 200 or n_t_foll < 200:
            continue
        balance = min(n_fdc, n_t_foll) / max(n_fdc, n_t_foll)
        roi_scores.append((roi, n_fdc, n_t_foll, n_s_foll, balance * n_fdc))

    roi_scores.sort(key=lambda x: x[4], reverse=True)

    # Pick 4 from different TMAs
    selected = []
    used_tmas = set()
    for roi, n_fdc, n_t_foll, n_s_foll, _ in roi_scores:
        tma = roi.split("_")[0]
        if tma not in used_tmas:
            selected.append(roi)
            used_tmas.add(tma)
        if len(selected) == 4:
            break
    for roi, *_ in roi_scores:
        if roi not in selected:
            selected.append(roi)
        if len(selected) == 4:
            break

    for roi in selected:
        sm = s_tumor & (s_sids == roi)
        tm = t_tumor & (t_sids == roi)
        n_fdc = (s_comps[sm] == S_FDC_ZONE).sum()
        n_t_foll = np.isin(t_comps[tm], T_FOLLICULAR).sum()
        n_s_foll = np.isin(s_comps[sm], list(S_B_ZONES)).sum()
        dy, dx = reg.loc[roi, "shift_dy"], reg.loc[roi, "shift_dx"]
        ovl = reg.loc[roi, "pct_overlap"]
        print(f"  {roi}: FDC zone={n_fdc:,}, T-foll={n_t_foll:,}, "
              f"S-B zones={n_s_foll:,}, shift=({dy:.1f},{dx:.1f}), overlap={ovl:.0f}%")

    # Create figure: 4 ROIs, each with 2 columns
    # Left: S-panel (FDC zone orange, B cell zones light blue, rest gray)
    # Right: same S-panel background + registered T-panel follicular domains (blue)
    fig, axes = plt.subplots(4, 2, figsize=(10, 18))

    for row, roi in enumerate(selected):
        dy = reg.loc[roi, "shift_dy"]
        dx = reg.loc[roi, "shift_dx"]

        # S-panel cells
        sm = s_tumor & (s_sids == roi)
        sx, sy = s_cx[sm], s_cy[sm]
        sc = s_comps[sm]
        is_fdc = sc == S_FDC_ZONE
        is_s_foll = np.isin(sc, list(S_B_ZONES))
        is_s_other = ~is_fdc & ~is_s_foll

        # T-panel cells (registered → shift into S-panel space)
        tm = t_tumor & (t_sids == roi)
        tx_reg = t_cx[tm] + dx
        ty_reg = t_cy[tm] + dy
        tc = t_comps[tm]
        is_t_foll = np.isin(tc, T_FOLLICULAR)

        # --- Left panel: S-panel compartments only ---
        ax = axes[row, 0]
        ax.scatter(sx[is_s_other], sy[is_s_other], c=COLOR_OTHER, s=1.5,
                   alpha=0.3, edgecolors="none", rasterized=True, zorder=1)
        ax.scatter(sx[is_s_foll], sy[is_s_foll], c=COLOR_S_FOLL, s=2.5,
                   alpha=0.6, edgecolors="none", rasterized=True, zorder=2)
        ax.scatter(sx[is_fdc], sy[is_fdc], c=COLOR_FDC_ZONE, s=2.5,
                   alpha=0.7, edgecolors="none", rasterized=True, zorder=3)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{roi} — S-panel", fontsize=10, fontweight="bold")
        panel_label(ax, chr(ord("a") + row * 2))

        # --- Right panel: S-panel FDC zone + registered T-panel follicular ---
        ax = axes[row, 1]
        # S-panel background (all gray)
        ax.scatter(sx, sy, c=COLOR_OTHER, s=1.5,
                   alpha=0.2, edgecolors="none", rasterized=True, zorder=1)
        # S-panel FDC zone (orange)
        ax.scatter(sx[is_fdc], sy[is_fdc], c=COLOR_FDC_ZONE, s=2.5,
                   alpha=0.7, edgecolors="none", rasterized=True, zorder=3)
        # T-panel follicular domains (blue, registered)
        ax.scatter(tx_reg[is_t_foll], ty_reg[is_t_foll], c=COLOR_T_FOLL, s=2.5,
                   alpha=0.5, edgecolors="none", rasterized=True, zorder=2)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{roi} — S FDC zone + registered T follicular",
                     fontsize=9, fontweight="bold")
        panel_label(ax, chr(ord("a") + row * 2 + 1))

    # Legend
    handles = [
        Patch(facecolor=COLOR_FDC_ZONE, label="FDC network zone (S-panel)"),
        Patch(facecolor=COLOR_S_FOLL, label="B cell zones (S-panel)"),
        Patch(facecolor=COLOR_T_FOLL, label="Follicular domains (T-panel, registered)"),
        Patch(facecolor=COLOR_OTHER, alpha=0.4, label="Other"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("FDC Network Zone (S-panel) vs Follicular Domains (T-panel, registered)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])

    out = outdir / "fig_fdc_zone_anatomy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return str(out)


if __name__ == "__main__":
    out = main()
    import subprocess
    subprocess.run(["open", "-a", "Preview", out])
