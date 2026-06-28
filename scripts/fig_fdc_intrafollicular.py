#!/usr/bin/env python3
"""Intrafollicular CD14+ FDC functional characterization.

Addresses: what do CD14-high FDCs do within follicular compartments?

Panels:
  (a) Follicular sub-compartment localization: where within follicles?
  (b) Intrafollicular marker profile: CD14-high vs CD14-low FDCs (follicular only)
  (c) Context comparison: same CD14-high FDC phenotype intrafollicular vs interfollicular?
  (d) Intrafollicular neighbors: who surrounds CD14-high FDCs within follicles?
  (e) B cell relationship: tumor B cell proximity by FDC CD14 status
  (f) Cross-panel: follicular CD14-high FDC density vs T-panel cell states

Usage:
    .venv/bin/python scripts/fig_fdc_intrafollicular.py \
        --s-utag output/all_TMA_S_utag_ct_merged.h5ad \
        --t-utag output/all_TMA_T_utag_ct_merged.h5ad \
        --output-dir output/hypotheses_v8
"""

import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.spatial import KDTree

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


def panel_label(ax, letter, x=-0.08, y=1.05):
    ax.text(x, y, f"$\\bf{{{letter}}}$",
            transform=ax.transAxes, fontsize=14, va="top", ha="left")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S_FOLLICULAR = [
    "B cell zone (BCL2+)", "B cell zone (PAX5+)",
    "FDC network zone", "FDC / myeloid zone",
    "B/T mixed zone",
]
S_INTERFOLLICULAR = [
    "T cell zone", "Stromal / CAF zone",
    "Other / myeloid zone",
]

# Markers grouped by function
FUNCTIONAL_MARKERS = {
    "Antigen\npresentation": ["HLA_DR", "HLA_Class_I", "CD11c"],
    "Immune\nsuppression":   ["VISTA", "IDO"],
    "FDC\nnetwork":          ["CD21", "PDPN"],
    "Chemokines":            ["CXCL13", "CXCL12", "CCL21"],
    "Myeloid":               ["CD68", "CD11b", "S100A9"],
    "Tumor /\nproliferation": ["BCL_2", "Ki-67", "PAX5"],
    "Stromal":               ["Vimentin", "CD146", "Fibronectin"],
}

MARKER_DISPLAY = {
    "HLA_Class_I": "HLA-I", "HLA_DR": "HLA-DR", "BCL_2": "BCL-2",
    "PD_L1": "PD-L1", "BCL_6": "BCL-6", "Ki-67": "Ki67",
    "CD11b": "CD11b", "S100A9": "S100A9",
}

SKIP_MARKERS = {"DNA1", "DNA2", "HistoneH3", "p_H3s28"}

# B cell types (tumor)
B_CELL_TYPES = [
    "B cells (CD20+)", "B cells (BCL2+)", "B cells (CXCR5+)",
    "B cells (PAX5+)", "B cells",
]

# T-panel cell types for cross-panel
T_CELL_TYPES_OF_INTEREST = [
    "CD8 T exhausted", "CD8 T pre-exhausted (TOX+)", "CD8 T cells",
    "CD4 T cells", "Tfh", "Treg",
]

CD8_ALL = ["CD8 T cells", "CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]
CD8_EXHAUSTED = ["CD8 T exhausted", "CD8 T pre-exhausted (TOX+)"]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_s(path):
    print("Loading S-panel UTAG...")
    f = h5py.File(path, "r")
    X = f["X"][:]
    markers = [v.decode() if isinstance(v, bytes) else str(v)
               for v in f["var"]["_index"][:]]
    ct = load_array(f, "cell_type")
    sid = load_array(f, "sample_id")
    comp = load_array(f, "compartment_name")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()
    midx = {m: i for i, m in enumerate(markers)}
    mask = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS
                     and not s.startswith("Biomax") for s in sid])
    print(f"  {mask.sum():,} tumor cells")
    return {"X": X[mask], "markers": markers, "midx": midx,
            "ct": ct[mask], "sid": sid[mask], "comp": comp[mask],
            "cx": cx[mask], "cy": cy[mask]}


def extract_t(path):
    print("Loading T-panel UTAG...")
    f = h5py.File(path, "r")
    ct = load_array(f, "cell_type")
    sid = load_array(f, "sample_id")
    comp = load_array(f, "compartment_name")
    f.close()
    mask = np.array([is_tumor_core(s) and s not in EXCLUDE_ROIS
                     and not s.startswith("Biomax") for s in sid])
    print(f"  {mask.sum():,} tumor cells")
    return {"ct": ct[mask], "sid": sid[mask], "comp": comp[mask]}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_all(s, t):
    """Run all analyses. Returns dict of results."""
    cd14_col = s["midx"]["CD14"]
    fdc = s["ct"] == "FDC"
    fdc_cd14 = s["X"][fdc, cd14_col]
    q25, q75 = np.percentile(fdc_cd14, [25, 75])
    fdc_comp = s["comp"][fdc]
    fdc_is_foll = np.isin(fdc_comp, S_FOLLICULAR)
    fdc_is_ifoll = np.isin(fdc_comp, S_INTERFOLLICULAR)

    print(f"FDCs: {fdc.sum():,} total, {fdc_is_foll.sum():,} follicular, "
          f"{fdc_is_ifoll.sum():,} interfollicular")
    print(f"CD14 Q25={q25:.3f}, Q75={q75:.3f}")

    hi = fdc_cd14 >= q75
    lo = fdc_cd14 <= q25

    results = {"q25": q25, "q75": q75}

    # ── (a) Follicular vs interfollicular localization ──
    print("\n--- (a) Follicular vs interfollicular localization ---")
    # Simple: what fraction of CD14-high vs CD14-low FDCs are in follicular domains?
    foll_frac_hi = fdc_is_foll[hi].sum() / hi.sum()
    foll_frac_lo = fdc_is_foll[lo].sum() / lo.sum()
    ifoll_frac_hi = fdc_is_ifoll[hi].sum() / hi.sum()
    ifoll_frac_lo = fdc_is_ifoll[lo].sum() / lo.sum()
    other_frac_hi = 1.0 - foll_frac_hi - ifoll_frac_hi
    other_frac_lo = 1.0 - foll_frac_lo - ifoll_frac_lo
    print(f"  CD14-high FDCs: {foll_frac_hi:.1%} follicular, {ifoll_frac_hi:.1%} interfollicular, {other_frac_hi:.1%} other")
    print(f"  CD14-low FDCs:  {foll_frac_lo:.1%} follicular, {ifoll_frac_lo:.1%} interfollicular, {other_frac_lo:.1%} other")
    # Chi-square test
    from scipy.stats import chi2_contingency
    cont = np.array([
        [fdc_is_foll[hi].sum(), (~fdc_is_foll)[hi].sum()],
        [fdc_is_foll[lo].sum(), (~fdc_is_foll)[lo].sum()],
    ])
    chi2, chi2_p, _, _ = chi2_contingency(cont)
    print(f"  Chi-square: {chi2:.1f}, p={chi2_p:.2e}")
    results["foll_localization"] = {
        "foll_frac_hi": foll_frac_hi, "foll_frac_lo": foll_frac_lo,
        "ifoll_frac_hi": ifoll_frac_hi, "ifoll_frac_lo": ifoll_frac_lo,
        "other_frac_hi": other_frac_hi, "other_frac_lo": other_frac_lo,
        "n_hi": int(hi.sum()), "n_lo": int(lo.sum()),
        "chi2": chi2, "chi2_p": chi2_p,
    }
    # Also keep sub-compartment data for reference
    all_comps = sorted(set(s["comp"]))
    comp_data = {}
    for c in all_comps:
        n_hi = (fdc_comp[hi] == c).sum()
        n_lo = (fdc_comp[lo] == c).sum()
        frac_hi = n_hi / hi.sum()
        frac_lo = n_lo / lo.sum()
        comp_data[c] = {"frac_hi": frac_hi, "frac_lo": frac_lo,
                        "n_hi": int(n_hi), "n_lo": int(n_lo)}
    results["comp_data"] = comp_data

    # ── (b) Intrafollicular marker profile ──
    print("\n--- (b) Intrafollicular marker profile ---")
    fdc_X = s["X"][fdc]
    foll_hi = fdc_is_foll & hi
    foll_lo = fdc_is_foll & lo
    print(f"  Follicular CD14-high: {foll_hi.sum():,}")
    print(f"  Follicular CD14-low:  {foll_lo.sum():,}")

    marker_foll = {}
    for m in s["markers"]:
        if m in SKIP_MARKERS or m == "CD14":
            continue
        mi = s["midx"][m]
        hi_vals = fdc_X[foll_hi, mi]
        lo_vals = fdc_X[foll_lo, mi]
        diff = float(hi_vals.mean() - lo_vals.mean())
        u, p = stats.mannwhitneyu(hi_vals, lo_vals, alternative="two-sided")
        marker_foll[m] = {"diff": diff, "hi": float(hi_vals.mean()),
                          "lo": float(lo_vals.mean()), "p": p}
    results["marker_foll"] = marker_foll

    # Top markers
    top = sorted(marker_foll.items(), key=lambda x: -abs(x[1]["diff"]))[:10]
    for m, v in top:
        disp = MARKER_DISPLAY.get(m, m)
        print(f"  {disp:12s} diff={v['diff']:+.3f} (hi={v['hi']:.3f}, lo={v['lo']:.3f}, p={v['p']:.1e})")

    # ── (c) Intrafollicular vs interfollicular phenotype ──
    print("\n--- (c) Context comparison ---")
    ifoll_hi = fdc_is_ifoll & hi
    ifoll_lo = fdc_is_ifoll & lo
    print(f"  Interfollicular CD14-high: {ifoll_hi.sum():,}")
    print(f"  Interfollicular CD14-low:  {ifoll_lo.sum():,}")

    marker_ifoll = {}
    for m in s["markers"]:
        if m in SKIP_MARKERS or m == "CD14":
            continue
        mi = s["midx"][m]
        hi_vals = fdc_X[ifoll_hi, mi]
        lo_vals = fdc_X[ifoll_lo, mi]
        if len(hi_vals) < 10 or len(lo_vals) < 10:
            continue
        diff = float(hi_vals.mean() - lo_vals.mean())
        marker_ifoll[m] = {"diff": diff, "hi": float(hi_vals.mean()),
                           "lo": float(lo_vals.mean())}
    results["marker_ifoll"] = marker_ifoll

    # Also: compare intrafollicular CD14-high vs interfollicular CD14-high
    marker_context = {}
    for m in s["markers"]:
        if m in SKIP_MARKERS or m == "CD14":
            continue
        mi = s["midx"][m]
        foll_vals = fdc_X[foll_hi, mi]
        ifoll_vals = fdc_X[ifoll_hi, mi]
        if len(foll_vals) < 10 or len(ifoll_vals) < 10:
            continue
        diff = float(foll_vals.mean() - ifoll_vals.mean())
        u, p = stats.mannwhitneyu(foll_vals, ifoll_vals, alternative="two-sided")
        marker_context[m] = {"diff": diff, "foll": float(foll_vals.mean()),
                             "ifoll": float(ifoll_vals.mean()), "p": p}
    results["marker_context"] = marker_context

    top_ctx = sorted(marker_context.items(), key=lambda x: -abs(x[1]["diff"]))[:8]
    print("  CD14-high FDC: intrafollicular vs interfollicular")
    for m, v in top_ctx:
        disp = MARKER_DISPLAY.get(m, m)
        print(f"    {disp:12s} foll={v['foll']:.3f} ifoll={v['ifoll']:.3f} "
              f"diff={v['diff']:+.3f} p={v['p']:.1e}")

    # ── (d) Intrafollicular neighbors ──
    print("\n--- (d) Intrafollicular neighbors ---")
    fdc_idx_global = np.where(fdc)[0]
    fdc_foll_idx = fdc_idx_global[fdc_is_foll]
    fdc_foll_cd14 = fdc_cd14[fdc_is_foll]
    fdc_foll_hi_mask = fdc_foll_cd14 >= q75
    fdc_foll_lo_mask = fdc_foll_cd14 <= q25

    # Sample ROIs with enough follicular FDCs
    roi_counts = Counter(s["sid"][fdc_foll_idx])
    big_rois = [r for r, c in roi_counts.items() if c >= 30]
    big_rois.sort(key=lambda r: -roi_counts[r])
    sel_rois = big_rois[:20]  # top 20 ROIs by follicular FDC count
    print(f"  Using {len(sel_rois)} ROIs with ≥30 follicular FDCs")

    nbr_hi = Counter()
    nbr_lo = Counter()
    n_hi_total = 0
    n_lo_total = 0
    K = 10

    # Also track FDC-FDC homotypic clustering (separate counters — with FDC neighbors)
    fdc_nbr_hi = Counter()  # CD14-high FDC's FDC-class neighbors
    fdc_nbr_lo = Counter()  # CD14-low FDC's FDC-class neighbors
    fdc_n_hi_total = 0
    fdc_n_lo_total = 0

    for roi in sel_rois:
        rmask = s["sid"] == roi
        roi_idx = np.where(rmask)[0]
        roi_cx = s["cx"][roi_idx]
        roi_cy = s["cy"][roi_idx]
        roi_ct = s["ct"][roi_idx]
        roi_comp = s["comp"][roi_idx]
        roi_cd14 = s["X"][roi_idx, cd14_col]
        roi_fdc = roi_ct == "FDC"
        roi_fdc_foll = roi_fdc & np.isin(roi_comp, S_FOLLICULAR)

        # Non-FDC cells as neighbor candidates (for panel e — niche recruitment)
        non_fdc = np.where(~roi_fdc)[0]
        if len(non_fdc) < 20:
            continue
        fdc_foll_local = np.where(roi_fdc_foll)[0]
        if len(fdc_foll_local) < 10:
            continue

        tree = KDTree(np.column_stack([roi_cx[non_fdc], roi_cy[non_fdc]]))
        fdc_coords = np.column_stack([roi_cx[fdc_foll_local], roi_cy[fdc_foll_local]])
        _, idxs = tree.query(fdc_coords, k=K)

        fdc_cd14_local = roi_cd14[fdc_foll_local]
        for j in range(len(fdc_foll_local)):
            nbr_cts = roi_ct[non_fdc[idxs[j]]]
            if fdc_cd14_local[j] >= q75:
                nbr_hi.update(nbr_cts)
                n_hi_total += K
            elif fdc_cd14_local[j] <= q25:
                nbr_lo.update(nbr_cts)
                n_lo_total += K

        # --- Homotypic FDC-FDC clustering (all cells as candidates, label FDCs by CD14) ---
        all_tree = KDTree(np.column_stack([roi_cx, roi_cy]))
        _, all_idxs = all_tree.query(fdc_coords, k=K + 1)  # K+1 to skip self
        fdc_pos_local = roi_fdc & (roi_cd14 >= q75)
        fdc_neg_local = roi_fdc & (roi_cd14 <= q25)
        for j in range(len(fdc_foll_local)):
            nbr_idx = all_idxs[j, 1:]  # skip self
            for idx in nbr_idx:
                if not roi_fdc[idx]:
                    label = "non-FDC"
                elif fdc_pos_local[idx]:
                    label = "CD14+ FDC"
                elif fdc_neg_local[idx]:
                    label = "CD14- FDC"
                else:
                    label = "FDC (intermediate)"
                if fdc_cd14_local[j] >= q75:
                    fdc_nbr_hi[label] += 1
                    fdc_n_hi_total += 1
                elif fdc_cd14_local[j] <= q25:
                    fdc_nbr_lo[label] += 1
                    fdc_n_lo_total += 1

    print(f"  CD14-high neighbors: {n_hi_total:,}")
    print(f"  CD14-low neighbors:  {n_lo_total:,}")

    nbr_types = sorted(set(nbr_hi.keys()) | set(nbr_lo.keys()))
    nbr_data = []
    for ct in nbr_types:
        hi_f = nbr_hi.get(ct, 0) / n_hi_total if n_hi_total > 0 else 0
        lo_f = nbr_lo.get(ct, 0) / n_lo_total if n_lo_total > 0 else 0
        nbr_data.append((ct, hi_f, lo_f, hi_f - lo_f))
    nbr_data.sort(key=lambda x: -abs(x[3]))

    # Print top differences
    for ct, hi_f, lo_f, diff in nbr_data[:10]:
        print(f"  {ct:35s} hi={hi_f:.4f} lo={lo_f:.4f} diff={diff:+.4f}")
    results["nbr_data"] = nbr_data

    # Homotypic FDC clustering summary (separate from panel e)
    fdc_homo = {}
    for label in ["CD14+ FDC", "CD14- FDC", "FDC (intermediate)", "non-FDC"]:
        hi_f = fdc_nbr_hi.get(label, 0) / fdc_n_hi_total if fdc_n_hi_total > 0 else 0
        lo_f = fdc_nbr_lo.get(label, 0) / fdc_n_lo_total if fdc_n_lo_total > 0 else 0
        fdc_homo[label] = {"hi": hi_f, "lo": lo_f, "diff": hi_f - lo_f}
    print("\n  Homotypic FDC clustering (all cells as neighbors, K=10):")
    for label, v in fdc_homo.items():
        print(f"    {label:20s} hi={v['hi']*100:5.1f}% lo={v['lo']*100:5.1f}% diff={v['diff']*100:+.1f}pp")
    results["fdc_homotypic"] = fdc_homo
    results["n_hi_nbr"] = n_hi_total
    results["n_lo_nbr"] = n_lo_total

    # ── (e) B cell proximity ──
    print("\n--- (e) B cell proximity ---")
    b_cell_mask = np.isin(s["ct"], B_CELL_TYPES)

    dist_hi_to_b = []
    dist_lo_to_b = []
    for roi in sel_rois:
        rmask = s["sid"] == roi
        roi_idx = np.where(rmask)[0]
        roi_cx = s["cx"][roi_idx]
        roi_cy = s["cy"][roi_idx]
        roi_ct = s["ct"][roi_idx]
        roi_comp = s["comp"][roi_idx]
        roi_cd14 = s["X"][roi_idx, cd14_col]

        roi_fdc = roi_ct == "FDC"
        roi_fdc_foll = roi_fdc & np.isin(roi_comp, S_FOLLICULAR)
        roi_b = np.isin(roi_ct, B_CELL_TYPES)

        fdc_local = np.where(roi_fdc_foll)[0]
        b_local = np.where(roi_b)[0]
        if len(fdc_local) < 5 or len(b_local) < 20:
            continue

        b_tree = KDTree(np.column_stack([roi_cx[b_local], roi_cy[b_local]]))
        fdc_coords = np.column_stack([roi_cx[fdc_local], roi_cy[fdc_local]])
        dists, _ = b_tree.query(fdc_coords, k=1)

        fdc_cd14_local = roi_cd14[fdc_local]
        for j, d in enumerate(dists):
            if fdc_cd14_local[j] >= q75:
                dist_hi_to_b.append(d)
            elif fdc_cd14_local[j] <= q25:
                dist_lo_to_b.append(d)

    dist_hi_to_b = np.array(dist_hi_to_b)
    dist_lo_to_b = np.array(dist_lo_to_b)
    u, p = stats.mannwhitneyu(dist_hi_to_b, dist_lo_to_b, alternative="two-sided")
    print(f"  Distance to nearest B cell:")
    print(f"    CD14-high FDC: median={np.median(dist_hi_to_b):.1f} µm (n={len(dist_hi_to_b):,})")
    print(f"    CD14-low FDC:  median={np.median(dist_lo_to_b):.1f} µm (n={len(dist_lo_to_b):,})")
    print(f"    Mann-Whitney P={p:.2e}")
    results["dist_hi_b"] = dist_hi_to_b
    results["dist_lo_b"] = dist_lo_to_b

    # ── (f) Cross-panel ──
    print("\n--- (f) Cross-panel: follicular CD14-high FDC density vs T cell states ---")
    # Per-ROI: fraction of follicular FDCs that are CD14-high
    roi_fdc_hi_frac = {}
    for roi in np.unique(s["sid"]):
        rmask = s["sid"] == roi
        roi_fdc_local = (s["ct"][rmask] == "FDC") & np.isin(s["comp"][rmask], S_FOLLICULAR)
        n_fdc_foll = roi_fdc_local.sum()
        if n_fdc_foll < 10:
            continue
        roi_cd14_vals = s["X"][rmask][roi_fdc_local, cd14_col]
        frac_hi = (roi_cd14_vals >= q75).sum() / n_fdc_foll
        roi_fdc_hi_frac[roi] = frac_hi

    # T-panel per-ROI cell type fractions
    t_roi_fracs = {}
    for ct in T_CELL_TYPES_OF_INTEREST:
        t_roi_fracs[ct] = {}
    for roi in np.unique(t["sid"]):
        rmask = t["sid"] == roi
        n = rmask.sum()
        if n < 500:
            continue
        roi_ct = t["ct"][rmask]
        for ct in T_CELL_TYPES_OF_INTEREST:
            t_roi_fracs[ct][roi] = (roi_ct == ct).sum() / n

    # Also compute CD8 exhaustion fraction
    t_roi_exh_frac = {}
    for roi in np.unique(t["sid"]):
        rmask = t["sid"] == roi
        roi_ct = t["ct"][rmask]
        n_cd8 = sum(1 for c in roi_ct if c in CD8_ALL)
        n_exh = sum(1 for c in roi_ct if c in CD8_EXHAUSTED)
        if n_cd8 >= 10:
            t_roi_exh_frac[roi] = n_exh / n_cd8

    common = sorted(set(roi_fdc_hi_frac.keys()) & set(t_roi_exh_frac.keys()))
    print(f"  {len(common)} paired ROIs")

    cross_results = {}
    if len(common) >= 20:
        fdc_arr = np.array([roi_fdc_hi_frac[r] for r in common])

        # CD8 exhaustion fraction
        exh_arr = np.array([t_roi_exh_frac[r] for r in common])
        rho, p = stats.spearmanr(fdc_arr, exh_arr)
        cross_results["exh_frac"] = {"rho": rho, "p": p, "x": fdc_arr, "y": exh_arr}
        print(f"  CD8 exhaustion frac: rho={rho:.3f}, p={p:.2e}")

        # Individual cell types
        for ct in T_CELL_TYPES_OF_INTEREST:
            ct_arr = np.array([t_roi_fracs[ct].get(r, 0) for r in common])
            rho, p = stats.spearmanr(fdc_arr, ct_arr)
            cross_results[ct] = {"rho": rho, "p": p}
            print(f"  {ct:35s} rho={rho:+.3f}, p={p:.2e}")

    results["cross"] = cross_results
    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results, output_dir):
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35,
                  left=0.06, right=0.97, top=0.92, bottom=0.07)

    # ── (a) Follicular vs interfollicular localization ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")

    fl = results["foll_localization"]
    categories = ["Follicular", "Interfollicular", "Other"]
    hi_vals = [fl["foll_frac_hi"] * 100, fl["ifoll_frac_hi"] * 100, fl["other_frac_hi"] * 100]
    lo_vals = [fl["foll_frac_lo"] * 100, fl["ifoll_frac_lo"] * 100, fl["other_frac_lo"] * 100]

    x_pos = np.arange(len(categories))
    bar_w = 0.35
    bars_hi = ax_a.bar(x_pos - bar_w/2, hi_vals, bar_w, color="#FFD700",
                       edgecolor="black", linewidth=0.5, label=f"CD14-high (n={fl['n_hi']:,})")
    bars_lo = ax_a.bar(x_pos + bar_w/2, lo_vals, bar_w, color="#1976D2",
                       edgecolor="black", linewidth=0.5, label=f"CD14-low (n={fl['n_lo']:,})")

    # Add value labels on bars
    for bar in bars_hi:
        h = bar.get_height()
        if h > 3:
            ax_a.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_lo:
        h = bar.get_height()
        if h > 3:
            ax_a.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(categories, fontsize=10)
    ax_a.set_ylabel("% of FDCs", fontsize=10)
    ax_a.legend(fontsize=8, loc="upper right")
    p_str = f"P={fl['chi2_p']:.1e}" if fl["chi2_p"] < 0.001 else f"P={fl['chi2_p']:.3f}"
    ax_a.set_title(f"FDC compartment localization\n(\u03c7\u00b2={fl['chi2']:.0f}, {p_str})",
                    fontsize=10)
    ax_a.set_ylim(0, max(hi_vals + lo_vals) * 1.15)

    # ── (b) Intrafollicular marker profile ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")

    mf = results["marker_foll"]
    # Organize by functional category
    cat_markers = []
    cat_labels = []
    cat_colors_map = {
        "Antigen\npresentation": "#4CAF50",
        "Immune\nsuppression": "#D32F2F",
        "FDC\nnetwork": "#FF9800",
        "Chemokines": "#9C27B0",
        "Myeloid": "#795548",
        "Tumor /\nproliferation": "#607D8B",
        "Stromal": "#00BCD4",
    }
    bar_colors = []
    for cat, mkrs in FUNCTIONAL_MARKERS.items():
        for m in mkrs:
            if m in mf:
                cat_markers.append(m)
                cat_labels.append(cat)
                bar_colors.append(cat_colors_map.get(cat, "#999"))

    diffs = [mf[m]["diff"] for m in cat_markers]
    pvals = [mf[m]["p"] for m in cat_markers]
    names = [MARKER_DISPLAY.get(m, m) for m in cat_markers]

    y_pos_b = np.arange(len(names))
    bars = ax_b.barh(y_pos_b, diffs, color=bar_colors, alpha=0.85,
                     edgecolor="black", linewidth=0.3)
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(names, fontsize=8)
    ax_b.axvline(0, color="black", lw=0.5)
    ax_b.set_xlabel("Mean difference (CD14-high − CD14-low)")
    ax_b.set_title("Intrafollicular FDC marker profile\n(follicular compartments only)", fontsize=10)
    ax_b.invert_yaxis()

    # Significance stars
    for i, (d, p) in enumerate(zip(diffs, pvals)):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = ""
        if star:
            x_off = 0.01 if d >= 0 else -0.01
            ha = "left" if d >= 0 else "right"
            ax_b.text(d + x_off, i, star, va="center", ha=ha, fontsize=8, color="red")

    # Category labels on right
    prev_cat = ""
    for i, cat in enumerate(cat_labels):
        if cat != prev_cat:
            ax_b.text(1.02, i, cat.replace("\n", " "), transform=ax_b.get_yaxis_transform(),
                      va="center", ha="left", fontsize=7, color="#666",
                      fontstyle="italic")
            prev_cat = cat

    # ── (c) Context: intrafollicular vs interfollicular CD14-high phenotype ──
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c")

    mc = results["marker_context"]
    # Show markers where context matters (|diff| > 0.05 or p < 0.01)
    ctx_markers = [(m, mc[m]) for m in mc
                   if abs(mc[m]["diff"]) > 0.03 or mc[m]["p"] < 0.01]
    ctx_markers.sort(key=lambda x: x[1]["diff"])

    if ctx_markers:
        c_names = [MARKER_DISPLAY.get(m, m) for m, _ in ctx_markers]
        c_diffs = [v["diff"] for _, v in ctx_markers]
        c_pvals = [v["p"] for _, v in ctx_markers]

        y_pos_c = np.arange(len(c_names))
        colors_c = ["#B22222" if d > 0 else "#1976D2" for d in c_diffs]
        ax_c.barh(y_pos_c, c_diffs, color=colors_c, alpha=0.8,
                  edgecolor="black", linewidth=0.3)
        ax_c.set_yticks(y_pos_c)
        ax_c.set_yticklabels(c_names, fontsize=8)
        ax_c.axvline(0, color="black", lw=0.5)
        ax_c.set_xlabel("Δ expression (intrafollicular − interfollicular)")
        for i, p in enumerate(c_pvals):
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if star:
                d = c_diffs[i]
                ax_c.text(d + (0.005 if d >= 0 else -0.005), i, star,
                          va="center", ha="left" if d >= 0 else "right",
                          fontsize=8, color="red")
    ax_c.set_title("CD14-high FDC phenotype:\nintrafollicular vs interfollicular", fontsize=10)
    ax_c.text(0.02, 0.98, "Red = higher in follicular\nBlue = higher in interfollicular",
              transform=ax_c.transAxes, fontsize=7, va="top", color="#666")

    # ── (d) Intrafollicular neighbors ──
    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, "d")

    nd = results["nbr_data"]
    # Show top 12 by absolute difference
    top_nbr = nd[:12]
    n_names = [ct[:30] for ct, _, _, _ in top_nbr]
    n_diffs = [diff * 100 for _, _, _, diff in top_nbr]  # as percentage points
    n_colors = ["#D32F2F" if d > 0 else "#1976D2" for d in n_diffs]

    y_pos_d = np.arange(len(n_names))
    ax_d.barh(y_pos_d, n_diffs, color=n_colors, alpha=0.8,
              edgecolor="black", linewidth=0.3)
    ax_d.set_yticks(y_pos_d)
    ax_d.set_yticklabels(n_names, fontsize=8)
    ax_d.axvline(0, color="black", lw=0.5)
    ax_d.set_xlabel("Δ neighbor fraction (pp)\n(CD14-high − CD14-low FDC)")
    ax_d.set_title(
        f"Intrafollicular FDC neighbors (k={10})\n"
        f"(n_hi={results['n_hi_nbr']:,}, n_lo={results['n_lo_nbr']:,})",
        fontsize=10)
    ax_d.invert_yaxis()
    ax_d.text(0.02, 0.98, "Red = enriched near CD14-high\nBlue = depleted",
              transform=ax_d.transAxes, fontsize=7, va="top", color="#666")

    # ── (e) B cell proximity ──
    ax_e = fig.add_subplot(gs[1, 1])
    panel_label(ax_e, "e")

    dh = results["dist_hi_b"]
    dl = results["dist_lo_b"]
    clip = 100
    ax_e.hist(np.clip(dh, 0, clip), bins=40, range=(0, clip), alpha=0.6,
              color="#FFD700", density=True, label=f"CD14-high (n={len(dh):,})")
    ax_e.hist(np.clip(dl, 0, clip), bins=40, range=(0, clip), alpha=0.6,
              color="#1976D2", density=True, label=f"CD14-low (n={len(dl):,})")
    ax_e.axvline(np.median(dh), color="#B8860B", ls="--", lw=1.5)
    ax_e.axvline(np.median(dl), color="#0D47A1", ls="--", lw=1.5)
    u, p = stats.mannwhitneyu(dh, dl)
    ax_e.set_xlabel("Distance to nearest B cell (µm)")
    ax_e.set_ylabel("Density")
    ax_e.set_title(
        f"Intrafollicular FDC → nearest B cell\n"
        f"(hi median={np.median(dh):.1f}, lo={np.median(dl):.1f} µm, P={p:.1e})",
        fontsize=10)
    ax_e.legend(fontsize=8)

    # ── (f) Cross-panel ──
    ax_f = fig.add_subplot(gs[1, 2])
    panel_label(ax_f, "f")

    cr = results["cross"]
    if cr:
        ct_names = []
        rhos = []
        sig = []
        for ct in T_CELL_TYPES_OF_INTEREST:
            if ct in cr:
                ct_names.append(ct[:25])
                rhos.append(cr[ct]["rho"])
                sig.append(cr[ct]["p"] < 0.05)
        if "exh_frac" in cr:
            ct_names.append("CD8 exhaustion frac")
            rhos.append(cr["exh_frac"]["rho"])
            sig.append(cr["exh_frac"]["p"] < 0.05)

        y_pos_f = np.arange(len(ct_names))
        colors_f = ["#D32F2F" if r > 0 else "#1976D2" for r in rhos]
        alphas = [0.9 if s else 0.4 for s in sig]
        for i in range(len(ct_names)):
            ax_f.barh(i, rhos[i], color=colors_f[i], alpha=alphas[i],
                      edgecolor="black", linewidth=0.3)
            if sig[i]:
                ax_f.text(rhos[i] + 0.01 * (1 if rhos[i] >= 0 else -1), i,
                          "*", va="center", fontsize=10, color="red")
        ax_f.set_yticks(y_pos_f)
        ax_f.set_yticklabels(ct_names, fontsize=8)
        ax_f.axvline(0, color="black", lw=0.5)
        ax_f.set_xlabel("Spearman ρ with follicular CD14-high FDC fraction")
        ax_f.set_title("Cross-panel: follicular CD14-high\nFDC density vs T cell states",
                        fontsize=10)
        ax_f.invert_yaxis()
        ax_f.text(0.02, 0.02, "* p < 0.05\nfaded = n.s.",
                  transform=ax_f.transAxes, fontsize=7, va="bottom", color="#666")

    fig.suptitle(
        "What do intrafollicular CD14+ FDCs do?",
        fontsize=15, fontweight="bold", y=0.98)

    out = Path(output_dir) / "fig_fdc_intrafollicular.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s-utag", default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--t-utag", default="output/all_TMA_T_utag_ct_merged.h5ad")
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    s = extract_s(args.s_utag)
    t = extract_t(args.t_utag)
    results = analyze_all(s, t)
    make_figure(results, args.output_dir)


if __name__ == "__main__":
    main()
