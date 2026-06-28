#!/usr/bin/env python3
"""
Exploratory analyses: What do S100A9+ MDSC-like myeloid cells DO?

Five analyses to understand S100A9+ cell functional impact:
  Q1. Proximity-dependent T cell suppression: CD8 T near S100A9+ show higher exhaustion?
  Q2. Per-ROI correlation: S100A9+ fraction vs CD8 exhaustion rate (cross-panel)
  Q3. Co-occurrence patterns: which cell types scale with S100A9+ density?
  Q4. Compartment-specific phenotype: S100A9+ marker expression varies by compartment?
  Q5. Transformation niche: S100A9+ proximity to Ki-67+ proliferating cells

Usage:
    .venv/bin/python scripts/explore_s100a9_biology.py \
        --s-panel output/all_TMA_S_global_v8.h5ad \
        --s-utag output/all_TMA_S_utag_ct_merged.h5ad \
        --t-panel output/all_TMA_T_global_v8.h5ad \
        --output-dir output/hypotheses_v8
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree


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


FOLL_COMPARTMENTS = {
    "B cell zone (BCL2+)", "B cell zone (PAX5+)",
    "FDC network zone", "FDC / myeloid zone",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panel(path, panel_name):
    """Load a panel h5ad: X, markers, cell_types, sample_ids, cx, cy."""
    print(f"Loading {panel_name} from {path}...")
    f = h5py.File(path, "r")
    X = f["X"][:]
    markers = [v.decode() for v in f["var"]["_index"][:]]
    cell_types = load_array(f, "cell_type")
    sample_ids = load_array(f, "sample_id")
    cx = f["obs"]["centroid_x"][:]
    cy = f["obs"]["centroid_y"][:]
    f.close()

    marker_idx = {m: i for i, m in enumerate(markers)}
    # Exclude Biomax (no clinical) for consistency with fig script
    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS and not s.startswith("Biomax")
        for s in sample_ids
    ])
    return {
        "X": X[tumor_mask], "markers": markers, "marker_idx": marker_idx,
        "cell_types": cell_types[tumor_mask],
        "sample_ids": sample_ids[tumor_mask],
        "cx": cx[tumor_mask], "cy": cy[tumor_mask],
    }


def load_utag(path):
    """Load UTAG compartment labels."""
    print(f"Loading UTAG from {path}...")
    f = h5py.File(path, "r")
    comp_names = load_array(f, "compartment_name")
    sample_ids = load_array(f, "sample_id")
    cell_types = load_array(f, "cell_type")
    f.close()

    tumor_mask = np.array([
        is_tumor_core(s) and s not in EXCLUDE_ROIS and not s.startswith("Biomax")
        for s in sample_ids
    ])
    return {
        "comps": comp_names[tumor_mask],
        "cell_types": cell_types[tumor_mask],
        "sample_ids": sample_ids[tumor_mask],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Q1: Proximity-dependent T cell suppression
# ═══════════════════════════════════════════════════════════════════════════

def q1_proximity_suppression(s_data, t_data):
    """Do CD8 T cells near S100A9+ cells show higher exhaustion (TOX+PD-1+)?

    Uses cross-panel spatial proximity: for each T-panel CD8 T cell, find
    the distance to the nearest S100A9+ cell in the S-panel (serial section).
    Compare exhaustion markers (TOX, PD-1) between near (<30µm) and far (>50µm).
    """
    print("\n" + "="*70)
    print("Q1: PROXIMITY-DEPENDENT T CELL SUPPRESSION")
    print("="*70)

    s_ct = s_data["cell_types"]
    s_sid = s_data["sample_ids"]
    t_ct = t_data["cell_types"]
    t_sid = t_data["sample_ids"]
    t_X = t_data["X"]
    t_markers = t_data["markers"]

    tox_idx = t_markers.index("TOX")
    pd1_idx = t_markers.index("PD_1")

    # Find shared ROIs (serial sections share same sample_id prefix)
    s_rois = set(s_sid)
    t_rois = set(t_sid)
    shared = s_rois & t_rois
    print(f"  Shared ROIs: {len(shared)}")

    near_tox, far_tox = [], []
    near_pd1, far_pd1 = [], []
    near_exh, far_exh = [], []  # dual-positive: TOX>0.8 AND PD-1>0.5

    for roi in sorted(shared):
        s_mask = s_sid == roi
        t_mask = t_sid == roi

        # S100A9+ positions in S-panel
        s100_mask = s_mask & (s_ct == "Myeloid (S100A9+)")
        if s100_mask.sum() < 5:
            continue

        s100_xy = np.column_stack([s_data["cx"][s100_mask], s_data["cy"][s100_mask]])

        # CD8 T cell positions in T-panel
        cd8_mask = t_mask & (t_ct == "CD8 T cells")
        if cd8_mask.sum() < 10:
            continue

        cd8_xy = np.column_stack([t_data["cx"][cd8_mask], t_data["cy"][cd8_mask]])
        cd8_tox = t_X[cd8_mask, tox_idx]
        cd8_pd1 = t_X[cd8_mask, pd1_idx]
        cd8_exhausted = (cd8_tox > 0.8) & (cd8_pd1 > 0.5)

        # Distance from each CD8 T to nearest S100A9+
        tree = cKDTree(s100_xy)
        dists, _ = tree.query(cd8_xy, k=1)

        near = dists < 30  # within 30µm
        far = dists > 50   # beyond 50µm

        near_tox.extend(cd8_tox[near].tolist())
        far_tox.extend(cd8_tox[far].tolist())
        near_pd1.extend(cd8_pd1[near].tolist())
        far_pd1.extend(cd8_pd1[far].tolist())
        near_exh.extend(cd8_exhausted[near].tolist())
        far_exh.extend(cd8_exhausted[far].tolist())

    near_tox = np.array(near_tox)
    far_tox = np.array(far_tox)
    near_pd1 = np.array(near_pd1)
    far_pd1 = np.array(far_pd1)
    near_exh = np.array(near_exh, dtype=bool)
    far_exh = np.array(far_exh, dtype=bool)

    print(f"  CD8 T cells near S100A9+ (<30µm): {len(near_tox):,}")
    print(f"  CD8 T cells far from S100A9+ (>50µm): {len(far_tox):,}")

    if len(near_tox) > 20 and len(far_tox) > 20:
        # TOX
        u_tox, p_tox = stats.mannwhitneyu(near_tox, far_tox, alternative="two-sided")
        print(f"\n  TOX expression:")
        print(f"    Near:  mean={near_tox.mean():.3f}, median={np.median(near_tox):.3f}")
        print(f"    Far:   mean={far_tox.mean():.3f}, median={np.median(far_tox):.3f}")
        print(f"    Δ = {near_tox.mean() - far_tox.mean():+.3f}, P={p_tox:.2e}")

        # PD-1
        u_pd1, p_pd1 = stats.mannwhitneyu(near_pd1, far_pd1, alternative="two-sided")
        print(f"\n  PD-1 expression:")
        print(f"    Near:  mean={near_pd1.mean():.3f}, median={np.median(near_pd1):.3f}")
        print(f"    Far:   mean={far_pd1.mean():.3f}, median={np.median(far_pd1):.3f}")
        print(f"    Δ = {near_pd1.mean() - far_pd1.mean():+.3f}, P={p_pd1:.2e}")

        # Exhaustion rate
        near_rate = near_exh.mean()
        far_rate = far_exh.mean()
        # Fisher's exact for rates
        a = near_exh.sum()
        b = len(near_exh) - a
        c = far_exh.sum()
        d = len(far_exh) - c
        _, p_exh = stats.fisher_exact([[a, b], [c, d]])
        print(f"\n  Exhaustion rate (TOX>0.8 & PD-1>0.5):")
        print(f"    Near:  {100*near_rate:.1f}% ({a}/{len(near_exh)})")
        print(f"    Far:   {100*far_rate:.1f}% ({c}/{len(far_exh)})")
        print(f"    OR = {(a*d)/(b*c+1e-9):.2f}, P={p_exh:.2e}")
    else:
        print("  Too few cells for comparison")

    # Also compare to distance from M1 Mac (control)
    print(f"\n  --- Control: distance from M1 Mac ---")
    near_m1_exh, far_m1_exh = [], []
    for roi in sorted(shared):
        s_mask = s_sid == roi
        t_mask = t_sid == roi

        m1_mask = s_mask & (s_ct == "M1 Macrophages")
        if m1_mask.sum() < 5:
            continue

        m1_xy = np.column_stack([s_data["cx"][m1_mask], s_data["cy"][m1_mask]])

        cd8_mask = t_mask & (t_ct == "CD8 T cells")
        if cd8_mask.sum() < 10:
            continue

        cd8_xy = np.column_stack([t_data["cx"][cd8_mask], t_data["cy"][cd8_mask]])
        cd8_tox = t_X[cd8_mask, tox_idx]
        cd8_pd1 = t_X[cd8_mask, pd1_idx]
        cd8_exhausted = (cd8_tox > 0.8) & (cd8_pd1 > 0.5)

        tree = cKDTree(m1_xy)
        dists, _ = tree.query(cd8_xy, k=1)

        near_m1_exh.extend(cd8_exhausted[dists < 30].tolist())
        far_m1_exh.extend(cd8_exhausted[dists > 50].tolist())

    near_m1_exh = np.array(near_m1_exh, dtype=bool)
    far_m1_exh = np.array(far_m1_exh, dtype=bool)
    if len(near_m1_exh) > 20 and len(far_m1_exh) > 20:
        r1 = near_m1_exh.mean()
        r2 = far_m1_exh.mean()
        a2 = near_m1_exh.sum()
        b2 = len(near_m1_exh) - a2
        c2 = far_m1_exh.sum()
        d2 = len(far_m1_exh) - c2
        _, p_m1 = stats.fisher_exact([[a2, b2], [c2, d2]])
        print(f"  M1 Mac near (<30µm) exhaustion: {100*r1:.1f}% ({a2}/{len(near_m1_exh)})")
        print(f"  M1 Mac far (>50µm) exhaustion:  {100*r2:.1f}% ({c2}/{len(far_m1_exh)})")
        print(f"  OR = {(a2*d2)/(b2*c2+1e-9):.2f}, P={p_m1:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# Q2: Per-ROI S100A9+ fraction vs CD8 exhaustion rate
# ═══════════════════════════════════════════════════════════════════════════

def q2_roi_correlation(s_data, t_data):
    """Cross-panel correlation: S100A9+ fraction (S-panel) vs CD8 exhaustion (T-panel)."""
    print("\n" + "="*70)
    print("Q2: PER-ROI S100A9+ FRACTION vs CD8 EXHAUSTION")
    print("="*70)

    s_ct = s_data["cell_types"]
    s_sid = s_data["sample_ids"]
    t_ct = t_data["cell_types"]
    t_sid = t_data["sample_ids"]
    t_X = t_data["X"]
    t_markers = t_data["markers"]

    tox_idx = t_markers.index("TOX")
    pd1_idx = t_markers.index("PD_1")

    shared = set(s_sid) & set(t_sid)

    s100_fracs = []
    exh_rates = []
    cd8_fracs = []
    roi_names = []

    for roi in sorted(shared):
        s_mask = s_sid == roi
        t_mask = t_sid == roi

        n_s = s_mask.sum()
        n_t = t_mask.sum()
        if n_s < 100 or n_t < 100:
            continue

        # S100A9+ fraction in S-panel
        s100_n = (s_ct[s_mask] == "Myeloid (S100A9+)").sum()
        s100_frac = s100_n / n_s

        # CD8 T exhaustion in T-panel
        cd8_mask_roi = t_mask & (t_ct == "CD8 T cells")
        n_cd8 = cd8_mask_roi.sum()
        if n_cd8 < 5:
            continue

        cd8_tox = t_X[cd8_mask_roi, tox_idx]
        cd8_pd1 = t_X[cd8_mask_roi, pd1_idx]
        exh = ((cd8_tox > 0.8) & (cd8_pd1 > 0.5)).sum()
        exh_rate = exh / n_cd8
        cd8_frac = n_cd8 / n_t

        s100_fracs.append(s100_frac)
        exh_rates.append(exh_rate)
        cd8_fracs.append(cd8_frac)
        roi_names.append(roi)

    s100_fracs = np.array(s100_fracs)
    exh_rates = np.array(exh_rates)
    cd8_fracs = np.array(cd8_fracs)

    print(f"  ROIs with both panels: {len(s100_fracs)}")

    # Spearman correlations
    rho, p = stats.spearmanr(s100_fracs, exh_rates)
    print(f"\n  S100A9+ fraction vs CD8 exhaustion rate:")
    print(f"    rho = {rho:+.3f}, P = {p:.2e}")

    rho2, p2 = stats.spearmanr(s100_fracs, cd8_fracs)
    print(f"\n  S100A9+ fraction vs CD8 T fraction:")
    print(f"    rho = {rho2:+.3f}, P = {p2:.2e}")

    # Also check: S100A9+ vs M1 Mac fraction, M2 Mac fraction
    m1_fracs = []
    m2_fracs = []
    for roi in roi_names:
        s_mask = s_sid == roi
        n_s = s_mask.sum()
        m1_fracs.append((s_ct[s_mask] == "M1 Macrophages").sum() / n_s)
        m2_fracs.append((s_ct[s_mask] == "M2 Macrophages").sum() / n_s)

    m1_fracs = np.array(m1_fracs)
    m2_fracs = np.array(m2_fracs)
    rho3, p3 = stats.spearmanr(s100_fracs, m1_fracs)
    rho4, p4 = stats.spearmanr(s100_fracs, m2_fracs)
    print(f"\n  S100A9+ vs M1 Mac fraction: rho={rho3:+.3f}, P={p3:.2e}")
    print(f"  S100A9+ vs M2 Mac fraction: rho={rho4:+.3f}, P={p4:.2e}")

    # Mean exhaustion rate in S100A9-high vs S100A9-low ROIs
    med = np.median(s100_fracs[s100_fracs > 0])
    hi = s100_fracs > med
    lo = s100_fracs <= med
    if hi.sum() > 5 and lo.sum() > 5:
        print(f"\n  CD8 exhaustion rate in S100A9-high vs -low ROIs (median split at {med:.4f}):")
        print(f"    S100A9-high: {100*exh_rates[hi].mean():.1f}% (n={hi.sum()})")
        print(f"    S100A9-low:  {100*exh_rates[lo].mean():.1f}% (n={lo.sum()})")
        _, p_mw = stats.mannwhitneyu(exh_rates[hi], exh_rates[lo], alternative="two-sided")
        print(f"    P = {p_mw:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# Q3: Co-occurrence patterns
# ═══════════════════════════════════════════════════════════════════════════

def q3_cooccurrence(s_data, t_data):
    """Which cell types increase/decrease per ROI as S100A9+ density increases?"""
    print("\n" + "="*70)
    print("Q3: CO-OCCURRENCE PATTERNS (ROI-LEVEL ECOLOGY)")
    print("="*70)

    s_ct = s_data["cell_types"]
    s_sid = s_data["sample_ids"]
    t_ct = t_data["cell_types"]
    t_sid = t_data["sample_ids"]

    # S-panel cell types per ROI
    s_rois = sorted(set(s_sid))
    s_all_cts = sorted(set(s_ct))
    s_roi_data = {}
    for roi in s_rois:
        m = s_sid == roi
        n = m.sum()
        if n < 200:
            continue
        ct_counts = Counter(s_ct[m])
        s_roi_data[roi] = {ct: ct_counts.get(ct, 0) / n for ct in s_all_cts}

    # T-panel cell types per ROI
    t_rois = sorted(set(t_sid))
    t_all_cts = sorted(set(t_ct))
    t_roi_data = {}
    for roi in t_rois:
        m = t_sid == roi
        n = m.sum()
        if n < 200:
            continue
        ct_counts = Counter(t_ct[m])
        t_roi_data[roi] = {ct: ct_counts.get(ct, 0) / n for ct in t_all_cts}

    # S100A9+ fraction per ROI
    s100_per_roi = {roi: d.get("Myeloid (S100A9+)", 0) for roi, d in s_roi_data.items()}
    s100_arr = np.array([s100_per_roi[roi] for roi in s_roi_data])

    # S-panel correlations
    print(f"\n  S-panel ({len(s_roi_data)} ROIs):")
    print(f"  {'Cell type':40s} {'rho':>6s} {'P':>10s}")
    s_results = []
    for ct in s_all_cts:
        if ct == "Myeloid (S100A9+)" or ct == "Unassigned":
            continue
        vals = np.array([s_roi_data[roi].get(ct, 0) for roi in s_roi_data])
        rho, p = stats.spearmanr(s100_arr, vals)
        s_results.append((ct, rho, p))
    s_results.sort(key=lambda x: -abs(x[1]))
    for ct, rho, p in s_results:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {ct:38s} {rho:+6.3f} {p:10.2e} {sig}")

    # Cross-panel: S100A9+ (S-panel) vs T-panel cell types
    shared = set(s_roi_data.keys()) & set(t_roi_data.keys())
    if len(shared) > 20:
        print(f"\n  T-panel cross-correlation ({len(shared)} shared ROIs):")
        print(f"  {'Cell type':40s} {'rho':>6s} {'P':>10s}")
        s100_shared = np.array([s100_per_roi[roi] for roi in shared])
        t_results = []
        for ct in t_all_cts:
            if ct == "Unassigned":
                continue
            vals = np.array([t_roi_data[roi].get(ct, 0) for roi in shared])
            rho, p = stats.spearmanr(s100_shared, vals)
            t_results.append((ct, rho, p))
        t_results.sort(key=lambda x: -abs(x[1]))
        for ct, rho, p in t_results:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {ct:38s} {rho:+6.3f} {p:10.2e} {sig}")


# ═══════════════════════════════════════════════════════════════════════════
# Q4: Compartment-specific phenotype
# ═══════════════════════════════════════════════════════════════════════════

def q4_compartment_phenotype(s_data, utag_data):
    """Do S100A9+ cells express different markers in different compartments?"""
    print("\n" + "="*70)
    print("Q4: COMPARTMENT-SPECIFIC S100A9+ PHENOTYPE")
    print("="*70)

    s_ct = s_data["cell_types"]
    s_sid = s_data["sample_ids"]
    s_X = s_data["X"]
    markers = s_data["markers"]

    u_ct = utag_data["cell_types"]
    u_sid = utag_data["sample_ids"]
    u_comps = utag_data["comps"]

    # Build compartment lookup from UTAG
    # Need to match cells between S-panel and UTAG — same h5ad, just UTAG has compartments
    # Since UTAG uses same cell ordering, we can match by index if same filter was applied
    # Actually, S-panel global and S-panel UTAG may have different cells (UTAG adds clustering)
    # Safest: match by sample_id + position. But for speed, use UTAG data directly since
    # UTAG has cell_type + compartment_name

    # Find S100A9+ cells in UTAG data
    s100_utag = u_ct == "Myeloid (S100A9+)"
    print(f"  S100A9+ cells in UTAG data: {s100_utag.sum():,}")

    # Get unique compartments for S100A9+
    comps_s100 = u_comps[s100_utag]
    comp_counts = Counter(comps_s100)
    print(f"  Compartment distribution:")
    for comp, n in comp_counts.most_common(10):
        print(f"    {comp:45s} {n:6,} ({100*n/s100_utag.sum():.1f}%)")

    # For marker comparison, we need expression from S-panel global (has X matrix)
    # The S-panel global has all cells; UTAG has all cells too.
    # Both filtered to tumor. Let's use S-panel directly by matching.

    # Group S100A9+ cells by compartment using UTAG
    # The ordering may differ, so build a lookup: (sample_id, cell_type) → compartment
    # Since we can't match individual cells easily, use UTAG's own S100A9+ markers
    # by computing mean expression from S-panel global for cells matching each compartment

    # Actually: load S-UTAG h5ad directly for expression
    # But UTAG h5ad may not have the raw expression matrix (it uses one-hot features)
    # So we need to use S-panel global and match cell indices

    # Alternative approach: S-panel has cell_type and sample_id. UTAG has those plus compartment.
    # If they have the same cells in the same order, we can just use the UTAG compartment
    # directly with S-panel expression.

    # Check alignment
    if len(s_ct) == len(u_ct):
        match_rate = (s_ct == u_ct).mean()
        print(f"\n  Alignment check: {100*match_rate:.1f}% cell types match")
        if match_rate > 0.99:
            print("  ✓ Perfect alignment — using S-panel expression with UTAG compartments")
            aligned = True
        else:
            aligned = False
    else:
        print(f"\n  Length mismatch: S-panel {len(s_ct)}, UTAG {len(u_ct)}")
        aligned = False

    if not aligned:
        print("  Cannot align panels. Using UTAG data only (no expression).")
        return

    # S100A9+ cells with compartment labels and expression
    s100_mask = s_ct == "Myeloid (S100A9+)"
    s100_comps = u_comps[s100_mask]
    s100_X = s_X[s100_mask]

    # Key markers to compare
    key_markers = ["S100A9", "CD14", "CD11b", "VISTA", "HLA_DR", "IDO",
                   "CD68", "CD11c", "CD34", "Ki-67", "CD163", "CD206"]
    available = [(m, markers.index(m)) for m in key_markers if m in markers]

    # Group by compartment categories
    compartment_groups = {
        "T cell zone": {"T cell zone"},
        "B/T mixed": {"B/T mixed zone", "Other / myeloid zone"},
        "Follicular": {"B cell zone (BCL2+)", "B cell zone (PAX5+)",
                       "FDC network zone", "FDC / myeloid zone"},
        "Stromal": {"Stromal / CAF zone"},
    }

    print(f"\n  {'Marker':12s}", end="")
    for group in compartment_groups:
        print(f" {group:>12s}", end="")
    print(f" {'T vs Foll P':>12s}")

    for mk_name, mk_idx in available:
        print(f"  {mk_name:12s}", end="")
        group_vals = {}
        for group, comp_set in compartment_groups.items():
            mask = np.isin(s100_comps, list(comp_set))
            if mask.sum() > 5:
                vals = s100_X[mask, mk_idx]
                print(f" {vals.mean():12.2f}", end="")
                group_vals[group] = vals
            else:
                print(f" {'n/a':>12s}", end="")

        # Compare T cell zone vs Follicular
        if "T cell zone" in group_vals and "Follicular" in group_vals:
            _, p = stats.mannwhitneyu(group_vals["T cell zone"],
                                      group_vals["Follicular"],
                                      alternative="two-sided")
            print(f" {p:12.2e}", end="")
        else:
            print(f" {'n/a':>12s}", end="")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# Q5: Transformation niche — S100A9+ proximity to proliferating cells
# ═══════════════════════════════════════════════════════════════════════════

def q5_proliferation_niche(s_data):
    """S100A9+ spatial relationship to Ki-67+ proliferating cells."""
    print("\n" + "="*70)
    print("Q5: S100A9+ PROXIMITY TO PROLIFERATING CELLS")
    print("="*70)

    s_ct = s_data["cell_types"]
    s_sid = s_data["sample_ids"]
    s_X = s_data["X"]
    markers = s_data["markers"]
    cx = s_data["cx"]
    cy = s_data["cy"]

    ki67_idx = markers.index("Ki-67")

    # Ki-67+ threshold: use top quartile of all cells
    ki67_vals = s_X[:, ki67_idx]
    ki67_thresh = np.percentile(ki67_vals[ki67_vals > 0], 75)
    ki67_pos = ki67_vals > ki67_thresh
    print(f"  Ki-67 threshold (p75 of positive): {ki67_thresh:.2f}")
    print(f"  Ki-67+ cells: {ki67_pos.sum():,} ({100*ki67_pos.mean():.1f}%)")

    # Among S100A9+ cells, what fraction are Ki-67+?
    s100_mask = s_ct == "Myeloid (S100A9+)"
    s100_ki67 = ki67_pos[s100_mask]
    print(f"\n  S100A9+ Ki-67+: {s100_ki67.sum():,} / {s100_mask.sum():,} ({100*s100_ki67.mean():.1f}%)")

    # Compare: Ki-67+ rate by myeloid subtype
    for subtype in ["Myeloid (S100A9+)", "M1 Macrophages", "M2 Macrophages",
                     "Macrophages", "FDC"]:
        sub_mask = s_ct == subtype
        sub_ki67 = ki67_pos[sub_mask]
        if sub_mask.sum() > 0:
            print(f"    {subtype:25s}: {100*sub_ki67.mean():.1f}% Ki-67+ ({sub_ki67.sum()}/{sub_mask.sum()})")

    # Distance from S100A9+ to nearest Ki-67+ cell
    print(f"\n  Distance to nearest Ki-67+ cell (per ROI, sampled):")
    rois = sorted(set(s_sid))
    s100_to_ki67_dists = []
    m1_to_ki67_dists = []
    m2_to_ki67_dists = []

    for roi in rois:
        rmask = s_sid == roi
        r_ki67 = rmask & ki67_pos
        if r_ki67.sum() < 5:
            continue

        ki67_xy = np.column_stack([cx[r_ki67], cy[r_ki67]])
        tree = cKDTree(ki67_xy)

        # S100A9+ to Ki-67+
        r_s100 = rmask & s100_mask
        if r_s100.sum() > 0:
            s100_xy = np.column_stack([cx[r_s100], cy[r_s100]])
            dists, _ = tree.query(s100_xy, k=1)
            s100_to_ki67_dists.extend(dists.tolist())

        # M1 Mac to Ki-67+ (control)
        r_m1 = rmask & (s_ct == "M1 Macrophages")
        if r_m1.sum() > 0:
            m1_xy = np.column_stack([cx[r_m1], cy[r_m1]])
            dists, _ = tree.query(m1_xy, k=1)
            m1_to_ki67_dists.extend(dists.tolist())

        # M2 Mac to Ki-67+ (control)
        r_m2 = rmask & (s_ct == "M2 Macrophages")
        if r_m2.sum() > 0:
            m2_xy = np.column_stack([cx[r_m2], cy[r_m2]])
            dists, _ = tree.query(m2_xy, k=1)
            m2_to_ki67_dists.extend(dists.tolist())

    s100_d = np.array(s100_to_ki67_dists)
    m1_d = np.array(m1_to_ki67_dists)
    m2_d = np.array(m2_to_ki67_dists)

    print(f"    S100A9+ → Ki-67+:  median={np.median(s100_d):.1f}µm, mean={s100_d.mean():.1f}µm (n={len(s100_d):,})")
    print(f"    M1 Mac → Ki-67+:   median={np.median(m1_d):.1f}µm, mean={m1_d.mean():.1f}µm (n={len(m1_d):,})")
    print(f"    M2 Mac → Ki-67+:   median={np.median(m2_d):.1f}µm, mean={m2_d.mean():.1f}µm (n={len(m2_d):,})")

    # Test S100A9+ vs M1
    if len(s100_d) > 20 and len(m1_d) > 20:
        _, p = stats.mannwhitneyu(s100_d, m1_d, alternative="two-sided")
        print(f"    S100A9+ vs M1: P={p:.2e}")
    if len(s100_d) > 20 and len(m2_d) > 20:
        _, p = stats.mannwhitneyu(s100_d, m2_d, alternative="two-sided")
        print(f"    S100A9+ vs M2: P={p:.2e}")

    # Per-ROI: S100A9+ fraction vs Ki-67+ fraction
    print(f"\n  Per-ROI correlation: S100A9+ vs Ki-67+ fractions:")
    s100_fracs, ki67_fracs = [], []
    for roi in rois:
        rmask = s_sid == roi
        n = rmask.sum()
        if n < 200:
            continue
        s100_fracs.append((s_ct[rmask] == "Myeloid (S100A9+)").sum() / n)
        ki67_fracs.append(ki67_pos[rmask].sum() / n)

    s100_fracs = np.array(s100_fracs)
    ki67_fracs = np.array(ki67_fracs)
    rho, p = stats.spearmanr(s100_fracs, ki67_fracs)
    print(f"    rho = {rho:+.3f}, P = {p:.2e} ({len(s100_fracs)} ROIs)")

    # What cell types are Ki-67+ near S100A9+?
    print(f"\n  Cell types of Ki-67+ cells near S100A9+ (<30µm):")
    ki67_near_s100_cts = []
    for roi in rois[:50]:  # sample 50 ROIs for speed
        rmask = s_sid == roi
        r_s100 = rmask & s100_mask
        r_ki67 = rmask & ki67_pos
        if r_s100.sum() < 3 or r_ki67.sum() < 3:
            continue

        s100_xy = np.column_stack([cx[r_s100], cy[r_s100]])
        ki67_xy = np.column_stack([cx[r_ki67], cy[r_ki67]])
        ki67_ct = s_ct[r_ki67]

        tree = cKDTree(s100_xy)
        dists, _ = tree.query(ki67_xy, k=1)

        for i, d in enumerate(dists):
            if d < 30:
                ki67_near_s100_cts.append(ki67_ct[i])

    if len(ki67_near_s100_cts) > 10:
        total = len(ki67_near_s100_cts)
        print(f"    Total Ki-67+ near S100A9+: {total}")
        for ct, n in Counter(ki67_near_s100_cts).most_common(10):
            print(f"      {ct:35s} {n:5,} ({100*n/total:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s-panel", required=True)
    parser.add_argument("--s-utag", required=True)
    parser.add_argument("--t-panel", required=True)
    parser.add_argument("--output-dir", default="output/hypotheses_v8")
    args = parser.parse_args()

    s_data = load_panel(args.s_panel, "S-panel")
    t_data = load_panel(args.t_panel, "T-panel")
    utag_data = load_utag(args.s_utag)

    q1_proximity_suppression(s_data, t_data)
    q2_roi_correlation(s_data, t_data)
    q3_cooccurrence(s_data, t_data)
    q4_compartment_phenotype(s_data, utag_data)
    q5_proliferation_niche(s_data)

    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
