"""Spatial features of macrophages in S-panel IMC data.

Analyses:
1. Domain enrichment (follicular vs interfollicular) per macrophage subtype
2. Spatial clustering via nearest-neighbor distances
3. Per-ROI macrophage density heterogeneity + correlation with s_CD14
4. Macrophage marker co-expression by subtype
"""

import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Helpers (from survival_analysis.py)
# ---------------------------------------------------------------------------

def load_array(f, key):
    ds = f["obs"][key]
    if isinstance(ds, h5py.Group) and "categories" in ds:
        cats = ds["categories"][:]
        codes = ds["codes"][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return cats_str[codes]
    vals = ds[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


def is_tumor_core(sid):
    s = sid.lower()
    if "_ton_" in s or "_adr_" in s:
        return False
    for tissue in ["tonsil", "prostate", "kidney", "spleen", "adrenal"]:
        if tissue in s:
            return False
    if sid == "Biomax_ROI_006":
        return False
    return True


MAC_TYPES = ["M1 Macrophages", "M2 Macrophages", "Macrophages",
             "Myeloid (S100A9+)", "Dendritic cells"]
B_TYPES = ["B cells", "B cells (BCL2+)", "B cells (PAX5+)"]
T_TYPES = ["CD4 T cells", "CD8 T cells"]
KEY_MARKERS = ["CD14", "CD68", "CD163", "S100A9", "CD206", "CD11c", "HLA_Class_I"]

# ---------------------------------------------------------------------------
# Analysis 1: Domain enrichment
# ---------------------------------------------------------------------------

def domain_enrichment(utag_path):
    print("=" * 60)
    print("ANALYSIS 1: Domain enrichment (follicular vs interfollicular)")
    print("=" * 60)
    with h5py.File(utag_path, "r") as f:
        sids = load_array(f, "sample_id")
        ctypes = load_array(f, "cell_type")
        utag = load_array(f, "UTAG Label_leiden_0.015")

    tumor_mask = np.array([is_tumor_core(s) for s in sids])
    sids, ctypes, utag = sids[tumor_mask], ctypes[tumor_mask], utag[tumor_mask]
    print(f"  Tumor cells: {tumor_mask.sum():,}")

    # Classify UTAG domains as follicular (>50% B cells)
    is_b = np.isin(ctypes, B_TYPES)
    utag_labels = np.unique(utag)
    foll_domains = set()
    print("\n  Domain classification (>50% B cells = follicular):")
    for u in sorted(utag_labels, key=int):
        mask = utag == u
        n = mask.sum()
        b_frac = is_b[mask].sum() / max(n, 1)
        label = "FOLL" if b_frac > 0.5 else "IFOLL"
        if b_frac > 0.5:
            foll_domains.add(u)
        print(f"    Domain {u:>2s}: {n:>8,} cells, B frac={b_frac:.3f} -> {label}")

    is_foll = np.isin(utag, list(foll_domains))
    n_foll, n_ifoll = is_foll.sum(), (~is_foll).sum()
    print(f"\n  Follicular: {n_foll:,} | Interfollicular: {n_ifoll:,}")

    # Enrichment per macrophage subtype
    print(f"\n  {'Subtype':25s} {'Foll%':>7s} {'IFoll%':>7s} {'Fold':>6s} {'Chi2':>10s} {'p':>12s}")
    print("  " + "-" * 70)
    for mt in MAC_TYPES:
        is_mt = ctypes == mt
        mt_foll = (is_mt & is_foll).sum()
        mt_ifoll = (is_mt & ~is_foll).sum()
        frac_foll = mt_foll / max(n_foll, 1)
        frac_ifoll = mt_ifoll / max(n_ifoll, 1)
        fold = frac_foll / max(frac_ifoll, 1e-10)
        # Chi-squared: observed vs expected under uniform distribution
        total_mt = mt_foll + mt_ifoll
        exp_foll = total_mt * n_foll / (n_foll + n_ifoll)
        exp_ifoll = total_mt * n_ifoll / (n_foll + n_ifoll)
        if exp_foll > 0 and exp_ifoll > 0:
            chi2, p = stats.chisquare([mt_foll, mt_ifoll], [exp_foll, exp_ifoll])
        else:
            chi2, p = 0, 1
        print(f"  {mt:25s} {frac_foll*100:6.2f}% {frac_ifoll*100:6.2f}% {fold:6.2f}x "
              f"{chi2:10.1f} {p:12.2e}")

# ---------------------------------------------------------------------------
# Analysis 2: Spatial clustering (NN distances)
# ---------------------------------------------------------------------------

def spatial_clustering(h5_path):
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Macrophage spatial clustering (NN distances)")
    print("=" * 60)
    with h5py.File(h5_path, "r") as f:
        sids = load_array(f, "sample_id")
        ctypes = load_array(f, "cell_type")
        cx = np.array(f["obs"]["centroid_x"][:], dtype=float)
        cy = np.array(f["obs"]["centroid_y"][:], dtype=float)

    tumor_mask = np.array([is_tumor_core(s) for s in sids])
    rois = np.unique(sids[tumor_mask])
    # Pick 8 representative ROIs spread across TMAs
    rng = np.random.default_rng(42)
    if len(rois) > 8:
        rois = rng.choice(rois, 8, replace=False)

    print(f"\n  Analyzing {len(rois)} ROIs. Median NN distance (um) to nearest:")
    print(f"  {'ROI':30s} {'nMac':>5s} {'Mac->Mac':>9s} {'Mac->B':>9s} {'Mac->T':>9s} {'B->B':>9s}")
    print("  " + "-" * 75)
    rows = []
    for roi in sorted(rois):
        mask = (sids == roi) & tumor_mask
        ct = ctypes[mask]
        x, y = cx[mask], cy[mask]
        is_mac = np.isin(ct, MAC_TYPES)
        is_b = np.isin(ct, B_TYPES)
        is_t = np.isin(ct, T_TYPES)
        n_mac = is_mac.sum()
        if n_mac < 10:
            continue
        coords_mac = np.column_stack([x[is_mac], y[is_mac]])
        coords_b = np.column_stack([x[is_b], y[is_b]]) if is_b.sum() > 10 else None
        coords_t = np.column_stack([x[is_t], y[is_t]]) if is_t.sum() > 10 else None
        bb_coords = np.column_stack([x[is_b], y[is_b]]) if is_b.sum() > 10 else None

        # Mac-Mac NN (k=2 to skip self)
        tree_mac = KDTree(coords_mac)
        dd_mm, _ = tree_mac.query(coords_mac, k=2)
        med_mm = np.median(dd_mm[:, 1])

        med_mb = med_mt = med_bb = np.nan
        if coords_b is not None:
            tree_b = KDTree(coords_b)
            dd_mb, _ = tree_b.query(coords_mac, k=1)
            med_mb = np.median(dd_mb)
            dd_bb, _ = tree_b.query(coords_b, k=2)
            med_bb = np.median(dd_bb[:, 1])
        if coords_t is not None:
            tree_t = KDTree(coords_t)
            dd_mt, _ = tree_t.query(coords_mac, k=1)
            med_mt = np.median(dd_mt)

        print(f"  {roi:30s} {n_mac:5d} {med_mm:9.1f} {med_mb:9.1f} {med_mt:9.1f} {med_bb:9.1f}")
        rows.append({"roi": roi, "n_mac": n_mac, "mac_mac": med_mm,
                      "mac_b": med_mb, "mac_t": med_mt, "b_b": med_bb})

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n  Summary across ROIs:")
        for col in ["mac_mac", "mac_b", "mac_t", "b_b"]:
            vals = df[col].dropna()
            print(f"    {col:10s}: mean={vals.mean():.1f}, median={vals.median():.1f}, std={vals.std():.1f}")
        # Are macrophages more dispersed than B cells?
        paired = df[["mac_mac", "b_b"]].dropna()
        if len(paired) >= 5:
            _, p = stats.wilcoxon(paired["mac_mac"], paired["b_b"])
            print(f"\n    Mac-Mac vs B-B NN distance (Wilcoxon): p={p:.4f}")
            print(f"    Mac more {'dispersed' if paired['mac_mac'].median() > paired['b_b'].median() else 'clustered'} than B cells")

# ---------------------------------------------------------------------------
# Analysis 3: Per-ROI macrophage density heterogeneity
# ---------------------------------------------------------------------------

def roi_density(h5_path, cov_csv):
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Per-ROI macrophage density heterogeneity")
    print("=" * 60)
    with h5py.File(h5_path, "r") as f:
        sids = load_array(f, "sample_id")
        ctypes = load_array(f, "cell_type")

    tumor_mask = np.array([is_tumor_core(s) for s in sids])
    sids_t, ctypes_t = sids[tumor_mask], ctypes[tumor_mask]
    rois = np.unique(sids_t)

    rows = []
    for roi in rois:
        mask = sids_t == roi
        ct = ctypes_t[mask]
        n = mask.sum()
        row = {"roi": roi, "n_cells": n}
        for mt in MAC_TYPES:
            row[mt] = (ct == mt).sum() / n
        row["all_mac"] = sum(row[mt] for mt in MAC_TYPES[:3])
        row["all_myeloid"] = sum(row[mt] for mt in MAC_TYPES)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  {len(df)} tumor ROIs")
    print(f"\n  Macrophage subtype fractions (% of cells):")
    print(f"  {'Subtype':25s} {'Mean%':>7s} {'Med%':>7s} {'Std%':>7s} {'Min%':>7s} {'Max%':>7s}")
    print("  " + "-" * 60)
    for mt in MAC_TYPES + ["all_mac", "all_myeloid"]:
        vals = df[mt] * 100
        print(f"  {mt:25s} {vals.mean():6.2f}% {vals.median():6.2f}% {vals.std():6.2f}% "
              f"{vals.min():6.2f}% {vals.max():6.2f}%")

    # Coefficient of variation
    print(f"\n  CV (coefficient of variation) across ROIs:")
    for mt in MAC_TYPES:
        vals = df[mt]
        cv = vals.std() / max(vals.mean(), 1e-10)
        print(f"    {mt:25s}: CV={cv:.2f}")

    # Correlation with s_CD14
    cov = pd.read_csv(cov_csv)
    if "s_CD14" in cov.columns and "slide_ID" in cov.columns:
        # Need to map ROI -> slide_ID for merge; use sample_id from covariates
        # covariates already have s_CD14 per patient, we need per-ROI s_CD14
        # Actually the covariates CSV is patient-level. Extract from h5ad directly.
        print("\n  Extracting per-ROI mean CD14 from h5ad...")
        with h5py.File(h5_path, "r") as f:
            var_key = "_index" if "_index" in f["var"] else "index"
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["var"][var_key][:]]
            if "CD14" in names:
                cd14_idx = names.index("CD14")
                X = f["X"]
                roi_cd14 = {}
                for roi in rois:
                    idx = np.where((sids == roi) & np.array([is_tumor_core(s) for s in sids]))[0]
                    if len(idx) > 0:
                        vals = X[idx[0]:idx[-1]+1, cd14_idx]
                        roi_cd14[roi] = float(np.mean(vals))
                df["mean_CD14"] = df["roi"].map(roi_cd14)

                print(f"\n  Spearman correlation: macrophage fraction vs mean CD14 intensity:")
                for mt in MAC_TYPES + ["all_mac", "all_myeloid"]:
                    sub = df[[mt, "mean_CD14"]].dropna()
                    if len(sub) >= 10:
                        rho, p = stats.spearmanr(sub[mt], sub["mean_CD14"])
                        sig = " *" if p < 0.05 else ""
                        print(f"    {mt:25s}: rho={rho:+.3f}, p={p:.4f}{sig}")

    # Top / bottom 5 ROIs by total macrophage fraction
    print(f"\n  Top 5 macrophage-rich ROIs:")
    top = df.nlargest(5, "all_myeloid")
    for _, r in top.iterrows():
        print(f"    {r['roi']:30s}: {r['all_myeloid']*100:.1f}% myeloid, {r['n_cells']:,} cells")
    print(f"\n  Bottom 5 macrophage-poor ROIs:")
    bot = df.nsmallest(5, "all_myeloid")
    for _, r in bot.iterrows():
        print(f"    {r['roi']:30s}: {r['all_myeloid']*100:.1f}% myeloid, {r['n_cells']:,} cells")

# ---------------------------------------------------------------------------
# Analysis 4: Macrophage marker co-expression
# ---------------------------------------------------------------------------

def marker_coexpression(h5_path):
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Macrophage marker co-expression by subtype")
    print("=" * 60)
    with h5py.File(h5_path, "r") as f:
        ctypes = load_array(f, "cell_type")
        sids = load_array(f, "sample_id")
        var_key = "_index" if "_index" in f["var"] else "index"
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["var"][var_key][:]]

        # Map marker names to indices (handle underscores in h5ad names)
        name_map = {}
        for km in KEY_MARKERS:
            # Try exact, then with underscores
            for n in names:
                if n.replace("_", "") == km.replace("_", ""):
                    name_map[km] = names.index(n)
                    break
        avail = [m for m in KEY_MARKERS if m in name_map]
        print(f"  Available markers: {avail}")
        midx = [name_map[m] for m in avail]

        tumor_mask = np.array([is_tumor_core(s) for s in sids])
        X = f["X"]

        print(f"\n  Mean expression (arcsinh-transformed, scaled) per macrophage subtype:")
        print(f"  {'Subtype':25s} " + " ".join(f"{m:>12s}" for m in avail) + f" {'n':>8s}")
        print("  " + "-" * (25 + 13 * len(avail) + 9))

        for mt in MAC_TYPES:
            mask = (ctypes == mt) & tumor_mask
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            # Sample up to 50k for speed
            if len(idx) > 50000:
                idx = np.random.default_rng(42).choice(idx, 50000, replace=False)
                idx.sort()
            vals = X[idx[0]:idx[-1]+1, :]
            # Only keep rows that are actually in our mask
            local_mask = mask[idx[0]:idx[-1]+1]
            vals = vals[local_mask]
            means = vals[:, midx].mean(axis=0)
            line = f"  {mt:25s} " + " ".join(f"{float(means[i]):12.3f}" for i in range(len(avail)))
            line += f" {len(idx):8,}"
            print(line)

        # Also show "Other" and B cells for reference
        for ref_ct in ["B cells", "CD4 T cells", "Other"]:
            mask = (ctypes == ref_ct) & tumor_mask
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            if len(idx) > 50000:
                idx = np.random.default_rng(42).choice(idx, 50000, replace=False)
                idx.sort()
            vals = X[idx[0]:idx[-1]+1, :]
            local_mask = mask[idx[0]:idx[-1]+1]
            vals = vals[local_mask]
            means = vals[:, midx].mean(axis=0)
            line = f"  {ref_ct + ' (ref)':25s} " + " ".join(f"{float(means[i]):12.3f}" for i in range(len(avail)))
            line += f" {len(idx):8,}"
            print(line)

        # Pairwise comparison: M1 vs M2
        print(f"\n  M1 vs M2 marker differences (Mann-Whitney, sampled):")
        m1_mask = (ctypes == "M1 Macrophages") & tumor_mask
        m2_mask = (ctypes == "M2 Macrophages") & tumor_mask
        m1_idx = np.where(m1_mask)[0]
        m2_idx = np.where(m2_mask)[0]
        if len(m1_idx) > 20000:
            m1_idx = np.random.default_rng(42).choice(m1_idx, 20000, replace=False)
            m1_idx.sort()
        if len(m2_idx) > 20000:
            m2_idx = np.random.default_rng(42).choice(m2_idx, 20000, replace=False)
            m2_idx.sort()
        m1_vals = X[m1_idx[0]:m1_idx[-1]+1, :]
        m1_local = m1_mask[m1_idx[0]:m1_idx[-1]+1]
        m1_vals = m1_vals[m1_local]
        m2_vals = X[m2_idx[0]:m2_idx[-1]+1, :]
        m2_local = m2_mask[m2_idx[0]:m2_idx[-1]+1]
        m2_vals = m2_vals[m2_local]

        print(f"  {'Marker':15s} {'M1 mean':>10s} {'M2 mean':>10s} {'Diff':>10s} {'p':>12s}")
        print("  " + "-" * 60)
        for i, m in enumerate(avail):
            v1 = m1_vals[:, midx[i]]
            v2 = m2_vals[:, midx[i]]
            _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            diff = float(np.mean(v1)) - float(np.mean(v2))
            sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            print(f"  {m:15s} {float(np.mean(v1)):10.3f} {float(np.mean(v2)):10.3f} "
                  f"{diff:+10.3f} {p:12.2e}{sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    s_global = base / "output" / "all_TMA_S_global_v8.h5ad"
    s_utag = base / "output" / "all_TMA_S_utag.h5ad"
    cov_csv = base / "output" / "hypotheses_v8" / "survival_covariates.csv"

    domain_enrichment(str(s_utag))
    spatial_clustering(str(s_global))
    roi_density(str(s_global), str(cov_csv))
    marker_coexpression(str(s_global))
    print("\nDone.")
