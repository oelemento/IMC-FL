"""
Deep investigation of macrophage biology in follicular lymphoma.

Analyses:
  1. Marker profiles: What distinguishes M1 vs M2 vs generic Mac vs S100A9+ myeloid?
  2. Domain depth: How far into the interfollicular zone are they?
  3. Spatial neighborhoods: What cell types surround each macrophage subtype?
  4. M1-M2 spatial relationship: Co-localized or segregated?
  5. Functional markers on macrophages: IDO, VISTA, HLA-DR, chemokines
  6. Macrophage-lymphocyte interactions: Proximity to B, CD4, CD8 T cells
  7. Per-ROI heterogeneity: What drives variation in macrophage content?
"""

import h5py
import numpy as np
from collections import Counter
from scipy import stats
from scipy.spatial import KDTree


def load_array(f, key):
    ds = f["obs"][key]
    if isinstance(ds, h5py.Group) and "categories" in ds:
        cats = ds["categories"][:]
        codes = ds["codes"][:]
        cats_str = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
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


MAC_SUBTYPES = ["M1 Macrophages", "M2 Macrophages", "Macrophages",
                "Myeloid (S100A9+)", "Dendritic cells"]
SKIP = {"DNA1", "DNA2", "HistoneH3"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading S-panel data...")
f = h5py.File("output/all_TMA_S_global_v8.h5ad", "r")
X = f["X"][:]
markers = [v.decode() for v in f["var"]["_index"][:]]
cell_types = load_array(f, "cell_type")
sample_ids = load_array(f, "sample_id")
cx = f["obs"]["centroid_x"][:]
cy = f["obs"]["centroid_y"][:]
f.close()

marker_idx = {m: i for i, m in enumerate(markers)}

# Filter to tumor
tumor_mask = np.array([is_tumor_core(s) and not s.startswith("Biomax") for s in sample_ids])
X = X[tumor_mask]; cell_types = cell_types[tumor_mask]
sample_ids = sample_ids[tumor_mask]; cx = cx[tumor_mask]; cy = cy[tumor_mask]
print(f"  {len(X):,} tumor cells")

for mt in MAC_SUBTYPES:
    print(f"  {mt}: {(cell_types == mt).sum():,}")

# Also load UTAG domains
print("\nLoading UTAG domains...")
f2 = h5py.File("output/all_TMA_S_utag.h5ad", "r")
u_ct = load_array(f2, "cell_type")
u_sids = load_array(f2, "sample_id")
obs_keys = list(f2["obs"].keys())
domain_key = None
for k in ["UTAG Label_leiden_0.015", "compartment_name", "utag_domain", "leiden_0.015"]:
    if k in obs_keys:
        domain_key = k
        break
if domain_key:
    u_domains = load_array(f2, domain_key)
    u_cx = f2["obs"]["centroid_x"][:]
    u_cy = f2["obs"]["centroid_y"][:]
    print(f"  Domain key: {domain_key}")
else:
    print("  WARNING: No domain key found")
    u_domains = None
f2.close()

# Classify UTAG domains as follicular
if u_domains is not None:
    u_tumor = np.array([is_tumor_core(s) and not s.startswith("Biomax") for s in u_sids])
    u_ct_t = u_ct[u_tumor]
    u_dom_t = u_domains[u_tumor]
    u_cx_t = u_cx[u_tumor]
    u_cy_t = u_cy[u_tumor]
    u_sid_t = u_sids[u_tumor]

    all_ct_u = np.unique(u_ct_t)
    b_lineage = [ct for ct in all_ct_u
                 if "B cell" in ct or "B " in ct or ct.startswith("B ")
                 or "GC" in ct or "Plasma" in ct]
    foll_domains = set()
    for d in np.unique(u_dom_t):
        d_mask = u_dom_t == d
        b_frac = np.mean(np.isin(u_ct_t[d_mask], b_lineage))
        if b_frac > 0.5:
            foll_domains.add(d)
    is_foll = np.isin(u_dom_t, list(foll_domains))
    print(f"  Follicular domains: {foll_domains}")
    print(f"  {is_foll.sum():,} follicular / {(~is_foll).sum():,} interfollicular cells")

# ======================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: MACROPHAGE SUBTYPE MARKER PROFILES")
print("=" * 70)

func_markers = [m for m in markers if m not in SKIP]
print(f"\n{'Marker':<16s}", end="")
for mt in MAC_SUBTYPES:
    short = mt.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+")
    short = short.replace("Dendritic cells", "DC")
    print(f" {short:>10s}", end="")
print(f" {'FDC':>10s}")
print("-" * (16 + 11 * (len(MAC_SUBTYPES) + 1)))

# Sort by max difference across subtypes
profile_data = {}
for m in func_markers:
    mi = marker_idx[m]
    vals = {}
    for mt in MAC_SUBTYPES:
        mask = cell_types == mt
        if mask.sum() > 0:
            vals[mt] = float(X[mask, mi].mean())
        else:
            vals[mt] = 0.0
    # Also FDC for comparison
    fdc_mask = cell_types == "FDC"
    vals["FDC"] = float(X[fdc_mask, mi].mean()) if fdc_mask.sum() > 0 else 0.0
    profile_data[m] = vals

# Sort by variance across subtypes (most distinguishing markers first)
marker_var = [(m, np.var([profile_data[m][mt] for mt in MAC_SUBTYPES]))
              for m in func_markers]
marker_var.sort(key=lambda x: -x[1])

for m, _ in marker_var:
    vals = profile_data[m]
    print(f"{m:<16s}", end="")
    for mt in MAC_SUBTYPES:
        print(f" {vals[mt]:>10.3f}", end="")
    print(f" {vals['FDC']:>10.3f}")

# ======================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: DISTANCE TO FOLLICULAR DOMAIN BOUNDARY")
print("=" * 70)

if u_domains is not None:
    # For each ROI, find the boundary between follicular and interfollicular
    # Then measure how far each macrophage is from the nearest boundary cell
    rois_with_both = set()
    for roi in np.unique(u_sid_t):
        roi_mask = u_sid_t == roi
        roi_foll = is_foll[roi_mask]
        if roi_foll.sum() > 50 and (~roi_foll).sum() > 50:
            rois_with_both.add(roi)
    print(f"\n  ROIs with both foll + ifoll domains: {len(rois_with_both)}")

    # Sample up to 20 ROIs for speed
    rng = np.random.default_rng(42)
    sample_rois = sorted(rois_with_both)
    if len(sample_rois) > 20:
        sample_rois = list(rng.choice(sample_rois, 20, replace=False))

    mac_dist_to_boundary = {mt: [] for mt in MAC_SUBTYPES}
    fdc_dist_to_boundary = []
    bcell_dist_to_boundary = []

    for roi in sample_rois:
        roi_mask = u_sid_t == roi
        roi_ct = u_ct_t[roi_mask]
        roi_foll = is_foll[roi_mask]
        roi_cx_r = u_cx_t[roi_mask]
        roi_cy_r = u_cy_t[roi_mask]

        # Find boundary cells: follicular cells with >=1 interfollicular neighbor within 30μm
        foll_idx = np.where(roi_foll)[0]
        ifoll_idx = np.where(~roi_foll)[0]
        if len(foll_idx) < 10 or len(ifoll_idx) < 10:
            continue

        # Build KDTree of interfollicular cells
        ifoll_tree = KDTree(np.column_stack([roi_cx_r[ifoll_idx], roi_cy_r[ifoll_idx]]))
        # Boundary = follicular cells within 30μm of interfollicular
        foll_coords = np.column_stack([roi_cx_r[foll_idx], roi_cy_r[foll_idx]])
        dists_to_ifoll, _ = ifoll_tree.query(foll_coords, k=1)
        boundary_foll = foll_idx[dists_to_ifoll < 30]

        if len(boundary_foll) < 5:
            continue

        boundary_coords = np.column_stack([roi_cx_r[boundary_foll], roi_cy_r[boundary_foll]])
        boundary_tree = KDTree(boundary_coords)

        # Distance from each macrophage subtype to boundary
        for mt in MAC_SUBTYPES:
            mt_mask = roi_ct == mt
            if mt_mask.sum() < 5:
                continue
            mt_coords = np.column_stack([roi_cx_r[mt_mask], roi_cy_r[mt_mask]])
            d, _ = boundary_tree.query(mt_coords, k=1)
            # Sign: positive = interfollicular side, negative = inside follicle
            # Use: check if the cell is in a follicular domain
            mt_in_foll = roi_foll[mt_mask]
            signed_d = np.where(mt_in_foll, -d, d)  # negative = inside follicle
            mac_dist_to_boundary[mt].extend(signed_d.tolist())

        # FDC distance for comparison
        fdc_m = roi_ct == "FDC"
        if fdc_m.sum() >= 5:
            fdc_coords = np.column_stack([roi_cx_r[fdc_m], roi_cy_r[fdc_m]])
            d, _ = boundary_tree.query(fdc_coords, k=1)
            fdc_in_foll = roi_foll[fdc_m]
            signed_d = np.where(fdc_in_foll, -d, d)
            fdc_dist_to_boundary.extend(signed_d.tolist())

        # B cell distance
        b_m = np.isin(roi_ct, ["B cells (BCL2+)", "B cells (PAX5+)", "B cells"])
        if b_m.sum() >= 5:
            b_coords = np.column_stack([roi_cx_r[b_m], roi_cy_r[b_m]])
            d, _ = boundary_tree.query(b_coords, k=1)
            b_in_foll = roi_foll[b_m]
            signed_d = np.where(b_in_foll, -d, d)
            bcell_dist_to_boundary.extend(signed_d.tolist())

    print(f"\n  Signed distance to follicle boundary (negative = inside follicle, positive = outside):")
    print(f"  {'Cell type':<25s} {'n':>8s} {'median':>8s} {'mean':>8s} {'%inside':>8s}")
    print(f"  {'-'*60}")
    for mt in MAC_SUBTYPES:
        d = np.array(mac_dist_to_boundary[mt])
        if len(d) > 0:
            short = mt.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+ myeloid")
            short = short.replace("Dendritic cells", "DCs")
            pct_inside = (d < 0).mean() * 100
            print(f"  {short:<25s} {len(d):>8,d} {np.median(d):>8.1f} {d.mean():>8.1f} {pct_inside:>7.1f}%")
    d_fdc = np.array(fdc_dist_to_boundary)
    if len(d_fdc) > 0:
        print(f"  {'FDC':<25s} {len(d_fdc):>8,d} {np.median(d_fdc):>8.1f} {d_fdc.mean():>8.1f} {(d_fdc < 0).mean()*100:>7.1f}%")
    d_b = np.array(bcell_dist_to_boundary)
    if len(d_b) > 0:
        print(f"  {'B cells (all)':<25s} {len(d_b):>8,d} {np.median(d_b):>8.1f} {d_b.mean():>8.1f} {(d_b < 0).mean()*100:>7.1f}%")

# ======================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: SPATIAL NEIGHBORHOODS OF EACH MACROPHAGE SUBTYPE")
print("=" * 70)

# Use 15 ROIs with most myeloid cells
mac_any = np.isin(cell_types, MAC_SUBTYPES)
mac_per_roi = Counter(sample_ids[mac_any])
top_rois = [r for r, _ in mac_per_roi.most_common(15)]

for mt in MAC_SUBTYPES:
    nbr_counts = Counter()
    n_total = 0
    for roi in top_rois:
        rmask = sample_ids == roi
        roi_ct = cell_types[rmask]
        roi_cx_r = cx[rmask]
        roi_cy_r = cy[rmask]

        mt_local = np.where(roi_ct == mt)[0]
        non_mt_local = np.where(roi_ct != mt)[0]
        if len(mt_local) < 10 or len(non_mt_local) < 20:
            continue

        tree = KDTree(np.column_stack([roi_cx_r[non_mt_local], roi_cy_r[non_mt_local]]))
        mt_coords = np.column_stack([roi_cx_r[mt_local], roi_cy_r[mt_local]])
        _, idxs = tree.query(mt_coords, k=10)
        for j in range(len(mt_local)):
            nbr_cts = roi_ct[non_mt_local[idxs[j]]]
            nbr_counts.update(nbr_cts)
            n_total += 10

    if n_total == 0:
        continue

    short_mt = mt.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+")
    short_mt = short_mt.replace("Dendritic cells", "DC")
    print(f"\n  {short_mt} neighbors ({n_total:,} total):")
    top_nbrs = nbr_counts.most_common(10)
    for ct_n, count in top_nbrs:
        frac = count / n_total * 100
        print(f"    {ct_n:<30s} {frac:5.1f}%")

# ======================================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: M1-M2 SPATIAL RELATIONSHIP")
print("=" * 70)

m1m2_dists = []
m2m1_dists = []
m1m1_dists = []
m2m2_dists = []

for roi in top_rois:
    rmask = sample_ids == roi
    roi_ct = cell_types[rmask]
    roi_cx_r = cx[rmask]
    roi_cy_r = cy[rmask]

    m1_local = np.where(roi_ct == "M1 Macrophages")[0]
    m2_local = np.where(roi_ct == "M2 Macrophages")[0]

    if len(m1_local) < 10 or len(m2_local) < 10:
        continue

    m1_coords = np.column_stack([roi_cx_r[m1_local], roi_cy_r[m1_local]])
    m2_coords = np.column_stack([roi_cx_r[m2_local], roi_cy_r[m2_local]])

    # M1 to nearest M2
    m2_tree = KDTree(m2_coords)
    d_m1_to_m2, _ = m2_tree.query(m1_coords, k=1)
    m1m2_dists.extend(d_m1_to_m2.tolist())

    # M2 to nearest M1
    m1_tree = KDTree(m1_coords)
    d_m2_to_m1, _ = m1_tree.query(m2_coords, k=1)
    m2m1_dists.extend(d_m2_to_m1.tolist())

    # M1 to nearest M1 (self-clustering)
    if len(m1_local) > 1:
        d_m1_m1, _ = m1_tree.query(m1_coords, k=2)  # k=2 because k=1 is self
        m1m1_dists.extend(d_m1_m1[:, 1].tolist())

    # M2 to nearest M2
    if len(m2_local) > 1:
        d_m2_m2, _ = m2_tree.query(m2_coords, k=2)
        m2m2_dists.extend(d_m2_m2[:, 1].tolist())

print(f"\n  M1→nearest M2: median={np.median(m1m2_dists):.1f}μm, mean={np.mean(m1m2_dists):.1f}μm (n={len(m1m2_dists):,})")
print(f"  M2→nearest M1: median={np.median(m2m1_dists):.1f}μm, mean={np.mean(m2m1_dists):.1f}μm (n={len(m2m1_dists):,})")
print(f"  M1→nearest M1: median={np.median(m1m1_dists):.1f}μm, mean={np.mean(m1m1_dists):.1f}μm (n={len(m1m1_dists):,})")
print(f"  M2→nearest M2: median={np.median(m2m2_dists):.1f}μm, mean={np.mean(m2m2_dists):.1f}μm (n={len(m2m2_dists):,})")

# Are M1 and M2 segregated or intermingled?
# If M1→M2 distance ≈ M1→M1 distance, they are intermingled
# If M1→M2 >> M1→M1, they are segregated
ratio_m1 = np.median(m1m2_dists) / np.median(m1m1_dists) if np.median(m1m1_dists) > 0 else 0
ratio_m2 = np.median(m2m1_dists) / np.median(m2m2_dists) if np.median(m2m2_dists) > 0 else 0
print(f"\n  Segregation index (cross/self distance ratio):")
print(f"    M1: {ratio_m1:.2f}x (>1 = segregated, ~1 = intermingled)")
print(f"    M2: {ratio_m2:.2f}x")

if ratio_m1 > 2.0 and ratio_m2 > 2.0:
    print("  → M1 and M2 are SPATIALLY SEGREGATED into different zones")
elif ratio_m1 < 1.5 and ratio_m2 < 1.5:
    print("  → M1 and M2 are INTERMINGLED in the same zones")
else:
    print("  → PARTIAL segregation: some mixing but spatial preference exists")

# ======================================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: FUNCTIONAL MARKER CO-EXPRESSION ON MACROPHAGES")
print("=" * 70)

func_markers_interest = ["IDO", "VISTA", "HLA_DR", "HLA_Class_I", "CXCL13",
                         "CXCL12", "CCL21", "CD14", "Ki-67", "BCL_2",
                         "CD47", "PD_L1"]

for mt in MAC_SUBTYPES:
    mask = cell_types == mt
    n = mask.sum()
    if n < 100:
        continue
    short = mt.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+")
    short = short.replace("Dendritic cells", "DC")
    print(f"\n  {short} (n={n:,}):")
    for fm in func_markers_interest:
        if fm not in marker_idx:
            continue
        vals = X[mask, marker_idx[fm]]
        mean_v = float(vals.mean())
        frac_pos = float((vals > 1.0).mean())
        p90 = float(np.percentile(vals, 90))
        print(f"    {fm:<16s} mean={mean_v:>6.3f}  %>1.0={frac_pos:>5.1%}  p90={p90:>6.3f}")

# Highlight: which macrophage subtype has highest IDO? VISTA? HLA-DR?
print("\n  Summary — which subtype leads each functional marker:")
for fm in func_markers_interest:
    if fm not in marker_idx:
        continue
    best_mt = None
    best_val = -999
    for mt in MAC_SUBTYPES:
        mask = cell_types == mt
        if mask.sum() < 100:
            continue
        v = float(X[mask, marker_idx[fm]].mean())
        if v > best_val:
            best_val = v
            best_mt = mt
    short = best_mt.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+")
    short = short.replace("Dendritic cells", "DC")
    print(f"    {fm:<16s} → {short} (mean={best_val:.3f})")

# ======================================================================
print("\n" + "=" * 70)
print("ANALYSIS 6: MACROPHAGE-LYMPHOCYTE SPATIAL INTERACTIONS")
print("=" * 70)

lymph_types = ["B cells (BCL2+)", "B cells (PAX5+)", "CD4 T cells", "CD8 T cells", "FDC"]

for mt in MAC_SUBTYPES:
    short = mt.replace("Macrophages", "Mac").replace("Myeloid (S100A9+)", "S100A9+")
    short = short.replace("Dendritic cells", "DC")
    print(f"\n  {short} → nearest lymphocyte/stromal distance:")

    dists_by_target = {lt: [] for lt in lymph_types}
    for roi in top_rois:
        rmask = sample_ids == roi
        roi_ct = cell_types[rmask]
        roi_cx_r = cx[rmask]
        roi_cy_r = cy[rmask]

        mt_local = np.where(roi_ct == mt)[0]
        if len(mt_local) < 10:
            continue
        mt_coords = np.column_stack([roi_cx_r[mt_local], roi_cy_r[mt_local]])

        for lt in lymph_types:
            lt_local = np.where(roi_ct == lt)[0]
            if len(lt_local) < 5:
                continue
            lt_tree = KDTree(np.column_stack([roi_cx_r[lt_local], roi_cy_r[lt_local]]))
            d, _ = lt_tree.query(mt_coords, k=1)
            dists_by_target[lt].extend(d.tolist())

    for lt in lymph_types:
        d = dists_by_target[lt]
        if len(d) > 0:
            print(f"    → {lt:<25s} median={np.median(d):>6.1f}μm  mean={np.mean(d):>6.1f}μm  (n={len(d):,})")

# ======================================================================
print("\n" + "=" * 70)
print("ANALYSIS 7: WHAT DRIVES PER-ROI MACROPHAGE VARIATION?")
print("=" * 70)

# Per-ROI macrophage fraction and marker means
unique_rois = np.unique(sample_ids)
roi_mac_frac = {}
roi_marker_means = {}
for roi in unique_rois:
    rmask = sample_ids == roi
    n = rmask.sum()
    if n < 100:
        continue
    roi_ct = cell_types[rmask]
    roi_mac_frac[roi] = np.isin(roi_ct, MAC_SUBTYPES).mean()
    for m in func_markers:
        if m in SKIP:
            continue
        roi_marker_means.setdefault(m, {})[roi] = float(X[rmask, marker_idx[m]].mean())

common_r = sorted(roi_mac_frac.keys())
mac_arr = np.array([roi_mac_frac[r] for r in common_r])

print(f"\n  {len(common_r)} ROIs, macrophage fraction: mean={mac_arr.mean():.3f}, "
      f"median={np.median(mac_arr):.3f}, range=[{mac_arr.min():.3f}, {mac_arr.max():.3f}]")

# What whole-ROI markers correlate with macrophage fraction?
print(f"\n  Top markers correlated with per-ROI macrophage fraction:")
corrs = []
for m in func_markers:
    if m in SKIP or m not in roi_marker_means:
        continue
    vals = np.array([roi_marker_means[m].get(r, 0) for r in common_r])
    rho, p = stats.spearmanr(mac_arr, vals)
    corrs.append((m, rho, p))
corrs.sort(key=lambda x: -abs(x[1]))

print(f"  {'Marker':<16s} {'rho':>8s} {'p':>12s}")
print(f"  {'-'*40}")
for m, rho, p in corrs[:15]:
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {m:<16s} {rho:>+8.3f} {p:>12.2e} {sig}")

# Also: which cell type fractions co-vary with macrophage fraction?
print(f"\n  Cell type fractions correlated with macrophage fraction:")
ct_list = sorted(set(cell_types) - {"Low quality / Unassigned"} - set(MAC_SUBTYPES))
ct_corrs = []
for ct in ct_list:
    ct_fracs = np.array([float((cell_types[sample_ids == r] == ct).mean()) for r in common_r])
    rho, p = stats.spearmanr(mac_arr, ct_fracs)
    ct_corrs.append((ct, rho, p))
ct_corrs.sort(key=lambda x: -abs(x[1]))

print(f"  {'Cell type':<30s} {'rho':>8s} {'p':>12s}")
print(f"  {'-'*55}")
for ct, rho, p in ct_corrs:
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {ct:<30s} {rho:>+8.3f} {p:>12.2e} {sig}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
