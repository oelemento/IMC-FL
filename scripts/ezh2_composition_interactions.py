#!/usr/bin/env python3
"""Wendy EZH2 mut vs WT analysis — slide 2 deliverable.

Produces a 4-panel figure on the T-panel (matches her PDF):
  (a) Overall cell type composition stacked bar — WT FL / Mut FL / Tonsil
  (b) Pairwise interaction enrichment z-score heatmap — EZH2 WT FL
  (c) Same heatmap — EZH2 Mut FL
  (d) Delta heatmap — (Mut - WT) z-scores

Interaction matrix: 12 specific cell-cell pairs Wendy listed in her PDF, by 9
paper-9 compartments. Per ROI we compute the 8x8 cell-type-pair z-score matrix
within each of the 9 compartments (k=10, n_perm=100). Patient-level aggregation
(mean across ROIs per patient), then group mean (WT / Mut). Tonsil is shown for
composition only (no patient EZH2 status for tonsil).

Cohort filter mirrors the rest of the project: exclude controls (tonsil/prostate/
kidney/spleen/adrenal/Biomax suffix patterns) for FL groups; tonsil group keeps
just the tonsil ROIs.
"""
import argparse, sys, os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

# --- Cohort definitions ---
MIN_CELLS_PER_ROI = 8000
MIN_CELLS_PER_COMPARTMENT = 200  # subset must have >=200 cells to run nhood
K_NN = 10
N_PERM = 100
UNASSIGNED_CT = ["Unassigned", "Low quality / Unassigned"]

# --- Cell type grouping ---
COMPOSITION_MAP = {  # broad 6-category for stacked bar (matches Wendy)
    "B cells": "B cells", "B cells (CXCR5hi)": "B cells",
    "B cells (CD20hi)": "B cells", "B cells (TOXhi)": "B cells",
    "GC B cells": "B cells", "Activated B / Plasmablast": "B cells",
    "CD4 T cells": "CD4 T",
    "CD8 T cells": "CD8 T",
    "CD8 T exhausted": "CD8 T", "CD8 T pre-exhausted (TOX+)": "CD8 T",
    "Treg": "Treg",
    "Macrophages": "Macrophage", "Macrophages (GzmB+)": "Macrophage",
    "T cells": "Other", "Mixed / Border cells": "Other", "Other": "Other",
    "Unassigned": "Unassigned",
}
COMPOSITION_ORDER = ["B cells", "CD4 T", "CD8 T", "Treg", "Macrophage", "Other"]
COMPOSITION_COLORS = {
    "B cells": "#4CAF50",     # green
    "CD4 T":   "#2196F3",     # blue
    "CD8 T":   "#1565C0",     # darker blue
    "Treg":    "#9C27B0",     # purple
    "Macrophage": "#8D6E63",  # brown
    "Other":   "#9E9E9E",     # gray
}

# --- 8 cell-type categories for interaction matrix (T-panel) ---
INTERACT_MAP = {
    "B cells": "B cells", "B cells (CXCR5hi)": "B cells",
    "B cells (CD20hi)": "B cells", "B cells (TOXhi)": "B cells",
    "GC B cells": "GC B cells", "Activated B / Plasmablast": "GC B cells",
    "CD4 T cells": "CD4 T",
    "CD8 T cells": "CD8 T",
    "CD8 T exhausted": "CD8 exh", "CD8 T pre-exhausted (TOX+)": "CD8 exh",
    "Treg": "Treg",
    "Macrophages": "Mac",
    "Macrophages (GzmB+)": "GzmB+ Mac",
}
INTERACT_TYPES = ["B cells", "GC B cells", "CD4 T", "CD8 T", "CD8 exh",
                  "Treg", "Mac", "GzmB+ Mac"]

# Wendy's 12 specific pairs (row labels on her heatmap)
INTERACT_PAIRS = [
    ("B cells", "Mac"),
    ("CD4 T", "Treg"),
    ("CD4 T", "CD8 T"),
    ("CD4 T", "CD8 exh"),
    ("CD4 T", "Mac"),
    ("Treg", "CD8 T"),
    ("Treg", "CD8 exh"),
    ("Treg", "Mac"),
    ("CD8 T", "CD8 exh"),
    ("CD8 T", "Mac"),
    ("CD8 exh", "Mac"),
    ("GzmB+ Mac", "Mac"),
]

# Paper-9 compartment ordering (matches Wendy's column order)
PAPER_9 = [
    "GC core",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)",
    "B cell follicle (CD20hi/CXCR5hi)",
    "B cell zone",
    "Follicle-T zone interface",
    "Treg-enriched T zone",
    "T cell zone (CD4/CD8)",
    "Macrophage-rich zone",
]
PAPER_9_SHORT = [
    "GC core", "Follicle core", "Follicle mantle",
    "B cell follicle", "B cell zone", "Foll-T interface",
    "Treg zone", "T cell zone", "Mac zone",
]


def is_tumor_core(sid):
    s = str(sid).lower()
    if any(t in s for t in ("tonsil", "prostate", "kidney", "spleen", "adrenal")):
        return False
    if any(t in s for t in ("_ton_", "_adr_", "_lym_", "_lym ")):
        return False
    if s.startswith("biomax"):
        return False
    if sid in EXCLUDE_ROIS:
        return False
    return True


def is_tonsil_core(sid):
    s = str(sid).lower()
    return "tonsil" in s or "_ton_" in s


def nhood_z(cx, cy, ct_codes, n_types, k=K_NN, n_perm=N_PERM, rng=None):
    """Per-ROI/per-compartment neighborhood enrichment z-score matrix."""
    n = len(cx)
    if n < k + 1:
        return None
    coords = np.column_stack([cx, cy])
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    neighbors = idx[:, 1:]
    rng = rng or np.random.default_rng()

    def count_pairs(codes):
        # vectorized: (n_types, n_types) matrix of source -> neighbor counts
        nb = codes[neighbors]  # (n, k)
        out = np.zeros((n_types, n_types), dtype=np.float64)
        for s in range(n_types):
            mask = codes == s
            if not mask.any():
                continue
            nb_of_s = nb[mask]
            nb_valid = nb_of_s[nb_of_s >= 0]
            if len(nb_valid) == 0:
                continue
            counts = np.bincount(nb_valid, minlength=n_types)
            out[s, :] = counts[:n_types]
        return out

    obs = count_pairs(ct_codes)
    perm_sums = np.zeros((n_types, n_types), dtype=np.float64)
    perm_sq = np.zeros((n_types, n_types), dtype=np.float64)
    for _ in range(n_perm):
        perm = rng.permutation(ct_codes)
        m = count_pairs(perm)
        perm_sums += m
        perm_sq += m**2
    mean = perm_sums / n_perm
    var = perm_sq / n_perm - mean**2
    std = np.sqrt(np.maximum(var, 0))
    std[std == 0] = 1
    return (obs - mean) / std


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t-panel", default="output/all_TMA_T_utag_ct_merged.h5ad")
    ap.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    ap.add_argument("--ezh2", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    ap.add_argument("--out", default="output/ezh2/composition_interactions")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.t_panel} ...")
    with h5py.File(args.t_panel, "r") as f:
        sid_codes = f["obs/sample_id/codes"][:]
        sid_cats = np.array([c.decode() for c in f["obs/sample_id/categories"][:]])
        sample_id = sid_cats[sid_codes]
        ct_codes_raw = f["obs/cell_type/codes"][:]
        ct_cats = np.array([c.decode() for c in f["obs/cell_type/categories"][:]])
        cell_type = ct_cats[ct_codes_raw]
        comp_codes = f["obs/compartment_name/codes"][:]
        comp_cats = np.array([c.decode() for c in f["obs/compartment_name/categories"][:]])
        compartment = comp_cats[comp_codes]
        cx = f["obs/centroid_x"][:]
        cy = f["obs/centroid_y"][:]

    print(f"  total cells: {len(sample_id):,}, unique ROIs: {len(np.unique(sample_id))}")

    # Build a working dataframe
    df = pd.DataFrame({
        "sample_id": sample_id, "cell_type": cell_type,
        "compartment": compartment, "cx": cx, "cy": cy,
    })

    # Group assignment
    df["is_tumor"] = df.sample_id.apply(is_tumor_core)
    df["is_tonsil"] = df.sample_id.apply(is_tonsil_core)
    df["sid_norm"] = df.sample_id.apply(normalize_sample_id)

    # EZH2 status (tumor only)
    clin = pd.read_csv(args.clinical)[["slide_ID", "Sample_ID", "Patient_ID"]]
    ezh = pd.read_excel(args.ezh2).rename(columns={"FL ID": "Sample_ID"})[["Sample_ID", "EZH2"]]
    sid_to_pt_ezh = (clin.merge(ezh, on="Sample_ID", how="inner")
                     [["slide_ID", "Patient_ID", "EZH2"]]
                     .drop_duplicates())
    df = df.merge(sid_to_pt_ezh, left_on="sid_norm", right_on="slide_ID", how="left")

    # ROI-level QC: min typed cells. Tonsil cores tend to be smaller; allow
    # a relaxed 5000-cell threshold (composition reference only — they are
    # not used in the interaction-heatmap computation).
    typed_per_roi = (df[~df.cell_type.isin(UNASSIGNED_CT)]
                     .groupby("sid_norm").size())
    tonsil_ids = set(df[df.is_tonsil].sid_norm)
    keep_rois = set(typed_per_roi[typed_per_roi >= MIN_CELLS_PER_ROI].index)
    keep_tonsil = set(typed_per_roi[(typed_per_roi >= 5000) & typed_per_roi.index.isin(tonsil_ids)].index)
    keep_all = keep_rois | keep_tonsil
    df = df[df.sid_norm.isin(keep_all)].copy()

    # Assign final group: WT, Mut, Tonsil, exclude
    def group(row):
        if row.is_tonsil:
            return "Tonsil"
        if not row.is_tumor:
            return "exclude"
        if row.EZH2 == "wt":
            return "WT"
        if row.EZH2 == "mut":
            return "Mut"
        return "exclude"
    df["group"] = df.apply(group, axis=1)
    df = df[df.group.isin(["WT", "Mut", "Tonsil"])].copy()
    # For tonsil rows without a clinical Patient_ID, use the sample_id as a
    # surrogate so groupby aggregations keep them.
    df.loc[df.group == "Tonsil", "Patient_ID"] = df.loc[df.group == "Tonsil", "sid_norm"]

    print(f"\n  ROIs by group (post-QC):")
    n_per_group = df.groupby("group").sid_norm.nunique().to_dict()
    print(f"    {n_per_group}")
    pt_per_group = df.groupby("group").Patient_ID.nunique().to_dict()
    print(f"  Patients by group (post-QC):")
    print(f"    {pt_per_group}")

    # ===================================================================
    # (a) Composition stacked bar (patient-level mean for FL groups,
    #     ROI-level mean for tonsil since tonsil has no patient mapping)
    #
    # IMPORTANT: drop LQ / Unassigned BEFORE applying COMPOSITION_MAP — otherwise
    # the `.fillna("Other")` quietly routes LQ cells into the "Other" bucket
    # (the bare key "Unassigned" in COMPOSITION_MAP never matches the actual
    # T-panel label "Low quality / Unassigned").
    # ===================================================================
    comp_df = df[~df.cell_type.isin(UNASSIGNED_CT)].copy()
    comp_df["composition_cat"] = comp_df.cell_type.map(COMPOSITION_MAP)
    unmapped = comp_df[comp_df.composition_cat.isna()].cell_type.unique()
    if len(unmapped) > 0:
        raise RuntimeError(
            f"COMPOSITION_MAP missing keys for cell types: {list(unmapped)}. "
            "Add them explicitly so they aren't silently absorbed into 'Other'."
        )

    # Per-ROI composition fractions
    roi_comp = (comp_df.groupby(["sid_norm", "group", "Patient_ID", "composition_cat"])
                .size().unstack(fill_value=0))
    roi_comp = roi_comp.div(roi_comp.sum(axis=1), axis=0)  # normalize per ROI
    roi_comp = roi_comp.reset_index()

    rows = []
    for g in ["WT", "Mut", "Tonsil"]:
        sub = roi_comp[roi_comp.group == g]
        if g == "Tonsil":
            # tonsil has no patient mapping — average ROIs
            row = {c: sub[c].mean() for c in COMPOSITION_ORDER if c in sub.columns}
            row["group"] = g; row["n"] = len(sub); row["unit"] = "ROIs"
        else:
            # patient-level mean
            pt = sub.groupby("Patient_ID")[[c for c in COMPOSITION_ORDER if c in sub.columns]].mean()
            row = {c: pt[c].mean() for c in pt.columns}
            row["group"] = g; row["n"] = len(pt); row["unit"] = "patients"
        rows.append(row)
    comp_summary = pd.DataFrame(rows).set_index("group").reindex(["WT", "Mut", "Tonsil"])
    # Fill missing categories with 0
    for c in COMPOSITION_ORDER:
        if c not in comp_summary.columns:
            comp_summary[c] = 0.0
    # Renormalize to sum to 1 across the 6 ordered categories (drop NaN col)
    comp_summary = comp_summary[COMPOSITION_ORDER + ["n", "unit"]]
    bar_data = comp_summary[COMPOSITION_ORDER].astype(float)
    bar_data = bar_data.div(bar_data.sum(axis=1), axis=0)
    comp_summary[COMPOSITION_ORDER] = bar_data

    comp_summary.to_csv(out_dir / "composition_summary.csv")
    print(f"\n  Saved composition summary: {out_dir / 'composition_summary.csv'}")
    print(comp_summary)

    # ===================================================================
    # (b)/(c)/(d) Per-ROI per-compartment neighborhood enrichment
    # ===================================================================
    # Pre-encode cell_type and compartment for fast access
    interact_map_arr = df.cell_type.map(INTERACT_MAP)
    valid_cells_mask = interact_map_arr.notna().to_numpy()
    type_to_idx = {t: i for i, t in enumerate(INTERACT_TYPES)}
    type_code = np.array([type_to_idx.get(t, -1) for t in interact_map_arr.fillna("__NONE__").to_numpy()])
    df["__type_code"] = type_code

    n_types = len(INTERACT_TYPES)
    n_comps = len(PAPER_9)
    rng = np.random.default_rng(0)

    # For each (ROI, compartment) — compute z matrix
    print(f"\n  Computing per-ROI per-compartment nhood enrichment "
          f"(k={K_NN}, n_perm={N_PERM}) ...")
    z_records = []  # (sid, group, Patient_ID, compartment, z_matrix)
    rois_done = 0
    for sid, sub in df.groupby("sid_norm"):
        g = sub.group.iloc[0]
        pt = sub.Patient_ID.iloc[0]
        for ci, comp_name in enumerate(PAPER_9):
            sub_c = sub[sub.compartment == comp_name]
            if len(sub_c) < MIN_CELLS_PER_COMPARTMENT:
                continue
            codes = sub_c["__type_code"].to_numpy()
            cx_c = sub_c.cx.to_numpy(); cy_c = sub_c.cy.to_numpy()
            z = nhood_z(cx_c, cy_c, codes, n_types, k=K_NN, n_perm=N_PERM, rng=rng)
            if z is None:
                continue
            z_records.append({
                "sid": sid, "group": g, "Patient_ID": pt,
                "compartment": comp_name,
                "compartment_idx": ci,
                "z": z,
            })
        rois_done += 1
        if rois_done % 25 == 0:
            print(f"    [{rois_done}] {len(z_records)} (ROI, compartment) z-matrices computed")
    print(f"  Done. {rois_done} ROIs, {len(z_records)} (ROI, compartment) z-matrices.")

    # Convert to long format: one row per (sid, compartment, pair_idx) with z value
    long_rows = []
    for r in z_records:
        for pi, (a, b) in enumerate(INTERACT_PAIRS):
            ai, bi = type_to_idx[a], type_to_idx[b]
            # Symmetrize: average of (a->b) and (b->a)
            z_pair = 0.5 * (r["z"][ai, bi] + r["z"][bi, ai])
            long_rows.append({
                "sid": r["sid"], "group": r["group"], "Patient_ID": r["Patient_ID"],
                "compartment": r["compartment"], "compartment_idx": r["compartment_idx"],
                "pair_idx": pi, "pair": f"{a}–{b}",
                "z": z_pair,
            })
    z_long = pd.DataFrame(long_rows)
    z_long.to_csv(out_dir / "interaction_z_per_roi.csv", index=False)

    # Patient-level mean per (compartment, pair) then group-level mean
    pt_z = (z_long.groupby(["group", "Patient_ID", "compartment", "compartment_idx",
                              "pair_idx", "pair"])
            .z.mean().reset_index())
    pt_z.to_csv(out_dir / "interaction_z_per_patient.csv", index=False)

    # Per-group heatmap: (12 pairs) x (9 compartments)
    def heatmap_for_group(g):
        sub = pt_z[pt_z.group == g]
        mat = np.full((len(INTERACT_PAIRS), len(PAPER_9)), np.nan)
        n_pts = sub.Patient_ID.nunique()
        for (pi, ci), sub2 in sub.groupby(["pair_idx", "compartment_idx"]):
            if len(sub2) >= 3:  # need >=3 patients
                mat[pi, ci] = sub2.z.mean()
        return mat, n_pts

    Z_WT, n_WT = heatmap_for_group("WT")
    Z_MUT, n_MUT = heatmap_for_group("Mut")
    Z_DIFF = Z_MUT - Z_WT

    # MW test on patient-level z values, per (pair, compartment)
    pmat = np.full((len(INTERACT_PAIRS), len(PAPER_9)), np.nan)
    for pi in range(len(INTERACT_PAIRS)):
        for ci in range(len(PAPER_9)):
            a = pt_z[(pt_z.group == "WT") & (pt_z.pair_idx == pi)
                      & (pt_z.compartment_idx == ci)].z.dropna().values
            b = pt_z[(pt_z.group == "Mut") & (pt_z.pair_idx == pi)
                      & (pt_z.compartment_idx == ci)].z.dropna().values
            if len(a) >= 3 and len(b) >= 3:
                try:
                    _, p = mannwhitneyu(a, b, alternative="two-sided")
                    pmat[pi, ci] = p
                except ValueError:
                    pass

    # Save matrices
    pair_labels = [f"{a}–{b}" for a, b in INTERACT_PAIRS]
    pd.DataFrame(Z_WT, index=pair_labels, columns=PAPER_9_SHORT).to_csv(out_dir / "Z_WT.csv")
    pd.DataFrame(Z_MUT, index=pair_labels, columns=PAPER_9_SHORT).to_csv(out_dir / "Z_Mut.csv")
    pd.DataFrame(Z_DIFF, index=pair_labels, columns=PAPER_9_SHORT).to_csv(out_dir / "Z_Mut_minus_WT.csv")
    pd.DataFrame(pmat, index=pair_labels, columns=PAPER_9_SHORT).to_csv(out_dir / "MW_p_WT_vs_Mut.csv")

    # ===================================================================
    # Figure
    # ===================================================================
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.6, 1.6], hspace=0.35,
                          wspace=0.30, height_ratios=[1, 1])

    # (a) Composition stacked bar
    ax_a = fig.add_subplot(gs[0, 0])
    groups_plot = ["WT", "Mut", "Tonsil"]
    labels = []
    for g in groups_plot:
        if g in comp_summary.index:
            n = int(comp_summary.loc[g, "n"])
            unit = comp_summary.loc[g, "unit"]
            short = "WT FL" if g == "WT" else ("Mut FL" if g == "Mut" else "Tonsil")
            labels.append(f"{short}\n(n={n} {unit[:-1] if unit=='patients' else unit})")
    bottoms = np.zeros(len(groups_plot))
    for c in COMPOSITION_ORDER:
        vals = comp_summary.loc[groups_plot, c].astype(float).values * 100
        ax_a.bar(labels, vals, bottom=bottoms, color=COMPOSITION_COLORS[c],
                 label=c, edgecolor="white", linewidth=0.6)
        for i, (b, v) in enumerate(zip(bottoms, vals)):
            if v > 4:
                ax_a.text(i, b + v/2, f"{v:.0f}%", ha="center", va="center",
                          color="white" if c in ("CD8 T", "Treg", "CD4 T", "Macrophage") else "black",
                          fontsize=9, fontweight="bold")
        bottoms += vals
    ax_a.set_ylabel("% of typed cells", fontsize=11)
    ax_a.set_title("(a) Overall cell type composition", fontsize=12)
    ax_a.set_ylim(0, 100)
    ax_a.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax_a.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax_a.spines[sp].set_visible(False)

    # (b) WT heatmap, (c) Mut heatmap, (d) Mut - WT heatmap
    vmax_abs = float(np.nanmax(np.abs([Z_WT, Z_MUT])))
    vmax_abs = max(min(vmax_abs, 15), 5)

    def draw_hm(ax, mat, title, vmin=-vmax_abs, vmax=vmax_abs, cmap="RdBu_r",
                show_p=False):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(PAPER_9_SHORT)))
        ax.set_xticklabels(PAPER_9_SHORT, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels, fontsize=9)
        ax.set_title(title, fontsize=12)
        # Numbers in cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:+.0f}" if abs(v) >= 1 else f"{v:+.1f}",
                        ha="center", va="center", fontsize=7.5, color=color)
                if show_p and not np.isnan(pmat[i, j]) and pmat[i, j] < 0.05:
                    ax.text(j + 0.32, i - 0.32, "*", ha="center", va="center",
                            fontsize=11, color="black", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    ax_b = fig.add_subplot(gs[0, 1])
    draw_hm(ax_b, Z_WT, f"(b) EZH2 WT — pairwise interaction z-score (n={n_WT} patients)")
    ax_c = fig.add_subplot(gs[0, 2])
    draw_hm(ax_c, Z_MUT, f"(c) EZH2 Mut — pairwise interaction z-score (n={n_MUT} patients)")
    ax_d = fig.add_subplot(gs[1, 1:])
    vmax_d = float(np.nanmax(np.abs(Z_DIFF)))
    vmax_d = max(min(vmax_d, 12), 3)
    draw_hm(ax_d, Z_DIFF, f"(d) Mut − WT  (z-score difference; *: MW p<0.05 per patient)",
            vmin=-vmax_d, vmax=vmax_d, show_p=True)

    # MW summary
    n_sig = int(np.nansum(pmat < 0.05))
    fig.suptitle(
        f"Wendy EZH2 mut vs WT — T-panel composition + interaction heatmaps "
        f"(patient-level; {n_sig} pair×compartment entries with MW p<0.05)",
        fontsize=13, y=0.995,
    )
    out = out_dir / "fig_ezh2_composition_interactions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out}")

    # Print top WT-Mut differences
    print(f"\nTop (pair, compartment) cells with |Mut - WT| z-difference (showing |diff|>=3):")
    rows = []
    for pi in range(len(INTERACT_PAIRS)):
        for ci in range(len(PAPER_9)):
            d = Z_DIFF[pi, ci]
            if np.isnan(d): continue
            rows.append({"pair": pair_labels[pi], "compartment": PAPER_9_SHORT[ci],
                         "diff": d, "WT_z": Z_WT[pi, ci], "Mut_z": Z_MUT[pi, ci],
                         "MW_p": pmat[pi, ci]})
    diff_df = pd.DataFrame(rows).sort_values("diff", key=lambda x: -np.abs(x))
    print(diff_df.head(15).to_string(index=False))
    diff_df.to_csv(out_dir / "delta_summary.csv", index=False)


if __name__ == "__main__":
    main()
