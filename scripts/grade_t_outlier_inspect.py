#!/usr/bin/env python3
"""Inspect T-panel low-shannon outliers — which compartment dominates?

For each ROI flagged by the diagnostic with shannon_compartment < THRESHOLD,
print the compartment composition (top 5 compartments by fraction) so we can
see if the dominant compartment is biological or quality-related (e.g., a core
that is mostly "Unidentified zone" or "LQ / B transitional").

Also reports cell-type composition of the dominant compartment to confirm
whether it's a meaningful biological zone.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clinical_linkage import EXCLUDE_ROIS, normalize_sample_id

LOW_SHANNON_THRESHOLD = 1.0  # ROIs with shannon below this are "outliers"
T_UNASSIGNED = ["Unassigned", "Low quality / Unassigned"]
MIN_CELLS = 8000


def is_tumor_core(sid: str) -> bool:
    s = str(sid).lower()
    if any(t in s for t in ("tonsil", "prostate", "kidney", "spleen", "adrenal")):
        return False
    if "_ton_" in s or "_adr_" in s or "_lym_" in s or "_lym " in s:
        return False
    if s.startswith("biomax"):
        return False
    if sid in EXCLUDE_ROIS:
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t-panel", default="output/all_TMA_T_utag_ct_merged.h5ad")
    p.add_argument("--clinical", default="data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")
    p.add_argument("--grade", default="data/clinicaldata/BCCA_tFL_clinical.xlsx")
    p.add_argument("--low-shannon", type=float, default=LOW_SHANNON_THRESHOLD)
    args = p.parse_args()

    print(f"Loading {args.t_panel} ...")
    with h5py.File(args.t_panel, "r") as f:
        sid_codes = f["obs/sample_id/codes"][:]
        sid_cats = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in f["obs/sample_id/categories"][:]])
        sample_id = sid_cats[sid_codes]
        ct_codes = f["obs/cell_type/codes"][:]
        ct_cats = np.array([c.decode() if isinstance(c, bytes) else c
                            for c in f["obs/cell_type/categories"][:]])
        cell_type = ct_cats[ct_codes]
        comp_codes = f["obs/compartment_name/codes"][:]
        comp_cats = np.array([c.decode() if isinstance(c, bytes) else c
                              for c in f["obs/compartment_name/categories"][:]])
        compartment = comp_cats[comp_codes]

    df = pd.DataFrame({"sample_id": sample_id, "cell_type": cell_type,
                       "compartment": compartment})
    df = df[df.sample_id.apply(is_tumor_core)].copy()
    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)

    # Compute per-ROI shannon, identify low-shannon ROIs (only those passing min_cells)
    rows = []
    for sid, sub in df.groupby("sample_id"):
        not_un = ~sub["cell_type"].isin(T_UNASSIGNED)
        n_typed = int(not_un.sum())
        if n_typed < MIN_CELLS:
            continue
        comp_fracs = sub["compartment"].value_counts(normalize=True)
        p_ = comp_fracs.values
        shannon = float(-(p_ * np.log2(p_ + 1e-12)).sum())
        rows.append({"sample_id": sid, "n_typed": n_typed,
                     "shannon": shannon})
    roi_df = pd.DataFrame(rows)

    # Join clinical for grade + Patient_ID
    clin = pd.read_csv(args.clinical)[["slide_ID", "Sample_ID", "Patient_ID"]]
    grade_df = pd.read_excel(args.grade).rename(columns={"FL ID": "Sample_ID", "DIAG": "grade"})
    grade_df = grade_df[["Sample_ID", "grade"]]
    roi_df = roi_df.merge(clin, left_on="sample_id", right_on="slide_ID", how="left")
    roi_df = roi_df.merge(grade_df, on="Sample_ID", how="left")

    low = roi_df[roi_df.shannon < args.low_shannon].sort_values("shannon")
    print(f"\nROIs with shannon_compartment < {args.low_shannon}: "
          f"{len(low)} of {len(roi_df)} ({100*len(low)/len(roi_df):.1f}%)")
    print()

    for _, row in low.iterrows():
        sid = row["sample_id"]
        sub = df[df.sample_id == sid]
        comp_fracs = sub["compartment"].value_counts(normalize=True)
        ct_fracs = sub["cell_type"].value_counts(normalize=True)
        # Dominant compartment composition
        top_comp = comp_fracs.index[0]
        comp_sub = sub[sub.compartment == top_comp]
        top_comp_ct = comp_sub["cell_type"].value_counts(normalize=True).head(5)

        grade = row.get("grade", "?")
        patient = row.get("Patient_ID", "?")
        print(f"=== {sid}  (Patient={patient}, grade={grade}, "
              f"shannon={row.shannon:.3f}, n_typed={int(row.n_typed)}) ===")
        print(f"  Top compartments (frac):")
        for c, f in comp_fracs.head(5).items():
            print(f"    {c:42s}  {f:.3f}")
        print(f"  Within '{top_comp}' ({len(comp_sub):,} cells), top cell types:")
        for c, f in top_comp_ct.items():
            print(f"    {c:42s}  {f:.3f}")
        # Overall typed-cell composition
        typed_frac = sub[~sub["cell_type"].isin(T_UNASSIGNED)]["cell_type"].value_counts(normalize=True).head(5)
        print(f"  Top typed cell types (across whole ROI):")
        for c, f in typed_frac.items():
            print(f"    {c:42s}  {f:.3f}")
        # Unassigned fraction
        un_frac = float(sub["cell_type"].isin(T_UNASSIGNED).mean())
        print(f"  Unassigned fraction: {un_frac:.3f}")
        print()


if __name__ == "__main__":
    main()
