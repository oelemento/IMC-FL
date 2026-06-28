"""
Clinical data linkage for IMC-FL.

Handles naming discrepancies between IMC ROI names and clinical slide_IDs,
loads clinical CSV, and joins clinical variables to h5ad objects.

Naming issues resolved:
  - A1 T-panel: ROI_001..ROI_010 → FL1..FL10
  - B1 T-panel: FL01..FL09 (zero-padded) → FL1..FL9
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

CLINICAL_CSV = Path("data/clinicaldata/BCCA_FL_clinical_merged.2.19.23_DWS.csv")
# Pre-DWS file path retained for diff/regression runs only:
CLINICAL_CSV_PRE_DWS = Path("data/clinicaldata/BCCA_FL_clinical_merged.2.19.23.csv")

# ROIs to exclude: partial/duplicate ablations that duplicate another ROI
# A1_ROI_005 is a partial ablation of the same core as A1_FL5 (patient 12-10614)
EXCLUDE_ROIS = {"A1_ROI_005"}


def normalize_sample_id(sample_id: str) -> str:
    """Normalize an IMC sample_id to match clinical slide_ID format.

    Fixes:
      A1_ROI_001 → A1_FL1  (ROI naming on A1 T-panel)
      B1_FL01    → B1_FL1  (zero-padded on B1 T-panel)
    """
    # A1 ROI_00X → FLX
    m = re.match(r'^(A1)_ROI_0*(\d+)$', sample_id)
    if m:
        return f"{m.group(1)}_FL{int(m.group(2))}"

    # B1 zero-padded FL0X → FLX (only single-digit, i.e. FL01-FL09)
    m = re.match(r'^(B1)_FL0(\d)$', sample_id)
    if m:
        return f"{m.group(1)}_FL{m.group(2)}"

    return sample_id


def build_sample_id_mapping(sample_ids: list[str]) -> dict[str, str]:
    """Build old→new mapping for all sample_ids that need normalization.

    Returns dict with only entries that actually changed.
    """
    mapping = {}
    for sid in sample_ids:
        normed = normalize_sample_id(sid)
        if normed != sid:
            mapping[sid] = normed
    return mapping


def load_clinical(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load clinical CSV. Returns DataFrame indexed by slide_ID.

    Backward-compat shims applied to the DWS-annotated file so existing
    survival/figure scripts run without code changes:
      * Restore the legacy 'Progression free survival (y)' column (renamed to
        'Time to progression (y)' in the DWS update; values are asserted
        identical to 'Time to progression (y).1' on load).
      * Clean POD24_event values: strip whitespace, replace 'NA' with NaN,
        store as object dtype (NOT pandas 'string' dtype, which carries
        pd.NA sentinels that break np.where / boolean-mask consumers).
      * Emit a loud cohort-diff warning vs the pre-DWS file if it exists,
        so silently dropped patients (e.g. 06-25647, whose only ROI was a
        T2-only slide_ID not in the DWS file) are visible at load time.
    """
    path = Path(csv_path) if csv_path else CLINICAL_CSV
    df = pd.read_csv(path)

    # Shim 1: alias renamed PFS column. Assert the two TTP-named columns are
    # value-equal so we fail loudly if BCCA ever diverges them.
    if "Time to progression (y)" in df.columns and "Time to progression (y).1" in df.columns:
        a = df["Time to progression (y)"]
        b = df["Time to progression (y).1"]
        if not a.equals(b):
            mismatch = (a.fillna(-999) != b.fillna(-999)).sum()
            raise ValueError(
                f"load_clinical: 'Time to progression (y)' and 'Time to progression (y).1' "
                f"differ in {mismatch}/{len(df)} rows — PFS/TTP semantics may have diverged. "
                "Update the shim to alias explicitly instead of relying on value-equality."
            )
    if ("Progression free survival (y)" not in df.columns
            and "Time to progression (y)" in df.columns):
        df["Progression free survival (y)"] = df["Time to progression (y)"]

    # Shim 2: POD24_event cleanup using object dtype + np.nan (not string + pd.NA)
    if "POD24_event" in df.columns:
        cleaned = df["POD24_event"].astype(object)
        cleaned = cleaned.where(cleaned.notna(), np.nan)
        cleaned = cleaned.apply(lambda x: x.strip() if isinstance(x, str) else x)
        cleaned = cleaned.replace({"": np.nan, "NA": np.nan})
        df["POD24_event"] = cleaned

    # Shim 3: cohort-diff audit vs pre-DWS file. Loud warning lists slide_IDs
    # and patients dropped/added so reviewers see silent cohort shifts.
    if csv_path is None and path == CLINICAL_CSV and CLINICAL_CSV_PRE_DWS.exists():
        try:
            old = pd.read_csv(CLINICAL_CSV_PRE_DWS)
            old_sids = set(old["slide_ID"].dropna()) if "slide_ID" in old.columns else set()
            old_pts = set(old["Patient_ID"].dropna()) if "Patient_ID" in old.columns else set()
            new_sids = set(df["slide_ID"].dropna()) if "slide_ID" in df.columns else set()
            new_pts = set(df["Patient_ID"].dropna()) if "Patient_ID" in df.columns else set()
            dropped_sids = sorted(old_sids - new_sids)
            dropped_pts = sorted(old_pts - new_pts)
            if dropped_sids or dropped_pts:
                msg = (
                    f"load_clinical: DWS cohort differs from pre-DWS baseline: "
                    f"{len(dropped_sids)} slide_IDs and {len(dropped_pts)} "
                    f"patients dropped."
                )
                if dropped_pts:
                    msg += f" Patients dropped: {dropped_pts}."
                if dropped_sids:
                    msg += f" slide_IDs dropped: {dropped_sids}."
                warnings.warn(msg, stacklevel=2)
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(f"load_clinical cohort-diff audit failed: {e}", stacklevel=2)
    return df


def get_clinical_for_sample(
    sample_id: str, clinical_df: pd.DataFrame
) -> Optional[pd.Series]:
    """Look up clinical data for a (possibly unnormalized) sample_id."""
    normed = normalize_sample_id(sample_id)
    rows = clinical_df[clinical_df["slide_ID"] == normed]
    if len(rows) == 1:
        return rows.iloc[0]
    elif len(rows) > 1:
        # Multiple rows = serial biopsies; return first timepoint
        return rows.sort_values("T").iloc[0]
    return None


def add_clinical_to_obs(
    adata,
    clinical_df: Optional[pd.DataFrame] = None,
    columns: Optional[list[str]] = None,
) -> None:
    """Add clinical columns to adata.obs, handling naming normalization.

    Modifies adata in place. Creates a 'slide_ID' column with normalized IDs.
    """
    if clinical_df is None:
        clinical_df = load_clinical()

    # Normalize sample_ids
    adata.obs["slide_ID"] = [
        normalize_sample_id(s) for s in adata.obs["sample_id"]
    ]

    # Select columns to merge
    if columns is None:
        columns = [
            "Patient_ID", "AGE", "SEX",
            "Overall survival (y)", "CODE_OS",
            "Progression free survival (y)", "CODE_PFS",
            "ANN ARBOR STAGE", "FLIPI", "FLIPI.1",
            "Transformation", "T",
        ]

    # Merge
    keep_cols = [c for c in columns if c in clinical_df.columns]
    clin_subset = clinical_df[
        ["slide_ID"] + keep_cols
    ].drop_duplicates(subset="slide_ID", keep="first")

    adata.obs = adata.obs.merge(
        clin_subset, on="slide_ID", how="left", suffixes=("", "_clin")
    )
    # merge resets index; restore it
    adata.obs.index = adata.obs.index.astype(str)


def linkage_summary(
    sample_ids: list[str],
    clinical_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Summarize clinical linkage for a set of sample_ids."""
    if clinical_df is None:
        clinical_df = load_clinical()

    clin_ids = set(clinical_df["slide_ID"].dropna())
    normed = [normalize_sample_id(s) for s in sample_ids]
    matched = [n for n in normed if n in clin_ids]
    unmatched = [n for n in normed if n not in clin_ids]

    patients = clinical_df[clinical_df["slide_ID"].isin(matched)][
        "Patient_ID"
    ].nunique()

    return {
        "total_rois": len(sample_ids),
        "matched": len(matched),
        "unmatched": len(unmatched),
        "unique_patients": patients,
        "unmatched_ids": sorted(set(unmatched)),
    }
