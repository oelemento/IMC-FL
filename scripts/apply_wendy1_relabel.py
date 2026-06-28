"""Wendy #1: rename "Cytotoxic (GzmB+)" → "Macrophages (GzmB+)" and
merge "B cells (weak CD20)" → "Low quality / Unassigned" on the T-panel h5ads.

Label-only swap — no re-gating. Verified backups exist as *_pre_wendy1.h5ad.bak.
"""
import sys
import anndata as ad

RENAMES = {
    "Cytotoxic (GzmB+)": "Macrophages (GzmB+)",
    "B cells (weak CD20)": "Low quality / Unassigned",
}

PATHS = [
    "output/all_TMA_T_global_v8.h5ad",
    "output/all_TMA_T_utag.h5ad",
    "output/all_TMA_T_utag_ct_merged.h5ad",
]


def relabel(path):
    a = ad.read_h5ad(path)
    col = a.obs["cell_type"].astype(str)
    before = col.value_counts()
    for old, new in RENAMES.items():
        col = col.replace(old, new)
    a.obs["cell_type"] = col.astype("category")
    after = a.obs["cell_type"].value_counts()
    print(f"\n{path}")
    for old, new in RENAMES.items():
        n_before = int(before.get(old, 0))
        n_after_old = int(after.get(old, 0))
        n_after_new = int(after.get(new, 0))
        print(f"  {old!r}: {n_before} → {n_after_old} (merged into {new!r}, now {n_after_new})")
    a.write_h5ad(path)
    print(f"  saved: {path}")


if __name__ == "__main__":
    for p in PATHS:
        relabel(p)
