"""Supplementary figure: grade-associated changes in proliferation, myeloid
composition, and follicle architecture (DWS GRADE).

Panel layout (3 rows x 3 cols):
  a  Ki-67+ B-cell fraction (S-panel; positive control validating DWS grade)
  b  M1 Macrophages fraction (S-panel)
  c  S100A9+ Myeloid fraction (S-panel)
  d  Macrophages fraction (T-panel)
  e  Macrophage-rich zone fraction (T-panel UTAG compartment)
  f  Ripley's L for macrophages at 25 um (T-panel; lower = decompartmentalized)
  g  Mean follicle compactness (4 pi A / P^2; T-panel)
  h  Representative FOLL1 ROI (compact follicles, clustered macrophages)
  i  Representative FOLL3A ROI (irregular follicles, dispersed macrophages)

All boxplot panels: KW p-value with significance stars (*** q<0.001, ** q<0.01,
* q<0.05); patient-level data (ROIs averaged per patient before testing).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.ndimage import (
    label as cclabel, generate_binary_structure,
    binary_closing, binary_opening, convolve,
)
from scipy.stats import kruskal

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clinical_linkage import load_clinical


# ---------------------------------------------------------------------------
# Standardized font sizes (match other manuscript figures)
# ---------------------------------------------------------------------------
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22

GRADE_ORDER = ["FOLL1", "FOLL2", "FOLL3A"]
GRADE_ORD = {"FOLL1": 1, "FOLL2": 2, "FOLL3A": 3}
GRADE_COLORS = {"FOLL1": "#1f77b4", "FOLL2": "#ff7f0e", "FOLL3A": "#d62728"}

FOLL_T_PANEL = [
    "GC core",
    "Follicle core (GC/CD20hi/CXCR5hi)",
    "Follicle mantle (CXCR5hi)",
    "B cell follicle (CD20hi/CXCR5hi)",
    "B cell zone",
]
B_S_PANEL = ["B cells", "B cells (BCL2+)", "B cells (PAX5+)"]
MAC_T_PANEL = ["Macrophages", "Macrophages (GzmB+)"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def panel_label(ax, letter, x=-0.12, y=1.05):
    ax.text(
        x, y, f"$\\bf{{{letter}}}$",
        transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
        va="bottom", ha="left",
    )


def control_mask(sid_series):
    sl = sid_series.str.lower()
    return ~(
        sl.str.contains("tonsil|prostate|kidney|spleen|adrenal")
        | sid_series.str.contains("_Ton_|_Adr_")
    )


def stars(p):
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def kw_groups(pt, col):
    sub = pt.dropna(subset=[col])
    groups = [sub.loc[sub["GRADE"] == g, col].values for g in GRADE_ORDER]
    if any(len(g) < 3 for g in groups):
        return groups, np.nan
    vals = np.concatenate(groups)
    if len(np.unique(vals)) < 2:
        return groups, np.nan
    return groups, float(kruskal(*groups).pvalue)


def box_panel(ax, pt, col, ylabel, title, ypct=False):
    groups, p_kw = kw_groups(pt, col)
    bp = ax.boxplot(
        groups, labels=GRADE_ORDER, patch_artist=True, widths=0.55,
        showfliers=False, medianprops=dict(color="black", linewidth=1.6),
    )
    for patch, g in zip(bp["boxes"], GRADE_ORDER):
        patch.set_facecolor(GRADE_COLORS[g])
        patch.set_alpha(0.42)
        patch.set_edgecolor("black")
    rng = np.random.default_rng(42)
    for i, (vals, g) in enumerate(zip(groups, GRADE_ORDER)):
        x = rng.normal(i + 1, 0.06, len(vals))
        ax.scatter(
            x, vals, color=GRADE_COLORS[g], alpha=0.7, s=18,
            edgecolors="black", linewidths=0.3, zorder=3,
        )
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    p_str = f"KW p={p_kw:.3f} {stars(p_kw)}" if np.isfinite(p_kw) else "KW: insufficient data"
    ax.set_title(
        f"{title}  ({p_str})",
        fontsize=TITLE_SIZE, fontweight="medium", pad=8,
    )
    if ypct:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
        )
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(axis="y", alpha=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return p_kw


def plot_roi(ax, sid_k, x_arr, y_arr, ci, cti, sid_val, title,
             foll_set, mac_types):
    m_roi = sid_k == sid_val
    x = x_arr[m_roi]; y = y_arr[m_roi]
    cci = ci[m_roi]; cct = cti[m_roi]
    is_foll = np.isin(cci, foll_set)
    is_mac = np.isin(cct, mac_types)
    sub_colors = {
        "GC core": "#8b0000",
        "Follicle core (GC/CD20hi/CXCR5hi)": "#d62728",
        "Follicle mantle (CXCR5hi)": "#fc9272",
        "B cell follicle (CD20hi/CXCR5hi)": "#fdae6b",
        "B cell zone": "#fed976",
    }
    bg = ~is_foll & ~is_mac
    ax.scatter(x[bg], y[bg], c="#dddddd", s=1.0, alpha=0.45,
               edgecolors="none", rasterized=True)
    fc = np.array([sub_colors.get(c, "#dddddd") for c in cci])
    foll_only = is_foll & ~is_mac
    ax.scatter(x[foll_only], y[foll_only], c=fc[foll_only], s=2.0,
               alpha=0.78, edgecolors="none", rasterized=True)
    ax.scatter(x[is_mac], y[is_mac], c="#1ec5d6", s=10, alpha=0.95,
               edgecolors="black", linewidths=0.3, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")  # TMA cores are physically circular; preserve aspect
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="medium", pad=6)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def extract_metrics(s_panel_path, t_panel_path):
    clin = load_clinical()
    slide_to_grade = dict(zip(clin["slide_ID"], clin["GRADE"].astype(str)))
    slide_to_patient = dict(zip(clin["slide_ID"], clin["Patient_ID"]))

    # S-panel: Ki-67 in B cells + cell-type fractions
    a_s = ad.read_h5ad(s_panel_path)
    sid_s = a_s.obs["sample_id"].astype(str)
    keep_s = control_mask(sid_s).values
    sid_sk = sid_s.values[keep_s]
    ct_s = a_s.obs["cell_type"].astype(str).values[keep_s]
    ki67_idx = list(a_s.var_names).index("Ki-67")
    x = a_s.X[:, ki67_idx]
    ki67 = (x.toarray().flatten() if hasattr(x, "toarray") else np.asarray(x).flatten())[keep_s]

    # Build S-panel cell-type fractions and the ≥8000 typed-cells QC list first;
    # apply the same QC to the Ki-67 panel so all S-panel panels share denominators.
    df_ct_s = pd.DataFrame({"sample_id": sid_sk, "ct": ct_s})
    tot_s = df_ct_s.groupby("sample_id").size().rename("n_total").reset_index()
    xt_s = df_ct_s.pivot_table(index="sample_id", columns="ct",
                                aggfunc="size", fill_value=0)
    fracs_s = xt_s.div(xt_s.sum(axis=1), axis=0).reset_index().merge(tot_s, on="sample_id")
    fracs_s = fracs_s[fracs_s["n_total"] >= 8000].copy()
    qc_pass_s = set(fracs_s["sample_id"])

    is_b_s = np.isin(ct_s, B_S_PANEL)
    df_ki = pd.DataFrame({"sample_id": sid_sk[is_b_s], "ki67": ki67[is_b_s]})
    ki67_roi = (df_ki.groupby("sample_id")
                .agg(n_b=("ki67", "size"),
                     ki67_pct=("ki67", lambda v: float((v > 0.5).mean())))
                .reset_index())
    ki67_roi = ki67_roi[(ki67_roi["n_b"] >= 200)
                        & (ki67_roi["sample_id"].isin(qc_pass_s))].copy()

    # T-panel: cell-type fractions, compartment fractions, Ripley L, compactness
    a_t = ad.read_h5ad(t_panel_path)
    sid_t = a_t.obs["sample_id"].astype(str)
    keep_t = control_mask(sid_t).values
    sid_tk = sid_t.values[keep_t]
    ct_t = a_t.obs["cell_type"].astype(str).values[keep_t]
    comp_t = a_t.obs["compartment_name"].astype(str).values[keep_t]
    cx_t = a_t.obs["centroid_x"].values[keep_t]
    cy_t = a_t.obs["centroid_y"].values[keep_t]

    df_ct_t = pd.DataFrame({"sample_id": sid_tk, "ct": ct_t})
    tot_t = df_ct_t.groupby("sample_id").size().rename("n_total").reset_index()
    xt_t = df_ct_t.pivot_table(index="sample_id", columns="ct",
                                aggfunc="size", fill_value=0)
    fracs_t = xt_t.div(xt_t.sum(axis=1), axis=0).reset_index().merge(tot_t, on="sample_id")
    fracs_t = fracs_t[fracs_t["n_total"] >= 8000].copy()

    df_comp_t = pd.DataFrame({"sample_id": sid_tk, "comp": comp_t})
    xtc_t = df_comp_t.pivot_table(index="sample_id", columns="comp",
                                    aggfunc="size", fill_value=0)
    comp_frac_t = xtc_t.div(xtc_t.sum(axis=1), axis=0).reset_index()
    if "Macrophage-rich zone" in comp_frac_t.columns:
        fracs_t = fracs_t.merge(
            comp_frac_t[["sample_id", "Macrophage-rich zone"]].rename(
                columns={"Macrophage-rich zone": "MacRich_frac"}
            ),
            on="sample_id", how="left",
        )

    # Per-ROI Ripley L_Mac_r25 and follicle compactness; also count distinct
    # follicles so the example-picker can prefer multi-follicle architecture.
    px_size = 25.0
    min_foll_px = 50
    rip_rows = []
    arch_rows = []
    for sid_val in pd.unique(sid_tk):
        m_roi = sid_tk == sid_val
        if m_roi.sum() < 8000:
            continue
        x = cx_t[m_roi]; y = cy_t[m_roi]
        cti = ct_t[m_roi]; ci = comp_t[m_roi]
        is_mac = np.isin(cti, MAC_T_PANEL)
        if is_mac.sum() >= 50:
            minx, miny = float(x.min()), float(y.min())
            maxx, maxy = float(x.max()), float(y.max())
            A_roi = (maxx - minx) * (maxy - miny)
            pts = np.c_[x[is_mac], y[is_mac]]
            tree = cKDTree(pts)
            n_p = len(pts)
            counts = tree.query_ball_point(pts, r=25.0, return_length=True)
            K_r = float((counts.sum() - n_p) * A_roi / (n_p * (n_p - 1)))
            L_r = np.sqrt(K_r / np.pi) - 25.0
            rip_rows.append({"sample_id": sid_val, "L_Mac_r25": L_r})
        foll = np.isin(ci, FOLL_T_PANEL)
        if foll.sum() >= 200:
            minx, miny = float(x.min()), float(y.min())
            maxx, maxy = float(x.max()), float(y.max())
            nx_ = int(np.ceil((maxx - minx) / px_size)) + 2
            ny_ = int(np.ceil((maxy - miny) / px_size)) + 2
            grid = np.zeros((nx_, ny_), dtype=int)
            ix = ((x - minx) / px_size).astype(int)
            iy = ((y - miny) / px_size).astype(int)
            np.add.at(grid, (ix[foll], iy[foll]), 1)
            np.add.at(grid, (ix[~foll], iy[~foll]), -1)
            foll_mask = binary_opening(
                binary_closing(grid > 0, iterations=2), iterations=1
            )
            lbl, n_comp = cclabel(foll_mask,
                                   structure=generate_binary_structure(2, 2))
            comp_sizes = np.bincount(lbl.ravel())[1:] if n_comp > 0 else []
            comps = []
            real_sizes = []
            for k in range(1, n_comp + 1):
                if comp_sizes[k - 1] < min_foll_px:
                    continue
                m = (lbl == k)
                sz = m.sum()
                neigh = convolve(m.astype(int), np.ones((3, 3), int),
                                  mode="constant")
                perim = ((neigh < 9) & m).sum()
                if perim <= 0:
                    continue
                comps.append(((4 * np.pi * sz) / (perim ** 2), sz))
                real_sizes.append(sz)
            if comps:
                v, w = zip(*comps)
                v = np.array(v); w = np.array(w)
                comp_val = float(np.sum(v * w) / np.sum(w))
                arch_rows.append({"sample_id": sid_val,
                                  "compactness": comp_val,
                                  "n_follicles": len(real_sizes)})

    rip_df = pd.DataFrame(rip_rows)
    arch_df = pd.DataFrame(arch_rows)
    if len(rip_df):
        fracs_t = fracs_t.merge(rip_df, on="sample_id", how="left")
    if len(arch_df):
        fracs_t = fracs_t.merge(arch_df, on="sample_id", how="left")

    def attach_grade(d):
        d = d.copy()
        d["GRADE"] = d["sample_id"].map(slide_to_grade)
        d["Patient_ID"] = d["sample_id"].map(slide_to_patient)
        return d[d["GRADE"].isin(GRADE_ORDER)]

    ki67_roi = attach_grade(ki67_roi)
    fracs_s_g = attach_grade(fracs_s)
    fracs_t_g = attach_grade(fracs_t)

    pt_ki67 = ki67_roi.groupby(["Patient_ID", "GRADE"])["ki67_pct"].mean().reset_index()
    pt_s = (fracs_s_g.groupby(["Patient_ID", "GRADE"])
            [["M1 Macrophages", "Myeloid (S100A9+)"]]
            .mean().reset_index())
    t_cols = ["Macrophages", "MacRich_frac", "L_Mac_r25", "compactness"]
    t_cols = [c for c in t_cols if c in fracs_t_g.columns]
    pt_t = fracs_t_g.groupby(["Patient_ID", "GRADE"])[t_cols].mean().reset_index()

    return dict(
        pt_ki67=pt_ki67, pt_s=pt_s, pt_t=pt_t,
        roi_t=fracs_t_g,
        sid_tk=sid_tk, ct_t=ct_t, comp_t=comp_t, cx_t=cx_t, cy_t=cy_t,
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def make_figure(out_path, s_panel_path, t_panel_path):
    data = extract_metrics(s_panel_path, t_panel_path)

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(3, 3, figure=fig, wspace=0.32, hspace=0.45,
                   left=0.06, right=0.97, top=0.97, bottom=0.05)

    # Row 1: Ki-67 + myeloid expansion (S-panel)
    ax_a = fig.add_subplot(gs[0, 0])
    box_panel(ax_a, data["pt_ki67"], "ki67_pct",
              ylabel="Ki-67$^+$ B-cell fraction",
              title="Ki-67$^+$ B cells (S-panel)", ypct=True)
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    box_panel(ax_b, data["pt_s"], "M1 Macrophages",
              ylabel="M1 macrophage fraction",
              title="M1 macrophages (S-panel)", ypct=True)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[0, 2])
    box_panel(ax_c, data["pt_s"], "Myeloid (S100A9+)",
              ylabel="S100A9$^+$ myeloid fraction",
              title="S100A9$^+$ myeloid (S-panel)", ypct=True)
    panel_label(ax_c, "c")

    # Row 2: T-panel macrophage expansion + decompartmentalization
    ax_d = fig.add_subplot(gs[1, 0])
    box_panel(ax_d, data["pt_t"], "Macrophages",
              ylabel="Macrophage fraction",
              title="Macrophages (T-panel)", ypct=True)
    panel_label(ax_d, "d")

    ax_e = fig.add_subplot(gs[1, 1])
    box_panel(ax_e, data["pt_t"], "MacRich_frac",
              ylabel="Macrophage-rich zone fraction",
              title="Macrophage-rich zone (T-panel)", ypct=True)
    panel_label(ax_e, "e")

    ax_f = fig.add_subplot(gs[1, 2])
    box_panel(ax_f, data["pt_t"], "L_Mac_r25",
              ylabel="Ripley's L (Mac, 25 µm)",
              title="Macrophage spatial\nclustering (T-panel)")
    panel_label(ax_f, "f")

    # Row 3: follicle compactness + representative ROIs
    ax_g = fig.add_subplot(gs[2, 0])
    box_panel(ax_g, data["pt_t"], "compactness",
              ylabel="Mean compactness (4πA/P²)",
              title="Follicle compactness (T-panel)")
    panel_label(ax_g, "g")

    # Pick representative ROIs by composite ranking on the grade signals
    # (mac fraction + Ripley clustering + compactness), so the spatial contrast
    # reflects the quantitative findings in panels (d)-(g) — not just compactness.
    # For FOLL1 we additionally prefer ROIs with multiple discrete follicles,
    # so the example shows the classical multi-follicle architecture rather
    # than one giant follicular mass.
    roi_arch = data["roi_t"][
        data["roi_t"]["compactness"].notna()
        & data["roi_t"]["L_Mac_r25"].notna()
        & data["roi_t"]["Macrophages"].notna()
    ].copy()
    has_nfoll = "n_follicles" in roi_arch.columns

    def best_example(pool, grade):
        if grade == "FOLL1":
            # Compact + clustered macs + FEW macs; prefer multi-follicle
            rank = (pool["compactness"].rank(ascending=False)
                    + pool["L_Mac_r25"].rank(ascending=False)
                    + pool["Macrophages"].rank(ascending=True))
            if has_nfoll:
                rank = rank + pool["n_follicles"].rank(ascending=False)
        else:
            # Irregular + dispersed macs + MANY macs
            rank = (pool["compactness"].rank(ascending=True)
                    + pool["L_Mac_r25"].rank(ascending=True)
                    + pool["Macrophages"].rank(ascending=False))
        return pool.assign(_rank=rank).sort_values("_rank").iloc[0]["sample_id"]

    f1_pool = roi_arch[roi_arch["GRADE"] == "FOLL1"]
    f3_pool = roi_arch[roi_arch["GRADE"] == "FOLL3A"]
    # Restrict FOLL1 pool to multi-follicle ROIs and require both individual
    # follicles to be reasonably compact (above the FOLL1 median); this picks
    # examples that show classical multi-follicle architecture AND demonstrate
    # the compactness signal in panel (g).
    if has_nfoll:
        multi_f1 = f1_pool[
            (f1_pool["n_follicles"] >= 2)
            & (f1_pool["compactness"] >= f1_pool["compactness"].median())
            & (f1_pool["Macrophages"] <= f1_pool["Macrophages"].median())
        ]
        if len(multi_f1) >= 1:
            f1_pool = multi_f1
    f1_pick = best_example(f1_pool, "FOLL1") if len(f1_pool) else None
    f3_pick = best_example(f3_pool, "FOLL3A") if len(f3_pool) else None

    ax_h = fig.add_subplot(gs[2, 1])
    if f1_pick:
        plot_roi(ax_h, data["sid_tk"], data["cx_t"], data["cy_t"],
                 data["comp_t"], data["ct_t"], f1_pick,
                 f"FOLL1 example ({f1_pick})", FOLL_T_PANEL, MAC_T_PANEL)
    panel_label(ax_h, "h")

    ax_i = fig.add_subplot(gs[2, 2])
    if f3_pick:
        plot_roi(ax_i, data["sid_tk"], data["cx_t"], data["cy_t"],
                 data["comp_t"], data["ct_t"], f3_pick,
                 f"FOLL3A example ({f3_pick})", FOLL_T_PANEL, MAC_T_PANEL)
    panel_label(ax_i, "i")

    # Horizontal legend below panels (h) and (i); does not overlap their content
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#8b0000",
               markersize=10, label="GC core"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=10, label="Follicle core"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fc9272",
               markersize=10, label="Mantle"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fdae6b",
               markersize=10, label="B-cell follicle"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fed976",
               markersize=10, label="B-cell zone"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1ec5d6",
               markeredgecolor="black", markersize=10, label="Macrophages"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#dddddd",
               markersize=10, label="Other"),
    ]
    # Anchor the legend to ax_h's axes coordinates (negative y = below panel).
    # This matches the convention in fig_macrophage_biology.py and is robust
    # to bbox_inches='tight' cropping (figure-level legend coords get squashed
    # to the bottom of the canvas when nothing else sits beneath them).
    ax_h.legend(
        handles=legend_elements, loc="upper center",
        bbox_to_anchor=(1.05, -0.05), ncol=7, fontsize=LEGEND_SIZE,
        frameon=True, framealpha=0.9, columnspacing=1.8, handletextpad=0.4,
    )

    out_pdf = Path(out_path).with_suffix(".pdf")
    out_png = Path(out_path).with_suffix(".png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-panel",
                        default="output/all_TMA_S_utag_ct_merged.h5ad")
    parser.add_argument("--t-panel",
                        default="output/all_TMA_T_utag_ct_merged.h5ad")
    parser.add_argument("--out",
                        default="output/hypotheses_v8/fig_grade_supplementary.png")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make_figure(args.out, args.s_panel, args.t_panel)


if __name__ == "__main__":
    main()
