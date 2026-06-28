#!/usr/bin/env python3
"""Add panel labels (a, b, c, d) to the 4 rows of fig_fdc_zone_vs_cd21_raw.png.

The figure is a 4-row × 3-column grid. Each row shows a different ROI.
Adds a bold panel letter at the top-left of each row.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


def add_labels(in_path: Path, out_path: Path):
    img = imread(str(in_path))
    h, w = img.shape[:2]

    # Figure sized to match the original aspect ratio
    fig_w = 14
    fig_h = fig_w * h / w
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")

    # 4 rows; add panel labels a, b, c, d at the left edge of each row
    # Rows are approximately evenly spaced after the title area (top ~3% is title)
    title_pad = 0.03
    row_height = (1.0 - title_pad) / 4

    for i, letter in enumerate(["a", "b", "c", "d"]):
        # y position in axes fraction (top of each row)
        # Note: axes fraction y=0 is bottom, y=1 is top
        y_top = 1.0 - title_pad - i * row_height
        ax.text(0.01, y_top - 0.005, f"$\\bf{{{letter}}}$",
                transform=ax.transAxes, fontsize=26,
                va="top", ha="left", color="black",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))

    fig.savefig(str(out_path), dpi=150, bbox_inches=None,
                facecolor="white", pad_inches=0)
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=300,
                bbox_inches=None, facecolor="white", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    root = Path("output/hypotheses_v8")
    in_path = root / "fig_fdc_zone_vs_cd21_raw.png"
    out_path = root / "fig_fdc_zone_vs_cd21_raw.png"
    add_labels(in_path, out_path)
