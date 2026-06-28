"""Shared figure style constants for all paper figures.

All figure scripts should import from here:
    from figure_style import STYLE, panel_label, apply_style

Direct-render convention: all panels rendered into a single GridSpec figure
(no PNG cache compositing). Fonts are final size, no scaling artifacts.
"""
import matplotlib.pyplot as plt

# Font sizes for ~20" wide composite figures (10 panels)
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 13
ANNOT_SIZE = 14
PANEL_LABEL_SIZE = 22

STYLE = {
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
}


def apply_style():
    """Apply standard style to all subsequent plots."""
    plt.rcParams.update(STYLE)


def panel_label(ax, letter, fontsize=PANEL_LABEL_SIZE):
    """Add bold panel label (no parentheses) at top-left of axes."""
    ax.text(-0.02, 1.02, f"$\\bf{{{letter}}}$", transform=ax.transAxes,
            fontsize=fontsize, va="bottom")


def save_figure(fig, path, dpi=200):
    """Save as PNG, PDF (vectorized), and compressed PDF (for review/ingestion)."""
    import subprocess
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_path = str(path).replace(".png", ".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Create compressed review copy via ghostscript
    review_path = pdf_path.replace(".pdf", "_review.pdf")
    try:
        subprocess.run([
            "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/screen", "-dNOPAUSE", "-dQUIET", "-dBATCH",
            f"-sOutputFile={review_path}", pdf_path,
        ], check=True)
        import os
        sz_full = os.path.getsize(pdf_path) / 1024
        sz_review = os.path.getsize(review_path) / 1024
        print(f"  Saved: {path} + {pdf_path} ({sz_full:.0f}KB) + {review_path} ({sz_review:.0f}KB)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  Saved: {path} + {pdf_path} (gs compression skipped)")
