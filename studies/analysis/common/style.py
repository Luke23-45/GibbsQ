"""
Publication-grade matplotlib style system for the GibbsQ thesis.

Design principles
-----------------
* **Premium feel**: generous whitespace, refined typography, subtle grid.
* **Colorblind-safe**: palette verified with Coblis & Viz Palette tools.
* **Thesis-compliant**: Type-42 fonts, serif family (Times), correct sizes.
* **Consistent**: every figure uses the same design tokens.

Usage::

    from studies.analysis.common.style import apply_thesis_style, create_figure, save_figure

    with apply_thesis_style():
        fig, ax = create_figure(width="single")
        ...
        save_figure(fig, Path("output/my_figure"))
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# Column widths (inches) for thesis format
_COL_WIDTH   = 4.5     # single column
_TEXT_WIDTH  = 6.5      # full text width
_DPI = 300

# Font sizes (pt)
_FONT_SIZE_TITLE  = 13
_FONT_SIZE_LABEL  = 12
_FONT_SIZE_TICK   = 10
_FONT_SIZE_LEGEND = 10
_FONT_SIZE_ANNOT  = 9


# Colorblind-safe palette for status bars
STATUS_PALETTE = {
    "PASS":   "#2ca02c",   # green
    "FAIL":   "#d62728",   # red
    "WARN":   "#ff7f0e",   # orange
}

# Hypothesis-specific palette
HYPOTHESIS_PALETTE = {
    "h1": "#332288",   # indigo
    "h2": "#117733",   # forest green
    "h3": "#882255",   # wine
    "h4": "#EE3377",   # magenta-pink
    "h5": "#88CCEE",   # sky blue
}


THESIS_RC: dict = {
    # Fonts
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":            _FONT_SIZE_LABEL,
    "mathtext.fontset":     "cm",

    # Axes
    "axes.titlesize":       _FONT_SIZE_TITLE,
    "axes.labelsize":       _FONT_SIZE_LABEL,
    "axes.titleweight":     "normal",
    "axes.titlepad":        8,
    "axes.labelpad":        5,
    "axes.linewidth":       0.6,
    "axes.edgecolor":       "#333333",
    "axes.facecolor":       "white",
    "axes.grid":            True,
    "axes.grid.which":      "major",
    "axes.axisbelow":       True,
    "axes.spines.top":      False,
    "axes.spines.right":    False,

    # Grid
    "grid.color":           "#E0E0E0",
    "grid.linewidth":       0.4,
    "grid.alpha":           0.7,
    "grid.linestyle":       "--",

    # Ticks
    "xtick.labelsize":      _FONT_SIZE_TICK,
    "ytick.labelsize":      _FONT_SIZE_TICK,
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.pad":      3,
    "ytick.major.pad":      3,

    # Legend
    "legend.fontsize":      _FONT_SIZE_LEGEND,
    "legend.frameon":        True,
    "legend.framealpha":     0.92,
    "legend.edgecolor":     "#CCCCCC",
    "legend.fancybox":      True,
    "legend.borderpad":     0.5,
    "legend.handlelength":  1.8,
    "legend.handletextpad": 0.5,

    # Lines
    "lines.linewidth":      1.5,
    "lines.markersize":     5,

    # Figure
    "figure.facecolor":     "white",
    "figure.dpi":           _DPI,
    "figure.constrained_layout.use": True,

    # Saving
    "savefig.dpi":          _DPI,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.08,
    "savefig.facecolor":    "white",
    "savefig.transparent":  False,

    # PDF
    "pdf.fonttype":         42,
    "ps.fonttype":          42,

    # Error bars
    "errorbar.capsize":     2.5,
}


@contextlib.contextmanager
def apply_thesis_style():
    """Context manager that applies the thesis publication style.

    All figures created within this block inherit the style. The
    previous style is restored on exit.

    Example::

        with apply_thesis_style():
            fig, ax = create_figure(width="single")
            ax.plot(x, y)
    """
    with mpl.rc_context(THESIS_RC):
        yield


def create_figure(
    width: str = "single",
    aspect: float = 0.618,
    nrows: int = 1,
    ncols: int = 1,
    height_override: Optional[float] = None,
    squeeze: bool = True,
) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, np.ndarray]]:
    """Create a figure with thesis-appropriate dimensions.

    Parameters
    ----------
    width : {"single", "double"}
        Column width preset.
    aspect : float
        Height/width ratio (default: golden ratio).
    nrows, ncols : int
        Subplot grid dimensions.
    height_override : float, optional
        Explicit height in inches (overrides aspect).
    squeeze : bool
        Whether to squeeze singleton dimensions.

    Returns
    -------
    (fig, axes)
    """
    w = _TEXT_WIDTH if width == "double" else _COL_WIDTH
    h = height_override if height_override else w * aspect

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(w, h),
        squeeze=squeeze,
    )

    return fig, axes


def save_figure(
    fig: plt.Figure,
    path: Path,
    formats: Sequence[str] = ("pdf", "png"),
    close: bool = True,
) -> None:
    """Save a figure to PDF and PNG.

    Parameters
    ----------
    fig : matplotlib Figure
    path : Path
        Base path (without extension). e.g. ``output/my_figure``
    formats : sequence of str
        File formats to export.
    close : bool
        Close the figure after saving to free memory.
    """
    for fmt in formats:
        out_path = path.parent / f"{path.name}.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), format=fmt)
        logger.info("Saved figure: %s", out_path)

    if close:
        plt.close(fig)


def get_status_color(status: str) -> str:
    """Get the color for a PASS/FAIL status, with fallback."""
    return STATUS_PALETTE.get(status, "#666666")
