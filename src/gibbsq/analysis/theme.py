"""
Publication-quality theme management for matplotlib charts.

Provides two themes:
- 'dark': Dark background for presentations/screens
- 'publication': White background for papers/reports

Features:
- Colorblind-safe color palettes
- High-DPI output for print quality
- Serif fonts for academic standards
- Vector format support
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib import font_manager

__all__ = [
    "apply_theme",
    "get_publication_params",
    "get_dark_params",
    "PublicationTheme",
    "THEMES",
]


# ──────────────────────────────────────────────────────────────
# Colorblind-Safe Palettes
# ──────────────────────────────────────────────────────────────

# Okabe-Ito colorblind-safe palette (optimized for common color vision deficiencies)
OKABE_ITO = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray
]

# Wong colorblind-safe palette (alternative)
WONG_PALETTE = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Purple
]


# ──────────────────────────────────────────────────────────────
# Theme Definitions
# ──────────────────────────────────────────────────────────────

@dataclass
class ThemeConfig:
    """Configuration for a matplotlib theme."""
    name: str
    figure_facecolor: str
    axes_facecolor: str
    text_color: str
    axes_edgecolor: str
    grid_color: str
    grid_alpha: float
    font_family: str
    font_serif: List[str]
    font_size_title: int
    font_size_label: int
    font_size_tick: int
    font_size_legend: int
    linewidth_axes: float
    linewidth_grid: float
    linewidth_lines: float
    dpi_figure: int
    dpi_save: int
    color_palette: List[str]
    

# Dark theme (for presentations/screens)
DARK_THEME = ThemeConfig(
    name="dark",
    figure_facecolor="#1a1a2e",
    axes_facecolor="#1a1a2e",
    text_color="white",
    axes_edgecolor="white",
    grid_color="white",
    grid_alpha=0.15,
    font_family="sans-serif",
    font_serif=["Inter", "Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"],
    font_size_title=14,
    font_size_label=12,
    font_size_tick=10,
    font_size_legend=10,
    linewidth_axes=1.2,
    linewidth_grid=0.5,
    linewidth_lines=1.5,
    dpi_figure=150,
    dpi_save=300,
    color_palette=OKABE_ITO,
)


# Publication theme (for papers/reports)
PUBLICATION_THEME = ThemeConfig(
    name="publication",
    figure_facecolor="white",
    axes_facecolor="white",
    text_color="black",
    axes_edgecolor="black",
    grid_color="gray",
    grid_alpha=0.3,
    font_family="serif",
    font_serif=["Times New Roman", "Computer Modern Roman", "DejaVu Serif", "serif"],
    font_size_title=12,
    font_size_label=11,
    font_size_tick=10,
    font_size_legend=9,
    linewidth_axes=0.8,
    linewidth_grid=0.5,
    linewidth_lines=1.2,
    dpi_figure=150,
    dpi_save=600,  # High DPI for print
    color_palette=OKABE_ITO,  # Colorblind-safe
)

THEMES = {
    "dark": DARK_THEME,
    "publication": PUBLICATION_THEME,
}

_current_theme: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Theme Application
# ──────────────────────────────────────────────────────────────

def apply_theme(theme: str = "publication") -> None:
    """
    Apply a matplotlib theme for all subsequent plots.
    
    Args:
        theme: Theme name ('dark' or 'publication')
    """
    global _current_theme
    
    if theme not in THEMES:
        raise ValueError(f"Unknown theme '{theme}'. Available: {list(THEMES.keys())}")
    
    _current_theme = theme
    config = THEMES[theme]
    
    # Get available fonts
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    
    # Build font stack
    if config.font_family == "serif":
        available_fonts = [f for f in config.font_serif if f in installed_fonts]
        font_stack = available_fonts + ["DejaVu Serif", "serif"]
    else:
        available_fonts = [f for f in config.font_serif if f in installed_fonts]
        font_stack = available_fonts + ["DejaVu Sans", "sans-serif"]
    
    # Apply rcParams
    params = {
        # Figure settings
        "figure.facecolor": config.figure_facecolor,
        "figure.dpi": config.dpi_figure,
        "figure.figsize": (8, 5),
        
        # Axes settings
        "axes.facecolor": config.axes_facecolor,
        "axes.edgecolor": config.axes_edgecolor,
        "axes.linewidth": config.linewidth_axes,
        "axes.titlesize": config.font_size_title,
        "axes.labelsize": config.font_size_label,
        "axes.labelcolor": config.text_color,
        "axes.titlecolor": config.text_color,
        "axes.grid": True,
        
        # Grid settings
        "grid.color": config.grid_color,
        "grid.alpha": config.grid_alpha,
        "grid.linestyle": "--",
        "grid.linewidth": config.linewidth_grid,
        
        # Tick settings
        "xtick.labelsize": config.font_size_tick,
        "ytick.labelsize": config.font_size_tick,
        "xtick.color": config.text_color,
        "ytick.color": config.text_color,
        
        # Font settings
        "font.family": config.font_family,
        "font.size": config.font_size_tick,
        
        # Legend settings
        "legend.fontsize": config.font_size_legend,
        "legend.frameon": True if theme == "publication" else False,
        "legend.edgecolor": config.axes_edgecolor,
        "legend.facecolor": config.figure_facecolor,
        
        # Line settings
        "lines.linewidth": config.linewidth_lines,
        "lines.markersize": 6,
        
        # Save settings
        "savefig.dpi": config.dpi_save,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": config.figure_facecolor,
        "savefig.edgecolor": config.figure_facecolor,
        
        # Text settings
        "text.color": config.text_color,
        
        # Image settings
        "image.cmap": "viridis",
    }
    
    # Add font stack only for the matching family
    if config.font_family == "serif":
        params["font.serif"] = font_stack
    else:
        params["font.sans-serif"] = font_stack
    
    plt.rcParams.update(params)
    
    # Set color cycle
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=config.color_palette)


def get_publication_params() -> Dict:
    """Get publication theme parameters as dictionary."""
    config = PUBLICATION_THEME
    return {
        "figure.facecolor": config.figure_facecolor,
        "axes.facecolor": config.axes_facecolor,
        "text.color": config.text_color,
        "font.family": config.font_family,
        "font.serif": config.font_serif,
        "savefig.dpi": config.dpi_save,
    }


def get_dark_params() -> Dict:
    """Get dark theme parameters as dictionary."""
    config = DARK_THEME
    return {
        "figure.facecolor": config.figure_facecolor,
        "axes.facecolor": config.axes_facecolor,
        "text.color": config.text_color,
        "font.family": config.font_family,
        "font.sans-serif": config.font_serif,
        "savefig.dpi": config.dpi_save,
    }


def get_current_theme() -> Optional[str]:
    """Get the currently applied theme name."""
    return _current_theme


# ──────────────────────────────────────────────────────────────
# Theme Class for Advanced Usage
# ──────────────────────────────────────────────────────────────

class PublicationTheme:
    """
    Publication theme manager for fine-grained control.
    
    Usage:
        theme = PublicationTheme()
        colors = theme.get_color_palette()
        params = theme.get_rc_params()
    """
    
    def __init__(self, name: str = "publication"):
        if name not in THEMES:
            raise ValueError(f"Unknown theme '{name}'")
        self.config = THEMES[name]
    
    def get_color_palette(self) -> List[str]:
        """Get the colorblind-safe color palette."""
        return self.config.color_palette.copy()
    
    def get_rc_params(self) -> Dict:
        """Get all rcParams for this theme."""
        return {
            "figure.facecolor": self.config.figure_facecolor,
            "axes.facecolor": self.config.axes_facecolor,
            "axes.edgecolor": self.config.axes_edgecolor,
            "axes.linewidth": self.config.linewidth_axes,
            "font.family": self.config.font_family,
            "savefig.dpi": self.config.dpi_save,
        }
    
    def apply(self) -> None:
        """Apply this theme."""
        apply_theme(self.config.name)
    
    def get_contrasting_color(self) -> str:
        """Get a color that contrasts with the background."""
        return self.config.text_color


# ──────────────────────────────────────────────────────────────
# Context Manager for Temporary Theme
# ──────────────────────────────────────────────────────────────

class temporary_theme:
    """
    Context manager to temporarily apply a theme.
    
    Usage:
        with temporary_theme('publication'):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            fig.savefig('output.png')
    """
    
    def __init__(self, theme: str):
        self.theme = theme
        self._original_rcparams = None
    
    def __enter__(self):
        self._original_rcparams = plt.rcParams.copy()
        apply_theme(self.theme)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_rcparams is not None:
            plt.rcParams.update(self._original_rcparams)
        return False


# ──────────────────────────────────────────────────────────────
# Chart-Style Integration (bridge to chart_styles module)
# ──────────────────────────────────────────────────────────────

def get_chart_style_for_theme(chart_type, theme: Optional[str] = None):
    """Get a chart style spec merged with the current theme settings.

    This is the primary entry point for plot functions that need
    chart-type-aware styling.

    Parameters
    ----------
    chart_type : ChartType
        The chart category enum value.
    theme : str, optional
        Theme name.  Falls back to the currently applied theme.

    Returns
    -------
    ChartStyleSpec
        A fresh copy of the style spec with theme overrides applied.
    """
    from gibbsq.analysis.chart_styles import get_chart_style

    if theme is None:
        theme = get_current_theme() or "publication"
    return get_chart_style(chart_type, theme=theme)
