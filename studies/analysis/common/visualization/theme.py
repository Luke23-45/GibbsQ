"""
Visual theme tokens and matplotlib RC configuration.
"""
import matplotlib as mpl

GIBBSQ_COLORS = {
    "primary": "#332288",
    "secondary": "#117733",
    "accent": "#CC6677",
    "background": "#FFFFFF",
    "grid": "#EEEEEE",
}

def apply_theme():
    """Apply the publication-quality theme to matplotlib."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.prop_cycle": mpl.cycler(color=list(GIBBSQ_COLORS.values())),
    })
