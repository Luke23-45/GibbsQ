"""
Chart-type-aware styling for the GibbsQ visualization system.

Provides per-chart-type style specifications so that heatmaps, scatter
plots, bar charts, training dashboards, and raincloud plots each receive
optimized colormaps, markers, grid behaviour, and layout settings.

Architecture
------------
The *theme* (dark / publication) controls global aesthetics: background
colour, fonts, DPI.  A *ChartStyleSpec* controls per-chart semantics:
colormap choice, marker shape, line widths, annotation sizes, subplot
layout.  The two are composed at render time via ``get_chart_style()``.

Colour Strategy
---------------
- **Qualitative** (Okabe-Ito): for series with categorical identity
  (line plots, multi-policy bars).
- **Diverging** (RdBu_r / RdYlGn): for heatmaps with a semantic
  midpoint (drift=0, improvement=1.0).
- **Sequential** (viridis / magma): for continuous encoding of a third
  dimension (Z-score gradient on a scatter plot).
- **Two-colour contrast**: for paired comparisons (GibbsQ vs Neural).

All qualitative palettes are colorblind-safe (Okabe-Ito).  All diverging
maps are perceptually uniform and print safely in greyscale.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ChartType",
    "ChartStyleSpec",
    "CHART_STYLES",
    "get_chart_style",
    "resolve_colormap",
    "get_semantic_colors",
]

class ChartType(Enum):
    """Enumeration of distinct chart categories in the project.

    Each value maps to a ``ChartStyleSpec`` in ``CHART_STYLES``.
    """

    SCATTER = "scatter"
    """Per-parameter scatter with identity line (gradient check, drift
    vs norm)."""

    HEATMAP_DIVERGING = "heatmap_diverging"
    """Diverging-colormap heatmap with contour overlay (drift landscape,
    generalisation sweep)."""

    BAR_COMPARISON = "bar_comparison"
    """Bar chart comparing discrete conditions on a single metric
    (baselines, ablation)."""

    LINE_SERIES = "line_series"
    """Multi-series line plot over a shared x-axis (alpha sweep,
    critical load)."""

    TRAINING_DASHBOARD = "training_dashboard"
    """Multi-panel training monitor (REINFORCE: reward, loss, critic,
    gradients)."""

    RAINCLOUD = "raincloud"
    """Raincloud plot: half-violin + jittered scatter + mini-box
    (stats_bench)."""

    STRESS_DASHBOARD = "stress_dashboard"
    """Three-panel small-multiples dashboard with heterogeneous panel
    types (stress_test)."""

    STATUS_TABLE = "status_table"
    """Documentation-only status table (check_configs). No chart
    generated."""

# Okabe-Ito colorblind-safe palette (8 colours)
OKABE_ITO: List[str] = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray
]

# Semantic two-colour pair (maximum perceptual contrast)
PAIR_GIBBSQ = "#0072B2"   # Blue  – GibbsQ / control
PAIR_NEURAL = "#D55E00"   # Vermillion – Neural / treatment

# Ablation degradation cascade (best → worst)
ABLATION_CASCADE: List[str] = [
    "#009E73",  # Green – Full model (best)
    "#E69F00",  # Orange – Ablation variant 1
    "#D55E00",  # Vermillion – Ablation variant 2
    "#999999",  # Gray – Baseline (worst)
]

TRAINING_PRIMARY = "#0072B2"    # Blue – primary metric (reward / PI)
TRAINING_SECONDARY = "#D55E00"  # Vermillion – secondary (loss)
TRAINING_CRITIC = "#009E73"     # Green – critic quality
TRAINING_GRADIENT = "#E69F00"   # Orange – gradient norms
TRAINING_ENTROPY = "#CC79A7"    # Purple – entropy

STRESS_SCALING_COLOR = "#0072B2"     # Blue
STRESS_CRITICAL_COLORS = [PAIR_NEURAL, PAIR_GIBBSQ]  # dual lines
STRESS_HETERO_COLORS: List[str] = list(OKABE_ITO[:4])

@dataclass
class ChartStyleSpec:
    """Per-chart-type visual specification.

    Fields that are ``None`` signal "use the global theme default".
    """

    colormap: Optional[str] = None
    """Matplotlib colormap name for continuous colour encoding.
    Set for heatmaps (``RdBu_r``, ``RdYlGn``) and scatter Z-dim
    (``viridis``).  ``None`` means "not applicable"."""

    colormap_center: Optional[float] = None
    """Normalisation midpoint for diverging colormaps (e.g. 0.0 for
    drift, 1.0 for improvement ratio)."""

    palette: List[str] = field(default_factory=lambda: OKABE_ITO.copy())
    """Qualitative discrete palette for categorical series."""

    marker_style: str = "o"
    marker_size: float = 6.0
    line_width: float = 1.5
    line_alpha: float = 0.9

    fill_alpha: float = 0.3
    """Alpha for ``fill_between``, violin interiors, etc."""

    grid_visible: bool = True
    grid_style: str = "--"
    grid_alpha: float = 0.25

    annotation_fontsize: int = 9
    value_format: str = ".2f"
    """Format string for cell/bar annotations (e.g. ``'.2f'``)."""

    error_bar_capsize: float = 4.0

    figsize: Tuple[float, float] = (8, 5)
    subplot_layout: Optional[Tuple[int, int]] = None
    """``(nrows, ncols)`` for multi-panel figures."""

    use_seaborn: bool = False
    """Whether this chart type benefits from seaborn functions."""

    seaborn_style: Optional[str] = None
    """Seaborn context style ('paper', 'notebook', 'talk', 'poster')."""

# Registry: ChartType → ChartStyleSpec

CHART_STYLES: Dict[ChartType, ChartStyleSpec] = {

    ChartType.SCATTER: ChartStyleSpec(
        colormap="viridis",            # continuous Z-score colour dim
        colormap_center=None,          # sequential, no midpoint
        palette=OKABE_ITO.copy(),
        marker_style="o",
        marker_size=20.0,              # scatter markers need to be visible
        line_width=2.0,                # identity/bound reference line
        line_alpha=0.7,
        fill_alpha=0.15,
        grid_visible=True,
        grid_style="--",
        grid_alpha=0.20,
        annotation_fontsize=9,
        value_format=".4f",            # high precision for gradient values
        error_bar_capsize=0.0,
        figsize=(7, 6),
    ),

    ChartType.HEATMAP_DIVERGING: ChartStyleSpec(
        colormap="RdBu_r",            # default diverging; gen_sweep overrides to RdYlGn
        colormap_center=0.0,           # default centre; gen_sweep overrides to 1.0
        palette=OKABE_ITO.copy(),
        marker_style="none",
        marker_size=0.0,
        line_width=2.0,                # contour line width
        line_alpha=1.0,
        fill_alpha=1.0,                # imshow is fully opaque
        grid_visible=False,            # no grid on heatmaps
        grid_style="-",
        grid_alpha=0.0,
        annotation_fontsize=10,        # cell text annotations
        value_format=".2f",
        error_bar_capsize=0.0,
        figsize=(7, 6),
    ),

    ChartType.BAR_COMPARISON: ChartStyleSpec(
        colormap=None,
        colormap_center=None,
        palette=ABLATION_CASCADE.copy(),
        marker_style="none",
        marker_size=0.0,
        line_width=1.0,                # bar edge width
        line_alpha=0.85,
        fill_alpha=0.85,
        grid_visible=True,
        grid_style="--",
        grid_alpha=0.25,
        annotation_fontsize=9,
        value_format=".2f",
        error_bar_capsize=5.0,         # visible error bar caps
        figsize=(10, 5),
    ),

    ChartType.LINE_SERIES: ChartStyleSpec(
        colormap=None,
        colormap_center=None,
        palette=OKABE_ITO.copy(),
        marker_style="o",
        marker_size=6.0,
        line_width=2.0,
        line_alpha=0.9,
        fill_alpha=0.15,               # for shaded CI regions
        grid_visible=True,
        grid_style="--",
        grid_alpha=0.25,
        annotation_fontsize=9,
        value_format=".2f",
        error_bar_capsize=3.0,
        figsize=(9, 6),
    ),

    ChartType.TRAINING_DASHBOARD: ChartStyleSpec(
        colormap=None,
        colormap_center=None,
        palette=[
            TRAINING_PRIMARY,
            TRAINING_SECONDARY,
            TRAINING_CRITIC,
            TRAINING_GRADIENT,
            TRAINING_ENTROPY,
        ],
        marker_style="none",           # dense time-series, no markers
        marker_size=0.0,
        line_width=1.5,
        line_alpha=0.85,
        fill_alpha=0.12,               # faint raw data ribbon
        grid_visible=True,
        grid_style=":",                # dotted grid for dashboards
        grid_alpha=0.20,
        annotation_fontsize=8,
        value_format=".3f",
        error_bar_capsize=0.0,
        figsize=(14, 10),
        subplot_layout=(2, 2),
    ),

    ChartType.RAINCLOUD: ChartStyleSpec(
        colormap=None,
        colormap_center=None,
        palette=[PAIR_GIBBSQ, PAIR_NEURAL],
        marker_style="o",
        marker_size=4.0,               # jittered scatter points
        line_width=1.0,
        line_alpha=0.8,
        fill_alpha=0.35,               # half-violin fill
        grid_visible=True,
        grid_style="--",
        grid_alpha=0.15,
        annotation_fontsize=9,
        value_format=".3f",
        error_bar_capsize=0.0,
        figsize=(8, 6),
        use_seaborn=True,
        seaborn_style="paper",
    ),

    ChartType.STRESS_DASHBOARD: ChartStyleSpec(
        colormap="magma",              # sequential for scaling panel
        colormap_center=None,
        palette=OKABE_ITO.copy(),
        marker_style="o",
        marker_size=5.0,
        line_width=1.8,
        line_alpha=0.9,
        fill_alpha=0.20,
        grid_visible=True,
        grid_style="--",
        grid_alpha=0.20,
        annotation_fontsize=8,
        value_format=".2f",
        error_bar_capsize=4.0,
        figsize=(16, 5),
        subplot_layout=(1, 3),
    ),

    ChartType.STATUS_TABLE: ChartStyleSpec(
        colormap=None,
        colormap_center=None,
        palette=[],
        marker_style="none",
        marker_size=0.0,
        line_width=0.0,
        line_alpha=0.0,
        fill_alpha=0.0,
        grid_visible=False,
        grid_style="-",
        grid_alpha=0.0,
        annotation_fontsize=10,
        value_format="",
        error_bar_capsize=0.0,
        figsize=(0, 0),
    ),
}

def get_chart_style(
    chart_type: ChartType,
    theme: str = "publication",
) -> ChartStyleSpec:
    """Return a *copy* of the style spec for ``chart_type``, with
    theme-specific overrides applied.

    Parameters
    ----------
    chart_type:
        Which chart category to style.
    theme:
        ``'dark'`` or ``'publication'``.  Affects annotation text
        colours, grid alpha, and contour colours.

    Returns
    -------
    ChartStyleSpec
        A fresh copy safe to mutate for one-off overrides.
    """
    if chart_type not in CHART_STYLES:
        raise ValueError(
            f"Unknown chart type {chart_type!r}. "
            f"Registered types: {[ct.value for ct in CHART_STYLES]}"
        )

    spec = copy.deepcopy(CHART_STYLES[chart_type])

    if theme == "dark":
        spec.grid_alpha = min(spec.grid_alpha, 0.15)
        spec.annotation_fontsize = max(spec.annotation_fontsize, 9)
    else:  # publication
        spec.grid_alpha = max(spec.grid_alpha, 0.20)

    return spec

def resolve_colormap(
    chart_type: ChartType,
    *,
    override: Optional[str] = None,
) -> Optional[str]:
    """Return the correct matplotlib colormap name for a chart type.

    Parameters
    ----------
    chart_type:
        Chart category.
    override:
        Explicit colormap name that takes precedence.

    Returns
    -------
    str or None
        Colormap name, or ``None`` if the chart type does not use
        continuous colour mapping.
    """
    if override is not None:
        return override
    return CHART_STYLES.get(chart_type, ChartStyleSpec()).colormap

def get_semantic_colors(chart_type: ChartType) -> Dict[str, object]:
    """Return a mapping of semantic names → hex colours for a chart type.

    This replaces the old ``_get_plot_colors()`` flat dict with
    chart-type-aware semantics.
    """
    spec = CHART_STYLES.get(chart_type, ChartStyleSpec())
    palette = spec.palette or OKABE_ITO

    base = {
        "palette": palette,
        "primary": palette[0] if len(palette) > 0 else "#000000",
        "secondary": palette[1] if len(palette) > 1 else "#666666",
        "tertiary": palette[2] if len(palette) > 2 else "#999999",
    }

    if chart_type == ChartType.RAINCLOUD:
        base["group_a"] = PAIR_GIBBSQ
        base["group_b"] = PAIR_NEURAL
    elif chart_type == ChartType.BAR_COMPARISON:
        base["best"] = ABLATION_CASCADE[0]
        base["ablation_1"] = ABLATION_CASCADE[1]
        base["ablation_2"] = ABLATION_CASCADE[2]
        base["baseline"] = ABLATION_CASCADE[3]
    elif chart_type == ChartType.TRAINING_DASHBOARD:
        base["reward"] = TRAINING_PRIMARY
        base["loss"] = TRAINING_SECONDARY
        base["critic"] = TRAINING_CRITIC
        base["gradient"] = TRAINING_GRADIENT
        base["entropy"] = TRAINING_ENTROPY
    elif chart_type == ChartType.LINE_SERIES:
        base["method_a"] = PAIR_GIBBSQ
        base["method_b"] = PAIR_NEURAL

    return base
