"""
Common utilities for the GibbsQ analysis framework.

Provides shared I/O, metrics, statistics, styling, LaTeX formatting,
configuration loading, and constant definitions used across all
study-specific modules.
"""

from analysis.common.config import (
    get_config,
    get_data_root,
    get_study_data_dir,
    get_output_root,
)
from analysis.common.constants import (
    HYPOTHESIS_DISPLAY,
    HYPOTHESIS_ORDER,
    HYPOTHESIS_COLORS,
    SEEDS,
)
from analysis.common.style import apply_thesis_style, create_figure, save_figure
