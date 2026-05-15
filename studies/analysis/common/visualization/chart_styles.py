"""
Enum-based chart type definitions and style specifications.
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, Any

class ChartType(Enum):
    BAR = auto()
    LINE = auto()
    SCATTER = auto()
    HEATMAP = auto()
    VIOLIN = auto()

@dataclass
class ChartStyleSpec:
    chart_type: ChartType
    title: str
    xlabel: str
    ylabel: str
    legend_loc: str = "best"
    use_log_scale: bool = False
    color_map: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None
