"""
Configuration profiles for different types of experiment plots.
"""
from dataclasses import dataclass
from typing import Dict, Any, List
from .chart_styles import ChartType

@dataclass
class ExperimentPlotContext:
    experiment_id: str
    data_source: str
    output_name: str

@dataclass
class ExperimentPlotProfile:
    name: str
    chart_type: ChartType
    defaults: Dict[str, Any]

def resolve_experiment_plot_profile(experiment_id: str) -> ExperimentPlotProfile:
    """Mock resolver for experiment plot profiles."""
    return ExperimentPlotProfile("default", ChartType.LINE, {})
