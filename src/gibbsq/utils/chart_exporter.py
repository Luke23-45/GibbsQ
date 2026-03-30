"""
Unified chart and data export system for publication-quality outputs.

Features:
- Multi-format chart export (PNG, PDF, SVG)
- High-DPI output for print quality
- Data export (CSV, JSON, NPZ)
- Consistent naming conventions
- Automatic metadata inclusion
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mfig

log = logging.getLogger(__name__)

__all__ = [
    "save_chart",
    "save_data",
    "ChartConfig",
    "DataConfig",
]

@dataclass
class ChartConfig:
    """Configuration for chart export."""
    dpi: int = 600
    transparent: bool = False
    bbox_inches: str = "tight"
    pad_inches: float = 0.1
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                "Creator": "GibbsQ Chart Exporter",
            }

@dataclass
class DataConfig:
    """Configuration for data export."""
    indent: int = 2  # For JSON
    compress: bool = False  # For NPZ
    include_index: bool = True  # For CSV
    metadata: Dict[str, Any] = field(default_factory=dict)

def save_chart(
    fig: mfig.Figure,
    output_path: Union[str, Path],
    formats: Optional[List[str]] = None,
    config: Optional[ChartConfig] = None,
    close_fig: bool = True,
) -> List[Path]:
    """
    Save a matplotlib figure in multiple formats.
    
    Args:
        fig: Matplotlib figure to save
        output_path: Base path (without extension)
        formats: List of formats ['png', 'pdf', 'svg']. Default: ['png']
        config: ChartConfig with export settings
        close_fig: Whether to close the figure after saving
    
    Returns:
        List of paths to saved files
    """
    if formats is None:
        formats = ['png']
    
    if config is None:
        config = ChartConfig()
    
    output_path = Path(output_path)
    output_dir = output_path.parent
    base_name = output_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for fmt in formats:
        fmt = fmt.lower().lstrip('.')
        file_path = output_dir / f"{base_name}.{fmt}"
        
        try:
            fig.savefig(
                file_path,
                format=fmt,
                dpi=config.dpi,
                transparent=config.transparent,
                bbox_inches=config.bbox_inches,
                pad_inches=config.pad_inches,
                metadata=config.metadata if fmt == 'pdf' else None,
            )
            saved_paths.append(file_path)
            log.debug(f"Saved chart: {file_path}")
        except Exception as e:
            log.error(f"Failed to save {fmt} chart: {e}")
    
    if close_fig:
        plt.close(fig)
    
    return saved_paths

def save_chart_with_data(
    fig: mfig.Figure,
    data: Dict[str, Any],
    output_path: Union[str, Path],
    chart_formats: Optional[List[str]] = None,
    data_format: str = "csv",
    chart_config: Optional[ChartConfig] = None,
    data_config: Optional[DataConfig] = None,
) -> Dict[str, List[Path]]:
    """
    Save both chart and associated data.
    
    Args:
        fig: Matplotlib figure
        data: Dictionary of data to save
        output_path: Base path (without extension)
        chart_formats: Chart formats ['png', 'pdf', 'svg']
        data_format: Data format ('csv', 'json', 'npz')
        chart_config: ChartConfig
        data_config: DataConfig
    
    Returns:
        Dictionary with 'charts' and 'data' keys containing paths
    """
    output_path = Path(output_path)
    
    chart_paths = save_chart(fig, output_path, chart_formats, chart_config, close_fig=False)
    
    data_path = output_path.parent / f"{output_path.stem}_data"
    data_paths = [save_data(data, data_path, data_format, data_config)]
    
    plt.close(fig)
    
    return {
        "charts": chart_paths,
        "data": data_paths,
    }

def save_data(
    data: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "csv",
    config: Optional[DataConfig] = None,
) -> Path:
    """
    Save data in specified format.
    
    Args:
        data: Dictionary of data to save
        output_path: Base path (without extension)
        format: Output format ('csv', 'json', 'npz')
        config: DataConfig with export settings
    
    Returns:
        Path to saved file
    """
    if config is None:
        config = DataConfig()
    
    output_path = Path(output_path)
    format = format.lower().lstrip('.')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path.with_suffix(f".{format}")
    
    if format == "csv":
        _save_csv(data, file_path, config)
    elif format == "json":
        _save_json(data, file_path, config)
    elif format == "npz":
        _save_npz(data, file_path, config)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', or 'npz'")
    
    log.debug(f"Saved data: {file_path}")
    return file_path

def _save_csv(data: Dict[str, Any], file_path: Path, config: DataConfig) -> None:
    """Save data as CSV file."""
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    if config.metadata:
        with open(file_path, 'w') as f:
            for key, value in config.metadata.items():
                f.write(f"# {key}: {value}\n")
        
        df.to_csv(file_path, mode='a', index=config.include_index)
    else:
        df.to_csv(file_path, index=config.include_index)

def _save_json(data: Dict[str, Any], file_path: Path, config: DataConfig) -> None:
    """Save data as JSON file."""
    serializable_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_data[key] = float(value)
        else:
            serializable_data[key] = value
    
    if config.metadata:
        serializable_data["_metadata"] = config.metadata
    
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f, indent=config.indent)

def _save_npz(data: Dict[str, Any], file_path: Path, config: DataConfig) -> None:
    """Save data as NPZ file (numpy compressed)."""
    np_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            np_data[key] = np.array(value)
        elif isinstance(value, np.ndarray):
            np_data[key] = value
        else:
            np_data[key] = np.array(value)
    
    if config.compress:
        np.savez_compressed(file_path, **np_data)
    else:
        np.savez(file_path, **np_data)

def export_experiment_results(
    output_dir: Union[str, Path],
    figures: Optional[Dict[str, mfig.Figure]] = None,
    data: Optional[Dict[str, Dict[str, Any]]] = None,
    chart_formats: List[str] = None,
    data_format: str = "csv",
    chart_config: Optional[ChartConfig] = None,
    data_config: Optional[DataConfig] = None,
) -> Dict[str, List[Path]]:
    """
    Export all experiment results (multiple figures and data).
    
    Args:
        output_dir: Directory for outputs
        figures: Dictionary mapping figure names to Figure objects
        data: Dictionary mapping data names to data dictionaries
        chart_formats: Chart formats
        data_format: Data format
        chart_config: ChartConfig
        data_config: DataConfig
    
    Returns:
        Dictionary with 'charts' and 'data' keys containing all paths
    """
    if chart_formats is None:
        chart_formats = ['png', 'pdf']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_chart_paths = []
    all_data_paths = []
    
    if figures:
        for name, fig in figures.items():
            chart_path = output_dir / name
            paths = save_chart(fig, chart_path, chart_formats, chart_config, close_fig=False)
            all_chart_paths.extend(paths)
            plt.close(fig)
    
    if data:
        for name, data_dict in data.items():
            data_path = output_dir / f"{name}_data"
            path = save_data(data_dict, data_path, data_format, data_config)
            all_data_paths.append(path)
    
    return {
        "charts": all_chart_paths,
        "data": all_data_paths,
    }

def get_default_chart_config() -> ChartConfig:
    """Get default chart configuration for publication."""
    return ChartConfig(
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.1,
    )

def get_default_data_config() -> DataConfig:
    """Get default data configuration."""
    return DataConfig(
        indent=2,
        compress=False,
        include_index=True,
    )

def ensure_output_dir(output_path: Union[str, Path]) -> Path:
    """Ensure output directory exists and return path."""
    output_path = Path(output_path)
    if output_path.suffix:
        output_dir = output_path.parent
    else:
        output_dir = output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_path
