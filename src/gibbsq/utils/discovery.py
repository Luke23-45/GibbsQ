"""Run Discovery Utilities: Functions for finding and loading experiment Run Capsules."""

import json
from pathlib import Path
import pandas as pd
import logging

from gibbsq.utils.run_artifacts import metrics_path

log = logging.getLogger(__name__)

def find_latest_run(base_dir: Path, experiment_type: str) -> Path | None:
    """Find the most recent timestamped run directory for an experiment type."""
    exp_dir = base_dir / experiment_type
    if not exp_dir.exists():
        return None
    
    runs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
    return runs[0] if runs else None

def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Load metrics.jsonl from a run directory."""
    metrics_file = metrics_path(run_dir)
    if not metrics_file.exists():
        metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return pd.DataFrame()
    
    records = []
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except Exception as e:
        log.error(f"Failed to load metrics from {metrics_file}: {e}")
        return pd.DataFrame()
        
    return pd.DataFrame(records)
