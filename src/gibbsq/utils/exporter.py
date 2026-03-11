"""
Data export utilities (Parquet, JSONL).

Replaces inefficient CSVs with Apache Parquet for numeric trajectory data.
Parquet provides:
1. ~20x compression via dictionary encoding.
2. Type preservation (float64 and int64).
3. Instant deserialization into pandas DataFrames.

JSON Lines (.jsonl) is used for appending scalar metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gibbsq.engines.numpy_engine import SimResult

log = logging.getLogger(__name__)

__all__ = [
    "save_trajectory_parquet",
    "append_metrics_jsonl",
]


def save_trajectory_parquet(res: SimResult, file_path: str | Path, compression: str = "snappy") -> None:
    """
    Save the dense Gillespie simulation trajectory |Q(t)| to a highly-compressed Parquet file.
    
    Generates a DataFrame with columns:
      - 'time': The timestamp  t.
      - 'total_q': The sum of the queues at  t.
      - 'q_0', 'q_1', ...: The individual server queue lengths.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pre-allocate column data
    data = {"time": res.times}
    
    # Add individual queues
    N = res.num_servers
    for i in range(N):
        data[f"q_{i}"] = res.states[:, i]
        
    # Add scalar total if they need it for quick processing
    data["total_q"] = res.states.sum(axis=1)
    
    # Create DataFrame and instantly encode
    df = pd.DataFrame(data)
    
    # Write to parquet natively
    df.to_parquet(path, engine="pyarrow", compression=compression, index=False)
    log.debug(f"Saved trajectory to {path} ({path.stat().st_size / 1024:.1f} KB)")


def append_metrics_jsonl(metrics: dict[str, Any], file_path: str | Path) -> None:
    """
    Append a scalar metric summary to a JSON Lines log for tracking multiple runs.
    
    This avoids massive CSV row locks and is natively iterable.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any raw NumPy scalars inside the dict to native Python types for JSON compatibility
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.generic, np.ndarray)):
            clean_metrics[k] = v.tolist()
        else:
            clean_metrics[k] = v

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean_metrics) + "\n")
