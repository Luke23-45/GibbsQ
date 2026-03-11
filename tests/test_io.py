import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
from gibbsq.utils.exporter import save_trajectory_parquet, append_metrics_jsonl
from gibbsq.engines.numpy_engine import SimResult

def test_save_trajectory_parquet(tmp_path):
    """Verify Parquet export preserves shape and types."""
    res = SimResult(
        times=np.linspace(0, 10, 11),
        states=np.random.randint(0, 10, (11, 4)),
        arrival_count=100,
        departure_count=90,
        final_time=10.0,
        num_servers=4
    )
    
    file_path = tmp_path / "test.parquet"
    save_trajectory_parquet(res, file_path)
    
    assert file_path.exists()
    
    # Load back
    df = pd.read_parquet(file_path)
    assert len(df) == 11
    assert list(df.columns) == ["time", "q_0", "q_1", "q_2", "q_3", "total_q"]
    assert np.allclose(df["time"], res.times)
    assert np.all(df["total_q"] == res.states.sum(axis=1))

def test_append_metrics_jsonl(tmp_path):
    """Verify JSONL append handles NumPy scalars and multiple lines."""
    file_path = tmp_path / "metrics.jsonl"
    
    m1 = {"rho": 0.8, "mean_q": np.float64(5.5), "is_ok": True}
    m2 = {"rho": 0.9, "mean_q": np.array([1.2, 3.4])} # Test array
    
    append_metrics_jsonl(m1, file_path)
    append_metrics_jsonl(m2, file_path)
    
    assert file_path.exists()
    
    with open(file_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        
        # Check types
        d1 = json.loads(lines[0])
        assert isinstance(d1["mean_q"], float) # NP scalar should become float
        
        d2 = json.loads(lines[1])
        assert isinstance(d2["mean_q"], list) # NP array should become list
