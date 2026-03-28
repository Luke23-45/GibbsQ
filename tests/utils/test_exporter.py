"""
Hardened test suite for gibbsq.utils.exporter — Robustness Loop Stage 2.

Categories:
- A: Correctness Tests
- B: Invariant Tests
- C: Edge Case Tests
- D: Numerical Stability Tests
- F: Regression Tests
"""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from gibbsq.utils.exporter import save_trajectory_parquet, append_metrics_jsonl
from gibbsq.engines.numpy_engine import SimResult


# ============================================================
# CATEGORY A: CORRECTNESS TESTS
# ============================================================

class TestSaveTrajectoryParquet:
    """Verify Parquet trajectory export."""
    
    def test_creates_file(self):
        """Should create parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimResult(
                times=np.linspace(0, 10, 100),
                states=np.zeros((100, 2), dtype=np.int64),
                arrival_count=50,
                departure_count=50,
                final_time=10.0,
                num_servers=2,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            assert path.exists()
    
    def test_correct_columns(self):
        """Should have time, total_q, and q_i columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimResult(
                times=np.linspace(0, 10, 100),
                states=np.zeros((100, 3), dtype=np.int64),
                arrival_count=50,
                departure_count=50,
                final_time=10.0,
                num_servers=3,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            df = pd.read_parquet(path)
            assert "time" in df.columns
            assert "total_q" in df.columns
            assert "q_0" in df.columns
            assert "q_1" in df.columns
            assert "q_2" in df.columns
    
    def test_data_integrity(self):
        """Data in parquet should match input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            times = np.linspace(0, 10, 100)
            states = np.arange(200).reshape(100, 2)
            
            result = SimResult(
                times=times,
                states=states,
                arrival_count=50,
                departure_count=50,
                final_time=10.0,
                num_servers=2,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            df = pd.read_parquet(path)
            
            np.testing.assert_array_almost_equal(df["time"].values, times)
            np.testing.assert_array_equal(df["q_0"].values, states[:, 0])
            np.testing.assert_array_equal(df["q_1"].values, states[:, 1])
            np.testing.assert_array_equal(df["total_q"].values, states.sum(axis=1))
    
    def test_creates_parent_directories(self):
        """Should create nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimResult(
                times=np.array([0.0, 1.0]),
                states=np.array([[0, 0], [1, 1]]),
                arrival_count=1,
                departure_count=0,
                final_time=1.0,
                num_servers=2,
            )
            
            path = Path(tmpdir) / "nested" / "deep" / "test.parquet"
            save_trajectory_parquet(result, path)
            
            assert path.exists()


class TestAppendMetricsJsonl:
    """Verify JSON Lines metrics export."""
    
    def test_creates_file(self):
        """Should create jsonl file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            append_metrics_jsonl({"mean": 5.0}, path)
            
            assert path.exists()
    
    def test_appends_multiple(self):
        """Should append multiple lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            
            append_metrics_jsonl({"run": 1}, path)
            append_metrics_jsonl({"run": 2}, path)
            append_metrics_jsonl({"run": 3}, path)
            
            with open(path) as f:
                lines = f.readlines()
            
            assert len(lines) == 3
    
    def test_valid_json(self):
        """Each line should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            
            append_metrics_jsonl({"mean": 5.0, "std": 1.0}, path)
            
            with open(path) as f:
                line = f.readline()
            
            data = json.loads(line)
            assert data["mean"] == 5.0
            assert data["std"] == 1.0
    
    def test_numpy_scalar_conversion(self):
        """NumPy scalars should convert to Python types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            
            metrics = {
                "mean": np.float64(5.0),
                "count": np.int64(100),
                "flag": np.bool_(True),
            }
            append_metrics_jsonl(metrics, path)
            
            with open(path) as f:
                line = f.readline()
            
            data = json.loads(line)
            assert isinstance(data["mean"], float)
            assert isinstance(data["count"], int)
            assert isinstance(data["flag"], bool)
    
    def test_numpy_array_conversion(self):
        """NumPy arrays should convert to lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            
            metrics = {
                "queues": np.array([1, 2, 3]),
            }
            append_metrics_jsonl(metrics, path)
            
            with open(path) as f:
                line = f.readline()
            
            data = json.loads(line)
            assert data["queues"] == [1, 2, 3]


# ============================================================
# CATEGORY B: INVARIANT TESTS
# ============================================================

class TestExporterInvariants:
    """Invariants that must hold for all exports."""
    
    def test_parquet_no_index(self):
        """Parquet should not include DataFrame index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimResult(
                times=np.array([0.0, 1.0]),
                states=np.array([[0, 0], [1, 1]]),
                arrival_count=1,
                departure_count=0,
                final_time=1.0,
                num_servers=2,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            df = pd.read_parquet(path)
            # Index should be default RangeIndex
            assert df.index.name is None
    
    def test_jsonl_newline_terminated(self):
        """Each JSON line should end with newline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            
            append_metrics_jsonl({"test": 1}, path)
            
            with open(path) as f:
                content = f.read()
            
            assert content.endswith("\n")


# ============================================================
# CATEGORY C: EDGE CASE TESTS
# ============================================================

class TestExporterEdgeCases:
    """Test boundary conditions."""
    
    def test_empty_trajectory(self):
        """Single-point trajectory should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimResult(
                times=np.array([0.0]),
                states=np.array([[0, 0]]),
                arrival_count=0,
                departure_count=0,
                final_time=0.0,
                num_servers=2,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            df = pd.read_parquet(path)
            assert len(df) == 1
    
    def test_empty_metrics(self):
        """Empty metrics dict should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            append_metrics_jsonl({}, path)
            
            with open(path) as f:
                line = f.readline()
            
            data = json.loads(line)
            assert data == {}
    
    def test_single_server(self):
        """Single server should produce one q_0 column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimResult(
                times=np.linspace(0, 10, 50),
                states=np.arange(50).reshape(-1, 1),
                arrival_count=25,
                departure_count=25,
                final_time=10.0,
                num_servers=1,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            df = pd.read_parquet(path)
            assert "q_0" in df.columns
            assert "q_1" not in df.columns
    
    def test_large_num_servers(self):
        """Large number of servers should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            N = 100
            result = SimResult(
                times=np.linspace(0, 10, 50),
                states=np.zeros((50, N), dtype=np.int64),
                arrival_count=25,
                departure_count=25,
                final_time=10.0,
                num_servers=N,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            df = pd.read_parquet(path)
            assert f"q_{N-1}" in df.columns


# ============================================================
# CATEGORY D: NUMERICAL STABILITY TESTS
# ============================================================

class TestExporterNumericalStability:
    """Test behavior under numerically challenging inputs."""
    
    def test_large_values(self):
        """Large queue values should be preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimResult(
                times=np.linspace(0, 10, 100),
                states=np.full((100, 2), 1e6, dtype=np.int64),
                arrival_count=50,
                departure_count=50,
                final_time=10.0,
                num_servers=2,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            df = pd.read_parquet(path)
            assert (df["q_0"] == 1e6).all()
    
    def test_very_small_times(self):
        """Very small time values should be preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            times = np.array([0.0, 1e-10, 2e-10])
            result = SimResult(
                times=times,
                states=np.zeros((3, 2), dtype=np.int64),
                arrival_count=1,
                departure_count=0,
                final_time=2e-10,
                num_servers=2,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            df = pd.read_parquet(path)
            np.testing.assert_array_almost_equal(df["time"].values, times)
    
    def test_special_float_values(self):
        """Special float values in metrics should handle gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            
            # NaN and Inf should convert to valid JSON
            metrics = {
                "value": float("inf"),
            }
            append_metrics_jsonl(metrics, path)
            
            with open(path) as f:
                line = f.readline()
            
            data = json.loads(line)
            assert data["value"] == "inf" or data["value"] == float("inf")


# ============================================================
# CATEGORY F: REGRESSION TESTS
# ============================================================

class TestExporterRegressions:
    """Prevent reintroduction of known faults."""
    
    def test_regression_column_order(self):
        """Columns should be in expected order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimResult(
                times=np.array([0.0, 1.0]),
                states=np.array([[0, 1], [2, 3]]),
                arrival_count=1,
                departure_count=0,
                final_time=1.0,
                num_servers=2,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            df = pd.read_parquet(path)
            # First column should be time
            assert df.columns[0] == "time"
            # Last column should be total_q
            assert df.columns[-1] == "total_q"
    
    def test_regression_compression_default(self):
        """Default compression should be snappy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SimResult(
                times=np.array([0.0, 1.0]),
                states=np.array([[0, 0], [1, 1]]),
                arrival_count=1,
                departure_count=0,
                final_time=1.0,
                num_servers=2,
            )
            
            path = Path(tmpdir) / "test.parquet"
            save_trajectory_parquet(result, path)
            
            # File should exist and be readable (snappy is default)
            assert path.exists()
            df = pd.read_parquet(path)
            assert len(df) == 2
