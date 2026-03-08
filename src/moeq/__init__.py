# Q-MoE: Queue-Aware Mixture of Experts
# High-Performance Soft-JSQ Routing Infrastructure

from moeq.core.config import ExperimentConfig, SystemConfig, SimulationConfig, PolicyConfig, DriftConfig
from moeq.core.config import validate, total_capacity, load_factor, drift_constant_R, drift_rate_epsilon
from moeq.core.config import hydra_to_config
from moeq.core.policies import make_policy, SoftmaxRouting, UniformRouting, ProportionalRouting, JSQRouting
from moeq.engines.numpy_engine import simulate, run_replications, SimResult
from moeq.engines.jax_engine import simulate_jax, run_replications_jax
from moeq.core.drift import generator_drift, upper_bound, simplified_bound, verify_single
from moeq.core.drift import evaluate_grid, evaluate_trajectory, lyapunov_V
from moeq.analysis.metrics import time_averaged_queue_lengths, gini_coefficient, stationarity_diagnostic
from moeq.utils.exporter import save_trajectory_parquet, append_metrics_jsonl
from moeq.utils.logging import setup_wandb, get_run_config

__all__ = [
    # Config
    "ExperimentConfig", "SystemConfig", "SimulationConfig", "PolicyConfig", "DriftConfig",
    "validate", "total_capacity", "load_factor", "drift_constant_R", "drift_rate_epsilon",
    "hydra_to_config",
    # Policies
    "make_policy", "SoftmaxRouting", "UniformRouting", "ProportionalRouting", "JSQRouting",
    # Simulator
    "simulate", "run_replications", "SimResult",
    "simulate_jax", "run_replications_jax",
    # Drift
    "generator_drift", "upper_bound", "simplified_bound", "verify_single",
    "evaluate_grid", "evaluate_trajectory", "lyapunov_V",
    # Metrics
    "time_averaged_queue_lengths", "gini_coefficient", "stationarity_diagnostic",
    # Exporter
    "save_trajectory_parquet", "append_metrics_jsonl",
    # Logging
    "setup_wandb", "get_run_config",
]
