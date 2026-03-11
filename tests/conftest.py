"""
Shared test fixtures and configuration.
"""

import pytest
import numpy as np
from gibbsq.core.config import ExperimentConfig, SystemConfig, SimulationConfig, PolicyConfig, DriftConfig


@pytest.fixture
def small_config():
    """N=2 configuration for quick tests."""
    return ExperimentConfig(
        system=SystemConfig(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=[1.0, 1.5],
            alpha=1.0,
        ),
        simulation=SimulationConfig(
            sim_time=1000.0,
            sample_interval=0.5,
            num_replications=2,
            seed=42,
            burn_in_fraction=0.2,
        ),
        policy=PolicyConfig(name="softmax", d=2),
        drift=DriftConfig(q_max=20, use_grid=True),
        output_dir="test_outputs",
    )


@pytest.fixture
def medium_config():
    """N=5 configuration for moderate tests."""
    return ExperimentConfig(
        system=SystemConfig(
            num_servers=5,
            arrival_rate=3.0,
            service_rates=[0.8, 1.0, 1.2, 1.4, 1.6],
            alpha=2.0,
        ),
        simulation=SimulationConfig(
            sim_time=5000.0,
            sample_interval=1.0,
            num_replications=3,
            seed=123,
            burn_in_fraction=0.2,
        ),
        policy=PolicyConfig(name="softmax", d=2),
        drift=DriftConfig(q_max=30, use_grid=False),
        output_dir="test_outputs",
    )


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)
