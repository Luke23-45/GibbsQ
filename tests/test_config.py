import pytest
from moeq.core.config import (
    SystemConfig, SimulationConfig, PolicyConfig, DriftConfig, ExperimentConfig,
    validate, total_capacity, load_factor, drift_constant_R, drift_rate_epsilon,
    hydra_to_config, JAXConfig
)
from omegaconf import OmegaConf

def test_valid_config():
    cfg = ExperimentConfig(
        system=SystemConfig(num_servers=2, arrival_rate=1.0, service_rates=[1.0, 1.5], alpha=1.0),
        simulation=SimulationConfig()
    )
    validate(cfg)  # Should not raise
    
    # Derived checks
    assert total_capacity(cfg) == 2.5
    assert load_factor(cfg) == 1.0 / 2.5

def test_invalid_capacity_condition():
    cfg = ExperimentConfig(
        system=SystemConfig(num_servers=2, arrival_rate=3.0, service_rates=[1.0, 1.5], alpha=1.0),
    )
    with pytest.raises(ValueError, match="Capacity condition violated"):
        validate(cfg)

def test_invalid_structural():
    cfg = ExperimentConfig(
        system=SystemConfig(num_servers=3, arrival_rate=1.0, service_rates=[1.0, 1.5], alpha=1.0),
    )
    with pytest.raises(ValueError, match="≠ num_servers"):
        validate(cfg)

def test_eps_R_bounds():
    cfg = ExperimentConfig(
        system=SystemConfig(num_servers=2, arrival_rate=1.0, service_rates=[1.0, 5.0], alpha=1.0),
    )
    # Λ = 6.0, λ = 1.0. ε = min((6-1)/2, 1) = min(2.5, 1) = 1.0
    assert drift_rate_epsilon(cfg) == 1.0
    
    # R = (1*ln(2))/1 + (1+6)/2 = ln(2) + 3.5
    import math
    assert drift_constant_R(cfg) == pytest.approx(math.log(2) + 3.5)


def test_hydra_single_service_rate_expands_for_large_n():
    raw = OmegaConf.create({
        "system": {
            "num_servers": 4,
            "arrival_rate": 2.0,
            "service_rates": [1.0],
            "alpha": 1.0,
        },
        "simulation": {},
        "policy": {},
        "drift": {},
        "wandb": {},
        "jax": {},
    })
    cfg = hydra_to_config(raw)
    assert cfg.system.service_rates == [1.0, 1.0, 1.0, 1.0]
    validate(cfg)


def test_invalid_jax_platform_rejected():
    cfg = ExperimentConfig(
        system=SystemConfig(num_servers=2, arrival_rate=1.0, service_rates=[1.0, 1.5], alpha=1.0),
        jax=JAXConfig(enabled=True, platform="quantum", precision="float64", fallback_to_cpu=True),
    )
    with pytest.raises(ValueError, match="jax.platform"):
        validate(cfg)


def test_cuda_alias_is_accepted_in_jax_config():
    cfg = ExperimentConfig(
        system=SystemConfig(num_servers=2, arrival_rate=1.0, service_rates=[1.0, 1.5], alpha=1.0),
        jax=JAXConfig(enabled=True, platform="cuda", precision="float32", fallback_to_cpu=True),
    )
    validate(cfg)
