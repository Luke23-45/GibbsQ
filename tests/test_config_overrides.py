import pytest
import math
from hydra import compose, initialize
from omegaconf import OmegaConf
from gibbsq.core.config import hydra_to_config, validate, ExperimentConfig

def test_default_loading():
    """Ensure the default configuration loads without errors and binds to our dataclass."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_raw = compose(config_name="default")
        cfg = hydra_to_config(cfg_raw)
        validate(cfg)
        
        # Verify baseline defaults from default.yaml
        assert cfg.system.num_servers == 10
        assert math.isclose(cfg.system.arrival_rate, 10.4)
        assert len(cfg.system.service_rates) == 10
        assert cfg.simulation.sim_time == 25000.0
        assert cfg.simulation.num_replications == 30
        assert cfg.wandb.mode == "offline"

def test_cli_overrides():
    """Ensure that CLI overrides correctly mutate the base configuration."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_raw = compose(config_name="default", overrides=[
            "system.num_servers=5", 
            "system.arrival_rate=4.5",
            "system.service_rates=[1.0, 1.0, 1.0, 1.0, 1.0]",
            "simulation.sim_time=500.0",
            "wandb.mode=online"
        ])
        cfg = hydra_to_config(cfg_raw)
        validate(cfg)
        
        assert cfg.system.num_servers == 5
        assert math.isclose(cfg.system.arrival_rate, 4.5)
        assert len(cfg.system.service_rates) == 5
        assert cfg.system.service_rates[0] == 1.0
        assert cfg.simulation.sim_time == 500.0
        assert cfg.wandb.mode == "online"

def test_experiment_profile_composition_stability_sweep():
    """Ensure the standard stability_sweep experiment profile maintains high-fidelity paper defaults."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_raw = compose(config_name="default", overrides=["+experiment=stability_sweep"])
        cfg = hydra_to_config(cfg_raw)
        validate(cfg)
        
        # Checking high-fidelity paper defaults
        assert cfg.simulation.num_replications == 50
        assert cfg.simulation.sim_time == 25000.0
        assert math.isclose(cfg.simulation.sample_interval, 0.1)
        assert cfg.jax.enabled is True
        assert cfg.jax.precision == "float32"

def test_experiment_profile_composition_stability_sweep_small():
    """Ensure the small stability_sweep variation uses efficient parameters."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_raw = compose(config_name="small", overrides=["+experiment=stability_sweep_small"])
        cfg = hydra_to_config(cfg_raw)
        validate(cfg)
        
        # Small config specialized values
        assert cfg.simulation.sim_time == 1000.0
        assert cfg.simulation.num_replications == 5
        assert math.isclose(cfg.simulation.sample_interval, 0.05)
        assert cfg.jax.enabled is True
        assert cfg.jax.precision == "float32"

def test_experiment_profile_composition_stability_sweep_large():
    """Ensure the large stability_sweep variation uses OOM-safe parameters."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_raw = compose(config_name="large", overrides=["+experiment=stability_sweep_large"])
        cfg = hydra_to_config(cfg_raw)
        validate(cfg)
        
        # Large config safe values
        assert cfg.simulation.sim_time == 5000.0
        assert cfg.simulation.num_replications == 10
        assert math.isclose(cfg.simulation.sample_interval, 0.5)
        assert cfg.jax.enabled is True
        assert cfg.jax.precision == "float32"

def test_experiment_profile_composition_policy_comparison():
    """Ensure the standard policy_comparison experiment profile uses runtime-balanced overrides."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_raw = compose(config_name="default", overrides=["+experiment=policy_comparison"])
        cfg = hydra_to_config(cfg_raw)
        validate(cfg)
        
        # Checking runtime-balanced defaults
        assert cfg.simulation.num_replications == 20
        assert cfg.simulation.sim_time == 5000.0
        assert math.isclose(cfg.simulation.sample_interval, 0.5)
        assert cfg.jax.enabled is True

def test_experiment_profile_composition_policy_comparison_small():
    """Ensure the small policy_comparison variation uses efficient parameters."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_raw = compose(config_name="small", overrides=["+experiment=policy_comparison_small"])
        cfg = hydra_to_config(cfg_raw)
        validate(cfg)
        
        # Small config specialized values
        assert cfg.simulation.sim_time == 1000.0
        assert cfg.simulation.num_replications == 3
        assert math.isclose(cfg.simulation.sample_interval, 0.05)
        assert cfg.jax.enabled is True

def test_experiment_profile_composition_policy_comparison_large():
    """Ensure the large policy_comparison variation uses OOM-safe parameters."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_raw = compose(config_name="large", overrides=["+experiment=policy_comparison_large"])
        cfg = hydra_to_config(cfg_raw)
        validate(cfg)
        
        # Large config safe values
        assert cfg.simulation.sim_time == 5000.0
        assert cfg.simulation.num_replications == 100
        assert math.isclose(cfg.simulation.sample_interval, 1.0)
        assert cfg.jax.enabled is True

def test_validation_failure_on_bad_override():
    """Ensure mathematical corruption (e.g. arrival > capacity) is blocked by the config system."""
    with initialize(version_base=None, config_path="../configs"):
        # We set arrival_rate to 100.0, but aggregate service rate defaults to ~13.0
        cfg_raw = compose(config_name="default", overrides=["system.arrival_rate=100.0"])
        cfg = hydra_to_config(cfg_raw)
        
        # The validation engine MUST reject this unstable configuration
        with pytest.raises(ValueError, match="Capacity condition violated"):
            validate(cfg)

def test_nested_dataclass_overrides():
    """Ensure complex nested overrides into sub-structs resolve cleanly."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_raw = compose(config_name="default", overrides=[
            "+stability_sweep.alpha_vals=[0.2, 0.4]",
            "+neural_training.dga_learning_rate=0.99"
        ])
        cfg = hydra_to_config(cfg_raw)
        validate(cfg)
        
        assert len(cfg.stability_sweep.alpha_vals) == 2
        assert cfg.stability_sweep.alpha_vals[0] == 0.2
        assert cfg.neural_training.dga_learning_rate == 0.99
    
if __name__ == "__main__":
    pytest.main(["-v", __file__])
