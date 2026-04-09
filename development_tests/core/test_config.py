import math
from pathlib import Path
import csv
import pytest
import numpy as np
from hydra import compose, initialize_config_dir
try:
    from hypothesis import given, strategies as st, assume, settings
    from hypothesis.extra.numpy import arrays
except ModuleNotFoundError:
    def given(*args, **kwargs):
        return pytest.mark.skip(reason="hypothesis is not installed in this environment")

    class _HypothesisStrategiesStub:
        def integers(self, *args, **kwargs):
            return None

        def floats(self, *args, **kwargs):
            return None

    st = _HypothesisStrategiesStub()

    def assume(condition):
        return None

    def settings(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def arrays(*args, **kwargs):
        return None

from gibbsq.core.config import (
    SystemConfig, SSAConfig, SimulationConfig, PolicyConfig, DriftConfig,
    ExperimentConfig, WandbConfig, JAXConfig,
    validate, total_capacity, load_factor, drift_constant_R, 
    drift_rate_epsilon, compact_set_radius, hydra_to_config, PolicyName,
    critical_load_required_sim_time, critical_load_sim_time,
    PROFILE_CONFIG_NAMES, EXPERIMENT_BLOCK_NAMES, validate_profile_config,
    resolve_experiment_config, resolve_experiment_config_chain, runtime_root_dict,
)
from omegaconf import DictConfig, OmegaConf


def _flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = str(v)
        elif v is None:
            items[new_key] = "null"
        else:
            items[new_key] = v
    return items

class TestDerivedQuantitiesCorrectness:
    def test_total_capacity_sum(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=3,
                arrival_rate=1.0,
                service_rates=[1.0, 2.0, 3.0],
                alpha=1.0,
            ),
        )
        assert total_capacity(cfg) == 6.0
    
    def test_load_factor_ratio(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=2.0,
                service_rates=[3.0, 7.0],
                alpha=1.0,
            ),
        )
        assert load_factor(cfg) == pytest.approx(0.2)
    
    def test_drift_constant_R_formula(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=4,
                arrival_rate=2.0,
                service_rates=[1.0, 1.0, 1.0, 1.0],
                alpha=0.5,
            ),
        )
        expected = (2.0 * math.log(4)) / 0.5 + (2.0 + 4.0) / 2.0
        assert drift_constant_R(cfg) == pytest.approx(expected, rel=1e-10)
    
    def test_epsilon_formula(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=3,
                arrival_rate=1.0,
                service_rates=[0.5, 2.0, 3.0],
                alpha=1.0,
            ),
        )
        assert drift_rate_epsilon(cfg) == pytest.approx(0.5)
    
    def test_compact_set_radius_formula(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[2.0, 2.0],
                alpha=1.0,
            ),
        )
        R = drift_constant_R(cfg)
        eps = drift_rate_epsilon(cfg)
        expected = math.ceil((R + 1.0) / eps)
        assert compact_set_radius(cfg) == expected

class TestConfigInvariants:
    @given(
        num_servers=st.integers(min_value=1, max_value=100),
        arrival_rate=st.floats(min_value=0.01, max_value=100.0, allow_infinity=False, allow_nan=False),
        alpha=st.floats(min_value=0.001, max_value=1000.0, allow_infinity=False, allow_nan=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_load_factor_always_in_open_interval(self, num_servers, arrival_rate, alpha):
        base_rate = arrival_rate / num_servers + 0.1  # Ensure Λ > λ
        service_rates = [base_rate] * num_servers
        
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=num_servers,
                arrival_rate=arrival_rate,
                service_rates=service_rates,
                alpha=alpha,
            ),
        )
        validate(cfg)
        
        rho = load_factor(cfg)
        assert 0.0 < rho < 1.0, f"Load factor {rho} not in (0, 1)"
    
    @given(
        num_servers=st.integers(min_value=1, max_value=50),
        arrival_rate=st.floats(min_value=0.1, max_value=10.0, allow_infinity=False, allow_nan=False),
        alpha=st.floats(min_value=0.1, max_value=10.0, allow_infinity=False, allow_nan=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_epsilon_always_positive(self, num_servers, arrival_rate, alpha):
        base_rate = (arrival_rate / num_servers) + 0.5
        service_rates = [base_rate + i * 0.1 for i in range(num_servers)]
        
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=num_servers,
                arrival_rate=arrival_rate,
                service_rates=service_rates,
                alpha=alpha,
            ),
        )
        validate(cfg)
        
        eps = drift_rate_epsilon(cfg)
        assert eps > 0.0, f"Epsilon {eps} not positive"
    
    def test_total_capacity_equals_sum(self):
        rates = [0.1] * 10
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=10,
                arrival_rate=0.5,
                service_rates=rates,
                alpha=1.0,
            ),
        )
        validate(cfg)
        assert total_capacity(cfg) == 1.0

class TestConfigEdgeCases:
    def test_single_server(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=1,
                arrival_rate=0.5,
                service_rates=[1.0],
                alpha=1.0,
            ),
        )
        validate(cfg)
        R = drift_constant_R(cfg)
        expected_R = (0.5 + 1.0) / 2.0  # C1 = (λ + Λ) / 2
        assert R == pytest.approx(expected_R)
    
    def test_near_capacity_violation(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[0.5000001, 0.5000001],
                alpha=1.0,
            ),
        )
        validate(cfg)  # Should not raise
        eps = drift_rate_epsilon(cfg)
        assert eps > 0.0  # Should be tiny but positive
    
    def test_capacity_violation_raises(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=2.0,
                service_rates=[0.5, 0.5],
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="Capacity condition violated"):
            validate(cfg)
    
    def test_exact_capacity_violation(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[0.5, 0.5],
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="Capacity condition violated"):
            validate(cfg)
    
    def test_zero_arrival_rate(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=0.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="arrival_rate.*must be > 0"):
            validate(cfg)
    
    def test_negative_arrival_rate(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=-1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="arrival_rate.*must be > 0"):
            validate(cfg)
    
    def test_zero_alpha(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=0.0,
            ),
        )
        with pytest.raises(ValueError, match="alpha.*must be > 0"):
            validate(cfg)
    
    def test_negative_alpha(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=-1.0,
            ),
        )
        with pytest.raises(ValueError, match="alpha.*must be > 0"):
            validate(cfg)
    
    def test_zero_service_rate(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[0.0, 1.0],
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="service_rates.*must be > 0"):
            validate(cfg)
    
    def test_negative_service_rate(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[-0.5, 1.0],
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="service_rates.*must be > 0"):
            validate(cfg)
    
    def test_mismatched_service_rates_count(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=3,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="≠ num_servers"):
            validate(cfg)
    
    def test_zero_num_servers(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=0,
                arrival_rate=1.0,
                service_rates=[],
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="num_servers must be ≥ 1"):
            validate(cfg)
    
    def test_invalid_policy_name(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
            policy=PolicyConfig(name="unknown_policy"),
        )
        with pytest.raises(ValueError, match="Unknown policy"):
            validate(cfg)
    
    def test_invalid_burn_in_fraction_negative(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
            simulation=SimulationConfig(burn_in_fraction=-0.1),
        )
        with pytest.raises(ValueError, match="burn_in_fraction"):
            validate(cfg)
    
    def test_invalid_burn_in_fraction_one(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
            simulation=SimulationConfig(burn_in_fraction=1.0),
        )
        with pytest.raises(ValueError, match="burn_in_fraction"):
            validate(cfg)
    
    def test_zero_sim_time(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
            simulation=SimulationConfig(ssa=SSAConfig(sim_time=0.0)),
        )
        with pytest.raises(ValueError, match="sim_time.*must be > 0"):
            validate(cfg)
    
    def test_negative_sim_time(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
            simulation=SimulationConfig(ssa=SSAConfig(sim_time=-100.0)),
        )
        with pytest.raises(ValueError, match="sim_time.*must be > 0"):
            validate(cfg)

class TestConfigNumericalStability:
    def test_very_small_alpha(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1e-10,
            ),
        )
        validate(cfg)
        R = drift_constant_R(cfg)
        assert math.isfinite(R)
        assert R > 1e9  # Should be huge
    
    def test_very_large_alpha(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1e10,
            ),
        )
        validate(cfg)
        R = drift_constant_R(cfg)
        assert math.isfinite(R)
        assert R == pytest.approx(1.5, rel=0.01)
    
    def test_very_large_num_servers(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=1000,
                arrival_rate=500.0,
                service_rates=[1.0] * 1000,
                alpha=1.0,
            ),
        )
        validate(cfg)
        R = drift_constant_R(cfg)
        assert math.isfinite(R)
    
    def test_very_small_epsilon(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[0.5000000001, 0.5000000001],
                alpha=1.0,
            ),
        )
        validate(cfg)
        eps = drift_rate_epsilon(cfg)
        assert eps > 0.0
        assert eps < 1e-9

    def test_uas_archimedean_constants_match_weighted_jensen_closure(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=3,
                arrival_rate=1.0,
                service_rates=[2.0, 3.0, 4.0],
                alpha=0.25,
            ),
            policy=PolicyConfig(name=PolicyName.UAS.value),
        )
        validate(cfg)
        R = drift_constant_R(cfg)
        eps = drift_rate_epsilon(cfg)
        assert R == pytest.approx((1.0 * 3.0) / 9.0 + 1.5)
        assert eps == pytest.approx((9.0 - 1.0) / 9.0)
    
    def test_floating_point_service_rates(self):
        rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=5,
                arrival_rate=1.0,
                service_rates=rates,
                alpha=1.0,
            ),
        )
        validate(cfg)
        cap = total_capacity(cfg)
        assert cap == pytest.approx(1.5, rel=1e-15)

class TestConfigRegressions:
    def test_regression_capacity_check_uses_fsum(self):
        rates = [1e-15] * 1000
        rates.append(2.0)
        
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=1001,
                arrival_rate=1.0,
                service_rates=rates,
                alpha=1.0,
            ),
        )
        validate(cfg)
        cap = total_capacity(cfg)
        assert cap > 1.0

class TestHydraConversion:
    def test_hydra_to_config_basic(self):
        raw = OmegaConf.create({
            "system": {
                "num_servers": 2,
                "arrival_rate": 1.0,
                "service_rates": [1.0, 1.5],
                "alpha": 1.0,
            },
            "simulation": {
                "ssa": {
                    "sim_time": 1000.0,
                    "sample_interval": 0.5,
                },
            },
            "policy": {"name": "softmax"},
            "drift": {"q_max": 50},
        })
        cfg = hydra_to_config(raw)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.system.num_servers == 2
        validate(cfg)
    
    def test_hydra_to_config_missing_optional(self):
        raw = OmegaConf.create({
            "system": {
                "num_servers": 2,
                "arrival_rate": 1.0,
                "service_rates": [1.0, 1.0],
                "alpha": 1.0,
            },
        })
        cfg = hydra_to_config(raw)
        assert cfg.simulation.ssa.sim_time == 5000.0
        assert cfg.policy.name == "softmax"

    def test_hydra_single_service_rate_expands_for_large_n(self):
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

    def test_cuda_alias_is_accepted_in_jax_config(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.5],
                alpha=1.0,
            ),
            jax=JAXConfig(enabled=True, platform="cuda", precision="float32", fallback_to_cpu=True),
        )
        validate(cfg)

    def test_default_stationarity_threshold_requires_all_replications(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.5],
                alpha=1.0,
            ),
        )
        assert cfg.verification.stationarity_threshold == pytest.approx(1.0)

    def test_profile_configs_define_all_experiment_blocks(self):
        config_dir = str(Path("configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            for profile_name in PROFILE_CONFIG_NAMES:
                raw_cfg = compose(config_name=profile_name)
                validate_profile_config(raw_cfg)
                assert sorted(raw_cfg.experiments.keys()) == sorted(EXPERIMENT_BLOCK_NAMES)

    def test_experiment_resolution_keeps_cli_override_precedence(self):
        config_dir = str(Path("configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(
                config_name="default",
                overrides=[
                    "system.alpha=7.0",
                    "simulation.ssa.sim_time=1234.0",
                    "++active_profile=default",
                ],
            )
            resolved = resolve_experiment_config(raw_cfg, "drift", profile_name="default")

        assert float(resolved.system.alpha) == pytest.approx(7.0)
        assert float(resolved.simulation.ssa.sim_time) == pytest.approx(1234.0)
        assert resolved.policy.name == "softmax"

    def test_layered_experiment_resolution_inherits_reinforce_budget_for_ablation(self):
        config_dir = str(Path("configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(
                config_name="debug",
                overrides=["++active_profile=debug"],
            )
            reinforce = resolve_experiment_config(raw_cfg, "reinforce_train", profile_name="debug")
            ablation = resolve_experiment_config_chain(
                raw_cfg,
                ["reinforce_train", "ablation"],
                profile_name="debug",
            )

        assert int(ablation.train_epochs) == int(reinforce.train_epochs) == 5
        assert int(ablation.batch_size) == int(reinforce.batch_size) == 4
        assert float(ablation.simulation.ssa.sim_time) == pytest.approx(
            float(reinforce.simulation.ssa.sim_time)
        )
        assert int(ablation.neural_training.eval_batches) == int(reinforce.neural_training.eval_batches)
        assert int(ablation.neural_training.eval_trajs_per_batch) == int(
            reinforce.neural_training.eval_trajs_per_batch
        )
        assert OmegaConf.select(ablation, "ablation_training") is None

    def test_layered_experiment_resolution_preserves_ablation_specific_overrides(self):
        raw_cfg = OmegaConf.create(
            {
                "active_profile": "debug",
                "system": {
                    "num_servers": 2,
                    "arrival_rate": 1.0,
                    "service_rates": [1.0, 1.5],
                    "alpha": 1.0,
                },
                "simulation": {
                    "num_replications": 10,
                    "seed": 42,
                    "burn_in_fraction": 0.2,
                    "export_trajectories": False,
                    "ssa": {"sim_time": 1000.0, "sample_interval": 1.0},
                    "dga": {"sim_steps": 1000, "temperature": 0.5},
                },
                "policy": {"name": "uas", "d": 2},
                "drift": {"q_max": 50, "use_grid": True},
                "wandb": {"enabled": False, "project": "GibbsQ-Debug", "mode": "offline", "run_name": None},
                "jax": {"enabled": False, "platform": "auto", "precision": "float32", "fallback_to_cpu": True},
                "jax_engine": {
                    "max_events_safety_multiplier": 1.5,
                    "max_events_additive_buffer": 1000,
                    "scan_sampling_chunk": 16,
                },
                "neural": {
                    "hidden_size": 64,
                    "preprocessing": "log1p",
                    "init_type": "zero_final",
                    "use_rho": True,
                    "use_service_rates": True,
                    "rho_input_scale": 10.0,
                    "entropy_bonus": 0.01,
                    "entropy_final": 0.001,
                    "clip_global_norm": 0.5,
                    "actor_lr": 0.001,
                    "critic_lr": 0.005,
                    "lr_decay_rate": 0.9,
                    "weight_decay": 0.0001,
                },
                "verification": {
                    "parity_threshold_percent": 25.0,
                    "jacobian_rel_tol": 0.05,
                    "alpha_significance": 0.05,
                    "confidence_interval": 0.95,
                    "stationarity_threshold": 1.0,
                    "parity_z_score": 1.96,
                    "gradient_check_chunk_size": 100,
                    "gradient_check_max_steps": 50,
                    "gradient_check_n_test": 50,
                    "gradient_check_hidden_size": 128,
                    "gradient_check_sim_time": 10.0,
                    "gradient_check_n_samples": 5000,
                    "gradient_check_epsilon": 0.05,
                    "gradient_check_cosine_threshold": 0.9,
                    "gradient_check_error_threshold": 0.30,
                    "gradient_shake_scale": 0.1,
                },
                "generalization": {
                    "rho_boundary_vals": [0.95],
                    "scale_vals": [0.5, 2.0],
                    "rho_grid_vals": [0.5, 0.85],
                },
                "stress": {
                    "n_values": [4, 8],
                    "critical_rhos": [0.9],
                    "mu_het": [10.0, 0.1, 0.1, 0.1],
                    "sample_interval": 1.0,
                    "massive_n_rho": 0.8,
                    "massive_n_sim_time": 500.0,
                    "critical_load_n": 10,
                    "critical_load_base_rho": 0.8,
                    "critical_load_max_sim_time": 100000.0,
                    "heterogeneity_rho": 0.5,
                    "heterogeneity_sim_time": 1000.0,
                },
                "stability_sweep": {
                    "alpha_vals": [0.1, 1.0, 5.0],
                    "rho_vals": [0.5, 0.8, 0.9],
                },
                "domain_randomization": {"enabled": True, "rho_min": 0.4, "rho_max": 0.98},
                "neural_training": {
                    "learning_rate": 3e-3,
                    "dga_learning_rate": 0.1,
                    "weight_decay": 1e-4,
                    "min_temperature": 0.3,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "curriculum": [[20, 1000]],
                    "eval_batches": 1,
                    "eval_trajs_per_batch": 3,
                    "perf_index_min_denom": 0.5,
                    "perf_index_jsq_margin": 0.05,
                    "bc_num_steps": 200,
                    "bc_lr": 0.002,
                    "bc_label_smoothing": 0.1,
                    "shake_scale": 0.01,
                    "checkpoint_freq": 10,
                    "squash_scale": 100.0,
                    "squash_threshold": 500.0,
                },
                "output_dir": "outputs/debug",
                "log_dir": "${output_dir}/logs",
                "train_epochs": 30,
                "batch_size": 16,
                "experiments": {name: {} for name in EXPERIMENT_BLOCK_NAMES},
            }
        )
        raw_cfg.experiments.reinforce_train = {
            "simulation": {"ssa": {"sim_time": 1000.0}},
            "neural_training": {"eval_batches": 1, "eval_trajs_per_batch": 3},
            "train_epochs": 5,
            "batch_size": 4,
        }
        raw_cfg.experiments.ablation = {
            "simulation": {"num_replications": 7},
            "neural_training": {"checkpoint_freq": 99},
            "ablation_training": {
                "train_epochs": 2,
                "batch_size": 3,
                "simulation": {"ssa": {"sim_time": 250.0}},
                "neural_training": {"eval_batches": 1, "eval_trajs_per_batch": 2},
            },
        }

        resolved = resolve_experiment_config_chain(
            raw_cfg,
            ["reinforce_train", "ablation"],
            profile_name="debug",
        )

        assert int(resolved.train_epochs) == 5
        assert int(resolved.batch_size) == 4
        assert float(resolved.simulation.ssa.sim_time) == pytest.approx(1000.0)
        assert int(resolved.simulation.num_replications) == 7
        assert int(resolved.neural_training.checkpoint_freq) == 99
        assert int(resolved.neural_training.eval_batches) == 1
        assert int(resolved.neural_training.eval_trajs_per_batch) == 3
        assert int(OmegaConf.select(resolved, "ablation_training.train_epochs")) == 2
        assert int(OmegaConf.select(resolved, "ablation_training.batch_size")) == 3
        assert float(OmegaConf.select(resolved, "ablation_training.simulation.ssa.sim_time")) == pytest.approx(250.0)
        assert int(OmegaConf.select(resolved, "ablation_training.neural_training.eval_batches")) == 1
        assert int(OmegaConf.select(resolved, "ablation_training.neural_training.eval_trajs_per_batch")) == 2

    def test_profile_invariants_hold_across_active_profiles(self):
        config_dir = str(Path("configs").resolve())
        invariant_paths = {
            "system.alpha": 1.0,
            "policy.name": "uas",
            "policy.d": 2,
            "verification.stationarity_threshold": 1.0,
            "verification.jacobian_rel_tol": 0.05,
            "jax.platform": "auto",
            "jax.fallback_to_cpu": True,
            "jax_engine.max_events_safety_multiplier": 1.5,
            "jax_engine.max_events_additive_buffer": 1000,
            "jax_engine.scan_sampling_chunk": 16,
        }

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raws = {name: compose(config_name=name) for name in PROFILE_CONFIG_NAMES}

        for path, expected in invariant_paths.items():
            values = {name: OmegaConf.select(raw, path) for name, raw in raws.items()}
            assert all(value == expected for value in values.values()), f"{path} diverged: {values}"

    def test_profile_scaled_runtime_budgets_are_monotonic(self):
        config_dir = str(Path("configs").resolve())
        ordered_profiles = ["debug", "small", "default", "final_experiment"]
        monotonic_paths = [
            "simulation.num_replications",
            "simulation.ssa.sim_time",
            "simulation.dga.sim_steps",
            "train_epochs",
            "batch_size",
            "verification.gradient_check_chunk_size",
            "verification.gradient_check_max_steps",
            "verification.gradient_check_n_samples",
        ]

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raws = {name: compose(config_name=name) for name in ordered_profiles}

        for path in monotonic_paths:
            values = [float(OmegaConf.select(raws[name], path)) for name in ordered_profiles]
            assert values == sorted(values), f"{path} is not monotonic: {values}"

        stress_lengths = [len(OmegaConf.select(raws[name], "stress.n_values")) for name in ordered_profiles]
        assert stress_lengths == sorted(stress_lengths), f"stress.n_values is not monotonic: {stress_lengths}"

    def test_final_experiment_locked_runtime_budgets_resolve_exactly(self):
        config_dir = str(Path("configs").resolve())

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(config_name="final_experiment")

            reinforce = resolve_experiment_config(raw_cfg, "reinforce_train", profile_name="final_experiment")
            assert float(reinforce.simulation.ssa.sim_time) == pytest.approx(1000.0)
            assert int(reinforce.train_epochs) == 15
            assert int(reinforce.batch_size) == 16

            generalize = resolve_experiment_config(raw_cfg, "generalize", profile_name="final_experiment")
            assert int(generalize.simulation.num_replications) == 3
            assert float(generalize.simulation.ssa.sim_time) == pytest.approx(15000.0)
            assert list(generalize.generalization.scale_vals) == [0.5, 1.0, 2.0, 5.0]
            assert list(generalize.generalization.rho_grid_vals) == [0.5, 0.7, 0.85, 0.95]

            critical = resolve_experiment_config(raw_cfg, "critical", profile_name="final_experiment")
            assert int(critical.simulation.num_replications) == 2
            assert list(critical.generalization.rho_boundary_vals) == [0.9, 0.95, 0.97, 0.98]

            ablation = resolve_experiment_config_chain(
                raw_cfg,
                ["reinforce_train", "ablation"],
                profile_name="final_experiment",
            )
            assert float(ablation.simulation.ssa.sim_time) == pytest.approx(1000.0)
            assert int(ablation.train_epochs) == 15
            assert int(ablation.batch_size) == 16
            assert int(OmegaConf.select(ablation, "ablation_training.train_epochs")) == 8
            assert int(OmegaConf.select(ablation, "ablation_training.batch_size")) == 8
            assert float(OmegaConf.select(ablation, "ablation_training.simulation.ssa.sim_time")) == pytest.approx(750.0)
            assert int(OmegaConf.select(ablation, "ablation_training.neural_training.eval_batches")) == 1
            assert int(OmegaConf.select(ablation, "ablation_training.neural_training.eval_trajs_per_batch")) == 3
            assert int(OmegaConf.select(ablation, "ablation_training.neural_training.checkpoint_freq")) == 25

    def test_final_experiment_budget_drift_is_rejected_during_resolution(self):
        config_dir = str(Path("configs").resolve())

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(config_name="final_experiment")

        drifted_reinforce = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
        OmegaConf.update(
            drifted_reinforce,
            "experiments.reinforce_train.train_epochs",
            20,
            force_add=True,
        )
        with pytest.raises(ValueError, match="final_experiment budget drift for reinforce_train"):
            resolve_experiment_config(drifted_reinforce, "reinforce_train", profile_name="final_experiment")

        drifted_ablation = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
        OmegaConf.update(
            drifted_ablation,
            "experiments.ablation.ablation_training.train_epochs",
            9,
            force_add=True,
        )
        with pytest.raises(ValueError, match="final_experiment budget drift for ablation"):
            resolve_experiment_config_chain(
                drifted_ablation,
                ["reinforce_train", "ablation"],
                profile_name="final_experiment",
            )

    def test_parameter_freeze_ledger_covers_all_active_parameters(self):
        config_dir = str(Path("configs").resolve())
        ledger_path = Path("configs") / "notes" / "parameter_freeze_ledger.csv"
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raws = {name: compose(config_name=name) for name in PROFILE_CONFIG_NAMES}

        all_paths = set()
        for raw in raws.values():
            runtime_root = _flatten_dict(runtime_root_dict(raw))
            all_paths.update(runtime_root.keys())

            raw_flat = OmegaConf.to_container(raw, resolve=True)
            experiments = raw_flat.get("experiments", {})
            if isinstance(experiments, dict):
                all_paths.update(_flatten_dict({"experiments": experiments}).keys())

        with ledger_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        allowed_classes = {
            "Invariant",
            "Profile-Scaled",
            "Workload-Defining",
            "Experiment-Specific Override",
            "Metadata/Output",
        }
        ledger_paths = {row["Parameter"] for row in rows}
        assert all_paths == ledger_paths
        assert all(row["Classification"] in allowed_classes for row in rows)

class TestPolicyNameEnum:
    def test_all_valid_names(self):
        for policy_name in PolicyName:
            cfg = ExperimentConfig(
                system=SystemConfig(
                    num_servers=2,
                    arrival_rate=1.0,
                    service_rates=[1.0, 1.0],
                    alpha=1.0,
                ),
                policy=PolicyConfig(name=policy_name.value),
            )
            validate(cfg)  # Should not raise
    
    def test_string_to_enum(self):
        assert PolicyName("softmax") == PolicyName.SOFTMAX
        assert PolicyName("uniform") == PolicyName.UNIFORM
        assert PolicyName("jsq") == PolicyName.JSQ

class TestSweepDomainGuards:
    def test_rejects_out_of_range_stability_rho(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.stability_sweep.rho_vals = [0.9, 1.0]
        with pytest.raises(ValueError, match="stability_sweep.rho_vals\\[1\\]"):
            validate(cfg)

    def test_rejects_non_positive_sweep_alpha(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.stability_sweep.alpha_vals = [0.5, 0.0]
        with pytest.raises(ValueError, match="stability_sweep.alpha_vals\\[1\\]"):
            validate(cfg)

    def test_rejects_out_of_range_generalization_rho(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.generalization.rho_grid_vals = [0.5, 1.2]
        with pytest.raises(ValueError, match="generalization.rho_grid_vals\\[1\\]"):
            validate(cfg)

class TestCriticalLoadGuards:
    def test_required_sim_time_matches_scaling_formula(self):
        required = critical_load_required_sim_time(
            base_sim_time=5000.0,
            rho=0.99,
            base_rho=0.8,
        )
        assert required == pytest.approx(100000.0)

    def test_critical_load_sim_time_fails_closed_when_cap_too_small(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.simulation.ssa.sim_time = 5000.0
        cfg.stress.critical_load_base_rho = 0.8
        cfg.stress.critical_load_max_sim_time = 100000.0

        with pytest.raises(ValueError, match="Refusing to truncate the horizon"):
            critical_load_sim_time(cfg, 0.999)

    def test_validate_rejects_critical_load_grid_that_exceeds_cap(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.generalization.rho_boundary_vals = [0.99, 0.999]
        cfg.simulation.ssa.sim_time = 5000.0
        cfg.stress.critical_load_base_rho = 0.8
        cfg.stress.critical_load_max_sim_time = 100000.0

        with pytest.raises(ValueError, match=r"generalization\.rho_boundary_vals\[1\]=0.999"):
            validate(cfg)


class TestConfigValidationCoverage:
    def test_rejects_non_positive_root_training_budget(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.train_epochs = 0
        with pytest.raises(ValueError, match="train_epochs"):
            validate(cfg)

    def test_rejects_invalid_neural_training_curriculum_shape(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.neural_training.curriculum = [[20, 500, 1]]
        with pytest.raises(ValueError, match="neural_training.curriculum\\[0\\]"):
            validate(cfg)

    def test_rejects_non_positive_gradient_check_budget(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.verification.gradient_check_n_test = 0
        with pytest.raises(ValueError, match="verification.gradient_check_n_test"):
            validate(cfg)

    def test_rejects_invalid_stress_targets(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.stress.n_values = [4, 0]
        with pytest.raises(ValueError, match=r"stress\.n_values\[1\]"):
            validate(cfg)

    def test_rejects_invalid_domain_randomization_phase_horizon(self):
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
        )
        cfg.domain_randomization.phases = [
            type(cfg.domain_randomization.phases[0])(rho_min=0.5, rho_max=0.8, epochs=10, horizon=0)
        ]
        with pytest.raises(ValueError, match="horizon"):
            validate(cfg)
