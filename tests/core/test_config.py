import math
import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from gibbsq.core.config import (
    SystemConfig, SSAConfig, SimulationConfig, PolicyConfig, DriftConfig,
    ExperimentConfig, WandbConfig, JAXConfig,
    validate, total_capacity, load_factor, drift_constant_R, 
    drift_rate_epsilon, compact_set_radius, hydra_to_config, PolicyName,
    critical_load_required_sim_time, critical_load_sim_time,
)
from omegaconf import DictConfig, OmegaConf

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
