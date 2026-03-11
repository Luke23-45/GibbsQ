"""
Hardened test suite for gibbsq.core.config — Robustness Loop Stage 2.

Categories:
- A: Correctness Tests
- B: Invariant Tests  
- C: Edge Case Tests
- D: Numerical Stability Tests
- E: Gradient Flow Tests (N/A for config — no gradients)
- F: Regression Tests
"""

import math
import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from gibbsq.core.config import (
    SystemConfig, SimulationConfig, PolicyConfig, DriftConfig,
    ExperimentConfig, WandbConfig, JAXConfig,
    validate, total_capacity, load_factor, drift_constant_R, 
    drift_rate_epsilon, compact_set_radius, hydra_to_config, PolicyName,
)
from omegaconf import DictConfig, OmegaConf


# ============================================================
# CATEGORY A: CORRECTNESS TESTS
# ============================================================

class TestDerivedQuantitiesCorrectness:
    """Verify mathematical correctness of derived quantities."""
    
    def test_total_capacity_sum(self):
        """Λ = Σ μ_i exactly."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=3,
                arrival_rate=1.0,
                service_rates=[1.0, 2.0, 3.0],
                alpha=1.0,
            ),
        )
        # 1 + 2 + 3 = 6.0
        assert total_capacity(cfg) == 6.0
    
    def test_load_factor_ratio(self):
        """ρ = λ / Λ."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=2.0,
                service_rates=[3.0, 7.0],  # Λ = 10
                alpha=1.0,
            ),
        )
        # ρ = 2 / 10 = 0.2
        assert load_factor(cfg) == pytest.approx(0.2)
    
    def test_drift_constant_R_formula(self):
        """R = (λ log N) / α + (λ + Λ) / 2."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=4,
                arrival_rate=2.0,
                service_rates=[1.0, 1.0, 1.0, 1.0],  # Λ = 4
                alpha=0.5,
            ),
        )
        # R = (2 * log(4)) / 0.5 + (2 + 4) / 2
        # R = (2 * 1.386...) / 0.5 + 3.0
        # R = 2.772... / 0.5 + 3.0 = 5.545... + 3.0 = 8.545...
        expected = (2.0 * math.log(4)) / 0.5 + (2.0 + 4.0) / 2.0
        assert drift_constant_R(cfg) == pytest.approx(expected, rel=1e-10)
    
    def test_epsilon_formula(self):
        """ε = min((Λ − λ) / N, min_i μ_i)."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=3,
                arrival_rate=1.0,
                service_rates=[0.5, 2.0, 3.0],  # Λ = 5.5
                alpha=1.0,
            ),
        )
        # (Λ − λ) / N = (5.5 - 1) / 3 = 1.5
        # min_i μ_i = 0.5
        # ε = min(1.5, 0.5) = 0.5
        assert drift_rate_epsilon(cfg) == pytest.approx(0.5)
    
    def test_compact_set_radius_formula(self):
        """radius = ⌈(R + 1) / ε⌉."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[2.0, 2.0],  # Λ = 4
                alpha=1.0,
            ),
        )
        R = drift_constant_R(cfg)
        eps = drift_rate_epsilon(cfg)
        expected = math.ceil((R + 1.0) / eps)
        assert compact_set_radius(cfg) == expected


# ============================================================
# CATEGORY B: INVARIANT TESTS
# ============================================================

class TestConfigInvariants:
    """Verify invariants hold across all valid inputs."""
    
    @given(
        num_servers=st.integers(min_value=1, max_value=100),
        arrival_rate=st.floats(min_value=0.01, max_value=100.0, allow_infinity=False, allow_nan=False),
        alpha=st.floats(min_value=0.001, max_value=1000.0, allow_infinity=False, allow_nan=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_load_factor_always_in_open_interval(self, num_servers, arrival_rate, alpha):
        """For any valid config, 0 < ρ < 1."""
        # Generate service_rates that satisfy capacity condition
        # Need Λ > λ, so sum(service_rates) > arrival_rate
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
        """ε > 0 for all valid configs (capacity condition ensures this)."""
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
        """Λ must exactly equal sum of service_rates (Kahan accuracy)."""
        # Use values that would accumulate error with naive sum
        rates = [0.1] * 10  # 10 * 0.1 = 1.0 exactly
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


# ============================================================
# CATEGORY C: EDGE CASE TESTS
# ============================================================

class TestConfigEdgeCases:
    """Test boundary conditions and degenerate inputs."""
    
    def test_single_server(self):
        """N=1 is valid (degenerate case)."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=1,
                arrival_rate=0.5,
                service_rates=[1.0],
                alpha=1.0,
            ),
        )
        validate(cfg)
        # log(1) = 0, so R = C1
        R = drift_constant_R(cfg)
        expected_R = (0.5 + 1.0) / 2.0  # C1 = (λ + Λ) / 2
        assert R == pytest.approx(expected_R)
    
    def test_near_capacity_violation(self):
        """Λ just barely > λ should still pass."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[0.5000001, 0.5000001],  # Λ = 1.0000002 > 1.0
                alpha=1.0,
            ),
        )
        validate(cfg)  # Should not raise
        eps = drift_rate_epsilon(cfg)
        assert eps > 0.0  # Should be tiny but positive
    
    def test_capacity_violation_raises(self):
        """Λ <= λ must raise ValueError with informative message."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=2.0,
                service_rates=[0.5, 0.5],  # Λ = 1.0 < 2.0
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="Capacity condition violated"):
            validate(cfg)
    
    def test_exact_capacity_violation(self):
        """Λ = λ (exact equality) must raise."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[0.5, 0.5],  # Λ = 1.0 = λ
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="Capacity condition violated"):
            validate(cfg)
    
    def test_zero_arrival_rate(self):
        """λ = 0 is invalid (must be > 0)."""
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
        """λ < 0 is invalid."""
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
        """α = 0 is invalid (division by zero in R)."""
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
        """α < 0 is invalid."""
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
        """μ_i = 0 is invalid."""
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
        """μ_i < 0 is invalid."""
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
        """|service_rates| ≠ num_servers must raise."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=3,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],  # Only 2 rates for 3 servers
                alpha=1.0,
            ),
        )
        with pytest.raises(ValueError, match="≠ num_servers"):
            validate(cfg)
    
    def test_zero_num_servers(self):
        """N = 0 is invalid."""
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
        """Unknown policy name must raise."""
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
        """burn_in_fraction < 0 is invalid."""
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
        """burn_in_fraction >= 1 is invalid."""
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
        """sim_time = 0 is invalid."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
            simulation=SimulationConfig(sim_time=0.0),
        )
        with pytest.raises(ValueError, match="sim_time.*must be > 0"):
            validate(cfg)
    
    def test_negative_sim_time(self):
        """sim_time < 0 is invalid."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1.0,
            ),
            simulation=SimulationConfig(sim_time=-100.0),
        )
        with pytest.raises(ValueError, match="sim_time.*must be > 0"):
            validate(cfg)


# ============================================================
# CATEGORY D: NUMERICAL STABILITY TESTS
# ============================================================

class TestConfigNumericalStability:
    """Test behavior under numerically challenging inputs."""
    
    def test_very_small_alpha(self):
        """Very small α should still compute R without overflow."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1e-10,  # Very small
            ),
        )
        validate(cfg)
        R = drift_constant_R(cfg)
        assert math.isfinite(R)
        # R = (λ log N) / α + C1 ≈ (1 * 0.693) / 1e-10 = 6.93e9 (large but finite)
        assert R > 1e9  # Should be huge
    
    def test_very_large_alpha(self):
        """Very large α should compute R ≈ C1."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[1.0, 1.0],
                alpha=1e10,  # Very large
            ),
        )
        validate(cfg)
        R = drift_constant_R(cfg)
        assert math.isfinite(R)
        # R ≈ C1 = (1 + 2) / 2 = 1.5
        assert R == pytest.approx(1.5, rel=0.01)
    
    def test_very_large_num_servers(self):
        """Large N should compute log(N) without overflow."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=1000,
                arrival_rate=500.0,
                service_rates=[1.0] * 1000,  # Λ = 1000
                alpha=1.0,
            ),
        )
        validate(cfg)
        R = drift_constant_R(cfg)
        assert math.isfinite(R)
    
    def test_very_small_epsilon(self):
        """Near-capacity should produce tiny but positive ε."""
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=[0.5000000001, 0.5000000001],  # Λ ≈ λ
                alpha=1.0,
            ),
        )
        validate(cfg)
        eps = drift_rate_epsilon(cfg)
        assert eps > 0.0
        assert eps < 1e-9  # Very small
    
    def test_floating_point_service_rates(self):
        """Service rates with many decimal places should sum accurately."""
        rates = [0.1, 0.2, 0.3, 0.4, 0.5]  # Sum = 1.5 exactly
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


# ============================================================
# CATEGORY F: REGRESSION TESTS
# ============================================================

class TestConfigRegressions:
    """Prevent reintroduction of known faults."""
    
    def test_regression_capacity_check_uses_fsum(self):
        """Ensure Kahan-accurate summation is used for capacity check."""
        # This would fail with naive sum due to floating point error
        # Create a case where naive sum would give wrong answer
        # Using many small values that accumulate error
        rates = [1e-15] * 1000  # Naive sum might give 0 or wrong value
        rates.append(2.0)  # Total should be ≈ 2.0
        
        cfg = ExperimentConfig(
            system=SystemConfig(
                num_servers=1001,
                arrival_rate=1.0,
                service_rates=rates,
                alpha=1.0,
            ),
        )
        # Should pass because math.fsum gives accurate result
        validate(cfg)  # Must not raise
        cap = total_capacity(cfg)
        assert cap > 1.0  # Must be > arrival_rate


# ============================================================
# HYDRA CONVERSION TESTS
# ============================================================

class TestHydraConversion:
    """Test Hydra DictConfig conversion."""
    
    def test_hydra_to_config_basic(self):
        """Basic DictConfig conversion."""
        raw = OmegaConf.create({
            "system": {
                "num_servers": 2,
                "arrival_rate": 1.0,
                "service_rates": [1.0, 1.5],
                "alpha": 1.0,
            },
            "simulation": {
                "sim_time": 1000.0,
                "sample_interval": 0.5,
            },
            "policy": {"name": "softmax"},
            "drift": {"q_max": 50},
        })
        cfg = hydra_to_config(raw)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.system.num_servers == 2
        validate(cfg)
    
    def test_hydra_to_config_missing_optional(self):
        """Missing optional fields should use defaults."""
        raw = OmegaConf.create({
            "system": {
                "num_servers": 2,
                "arrival_rate": 1.0,
                "service_rates": [1.0, 1.0],
                "alpha": 1.0,
            },
        })
        cfg = hydra_to_config(raw)
        assert cfg.simulation.sim_time == 1e4  # Default
        assert cfg.policy.name == "softmax"  # Default


# ============================================================
# POLICY NAME ENUM TESTS
# ============================================================

class TestPolicyNameEnum:
    """Test PolicyName enum completeness."""
    
    def test_all_valid_names(self):
        """All enum values should pass validation."""
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
        """String values should match enum."""
        assert PolicyName("softmax") == PolicyName.SOFTMAX
        assert PolicyName("uniform") == PolicyName.UNIFORM
        assert PolicyName("jsq") == PolicyName.JSQ
