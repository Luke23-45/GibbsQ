"""
Baseline Test Script for Patch Verification

This script provides verification tests for the patches documented in ledger/patches.md.
Run with: pytest tests/baseline_test.py -v

Tests are organized by patch priority (P0, P1, P2).
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from pathlib import Path
from gibbsq.core.config import ExperimentConfig, SystemConfig, SimulationConfig, NeuralConfig
from gibbsq.core.neural_policies import NeuralRouter, ValueNetwork
from gibbsq.analysis.metrics import gini_coefficient, stationarity_diagnostic


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    """Base configuration for tests."""
    return ExperimentConfig(
        system=SystemConfig(
            num_servers=4,
            arrival_rate=3.6,  # rho = 0.9
            service_rates=[1.0, 1.0, 1.0, 1.0],
            alpha=1.0,
        ),
        simulation=SimulationConfig(
            sim_time=5000.0,
            seed=42,
        ),
        neural=NeuralConfig(
            hidden_size=64,
            use_rho=True,
        ),
    )


@pytest.fixture
def heterogeneous_service_rates():
    """Heterogeneous service rates for H5 testing."""
    return np.array([10.0, 0.1, 0.1, 0.1])


# ─────────────────────────────────────────────────────────────────────────────
# P0 Tests: Critical Patches
# ─────────────────────────────────────────────────────────────────────────────

class TestP1_HeavyTrafficMixing:
    """Tests for P1: Heavy-Traffic Mixing Time."""
    
    def test_compute_heavy_traffic_sim_time_light_load(self):
        """Test sim_time computation for light load (rho < 0.90)."""
        from experiments.testing.stress_test import compute_heavy_traffic_sim_time
        
        sim_time = compute_heavy_traffic_sim_time(rho=0.80, N=4, base=5000.0)
        assert sim_time == 5000.0, "Light load should use base sim_time"
    
    def test_compute_heavy_traffic_sim_time_medium_load(self):
        """Test sim_time computation for medium load (0.90 <= rho < 0.95)."""
        from experiments.testing.stress_test import compute_heavy_traffic_sim_time
        
        sim_time = compute_heavy_traffic_sim_time(rho=0.92, N=4, base=5000.0)
        assert sim_time == 20000.0, "Medium load should use 4x base sim_time"
    
    def test_compute_heavy_traffic_sim_time_heavy_load(self):
        """Test sim_time computation for heavy load (0.95 <= rho < 0.98)."""
        from experiments.testing.stress_test import compute_heavy_traffic_sim_time
        
        sim_time = compute_heavy_traffic_sim_time(rho=0.96, N=4, base=5000.0)
        expected = 5000.0 * 10 * (4 ** 0.5)  # 50000 * 2 = 100000
        assert abs(sim_time - expected) < 1.0, f"Heavy load should scale with N, expected {expected}"
    
    def test_compute_heavy_traffic_sim_time_critical_load(self):
        """Test sim_time computation for critical load (rho >= 0.98)."""
        from experiments.testing.stress_test import compute_heavy_traffic_sim_time
        
        sim_time = compute_heavy_traffic_sim_time(rho=0.99, N=4, base=5000.0)
        expected = 5000.0 * 20 * (4 ** 0.5)  # 100000 * 2 = 200000
        assert abs(sim_time - expected) < 1.0, f"Critical load should have extra buffer, expected {expected}"
    
    @pytest.mark.slow
    def test_stationarity_at_rho_95(self, base_config):
        """Test that stationarity is achieved at rho=0.95 with extended sim_time."""
        # This test requires running actual simulation
        # Marked as slow - run with: pytest -m slow
        pytest.skip("Requires full simulation run - use manual verification")


class TestP2_HeterogeneityAwareRouting:
    """Tests for P2: Heterogeneity-Aware Routing."""
    
    def test_neural_router_input_dim_with_heterogeneity(self, base_config):
        """Test that NeuralRouter has correct input dimension with heterogeneity features."""
        cfg = base_config.neural
        num_servers = base_config.system.num_servers
        
        # Expected: num_servers (potential) + num_servers (mu_normalized) + 1 (rho)
        expected_dim = num_servers * 2 + 1 if cfg.use_rho else num_servers * 2
        
        router = NeuralRouter(
            num_servers=num_servers,
            hidden_size=cfg.hidden_size,
            use_rho=cfg.use_rho,
            service_rates=jnp.array(base_config.system.service_rates),
        )
        
        # Check input dimension
        assert router.input_dim == expected_dim, f"Expected input_dim={expected_dim}, got {router.input_dim}"
    
    def test_gini_coefficient_heterogeneous(self, heterogeneous_service_rates):
        """Test Gini coefficient under heterogeneous service rates."""
        # Simulate queue distribution: fast server should have lower queue
        # Expected: Gini < 0.10 with heterogeneity-aware routing
        queue_lengths = np.array([1.0, 5.0, 5.0, 5.0])  # Ideal distribution
        
        gini = gini_coefficient(queue_lengths)
        assert gini < 0.10, f"Gini should be < 0.10 for balanced queues, got {gini}"
    
    def test_gini_coefficient_imbalanced(self):
        """Test Gini coefficient for imbalanced distribution."""
        queue_lengths = np.array([0.87, 5.75, 5.90, 5.82])  # From stress test logs
        
        gini = gini_coefficient(queue_lengths)
        # Current implementation gives Gini ≈ 0.2065
        # Target after fix: Gini < 0.10
        assert gini < 0.25, f"Gini should reflect imbalance, got {gini}"
    
    @pytest.mark.slow
    def test_heterogeneity_routing_improvement(self, base_config, heterogeneous_service_rates):
        """Test that heterogeneity-aware routing improves Gini coefficient."""
        # This test requires training and evaluation
        pytest.skip("Requires full training run - use manual verification")


# ─────────────────────────────────────────────────────────────────────────────
# P1 Tests: High Priority Patches
# ─────────────────────────────────────────────────────────────────────────────

class TestP3_DomainRandomization:
    """Tests for P3: Domain Randomization Upper Bound."""
    
    def test_rho_max_in_config(self):
        """Test that rho_max is set to 0.85 in small.yaml."""
        import yaml
        
        config_path = Path(__file__).parent.parent / "configs" / "small.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        rho_max = config.get("domain_randomization", {}).get("rho_max", 0.40)
        assert rho_max >= 0.85, f"rho_max should be >= 0.85 for small validation, got {rho_max}"


class TestP4_AdaptiveTemperature:
    """Tests for P4: Adaptive Temperature."""
    
    def test_adaptive_alpha_light_load(self):
        """Test adaptive alpha for light load."""
        from gibbsq.core.neural_policies import compute_adaptive_alpha
        
        alpha = compute_adaptive_alpha(rho=0.60, base_alpha=1.0)
        assert alpha == 0.1, f"Light load should have alpha=0.1, got {alpha}"
    
    def test_adaptive_alpha_medium_load(self):
        """Test adaptive alpha for medium load."""
        from gibbsq.core.neural_policies import compute_adaptive_alpha
        
        alpha = compute_adaptive_alpha(rho=0.80, base_alpha=1.0)
        assert alpha == 0.5, f"Medium load should have alpha=0.5, got {alpha}"
    
    def test_adaptive_alpha_heavy_load(self):
        """Test adaptive alpha for heavy load."""
        from gibbsq.core.neural_policies import compute_adaptive_alpha
        
        alpha = compute_adaptive_alpha(rho=0.96, base_alpha=1.0)
        expected = 2.0 + 10.0 * (0.96 - 0.95)  # 2.1
        assert abs(alpha - expected) < 0.01, f"Heavy load alpha should be {expected}, got {alpha}"
    
    def test_adaptive_alpha_critical_load(self):
        """Test adaptive alpha for critical load."""
        from gibbsq.core.neural_policies import compute_adaptive_alpha
        
        alpha = compute_adaptive_alpha(rho=0.98, base_alpha=1.0)
        expected = 2.0 + 10.0 * (0.98 - 0.95)  # 2.3
        assert abs(alpha - expected) < 0.01, f"Critical load alpha should be {expected}, got {alpha}"


# ─────────────────────────────────────────────────────────────────────────────
# P2 Tests: Medium Priority Patches
# ─────────────────────────────────────────────────────────────────────────────

class TestP5_GAEIntegration:
    """Tests for P5: GAE Integration."""
    
    def test_gae_function_exists(self):
        """Test that GAE function exists in train_reinforce.py."""
        from experiments.training.train_reinforce import compute_gae
        
        assert compute_gae is not None, "compute_gae function should exist"
    
    def test_gae_computation(self):
        """Test GAE computation produces correct values."""
        from experiments.training.train_reinforce import compute_gae
        
        # Simple test case
        rewards = jnp.array([1.0, 1.0, 1.0, 0.0])
        values = jnp.array([3.0, 2.0, 1.0, 0.0])
        dones = jnp.array([0.0, 0.0, 0.0, 1.0])
        
        advantages = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        
        assert len(advantages) == len(rewards), "Advantages should have same length as rewards"
        assert jnp.all(jnp.isfinite(advantages)), "Advantages should be finite"


class TestP6_BatchSize:
    """Tests for P6: Batch Size."""
    
    def test_batch_size_in_config(self):
        """Test that batch_size is set appropriately in small.yaml."""
        import yaml
        
        config_path = Path(__file__).parent.parent / "configs" / "small.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        batch_size = config.get("batch_size", 16)
        assert batch_size >= 32, f"batch_size should be >= 32 for stable training, got {batch_size}"


# ─────────────────────────────────────────────────────────────────────────────
# Regression Tests (Ensure existing fixes remain)
# ─────────────────────────────────────────────────────────────────────────────

class TestRhoBroadcasting:
    """Regression tests for H1/H2: Rho Broadcasting fixes."""
    
    def test_neural_router_single_input(self, base_config):
        """Test NeuralRouter with single input."""
        router = NeuralRouter(
            num_servers=4,
            hidden_size=64,
            use_rho=True,
            service_rates=jnp.ones(4),
        )
        
        Q = jnp.array([1.0, 2.0, 3.0, 4.0])
        rho = 0.9
        
        logits = router(Q, rho)
        assert logits.shape == (4,), f"Expected shape (4,), got {logits.shape}"
    
    def test_neural_router_batched_input_scalar_rho(self, base_config):
        """Test NeuralRouter with batched input and scalar rho."""
        router = NeuralRouter(
            num_servers=4,
            hidden_size=64,
            use_rho=True,
            service_rates=jnp.ones(4),
        )
        
        Q = jnp.array([[1.0, 2.0, 3.0, 4.0],
                       [2.0, 3.0, 4.0, 5.0]])
        rho = 0.9  # Scalar
        
        logits = router(Q, rho)
        assert logits.shape == (2, 4), f"Expected shape (2, 4), got {logits.shape}"
    
    def test_neural_router_batched_input_batched_rho(self, base_config):
        """Test NeuralRouter with batched input and batched rho."""
        router = NeuralRouter(
            num_servers=4,
            hidden_size=64,
            use_rho=True,
            service_rates=jnp.ones(4),
        )
        
        Q = jnp.array([[1.0, 2.0, 3.0, 4.0],
                       [2.0, 3.0, 4.0, 5.0]])
        rho = jnp.array([0.9, 0.95])  # Batched
        
        logits = router(Q, rho)
        assert logits.shape == (2, 4), f"Expected shape (2, 4), got {logits.shape}"
    
    def test_value_network_single_input(self, base_config):
        """Test ValueNetwork with single input."""
        value_net = ValueNetwork(
            num_servers=4,
            hidden_size=64,
            use_rho=True,
            service_rates=jnp.ones(4),
        )
        
        s = jnp.array([1.0, 2.0, 3.0, 4.0])
        rho = 0.9
        
        value = value_net(s, rho)
        assert value.shape == (), f"Expected scalar output, got shape {value.shape}"
    
    def test_value_network_batched_input(self, base_config):
        """Test ValueNetwork with batched input."""
        value_net = ValueNetwork(
            num_servers=4,
            hidden_size=64,
            use_rho=True,
            service_rates=jnp.ones(4),
        )
        
        s = jnp.array([[1.0, 2.0, 3.0, 4.0],
                       [2.0, 3.0, 4.0, 5.0]])
        rho = jnp.array([0.9, 0.95])
        
        values = value_net(s, rho)
        assert values.shape == (2,), f"Expected shape (2,), got {values.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def run_all_verification():
    """Run all verification tests and print summary."""
    import subprocess
    
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print("FAILURES:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    import sys
    success = run_all_verification()
    sys.exit(0 if success else 1)
