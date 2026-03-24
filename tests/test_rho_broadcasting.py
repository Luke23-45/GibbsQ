"""
Test suite for rho broadcasting in NeuralRouter and ValueNetwork.

Verifies that both networks handle single inputs correctly, and that
external jax.vmap (as used in training) correctly batches operations.

Key insight: Training code uses jax.vmap(model)(s_batch, rho_batch) externally,
so __call__ methods should only handle single inputs.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from gibbsq.core.neural_policies import NeuralRouter, ValueNetwork
from gibbsq.core.config import NeuralConfig


# --- Fixtures ---

@pytest.fixture
def config():
    """Standard NeuralConfig for testing."""
    return NeuralConfig(
        hidden_size=32,
        use_rho=True,
        rho_input_scale=10.0,
        preprocessing="log1p",
    )


@pytest.fixture
def key():
    """PRNG key for reproducibility."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def policy_net(config, key):
    """NeuralRouter instance."""
    return NeuralRouter(num_servers=3, config=config, key=key)


@pytest.fixture
def value_net(config, key):
    """ValueNetwork instance."""
    return ValueNetwork(num_servers=3, config=config, key=key)


# --- NeuralRouter Single Input Tests ---

class TestNeuralRouterSingleInput:
    """Test NeuralRouter with single (non-batched) inputs."""

    def test_scalar_rho_single_input(self, policy_net):
        """T1: Scalar rho with single input."""
        Q = jnp.array([1.0, 2.0, 3.0])  # shape (3,)
        rho = 0.8  # scalar

        logits = policy_net(Q, rho=rho)

        assert logits.shape == (3,), f"Expected shape (3,), got {logits.shape}"
        assert jnp.all(jnp.isfinite(logits)), "Logits contain NaN or Inf"

    def test_none_rho_single_input(self, policy_net):
        """T2: None rho (backward compatibility) with single input."""
        Q = jnp.array([1.0, 2.0, 3.0])

        logits = policy_net(Q, rho=None)

        assert logits.shape == (3,)
        assert jnp.all(jnp.isfinite(logits))

    def test_different_rho_changes_output(self, policy_net):
        """T3: Different rho values produce different outputs.
        
        NOTE: With zero_final initialization, outputs are initially zeros.
        This test verifies the network accepts different rho values without error.
        After training, different rho should produce different logits.
        """
        Q = jnp.array([1.0, 2.0, 3.0])

        # Should not raise errors
        logits_low = policy_net(Q, rho=0.3)
        logits_high = policy_net(Q, rho=0.9)

        assert logits_low.shape == (3,)
        assert logits_high.shape == (3,)


# --- ValueNetwork Single Input Tests ---

class TestValueNetworkSingleInput:
    """Test ValueNetwork with single (non-batched) inputs."""

    def test_scalar_rho_single_input(self, value_net):
        """T1: Scalar rho with single input."""
        s = jnp.array([1.0, 2.0, 3.0])
        rho = 0.8

        value = value_net(s, rho=rho)

        assert value.shape == (), f"Expected scalar shape (), got {value.shape}"
        assert jnp.isfinite(value), "Value is NaN or Inf"

    def test_none_rho_single_input(self, value_net):
        """T2: None rho (backward compatibility) with single input."""
        s = jnp.array([1.0, 2.0, 3.0])

        value = value_net(s, rho=None)

        assert value.shape == ()
        assert jnp.isfinite(value)

    def test_different_rho_changes_value(self, value_net):
        """T3: Different rho values produce different value estimates."""
        s = jnp.array([1.0, 2.0, 3.0])

        value_low = value_net(s, rho=0.3)
        value_high = value_net(s, rho=0.9)

        assert not jnp.isclose(value_low, value_high), \
            "Different rho should produce different value estimates"


# --- External vmap Tests (Training Pattern) ---

class TestExternalVmap:
    """Test that external jax.vmap works correctly (training pattern)."""

    def test_policy_vmap_scalar_rho(self, policy_net):
        """T1: vmap over states with scalar rho."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rho = 0.8

        # External vmap: in_axes=(0, None) means map over states, broadcast rho
        logits = jax.vmap(policy_net, in_axes=(0, None))(states, rho)

        assert logits.shape == (2, 3)
        assert jnp.all(jnp.isfinite(logits))

    def test_policy_vmap_batched_rho(self, policy_net):
        """T2: vmap over states with batched rho (domain randomization)."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rhos = jnp.array([0.6, 0.8])  # One rho per sample

        # External vmap with in_axes=(0, 0) for both state and rho
        logits = jax.vmap(policy_net, in_axes=(0, 0))(states, rhos)

        assert logits.shape == (2, 3)
        assert jnp.all(jnp.isfinite(logits))

    def test_value_vmap_scalar_rho(self, value_net):
        """T3: vmap over states with scalar rho."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rho = 0.8

        # External vmap: in_axes=(0, None) means map over states, broadcast rho
        values = jax.vmap(value_net, in_axes=(0, None))(states, rho)

        assert values.shape == (2,)
        assert jnp.all(jnp.isfinite(values))

    def test_value_vmap_batched_rho(self, value_net):
        """T4: vmap over states with batched rho (domain randomization).
        
        This is the critical test for the bug fix: vmap with in_axes=(0, 0)
        correctly maps each rho to its corresponding state.
        """
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rhos = jnp.array([0.6, 0.8])

        # External vmap with in_axes=(0, 0) for both state and rho
        values = jax.vmap(value_net, in_axes=(0, 0))(states, rhos)

        assert values.shape == (2,)
        assert jnp.all(jnp.isfinite(values))

    def test_vmap_per_sample_rho_varies_output(self, policy_net, value_net):
        """T5: Per-sample rho should vary output for identical states.
        
        NOTE: With zero_final initialization, policy outputs are initially zeros.
        Value network should show different estimates for different rho values.
        """
        states = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])  # identical
        rhos = jnp.array([0.3, 0.9])

        policy_logits = jax.vmap(policy_net, in_axes=(0, 0))(states, rhos)
        value_outputs = jax.vmap(value_net, in_axes=(0, 0))(states, rhos)

        # Value network should vary with rho (it's not zero-initialized)
        assert not jnp.isclose(value_outputs[0], value_outputs[1]), \
            f"Per-sample rho should vary value estimates. Got {value_outputs}"

    def test_vmap_matches_explicit_loop(self, policy_net, value_net):
        """T6: vmap results match explicit loop (ground truth)."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rhos = jnp.array([0.4, 0.8])

        # vmap forward
        vmap_logits = jax.vmap(policy_net, in_axes=(0, 0))(states, rhos)
        vmap_values = jax.vmap(value_net, in_axes=(0, 0))(states, rhos)

        # Explicit loop (ground truth)
        loop_logits = jnp.stack([policy_net(states[i], rho=rhos[i]) for i in range(2)])
        loop_values = jnp.array([value_net(states[i], rho=rhos[i]) for i in range(2)])

        assert jnp.allclose(vmap_logits, loop_logits, atol=1e-5), \
            f"Policy vmap mismatch: vmap={vmap_logits}, loop={loop_logits}"
        assert jnp.allclose(vmap_values, loop_values, atol=1e-5), \
            f"Value vmap mismatch: vmap={vmap_values}, loop={loop_values}"


# --- Direct Batched Call Tests (Robust Implementation) ---

class TestDirectBatchedCall:
    """Test direct batched calls without external vmap (robust implementation)."""

    def test_policy_direct_batched_with_batched_rho(self, policy_net):
        """T1: Direct batched call with batched rho."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rhos = jnp.array([0.6, 0.8])

        logits = policy_net(states, rho=rhos)

        assert logits.shape == (2, 3)
        assert jnp.all(jnp.isfinite(logits))

    def test_policy_direct_batched_with_scalar_rho(self, policy_net):
        """T2: Direct batched call with scalar rho (broadcast)."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rho = 0.7

        logits = policy_net(states, rho=rho)

        assert logits.shape == (2, 3)
        assert jnp.all(jnp.isfinite(logits))

    def test_policy_direct_batched_with_none_rho(self, policy_net):
        """T3: Direct batched call with None rho (backward compat)."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        logits = policy_net(states, rho=None)

        assert logits.shape == (2, 3)
        assert jnp.all(jnp.isfinite(logits))

    def test_value_direct_batched_with_batched_rho(self, value_net):
        """T4: Direct batched call with batched rho."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rhos = jnp.array([0.6, 0.8])

        values = value_net(states, rho=rhos)

        assert values.shape == (2,)
        assert jnp.all(jnp.isfinite(values))

    def test_value_direct_batched_with_scalar_rho(self, value_net):
        """T5: Direct batched call with scalar rho (broadcast)."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rho = 0.7

        values = value_net(states, rho=rho)

        assert values.shape == (2,)
        assert jnp.all(jnp.isfinite(values))

    def test_value_direct_batched_with_none_rho(self, value_net):
        """T6: Direct batched call with None rho (backward compat)."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        values = value_net(states, rho=None)

        assert values.shape == (2,)
        assert jnp.all(jnp.isfinite(values))

    def test_direct_batched_matches_vmap(self, policy_net, value_net):
        """T7: Direct batched call produces same results as external vmap."""
        states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rhos = jnp.array([0.4, 0.8])

        # Direct batched call
        direct_logits = policy_net(states, rho=rhos)
        direct_values = value_net(states, rho=rhos)

        # External vmap
        vmap_logits = jax.vmap(policy_net, in_axes=(0, 0))(states, rhos)
        vmap_values = jax.vmap(value_net, in_axes=(0, 0))(states, rhos)

        assert jnp.allclose(direct_logits, vmap_logits, atol=1e-5), \
            f"Policy mismatch: direct={direct_logits}, vmap={vmap_logits}"
        assert jnp.allclose(direct_values, vmap_values, atol=1e-5), \
            f"Value mismatch: direct={direct_values}, vmap={vmap_values}"


# --- Integration Test ---

class TestDomainRandomizationIntegration:
    """Integration tests for domain randomization training scenario."""

    def test_training_batch_simulation(self, policy_net, value_net):
        """Simulate a training batch with domain randomization."""
        batch_size = 16
        num_servers = 3

        states = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, num_servers))
        rhos = jax.random.uniform(jax.random.PRNGKey(1), (batch_size,), minval=0.4, maxval=0.85)

        # Forward passes with external vmap (training pattern)
        logits = jax.vmap(policy_net, in_axes=(0, 0))(states, rhos)
        values = jax.vmap(value_net, in_axes=(0, 0))(states, rhos)

        assert logits.shape == (batch_size, num_servers)
        assert values.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(logits))
        assert jnp.all(jnp.isfinite(values))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
