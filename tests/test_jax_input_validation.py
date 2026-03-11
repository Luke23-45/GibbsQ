import pytest
import jax
import jax.numpy as jnp

from gibbsq.engines.jax_engine import simulate_jax, run_replications_jax


def test_simulate_jax_rejects_invalid_service_rate_shape():
    with pytest.raises(ValueError, match="service_rates must have shape"):
        simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([1.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(0),
            max_samples=20,
        )


def test_simulate_jax_rejects_invalid_policy_type():
    with pytest.raises(ValueError, match="policy_type must be one of 0..4"):
        simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([1.0, 1.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(0),
            max_samples=20,
            policy_type=99,
        )


def test_run_replications_jax_rejects_non_positive_replications():
    with pytest.raises(ValueError, match="num_replications must be >= 1"):
        run_replications_jax(
            num_replications=0,
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([1.0, 1.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            base_seed=0,
            max_samples=20,
        )
