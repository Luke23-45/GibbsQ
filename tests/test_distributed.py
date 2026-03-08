import pytest
import jax
import jax.numpy as jnp
import numpy as np
from moeq.engines.distributed import sharded_replications

def test_sharded_replications_basic():
    """Verify sharded replications produce expected batch shapes."""
    num_reps = 4
    num_servers = 2
    sim_time = 10.0
    sample_interval = 1.0
    max_samples = int(sim_time / sample_interval) + 1
    
    # Run with small parameters
    times, states, (arrs, deps) = sharded_replications(
        num_replications=num_reps,
        num_servers=num_servers,
        arrival_rate=2.0,
        service_rates=jnp.array([3.0, 3.0]),
        alpha=1.0,
        sim_time=sim_time,
        sample_interval=sample_interval,
        base_seed=42,
        max_samples=max_samples
    )
    
    # Check shapes
    assert times.shape == (num_reps, max_samples)
    assert states.shape == (num_reps, max_samples, num_servers)
    assert arrs.shape == (num_reps,)
    assert deps.shape == (num_reps,)
    
    # Check finite
    assert jnp.all(jnp.isfinite(states))
    assert jnp.all(states >= 0)

def test_sharding_device_count():
    """Verify logic doesn't crash regardless of device count."""
    devices = jax.devices()
    num_devices = len(devices)
    
    # Even if 1 device (CPU), the sharding logic should be transparent
    test_sharded_replications_basic()
