"""
Distributed Queueing Engine using JAX Sharding.

Enables scaling to massive N-server regimes (e.g., N=2048) and millions
of replications by partitioning the simulation state across TPUs/GPUs.
"""
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
# Import the core simulator from jax_engine
from gibbsq.engines.jax_engine import simulate_jax

def sharded_replications(
    num_replications: int,
    num_servers: int,
    arrival_rate: float,
    service_rates: jnp.ndarray,
    alpha: float,
    sim_time: float,
    sample_interval: float,
    base_seed: int,
    max_samples: int,
    policy_type: int = 3,
):
    """
    Run N replications in parallel across ALL available hardware devices.
    Replaces jax.vmap with jax.jit + NamedSharding for multi-host execution.
    """
    if num_replications < 1:
        raise ValueError(f"num_replications must be >= 1, got {num_replications}")
    if num_servers < 1:
        raise ValueError(f"num_servers must be >= 1, got {num_servers}")
    if sample_interval <= 0.0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")
    if sim_time < 0.0:
        raise ValueError(f"sim_time must be >= 0, got {sim_time}")
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")
    if service_rates.shape != (num_servers,):
        raise ValueError(
            f"service_rates must have shape ({num_servers},), got {service_rates.shape}"
        )

    devices = jax.devices()

    # Define a 1D mesh of all available devices (e.g., 8 GPUs or TPU slices)
    mesh = Mesh(devices, axis_names=('batch',))
    
    # We partition the replications (batch dimension) across the devices
    sharding = NamedSharding(mesh, PartitionSpec('batch'))
    
    # 1. Create and shard the PRNG keys
    # Creating them on CPU first, then sharding them to devices
    keys = jax.random.split(jax.random.PRNGKey(base_seed), num_replications)
    sharded_keys = jax.device_put(keys, sharding)
    
    # 2. Define the vectorized simulation step
    v_sim = jax.vmap(
        lambda k: simulate_jax(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            alpha=alpha,
            sim_time=sim_time,
            sample_interval=sample_interval,
            key=k,
            max_samples=max_samples,
            policy_type=policy_type,
            d=2
        )
    )
    
    # 3. Run. JAX distributes computation via SPMD since inputs are sharded.
    
    times_buf, states_buf, counts = v_sim(sharded_keys)
    return times_buf, states_buf, counts
