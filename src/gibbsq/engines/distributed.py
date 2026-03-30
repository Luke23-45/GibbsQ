"""
Distributed Queueing Engine.

Enables parallel replications across available hardware devices.

Note: This module delegates to run_replications_jax from jax_engine.py, which uses
the provably-correct pattern of @jax.jit wrapping jax.vmap over the raw simulation
kernel. This avoids the XLA compilation hang that occurs when combining:
- jax.vmap over a @jax.jit-decorated function without an outer jit
- NamedSharding annotation on the vmap input
- Nested lax.while_loop with loop-carried buffers

For multi-host execution (TPU pods, multi-node GPU clusters), SPMD sharding
can be re-introduced inside a jax.jit context once single-device is stable.
"""

import jax.numpy as jnp

from gibbsq.engines.jax_engine import run_replications_jax

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
    max_events_multiplier: float = 1.5,
    max_events_buffer: int = 1000,
    scan_sampling_chunk: int = 16,
) -> tuple:
    """
    Run ``num_replications`` independent CTMC simulations in parallel.

    Delegates to :func:`~gibbsq.engines.jax_engine.run_replications_jax`
    which uses a ``@jax.jit``-wrapped ``jax.vmap`` over the raw (un-jitted)
    simulation kernel — the only pattern that compiles correctly on both
    single-device and multi-device setups without hanging.

    Parameters
    ----------
    num_replications : int   Number of independent replications  R ≥ 1.
    num_servers      : int   Number of parallel servers  N ≥ 1.
    arrival_rate     : float Poisson arrival rate  λ ≥ 0.
    service_rates    : ndarray, shape (N,)  Per-server service rates μ_i > 0.
    alpha            : float Softmax inverse temperature.
    sim_time         : float Simulation horizon T ≥ 0.
    sample_interval  : float Time between trajectory snapshots Δt > 0.
    base_seed        : int   Base PRNG seed; replication r uses seed+r.
    max_samples      : int   Pre-allocated trajectory buffer length ≥ 1.
    policy_type      : int   Routing policy index (3 = Softmax, default).

    Returns
    -------
    (times_buf, states_buf, (arrival_counts, departure_counts))
        Each array has a leading batch dimension of size ``num_replications``.
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
            f"service_rates must have shape ({num_servers},), "
            f"got {service_rates.shape}"
        )

    return run_replications_jax(
        num_replications=num_replications,
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        alpha=alpha,
        sim_time=sim_time,
        sample_interval=sample_interval,
        base_seed=base_seed,
        max_samples=max_samples,
        policy_type=policy_type,
        d=2,
        max_events_multiplier=max_events_multiplier,
        max_events_buffer=max_events_buffer,
        scan_sampling_chunk=scan_sampling_chunk,
    )
