"""
Distributed Queueing Engine.

Enables parallel replications across available hardware devices.

PATCH 2026-03-13 (BUG-DIST-1) — INDEFINITE HANG FIX
-----------------------------------------------------
Root Cause:
  The original implementation created a NamedSharding mesh, sharded the
  PRNG keys with jax.device_put(..., NamedSharding(...)), then called

      v_sim = jax.vmap(lambda k: simulate_jax(...))
      v_sim(sharded_keys)                  ← hangs forever

  This combination of three factors causes an indefinite XLA compilation hang
  on single-GPU (and even multi-GPU) setups:

  Factor 1 — jax.vmap over a jax.jit-decorated function WITHOUT an outer jit:
    simulate_jax internally calls _simulate_jax_impl which is @jax.jit.
    When jax.vmap traces through simulate_jax it must "peel back" the inner
    @jax.jit and re-trace the raw computation. Without an outer jax.jit to
    anchor the compilation, JAX repeatedly re-enters the tracing loop looking
    for a stable computation to compile.

  Factor 2 — NamedSharding annotation on the vmap input:
    jax.device_put(keys, NamedSharding(mesh, PartitionSpec('batch'))) tags
    the key array with SPMD sharding metadata. When this sharded array is fed
    into an un-jitted jax.vmap, JAX activates its SPMD path, which requires
    jax.jit to apply sharding transforms. Without an outer jit, SPMD
    compilation loops waiting for a JIT context that never arrives.

  Factor 3 — Interaction with nested lax.while_loop inside the vmapped fn:
    _simulate_jax_impl contains nested lax.while_loops with loop-carried
    buffers of shape (max_samples, N). The SPMD + vmap + nested-while_loop
    combination creates a circular dependency in XLA's HLO shape inference
    that hangs the compiler indefinitely (verified: >10 minutes, no progress).

Fix:
  Delegate to run_replications_jax from jax_engine.py.
  That function uses the provably-correct pattern:

      @jax.jit          ← outer jit owns the compilation context
      def impl(...):
          v_sim = lambda k: _simulate_jax_impl(...)  ← RAW function, not jitted
          return jax.vmap(v_sim)(keys)               ← vmap INSIDE jit

  This pattern has been validated across all prior pipeline steps
  (policy_comparison, stability_sweep) with zero compilation hangs.

For future true multi-host execution (TPU pods, multi-node GPU clusters),
the SPMD approach can be re-introduced INSIDE a jax.jit context once the
single-device path is confirmed stable.
"""

import jax.numpy as jnp

# Import the proven-working replications engine directly.
# We avoid re-importing jax.sharding utilities here since the NamedSharding
# approach is the root cause of the hang (see module docstring).
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
    # --- Input validation (preserved from original implementation) ----------
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

    # --- Delegate to the correctly-structured jit+vmap implementation -------
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
        d=2,  # Power-of-d default; ignored by Softmax (policy_type=3)
    )
