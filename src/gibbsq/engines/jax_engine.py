"""Hardware-accelerated CTMC simulator implemented in JAX.

This module mirrors the NumPy engine contract while using pure functional
state transitions suitable for JIT compilation and vectorization.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import NamedTuple
from dataclasses import dataclass


RATE_EPSILON = 1e-12


def _validate_inputs(
    *,
    num_servers: int,
    arrival_rate: float,
    service_rates: jnp.ndarray,
    sim_time: float,
    sample_interval: float,
    max_samples: int,
    policy_type: int,
    d: int,
) -> None:
    """Host-side validation for JAX simulator inputs."""
    if num_servers < 1:
        raise ValueError(f"num_servers must be >= 1, got {num_servers}")
    if arrival_rate < 0:
        raise ValueError(f"arrival_rate must be >= 0, got {arrival_rate}")
    if sim_time < 0:
        raise ValueError(f"sim_time must be >= 0, got {sim_time}")
    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")
    if policy_type not in (0, 1, 2, 3, 4):
        raise ValueError(f"policy_type must be one of 0..4, got {policy_type}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    if service_rates.shape != (num_servers,):
        raise ValueError(
            f"service_rates must have shape ({num_servers},), got {service_rates.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(service_rates))):
        raise ValueError("service_rates must be finite")
    if not bool(jnp.all(service_rates > 0)):
        raise ValueError("service_rates must be strictly positive")

# ──────────────────────────────────────────────────────────────
#  State & Configuration
# ──────────────────────────────────────────────────────────────

class SimState(NamedTuple):
    """Immutable state for the JAX lax.scan."""
    t:               jnp.ndarray    # scalar float
    Q:               jnp.ndarray    # shape (N,) int
    key:             jax.random.PRNGKey
    arrival_count:   jnp.ndarray    # scalar int
    departure_count: jnp.ndarray    # scalar int


@dataclass(frozen=True)
class SimParams:
    """Static parameters for the simulation."""
    num_servers:     int
    arrival_rate:    float
    service_rates:   jnp.ndarray    # shape (N,)
    alpha:           float
    sim_time:        float
    sample_interval: float
    max_events:      int            # Static bound for lax.scan
    policy_type:     int            # 0: Uniform, 1: Prop, 2: JSQ, 3: Softmax, 4: Power-of-d
    d:               int            # for Power-of-d (default 2)


# ──────────────────────────────────────────────────────────────
#  Routing Logic (compiled)
# ──────────────────────────────────────────────────────────────

def get_probs(Q: jnp.ndarray, params: SimParams, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Policy selector using static branch resolution."""
    # Because policy_type is in static_argnames, it is evaluated at compile time.
    # Using Python conditionals forces XLA to compile ONLY the active branch.
    
    if params.policy_type == 0:    # Uniform
        return jnp.ones(params.num_servers) / params.num_servers
        
    elif params.policy_type == 1:  # Proportional
        return params.service_rates / jnp.sum(params.service_rates)
        
    elif params.policy_type == 2:  # JSQ
        is_min = (Q == jnp.min(Q))
        noise = jax.random.uniform(key, shape=Q.shape)
        masked_noise = jnp.where(is_min, noise, -jnp.inf)
        idx = jnp.argmax(masked_noise)
        return jnp.zeros(params.num_servers).at[idx].set(1.0)
        
    elif params.policy_type == 3:  # Softmax
        logits = -params.alpha * Q.astype(params.service_rates.dtype)
        max_logit = jnp.max(logits)
        exp_logits = jnp.exp(logits - max_logit)
        return exp_logits / jnp.sum(exp_logits)
        
    elif params.policy_type == 4:  # Power-of-d
        N = params.num_servers
        d_actual = min(params.d, N)
        perm = jax.random.permutation(key, N)
        # Static slice is cleanly supported by JAX without dynamic_slice overhead
        candidates = perm[:d_actual]
        candidate_queues = Q[candidates]
        winner_local = jnp.argmin(candidate_queues)
        winner = candidates[winner_local]
        return jnp.zeros(N).at[winner].set(1.0)
        
    else:
        # Fallback to avoid compilation errors
        return jnp.ones(params.num_servers) / params.num_servers


# ──────────────────────────────────────────────────────────────
#  Gillespie Step (Scan Body)
# ──────────────────────────────────────────────────────────────

def scan_body(state: SimState, _, params: SimParams) -> tuple[SimState, tuple[jnp.ndarray, jnp.ndarray]]:
    """Single Gillespie event logic for lax.scan."""
    k1, k2, k3, k4 = jax.random.split(state.key, 4)
    
    # 1. Probabilities & Rates
    probs = get_probs(state.Q, params, k4)
    arrival_rates = params.arrival_rate * probs
    departure_rates = params.service_rates * (state.Q > 0).astype(jnp.float32)
    rates = jnp.concatenate([arrival_rates, departure_rates])
    
    a0 = jnp.sum(rates)
    safe_a0 = jnp.maximum(a0, RATE_EPSILON)
    
    # 2. Time update
    tau = jax.random.exponential(k1) / safe_a0
    new_t = state.t + tau
    
    # Mathematical Boundary Condition:
    # State updates ONLY if the event completes inside the simulation window.
    in_window = (new_t <= params.sim_time) & (a0 > RATE_EPSILON)
    # Time always advances so we cleanly cross the boundary and halt.
    time_advances = (state.t < params.sim_time) & (a0 > RATE_EPSILON)
    
    # 3. State update
    u = jax.random.uniform(k2) * safe_a0
    cumrates = jnp.cumsum(rates)
    event = jnp.sum(cumrates < u)
    
    is_arrival = event < params.num_servers
    srv_idx = jnp.where(is_arrival, event, event - params.num_servers)
    delta = jnp.where(is_arrival, 1, -1)
    
    new_Q = state.Q.at[srv_idx].add(delta)
    new_arrival_count = state.arrival_count + jnp.where(is_arrival, 1, 0)
    new_departure_count = state.departure_count + jnp.where(is_arrival, 0, 1)
    
    # 4. Mask updates safely
    final_t = jnp.where(time_advances, new_t, state.t)
    final_Q = jnp.where(in_window, new_Q, state.Q)
    final_arrival_count = jnp.where(in_window, new_arrival_count, state.arrival_count)
    final_departure_count = jnp.where(in_window, new_departure_count, state.departure_count)
    
    next_state = SimState(
        t=final_t,
        Q=final_Q,
        key=k3,
        arrival_count=final_arrival_count,
        departure_count=final_departure_count
    )
    
    return next_state, (state.t, state.Q)


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("num_servers", "max_samples", "max_events", "policy_type", "d"))
def _simulate_jax_impl(
    num_servers:     int,
    arrival_rate:    float,
    service_rates:   jnp.ndarray,
    alpha:           float,
    sim_time:        float,
    sample_interval: float,
    key:             jax.random.PRNGKey,
    max_samples:     int,
    max_events:      int,
    policy_type:     int = 3,
    d:               int = 2
):
    """Hardware-accelerated simulation of the GibbsQ network using SOTA scan+search."""
    params = SimParams(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        alpha=alpha,
        sim_time=sim_time,
        sample_interval=sample_interval,
        max_events=max_events,
        policy_type=policy_type,
        d=d
    )
    
    init_state = SimState(
        t=0.0,
        Q=jnp.zeros(num_servers, dtype=jnp.int32),
        key=key,
        arrival_count=0,
        departure_count=0
    )
    
    # O(E): Generate the raw stochastic trajectory exactly
    final_state, (all_times, all_states) = lax.scan(
        lambda s, _: scan_body(s, _, params),
        init_state,
        None,
        length=params.max_events
    )
    
    # O(S log E): Interpolate onto the unified time grid
    query_times = jnp.arange(max_samples) * sample_interval
    
    # searchsorted(side='right') gives index where query_time would be inserted
    # subtracting 1 gives the state interval that query_time falls into
    idxs = jnp.searchsorted(all_times, query_times, side='right') - 1
    idxs = jnp.clip(idxs, 0, params.max_events - 1)
    
    sampled_states = all_states[idxs]
    
    # We drop the massive all_times and all_states trajectory arrays here. 
    # JAX XLA naturally garbage collects intermediate arrays not returned.
    
    # Safety Check: Did we truncate because max_events was too small?
    # Valid if we reached sim_time OR if the simulation halted (arrival_rate = 0)
    is_valid = (final_state.t >= sim_time) | (params.arrival_rate == 0.0)
    
    return query_times, sampled_states, (final_state.arrival_count, final_state.departure_count), is_valid


def simulate_jax(
    num_servers:     int,
    arrival_rate:    float,
    service_rates:   jnp.ndarray,
    alpha:           float,
    sim_time:        float,
    sample_interval: float,
    key:             jax.random.PRNGKey,
    max_samples:     int,
    policy_type:     int = 3,
    d:               int = 2
):
    """Validated wrapper around the jitted JAX simulation kernel."""
    _validate_inputs(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        sim_time=sim_time,
        sample_interval=sample_interval,
        max_samples=max_samples,
        policy_type=policy_type,
        d=d,
    )
    
    import numpy as np
    # Calculate MaxEvents dynamically for mathematical safety
    max_theoretical_rate = arrival_rate + float(np.sum(np.array(service_rates)))
    max_events = int(max_theoretical_rate * sim_time * 1.5) + 1000
    
    times, states, counts, is_valid = _simulate_jax_impl(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        alpha=alpha,
        sim_time=sim_time,
        sample_interval=sample_interval,
        key=key,
        max_samples=max_samples,
        max_events=max_events,
        policy_type=policy_type,
        d=d,
    )
    
    import warnings
    if not np.asarray(is_valid).item():
        warnings.warn(f"JAX lax.scan MaxEvents truncation limit reached! Some paths did not finish sim_time={sim_time}. Increase the 1.5x multiplier.", RuntimeWarning, stacklevel=2)
        
    return times, states, counts


@partial(jax.jit, static_argnames=("num_replications", "num_servers", "max_samples", "max_events", "policy_type", "d"))
def _run_replications_jax_impl(
    num_replications: int,
    num_servers:      int,
    arrival_rate:     float,
    service_rates:    jnp.ndarray,
    alpha:            float,
    sim_time:         float,
    sample_interval:  float,
    base_seed:        int,
    max_samples:      int,
    max_events:       int,
    policy_type:      int = 3,
    d:                int = 2
):
    """Run replications in parallel across available accelerator lanes."""
    keys = jax.random.split(jax.random.PRNGKey(base_seed), num_replications)

    v_sim = lambda k: _simulate_jax_impl(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        alpha=alpha,
        sim_time=sim_time,
        sample_interval=sample_interval,
        key=k,
        max_samples=max_samples,
        max_events=max_events,
        policy_type=policy_type,
        d=d
    )

    return jax.vmap(v_sim)(keys)


def run_replications_jax(
    num_replications: int,
    num_servers:      int,
    arrival_rate:     float,
    service_rates:    jnp.ndarray,
    alpha:            float,
    sim_time:         float,
    sample_interval:  float,
    base_seed:        int,
    max_samples:      int,
    policy_type:      int = 3,
    d:                int = 2
):
    """Validated wrapper around vmapped JAX replications."""
    if num_replications < 1:
        raise ValueError(f"num_replications must be >= 1, got {num_replications}")
    _validate_inputs(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        sim_time=sim_time,
        sample_interval=sample_interval,
        max_samples=max_samples,
        policy_type=policy_type,
        d=d,
    )
    
    import numpy as np
    
    # Calculate MaxEvents dynamically for mathematical safety
    # The maximum theoretical event rate is the arrival rate plus the sum of all service rates
    max_theoretical_rate = arrival_rate + float(np.sum(np.array(service_rates)))
    # Apply a 1.5x multiplier to account for extreme stochasticity, plus a buffer
    max_events = int(max_theoretical_rate * sim_time * 1.5) + 1000
    
    times, states, counts, is_valid = _run_replications_jax_impl(
        num_replications=num_replications,
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        alpha=alpha,
        sim_time=sim_time,
        sample_interval=sample_interval,
        base_seed=base_seed,
        max_samples=max_samples,
        max_events=max_events,
        policy_type=policy_type,
        d=d,
    )
    
    # VRAM footprint safety check: Extract boolean back to Host
    import warnings
    # Verify that all replications ran to completion successfully
    if not np.all(np.asarray(is_valid)):
        warnings.warn(f"JAX lax.scan MaxEvents truncation limit reached! Some paths did not finish sim_time={sim_time}. Increase the 1.5x multiplier.", RuntimeWarning, stacklevel=2)
        
    return times, states, counts

