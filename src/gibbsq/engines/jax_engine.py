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


from gibbsq.core import constants
from gibbsq.engines.jax_ssa import compute_poisson_max_steps

RATE_EPSILON = constants.RATE_GUARD_EPSILON

POLICY_NAME_TO_TYPE = {
    "uniform": 0,
    "proportional": 1,
    "jsq": 2,
    "softmax": 3,
    "power_of_d": 4,
    "jssq": 5,
    "uas": 6,
}


def policy_name_to_type(name: str) -> int:
    """Map a public policy name onto the compiled JAX policy selector id."""
    try:
        return POLICY_NAME_TO_TYPE[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported JAX policy name '{name}'. "
            f"Expected one of {sorted(POLICY_NAME_TO_TYPE)}."
        ) from exc


def _validate_inputs(
    *,
    num_servers: int,
    arrival_rate: float,
    service_rates: jnp.ndarray,
    alpha: float,
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
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")
    required_samples = int(sim_time / sample_interval) + 1
    if max_samples < required_samples:
        raise ValueError(
            "max_samples must cover the full sampling grid through sim_time: "
            f"required >= {required_samples}, got {max_samples}"
        )
    if policy_type not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError(f"policy_type must be one of 0..6, got {policy_type}")
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
    sample_idx:      jnp.ndarray    # scalar int — index of next sample slot to write
    times_buf:       jnp.ndarray    # shape (max_samples,) — accumulated sample times
    states_buf:      jnp.ndarray    # shape (max_samples, N) — accumulated sample states


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
    max_samples:     int = 1        # Static bound for sample buffer
    scan_sampling_chunk: int = 16   # Max events recorded per step


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
        
    elif params.policy_type == 3:  # Softmax on queue lengths
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
        
    elif params.policy_type == 5:  # Look-Ahead Potential Softmax (heterogeneity-aware)
        # SG#7 FIX: Use (Q+1)/mu to match policies.py, drift.py, and
        # the look_ahead_potential used during neural network training.
        # Previously used Q/mu which was inconsistent (shifted input domain).
        potential = (Q.astype(params.service_rates.dtype) + 1.0) / params.service_rates
        logits = -params.alpha * potential
        max_logit = jnp.max(logits)
        exp_logits = jnp.exp(logits - max_logit)
        return exp_logits / jnp.sum(exp_logits)
        
    elif params.policy_type == 6:  # UAS (Unified Archimedean Softmax)
        # UAS formula: p_i ∝ μ_i * exp(-α * (Q_i + 1) / μ_i)
        # The μ_i weighting provides capacity-aware routing
        potential = (Q.astype(params.service_rates.dtype) + 1.0) / params.service_rates
        logits = -params.alpha * potential
        # Add log(μ_i) to logits = μ_i * exp(...) in log space
        logits = logits + jnp.log(params.service_rates)
        max_logit = jnp.max(logits)
        exp_logits = jnp.exp(logits - max_logit)
        return exp_logits / jnp.sum(exp_logits)
        
    else:
        # Fallback to avoid compilation errors
        return jnp.ones(params.num_servers) / params.num_servers


# ──────────────────────────────────────────────────────────────
#  Gillespie Step (Scan Body)
# ──────────────────────────────────────────────────────────────

def scan_body(state: SimState, _, params: SimParams) -> tuple[SimState, None]:
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
    # SG-5 FIX: Clamp event index to [0, 2N−1] before branching.
    # When a0 ≤ RATE_EPSILON, safe_a0 > a0 so u can exceed cumrates[-1],
    # making jnp.sum(cumrates < u) = 2N (out-of-bounds). The minimum
    # eliminates the latent OOB scatter and the silent JAX index clamp.
    event = jnp.minimum(jnp.sum(cumrates < u), 2 * params.num_servers - 1)
    
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
        departure_count=final_departure_count,
        sample_idx=state.sample_idx,
        times_buf=state.times_buf,
        states_buf=state.states_buf,
    )
    
    # ── Online sample recording ────────────────────────────────────────────
    # Write the state BEFORE this event for any sample boundary that final_t crossed.
    # At typical Gillespie rates (large N, high event density), tau << sample_interval,
    # so at most one boundary is crossed per step. A fori_loop handles the rare
    # multi-boundary case without dynamic shapes.
    def _write_one(carry, _):
        idx, t_buf, s_buf = carry
        sample_t = idx.astype(jnp.float32) * params.sample_interval
        should = (
            (idx < params.max_samples) &
            (sample_t <= params.sim_time) &
            (final_t >= sample_t)
        )
        t_buf   = t_buf.at[idx].set(jnp.where(should, sample_t, t_buf[idx]))
        s_buf   = s_buf.at[idx].set(jnp.where(should, state.Q,  s_buf[idx]))
        new_idx = idx + jnp.where(should, 1, 0)
        return (new_idx, t_buf, s_buf), None

    # Bound the inner loop to scan_sampling_chunk iterations — handles bursts of crossings
    # per step.
    (new_sample_idx, new_times_buf, new_states_buf), _ = lax.scan(
        _write_one,
        (next_state.sample_idx, next_state.times_buf, next_state.states_buf),
        xs=None,
        length=params.scan_sampling_chunk,
    )
    next_state = SimState(
        t=next_state.t,
        Q=next_state.Q,
        key=next_state.key,
        arrival_count=next_state.arrival_count,
        departure_count=next_state.departure_count,
        sample_idx=new_sample_idx,
        times_buf=new_times_buf,
        states_buf=new_states_buf,
    )
    return next_state, None


def _fill_remaining_samples(state: SimState, params: SimParams) -> SimState:
    """Fill the flat terminal tail so the sampled grid reaches sim_time."""
    required_samples = jnp.minimum(
        jnp.array(params.max_samples, dtype=jnp.int32),
        jnp.floor(params.sim_time / params.sample_interval).astype(jnp.int32) + 1,
    )

    def _body(i, carry):
        times_buf, states_buf = carry
        sample_t = i.astype(jnp.float32) * params.sample_interval
        times_buf = times_buf.at[i].set(sample_t)
        states_buf = states_buf.at[i].set(state.Q)
        return times_buf, states_buf

    filled_times, filled_states = lax.fori_loop(
        state.sample_idx,
        required_samples,
        _body,
        (state.times_buf, state.states_buf),
    )

    return SimState(
        t=state.t,
        Q=state.Q,
        key=state.key,
        arrival_count=state.arrival_count,
        departure_count=state.departure_count,
        sample_idx=required_samples,
        times_buf=filled_times,
        states_buf=filled_states,
    )


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("num_servers", "max_samples", "max_events", "policy_type", "d", "scan_sampling_chunk"))
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
    d:               int = 2,
    scan_sampling_chunk: int = 16
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
        max_samples=max_samples,
        policy_type=policy_type,
        d=d,
        scan_sampling_chunk=scan_sampling_chunk
    )
    
    # Pre-allocate sample buffers in init_state (O(max_samples × N), not O(max_events × N))
    init_state = SimState(
        t=0.0,
        Q=jnp.zeros(num_servers, dtype=jnp.int32),
        key=key,
        arrival_count=0,
        departure_count=0,
        sample_idx=jnp.array(0, dtype=jnp.int32),
        times_buf=jnp.zeros(max_samples, dtype=jnp.float32),
        states_buf=jnp.zeros((max_samples, num_servers), dtype=jnp.int32),
    )
    
    # O(E): Advance the CTMC; samples are recorded into carry-state buffers inline.
    # Per-step scan outputs are None — no O(max_events × N) array is ever materialised.
    final_state, _ = lax.scan(
        lambda s, _: scan_body(s, _, params),
        init_state,
        None,
        length=params.max_events
    )
    
    final_state = _fill_remaining_samples(final_state, params)

    # Extract the inline-recorded sample buffers directly from carry state.
    # Memory used: O(max_samples × N) — a ~5500x reduction for N=1024 vs prior design.
    query_times    = final_state.times_buf
    sampled_states = final_state.states_buf
    
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
    d:               int = 2,
    max_events_multiplier: float = 1.5,
    max_events_buffer: int = 1000,
    scan_sampling_chunk: int = 16
):
    """Validated wrapper around the jitted JAX simulation kernel."""
    _validate_inputs(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        alpha=alpha,
        sim_time=sim_time,
        sample_interval=sample_interval,
        max_samples=max_samples,
        policy_type=policy_type,
        d=d,
    )
    
    import numpy as np
    # Patch: Use Poisson 6-Sigma bound for absolute mathematical safety at high load
    max_events = compute_poisson_max_steps(arrival_rate, np.array(service_rates), sim_time)
    
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
        scan_sampling_chunk=scan_sampling_chunk
    )
    
    import warnings
    if not np.asarray(is_valid).item():
        warnings.warn(f"JAX lax.scan MaxEvents truncation limit reached! Some paths did not finish sim_time={sim_time}. Increase the multiplier (current={max_events_multiplier}x).", RuntimeWarning, stacklevel=2)
        
    return times, states, counts


@partial(jax.jit, static_argnames=("num_replications", "num_servers", "max_samples", "max_events", "policy_type", "d", "scan_sampling_chunk"))
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
    d:                int = 2,
    scan_sampling_chunk: int = 16
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
        d=d,
        scan_sampling_chunk=scan_sampling_chunk
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
    d:                int = 2,
    max_events_multiplier: float = 1.5,
    max_events_buffer: int = 1000,
    scan_sampling_chunk: int = 16
):
    """Validated wrapper around vmapped JAX replications."""
    if num_replications < 1:
        raise ValueError(f"num_replications must be >= 1, got {num_replications}")
    _validate_inputs(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        alpha=alpha,
        sim_time=sim_time,
        sample_interval=sample_interval,
        max_samples=max_samples,
        policy_type=policy_type,
        d=d,
    )
    
    import numpy as np
    
    # Patch: Use Poisson 6-Sigma bound for absolute mathematical safety at high load
    max_events = compute_poisson_max_steps(arrival_rate, np.array(service_rates), sim_time)
    
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
        scan_sampling_chunk=scan_sampling_chunk
    )
    
    # VRAM footprint safety check: Extract boolean back to Host
    import warnings
    # Verify that all replications ran to completion successfully
    if not np.all(np.asarray(is_valid)):
        warnings.warn(f"JAX lax.scan MaxEvents truncation limit reached! Some paths did not finish sim_time={sim_time}. Increase the multiplier (current={max_events_multiplier}x).", RuntimeWarning, stacklevel=2)
        
    return times, states, counts
