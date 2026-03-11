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
    """Immutable state for the JAX while_loop."""
    t:               jnp.ndarray    # scalar float
    Q:               jnp.ndarray    # shape (N,) int
    key:             jax.random.PRNGKey
    sample_idx:      jnp.ndarray    # scalar int
    times_buf:       jnp.ndarray    # shape (max_samples,) float
    states_buf:      jnp.ndarray    # shape (max_samples, N) int
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
    policy_type:     int            # 0: Uniform, 1: Prop, 2: JSQ, 3: Softmax, 4: Power-of-d
    d:               int            # for Power-of-d (default 2)


# ──────────────────────────────────────────────────────────────
#  Routing Logic (compiled)
# ──────────────────────────────────────────────────────────────

def get_probs(Q: jnp.ndarray, params: SimParams, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Policy selector using lax.switch.
    
    Args:
        Q: Queue lengths (shape N)
        params: Simulation parameters including policy_type and d
        key: PRNG key for Power-of-d sampling
    
    Returns:
        Routing probabilities (shape N)
    """
    
    def uniform_p(_):
        return jnp.ones(params.num_servers) / params.num_servers

    def proportional_p(_):
        return params.service_rates / jnp.sum(params.service_rates)

    def jsq_p(_):
        # Tie-break uniformly over minimum queues to prevent deterministic pileup on the first server
        is_min = (Q == jnp.min(Q))
        noise = jax.random.uniform(key, shape=Q.shape)
        masked_noise = jnp.where(is_min, noise, -jnp.inf)
        idx = jnp.argmax(masked_noise)
        return jnp.zeros(params.num_servers).at[idx].set(1.0)

    def softmax_p(_):
        logits = -params.alpha * Q.astype(params.service_rates.dtype)
        max_logit = jnp.max(logits)
        exp_logits = jnp.exp(logits - max_logit)
        return exp_logits / jnp.sum(exp_logits)
    
    def power_of_d_p(_):
        """Power-of-d choices: sample d servers uniformly, route to shortest.
        
        Returns a one-hot probability vector at the selected server.
        This is a deterministic routing policy given the sampled candidates.
        
        Note: d is a static argument for JIT compilation, enabling use of
        lax.dynamic_slice with static slice size.
        """
        N = params.num_servers
        d_actual = min(params.d, N)  # Clamp d to N (static)
        
        # Shuffle all server indices and take first d
        # This is equivalent to sampling d distinct indices without replacement
        perm = jax.random.permutation(key, N)
        
        # Use dynamic_slice with static size (d is static)
        candidates = lax.dynamic_slice(perm, (0,), (d_actual,))
        
        # Get queue lengths at candidate servers
        candidate_queues = Q[candidates]
        
        # Find the winner (index within candidates of minimum queue)
        winner_local = jnp.argmin(candidate_queues)
        
        # Map back to global server index
        winner = candidates[winner_local]
        
        # Return one-hot at winner
        return jnp.zeros(N).at[winner].set(1.0)

    # Policy types: 0=Uniform, 1=Proportional, 2=JSQ, 3=Softmax, 4=Power-of-d
    return lax.switch(
        params.policy_type, 
        [uniform_p, proportional_p, jsq_p, softmax_p, power_of_d_p], 
        None
    )


# ──────────────────────────────────────────────────────────────
#  Gillespie Step
# ──────────────────────────────────────────────────────────────

def cond_fun(state: SimState, params: SimParams) -> bool:
    """Loop while time < sim_time and we haven't overfilled buffers."""
    max_samples = state.times_buf.shape[0]
    return (state.t < params.sim_time) & (state.sample_idx < max_samples)


def body_fun(state: SimState, params: SimParams) -> SimState:
    """Single Gillespie event and snapshot logic."""
    k1, k2, k3, k4 = jax.random.split(state.key, 4)
    
    # 1. Probabilities (k4 used for Power-of-d sampling if needed)
    probs = get_probs(state.Q, params, k4)
    
    # 2. Rates
    # [Arrivals (N), Departures (N)]
    arrival_rates = params.arrival_rate * probs
    departure_rates = params.service_rates * (state.Q > 0).astype(jnp.float32)
    rates = jnp.concatenate([arrival_rates, departure_rates])
    
    a0 = jnp.sum(rates)
    

    def process_event(s: SimState) -> SimState:
        """Normal event processing when a0 > 0."""
        # 3. Draw holding time  ~ Exp(a0)
        tau = jax.random.exponential(k1) / a0  # a0 > 0 guaranteed here
        new_t = s.t + tau
        
        # 4. Draw event
        u = jax.random.uniform(k2) * a0
        cumrates = jnp.cumsum(rates)
        event = jnp.sum(cumrates < u)
        
        # 5. Apply transition
        is_arrival = event < params.num_servers
        srv_idx = jnp.where(is_arrival, event, event - params.num_servers)
        
        delta = jnp.where(is_arrival, 1, -1)
        new_Q = s.Q.at[srv_idx].add(delta)
        
        new_arrival_count = s.arrival_count + jnp.where(is_arrival, 1, 0)
        new_departure_count = s.departure_count + jnp.where(is_arrival, 0, 1)
        
        # Use a while_loop to fill all samples between old t and new_t
        def fill_samples(carry):
            """Fill all sample slots that fall between current t and new_t."""
            idx, times_b, states_b = carry
            next_sample_t = idx * params.sample_interval
            
            # Check if this sample time is crossed by the event
            should_record = (new_t >= next_sample_t) & (next_sample_t <= params.sim_time)
            
            # Record the state BEFORE the event (s.Q) at this sample time
            new_times_b = times_b.at[idx].set(
                jnp.where(should_record, next_sample_t, times_b[idx])
            )
            new_states_b = states_b.at[idx].set(
                jnp.where(should_record, s.Q, states_b[idx])
            )
            new_idx = idx + jnp.where(should_record, 1, 0)
            
            return (new_idx, new_times_b, new_states_b)
        
        
        # Fill samples until we reach the sample time that new_t falls before
        # Maximum iterations = max_samples to prevent infinite loop
        max_iters = s.times_buf.shape[0]
        final_idx, final_times, final_states = lax.while_loop(
            lambda carry: (carry[0] < max_iters) & 
                         (carry[0] * params.sample_interval <= new_t) & 
                         (carry[0] * params.sample_interval <= params.sim_time),
            fill_samples,
            (s.sample_idx, s.times_buf, s.states_buf)
        )
        
        return s._replace(
            t=new_t,
            Q=new_Q,
            key=k3,
            sample_idx=final_idx,
            times_buf=final_times,
            states_buf=final_states,
            arrival_count=new_arrival_count,
            departure_count=new_departure_count
        )
    
    def skip_event(s: SimState) -> SimState:
        """When a0=0, no events can occur - advance time past sim_time to exit loop."""
        # Record any remaining samples up to sim_time first
        max_iters = s.times_buf.shape[0]
        
        def fill_remaining(carry):
            idx, times_b, states_b = carry
            next_sample_t = idx * params.sample_interval
            should_record = next_sample_t <= params.sim_time
            
            new_times_b = times_b.at[idx].set(
                jnp.where(should_record, next_sample_t, times_b[idx])
            )
            new_states_b = states_b.at[idx].set(
                jnp.where(should_record, s.Q, states_b[idx])
            )
            new_idx = idx + jnp.where(should_record, 1, 0)
            return (new_idx, new_times_b, new_states_b)
        
        
        final_idx, final_times, final_states = lax.while_loop(
            lambda carry: (carry[0] < max_iters) & 
                         (carry[0] * params.sample_interval <= params.sim_time),
            fill_remaining,
            (s.sample_idx, s.times_buf, s.states_buf)
        )
        
        return s._replace(
            t=params.sim_time + 1.0,  # Force loop exit
            key=k3,
            sample_idx=final_idx,
            times_buf=final_times,
            states_buf=final_states,
        )
    
    # Use lax.cond to branch based on whether events are possible
    return lax.cond(
        a0 > RATE_EPSILON,  # Threshold for "effectively zero"
        process_event,
        skip_event,
        state
    )


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("num_servers", "max_samples", "policy_type", "d"))
def _simulate_jax_impl(
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
    """Hardware-accelerated simulation of the GibbsQ network."""
    params = SimParams(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        alpha=alpha,
        sim_time=sim_time,
        sample_interval=sample_interval,
        policy_type=policy_type,
        d=d
    )
    
    # Initialize buffers
    times_buf = jnp.zeros(max_samples)
    states_buf = jnp.zeros((max_samples, num_servers), dtype=jnp.int32)
    
    # Record initial state at t=0
    times_buf = times_buf.at[0].set(0.0)
    states_buf = states_buf.at[0].set(jnp.zeros(num_servers, dtype=jnp.int32))
    
    init_state = SimState(
        t=0.0,
        Q=jnp.zeros(num_servers, dtype=jnp.int32),
        key=key,
        sample_idx=1,  # Start at 1 since index 0 is already filled
        times_buf=times_buf,
        states_buf=states_buf,
        arrival_count=0,
        departure_count=0
    )
    
    final_state = lax.while_loop(
        lambda s: cond_fun(s, params),
        lambda s: body_fun(s, params),
        init_state
    )
    
    return final_state.times_buf, final_state.states_buf, (final_state.arrival_count, final_state.departure_count)


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
    return _simulate_jax_impl(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        alpha=alpha,
        sim_time=sim_time,
        sample_interval=sample_interval,
        key=key,
        max_samples=max_samples,
        policy_type=policy_type,
        d=d,
    )


@partial(jax.jit, static_argnames=("num_replications", "num_servers", "max_samples", "policy_type", "d"))
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
    policy_type:      int = 3,
    d:                int = 2
):
    """
    Run N replications in PARALLEL using jax.vmap.
    """
    keys = jax.random.split(jax.random.PRNGKey(base_seed), num_replications)
    
    # Vectorize across the keys
    v_sim = jax.vmap(
        lambda k: _simulate_jax_impl(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            alpha=alpha,
            sim_time=sim_time,
            sample_interval=sample_interval,
            key=k,
            max_samples=max_samples,
            policy_type=policy_type,
            d=d
        )
    )
    
    return v_sim(keys)


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
    return _run_replications_jax_impl(
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
        d=d,
    )
