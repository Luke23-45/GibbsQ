"""
JAX-Native Gillespie Stochastic Simulation Algorithm (SSA) Engine.

This module provides a JIT-compiled, heavily vectorized implementation of the 
N-GibbsQ forward simulation and causal return discounting. 

By migrating the SSA loop from Python `while` loops to `jax.lax.scan`, we keep 
all trajectory generation, feature calculation, and neural network forward passes 
entirely on the GPU/TPU SRAM. This achieves massive parallelization via `jax.vmap`.

Mathematical Equivalences Ensured:
1. Exact Jump-Time Integration: The calculation of dt and interval area precisely 
   matches the legacy Python `np.diff(jump_times)` logic.
2. Causal Returns-to-Go: Replicates the interval-based temporal credit assignment 
   via a reverse-scan segmented accumulator.
3. Random Number Generation: Fully deterministic and pure using `jax.random.split`.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, Dict

from gibbsq.core.neural_policies import NeuralRouter


class TrajectoryBatchResult(NamedTuple):
    """Batched results from a vectorized JAX SSA simulation."""
    states: jax.Array                # [Batch, MaxSteps, N] - Queue states before events
    actions: jax.Array               # [Batch, MaxSteps] - Chosen server index (or -1)
    log_probs: jax.Array             # [Batch, MaxSteps] - Log probability of chosen action
    returns: jax.Array               # [Batch, MaxSteps] - Discounted causal returns to go
    is_action_mask: jax.Array        # [Batch, MaxSteps] - True if step was an Arrival (decision)
    valid_mask: jax.Array            # [Batch, MaxSteps] - True if t < sim_time
    total_integrated_queue: jax.Array # [Batch] - True Reward for the whole trajectory
    arrival_count: jax.Array         # [Batch]
    departure_count: jax.Array       # [Batch]


@jax.jit
def compute_causal_returns_jax(
    q_integrals: jax.Array, 
    is_arrival: jax.Array, 
    valid_mask: jax.Array, 
    gamma: float = 0.99
) -> jax.Array:
    """
    Computes Discounted Causal Returns using a vectorized reverse-scan.
    
    This matches the Python logic exactly:
    1. It sums immediate Q-area costs between actions.
    2. It discounts backward ONLY at action boundaries.
    
    Parameters
    ----------
    q_integrals : jax.Array
        Array of shape (max_steps,) containing area under queue for each interval.
    is_arrival : jax.Array
        Boolean mask of shape (max_steps,) indicating if step was an action.
    valid_mask : jax.Array
        Boolean mask of shape (max_steps,) indicating if step happened before T.
    gamma : float
        Stationary discount factor.
        
    Returns
    -------
    jax.Array
        Array of returns aligned with action steps. Non-action steps are zeroed.
    """
    
    def backward_step(carry, inputs):
        R = carry
        c_k, is_act, _ = inputs
        
        # Immediate interval cost added to accumulator
        action_return = R + c_k
        
        # If this step is an action boundary, we discount the future accumulated 
        # returns for the *next* step backwards in time. Otherwise, just accumulate.
        next_R = jnp.where(is_act, action_return * gamma, action_return)
        
        return next_R, action_return

    # reverse=True iterates from the end of the trajectory to the beginning
    _, returns = jax.lax.scan(
        backward_step,
        0.0,
        (q_integrals, is_arrival, valid_mask),
        reverse=True
    )
    
    # Mask out values at non-action steps to preserve padding invariants
    action_returns = jnp.where(is_arrival & valid_mask, returns, 0.0)
    return action_returns


@eqx.filter_jit
def collect_trajectory_jax(
    policy_net: NeuralRouter,
    num_servers: int,
    arrival_rate: float,
    service_rates: jax.Array,
    sim_time: float,
    key: jax.Array,
    max_steps: int = 5000,
    gamma: float = 0.99
) -> Dict[str, jax.Array]:
    """
    Run one Gillespie SSA trajectory using JAX dynamic unrolling.
    
    Unlike Python loops, `lax.scan` strictly executes `max_steps`. Steps occurring 
    after `t >= sim_time` are computed but mathematically zeroed out via `valid_mask`.
    """
    
    def step_fn(carry, _):
        t, Q, rng_key, is_active = carry
        
        rng_key, key_tau, key_event = jax.random.split(rng_key, 3)
        
        # --- 1. Routing Logits via Policy Net ---
        # Feature formulation (Sojourn Time Proxy): s_i = (Q_i + 1) / mu_i
        # FIX: The scan iterates single trajectories, so s is strictly 1D shape (N,).
        # We do NOT use vmap here, as policy_net expects 1D input for a single step.
        s = (Q + 1.0) / service_rates
        logits = policy_net(s)
        
        # Log-sum-exp for numerical stability
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)
        
        # --- 2. Propensities ---
        arr_rates = arrival_rate * probs
        dep_rates = jnp.where(Q > 0, service_rates, 0.0)
        rates = jnp.concatenate([arr_rates, dep_rates])
        
        a0 = jnp.sum(rates)
        
        # --- 3. Stochastic Draw ---
        # Add epsilon to prevent division-by-zero NaN propagation on valid_event=False states
        safe_a0 = a0 + 1e-12 
        tau = jax.random.exponential(key_tau) / safe_a0
        
        event_probs = rates / safe_a0
        event = jax.random.choice(key_event, 2 * num_servers, p=event_probs)
        
        # --- 4. State Update with Horizon Masking ---
        t_new = t + tau
        
        # Step is valid only if system is active, within horizon, and not degenerate
        valid_event = is_active & (t_new < sim_time) & (a0 > 1e-8)
        
        is_arrival = event < num_servers
        action = jnp.where(is_arrival, event, -1)
        # FIXED: Only access log_probs for arrival events, not departure events
        chosen_log_prob = jnp.where(is_arrival, log_probs[event], 0.0)
        
        # FIX: Prevent array wrap-around negative indexing during tracing phase
        dep_idx = jnp.maximum(0, event - num_servers)
        
        Q_arr = Q.at[event].add(1)
        Q_dep = Q.at[dep_idx].add(-1)
        Q_updated = jnp.where(is_arrival, Q_arr, Q_dep)
        
        Q_next = jnp.where(valid_event, Q_updated, Q)
        t_next = jnp.where(valid_event, t_new, t)
        
        outputs = {
            "pre_jump_state": Q,
            "post_jump_state": Q_next,
            "jump_time": t_new,
            "action": action,
            "log_prob": chosen_log_prob,
            "is_arrival": is_arrival & valid_event,
            "valid_mask": valid_event
        }
        
        next_carry = (t_next, Q_next, rng_key, valid_event)
        return next_carry, outputs

    # Unroll the simulation loop statically
    init_Q = jnp.zeros(num_servers, dtype=jnp.int32)
    init_carry = (0.0, init_Q, key, True)
    
    _, outputs = jax.lax.scan(
        step_fn,
        init_carry,
        None,
        length=max_steps
    )
    
    # Extract historical vectors
    valid_mask = outputs["valid_mask"]
    jump_times = outputs["jump_time"]
    post_states = outputs["post_jump_state"]
    pre_states = outputs["pre_jump_state"]
    is_arrival = outputs["is_arrival"]
    actions = outputs["action"]
    log_probs = outputs["log_prob"]
    
    # --- 5. Interval Integration (Area under Q curve) ---
    # To match Python np.diff(jump_times, append=sim_time), we compute time deltas.
    # The interval for step i is [t_i, t_{i+1}], bounded by T at the horizon.
    next_jumps = jnp.pad(jump_times[1:], (0, 1), constant_values=sim_time)
    next_mask = jnp.pad(valid_mask[1:], (0, 1), constant_values=False)
    
    # The last valid jump links to sim_time, not the next generated jump
    is_last_valid = valid_mask & ~next_mask
    next_jumps = jnp.where(is_last_valid, sim_time, next_jumps)
    
    dt = jnp.where(valid_mask, next_jumps - jump_times, 0.0)
    
    q_totals = jnp.sum(post_states, axis=-1)
    q_integrals = q_totals * dt
    
    total_integrated_queue = jnp.sum(q_integrals)
    
    # --- 6. Apply Causal Return Tracking ---
    returns = compute_causal_returns_jax(q_integrals, is_arrival, valid_mask, gamma)
    
    return {
        "states": pre_states,
        "actions": actions,
        "log_probs": log_probs,
        "returns": returns,
        "is_action_mask": is_arrival,
        "valid_mask": valid_mask,
        "total_integrated_queue": total_integrated_queue,
        "arrival_count": jnp.sum(is_arrival),
        "departure_count": jnp.sum(valid_mask & ~is_arrival),
    }


@eqx.filter_jit
def vmap_collect_trajectories(
    policy_net: NeuralRouter,
    num_servers: int,
    arrival_rate: float,
    service_rates: jax.Array,
    sim_time: float,
    keys: jax.Array,
    max_steps: int = 5000,
    gamma: float = 0.99
) -> TrajectoryBatchResult:
    """
    Massively Parallel Execution of multiple CTMC trajectories.
    
    This function broadcasts `collect_trajectory_jax` over a batch of RNG keys.
    All processing is kept on-device, bypassing Python loop overheads entirely.
    """
    
    def single_trajectory(k):
        return collect_trajectory_jax(
            policy_net, 
            num_servers, 
            arrival_rate, 
            service_rates, 
            sim_time, 
            k, 
            max_steps, 
            gamma
        )
        
    # Vectorize ONLY over the keys parameter. 
    # JAX will automatically broadcast the unmapped variables inside the closure.
    batch_outputs = jax.vmap(single_trajectory)(keys)
    
    return TrajectoryBatchResult(
        states=batch_outputs["states"],
        actions=batch_outputs["actions"],
        log_probs=batch_outputs["log_probs"],
        returns=batch_outputs["returns"],
        is_action_mask=batch_outputs["is_action_mask"],
        valid_mask=batch_outputs["valid_mask"],
        total_integrated_queue=batch_outputs["total_integrated_queue"],
        arrival_count=batch_outputs["arrival_count"],
        departure_count=batch_outputs["departure_count"]
    )