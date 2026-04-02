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
import numpy as np
import math
import equinox as eqx
from typing import NamedTuple, Dict

from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.reinforce_objective import compute_action_interval_returns_jax

def compute_poisson_max_steps(arrival_rate: float, service_rates: np.ndarray, sim_time: float, sigma: float = 6.0) -> int:
    """Pure Python helper to safely calculate JAX unroll bounds without XLA tracer leaks."""
    max_rate = float(arrival_rate) + float(np.sum(service_rates))
    expected_events = max_rate * float(sim_time)
    # 6-sigma captures 99.9999998% of the Poisson tail. +100 is an absolute safety floor.
    return int(math.ceil(expected_events + sigma * math.sqrt(expected_events) + 100))

class TrajectoryBatchResult(NamedTuple):
    """Batched results from a vectorized JAX SSA simulation."""
    states: jax.Array                # [Batch, MaxSteps, N] - Queue states before events
    post_states: jax.Array           # [Batch, MaxSteps, N] - Queue states after events
    jump_times: jax.Array            # [Batch, MaxSteps] - Event times after each step
    actions: jax.Array               # [Batch, MaxSteps] - Chosen server index (or -1)
    log_probs: jax.Array             # [Batch, MaxSteps] - Log probability of chosen action
    returns: jax.Array               # [Batch, MaxSteps] - Discounted causal returns to go
    is_action_mask: jax.Array        # [Batch, MaxSteps] - True if step was an Arrival (decision)
    valid_mask: jax.Array            # [Batch, MaxSteps] - True if t < sim_time
    total_integrated_queue: jax.Array # [Batch] - True Reward for the whole trajectory
    arrival_count: jax.Array         # [Batch]
    departure_count: jax.Array       # [Batch]
    is_truncated: jax.Array          # [Batch] - True if simulation was cut off by max_steps

@jax.jit
def compute_causal_returns_jax(
    q_integrals: jax.Array,
    dt: jax.Array,
    is_arrival: jax.Array,
    valid_mask: jax.Array,
    gamma: float = 0.99
) -> jax.Array:
    """
    Computes the canonical continuous-time discounted returns over action intervals.
    
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
    
    returns = compute_action_interval_returns_jax(
        q_integrals=q_integrals,
        dt=dt,
        is_action=is_arrival,
        valid_mask=valid_mask,
        gamma=gamma,
    )
    return jnp.where(is_arrival & valid_mask, returns, 0.0)

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
        
        rho_val = arrival_rate / jnp.sum(service_rates)
        logits = policy_net(Q, mu=service_rates, rho=rho_val)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)

        arr_rates = arrival_rate * probs
        dep_rates = jnp.where(Q > 0, service_rates, 0.0)
        rates = jnp.concatenate([arr_rates, dep_rates])
        
        a0 = jnp.sum(rates)
        safe_a0 = a0 + 1e-12
        tau = jax.random.exponential(key_tau) / safe_a0
        
        event_probs = rates / safe_a0
        event = jax.random.choice(key_event, 2 * num_servers, p=event_probs)
        
        t_new = t + tau
        valid_event = is_active & (t_new < sim_time) & (a0 > 1e-8)
        
        is_arrival = event < num_servers
        action = jnp.where(is_arrival, event, -1)
        chosen_log_prob = jnp.where(is_arrival, log_probs[event], 0.0)
        
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

    init_Q = jnp.zeros(num_servers, dtype=jnp.int32)
    init_carry = (0.0, init_Q, key, True)
    
    _, outputs = jax.lax.scan(
        step_fn,
        init_carry,
        None,
        length=max_steps
    )
    
    valid_mask = outputs["valid_mask"]
    jump_times = outputs["jump_time"]
    post_states = outputs["post_jump_state"]
    pre_states = outputs["pre_jump_state"]
    is_arrival = outputs["is_arrival"]
    actions = outputs["action"]
    log_probs = outputs["log_prob"]
    
    next_jumps = jnp.pad(jump_times[1:], (0, 1), constant_values=sim_time)
    next_mask = jnp.pad(valid_mask[1:], (0, 1), constant_values=False)
    is_last_valid = valid_mask & ~next_mask
    next_jumps = jnp.where(is_last_valid, sim_time, next_jumps)
    
    dt = jnp.where(valid_mask, next_jumps - jump_times, 0.0)
    
    q_totals = jnp.sum(post_states, axis=-1)
    q_integrals = q_totals * dt
    
    total_integrated_queue = jnp.sum(q_integrals)
    is_truncated = valid_mask[-1]
    
    returns = compute_causal_returns_jax(q_integrals, dt, is_arrival, valid_mask, gamma)
    
    return {
        "states": pre_states,
        "post_states": post_states,
        "jump_times": jump_times,
        "actions": actions,
        "log_probs": log_probs,
        "returns": returns,
        "is_action_mask": is_arrival,
        "valid_mask": valid_mask,
        "total_integrated_queue": total_integrated_queue,
        "arrival_count": jnp.sum(is_arrival),
        "departure_count": jnp.sum(valid_mask & ~is_arrival),
        "is_truncated": is_truncated,
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
        
    batch_outputs = jax.vmap(single_trajectory)(keys)
    
    return TrajectoryBatchResult(
        states=batch_outputs["states"],
        post_states=batch_outputs["post_states"],
        jump_times=batch_outputs["jump_times"],
        actions=batch_outputs["actions"],
        log_probs=batch_outputs["log_probs"],
        returns=batch_outputs["returns"],
        is_action_mask=batch_outputs["is_action_mask"],
        valid_mask=batch_outputs["valid_mask"],
        total_integrated_queue=batch_outputs["total_integrated_queue"],
        arrival_count=batch_outputs["arrival_count"],
        departure_count=batch_outputs["departure_count"],
        is_truncated=batch_outputs["is_truncated"]
    )
