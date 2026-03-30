"""
Canonical REINFORCE objective utilities.

This module centralizes the action-interval return definition used by
training, the JAX SSA collector, and the gradient-check path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

def compute_action_interval_returns_from_trajectory_numpy(
    states: list[np.ndarray],
    jump_times: list[float],
    action_step_indices: list[int],
    sim_time: float,
    gamma: float = 0.99,
) -> np.ndarray:
    """Return discounted action-level returns for a Python SSA trajectory."""
    if not states or not action_step_indices:
        return np.array([], dtype=np.float64)

    q_totals = np.asarray([np.sum(s) for s in states], dtype=np.float64)
    dt = np.diff(np.asarray(jump_times, dtype=np.float64), append=sim_time)
    q_integrals = q_totals * dt
    is_action = np.zeros(len(q_integrals), dtype=bool)
    is_action[np.asarray(action_step_indices, dtype=int)] = True

    step_returns = compute_action_interval_returns_numpy(
        q_integrals=q_integrals,
        dt=dt,
        is_action=is_action,
        valid_mask=np.ones_like(is_action, dtype=bool),
        gamma=gamma,
    )
    return extract_action_returns_numpy(step_returns, is_action)

def compute_action_interval_returns_numpy(
    q_integrals: np.ndarray,
    dt: np.ndarray,
    is_action: np.ndarray,
    valid_mask: np.ndarray | None = None,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute step-aligned discounted returns using continuous-time action intervals."""
    q_integrals = np.asarray(q_integrals, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    is_action = np.asarray(is_action, dtype=bool)
    if valid_mask is None:
        valid_mask = np.ones_like(is_action, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    if not (len(q_integrals) == len(dt) == len(is_action) == len(valid_mask)):
        raise ValueError("q_integrals, dt, is_action, and valid_mask must have the same length.")

    q_integrals = np.where(valid_mask, q_integrals, 0.0)
    dt = np.where(valid_mask, dt, 0.0)
    is_action = is_action & valid_mask

    returns = np.zeros_like(q_integrals, dtype=np.float64)
    future_return = 0.0
    interval_cost = 0.0
    interval_duration = 0.0

    for idx in range(len(q_integrals) - 1, -1, -1):
        interval_cost += q_integrals[idx]
        interval_duration += dt[idx]
        if is_action[idx]:
            action_return = interval_cost + (gamma ** interval_duration) * future_return
            returns[idx] = action_return
            future_return = action_return
            interval_cost = 0.0
            interval_duration = 0.0

    return returns

@jax.jit
def compute_action_interval_returns_jax(
    q_integrals: jax.Array,
    dt: jax.Array,
    is_action: jax.Array,
    valid_mask: jax.Array,
    gamma: float = 0.99,
) -> jax.Array:
    """JAX equivalent of the canonical continuous-time action-interval return."""
    q_integrals = jnp.where(valid_mask, q_integrals, 0.0)
    dt = jnp.where(valid_mask, dt, 0.0)
    is_action = is_action & valid_mask

    def backward_step(carry, inputs):
        future_return, interval_cost, interval_duration = carry
        cost_t, dt_t, is_act = inputs

        interval_cost = interval_cost + cost_t
        interval_duration = interval_duration + dt_t
        action_return = interval_cost + (gamma ** interval_duration) * future_return
        emitted = jnp.where(is_act, action_return, 0.0)

        next_future = jnp.where(is_act, action_return, future_return)
        next_cost = jnp.where(is_act, 0.0, interval_cost)
        next_duration = jnp.where(is_act, 0.0, interval_duration)
        return (next_future, next_cost, next_duration), emitted

    _, step_returns = jax.lax.scan(
        backward_step,
        (0.0, 0.0, 0.0),
        (q_integrals, dt, is_action),
        reverse=True,
    )
    return step_returns

def extract_action_returns_numpy(
    step_returns: np.ndarray,
    is_action: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Extract action-only returns from a step-aligned NumPy return tensor."""
    is_action = np.asarray(is_action, dtype=bool)
    if valid_mask is not None:
        is_action = is_action & np.asarray(valid_mask, dtype=bool)
    return np.asarray(step_returns, dtype=np.float64)[is_action]

@jax.jit
def extract_first_action_returns_jax(
    step_returns: jax.Array,
    is_action: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """Extract the first action return for each batched trajectory."""
    mask = is_action & valid_mask
    first_action_idx = jnp.argmax(mask, axis=-1)
    has_action = jnp.any(mask, axis=-1)
    gathered = jnp.take_along_axis(step_returns, first_action_idx[..., None], axis=-1).squeeze(-1)
    return jnp.where(has_action, gathered, 0.0)
