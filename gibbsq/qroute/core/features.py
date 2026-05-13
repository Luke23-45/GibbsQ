"""
State representation functions for queueing systems.

This module provides feature transformations for routing policies in
systems with non-identical service rates.

The primary export is `look_ahead_potential`.
"""

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float

def look_ahead_potential(
    Q: Float[Array, "..."],
    mu: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Compute expected look-ahead potential representation for routing decisions.
    
    For a heterogeneous M/M/N queue, the correct state representation
    is the expected look-ahead potential a newly arriving job would experience
    at each server:
    
        s_i = (Q_i + 1) / μ_i
    
    This represents the expected time spent waiting (Q_i jobs ahead)
    plus service time (1/μ_i), under FCFS discipline.
    
    **Why this representation is used:**
    
    1. **Scale-invariant**: A server with μ=10 and Q=9 has the same
       look-ahead potential as a server with μ=1 and Q=0. Raw queue lengths
       cannot distinguish these states; potential correctly equates them.
    
    2. **Heterogeneity-aware**: Natively encodes capacity mismatch
       between servers, eliminating the "Heterogeneity Trap" where
       JSQ routes equally to fast and slow servers.
    
    3. **Heavy-traffic motivation**: The routing policy that minimizes expected
       M/M/N heterogeneous queues in heavy traffic minimizes expected
       sojourn time (Halfin-Whitt, 1981; Atar, Mandelbaum & Reiman, 2004).
    
    4. **Little's Law consistency**: By Little's Law, E[W_i] = E[Q_i]/λ_i.
       The potential representation uses the quantity that Little's
       Law connects to system performance.
    
    Parameters
    ----------
    Q : Float[Array, "..."]
        Queue length vector. Shape (N,) for single state, or (batch, N)
        for batched states. Can be numpy or JAX array.
    mu : Float[Array, "..."]
        Service rate vector. Shape (N,) or broadcastable to Q.
        Must be > 0 for all servers.
    
    Returns
    -------
    Float[Array, "..."]
        Look-ahead potential features s_i = (Q_i + 1) / μ_i.
        Same shape as input Q.
    
    Examples
    --------
    >>> import numpy as np
    >>> Q = np.array([5, 3, 0])  # Queue lengths
    >>> mu = np.array([1.0, 2.0, 3.0])  # Service rates
    >>> look_ahead_potential(Q, mu)
    array([6.        , 2.        , 0.33333333])
    
    The first server has Q=5 and μ=1, so expected potential is (5+1)/1 = 6.
    The third server has Q=0 and μ=3, so expected potential is (0+1)/3 ≈ 0.33.
    A job should route to server 3 (lowest potential), not server 2
    (lowest queue length), demonstrating why raw Q is wrong for heterogeneous systems.
    
    Notes
    -----
    This function is designed to work with both NumPy and JAX arrays.
    It uses jax.numpy operations which are compatible with both.
    
    References
    ----------
    .. [1] Halfin, S., & Whitt, W. (1981). Heavy-traffic limits for
           queues with many exponential servers. Operations Research.
    .. [2] Atar, R., Mandelbaum, A., & Reiman, M. I. (2004).
           Scheduling a multi-class queue with many exponential servers.
           Annals of Applied Probability.
    """
    Q_arr = jnp.asarray(Q)
    mu_arr = jnp.asarray(mu)
    return (Q_arr + 1.0) / mu_arr

def look_ahead_potential_numpy(
    Q: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """
    NumPy-only version of look_ahead_potential.
    
    Provided for compatibility with pure NumPy code paths.
    See `look_ahead_potential` for full documentation.
    
    Parameters
    ----------
    Q : np.ndarray
        Queue length vector, shape (N,) or (batch, N).
    mu : np.ndarray
        Service rate vector, shape (N,) or broadcastable.
    
    Returns
    -------
    np.ndarray
        Look-ahead potential features, same shape as Q.
    """
    return (Q.astype(np.float64) + 1.0) / mu.astype(np.float64)

def softmax_on_potential(
    Q: Float[Array, "..."],
    mu: Float[Array, "..."],
    alpha: float,
) -> Float[Array, "..."]:
    """
    Compute GibbsQ routing probabilities using look-ahead potential representation.
    
    This is the corrected GibbsQ policy for heterogeneous servers:
    
        p_i(Q) ∝ exp(-α · s_i) = exp(-α · (Q_i + 1) / μ_i)
    
    This replaces the incorrect raw-queue formulation:
    
        p_i(Q) ∝ exp(-α · Q_i)  # WRONG for heterogeneous systems
    
    Parameters
    ----------
    Q : Float[Array, "..."]
        Queue length vector, shape (N,).
    mu : Float[Array, "..."]
        Service rate vector, shape (N,).
    alpha : float
        Inverse temperature parameter. Higher α means more aggressive
        routing to the shortest-potential server.
    
    Returns
    -------
    Float[Array, "N"]
        Routing probabilities that sum to 1.0.
    
    Examples
    --------
    >>> Q = np.array([5, 3, 0])
    >>> mu = np.array([1.0, 2.0, 3.0])
    >>> softmax_on_potential(Q, mu, alpha=1.0)
    array([0.0024...])  # Probability heavily weighted to server 3
    """
    s = look_ahead_potential(Q, mu)
    logits = -alpha * s
    logits = logits - jnp.max(logits)
    weights = jnp.exp(logits)
    
    return weights / jnp.sum(weights)

def softmax_on_potential_numpy(
    Q: np.ndarray,
    mu: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    NumPy-only version of softmax_on_potential.
    
    See `softmax_on_potential` for full documentation.
    """
    s = look_ahead_potential_numpy(Q, mu)
    logits = -alpha * s
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    
    return weights / np.sum(weights)

def softmax_on_potential_uas(
    Q: Float[Array, "..."],
    mu: Float[Array, "..."],
    alpha: float,
) -> Float[Array, "..."]:
    """
    Compute UAS (Unified Archimedean Softmax) routing probabilities.
    
    This is the capacity-weighted GibbsQ policy for heterogeneous servers:
    
        p_i(Q) ∝ μ_i · exp(-α · s_i) = μ_i · exp(-α · (Q_i + 1) / μ_i)
    
    The μ_i weighting provides capacity-aware routing: faster servers
    receive proportionally more traffic even when look-ahead potentials are equal.
    
    Parameters
    ----------
    Q : Float[Array, "..."]
        Queue length vector, shape (N,).
    mu : Float[Array, "..."]
        Service rate vector, shape (N,).
    alpha : float
        Inverse temperature parameter. Higher α means more aggressive
        routing to the shortest-potential server.
    
    Returns
    -------
    Float[Array, "N"]
        Routing probabilities that sum to 1.0.
    
    Examples
    --------
    >>> Q = np.array([5, 3, 0])
    >>> mu = np.array([1.0, 2.0, 3.0])
    >>> softmax_on_potential_uas(Q, mu, alpha=1.0)
    array([...])  # Probability weighted by capacity
    """
    s = look_ahead_potential(Q, mu)
    logits = -alpha * s + jnp.log(mu)
    logits = logits - jnp.max(logits)
    weights = jnp.exp(logits)
    
    return weights / jnp.sum(weights)

def softmax_on_potential_uas_numpy(
    Q: np.ndarray,
    mu: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    NumPy-only version of softmax_on_potential_uas.
    
    See `softmax_on_potential_uas` for full documentation.
    """
    s = look_ahead_potential_numpy(Q, mu)
    logits = -alpha * s + np.log(mu)
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    
    return weights / np.sum(weights)

def compute_advantage(
    Q: Float[Array, "N"],
    server_idx: int,
    mu: Float[Array, "N"],
    value_estimate: float,
) -> float:
    """
    Compute the advantage for a routing decision in REINFORCE training.
    
    For continuous-time queueing systems, the advantage for routing
    an arrival to server i in state Q is:
    
        A(Q, i) = -s_i + V(Q)
    
    where s_i = (Q_i + 1) / μ_i is the expected look-ahead potential at server i,
    and V(Q) is the value function estimate of expected future queue length.
    
    The negative sign on s_i reflects that lower potential is better
    (we want to minimize expected queue length).
    
    Parameters
    ----------
    Q : Float[Array, "N"]
        Current queue length vector.
    server_idx : int
        Index of the server the job was routed to.
    mu : Float[Array, "N"]
        Service rate vector.
    value_estimate : float
        Value function estimate V(Q) from the critic network.
    
    Returns
    -------
    float
        Advantage estimate for this routing decision.
    """
    s = look_ahead_potential(Q, mu)
    immediate_cost = -s[server_idx]
    return immediate_cost + value_estimate

__all__ = [
    "look_ahead_potential",
    "look_ahead_potential_numpy",
    "softmax_on_potential",
    "softmax_on_potential_numpy",
    "softmax_on_potential_uas",
    "softmax_on_potential_uas_numpy",
    "compute_advantage",
]
