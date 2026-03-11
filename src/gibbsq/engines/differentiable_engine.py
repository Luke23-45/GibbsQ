"""
Differentiable Gillespie Algorithm (DGA) for Soft-JSQ.

Uses continuous-state relaxations and Gumbel-Softmax for jump selection,
allowing gradients (via jax.grad) to flow from the final queueing metrics
back to the routing parameters (alpha).
"""
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import NamedTuple, Any, Callable

def default_policy(params: jnp.float32, Q: jnp.float32) -> jnp.float32:
    return -params * Q

class DGASimState(NamedTuple):
    t: jnp.float32
    Q: jnp.float32  # Continuous queue lengths
    key: jax.random.PRNGKey
    expected_Q_tot: jnp.float32  # Accumulated metric for loss tracking

@partial(jax.jit, static_argnames=("num_servers", "sim_steps", "apply_fn"))
def simulate_dga_jax(
    num_servers: int,
    arrival_rate: float,
    service_rates: jnp.ndarray,
    params: Any,
    sim_steps: int,
    key: jax.random.PRNGKey,
    temperature: float = 0.5,
    apply_fn: Callable = default_policy,
) -> jnp.float32:
    """
    Runs a Differentiable Gillespie simulation for `sim_steps` jumps.
    
    Returns
    -------
    jnp.float32
        The time-averaged expected total queue length.
    """
    
    def body_fun(carry, _):
        state = carry
        k1, k2 = jax.random.split(state.key)
        
        # 1. Routing Probabilities (fully differentiable via apply_fn)
        logits = apply_fn(params, state.Q)
        max_logit = jnp.max(logits)
        exp_logits = jnp.exp(logits - lax.stop_gradient(max_logit))
        probs = exp_logits / jnp.sum(exp_logits)
        
        # 2. Relaxed Propensities
        arrival_rates = arrival_rate * probs
        # Sigmoid relaxation for (Q > 0): if Q is near 0, departure rate lowers smoothly
        departure_rates = service_rates * jax.nn.sigmoid(state.Q * 20.0)
        
        rates = jnp.concatenate([arrival_rates, departure_rates])
        a0 = jnp.sum(rates)
        
        # 3. Expected Holding Time (Deterministic for variance reduction)
        tau = 1.0 / jnp.maximum(a0, 1e-9)
        
        # 4. Gumbel-Softmax Event Selection
        # Instead of sampling a hard index, we get a smooth probability vector
        gumbels = -jnp.log(-jnp.log(jax.random.uniform(k1, shape=rates.shape) + 1e-9) + 1e-9)
        event_weights = jax.nn.softmax((jnp.log(rates + 1e-9) + gumbels) / temperature)
        
        # 5. Continuous State Update
        arr_updates = event_weights[:num_servers]
        dep_updates = -event_weights[num_servers:]
        
        new_Q = jax.nn.relu(state.Q + arr_updates + dep_updates)
        new_t = state.t + tau
        
        # Accumulate time-weighted sum of queues
        current_Q_tot = jnp.sum(new_Q)
        new_expected_Q = state.expected_Q_tot + (current_Q_tot * tau)
        
        next_state = DGASimState(
            t=new_t,
            Q=new_Q,
            key=k2,
            expected_Q_tot=new_expected_Q
        )
        return next_state, None

    # Match dtype of service_rates for AD consistency.
    dtype = service_rates.dtype if hasattr(service_rates, 'dtype') else jnp.float32

    init_state = DGASimState(
        t=jnp.zeros((), dtype=dtype),
        Q=jnp.zeros(num_servers, dtype=dtype),
        key=key,
        expected_Q_tot=jnp.zeros((), dtype=dtype)
    )
    
    # Use scan because while_loop doesn't natively support reverse-mode AD easily 
    # without explicitly defining iterations, and scan is preferred for RNN/DGA unrolling.
    final_state, _ = jax.lax.scan(body_fun, init_state, xs=None, length=sim_steps)
    
    # Return time-averaged E[Q_total]
    return final_state.expected_Q_tot / jnp.maximum(final_state.t, 1e-9)

def expected_queue_loss(
    params: Any,
    arrival_rate: float,
    service_rates: jnp.ndarray,
    key: jax.random.PRNGKey,
    num_servers: int,
    sim_steps: int,
    temperature: float = 0.5,
    apply_fn: Callable = default_policy,
) -> jnp.float32:
    """Wrapper to compute scalar loss for jax.grad"""
    return simulate_dga_jax(
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        params=params,
        sim_steps=sim_steps,
        key=key,
        temperature=temperature,
        apply_fn=apply_fn
    )
