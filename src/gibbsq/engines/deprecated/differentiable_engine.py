"""
Differentiable Gillespie Algorithm (DGA) for Soft-JSQ.

The DGA is a differentiable surrogate of the true CTMC (Gillespie SSA),
not an exact simulation. Key differences from the true process:

1. EVENT SIMULTANEITY: Each DGA step applies fractional arrival AND departure
   updates simultaneously via Gumbel-Softmax weights. A true CTMC has exactly
   one event per step.

2. CONTINUOUS STATE: Queue lengths Q are real-valued (not integer). The
   soft indicator replaces the hard 1(Q>0) predicate.

3. TEMPERATURE BIAS: At temperature=0.5 (default), Gumbel-Softmax weights
   are not sharply peaked. The surrogate bias decreases as temperature → 0,
   but gradient magnitude also decreases.

CONSEQUENCE FOR THE PAPER: DGA-measured E[Q] and SSA-measured E[Q] are
correlated but not identical. Results labeled "E[Q]" from this engine must
be described as "DGA surrogate E[Q]" in paper text. For true steady-state
E[Q] comparisons use the SSA engine (jax_engine.py or numpy_engine.py).
The SSA evaluation of the trained NeuralRouter is in policy_comparison.py.
"""
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import NamedTuple, Any, Callable
from gibbsq.core import constants

def default_policy(params: jnp.float32, Q: jnp.float32) -> jnp.float32:
    return -params * Q

class DGASimState(NamedTuple):
    t: jnp.float32
    Q: jnp.float32
    key: jax.random.PRNGKey
    expected_Q_tot: jnp.float32

def _dga_step(
    state: DGASimState,
    num_servers: int,
    arrival_rate: float,
    service_rates: jnp.ndarray,
    params: Any,
    temperature: float,
    apply_fn: Callable,
) -> DGASimState:
    """Single differentiable Gillespie step shared by scan/fori implementations."""
    k1, k2 = jax.random.split(state.key)

    logits = apply_fn(params, state.Q)
    max_logit = jnp.max(logits)
    exp_logits = jnp.exp(logits - lax.stop_gradient(max_logit))
    probs = exp_logits / jnp.sum(exp_logits)

    arrival_rates = arrival_rate * probs
    _soft_indicator = 1.0 - jnp.exp(-state.Q * constants.DGA_INDICATOR_STEEPNESS)
    departure_rates = service_rates * _soft_indicator

    rates = jnp.concatenate([arrival_rates, departure_rates])
    a0 = jnp.sum(rates)

    tau = 1.0 / jnp.maximum(a0, constants.NUMERICAL_STABILITY_EPSILON)

    gumbels = -jnp.log(-jnp.log(jax.random.uniform(k1, shape=rates.shape) + constants.GUMBEL_SMOOTHING) + constants.GUMBEL_SMOOTHING)
    event_weights = jax.nn.softmax((jnp.log(rates + constants.GUMBEL_SMOOTHING) + gumbels) / temperature)

    arr_updates = event_weights[:num_servers]
    dep_updates = -event_weights[num_servers:]

    new_Q = jax.nn.relu(state.Q + arr_updates + dep_updates)
    new_t = state.t + tau

    current_Q_tot = jnp.sum(state.Q)
    new_expected_Q = state.expected_Q_tot + (current_Q_tot * tau)
    return DGASimState(t=new_t, Q=new_Q, key=k2, expected_Q_tot=new_expected_Q)

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
    Runs a Differentiable Gillespie **surrogate** simulation for `sim_steps` steps.

    This is a biased but differentiable approximation of the true CTMC.
    See module docstring for the full disclosure of surrogate bias.
    Use the SSA engines (numpy_engine.py, jax_engine.py) for unbiased E[Q].

    Returns
    -------
    jnp.float32
        Time-averaged surrogate expected total queue length E_DGA[Q].
        This is NOT the same as the true CTMC steady-state E_SSA[Q].
    """
    
    def body_fun(carry, _):
        next_state = _dga_step(
            state=carry,
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            params=params,
            temperature=temperature,
            apply_fn=apply_fn,
        )
        return next_state, None

    dtype = service_rates.dtype if hasattr(service_rates, 'dtype') else jnp.float32

    init_state = DGASimState(
        t=jnp.zeros((), dtype=dtype),
        Q=jnp.zeros(num_servers, dtype=dtype),
        key=key,
        expected_Q_tot=jnp.zeros((), dtype=dtype)
    )
    
    final_state, _ = jax.lax.scan(body_fun, init_state, xs=None, length=sim_steps)
    
    return final_state.expected_Q_tot / jnp.maximum(final_state.t, constants.NUMERICAL_STABILITY_EPSILON)

def simulate_dga_jax_dynamic_steps(
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
    Differentiable Gillespie simulation with dynamic `sim_steps` support.

    This variant uses ``lax.fori_loop`` to allow dynamic step counts in a single
    compiled graph when using forward-mode differentiation (e.g., ``jax.jvp``).
    """
    dtype = service_rates.dtype if hasattr(service_rates, 'dtype') else jnp.float32

    init_state = DGASimState(
        t=jnp.zeros((), dtype=dtype),
        Q=jnp.zeros(num_servers, dtype=dtype),
        key=key,
        expected_Q_tot=jnp.zeros((), dtype=dtype)
    )

    def body_fun(_, carry):
        return _dga_step(
            state=carry,
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            params=params,
            temperature=temperature,
            apply_fn=apply_fn,
        )

    final_state = jax.lax.fori_loop(0, sim_steps, body_fun, init_state)
    return final_state.expected_Q_tot / jnp.maximum(final_state.t, constants.NUMERICAL_STABILITY_EPSILON)

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
