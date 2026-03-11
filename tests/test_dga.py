import jax
import jax.numpy as jnp
import pytest
from gibbsq.engines.differentiable_engine import expected_queue_loss

def test_dga_gradient():
    """Prove that JAX can compute gradients through the Gillespie simulator."""
    key = jax.random.PRNGKey(42)
    
    # Parameters matching expected_queue_loss signature
    num_servers = 2
    arrival_rate = 1.0
    service_rates = jnp.array([1.0, 1.0])
    sim_steps = 100  # Unroll 100 jumps
    temperature = 0.1
    
    alpha = jnp.float32(5.0)
    
    # Evaluate forward pass
    loss = expected_queue_loss(
        alpha=alpha,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        key=key,
        num_servers=num_servers,
        sim_steps=sim_steps,
        temperature=temperature,
    )
    assert not jnp.isnan(loss)
    assert loss >= 0.0
    
    # Compute derivative d(Loss) / d(alpha)
    # This mathematically proves DGA works in our library
    # Note: jax.grad requires the differentiated argument to be positional
    grad_fn = jax.grad(expected_queue_loss)
    alpha_grad = grad_fn(
        alpha,  # positional argument for gradient
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        key=key,
        num_servers=num_servers,
        sim_steps=sim_steps,
        temperature=temperature,
    )
    
    # The gradient must be a valid float (not nan/inf) and non-zero (since alpha affects routing)
    assert not jnp.isnan(alpha_grad)
    assert not jnp.isinf(alpha_grad)
    assert jnp.abs(alpha_grad) > 1e-6
