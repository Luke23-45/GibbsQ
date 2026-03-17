import jax
import jax.numpy as jnp

from gibbsq.engines.deprecated.differentiable_engine import (
    expected_queue_loss,
    simulate_dga_jax_dynamic_steps,
)


def _forward_loss_and_grad(alpha, arrival_rate, service_rates, key, num_servers, sim_steps, temperature):
    def loss_fn(alpha_param):
        return simulate_dga_jax_dynamic_steps(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            params=alpha_param,
            sim_steps=sim_steps,
            key=key,
            temperature=temperature,
        )

    tangent = jnp.array(1.0, dtype=alpha.dtype)
    return jax.jvp(loss_fn, (alpha,), (tangent,))


def test_forward_mode_matches_reverse_mode_for_fixed_horizon():
    num_servers = 3
    arrival_rate = 1.8
    service_rates = jnp.array([1.0, 1.2, 1.4], dtype=jnp.float32)
    alpha = jnp.array(0.5, dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    sim_steps = 200
    temperature = 0.5

    reverse_loss, reverse_grad = jax.value_and_grad(expected_queue_loss, argnums=0)(
        alpha,
        arrival_rate,
        service_rates,
        key,
        num_servers,
        sim_steps,
        temperature,
    )

    forward_loss, forward_grad = _forward_loss_and_grad(
        alpha,
        arrival_rate,
        service_rates,
        key,
        num_servers,
        sim_steps,
        temperature,
    )

    assert jnp.allclose(forward_loss, reverse_loss, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(forward_grad, reverse_grad, rtol=1e-5, atol=1e-6)


def test_jitted_forward_mode_supports_multiple_horizons():
    num_servers = 3
    arrival_rate = 1.8
    service_rates = jnp.array([1.0, 1.2, 1.4], dtype=jnp.float32)
    alpha = jnp.array(0.5, dtype=jnp.float32)
    temperature = 0.5

    jitted_fn = jax.jit(_forward_loss_and_grad, static_argnums=(4,))

    key = jax.random.PRNGKey(1)
    for sim_steps in (50, 125, 300):
        key, subkey = jax.random.split(key)
        loss_val, grad_val = jitted_fn(
            alpha,
            arrival_rate,
            service_rates,
            subkey,
            num_servers,
            sim_steps,
            temperature,
        )
        assert jnp.isfinite(loss_val)
        assert jnp.isfinite(grad_val)
