import jax
import jax.numpy as jnp
import pytest

from gibbsq.core.config import NeuralConfig
from gibbsq.core.neural_policies import NeuralRouter, ValueNetwork


@pytest.fixture
def config():
    return NeuralConfig(
        hidden_size=32,
        use_rho=True,
        use_service_rates=True,
        rho_input_scale=10.0,
        preprocessing="log1p",
    )


@pytest.fixture
def service_rates():
    return jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)


@pytest.fixture
def policy_net(config, service_rates):
    return NeuralRouter(
        num_servers=3,
        config=config,
        service_rates=service_rates,
        key=jax.random.PRNGKey(42),
    )


@pytest.fixture
def value_net(config):
    return ValueNetwork(
        num_servers=3,
        config=config,
        key=jax.random.PRNGKey(7),
    )


def test_policy_single_input_accepts_scalar_rho(policy_net):
    logits = policy_net(jnp.array([1.0, 2.0, 3.0]), rho=0.8)
    assert logits.shape == (3,)
    assert jnp.all(jnp.isfinite(logits))


def test_value_single_input_accepts_service_rates_and_rho(value_net, service_rates):
    value = value_net(jnp.array([1.0, 2.0, 3.0]), service_rates, 0.8)
    assert value.shape == ()
    assert jnp.isfinite(value)


def test_policy_vmap_matches_training_pattern(policy_net):
    states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    rhos = jnp.array([0.6, 0.8], dtype=jnp.float32)

    logits = jax.vmap(lambda s, r: policy_net(s, rho=r))(states, rhos)

    assert logits.shape == (2, 3)
    assert jnp.all(jnp.isfinite(logits))


def test_value_vmap_matches_training_callsite(value_net, service_rates):
    states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    rhos = jnp.array([0.6, 0.8], dtype=jnp.float32)

    values = jax.vmap(value_net, in_axes=(0, None, 0))(states, service_rates, rhos)

    assert values.shape == (2,)
    assert jnp.all(jnp.isfinite(values))


def test_policy_direct_batched_call_matches_keyword_vmap(policy_net):
    states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    rhos = jnp.array([0.4, 0.8], dtype=jnp.float32)

    direct = policy_net(states, rho=rhos)
    via_vmap = jax.vmap(lambda s, r: policy_net(s, rho=r))(states, rhos)

    assert direct.shape == (2, 3)
    assert jnp.allclose(direct, via_vmap, atol=1e-5)
