import numpy as np
import pytest


def test_action_interval_returns_match_handcrafted_trace():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from gibbsq.core.reinforce_objective import (
        compute_action_interval_returns_jax,
        compute_action_interval_returns_numpy,
        extract_action_returns_numpy,
        extract_first_action_returns_jax,
    )

    q_integrals = np.array([3.0, 4.0, 1.0, 2.1, 2.8], dtype=np.float64)
    dt = np.array([0.5, 1.0, 0.2, 0.7, 0.4], dtype=np.float64)
    is_action = np.array([True, False, True, False, True], dtype=bool)
    valid_mask = np.ones_like(is_action, dtype=bool)

    returns_np = compute_action_interval_returns_numpy(q_integrals, dt, is_action, valid_mask, gamma=0.99)
    returns_jax = np.array(
        compute_action_interval_returns_jax(
            jnp.asarray(q_integrals),
            jnp.asarray(dt),
            jnp.asarray(is_action),
            jnp.asarray(valid_mask),
            gamma=0.99,
        )
    )

    np.testing.assert_allclose(returns_jax, returns_np, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        returns_np[is_action],
        extract_action_returns_numpy(returns_np, is_action),
        rtol=1e-9,
        atol=1e-9,
    )
    first_jax = np.array(
        extract_first_action_returns_jax(
            jnp.asarray(returns_jax)[None, :],
            jnp.asarray(is_action)[None, :],
            jnp.asarray(valid_mask)[None, :],
        )
    )[0]
    assert np.isclose(first_jax, returns_np[is_action][0])


def test_extract_first_action_returns_jax_is_batchwise_and_skips_missing_actions():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from gibbsq.core.reinforce_objective import extract_first_action_returns_jax

    step_returns = jnp.asarray(
        [
            [10.0, 0.0, 4.0, 1.0],
            [0.0, 7.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    is_action = jnp.asarray(
        [
            [True, False, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    valid_mask = jnp.asarray(
        [
            [True, True, False, False],
            [True, True, True, True],
            [True, True, True, True],
        ],
        dtype=bool,
    )

    first_returns = np.array(
        extract_first_action_returns_jax(step_returns, is_action, valid_mask)
    )

    np.testing.assert_allclose(first_returns, np.array([10.0, 7.0, 0.0]))


def test_policy_distribution_matches_across_paths():
    pytest.importorskip("jax")
    pytest.importorskip("equinox")
    import jax
    import jax.numpy as jnp

    from gibbsq.core.config import NeuralConfig
    from gibbsq.core.neural_policies import NeuralRouter
    from gibbsq.core.policy_distribution import compute_numpy_policy_probs
    from gibbsq.utils.model_io import StochasticNeuralPolicy

    service_rates = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rho = 0.7
    Q = np.array([2.0, 0.0, 1.0], dtype=np.float64)

    net = NeuralRouter(
        num_servers=3,
        config=NeuralConfig(hidden_size=16, preprocessing="log1p", init_type="zero_final"),
        service_rates=service_rates,
        key=jax.random.PRNGKey(7),
    )

    probs_numpy = compute_numpy_policy_probs(net, Q, service_rates, rho, deterministic=False)
    wrapper_probs = StochasticNeuralPolicy(net, service_rates, rho=rho)(Q, np.random.default_rng(0))
    logits_jax = net(
        jnp.asarray(Q, dtype=jnp.float32),
        mu=jnp.asarray(service_rates, dtype=jnp.float32),
        rho=jnp.asarray(rho, dtype=jnp.float32),
    )
    probs_jax = np.array(jax.nn.softmax(logits_jax, axis=-1), dtype=np.float64)

    np.testing.assert_allclose(wrapper_probs, probs_numpy, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(probs_jax, probs_numpy, rtol=1e-6, atol=1e-6)
