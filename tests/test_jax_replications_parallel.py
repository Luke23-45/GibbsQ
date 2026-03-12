import jax
import jax.numpy as jnp

from gibbsq.engines.jax_engine import _run_replications_jax_impl, _simulate_jax_impl


def _reference_lax_map(
    num_replications,
    num_servers,
    arrival_rate,
    service_rates,
    alpha,
    sim_time,
    sample_interval,
    base_seed,
    max_samples,
    policy_type,
    d,
):
    keys = jax.random.split(jax.random.PRNGKey(base_seed), num_replications)

    def v_sim(k):
        return _simulate_jax_impl(
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates,
            alpha=alpha,
            sim_time=sim_time,
            sample_interval=sample_interval,
            key=k,
            max_samples=max_samples,
            policy_type=policy_type,
            d=d,
        )

    return jax.lax.map(v_sim, keys)


def test_vmap_replications_match_lax_map_reference():
    args = dict(
        num_replications=3,
        num_servers=2,
        arrival_rate=1.0,
        service_rates=jnp.array([1.0, 1.5], dtype=jnp.float32),
        alpha=1.0,
        sim_time=5.0,
        sample_interval=0.5,
        base_seed=123,
        max_samples=11,
        policy_type=0,
        d=2,
    )

    out_vmap = _run_replications_jax_impl(**args)
    out_ref = _reference_lax_map(**args)

    for a, b in zip(out_vmap, out_ref):
        assert jnp.array_equal(a, b)
