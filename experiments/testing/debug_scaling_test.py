"""
GibbsQ Diagnostic: sim_time scaling test
=========================================
Tests the EXACT relationship between sim_time (= event count) and wall-clock time.
Uses R=1 replication, N=4 servers to match production N, but varies sim_time.
"""

import time
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def run_scaling_test():
    log.info("=" * 70)
    log.info("  sim_time Scaling Test (N=4, R=1)")
    log.info("=" * 70)

    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial
    from typing import NamedTuple
    from dataclasses import dataclass

    platform = jax.default_backend()
    log.info(f"JAX platform: {platform}")

    # ── Inline the kernel (identical to jax_engine.py) ──────────────────
    RATE_EPSILON = 1e-12

    class SimState(NamedTuple):
        t: jnp.ndarray; Q: jnp.ndarray; key: jax.random.PRNGKey
        sample_idx: jnp.ndarray; times_buf: jnp.ndarray
        states_buf: jnp.ndarray; arrival_count: jnp.ndarray
        departure_count: jnp.ndarray

    @dataclass(frozen=True)
    class SimParams:
        num_servers: int; arrival_rate: float; service_rates: jnp.ndarray
        alpha: float; sim_time: float; sample_interval: float
        policy_type: int; d: int

    def get_probs(Q, params, key):
        def uniform_p(_): return jnp.ones(params.num_servers) / params.num_servers
        def proportional_p(_): return params.service_rates / jnp.sum(params.service_rates)
        def jsq_p(_):
            is_min = (Q == jnp.min(Q))
            noise = jax.random.uniform(key, shape=Q.shape)
            masked_noise = jnp.where(is_min, noise, -jnp.inf)
            idx = jnp.argmax(masked_noise)
            return jnp.zeros(params.num_servers).at[idx].set(1.0)
        def softmax_p(_):
            logits = -params.alpha * Q.astype(params.service_rates.dtype)
            max_logit = jnp.max(logits)
            exp_logits = jnp.exp(logits - max_logit)
            return exp_logits / jnp.sum(exp_logits)
        def power_of_d_p(_):
            N = params.num_servers
            d_actual = min(params.d, N)
            perm = jax.random.permutation(key, N)
            candidates = lax.dynamic_slice(perm, (0,), (d_actual,))
            candidate_queues = Q[candidates]
            winner_local = jnp.argmin(candidate_queues)
            winner = candidates[winner_local]
            return jnp.zeros(N).at[winner].set(1.0)
        return lax.switch(params.policy_type, [uniform_p, proportional_p, jsq_p, softmax_p, power_of_d_p], None)

    def cond_fun(state, params):
        max_samples = state.times_buf.shape[0]
        return (state.t < params.sim_time) & (state.sample_idx < max_samples)

    def body_fun(state, params):
        k1, k2, k3, k4 = jax.random.split(state.key, 4)
        probs = get_probs(state.Q, params, k4)
        arrival_rates = params.arrival_rate * probs
        departure_rates = params.service_rates * (state.Q > 0).astype(jnp.float32)
        rates = jnp.concatenate([arrival_rates, departure_rates])
        a0 = jnp.sum(rates)

        def process_event(s):
            tau = jax.random.exponential(k1) / a0; new_t = s.t + tau
            u = jax.random.uniform(k2) * a0; cumrates = jnp.cumsum(rates)
            event = jnp.sum(cumrates < u)
            is_arrival = event < params.num_servers
            srv_idx = jnp.where(is_arrival, event, event - params.num_servers)
            delta = jnp.where(is_arrival, 1, -1)
            new_Q = s.Q.at[srv_idx].add(delta)
            new_arr = s.arrival_count + jnp.where(is_arrival, 1, 0)
            new_dep = s.departure_count + jnp.where(is_arrival, 0, 1)
            def fill_samples(carry):
                idx, tb, sb = carry
                nst = idx * params.sample_interval
                sr = (new_t >= nst) & (nst <= params.sim_time)
                tb = tb.at[idx].set(jnp.where(sr, nst, tb[idx]))
                sb = sb.at[idx].set(jnp.where(sr, s.Q, sb[idx]))
                return (idx + jnp.where(sr, 1, 0), tb, sb)
            mi = s.times_buf.shape[0]
            fi, ft, fs = lax.while_loop(
                lambda c: (c[0] < mi) & (c[0] * params.sample_interval <= new_t) & (c[0] * params.sample_interval <= params.sim_time),
                fill_samples, (s.sample_idx, s.times_buf, s.states_buf))
            return s._replace(t=new_t, Q=new_Q, key=k3, sample_idx=fi, times_buf=ft, states_buf=fs, arrival_count=new_arr, departure_count=new_dep)

        def skip_event(s):
            mi = s.times_buf.shape[0]
            def fill_r(carry):
                idx, tb, sb = carry
                nst = idx * params.sample_interval; sr = nst <= params.sim_time
                tb = tb.at[idx].set(jnp.where(sr, nst, tb[idx]))
                sb = sb.at[idx].set(jnp.where(sr, s.Q, sb[idx]))
                return (idx + jnp.where(sr, 1, 0), tb, sb)
            fi, ft, fs = lax.while_loop(
                lambda c: (c[0] < mi) & (c[0] * params.sample_interval <= params.sim_time),
                fill_r, (s.sample_idx, s.times_buf, s.states_buf))
            return s._replace(t=params.sim_time + 1.0, key=k3, sample_idx=fi, times_buf=ft, states_buf=fs)

        return lax.cond(a0 > RATE_EPSILON, process_event, skip_event, state)

    def _simulate(num_servers, arrival_rate, service_rates, alpha, sim_time, sample_interval, key, max_samples, policy_type, d):
        params = SimParams(num_servers=num_servers, arrival_rate=arrival_rate, service_rates=service_rates, alpha=alpha, sim_time=sim_time, sample_interval=sample_interval, policy_type=policy_type, d=d)
        tb = jnp.zeros(max_samples); sb = jnp.zeros((max_samples, num_servers), dtype=jnp.int32)
        tb = tb.at[0].set(0.0); sb = sb.at[0].set(jnp.zeros(num_servers, dtype=jnp.int32))
        init = SimState(t=0.0, Q=jnp.zeros(num_servers, dtype=jnp.int32), key=key, sample_idx=1, times_buf=tb, states_buf=sb, arrival_count=0, departure_count=0)
        final = lax.while_loop(lambda s: cond_fun(s, params), lambda s: body_fun(s, params), init)
        return final.times_buf, final.states_buf, (final.arrival_count, final.departure_count)

    # Single replication (no vmap), N=4 — test scaling with sim_time
    @partial(jax.jit, static_argnames=("num_servers", "max_samples", "policy_type", "d"))
    def run_single(num_servers, arrival_rate, service_rates, alpha, sim_time, sample_interval, key, max_samples, policy_type=3, d=2):
        return _simulate(num_servers, arrival_rate, service_rates, alpha, sim_time, sample_interval, key, max_samples, policy_type, d)

    # vmap version (R=3), N=4
    @partial(jax.jit, static_argnames=("num_replications", "num_servers", "max_samples", "policy_type", "d"))
    def run_vmap(num_replications, num_servers, arrival_rate, service_rates, alpha, sim_time, sample_interval, base_seed, max_samples, policy_type=3, d=2):
        keys = jax.random.split(jax.random.PRNGKey(base_seed), num_replications)
        v = lambda k: _simulate(num_servers=num_servers, arrival_rate=arrival_rate, service_rates=service_rates, alpha=alpha, sim_time=sim_time, sample_interval=sample_interval, key=k, max_samples=max_samples, policy_type=policy_type, d=d)
        return jax.vmap(v)(keys)

    N = 4
    mu = jnp.ones(N) * 2.0
    lam = 0.8 * float(jnp.sum(mu))  # rho = 0.8

    # ── Part A: Single replication, varying sim_time ────────────────────
    log.info("")
    log.info("Part A: Single replication (R=1), N=4, varying sim_time")
    log.info(f"  mu={[2.0]*N}, lam={lam:.1f}, rho=0.8, sample_interval=1.0")
    log.info(f"  {'sim_time':>10} {'max_sam':>8} {'~events':>8} {'1st_call':>10} {'2nd_call':>10} {'compile':>10}")
    log.info("  " + "-" * 60)

    results_single = []
    for sim_time_val in [10, 50, 100, 500, 1000, 2000, 5000]:
        max_samples = int(sim_time_val / 1.0) + 1
        expected_events = int((lam + float(jnp.sum(mu))) * sim_time_val)

        key = jax.random.PRNGKey(42)
        t0 = time.perf_counter()
        tb, sb, (a, d) = run_single(N, lam, mu, 1.0, float(sim_time_val), 1.0, key, max_samples, 3, 2)
        tb.block_until_ready()
        t1 = time.perf_counter() - t0

        key2 = jax.random.PRNGKey(99)
        t0 = time.perf_counter()
        tb2, sb2, (a2, d2) = run_single(N, lam, mu, 1.0, float(sim_time_val), 1.0, key2, max_samples, 3, 2)
        tb2.block_until_ready()
        t2 = time.perf_counter() - t0

        tc = t1 - t2
        log.info(f"  {sim_time_val:>10} {max_samples:>8} {expected_events:>8,} {t1:>10.2f}s {t2:>10.2f}s {tc:>10.2f}s")
        results_single.append((sim_time_val, max_samples, expected_events, t1, t2, tc))

    # ── Part B: R=3 vmap, N=4, key sim_time values ─────────────────────
    log.info("")
    log.info("Part B: vmap R=3, N=4, key sim_time values")
    log.info(f"  {'sim_time':>10} {'max_sam':>8} {'~events':>8} {'1st_call':>10} {'2nd_call':>10} {'compile':>10}")
    log.info("  " + "-" * 60)

    results_vmap = []
    for sim_time_val in [10, 100, 500, 1000]:
        max_samples = int(sim_time_val / 1.0) + 1
        expected_events = int((lam + float(jnp.sum(mu))) * sim_time_val) * 3

        t0 = time.perf_counter()
        tb, sb, (a, d) = run_vmap(3, N, lam, mu, 1.0, float(sim_time_val), 1.0, 42, max_samples, 3, 2)
        tb.block_until_ready()
        t1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        tb2, sb2, (a2, d2) = run_vmap(3, N, lam, mu, 1.0, float(sim_time_val), 1.0, 99, max_samples, 3, 2)
        tb2.block_until_ready()
        t2 = time.perf_counter() - t0

        tc = t1 - t2
        log.info(f"  {sim_time_val:>10} {max_samples:>8} {expected_events:>8,} {t1:>10.2f}s {t2:>10.2f}s {tc:>10.2f}s")
        results_vmap.append((sim_time_val, max_samples, expected_events, t1, t2, tc))

    log.info("")
    log.info("=" * 70)
    log.info("  ANALYSIS")
    log.info("=" * 70)

    # Compute scaling ratios for single replication
    if len(results_single) >= 2:
        for i in range(1, len(results_single)):
            prev_t, prev_events = results_single[i-1][4], results_single[i-1][2]
            curr_t, curr_events = results_single[i][4], results_single[i][2]
            if prev_t > 0.001:
                event_ratio = curr_events / max(prev_events, 1)
                time_ratio = curr_t / prev_t
                log.info(f"  T={results_single[i][0]:>5}: events {event_ratio:.1f}x -> runtime {time_ratio:.1f}x")

    log.info("=" * 70)


if __name__ == "__main__":
    run_scaling_test()
