"""
DEPRECATED: GibbsQ Stress Test Diagnostic Runner (Standalone)
===================================================
This file contains the outdated `lax.while_loop` engine logic, which 
has been replaced by `lax.scan` in the main `jax_engine.py`.
It is retained only as a historical artifact and will not run.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

def run_diagnostic():
    log.error("This diagnostic script is DEPRECATED and Disabled.")
    log.error("The JAX engine has been fully rewritten using lax.scan.")
    sys.exit(1)

if __name__ == "__main__":
    run_diagnostic()


import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def _legacy_run_diagnostic_DEPRECATED():
    log.info("=" * 70)
    log.info("  GibbsQ Stress Test Diagnostic Runner")
    log.info("=" * 70)

    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial
    from typing import NamedTuple
    from dataclasses import dataclass

    platform = jax.default_backend()
    devices = jax.devices()
    log.info(f"JAX platform: {platform} ({len(devices)} device(s))")
    log.info(f"Devices: {devices}")

    # ── Inline the essential simulation kernel from jax_engine.py ───────
    RATE_EPSILON = 1e-12

    class SimState(NamedTuple):
        t:               jnp.ndarray
        Q:               jnp.ndarray
        key:             jax.random.PRNGKey
        sample_idx:      jnp.ndarray
        times_buf:       jnp.ndarray
        states_buf:      jnp.ndarray
        arrival_count:   jnp.ndarray
        departure_count: jnp.ndarray

    @dataclass(frozen=True)
    class SimParams:
        num_servers:     int
        arrival_rate:    float
        service_rates:   jnp.ndarray
        alpha:           float
        sim_time:        float
        sample_interval: float
        policy_type:     int
        d:               int

    def get_probs(Q, params, key):
        def uniform_p(_):
            return jnp.ones(params.num_servers) / params.num_servers
        def proportional_p(_):
            return params.service_rates / jnp.sum(params.service_rates)
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
        return lax.switch(
            params.policy_type,
            [uniform_p, proportional_p, jsq_p, softmax_p, power_of_d_p],
            None
        )

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
            tau = jax.random.exponential(k1) / a0
            new_t = s.t + tau
            u = jax.random.uniform(k2) * a0
            cumrates = jnp.cumsum(rates)
            event = jnp.sum(cumrates < u)
            is_arrival = event < params.num_servers
            srv_idx = jnp.where(is_arrival, event, event - params.num_servers)
            delta = jnp.where(is_arrival, 1, -1)
            new_Q = s.Q.at[srv_idx].add(delta)
            new_arrival_count = s.arrival_count + jnp.where(is_arrival, 1, 0)
            new_departure_count = s.departure_count + jnp.where(is_arrival, 0, 1)

            def fill_samples(carry):
                idx, times_b, states_b = carry
                next_sample_t = idx * params.sample_interval
                should_record = (new_t >= next_sample_t) & (next_sample_t <= params.sim_time)
                new_times_b = times_b.at[idx].set(
                    jnp.where(should_record, next_sample_t, times_b[idx])
                )
                new_states_b = states_b.at[idx].set(
                    jnp.where(should_record, s.Q, states_b[idx])
                )
                new_idx = idx + jnp.where(should_record, 1, 0)
                return (new_idx, new_times_b, new_states_b)

            max_iters = s.times_buf.shape[0]
            final_idx, final_times, final_states = lax.while_loop(
                lambda carry: (carry[0] < max_iters) &
                             (carry[0] * params.sample_interval <= new_t) &
                             (carry[0] * params.sample_interval <= params.sim_time),
                fill_samples,
                (s.sample_idx, s.times_buf, s.states_buf)
            )

            return s._replace(
                t=new_t, Q=new_Q, key=k3,
                sample_idx=final_idx,
                times_buf=final_times, states_buf=final_states,
                arrival_count=new_arrival_count,
                departure_count=new_departure_count
            )

        def skip_event(s):
            max_iters = s.times_buf.shape[0]
            def fill_remaining(carry):
                idx, times_b, states_b = carry
                next_sample_t = idx * params.sample_interval
                should_record = next_sample_t <= params.sim_time
                new_times_b = times_b.at[idx].set(
                    jnp.where(should_record, next_sample_t, times_b[idx])
                )
                new_states_b = states_b.at[idx].set(
                    jnp.where(should_record, s.Q, states_b[idx])
                )
                new_idx = idx + jnp.where(should_record, 1, 0)
                return (new_idx, new_times_b, new_states_b)

            final_idx, final_times, final_states = lax.while_loop(
                lambda carry: (carry[0] < max_iters) &
                             (carry[0] * params.sample_interval <= params.sim_time),
                fill_remaining,
                (s.sample_idx, s.times_buf, s.states_buf)
            )
            return s._replace(
                t=params.sim_time + 1.0, key=k3,
                sample_idx=final_idx,
                times_buf=final_times, states_buf=final_states,
            )

        return lax.cond(a0 > RATE_EPSILON, process_event, skip_event, state)

    def _simulate_impl(num_servers, arrival_rate, service_rates, alpha,
                        sim_time, sample_interval, key, max_samples,
                        policy_type, d):
        params = SimParams(
            num_servers=num_servers, arrival_rate=arrival_rate,
            service_rates=service_rates, alpha=alpha,
            sim_time=sim_time, sample_interval=sample_interval,
            policy_type=policy_type, d=d
        )
        times_buf = jnp.zeros(max_samples)
        states_buf = jnp.zeros((max_samples, num_servers), dtype=jnp.int32)
        times_buf = times_buf.at[0].set(0.0)
        states_buf = states_buf.at[0].set(jnp.zeros(num_servers, dtype=jnp.int32))

        init_state = SimState(
            t=0.0, Q=jnp.zeros(num_servers, dtype=jnp.int32),
            key=key, sample_idx=1,
            times_buf=times_buf, states_buf=states_buf,
            arrival_count=0, departure_count=0
        )
        final_state = lax.while_loop(
            lambda s: cond_fun(s, params),
            lambda s: body_fun(s, params),
            init_state
        )
        return final_state.times_buf, final_state.states_buf, (final_state.arrival_count, final_state.departure_count)

    @partial(jax.jit, static_argnames=("num_replications", "num_servers", "max_samples", "policy_type", "d"))
    def run_replications(num_replications, num_servers, arrival_rate,
                         service_rates, alpha, sim_time, sample_interval,
                         base_seed, max_samples, policy_type=3, d=2):
        keys = jax.random.split(jax.random.PRNGKey(base_seed), num_replications)
        v_sim = lambda k: _simulate_impl(
            num_servers=num_servers, arrival_rate=arrival_rate,
            service_rates=service_rates, alpha=alpha,
            sim_time=sim_time, sample_interval=sample_interval,
            key=k, max_samples=max_samples, policy_type=policy_type, d=d
        )
        return jax.vmap(v_sim)(keys)

    # ── Define test cases ───────────────────────────────────────────────
    tests = [
        # (label, num_reps, num_servers, sim_time, sample_interval, max_samples)
        ("H1: JIT baseline (N=2, T=10, R=1, M=11)",       1, 2,   10.0, 1.0,    11),
        ("H2: sim_time 10x (N=2, T=100, R=1, M=101)",     1, 2,  100.0, 1.0,   101),
        ("H3a: large buf (N=2, T=100, R=1, M=5001)",      1, 2,  100.0, 1.0,  5001),
        ("H3b: huge buf (N=2, T=100, R=1, M=10001)",      1, 2,  100.0, 1.0, 10001),
        ("H4: vmap batch (N=2, T=10, R=3, M=11)",         3, 2,   10.0, 1.0,    11),
        ("H5: production (N=4, T=5000, R=3, M=5001)",     3, 4, 5000.0, 1.0,  5001),
    ]

    results = []

    for i, (label, num_reps, num_servers, sim_time, sample_interval, max_samples) in enumerate(tests):
        log.info("")
        log.info(f"--- TEST {i+1}/{len(tests)}: {label} ---")

        mu = jnp.ones(num_servers) * 2.0
        lam = 0.8 * float(jnp.sum(mu))

        total_rate = lam + float(jnp.sum(mu))
        expected_events = int(total_rate * sim_time)
        log.info(f"  Expected events/replication: ~{expected_events:,}")
        log.info(f"  Buffer shape: ({max_samples}, {num_servers})")
        log.info(f"  Replications: {num_reps}")

        # Timeout: 120s for production, 300s for others
        timeout_label = "production" in label.lower()
        log.info(f"  Starting... (will report time)")

        t_start = time.perf_counter()

        try:
            # First call: JIT compile + execute
            result = run_replications(
                num_replications=num_reps,
                num_servers=num_servers,
                arrival_rate=lam,
                service_rates=mu,
                alpha=1.0,
                sim_time=sim_time,
                sample_interval=sample_interval,
                base_seed=42,
                max_samples=max_samples,
                policy_type=3,
                d=2,
            )

            times_buf, states_buf, (arrs, deps) = result
            times_buf.block_until_ready()
            states_buf.block_until_ready()

            t_first_call = time.perf_counter() - t_start
            log.info(f"  First call (compile+run): {t_first_call:.2f}s")

            # Second call: cached JIT, pure execution
            t_start2 = time.perf_counter()
            result2 = run_replications(
                num_replications=num_reps,
                num_servers=num_servers,
                arrival_rate=lam,
                service_rates=mu,
                alpha=1.0,
                sim_time=sim_time,
                sample_interval=sample_interval,
                base_seed=99,
                max_samples=max_samples,
                policy_type=3,
                d=2,
            )
            times_buf2, states_buf2, (arrs2, deps2) = result2
            times_buf2.block_until_ready()
            states_buf2.block_until_ready()

            t_second_call = time.perf_counter() - t_start2
            t_compile_est = t_first_call - t_second_call

            log.info(f"  Second call (run only):   {t_second_call:.2f}s")
            log.info(f"  Estimated compile time:   {t_compile_est:.2f}s")
            log.info(f"  Arrivals[0]: {int(arrs[0])}, Departures[0]: {int(deps[0])}")

            results.append((label, t_first_call, t_second_call, t_compile_est, "OK"))

        except Exception as e:
            t_elapsed = time.perf_counter() - t_start
            log.error(f"  FAILED after {t_elapsed:.2f}s: {e}")
            results.append((label, t_elapsed, -1, -1, f"FAIL: {e}"))

    # ── Print diagnostic table ──────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("  DIAGNOSTIC RESULTS")
    log.info("=" * 70)
    log.info(f"  {'Test':<48} {'1stCall':>8} {'2ndCall':>8} {'Compile':>8} {'Status':>6}")
    log.info("  " + "-" * 80)

    for label, t1, t2, tc, status in results:
        t1_s = f"{t1:.2f}s" if t1 >= 0 else "N/A"
        t2_s = f"{t2:.2f}s" if t2 >= 0 else "N/A"
        tc_s = f"{tc:.2f}s" if tc >= 0 else "N/A"
        log.info(f"  {label:<48} {t1_s:>8} {t2_s:>8} {tc_s:>8} {status:>6}")

    log.info("")
    log.info("=" * 70)
    log.info("  INTERPRETATION GUIDE")
    log.info("=" * 70)
    log.info("  If H1 compile time >> 5s       -> nested JIT is root cause")
    log.info("  If H2-H1 runtime diff >> 10x   -> event count is bottleneck")
    log.info("  If H3a >> H2 (compile or run)  -> buffer shape causes blowup")
    log.info("  If H4 >> H1                    -> vmap batch overhead")
    log.info("  If H5 hangs / very slow        -> combination of factors")
    log.info("=" * 70)


if __name__ == "__main__":
    run_diagnostic()
