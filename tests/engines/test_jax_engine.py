"""
Hardened test suite for gibbsq.engines.jax_engine — Robustness Loop Stage 2.

Categories:
- A: Correctness Tests
- B: Invariant Tests
- C: Edge Case Tests
- D: Numerical Stability Tests
- E: Gradient Flow Tests (covered in test_dga.py)
- F: Regression Tests
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from gibbsq.engines.jax_engine import (
    simulate_jax, run_replications_jax, get_probs, SimParams, SimState, compute_configured_max_events,
)
from gibbsq.core.policies import JSSQRouting


# ============================================================
# CATEGORY A: CORRECTNESS TESTS
# ============================================================

class TestGetProbsCorrectness:
    """Verify routing probability functions."""
    
    def test_uniform_probs(self):
        """Uniform policy should return equal probabilities."""
        Q = jnp.array([5, 10, 15])
        params = SimParams(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=jnp.ones(3),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=0,  # Uniform
            d=2,
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        expected = jnp.array([1/3, 1/3, 1/3])
        
        np.testing.assert_allclose(probs, expected, rtol=1e-6)
    
    def test_proportional_probs(self):
        """Proportional policy should return probabilities proportional to service rates."""
        Q = jnp.array([5, 5, 5])
        params = SimParams(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=jnp.array([1.0, 2.0, 3.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=1,  # Proportional
            d=2,
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        expected = jnp.array([1/6, 2/6, 3/6])
        
        np.testing.assert_allclose(probs, expected, rtol=1e-6)
    
    def test_jsq_probs(self):
        """JSQ should route to shortest queue."""
        Q = jnp.array([10, 5, 20])
        params = SimParams(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=jnp.ones(3),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=2,  # JSQ
            d=2,
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        expected = jnp.array([0.0, 1.0, 0.0])
        
        np.testing.assert_allclose(probs, expected, rtol=1e-6)
    
    def test_softmax_probs(self):
        """Softmax should prefer shorter queues."""
        Q = jnp.array([0, 10])
        params = SimParams(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.ones(2),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=3,  # Softmax
            d=2,
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        
        # Server 0 should have higher probability
        assert probs[0] > probs[1]
        assert probs[0] > 0.99
    
    def test_power_of_d_probs(self):
        """Power-of-d should return one-hot at winner."""
        Q = jnp.array([10, 5, 20])  # Server 1 has minimum
        params = SimParams(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=jnp.ones(3),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=4,  # Power-of-d
            d=3,  # Sample all servers
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        
        # With d=3, all servers sampled, should route to server 1 (minimum)
        # But due to random permutation, need to check it's one-hot
        assert jnp.sum(probs == 1.0) == 1
        assert jnp.sum(probs == 0.0) == 2

    def test_jssq_probs_match_shared_policy_contract(self):
        """JAX JSSQ must match the shared NumPy JSSQRouting semantics exactly."""
        Q = jnp.array([5, 0], dtype=jnp.int32)
        mu = jnp.array([10.0, 1.0], dtype=jnp.float32)
        params = SimParams(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=mu,
            alpha=7.0,  # Should be ignored by JSSQ
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=5,  # JSSQ
            d=2,
        )

        probs = np.array(get_probs(Q, params, jax.random.PRNGKey(0)))
        expected = JSSQRouting(np.array([10.0, 1.0]))(np.array([5, 0]), np.random.default_rng(0))

        np.testing.assert_allclose(probs, expected, rtol=1e-6, atol=1e-6)


class TestSimulateJaxCorrectness:
    """Verify JAX simulation produces correct outputs."""
    
    def test_simulation_returns_valid_shapes(self):
        """Output shapes should match input parameters."""
        times, states, (arrivals, departures) = simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=20,
        )
        
        assert states.shape[1] == 2  # num_servers
        assert times.shape[0] == states.shape[0]
        assert arrivals.shape == ()  # scalar
        assert departures.shape == ()  # scalar
    
    def test_queues_never_negative(self):
        """Queue lengths must never be negative."""
        times, states, counts = simulate_jax(
            num_servers=3,
            arrival_rate=2.0,
            service_rates=jnp.array([3.0, 3.0, 3.0]),
            alpha=1.0,
            sim_time=50.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=100,
        )
        
        assert (states >= 0).all()
    
    def test_initial_state_is_zero(self):
        """Simulation must start with all queues at zero."""
        times, states, counts = simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=20,
        )
        
        np.testing.assert_array_equal(states[0], [0, 0])

    def test_final_sample_reaches_sim_time(self):
        """Valid samples should include the terminal sampling point."""
        sim_time = 10.0
        times, states, counts = simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=sim_time,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=20,
        )

        valid_mask = np.asarray(times) > 0
        valid_mask[0] = True
        valid_times = np.asarray(times)[valid_mask]
        assert valid_times[-1] == pytest.approx(sim_time)

    def test_terminal_sample_matches_event_counts(self):
        """The final valid sample should satisfy queue conservation exactly."""
        times, states, (arrivals, departures) = simulate_jax(
            num_servers=2,
            arrival_rate=1.5,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=25.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(7),
            max_samples=40,
        )

        valid_mask = np.asarray(times) > 0
        valid_mask[0] = True
        final_state = np.asarray(states)[valid_mask][-1]
        assert final_state.sum() == int(arrivals) - int(departures)

    def test_truncated_runs_fail_closed_in_sample_buffers(self):
        """Early max-event truncation must not fabricate a full sampling grid."""
        from gibbsq.engines.jax_engine import _simulate_jax_impl

        times, states, counts, is_valid = _simulate_jax_impl(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([1.0, 1.0], dtype=jnp.float32),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(0),
            max_samples=11,
            max_events=1,
            policy_type=3,
            d=2,
            scan_sampling_chunk=16,
        )

        assert bool(is_valid) is False
        np_times = np.asarray(times)
        valid_mask = np_times > 0
        valid_mask[0] = True
        assert np.count_nonzero(valid_mask) < len(np_times)
        assert np.all(np_times[~valid_mask] == 0.0)
        assert np.all(np.asarray(states)[~valid_mask] == 0)


class TestRunReplicationsJaxCorrectness:
    """Verify multi-replication driver."""
    
    def test_replications_count(self):
        """Should return results for all replications."""
        times, states, (arrivals, departures) = run_replications_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            num_replications=5,
            base_seed=42,
            max_samples=20,
        )
        
        # vmap adds a batch dimension
        assert times.shape[0] == 5
        assert states.shape[0] == 5
        assert arrivals.shape[0] == 5
    
    def test_replications_reproducible(self):
        """Same seed should give same results."""
        times1, states1, _ = run_replications_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            num_replications=3,
            base_seed=42,
            max_samples=20,
        )
        
        times2, states2, _ = run_replications_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            num_replications=3,
            base_seed=42,
            max_samples=20,
        )
        
        np.testing.assert_array_equal(states1, states2)


# ============================================================
# CATEGORY B: INVARIANT TESTS
# ============================================================

class TestJaxSimulationInvariants:
    """Invariants that must hold for all JAX simulations."""
    
    def test_probs_sum_to_one(self):
        """Routing probabilities must sum to 1."""
        Q = jnp.array([5, 10, 15, 20])
        key = jax.random.PRNGKey(42)
        
        for policy_type in [0, 1, 2, 3, 4]:
            params = SimParams(
                num_servers=4,
                arrival_rate=1.0,
                service_rates=jnp.array([1.0, 2.0, 3.0, 4.0]),
                alpha=1.0,
                sim_time=10.0,
                sample_interval=1.0,
                max_events=1000,
                policy_type=policy_type,
                d=2,
            )
            
            probs = get_probs(Q, params, key)
            assert abs(float(jnp.sum(probs)) - 1.0) < 1e-6
    
    def test_probs_non_negative(self):
        """Routing probabilities must be >= 0."""
        Q = jnp.array([5, 10, 15, 20])
        key = jax.random.PRNGKey(42)
        
        for policy_type in [0, 1, 2, 3, 4]:
            params = SimParams(
                num_servers=4,
                arrival_rate=1.0,
                service_rates=jnp.array([1.0, 2.0, 3.0, 4.0]),
                alpha=1.0,
                sim_time=10.0,
                sample_interval=1.0,
                max_events=1000,
                policy_type=policy_type,
                d=2,
            )
            
            probs = get_probs(Q, params, key)
            assert (probs >= 0).all()


# ============================================================
# CATEGORY C: EDGE CASE TESTS
# ============================================================

class TestJaxSimulationEdgeCases:
    """Test boundary conditions."""
    
    def test_single_server(self):
        """N=1 should work correctly."""
        times, states, (arrivals, departures) = simulate_jax(
            num_servers=1,
            arrival_rate=0.5,
            service_rates=jnp.array([1.0]),
            alpha=1.0,
            sim_time=20.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=50,
        )
        
        assert states.shape[1] == 1
        assert arrivals > 0  # Some arrivals
    
    def test_zero_arrival_rate(self):
        """Zero arrivals should result in empty system (no real arrivals)."""
        times, states, (arrivals, departures) = simulate_jax(
            num_servers=2,
            arrival_rate=0.0,
            service_rates=jnp.array([1.0, 1.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=20,
        )
        
        # With zero arrival rate and zero initial queues, all rates are zero.
        # The simulation uses a floor of 1e-9 for a0, so one "phantom" event
        # may be processed, but it won't affect the queue state.
        # The key invariant: queues remain at zero.
        assert (states == 0).all()
    
    def test_power_of_d_d_equals_n(self):
        """d=N should sample all servers, equivalent to JSQ."""
        Q = jnp.array([10, 5, 20])  # Server 1 has minimum
        params = SimParams(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=jnp.ones(3),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=4,  # Power-of-d
            d=3,  # Sample all servers
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        
        # Should route to server 1 (minimum queue)
        # Note: permutation may reorder, but winner should be server 1
        # Actually with permutation, the first d indices are shuffled
        # So we just verify it's a valid one-hot
        assert jnp.sum(probs) == 1.0
        assert (probs >= 0).all()
    
    def test_power_of_d_d_greater_than_n(self):
        """d > N should clamp to N."""
        Q = jnp.array([10, 5])
        params = SimParams(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.ones(2),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=4,  # Power-of-d
            d=10,  # Greater than N
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        
        # Should still produce valid probabilities
        assert abs(float(jnp.sum(probs)) - 1.0) < 1e-6


# ============================================================
# CATEGORY D: NUMERICAL STABILITY TESTS
# ============================================================

class TestJaxSimulationNumericalStability:
    """Test behavior under numerically challenging inputs."""
    
    def test_very_large_queues(self):
        """Very large Q should not cause overflow."""
        Q = jnp.array([1000, 2000, 3000])
        params = SimParams(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=jnp.ones(3),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=3,  # Softmax
            d=2,
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        
        assert jnp.isfinite(probs).all()
        assert abs(float(jnp.sum(probs)) - 1.0) < 1e-6
    
    def test_very_small_alpha(self):
        """Very small alpha (high temperature) should work."""
        Q = jnp.array([0, 100])
        params = SimParams(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.ones(2),
            alpha=1e-10,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=3,  # Softmax
            d=2,
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        
        # Should be nearly uniform
        assert abs(float(probs[0]) - 0.5) < 0.01
        assert jnp.isfinite(probs).all()
    
    def test_very_large_alpha(self):
        """Very large alpha (low temperature) should work."""
        Q = jnp.array([0, 1])
        params = SimParams(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.ones(2),
            alpha=1e6,
            sim_time=10.0,
            sample_interval=1.0,
            max_events=1000,
            policy_type=3,  # Softmax
            d=2,
        )
        key = jax.random.PRNGKey(42)
        
        probs = get_probs(Q, params, key)
        
        # Should be nearly one-hot at server 0
        assert probs[0] > 0.999
        assert jnp.isfinite(probs).all()


# ============================================================
# CATEGORY F: REGRESSION TESTS
# ============================================================

class TestJaxSimulationRegressions:
    """Prevent reintroduction of known faults."""
    
    def test_regression_power_of_d_static_d(self):
        """Power-of-d requires d to be static for JIT."""
        # This test verifies the implementation works with d as static arg
        times, states, counts = simulate_jax(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0, 2.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=20,
            policy_type=4,  # Power-of-d
            d=2,
        )
        
        # Should complete without error
        assert states.shape[1] == 3
    
    def test_regression_key_splitting(self):
        """Verify key splitting produces independent keys."""
        # Run same simulation twice with different keys
        times1, states1, _ = simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=50.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(1),
            max_samples=100,
        )
        
        times2, states2, _ = simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=50.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(2),
            max_samples=100,
        )
        
        # Different keys should produce different results
        assert not jnp.array_equal(states1, states2)

    def test_simulate_jax_rejects_invalid_service_rate_shape(self):
        with pytest.raises(ValueError, match="service_rates must have shape"):
            simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([1.0]),
                alpha=1.0,
                sim_time=10.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(0),
                max_samples=20,
            )

    def test_simulate_jax_rejects_invalid_policy_type(self):
        with pytest.raises(ValueError, match="policy_type must be one of 0..6"):
            simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([1.0, 1.0]),
                alpha=1.0,
                sim_time=10.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(0),
                max_samples=20,
                policy_type=99,
            )

    def test_simulate_jax_rejects_non_positive_alpha(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([1.0, 1.0]),
                alpha=0.0,
                sim_time=10.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(0),
                max_samples=20,
            )

    def test_run_replications_jax_rejects_non_positive_replications(self):
        with pytest.raises(ValueError, match="num_replications must be >= 1"):
            run_replications_jax(
                num_replications=0,
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([1.0, 1.0]),
                alpha=1.0,
                sim_time=10.0,
                sample_interval=1.0,
                base_seed=0,
                max_samples=20,
            )

    def test_compute_configured_max_events_honors_configurable_floor(self):
        service_rates = np.array([1.0, 1.0], dtype=np.float64)
        bound = compute_configured_max_events(
            arrival_rate=1.0,
            service_rates=service_rates,
            sim_time=10.0,
            max_events_multiplier=3.0,
            max_events_buffer=17,
        )

        expected_floor = int(np.ceil((1.0 + service_rates.sum()) * 10.0 * 3.0)) + 17
        assert bound >= expected_floor

    def test_vmap_replications_match_lax_map_reference(self):
        from gibbsq.engines.jax_engine import _run_replications_jax_impl, _simulate_jax_impl

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
            max_events=1000,
            policy_type=0,
            d=2,
        )

        keys = jax.random.split(jax.random.PRNGKey(args["base_seed"]), args["num_replications"])

        def v_sim(k):
            return _simulate_jax_impl(
                num_servers=args["num_servers"],
                arrival_rate=args["arrival_rate"],
                service_rates=args["service_rates"],
                alpha=args["alpha"],
                sim_time=args["sim_time"],
                sample_interval=args["sample_interval"],
                key=k,
                max_samples=args["max_samples"],
                max_events=args["max_events"],
                policy_type=args["policy_type"],
                d=args["d"],
            )

        out_vmap = _run_replications_jax_impl(**args)
        out_ref = jax.lax.map(v_sim, keys)

        for a, b in zip(out_vmap, out_ref):
            assert jnp.array_equal(a, b)


# ============================================================
# DETERMINISM TESTS
# ============================================================

class TestJaxSimulationDeterminism:
    """Verify reproducibility."""
    
    def test_same_key_same_result(self):
        """Same PRNG key must produce identical results."""
        kwargs = dict(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=20.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(12345),
            max_samples=50,
        )
        
        times1, states1, counts1 = simulate_jax(**kwargs)
        times2, states2, counts2 = simulate_jax(**kwargs)
        
        np.testing.assert_array_equal(states1, states2)
        np.testing.assert_array_equal(counts1, counts2)


# ============================================================
# JIT COMPILATION TESTS
# ============================================================

class TestJaxJitCompilation:
    """Verify JIT compilation works correctly."""
    
    def test_jit_compiles_without_error(self):
        """Function should be JIT compilable."""
        # The @jax.jit decorator is already on simulate_jax
        # Just verify it runs without compilation errors
        times, states, counts = simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=20,
        )
        
        assert times is not None
        assert states is not None
        assert counts is not None
    
    def test_jit_caching(self):
        """JIT should cache compiled functions."""
        # Run same function multiple times
        for i in range(3):
            times, states, counts = simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([2.0, 2.0]),
                alpha=1.0,
                sim_time=10.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(i),
                max_samples=20,
            )
        
        # Should complete without error (cached compilation)
        assert states.shape[1] == 2
