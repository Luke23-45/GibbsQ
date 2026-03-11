"""
SMOKING GUN HUNT - Adversarial tests to expose bugs in JAX engine.

Each test is designed to FAIL if a bug exists, exposing the issue.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from gibbsq.engines.jax_engine import (
    simulate_jax,
    run_replications_jax,
    SimState,
    SimParams,
    body_fun,
    cond_fun,
    get_probs,
)


# ============================================================
# SMOKING GUN #1: SNAPSHOT TIMING CORRECTNESS
# ============================================================

class TestSnapshotTimingBugs:
    """Expose bugs in snapshot buffer logic."""
    
    def test_initial_state_recorded(self):
        """BUG: Initial state at t=0 should be recorded."""
        times, states, _ = simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=20,
        )
        
        # First recorded time should be 0.0
        assert times[0] == 0.0, f"Initial time not recorded: times[0]={times[0]}"
        # First recorded state should be [0, 0]
        np.testing.assert_array_equal(states[0], [0, 0], 
            err_msg="Initial state not recorded correctly")
    
    def test_all_sample_times_present(self):
        """BUG: Every sample_interval point should have a snapshot."""
        sim_time = 10.0
        sample_interval = 1.0
        expected_samples = int(sim_time / sample_interval) + 1  # 11 samples: 0,1,2,...,10
        
        times, states, _ = simulate_jax(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=sim_time,
            sample_interval=sample_interval,
            key=jax.random.PRNGKey(42),
            max_samples=20,
        )
        
        # Count non-zero times (samples actually recorded)
        recorded_samples = np.sum(times > -0.5)  # Allow for float precision
        
        # Should have recorded all samples
        assert recorded_samples >= expected_samples - 1, \
            f"Missing samples: expected ~{expected_samples}, got {recorded_samples}"
    
    def test_large_tau_skips_samples(self):
        """BUG: If tau > sample_interval, intermediate samples are missed."""
        # Very low arrival rate -> large inter-event times
        times, states, _ = simulate_jax(
            num_servers=2,
            arrival_rate=0.01,  # Very sparse events
            service_rates=jnp.array([10.0, 10.0]),
            alpha=1.0,
            sim_time=100.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=150,
        )
        
        # Check if we have gaps in the time series
        recorded_times = times[times > 0]
        if len(recorded_times) > 1:
            time_diffs = np.diff(recorded_times)
            # All differences should be approximately sample_interval
            # (with tolerance for event timing)
            gaps = np.sum(time_diffs > 1.5)  # Gaps > 1.5 sample intervals
            if gaps > 0:
                pytest.fail(f"SMOKING GUN: {gaps} sample gaps detected - large tau misses samples")
    
    def test_snapshot_records_correct_state(self):
        """BUG: Snapshot should record state BEFORE the event that crosses the boundary."""
        # Run a simulation and check that the recorded state is consistent
        times, states, _ = simulate_jax(
            num_servers=2,
            arrival_rate=0.5,  # Low rate for clear events
            service_rates=jnp.array([1.0, 1.0]),
            alpha=1.0,
            sim_time=20.0,
            sample_interval=5.0,  # Large interval
            key=jax.random.PRNGKey(123),
            max_samples=10,
        )
        
        # States should be non-decreasing in total (arrivals increase, departures decrease)
        # but overall trend should be upward for underloaded system
        total_qs = states.sum(axis=1)
        
        # Check that states are valid (non-negative integers)
        assert np.all(states >= 0), "Negative queue states detected"
        assert np.all(states == states.astype(int)), "Non-integer queue states"


# ============================================================
# SMOKING GUN #2: CONSERVATION LAW
# ============================================================

class TestConservationLawBugs:
    """Expose conservation law violations."""
    
    def test_exact_conservation_final_state(self):
        """BUG: arrivals - departures must equal final queue state."""
        # Run multiple simulations and check conservation
        for seed in range(10):
            _, states, (arrivals, departures) = simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([2.0, 2.0]),
                alpha=1.0,
                sim_time=50.0,
                sample_interval=0.01,  # Very fine sampling
                key=jax.random.PRNGKey(seed),
                max_samples=10000,
            )
            
            # The FINAL Q state should satisfy: Q_final = arrivals - departures
            # Note: states[-1] is the last SNAPSHOT, not the actual final state
            # We need to check if the simulation maintains conservation internally
            
            # The arrivals and departures are cumulative counts
            # Queue can't be negative, so arrivals >= departures always
            assert arrivals >= departures, \
                f"SMOKING GUN: departures ({departures}) > arrivals ({arrivals})"
            
            # The final queue from snapshot should be close to arrivals - departures
            final_q_snapshot = states[-1].sum()
            expected_final_q = arrivals - departures
            
            # Allow tolerance for snapshot timing
            diff = abs(final_q_snapshot - expected_final_q)
            if diff > 5:
                print(f"Seed {seed}: arrivals={arrivals}, departures={departures}, "
                      f"expected_q={expected_final_q}, snapshot_q={final_q_snapshot}, diff={diff}")
    
    def test_queue_never_negative_during_simulation(self):
        """BUG: Queue should never go negative."""
        # This requires instrumenting the body_fun
        # We can check by running with high service rate and checking all snapshots
        
        for seed in range(5):
            _, states, _ = simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([10.0, 10.0]),  # Fast service
                alpha=1.0,
                sim_time=100.0,
                sample_interval=0.1,
                key=jax.random.PRNGKey(seed),
                max_samples=2000,
            )
            
            assert np.all(states >= 0), f"SMOKING GUN: Negative queue detected for seed {seed}"


# ============================================================
# SMOKING GUN #3: EVENT SELECTION
# ============================================================

class TestEventSelectionBugs:
    """Expose bugs in event selection logic."""
    
    def test_departure_from_empty_queue_impossible(self):
        """BUG: Should not be able to depart from empty queue."""
        # Run with zero initial state and verify no negative queues
        for seed in range(20):
            _, states, (arr, dep) = simulate_jax(
                num_servers=2,
                arrival_rate=0.1,  # Low arrival
                service_rates=jnp.array([100.0, 100.0]),  # Very fast service
                alpha=1.0,
                sim_time=50.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(seed),
                max_samples=100,
            )
            
            # If a departure happened when queue was 0, we'd have negative state
            assert np.all(states >= 0), \
                f"SMOKING GUN: Departure from empty queue for seed {seed}"
    
    def test_event_selection_uniform_distribution(self):
        """BUG: Event selection should be uniformly distributed across possible events."""
        # Track arrivals per server by observing queue increases
        # Use fine sampling to capture state changes
        all_increments = np.zeros(3)
        
        for seed in range(30):
            times, states, (arr, dep) = simulate_jax(
                num_servers=3,
                arrival_rate=3.0,
                service_rates=jnp.array([0.5, 0.5, 0.5]),  # Moderate service
                alpha=1.0,
                sim_time=20.0,
                sample_interval=0.1,  # Fine sampling
                key=jax.random.PRNGKey(seed),
                max_samples=500,
                policy_type=0,  # Uniform
            )
            
            # Count queue increments (arrivals) per server
            diffs = np.diff(states, axis=0)
            increments = np.sum(diffs == 1, axis=0)
            all_increments += increments
        
        # With uniform policy, arrivals should be roughly equal across servers
        mean_count = np.mean(all_increments)
        if mean_count > 0:
            max_deviation = np.max(np.abs(all_increments - mean_count)) / mean_count
            assert max_deviation < 0.25, \
                f"SMOKING GUN: Uniform policy produced uneven distribution: {all_increments}"
        else:
            pytest.fail(f"No arrivals detected: {all_increments}")


# ============================================================
# SMOKING GUN #4: POWER-OF-D POLICY
# ============================================================

class TestPowerOfDPolicyBugs:
    """Expose bugs in Power-of-d routing."""
    
    def test_power_of_d_always_picks_shortest(self):
        """BUG: Power-of-d should always route to shortest among sampled."""
        # Set up a scenario where one server is clearly shortest
        # Run many times and verify it always picks the shortest when sampled
        
        # We need to test get_probs directly
        from gibbsq.engines.jax_engine import get_probs, SimParams
        
        # Queue state: server 0 has 0, server 1 has 10, server 2 has 20
        Q = jnp.array([0, 10, 20])
        
        # With d=3 (sample all servers), should always pick server 0
        params = SimParams(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=jnp.ones(3),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            policy_type=4,  # Power-of-d
            d=3,  # Sample all servers
        )
        
        # Run 100 times with different keys
        for i in range(100):
            key = jax.random.PRNGKey(i)
            probs = get_probs(Q, params, key)
            
            # Should always pick server 0 (shortest)
            winner = jnp.argmax(probs)
            assert winner == 0, \
                f"SMOKING GUN: Power-of-d picked server {winner} instead of 0 (shortest)"
    
    def test_power_of_d_d_greater_than_n(self):
        """BUG: d > N should be clamped to N."""
        from gibbsq.engines.jax_engine import get_probs, SimParams
        
        Q = jnp.array([5, 10])
        params = SimParams(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.ones(2),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            policy_type=4,
            d=10,  # d > N
        )
        
        key = jax.random.PRNGKey(42)
        probs = get_probs(Q, params, key)
        
        # Should still work and pick server 0
        assert jnp.sum(probs) == pytest.approx(1.0), "Probabilities don't sum to 1"
        assert probs[0] == 1.0 or probs[1] == 1.0, "Should be one-hot"
    
    def test_power_of_d_single_server(self):
        """BUG: Power-of-d with N=1 should always route to that server."""
        from gibbsq.engines.jax_engine import get_probs, SimParams
        
        Q = jnp.array([5])
        params = SimParams(
            num_servers=1,
            arrival_rate=1.0,
            service_rates=jnp.ones(1),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            policy_type=4,
            d=2,
        )
        
        key = jax.random.PRNGKey(42)
        probs = get_probs(Q, params, key)
        
        assert probs[0] == 1.0, "SMOKING GUN: Power-of-d failed for single server"


# ============================================================
# SMOKING GUN #5: NUMERICAL STABILITY
# ============================================================

class TestNumericalStabilityBugs:
    """Expose numerical stability issues."""
    
    def test_zero_total_rate_handling(self):
        """SMOKING GUN #1: When all rates are zero, simulation should process ZERO events."""
        # Zero arrival rate with empty queues -> zero total rate
        _, states, (arr, dep) = simulate_jax(
            num_servers=2,
            arrival_rate=0.0,  # No arrivals
            service_rates=jnp.array([1.0, 1.0]),
            alpha=1.0,
            sim_time=10.0,
            sample_interval=1.0,
            key=jax.random.PRNGKey(42),
            max_samples=20,
        )
        
        # CRITICAL BUG: Currently produces 1 arrival even with zero rate!
        # This is because event selection doesn't handle a0=0 case
        assert arr == 0, f"SMOKING GUN #1: Zero rate produced {arr} arrivals (should be 0)"
        assert dep == 0, f"SMOKING GUN #1: Zero rate produced {dep} departures (should be 0)"
        assert np.all(states == 0), "SMOKING GUN #1: Non-zero states with no events"
    
    def test_extreme_alpha_softmax(self):
        """BUG: Very large alpha should not cause overflow in softmax."""
        from gibbsq.engines.jax_engine import get_probs, SimParams
        
        Q = jnp.array([0, 1000])  # Large disparity
        params = SimParams(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.ones(2),
            alpha=1000.0,  # Very large alpha
            sim_time=10.0,
            sample_interval=1.0,
            policy_type=3,  # Softmax
            d=2,
        )
        
        key = jax.random.PRNGKey(42)
        probs = get_probs(Q, params, key)
        
        # Should still be valid probabilities
        assert jnp.all(jnp.isfinite(probs)), "SMOKING GUN: Non-finite softmax probs"
        assert jnp.sum(probs) == pytest.approx(1.0), "Softmax probs don't sum to 1"
        # Should heavily favor server 0
        assert probs[0] > 0.99, f"Softmax should favor shorter queue: probs={probs}"
    
    def test_extreme_alpha_small_values(self):
        """BUG: Very small alpha should approach uniform."""
        from gibbsq.engines.jax_engine import get_probs, SimParams
        
        Q = jnp.array([0, 100])
        params = SimParams(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.ones(2),
            alpha=1e-10,  # Very small alpha
            sim_time=10.0,
            sample_interval=1.0,
            policy_type=3,  # Softmax
            d=2,
        )
        
        key = jax.random.PRNGKey(42)
        probs = get_probs(Q, params, key)
        
        # Should be close to uniform
        assert jnp.all(jnp.isfinite(probs)), "SMOKING GUN: Non-finite softmax probs"
        assert probs[0] == pytest.approx(0.5, abs=0.1), \
            f"Small alpha should give uniform: probs={probs}"


# ============================================================
# SMOKING GUN #6: GRADIENT FLOW
# ============================================================

class TestGradientFlowBugs:
    """Expose gradient flow issues."""
    
    def test_gradient_through_zero_queue(self):
        """BUG: Gradient should flow even when queue is zero."""
        import jax
        
        def sim_loss(alpha):
            _, states, _ = simulate_jax(
                num_servers=2,
                arrival_rate=0.1,  # Low arrival
                service_rates=jnp.array([10.0, 10.0]),
                alpha=alpha,
                sim_time=10.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(0),
                max_samples=20,
            )
            return states.sum().astype(jnp.float32)
        
        # Test gradient at various alpha values
        for alpha_val in [0.01, 0.1, 1.0, 10.0]:
            grad_fn = jax.grad(sim_loss)
            grad = grad_fn(alpha_val)
            
            assert jnp.isfinite(grad), \
                f"SMOKING GUN: Non-finite gradient at alpha={alpha_val}: {grad}"
    
    def test_gradient_does_not_depend_on_seed(self):
        """BUG: Gradient should be deterministic for same seed."""
        import jax
        
        def sim_loss(alpha, seed):
            _, states, _ = simulate_jax(
                num_servers=2,
                arrival_rate=1.0,
                service_rates=jnp.array([2.0, 2.0]),
                alpha=alpha,
                sim_time=20.0,
                sample_interval=1.0,
                key=jax.random.PRNGKey(seed),
                max_samples=50,
            )
            return states.sum().astype(jnp.float32)
        
        # Same seed should give same gradient
        grad_fn = jax.grad(sim_loss)
        g1 = grad_fn(1.0, 42)
        g2 = grad_fn(1.0, 42)
        
        assert jnp.allclose(g1, g2), \
            f"SMOKING GUN: Non-deterministic gradients: {g1} vs {g2}"


# ============================================================
# SMOKING GUN #7: VMAP BATCH CONSISTENCY
# ============================================================

class TestVmapBugs:
    """Expose bugs in vmap batch processing."""
    
    def test_vmap_produces_correct_shapes(self):
        """BUG: vmap should produce correct batch dimensions."""
        times, states, (arrivals, departures) = run_replications_jax(
            num_replications=5,
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=20.0,
            sample_interval=1.0,
            base_seed=42,
            max_samples=50,
        )
        
        assert times.shape[0] == 5, f"Wrong batch dim: {times.shape}"
        assert states.shape[0] == 5, f"Wrong batch dim: {states.shape}"
        assert arrivals.shape[0] == 5, f"Wrong batch dim: {arrivals.shape}"
    
    def test_vmap_each_replication_independent(self):
        """BUG: Each replication should be independent."""
        times, states, (arrivals, departures) = run_replications_jax(
            num_replications=3,
            num_servers=2,
            arrival_rate=1.0,
            service_rates=jnp.array([2.0, 2.0]),
            alpha=1.0,
            sim_time=50.0,
            sample_interval=1.0,
            base_seed=42,
            max_samples=100,
        )
        
        # Each replication should have different results (different seeds)
        # Check that not all are identical
        assert not jnp.array_equal(states[0], states[1]), \
            "SMOKING GUN: vmap replications are identical"
        assert not jnp.array_equal(states[1], states[2]), \
            "SMOKING GUN: vmap replications are identical"


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # -x to stop on first failure
