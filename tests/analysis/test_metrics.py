"""
Hardened test suite for gibbsq.analysis — Robustness Loop Stage 2.

Categories:
- A: Correctness Tests
- B: Invariant Tests
- C: Edge Case Tests
- D: Numerical Stability Tests
- E: Gradient Flow Tests (N/A)
- F: Regression Tests
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from gibbsq.analysis.metrics import (
    time_averaged_queue_lengths,
    total_queue_trajectory,
    running_average,
    queue_length_stats,
    gini_coefficient,
    sojourn_time_estimate,
    stationarity_diagnostic,
    mser5_truncation,
    gelman_rubin_diagnostic,
)
from gibbsq.engines.numpy_engine import SimResult


# ============================================================
# CATEGORY A: CORRECTNESS TESTS
# ============================================================

class TestTimeAveragedQueueLengths:
    """Verify time-averaged queue length computation."""
    
    def test_constant_trajectory(self):
        """Constant trajectory should return constant value."""
        result = SimResult(
            times=np.linspace(0, 10, 100),
            states=np.full((100, 2), 5.0),
            arrival_count=50,
            departure_count=50,
            final_time=10.0,
            num_servers=2,
        )
        
        avg = time_averaged_queue_lengths(result, burn_in_fraction=0.0)
        np.testing.assert_allclose(avg, [5.0, 5.0])
    
    def test_burn_in_discards_initial(self):
        """Burn-in should discard initial portion."""
        # First 50 samples: Q=10, Last 50 samples: Q=0
        states = np.vstack([np.full((50, 2), 10.0), np.zeros((50, 2))])
        result = SimResult(
            times=np.linspace(0, 10, 100),
            states=states,
            arrival_count=50,
            departure_count=50,
            final_time=10.0,
            num_servers=2,
        )
        
        # With 50% burn-in, should only see zeros
        avg = time_averaged_queue_lengths(result, burn_in_fraction=0.5)
        np.testing.assert_allclose(avg, [0.0, 0.0])
    
    def test_linear_growth(self):
        """Linearly growing trajectory should average correctly."""
        t = np.linspace(0, 10, 100)
        states = np.column_stack([t, t])  # Q grows linearly from 0 to 10
        result = SimResult(
            times=t,
            states=states,
            arrival_count=50,
            departure_count=50,
            final_time=10.0,
            num_servers=2,
        )
        
        avg = time_averaged_queue_lengths(result, burn_in_fraction=0.0)
        # Mean of [0, 10] is 5
        np.testing.assert_allclose(avg, [5.0, 5.0], rtol=0.1)


class TestTotalQueueTrajectory:
    """Verify total queue computation."""
    
    def test_sum(self):
        """Should sum across servers."""
        result = SimResult(
            times=np.array([0.0, 1.0, 2.0]),
            states=np.array([[1, 2], [3, 4], [5, 6]]),
            arrival_count=10,
            departure_count=10,
            final_time=2.0,
            num_servers=2,
        )
        
        t, q_tot = total_queue_trajectory(result)
        np.testing.assert_array_equal(q_tot, [3, 7, 11])
    
    def test_single_server(self):
        """Single server should return queue as-is."""
        result = SimResult(
            times=np.array([0.0, 1.0, 2.0]),
            states=np.array([[5], [10], [15]]),
            arrival_count=10,
            departure_count=10,
            final_time=2.0,
            num_servers=1,
        )
        
        t, q_tot = total_queue_trajectory(result)
        np.testing.assert_array_equal(q_tot, [5, 10, 15])


class TestRunningAverage:
    """Verify running average computation."""
    
    def test_convergence(self):
        """Running average should converge to steady state."""
        # Create trajectory that settles to 5
        states = np.column_stack([
            np.concatenate([np.linspace(0, 10, 50), np.full(50, 5.0)]),
            np.zeros(100)
        ])
        result = SimResult(
            times=np.linspace(0, 10, 100),
            states=states,
            arrival_count=50,
            departure_count=50,
            final_time=10.0,
            num_servers=2,
        )
        
        t, cum_avg = running_average(result)
        
        # Final value should be close to the mean
        assert abs(cum_avg[-1] - states[:, 0].mean()) < 1.0


class TestGiniCoefficient:
    """Verify Gini coefficient computation."""
    
    def test_perfect_equality(self):
        """Equal values should give Gini = 0."""
        gini = gini_coefficient(np.array([5, 5, 5, 5]))
        assert gini == pytest.approx(0.0, abs=1e-10)
    
    def test_maximum_inequality(self):
        """One non-zero value should approach Gini = 1."""
        gini = gini_coefficient(np.array([0, 0, 0, 100]))
        # Gini for [0,0,0,100] = (2*4*100)/(4*100) - 5/4 = 2 - 1.25 = 0.75
        assert gini == pytest.approx(0.75, abs=0.01)
    
    def test_all_zeros(self):
        """All zeros should give Gini = 0."""
        gini = gini_coefficient(np.array([0, 0, 0]))
        assert gini == 0.0
    
    def test_single_value(self):
        """Single value should give Gini = 0."""
        gini = gini_coefficient(np.array([42]))
        assert gini == 0.0
    
    def test_negative_raises(self):
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError):
            gini_coefficient(np.array([-1, 2, 3]))


class TestSojournTimeEstimate:
    """Verify Little's Law sojourn time estimate."""
    
    def test_littles_law(self):
        """E[W] = E[Q] / λ."""
        result = SimResult(
            times=np.linspace(0, 100, 1000),
            states=np.full((1000, 2), 5.0),  # E[Q] = 10
            arrival_count=500,
            departure_count=500,
            final_time=100.0,
            num_servers=2,
        )
        
        sojourn = sojourn_time_estimate(result, arrival_rate=2.0, burn_in_fraction=0.0)
        # E[W] = 10 / 2 = 5
        assert sojourn == pytest.approx(5.0, abs=0.1)


class TestStationarityDiagnostic:
    """Verify stationarity diagnostic."""
    
    def test_stationary_trajectory(self):
        """Stationary trajectory should pass."""
        rng = np.random.default_rng(42)
        # Constant mean with noise
        states = rng.normal(10, 1, (1000, 2))
        result = SimResult(
            times=np.linspace(0, 100, 1000),
            states=states,
            arrival_count=500,
            departure_count=500,
            final_time=100.0,
            num_servers=2,
        )
        
        diag = stationarity_diagnostic(result)
        assert diag["is_stationary"] is True
    
    def test_trending_trajectory_detected(self):
        """Trending trajectory should be detected."""
        # Linear growth
        t = np.linspace(0, 100, 1000)
        states = np.column_stack([t / 10, np.zeros(1000)])
        result = SimResult(
            times=t,
            states=states,
            arrival_count=500,
            departure_count=500,
            final_time=100.0,
            num_servers=2,
        )
        
        diag = stationarity_diagnostic(result)
        assert diag["is_stationary"] is False

    def test_negative_trend_is_not_marked_explosive(self):
        """Only positive significant slopes should fail the stationarity gate."""
        t = np.linspace(0, 100, 1000)
        states = np.column_stack([100 - (t / 10), np.zeros(1000)])
        result = SimResult(
            times=t,
            states=states,
            arrival_count=500,
            departure_count=500,
            final_time=100.0,
            num_servers=2,
        )

        diag = stationarity_diagnostic(result, burn_in_fraction=0.0)
        assert diag["slope"] < 0.0
        assert diag["p_value"] < 0.05
        assert diag["is_stationary"] is True


class TestMSER5Truncation:
    """Verify MSER-5 truncation point detection."""
    
    def test_immediate_convergence(self):
        """Already converged trajectory should truncate at 0."""
        traj = np.ones(100)
        trunc = mser5_truncation(traj, batch_size=5)
        # Variance of a constant is 0 everywhere, so argmin is usually 0
        assert trunc == 0
    
    def test_initialization_bias(self):
        """A sequence starting high and settling low should truncate after the drop."""
        # 20 samples of 100, then 80 samples of 1
        traj = np.concatenate([np.full(20, 100.0), np.full(80, 1.0)])
        # Add tiny noise to avoid exact zeros in variance calculation
        # Use seeded RNG for reproducibility
        rng = np.random.default_rng(42)
        traj += rng.normal(0, 0.01, size=100)
        trunc = mser5_truncation(traj, batch_size=5)
        # Should truncate at or slightly after index 20 (batch 4)
        # Allow wider range due to noise influence
        assert trunc >= 10 and trunc <= 30
    
    def test_short_trajectory(self):
        """Short trajectory should return 0."""
        traj = np.array([1, 2, 3])
        trunc = mser5_truncation(traj, batch_size=5)
        assert trunc == 0


class TestGelmanRubinDiagnostic:
    """Verify Gelman-Rubin R-hat computation."""
    
    def test_identical_chains(self):
        """Perfectly identical constant chains -> R-hat = 1.0"""
        chains = np.full((3, 50), 10.0)
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat == pytest.approx(1.0)
    
    def test_converged_chains(self):
        """Chains converged to same distribution should have R-hat ≈ 1."""
        rng = np.random.default_rng(42)
        # Three chains from same distribution
        chains = rng.normal(10, 1, (3, 1000))
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat < 1.1  # Standard threshold
    
    def test_diverged_chains(self):
        """Diverged chains should have R-hat > 1."""
        # Three chains with different means
        chains = np.array([
            np.full(100, 0.0),
            np.full(100, 10.0),
            np.full(100, 20.0),
        ])
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat > 1.1
    
    def test_single_chain(self):
        """Single chain should return 1.0."""
        chains = np.ones((1, 100))
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat == 1.0


# ============================================================
# CATEGORY B: INVARIANT TESTS
# ============================================================

class TestMetricsInvariants:
    """Invariants that must hold for all metrics."""
    
    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=0.0, max_value=100.0, allow_infinity=False, allow_nan=False),
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_gini_in_bounds(self, values):
        """Gini coefficient must be in [0, 1) (with tolerance for FP errors)."""
        gini = gini_coefficient(values)
        # Allow small negative values due to floating-point precision
        assert -1e-10 <= gini < 1.0
    
    @given(
        trajectories=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=2, max_value=10),
                st.integers(min_value=10, max_value=100)
            ),
            elements=st.floats(min_value=0.0, max_value=100.0, allow_infinity=False, allow_nan=False),
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_rhat_positive(self, trajectories):
        """R-hat must be positive."""
        r_hat = gelman_rubin_diagnostic(trajectories)
        assert r_hat > 0.0


# ============================================================
# CATEGORY C: EDGE CASE TESTS
# ============================================================

class TestMetricsEdgeCases:
    """Test boundary conditions."""
    
    def test_empty_trajectory(self):
        """Empty trajectory should handle gracefully."""
        # Create minimal valid result
        result = SimResult(
            times=np.array([0.0]),
            states=np.array([[0, 0]]),
            arrival_count=0,
            departure_count=0,
            final_time=0.0,
            num_servers=2,
        )
        
        avg = time_averaged_queue_lengths(result, burn_in_fraction=0.0)
        np.testing.assert_array_equal(avg, [0.0, 0.0])
    
    def test_full_burn_in(self):
        """Burn-in fraction close to 1 should handle gracefully."""
        result = SimResult(
            times=np.linspace(0, 10, 100),
            states=np.full((100, 2), 5.0),
            arrival_count=50,
            departure_count=50,
            final_time=10.0,
            num_servers=2,
        )
        
        # 99% burn-in should leave at least 1 sample
        avg = time_averaged_queue_lengths(result, burn_in_fraction=0.99)
        assert len(avg) == 2
    
    def test_mser_batch_size_larger_than_trajectory(self):
        """Batch size larger than trajectory should return 0."""
        traj = np.array([1, 2, 3])
        trunc = mser5_truncation(traj, batch_size=10)
        assert trunc == 0


# ============================================================
# CATEGORY D: NUMERICAL STABILITY TESTS
# ============================================================

class TestMetricsNumericalStability:
    """Test behavior under numerically challenging inputs."""
    
    def test_gini_very_large_values(self):
        """Very large values should not cause overflow."""
        gini = gini_coefficient(np.array([1e10, 2e10, 3e10]))
        assert np.isfinite(gini)
        assert 0.0 <= gini < 1.0
    
    def test_gini_very_small_values(self):
        """Very small values should work correctly."""
        gini = gini_coefficient(np.array([1e-10, 2e-10, 3e-10]))
        assert np.isfinite(gini)
        assert 0.0 <= gini < 1.0
    
    def test_rhat_very_large_values(self):
        """Very large values should not cause overflow."""
        # Use chains with same mean but large values
        chains = np.full((3, 100), 1e10)
        r_hat = gelman_rubin_diagnostic(chains)
        assert np.isfinite(r_hat)
        assert r_hat == pytest.approx(1.0, abs=0.01)
    
    def test_mser_constant_trajectory(self):
        """Constant trajectory should not cause division by zero."""
        traj = np.full(100, 42.0)
        trunc = mser5_truncation(traj, batch_size=5)
        assert np.isfinite(trunc)


# ============================================================
# CATEGORY F: REGRESSION TESTS
# ============================================================

class TestMetricsRegressions:
    """Prevent reintroduction of known faults."""
    
    def test_regression_gini_formula(self):
        """Verify Gini formula is correct."""
        # Known values with known Gini
        # Perfect equality: [1,1,1,1] -> Gini = 0
        assert gini_coefficient(np.array([1, 1, 1, 1])) == pytest.approx(0.0, abs=1e-10)
        
        # [1,2,3,4] -> Gini = 0.1
        # Formula: (2*(1*1 + 2*2 + 3*3 + 4*4))/(4*10) - 5/4
        # = (2*30)/40 - 1.25 = 1.5 - 1.25 = 0.25
        assert gini_coefficient(np.array([1, 2, 3, 4])) == pytest.approx(0.25, abs=0.01)
    
    def test_regression_rhat_identical_chains(self):
        """Identical chains should give exactly 1.0."""
        chains = np.full((5, 100), 42.0)
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat == pytest.approx(1.0, abs=1e-10)


# ============================================================
# QUEUE LENGTH STATS TESTS
# ============================================================

class TestQueueLengthStats:
    """Verify queue length statistics."""
    
    def test_basic_stats(self):
        """Should compute correct statistics."""
        result = SimResult(
            times=np.linspace(0, 10, 100),
            states=np.column_stack([np.arange(100), np.zeros(100)]),
            arrival_count=50,
            departure_count=50,
            final_time=10.0,
            num_servers=2,
        )
        
        stats = queue_length_stats(result, burn_in_fraction=0.0)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p99" in stats
        
        assert stats["min"] == 0.0
        assert stats["max"] == 99.0  # Sum of queues: 0+0, 1+0, ..., 99+0
