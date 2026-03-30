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

class TestTimeAveragedQueueLengths:
    def test_constant_trajectory(self):
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
        states = np.vstack([np.full((50, 2), 10.0), np.zeros((50, 2))])
        result = SimResult(
            times=np.linspace(0, 10, 100),
            states=states,
            arrival_count=50,
            departure_count=50,
            final_time=10.0,
            num_servers=2,
        )
        
        avg = time_averaged_queue_lengths(result, burn_in_fraction=0.5)
        np.testing.assert_allclose(avg, [0.0, 0.0])
    
    def test_linear_growth(self):
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
        np.testing.assert_allclose(avg, [5.0, 5.0], rtol=0.1)

class TestTotalQueueTrajectory:
    def test_sum(self):
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
    def test_convergence(self):
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
        
        assert abs(cum_avg[-1] - states[:, 0].mean()) < 1.0

class TestGiniCoefficient:
    def test_perfect_equality(self):
        gini = gini_coefficient(np.array([5, 5, 5, 5]))
        assert gini == pytest.approx(0.0, abs=1e-10)
    
    def test_maximum_inequality(self):
        gini = gini_coefficient(np.array([0, 0, 0, 100]))
        # Gini for [0,0,0,100] = (2*4*100)/(4*100) - 5/4 = 2 - 1.25 = 0.75
        assert gini == pytest.approx(0.75, abs=0.01)
    
    def test_all_zeros(self):
        gini = gini_coefficient(np.array([0, 0, 0]))
        assert gini == 0.0
    
    def test_single_value(self):
        gini = gini_coefficient(np.array([42]))
        assert gini == 0.0
    
    def test_negative_raises(self):
        with pytest.raises(ValueError):
            gini_coefficient(np.array([-1, 2, 3]))

class TestSojournTimeEstimate:
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
        assert sojourn == pytest.approx(5.0, abs=0.1)

class TestStationarityDiagnostic:
    def test_stationary_trajectory(self):
        rng = np.random.default_rng(42)
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
    def test_immediate_convergence(self):
        traj = np.ones(100)
        trunc = mser5_truncation(traj, batch_size=5)
        # Variance of a constant is 0 everywhere, so argmin is usually 0
        assert trunc == 0
    
    def test_initialization_bias(self):
        # 20 samples of 100, then 80 samples of 1
        traj = np.concatenate([np.full(20, 100.0), np.full(80, 1.0)])
        # Add tiny noise to avoid exact zeros in variance calculation
        rng = np.random.default_rng(42)
        traj += rng.normal(0, 0.01, size=100)
        trunc = mser5_truncation(traj, batch_size=5)
        assert trunc >= 10 and trunc <= 30
    
    def test_short_trajectory(self):
        traj = np.array([1, 2, 3])
        trunc = mser5_truncation(traj, batch_size=5)
        assert trunc == 0

class TestGelmanRubinDiagnostic:
    def test_identical_chains(self):
        chains = np.full((3, 50), 10.0)
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat == pytest.approx(1.0)
    
    def test_converged_chains(self):
        rng = np.random.default_rng(42)
        chains = rng.normal(10, 1, (3, 1000))
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat < 1.1  # Standard threshold
    
    def test_diverged_chains(self):
        chains = np.array([
            np.full(100, 0.0),
            np.full(100, 10.0),
            np.full(100, 20.0),
        ])
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat > 1.1
    
    def test_single_chain(self):
        chains = np.ones((1, 100))
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat == 1.0

class TestMetricsInvariants:
    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=0.0, max_value=100.0, allow_infinity=False, allow_nan=False),
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_gini_in_bounds(self, values):
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
        r_hat = gelman_rubin_diagnostic(trajectories)
        assert r_hat > 0.0

class TestMetricsEdgeCases:
    def test_empty_trajectory(self):
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
        traj = np.array([1, 2, 3])
        trunc = mser5_truncation(traj, batch_size=10)
        assert trunc == 0

class TestMetricsNumericalStability:
    def test_gini_very_large_values(self):
        gini = gini_coefficient(np.array([1e10, 2e10, 3e10]))
        assert np.isfinite(gini)
        assert 0.0 <= gini < 1.0
    
    def test_gini_very_small_values(self):
        gini = gini_coefficient(np.array([1e-10, 2e-10, 3e-10]))
        assert np.isfinite(gini)
        assert 0.0 <= gini < 1.0
    
    def test_rhat_very_large_values(self):
        chains = np.full((3, 100), 1e10)
        r_hat = gelman_rubin_diagnostic(chains)
        assert np.isfinite(r_hat)
        assert r_hat == pytest.approx(1.0, abs=0.01)
    
    def test_mser_constant_trajectory(self):
        traj = np.full(100, 42.0)
        trunc = mser5_truncation(traj, batch_size=5)
        assert np.isfinite(trunc)

class TestMetricsRegressions:
    def test_regression_gini_formula(self):
        # Perfect equality: [1,1,1,1] -> Gini = 0
        assert gini_coefficient(np.array([1, 1, 1, 1])) == pytest.approx(0.0, abs=1e-10)
        
        # [1,2,3,4] -> Gini = 0.1
        # Formula: (2*(1*1 + 2*2 + 3*3 + 4*4))/(4*10) - 5/4
        # = (2*30)/40 - 1.25 = 1.5 - 1.25 = 0.25
        assert gini_coefficient(np.array([1, 2, 3, 4])) == pytest.approx(0.25, abs=0.01)
    
    def test_regression_rhat_identical_chains(self):
        chains = np.full((5, 100), 42.0)
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat == pytest.approx(1.0, abs=1e-10)

class TestQueueLengthStats:
    def test_basic_stats(self):
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
