"""Tests for src/metrics.py"""

import numpy as np
import pytest
from moeq.engines.numpy_engine import SimResult
from moeq.analysis.metrics import (
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


def _make_result(times, states):
    """Helper to create a SimResult from arrays."""
    return SimResult(
        times=np.array(times, dtype=np.float64),
        states=np.array(states, dtype=np.int64),
        arrival_count=0,
        departure_count=0,
        final_time=times[-1] if len(times) > 0 else 0.0,
        num_servers=np.array(states).shape[1] if len(states) > 0 else 0,
    )


class TestTimeAveragedQueueLengths:
    def test_constant_trajectory(self):
        """Constant queue lengths → average equals the constant."""
        states = np.full((100, 3), 5, dtype=np.int64)
        times = np.arange(100, dtype=np.float64)
        result = _make_result(times, states)
        avg = time_averaged_queue_lengths(result, burn_in_fraction=0.0)
        np.testing.assert_allclose(avg, [5.0, 5.0, 5.0])

    def test_burn_in_discards_initial(self):
        """First 20% of trajectory should be ignored."""
        N = 100
        states = np.zeros((N, 2), dtype=np.int64)
        # First 20 samples have Q = [100, 100], rest have Q = [1, 1]
        states[:20] = 100
        states[20:] = 1
        times = np.arange(N, dtype=np.float64)
        result = _make_result(times, states)
        avg = time_averaged_queue_lengths(result, burn_in_fraction=0.2)
        np.testing.assert_allclose(avg, [1.0, 1.0])


class TestTotalQueueTrajectory:
    def test_sum(self):
        states = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        times = np.array([0.0, 1.0, 2.0])
        result = _make_result(times, states)
        t, q = total_queue_trajectory(result)
        np.testing.assert_array_equal(q, [3, 7, 11])


class TestRunningAverage:
    def test_convergence(self):
        """Running average of constant should be that constant."""
        states = np.full((200, 2), 3, dtype=np.int64)
        times = np.arange(200, dtype=np.float64)
        result = _make_result(times, states)
        t, cumavg = running_average(result)
        np.testing.assert_allclose(cumavg, 6.0)  # 3+3=6 total


class TestGiniCoefficient:
    def test_perfect_equality(self):
        assert gini_coefficient(np.array([5.0, 5.0, 5.0, 5.0])) == pytest.approx(0.0)

    def test_maximum_inequality(self):
        """One nonzero value among N → Gini = (N-1)/N."""
        Q = np.array([0.0, 0.0, 0.0, 10.0])
        expected = 3.0 / 4.0
        assert gini_coefficient(Q) == pytest.approx(expected)

    def test_all_zeros(self):
        assert gini_coefficient(np.array([0.0, 0.0, 0.0])) == 0.0


class TestSojournTimeEstimate:
    def test_littles_law(self):
        """E[W] = E[Q_total] / λ."""
        states = np.full((100, 2), 4, dtype=np.int64)  # Q_total = 8
        times = np.arange(100, dtype=np.float64)
        result = _make_result(times, states)
        W = sojourn_time_estimate(result, arrival_rate=2.0, burn_in_fraction=0.0)
        assert W == pytest.approx(4.0)  # 8 / 2


class TestStationarityDiagnostic:
    def test_stationary_trajectory(self):
        """Constant trajectory should pass stationarity test."""
        rng = np.random.default_rng(42)
        # Add small noise to make linregress work (constant gives degenerate fit)
        noise = rng.normal(0, 0.01, size=(1000, 2))
        states = (np.full((1000, 2), 5) + noise).astype(np.int64)
        times = np.arange(1000, dtype=np.float64)
        result = _make_result(times, states)
        diag = stationarity_diagnostic(result, num_windows=10, burn_in_fraction=0.0)
        assert diag["is_stationary"]

    def test_trending_trajectory_detected(self):
        """Linearly increasing trajectory should fail stationarity."""
        states = np.column_stack([
            np.arange(10000),
            np.arange(10000),
        ]).astype(np.int64)
        times = np.arange(10000, dtype=np.float64)
        result = _make_result(times, states)
        diag = stationarity_diagnostic(result, num_windows=10, burn_in_fraction=0.0)
        assert not diag["is_stationary"]


class TestMSER5Truncation:
    def test_mser_immediate_convergence(self):
        """Constant sequence should ideally truncate at 0, or close to it."""
        traj = np.ones(100)
        trunc = mser5_truncation(traj, batch_size=5)
        # Variance of a constant is 0 everywhere, so argmin is usually 0
        assert trunc == 0

    def test_mser_initialization_bias(self):
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


class TestGelmanRubinDiagnostic:
    def test_identical_chains(self):
        """Perfectly identical constant chains -> R-hat = 1.0"""
        chains = np.full((3, 50), 10.0)
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat == pytest.approx(1.0)

    def test_converged_chains(self):
        """Chains from the same distribution -> R-hat close to 1.0"""
        rng = np.random.default_rng(42)
        # 5 chains, 1000 length, all Normal(5, 1)
        chains = rng.normal(5, 1, size=(5, 1000))
        r_hat = gelman_rubin_diagnostic(chains)
        # Should be very close to 1.0, safely under 1.1
        assert r_hat < 1.1

    def test_diverged_chains(self):
        """Chains from different distributions -> R-hat >> 1.0"""
        rng = np.random.default_rng(42)
        chain1 = rng.normal(0, 1, size=100)
        chain2 = rng.normal(10, 1, size=100)
        chain3 = rng.normal(50, 1, size=100)
        chains = np.vstack([chain1, chain2, chain3])
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat > 2.0
