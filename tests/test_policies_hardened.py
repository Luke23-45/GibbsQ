"""
Hardened test suite for moeq.core.policies — Robustness Loop Stage 2.

Categories:
- A: Correctness Tests
- B: Invariant Tests
- C: Edge Case Tests
- D: Numerical Stability Tests
- E: Gradient Flow Tests (N/A — policies are not differentiable in NumPy)
- F: Regression Tests
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from moeq.core.policies import (
    SoftmaxRouting, UniformRouting, ProportionalRouting,
    JSQRouting, PowerOfDRouting, make_policy, RoutingPolicy,
)


# ============================================================
# CATEGORY A: CORRECTNESS TESTS
# ============================================================

class TestSoftmaxCorrectness:
    """Verify softmax routing produces mathematically correct outputs."""
    
    def test_softmax_uniform_for_equal_queues(self):
        """When all Q_i equal, p_i = 1/N."""
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([5.0, 5.0, 5.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])
    
    def test_softmax_prefers_shorter_queue(self):
        """Lower Q_i should get higher probability."""
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 10.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] > probs[1]
        assert probs[0] > 0.99  # Should be almost 1
    
    def test_softmax_boltzmann_distribution(self):
        """Verify p_i = exp(-α Q_i) / Σ exp(-α Q_j)."""
        policy = SoftmaxRouting(alpha=2.0)
        Q = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        
        # Manual calculation
        logits = -2.0 * np.array([1.0, 2.0, 3.0])
        logits -= logits.max()
        expected = np.exp(logits) / np.exp(logits).sum()
        
        np.testing.assert_allclose(probs, expected, rtol=1e-10)
    
    def test_softmax_sums_to_one(self):
        """Output must be a valid probability distribution."""
        policy = SoftmaxRouting(alpha=0.5)
        Q = np.array([1.0, 5.0, 10.0, 20.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert abs(probs.sum() - 1.0) < 1e-10


class TestUniformCorrectness:
    """Verify uniform routing."""
    
    def test_uniform_always_equal(self):
        """p_i = 1/N regardless of state."""
        policy = UniformRouting()
        rng = np.random.default_rng(42)
        
        for Q in [np.array([0, 0]), np.array([100, 0]), np.array([5, 5, 5, 5])]:
            probs = policy(Q, rng)
            expected = np.ones(len(Q)) / len(Q)
            np.testing.assert_allclose(probs, expected)


class TestProportionalCorrectness:
    """Verify proportional-to-capacity routing."""
    
    def test_proportional_matches_service_rates(self):
        """p_i = μ_i / Λ."""
        policy = ProportionalRouting(mu=np.array([1.0, 3.0, 6.0]))
        rng = np.random.default_rng(42)
        probs = policy(np.array([10, 10, 10]), rng)
        
        expected = np.array([1.0, 3.0, 6.0]) / 10.0
        np.testing.assert_allclose(probs, expected)
    
    def test_proportional_sums_to_one(self):
        """Output must be valid probability."""
        policy = ProportionalRouting(mu=np.array([0.5, 1.5, 2.0, 1.0]))
        rng = np.random.default_rng(42)
        probs = policy(np.zeros(4), rng)
        assert abs(probs.sum() - 1.0) < 1e-10


class TestJSQCorrectness:
    """Verify Join-Shortest-Queue routing."""
    
    def test_jsq_single_minimum(self):
        """When one server has minimum, route entirely there."""
        policy = JSQRouting()
        Q = np.array([5, 1, 10])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(probs, expected)
    
    def test_jsq_tie_breaking(self):
        """When k servers tie for minimum, each gets 1/k."""
        policy = JSQRouting()
        Q = np.array([3, 1, 4, 1, 5])  # Servers 1 and 3 tie at 1
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        expected = np.array([0.0, 0.5, 0.0, 0.5, 0.0])
        np.testing.assert_allclose(probs, expected)
    
    def test_jsq_all_equal(self):
        """When all queues equal, uniform distribution."""
        policy = JSQRouting()
        Q = np.array([2, 2, 2, 2])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25])


class TestPowerOfDCorrectness:
    """Verify Power-of-d-choices routing."""
    
    def test_power_of_d_samples_d_servers(self):
        """Should sample exactly d servers."""
        policy = PowerOfDRouting(d=2)
        Q = np.array([10, 0, 10, 10])
        rng = np.random.default_rng(42)
        
        probs = policy(Q, rng)
        # Exactly one server should have probability 1
        assert np.sum(probs == 1.0) == 1
        assert np.sum(probs == 0.0) == len(Q) - 1
    
    def test_power_of_d_clamps_to_n(self):
        """If d > N, should use N."""
        policy = PowerOfDRouting(d=10)
        Q = np.array([1, 2])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        # Should still work, sampling all servers
        assert probs.sum() == 1.0


# ============================================================
# CATEGORY B: INVARIANT TESTS
# ============================================================

class TestPolicyInvariants:
    """Invariants that must hold for all policies."""
    
    @given(
        alpha=st.floats(min_value=0.001, max_value=100.0, allow_infinity=False, allow_nan=False),
        Q=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=20),
            elements=st.floats(min_value=0.0, max_value=1000.0, allow_infinity=False, allow_nan=False),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_softmax_probs_sum_to_one(self, alpha, Q):
        """Softmax output must always sum to 1."""
        policy = SoftmaxRouting(alpha=alpha)
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert abs(float(probs.sum()) - 1.0) < 1e-9
    
    @given(
        mu=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=20),
            elements=st.floats(min_value=0.001, max_value=100.0, allow_infinity=False, allow_nan=False),
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_proportional_probs_sum_to_one(self, mu):
        """Proportional output must always sum to 1."""
        policy = ProportionalRouting(mu=mu)
        rng = np.random.default_rng(42)
        probs = policy(np.zeros(len(mu)), rng)
        assert abs(float(probs.sum()) - 1.0) < 1e-9
    
    @given(
        Q=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=20),
            elements=st.floats(min_value=0.0, max_value=1000.0, allow_infinity=False, allow_nan=False),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_jsq_probs_sum_to_one(self, Q):
        """JSQ output must always sum to 1."""
        policy = JSQRouting()
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert abs(float(probs.sum()) - 1.0) < 1e-9
    
    @given(
        d=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50, deadline=None)
    def test_power_of_d_probs_sum_to_one(self, d, N):
        """Power-of-d output must always sum to 1."""
        assume(d >= 1 and N >= 1)
        policy = PowerOfDRouting(d=d)
        rng = np.random.default_rng(42)
        Q = rng.integers(0, 100, size=N).astype(np.float64)
        probs = policy(Q, rng)
        assert abs(float(probs.sum()) - 1.0) < 1e-9
    
    @given(
        alpha=st.floats(min_value=0.001, max_value=100.0, allow_infinity=False, allow_nan=False),
        Q=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=20),
            elements=st.floats(min_value=0.0, max_value=1000.0, allow_infinity=False, allow_nan=False),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_softmax_probs_non_negative(self, alpha, Q):
        """All probabilities must be >= 0."""
        policy = SoftmaxRouting(alpha=alpha)
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert (probs >= 0.0).all()


# ============================================================
# CATEGORY C: EDGE CASE TESTS
# ============================================================

class TestPolicyEdgeCases:
    """Test boundary conditions and degenerate inputs."""
    
    def test_softmax_single_server(self):
        """N=1 should always return [1.0]."""
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([100.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1.0])
    
    def test_softmax_zero_queues(self):
        """All zeros should give uniform."""
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])
    
    def test_softmax_large_queue_difference(self):
        """Large differences should not cause overflow."""
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 1e6])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] > 0.999999
        assert probs[1] < 1e-6
        assert np.isfinite(probs).all()
    
    def test_jsq_single_server(self):
        """N=1 should always return [1.0]."""
        policy = JSQRouting()
        Q = np.array([50])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1.0])
    
    def test_jsq_zero_queues(self):
        """All zeros = all minimum = uniform."""
        policy = JSQRouting()
        Q = np.array([0, 0, 0, 0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25])
    
    def test_power_of_d_d_equals_n(self):
        """d=N means all servers sampled, equivalent to JSQ."""
        policy = PowerOfDRouting(d=3)
        Q = np.array([5, 1, 10])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        # Server 1 (index 1) has minimum, should always win
        assert probs[1] == 1.0
    
    def test_power_of_d_d_equals_1(self):
        """d=1 means random server chosen uniformly."""
        policy = PowerOfDRouting(d=1)
        Q = np.array([10, 10, 10])  # All equal, so any choice is valid
        rng = np.random.default_rng(42)
        
        # Run multiple times, should get different servers
        winners = set()
        for _ in range(100):
            probs = policy(Q, rng)
            winners.add(np.argmax(probs))
        
        # Should have sampled all 3 servers over 100 runs
        assert len(winners) == 3
    
    def test_proportional_single_server(self):
        """N=1 should always return [1.0]."""
        policy = ProportionalRouting(mu=np.array([5.0]))
        rng = np.random.default_rng(42)
        probs = policy(np.array([100]), rng)
        np.testing.assert_allclose(probs, [1.0])
    
    def test_invalid_alpha_zero(self):
        """alpha=0 must raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            SoftmaxRouting(alpha=0.0)
    
    def test_invalid_alpha_negative(self):
        """alpha < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            SoftmaxRouting(alpha=-1.0)
    
    def test_invalid_proportional_zero_mu(self):
        """mu=0 must raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            ProportionalRouting(mu=np.array([1.0, 0.0]))
    
    def test_invalid_proportional_negative_mu(self):
        """mu < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            ProportionalRouting(mu=np.array([1.0, -0.5]))
    
    def test_invalid_power_of_d_zero(self):
        """d=0 must raise ValueError."""
        with pytest.raises(ValueError, match="d must be ≥ 1"):
            PowerOfDRouting(d=0)
    
    def test_invalid_power_of_d_negative(self):
        """d < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="d must be ≥ 1"):
            PowerOfDRouting(d=-5)


# ============================================================
# CATEGORY D: NUMERICAL STABILITY TESTS
# ============================================================

class TestPolicyNumericalStability:
    """Test behavior under numerically challenging inputs."""
    
    def test_softmax_very_large_queues(self):
        """Very large Q values should not overflow."""
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([1e10, 1e10, 1e10])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])
        assert np.isfinite(probs).all()
    
    def test_softmax_very_small_alpha(self):
        """Very small alpha → near-uniform (high temperature)."""
        policy = SoftmaxRouting(alpha=1e-10)
        Q = np.array([0.0, 1000.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        # Should be nearly uniform
        assert abs(probs[0] - 0.5) < 0.01
        assert np.isfinite(probs).all()
    
    def test_softmax_very_large_alpha(self):
        """Very large alpha → near-JSQ (low temperature)."""
        policy = SoftmaxRouting(alpha=1e6)
        Q = np.array([0.0, 1.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] > 0.999
        assert np.isfinite(probs).all()
    
    def test_softmax_extreme_queue_difference(self):
        """Extreme difference should not cause underflow issues."""
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 1e15])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] == 1.0  # Numerically exact
        assert probs[1] == 0.0
        assert np.isfinite(probs).all()
    
    def test_jsq_very_large_queues(self):
        """Very large queues should work correctly."""
        policy = JSQRouting()
        Q = np.array([1e15, 1e15 - 1, 1e15])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        # Server 1 has minimum
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(probs, expected)


# ============================================================
# CATEGORY F: REGRESSION TESTS
# ============================================================

class TestPolicyRegressions:
    """Prevent reintroduction of known faults."""
    
    def test_regression_softmax_log_sum_exp_trick(self):
        """Ensure log-sum-exp trick is applied (shift by max)."""
        # Without the shift, exp(-1000) would underflow to 0
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([1000.0, 1000.0, 1000.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])
    
    def test_regression_softmax_mixed_extremes(self):
        """Mix of very large and very small values."""
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 1000.0, 2000.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] > 0.999
        assert probs[1] < 1e-10
        assert probs[2] < 1e-100
        assert np.isfinite(probs).all()


# ============================================================
# FACTORY TESTS
# ============================================================

class TestMakePolicy:
    """Test the policy factory function."""
    
    def test_make_softmax(self):
        p = make_policy("softmax", alpha=2.5)
        assert isinstance(p, SoftmaxRouting)
        assert p.alpha == 2.5
    
    def test_make_uniform(self):
        p = make_policy("uniform")
        assert isinstance(p, UniformRouting)
    
    def test_make_proportional(self):
        p = make_policy("proportional", mu=np.array([1.0, 2.0, 3.0]))
        assert isinstance(p, ProportionalRouting)
    
    def test_make_jsq(self):
        p = make_policy("jsq")
        assert isinstance(p, JSQRouting)
    
    def test_make_power_of_d(self):
        p = make_policy("power_of_d", d=3)
        assert isinstance(p, PowerOfDRouting)
    
    def test_make_proportional_missing_mu(self):
        with pytest.raises(ValueError, match="requires 'mu'"):
            make_policy("proportional")
    
    def test_make_unknown_policy(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            make_policy("nonexistent")


# ============================================================
# PROTOCOL TESTS
# ============================================================

class TestRoutingPolicyProtocol:
    """Verify all policies conform to the RoutingPolicy protocol."""
    
    def test_softmax_is_routing_policy(self):
        assert isinstance(SoftmaxRouting(alpha=1.0), RoutingPolicy)
    
    def test_uniform_is_routing_policy(self):
        assert isinstance(UniformRouting(), RoutingPolicy)
    
    def test_proportional_is_routing_policy(self):
        assert isinstance(ProportionalRouting(mu=np.array([1.0, 2.0])), RoutingPolicy)
    
    def test_jsq_is_routing_policy(self):
        assert isinstance(JSQRouting(), RoutingPolicy)
    
    def test_power_of_d_is_routing_policy(self):
        assert isinstance(PowerOfDRouting(d=2), RoutingPolicy)
    
    def test_all_policies_callable(self):
        """All policies must be callable with (Q, rng) signature."""
        policies = [
            SoftmaxRouting(alpha=1.0),
            UniformRouting(),
            ProportionalRouting(mu=np.array([1.0, 2.0])),
            JSQRouting(),
            PowerOfDRouting(d=2),
        ]
        Q = np.array([1.0, 2.0])
        rng = np.random.default_rng(42)
        
        for policy in policies:
            probs = policy(Q, rng)
            assert isinstance(probs, np.ndarray)
            assert probs.shape == Q.shape
            assert abs(probs.sum() - 1.0) < 1e-10
