import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from gibbsq.core.policies import (
    SoftmaxRouting, UniformRouting, ProportionalRouting,
    JSQRouting, PowerOfDRouting, make_policy, RoutingPolicy,
)

class TestSoftmaxCorrectness:
    def test_softmax_uniform_for_equal_queues(self):
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([5.0, 5.0, 5.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])
    
    def test_softmax_prefers_shorter_queue(self):
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 10.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] > probs[1]
        assert probs[0] > 0.99  # Should be almost 1
    
    def test_softmax_boltzmann_distribution(self):
        policy = SoftmaxRouting(alpha=2.0)
        Q = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        
        logits = -2.0 * np.array([1.0, 2.0, 3.0])
        logits -= logits.max()
        expected = np.exp(logits) / np.exp(logits).sum()
        
        np.testing.assert_allclose(probs, expected, rtol=1e-10)
    
    def test_softmax_sums_to_one(self):
        policy = SoftmaxRouting(alpha=0.5)
        Q = np.array([1.0, 5.0, 10.0, 20.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert abs(probs.sum() - 1.0) < 1e-10

class TestUniformCorrectness:
    def test_uniform_always_equal(self):
        policy = UniformRouting()
        rng = np.random.default_rng(42)
        
        for Q in [np.array([0, 0]), np.array([100, 0]), np.array([5, 5, 5, 5])]:
            probs = policy(Q, rng)
            expected = np.ones(len(Q)) / len(Q)
            np.testing.assert_allclose(probs, expected)

class TestProportionalCorrectness:
    def test_proportional_matches_service_rates(self):
        policy = ProportionalRouting(mu=np.array([1.0, 3.0, 6.0]))
        rng = np.random.default_rng(42)
        probs = policy(np.array([10, 10, 10]), rng)
        
        expected = np.array([1.0, 3.0, 6.0]) / 10.0
        np.testing.assert_allclose(probs, expected)
    
    def test_proportional_sums_to_one(self):
        policy = ProportionalRouting(mu=np.array([0.5, 1.5, 2.0, 1.0]))
        rng = np.random.default_rng(42)
        probs = policy(np.zeros(4), rng)
        assert abs(probs.sum() - 1.0) < 1e-10

class TestJSQCorrectness:
    def test_jsq_single_minimum(self):
        policy = JSQRouting()
        Q = np.array([5, 1, 10])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(probs, expected)
    
    def test_jsq_tie_breaking(self):
        policy = JSQRouting()
        Q = np.array([3, 1, 4, 1, 5])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        expected = np.array([0.0, 0.5, 0.0, 0.5, 0.0])
        np.testing.assert_allclose(probs, expected)
    
    def test_jsq_all_equal(self):
        policy = JSQRouting()
        Q = np.array([2, 2, 2, 2])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25])

class TestPowerOfDCorrectness:
    def test_power_of_d_samples_d_servers(self):
        policy = PowerOfDRouting(d=2)
        Q = np.array([10, 0, 10, 10])
        rng = np.random.default_rng(42)
        
        probs = policy(Q, rng)
        assert np.sum(probs == 1.0) == 1
        assert np.sum(probs == 0.0) == len(Q) - 1
    
    def test_power_of_d_clamps_to_n(self):
        policy = PowerOfDRouting(d=10)
        Q = np.array([1, 2])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs.sum() == 1.0

class TestPolicyInvariants:
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
        policy = SoftmaxRouting(alpha=alpha)
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert (probs >= 0.0).all()

class TestPolicyEdgeCases:
    def test_softmax_single_server(self):
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([100.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1.0])
    
    def test_softmax_zero_queues(self):
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])
    
    def test_softmax_large_queue_difference(self):
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 1e6])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] > 0.999999
        assert probs[1] < 1e-6
        assert np.isfinite(probs).all()
    
    def test_jsq_single_server(self):
        policy = JSQRouting()
        Q = np.array([50])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1.0])
    
    def test_jsq_zero_queues(self):
        policy = JSQRouting()
        Q = np.array([0, 0, 0, 0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25])
    
    def test_power_of_d_d_equals_n(self):
        policy = PowerOfDRouting(d=3)
        Q = np.array([5, 1, 10])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[1] == 1.0
    
    def test_power_of_d_d_equals_1(self):
        policy = PowerOfDRouting(d=1)
        Q = np.array([10, 10, 10])
        rng = np.random.default_rng(42)
        
        winners = set()
        for _ in range(100):
            probs = policy(Q, rng)
            winners.add(np.argmax(probs))
        
        assert len(winners) == 3
    
    def test_proportional_single_server(self):
        policy = ProportionalRouting(mu=np.array([5.0]))
        rng = np.random.default_rng(42)
        probs = policy(np.array([100]), rng)
        np.testing.assert_allclose(probs, [1.0])
    
    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            SoftmaxRouting(alpha=0.0)
    
    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            SoftmaxRouting(alpha=-1.0)
    
    def test_invalid_proportional_zero_mu(self):
        with pytest.raises(ValueError, match="must be > 0"):
            ProportionalRouting(mu=np.array([1.0, 0.0]))
    
    def test_invalid_proportional_negative_mu(self):
        with pytest.raises(ValueError, match="must be > 0"):
            ProportionalRouting(mu=np.array([1.0, -0.5]))
    
    def test_invalid_power_of_d_zero(self):
        with pytest.raises(ValueError, match="d must be ≥ 1"):
            PowerOfDRouting(d=0)
    
    def test_invalid_power_of_d_negative(self):
        with pytest.raises(ValueError, match="d must be ≥ 1"):
            PowerOfDRouting(d=-5)

class TestPolicyNumericalStability:
    def test_softmax_very_large_queues(self):
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([1e10, 1e10, 1e10])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])
        assert np.isfinite(probs).all()
    
    def test_softmax_very_small_alpha(self):
        policy = SoftmaxRouting(alpha=1e-10)
        Q = np.array([0.0, 1000.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert abs(probs[0] - 0.5) < 0.01
        assert np.isfinite(probs).all()
    
    def test_softmax_very_large_alpha(self):
        policy = SoftmaxRouting(alpha=1e6)
        Q = np.array([0.0, 1.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] > 0.999
        assert np.isfinite(probs).all()
    
    def test_softmax_extreme_queue_difference(self):
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 1e15])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] == 1.0  # Numerically exact
        assert probs[1] == 0.0
        assert np.isfinite(probs).all()
    
    def test_jsq_very_large_queues(self):
        policy = JSQRouting()
        Q = np.array([1e15, 1e15 - 1, 1e15])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(probs, expected)

class TestPolicyRegressions:
    def test_regression_softmax_log_sum_exp_trick(self):
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([1000.0, 1000.0, 1000.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])
    
    def test_regression_softmax_mixed_extremes(self):
        policy = SoftmaxRouting(alpha=1.0)
        Q = np.array([0.0, 1000.0, 2000.0])
        rng = np.random.default_rng(42)
        probs = policy(Q, rng)
        assert probs[0] > 0.999
        assert probs[1] < 1e-10
        assert probs[2] < 1e-100
        assert np.isfinite(probs).all()

class TestMakePolicy:
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
        with pytest.raises(KeyError, match="Unknown policy"):
            make_policy("nonexistent")

class TestRoutingPolicyProtocol:
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
