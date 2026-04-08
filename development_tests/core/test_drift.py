import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from gibbsq.core.drift import (
    lyapunov_V, generator_drift, upper_bound, simplified_bound,
    verify_single, evaluate_grid, evaluate_trajectory, DriftResult,
)

class TestLyapunovVCcorrectness:
    def test_lyapunov_V_basic(self):
        Q = np.array([3, 4])
        assert lyapunov_V(Q) == 12.5
    
    def test_lyapunov_V_zero(self):
        Q = np.array([0, 0, 0])
        assert lyapunov_V(Q) == 0.0
    
    def test_lyapunov_V_single_element(self):
        Q = np.array([5])
        assert lyapunov_V(Q) == 12.5
    
    def test_lyapunov_V_larger_array(self):
        Q = np.array([1, 2, 3, 4, 5])
        assert lyapunov_V(Q) == 27.5

class TestGeneratorDriftCorrectness:
    def test_drift_at_origin(self):
        Q = np.array([0, 0])
        lam = 2.0
        mu = np.array([1.0, 1.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha, mode="raw")
        assert exact == pytest.approx(1.0)
    
    def test_drift_at_origin_with_active_servers(self):
        Q = np.array([0, 0, 0])
        lam = 3.0
        mu = np.array([1.0, 2.0, 3.0])
        alpha = 0.5
        
        exact = generator_drift(Q, lam, mu, alpha, mode="raw")
        assert exact == pytest.approx(1.5)
    
    def test_drift_with_positive_queues(self):
        Q = np.array([10, 20])
        lam = 1.0
        mu = np.array([2.0, 3.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha, mode="raw")
        
        
        assert exact < 0  # Should be negative (draining)
        assert exact == pytest.approx(-67.0, rel=0.01)

class TestBoundsCorrectness:
    def test_upper_bound_formula(self):
        Q = np.array([5, 10, 15])
        lam = 1.0
        mu = np.array([2.0, 3.0, 4.0])
        alpha = 1.0
        
        ub = upper_bound(Q, lam, mu, alpha)
        
        
        import math
        R = (lam * math.log(3)) / alpha + (lam + 9) / 2
        expected = -(9-1)*5 - (2*0 + 3*5 + 4*10) + R
        assert ub == pytest.approx(expected)
    
    def test_simplified_bound_formula(self):
        Q = np.array([5, 10, 15])
        lam = 1.0
        mu = np.array([2.0, 3.0, 4.0])
        alpha = 1.0
        
        sb = simplified_bound(Q, lam, mu, alpha)
        
        
        import math
        eps = min((9-1)/3, 2.0)
        R = (lam * math.log(3)) / alpha + (lam + 9) / 2
        expected = -eps * 30 + R
        assert sb == pytest.approx(expected)

    def test_uas_upper_bound_matches_weighted_jensen_formula(self):
        Q = np.array([5, 10, 15])
        lam = 1.0
        mu = np.array([2.0, 3.0, 4.0])
        alpha = 1.0

        ub = upper_bound(Q, lam, mu, alpha, mode="uas")

        import math
        cap = mu.sum()
        eps = (cap - lam) / cap
        R = (lam * len(mu)) / cap + (len(mu) / 2.0)
        expected = -eps * Q.sum() + R
        assert ub == pytest.approx(expected)

    def test_uas_bound_dominates_exact_drift_on_small_grid(self):
        lam = 0.9
        mu = np.array([0.8, 1.1])
        alpha = 1.3

        for q0 in range(12):
            for q1 in range(12):
                Q = np.array([q0, q1])
                exact = generator_drift(Q, lam, mu, alpha, mode="uas")
                ub = upper_bound(Q, lam, mu, alpha, mode="uas")
                assert exact <= ub + 1e-12, f"Q={Q}, exact={exact}, ub={ub}"

    def test_uas_grid_verification_has_zero_violations(self):
        lam = 1.0
        mu = np.array([2.0, 3.0, 4.0])
        alpha = 0.7

        result = evaluate_grid(lam, mu, alpha, q_max=8, mode="uas")

        assert result.violations == 0

class TestVerifySingleCorrectness:
    def test_verify_single_structure(self):
        Q = np.array([10, 20])
        lam = 1.0
        mu = np.array([2.0, 3.0])
        alpha = 1.0
        
        result = verify_single(Q, lam, mu, alpha)
        
        assert "exact_drift" in result
        assert "upper_bound" in result
        assert "simplified_bound" in result
        assert "bound_holds" in result
        assert "simplified_holds" in result
        
        assert result["bound_holds"] in (True, False, np.True_, np.False_)
        assert result["simplified_holds"] in (True, False, np.True_, np.False_)

class TestDriftInvariants:
    @given(
        Q=arrays(
            dtype=np.int64,
            shape=st.integers(min_value=1, max_value=10),
            elements=st.integers(min_value=0, max_value=100),
        ),
        lam=st.floats(min_value=0.1, max_value=10.0, allow_infinity=False, allow_nan=False),
        alpha=st.floats(min_value=0.1, max_value=10.0, allow_infinity=False, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_exact_drift_is_finite(self, Q, lam, alpha):
        assume(len(Q) >= 1)
        N = len(Q)
        mu = np.full(N, (lam + 1.0) / N + 0.1)
        
        exact = generator_drift(Q, lam, mu, alpha, mode="raw")
        assert np.isfinite(exact)
    
    @given(
        Q=arrays(
            dtype=np.int64,
            shape=st.integers(min_value=2, max_value=10),
            elements=st.integers(min_value=0, max_value=100),
        ),
        lam=st.floats(min_value=0.1, max_value=5.0, allow_infinity=False, allow_nan=False),
        alpha=st.floats(min_value=0.1, max_value=10.0, allow_infinity=False, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_upper_bound_bounds_exact(self, Q, lam, alpha):
        assume(len(Q) >= 2)
        N = len(Q)
        mu = np.full(N, (lam + 1.0) / N + 0.5)
        
        exact = generator_drift(Q, lam, mu, alpha, mode="raw")
        ub = upper_bound(Q, lam, mu, alpha)
        
        assert exact <= ub + 1e-10, f"exact={exact} > ub={ub}"
    
    @given(
        Q=arrays(
            dtype=np.int64,
            shape=st.integers(min_value=2, max_value=10),
            elements=st.integers(min_value=0, max_value=100),
        ),
        lam=st.floats(min_value=0.1, max_value=5.0, allow_infinity=False, allow_nan=False),
        alpha=st.floats(min_value=0.1, max_value=10.0, allow_infinity=False, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_simplified_bound_bounds_exact(self, Q, lam, alpha):
        assume(len(Q) >= 2)
        N = len(Q)
        mu = np.full(N, (lam + 1.0) / N + 0.5)
        
        exact = generator_drift(Q, lam, mu, alpha)
        sb = simplified_bound(Q, lam, mu, alpha)
        
        assert exact <= sb + 1e-10, f"exact={exact} > sb={sb}"
    
    def test_bounds_hierarchy(self):
        Q = np.array([100, 200])
        lam = 1.0
        mu = np.array([2.0, 3.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        ub = upper_bound(Q, lam, mu, alpha)
        sb = simplified_bound(Q, lam, mu, alpha)
        
        assert exact <= ub
        assert sb >= ub - 1e-10

class TestDriftEdgeCases:
    def test_single_server_drift(self):
        Q = np.array([10])
        lam = 0.5
        mu = np.array([1.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        ub = upper_bound(Q, lam, mu, alpha)
        sb = simplified_bound(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
        assert exact <= ub + 1e-10
        assert exact <= sb + 1e-10
    
    def test_zero_queue_drift(self):
        Q = np.array([0, 0, 0])
        lam = 1.0
        mu = np.array([1.0, 1.0, 1.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        assert exact > 0
        assert exact == pytest.approx(lam / 2)
    
    def test_very_large_queue_drift(self):
        Q = np.array([1000, 1000])
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        assert exact < 0
        assert exact < -1000  # Strongly negative
    
    def test_identical_queues(self):
        Q = np.array([10, 10, 10])
        lam = 1.0
        mu = np.array([2.0, 3.0, 4.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        assert np.isfinite(exact)
    
    def test_near_capacity_drift(self):
        Q = np.array([100, 100])
        lam = 1.0
        mu = np.array([0.51, 0.51])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        ub = upper_bound(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
        assert exact <= ub + 1e-10

class TestGridEvaluationEdgeCases:
    def test_grid_n_equals_2(self):
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        q_max = 5
        
        result = evaluate_grid(lam, mu, alpha, q_max)
        
        assert result.states.shape == ((q_max+1)**2, 2)
        assert result.violations == 0
    
    def test_grid_n_equals_3(self):
        lam = 1.0
        mu = np.array([2.0, 2.0, 2.0])
        alpha = 1.0
        q_max = 3
        
        result = evaluate_grid(lam, mu, alpha, q_max)
        
        assert result.states.shape == ((q_max+1)**3, 3)
        assert result.violations == 0
    
    def test_grid_n_equals_4_raises(self):
        lam = 1.0
        mu = np.array([1.0, 1.0, 1.0, 1.0])
        alpha = 1.0
        
        with pytest.raises(ValueError, match="infeasible"):
            evaluate_grid(lam, mu, alpha, q_max=5)
    
    def test_grid_q_max_zero(self):
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        q_max = 0
        
        result = evaluate_grid(lam, mu, alpha, q_max)
        
        assert result.states.shape == (1, 2)
        assert np.array_equal(result.states[0], [0, 0])

class TestDriftNumericalStability:
    def test_very_large_queues_no_overflow(self):
        Q = np.array([1e10, 1e10])
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        ub = upper_bound(Q, lam, mu, alpha)
        sb = simplified_bound(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
        assert np.isfinite(ub)
        assert np.isfinite(sb)
    
    def test_very_small_alpha_drift(self):
        Q = np.array([10, 20])
        lam = 1.0
        mu = np.array([2.0, 3.0])
        alpha = 1e-10
        
        exact = generator_drift(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
    
    def test_very_large_alpha_drift(self):
        Q = np.array([10, 20])
        lam = 1.0
        mu = np.array([2.0, 3.0])
        alpha = 1e10
        
        exact = generator_drift(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
    
    def test_extreme_queue_difference(self):
        Q = np.array([0, 1e15])
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        ub = upper_bound(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
        assert np.isfinite(ub)
    
    def test_vectorised_drift_large_batch(self):
        M = 10000
        N = 3
        Q_all = np.random.randint(0, 100, size=(M, N))
        lam = 1.0
        mu = np.array([2.0, 2.0, 2.0])
        alpha = 1.0
        
        result = evaluate_trajectory(Q_all, lam, mu, alpha)
        
        assert result.exact_drifts.shape == (M,)
        assert np.isfinite(result.exact_drifts).all()
        assert result.violations == 0

class TestDriftRegressions:
    def test_regression_softmax_stability_in_drift(self):
        Q = np.array([1000, 1000, 1000])
        lam = 1.0
        mu = np.array([2.0, 2.0, 2.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
    
    def test_regression_bound_tolerance(self):
        Q = np.array([100, 100])
        lam = 1.0
        mu = np.array([1.0000001, 1.0000001])
        alpha = 1.0
        
        result = verify_single(Q, lam, mu, alpha)
        
        assert result["bound_holds"]

class TestEvaluateTrajectory:
    def test_evaluate_trajectory_basic(self):
        states = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [2, 1],
            [2, 2],
        ])
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        
        result = evaluate_trajectory(states, lam, mu, alpha)
        
        assert result.states.shape == (5, 2)
        assert result.exact_drifts.shape == (5,)
        assert result.violations == 0
    
    def test_evaluate_trajectory_empty(self):
        states = np.array([]).reshape(0, 2)
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        
        result = evaluate_trajectory(states, lam, mu, alpha)
        
        assert result.states.shape == (0, 2)
        assert result.exact_drifts.shape == (0,)

class TestDriftResult:
    def test_drift_result_attributes(self):
        result = DriftResult(
            states=np.array([[0, 0], [1, 1]]),
            exact_drifts=np.array([0.5, -1.0]),
            upper_bounds=np.array([1.0, 0.0]),
            simplified_bounds=np.array([2.0, 1.0]),
            violations=0,
            norms=np.array([0.0, 2.0]),
        )
        
        assert result.states.shape == (2, 2)
        assert result.violations == 0
    
    def test_drift_result_frozen(self):
        result = DriftResult(
            states=np.array([[0, 0]]),
            exact_drifts=np.array([0.5]),
            upper_bounds=np.array([1.0]),
            simplified_bounds=np.array([2.0]),
            violations=0,
            norms=np.array([0.0]),
        )
        
        with pytest.raises(AttributeError):
            result.violations = 5
