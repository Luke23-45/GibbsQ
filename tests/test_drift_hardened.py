"""
Hardened test suite for gibbsq.core.drift — Robustness Loop Stage 2.

Categories:
- A: Correctness Tests
- B: Invariant Tests
- C: Edge Case Tests
- D: Numerical Stability Tests
- E: Gradient Flow Tests (N/A — NumPy implementation)
- F: Regression Tests
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from gibbsq.core.drift import (
    lyapunov_V, generator_drift, upper_bound, simplified_bound,
    verify_single, evaluate_grid, evaluate_trajectory, DriftResult,
)


# ============================================================
# CATEGORY A: CORRECTNESS TESTS
# ============================================================

class TestLyapunovVCcorrectness:
    """Verify V(Q) = ½‖Q‖₂²."""
    
    def test_lyapunov_V_basic(self):
        """V([3, 4]) = 0.5 * (9 + 16) = 12.5."""
        Q = np.array([3, 4])
        assert lyapunov_V(Q) == 12.5
    
    def test_lyapunov_V_zero(self):
        """V([0, 0, 0]) = 0."""
        Q = np.array([0, 0, 0])
        assert lyapunov_V(Q) == 0.0
    
    def test_lyapunov_V_single_element(self):
        """V([5]) = 0.5 * 25 = 12.5."""
        Q = np.array([5])
        assert lyapunov_V(Q) == 12.5
    
    def test_lyapunov_V_larger_array(self):
        """V([1, 2, 3, 4, 5]) = 0.5 * (1+4+9+16+25) = 27.5."""
        Q = np.array([1, 2, 3, 4, 5])
        assert lyapunov_V(Q) == 27.5


class TestGeneratorDriftCorrectness:
    """Verify 𝓛V(Q) = λ⟨p, Q⟩ − ⟨μ, Q⟩ + C(Q)."""
    
    def test_drift_at_origin(self):
        """At Q=0: p_i = 1/N, ⟨p,Q⟩=0, ⟨μ,Q⟩=0, C(Q) = λ/2."""
        Q = np.array([0, 0])
        lam = 2.0
        mu = np.array([1.0, 1.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        # 𝓛V(0) = λ/2 = 1.0
        assert exact == pytest.approx(1.0)
    
    def test_drift_at_origin_with_active_servers(self):
        """C(Q) = λ/2 + ½⟨μ, 𝟙(Q>0)⟩, but at Q=0, no servers active."""
        Q = np.array([0, 0, 0])
        lam = 3.0
        mu = np.array([1.0, 2.0, 3.0])
        alpha = 0.5
        
        exact = generator_drift(Q, lam, mu, alpha)
        # 𝓛V(0) = λ/2 + 0 = 1.5
        assert exact == pytest.approx(1.5)
    
    def test_drift_with_positive_queues(self):
        """Verify full formula with non-zero queues."""
        Q = np.array([10, 20])
        lam = 1.0
        mu = np.array([2.0, 3.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        
        # Manual calculation:
        # p = softmax: logits = [-10, -20], shifted to [0, -10]
        # w = [1, e^-10] ≈ [1, 4.5e-5]
        # p ≈ [0.99995, 4.5e-5]
        # ⟨p, Q⟩ ≈ 10 * 0.99995 + 20 * 4.5e-5 ≈ 10.0
        # ⟨μ, Q⟩ = 2*10 + 3*20 = 80
        # C(Q) = 0.5 + 0.5*(2+3) = 3.0 (both servers active)
        # 𝓛V ≈ 1*10 - 80 + 3 = -67
        
        assert exact < 0  # Should be negative (draining)
        assert exact == pytest.approx(-67.0, rel=0.01)


class TestBoundsCorrectness:
    """Verify bound formulas."""
    
    def test_upper_bound_formula(self):
        """Verify upper bound: −(Λ−λ)Q_min − Σμ_iΔ_i + R."""
        Q = np.array([5, 10, 15])
        lam = 1.0
        mu = np.array([2.0, 3.0, 4.0])  # Λ = 9
        alpha = 1.0
        
        ub = upper_bound(Q, lam, mu, alpha)
        
        # Q_min = 5
        # Δ = [0, 5, 10]
        # R = (1*log(3))/1 + (1+9)/2 = 1.099 + 5 = 6.099
        # ub = -(9-1)*5 - (2*0 + 3*5 + 4*10) + 6.099
        #    = -40 - 55 + 6.099 = -88.9
        
        import math
        R = (lam * math.log(3)) / alpha + (lam + 9) / 2
        expected = -(9-1)*5 - (2*0 + 3*5 + 4*10) + R
        assert ub == pytest.approx(expected)
    
    def test_simplified_bound_formula(self):
        """Verify simplified bound: −ε|Q|₁ + R."""
        Q = np.array([5, 10, 15])
        lam = 1.0
        mu = np.array([2.0, 3.0, 4.0])  # Λ = 9
        alpha = 1.0
        
        sb = simplified_bound(Q, lam, mu, alpha)
        
        # ε = min((9-1)/3, 2) = min(2.67, 2) = 2
        # |Q|₁ = 30
        # R = 6.099
        # sb = -2 * 30 + 6.099 = -53.9
        
        import math
        eps = min((9-1)/3, 2.0)
        R = (lam * math.log(3)) / alpha + (lam + 9) / 2
        expected = -eps * 30 + R
        assert sb == pytest.approx(expected)


class TestVerifySingleCorrectness:
    """Verify verify_single returns correct structure."""
    
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
        
        # bound_holds is np.True_ or np.False_, which are truthy
        assert result["bound_holds"] in (True, False, np.True_, np.False_)
        assert result["simplified_holds"] in (True, False, np.True_, np.False_)


# ============================================================
# CATEGORY B: INVARIANT TESTS
# ============================================================

class TestDriftInvariants:
    """Invariants that must hold for drift computations."""
    
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
        """𝓛V(Q) must always be finite for valid inputs."""
        assume(len(Q) >= 1)
        # Generate mu such that cap > lam
        N = len(Q)
        mu = np.full(N, (lam + 1.0) / N + 0.1)
        
        exact = generator_drift(Q, lam, mu, alpha)
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
        """exact <= upper_bound must always hold."""
        assume(len(Q) >= 2)
        N = len(Q)
        mu = np.full(N, (lam + 1.0) / N + 0.5)
        
        exact = generator_drift(Q, lam, mu, alpha)
        ub = upper_bound(Q, lam, mu, alpha)
        
        # Allow tiny tolerance for floating point
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
        """exact <= simplified_bound must always hold."""
        assume(len(Q) >= 2)
        N = len(Q)
        mu = np.full(N, (lam + 1.0) / N + 0.5)
        
        exact = generator_drift(Q, lam, mu, alpha)
        sb = simplified_bound(Q, lam, mu, alpha)
        
        assert exact <= sb + 1e-10, f"exact={exact} > sb={sb}"
    
    def test_bounds_hierarchy(self):
        """For large |Q|₁, simplified_bound should be looser than upper_bound."""
        Q = np.array([100, 200])
        lam = 1.0
        mu = np.array([2.0, 3.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        ub = upper_bound(Q, lam, mu, alpha)
        sb = simplified_bound(Q, lam, mu, alpha)
        
        # exact <= ub <= sb (typically, but not guaranteed for all Q)
        assert exact <= ub
        # sb should be looser (larger/more positive)
        assert sb >= ub - 1e-10


# ============================================================
# CATEGORY C: EDGE CASE TESTS
# ============================================================

class TestDriftEdgeCases:
    """Test boundary conditions and degenerate inputs."""
    
    def test_single_server_drift(self):
        """N=1 edge case."""
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
        """Q=0 should give positive drift (arrival dominates)."""
        Q = np.array([0, 0, 0])
        lam = 1.0
        mu = np.array([1.0, 1.0, 1.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        # At origin, drift = λ/2 > 0
        assert exact > 0
        assert exact == pytest.approx(lam / 2)
    
    def test_very_large_queue_drift(self):
        """Large Q should give strongly negative drift."""
        Q = np.array([1000, 1000])
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        assert exact < 0
        assert exact < -1000  # Strongly negative
    
    def test_identical_queues(self):
        """When all Q_i equal, softmax gives uniform."""
        Q = np.array([10, 10, 10])
        lam = 1.0
        mu = np.array([2.0, 3.0, 4.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        assert np.isfinite(exact)
    
    def test_near_capacity_drift(self):
        """When Λ ≈ λ, drift should still be negative for large Q."""
        Q = np.array([100, 100])
        lam = 1.0
        mu = np.array([0.51, 0.51])  # Λ = 1.02 ≈ λ
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        ub = upper_bound(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
        assert exact <= ub + 1e-10


class TestGridEvaluationEdgeCases:
    """Test grid evaluation edge cases."""
    
    def test_grid_n_equals_2(self):
        """N=2 grid should work."""
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        q_max = 5
        
        result = evaluate_grid(lam, mu, alpha, q_max)
        
        assert result.states.shape == ((q_max+1)**2, 2)
        assert result.violations == 0
    
    def test_grid_n_equals_3(self):
        """N=3 grid should work."""
        lam = 1.0
        mu = np.array([2.0, 2.0, 2.0])
        alpha = 1.0
        q_max = 3
        
        result = evaluate_grid(lam, mu, alpha, q_max)
        
        assert result.states.shape == ((q_max+1)**3, 3)
        assert result.violations == 0
    
    def test_grid_n_equals_4_raises(self):
        """N=4 grid should raise ValueError (too large)."""
        lam = 1.0
        mu = np.array([1.0, 1.0, 1.0, 1.0])
        alpha = 1.0
        
        with pytest.raises(ValueError, match="infeasible"):
            evaluate_grid(lam, mu, alpha, q_max=5)
    
    def test_grid_q_max_zero(self):
        """q_max=0 should give single state (origin)."""
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        q_max = 0
        
        result = evaluate_grid(lam, mu, alpha, q_max)
        
        assert result.states.shape == (1, 2)
        assert np.array_equal(result.states[0], [0, 0])


# ============================================================
# CATEGORY D: NUMERICAL STABILITY TESTS
# ============================================================

class TestDriftNumericalStability:
    """Test behavior under numerically challenging inputs."""
    
    def test_very_large_queues_no_overflow(self):
        """Very large Q values should not cause overflow."""
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
        """Very small α (high temperature) should compute without overflow."""
        Q = np.array([10, 20])
        lam = 1.0
        mu = np.array([2.0, 3.0])
        alpha = 1e-10
        
        exact = generator_drift(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
    
    def test_very_large_alpha_drift(self):
        """Very large α (low temperature) should compute without underflow."""
        Q = np.array([10, 20])
        lam = 1.0
        mu = np.array([2.0, 3.0])
        alpha = 1e10
        
        exact = generator_drift(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
    
    def test_extreme_queue_difference(self):
        """Extreme difference in queue lengths should be stable."""
        Q = np.array([0, 1e15])
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        ub = upper_bound(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
        assert np.isfinite(ub)
    
    def test_vectorised_drift_large_batch(self):
        """Vectorized drift should handle large batches."""
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


# ============================================================
# CATEGORY F: REGRESSION TESTS
# ============================================================

class TestDriftRegressions:
    """Prevent reintroduction of known faults."""
    
    def test_regression_softmax_stability_in_drift(self):
        """Ensure softmax in drift uses log-sum-exp trick."""
        # Without the trick, exp(-1000) would underflow
        Q = np.array([1000, 1000, 1000])
        lam = 1.0
        mu = np.array([2.0, 2.0, 2.0])
        alpha = 1.0
        
        exact = generator_drift(Q, lam, mu, alpha)
        
        assert np.isfinite(exact)
    
    def test_regression_bound_tolerance(self):
        """Bounds should hold with small tolerance for floating point."""
        # Construct a case where floating point might cause tiny violation
        Q = np.array([100, 100])
        lam = 1.0
        mu = np.array([1.0000001, 1.0000001])
        alpha = 1.0
        
        result = verify_single(Q, lam, mu, alpha)
        
        # Should pass with tolerance
        assert result["bound_holds"]


# ============================================================
# TRAJECTORY EVALUATION TESTS
# ============================================================

class TestEvaluateTrajectory:
    """Test trajectory-based drift evaluation."""
    
    def test_evaluate_trajectory_basic(self):
        """Evaluate drift on a simple trajectory."""
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
        """Empty trajectory should work."""
        states = np.array([]).reshape(0, 2)
        lam = 1.0
        mu = np.array([2.0, 2.0])
        alpha = 1.0
        
        result = evaluate_trajectory(states, lam, mu, alpha)
        
        assert result.states.shape == (0, 2)
        assert result.exact_drifts.shape == (0,)


# ============================================================
# DRIFT RESULT DATACLASS TESTS
# ============================================================

class TestDriftResult:
    """Test DriftResult dataclass."""
    
    def test_drift_result_attributes(self):
        """DriftResult should have all required attributes."""
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
        """DriftResult should be immutable."""
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
