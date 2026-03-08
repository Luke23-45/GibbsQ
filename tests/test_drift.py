import pytest
import numpy as np
from moeq.core.drift import (
    lyapunov_V, generator_drift, upper_bound, simplified_bound,
    verify_single, evaluate_grid
)

def test_lyapunov_V():
    Q = np.array([3, 4])
    # V(Q) = 0.5 * (3^2 + 4^2) = 0.5 * 25 = 12.5
    assert lyapunov_V(Q) == 12.5
    
    Q_zero = np.array([0, 0, 0])
    assert lyapunov_V(Q_zero) == 0.0

def test_drift_origin():
    # At Q = 0, p_i = 1/N. μ term is 0. C(Q) = λ/2 + 0 = λ/2.
    # LV(0) = λ * (1/N * 0) - 0 + λ/2 = λ/2
    Q = np.array([0, 0])
    lam = 2.0
    mu = np.array([1.0, 1.0])
    alpha = 1.0
    
    exact = generator_drift(Q, lam, mu, alpha)
    assert exact == pytest.approx(1.0)  # lam/2
    
def test_bounds_hierarchy():
    # Test that exact <= upper_bound <= simplified_bound (usually, though sb can be slighly looser/tighter depending on Q_min vs |Q|_1 scaling, the theorem guarantees exact <= ub and ub <= sb outside the compact set)
    lam = 1.0
    mu = np.array([2.0, 3.0])  # Cap = 5.0
    alpha = 1.5
    
    Q = np.array([10, 20])
    
    res = verify_single(Q, lam, mu, alpha)
    assert res["bound_holds"]
    assert res["simplified_holds"]
    
def test_vectorised_grid():
    lam = 1.0
    mu = np.array([2.0, 2.0])
    alpha = 1.0
    q_max = 10  # 11x11 = 121 states
    
    res = evaluate_grid(lam, mu, alpha, q_max=q_max)
    
    assert res.states.shape == (121, 2)
    assert res.exact_drifts.shape == (121,)
    assert res.upper_bounds.shape == (121,)
    assert res.simplified_bounds.shape == (121,)
    assert res.violations == 0

def test_grid_too_large():
    lam = 1.0
    mu = np.array([2.0, 2.0, 2.0, 2.0])  # N=4
    with pytest.raises(ValueError, match="infeasible"):
        evaluate_grid(lam, mu, 1.0, q_max=5)
