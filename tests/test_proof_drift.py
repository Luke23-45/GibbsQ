import numpy as np
import pytest

from gibbsq.core.drift import evaluate_grid, generator_drift, simplified_bound, upper_bound


def test_raw_upper_bound_formula():
    Q = np.array([5, 10, 15])
    lam = 1.0
    mu = np.array([2.0, 3.0, 4.0])
    alpha = 1.0

    ub = upper_bound(Q, lam, mu, alpha)

    import math

    R = (lam * math.log(3)) / alpha + (lam + 9) / 2
    expected = -(9 - 1) * 5 - (2 * 0 + 3 * 5 + 4 * 10) + R
    assert ub == pytest.approx(expected)


def test_raw_simplified_bound_formula():
    Q = np.array([5, 10, 15])
    lam = 1.0
    mu = np.array([2.0, 3.0, 4.0])
    alpha = 1.0

    sb = simplified_bound(Q, lam, mu, alpha)

    import math

    eps = min((9 - 1) / 3, 2.0)
    R = (lam * math.log(3)) / alpha + (lam + 9) / 2
    expected = -eps * 30 + R
    assert sb == pytest.approx(expected)


def test_uas_upper_bound_matches_weighted_jensen_formula():
    Q = np.array([5, 10, 15])
    lam = 1.0
    mu = np.array([2.0, 3.0, 4.0])
    alpha = 1.0

    ub = upper_bound(Q, lam, mu, alpha, mode="uas")

    cap = mu.sum()
    eps = (cap - lam) / cap
    R = (lam * len(mu)) / cap + (len(mu) / 2.0)
    expected = -eps * Q.sum() + R
    assert ub == pytest.approx(expected)


def test_uas_bound_dominates_exact_drift_on_small_grid():
    lam = 0.9
    mu = np.array([0.8, 1.1])
    alpha = 1.3

    for q0 in range(12):
        for q1 in range(12):
            Q = np.array([q0, q1])
            exact = generator_drift(Q, lam, mu, alpha, mode="uas")
            ub = upper_bound(Q, lam, mu, alpha, mode="uas")
            assert exact <= ub + 1e-12, f"Q={Q}, exact={exact}, ub={ub}"


def test_uas_grid_verification_has_zero_violations():
    lam = 1.0
    mu = np.array([2.0, 3.0, 4.0])
    alpha = 0.7

    result = evaluate_grid(lam, mu, alpha, q_max=8, mode="uas")

    assert result.violations == 0
