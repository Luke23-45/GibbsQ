import pytest
import numpy as np
from moeq.core.policies import (
    SoftmaxRouting, UniformRouting, ProportionalRouting, 
    JSQRouting, PowerOfDRouting, make_policy
)

@pytest.fixture
def rng():
    return np.random.default_rng(42)

def test_softmax_stable(rng):
    # Log-sum-exp trick must handle very large Q values without overflow
    policy = SoftmaxRouting(alpha=1.0)
    Q = np.array([1000, 1000, 1000])  # Without shift, exp(-1000) underflows
    probs = policy(Q, rng)
    np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])

    Q2 = np.array([0, 1000])  # Server 0 should get essentially probability 1
    probs2 = policy(Q2, rng)
    assert probs2[0] > 0.999999

def test_softmax_temperature(rng):
    policy1 = SoftmaxRouting(alpha=0.01)  # High temp -> ~uniform
    policy2 = SoftmaxRouting(alpha=10.0)  # Low temp -> ~JSQ
    
    Q = np.array([1, 2, 3])
    
    p1 = policy1(Q, rng)
    assert abs(p1[0] - 1/3) < 0.05
    
    p2 = policy2(Q, rng)
    assert p2[0] > 0.99  # Almost entirely to the shortest queue.

def test_uniform(rng):
    policy = UniformRouting()
    Q = np.array([10, 0, 5, 2])
    probs = policy(Q, rng)
    np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25])

def test_proportional(rng):
    policy = ProportionalRouting(mu=[1.0, 3.0])
    probs = policy(np.array([5, 5]), rng)
    np.testing.assert_allclose(probs, [0.25, 0.75])

def test_jsq(rng):
    policy = JSQRouting()
    # Tie breaking case
    Q = np.array([3, 1, 4, 1])
    probs = policy(Q, rng)
    np.testing.assert_allclose(probs, [0.0, 0.5, 0.0, 0.5])

def test_power_of_d():
    policy = PowerOfDRouting(d=2)
    Q = np.array([10, 0, 10, 10])
    
    # Run 1000 times, Server 1 (queue length 0) should win roughly 50% of the time
    # Specifically: 1 - P(server 1 not sampled) = 1 - (3/4 * 2/3) = 1 - 0.5 = 0.5
    wins = 0
    rng = np.random.default_rng(123)
    for _ in range(1000):
        probs = policy(Q, rng)
        if probs[1] == 1.0:
            wins += 1
            
    assert 450 < wins < 550

def test_factory():
    p = make_policy("softmax", alpha=2.5)
    assert isinstance(p, SoftmaxRouting)
    assert p.alpha == 2.5
    
    p = make_policy("jsq")
    assert isinstance(p, JSQRouting)
