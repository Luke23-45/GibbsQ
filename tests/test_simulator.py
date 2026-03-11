import pytest
import numpy as np
from gibbsq.engines.numpy_engine import simulate
from gibbsq.core.policies import JSQRouting

def test_mm1_conservation():
    # 1 server, guaranteed stability (λ=0.5, μ=1.0)
    res = simulate(
        num_servers=1,
        arrival_rate=0.5,
        service_rates=np.array([1.0]),
        policy=JSQRouting(),  # JSQ on 1 server is just passthrough
        sim_time=100.0,
        rng=np.random.default_rng(1)
    )
    
    final_q = res.states[-1, 0]
    expected_q = res.arrival_count - res.departure_count
    assert final_q == expected_q
    
    assert res.arrival_count > 0
    assert res.departure_count > 0

def test_symmetric_servers():
    res = simulate(
        num_servers=3,
        arrival_rate=1.0,
        service_rates=np.array([1.0, 1.0, 1.0]),
        policy=JSQRouting(),
        sim_time=50.0,
        rng=np.random.default_rng(2)
    )
    # JSQ should keep queues roughly balanced
    final_state = res.states[-1]
    assert np.max(final_state) - np.min(final_state) <= 2
    
def test_zero_arrivals():
    res = simulate(
        num_servers=2,
        arrival_rate=0.0,  # No arrivals
        service_rates=np.array([1.0, 1.0]),
        policy=JSQRouting(),
        sim_time=10.0,
    )
    assert res.arrival_count == 0
    assert res.departure_count == 0
    np.testing.assert_array_equal(res.states[-1], [0, 0])


def test_zero_arrivals_records_full_timeline():
    res = simulate(
        num_servers=2,
        arrival_rate=0.0,
        service_rates=np.array([1.0, 1.0]),
        policy=JSQRouting(),
        sim_time=10.0,
        sample_interval=1.0,
    )
    assert res.times[-1] == pytest.approx(10.0)


def test_policy_output_validation():
    class BadPolicy:
        def __call__(self, Q, rng):
            return np.array([0.2, 0.2])

    with pytest.raises(ValueError, match="sum to 1.0"):
        simulate(
            num_servers=2,
            arrival_rate=1.0,
            service_rates=np.array([1.0, 1.0]),
            policy=BadPolicy(),
            sim_time=1.0,
            rng=np.random.default_rng(0),
        )
