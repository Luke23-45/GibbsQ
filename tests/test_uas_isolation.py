import numpy as np

from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.core.builders import build_policy_by_name
from gibbsq.core.policies import JSSQRouting, UASRouting
from gibbsq.engines.numpy_engine import run_replications


def test_uas_routing_matches_documented_probability_formula():
    mu = np.array([0.5, 0.7, 0.9, 1.1], dtype=np.float64)
    Q = np.array([3, 1, 0, 2], dtype=np.int64)
    alpha = 10.0

    policy = UASRouting(mu, alpha=alpha)
    probs = policy(Q, np.random.default_rng(0))

    expected_logits = np.log(mu) - alpha * ((Q.astype(np.float64) + 1.0) / mu)
    expected_logits -= expected_logits.max()
    expected = np.exp(expected_logits)
    expected /= expected.sum()

    assert np.allclose(probs, expected, rtol=1e-12, atol=1e-12)


def test_uas_and_jssq_share_the_same_best_server_on_representative_states():
    mu = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3], dtype=np.float64)
    uas = UASRouting(mu, alpha=10.0)
    jssq = JSSQRouting(mu)
    rng = np.random.default_rng(0)

    states = [
        np.zeros(10, dtype=np.int64),
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 3], dtype=np.int64),
        np.array([5, 4, 3, 2, 1, 0, 0, 0, 0, 0], dtype=np.int64),
        np.array([10, 8, 6, 4, 2, 0, 0, 0, 0, 0], dtype=np.int64),
    ]

    for Q in states:
        uas_best = int(np.argmax(uas(Q, rng)))
        jssq_best = int(np.argmax(jssq(Q, rng)))
        assert uas_best == jssq_best


def test_isolated_numpy_ssa_reproduces_uas_below_jssq_on_policy_benchmark():
    mu = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3], dtype=np.float64)
    arrival_rate = 11.2
    sim_kwargs = dict(
        num_servers=10,
        arrival_rate=arrival_rate,
        service_rates=mu,
        num_replications=2,
        sim_time=1000.0,
        sample_interval=1.0,
        base_seed=42,
        progress_desc=None,
    )

    jssq = build_policy_by_name("jssq", mu=mu)
    uas = build_policy_by_name("uas", mu=mu, alpha=10.0)

    jssq_results = run_replications(policy=jssq, **sim_kwargs)
    uas_results = run_replications(policy=uas, **sim_kwargs)

    burn_in = 0.2
    jssq_mean = float(np.mean([time_averaged_queue_lengths(r, burn_in).sum() for r in jssq_results]))
    uas_mean = float(np.mean([time_averaged_queue_lengths(r, burn_in).sum() for r in uas_results]))

    assert uas_mean > jssq_mean
