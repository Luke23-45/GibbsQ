import numpy as np

from studies.analysis.common.metrics import time_averaged_queue_lengths
from gibbsq.qroute.core.builders import build_policy_by_name
from gibbsq.qroute.core.policies import ReflectedUASRouting, UASRouting
from gibbsq.qroute.engines.numpy_engine import run_replications


def _isolated_benchmark_mean(policy) -> float:
    mu = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3], dtype=np.float64)
    reps = run_replications(
        num_servers=10,
        arrival_rate=11.2,
        service_rates=mu,
        policy=policy,
        num_replications=2,
        sim_time=1000.0,
        sample_interval=1.0,
        base_seed=42,
        progress_desc=None,
    )
    burn_in = 0.2
    return float(np.mean([time_averaged_queue_lengths(r, burn_in).sum() for r in reps]))


def test_reflected_archimedean_family_contains_current_uas():
    mu = np.array([0.5, 0.7, 0.9, 1.1], dtype=np.float64)
    Q = np.array([3, 1, 0, 2], dtype=np.int64)
    rng = np.random.default_rng(0)

    uas = UASRouting(mu, alpha=10.0)
    reflected = ReflectedUASRouting(mu, alpha=10.0, beta=1.0, gamma=1.0, c=1.0)

    assert np.allclose(uas(Q, rng), reflected(Q, rng), rtol=1e-12, atol=1e-12)


def test_reflected_candidates_outperform_current_uas_in_isolated_benchmark():
    mu = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3], dtype=np.float64)

    uas = build_policy_by_name("uas", mu=mu, alpha=10.0)
    reflected_uas = build_policy_by_name("reflected_uas", mu=mu, alpha=20.0)
    current_uas_mean = _isolated_benchmark_mean(uas)

    candidates = {
        "reflected_uas": reflected_uas,
        "shifted_tempered_1": ReflectedUASRouting(mu, alpha=20.0, beta=0.85, gamma=1.0, c=0.5),
        "shifted_tempered_2": ReflectedUASRouting(mu, alpha=10.0, beta=0.70, gamma=1.0, c=0.5),
    }

    candidate_means = {name: _isolated_benchmark_mean(policy) for name, policy in candidates.items()}

    assert min(candidate_means.values()) < current_uas_mean
    assert any(mean < current_uas_mean - 0.5 for mean in candidate_means.values())


def test_best_reflected_candidate_beats_jssq_in_short_isolated_reproduction():
    mu = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3], dtype=np.float64)

    jssq = build_policy_by_name("jssq", mu=mu)
    reflected = build_policy_by_name("reflected_uas", mu=mu, alpha=20.0)
    rng = np.random.default_rng(42)

    jssq_mean = _isolated_benchmark_mean(jssq)
    reflected_mean = _isolated_benchmark_mean(reflected)

    assert reflected_mean < jssq_mean


