import numpy as np

from experiments.evaluation import policy_comparison as pc
from experiments.evaluation.baselines_comparison import _compute_metrics_from_arrays, _standard_error
from gibbsq.analysis.metrics import gini_coefficient, sojourn_time_estimate, time_averaged_queue_lengths
from gibbsq.engines.numpy_engine import SimResult


def test_iter_with_progress_delegates_to_shared_helper(monkeypatch):
    calls = {}

    def fake_iter_progress(items, **kwargs):
        calls["items"] = list(items)
        calls["kwargs"] = kwargs
        return calls["items"]

    monkeypatch.setattr(pc, "iter_progress", fake_iter_progress)
    wrapped = pc._iter_with_progress(["a", "b"], desc="Policies", total=2)
    assert list(wrapped) == ["a", "b"]
    assert calls["kwargs"]["desc"] == "Policies"
    assert calls["kwargs"]["total"] == 2


def test_collect_trajectory_ssa_jsq_path_uses_shared_jsq_contract(monkeypatch):
    from experiments.training import train_reinforce as tr

    class FakeJSQPolicy:
        def __call__(self, q, rng):
            return np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float64)

    class FakeRng:
        def exponential(self, scale):
            return 0.1

        def uniform(self, low, high):
            return high * 0.99

    monkeypatch.setattr(tr, "JSQRouting", lambda: FakeJSQPolicy())

    traj = tr.collect_trajectory_ssa(
        policy_net=None,
        num_servers=4,
        arrival_rate=1.0,
        service_rates=np.ones(4, dtype=np.float64),
        sim_time=1.0,
        rng=FakeRng(),
        use_jsq=True,
    )

    assert traj.actions[0] == 3


def test_compute_metrics_from_arrays_matches_reference_loop():
    reps, t_steps, n = 3, 16, 2
    rng = np.random.default_rng(7)

    times = np.tile(np.linspace(0.0, 3.0, t_steps, dtype=np.float64), (reps, 1))
    states = rng.integers(0, 6, size=(reps, t_steps, n), dtype=np.int32)
    arrs = rng.integers(10, 20, size=(reps,), dtype=np.int32)
    deps = rng.integers(10, 20, size=(reps,), dtype=np.int32)

    burn = 0.2
    lam = 1.0

    q_vals, g_vals, w_vals, last_res = _compute_metrics_from_arrays(
        times=times,
        states=states,
        arrs=arrs,
        deps=deps,
        num_servers=n,
        arrival_rate=lam,
        burn_in_fraction=burn,
    )

    q_ref, g_ref, w_ref = [], [], []
    for r in range(reps):
        res = SimResult(
            times=times[r],
            states=states[r],
            arrival_count=int(arrs[r]),
            departure_count=int(deps[r]),
            final_time=float(times[r][-1]),
            num_servers=n,
        )
        avg_q = time_averaged_queue_lengths(res, burn)
        q_ref.append(float(avg_q.sum()))
        g_ref.append(float(gini_coefficient(avg_q)))
        w_ref.append(float(sojourn_time_estimate(res, lam, burn)))

    assert np.allclose(q_vals, q_ref)
    assert np.allclose(g_vals, g_ref)
    assert np.allclose(w_vals, w_ref)
    assert np.array_equal(last_res.times, times[-1])
    assert np.array_equal(last_res.states, states[-1])


def test_publication_standard_error_uses_sample_std():
    values = np.array([1.0, 2.0, 5.0], dtype=np.float64)
    expected = np.std(values, ddof=1) / np.sqrt(len(values))
    assert _standard_error(values) == expected


def test_publication_standard_error_returns_zero_for_single_observation():
    assert _standard_error([3.5]) == 0.0
