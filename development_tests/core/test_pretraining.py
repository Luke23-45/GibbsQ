import os

import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir

from gibbsq.core.config import load_experiment_config
from gibbsq.core.pretraining import (
    collect_robust_expert_data,
    compute_value_bootstrap_targets,
)


def load_config(name: str, experiment: str = "bc_train"):
    config_dir = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        raw_cfg = compose(config_name=name, overrides=[f"++active_profile={name}"])
        cfg, _ = load_experiment_config(raw_cfg, experiment, profile_name=name)
        return cfg


def test_compute_value_bootstrap_targets_matches_linear_pi_scale():
    queue_totals = jnp.asarray([0.0, 1.5, 3.0], dtype=jnp.float32)
    targets = compute_value_bootstrap_targets(
        queue_totals=queue_totals,
        random_limit=1.5,
        denom=0.5,
    )
    np.testing.assert_allclose(
        np.asarray(targets),
        np.asarray([300.0, 0.0, -300.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_debug_config_value_bootstrap_targets_stay_finite_and_reasonable():
    cfg = load_config("debug")
    service_rates = np.asarray(cfg.system.service_rates, dtype=float)
    _, _, _, queue_totals, _ = collect_robust_expert_data(
        num_servers=len(service_rates),
        service_rates=service_rates,
        samples_per_rho=120,
        seed=cfg.simulation.seed,
    )

    jsq_limit = 1.129862627702106
    random_limit = 1.5
    denom = max(
        cfg.neural_training.perf_index_min_denom,
        jsq_limit * cfg.neural_training.perf_index_jsq_margin,
        random_limit - jsq_limit,
    )
    targets = np.asarray(
        compute_value_bootstrap_targets(
            queue_totals=queue_totals,
            random_limit=random_limit,
            denom=denom,
        )
    )

    assert np.isfinite(targets).all()
    assert float(np.max(np.abs(targets))) < 1.0e4


def test_collect_robust_expert_data_uses_requested_alpha(monkeypatch):
    captured_alphas = []

    class FakeExpert:
        def __init__(self, service_rates, alpha):
            captured_alphas.append(float(alpha))

        def __call__(self, state, rng):
            return np.array([1.0, 0.0], dtype=np.float64)

    class FakeResult:
        states = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)

    monkeypatch.setattr("gibbsq.core.pretraining.UASRouting", FakeExpert)
    monkeypatch.setattr("gibbsq.core.pretraining.simulate", lambda **kwargs: FakeResult())

    collect_robust_expert_data(
        num_servers=2,
        service_rates=np.array([1.0, 2.0], dtype=np.float64),
        rhos=[0.5],
        samples_per_rho=3,
        seed=7,
        alpha=2.5,
    )

    assert captured_alphas
    assert set(captured_alphas) == {2.5}
