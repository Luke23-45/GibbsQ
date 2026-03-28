import os

import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir

from gibbsq.core.config import hydra_to_config
from gibbsq.core.pretraining import (
    collect_robust_expert_data,
    compute_value_bootstrap_targets,
)


def load_config(name: str):
    config_dir = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return hydra_to_config(compose(config_name=name))


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


def test_fast_config_value_bootstrap_targets_stay_finite_and_reasonable():
    cfg = load_config("fast")
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
