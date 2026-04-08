import math

import pytest

from gibbsq.core.config import (
    ExperimentConfig,
    PolicyConfig,
    PolicyName,
    SystemConfig,
    compact_set_radius,
    drift_constant_R,
    drift_rate_epsilon,
    validate,
)


def test_raw_drift_constant_formula():
    cfg = ExperimentConfig(
        system=SystemConfig(
            num_servers=4,
            arrival_rate=2.0,
            service_rates=[1.0, 1.0, 1.0, 1.0],
            alpha=0.5,
        ),
    )

    expected = (2.0 * math.log(4)) / 0.5 + (2.0 + 4.0) / 2.0
    assert drift_constant_R(cfg) == pytest.approx(expected, rel=1e-10)


def test_raw_epsilon_and_compact_set_radius_formulas():
    cfg = ExperimentConfig(
        system=SystemConfig(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=[0.5, 2.0, 3.0],
            alpha=1.0,
        ),
    )

    eps = drift_rate_epsilon(cfg)
    assert eps == pytest.approx(0.5)

    R = drift_constant_R(cfg)
    expected_radius = math.ceil((R + 1.0) / eps)
    assert compact_set_radius(cfg) == expected_radius


def test_uas_archimedean_constants_match_weighted_jensen_closure():
    cfg = ExperimentConfig(
        system=SystemConfig(
            num_servers=3,
            arrival_rate=1.0,
            service_rates=[2.0, 3.0, 4.0],
            alpha=0.25,
        ),
        policy=PolicyConfig(name=PolicyName.UAS.value),
    )
    validate(cfg)

    R = drift_constant_R(cfg)
    eps = drift_rate_epsilon(cfg)
    assert R == pytest.approx((1.0 * 3.0) / 9.0 + 1.5)
    assert eps == pytest.approx((9.0 - 1.0) / 9.0)


def test_uas_compact_set_radius_uses_proved_constants():
    cfg = ExperimentConfig(
        system=SystemConfig(
            num_servers=2,
            arrival_rate=0.9,
            service_rates=[0.8, 1.1],
            alpha=1.3,
        ),
        policy=PolicyConfig(name=PolicyName.UAS.value),
    )
    validate(cfg)

    R = drift_constant_R(cfg)
    eps = drift_rate_epsilon(cfg)
    expected_radius = math.ceil((R + 1.0) / eps)
    assert compact_set_radius(cfg) == expected_radius
