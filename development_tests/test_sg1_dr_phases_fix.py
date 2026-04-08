import pytest

from gibbsq.core.config import (
    DomainRandomizationConfig,
    DomainRandomizationPhase,
    ExperimentConfig,
    _build_dr_config,
    validate,
)

class TestBuildDRConfig:
    def test_no_phases_in_yaml_returns_empty_phases(self):
        dr = _build_dr_config({"enabled": True, "rho_min": 0.7, "rho_max": 0.98})
        assert dr.phases == [], (
            f"Expected empty phases list when YAML omits phases, "
            f"got {len(dr.phases)} phases: {dr.phases}"
        )
        assert dr.rho_min == 0.7
        assert dr.rho_max == 0.98
        assert dr.enabled is True

    def test_explicit_phases_in_yaml_are_preserved(self):
        phases_raw = [
            {"rho_min": 0.5, "rho_max": 0.8, "epochs": 10, "horizon": 500},
            {"rho_min": 0.6, "rho_max": 0.9, "epochs": 20, "horizon": 1000},
        ]
        dr = _build_dr_config({"enabled": True, "phases": phases_raw})
        assert len(dr.phases) == 2
        assert dr.phases[0].rho_min == 0.5
        assert dr.phases[0].rho_max == 0.8
        assert dr.phases[1].epochs == 20

    def test_empty_phases_list_in_yaml_stays_empty(self):
        dr = _build_dr_config({"enabled": True, "rho_min": 0.4, "rho_max": 0.85, "phases": []})
        assert dr.phases == []
        assert dr.rho_min == 0.4

    def test_disabled_dr_no_phases(self):
        dr = _build_dr_config({"enabled": False, "rho_min": 0.3, "rho_max": 0.7})
        assert dr.phases == []
        assert dr.enabled is False

    def test_empty_dict_produces_empty_phases(self):
        dr = _build_dr_config({})
        assert dr.phases == []

def _make_cfg(**dr_kwargs) -> ExperimentConfig:
    dr = DomainRandomizationConfig(**dr_kwargs)
    cfg = ExperimentConfig()
    cfg.system.num_servers = 2
    cfg.system.arrival_rate = 1.0
    cfg.system.service_rates = [1.0, 1.5]
    cfg.system.alpha = 1.0
    cfg.domain_randomization = dr
    return cfg

class TestValidateDR:
    def test_valid_top_level_rho_bounds_accepted(self):
        cfg = _make_cfg(enabled=True, rho_min=0.4, rho_max=0.85, phases=[])
        validate(cfg)

    def test_valid_explicit_phases_accepted(self):
        phases = [DomainRandomizationPhase(rho_min=0.5, rho_max=0.8, epochs=10)]
        cfg = _make_cfg(enabled=True, phases=phases)
        validate(cfg)

    def test_invalid_top_level_rho_min_ge_rho_max_rejected(self):
        cfg = _make_cfg(enabled=True, rho_min=0.9, rho_max=0.5, phases=[])
        with pytest.raises(ValueError, match="invalid rho range"):
            validate(cfg)

    def test_invalid_top_level_rho_min_zero_rejected(self):
        cfg = _make_cfg(enabled=True, rho_min=0.0, rho_max=0.5, phases=[])
        with pytest.raises(ValueError, match="invalid rho range"):
            validate(cfg)

    def test_invalid_top_level_rho_max_one_rejected(self):
        cfg = _make_cfg(enabled=True, rho_min=0.4, rho_max=1.0, phases=[])
        with pytest.raises(ValueError, match="invalid rho range"):
            validate(cfg)

    def test_invalid_phase_rho_rejected(self):
        phases = [DomainRandomizationPhase(rho_min=0.9, rho_max=0.5, epochs=10)]
        cfg = _make_cfg(enabled=True, phases=phases)
        with pytest.raises(ValueError, match="invalid rho range"):
            validate(cfg)

    def test_disabled_dr_skips_all_validation(self):
        cfg = _make_cfg(enabled=False, rho_min=999.0, rho_max=-1.0, phases=[])
        validate(cfg)
