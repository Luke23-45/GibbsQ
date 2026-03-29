"""
SG#1 Regression Test: Domain Randomization phases vs top-level rho bounds.

Confirms that _build_dr_config() does NOT inject the dataclass default
phases when the YAML omits them, so the top-level rho_min/rho_max are
honoured by the training loop.
"""

import pytest

from gibbsq.core.config import (
    DomainRandomizationConfig,
    DomainRandomizationPhase,
    ExperimentConfig,
    _build_dr_config,
    validate,
)


# ── _build_dr_config unit tests ──────────────────────────────


class TestBuildDRConfig:
    """Verify the config builder produces correct phases in both modes."""

    def test_no_phases_in_yaml_returns_empty_phases(self):
        """SG#1 core fix: absent phases → phases=[], NOT the dataclass default."""
        dr = _build_dr_config({"enabled": True, "rho_min": 0.7, "rho_max": 0.98})
        assert dr.phases == [], (
            f"Expected empty phases list when YAML omits phases, "
            f"got {len(dr.phases)} phases: {dr.phases}"
        )
        assert dr.rho_min == 0.7
        assert dr.rho_max == 0.98
        assert dr.enabled is True

    def test_explicit_phases_in_yaml_are_preserved(self):
        """Regression: explicit phases in YAML must be forwarded verbatim."""
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
        """If user explicitly passes phases: [], it must stay empty."""
        dr = _build_dr_config({"enabled": True, "rho_min": 0.4, "rho_max": 0.85, "phases": []})
        assert dr.phases == []
        assert dr.rho_min == 0.4

    def test_disabled_dr_no_phases(self):
        """Disabled DR with no phases should still produce phases=[]."""
        dr = _build_dr_config({"enabled": False, "rho_min": 0.3, "rho_max": 0.7})
        assert dr.phases == []
        assert dr.enabled is False

    def test_empty_dict_produces_empty_phases(self):
        """Edge case: completely empty dict (all defaults) → phases=[]."""
        dr = _build_dr_config({})
        assert dr.phases == []


# ── validate() integration tests ─────────────────────────────


def _make_cfg(**dr_kwargs) -> ExperimentConfig:
    """Build a minimal valid ExperimentConfig with custom DR settings."""
    dr = DomainRandomizationConfig(**dr_kwargs)
    cfg = ExperimentConfig()
    # Fill required MISSING fields
    cfg.system.num_servers = 2
    cfg.system.arrival_rate = 1.0
    cfg.system.service_rates = [1.0, 1.5]
    cfg.system.alpha = 1.0
    cfg.domain_randomization = dr
    return cfg


class TestValidateDR:
    """Verify validate() accepts valid DR configs and rejects invalid ones."""

    def test_valid_top_level_rho_bounds_accepted(self):
        """validate() must accept phases=[] with valid top-level rho bounds."""
        cfg = _make_cfg(enabled=True, rho_min=0.4, rho_max=0.85, phases=[])
        validate(cfg)  # Should not raise

    def test_valid_explicit_phases_accepted(self):
        """validate() must accept explicit phases with valid rho ranges."""
        phases = [DomainRandomizationPhase(rho_min=0.5, rho_max=0.8, epochs=10)]
        cfg = _make_cfg(enabled=True, phases=phases)
        validate(cfg)  # Should not raise

    def test_invalid_top_level_rho_min_ge_rho_max_rejected(self):
        """validate() must reject rho_min >= rho_max in top-level mode."""
        cfg = _make_cfg(enabled=True, rho_min=0.9, rho_max=0.5, phases=[])
        with pytest.raises(ValueError, match="invalid rho range"):
            validate(cfg)

    def test_invalid_top_level_rho_min_zero_rejected(self):
        """validate() must reject rho_min <= 0 in top-level mode."""
        cfg = _make_cfg(enabled=True, rho_min=0.0, rho_max=0.5, phases=[])
        with pytest.raises(ValueError, match="invalid rho range"):
            validate(cfg)

    def test_invalid_top_level_rho_max_one_rejected(self):
        """validate() must reject rho_max >= 1.0 in top-level mode."""
        cfg = _make_cfg(enabled=True, rho_min=0.4, rho_max=1.0, phases=[])
        with pytest.raises(ValueError, match="invalid rho range"):
            validate(cfg)

    def test_invalid_phase_rho_rejected(self):
        """validate() must reject a phase with rho_min >= rho_max."""
        phases = [DomainRandomizationPhase(rho_min=0.9, rho_max=0.5, epochs=10)]
        cfg = _make_cfg(enabled=True, phases=phases)
        with pytest.raises(ValueError, match="invalid rho range"):
            validate(cfg)

    def test_disabled_dr_skips_all_validation(self):
        """validate() must not check rho bounds when DR is disabled."""
        cfg = _make_cfg(enabled=False, rho_min=999.0, rho_max=-1.0, phases=[])
        validate(cfg)  # Should not raise — DR is disabled
