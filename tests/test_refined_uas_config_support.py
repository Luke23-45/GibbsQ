from pathlib import Path

from omegaconf import OmegaConf

from gibbsq.core.builders import build_policy_by_name
from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.jax_engine import policy_name_to_type


def test_jax_engine_supports_refined_uas_policy_name():
    assert policy_name_to_type("refined_uas") == 7


def test_all_profile_configs_validate_with_refined_uas_override():
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "configs"

    for profile_name in ("debug", "small", "default", "final_experiment"):
        raw = OmegaConf.load(config_dir / f"{profile_name}.yaml")
        raw.policy.name = "refined_uas"
        cfg = hydra_to_config(raw)
        validate(cfg)

        policy = build_policy_by_name(
            cfg.policy.name,
            alpha=20.0,
            mu=cfg.system.service_rates,
            d=cfg.policy.d,
        )
        assert policy.__class__.__name__ == "RefinedUASRouting"
