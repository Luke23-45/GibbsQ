import sys
import logging
from pathlib import Path
from hydra import compose, initialize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gibbsq.core.config import (
    PROFILE_CONFIG_NAMES,
    EXPERIMENT_BLOCK_NAMES,
    hydra_to_config,
    validate,
    validate_profile_config,
    resolve_experiment_config,
    resolve_experiment_config_chain,
)
from gibbsq.utils.progress import iter_progress
from scripts.execution.experiment_runner import (
    EXPERIMENTS,
    default_hydra_overrides_for_experiment,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def _discover_root_config_names() -> list[str]:
    return list(PROFILE_CONFIG_NAMES)


PUBLIC_EXPERIMENT_BASE_CONFIGS = {
    experiment_name: "default"
    for experiment_name in EXPERIMENTS
    if experiment_name != "check_configs"
}


def _public_experiment_overrides(experiment_name: str) -> list[str]:
    return default_hydra_overrides_for_experiment(experiment_name, [])


def _resolve_for_validation(raw_cfg, experiment_name: str, profile_name: str):
    if experiment_name == "ablation":
        return resolve_experiment_config_chain(
            raw_cfg,
            ["reinforce_train", "ablation"],
            profile_name=profile_name,
        )
    return resolve_experiment_config(raw_cfg, experiment_name, profile_name=profile_name)

def main():
    root_config_names = _discover_root_config_names()

    failed = False
    with initialize(version_base=None, config_path="../../configs"):
        for name in iter_progress(
            root_config_names,
            total=len(root_config_names),
            desc="check_configs: roots",
            unit="config",
            leave=False,
        ):
            try:
                cfg = compose(config_name=name)
                validate_profile_config(cfg)
                print(f"[OK] Config {name} validated successfully.")
            except Exception as e:
                print(f"[FAIL] Config {name} failed validation: {e}")
                failed = True
        profile_paths = [
            (profile_name, experiment_name)
            for profile_name in root_config_names
            for experiment_name in EXPERIMENT_BLOCK_NAMES
            if experiment_name != "check_configs"
        ]
        for profile_name, experiment_name in iter_progress(
            profile_paths,
            total=len(profile_paths),
            desc="check_configs: resolved paths",
            unit="path",
            leave=False,
        ):
            overrides = [f"++active_profile={profile_name}"] + _public_experiment_overrides(experiment_name)
            try:
                cfg = compose(config_name=profile_name, overrides=overrides)
                validate_profile_config(cfg)
                resolved = _resolve_for_validation(cfg, experiment_name, profile_name=profile_name)
                validated = hydra_to_config(resolved)
                validate(validated)
                print(
                    f"[OK] Resolved experiment path {experiment_name} "
                    f"(profile={profile_name}, overrides={overrides}) validated successfully."
                )
            except Exception as e:
                print(
                    f"[FAIL] Resolved experiment path {experiment_name} "
                    f"(profile={profile_name}, overrides={overrides}) failed validation: {e}"
                )
                failed = True
        public_paths = list(PUBLIC_EXPERIMENT_BASE_CONFIGS.items())
        for experiment_name, base_config in iter_progress(
            public_paths,
            total=len(public_paths),
            desc="check_configs: public paths",
            unit="path",
            leave=False,
        ):
            overrides = [f"++active_profile={base_config}"] + _public_experiment_overrides(experiment_name)
            try:
                cfg = compose(config_name=base_config, overrides=overrides)
                validate_profile_config(cfg)
                resolved = _resolve_for_validation(cfg, experiment_name, profile_name=base_config)
                validated = hydra_to_config(resolved)
                validate(validated)
                print(
                    f"[OK] Public experiment path {experiment_name} "
                    f"(base={base_config}, overrides={overrides}) validated successfully."
                )
            except Exception as e:
                print(
                    f"[FAIL] Public experiment path {experiment_name} "
                    f"(base={base_config}, overrides={overrides}) failed validation: {e}"
                )
                failed = True
    
    if failed:
        print("\n[ERROR] One or more configs failed validation.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All configs passed validation.")

if __name__ == "__main__":
    main()
