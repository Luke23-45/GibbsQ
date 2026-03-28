import sys
import logging
from pathlib import Path
from hydra import compose, initialize
from gibbsq.core.config import hydra_to_config, validate
from gibbsq.utils.progress import iter_progress
from scripts.execution.experiment_runner import (
    EXPERIMENTS,
    default_hydra_overrides_for_experiment,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def _discover_root_config_names() -> list[str]:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    names = sorted(
        path.stem
        for path in config_dir.glob("*.yaml")
        if path.is_file()
    )
    return names


def _discover_experiment_profiles() -> list[str]:
    experiment_dir = Path(__file__).resolve().parents[2] / "configs" / "experiment"
    return sorted(
        path.stem
        for path in experiment_dir.glob("*.yaml")
        if path.is_file()
    )


def _base_config_for_experiment(profile_name: str) -> str:
    if profile_name.endswith("_small"):
        return "small"
    if profile_name.endswith("_large"):
        return "large"
    return "default"


PUBLIC_EXPERIMENT_BASE_CONFIGS = {
    experiment_name: ("drift" if experiment_name == "drift" else "default")
    for experiment_name in EXPERIMENTS
    if experiment_name != "check_configs"
}


def _public_experiment_overrides(experiment_name: str) -> list[str]:
    return default_hydra_overrides_for_experiment(experiment_name, [])

def main():
    root_config_names = _discover_root_config_names()
    experiment_profiles = _discover_experiment_profiles()

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
                validated = hydra_to_config(cfg)
                validate(validated)
                print(f"[OK] Config {name} validated successfully.")
            except Exception as e:
                print(f"[FAIL] Config {name} failed validation: {e}")
                failed = True
        for profile_name in iter_progress(
            experiment_profiles,
            total=len(experiment_profiles),
            desc="check_configs: profiles",
            unit="profile",
            leave=False,
        ):
            base_config = _base_config_for_experiment(profile_name)
            try:
                cfg = compose(config_name=base_config, overrides=[f"+experiment={profile_name}"])
                validated = hydra_to_config(cfg)
                validate(validated)
                print(
                    f"[OK] Experiment profile {profile_name} "
                    f"(base={base_config}) validated successfully."
                )
            except Exception as e:
                print(
                    f"[FAIL] Experiment profile {profile_name} "
                    f"(base={base_config}) failed validation: {e}"
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
            overrides = _public_experiment_overrides(experiment_name)
            try:
                cfg = compose(config_name=base_config, overrides=overrides)
                validated = hydra_to_config(cfg)
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
