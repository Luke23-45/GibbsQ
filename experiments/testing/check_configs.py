import logging
from hydra import compose, initialize
from gibbsq.core.config import hydra_to_config, validate

logging.basicConfig(level=logging.INFO)

def main():
    initialize(version_base=None, config_path="../../configs")
    for name in ["default", "small", "large"]:
        try:
            cfg = compose(config_name=name)
            validated = hydra_to_config(cfg)
            validate(validated)
            print(f"[OK] Config {name} validated successfully.")
        except Exception as e:
            print(f"[FAIL] Config {name} failed validation: {e}")

if __name__ == "__main__":
    main()
