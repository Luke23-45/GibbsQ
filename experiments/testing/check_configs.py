import sys
import logging
from hydra import compose, initialize
from gibbsq.core.config import hydra_to_config, validate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    initialize(version_base=None, config_path="../../configs")
    
    failed = False
    for name in ["default", "small", "large"]:
        try:
            cfg = compose(config_name=name)
            validated = hydra_to_config(cfg)
            validate(validated)
            print(f"[OK] Config {name} validated successfully.")
        except Exception as e:
            print(f"[FAIL] Config {name} failed validation: {e}")
            failed = True
    
    if failed:
        print("\n[ERROR] One or more configs failed validation.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All configs passed validation.")

if __name__ == "__main__":
    main()
