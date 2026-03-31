import logging

import hydra
from omegaconf import DictConfig

from gibbsq.core.config import load_experiment_config
from gibbsq.studies.hyperparameter_qualification import (
    _write_json,
    normalize_study_config,
    run_stage,
)
from gibbsq.utils.logging import get_run_config

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "hyperqual")
    study_cfg, preset = normalize_study_config(resolved_raw_cfg.get("study"))

    run_dir, _ = get_run_config(cfg, "hyperqual", resolved_raw_cfg)
    _write_json(run_dir / "study_config.json", {"study": study_cfg.__dict__, "preset": preset})

    log.info("=" * 60)
    log.info("  Hyperparameter Qualification Study")
    log.info("=" * 60)
    log.info(f"Requested stage: {study_cfg.stage}")
    log.info(f"Execution profile: {preset['profile_name']}")
    log.info(f"Mode: {study_cfg.mode}")

    return run_stage(
        stage=study_cfg.stage,
        study_cfg=study_cfg,
        preset=preset,
        hyperqual_run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
