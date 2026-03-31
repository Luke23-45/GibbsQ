"""
Unified Behavior Cloning (BC) Pretraining.

This script replaces the legacy 'optimize_bc.py' and 'train_domain_randomized.py'
by providing a clean entry point to the core 'gibbsq.core.pretraining' utility.
It is mapped to 'dr_train' and 'bc_train' in the experiment runner.
"""

import logging
from pathlib import Path
import jax
import numpy as np
from omegaconf import DictConfig

from gibbsq.core.config import ExperimentConfig, load_experiment_config
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.pretraining import extract_bc_data_config, train_robust_bc_policy
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.model_io import BC_POINTER, save_model_pointer, write_bc_reuse_metadata

log = logging.getLogger(__name__)

def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "bc_train")
    bc_data_config = extract_bc_data_config(resolved_raw_cfg)
    
    run_dir, run_id = get_run_config(cfg, "bc_train", resolved_raw_cfg)
    setup_wandb(cfg, resolved_raw_cfg, default_group="pretraining", run_id=run_id, run_dir=run_dir)

    key = jax.random.PRNGKey(cfg.simulation.seed)
    key, actor_key = jax.random.split(key)
    
    policy_net = NeuralRouter(
        num_servers=cfg.system.num_servers,
        config=cfg.neural,
        service_rates=cfg.system.service_rates,
        key=actor_key,
    )
    
    policy_net = train_robust_bc_policy(
        policy_net=policy_net,
        service_rates=np.asarray(cfg.system.service_rates, dtype=np.float32),
        key=key,
        num_steps=cfg.neural_training.bc_num_steps,
        lr=cfg.neural_training.bc_lr,
        weight_decay=cfg.neural_training.weight_decay,
        label_smoothing=cfg.neural_training.bc_label_smoothing,
        seed=cfg.simulation.seed,
        alpha=cfg.system.alpha,
        bc_data_config=bc_data_config,
    )
    
    model_path = run_dir / "n_gibbsq_platinum_bc_weights.eqx"
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, policy_net)
    import equinox as eqx
    eqx.tree_serialise_leaves(model_path, policy_net)
    metadata_path = write_bc_reuse_metadata(
        model_path,
        cfg=cfg,
        bc_data_config=bc_data_config,
    )
    log.info(f"\n[DONE] Platinum BC Weights saved to {model_path}")
    log.info("[Metadata] BC warm-start compatibility metadata saved to %s", metadata_path)
    
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    pointer_dir = run_dir.parent.parent
    save_model_pointer(
        model_path=model_path,
        project_root=_PROJECT_ROOT,
        output_root=pointer_dir,
        pointer_name=BC_POINTER,
    )

if __name__ == "__main__":
    import hydra
    hydra.main(version_base=None, config_path="../../configs", config_name="default")(main)()
