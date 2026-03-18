"""
Unified Behavior Cloning (BC) Pretraining.

This script replaces the legacy 'optimize_bc.py' and 'train_domain_randomized.py'
by providing a clean entry point to the core 'gibbsq.core.pretraining' utility.
It is mapped to 'dr_train' and 'bc_train' in the experiment runner.
"""

import logging
from pathlib import Path
import jax
from omegaconf import DictConfig

from gibbsq.core.config import ExperimentConfig, hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.pretraining import train_robust_bc_policy
from gibbsq.utils.logging import setup_wandb, get_run_config

log = logging.getLogger(__name__)

def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)
    
    # Use standard run config
    run_dir, run_id = get_run_config(cfg, "platinum_pretraining", raw_cfg)
    setup_wandb(cfg, raw_cfg, default_group="pretraining", run_id=run_id, run_dir=run_dir)
    
    # Initialize policy
    key = jax.random.PRNGKey(cfg.simulation.seed)
    key, actor_key = jax.random.split(key)
    
    policy_net = NeuralRouter(
        num_servers=cfg.system.num_servers,
        config=cfg.neural,
        key=actor_key,
    )
    
    # Execute robust training
    policy_net = train_robust_bc_policy(
        policy_net=policy_net,
        service_rates=cfg.system.service_rates,
        key=key,
        num_steps=1000,
        lr=0.002,
        weight_decay=1e-4,
        label_smoothing=0.1
    )
    
    # Save assets
    model_path = run_dir / "n_gibbsq_platinum_bc_weights.eqx"
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, policy_net)
    import equinox as eqx
    eqx.tree_serialise_leaves(model_path, policy_net)
    log.info(f"\n[DONE] Platinum BC Weights saved to {model_path}")
    
    # Update pointer for evaluation
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    pointer_dir = run_dir.parent.parent
    pointer_dir.mkdir(parents=True, exist_ok=True)
    
    ptr_path = pointer_dir / "latest_domain_randomized_weights.txt"
    relative_path = model_path.resolve().relative_to(_PROJECT_ROOT)
    with open(ptr_path, "w", encoding='utf-8') as f:
        f.write(str(relative_path))
    log.info(f"[Pointer] Updated latest weights pointer at {ptr_path}")

if __name__ == "__main__":
    import hydra
    hydra.main(version_base=None, config_path="../../configs", config_name="default")(main)()
