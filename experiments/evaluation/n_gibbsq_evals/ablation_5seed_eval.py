"""
Multi-seed ablation evaluation script for N-GibbsQ.

This script isolates the best-performing neural ablation variant (BC: Calibrated UAS -> REINFORCE)
and evaluates its end-to-end training stability across 5 independent seeds. This is required to 
establish the expected performance variance and populate the 5-seed mean row in empirical results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import json

import equinox as eqx
import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Path setup to ensure imports work when run as a module or script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gibbsq.core.config import ExperimentConfig, load_experiment_config_chain
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.run_artifacts import artifacts_dir

# Import shared routines from the main ablation script
from experiments.evaluation.n_gibbsq_evals.ablation_ssa import (
    NEURAL_VARIANTS,
    AblationReinforceTrainer,
    _build_ablation_eval_policy,
    _build_ablation_training_cfg,
    _ci95_half_width,
    _standard_error,
    _variant_bc_data_config,
    _variant_cfg,
    evaluate_policy_ssa,
)

log = logging.getLogger(__name__)

def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config_chain(
        raw_cfg,
        ["reinforce_train", "ablation"],
    )

    run_dir, run_id = get_run_config(cfg, "ablation_5seed", resolved_raw_cfg)
    run_logger = setup_wandb(
        cfg,
        resolved_raw_cfg,
        default_group="ablation_5seed",
        run_id=run_id,
        run_dir=run_dir,
    )

    log.info("=" * 60)
    log.info("  5-Seed Neural Ablation Evaluation")
    log.info("=" * 60)

    # We specifically isolate the best variant from NEURAL_VARIANTS.
    # Variant index 3 is: BC from Calibrated UAS -> REINFORCE.
    spec = NEURAL_VARIANTS[3]
    
    log.info("Isolating variant: %s", spec.name)
    
    training_cfg = _build_ablation_training_cfg(cfg, resolved_raw_cfg)
    v_cfg = _variant_cfg(cfg, spec)
    trainer_cfg = _variant_cfg(training_cfg, spec)
    
    # Seeds specifically bounded as 42, 43, 44, 45, 46
    seeds = [42, 43, 44, 45, 46]
    metrics_results = []
    
    for _, seed in enumerate(seeds):
        log.info("-" * 60)
        log.info("Training Seed: %d", seed)
        
        # Create a specific directory for this seed's artifacts
        v_dir = artifacts_dir(run_dir) / f"{spec.artifact_dir}_seed_{seed}"
        v_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir(v_dir).mkdir(parents=True, exist_ok=True)
        
        trainer = AblationReinforceTrainer(
            trainer_cfg,
            v_dir,
            run_logger=None, # Supressing wandb spam for sub-runs
            bc_data_config=_variant_bc_data_config(resolved_raw_cfg, spec),
            bootstrap_mode=spec.bootstrap_mode,
        )
        
        # Train
        seed_key = jax.random.PRNGKey(seed)
        trainer.execute(seed_key, n_epochs=trainer_cfg.train_epochs)
        
        # Evaluate
        model_path = artifacts_dir(v_dir) / "n_gibbsq_reinforce_weights.eqx"
        skeleton = NeuralRouter(
            num_servers=v_cfg.system.num_servers,
            config=v_cfg.neural,
            service_rates=v_cfg.system.service_rates,
            key=jax.random.PRNGKey(seed + 10_000), 
        )
        
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        mu_arr = np.array(v_cfg.system.service_rates, dtype=np.float64)
        rho = v_cfg.system.arrival_rate / float(mu_arr.sum())
        policy = _build_ablation_eval_policy(model, mu_arr, rho)
        
        log.info("Evaluating Seed %d...", seed)
        eval_metrics = evaluate_policy_ssa(policy, v_cfg)
        log.info("Seed %d Mean Q: %.4f", seed, eval_metrics["mean_q_total"])
        
        metrics_results.append(eval_metrics["mean_q_total"])

    log.info("=" * 60)
    
    final_mean = float(np.mean(metrics_results))
    final_se = _standard_error(metrics_results)
    final_ci = _ci95_half_width(metrics_results)
    
    # Delta vs Cal UAS for the ablation context (Cal UAS mean queue was 9.898 in protocol B)
    REFERENCE_CAL_UAS_MEAN = 9.898 
    delta_vs_cal_uas = final_mean - REFERENCE_CAL_UAS_MEAN
    
    log.info("FINAL 5-SEED RESULTS FOR MANUSCRIPT TABLE 7:")
    log.info("  Variant:     %s", spec.name)
    log.info("  5-Seed Mean: %.4f", final_mean)
    log.info("  SE:          %.4f", final_se)
    log.info("  95%% CI:      %.4f", final_ci)
    log.info("  Delta:       %+.4f", delta_vs_cal_uas)
    log.info("=" * 60)
    
    summary_payload = {
        "variant": spec.name,
        "seeds_evaluated": seeds,
        "q_totals": metrics_results,
        "summary": {
            "mean": final_mean,
            "se": final_se,
            "ci95_half_width": final_ci,
            "delta_vs_calibrated_uas": delta_vs_cal_uas
        }
    }
    
    summary_path = artifacts_dir(run_dir) / "ablation_5seed_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    log.info("Saved 5-seed metrics payload to %s", summary_path)
    
    if run_logger:
        run_logger.finish()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        hydra.main(version_base=None, config_path="../../../configs", config_name="default")(main)()
    else:
        from hydra import compose, initialize_config_dir

        config_dir = str(PROJECT_ROOT / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(config_name="default")
            main(raw_cfg)
