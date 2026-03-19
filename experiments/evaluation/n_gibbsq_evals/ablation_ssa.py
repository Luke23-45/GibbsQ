"""
SSA-based ablation study for N-GibbsQ.

This replaces the legacy DGA-only ablation path with a REINFORCE-trained,
true-SSA evaluation loop.

Variants:
1. Full Model: default neural config
2. Ablated: No Log-Norm (preprocessing='none')
3. Ablated: No Zero-Init (init_type='standard')
4. Uniform Routing baseline
"""

import copy
import dataclasses
import logging
from pathlib import Path

import equinox as eqx
import hydra
import jax
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig

from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.core.config import ExperimentConfig, hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.numpy_engine import run_replications
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.model_io import StochasticNeuralPolicy
from experiments.training.train_reinforce import ReinforceTrainer

log = logging.getLogger(__name__)


class AblationReinforceTrainer(ReinforceTrainer):
    """REINFORCE trainer variant that does not rewrite global model pointers."""

    def _save_assets(self, policy_net, value_net, history_loss, history_reward):
        import matplotlib.pyplot as plt

        policy_path = self.run_dir / "n_gibbsq_reinforce_weights.eqx"
        eqx.tree_serialise_leaves(policy_path, policy_net)

        value_path = self.run_dir / "value_network_weights.eqx"
        eqx.tree_serialise_leaves(value_path, value_net)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(history_loss, color='blue', linewidth=2)
        axes[0].set_title('REINFORCE Policy Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history_reward, color='green', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title('Mean Reward (-E[Q])')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reward')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.run_dir / "reinforce_training_curve.png", dpi=300)
        plt.close()

        log.info(f"Saved variant artifacts in {self.run_dir}")


class UniformRouting:
    """Uniform routing baseline policy for NumPy SSA engine."""

    def __call__(self, q, rng):
        n = len(q)
        return np.full(n, 1.0 / n, dtype=np.float64)


# NeuralPolicyWrapper moved to gibbsq.utils.model_io.StochasticNeuralPolicy
NeuralPolicyWrapper = StochasticNeuralPolicy  # Backward-compat alias


def evaluate_policy_ssa(policy, cfg: ExperimentConfig) -> dict:
    """Evaluate a policy on true SSA replications."""
    results = run_replications(
        num_servers=cfg.system.num_servers,
        arrival_rate=cfg.system.arrival_rate,
        service_rates=np.array(cfg.system.service_rates, dtype=np.float64),
        policy=policy,
        num_replications=cfg.simulation.num_replications,
        sim_time=cfg.simulation.ssa.sim_time,
        sample_interval=cfg.simulation.ssa.sample_interval,
        base_seed=cfg.simulation.seed,
    )
    q_totals = [
        float(time_averaged_queue_lengths(r, cfg.simulation.burn_in_fraction).sum())
        for r in results
    ]
    return {
        "mean_q_total": float(np.mean(q_totals)),
        "se_q_total": float(np.std(q_totals) / np.sqrt(len(q_totals))),
    }


def _variant_cfg(base_cfg: ExperimentConfig, preprocessing: str | None, init_type: str | None) -> ExperimentConfig:
    cfg = copy.deepcopy(base_cfg)
    neural_cfg = cfg.neural
    if preprocessing is not None:
        neural_cfg = dataclasses.replace(neural_cfg, preprocessing=preprocessing)
    if init_type is not None:
        neural_cfg = dataclasses.replace(neural_cfg, init_type=init_type)
    cfg.neural = neural_cfg
    return cfg


def run_ablation(cfg: ExperimentConfig, run_dir: Path, run_logger=None):
    variants = [
        ("Full Model", None, None),
        ("Ablated: No Log-Norm", "none", None),
        ("Ablated: No Zero-Init", None, "standard"),
    ]

    summary = {}

    for idx, (name, preproc, init_type) in enumerate(variants):
        v_cfg = _variant_cfg(cfg, preproc, init_type)
        v_dir = run_dir / f"variant_{idx+1}_{name.lower().replace(' ', '_').replace(':', '')}"
        v_dir.mkdir(parents=True, exist_ok=True)

        log.info("-" * 60)
        log.info(f"Training variant: {name}")
        log.info(f"  preprocessing={v_cfg.neural.preprocessing}, init_type={v_cfg.neural.init_type}")

        trainer = AblationReinforceTrainer(v_cfg, v_dir, run_logger=None)
        seed_key = jax.random.PRNGKey(v_cfg.simulation.seed + idx)
        trainer.execute(seed_key, n_epochs=v_cfg.train_epochs)

        model_path = v_dir / "n_gibbsq_reinforce_weights.eqx"
        skeleton = NeuralRouter(
            num_servers=v_cfg.system.num_servers,
            config=v_cfg.neural,
            key=jax.random.PRNGKey(v_cfg.simulation.seed + 10_000 + idx),
        )
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        policy = NeuralPolicyWrapper(model, np.array(v_cfg.system.service_rates, dtype=np.float64))
        metrics = evaluate_policy_ssa(policy, v_cfg)
        summary[name] = metrics

        append_metrics_jsonl(
            {
                "variant": name,
                "preprocessing": v_cfg.neural.preprocessing,
                "init_type": v_cfg.neural.init_type,
                **metrics,
            },
            run_dir / "ablation_ssa_metrics.jsonl",
        )

        log.info(f"  SSA E[Q_total] = {metrics['mean_q_total']:.4f} +/- {metrics['se_q_total']:.4f}")

    uniform_metrics = evaluate_policy_ssa(UniformRouting(), cfg)
    summary["Uniform Routing (Baseline)"] = uniform_metrics
    append_metrics_jsonl(
        {
            "variant": "Uniform Routing (Baseline)",
            "preprocessing": "n/a",
            "init_type": "n/a",
            **uniform_metrics,
        },
        run_dir / "ablation_ssa_metrics.jsonl",
    )

    names = list(summary.keys())
    values = [summary[n]["mean_q_total"] for n in names]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(names, values, color=['#2ecc71', '#e67e22', '#e74c3c', '#95a5a6'])
    plt.title('N-GibbsQ SSA Ablation Study')
    plt.ylabel('Expected Total Queue Length E[Q_total]')
    plt.xticks(rotation=15)
    plt.grid(True, axis='y', alpha=0.3)
    for bar, name in zip(bars, names):
        y = bar.get_height()
        se = summary[name]["se_q_total"]
        plt.text(bar.get_x() + bar.get_width() / 2, y + 0.5, f"{y:.2f}\n±{se:.2f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plot_path = run_dir / "ablation_ssa.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    if run_logger:
        run_logger.log({"ablation_ssa_plot": str(plot_path)})


def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "ablation_ssa", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="ablation_ssa", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  SSA-Based Ablation Study")
    log.info("=" * 60)

    run_ablation(cfg, run_dir, run_logger)

    if run_logger:
        run_logger.finish()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        hydra.main(version_base=None, config_path="../../../configs", config_name="default")(main)()
    else:
        from hydra import compose, initialize_config_dir

        config_dir = str(Path(__file__).resolve().parents[3] / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(config_name="default")
            main(raw_cfg)
