"""
SSA-based ablation study for N-GibbsQ.

This replaces the legacy DGA-only ablation path with a REINFORCE-trained,
true-SSA evaluation loop.

Variants:
1. Full Model: theorem-aligned reference architecture
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
from omegaconf import DictConfig, OmegaConf

from gibbsq.analysis.plot_profiles import ExperimentPlotContext
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.core.config import ExperimentConfig, load_experiment_config_chain, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.numpy_engine import run_replications
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.model_io import build_neural_eval_policy
from gibbsq.utils.progress import create_progress
from gibbsq.utils.run_artifacts import artifacts_dir, figure_path, metrics_path
from experiments.training.train_reinforce import ReinforceTrainer

log = logging.getLogger(__name__)


class AblationReinforceTrainer(ReinforceTrainer):
    """REINFORCE trainer variant that does not rewrite global model pointers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_global_pointer = False

    def _save_assets(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        from gibbsq.analysis.plotting import plot_ablation_training_curve

        policy_net = args[0] if len(args) > 0 else kwargs.get('policy_net')
        value_net = args[1] if len(args) > 1 else kwargs.get('value_net')
        history_loss = args[2] if len(args) > 2 else kwargs.get('history_loss')
        history_reward = args[3] if len(args) > 3 else kwargs.get('history_reward')

        artifacts = artifacts_dir(self.run_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        policy_path = artifacts / "n_gibbsq_reinforce_weights.eqx"
        eqx.tree_serialise_leaves(policy_path, policy_net)

        value_path = artifacts / "value_network_weights.eqx"
        eqx.tree_serialise_leaves(value_path, value_net)

        plot_path = figure_path(self.run_dir, "ablation_training_curve")
        fig = plot_ablation_training_curve(
            metrics={
                "epoch": list(range(len(history_loss))),
                "training_loss": history_loss,
                "performance_index": history_reward,
                "variant_label": self.run_dir.name,
                "preprocessing": self.cfg.neural.preprocessing,
                "init_type": self.cfg.neural.init_type,
                "train_epochs": len(history_loss),
            },
            save_path=plot_path,
            theme="publication",
            formats=["png", "pdf"],
            context=ExperimentPlotContext(
                experiment_id="ablation",
                chart_name="plot_ablation_training_curve",
            ),
        )
        plt.close(fig)

        log.info(f"Saved variant artifacts in {self.run_dir}")


class UniformRouting:
    """Uniform routing baseline policy for NumPy SSA engine."""

    def __call__(self, q, rng):
        n = len(q)
        return np.full(n, 1.0 / n, dtype=np.float64)


NEURAL_EVAL_MODE = "deterministic"


def _standard_error(values) -> float:
    """Return sample standard error, guarding the single-observation case."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _build_ablation_eval_policy(model, mu_arr: np.ndarray, rho: float):
    return build_neural_eval_policy(
        model,
        mu_arr,
        rho=rho,
        mode=NEURAL_EVAL_MODE,
    )


def _build_ablation_training_cfg(
    base_cfg: ExperimentConfig,
    resolved_raw_cfg: DictConfig | dict | None,
) -> ExperimentConfig:
    """Return the ablation-local training budget without mutating evaluation config."""
    cfg = copy.deepcopy(base_cfg)
    if resolved_raw_cfg is None:
        return cfg

    raw_data = OmegaConf.to_container(resolved_raw_cfg, resolve=True) if isinstance(resolved_raw_cfg, DictConfig) else resolved_raw_cfg
    if not isinstance(raw_data, dict):
        return cfg

    overrides = raw_data.get("ablation_training")
    if overrides is None:
        return cfg
    if not isinstance(overrides, dict):
        raise ValueError("ablation_training must be a mapping when provided.")

    allowed_keys = {"train_epochs", "batch_size", "simulation", "neural_training"}
    unknown = sorted(set(overrides) - allowed_keys)
    if unknown:
        raise ValueError(f"Unsupported ablation_training override(s): {unknown}")

    if "train_epochs" in overrides:
        cfg.train_epochs = int(overrides["train_epochs"])
    if "batch_size" in overrides:
        cfg.batch_size = int(overrides["batch_size"])

    sim_overrides = overrides.get("simulation")
    if sim_overrides is not None:
        if not isinstance(sim_overrides, dict):
            raise ValueError("ablation_training.simulation must be a mapping when provided.")
        unknown_sim = sorted(set(sim_overrides) - {"ssa"})
        if unknown_sim:
            raise ValueError(f"Unsupported ablation_training.simulation override(s): {unknown_sim}")
        ssa_overrides = sim_overrides.get("ssa")
        if ssa_overrides is not None:
            if not isinstance(ssa_overrides, dict):
                raise ValueError("ablation_training.simulation.ssa must be a mapping when provided.")
            unknown_ssa = sorted(set(ssa_overrides) - {"sim_time"})
            if unknown_ssa:
                raise ValueError(f"Unsupported ablation_training.simulation.ssa override(s): {unknown_ssa}")
            if "sim_time" in ssa_overrides:
                cfg.simulation.ssa.sim_time = float(ssa_overrides["sim_time"])

    neural_training_overrides = overrides.get("neural_training")
    if neural_training_overrides is not None:
        if not isinstance(neural_training_overrides, dict):
            raise ValueError("ablation_training.neural_training must be a mapping when provided.")
        allowed_neural_training = {"eval_batches", "eval_trajs_per_batch", "checkpoint_freq"}
        unknown_nt = sorted(set(neural_training_overrides) - allowed_neural_training)
        if unknown_nt:
            raise ValueError(f"Unsupported ablation_training.neural_training override(s): {unknown_nt}")
        if "eval_batches" in neural_training_overrides:
            cfg.neural_training.eval_batches = int(neural_training_overrides["eval_batches"])
        if "eval_trajs_per_batch" in neural_training_overrides:
            cfg.neural_training.eval_trajs_per_batch = int(neural_training_overrides["eval_trajs_per_batch"])
        if "checkpoint_freq" in neural_training_overrides:
            cfg.neural_training.checkpoint_freq = int(neural_training_overrides["checkpoint_freq"])

    validate(cfg)
    return cfg


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
        progress_desc="ablation eval",
    )
    q_totals = [
        float(time_averaged_queue_lengths(r, cfg.simulation.burn_in_fraction).sum())
        for r in results
    ]
    return {
        "mean_q_total": float(np.mean(q_totals)),
        "se_q_total": _standard_error(q_totals),
    }


def _variant_cfg(base_cfg: ExperimentConfig, preprocessing: str | None, init_type: str | None) -> ExperimentConfig:
    cfg = copy.deepcopy(base_cfg)
    neural_cfg = dataclasses.replace(
        cfg.neural,
        preprocessing="log1p",
        init_type="zero_final",
    )
    if preprocessing is not None:
        neural_cfg = dataclasses.replace(neural_cfg, preprocessing=preprocessing)
    if init_type is not None:
        neural_cfg = dataclasses.replace(neural_cfg, init_type=init_type)
    cfg.neural = neural_cfg
    return cfg


def run_ablation(cfg: ExperimentConfig, run_dir: Path, run_logger=None, resolved_raw_cfg: DictConfig | dict | None = None):
    variants = [
        ("Full Model", None, None),
        ("Ablated: No Log-Norm", "none", None),
        ("Ablated: No Zero-Init", None, "standard"),
    ]

    summary = {}
    training_cfg = _build_ablation_training_cfg(cfg, resolved_raw_cfg)

    with create_progress(total=len(variants) + 1, desc="ablation", unit="variant") as variant_bar:
        for idx, (name, preproc, init_type) in enumerate(variants):
            variant_bar.set_postfix({"variant": name}, refresh=False)
            v_cfg = _variant_cfg(cfg, preproc, init_type)
            trainer_cfg = _variant_cfg(training_cfg, preproc, init_type)
            v_dir = artifacts_dir(run_dir) / f"variant_{idx+1}_{name.lower().replace(' ', '_').replace(':', '')}"
            v_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir(v_dir).mkdir(parents=True, exist_ok=True)

            log.info("-" * 60)
            log.info(f"Training variant: {name}")
            log.info(f"  preprocessing={trainer_cfg.neural.preprocessing}, init_type={trainer_cfg.neural.init_type}")

            trainer = AblationReinforceTrainer(trainer_cfg, v_dir, run_logger=None)
            seed_key = jax.random.PRNGKey(trainer_cfg.simulation.seed + idx)
            trainer.execute(seed_key, n_epochs=trainer_cfg.train_epochs)

            model_path = artifacts_dir(v_dir) / "n_gibbsq_reinforce_weights.eqx"
            skeleton = NeuralRouter(
                num_servers=v_cfg.system.num_servers,
                config=v_cfg.neural,
                service_rates=v_cfg.system.service_rates,
                key=jax.random.PRNGKey(v_cfg.simulation.seed + 10_000 + idx),
            )
            model = eqx.tree_deserialise_leaves(model_path, skeleton)
            mu_arr = np.array(v_cfg.system.service_rates, dtype=np.float64)
            rho = v_cfg.system.arrival_rate / float(mu_arr.sum())
            policy = _build_ablation_eval_policy(model, mu_arr, rho)
            metrics = evaluate_policy_ssa(policy, v_cfg)
            summary[name] = metrics

            append_metrics_jsonl(
                {
                    "variant": name,
                    "preprocessing": v_cfg.neural.preprocessing,
                    "init_type": v_cfg.neural.init_type,
                    **metrics,
                },
                metrics_path(run_dir, "ablation_ssa_metrics.jsonl"),
            )

            log.info(f"  SSA E[Q_total] = {metrics['mean_q_total']:.4f} +/- {metrics['se_q_total']:.4f}")
            variant_bar.update(1)

        uniform_metrics = evaluate_policy_ssa(UniformRouting(), cfg)
        summary["Uniform Routing (Baseline)"] = uniform_metrics
        append_metrics_jsonl(
            {
                "variant": "Uniform Routing (Baseline)",
                "preprocessing": "n/a",
                "init_type": "n/a",
                **uniform_metrics,
            },
            metrics_path(run_dir, "ablation_ssa_metrics.jsonl"),
        )
        variant_bar.update(1)

    names = list(summary.keys())
    values = [summary[n]["mean_q_total"] for n in names]
    se_values = [summary[n]["se_q_total"] for n in names]

    from gibbsq.analysis.plotting import plot_ablation_bars

    plot_path = figure_path(run_dir, "ablation_ssa")
    fig = plot_ablation_bars(
        variant_names=names,
        mean_values=values,
        se_values=se_values,
        save_path=plot_path,
        theme="publication",
        formats=["png", "pdf"],
        context=ExperimentPlotContext(
            experiment_id="ablation",
            chart_name="plot_ablation_bars",
        ),
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    if run_logger:
        try:
            import wandb
            run_logger.log({"ablation_ssa_plot": wandb.Image(str(plot_path.with_suffix(".png")))})
        except Exception:
            pass


def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config_chain(
        raw_cfg,
        ["reinforce_train", "ablation"],
    )

    run_dir, run_id = get_run_config(cfg, "ablation", resolved_raw_cfg)
    run_logger = setup_wandb(cfg, resolved_raw_cfg, default_group="ablation_ssa", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  SSA-Based Ablation Study")
    log.info("=" * 60)

    run_ablation(cfg, run_dir, run_logger, resolved_raw_cfg)

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
