"""
SSA-based empirical ablation study for N-GibbsQ.

This ablation is framed as empirical model selection, not theorem validation.
It compares neural design choices against the strongest closed-form baselines:

- JSSQ
- Calibrated UAS
- best legacy UAS

and against a compact matrix of neural training variants.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import hydra
import jax
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig, OmegaConf

from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.analysis.plot_profiles import ExperimentPlotContext
from gibbsq.analysis.plotting import plot_ablation_bars
from gibbsq.core.config import ExperimentConfig, load_experiment_config_chain, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.policies import CalibratedUASRouting, JSSQRouting, UASRouting
from gibbsq.engines.numpy_engine import run_replications
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.model_io import build_neural_eval_policy
from gibbsq.utils.progress import create_progress
from gibbsq.utils.run_artifacts import artifacts_dir, figure_path, metrics_path
from experiments.training.train_reinforce import ReinforceTrainer

log = logging.getLogger(__name__)

NEURAL_EVAL_MODE = "deterministic"
LEGACY_UAS_ALPHA = 10.0
CI_Z_SCORE = 1.96


@dataclass(frozen=True)
class AblationVariantSpec:
    name: str
    variant_kind: str
    panel: str
    preprocessing: str | None = None
    init_type: str | None = None
    bootstrap_mode: str = "expert"
    expert_policy_name: str | None = None
    expert_policy_params: dict | None = None
    artifact_dir: str | None = None


NEURAL_VARIANTS: list[AblationVariantSpec] = [
    AblationVariantSpec(
        name="No Log-Norm",
        variant_kind="neural",
        panel="architecture",
        preprocessing="none",
        init_type="standard",
        bootstrap_mode="expert",
        expert_policy_name="uas",
        expert_policy_params={"alpha": LEGACY_UAS_ALPHA},
        artifact_dir="variant_1_no_log_norm",
    ),
    AblationVariantSpec(
        name="Zero-Init Final",
        variant_kind="neural",
        panel="architecture",
        preprocessing="log1p",
        init_type="zero_final",
        bootstrap_mode="expert",
        expert_policy_name="uas",
        expert_policy_params={"alpha": LEGACY_UAS_ALPHA},
        artifact_dir="variant_2_zero_init_final",
    ),
    AblationVariantSpec(
        name="BC from UAS -> REINFORCE",
        variant_kind="neural",
        panel="teacher",
        preprocessing="log1p",
        init_type="standard",
        bootstrap_mode="expert",
        expert_policy_name="uas",
        expert_policy_params={"alpha": LEGACY_UAS_ALPHA},
        artifact_dir="variant_3_bc_from_uas_to_reinforce",
    ),
    AblationVariantSpec(
        name="BC from Calibrated UAS -> REINFORCE",
        variant_kind="neural",
        panel="teacher",
        preprocessing="log1p",
        init_type="standard",
        bootstrap_mode="expert",
        expert_policy_name="calibrated_uas",
        expert_policy_params={"alpha": 20.0, "beta": 0.85, "gamma": 0.5, "c": 0.5},
        artifact_dir="variant_4_bc_from_calibrated_uas_to_reinforce",
    ),
    AblationVariantSpec(
        name="REINFORCE from Scratch",
        variant_kind="neural",
        panel="architecture",
        preprocessing="log1p",
        init_type="standard",
        bootstrap_mode="scratch",
        expert_policy_name=None,
        expert_policy_params=None,
        artifact_dir="variant_5_reinforce_from_scratch",
    ),
]

REFERENCE_VARIANTS: list[AblationVariantSpec] = [
    AblationVariantSpec(name="JSSQ", variant_kind="reference", panel="teacher"),
    AblationVariantSpec(name="Calibrated UAS", variant_kind="reference", panel="teacher"),
    AblationVariantSpec(name=f"UAS (alpha={LEGACY_UAS_ALPHA:.1f})", variant_kind="reference", panel="teacher"),
]

ALL_VARIANTS: list[AblationVariantSpec] = [*NEURAL_VARIANTS, *REFERENCE_VARIANTS]


class AblationReinforceTrainer(ReinforceTrainer):
    """REINFORCE trainer variant that does not rewrite global model pointers."""

    def __init__(self, *args, bootstrap_mode: str = "expert", **kwargs):
        super().__init__(*args, **kwargs)
        self.save_global_pointer = False
        self.bootstrap_mode = bootstrap_mode

    def bootstrap_from_expert(self, policy_net, value_net, key, jsq_limit, random_limit, denom):
        if self.bootstrap_mode == "scratch":
            self.last_warm_start_meta = {
                "source": "scratch",
                "pointer_path": None,
                "model_path": None,
                "loaded": False,
                "reason": "bootstrap_disabled",
            }
            return policy_net, value_net
        return super().bootstrap_from_expert(policy_net, value_net, key, jsq_limit, random_limit, denom)

    def _save_assets(self, *args, **kwargs):
        from gibbsq.analysis.plotting import plot_ablation_training_curve

        policy_net = args[0] if len(args) > 0 else kwargs.get("policy_net")
        value_net = args[1] if len(args) > 1 else kwargs.get("value_net")
        history_loss = args[2] if len(args) > 2 else kwargs.get("history_loss")
        history_reward = args[3] if len(args) > 3 else kwargs.get("history_reward")

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
        log.info("Saved variant artifacts in %s", self.run_dir)


def variant_catalog() -> list[dict[str, str]]:
    """Return a lightweight public summary of the ablation study design."""
    return [
        {
            "name": spec.name,
            "variant_kind": spec.variant_kind,
            "panel": spec.panel,
            "bootstrap_mode": spec.bootstrap_mode,
            "expert_policy_name": spec.expert_policy_name or "none",
        }
        for spec in ALL_VARIANTS
    ]


def _standard_error(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _ci95_half_width(values) -> float:
    return CI_Z_SCORE * _standard_error(values)


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
    cfg = copy.deepcopy(base_cfg)
    if resolved_raw_cfg is None:
        return cfg

    raw_data = (
        OmegaConf.to_container(resolved_raw_cfg, resolve=True)
        if isinstance(resolved_raw_cfg, DictConfig)
        else resolved_raw_cfg
    )
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


def _variant_cfg(base_cfg: ExperimentConfig, spec: AblationVariantSpec) -> ExperimentConfig:
    cfg = copy.deepcopy(base_cfg)
    neural_cfg = dataclasses.replace(
        cfg.neural,
        preprocessing=spec.preprocessing or cfg.neural.preprocessing,
        init_type=spec.init_type or cfg.neural.init_type,
    )
    cfg.neural = neural_cfg
    return cfg


def _variant_bc_data_config(resolved_raw_cfg: DictConfig | dict | None, spec: AblationVariantSpec) -> dict | None:
    if spec.expert_policy_name is None:
        return None
    if isinstance(resolved_raw_cfg, DictConfig):
        raw_data = OmegaConf.to_container(resolved_raw_cfg, resolve=True)
    else:
        raw_data = resolved_raw_cfg
    base_bc = {}
    if isinstance(raw_data, dict):
        candidate = raw_data.get("bc_data")
        if isinstance(candidate, dict):
            base_bc = dict(candidate)
    base_bc["expert_policy_name"] = spec.expert_policy_name
    if spec.expert_policy_params:
        base_bc["expert_policy_params"] = dict(spec.expert_policy_params)
    return base_bc


def evaluate_policy_ssa(policy, cfg: ExperimentConfig) -> dict:
    """Evaluate a policy on true SSA replications and retain paired raw results."""
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
        "ci95_half_width": _ci95_half_width(q_totals),
        "q_totals": q_totals,
    }


def _reference_policy(spec: AblationVariantSpec, cfg: ExperimentConfig):
    mu_arr = np.array(cfg.system.service_rates, dtype=np.float64)
    if spec.name == "JSSQ":
        return JSSQRouting(mu_arr)
    if spec.name == "Calibrated UAS":
        return CalibratedUASRouting(mu_arr, alpha=20.0, beta=0.85, gamma=0.5, c=0.5)
    if spec.name.startswith("UAS"):
        return UASRouting(mu_arr, alpha=LEGACY_UAS_ALPHA)
    raise ValueError(f"Unsupported reference policy '{spec.name}'")


def _compute_delta_metrics(candidate_values: list[float], reference_values: list[float]) -> dict[str, float]:
    cand = np.asarray(candidate_values, dtype=np.float64)
    ref = np.asarray(reference_values, dtype=np.float64)
    if cand.shape != ref.shape:
        raise ValueError("Paired delta metrics require equal-length replication arrays.")
    deltas = cand - ref
    return {
        "mean": float(np.mean(deltas)),
        "se": _standard_error(deltas.tolist()),
        "ci95_half_width": _ci95_half_width(deltas.tolist()),
    }


def _summary_row(
    spec: AblationVariantSpec,
    metrics: dict,
    *,
    delta_vs_calibrated_uas: dict[str, float] | None,
    delta_vs_best_neural: dict[str, float] | None,
) -> dict:
    row = {
        "variant": spec.name,
        "variant_kind": spec.variant_kind,
        "panel": spec.panel,
        "preprocessing": spec.preprocessing or "n/a",
        "init_type": spec.init_type or "n/a",
        "bootstrap_mode": spec.bootstrap_mode,
        "teacher_policy": spec.expert_policy_name or "n/a",
        "mean_q_total": float(metrics["mean_q_total"]),
        "se_q_total": float(metrics["se_q_total"]),
        "ci95_half_width": float(metrics["ci95_half_width"]),
    }
    if delta_vs_calibrated_uas is not None:
        row.update(
            {
                "delta_vs_calibrated_uas_mean": delta_vs_calibrated_uas["mean"],
                "delta_vs_calibrated_uas_se": delta_vs_calibrated_uas["se"],
                "delta_vs_calibrated_uas_ci95_half_width": delta_vs_calibrated_uas["ci95_half_width"],
            }
        )
    if delta_vs_best_neural is not None:
        row.update(
            {
                "delta_vs_best_neural_mean": delta_vs_best_neural["mean"],
                "delta_vs_best_neural_se": delta_vs_best_neural["se"],
                "delta_vs_best_neural_ci95_half_width": delta_vs_best_neural["ci95_half_width"],
            }
        )
    return row


def _write_summary_artifacts(run_dir: Path, summary_rows: list[dict]) -> None:
    summary_path = metrics_path(run_dir, "ablation_ssa_summary.json")
    ranking = sorted(summary_rows, key=lambda row: row["mean_q_total"])
    summary_payload = {
        "variants": summary_rows,
        "ranking": [
            {
                "rank": idx + 1,
                "variant": row["variant"],
                "variant_kind": row["variant_kind"],
                "mean_q_total": row["mean_q_total"],
            }
            for idx, row in enumerate(ranking)
        ],
        "best_neural_variant": next(
            (
                row["variant"]
                for row in ranking
                if row["variant_kind"] == "neural"
            ),
            None,
        ),
        "best_reference_variant": next(
            (
                row["variant"]
                for row in ranking
                if row["variant_kind"] == "reference"
            ),
            None,
        ),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def _plot_delta_vs_calibrated_uas(summary_rows: list[dict], run_dir: Path) -> None:
    names = [row["variant"] for row in summary_rows]
    delta_means = [float(row.get("delta_vs_calibrated_uas_mean", 0.0)) for row in summary_rows]
    delta_cis = [float(row.get("delta_vs_calibrated_uas_ci95_half_width", 0.0)) for row in summary_rows]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    colors = ["#009E73" if value <= 0 else "#D55E00" for value in delta_means]
    ax.bar(x, delta_means, yerr=delta_cis, color=colors, capsize=4, edgecolor="#333333", alpha=0.9)
    ax.axhline(0.0, color="#444444", linewidth=1.1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right")
    ax.set_ylabel(r"$\Delta \mathbb{E}[Q_{total}]$ vs Calibrated UAS")
    ax.set_title("Ablation Study: Paired Delta vs Calibrated UAS")
    ax.grid(True, axis="y", linestyle=(0, (3, 3)), alpha=0.25)
    fig.tight_layout()
    save_path = figure_path(run_dir, "ablation_delta_vs_calibrated_uas")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_ablation(
    cfg: ExperimentConfig,
    run_dir: Path,
    run_logger=None,
    resolved_raw_cfg: DictConfig | dict | None = None,
):
    summary_metrics: dict[str, dict] = {}
    training_cfg = _build_ablation_training_cfg(cfg, resolved_raw_cfg)

    total_steps = len(NEURAL_VARIANTS) + len(REFERENCE_VARIANTS)
    with create_progress(total=total_steps, desc="ablation", unit="variant") as variant_bar:
        for idx, spec in enumerate(NEURAL_VARIANTS):
            variant_bar.set_postfix({"variant": spec.name}, refresh=False)
            v_cfg = _variant_cfg(cfg, spec)
            trainer_cfg = _variant_cfg(training_cfg, spec)
            v_dir = artifacts_dir(run_dir) / (spec.artifact_dir or f"variant_{idx + 1}")
            v_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir(v_dir).mkdir(parents=True, exist_ok=True)

            log.info("-" * 60)
            log.info("Training variant: %s", spec.name)
            log.info(
                "  preprocessing=%s, init_type=%s, bootstrap_mode=%s, expert_policy=%s",
                trainer_cfg.neural.preprocessing,
                trainer_cfg.neural.init_type,
                spec.bootstrap_mode,
                spec.expert_policy_name or "none",
            )

            trainer = AblationReinforceTrainer(
                trainer_cfg,
                v_dir,
                run_logger=None,
                bc_data_config=_variant_bc_data_config(resolved_raw_cfg, spec),
                bootstrap_mode=spec.bootstrap_mode,
            )
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
            summary_metrics[spec.name] = metrics
            variant_bar.update(1)

        for spec in REFERENCE_VARIANTS:
            variant_bar.set_postfix({"variant": spec.name}, refresh=False)
            log.info("-" * 60)
            log.info("Evaluating reference: %s", spec.name)
            metrics = evaluate_policy_ssa(_reference_policy(spec, cfg), cfg)
            summary_metrics[spec.name] = metrics
            variant_bar.update(1)

    calibrated_metrics = summary_metrics["Calibrated UAS"]
    best_neural_name = min(
        (spec.name for spec in NEURAL_VARIANTS),
        key=lambda name: summary_metrics[name]["mean_q_total"],
    )
    best_neural_metrics = summary_metrics[best_neural_name]

    summary_rows: list[dict] = []
    metrics_jsonl_path = metrics_path(run_dir, "ablation_ssa_metrics.jsonl")
    if metrics_jsonl_path.exists():
        metrics_jsonl_path.unlink()

    for spec in ALL_VARIANTS:
        metrics = summary_metrics[spec.name]
        delta_vs_calibrated = _compute_delta_metrics(metrics["q_totals"], calibrated_metrics["q_totals"])
        delta_vs_best_neural = _compute_delta_metrics(metrics["q_totals"], best_neural_metrics["q_totals"])
        row = _summary_row(
            spec,
            metrics,
            delta_vs_calibrated_uas=delta_vs_calibrated,
            delta_vs_best_neural=delta_vs_best_neural,
        )
        summary_rows.append(row)
        append_metrics_jsonl(row, metrics_jsonl_path)
        log.info(
            "  %-30s E[Q_total] = %.4f +/- %.4f | delta vs Calibrated UAS = %+0.4f",
            spec.name,
            row["mean_q_total"],
            row["se_q_total"],
            row["delta_vs_calibrated_uas_mean"],
        )

    _write_summary_artifacts(run_dir, summary_rows)

    names = [row["variant"] for row in summary_rows]
    values = [row["mean_q_total"] for row in summary_rows]
    ci_values = [row["ci95_half_width"] for row in summary_rows]
    plot_path = figure_path(run_dir, "ablation_ssa")
    fig = plot_ablation_bars(
        variant_names=names,
        mean_values=values,
        se_values=ci_values,
        save_path=plot_path,
        theme="publication",
        formats=["png", "pdf"],
        context=ExperimentPlotContext(
            experiment_id="ablation",
            chart_name="plot_ablation_bars",
        ),
    )
    plt.close(fig)

    _plot_delta_vs_calibrated_uas(summary_rows, run_dir)

    if run_logger:
        try:
            import wandb

            run_logger.log(
                {
                    "ablation_ssa_plot": wandb.Image(str(plot_path.with_suffix(".png"))),
                    "best_neural_variant": best_neural_name,
                    "best_neural_mean_q_total": float(best_neural_metrics["mean_q_total"]),
                }
            )
        except Exception:
            pass


def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config_chain(
        raw_cfg,
        ["reinforce_train", "ablation"],
    )

    run_dir, run_id = get_run_config(cfg, "ablation", resolved_raw_cfg)
    run_logger = setup_wandb(
        cfg,
        resolved_raw_cfg,
        default_group="ablation_ssa",
        run_id=run_id,
        run_dir=run_dir,
    )

    log.info("=" * 60)
    log.info("  SSA-Based Empirical Ablation Study")
    log.info("=" * 60)
    for item in variant_catalog():
        log.info(
            "  %-30s | kind=%s | panel=%s | bootstrap=%s | teacher=%s",
            item["name"],
            item["variant_kind"],
            item["panel"],
            item["bootstrap_mode"],
            item["expert_policy_name"],
        )

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
