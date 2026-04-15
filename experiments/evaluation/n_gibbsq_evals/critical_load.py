"""
N-GibbsQ critical load test.

Tests N-GibbsQ as rho -> 1 against the publication closed-form baseline,
Calibrated UAS, where queueing systems become unstable.
"""

import logging
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import DictConfig

from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.analysis.plot_profiles import ExperimentPlotContext
from gibbsq.core.builders import build_policy_by_name
from gibbsq.core.config import critical_load_sim_time, load_experiment_config
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_ssa import compute_poisson_max_steps
from gibbsq.engines.numpy_engine import run_replications, simulate
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.model_io import build_neural_eval_policy, resolve_model_pointer
from gibbsq.utils.progress import create_progress, iter_progress
from gibbsq.utils.run_artifacts import figure_path, metrics_path


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
NEURAL_EVAL_MODE = "deterministic"
PUBLICATION_BASELINE_POLICY_NAME = "calibrated_uas"
PUBLICATION_BASELINE_LABEL = "Calibrated UAS"


def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Pure functional bridge."""
    return model(Q)


def _publication_baseline_spec() -> tuple[str, int]:
    """Return the canonical publication baseline for critical-load evaluation."""
    return PUBLICATION_BASELINE_POLICY_NAME, int(0)


class CriticalLoadTest:
    """Evaluate N-GibbsQ at high load factors."""

    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float32)
        self.ssa_sim_time = cfg.simulation.ssa.sim_time
        self.ssa_sample_interval = cfg.simulation.ssa.sample_interval
        self.rho_vals = list(cfg.generalization.rho_boundary_vals)

    def execute(self, key: PRNGKeyArray):
        """Sweep rho and measure stability."""
        k_load, _ = jax.random.split(key)

        project_root = Path(__file__).resolve().parents[3]
        output_root = self.run_dir.parent.parent
        model_path = resolve_model_pointer(project_root, output_root, allow_bc=False, allow_legacy=False)
        skeleton = NeuralRouter(
            num_servers=self.num_servers,
            config=self.cfg.neural,
            service_rates=self.service_rates,
            key=k_load,
        )
        model = eqx.tree_deserialise_leaves(model_path, skeleton)

        from gibbsq.utils.model_io import validate_neural_model_shape

        try:
            validate_neural_model_shape(model, self.cfg.neural, self.num_servers)
        except ValueError as e:
            raise RuntimeError(f"Model shape mismatch: {e}") from e

        total_capacity = jnp.sum(self.service_rates)
        neural_results = []
        baseline_results = []
        num_reps = int(self.cfg.simulation.num_replications)
        baseline_policy_name, _ = _publication_baseline_spec()

        log.info(f"System Capacity: {total_capacity:.2f}")
        log.info(f"Targeting Load Boundary: {self.rho_vals}")
        log.info(f"Publication baseline: {PUBLICATION_BASELINE_LABEL} ({baseline_policy_name})")
        try:
            with create_progress(total=len(self.rho_vals), desc="critical", unit="rho") as rho_bar:
                for idx, rho in enumerate(self.rho_vals):
                    rho_bar.set_postfix({"rho": f"{rho:.3f}"}, refresh=False)
                    arrival_rate = rho * total_capacity
                    log.info(f"Evaluating Boundary rho={rho:.3f} (Arrival={arrival_rate:.3f})...")

                    rho_sim_time = critical_load_sim_time(self.cfg, float(rho))
                    mu_np = np.array(self.service_rates, dtype=np.float64)
                    rho_seed = self.cfg.simulation.seed + idx * 1000

                    baseline_policy = build_policy_by_name(
                        baseline_policy_name,
                        alpha=float(self.cfg.system.alpha),
                        mu=mu_np,
                    )
                    max_events = compute_poisson_max_steps(float(arrival_rate), mu_np, rho_sim_time)
                    baseline_results = run_replications(
                        num_replications=num_reps,
                        num_servers=self.num_servers,
                        arrival_rate=float(arrival_rate),
                        service_rates=mu_np,
                        policy=baseline_policy,
                        sim_time=rho_sim_time,
                        sample_interval=self.ssa_sample_interval,
                        base_seed=rho_seed,
                        max_events=max_events,
                        progress_desc=f"critical {PUBLICATION_BASELINE_LABEL} rho={rho:.3f}",
                    )

                    baseline_vals = [
                        float(time_averaged_queue_lengths(res, self.cfg.simulation.burn_in_fraction).sum())
                        for res in baseline_results
                    ]
                    baseline_loss = float(np.mean(baseline_vals))

                    neural_policy = build_neural_eval_policy(
                        model,
                        mu=mu_np,
                        rho=rho,
                        mode=NEURAL_EVAL_MODE,
                    )
                    neural_vals = []
                    for rep_idx in iter_progress(
                        range(num_reps),
                        total=num_reps,
                        desc=f"critical neural rho={rho:.3f}",
                        unit="rep",
                        leave=False,
                    ):
                        rng = np.random.default_rng(rho_seed + rep_idx)
                        res_n = simulate(
                            num_servers=self.num_servers,
                            arrival_rate=float(arrival_rate),
                            service_rates=mu_np,
                            policy=neural_policy,
                            sim_time=rho_sim_time,
                            sample_interval=self.ssa_sample_interval,
                            rng=rng,
                            max_events=max_events,
                        )
                        neural_vals.append(
                            float(time_averaged_queue_lengths(res_n, self.cfg.simulation.burn_in_fraction).sum())
                        )
                    neural_loss = float(np.mean(neural_vals))

                    neural_results.append(neural_loss)
                    baseline_results.append(baseline_loss)
                    log.info(
                        f"   => N-GibbsQ E[Q]: {neural_loss:.2f} | "
                        f"{PUBLICATION_BASELINE_LABEL} E[Q]: {baseline_loss:.2f}"
                    )

                    append_metrics_jsonl(
                        {
                            "rho": float(rho),
                            "baseline_policy": PUBLICATION_BASELINE_POLICY_NAME,
                            "baseline_label": PUBLICATION_BASELINE_LABEL,
                            "neural_eq": neural_loss,
                            "baseline_eq": baseline_loss,
                            "calibrated_uas_eq": baseline_loss,
                            "gibbs_eq": baseline_loss,
                        },
                        metrics_path(self.run_dir),
                    )
                    self._plot_progress(self.rho_vals[: len(neural_results)], neural_results, baseline_results)
                    rho_bar.update(1)
        finally:
            self._plot_progress(self.rho_vals[: len(neural_results)], neural_results, baseline_results)

        if neural_results:
            self._assert_curve_artifacts()

        relative_ratios = [float(n / max(b, 1e-8)) for n, b in zip(neural_results, baseline_results)]
        return {
            "rho_vals": list(self.rho_vals),
            "baseline_policy": PUBLICATION_BASELINE_POLICY_NAME,
            "baseline_label": PUBLICATION_BASELINE_LABEL,
            "neural_eq": neural_results,
            "baseline_eq": baseline_results,
            "calibrated_uas_eq": baseline_results,
            "gibbs_eq": baseline_results,
            "max_neural_to_gibbs_ratio": max(relative_ratios) if relative_ratios else float("inf"),
            "mean_neural_to_gibbs_ratio": float(np.mean(relative_ratios)) if relative_ratios else float("inf"),
        }

    def _plot_progress(self, rho_vals, neural_r, baseline_r) -> None:
        """Persist the best available critical-load curve when results exist."""
        if not rho_vals or not neural_r or not baseline_r:
            return
        try:
            self._plot(rho_vals, neural_r, baseline_r)
        except Exception:
            log.exception(
                "Failed to persist critical-load curve for %d point(s). "
                "rho_vals=%s neural_eq=%s baseline_eq=%s",
                len(rho_vals),
                rho_vals,
                neural_r,
                baseline_r,
            )
            raise

    def _plot(self, rho_vals, neural_r, baseline_r):
        """Generate the critical-load curve."""
        from gibbsq.analysis.plotting import plot_critical_load

        plot_path = figure_path(self.run_dir, "critical_load_curve")
        fig = plot_critical_load(
            rho_values=np.array(rho_vals),
            neural_eq=np.array(neural_r),
            gibbs_eq=np.array(baseline_r),
            save_path=plot_path,
            theme="publication",
            formats=["png", "pdf"],
            context=ExperimentPlotContext(
                experiment_id="critical",
                chart_name="plot_critical_load",
                semantic_overrides={
                    "thresholds": {"critical_rho": float(self.cfg.generalization.rho_boundary_threshold)},
                },
            ),
        )
        plt.close(fig)
        self._assert_curve_artifacts()

        log.info(f"Critical load test complete. Curve saved to {plot_path}.png, {plot_path}.pdf")

        if self.run_logger:
            self.run_logger.log(
                {
                    "critical_load/rho": rho_vals,
                    "critical_load/neural_eq": neural_r,
                    "critical_load/baseline_eq": baseline_r,
                    "critical_load/calibrated_uas_eq": baseline_r,
                    "critical_load/gibbs_eq": baseline_r,
                }
            )
            try:
                import wandb

                self.run_logger.log(
                    {"critical_load_curve": wandb.Image(str(figure_path(self.run_dir, "critical_load_curve").with_suffix(".png")))}
                )
            except Exception:
                pass

    def _assert_curve_artifacts(self) -> None:
        """Fail fast when the critical-load figure was not actually written."""
        plot_path = figure_path(self.run_dir, "critical_load_curve")
        missing = [
            str(path)
            for path in (plot_path.with_suffix(".png"), plot_path.with_suffix(".pdf"))
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "Critical-load plotting did not produce the required figure artifact(s): "
                + ", ".join(missing)
            )


@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "critical")

    run_dir, run_id = get_run_config(cfg, "critical", resolved_raw_cfg)
    run_logger = setup_wandb(
        cfg,
        resolved_raw_cfg,
        default_group="n_gibbsq_verification",
        run_id=run_id,
        run_dir=run_dir,
    )

    log.info("=" * 60)
    log.info("  Phase VIII: The Critical Stability Boundary")
    log.info("=" * 60)

    test = CriticalLoadTest(cfg, run_dir, run_logger)
    return test.execute(jax.random.PRNGKey(cfg.simulation.seed))


if __name__ == "__main__":
    main()
