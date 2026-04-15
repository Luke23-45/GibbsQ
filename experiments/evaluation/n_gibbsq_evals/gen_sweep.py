"""
N-GibbsQ generalization sweep.

Tests N-GibbsQ generalization to unseen configurations at FIXED server count
against the publication closed-form baseline, Calibrated UAS.

PREREQUISITE: This sweep evaluates a model trained with the SAME
config that is active when this script runs. Train first via one of the
public entry points:
  python scripts/execution/experiment_runner.py bc_train --config-name <your-config>
  python scripts/execution/experiment_runner.py reinforce_train --config-name <your-config>
The model architecture (num_servers, hidden_size) must match the active config.
The shape check (model.layers[0].weight.shape[1] == num_servers) enforces this.

The "rate-scale" and "load" axes below vary the OPERATING CONDITIONS, not the
model architecture. The model is never re-trained during the sweep - it is
evaluated zero-shot on each (scale, rho) grid cell.

Sweeps (2-D grid):
- Rate-Scale Axis: Uniform scaling of all service rates (config: generalization.scale_vals).
- Load Axis:       Arrival-rate rho (config: generalization.rho_grid_vals).

This sweep does not vary server count.
The model is evaluated only at the same N used for training.

Both sides measured on the true Gillespie SSA (not DGA surrogate).
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
from gibbsq.core import constants
from gibbsq.core.builders import build_policy_by_name
from gibbsq.core.config import load_experiment_config
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
    """Return the canonical publication baseline for generalization sweeps."""
    return PUBLICATION_BASELINE_POLICY_NAME, int(0)


class GeneralizationSweeper:
    """Zero-shot transfer evaluation engine for N-GibbsQ."""

    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.temperature = float(cfg.simulation.dga.temperature)
        self.sim_steps = cfg.simulation.dga.sim_steps
        self.ssa_sim_time = cfg.simulation.ssa.sim_time
        self.ssa_sample_interval = cfg.simulation.ssa.sample_interval

    def execute(self, key: PRNGKeyArray):
        """Run the multi-dimensional sweep."""
        k_load, k_grid = jax.random.split(key)

        project_root = Path(__file__).resolve().parents[3]
        output_root = self.run_dir.parent.parent
        model_path = resolve_model_pointer(project_root, output_root, allow_bc=False, allow_legacy=False)

        scale_vals = list(self.cfg.generalization.scale_vals)
        rho_vals = list(self.cfg.generalization.rho_grid_vals)
        base_rates = jnp.array(self.cfg.system.service_rates, dtype=jnp.float32)
        grid = np.zeros((len(scale_vals), len(rho_vals)))

        log.info(f"Initiating Generalization Sweep (Scales={scale_vals}, rho={rho_vals})")

        skeleton = NeuralRouter(
            num_servers=self.cfg.system.num_servers,
            config=self.cfg.neural,
            service_rates=self.cfg.system.service_rates,
            key=k_load,
        )
        model = eqx.tree_deserialise_leaves(model_path, skeleton)

        from gibbsq.utils.model_io import validate_neural_model_shape

        try:
            validate_neural_model_shape(model, self.cfg.neural, self.cfg.system.num_servers)
        except ValueError as e:
            raise RuntimeError(f"Model shape mismatch: {e}") from e

        cell_reps = int(self.cfg.simulation.num_replications)
        total_cells = len(scale_vals) * len(rho_vals)
        cell_keys = jax.random.split(k_grid, total_cells)
        baseline_policy_name, _ = _publication_baseline_spec()

        log.info(
            f"Evaluating N-GibbsQ improvement ratio ({PUBLICATION_BASELINE_LABEL} / Neural) "
            f"on {len(scale_vals)}x{len(rho_vals)} grid with policy='{baseline_policy_name}'..."
        )

        cell_idx = 0
        with create_progress(total=total_cells, desc="generalize", unit="cell") as cell_bar:
            for i, scale in enumerate(scale_vals):
                mu = base_rates * scale
                total_cap = jnp.sum(mu)
                for j, rho in enumerate(rho_vals):
                    cell_bar.set_postfix({"scale": f"{scale:.2f}x", "rho": f"{rho:.2f}"}, refresh=False)
                    lambda_rate = float(rho * total_cap)
                    cell_seed = int(jax.random.bits(jax.random.fold_in(cell_keys[cell_idx], 0))) & 0x7FFFFFFF
                    cell_idx += 1
                    mu_np = np.array(mu, dtype=np.float64)

                    neural_ssa = build_neural_eval_policy(
                        model,
                        mu=mu_np,
                        rho=rho,
                        mode=NEURAL_EVAL_MODE,
                    )
                    baseline_policy = build_policy_by_name(
                        baseline_policy_name,
                        alpha=float(self.cfg.system.alpha),
                        mu=mu_np,
                    )
                    max_events = compute_poisson_max_steps(lambda_rate, mu_np, self.ssa_sim_time)
                    baseline_results = run_replications(
                        num_replications=cell_reps,
                        num_servers=self.cfg.system.num_servers,
                        arrival_rate=lambda_rate,
                        service_rates=mu_np,
                        policy=baseline_policy,
                        sim_time=self.ssa_sim_time,
                        sample_interval=self.ssa_sample_interval,
                        base_seed=cell_seed,
                        max_events=max_events,
                        progress_desc=f"generalize baseline scale={scale:.2f} rho={rho:.2f}",
                    )

                    baseline_vals = [
                        float(time_averaged_queue_lengths(res, self.cfg.simulation.burn_in_fraction).sum())
                        for res in baseline_results
                    ]
                    baseline_loss = float(np.mean(baseline_vals))

                    neural_vals = []
                    for rep_idx in iter_progress(
                        range(cell_reps),
                        total=cell_reps,
                        desc=f"generalize reps scale={scale:.2f} rho={rho:.2f}",
                        unit="rep",
                        leave=False,
                    ):
                        rng = np.random.default_rng(cell_seed + rep_idx)
                        res_n = simulate(
                            num_servers=self.cfg.system.num_servers,
                            arrival_rate=lambda_rate,
                            service_rates=mu_np,
                            policy=neural_ssa,
                            sim_time=self.ssa_sim_time,
                            sample_interval=self.ssa_sample_interval,
                            rng=rng,
                            max_events=max_events,
                        )
                        neural_vals.append(
                            float(time_averaged_queue_lengths(res_n, self.cfg.simulation.burn_in_fraction).sum())
                        )
                    neural_loss = float(np.mean(neural_vals))

                    ratio = float(baseline_loss / max(neural_loss, constants.NUMERICAL_STABILITY_EPSILON))
                    grid[i, j] = ratio
                    log.info(
                        f"   Scale={scale:5.1f}x | rho={rho:.2f} | "
                        f"{PUBLICATION_BASELINE_LABEL}/Neural={ratio:.2f}x"
                    )
                    cell_bar.update(1)

        self._plot_heatmap(grid, scale_vals, rho_vals)
        return {
            "baseline_policy": PUBLICATION_BASELINE_POLICY_NAME,
            "baseline_label": PUBLICATION_BASELINE_LABEL,
            "grid": grid.tolist(),
            "scale_vals": scale_vals,
            "rho_vals": rho_vals,
            "min_improvement_ratio": float(np.min(grid)) if grid.size else 0.0,
            "mean_improvement_ratio": float(np.mean(grid)) if grid.size else 0.0,
        }

    def _plot_heatmap(self, grid, scale_vals, rho_vals):
        """Generate generalization heatmap."""
        from gibbsq.analysis.plotting import plot_improvement_heatmap

        plot_path = figure_path(self.run_dir, "generalization_heatmap")
        fig = plot_improvement_heatmap(
            grid=grid,
            x_labels=[str(r) for r in rho_vals],
            y_labels=[f"{s}x" for s in scale_vals],
            x_axis_name=r"Load Factor $\rho$",
            y_axis_name="Service Rate Scale (x base distribution)",
            save_path=plot_path,
            theme="publication",
            formats=["png", "pdf"],
            context=ExperimentPlotContext(
                experiment_id="generalize",
                chart_name="plot_improvement_heatmap",
                semantic_overrides={
                    "axis_labels": {
                        "y": "Service Rate Scale (x base distribution)",
                        "colorbar": f"Improvement Ratio ({PUBLICATION_BASELINE_LABEL} / Neural)",
                    },
                },
            ),
        )
        plt.close(fig)

        log.info(f"Generalization analysis complete. Heatmap saved to {plot_path}.png, {plot_path}.pdf")

        if self.run_logger:
            try:
                import wandb

                self.run_logger.log(
                    {"generalization_heatmap": wandb.Image(str(figure_path(self.run_dir, "generalization_heatmap").with_suffix(".png")))}
                )
            except Exception:
                pass

        append_metrics_jsonl(
            {
                "baseline_policy": PUBLICATION_BASELINE_POLICY_NAME,
                "baseline_label": PUBLICATION_BASELINE_LABEL,
                "grid": grid.tolist(),
                "scale_vals": scale_vals,
                "rho_vals": rho_vals,
            },
            metrics_path(self.run_dir),
        )


@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "generalize")

    run_dir, run_id = get_run_config(cfg, "generalize", resolved_raw_cfg)
    run_logger = setup_wandb(
        cfg,
        resolved_raw_cfg,
        default_group="n_gibbsq_verification",
        run_id=run_id,
        run_dir=run_dir,
    )

    log.info("=" * 60)
    log.info("  Phase VIII: Generalization & Stress Heatmap")
    log.info("=" * 60)

    sweeper = GeneralizationSweeper(cfg, run_dir, run_logger)
    return sweeper.execute(jax.random.PRNGKey(cfg.simulation.seed))


if __name__ == "__main__":
    main()
