"""
N-GibbsQ generalization sweep.

Tests N-GibbsQ generalization to unseen configurations at FIXED server count.

PREREQUISITE: This sweep evaluates a model trained with the SAME
config that is active when this script runs. Train first via one of the
public entry points:
  python scripts/execution/experiment_runner.py bc_train --config-name <your-config>
  python scripts/execution/experiment_runner.py reinforce_train --config-name <your-config>
The model architecture (num_servers, hidden_size) must match the active config.
The shape check (model.layers[0].weight.shape[1] == num_servers) enforces this.

The "rate-scale" and "load" axes below vary the OPERATING CONDITIONS, not the
model architecture. The model is never re-trained during the sweep — it is
evaluated zero-shot on each (scale, rho) grid cell.

Sweeps (2-D grid):
- Rate-Scale Axis: Uniform scaling of all service rates (config: generalization.scale_vals).
- Load Axis:       Arrival-rate ρ (config: generalization.rho_grid_vals).

This sweep does not vary server count.
The model is evaluated only at the same N used for training.

Both sides measured on the true Gillespie SSA (not DGA surrogate).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
from jaxtyping import Array, Float, PRNGKeyArray
import numpy as np
import matplotlib.pyplot as plt
import functools
from gibbsq.analysis.plot_profiles import ExperimentPlotContext
from gibbsq.core import constants

from gibbsq.core.config import load_experiment_config
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_engine import policy_name_to_type, run_replications_jax
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.model_io import build_neural_eval_policy, resolve_model_pointer
from gibbsq.engines.jax_ssa import compute_poisson_max_steps
from gibbsq.utils.progress import create_progress, iter_progress
from gibbsq.utils.run_artifacts import figure_path, metrics_path


logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)
NEURAL_EVAL_MODE = "deterministic"


def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Pure functional bridge."""
    return model(Q)

class GeneralizationSweeper:
    """
    Zero-shot transfer evaluation engine for N-GibbsQ.
    """
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.temperature = float(cfg.simulation.dga.temperature)
        self.sim_steps = cfg.simulation.dga.sim_steps  # Enough to see steady state
        self.ssa_sim_time = cfg.simulation.ssa.sim_time
        self.ssa_sample_interval = cfg.simulation.ssa.sample_interval


    def execute(self, key: PRNGKeyArray):
        """Runs the multi-dimensional sweep."""
        k_load, k_grid = jax.random.split(key)
        
        _PROJECT_ROOT = Path(__file__).resolve().parents[3]
        output_root = self.run_dir.parent.parent
        model_path = resolve_model_pointer(_PROJECT_ROOT, output_root, allow_bc=False, allow_legacy=False)

        scale_vals = list(self.cfg.generalization.scale_vals)
        rho_vals = list(self.cfg.generalization.rho_grid_vals)
        base_rates = jnp.array(self.cfg.system.service_rates, dtype=jnp.float32)
        
        grid = np.zeros((len(scale_vals), len(rho_vals)))
        
        log.info(f"Initiating Generalization Sweep (Scales={scale_vals}, rho={rho_vals})")
        
        skeleton = NeuralRouter(num_servers=self.cfg.system.num_servers, config=self.cfg.neural, service_rates=self.cfg.system.service_rates, key=k_load)
        model = eqx.tree_deserialise_leaves(model_path, skeleton)

        from gibbsq.utils.model_io import validate_neural_model_shape
        try:
            validate_neural_model_shape(model, self.cfg.neural, self.cfg.system.num_servers)
        except ValueError as e:
            raise RuntimeError(f"Model shape mismatch: {e}") from e
        
        _cell_reps   = int(self.cfg.simulation.num_replications)
        _max_s_cell  = int(self.ssa_sim_time / self.ssa_sample_interval) + 1

        total_cells = len(scale_vals) * len(rho_vals)
        cell_keys = jax.random.split(k_grid, total_cells)
        
        log.info(
            "Evaluating N-GibbsQ improvement ratio (GibbsQ / Neural) "
            f"on {len(scale_vals)}x{len(rho_vals)} grid..."
        )
        
        cell_idx = 0
        baseline_policy_name = self.cfg.policy.name
        baseline_policy_type = policy_name_to_type(baseline_policy_name)

        with create_progress(total=total_cells, desc="generalize", unit="cell") as cell_bar:
            for i, scale in enumerate(scale_vals):
                mu = base_rates * scale
                total_cap = jnp.sum(mu)
                for j, rho in enumerate(rho_vals):
                    cell_bar.set_postfix(
                        {"scale": f"{scale:.2f}x", "rho": f"{rho:.2f}"},
                        refresh=False,
                    )
                    lambda_rate = float(rho * total_cap)

                    _cell_seed = int(jax.random.bits(jax.random.fold_in(cell_keys[cell_idx], 0))) & 0x7FFFFFFF
                    cell_idx += 1
                    _mu_np = np.array(mu, dtype=np.float64)

                    _neural_ssa = build_neural_eval_policy(
                        model,
                        mu=_mu_np,
                        rho=rho,
                        mode=NEURAL_EVAL_MODE,
                    )
                    times_g, states_g, (arrs_g, deps_g) = run_replications_jax(
                        num_replications=_cell_reps,
                        num_servers=self.cfg.system.num_servers,
                        arrival_rate=lambda_rate,
                        service_rates=jnp.array(_mu_np),
                        alpha=float(self.cfg.system.alpha),
                        sim_time=self.ssa_sim_time,
                        sample_interval=self.ssa_sample_interval,
                        base_seed=_cell_seed,
                        max_samples=_max_s_cell,
                        policy_type=baseline_policy_type,
                        max_events_multiplier=self.cfg.jax_engine.max_events_safety_multiplier,
                        max_events_buffer=self.cfg.jax_engine.max_events_additive_buffer,
                        scan_sampling_chunk=self.cfg.jax_engine.scan_sampling_chunk,
                    )
                    g_vals = []
                    for _r in range(_cell_reps):
                        _np_times = np.array(times_g[_r])
                        _np_states = np.array(states_g[_r])
                        _valid_mask = _np_times > 0
                        _valid_mask[0] = True
                        _vl = int(np.sum(_valid_mask))
                        _np_times = _np_times[:_vl]
                        _np_states = _np_states[:_vl]

                        _res = SimResult(
                            times=_np_times, states=_np_states,
                            arrival_count=int(arrs_g[_r]), departure_count=int(deps_g[_r]),
                            final_time=float(_np_times[-1]),
                            num_servers=self.cfg.system.num_servers,
                        )
                        g_vals.append(float(time_averaged_queue_lengths(
                            _res, self.cfg.simulation.burn_in_fraction).sum()))
                    g_loss = float(np.mean(g_vals))

                    _max_ev = compute_poisson_max_steps(lambda_rate, _mu_np, self.ssa_sim_time)
                    n_vals = []
                    for _rep in iter_progress(
                        range(_cell_reps),
                        total=_cell_reps,
                        desc=f"generalize reps scale={scale:.2f} rho={rho:.2f}",
                        unit="rep",
                        leave=False,
                    ):
                        _rng = np.random.default_rng(_cell_seed + _rep)
                        _res_n = simulate(
                            num_servers=self.cfg.system.num_servers,
                            arrival_rate=lambda_rate, service_rates=_mu_np,
                            policy=_neural_ssa, sim_time=self.ssa_sim_time,
                            sample_interval=self.ssa_sample_interval,
                            rng=_rng, max_events=_max_ev,
                        )
                        n_vals.append(float(time_averaged_queue_lengths(
                            _res_n, self.cfg.simulation.burn_in_fraction).sum()))
                    n_loss = float(np.mean(n_vals))

                    ratio = float(g_loss / max(n_loss, constants.NUMERICAL_STABILITY_EPSILON))
                    grid[i, j] = ratio
                    log.info(f"   Scale={scale:5.1f}x | rho={rho:.2f} | Improvement={ratio:.2f}x")
                    cell_bar.update(1)

        self._plot_heatmap(grid, scale_vals, rho_vals)
        return {
            "grid": grid.tolist(),
            "scale_vals": scale_vals,
            "rho_vals": rho_vals,
            "min_improvement_ratio": float(np.min(grid)) if grid.size else 0.0,
            "mean_improvement_ratio": float(np.mean(grid)) if grid.size else 0.0,
        }

    def _plot_heatmap(self, grid, scale_vals, rho_vals):
        """Generates generalization heatmap."""
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
                    },
                },
            ),
        )
        plt.close(fig)

        log.info(f"Generalization analysis complete. Heatmap saved to {plot_path}.png, {plot_path}.pdf")
        
        if self.run_logger:
            try:
                import wandb
                self.run_logger.log({"generalization_heatmap": wandb.Image(str(figure_path(self.run_dir, "generalization_heatmap").with_suffix(".png")))})
            except Exception:
                pass

        append_metrics_jsonl({
            "grid": grid.tolist(),
            "scale_vals": scale_vals,
            "rho_vals": rho_vals
        }, metrics_path(self.run_dir))

@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "generalize")

    run_dir, run_id = get_run_config(cfg, "generalize", resolved_raw_cfg)
    run_logger = setup_wandb(cfg, resolved_raw_cfg, default_group="n_gibbsq_verification", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase VIII: Generalization & Stress Heatmap")
    log.info("=" * 60)
    
    sweeper = GeneralizationSweeper(cfg, run_dir, run_logger)
    return sweeper.execute(jax.random.PRNGKey(cfg.simulation.seed))

if __name__ == "__main__":
    main()
