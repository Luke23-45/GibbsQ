"""
N-GibbsQ generalization sweep.

Tests N-GibbsQ generalization to unseen configurations at FIXED server count.

PREREQUISITE (SG#8): This sweep evaluates a model trained with the SAME
config that is active when this script runs. Train first:
  python -m experiments.n_gibbsq.train --config-name <your-config>
The model architecture (num_servers, hidden_size) must match the active config.
The SG#16 check (model.layers[0].weight.shape[1] == num_servers) enforces this.

The "rate-scale" and "load" axes below vary the OPERATING CONDITIONS, not the
model architecture. The model is never re-trained during the sweep — it is
evaluated zero-shot on each (scale, rho) grid cell.

Sweeps (2-D grid):
- Rate-Scale Axis: Uniform scaling of all service rates (config: generalization.scale_vals).
- Load Axis:       Arrival-rate ρ (config: generalization.rho_grid_vals).

NOTE: N-scaling (varying server count) is NOT implemented in this sweep.
      The model is evaluated only on the same N it was trained on.

SG-D FIX: Both sides now measured on the true Gillespie SSA (not DGA surrogate).
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
from gibbsq.core import constants

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_engine import run_replications_jax
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.model_io import NeuralSSAPolicy, resolve_model_pointer


# _NeuralSSAPolicy moved to gibbsq.utils.model_io.NeuralSSAPolicy


logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


# _resolve_model_pointer moved to gibbsq.utils.model_io.resolve_model_pointer

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
        
        # 1. Load trained model
        _PROJECT_ROOT = Path(__file__).resolve().parents[3]
        output_root = self.run_dir.parent.parent
        model_path = resolve_model_pointer(_PROJECT_ROOT, output_root)
        
        # SG#8 FIX: Removed stale comment referencing "[100,1,1,1]" training
        # config (not part of this codebase). The sweep design is:
        #   base_rates  = cfg.system.service_rates (matches training config)
        #   Scale axis  = base_rates × generalization.scale_vals
        #   Load axis   = generalization.rho_grid_vals × total_capacity(scaled)
        # The model evaluates zero-shot on each (scale, rho) grid cell.
        
        scale_vals = list(self.cfg.generalization.scale_vals)  # Multiply base rates
        rho_vals = list(self.cfg.generalization.rho_grid_vals)
        base_rates = jnp.array(self.cfg.system.service_rates, dtype=jnp.float32)  # Same structure as training
        
        grid = np.zeros((len(scale_vals), len(rho_vals)))
        
        log.info(f"Initiating Generalization Sweep (Scales={scale_vals}, rho={rho_vals})")
        
        # Load Neural model matching training dimensions using validated NeuralConfig
        skeleton = NeuralRouter(num_servers=self.cfg.system.num_servers, config=self.cfg.neural, service_rates=self.cfg.system.service_rates, key=k_load)
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        
        # SG#16 Fix: Validate that the loaded model matches the current config
        from gibbsq.utils.model_io import validate_neural_model_shape
        try:
            validate_neural_model_shape(model, self.cfg.neural, self.cfg.system.num_servers)
        except ValueError as e:
            log.error(f"Model shape mismatch! {e}")
            return
        
        # SG-4 FIX: Replace single-sample DGA calls with replicated SSA.
        _cell_reps   = int(self.cfg.simulation.num_replications)
        _max_s_cell  = int(self.ssa_sim_time / self.ssa_sample_interval) + 1

        # Pre-generate unique keys for each grid cell (stochastic independence)
        total_cells = len(scale_vals) * len(rho_vals)
        cell_keys = jax.random.split(k_grid, total_cells)
        
        log.info("Evaluating N-GibbsQ improvement ratio (GibbsQ / Neural) on 5x5 Grid...")
        
        cell_idx = 0
        for i, scale in enumerate(scale_vals):
            mu = base_rates * scale  # Scale the rates but keep structure
            total_cap = jnp.sum(mu)
            for j, rho in enumerate(rho_vals):
                lambda_rate = float(rho * total_cap)
                
                _cell_seed = int(jax.random.bits(jax.random.fold_in(cell_keys[cell_idx], 0))) & 0x7FFFFFFF
                cell_idx += 1
                _mu_np = np.array(mu, dtype=np.float64)

                # Neural on true SSA
                _neural_ssa = NeuralSSAPolicy(model, mu=_mu_np, rho=rho)
                _pmap = {"uniform": 0, "proportional": 1, "jsq": 2, "softmax": 3, "power_of_d": 4, "sojourn_softmax": 5}
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
                    policy_type=_pmap.get(self.cfg.policy.name, 3),
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
                        final_time=float(times_g[_r][-1]),
                        num_servers=self.cfg.system.num_servers,
                    )
                    g_vals.append(float(time_averaged_queue_lengths(
                        _res, self.cfg.simulation.burn_in_fraction).sum()))
                g_loss = float(np.mean(g_vals))

                # Neural on true SSA
                _max_ev = int((lambda_rate + _mu_np.sum()) * self.ssa_sim_time * 1.5) + 1000
                n_vals = []
                for _rep in range(_cell_reps):
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
                
                # Ratio: GibbsQ / Neural. > 1.0 means Neural is better.
                ratio = float(g_loss / max(n_loss, constants.NUMERICAL_STABILITY_EPSILON))
                grid[i, j] = ratio
                log.info(f"   Scale={scale:5.1f}x | rho={rho:.2f} | Improvement={ratio:.2f}x")

        self._plot_heatmap(grid, scale_vals, rho_vals)

    def _plot_heatmap(self, grid, scale_vals, rho_vals):
        """Generates generalization heatmap using chart-type-aware styling."""
        from gibbsq.analysis.plotting import plot_improvement_heatmap

        plot_path = self.run_dir / "generalization_heatmap"
        fig = plot_improvement_heatmap(
            grid=grid,
            x_labels=[str(r) for r in rho_vals],
            y_labels=[f"{s}x" for s in scale_vals],
            x_axis_name=r"Load Factor $\rho$",
            y_axis_name="Service Rate Scale (x base distribution)",
            save_path=plot_path,
            theme="publication",
            formats=["png", "pdf"],
        )
        plt.close(fig)

        log.info(f"Generalization analysis complete. Heatmap saved to {plot_path}.png, {plot_path}.pdf")
        
        if self.run_logger:
            try:
                import wandb
                self.run_logger.log({"generalization_heatmap": wandb.Image(str(self.run_dir / "generalization_heatmap.png"))})
            except Exception:
                pass

        # Persist metrics locally
        append_metrics_jsonl({
            "grid": grid.tolist(),
            "scale_vals": scale_vals,
            "rho_vals": rho_vals
        }, self.run_dir / "metrics.jsonl")

@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "generalization_sweep", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_verification", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase VIII: Generalization & Stress Heatmap")
    log.info("=" * 60)
    
    sweeper = GeneralizationSweeper(cfg, run_dir, run_logger)
    sweeper.execute(jax.random.PRNGKey(cfg.simulation.seed))

if __name__ == "__main__":
    main()
