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


class _NeuralSSAPolicy:
    """Identical to eval.py."""
    def __init__(self, model):
        import jax as _jax
        self._model = model
        @eqx.filter_jit
        def _forward(m, x): return _jax.nn.softmax(m(x))
        self._forward = _forward
        @functools.lru_cache(maxsize=131072)
        def _get_probs(q_tuple):
            probs = self._forward(self._model, jnp.array(q_tuple, dtype=jnp.float32))
            probs_np = np.array(probs, dtype=np.float64)
            return probs_np / probs_np.sum()
        self._get_probs = _get_probs
    def __call__(self, Q, rng): return self._get_probs(tuple(Q))


logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

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
        # SG#5 FIX: Use fixed canonical pointer path (config-independent).
        pointer_path = Path("outputs") / "small" / "latest_weights.txt"
        if not pointer_path.exists():
            raise FileNotFoundError(
                f"Model pointer not found at '{pointer_path.resolve()}'. "
                f"Run training first: python -m experiments.n_gibbsq.train"
            )
        model_path = Path(pointer_path.read_text(encoding='utf-8').strip())
        if not model_path.exists():
            raise FileNotFoundError(
                f"Weight file missing at '{model_path}'. Rerun training."
            )
        
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
        skeleton = NeuralRouter(num_servers=self.cfg.system.num_servers, config=self.cfg.neural, key=k_load)
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        
        # SG#16 Fix: Validate that the loaded model matches the current config
        if model.layers[0].weight.shape[1] != self.cfg.system.num_servers:
            log.error(f"Model shape mismatch! Loaded model expects N={model.layers[0].weight.shape[1]}, but eval config requires N={self.cfg.system.num_servers}.")
            return
        
        # SG-4 FIX: Replace single-sample DGA calls with replicated SSA.
        _cell_reps   = int(self.cfg.simulation.num_replications)
        _neural_ssa  = _NeuralSSAPolicy(model)  # one instance, reused across all cells
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

                # GibbsQ on true SSA
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
                    policy_type=3,
                )
                g_vals = []
                for _r in range(_cell_reps):
                    _res = SimResult(
                        times=np.array(times_g[_r]), states=np.array(states_g[_r]),
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
        """Generates generalization heatmap."""
        plt.figure(figsize=(10, 8))
        # RdYlGn: Green = high ratio = Neural strongly wins
        im = plt.imshow(grid, cmap="RdYlGn", aspect='auto', vmin=0.5, vmax=max(3.0, np.max(grid)))
        
        # Add values in cells
        for i in range(len(scale_vals)):
            for j in range(len(rho_vals)):
                val = grid[i, j]
                plt.text(j, i, f"{val:.2f}x", ha="center", va="center", 
                         color="black" if 0.8 < val < 2.5 else "white", fontweight='bold')
        
        plt.colorbar(im, label='Improvement Factor (GibbsQ / Neural). > 1.0 = Neural Wins')
        
        plt.xticks(np.arange(len(rho_vals)), rho_vals)
        plt.yticks(np.arange(len(scale_vals)), [f"{s}x" for s in scale_vals])
        
        plt.title('N-GibbsQ Zero-Shot Generalization\nImprovement Factor over GibbsQ (Higher = Better)')
        plt.xlabel('Load Factor ($\\rho$)')
        plt.ylabel('Service Rate Scale (x base distribution)')
        
        plot_path = self.run_dir / "generalization_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Generalization analysis complete. Heatmap saved to {plot_path}")
        
        if self.run_logger:
            try:
                import wandb
                self.run_logger.log({"generalization_heatmap": wandb.Image(str(plot_path))})
            except Exception:
                pass

        # Persist metrics locally
        append_metrics_jsonl({
            "grid": grid.tolist(),
            "scale_vals": scale_vals,
            "rho_vals": rho_vals
        }, self.run_dir / "metrics.jsonl")

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
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
