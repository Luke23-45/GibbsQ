"""
N-GibbsQ generalization sweep.

Tests N-GibbsQ generalization to unseen configurations at FIXED server count.

Sweeps (2-D grid):
- Rate-Scale Axis: Uniform scaling of all service rates (config: generalization.scale_vals).
- Load Axis:       Arrival-rate ρ (config: generalization.rho_grid_vals).

NOTE: N-scaling (varying server count) is NOT implemented in this sweep.
      The model is evaluated only on the same N it was trained on.
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

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.differentiable_engine import simulate_dga_jax, default_policy
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

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


    def execute(self, key: PRNGKeyArray):
        """Runs the multi-dimensional sweep."""
        k_load, k_grid = jax.random.split(key)
        
        # 1. Load trained model (Trained on N=4, [100,1,1,1], arrival=95)
        pointer_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "neural_training" / "latest_weights.txt"
        if not pointer_path.exists():
            log.error("Latest weights not found. Run training first.")
            return
        
        model_path = Path(pointer_path.read_text(encoding='utf-8').strip())
        
        # Generalization Test Design:
        # The model was trained on [100,1,1,1] at ρ≈0.92.
        # We test: can it handle UNSEEN load conditions and service rate SCALES?
        # We preserve the STRUCTURE (1 fast + 3 slow) but vary:
        #   - Service Rate Scale: [100,1,1,1] x {0.5, 1.0, 2.0, 5.0, 10.0}
        #   - Load Factor ρ: {0.5, 0.7, 0.85, 0.95, 0.98}
        # This tests if the model learned the PRINCIPLE of "route to fast server"
        # not just specific weight values for one config.
        
        scale_vals = list(self.cfg.generalization.scale_vals)  # Multiply base rates
        rho_vals = list(self.cfg.generalization.rho_grid_vals)
        base_rates = jnp.array(self.cfg.system.service_rates, dtype=jnp.float32)  # Same structure as training
        
        grid = np.zeros((len(scale_vals), len(rho_vals)))
        
        log.info(f"Initiating Generalization Sweep (Scales={scale_vals}, rho={rho_vals})")
        
        # Load Neural model matching training dimensions
        skeleton = NeuralRouter(num_servers=self.cfg.system.num_servers, hidden_size=64, key=k_load)
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        
        # SG#16 Fix: Validate that the loaded model matches the current config
        if model.l1.weight.shape[1] != self.cfg.system.num_servers:
            log.error(f"Model shape mismatch! Loaded model expects N={model.l1.weight.shape[1]}, but eval config requires N={self.cfg.system.num_servers}.")
            return
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
                
                # CRN: use the SAME key for both neural and GibbsQ (SG-11.5 fix)
                k_shared = cell_keys[cell_idx]
                cell_idx += 1
                
                n_loss = simulate_dga_jax(self.cfg.system.num_servers, lambda_rate, mu, model, self.sim_steps, k_shared, self.temperature, evaluate_model)
                g_loss = simulate_dga_jax(self.cfg.system.num_servers, lambda_rate, mu, jnp.float32(self.cfg.system.alpha), self.sim_steps, k_shared, self.temperature, default_policy)
                
                # Ratio: GibbsQ / Neural. > 1.0 means Neural is better.
                ratio = float(g_loss / jnp.maximum(n_loss, 1e-9))
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
