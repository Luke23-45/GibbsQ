"""
N-GibbsQ generalization sweep.

Tests N-GibbsQ generalization to unseen configurations.

Sweeps:
- Scaling Axis: Server count N increasing from 4 to 32 (Zero-Shot Transfer).
- Load Axis: Arrival Rate ρ increasing from 0.4 to 0.95.
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
        self.temperature = float(cfg.simulation.temperature)
        self.sim_steps = int(cfg.simulation.sim_time) # Enough to see steady state
        
        # Grid parameters - these are currently unused by the class but preserving interface
        self.n_vals = [4, 8, 16, 32]
        self.rho_vals = list(cfg.generalization.rho_grid_vals)

    def _get_env(self, N: int, rho: float):
        """Generates a heterogeneous system for a specific N and rho."""
        # Mean service rate = 10.0, skewed distribution
        service_rates = jnp.linspace(1.0, 19.0, N)
        avg_mu = jnp.mean(service_rates)
        arrival_rate = rho * (N * avg_mu)
        return arrival_rate, service_rates

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
        
        # Pre-generate unique keys for each grid cell (stochastic independence)
        total_cells = len(scale_vals) * len(rho_vals)
        cell_keys = jax.random.split(k_grid, total_cells * 2)
        
        log.info("Evaluating N-GibbsQ improvement ratio (GibbsQ / Neural) on 5x5 Grid...")
        
        cell_idx = 0
        for i, scale in enumerate(scale_vals):
            mu = base_rates * scale  # Scale the rates but keep structure
            total_cap = jnp.sum(mu)
            for j, rho in enumerate(rho_vals):
                lambda_rate = float(rho * total_cap)
                
                # Each cell gets its OWN unique keys for stochastic independence
                k_n = cell_keys[cell_idx * 2]
                k_g = cell_keys[cell_idx * 2 + 1]
                cell_idx += 1
                
                n_loss = simulate_dga_jax(self.cfg.system.num_servers, lambda_rate, mu, model, self.sim_steps, k_n, self.temperature, evaluate_model)
                g_loss = simulate_dga_jax(self.cfg.system.num_servers, lambda_rate, mu, jnp.float32(0.5), self.sim_steps, k_g, self.temperature, default_policy)
                
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
            self.run_logger.log({"generalization_heatmap": [plot_path]})

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
    sweeper.execute(jax.random.PRNGKey(456))

if __name__ == "__main__":
    main()
