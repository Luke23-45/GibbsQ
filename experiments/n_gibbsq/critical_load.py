"""
N-GibbsQ critical load test.

Tests N-GibbsQ as ρ → 1, where queueing systems become unstable.

Compares the neural router's expected queue length against GibbsQ
at load factors approaching the critical boundary.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
from jaxtyping import Array, Float, PRNGKeyArray
import matplotlib.pyplot as plt
import numpy as np

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

class CriticalLoadTest:
    """Evaluates N-GibbsQ at high load factors."""
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float32)
        self.sim_steps = cfg.simulation.dga.sim_steps
        self.temperature = float(cfg.simulation.dga.temperature)
        
        # Symmetrical and Extreme ρ range
        self.rho_vals = list(cfg.generalization.rho_boundary_vals)

    def execute(self, key: PRNGKeyArray):
        """Sweeps ρ and measures stability."""
        k_load, k_sweep = jax.random.split(key)
        
        # 1. Load trained model
        pointer_path = Path(self.cfg.output_dir) / "small" / "latest_weights.txt"
        if not pointer_path.exists():
            log.error(f"Latest weights not found at {pointer_path}. Run training first.")
            return
        
        model_path = Path(pointer_path.read_text(encoding='utf-8').strip())
        if not model_path.exists():
            log.error(f"Trained weights not found at {model_path}. Run training first.")
            return
        skeleton = NeuralRouter(num_servers=self.num_servers, config=self.cfg.neural, key=k_load)
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        
        # SG#16 Fix: Validate that the loaded model matches the current config
        if model.layers[0].weight.shape[1] != self.num_servers:
            log.error(f"Model shape mismatch! Loaded model expects N={model.layers[0].weight.shape[1]}, but eval config requires N={self.num_servers}.")
            return
        
        total_capacity = jnp.sum(self.service_rates)
        
        neural_results = []
        gibbs_results = []
        
        log.info(f"System Capacity: {total_capacity:.2f}")
        log.info(f"Targeting Load Boundary: {self.rho_vals}")
        
        # Pre-generate shared CRN keys for each ρ value; static_argnums replaces static_argnames
        rho_keys = jax.random.split(k_sweep, len(self.rho_vals))
        
        # Vectorized simulation (static_argnums: positions 0=num_servers, 4=sim_steps, 7=apply_fn)
        num_reps = int(self.cfg.simulation.num_replications)
        vmap_simulate = jax.jit(
            jax.vmap(simulate_dga_jax, in_axes=(None, None, None, None, None, 0, None, None)),
            static_argnums=(0, 4, 7)
        )
        
        for idx, rho in enumerate(self.rho_vals):
            arrival_rate = rho * total_capacity
            
            # CRN: shared key for both neural and GibbsQ (SG-13.5 fix)
            k_shared = rho_keys[idx]
            rep_keys = jax.random.split(k_shared, num_reps)
            
            log.info(f"Evaluating Boundary rho={rho:.3f} (Arrival={arrival_rate:.3f})...")
            
            # Neural Evaluation with vmap replications (SG-13.4 fix)
            n_loss_array = vmap_simulate(
                self.num_servers, arrival_rate, self.service_rates, model, self.sim_steps, rep_keys, self.temperature, evaluate_model
            )
            n_loss = float(jnp.mean(n_loss_array))
            neural_results.append(n_loss)
            
            # GibbsQ Evaluation with vmap replications and CRN
            g_loss_array = vmap_simulate(
                self.num_servers, arrival_rate, self.service_rates, jnp.float32(self.cfg.system.alpha), self.sim_steps, rep_keys, self.temperature, default_policy
            )
            g_loss = float(jnp.mean(g_loss_array))
            gibbs_results.append(g_loss)
            
            log.info(f"   => N-GibbsQ E[Q]: {n_loss:.2f} | GibbsQ E[Q]: {g_loss:.2f}")

            append_metrics_jsonl({
                "rho": float(rho),
                "neural_eq": n_loss,
                "gibbs_eq": g_loss
            }, self.run_dir / "metrics.jsonl")

        self._plot(self.rho_vals, neural_results, gibbs_results)

    def _plot(self, rho_vals, neural_r, gibbs_r):
        """Generates the stability breakdown plot."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(rho_vals, neural_r, marker='s', color='#2ecc71', linewidth=2, label='N-GibbsQ (Neural Router)')
        plt.plot(rho_vals, gibbs_r, marker='o', color='#e74c3c', linestyle='--', linewidth=2, label='GibbsQ (Baseline)')
        
        plt.yscale('log')
        plt.title('N-GibbsQ Stability Boundary Performance ($\mathbb{E}[Q]$ vs $\\rho$)')
        plt.xlabel('Load Factor $\\rho = \lambda / \sum \mu_i$')
        plt.ylabel('Expected Queue Length (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        
        plot_path = self.run_dir / "critical_load_curve.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        log.info(f"Critical load test complete. Curve saved to {plot_path}")
        
        if self.run_logger:
            self.run_logger.log({
                "critical_load/rho": rho_vals,
                "critical_load/neural_eq": neural_r,
                "critical_load/gibbs_eq": gibbs_r
            })
            try:
                import wandb
                self.run_logger.log({"critical_load_curve": wandb.Image(str(plot_path))})
            except Exception:
                pass

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "critical_load", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_verification", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase VIII: The Critical Stability Boundary")
    log.info("=" * 60)
    
    test = CriticalLoadTest(cfg, run_dir, run_logger)
    test.execute(jax.random.PRNGKey(cfg.simulation.seed))

if __name__ == "__main__":
    main()
