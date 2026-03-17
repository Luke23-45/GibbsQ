"""
N-GibbsQ Phase 1: Gradient Fidelity Check
---------------------------------------
Verifies that the gradient of the simulated expected queue length
survives backpropagation through time across increasing simulation horizons.
If gradients vanish, neural routing will fail to learn.

SG-8 PATCH: Scope & Limitations Documentation
-------------------------------------------
This script validates gradient survival through the JAX-based forward-mode
differentiable engine (DGA). It is a diagnostic for the "Tracing Death"
bottleneck and vanishing gradients in deterministic surrogate models.

IMPORTANT: This does NOT validate the REINFORCE stochastic gradient estimator.
For likelihood-ratio gradient validation (unbiasedness check), use:
experiments/testing/reinforce_gradient_check.py.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.differentiable_engine import (
    simulate_dga_jax_dynamic_steps,
)
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

class GradientFidelityChecker:
    """
    Object-Oriented suite to mathematically verify gradient survival over long horizons.
    """
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        precision = jnp.float32 if cfg.jax.precision == "float32" else jnp.float64
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=precision)
        self.arrival_rate = float(cfg.system.arrival_rate)
        self.temperature = float(cfg.simulation.dga.temperature)
        
        # Test gradient at a stable non-zero scalar routing parameter
        self.alpha_test = jnp.array(cfg.system.alpha, dtype=precision) 
        
        # Horizons to sweep, capped by config
        max_h = cfg.simulation.dga.sim_steps
        self.horizons = [h for h in [100, 500, 1000, 2500, 5000] if h <= max_h]

        # Forward-mode JVP requires sim_steps to be static: lax.fori_loop inside
        # simulate_dga_jax_dynamic_steps needs a concrete upper bound when traced
        # under jax.jvp (JAX >= 0.4.7 raises ConcretizationTypeError otherwise).
        self.loss_grad_fn = jax.jit(
            self._loss_and_grad,
            static_argnums=(4, 5),
        )

    @staticmethod
    def _loss_and_grad(alpha, arrival_rate, service_rates, key, num_servers, sim_steps, temperature):
        def _loss_fn(alpha_param):
            return simulate_dga_jax_dynamic_steps(
                num_servers=num_servers,
                arrival_rate=arrival_rate,
                service_rates=service_rates,
                params=alpha_param,
                sim_steps=sim_steps,
                key=key,
                temperature=temperature,
            )

        tangent = jnp.array(1.0, dtype=alpha.dtype)
        loss_val, grad_val = jax.jvp(_loss_fn, (alpha,), (tangent,))
        return loss_val, grad_val

    def execute(self, key: jax.random.PRNGKey):
        """Sweeps horizons and measures gradient survival."""
        log.info(f"{'Horizon (T)':<15} | {'Loss (E[Q])':<15} | {'Gradient ||grad||':<20}")
        log.info("-" * 55)
        
        grad_norms = []
        losses = []

        for T in self.horizons:
            key, subkey = jax.random.split(key)
            
            try:
                loss, grad = self.loss_grad_fn(
                    self.alpha_test, 
                    self.arrival_rate, 
                    self.service_rates, 
                    subkey, 
                    self.num_servers, 
                    T, 
                    self.temperature
                )
                grad_norm = float(jnp.abs(grad))
                loss_val = float(loss)
                
                grad_norms.append(grad_norm)
                losses.append(loss_val)
                
                log.info(f"{T:<15} | {loss_val:<15.4f} | {grad_norm:<20.6f}")
                
                metrics = {"horizon": T, "loss": loss_val, "gradient_norm": grad_norm}
                append_metrics_jsonl(metrics, self.run_dir / "gradient_fidelity_metrics.jsonl")
                if self.run_logger:
                    self.run_logger.log(metrics)
                    
            except Exception as e:
                log.error(f"{T:<15} | FAILED: {str(e)}")
                grad_norms.append(0.0)
                losses.append(0.0)

        self._save_assets(grad_norms, losses)
        self._automated_analysis(grad_norms)

    def _save_assets(self, grad_norms: list, losses: list):
        """Persists the diagnostic plots."""
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Simulation Steps (Horizon T)', fontsize=12)
        ax1.set_ylabel('Gradient Magnitude ||grad||', color=color, fontsize=12)
        ax1.plot(self.horizons, grad_norms, marker='o', color=color, linewidth=2, label='||Gradient||')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Loss (Expected Queue Length)', color=color, fontsize=12)  
        ax2.plot(self.horizons, losses, marker='s', color=color, linestyle='--', alpha=0.7, label='Loss')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.suptitle('Gradient Fidelity Over Time (CTMC Horizon)', fontsize=14)
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        
        fig.tight_layout()
        plot_path = self.run_dir / "gradient_decay.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        log.info("-" * 55)
        log.info(f"Fidelity check complete. Plot saved to {plot_path}")

        if self.run_logger:
            try:
                import wandb
                self.run_logger.log({"gradient_decay": wandb.Image(str(plot_path))})
            except Exception:
                pass

    def _automated_analysis(self, grad_norms: list):
        """Log diagnostic conclusions."""
        if len(grad_norms) > 0 and grad_norms[-1] < 1e-4 and grad_norms[0] > 1e-2:
            log.warning("\n[!] WARNING: Severe Vanishing Gradient detected at high T.")
            log.warning("    A neural network will fail to learn over infinite horizons without TBPTT or Curriculum.")
        elif len(grad_norms) > 0 and grad_norms[-1] > 1e-4:
            log.info("\n[+] Gradients survive long horizons. Safe to proceed to training.")

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "gradient_fidelity", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_diagnostics", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase 1: N-GibbsQ Gradient Fidelity Check")
    log.info("=" * 60)
    
    checker = GradientFidelityChecker(cfg, run_dir, run_logger)
    
    seed_key = jax.random.PRNGKey(cfg.simulation.seed)
    checker.execute(seed_key)

if __name__ == "__main__":
    main()
