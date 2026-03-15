"""
N-GibbsQ Phase VI: Jacobian Check (AD vs Finite Differences)
-----------------------------------------------------------
Compares JAX autodiff gradients against central finite differences at FP64.
A relative error below 1e-3 confirms that the DGA produces correct gradients.
"""

import jax
import jax.flatten_util
# Enable FP64 before anything else
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import equinox as eqx
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
from jaxtyping import Array, Float, PRNGKeyArray
import time
import dataclasses

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.differentiable_engine import expected_queue_loss
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Pure functional bridge."""
    return model(Q)

class JacobianValidator:
    """
    Precision verification engine for N-GibbsQ gradients.
    """
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float64)
        self.arrival_rate = float(cfg.system.arrival_rate)
        self.temperature = float(cfg.simulation.dga.temperature)
        
        # CFD Parameters
        self.epsilon = 1e-7  # Optimal for FP64 central difference
        self.sim_steps = cfg.simulation.dga.sim_steps

    def _loss_wrapper(self, model: NeuralRouter, key: PRNGKeyArray) -> jnp.float64:
        """Scalar loss function for gradient check."""
        return expected_queue_loss(
            params=model,
            arrival_rate=self.arrival_rate,
            service_rates=self.service_rates,
            key=key,
            num_servers=self.num_servers,
            sim_steps=self.sim_steps,
            temperature=self.temperature,
            apply_fn=evaluate_model
        )

    def run_verification(self, key: PRNGKeyArray):
        """Performs the full Jacobian vs CFD showdown."""
        k1, k2 = jax.random.split(key)
        
        # SG#13 FIX: Use actual model hidden_size (from config) instead of 16.
        # This makes the Jacobian check validate the EXACT architecture used in training.
        check_config = dataclasses.replace(self.cfg.neural)
        log.info(f"Jacobian check using hidden_size={check_config.hidden_size} (matching training)")
        model = NeuralRouter(num_servers=self.num_servers, config=check_config, key=k1)
        # PATCH SG3: Cast all model parameters to float64 to match jax_enable_x64 mode.
        # eqx.nn.Linear initialises weights as float32 by default even when x64 is enabled.
        # With float32 weights and epsilon=1e-7 (near float32 machine epsilon ~1.19e-7),
        # central finite differences suffer catastrophic cancellation.
        params_f64, static = eqx.partition(model, eqx.is_array)
        params_f64 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params_f64)
        model = eqx.combine(params_f64, static)
        
        log.info(f"Numerical Precision: FP64 (Enforced)")
        log.info(f"Perturbation (epsilon): {self.epsilon}")
        log.info(f"Simulation Steps (T): {self.sim_steps}")
        log.info("-" * 40)

        # 1. Compute JAX Autodiff Gradients
        log.info("[1/3] Computing JAX Gradients (Reverse-Mode AD)...")
        loss_fn = lambda m: self._loss_wrapper(m, k2)
        jax_loss, jax_grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        # 2. Compute Numerical Central Finite Difference Gradients
        log.info("[2/3] Computing Numerical Gradients (Finite Differences)...")
        # Flatten model to iterate over every scalar parameter
        params, static = eqx.partition(model, eqx.is_array)
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        
        num_params = flat_params.size
        cfd_grads_flat = jnp.zeros_like(flat_params)
        
        start_time = time.time()
        # We sample indices if num_params is huge, but here hidden_size=16 is small (~500 params)
        for i in range(num_params):
            if i % 100 == 0 and i > 0:
                log.info(f"   Progress: {i}/{num_params} parameters checked...")
            
            # Plus perturbation
            flat_plus = flat_params.at[i].add(self.epsilon)
            model_plus = eqx.combine(unravel_fn(flat_plus), static)
            l_plus = self._loss_wrapper(model_plus, k2)
            
            # Minus perturbation
            flat_minus = flat_params.at[i].add(-self.epsilon)
            model_minus = eqx.combine(unravel_fn(flat_minus), static)
            l_minus = self._loss_wrapper(model_minus, k2)
            
            # Central difference: (f(x+h) - f(x-h)) / 2h
            cfd_grads_flat = cfd_grads_flat.at[i].set((l_plus - l_minus) / (2 * self.epsilon))
            
        duration = time.time() - start_time
        log.info(f"Numerical integration completed in {duration:.2f}s.")

        # 3. Comparative Diagnostics
        log.info("[3/3] Performing Cross-Validation Diagnostics...")
        
        # Flatten JAX grads
        jax_grads_flat, _ = jax.flatten_util.ravel_pytree(eqx.filter(jax_grads, eqx.is_array))
        
        # Calculate Errors
        abs_error = jnp.abs(jax_grads_flat - cfd_grads_flat)
        denom = jnp.maximum(jnp.maximum(jnp.abs(jax_grads_flat), jnp.abs(cfd_grads_flat)), 1e-12)
        rel_error = abs_error / denom
        
        max_rel_err = float(jnp.max(rel_error))
        mean_rel_err = float(jnp.mean(rel_error))
        
        log.info("\n" + "=" * 50)
        log.info(f"  JACOBIAN FIDELITY REPORT")
        log.info("=" * 50)
        log.info(f"Max Relative Error:  {max_rel_err:.2e}")
        log.info(f"Mean Relative Error: {mean_rel_err:.2e}")
        log.info(f"Status: {'PASS' if max_rel_err < self.cfg.verification.jacobian_rel_tol else 'FAIL'}")
        log.info("=" * 50)

        # Logging to WandB
        if self.run_logger:
            self.run_logger.log({
                "max_rel_gradient_error": max_rel_err,
                "mean_rel_gradient_error": mean_rel_err,
                "sim_steps": self.sim_steps
            })

        append_metrics_jsonl({
            "max_rel_gradient_error": max_rel_err,
            "mean_rel_gradient_error": mean_rel_err,
            "sim_steps": self.sim_steps,
            "status": "PASS" if max_rel_err < self.cfg.verification.jacobian_rel_tol else "FAIL"
        }, self.run_dir / "metrics.jsonl")

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    # Force debug settings if requested to keep FD check fast
    if raw_cfg.get("debug", False):
        cfg.simulation.ssa.sim_time = 100
    
    run_dir, run_id = get_run_config(cfg, "jacobian_check", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_verification", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase VI: Jacobian Fidelity (Numerical AD Check)")
    log.info("=" * 60)
    
    validator = JacobianValidator(cfg, run_dir, run_logger)
    
    seed_key = jax.random.PRNGKey(cfg.simulation.seed)
    validator.run_verification(seed_key)

if __name__ == "__main__":
    main()
