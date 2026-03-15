"""
Differentiable Routing Agent Training
-------------------------------------
Optimizes the routing parameter (alpha) for Soft-JSQ via gradient descent
through a differentiable Gillespie simulation (DGA).

Instead of sweeping alpha manually, we compute the gradient of the 
continuous-time expected queue length with respect to alpha and minimize it.
"""

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import logging
import hydra
import wandb
from pathlib import Path
from omegaconf import DictConfig

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.differentiable_engine import expected_queue_loss, default_policy
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import setup_wandb, get_run_config

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def train_routing_agent(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    # Initialize Run Capsule (Dynamic Directory + Config Persistence)
    run_dir, run_id = get_run_config(cfg, "dga_training", raw_cfg)

    # Initialize WandB via centralized utility
    run = setup_wandb(cfg, raw_cfg, default_group="dga_training", run_id=run_id, run_dir=run_dir)

    # Use the isolated Run Directory for all outputs
    out_dir = run_dir

    log.info("=" * 60)
    log.info("  Training Differentiable Routing Agent (Soft-JSQ)")
    log.info("=" * 60)
    
    # --- Environment Setup (Heterogenous Servers) ---
    sc = cfg.system
    N = sc.num_servers
    service_rates = jnp.array(sc.service_rates, dtype=jnp.float32) 
    arrival_rate = float(sc.arrival_rate)
    
    params = {
        'num_servers': N,
        'arrival_rate': arrival_rate,
        'service_rates': service_rates,
        'sim_steps': cfg.simulation.dga.sim_steps, # Max jumps per trajectory during training
        'temperature': float(cfg.simulation.dga.temperature)
    }
    
    # SG-2 FIX: Reparametrise via softplus so alpha = softplus(raw_alpha) > 0 always.
    # This eliminates the alpha ≥ 0 hard clip that was corrupting Adam's momentum state
    # (Optax invariant: opt_state must correspond to the unmodified parameter trajectory).
    # softplus^{-1}(0.1) = log(exp(0.1) − 1) ≈ −2.2518, giving alpha₀ ≈ 0.1 at epoch 0.
    raw_alpha = jnp.float32(-2.2518)

    # --- Optimizer Setup ---
    learning_rate = float(cfg.neural_training.dga_learning_rate)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(raw_alpha)
    
    # --- Training Loop ---
    # Wrapper: differentiates through softplus so JAX computes dL/d(raw_alpha)
    # via the chain rule dL/d(alpha) · sigmoid(raw_alpha) automatically.
    def _raw_alpha_loss(raw_a, arrival_rate, service_rates, key,
                        num_servers, sim_steps, temperature, apply_fn):
        return expected_queue_loss(
            jax.nn.softplus(raw_a),
            arrival_rate, service_rates, key,
            num_servers, sim_steps, temperature, apply_fn,
        )

    loss_grad_fn = jax.jit(
        jax.value_and_grad(_raw_alpha_loss, argnums=0),
        static_argnums=(4, 5, 7),  # num_servers, sim_steps, apply_fn
    )
    
    num_epochs = cfg.train_epochs
    key = jax.random.PRNGKey(cfg.simulation.seed)
    
    history_alpha = []
    history_loss = []
    
    log.info(f"Initial setup: Arrival = {arrival_rate}, Service = {service_rates}")
    log.info("Starting Adam optimization...\n")
    log.info(f"{'Epoch':<10} | {'Alpha':<10} | {'Loss (E[Q])':<15} | {'Gradient':<15}")
    log.info("-" * 55)
    
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        
        # alpha derived from raw_alpha for use in routing (and for logging).
        alpha = jax.nn.softplus(raw_alpha)

        loss, grad = loss_grad_fn(
            raw_alpha, 
            params['arrival_rate'], 
            params['service_rates'], 
            subkey, 
            params['num_servers'], 
            params['sim_steps'], 
            params['temperature'],
            default_policy,  # explicit static arg at index 7 — satisfies static_argnums=(4,5,7)
        )
        
        history_alpha.append(float(alpha))
        history_loss.append(float(loss))
        
        log.info(f"{epoch:<10} | {float(alpha):<10.4f} | {float(loss):<15.4f} | {float(grad):<15.4f}")
        
        # Optax update step
        updates, opt_state = optimizer.update(grad, opt_state)
        raw_alpha = optax.apply_updates(raw_alpha, updates)
        # No clip needed: softplus(raw_alpha) > 0 for all finite raw_alpha.
        alpha = jax.nn.softplus(raw_alpha)  # updated value for WandB / jsonl logging

        # Log to WandB
        if run:
            run.log({
                "epoch": epoch,
                "alpha": float(alpha),
                "loss": float(loss),
                "gradient": float(grad)
            })
        
        append_metrics_jsonl({
            "epoch": int(epoch),
            "alpha": float(alpha),
            "loss": float(loss),
            "gradient": float(grad)
        }, out_dir / "metrics.jsonl")

    # --- Plotting ---
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, color='red', marker='o')
    plt.title('Training Loss (Expected Queue Length)')
    plt.xlabel('Epoch')
    plt.ylabel('E[Q]')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_alpha, color='blue', marker='s')
    plt.title('Learned Routing Parameter (Alpha)')
    plt.xlabel('Epoch')
    plt.ylabel('Alpha')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = out_dir / "dga_training_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    if run:
        run.log({"training_curves": wandb.Image(str(plot_path))})
        run.finish()
    
    log.info("-" * 55)
    log.info(f"Training complete. Final alpha: {float(alpha):.4f}")
    log.info(f"Training plots saved to: {plot_path}")

    # SG#7 FIX: Persist the gradient-optimal alpha to a JSON file.
    # All downstream experiments (eval.py, policy_comparison.py) must read this
    # instead of the arbitrary cfg.system.alpha default (1.0).
    import json
    alpha_path = out_dir / "optimal_alpha.json"
    alpha_payload = {
        "alpha": float(alpha),
        "final_loss": float(loss),
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
    }
    alpha_path.write_text(json.dumps(alpha_payload, indent=2), encoding="utf-8")
    log.info(f"[SG#7] Optimal alpha={float(alpha):.4f} persisted to {alpha_path}")

if __name__ == "__main__":
    train_routing_agent()
