"""
Differentiable Routing Agent Training
-------------------------------------
This script demonstrates the "Tier-1 Idea Project" functionality: using 
Gradient Descent (via Optax and JAX) to automatically learn the optimal
routing parameter (alpha) for Soft-JSQ in a heterogenous server environment.

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

from moeq.core.config import hydra_to_config, validate
from moeq.engines.differentiable_engine import expected_queue_loss
from moeq.utils.exporter import append_metrics_jsonl
from moeq.utils.logging import setup_wandb, get_run_config

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
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
    N = 4
    # One highly efficient server, 3 slow servers.
    # A naive router will overload the slow ones. JSQ will underutilize the fast one.
    # Soft-JSQ needs to find the perfect alpha.
    service_rates = jnp.array([10.0, 1.0, 1.0, 1.0]) 
    arrival_rate = 8.0 # High load
    
    params = {
        'num_servers': N,
        'arrival_rate': arrival_rate,
        'service_rates': service_rates,
        'sim_steps': 500, # Max jumps per trajectory during training
        'temperature': 0.1
    }
    
    # Initialize alpha (start at 0.0 = Uniform Routing)
    alpha = jnp.float32(0.0) 
    
    # --- Optimizer Setup ---
    learning_rate = 0.5 # Aggressive LR for demonstration
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(alpha)
    
    # --- Training Loop ---
    # We use value_and_grad and mark num_servers, sim_steps as static
    # signature: alpha, arrival_rate, service_rates, key, num_servers, sim_steps, temperature
    loss_grad_fn = jax.jit(
        jax.value_and_grad(expected_queue_loss, argnums=0),
        static_argnums=(4, 5) 
    )
    
    num_epochs = 30
    key = jax.random.PRNGKey(42)
    
    history_alpha = []
    history_loss = []
    
    log.info(f"Initial setup: Arrival = {arrival_rate}, Service = {service_rates}")
    log.info("Starting Adam optimization...\n")
    log.info(f"{'Epoch':<10} | {'Alpha':<10} | {'Loss (E[Q])':<15} | {'Gradient':<15}")
    log.info("-" * 55)
    
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        
        # Forward pass and backpropagation through the CTMC!
        loss, grad = loss_grad_fn(
            alpha, 
            params['arrival_rate'], 
            params['service_rates'], 
            subkey, 
            params['num_servers'], 
            params['sim_steps'], 
            params['temperature']
        )
        
        history_alpha.append(float(alpha))
        history_loss.append(float(loss))
        
        log.info(f"{epoch:<10} | {float(alpha):<10.4f} | {float(loss):<15.4f} | {float(grad):<15.4f}")
        
        # Optax update step
        updates, opt_state = optimizer.update(grad, opt_state)
        alpha = optax.apply_updates(alpha, updates)
        
        # Clip alpha to be non-negative (can't route away from short queues)
        alpha = jnp.maximum(alpha, 0.0)

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
    log.info(f"Training Complete! Final optimally learned Alpha: {float(alpha):.4f}")
    log.info(f"Training plots saved to: {plot_path}")

if __name__ == "__main__":
    train_routing_agent()
