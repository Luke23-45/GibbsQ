"""
N-GibbsQ Phase III: Curriculum Training
---------------------------------------
Trains the NeuralRouter using curriculum scheduling and AdamW.
Starts with short simulation horizons and increases over epochs.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
from jaxtyping import Array, Float, PRNGKeyArray

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.differentiable_engine import expected_queue_loss
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Pure functional bridge required by the DGA engine."""
    return model(Q)

class NeuralCurriculumTrainer:
    """Trains a NeuralRouter using AdamW with curriculum scheduling.
    Handles the Optax trajectory, model state, and curriculum scheduling.
    """
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float32)
        self.arrival_rate = float(cfg.system.arrival_rate)
        self.temperature = float(cfg.simulation.temperature)
        
        # Weight decay regularizes large routing weights.
        self.learning_rate = float(cfg.neural_training.learning_rate)
        self.optimizer = optax.adamw(learning_rate=self.learning_rate, weight_decay=float(cfg.neural_training.weight_decay))

    def _loss_fn(self, model: NeuralRouter, key: PRNGKeyArray, sim_steps: int) -> jnp.float32:
        """Computes the expected queue length over a specific horizon."""
        return expected_queue_loss(
            params=model,
            arrival_rate=self.arrival_rate,
            service_rates=self.service_rates,
            key=key,
            num_servers=self.num_servers,
            sim_steps=sim_steps,
            temperature=self.temperature,
            apply_fn=evaluate_model
        )

    def execute(self, key: PRNGKeyArray):
        """Executes the curriculum training loop."""
        key, subkey = jax.random.split(key)
        model = NeuralRouter(num_servers=self.num_servers, hidden_size=64, key=subkey)
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        # Curriculum Schedule: (Epochs, Simulation_Horizon)
        # This prevents the initial vanishing gradient shock.
        curriculum = [tuple(phase) for phase in self.cfg.neural_training.curriculum]

        history_loss = []
        epoch_count = 0

        log.info(f"{'Epoch':<8} | {'Horizon':<8} | {'Loss (E[Q])':<15} | {'Max ||w||':<15}")
        log.info("-" * 55)

        for phase_epochs, T_horizon in curriculum:
            log.info(f"--- Entering Curriculum Phase: Horizon T={T_horizon} ---")
            
            # Re-compile JIT for the specific static horizon T
            @eqx.filter_jit
            def train_step(model_t: NeuralRouter, opt_state_t: optax.OptState, key_t: PRNGKeyArray):
                # Dynamically bind the filter_value_and_grad so it acts on `model_t` (the first arg)
                loss_fn = lambda m, k, s: self._loss_fn(m, k, s)
                loss, grads = eqx.filter_value_and_grad(loss_fn)(model_t, key_t, T_horizon)
                
                updates, new_opt_state = self.optimizer.update(grads, opt_state_t, model_t)
                new_model = eqx.apply_updates(model_t, updates)
                return loss, new_model, new_opt_state

            for _ in range(phase_epochs):
                key, subkey = jax.random.split(key)
                loss, model, opt_state = train_step(model, opt_state, subkey)
                loss_val = float(loss)
                
                # Model health telemetry
                max_w = float(max([jnp.max(jnp.abs(l.weight)) for l in model.layers if hasattr(l, 'weight')]))
                
                history_loss.append(loss_val)
                log.info(f"{epoch_count:<8} | {T_horizon:<8} | {loss_val:<15.4f} | {max_w:<15.4f}")
                
                metrics = {
                    "epoch": epoch_count,
                    "horizon": T_horizon,
                    "loss": loss_val,
                    "max_weight": max_w
                }
                append_metrics_jsonl(metrics, self.run_dir / "n_gibbsq_metrics.jsonl")
                if self.run_logger:
                    self.run_logger.log(metrics)
                    
                epoch_count += 1

        self._save_assets(model, history_loss)

    def _save_assets(self, model: NeuralRouter, history_loss: list):
        """Persists model weights and loss trajectory."""
        plt.figure(figsize=(8, 5))
        plt.plot(history_loss, color='purple', linewidth=2)
        plt.title('N-GibbsQ Curriculum Training Convergence')
        plt.xlabel('Global Epoch')
        plt.ylabel('Expected Queue Length $\\mathbb{E}[Q]$')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Shift to T=2000')
        plt.axvline(x=50, color='gray', linestyle=':', alpha=0.5, label='Shift to T=5000')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(self.run_dir / "n_gibbsq_training_curve.png", dpi=300)
        plt.close()

        model_path = self.run_dir / "n_gibbsq_weights.eqx"
        eqx.tree_serialise_leaves(model_path, model)
        
        # Write a pointer to the latest weights directory for downstream phases.
        pointer_path = self.run_dir.parent / "latest_weights.txt"
        pointer_path.write_text(str(model_path.resolve()), encoding='utf-8')
        
        log.info("-" * 55)
        log.info(f"Training Complete! Final Loss: {history_loss[-1]:.4f}")
        log.info(f"Neural weights preserved at {model_path}")

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "neural_training", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_training", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase 3: N-GibbsQ Curriculum Training")
    log.info("=" * 60)
    
    trainer = NeuralCurriculumTrainer(cfg, run_dir, run_logger)
    
    # Execute deterministic seeded training
    seed_key = jax.random.PRNGKey(42)
    trainer.execute(seed_key)

if __name__ == "__main__":
    main()
