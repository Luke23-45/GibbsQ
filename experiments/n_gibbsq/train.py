"""
N-GibbsQ Phase III: Curriculum Training
---------------------------------------
Trains the NeuralRouter using curriculum scheduling and AdamW.
Starts with short simulation horizons and increases over epochs.
"""

import jax
import jax.numpy as jnp
import optax
import math
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
        self.temperature = float(cfg.simulation.dga.temperature)
        
        # Weight decay regularizes large routing weights.
        self.learning_rate = float(cfg.neural_training.learning_rate)
        self.optimizer = optax.adamw(learning_rate=self.learning_rate, weight_decay=float(cfg.neural_training.weight_decay))

    def _loss_fn(self, model: NeuralRouter, key: PRNGKeyArray, sim_steps: int, temperature: float) -> jnp.float32:
        """Computes the expected queue length over a specific horizon."""
        return expected_queue_loss(
            params=model,
            arrival_rate=self.arrival_rate,
            service_rates=self.service_rates,
            key=key,
            num_servers=self.num_servers,
            sim_steps=sim_steps,
            temperature=temperature,
            apply_fn=evaluate_model
        )

    def execute(self, key: PRNGKeyArray):
        """Executes the curriculum training loop."""
        key, subkey = jax.random.split(key)
        model = NeuralRouter(num_servers=self.num_servers, config=self.cfg.neural, key=subkey)
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        # Curriculum Schedule: (Epochs, Simulation_Horizon)
        # This prevents the initial vanishing gradient shock.
        curriculum = [tuple(phase) for phase in self.cfg.neural_training.curriculum]

        history_loss = []
        epoch_count = 0

        log.info(f"{'Epoch':<8} | {'Horizon':<8} | {'Loss (E[Q])':<15} | {'Max ||w||':<15}")
        log.info("-" * 55)

        # Temperature annealing: linearly decay from base temperature to
        # a near-hard assignment (0.05) across curriculum phases.  This
        # shrinks the DGA→SSA domain gap because the final training
        # phases use Gumbel-Softmax closer to a hard argmax.
        # SG#20 FIX: Read min_temperature from config instead of hardcoded value.
        # Old value was 0.05, which risks catastrophic forgetting at low temperatures.
        _min_temp = float(getattr(self.cfg.neural_training, 'min_temperature', 0.1))
        num_phases = len(curriculum)
        
        for phase_idx, (phase_epochs, T_horizon) in enumerate(curriculum):
            if num_phases > 1:
                phase_temp = self.temperature - (self.temperature - _min_temp) * (phase_idx / (num_phases - 1))
            else:
                phase_temp = self.temperature
            log.info(f"--- Entering Curriculum Phase: Horizon T={T_horizon}, temp={phase_temp:.3f} ---")
            
            # Re-compile JIT for the specific static horizon T
            @eqx.filter_jit
            def train_step(model_t: NeuralRouter, opt_state_t: optax.OptState, key_t: PRNGKeyArray):
                # Dynamically bind the filter_value_and_grad so it acts on `model_t` (the first arg)
                loss_fn = lambda m, k, s: self._loss_fn(m, k, s, phase_temp)
                loss, grads = eqx.filter_value_and_grad(loss_fn)(model_t, key_t, T_horizon)
                
                updates, new_opt_state = self.optimizer.update(grads, opt_state_t, model_t)
                new_model = eqx.apply_updates(model_t, updates)
                return loss, new_model, new_opt_state

            for _ in range(phase_epochs):
                key, subkey = jax.random.split(key)
                loss, model, opt_state = train_step(model, opt_state, subkey)
                loss_val = float(loss)
                
                # SG#10 FIX: Skip update and warn on non-finite loss.
                # A NaN/Inf loss means the DGA trajectory overflowed (tau→∞
                # when a0→0). Applying NaN gradients corrupts the optimizer
                # momentum state and produces invalid weight files.
                if not jnp.isfinite(loss):
                    log.warning(
                        f"  [!] Epoch {epoch_count}: Non-finite loss={loss_val:.4f}. "
                        f"Skipping gradient update. Check arrival_rate/service_rates ratio."
                    )
                    epoch_count += 1
                    continue

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
        # SG#10 FIX: Refuse to serialise if the training run produced
        # no finite losses (all steps were NaN/skipped). An empty or
        # all-NaN history_loss means the model was never updated.
        valid_losses = [l for l in history_loss if math.isfinite(l)]
        if not valid_losses:
            log.error(
                "[!] _save_assets: No finite loss recorded. "
                "Model weights NOT saved. Re-run training with a stable config."
            )
            return

        plt.figure(figsize=(8, 5))
        plt.plot(history_loss, color='purple', linewidth=2)
        plt.title('N-GibbsQ Curriculum Training Convergence')
        plt.xlabel('Global Epoch')
        plt.ylabel('Expected Queue Length $\\mathbb{E}[Q]$')
        plt.grid(True, alpha=0.3)
        # Plot phase transition markers from curriculum config
        cumulative = 0
        for phase_idx, phase in enumerate(self.cfg.neural_training.curriculum[:-1]):
            cumulative += phase[0]
            next_T = self.cfg.neural_training.curriculum[phase_idx + 1][1]
            plt.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5, label=f'Shift to T={next_T}')
        plt.legend()
        plt.tight_layout()
        
        plot_path = self.run_dir / "n_gibbsq_training_curve.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        if self.run_logger:
            try:
                import wandb
                self.run_logger.log({"training_curve": wandb.Image(str(plot_path))})
            except Exception:
                pass

        model_path = self.run_dir / "n_gibbsq_weights.eqx"
        eqx.tree_serialise_leaves(model_path, model)
        
        # SG#5 FIX: Write pointer to a FIXED canonical path, independent of
        # CWD.  Path(__file__).resolve().parents[2] is the project root
        # regardless of which directory Python is invoked from (Jupyter,
        # CI runners, direct sub-directory invocation, etc.).
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
        pointer_dir = _PROJECT_ROOT / "outputs" / "small"

        pointer_dir.mkdir(parents=True, exist_ok=True)
        pointer_path = pointer_dir / "latest_weights.txt"
        pointer_path.write_text(str(model_path.resolve()), encoding='utf-8')
        log.info(f"[SG#5] Model pointer written to: {pointer_path.resolve()}")
        
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
    seed_key = jax.random.PRNGKey(cfg.simulation.seed)
    trainer.execute(seed_key)

if __name__ == "__main__":
    main()
