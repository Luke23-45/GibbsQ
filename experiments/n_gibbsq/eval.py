"""
N-GibbsQ evaluation: neural vs analytical parity.

Compares trained NeuralRouter against scalar-alpha GibbsQ
routing policy in a heterogeneous server environment.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
from jaxtyping import Array, Float, PRNGKeyArray

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.differentiable_engine import simulate_dga_jax, default_policy
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Pure functional bridge required by the DGA engine."""
    return model(Q)

class NeuralTuringTest:
    """
    Verification suite comparing NeuralRouter to the analytical baseline.
    """
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float32)
        self.arrival_rate = float(cfg.system.arrival_rate)
        self.temperature = float(cfg.simulation.dga.temperature)
        self.sim_steps = cfg.simulation.dga.sim_steps

        # We run multiple replications to get statistically stable results
        self.num_reps = int(cfg.simulation.num_replications)
        
        # JIT-compile vectorized simulation
        self.vmap_simulate = jax.jit(
            jax.vmap(simulate_dga_jax, in_axes=(None, None, None, None, None, 0, None, None)), 
            static_argnames=("num_servers", "sim_steps", "apply_fn")
        )

    def execute(self, key: PRNGKeyArray):
        """Executes the parity test."""
        k1, k2 = jax.random.split(key, 2)
        
        log.info(f"Environment: N={self.num_servers}, Load={self.arrival_rate}")
        
        # --- 1. The Expert: GibbsQ (Analytically Optimal) ---
        optimal_alpha = jnp.float32(self.cfg.system.alpha)
        log.info(f"\n[GibbsQ baseline (alpha={optimal_alpha})]")
        
        # CRN: use shared keys for both neural and GibbsQ for variance reduction
        shared_keys = jax.random.split(k1, self.num_reps)
        gibbsq_loss_array = self.vmap_simulate(
            self.num_servers, self.arrival_rate, self.service_rates, optimal_alpha, self.sim_steps, shared_keys, self.temperature, default_policy
        )
        mean_gibbsq_loss = float(jnp.mean(gibbsq_loss_array))
        
        # --- 2. The Challenger: N-GibbsQ (Neural Network) ---
        log.info("\n[Loading Challenger: N-GibbsQ Neural Router]")
        
        # Read the pointer to the latest dynamically generated run
        pointer_path = Path(self.cfg.output_dir) / "neural_training" / "latest_weights.txt"
        
        if not pointer_path.exists():
            log.error(f"Weights pointer not found at {pointer_path}. You must run training (n_gibbsq/train.py) first.")
            return
            
        model_path_str = pointer_path.read_text(encoding='utf-8').strip()
        model_path = Path(model_path_str)
        
        if not model_path.exists():
            log.error(f"Trained weight file missing at {model_path}. Please rerun training (train.py).")
            return

        # Re-initialize skeleton and load weights securely
        skeleton = NeuralRouter(num_servers=self.num_servers, hidden_size=64, key=k2)
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        
        # SG#16 Fix: Validate that the loaded model matches the current config
        if model.layers[0].weight.shape[1] != self.num_servers:
            log.error(f"Model shape mismatch! Loaded model expects N={model.layers[0].weight.shape[1]}, but eval config requires N={self.num_servers}.")
            return

        
        neural_keys = shared_keys  # CRN: same keys as GibbsQ
        n_gibbsq_loss_array = self.vmap_simulate(
            self.num_servers, self.arrival_rate, self.service_rates, model, self.sim_steps, neural_keys, self.temperature, evaluate_model
        )
        mean_neural_loss = float(jnp.mean(n_gibbsq_loss_array))
        
        self._report_results(mean_gibbsq_loss, mean_neural_loss)

    def _report_results(self, mean_gibbsq_loss: float, mean_neural_loss: float):
        """Calculates parity metrics and logs the final showdown string."""
        log.info("\n" + "=" * 60)
        log.info("  PARITY RESULTS")
        log.info("=" * 60)
        log.info(f"GibbsQ (Scalar Math) Expected Queue: {mean_gibbsq_loss:.4f}")
        log.info(f"N-GibbsQ (Neural Net) Expected Queue:  {mean_neural_loss:.4f}")
        
        diff = mean_neural_loss - mean_gibbsq_loss
        perc = (diff / mean_gibbsq_loss) * 100 if mean_gibbsq_loss > 0 else 0
        
        status = ""
        if diff <= 0:
            log.info("\n[+] OUTCOME: Neural router matched or exceeded analytical baseline.")
            log.info("    Performance gap: <= 0%.")
            status = "MATCHED"
        elif perc < 25.0:
            log.info(f"\n[+] OUTCOME: Neural router within {perc:.1f}% of analytical baseline.")
            log.info("    Neural router converged near analytical optimum.")
            status = "SUCCESS"
        else:
            log.warning(f"\n[-] OUTCOME: The Neural Network failed to match GibbsQ by {perc:.1f}%.")
            status = "FAILED"
            
        # Log to WandB
        if self.run_logger:
            self.run_logger.log({
                "gibbsq_loss": mean_gibbsq_loss,
                "n_gibbsq_loss": mean_neural_loss,
                "parity_diff_percentage": perc
            })

        summary_path = self.run_dir / "parity_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Status: {status}\n")
            f.write(f"GibbsQ E[Q]: {mean_gibbsq_loss:.4f}\n")
            f.write(f"N-GibbsQ E[Q]: {mean_neural_loss:.4f}\n")
            f.write(f"Performance Gap: {perc:.2f}%\n")

        append_metrics_jsonl({
            "gibbsq_loss": mean_gibbsq_loss,
            "n_gibbsq_loss": mean_neural_loss,
            "parity_diff_percentage": perc,
            "status": status
        }, self.run_dir / "metrics.jsonl")


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "neural_parity", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_parity", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase 4: N-GibbsQ Parity Evaluation")
    log.info("=" * 60)
    
    test_suite = NeuralTuringTest(cfg, run_dir, run_logger)
    
    seed_key = jax.random.PRNGKey(cfg.simulation.seed)
    test_suite.execute(seed_key)

if __name__ == "__main__":
    main()
