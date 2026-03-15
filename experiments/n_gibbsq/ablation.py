"""
N-GibbsQ ablation study.

Evaluates contributions of log-state normalization and zero-initialized
final layers to overall performance.

Variants Tested:
1. Full model
2. No Preprocessing (Remove log1p normalization)
3. No Zero Init (Use default random initialization for the final layer)
4. Random Policy (Baseline comparison)
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
import dataclasses

import optax

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.differentiable_engine import simulate_dga_jax, default_policy, expected_queue_loss
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Pure functional bridge."""
    return model(Q)

class AblationRouter(NeuralRouter):
    """Modified router for ablation testing."""
    def __init__(self, num_servers, config, key, ablate_log=False, ablate_zero=False):
        # Create a modified config for this ablation variant.
        # Use dataclasses.replace because NeuralConfig is a plain @dataclass, not a JAX pytree.
        variant_config = dataclasses.replace(
            config,
            preprocessing="none" if ablate_log else config.preprocessing,
            init_type="standard" if ablate_zero else config.init_type
        )
        
        super().__init__(num_servers=num_servers, config=variant_config, key=key)

    def __call__(self, Q):
        # Delegate to the base class which now supports config-driven preprocessing
        return super().__call__(Q)

class AblationStudy:
    """
    Scientific suite to decompose N-GibbsQ performance.
    """
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float32)
        self.arrival_rate = float(cfg.system.arrival_rate)
        self.sim_steps = cfg.simulation.dga.sim_steps
        self.temperature = float(cfg.simulation.dga.temperature)

    def execute(self, key: PRNGKeyArray):
        """Runs all ablation variants."""
        keys = jax.random.split(key, 5)
        
        # Variants: (Name, ablate_log, ablate_zero)
        variants = [
            ("Full Model", False, False),
            ("Ablated: No Log-Norm", True, False),
            ("Ablated: No Zero-Init", False, True),
            ("Uniform Routing (Baseline)", "RANDOM", "RANDOM")
        ]
        
        results = {}
        
        log.info("Starting Ablation Benchmark...")

        # Horizon for BOTH training and evaluation.  Must be defined before the loop
        # so both the uniform baseline branch and the neural training branch can use
        # it consistently.  See PATCH SG4 comment below for rationale.
        _ABLATION_STEPS = 500
        
        for name, a_log, a_zero in variants:
            if name == "Uniform Routing (Baseline)":
                # Uniform routing baseline: softmax(zeros) = 1/N for all servers.
                # Uses scalar params=0.0 to satisfy default_policy's type contract
                # (params: jnp.float32) and share the JIT cache entry with other callers.
                loss = simulate_dga_jax(
                    self.num_servers, self.arrival_rate, self.service_rates, jnp.float32(0.0),
                    _ABLATION_STEPS, keys[2], self.temperature, default_policy
                )
            else:
                router = AblationRouter(self.num_servers, self.cfg.neural, keys[0], a_log, a_zero)
                
                # Use hyperparameters from config
                optimizer = optax.adamw(learning_rate=self.cfg.neural_training.learning_rate, weight_decay=self.cfg.neural_training.weight_decay)
                opt_state = optimizer.init(eqx.filter(router, eqx.is_array))
                
                train_key = keys[1]
                
                # PATCH SG4: Use the literal horizon and epoch count from the comment,
                # not cfg.train_epochs (30) and self.sim_steps (5000).
                # The ablation is designed to measure initialization effects, which are
                # only visible in the early-learning regime (T=500, 15 epochs).
                # Over-training (T=5000, 30 epochs) allows all variants to converge,
                # erasing the initialization signal the study is designed to isolate.
                def _loss_fn(m, k):
                    return expected_queue_loss(m, self.arrival_rate, self.service_rates, k, self.num_servers, _ABLATION_STEPS, self.temperature, evaluate_model)
                
                @eqx.filter_jit
                def train_step(model_t, opt_state_t, key_t):
                    l, grads = eqx.filter_value_and_grad(_loss_fn)(model_t, key_t)
                    updates, new_opt_state = optimizer.update(grads, opt_state_t, model_t)
                    new_model = eqx.apply_updates(model_t, updates)
                    return l, new_model, new_opt_state
                
                ablation_epochs = 15
                log.info(f"   [Training] {name} for {ablation_epochs} epochs at T={_ABLATION_STEPS}...")
                for ep in range(ablation_epochs):
                    train_key, subkey = jax.random.split(train_key)
                    l, router, opt_state = train_step(router, opt_state, subkey)
                    log.info(f"      Epoch {ep:2d} | Loss: {l:7.4f}")
                
                # Evaluate at the same T=_ABLATION_STEPS horizon used for training
                # (apples-to-apples initialization-regime comparison)
                loss = simulate_dga_jax(
                    self.num_servers, self.arrival_rate, self.service_rates, router, 
                    _ABLATION_STEPS, keys[2], self.temperature, evaluate_model
                )
            
            results[name] = float(loss)
            log.info(f"   {name:<25} | Final Eval Loss: {results[name]:.4f}")

            append_metrics_jsonl({
                "variant": name,
                "loss": float(loss)
            }, self.run_dir / "metrics.jsonl")

        self._plot(results)

    def _plot(self, results):
        """Generates the ablation bar chart."""
        plt.figure(figsize=(10, 6))
        names = list(results.keys())
        values = list(results.values())
        
        bars = plt.bar(names, values, color=['#2ecc71', '#e67e22', '#e74c3c', '#95a5a6'])
        plt.title('N-GibbsQ Component Ablation Study (Initialization Regime)')
        plt.ylabel('Expected Queue Length $\mathbb{E}[Q]$')
        plt.xticks(rotation=15)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add labels on top
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom')
            
        plot_path = self.run_dir / "ablation_study.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        log.info(f"Ablation complete. Plot saved to {plot_path}")
        
        if self.run_logger:
            self.run_logger.log({"ablation_results": results})
            try:
                import wandb
                self.run_logger.log({"ablation_chart": wandb.Image(str(plot_path))})
            except Exception:
                pass

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "ablation_study", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_verification", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase IX: Scientific Ablation Study")
    log.info("=" * 60)
    
    study = AblationStudy(cfg, run_dir, run_logger)
    study.execute(jax.random.PRNGKey(cfg.simulation.seed))

if __name__ == "__main__":
    main()
