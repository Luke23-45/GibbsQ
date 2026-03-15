"""
N-GibbsQ critical load test.

Tests N-GibbsQ as ρ → 1, where queueing systems become unstable.

Compares the neural router's expected queue length against GibbsQ
at load factors approaching the critical boundary.

SG-D FIX: Both sides now measured on the true Gillespie SSA (not DGA surrogate).
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
import functools

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_engine import run_replications_jax
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl


class _NeuralSSAPolicy:
    """Identical to eval.py."""
    def __init__(self, model):
        import jax as _jax
        self._model = model
        @eqx.filter_jit
        def _forward(m, x): return _jax.nn.softmax(m(x))
        self._forward = _forward
        @functools.lru_cache(maxsize=131072)
        def _get_probs(q_tuple):
            probs = self._forward(self._model, jnp.array(q_tuple, dtype=jnp.float32))
            probs_np = np.array(probs, dtype=np.float64)
            return probs_np / probs_np.sum()
        self._get_probs = _get_probs
    def __call__(self, Q, rng): return self._get_probs(tuple(Q))


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
        self.ssa_sim_time = cfg.simulation.ssa.sim_time
        self.ssa_sample_interval = cfg.simulation.ssa.sample_interval
        
        # Symmetrical and Extreme ρ range
        self.rho_vals = list(cfg.generalization.rho_boundary_vals)

    def execute(self, key: PRNGKeyArray):
        """Sweeps ρ and measures stability."""
        k_load, k_sweep = jax.random.split(key)
        
        # 1. Load trained model
        # SG#5 FIX: Use fixed canonical pointer path (config-independent).
        pointer_path = Path("outputs") / "small" / "latest_weights.txt"
        if not pointer_path.exists():
            raise FileNotFoundError(
                f"Model pointer not found at '{pointer_path.resolve()}'. "
                f"Run training first: python -m experiments.n_gibbsq.train"
            )
        model_path = Path(pointer_path.read_text(encoding='utf-8').strip())
        if not model_path.exists():
            raise FileNotFoundError(
                f"Weight file missing at '{model_path}'. Rerun training."
            )
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
        
        num_reps = int(self.cfg.simulation.num_replications)
        _neural_ssa = _NeuralSSAPolicy(model)
        
        for idx, rho in enumerate(self.rho_vals):
            arrival_rate = rho * total_capacity
            
            log.info(f"Evaluating Boundary rho={rho:.3f} (Arrival={arrival_rate:.3f})...")
            
            # SG#1 FIX: Scale sim_time with theoretical CTMC mixing time.
            # Mixing time ~ O(1/(1-rho)^2) (Meyn & Tweedie 1993, §4).
            # Use min(100/(1-rho)^2, 100_000) as a compute-budget cap.
            _mixing_budget = 100.0 / max((1.0 - float(rho)) ** 2, 1e-12)
            _rho_sim_time = min(_mixing_budget, 100_000.0)
            if _rho_sim_time >= 100_000.0:
                log.warning(
                    f"  [!] rho={rho:.4f}: sim_time capped at 100,000s "
                    f"(theoretical mixing time ~ {_mixing_budget:.0f}s). "
                    f"E[Q] near criticality may be underestimated."
                )
            _max_s = int(_rho_sim_time / self.ssa_sample_interval) + 1
            _mu_np = np.array(self.service_rates, dtype=np.float64)
            _rho_seed = self.cfg.simulation.seed + idx * 1000

            # GibbsQ on true SSA
            times_g, states_g, (arrs_g, deps_g) = run_replications_jax(
                num_replications=num_reps, num_servers=self.num_servers,
                arrival_rate=float(arrival_rate), service_rates=jnp.array(_mu_np),
                alpha=float(self.cfg.system.alpha), sim_time=_rho_sim_time,
                sample_interval=self.ssa_sample_interval, base_seed=_rho_seed,
                max_samples=_max_s, policy_type=3,
            )
            g_vals = []
            for _r in range(num_reps):
                _res = SimResult(
                    times=np.array(times_g[_r]), states=np.array(states_g[_r]),
                    arrival_count=int(arrs_g[_r]), departure_count=int(deps_g[_r]),
                    final_time=float(times_g[_r][-1]), num_servers=self.num_servers,
                )
                g_vals.append(float(time_averaged_queue_lengths(
                    _res, self.cfg.simulation.burn_in_fraction).sum()))
            g_loss = float(np.mean(g_vals))

            # Neural on true SSA
            _max_ev = int((float(arrival_rate) + _mu_np.sum()) * _rho_sim_time * 1.5) + 1000
            n_vals = []
            for _rep in range(num_reps):
                _rng = np.random.default_rng(_rho_seed + _rep)
                _res_n = simulate(
                    num_servers=self.num_servers, arrival_rate=float(arrival_rate),
                    service_rates=_mu_np, policy=_neural_ssa,
                    sim_time=_rho_sim_time, sample_interval=self.ssa_sample_interval,
                    rng=_rng, max_events=_max_ev,
                )
                n_vals.append(float(time_averaged_queue_lengths(
                    _res_n, self.cfg.simulation.burn_in_fraction).sum()))
            n_loss = float(np.mean(n_vals))

            neural_results.append(n_loss)
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
        plt.title('N-GibbsQ Stability Boundary Performance ($\\mathbb{E}[Q]$ vs $\\rho$)')
        plt.xlabel('Load Factor $\\rho = \\lambda / \\sum \\mu_i$')
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
