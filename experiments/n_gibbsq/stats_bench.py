"""
N-GibbsQ Phase VII: Statistical Benchmark
-----------------------------------------
Statistical comparison of N-GibbsQ vs GibbsQ over 30 seeds.

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
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import functools

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_engine import run_replications_jax
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl


class _NeuralSSAPolicy:
    """Bridges NeuralRouter → NumPy SSA. See eval.py for canonical version."""
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

class StatsBenchmark:
    """Statistical benchmark for N-GibbsQ."""
    def __init__(self, cfg, run_dir: Path, run_logger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.num_servers = cfg.system.num_servers
        self.service_rates = jnp.array(cfg.system.service_rates, dtype=jnp.float32)
        self.arrival_rate = float(cfg.system.arrival_rate)
        self.temperature = float(cfg.simulation.dga.temperature)
        self.sim_steps = cfg.simulation.dga.sim_steps
        
        # Pull replicates from simulation config.
        self.num_samples = int(cfg.simulation.num_replications)

    def execute(self, key: PRNGKeyArray):
        """Runs the 30-seed showdown."""
        _k1, _k2, k_load = jax.random.split(key, 3)
        
        log.info(f"Initiating statistical comparison (n={self.num_samples} seeds).")
        log.info(f"Environment: N={self.num_servers}, rho={self.arrival_rate / jnp.sum(self.service_rates):.2f}")
        
        # --- 1. Load Model ---
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
        
        # --- 2. Benchmark on True Gillespie SSA ---
        _sc  = self.cfg.simulation
        _ssa = _sc.ssa
        _max_samples = int(_ssa.sim_time / _ssa.sample_interval) + 1
        _mu_np = np.array(self.service_rates, dtype=np.float64)

        log.info(f"Running {self.num_samples} GibbsQ SSA simulations...")
        times_g, states_g, (arrs_g, deps_g) = run_replications_jax(
            num_replications=self.num_samples,
            num_servers=self.num_servers,
            arrival_rate=self.arrival_rate,
            service_rates=jnp.array(_mu_np),
            alpha=float(self.cfg.system.alpha),
            sim_time=_ssa.sim_time,
            sample_interval=_ssa.sample_interval,
            base_seed=_sc.seed,
            max_samples=_max_samples,
            policy_type=3,
        )
        gibbs_list = []
        for _r in range(self.num_samples):
            _res = SimResult(
                times=np.array(times_g[_r]), states=np.array(states_g[_r]),
                arrival_count=int(arrs_g[_r]), departure_count=int(deps_g[_r]),
                final_time=float(times_g[_r][-1]), num_servers=self.num_servers,
            )
            gibbs_list.append(float(
                time_averaged_queue_lengths(_res, _sc.burn_in_fraction).sum()))
        gibbs_data = np.array(gibbs_list)

        log.info(f"Running {self.num_samples} Neural SSA simulations...")
        _neural_policy = _NeuralSSAPolicy(model)
        _np_max_ev = int(
            (self.arrival_rate + float(_mu_np.sum())) * _ssa.sim_time * 1.5
        ) + 1000
        neural_list = []
        for _rep in range(self.num_samples):
            _rng = np.random.default_rng(_sc.seed + _rep)
            _res = simulate(
                num_servers=self.num_servers, arrival_rate=self.arrival_rate,
                service_rates=_mu_np, policy=_neural_policy,
                sim_time=_ssa.sim_time, sample_interval=_ssa.sample_interval,
                rng=_rng, max_events=_np_max_ev,
            )
            neural_list.append(float(
                time_averaged_queue_lengths(_res, _sc.burn_in_fraction).sum()))
        neural_data = np.array(neural_list)
        
        self._analyze(neural_data, gibbs_data)

    def _analyze(self, neural_data, gibbs_data):
        """Perform rigorous scientific analysis."""
        # Descriptive Statistics
        n_mean, n_std = np.mean(neural_data), np.std(neural_data, ddof=1)
        g_mean, g_std = np.mean(gibbs_data), np.std(gibbs_data, ddof=1)
        
        # 1. Paired T-Test
        t_stat, p_val = stats.ttest_rel(neural_data, gibbs_data)
        
        # 2. Cohen's d — paired design: d = mean(diff) / std(diff, ddof=1)
        #    (Cohen 1988, Ch.2; pooled-SD formula is only valid for independent samples)
        diff = neural_data - gibbs_data
        _diff_std = float(np.std(diff, ddof=1))
        cohen_d = float(np.mean(diff) / _diff_std) if _diff_std > 0.0 else 0.0
        
        # 3. 95% Confidence Interval for the difference (reuses diff from above)
        ci_low, ci_high = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
        
        improvement = (g_mean - n_mean) / g_mean * 100 if g_mean > 0 else 0
        
        log.info("\n" + "=" * 60)
        log.info("  STATISTICAL SUMMARY")
        log.info("=" * 60)
        log.info(f"GibbsQ E[Q]:   {g_mean:.4f} ± {g_std:.4f}")
        log.info(f"N-GibbsQ E[Q]:   {n_mean:.4f} ± {n_std:.4f}")
        log.info(f"Rel. Improve:  {improvement:.2f}%")
        log.info("-" * 40)
        log.info(f"P-Value:       {p_val:.2e} ({'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'})")
        log.info(f"Effect Size:   {cohen_d:.2f} (Cohen's d)")
        log.info(f"95% CI (Diff): [{ci_low:.4f}, {ci_high:.4f}]")
        log.info("=" * 60)
        
        # Assets
        plt.figure(figsize=(10, 6))
        plt.boxplot([gibbs_data, neural_data], tick_labels=['GibbsQ (Baseline)', 'N-GibbsQ (Proposed)'])
        plt.title(f'Performance Distribution Comparison (n={self.num_samples} seeds)')
        plt.ylabel('Expected Queue Length $\\mathbb{E}[Q]$')
        plt.grid(True, alpha=0.3)
        plot_path = self.run_dir / "stats_boxplot.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        append_metrics_jsonl({
            "gibbs_mean": float(g_mean), "gibbs_std": float(g_std),
            "neural_mean": float(n_mean), "neural_std": float(n_std),
            "p_value": float(p_val), "cohen_d": float(cohen_d),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "improvement_pct": float(improvement)
        }, self.run_dir / "metrics.jsonl")

        if self.run_logger:
            self.run_logger.log({
                "stats/p_value": p_val,
                "stats/cohen_d": cohen_d,
                "stats/ci_low": ci_low,
                "stats/ci_high": ci_high,
                "stats/improvement_pct": improvement
            })
            try:
                import wandb
                self.run_logger.log({"stats_boxplot": wandb.Image(str(plot_path))})
            except Exception:
                pass

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "stats_benchmark", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="n_gibbsq_verification", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase VII: Statistical Summary")
    log.info("=" * 60)
    
    bench = StatsBenchmark(cfg, run_dir, run_logger)
    bench.execute(jax.random.PRNGKey(cfg.simulation.seed))

if __name__ == "__main__":
    main()
