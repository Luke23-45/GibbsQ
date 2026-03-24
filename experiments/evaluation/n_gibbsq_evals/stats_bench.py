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
from gibbsq.utils.model_io import NeuralSSAPolicy, resolve_model_pointer_or_none


# _NeuralSSAPolicy moved to gibbsq.utils.model_io.NeuralSSAPolicy


logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


# _resolve_model_pointer moved to gibbsq.utils.model_io.resolve_model_pointer

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
        _PROJECT_ROOT = Path(__file__).resolve().parents[3]
        output_root = self.run_dir.parent.parent
        
        # PATCH H#3: Graceful fallback for missing pointers
        model_path = resolve_model_pointer_or_none(_PROJECT_ROOT, output_root)
        skeleton = NeuralRouter(num_servers=self.num_servers, config=self.cfg.neural, key=k_load)
        
        if model_path is None:
            log.warning("No trained model found. Creating fresh JSQ-initialized model for evaluation.")
            # Create a fresh model and bootstrap it to JSQ behavior
            from gibbsq.core.pretraining import train_robust_bc_policy
            model = train_robust_bc_policy(
                skeleton, np.array(self.service_rates), k_load, num_steps=500
            )
            log.info("Fresh JSQ-initialized model created successfully.")
        else:
            model = eqx.tree_deserialise_leaves(model_path, skeleton)
        
        # SG#16 Fix: Validate that the loaded model matches the current config
        from gibbsq.utils.model_io import validate_neural_model_shape
        try:
            validate_neural_model_shape(model, self.cfg.neural, self.num_servers)
        except ValueError as e:
            log.error(f"Model shape mismatch! {e}")
            return
        
        # --- 2. Benchmark on True Gillespie SSA ---
        _sc  = self.cfg.simulation
        _ssa = _sc.ssa
        _max_samples = int(_ssa.sim_time / _ssa.sample_interval) + 1
        _mu_np = np.array(self.service_rates, dtype=np.float64)

        log.info(f"Running {self.num_samples} GibbsQ SSA simulations...")
        _pmap = {"uniform": 0, "proportional": 1, "jsq": 2, "softmax": 3, "power_of_d": 4, "sojourn_softmax": 5}
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
            policy_type=_pmap.get(self.cfg.policy.name, 3),
        )
        gibbs_list = []
        for _r in range(self.num_samples):
            _np_times = np.array(times_g[_r])
            _np_states = np.array(states_g[_r])
            _valid_mask = _np_times > 0
            _valid_mask[0] = True
            _vl = int(np.sum(_valid_mask))
            _np_times = _np_times[:_vl]
            _np_states = _np_states[:_vl]

            _res = SimResult(
                times=_np_times, states=_np_states,
                arrival_count=int(arrs_g[_r]), departure_count=int(deps_g[_r]),
                final_time=float(times_g[_r][-1]), num_servers=self.num_servers,
            )
            gibbs_list.append(float(
                time_averaged_queue_lengths(_res, _sc.burn_in_fraction).sum()))
        gibbs_data = np.array(gibbs_list)

        log.info(f"Running {self.num_samples} Neural SSA simulations...")
        _neural_policy = NeuralSSAPolicy(model, mu=_mu_np, rho=self.arrival_rate / float(_mu_np.sum()))
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
        
        # SG#14 FIX: Use independent-samples t-test.
        # GibbsQ (JAX PRNG) and Neural (NumPy PRNG) use different random backends,
        # so samples from matching replication indices are NOT truly paired.
        t_stat, p_val = stats.ttest_ind(neural_data, gibbs_data)
        
        # SG#14 FIX: Cohen's d for independent samples uses pooled SD
        _n1, _n2 = len(neural_data), len(gibbs_data)
        _pooled_var = (
            (_n1 - 1) * np.var(neural_data, ddof=1) + (_n2 - 1) * np.var(gibbs_data, ddof=1)
        ) / (_n1 + _n2 - 2)
        _pooled_std = float(np.sqrt(_pooled_var))
        cohen_d = float((np.mean(neural_data) - np.mean(gibbs_data)) / _pooled_std) if _pooled_std > 0 else 0.0
        
        # 3. Confidence Interval for the difference (independent samples)
        _diff_mean = np.mean(neural_data) - np.mean(gibbs_data)
        _diff_se = _pooled_std * np.sqrt(1.0/_n1 + 1.0/_n2)
        _df = _n1 + _n2 - 2
        ci_low, ci_high = stats.t.interval(self.cfg.verification.confidence_interval, _df, loc=_diff_mean, scale=_diff_se)
        
        improvement = (g_mean - n_mean) / g_mean * 100 if g_mean > 0 else 0
        
        log.info("\n" + "=" * 60)
        log.info("  STATISTICAL SUMMARY")
        log.info("=" * 60)
        log.info(f"GibbsQ E[Q]:   {g_mean:.4f} ± {g_std:.4f}")
        log.info(f"N-GibbsQ E[Q]:   {n_mean:.4f} ± {n_std:.4f}")
        log.info(f"Rel. Improve:  {improvement:.2f}%")
        log.info("-" * 40)
        log.info(f"P-Value:       {p_val:.2e} ({'SIGNIFICANT' if p_val < self.cfg.verification.alpha_significance else 'NOT SIGNIFICANT'})")
        log.info(f"Effect Size:   {cohen_d:.2f} (Cohen's d)")
        log.info(f"{int(self.cfg.verification.confidence_interval*100)}% CI (Diff): [{ci_low:.4f}, {ci_high:.4f}]")
        log.info("=" * 60)
        
        # Assets
        from gibbsq.analysis.plotting import plot_raincloud

        plot_path = self.run_dir / "stats_boxplot"
        fig = plot_raincloud(
            group_a_data=gibbs_data,
            group_b_data=neural_data,
            group_a_label="GibbsQ (Baseline)",
            group_b_label="N-GibbsQ (Proposed)",
            stats={
                "p_value": float(p_val),
                "cohen_d": float(cohen_d),
                "improvement_pct": float(improvement),
            },
            save_path=plot_path,
            theme="publication",
            formats=["png", "pdf"],
        )
        plt.close(fig)

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
                self.run_logger.log({"stats_boxplot": wandb.Image(str(self.run_dir / "stats_boxplot.png"))})
            except Exception:
                pass

@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
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
