"""
N-GibbsQ Phase VII: Statistical Benchmark

Statistical comparison of N-GibbsQ vs GibbsQ over 30 seeds.
Both sides measured on the true Gillespie SSA (not DGA surrogate).
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

from gibbsq.core.config import load_experiment_config
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_engine import policy_name_to_type, run_replications_jax
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.model_io import build_neural_eval_policy, resolve_model_pointer
from gibbsq.utils.progress import create_progress, iter_progress


# _NeuralSSAPolicy moved to gibbsq.utils.model_io.NeuralSSAPolicy


logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)
NEURAL_EVAL_MODE = "deterministic"


# _resolve_model_pointer moved to gibbsq.utils.model_io.resolve_model_pointer

def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Pure functional bridge."""
    return model(Q)


def _load_trained_model_or_fail(cfg, num_servers: int, service_rates, load_key, project_root: Path, output_root: Path):
    model_path = resolve_model_pointer(project_root, output_root, allow_bc=False, allow_legacy=False)
    skeleton = NeuralRouter(num_servers=num_servers, config=cfg.neural, service_rates=service_rates, key=load_key)
    model = eqx.tree_deserialise_leaves(model_path, skeleton)
    return model, model_path

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
        
        self.num_samples = int(cfg.simulation.num_replications)

    def execute(self, key: PRNGKeyArray):
        """Runs the 30-seed showdown."""
        _k1, _k2, k_load = jax.random.split(key, 3)
        
        log.info(f"Initiating statistical comparison (n={self.num_samples} seeds).")
        log.info(f"Environment: N={self.num_servers}, rho={self.arrival_rate / jnp.sum(self.service_rates):.2f}")
        
        _PROJECT_ROOT = Path(__file__).resolve().parents[3]
        output_root = self.run_dir.parent.parent
        
        model, model_path = _load_trained_model_or_fail(
            self.cfg,
            self.num_servers,
            self.service_rates,
            k_load,
            _PROJECT_ROOT,
            output_root,
        )
        log.info(f"Loaded trained model from {model_path}")
        
        from gibbsq.utils.model_io import validate_neural_model_shape
        try:
            validate_neural_model_shape(model, self.cfg.neural, self.num_servers)
        except ValueError as e:
            raise RuntimeError(f"Model shape mismatch: {e}") from e
        
        _sc  = self.cfg.simulation
        _ssa = _sc.ssa
        _max_samples = int(_ssa.sim_time / _ssa.sample_interval) + 1
        _mu_np = np.array(self.service_rates, dtype=np.float64)

        baseline_policy_name = self.cfg.policy.name
        baseline_policy_type = policy_name_to_type(baseline_policy_name)

        with create_progress(total=2, desc="stats", unit="stage") as stage_bar:
            log.info(
                f"Running {self.num_samples} GibbsQ SSA simulations "
                f"with policy='{baseline_policy_name}'..."
            )
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
                policy_type=baseline_policy_type,
                max_events_multiplier=self.cfg.jax_engine.max_events_safety_multiplier,
                max_events_buffer=self.cfg.jax_engine.max_events_additive_buffer,
                scan_sampling_chunk=self.cfg.jax_engine.scan_sampling_chunk,
            )
            gibbs_list = []
            for _r in iter_progress(
                range(self.num_samples),
                total=self.num_samples,
                desc="stats: GibbsQ reps",
                unit="rep",
                leave=False,
            ):
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
                    final_time=float(_np_times[-1]), num_servers=self.num_servers,
                )
                gibbs_list.append(float(
                    time_averaged_queue_lengths(_res, _sc.burn_in_fraction).sum()))
            gibbs_data = np.array(gibbs_list)
            stage_bar.update(1)

            log.info(f"Running {self.num_samples} Neural SSA simulations...")
            log.info(f"Neural evaluation mode: {NEURAL_EVAL_MODE}")
            _neural_policy = build_neural_eval_policy(
                model,
                mu=_mu_np,
                rho=self.arrival_rate / float(_mu_np.sum()),
                mode=NEURAL_EVAL_MODE,
            )
            _np_max_ev = int(
                (self.arrival_rate + float(_mu_np.sum())) * _ssa.sim_time * 1.5
            ) + 1000
            neural_list = []
            for _rep in iter_progress(
                range(self.num_samples),
                total=self.num_samples,
                desc="stats: neural reps",
                unit="rep",
                leave=False,
            ):
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
            stage_bar.update(1)
        
        self._analyze(neural_data, gibbs_data)

    def _analyze(self, neural_data, gibbs_data):
        """Compute t-test and effect size."""
        n_mean, n_std = np.mean(neural_data), np.std(neural_data, ddof=1)
        g_mean, g_std = np.mean(gibbs_data), np.std(gibbs_data, ddof=1)
        
        # GibbsQ (JAX PRNG) and Neural (NumPy PRNG) use different random backends,
        # so samples from matching replication indices are NOT truly paired.
        t_stat, p_val = stats.ttest_ind(neural_data, gibbs_data)
        
        # Cohen's d for independent samples uses pooled SD
        _n1, _n2 = len(neural_data), len(gibbs_data)
        _pooled_var = (
            (_n1 - 1) * np.var(neural_data, ddof=1) + (_n2 - 1) * np.var(gibbs_data, ddof=1)
        ) / (_n1 + _n2 - 2)
        _pooled_std = float(np.sqrt(_pooled_var))
        cohen_d = float((np.mean(neural_data) - np.mean(gibbs_data)) / _pooled_std) if _pooled_std > 0 else 0.0
        
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
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "stats")

    run_dir, run_id = get_run_config(cfg, "stats", resolved_raw_cfg)
    run_logger = setup_wandb(cfg, resolved_raw_cfg, default_group="n_gibbsq_verification", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase VII: Statistical Summary")
    log.info("=" * 60)
    
    bench = StatsBenchmark(cfg, run_dir, run_logger)
    bench.execute(jax.random.PRNGKey(cfg.simulation.seed))

if __name__ == "__main__":
    main()
