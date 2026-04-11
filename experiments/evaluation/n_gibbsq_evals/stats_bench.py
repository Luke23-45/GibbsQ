"""
N-GibbsQ Phase VII: Statistical Benchmark

Statistical comparison of N-GibbsQ vs Calibrated UAS over multiple seeds.
Both sides measured on the true Gillespie SSA (not DGA surrogate).
"""

import logging
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import DictConfig
from scipy import stats

from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.analysis.plot_profiles import ExperimentPlotContext
from gibbsq.core.config import load_experiment_config
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_engine import policy_name_to_type, run_replications_jax
from gibbsq.engines.numpy_engine import SimResult, simulate
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.model_io import build_neural_eval_policy, resolve_model_pointer
from gibbsq.utils.progress import create_progress, iter_progress
from gibbsq.utils.run_artifacts import figure_path, metrics_path


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
NEURAL_EVAL_MODE = "deterministic"
PUBLICATION_BASELINE_POLICY_NAME = "calibrated_uas"
PUBLICATION_BASELINE_LABEL = "Calibrated UAS"


def evaluate_model(model: NeuralRouter, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
    """Pure functional bridge."""
    return model(Q)


def _publication_baseline_spec() -> tuple[str, int]:
    """Return the canonical publication baseline for this runner."""
    return PUBLICATION_BASELINE_POLICY_NAME, policy_name_to_type(PUBLICATION_BASELINE_POLICY_NAME)


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
        """Run the multi-seed statistical comparison."""
        _k1, _k2, k_load = jax.random.split(key, 3)

        log.info(f"Initiating statistical comparison (n={self.num_samples} seeds).")
        log.info(f"Environment: N={self.num_servers}, rho={self.arrival_rate / jnp.sum(self.service_rates):.2f}")

        project_root = Path(__file__).resolve().parents[3]
        output_root = self.run_dir.parent.parent

        model, model_path = _load_trained_model_or_fail(
            self.cfg,
            self.num_servers,
            self.service_rates,
            k_load,
            project_root,
            output_root,
        )
        log.info(f"Loaded trained model from {model_path}")

        from gibbsq.utils.model_io import validate_neural_model_shape

        try:
            validate_neural_model_shape(model, self.cfg.neural, self.num_servers)
        except ValueError as e:
            raise RuntimeError(f"Model shape mismatch: {e}") from e

        sim_cfg = self.cfg.simulation
        ssa_cfg = sim_cfg.ssa
        max_samples = int(ssa_cfg.sim_time / ssa_cfg.sample_interval) + 1
        mu_np = np.array(self.service_rates, dtype=np.float64)
        baseline_policy_name, baseline_policy_type = _publication_baseline_spec()

        with create_progress(total=2, desc="stats", unit="stage") as stage_bar:
            log.info(
                f"Running {self.num_samples} {PUBLICATION_BASELINE_LABEL} SSA simulations "
                f"with policy='{baseline_policy_name}'..."
            )
            times_g, states_g, (arrs_g, deps_g) = run_replications_jax(
                num_replications=self.num_samples,
                num_servers=self.num_servers,
                arrival_rate=self.arrival_rate,
                service_rates=jnp.array(mu_np),
                alpha=float(self.cfg.system.alpha),
                sim_time=ssa_cfg.sim_time,
                sample_interval=ssa_cfg.sample_interval,
                base_seed=sim_cfg.seed,
                max_samples=max_samples,
                policy_type=baseline_policy_type,
                max_events_multiplier=self.cfg.jax_engine.max_events_safety_multiplier,
                max_events_buffer=self.cfg.jax_engine.max_events_additive_buffer,
                scan_sampling_chunk=self.cfg.jax_engine.scan_sampling_chunk,
            )

            baseline_list = []
            for rep_idx in iter_progress(
                range(self.num_samples),
                total=self.num_samples,
                desc=f"stats: {PUBLICATION_BASELINE_LABEL} reps",
                unit="rep",
                leave=False,
            ):
                np_times = np.array(times_g[rep_idx])
                np_states = np.array(states_g[rep_idx])
                valid_mask = np_times > 0
                valid_mask[0] = True
                valid_len = int(np.sum(valid_mask))
                np_times = np_times[:valid_len]
                np_states = np_states[:valid_len]

                res = SimResult(
                    times=np_times,
                    states=np_states,
                    arrival_count=int(arrs_g[rep_idx]),
                    departure_count=int(deps_g[rep_idx]),
                    final_time=float(np_times[-1]),
                    num_servers=self.num_servers,
                )
                baseline_list.append(float(time_averaged_queue_lengths(res, sim_cfg.burn_in_fraction).sum()))
            baseline_data = np.array(baseline_list)
            stage_bar.update(1)

            log.info(f"Running {self.num_samples} Neural SSA simulations...")
            log.info(f"Neural evaluation mode: {NEURAL_EVAL_MODE}")
            neural_policy = build_neural_eval_policy(
                model,
                mu=mu_np,
                rho=self.arrival_rate / float(mu_np.sum()),
                mode=NEURAL_EVAL_MODE,
            )
            max_events = int((self.arrival_rate + float(mu_np.sum())) * ssa_cfg.sim_time * 1.5) + 1000
            neural_list = []
            for rep_idx in iter_progress(
                range(self.num_samples),
                total=self.num_samples,
                desc="stats: neural reps",
                unit="rep",
                leave=False,
            ):
                rng = np.random.default_rng(sim_cfg.seed + rep_idx)
                res = simulate(
                    num_servers=self.num_servers,
                    arrival_rate=self.arrival_rate,
                    service_rates=mu_np,
                    policy=neural_policy,
                    sim_time=ssa_cfg.sim_time,
                    sample_interval=ssa_cfg.sample_interval,
                    rng=rng,
                    max_events=max_events,
                )
                neural_list.append(float(time_averaged_queue_lengths(res, sim_cfg.burn_in_fraction).sum()))
            neural_data = np.array(neural_list)
            stage_bar.update(1)

        self._analyze(neural_data, baseline_data)

    def _analyze(self, neural_data, baseline_data):
        """Compute t-test and effect size."""
        n_mean, n_std = np.mean(neural_data), np.std(neural_data, ddof=1)
        b_mean, b_std = np.mean(baseline_data), np.std(baseline_data, ddof=1)

        # The baseline and neural paths use different random backends, so the
        # samples are treated as independent rather than paired.
        _t_stat, p_val = stats.ttest_ind(neural_data, baseline_data)

        n1, n2 = len(neural_data), len(baseline_data)
        pooled_var = (
            (n1 - 1) * np.var(neural_data, ddof=1) + (n2 - 1) * np.var(baseline_data, ddof=1)
        ) / (n1 + n2 - 2)
        pooled_std = float(np.sqrt(pooled_var))
        cohen_d = float((np.mean(neural_data) - np.mean(baseline_data)) / pooled_std) if pooled_std > 0 else 0.0

        diff_mean = np.mean(neural_data) - np.mean(baseline_data)
        diff_se = pooled_std * np.sqrt(1.0 / n1 + 1.0 / n2)
        df = n1 + n2 - 2
        ci_low, ci_high = stats.t.interval(
            self.cfg.verification.confidence_interval,
            df,
            loc=diff_mean,
            scale=diff_se,
        )

        improvement = (b_mean - n_mean) / b_mean * 100 if b_mean > 0 else 0.0

        log.info("\n" + "=" * 60)
        log.info("  STATISTICAL SUMMARY")
        log.info("=" * 60)
        log.info(f"{PUBLICATION_BASELINE_LABEL} E[Q]:   {b_mean:.4f} +/- {b_std:.4f}")
        log.info(f"N-GibbsQ E[Q]:   {n_mean:.4f} +/- {n_std:.4f}")
        log.info(f"Rel. Improve:  {improvement:.2f}%")
        log.info("-" * 40)
        log.info(
            f"P-Value:       {p_val:.2e} "
            f"({'SIGNIFICANT' if p_val < self.cfg.verification.alpha_significance else 'NOT SIGNIFICANT'})"
        )
        log.info(f"Effect Size:   {cohen_d:.2f} (Cohen's d)")
        log.info(f"{int(self.cfg.verification.confidence_interval * 100)}% CI (Diff): [{ci_low:.4f}, {ci_high:.4f}]")
        log.info("=" * 60)

        from gibbsq.analysis.plotting import plot_raincloud

        plot_path = figure_path(self.run_dir, "stats_boxplot")
        fig = plot_raincloud(
            group_a_data=baseline_data,
            group_b_data=neural_data,
            group_a_label=f"{PUBLICATION_BASELINE_LABEL} (Baseline)",
            group_b_label="N-GibbsQ (Proposed)",
            stats={
                "p_value": float(p_val),
                "cohen_d": float(cohen_d),
                "improvement_pct": float(improvement),
            },
            save_path=plot_path,
            theme="publication",
            formats=["png", "pdf"],
            context=ExperimentPlotContext(
                experiment_id="stats",
                chart_name="plot_raincloud",
                semantic_overrides={
                    "figure_title": f"{PUBLICATION_BASELINE_LABEL} vs N-GibbsQ: Distribution Comparison",
                },
            ),
        )
        plt.close(fig)

        append_metrics_jsonl(
            {
                "baseline_policy": PUBLICATION_BASELINE_POLICY_NAME,
                "baseline_label": PUBLICATION_BASELINE_LABEL,
                "baseline_mean": float(b_mean),
                "baseline_std": float(b_std),
                "calibrated_uas_mean": float(b_mean),
                "calibrated_uas_std": float(b_std),
                "gibbs_mean": float(b_mean),
                "gibbs_std": float(b_std),
                "neural_mean": float(n_mean),
                "neural_std": float(n_std),
                "p_value": float(p_val),
                "cohen_d": float(cohen_d),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "improvement_pct": float(improvement),
            },
            metrics_path(self.run_dir),
        )

        if self.run_logger:
            self.run_logger.log(
                {
                    "stats/p_value": p_val,
                    "stats/cohen_d": cohen_d,
                    "stats/ci_low": ci_low,
                    "stats/ci_high": ci_high,
                    "stats/improvement_pct": improvement,
                }
            )
            try:
                import wandb

                self.run_logger.log(
                    {"stats_boxplot": wandb.Image(str(figure_path(self.run_dir, "stats_boxplot").with_suffix(".png")))}
                )
            except Exception:
                pass


@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "stats")

    run_dir, run_id = get_run_config(cfg, "stats", resolved_raw_cfg)
    run_logger = setup_wandb(
        cfg,
        resolved_raw_cfg,
        default_group="n_gibbsq_verification",
        run_id=run_id,
        run_dir=run_dir,
    )

    log.info("=" * 60)
    log.info("  Phase VII: Statistical Summary")
    log.info("=" * 60)

    bench = StatsBenchmark(cfg, run_dir, run_logger)
    bench.execute(jax.random.PRNGKey(cfg.simulation.seed))


if __name__ == "__main__":
    main()
