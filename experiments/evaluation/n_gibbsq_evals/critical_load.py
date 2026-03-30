"""
N-GibbsQ critical load test.

Tests N-GibbsQ as rho -> 1, where queueing systems become unstable.
"""

from pathlib import Path
import logging

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import DictConfig

from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.core.config import critical_load_sim_time, load_experiment_config
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.engines.jax_engine import policy_name_to_type, run_replications_jax
from gibbsq.engines.jax_ssa import compute_poisson_max_steps
from gibbsq.engines.numpy_engine import SimResult, simulate
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.model_io import build_neural_eval_policy, resolve_model_pointer
from gibbsq.utils.progress import create_progress, iter_progress

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
NEURAL_EVAL_MODE = "deterministic"


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
        self.ssa_sim_time = cfg.simulation.ssa.sim_time
        self.ssa_sample_interval = cfg.simulation.ssa.sample_interval
        self.rho_vals = list(cfg.generalization.rho_boundary_vals)

    def execute(self, key: PRNGKeyArray):
        """Sweeps rho and measures stability."""
        k_load, _ = jax.random.split(key)

        project_root = Path(__file__).resolve().parents[3]
        output_root = self.run_dir.parent.parent
        model_path = resolve_model_pointer(project_root, output_root, allow_bc=False, allow_legacy=False)
        skeleton = NeuralRouter(
            num_servers=self.num_servers,
            config=self.cfg.neural,
            service_rates=self.service_rates,
            key=k_load,
        )
        model = eqx.tree_deserialise_leaves(model_path, skeleton)

        from gibbsq.utils.model_io import validate_neural_model_shape

        try:
            validate_neural_model_shape(model, self.cfg.neural, self.num_servers)
        except ValueError as e:
            raise RuntimeError(f"Model shape mismatch: {e}") from e

        total_capacity = jnp.sum(self.service_rates)
        neural_results = []
        gibbs_results = []
        num_reps = int(self.cfg.simulation.num_replications)
        baseline_policy_name = self.cfg.policy.name
        baseline_policy_type = policy_name_to_type(baseline_policy_name)

        log.info(f"System Capacity: {total_capacity:.2f}")
        log.info(f"Targeting Load Boundary: {self.rho_vals}")

        with create_progress(total=len(self.rho_vals), desc="critical", unit="rho") as rho_bar:
            for idx, rho in enumerate(self.rho_vals):
                rho_bar.set_postfix({"rho": f"{rho:.3f}"}, refresh=False)
                arrival_rate = rho * total_capacity
                log.info(f"Evaluating Boundary rho={rho:.3f} (Arrival={arrival_rate:.3f})...")

                rho_sim_time = critical_load_sim_time(self.cfg, float(rho))

                max_samples = int(rho_sim_time / self.ssa_sample_interval) + 1
                mu_np = np.array(self.service_rates, dtype=np.float64)
                rho_seed = self.cfg.simulation.seed + idx * 1000

                times_g, states_g, (arrs_g, deps_g) = run_replications_jax(
                    num_replications=num_reps,
                    num_servers=self.num_servers,
                    arrival_rate=float(arrival_rate),
                    service_rates=jnp.array(mu_np),
                    alpha=float(self.cfg.system.alpha),
                    sim_time=rho_sim_time,
                    sample_interval=self.ssa_sample_interval,
                    base_seed=rho_seed,
                    max_samples=max_samples,
                    policy_type=baseline_policy_type,
                    max_events_multiplier=self.cfg.jax_engine.max_events_safety_multiplier,
                    max_events_buffer=self.cfg.jax_engine.max_events_additive_buffer,
                    scan_sampling_chunk=self.cfg.jax_engine.scan_sampling_chunk,
                )
                g_vals = []
                for rep_idx in iter_progress(
                    range(num_reps),
                    total=num_reps,
                    desc=f"critical GibbsQ rho={rho:.3f}",
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
                    g_vals.append(float(time_averaged_queue_lengths(
                        res, self.cfg.simulation.burn_in_fraction).sum()))
                g_loss = float(np.mean(g_vals))

                neural_policy = build_neural_eval_policy(
                    model,
                    mu=mu_np,
                    rho=rho,
                    mode=NEURAL_EVAL_MODE,
                )
                max_events = compute_poisson_max_steps(float(arrival_rate), mu_np, rho_sim_time)
                n_vals = []
                for rep_idx in iter_progress(
                    range(num_reps),
                    total=num_reps,
                    desc=f"critical neural rho={rho:.3f}",
                    unit="rep",
                    leave=False,
                ):
                    rng = np.random.default_rng(rho_seed + rep_idx)
                    res_n = simulate(
                        num_servers=self.num_servers,
                        arrival_rate=float(arrival_rate),
                        service_rates=mu_np,
                        policy=neural_policy,
                        sim_time=rho_sim_time,
                        sample_interval=self.ssa_sample_interval,
                        rng=rng,
                        max_events=max_events,
                    )
                    n_vals.append(float(time_averaged_queue_lengths(
                        res_n, self.cfg.simulation.burn_in_fraction).sum()))
                n_loss = float(np.mean(n_vals))

                neural_results.append(n_loss)
                gibbs_results.append(g_loss)
                log.info(f"   => N-GibbsQ E[Q]: {n_loss:.2f} | GibbsQ E[Q]: {g_loss:.2f}")

                append_metrics_jsonl(
                    {"rho": float(rho), "neural_eq": n_loss, "gibbs_eq": g_loss},
                    self.run_dir / "metrics.jsonl",
                )
                rho_bar.update(1)

        self._plot(self.rho_vals, neural_results, gibbs_results)

    def _plot(self, rho_vals, neural_r, gibbs_r):
        """Generate the critical-load curve."""
        from gibbsq.analysis.plotting import plot_critical_load

        plot_path = self.run_dir / "critical_load_curve"
        fig = plot_critical_load(
            rho_values=np.array(rho_vals),
            neural_eq=np.array(neural_r),
            gibbs_eq=np.array(gibbs_r),
            save_path=plot_path,
            theme="publication",
            formats=["png", "pdf"],
        )
        plt.close(fig)

        log.info(f"Critical load test complete. Curve saved to {plot_path}.png, {plot_path}.pdf")

        if self.run_logger:
            self.run_logger.log({
                "critical_load/rho": rho_vals,
                "critical_load/neural_eq": neural_r,
                "critical_load/gibbs_eq": gibbs_r,
            })
            try:
                import wandb
                self.run_logger.log({"critical_load_curve": wandb.Image(str(self.run_dir / "critical_load_curve.png"))})
            except Exception:
                pass


@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "critical")

    run_dir, run_id = get_run_config(cfg, "critical", resolved_raw_cfg)
    run_logger = setup_wandb(cfg, resolved_raw_cfg, default_group="n_gibbsq_verification", run_id=run_id, run_dir=run_dir)

    log.info("=" * 60)
    log.info("  Phase VIII: The Critical Stability Boundary")
    log.info("=" * 60)

    test = CriticalLoadTest(cfg, run_dir, run_logger)
    test.execute(jax.random.PRNGKey(cfg.simulation.seed))


if __name__ == "__main__":
    main()
