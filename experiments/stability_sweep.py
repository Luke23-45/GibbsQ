import logging
import hydra
import numpy as np
import jax.numpy as jnp
from omegaconf import DictConfig

from moeq.core.config import hydra_to_config, validate
from moeq.core.policies import make_policy
from moeq.engines.numpy_engine import simulate, SimResult
from moeq.engines.jax_engine import run_replications_jax
from moeq.analysis.metrics import time_averaged_queue_lengths, stationarity_diagnostic
from moeq.analysis.plotting import plot_alpha_sweep
from moeq.utils.exporter import save_trajectory_parquet, append_metrics_jsonl
from moeq.utils.logging import setup_wandb, get_run_config

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)

ALPHA_VALUES = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
RHO_VALUES = np.array([0.5, 0.7, 0.9, 0.95, 0.99])


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    # Initialize Run Capsule (Dynamic Directory + Config Persistence)
    run_dir, run_id = get_run_config(cfg, "stability_sweep", raw_cfg)

    # Initialize WandB via centralized utility
    run = setup_wandb(cfg, raw_cfg, default_group="stability_sweep", run_id=run_id, run_dir=run_dir)

    sc = cfg.system
    N = sc.num_servers
    mu = np.asarray(sc.service_rates, dtype=np.float64)
    mu_jax = jnp.asarray(mu)
    cap = float(mu.sum())

    # Use the isolated Run Directory for all outputs
    out_dir = run_dir
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    log.info(f"System: N={N}, cap={cap:.4f} | Backend: {'JAX' if cfg.jax.enabled else 'NumPy'}")
    log.info(f"Grid: {len(ALPHA_VALUES)} alpha x {len(RHO_VALUES)} rho "
             f"x {cfg.simulation.num_replications} reps")

    n_rho, n_alpha = len(RHO_VALUES), len(ALPHA_VALUES)
    mean_Q = np.zeros((n_rho, n_alpha))
    stationary = np.zeros((n_rho, n_alpha), dtype=bool)

    total_runs = n_rho * n_alpha
    completed = 0

    for i, rho in enumerate(RHO_VALUES):
        lam = rho * cap
        log.info(f"\n{'-'*60}\n  rho = {rho:.2f}  (lam = {lam:.4f})\n{'-'*60}")

        for j, alpha in enumerate(ALPHA_VALUES):
            completed += 1
            rep_means = []
            rep_stationary_flags: list[bool] = []
            last_res = None

            if cfg.jax.enabled:
                # --- SOTA JAX EXECUTION (Vectorized Replications) ---
                max_samples = int(cfg.simulation.sim_time / cfg.simulation.sample_interval) + 1
                times, states, (arrs, deps) = run_replications_jax(
                    num_replications=cfg.simulation.num_replications,
                    num_servers=N,
                    arrival_rate=lam,
                    service_rates=mu_jax,
                    alpha=float(alpha),
                    sim_time=cfg.simulation.sim_time,
                    sample_interval=cfg.simulation.sample_interval,
                    base_seed=cfg.simulation.seed,
                    max_samples=max_samples
                )
                
                # Convert JAX results to NumPy-friendly lists for metric computation
                for r in range(cfg.simulation.num_replications):
                    res = SimResult(
                        times=np.array(times[r]),
                        states=np.array(states[r]),
                        arrival_count=int(arrs[r]),
                        departure_count=int(deps[r]),
                        final_time=float(times[r][-1]),
                        num_servers=N
                    )
                    avg_q = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
                    rep_means.append(avg_q.sum())
                    rep_diag = stationarity_diagnostic(res, burn_in_fraction=cfg.simulation.burn_in_fraction)
                    rep_stationary_flags.append(bool(rep_diag["is_stationary"]))
                    last_res = res
            else:
                # --- STANDARD NUMPY EXECUTION (Sequential) ---
                policy = make_policy("softmax", alpha=float(alpha))
                for rep in range(cfg.simulation.num_replications):
                    rng = np.random.default_rng(cfg.simulation.seed + rep)
                    res = simulate(
                        num_servers=N,
                        arrival_rate=lam,
                        service_rates=mu,
                        policy=policy,
                        sim_time=cfg.simulation.sim_time,
                        sample_interval=cfg.simulation.sample_interval,
                        rng=rng,
                    )
                    avg_q = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
                    rep_means.append(avg_q.sum())
                    rep_diag = stationarity_diagnostic(res, burn_in_fraction=cfg.simulation.burn_in_fraction)
                    rep_stationary_flags.append(bool(rep_diag["is_stationary"]))
                    last_res = res

            mean_Q[i, j] = np.mean(rep_means)
            stationarity_rate = float(np.mean(rep_stationary_flags)) if rep_stationary_flags else 0.0
            stationary[i, j] = stationarity_rate >= 0.5
            
            # Export scalar metrics
            metrics = {
                "rho": float(rho),
                "lam": float(lam),
                "alpha": float(alpha),
                "mean_q_total": float(mean_Q[i, j]),
                "is_stationary": bool(stationary[i, j]),
                "stationarity_rate": stationarity_rate,
            }
            append_metrics_jsonl(metrics, out_dir / "metrics.jsonl")
            if run:
                run.log(metrics)
            
            # Conditionally export the last trajectory
            if cfg.simulation.export_trajectories and last_res is not None:
                fname = out_dir / f"trajectories/rho{rho:.2f}_alpha{alpha:.2f}.parquet"
                save_trajectory_parquet(last_res, fname)

            stat_str = "OK" if stationary[i, j] else "NONSTATIONARY"
            log.info(f"  alpha={alpha:6.2f} | E[Q_total]={mean_Q[i,j]:8.2f} | {stat_str}  ({completed}/{total_runs})")

    # Outputs
    rho_labels = [f"ρ={r:.2f}" for r in RHO_VALUES]
    f_plot = out_dir / "alpha_sweep.png"
    plot_alpha_sweep(ALPHA_VALUES, mean_Q, rho_labels, save_path=f_plot)
    log.info(f"\nSaved plot: {f_plot}")
    if run:
        run.log({"alpha_sweep_plot": wandb.Image(str(f_plot))})

    # Summary Log
    n_fail = np.sum(~stationary)
    log.info(f"\nSummary: {n_fail}/{stationary.size} configurations non-stationary.")
    
    if run:
        run.finish()


if __name__ == "__main__":
    main()
