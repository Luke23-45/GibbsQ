import logging
import hydra
import numpy as np
import jax.numpy as jnp
from omegaconf import DictConfig

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.core.builders import build_policy_by_name
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.engines.jax_engine import run_replications_jax
from gibbsq.analysis.metrics import time_averaged_queue_lengths, stationarity_diagnostic
from gibbsq.analysis.plotting import plot_alpha_sweep
from gibbsq.utils.exporter import save_trajectory_parquet, append_metrics_jsonl
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.analysis.theme import apply_theme

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)




@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    # Initialize Run Capsule (Dynamic Directory + Config Persistence)
    run_dir, run_id = get_run_config(cfg, "stability_sweep", raw_cfg)

    # Initialize WandB via centralized utility
    run = setup_wandb(cfg, raw_cfg, default_group="stability_sweep", run_id=run_id, run_dir=run_dir)

    # Apply publication theme for paper-ready charts
    apply_theme('publication')

    sc = cfg.system
    N = sc.num_servers
    mu = np.asarray(sc.service_rates, dtype=np.float64)
    mu_jax = jnp.asarray(mu)
    cap = float(mu.sum())

    # Use the isolated Run Directory for all outputs
    out_dir = run_dir
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    # SG#16 FIX: Replace bare assert with if/raise to survive python -O.
    if cfg.policy.name not in ["softmax", "sojourn_softmax"]:
        raise ValueError(
            f"Stability sweep requires policy.name == 'sojourn_softmax', "
            f"but config has '{cfg.policy.name}'."
        )

    alpha_values = np.array(cfg.stability_sweep.alpha_vals)
    rho_values = np.array(cfg.stability_sweep.rho_vals)

    log.info(f"System: N={N}, cap={cap:.4f} | Backend: {'JAX' if cfg.jax.enabled else 'NumPy'}")
    log.info(f"Grid: {len(alpha_values)} alpha x {len(rho_values)} rho "
             f"x {cfg.simulation.num_replications} reps")

    n_rho, n_alpha = len(rho_values), len(alpha_values)
    mean_Q = np.zeros((n_rho, n_alpha))
    stationary = np.zeros((n_rho, n_alpha), dtype=bool)

    total_runs = n_rho * n_alpha
    completed = 0

    for i, rho in enumerate(rho_values):
        lam = rho * cap
        log.info(f"\n{'-'*60}\n  rho = {rho:.2f}  (lam = {lam:.4f})\n{'-'*60}")

        for j, alpha in enumerate(alpha_values):
            completed += 1
            # SG-7: PRNG Safety Warning.
            # Using a linear seed increment (seed + index * replications) is safe only if
            # num_replications is static across all cells. If replications vary, 
            # seeds may overlap, violating independence across grid cells.
            _cell_seed = cfg.simulation.seed + (i * n_alpha + j) * cfg.simulation.num_replications
            
            rep_means = []
            rep_stationary_flags: list[bool] = []
            last_res = None

            if cfg.jax.enabled:
                # --- JAX backend (vectorized replications) ---
                # ─────────────────────────────────────────────────────────────────────────
                # JAX Simulation
                # ─────────────────────────────────────────────────────────────────────────
                
                _sim_time = cfg.simulation.ssa.sim_time
                _sample_interval = cfg.simulation.ssa.sample_interval
                max_samples = int(_sim_time / _sample_interval) + 1
                _pmap = {"uniform": 0, "proportional": 1, "jsq": 2, "softmax": 3, "power_of_d": 4, "sojourn_softmax": 5}
                times, states, (arrs, deps) = run_replications_jax(
                    num_replications=cfg.simulation.num_replications,
                    num_servers=N,
                    arrival_rate=lam,
                    service_rates=mu_jax,
                    alpha=float(alpha),
                    sim_time=_sim_time,
                    sample_interval=_sample_interval,
                    base_seed=_cell_seed,
                    max_samples=max_samples,
                    policy_type=_pmap.get(cfg.policy.name, 3),
                )
                
                # Convert JAX results to NumPy-friendly lists for metric computation
                for r in range(cfg.simulation.num_replications):
                    _np_times = np.array(times[r])
                    _np_states = np.array(states[r])
                    _valid_mask = _np_times > 0
                    _valid_mask[0] = True
                    _vl = int(np.sum(_valid_mask))
                    if _vl < _np_states.shape[0]:
                        _np_times = _np_times[:_vl]
                        _np_states = _np_states[:_vl]

                    res = SimResult(
                        times=_np_times,
                        states=_np_states,
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
                policy = build_policy_by_name(cfg.policy.name, alpha=float(alpha), mu=mu)
                # Dynamic max_events ceiling matches the JAX engine formula and
                # makes the NumPy path consistent with policy_comparison.py (SG#2/3 fix).
                _np_max_events = int(
                    (lam + float(mu.sum()))
                    * cfg.simulation.ssa.sim_time
                    * 1.5
                ) + 1000
                for rep in range(cfg.simulation.num_replications):
                    rng = np.random.default_rng(_cell_seed + rep)
                    res = simulate(
                        num_servers=N,
                        arrival_rate=lam,
                        service_rates=mu,
                        policy=policy,
                        sim_time=cfg.simulation.ssa.sim_time,
                        sample_interval=cfg.simulation.ssa.sample_interval,
                        rng=rng,
                        max_events=_np_max_events,
                    )
                    avg_q = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
                    rep_means.append(avg_q.sum())
                    rep_diag = stationarity_diagnostic(res, burn_in_fraction=cfg.simulation.burn_in_fraction)
                    rep_stationary_flags.append(bool(rep_diag["is_stationary"]))
                    last_res = res

            # SG#1 FIX: Aggregate per-replication means into mean_Q
            # Previously, rep_means was collected but never averaged into mean_Q[i,j],
            # causing all E[Q_total] values to be 0.0 across all configurations.
            mean_Q[i, j] = float(np.mean(rep_means)) if rep_means else 0.0

            stationarity_rate = float(np.mean(rep_stationary_flags)) if rep_stationary_flags else 0.0
            stationary[i, j] = stationarity_rate >= cfg.verification.stationarity_threshold
            
            # Export scalar metrics
            metrics = {
                "rho": float(rho),
                "lam": float(lam),
                "alpha": float(alpha),
                "mean_q_total": float(mean_Q[i, j]),
                "is_stationary": bool(stationary[i, j]),
                "stationarity_rate": stationarity_rate,
                "backend": "JAX" if cfg.jax.enabled else "NumPy",  # SG-7: Add backend tag for traceability
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
    rho_labels = [f"ρ={r:.2f}" for r in rho_values]
    f_plot = out_dir / "alpha_sweep"
    plot_alpha_sweep(alpha_values, mean_Q, rho_labels, save_path=f_plot, theme='publication', formats=['png', 'pdf'])
    log.info(f"\nSaved plot: {f_plot}.png, {f_plot}.pdf")
    if run:
        run.log({"alpha_sweep_plot": wandb.Image(str(out_dir / "alpha_sweep.png"))})

    # Summary Log
    n_fail = np.sum(~stationary)
    log.info(f"\nSummary: {n_fail}/{stationary.size} configurations non-stationary.")
    
    if run:
        run.finish()


if __name__ == "__main__":
    main()
