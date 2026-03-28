import logging

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from gibbsq.analysis.metrics import stationarity_diagnostic, time_averaged_queue_lengths
from gibbsq.analysis.plotting import plot_alpha_sweep
from gibbsq.analysis.theme import apply_theme
from gibbsq.core.builders import build_policy_by_name
from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.jax_engine import policy_name_to_type, run_replications_jax
from gibbsq.engines.numpy_engine import SimResult, simulate
from gibbsq.utils.exporter import append_metrics_jsonl, save_trajectory_parquet
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.progress import create_progress, iter_progress

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "sweep", raw_cfg)
    run = setup_wandb(cfg, raw_cfg, default_group="stability_sweep", run_id=run_id, run_dir=run_dir)

    apply_theme("publication")

    sc = cfg.system
    N = sc.num_servers
    mu = np.asarray(sc.service_rates, dtype=np.float64)
    mu_jax = jnp.asarray(mu)
    cap = float(mu.sum())

    out_dir = run_dir
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    if cfg.policy.name not in ["softmax", "uas"]:
        raise ValueError(
            f"Stability sweep requires policy.name == 'softmax' or 'uas', "
            f"but config has '{cfg.policy.name}'."
        )

    alpha_values = np.array(cfg.stability_sweep.alpha_vals)
    rho_values = np.array(cfg.stability_sweep.rho_vals)

    log.info(f"System: N={N}, cap={cap:.4f} | Backend: {'JAX' if cfg.jax.enabled else 'NumPy'}")
    log.info(
        f"Grid: {len(alpha_values)} alpha x {len(rho_values)} rho "
        f"x {cfg.simulation.num_replications} reps"
    )

    n_rho, n_alpha = len(rho_values), len(alpha_values)
    mean_Q = np.zeros((n_rho, n_alpha))
    stationary = np.zeros((n_rho, n_alpha), dtype=bool)

    total_runs = n_rho * n_alpha
    completed = 0

    with create_progress(total=total_runs, desc="sweep", unit="cell") as cell_bar:
        for i, rho in enumerate(rho_values):
            lam = rho * cap
            log.info(f"\n{'-' * 60}\n  rho = {rho:.2f}  (lam = {lam:.4f})\n{'-' * 60}")

            for j, alpha in enumerate(alpha_values):
                completed += 1
                cell_bar.set_postfix(
                    {"rho": f"{rho:.2f}", "alpha": f"{alpha:.2f}", "done": f"{completed}/{total_runs}"},
                    refresh=False,
                )

                _cell_seed = cfg.simulation.seed + (i * n_alpha + j) * cfg.simulation.num_replications
                rep_means = []
                rep_stationary_flags: list[bool] = []
                last_res = None

                if cfg.jax.enabled:
                    _sim_time = cfg.simulation.ssa.sim_time
                    _sample_interval = cfg.simulation.ssa.sample_interval
                    max_samples = int(_sim_time / _sample_interval) + 1
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
                        policy_type=policy_name_to_type(cfg.policy.name),
                    )

                    for r in iter_progress(
                        range(cfg.simulation.num_replications),
                        total=cfg.simulation.num_replications,
                        desc=f"sweep reps rho={rho:.2f} alpha={alpha:.2f}",
                        unit="rep",
                        leave=False,
                    ):
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
                            final_time=float(_np_times[-1]),
                            num_servers=N,
                        )
                        avg_q = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
                        rep_means.append(avg_q.sum())
                        rep_diag = stationarity_diagnostic(res, burn_in_fraction=cfg.simulation.burn_in_fraction)
                        rep_stationary_flags.append(bool(rep_diag["is_stationary"]))
                        last_res = res
                else:
                    policy = build_policy_by_name(cfg.policy.name, alpha=float(alpha), mu=mu)
                    _np_max_events = int(
                        (lam + float(mu.sum()))
                        * cfg.simulation.ssa.sim_time
                        * 1.5
                    ) + 1000
                    for rep in iter_progress(
                        range(cfg.simulation.num_replications),
                        total=cfg.simulation.num_replications,
                        desc=f"sweep reps rho={rho:.2f} alpha={alpha:.2f}",
                        unit="rep",
                        leave=False,
                    ):
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

                mean_Q[i, j] = float(np.mean(rep_means)) if rep_means else 0.0
                stationarity_rate = float(np.mean(rep_stationary_flags)) if rep_stationary_flags else 0.0
                threshold = float(cfg.verification.stationarity_threshold)
                stationary[i, j] = stationarity_rate >= threshold

                metrics = {
                    "rho": float(rho),
                    "lam": float(lam),
                    "alpha": float(alpha),
                    "mean_q_total": float(mean_Q[i, j]),
                    "is_stationary": bool(stationary[i, j]),
                    "stationarity_rate": stationarity_rate,
                    "backend": "JAX" if cfg.jax.enabled else "NumPy",
                }
                append_metrics_jsonl(metrics, out_dir / "metrics.jsonl")
                if run:
                    run.log(metrics)

                if cfg.simulation.export_trajectories and last_res is not None:
                    fname = out_dir / f"trajectories/rho{rho:.2f}_alpha{alpha:.2f}.parquet"
                    save_trajectory_parquet(last_res, fname)

                stat_str = "OK" if stationary[i, j] else "NONSTATIONARY"
                log.info(
                    f"  alpha={alpha:6.2f} | E[Q_total]={mean_Q[i, j]:8.2f} | "
                    f"{stat_str}  ({completed}/{total_runs})"
                )
                cell_bar.update(1)

    rho_labels = [f"ρ={r:.2f}" for r in rho_values]
    f_plot = out_dir / "alpha_sweep"
    plot_alpha_sweep(alpha_values, mean_Q, rho_labels, save_path=f_plot, theme="publication", formats=["png", "pdf"])
    log.info(f"\nSaved plot: {f_plot}.png, {f_plot}.pdf")
    if run:
        run.log({"alpha_sweep_plot": wandb.Image(str(out_dir / "alpha_sweep.png"))})

    n_fail = np.sum(~stationary)
    log.info(f"\nSummary: {n_fail}/{stationary.size} configurations non-stationary.")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
