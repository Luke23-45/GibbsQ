"""
Stress test: GibbsQ scaling, critical load, and heterogeneity.
"""

import logging

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from gibbsq.analysis.metrics import (
    gelman_rubin_diagnostic,
    gini_coefficient,
    mser5_truncation,
    stationarity_diagnostic_from_index,
    time_averaged_queue_lengths,
    time_averaged_queue_lengths_from_index,
)
from gibbsq.core.config import critical_load_sim_time, hydra_to_config, validate
from gibbsq.engines.distributed import sharded_replications
from gibbsq.engines.jax_engine import policy_name_to_type
from gibbsq.engines.numpy_engine import SimResult
from gibbsq.utils.exporter import append_metrics_jsonl
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

    if not cfg.jax.enabled:
        raise ValueError("stress_test requires cfg.jax.enabled=True")

    run_dir, run_id = get_run_config(cfg, "stress", raw_cfg)
    run = setup_wandb(cfg, raw_cfg, default_group="scientific_stress", run_id=run_id, run_dir=run_dir)
    out_dir = run_dir

    log.info("=" * 60)
    log.info("  GibbsQ Stress Test (JAX Accelerator Active)")
    log.info("=" * 60)

    scaling_data = {"n_values": [], "mean_q": [], "gini": []}
    critical_data = {"rho_values": [], "mean_q": [], "stationary": []}
    hetero_data = {"scenario_names": [], "mean_q": [], "gini": []}

    with create_progress(total=3, desc="stress", unit="stage") as stage_bar:
        log.info("\n[TEST 1] Massive-N Scaling Analysis")
        n_targets = cfg.stress.n_values
        with create_progress(total=len(n_targets), desc="stress: massive-N", unit="N", leave=False) as scaling_bar:
            for N in n_targets:
                scaling_bar.set_postfix({"N": int(N)}, refresh=False)
                mu = jnp.ones(N) * 2.0
                lam = cfg.stress.massive_n_rho * float(jnp.sum(mu))
                sim_time_t1 = cfg.stress.massive_n_sim_time
                max_samples_t1 = int(sim_time_t1 / cfg.stress.sample_interval) + 1

                log.info(f"  Simulating N={N} experts (rho={cfg.stress.massive_n_rho})...")
                times, states, (arrs, deps) = sharded_replications(
                    num_replications=cfg.simulation.num_replications,
                    num_servers=N,
                    arrival_rate=lam,
                    service_rates=mu,
                    alpha=cfg.system.alpha,
                    sim_time=sim_time_t1,
                    sample_interval=cfg.stress.sample_interval,
                    base_seed=cfg.simulation.seed,
                    max_samples=max_samples_t1,
                    policy_type=policy_name_to_type(cfg.policy.name),
                )

                ginis = []
                avg_q_vals = []
                for rep_idx in iter_progress(
                    range(cfg.simulation.num_replications),
                    total=cfg.simulation.num_replications,
                    desc=f"stress massive-N N={int(N)}",
                    unit="rep",
                    leave=False,
                ):
                    np_t = np.array(times[rep_idx])
                    np_s = np.array(states[rep_idx])
                    valid_mask = np_t > 0
                    valid_mask[0] = True
                    valid_len = int(np.sum(valid_mask))
                    if valid_len < np_s.shape[0]:
                        np_t = np_t[:valid_len]
                        np_s = np_s[:valid_len]
                    res = SimResult(
                        times=np_t,
                        states=np_s,
                        arrival_count=int(arrs[rep_idx]),
                        departure_count=int(deps[rep_idx]),
                        final_time=float(np_t[-1]),
                        num_servers=N,
                    )
                    avg_q = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
                    ginis.append(gini_coefficient(avg_q))
                    avg_q_vals.append(avg_q.sum())

                avg_gini = float(np.mean(ginis))
                mean_q = float(np.mean(avg_q_vals))
                log.info(f"    -> Average Gini Imbalance: {avg_gini:.4f}")
                if wandb and wandb.run:
                    wandb.log({"massive_n/N": N, "massive_n/avg_gini": avg_gini})

                scaling_data["n_values"].append(int(N))
                scaling_data["mean_q"].append(mean_q)
                scaling_data["gini"].append(avg_gini)
                append_metrics_jsonl({"test": "massive_n", "N": int(N), "avg_gini": avg_gini}, out_dir / "metrics.jsonl")
                scaling_bar.update(1)
        stage_bar.update(1)

        log.info(f"\n[TEST 2] Critical Load Analysis (rho up to {cfg.stress.critical_rhos[-1]})")
        N_fixed = cfg.stress.critical_load_n
        mu_fixed = jnp.ones(N_fixed)
        cap_fixed = float(jnp.sum(mu_fixed))
        rho_targets = cfg.stress.critical_rhos

        with create_progress(total=len(rho_targets), desc="stress: critical", unit="rho", leave=False) as critical_bar:
            for rho in rho_targets:
                critical_bar.set_postfix({"rho": f"{rho:.3f}"}, refresh=False)
                lam = rho * cap_fixed
                sim_time_crit = critical_load_sim_time(cfg, float(rho))

                max_samples_crit = int(sim_time_crit / cfg.stress.sample_interval) + 1
                log.info(f"  Simulating rho={rho:.3f} (T={sim_time_crit})...")
                times, states, (arrs, deps) = sharded_replications(
                    num_replications=cfg.simulation.num_replications,
                    num_servers=N_fixed,
                    arrival_rate=lam,
                    service_rates=mu_fixed,
                    alpha=cfg.system.alpha,
                    sim_time=sim_time_crit,
                    sample_interval=cfg.stress.sample_interval,
                    base_seed=cfg.simulation.seed,
                    max_samples=max_samples_crit,
                    policy_type=policy_name_to_type(cfg.policy.name),
                )

                total_q_trajectories = np.array(states).sum(axis=2)
                valid_lens = []
                for rep_idx in range(cfg.simulation.num_replications):
                    valid_mask = np.array(times[rep_idx]) > 0
                    valid_mask[0] = True
                    valid_len = int(np.sum(valid_mask))
                    valid_lens.append(valid_len)
                    if valid_len < total_q_trajectories.shape[1]:
                        total_q_trajectories[rep_idx, valid_len:] = total_q_trajectories[rep_idx, valid_len - 1]

                trunc_samples = [mser5_truncation(traj) for traj in total_q_trajectories]
                max_burn_samples = max(trunc_samples) if trunc_samples else 0
                truncated_trajectories = total_q_trajectories[:, max_burn_samples:]
                r_hat = gelman_rubin_diagnostic(truncated_trajectories)
                log.info(f"    -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): {r_hat:.4f}")

                q_totals = []
                stationary_count = 0
                for rep_idx in iter_progress(
                    range(cfg.simulation.num_replications),
                    total=cfg.simulation.num_replications,
                    desc=f"stress critical rho={rho:.3f}",
                    unit="rep",
                    leave=False,
                ):
                    np_times = np.array(times[rep_idx])
                    np_states = np.array(states[rep_idx])
                    valid_len = valid_lens[rep_idx]
                    if valid_len < np_states.shape[0]:
                        np_times = np_times[:valid_len]
                        np_states = np_states[:valid_len]

                    res = SimResult(
                        np_times,
                        np_states,
                        int(arrs[rep_idx]),
                        int(deps[rep_idx]),
                        float(np_times[-1]),
                        N_fixed,
                    )
                    d_star = trunc_samples[rep_idx]
                    avg_q = time_averaged_queue_lengths_from_index(res, d_star).sum()
                    q_totals.append(avg_q)
                    diag = stationarity_diagnostic_from_index(res, start_idx=d_star)
                    if diag["is_stationary"]:
                        stationary_count += 1

                mean_q = float(np.mean(q_totals))
                stationary_rate = float(stationary_count / cfg.simulation.num_replications)
                log.info(
                    f"    -> Avg E[Q_total]: {mean_q:.2f} | "
                    f"Stationarity: {stationary_count}/{cfg.simulation.num_replications}"
                )
                if wandb and wandb.run:
                    wandb.log({
                        "critical_load/rho": rho,
                        "critical_load/mean_q": mean_q,
                        "critical_load/stationary_rate": stationary_rate,
                    })

                critical_data["rho_values"].append(float(rho))
                critical_data["mean_q"].append(mean_q)
                critical_data["stationary"].append(stationary_count == cfg.simulation.num_replications)
                append_metrics_jsonl(
                    {
                        "test": "critical_load",
                        "rho": float(rho),
                        "mean_q": mean_q,
                        "stationary_rate": stationary_rate,
                    },
                    out_dir / "metrics.jsonl",
                )
                critical_bar.update(1)
        stage_bar.update(1)

        log.info("\n[TEST 3] Extreme Heterogeneity Resilience (100x Speed Gap)")
        mu_het = jnp.array(cfg.stress.mu_het)
        cap_het = float(jnp.sum(mu_het))
        lam_het = cfg.stress.heterogeneity_rho * cap_het
        sim_time_het = cfg.stress.heterogeneity_sim_time
        max_samples_het = int(sim_time_het / cfg.stress.sample_interval) + 1
        log.info(f"  Simulating heterogenous setup: mu={mu_het}")

        times, states, (arrs, deps) = sharded_replications(
            num_replications=cfg.simulation.num_replications,
            num_servers=4,
            arrival_rate=lam_het,
            service_rates=mu_het,
            alpha=cfg.system.alpha,
            sim_time=sim_time_het,
            sample_interval=cfg.stress.sample_interval,
            base_seed=cfg.simulation.seed,
            max_samples=max_samples_het,
            policy_type=6,
        )

        work_dist = []
        for rep_idx in iter_progress(
            range(cfg.simulation.num_replications),
            total=cfg.simulation.num_replications,
            desc="stress heterogeneity",
            unit="rep",
            leave=False,
        ):
            np_t = np.array(times[rep_idx])
            np_s = np.array(states[rep_idx])
            valid_mask = np_t > 0
            valid_mask[0] = True
            valid_len = int(np.sum(valid_mask))
            if valid_len < np_s.shape[0]:
                np_t = np_t[:valid_len]
                np_s = np_s[:valid_len]
            res = SimResult(np_t, np_s, int(arrs[rep_idx]), int(deps[rep_idx]), float(np_t[-1]), 4)
            avg_q_per_srv = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
            work_dist.append(avg_q_per_srv)

        mean_dist = np.mean(work_dist, axis=0)
        mean_gini = float(gini_coefficient(mean_dist))
        log.info(f"    -> Mean Queue per Expert: {mean_dist}")
        log.info(f"    -> Gini: {mean_gini:.4f}")

        if wandb and wandb.run:
            wandb.log({"heterogeneity/gini": mean_gini})

        hetero_data["scenario_names"].append("100x Gap")
        hetero_data["mean_q"].append(float(mean_dist.sum()))
        hetero_data["gini"].append(mean_gini)
        append_metrics_jsonl(
            {"test": "heterogeneity", "gini": mean_gini, "mean_dist": [float(x) for x in mean_dist]},
            out_dir / "metrics.jsonl",
        )
        stage_bar.update(1)

    from gibbsq.analysis.plotting import plot_stress_dashboard
    import matplotlib.pyplot as plt

    plot_path = out_dir / "stress_dashboard"
    fig = plot_stress_dashboard(
        scaling_data=scaling_data,
        critical_data=critical_data,
        hetero_data=hetero_data,
        save_path=plot_path,
        theme="publication",
        formats=["png", "pdf"],
    )
    plt.close(fig)
    log.info(f"Stress dashboard saved to {plot_path}.png, {plot_path}.pdf")
    log.info("\nStress test complete.")

    if wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
