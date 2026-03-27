"""
Stress test: GibbsQ scaling, critical load, and heterogeneity.

Tests:
1. Massive-N Scaling: N=1024 servers.
2. Critical Load: rho=0.999.
3. Extreme Heterogeneity: 100x variance in service rates.

Uses JAX vmap for parallel replications.

PATCH NOTES (2026-03-12):
  BUG-1/2/3 FIX: All three tests previously computed max_samples using
  cfg.simulation.sample_interval (0.05 in small.yaml), producing buffers of
  100,001–1,000,001 slots. JAX's XLA compiler hangs when loop-carried state
  tensors are this large. Fix: use _STRESS_SAMPLE_INTERVAL = 1.0 uniformly
  for all stress tests. Accuracy is unaffected — stress tests measure Gini
  coefficients and queue-length means, not fine time-series resolution.
"""

import logging
import hydra
import numpy as np
import jax.numpy as jnp
from omegaconf import DictConfig

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.engines.distributed import sharded_replications
from gibbsq.engines.numpy_engine import SimResult
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.analysis.metrics import (
    time_averaged_queue_lengths,
    stationarity_diagnostic,
    gini_coefficient,
    mser5_truncation,
    gelman_rubin_diagnostic
)
from gibbsq.utils.logging import setup_wandb, get_run_config

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATCH: Fixed coarse sampling interval for ALL stress-test simulations.
#
# Rationale: stress_test.py overrides sim_time independently of the config
# (5 000 s, 10 000 s, 50 000 s). If max_samples is derived from
# cfg.simulation.sample_interval (0.05 s in small.yaml), the resulting buffer
# lengths (100 001 – 1 000 001) make XLA hang during JIT compilation because
# those arrays are loop-carried state in a nested lax.while_loop.
#
# 1.0 s gives 5 001 – 50 001 slots — fast to compile, statistically sound
# Removed scripted hardcode, now using cfg.stress.sample_interval

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2026-03-14: Reduced sim_time for all stress tests.
#
# ROOT CAUSE (confirmed via experiments/testing/debug_stress_runner.py):
#   The Gillespie SSA in JAX runs inside lax.while_loop which is STRICTLY
#   SEQUENTIAL — each event iteration executes one after another.  With
#   N=4, rho=0.8, the event rate is ~14.4/s.  At sim_time=5000, that's
#   ~72,000 sequential while_loop iterations per replication.  Combined
#   with vmap(R=3), this takes >15 minutes on both GPU (Colab) and CPU.
#
#   Diagnostic results (H1-H4 all <1.3s, H5 60x smaller but hung 10+ min)
#   confirmed that event count is the sole bottleneck — JIT compilation,
#   buffer shapes, and vmap batch overhead are negligible.
#
# FIX: Reduce sim_time to 500s for N-scaling, 1000s/5000s for critical load,
#   and 1000s for heterogeneity.  These still provide >10x the mixing time
#   for the CTMC at the given rho values, which is sufficient for Gini and
#   mean-queue metrics.
# ─────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    if not cfg.jax.enabled:
        raise ValueError("stress_test requires cfg.jax.enabled=True")

    # Initialize Run Capsule (Dynamic Directory + Config Persistence)
    run_dir, run_id = get_run_config(cfg, "scientific_stress", raw_cfg)

    # Initialize WandB via centralized utility
    run = setup_wandb(cfg, raw_cfg, default_group="scientific_stress", run_id=run_id, run_dir=run_dir)

    # Use the isolated Run Directory for all outputs
    out_dir = run_dir

    log.info("=" * 60)
    log.info("  GibbsQ Stress Test (JAX Accelerator Active)")
    log.info("=" * 60)

    # Dashboard data accumulators (for plot_stress_dashboard at the end)
    _scaling_data = {"n_values": [], "mean_q": [], "gini": []}
    _critical_data = {"rho_values": [], "mean_q": [], "stationary": []}
    _hetero_data = {"scenario_names": [], "mean_q": [], "gini": []}

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: MASSIVE-N SCALING
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n[TEST 1] Massive-N Scaling Analysis")
    n_targets = [cfg.stress.n_values[0]] if raw_cfg.get("debug", False) else cfg.stress.n_values

    for N in n_targets:
        mu = jnp.ones(N) * 2.0          # normalised service rate
        lam = cfg.stress.massive_n_rho * float(jnp.sum(mu))

        # PATCH SG1: Apply the documented reduction to 500 s (non-debug).
        # cfg.simulation.ssa.sim_time (5000 s) causes O(N×T) OOM for N>=512.
        _sim_time_t1 = 100.0 if raw_cfg.get("debug", False) else cfg.stress.massive_n_sim_time
        max_samples_t1 = int(_sim_time_t1 / cfg.stress.sample_interval) + 1

        log.info(f"  Simulating N={N} experts (rho={cfg.stress.massive_n_rho})...")
        _pmap = {"uniform": 0, "proportional": 1, "jsq": 2, "softmax": 3, "power_of_d": 4, "uas": 6}

        times, states, (arrs, deps) = sharded_replications(
            num_replications=cfg.simulation.num_replications,
            num_servers=N,
            arrival_rate=lam,
            service_rates=mu,
            alpha=cfg.system.alpha,
            sim_time=_sim_time_t1,
            sample_interval=cfg.stress.sample_interval,
            base_seed=cfg.simulation.seed,
            max_samples=max_samples_t1,
            policy_type=_pmap.get(cfg.policy.name, 3)
        )

        ginis = []
        avg_q_vals_t1 = []
        for r in range(cfg.simulation.num_replications):
            _np_t = np.array(times[r])
            _np_s = np.array(states[r])
            # Truncate invalid trailing JAX buffer slots (SG#5 fix)
            _vm = _np_t > 0; _vm[0] = True
            _vl = int(np.sum(_vm))
            if _vl < _np_s.shape[0]:
                _np_t = _np_t[:_vl]; _np_s = _np_s[:_vl]
            res = SimResult(
                times=_np_t, states=_np_s,
                arrival_count=int(arrs[r]),
                departure_count=int(deps[r]),
                final_time=float(times[r][-1]),
                num_servers=N
            )
            avg_q = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
            ginis.append(gini_coefficient(avg_q))
            avg_q_vals_t1.append(avg_q.sum())

        avg_gini = np.mean(ginis)
        log.info(f"    -> Average Gini Imbalance: {avg_gini:.4f}")
        if wandb and wandb.run:
            wandb.log({"massive_n/N": N, "massive_n/avg_gini": avg_gini})

        # Accumulate for dashboard
        _mean_q_t1 = float(np.mean(avg_q_vals_t1))
        _scaling_data["n_values"].append(int(N))
        _scaling_data["mean_q"].append(_mean_q_t1)
        _scaling_data["gini"].append(float(avg_gini))

        append_metrics_jsonl(
            {"test": "massive_n", "N": int(N), "avg_gini": float(avg_gini)},
            out_dir / "metrics.jsonl"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: CRITICAL LOAD ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n[TEST 2] Critical Load Analysis (rho up to {cfg.stress.critical_rhos[-1]})")
    N_fixed = cfg.stress.critical_load_n
    mu_fixed = jnp.ones(N_fixed)
    cap_fixed = float(jnp.sum(mu_fixed))

    rho_targets = [cfg.stress.critical_rhos[0]] if raw_cfg.get("debug", False) else cfg.stress.critical_rhos

    for rho in rho_targets:
        lam = rho * cap_fixed

        # Near-critical systems need longer horizons to reach stationarity.
        # PATCH BUG-2: use _STRESS_SAMPLE_INTERVAL, not cfg.simulation.sample_interval.
        # old:  max_samples = int(sim_time_critical / cfg.simulation.sample_interval) + 1
        #       → up to 1 000 001 for rho > 0.99
        # new:  max_samples = int(sim_time_critical / 1.0) + 1
        #       → up to 50 001 for rho > 0.99
        if raw_cfg.get("debug", False):
            _sim_time_crit = 500.0
        else:
            # PATCH SG-B: linear mixing-time scaling O(1/(1-ρ)) for fixed-N
            # birth-death chains (spectral gap γ = Θ(1-ρ) → τ_mix = O(1/(1-ρ))).
            # The quadratic O(1/(1-ρ)²) is only valid in the Halfin-Whitt
            # many-server regime (N→∞ simultaneously) — not this fixed-N setting.
            # Formula now matches critical_load.py exactly (SG-B consistency fix).
            # Halfin-Whitt quadratic scaling O(1/(1-ρ)²) applies only to
            # the many-server asymptotic regime where both N and λ grow.
            _base_rho_crit = cfg.stress.critical_load_base_rho
            _rho_factor_crit = max(1.0, ((1.0 - _base_rho_crit) / max(1.0 - rho, 1e-6)))
            _sim_time_crit = min(cfg.simulation.ssa.sim_time * _rho_factor_crit, cfg.stress.critical_load_max_sim_time)
            if _sim_time_crit >= cfg.stress.critical_load_max_sim_time:
                log.warning(
                    f"  [!] rho={rho:.4f}: sim_time capped at 100,000s "
                    f"(linear mixing time ~ {cfg.simulation.ssa.sim_time * _rho_factor_crit:.0f}s). "
                    f"E[Q] near criticality may be underestimated. "
                    f"Report only rho<=0.999 and add mixing-time caveat."
                )

        max_samples_crit = int(_sim_time_crit / cfg.stress.sample_interval) + 1

        log.info(f"  Simulating rho={rho:.3f} (T={_sim_time_crit})...")
        _pmap = {"uniform": 0, "proportional": 1, "jsq": 2, "softmax": 3, "power_of_d": 4, "uas": 6}

        times, states, (arrs, deps) = sharded_replications(
            num_replications=cfg.simulation.num_replications,
            num_servers=N_fixed,
            arrival_rate=lam,
            service_rates=mu_fixed,
            alpha=cfg.system.alpha,
            sim_time=_sim_time_crit,
            sample_interval=cfg.stress.sample_interval,
            base_seed=cfg.simulation.seed,
            max_samples=max_samples_crit,
            policy_type=_pmap.get(cfg.policy.name, 3)
        )

        q_totals = []
        stationary_count = 0

        total_q_trajectories = np.array(states).sum(axis=2)  # (Reps, TimeSteps)

        # construction below.  total_q_trajectories was already patched but
        # SimResult.states was not — causing trailing-zero bias in E[Q].
        valid_lens = []
        for r in range(cfg.simulation.num_replications):
            _valid_mask = np.array(times[r]) > 0
            _valid_mask[0] = True  # t=0 initial snapshot is always valid
            _valid_len = int(np.sum(_valid_mask))
            valid_lens.append(_valid_len)
            if _valid_len < total_q_trajectories.shape[1]:
                # Fill trailing zeros with the last valid sample (flat tail, not biased)
                total_q_trajectories[r, _valid_len:] = total_q_trajectories[r, _valid_len - 1]


        # SG#11 Fix: Use data-driven MSER-5 truncation consistently instead of fixed 20%
        # Compute MSER-5 truncations for all replicas to find the maximum burn-in period
        trunc_samples = [mser5_truncation(traj) for traj in total_q_trajectories]
        max_burn_samples = max(trunc_samples) if trunc_samples else 0
        
        truncated_trajectories = total_q_trajectories[:, max_burn_samples:]
        r_hat = gelman_rubin_diagnostic(truncated_trajectories)
        log.info(f"    -> Gelman-Rubin R-hat across replicas (post MSER-5 burn-in): {r_hat:.4f}")

        for r in range(cfg.simulation.num_replications):
            # SG-2 FIX (v2): Truncate invalid trailing buffer slots instead
            # of padding with the last valid state. Padding gives the final
            # state disproportionate weight in time_averaged_queue_lengths.
            _np_times_r = np.array(times[r])
            _np_states_r = np.array(states[r])
            _vl = valid_lens[r]
            if _vl < _np_states_r.shape[0]:
                _np_times_r = _np_times_r[:_vl]
                _np_states_r = _np_states_r[:_vl]

            res = SimResult(
                _np_times_r, _np_states_r,
                int(arrs[r]), int(deps[r]), float(times[r][-1]), N_fixed
            )


            traj = total_q_trajectories[r]
            d_star = trunc_samples[r]
            trunc_fraction = d_star / max(len(traj), 1)

            avg_q = time_averaged_queue_lengths(res, trunc_fraction).sum()
            q_totals.append(avg_q)

            diag = stationarity_diagnostic(res, burn_in_fraction=trunc_fraction)
            if diag["is_stationary"]:
                stationary_count += 1

        log.info(
            f"    -> Avg E[Q_total]: {np.mean(q_totals):.2f} | "
            f"Stationarity: {stationary_count}/{cfg.simulation.num_replications}"
        )
        if wandb and wandb.run:
            wandb.log({
                "critical_load/rho": rho,
                "critical_load/mean_q": np.mean(q_totals),
                "critical_load/stationary_rate": stationary_count / cfg.simulation.num_replications
            })

        # Accumulate for dashboard
        _critical_data["rho_values"].append(float(rho))
        _critical_data["mean_q"].append(float(np.mean(q_totals)))
        _critical_data["stationary"].append(stationary_count == cfg.simulation.num_replications)

        append_metrics_jsonl(
            {
                "test": "critical_load",
                "rho": float(rho),
                "mean_q": float(np.mean(q_totals)),
                "stationary_rate": float(stationary_count / cfg.simulation.num_replications)
            },
            out_dir / "metrics.jsonl"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: EXTREME HETEROGENEITY
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n[TEST 3] Extreme Heterogeneity Resilience (100x Speed Gap)")
    mu_het = jnp.array(cfg.stress.mu_het)
    cap_het = float(jnp.sum(mu_het))
    lam_het = cfg.stress.heterogeneity_rho * cap_het

    log.info(f"  Simulating heterogenous setup: mu={mu_het}")

    # PATCH BUG-3 (two fixes):
    #   Fix A — derive sim_time into a named variable so max_samples stays consistent.
    #   Fix B — use _STRESS_SAMPLE_INTERVAL instead of cfg.simulation.sample_interval.
    # old:  sim_time=... (conditional)
    #       max_samples=int(10000.0 / cfg.simulation.sample_interval) + 1  ← hardcoded 10000!
    # new:  _sim_time_het = ... (same conditional)
    #       max_samples=int(_sim_time_het / _STRESS_SAMPLE_INTERVAL) + 1   ← consistent
    # PATCH 2026-03-14: reduced from 10000→1000 (debug: 1000→500).
    _sim_time_het = 500.0 if raw_cfg.get("debug", False) else cfg.stress.heterogeneity_sim_time
    max_samples_het = int(_sim_time_het / cfg.stress.sample_interval) + 1    # PATCH

    times, states, (arrs, deps) = sharded_replications(
        num_replications=cfg.simulation.num_replications,
        num_servers=4,
        arrival_rate=lam_het,
        service_rates=mu_het,
        alpha=cfg.system.alpha,
        sim_time=_sim_time_het,
        sample_interval=cfg.stress.sample_interval,    # PATCH: was cfg.simulation.sample_interval
        base_seed=cfg.simulation.seed,
        max_samples=max_samples_het,               # PATCH: consistent with _sim_time_het
        policy_type=6  # Use UAS for heterogeneity-aware routing
    )

    work_dist = []
    for r in range(cfg.simulation.num_replications):
        _np_t3 = np.array(times[r])
        _np_s3 = np.array(states[r])
        # Truncate invalid trailing JAX buffer slots (SG#5 fix)
        _vm3 = _np_t3 > 0; _vm3[0] = True
        _vl3 = int(np.sum(_vm3))
        if _vl3 < _np_s3.shape[0]:
            _np_t3 = _np_t3[:_vl3]; _np_s3 = _np_s3[:_vl3]
        res = SimResult(
            _np_t3, _np_s3,
            int(arrs[r]), int(deps[r]), float(times[r][-1]), 4
        )
        avg_q_per_srv = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
        work_dist.append(avg_q_per_srv)

    mean_dist = np.mean(work_dist, axis=0)
    log.info(f"    -> Mean Queue per Expert: {mean_dist}")
    log.info(f"    -> Gini: {gini_coefficient(mean_dist):.4f}")

    if wandb and wandb.run:
        wandb.log({"heterogeneity/gini": gini_coefficient(mean_dist)})
        wandb.finish()

    # Accumulate for dashboard
    _hetero_data["scenario_names"].append("100x Gap")
    _hetero_data["mean_q"].append(float(mean_dist.sum()))
    _hetero_data["gini"].append(float(gini_coefficient(mean_dist)))

    append_metrics_jsonl(
        {
            "test": "heterogeneity",
            "gini": float(gini_coefficient(mean_dist)),
            "mean_dist": [float(x) for x in mean_dist]
        },
        out_dir / "metrics.jsonl"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STRESS TEST DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    from gibbsq.analysis.plotting import plot_stress_dashboard
    import matplotlib.pyplot as plt

    plot_path = out_dir / "stress_dashboard"
    fig = plot_stress_dashboard(
        scaling_data=_scaling_data,
        critical_data=_critical_data,
        hetero_data=_hetero_data,
        save_path=plot_path,
        theme="publication",
        formats=["png", "pdf"],
    )
    plt.close(fig)
    log.info(f"Stress dashboard saved to {plot_path}.png, {plot_path}.pdf")

    log.info("\nStress test complete.")


if __name__ == "__main__":
    main()
