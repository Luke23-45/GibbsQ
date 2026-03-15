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
# for Gini and mean-queue metrics (stress tests do not need fine resolution).
# ─────────────────────────────────────────────────────────────────────────────
_STRESS_SAMPLE_INTERVAL: float = 1.0

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

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: MASSIVE-N SCALING
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n[TEST 1] Massive-N Scaling Analysis")
    n_targets = [cfg.stress.n_values[0]] if raw_cfg.get("debug", False) else cfg.stress.n_values

    for N in n_targets:
        mu = jnp.ones(N) * 2.0          # normalised service rate
        lam = 0.8 * float(jnp.sum(mu))  # rho = 0.8

        # PATCH SG1: Apply the documented reduction to 500 s (non-debug).
        # cfg.simulation.ssa.sim_time (5000 s) causes O(N×T) OOM for N>=512.
        _sim_time_t1 = 100.0 if raw_cfg.get("debug", False) else 500.0
        max_samples_t1 = int(_sim_time_t1 / _STRESS_SAMPLE_INTERVAL) + 1

        log.info(f"  Simulating N={N} experts (rho=0.8)...")

        times, states, (arrs, deps) = sharded_replications(
            num_replications=cfg.simulation.num_replications,
            num_servers=N,
            arrival_rate=lam,
            service_rates=mu,
            alpha=cfg.system.alpha,
            sim_time=_sim_time_t1,
            sample_interval=_STRESS_SAMPLE_INTERVAL,
            base_seed=cfg.simulation.seed,
            max_samples=max_samples_t1,
            policy_type=3  # Softmax
        )

        ginis = []
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
            ginis.append(gini_coefficient(avg_q))

        avg_gini = np.mean(ginis)
        log.info(f"    -> Average Gini Imbalance: {avg_gini:.4f}")
        if wandb and wandb.run:
            wandb.log({"massive_n/N": N, "massive_n/avg_gini": avg_gini})

        append_metrics_jsonl(
            {"test": "massive_n", "N": int(N), "avg_gini": float(avg_gini)},
            out_dir / "metrics.jsonl"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: CRITICAL LOAD ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n[TEST 2] Critical Load Analysis (rho up to 0.999)")
    N_fixed = 10
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
            # SG#1 FIX: Scale sim_time with theoretical CTMC mixing time.
            # Mixing time ~ O(1/(1-rho)^2) (Meyn & Tweedie 1993, §4).
            # Use min(100/(1-rho)^2, 100_000) as a compute-budget cap.
            #   rho=0.90 → 10,000s    rho=0.95 → 40,000s
            #   rho=0.99 → 100,000s   rho=0.999 → 100,000s (capped)
            _mixing_budget = 100.0 / max((1.0 - rho) ** 2, 1e-12)
            _sim_time_crit = min(_mixing_budget, 100_000.0)
            if _sim_time_crit >= 100_000.0:
                log.warning(
                    f"  [!] rho={rho:.4f}: sim_time capped at 100,000s "
                    f"(theoretical mixing time ~ {_mixing_budget:.0f}s). "
                    f"E[Q] may still be underestimated. Report only rho<=0.999 "
                    f"and add mixing-time caveat."
                )

        max_samples_crit = int(_sim_time_crit / _STRESS_SAMPLE_INTERVAL) + 1

        log.info(f"  Simulating rho={rho:.3f} (T={_sim_time_crit})...")

        times, states, (arrs, deps) = sharded_replications(
            num_replications=cfg.simulation.num_replications,
            num_servers=N_fixed,
            arrival_rate=lam,
            service_rates=mu_fixed,
            alpha=cfg.system.alpha,
            sim_time=_sim_time_crit,
            sample_interval=_STRESS_SAMPLE_INTERVAL,
            base_seed=cfg.simulation.seed,
            max_samples=max_samples_crit,
            policy_type=3
        )

        q_totals = []
        stationary_count = 0

        total_q_trajectories = np.array(states).sum(axis=2)  # (Reps, TimeSteps)

        # SG#4 FIX: Mask zero-padded trailing entries from JAX pre-allocated buffers.
        # JAX states_buf is initialized with jnp.zeros; unfilled slots appear as Q=0.
        # MSER-5 interprets trailing zeros as a downward trend and inflates truncation.
        # Clip each trajectory to its valid length using times_buf > 0 as a mask.
        for r in range(cfg.simulation.num_replications):
            _valid_mask = np.array(times[r]) > 0
            _valid_mask[0] = True  # t=0 initial snapshot is always valid
            _valid_len = int(np.sum(_valid_mask))
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
            res = SimResult(
                np.array(times[r]), np.array(states[r]),
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
    mu_het = jnp.array([10.0, 0.1, 0.1, 0.1])
    cap_het = float(jnp.sum(mu_het))
    lam_het = 0.5 * cap_het  # 50% load

    log.info(f"  Simulating heterogenous setup: mu={mu_het}")

    # PATCH BUG-3 (two fixes):
    #   Fix A — derive sim_time into a named variable so max_samples stays consistent.
    #   Fix B — use _STRESS_SAMPLE_INTERVAL instead of cfg.simulation.sample_interval.
    # old:  sim_time=... (conditional)
    #       max_samples=int(10000.0 / cfg.simulation.sample_interval) + 1  ← hardcoded 10000!
    # new:  _sim_time_het = ... (same conditional)
    #       max_samples=int(_sim_time_het / _STRESS_SAMPLE_INTERVAL) + 1   ← consistent
    # PATCH 2026-03-14: reduced from 10000→1000 (debug: 1000→500).
    _sim_time_het = 500.0 if raw_cfg.get("debug", False) else 1000.0
    max_samples_het = int(_sim_time_het / _STRESS_SAMPLE_INTERVAL) + 1    # PATCH

    times, states, (arrs, deps) = sharded_replications(
        num_replications=cfg.simulation.num_replications,
        num_servers=4,
        arrival_rate=lam_het,
        service_rates=mu_het,
        alpha=cfg.system.alpha,
        sim_time=_sim_time_het,
        sample_interval=_STRESS_SAMPLE_INTERVAL,    # PATCH: was cfg.simulation.sample_interval
        base_seed=cfg.simulation.seed,
        max_samples=max_samples_het,               # PATCH: consistent with _sim_time_het
        policy_type=3
    )

    work_dist = []
    for r in range(cfg.simulation.num_replications):
        res = SimResult(
            np.array(times[r]), np.array(states[r]),
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

    append_metrics_jsonl(
        {
            "test": "heterogeneity",
            "gini": float(gini_coefficient(mean_dist)),
            "mean_dist": [float(x) for x in mean_dist]
        },
        out_dir / "metrics.jsonl"
    )

    log.info("\nStress test complete.")


if __name__ == "__main__":
    main()
