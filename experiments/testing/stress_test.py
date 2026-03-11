"""
Stress test: GibbsQ scaling, critical load, and heterogeneity.

Tests:
1. Massive-N Scaling: N=1024 servers.
2. Critical Load: rho=0.999.
3. Extreme Heterogeneity: 100x variance in service rates.

Uses JAX vmap for parallel replications.
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

# --- Stress Test Regimes ---
# Arrays now pulled dynamically from Hydra configs (e.g., cfg.stress)


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

    # 1. MASSIVE-N SCALING TEST
    log.info("\n[TEST 1] Massive-N Scaling Analysis")
    n_targets = [cfg.stress.n_values[0]] if raw_cfg.get("debug", False) else cfg.stress.n_values
    for N in n_targets:
        # Construct parameters for N experts
        mu = jnp.ones(N) * 2.0  # Normalized service rate
        lam = 0.8 * float(jnp.sum(mu)) # Fixed rho=0.8
        
        sim_time = 5000.0  # Reduced for scaling tests
        max_samples = int(sim_time / cfg.simulation.sample_interval) + 1
        
        log.info(f"  Simulating N={N} experts (rho=0.8)...")
        
        # JAX backend
        times, states, (arrs, deps) = sharded_replications(
            num_replications=cfg.simulation.num_replications,
            num_servers=N,
            arrival_rate=lam,
            service_rates=mu,
            alpha=1.0,
            sim_time=sim_time,
            sample_interval=cfg.simulation.sample_interval,
            base_seed=cfg.simulation.seed,
            max_samples=max_samples,
            policy_type=3 # Softmax
        )
        
        # Calculate aggregate Gini across replications
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
            avg_q = time_averaged_queue_lengths(res, 0.2)
            ginis.append(gini_coefficient(avg_q))
        
        avg_gini = np.mean(ginis)
        log.info(f"    -> Average Gini Imbalance: {avg_gini:.4f}")
        if wandb and wandb.run:
            wandb.log({"massive_n/N": N, "massive_n/avg_gini": avg_gini})
        
        append_metrics_jsonl(
            {"test": "massive_n", "N": int(N), "avg_gini": float(avg_gini)}, 
            out_dir / "metrics.jsonl"
        )

    # 2. CRITICAL LOAD ANALYSIS
    log.info("\n[TEST 2] Critical Load Analysis (rho up to 0.999)")
    N_fixed = 10
    mu_fixed = jnp.ones(N_fixed)
    cap_fixed = float(jnp.sum(mu_fixed))
    
    rho_targets = [cfg.stress.critical_rhos[0]] if raw_cfg.get("debug", False) else cfg.stress.critical_rhos
    for rho in rho_targets:
        lam = rho * cap_fixed
        # Near stability limits, we need longer horizons
        sim_time_critical = 1000.0 if raw_cfg.get("debug", False) else (50000.0 if rho > 0.99 else 10000.0)
        max_samples_crit = int(sim_time_critical / cfg.simulation.sample_interval) + 1
        
        log.info(f"  Simulating rho={rho:.3f} (T={sim_time_critical})...")
        
        times, states, (arrs, deps) = sharded_replications(
            num_replications=cfg.simulation.num_replications,
            num_servers=N_fixed,
            arrival_rate=lam,
            service_rates=mu_fixed,
            alpha=1.0,
            sim_time=sim_time_critical,
            sample_interval=cfg.simulation.sample_interval,
            base_seed=cfg.simulation.seed,
            max_samples=max_samples_crit,
            policy_type=3
        )
        
        q_totals = []
        stationary_count = 0
        
        # Convergence diagnostics
        total_q_trajectories = np.array(states).sum(axis=2) # Shape: (Reps, TimeSteps)
        r_hat = gelman_rubin_diagnostic(total_q_trajectories)
        log.info(f"    -> Gelman-Rubin R-hat across replicas: {r_hat:.4f}")
        
        for r in range(cfg.simulation.num_replications):
            res = SimResult(np.array(times[r]), np.array(states[r]), int(arrs[r]), int(deps[r]), float(times[r][-1]), N_fixed)
            
            # Use MSER-5 to dynamically find initialization bias truncation instead of arbitrary 20%
            traj = total_q_trajectories[r]
            d_star = mser5_truncation(traj)
            trunc_fraction = d_star / len(traj)
            
            avg_q = time_averaged_queue_lengths(res, trunc_fraction).sum()
            q_totals.append(avg_q)
            
            diag = stationarity_diagnostic(res, burn_in_fraction=trunc_fraction)
            if diag["is_stationary"]:
                stationary_count += 1
        
        log.info(f"    -> Avg E[Q_total]: {np.mean(q_totals):.2f} | Stationarity: {stationary_count}/{cfg.simulation.num_replications}")
        if wandb and wandb.run:
            wandb.log({"critical_load/rho": rho, "critical_load/mean_q": np.mean(q_totals), "critical_load/stationary_rate": stationary_count/cfg.simulation.num_replications})
        
        append_metrics_jsonl(
            {
                "test": "critical_load", 
                "rho": float(rho), 
                "mean_q": float(np.mean(q_totals)), 
                "stationary_rate": float(stationary_count/cfg.simulation.num_replications)
            }, 
            out_dir / "metrics.jsonl"
        )

    # 3. EXTREME HETEROGENEITY TEST
    log.info("\n[TEST 3] Extreme Heterogeneity Resilience (100x Speed Gap)")
    mu_het = jnp.array([10.0, 0.1, 0.1, 0.1]) # One expert is 100x faster than the peers
    cap_het = float(jnp.sum(mu_het))
    lam_het = 0.5 * cap_het # 50% load
    
    log.info(f"  Simulating heterogenous setup: mu={mu_het}")
    
    times, states, (arrs, deps) = sharded_replications(
        num_replications=cfg.simulation.num_replications,
        num_servers=4,
        arrival_rate=lam_het,
        service_rates=mu_het,
        alpha=1.0, # Softmax helps here
        sim_time=1000.0 if raw_cfg.get("debug", False) else 10000.0,
        sample_interval=cfg.simulation.sample_interval,
        base_seed=cfg.simulation.seed,
        max_samples=int(10000.0 / cfg.simulation.sample_interval) + 1,
        policy_type=3
    )
    
    # Analyze workload distribution
    # Under Soft-JSQ, the fast server should receive most arrivals
    work_dist = []
    for r in range(cfg.simulation.num_replications):
        res = SimResult(np.array(times[r]), np.array(states[r]), int(arrs[r]), int(deps[r]), float(times[r][-1]), 4)
        avg_q_per_srv = time_averaged_queue_lengths(res, 0.2)
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
