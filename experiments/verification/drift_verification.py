"""
Experiment: Drift Verification (Experiment 2)

- Grid evaluation for N=2: outputs heatmap and scatter plot.
- Trajectory evaluation for N>2: outputs scatter plot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import hydra
from omegaconf import DictConfig

from gibbsq.core.config import hydra_to_config, validate, drift_constant_R, drift_rate_epsilon
from gibbsq.core.policies import make_policy
from gibbsq.engines.numpy_engine import simulate
from gibbsq.core.drift import evaluate_grid, evaluate_trajectory
from gibbsq.analysis.plotting import plot_drift_landscape, plot_drift_vs_norm
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import setup_wandb, get_run_config

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="small")
def main(raw_cfg: DictConfig) -> None:
    # 1. Parse and validate
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    # Initialize Run Capsule (Dynamic Directory + Config Persistence)
    run_dir, run_id = get_run_config(cfg, "drift_verification", raw_cfg)

    # Initialize WandB via centralized utility
    run = setup_wandb(cfg, raw_cfg, default_group="drift_verification", run_id=run_id, run_dir=run_dir)

    # Use the isolated Run Directory for all outputs
    out_dir = run_dir

    sc = cfg.system
    N = sc.num_servers
    lam = sc.arrival_rate
    mu = sc.service_rates
    alpha = sc.alpha

    # Derived proof constants
    R = drift_constant_R(cfg)
    eps = drift_rate_epsilon(cfg)
    log.info(f"System: N={N}, lam={lam}, alpha={alpha}, cap={sum(mu):.4f}")
    log.info(f"Proof bounds: R={R:.4f}, eps={eps:.6f}")

    if N <= 3 and cfg.drift.use_grid:
        log.info(f"--- Grid Evaluation (q_max={cfg.drift.q_max}) ---")
        res = evaluate_grid(lam, mu, alpha, q_max=cfg.drift.q_max)
        
        log.info(f"States evaluated: {len(res.states):,}")
        log.info(f"Bound violations: {res.violations:,}")
        
        if res.violations > 0:
            log.error("VIOLATIONS DETECTED: Exact drift exceeds analytical upper bound.")

        if N == 2:
            f1 = out_dir / "drift_heatmap.png"
            plot_drift_landscape(res, alpha, save_path=f1)
            log.info(f"Saved: {f1}")

        f2 = out_dir / "drift_vs_norm.png"
        plot_drift_vs_norm(res, eps, R, save_path=f2)
        log.info(f"Saved: {f2}")
        
        if run:
            run.log({
                "drift_heatmap": wandb.Image(str(f1)) if N == 2 else None,
                "drift_vs_norm": wandb.Image(str(f2))
            })

        # Persist metrics locally for capsule integrity
        append_metrics_jsonl({
            "num_servers": int(N),
            "arrival_rate": float(lam),
            "alpha": float(alpha),
            "R": float(R),
            "epsilon": float(eps),
            "violations": int(res.violations),
            "states_evaluated": int(len(res.states))
        }, out_dir / "metrics.jsonl")

    else:
        log.info("--- Trajectory Evaluation ---")
        policy = make_policy("softmax", alpha=alpha)
        
        sim_res = simulate(
            num_servers=N,
            arrival_rate=lam,
            service_rates=mu,
            policy=policy,
            sim_time=cfg.simulation.ssa.sim_time,
            sample_interval=cfg.simulation.ssa.sample_interval,
            log_interval=cfg.simulation.ssa.sim_time / 10.0,
            rng=np.random.default_rng(cfg.simulation.seed),
        )
        log.info(f"Simulation done: {sim_res.arrival_count:,} arrivals.")

        res = evaluate_trajectory(sim_res.states, lam, mu, alpha)
        log.info(f"States evaluated: {len(res.states):,}")
        log.info(f"Bound violations: {res.violations:,}")

        f = out_dir / "drift_vs_norm.png"
        plot_drift_vs_norm(res, eps, R, save_path=f)
        log.info(f"Saved: {f}")
        
        if run:
            run.log({"drift_vs_norm": wandb.Image(str(f))})

        # Persist metrics locally for capsule integrity
        append_metrics_jsonl({
            "num_servers": int(N),
            "arrival_rate": float(lam),
            "alpha": float(alpha),
            "R": float(R),
            "epsilon": float(eps),
            "violations": int(res.violations),
            "states_evaluated": int(len(res.states))
        }, out_dir / "metrics.jsonl")

    log.info("Drift verification complete.")
    
    if run:
        run.finish()


if __name__ == "__main__":
    main()
