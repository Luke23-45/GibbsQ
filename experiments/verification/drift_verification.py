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

from gibbsq.core.config import load_experiment_config, drift_constant_R, drift_rate_epsilon
from gibbsq.core.builders import build_policy_by_name
from gibbsq.engines.numpy_engine import simulate
from gibbsq.core.drift import evaluate_grid, evaluate_trajectory
from gibbsq.analysis.plotting import plot_drift_landscape, plot_drift_vs_norm
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.progress import create_progress
from gibbsq.analysis.theme import apply_theme

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)


def _require_theorem_supported_policy(policy_name: str) -> str:
    theorem_modes = {
        "softmax": "raw",
        "uas": "uas",
    }
    try:
        return theorem_modes[policy_name]
    except KeyError as exc:
        raise ValueError(
            "drift_verification certifies only theorem-backed policy paths "
            "(policy.name in {'softmax', 'uas'}); "
            f"got policy.name='{policy_name}'."
        ) from exc


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "drift")

    run_dir, run_id = get_run_config(cfg, "drift", resolved_raw_cfg)

    run = setup_wandb(cfg, resolved_raw_cfg, default_group="drift_verification", run_id=run_id, run_dir=run_dir)

    apply_theme('publication')

    out_dir = run_dir

    sc = cfg.system
    N = sc.num_servers
    lam = sc.arrival_rate
    mu = sc.service_rates
    alpha = sc.alpha
    drift_mode = _require_theorem_supported_policy(cfg.policy.name)

    R = drift_constant_R(cfg)
    eps = drift_rate_epsilon(cfg)
    log.info(f"System: N={N}, lam={lam}, alpha={alpha}, cap={sum(mu):.4f}")
    log.info(f"Proof bounds: R={R:.4f}, eps={eps:.6f}")

    if N <= 3 and cfg.drift.use_grid:
        log.info(f"--- Grid Evaluation (q_max={cfg.drift.q_max}) ---")
        with create_progress(total=3, desc="drift: grid", unit="stage") as progress:
            progress.set_postfix({"N": N, "mode": "grid"}, refresh=False)
            res = evaluate_grid(lam, mu, alpha, q_max=cfg.drift.q_max, mode=drift_mode)
            progress.update(1)

            log.info(f"States evaluated: {len(res.states):,}")
            log.info(f"Bound violations: {res.violations:,}")

            if res.violations > 0:
                log.error(
                    f"VIOLATIONS DETECTED: {res.violations:,} states have exact drift "
                    f"exceeding the analytical upper bound. "
                    f"Foster-Lyapunov proof FAILS for alpha={alpha}, "
                    f"lam={lam}, cap={sum(mu):.4f}. All results are INVALID."
                )
                raise RuntimeError(
                    f"Proof violation at {res.violations} states. "
                    "Halt: paper results cannot be generated from an invalid proof basis."
                )


            if N == 2:
                f1 = out_dir / "drift_heatmap"
                plot_drift_landscape(res, alpha, save_path=f1, theme='publication', formats=['png', 'pdf'])
                log.info(f"Saved: {f1}.png, {f1}.pdf")

            f2 = out_dir / "drift_vs_norm"
            plot_drift_vs_norm(res, eps, R, save_path=f2, theme='publication', formats=['png', 'pdf'])
            log.info(f"Saved: {f2}.png, {f2}.pdf")
            progress.update(1)

            if run:
                png_path = str(out_dir / "drift_heatmap.png") if N == 2 else None
                run.log({
                    "drift_heatmap": wandb.Image(png_path) if png_path else None,
                    "drift_vs_norm": wandb.Image(str(out_dir / "drift_vs_norm.png"))
                })

            append_metrics_jsonl({
                "num_servers": int(N),
                "arrival_rate": float(lam),
                "alpha": float(alpha),
                "R": float(R),
                "epsilon": float(eps),
                "violations": int(res.violations),
                "states_evaluated": int(len(res.states))
            }, out_dir / "metrics.jsonl")
            progress.update(1)

    else:
        log.info("--- Trajectory Evaluation ---")
        policy = build_policy_by_name(cfg.policy.name, alpha=alpha, mu=np.asarray(mu, dtype=np.float64))
        with create_progress(total=3, desc="drift: trajectory", unit="stage") as progress:
            progress.set_postfix({"N": N, "mode": "trajectory"}, refresh=False)
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
            progress.update(1)
            log.info(f"Simulation done: {sim_res.arrival_count:,} arrivals.")

            res = evaluate_trajectory(sim_res.states, lam, mu, alpha, mode=drift_mode)
            progress.update(1)
            log.info(f"States evaluated: {len(res.states):,}")
            log.info(f"Bound violations: {res.violations:,}")

            f = out_dir / "drift_vs_norm"
            plot_drift_vs_norm(res, eps, R, save_path=f, theme='publication', formats=['png', 'pdf'])
            log.info(f"Saved: {f}.png, {f}.pdf")

            if run:
                run.log({"drift_vs_norm": wandb.Image(str(out_dir / "drift_vs_norm.png"))})

            append_metrics_jsonl({
                "num_servers": int(N),
                "arrival_rate": float(lam),
                "alpha": float(alpha),
                "R": float(R),
                "epsilon": float(eps),
                "violations": int(res.violations),
                "states_evaluated": int(len(res.states))
            }, out_dir / "metrics.jsonl")
            progress.update(1)

    log.info("Drift verification complete.")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
