import logging
import time
import hydra
import numpy as np
import jax.numpy as jnp
from omegaconf import DictConfig

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.core.policies import make_policy
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.engines.jax_engine import run_replications_jax
from gibbsq.analysis.metrics import time_averaged_queue_lengths, gini_coefficient, sojourn_time_estimate
from gibbsq.analysis.plotting import plot_policy_comparison
from gibbsq.utils.exporter import save_trajectory_parquet, append_metrics_jsonl
from gibbsq.utils.logging import setup_wandb, get_run_config

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)

POLICIES = [
    {"name": "uniform",      "label": "Uniform (1/N)",        "jax_idx": 0},
    {"name": "proportional", "label": "Proportional (mu/cap)", "jax_idx": 1},
    {"name": "jsq",          "label": "JSQ (Exact Min)",      "jax_idx": 2},
    {"name": "power_of_d",   "label": "Power-of-d (d=2)",     "jax_idx": 4, "d": 2}, # Index 4 maps to Power-of-d in jax_engine.py
    # Intentional alpha sweep for softmax policies (independent of cfg.system.alpha)
    {"name": "softmax",      "label": "Softmax (alpha=0.1)",  "jax_idx": 3, "alpha": 0.1},
    {"name": "softmax",      "label": "Softmax (alpha=1.0)",  "jax_idx": 3, "alpha": 1.0},
    {"name": "softmax",      "label": "Softmax (alpha=10.0)", "jax_idx": 3, "alpha": 10.0},
]


def _iter_with_progress(items, desc: str, total: int | None = None):
    """Return iterable wrapped with tqdm when available."""
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, total=total, dynamic_ncols=True, leave=False)


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    # Initialize Run Capsule (Dynamic Directory + Config Persistence)
    run_dir, run_id = get_run_config(cfg, "policy_comparison", raw_cfg)

    # Initialize WandB via centralized utility
    run = setup_wandb(cfg, raw_cfg, default_group="policy_comparison", run_id=run_id, run_dir=run_dir)

    sc = cfg.system
    N = sc.num_servers
    mu = np.asarray(sc.service_rates, dtype=np.float64)
    mu_jax = jnp.asarray(mu)
    cap = float(mu.sum())
    rho = sc.arrival_rate / cap

    # Use the isolated Run Directory for all outputs
    out_dir = run_dir
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    log.info(f"System: N={N}, lam={sc.arrival_rate:.4f}, rho={rho:.4f} | Backend: {'JAX' if cfg.jax.enabled else 'NumPy'}")
    log.info(f"Comparing {len(POLICIES)} policies across {cfg.simulation.num_replications} reps.")

    q_res: dict[str, list[float]] = {}
    gini_res: dict[str, list[float]] = {}
    sojourn_res: dict[str, list[float]] = {}

    for p in _iter_with_progress(POLICIES, desc="Policies", total=len(POLICIES)):
        lbl = p["label"]
        log.info(f"\n--- {lbl} ---")
        
        q_res[lbl] = []
        gini_res[lbl] = []
        sojourn_res[lbl] = []

        # Selective use of JAX backend (all policies 0-4 now supported)
        use_jax = cfg.jax.enabled
        last_res = None   # initialise before if/else so export guard is always defined
        
        if use_jax:
            # --- JAX backend ---
            max_samples = int(cfg.simulation.ssa.sim_time / cfg.simulation.ssa.sample_interval) + 1
            t_sim_start = time.perf_counter()
            times, states, (arrs, deps) = run_replications_jax(
                num_replications=cfg.simulation.num_replications,
                num_servers=N,
                arrival_rate=sc.arrival_rate,
                service_rates=mu_jax,
                alpha=float(p.get("alpha", 1.0)),
                sim_time=cfg.simulation.ssa.sim_time,
                sample_interval=cfg.simulation.ssa.sample_interval,
                base_seed=cfg.simulation.seed,
                max_samples=max_samples,
                policy_type=p["jax_idx"],
                d=p.get("d", 2)
            )
            for r in _iter_with_progress(range(cfg.simulation.num_replications), desc=f"{lbl} reps", total=cfg.simulation.num_replications):
                res = SimResult(
                    times=np.array(times[r]),
                    states=np.array(states[r]),
                    arrival_count=int(arrs[r]),
                    departure_count=int(deps[r]),
                    final_time=float(times[r][-1]),
                    num_servers=N
                )
                avg_q = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
                q_res[lbl].append(float(avg_q.sum()))
                gini_res[lbl].append(gini_coefficient(avg_q))
                sojourn_res[lbl].append(sojourn_time_estimate(res, sc.arrival_rate, cfg.simulation.burn_in_fraction))
                last_res = res
            log.info("  timing[%s]: jax_total=%.2fs", lbl, time.perf_counter() - t_sim_start)
        else:
            # --- STANDARD NUMPY EXECUTION ---
            # Compute a safe dynamic max_events ceiling that mirrors the JAX engine
            # formula: int(max_theoretical_rate * sim_time * 1.5) + 1000.
            # The static cfg.simulation.ssa.max_events (100 000) is below the
            # expected event count for the default config (~117 000), causing silent
            # trajectory truncation and biased E[Q] estimates.
            _np_max_events = int(
                (sc.arrival_rate + float(mu.sum()))
                * cfg.simulation.ssa.sim_time
                * 1.5
            ) + 1000
            t_np_start = time.perf_counter()
            for rep in _iter_with_progress(range(cfg.simulation.num_replications), desc=f"{lbl} reps", total=cfg.simulation.num_replications):
                rng = np.random.default_rng(cfg.simulation.seed + rep)
                policy = make_policy(
                    p["name"],
                    alpha=p.get("alpha", 1.0),
                    mu=mu,
                    d=p.get("d", 2),
                )
                res = simulate(
                    num_servers=N,
                    arrival_rate=sc.arrival_rate,
                    service_rates=mu,
                    policy=policy,
                    sim_time=cfg.simulation.ssa.sim_time,
                    sample_interval=cfg.simulation.ssa.sample_interval,
                    rng=rng,
                    max_events=_np_max_events,
                )
                avg_q = time_averaged_queue_lengths(res, cfg.simulation.burn_in_fraction)
                q_res[lbl].append(float(avg_q.sum()))
                gini_res[lbl].append(gini_coefficient(avg_q))
                sojourn_res[lbl].append(sojourn_time_estimate(res, sc.arrival_rate, cfg.simulation.burn_in_fraction))
                last_res = res
            log.info("  timing[%s]: numpy_total=%.2fs", lbl, time.perf_counter() - t_np_start)

        m_q = np.mean(q_res[lbl])
        se_q = np.std(q_res[lbl]) / np.sqrt(cfg.simulation.num_replications)
        m_g = np.mean(gini_res[lbl])
        m_w = np.mean(sojourn_res[lbl])
        log.info(f"  E[Q_total] = {m_q:8.2f} +/- {se_q:5.2f}  |  Gini = {m_g:.4f}  |  E[W] = {m_w:.4f}")
        
        metrics = {
            "policy": p["name"],
            "label": lbl,
            "mean_q_total": float(m_q),
            "se_q_total": float(se_q),
            "mean_gini": float(m_g),
            "mean_sojourn": float(m_w),
        }
        append_metrics_jsonl(metrics, out_dir / "metrics.jsonl")
        if run:
            label_slug = lbl.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            run.log({f"policy_{label_slug}/{k}": v for k, v in metrics.items() if k not in ["policy", "label"]})
        
        # Conditionally export the last trajectory for this policy
        if cfg.simulation.export_trajectories and last_res is not None:
            fname = out_dir / f"trajectories/{p['name']}_alpha{p.get('alpha', 1.0)}.parquet"
            save_trajectory_parquet(last_res, fname)

    # Plots
    plot_policy_comparison(q_res, "Expected Total Queue Length E[Q_total]", out_dir / "qtotal_compare.png")
    plot_policy_comparison(gini_res, "Gini Coefficient (Imbalance)", out_dir / "gini_compare.png")
    
    if run:
        run.log({
            "q_total_comparison": wandb.Image(str(out_dir / "qtotal_compare.png")),
            "gini_comparison": wandb.Image(str(out_dir / "gini_compare.png"))
        })
        run.finish()

    log.info("\nComparison complete. Plots saved to output directory.")


if __name__ == "__main__":
    main()
