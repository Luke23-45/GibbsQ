import logging
import time
import hydra
import warnings
import numpy as np
import jax.numpy as jnp
from omegaconf import DictConfig

from gibbsq.core.config import hydra_to_config, validate
from gibbsq.core.policies import make_policy
from gibbsq.engines.numpy_engine import simulate, SimResult
from gibbsq.engines.numpy_engine import run_replications
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

# SG#7 FIX: Ground-truth mapping from policy name to JAX engine policy_type index.
# This is the single authoritative source; the POLICIES list is validated against it.
_POLICY_NAME_TO_JAX_IDX: dict[str, int] = {
    "uniform":     0,
    "proportional": 1,
    "jsq":         2,
    "softmax":     3,
    "power_of_d":  4,
}

# SG#1 FIX: Adapter that wraps a trained NeuralRouter for the NumPy SSA engine.
# The NumPy engine's RoutingPolicy protocol requires __call__(Q, rng) -> probs.
# This is the ONLY correct way to evaluate the neural router on the true CTMC:
# train on the DGA surrogate, but measure steady-state performance on the SSA.
class _NeuralSSAPolicy:
    """
    Bridges NeuralRouter → NumPy SSA engine for true-CTMC evaluation.

    Uses an LRU cache and JIT-compiled inference to mitigate the heavy
    dispatch overhead of calling JAX inside a Python simulation loop.
    """
    def __init__(self, model: "NeuralRouter") -> None:
        import jax
        import equinox as eqx
        import functools
        import numpy as np
        
        self._model = model
        
        @eqx.filter_jit
        def _forward(m, x):
            return jax.nn.softmax(m(x))
            
        self._forward = _forward
        
        @functools.lru_cache(maxsize=131072)
        def _get_probs(q_tuple):
            import jax.numpy as jnp
            Q_jax = jnp.array(q_tuple, dtype=jnp.float32)
            probs = self._forward(self._model, Q_jax)
            probs_np = np.array(probs, dtype=np.float64)
            return probs_np / probs_np.sum()
            
        self._get_probs = _get_probs

    def __call__(self, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self._get_probs(tuple(Q))

POLICIES = [
    {"name": "uniform",      "label": "Uniform (1/N)",        "jax_idx": 0},
    {"name": "proportional", "label": "Proportional (mu/cap)", "jax_idx": 1},
    {"name": "jsq",          "label": "JSQ (Exact Min)",      "jax_idx": 2},
    {"name": "power_of_d",   "label": "Power-of-d (d=2)",     "jax_idx": 4, "d": 2},
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


def _compute_metrics_from_arrays(times, states, arrs, deps, num_servers, arrival_rate, burn_in_fraction):
    """Helper for legacy test compatibility."""
    q_vals, g_vals, w_vals = [], [], []
    last_res = None
    for r in range(len(times)):
        res = SimResult(
            times=np.array(times[r]),
            states=np.array(states[r]),
            arrival_count=int(arrs[r]),
            departure_count=int(deps[r]),
            final_time=float(times[r][-1]),
            num_servers=num_servers
        )
        avg_q = time_averaged_queue_lengths(res, burn_in_fraction)
        q_vals.append(float(avg_q.sum()))
        g_vals.append(float(gini_coefficient(avg_q)))
        w_vals.append(float(sojourn_time_estimate(res, arrival_rate, burn_in_fraction)))
        last_res = res
    return q_vals, g_vals, w_vals, last_res


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    # SG#7 FIX: Validate that every entry in POLICIES has consistent name↔jax_idx.
    for _p in POLICIES:
        _name, _idx = _p["name"], _p["jax_idx"]
        assert _name in _POLICY_NAME_TO_JAX_IDX, (
            f"Unknown policy name '{_name}' in POLICIES list."
        )
        assert _POLICY_NAME_TO_JAX_IDX[_name] == _idx, (
            f"POLICIES entry '{_p['label']}': name='{_name}' maps to "
            f"jax_idx={_POLICY_NAME_TO_JAX_IDX[_name]} but entry has "
            f"jax_idx={_idx}. Fix the POLICIES list."
        )

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

    # ─────────────────────────────────────────────────────────────────────
    # SG#1 FIX: True-CTMC (SSA) evaluation of the trained NeuralRouter.
    # This is the scientifically required comparison: train on the DGA
    # surrogate, but measure E[Q] on the true Gillespie SSA. Without this,
    # "N-GibbsQ parity with GibbsQ" is unmeasured on the real objective.
    # ─────────────────────────────────────────────────────────────────────
    _neural_ssa_label = "N-GibbsQ (SSA — true CTMC)"
    try:
        import equinox as eqx
        import jax as _jax
        from gibbsq.core.neural_policies import NeuralRouter
        from pathlib import Path as _Path

        _PROJECT_ROOT = _Path(__file__).resolve().parents[2]
        _ptr = _PROJECT_ROOT / "outputs" / "small" / "latest_weights.txt"

        if _ptr.exists():
            _ptr_content = _ptr.read_text(encoding="utf-8").strip()
            _ptr_raw = _Path(_ptr_content)
            # PR#1 FIX: pointer may be relative (new train.py) or absolute (legacy).
            _model_path = _ptr_raw if _ptr_raw.is_absolute() else (_PROJECT_ROOT / _ptr_raw)
            if _model_path.exists():
                _sk = NeuralRouter(
                    num_servers=N, config=cfg.neural,
                    key=_jax.random.PRNGKey(cfg.simulation.seed)
                )
                _neural_model = eqx.tree_deserialise_leaves(_model_path, _sk)

                # SG-9 PATCH: Validate BOTH num_servers AND hidden_size.
                if _neural_model.layers[0].weight.shape[1] != N:
                    log.warning(
                        f"[SG-9] Neural model N-mismatch: model expects "
                        f"N={_neural_model.layers[0].weight.shape[1]}, system has N={N}. "
                        f"Skipping neural evaluation."
                    )
                elif _neural_model.layers[0].weight.shape[0] != cfg.neural.hidden_size:
                    log.warning(
                        f"[SG-9] Neural model hidden_size mismatch: "
                        f"model={_neural_model.layers[0].weight.shape[0]}, "
                        f"config expects={cfg.neural.hidden_size}. "
                        f"Skipping neural evaluation to avoid corrupt weights."
                    )
                else:
                    _neural_policy = _NeuralSSAPolicy(_neural_model)
                    _np_max_events = int(
                        (sc.arrival_rate + float(mu.sum()))
                        * cfg.simulation.ssa.sim_time * 1.5
                    ) + 1000

                    q_res[_neural_ssa_label] = []
                    gini_res[_neural_ssa_label] = []
                    sojourn_res[_neural_ssa_label] = []

                    log.info(f"\n--- {_neural_ssa_label} (NumPy SSA) ---")
                    for _rep in range(cfg.simulation.num_replications):
                        _rng = np.random.default_rng(cfg.simulation.seed + _rep)
                        _res = simulate(
                            num_servers=N, arrival_rate=sc.arrival_rate,
                            service_rates=mu, policy=_neural_policy,
                            sim_time=cfg.simulation.ssa.sim_time,
                            sample_interval=cfg.simulation.ssa.sample_interval,
                            rng=_rng, max_events=_np_max_events,
                        )
                        _avg_q = time_averaged_queue_lengths(_res, cfg.simulation.burn_in_fraction)
                        q_res[_neural_ssa_label].append(float(_avg_q.sum()))
                        gini_res[_neural_ssa_label].append(gini_coefficient(_avg_q))
                        sojourn_res[_neural_ssa_label].append(
                            sojourn_time_estimate(_res, sc.arrival_rate, cfg.simulation.burn_in_fraction)
                        )

                    _m_q = np.mean(q_res[_neural_ssa_label])
                    _se_q = np.std(q_res[_neural_ssa_label]) / np.sqrt(cfg.simulation.num_replications)
                    log.info(
                        f"  E[Q_total] = {_m_q:8.2f} +/- {_se_q:5.2f}  "
                        f"(SSA; {cfg.simulation.num_replications} reps)"
                    )
                    metrics = {
                        "policy": "neural_ssa",
                        "label": _neural_ssa_label,
                        "mean_q_total": float(_m_q),
                        "se_q_total": float(_se_q),
                    }
                    append_metrics_jsonl(metrics, out_dir / "metrics.jsonl")
                    # Re-generate plots to include neural entry
                    plot_policy_comparison(
                        q_res, "Expected Total Queue Length E[Q_total] (incl. N-GibbsQ SSA)",
                        out_dir / "qtotal_compare.png"
                    )
            else:
                log.warning(
                    "[SG#1] INCOMPLETE RESULTS: Neural weight file not found — "
                    "SSA neural eval SKIPPED. Parity figures absent from this run. "
                    "Run: python -m experiments.n_gibbsq.train first."
                )
        else:
            log.warning(
                "[SG#1] INCOMPLETE RESULTS: No trained NeuralRouter found — "
                "SSA neural eval SKIPPED. Parity figures absent from this run. "
                "Run: python -m experiments.n_gibbsq.train first."
            )
    except ImportError as _e:
        log.warning(f"[SG#1] Could not load neural policy ({_e}). Skipping SSA neural eval.")

    if run:
        run.log({
            "q_total_comparison": wandb.Image(str(out_dir / "qtotal_compare.png")),
            "gini_comparison": wandb.Image(str(out_dir / "gini_compare.png"))
        })
        run.finish()

    log.info("\nComparison complete. Plots saved to output directory.")


if __name__ == "__main__":
    main()
