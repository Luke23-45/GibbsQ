"""
Corrected Policy Comparison for N-GibbsQ.

This module implements the corrected baseline hierarchy for policy comparison,
addressing Smoking Gun #3 (Critical Load Advantage Is Baseline Collapse) and
Smoking Gun #4 (Parity Benchmark Is Self-Referential).

The corrected baseline hierarchy:
- Tier 1: Blind policies (Uniform, Proportional)
- Tier 2: Queue-length-based (JSQ, Power-of-d)
- Tier 3: Sojourn-time-based (JSSQ, UAS)
- Tier 4: Fixed-weight baselines (Proportional, Uniform)
- Tier 5: Neural (N-GibbsQ trained with REINFORCE)

The parity criterion is now: N-GibbsQ must match or beat Tier 3 (JSSQ),
not the fixed-weight Tier 4 baselines.
"""

import logging
from pathlib import Path
import numpy as np
import jax
from omegaconf import DictConfig
from gibbsq.core.config import ExperimentConfig, hydra_to_config, validate
from gibbsq.core.builders import build_policy_by_name
from gibbsq.utils.model_io import build_neural_eval_policy, resolve_model_pointer
from gibbsq.engines.numpy_engine import simulate, run_replications, SimResult
from gibbsq.analysis.metrics import (
    time_averaged_queue_lengths, gini_coefficient, sojourn_time_estimate
)
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.progress import iter_progress
from gibbsq.analysis.theme import apply_theme, THEMES
from gibbsq.utils.chart_exporter import save_chart
import pandas as pd
import matplotlib.pyplot as plt
import sys
import equinox as eqx
from gibbsq.core.neural_policies import NeuralRouter

log = logging.getLogger(__name__)
NEURAL_EVAL_MODE = "deterministic"


def _iter_with_progress(items, **kwargs):
    """Compatibility wrapper for optional progress bars."""
    return iter_progress(items, **kwargs)


def _standard_error(values) -> float:
    """Return sample standard error, guarding the single-observation case."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _compute_metrics_from_arrays(
    times: np.ndarray,
    states: np.ndarray,
    arrs: np.ndarray,
    deps: np.ndarray,
    num_servers: int,
    arrival_rate: float,
    burn_in_fraction: float,
):
    """Compatibility helper for array-based policy-comparison tests."""
    q_vals, g_vals, w_vals = [], [], []
    last_res = None
    for r in range(times.shape[0]):
        np_times = np.asarray(times[r])
        np_states = np.asarray(states[r])
        valid_mask = np_times > 0
        valid_mask[0] = True
        valid_len = int(np.sum(valid_mask))
        last_res = SimResult(
            times=np_times[:valid_len],
            states=np_states[:valid_len],
            arrival_count=int(arrs[r]),
            departure_count=int(deps[r]),
            final_time=float(np_times[valid_len - 1]),
            num_servers=num_servers,
        )
        avg_q = time_averaged_queue_lengths(last_res, burn_in_fraction)
        q_vals.append(float(avg_q.sum()))
        g_vals.append(float(gini_coefficient(avg_q)))
        w_vals.append(float(sojourn_time_estimate(last_res, arrival_rate, burn_in_fraction)))

    return np.asarray(q_vals), np.asarray(g_vals), np.asarray(w_vals), last_res


CORRECTED_POLICIES = [
    {"tier": 2, "name": "jsq", "label": "JSQ (Min Queue)", "requires_mu": False},
    {"tier": 2, "name": "jssq", "label": "JSSQ (Min Sojourn)", "requires_mu": True},

    {"tier": 3, "name": "uas", "label": "UAS (alpha=1.0)", "requires_mu": True, "alpha": 1.0},
    {"tier": 3, "name": "uas", "label": "UAS (alpha=10.0)", "requires_mu": True, "alpha": 10.0},
    {"tier": 3, "name": "uas", "label": "UAS (alpha=5.0)", "requires_mu": True, "alpha": 5.0},

    # Tier 4: Fixed-Weight Baselines (Blind Policies)
    {"tier": 4, "name": "proportional", "label": "Proportional (mu/Lambda)", "requires_mu": True},
    {"tier": 4, "name": "uniform", "label": "Uniform (1/N)", "requires_mu": False},

    # Tier 5: Neural Policies (evaluated separately)
]


# make_corrected_policy deprecated: using build_policy_by_name from Registry


def evaluate_single_policy(
    policy,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> dict:
    """Evaluate a single policy via SSA simulation."""
    burn_in = cfg.simulation.burn_in_fraction
    results = run_replications(
        num_servers=cfg.system.num_servers,
        arrival_rate=cfg.system.arrival_rate,
        service_rates=np.array(cfg.system.service_rates),
        policy=policy,
        num_replications=cfg.simulation.num_replications,
        sim_time=cfg.simulation.ssa.sim_time,
        sample_interval=cfg.simulation.ssa.sample_interval,
        base_seed=cfg.simulation.seed,
        progress_desc=f"policy eval ({type(policy).__name__})",
    )

    q_totals = [time_averaged_queue_lengths(r, burn_in).sum()
                for r in results]
    ginis = [gini_coefficient(time_averaged_queue_lengths(r, burn_in))
             for r in results]
    sojourns = [sojourn_time_estimate(r, cfg.system.arrival_rate, burn_in)
                for r in results]

    return {
        "mean_q_total": float(np.mean(q_totals)),
        "se_q_total": _standard_error(q_totals),
        "mean_gini": float(np.mean(ginis)),
        "se_gini": _standard_error(ginis),
        "mean_sojourn": float(np.mean(sojourns)),
        "se_sojourn": _standard_error(sojourns),
    }


def run_corrected_comparison(
    cfg: ExperimentConfig,
    run_dir: Path,
    run_logger=None,
):
    """Run the corrected policy comparison."""
    log.info("=" * 60)
    log.info("  Corrected Policy Comparison")
    log.info("=" * 60)

    sc = cfg.system
    N = sc.num_servers
    mu = np.asarray(sc.service_rates, dtype=np.float64)
    cap = float(mu.sum())
    rho = sc.arrival_rate / cap

    log.info(f"System: N={N}, lambda={sc.arrival_rate:.4f}, Lambda={cap:.4f}, rho={rho:.4f}")
    log.info("-" * 60)

    results = {}

    for entry in _iter_with_progress(
        CORRECTED_POLICIES,
        total=len(CORRECTED_POLICIES),
        desc="policy: tiers",
        unit="policy",
    ):
        tier = entry["tier"]
        name = entry["name"]
        label = entry["label"]

        log.info(f"Evaluating Tier {tier}: {label}...")

        kwargs = {}
        if entry.get("requires_mu", False):
            kwargs["mu"] = mu
        if "alpha" in entry:
            kwargs["alpha"] = entry["alpha"]
        if "d" in entry:
            kwargs["d"] = entry["d"]

        policy = build_policy_by_name(entry["name"], **kwargs)

        rng = np.random.default_rng(cfg.simulation.seed)
        metrics = evaluate_single_policy(policy, cfg, rng)

        results[label] = {
            "tier": tier,
            "name": name,
            **metrics,
        }

        log.info(f"  E[Q_total] = {metrics['mean_q_total']:.4f} ± {metrics['se_q_total']:.4f}")

        append_metrics_jsonl({
            "policy": label,
            "tier": tier,
            **metrics,
        }, run_dir / "corrected_comparison_metrics.jsonl")

    # Resolve pointer directory from active run output layout
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # run_dir = output_dir / experiment_type / run_id, so run_dir.parent.parent = output_dir
    pointer_dir = run_dir.parent.parent

    model_path = resolve_model_pointer(
        _PROJECT_ROOT,
        pointer_dir,
        allow_bc=False,
        allow_legacy=False,
    )
    log.info("\nEvaluating Tier 5: N-GibbsQ (REINFORCE trained)...")
    log.info(f"Using neural weights from {model_path}")

    key = jax.random.PRNGKey(cfg.simulation.seed)
    policy_net = NeuralRouter(num_servers=N, config=cfg.neural, service_rates=mu, key=key)
    policy_net = eqx.tree_deserialise_leaves(model_path, policy_net)

    from gibbsq.utils.model_io import validate_neural_model_shape

    validate_neural_model_shape(policy_net, cfg.neural, N)

    neural_policy = build_neural_eval_policy(
        policy_net,
        mu,
        rho=rho,
        mode=NEURAL_EVAL_MODE,
    )

    log.info(f"Evaluating N-GibbsQ ({NEURAL_EVAL_MODE})...")
    metrics = evaluate_single_policy(neural_policy, cfg, np.random.default_rng(cfg.simulation.seed))

    results["N-GibbsQ (Platinum)"] = {
        "tier": 5,
        "name": f"neural_platinum_{NEURAL_EVAL_MODE}",
        "eval_mode": NEURAL_EVAL_MODE,
        **metrics,
    }

    log.info(f"  E[Q_total] = {metrics['mean_q_total']:.4f} ± {metrics['se_q_total']:.4f}")

    # Compute parity result using corrected tiered criteria
    log.info("\n" + "=" * 60)
    log.info("  Parity Analysis (Corrected Criteria)")
    log.info("=" * 60)

    # Professor's spec at suggestions.md:547-551:
    # GOLD: N-GibbsQ E[Q] ≤ JSSQ E[Q] (matches asymptotic optimum)
    # SILVER: N-GibbsQ E[Q] ≤ fixed-alpha UAS E[Q] (matches empirical softmax baseline)
    # BRONZE: N-GibbsQ E[Q] ≤ Proportional E[Q] (exceeds static baseline)
    # FAILED: N-GibbsQ E[Q] > Proportional E[Q]

    jssq_result = results.get("JSSQ (Min Sojourn)")
    sojourn_result = results.get("UAS (alpha=1.0)")
    proportional_result = results.get("Proportional (mu/Lambda)")

    neural_result = results.get("N-GibbsQ (Platinum)")

    if neural_result:
        neural_q = neural_result["mean_q_total"]
        log.info(f"N-GibbsQ (Platinum/Greedy): E[Q] = {neural_q:.4f}")

        jssq_q = jssq_result["mean_q_total"] if jssq_result else float('inf')
        sojourn_q = sojourn_result["mean_q_total"] if sojourn_result else float('inf')
        proportional_q = proportional_result["mean_q_total"] if proportional_result else float('inf')

        log.info(f"Reference thresholds:")
        log.info(f"  JSSQ (Tier 2): E[Q] = {jssq_q:.4f}")
        log.info(f"  UAS (Tier 3): E[Q] = {sojourn_q:.4f}")
        log.info(f"  Proportional (Tier 4): E[Q] = {proportional_q:.4f}")

        def has_parity(q_agent, se_agent, q_base, se_base):
            # Parity applies if the Agent performs better OR is within the statistical Margin of Error
            # We calculate a Confidence limit (configurable via parity_z_score) for the difference of two means
            margin_of_error = cfg.verification.parity_z_score * np.sqrt(se_agent**2 + se_base**2)
            return q_agent <= (q_base + margin_of_error)

        se_neural = neural_result["se_q_total"]

        jssq_q = jssq_result["mean_q_total"] if jssq_result else float('inf')
        jssq_se = jssq_result["se_q_total"] if jssq_result else 0.0

        sojourn_q = sojourn_result["mean_q_total"] if sojourn_result else float('inf')
        sojourn_se = sojourn_result["se_q_total"] if sojourn_result else 0.0

        proportional_q = proportional_result["mean_q_total"] if proportional_result else float('inf')
        proportional_se = proportional_result["se_q_total"] if proportional_result else 0.0

        log.info(f"Reference statistical bounds (95% CI):")
        log.info(f"  JSSQ (Tier 2): E[Q] = {jssq_q:.4f} ± {jssq_se:.4f}")
        log.info(f"  UAS (Tier 3): E[Q] = {sojourn_q:.4f} ± {sojourn_se:.4f}")
        log.info(f"  Proportional (Tier 4): E[Q] = {proportional_q:.4f} ± {proportional_se:.4f}")

        if has_parity(neural_q, se_neural, jssq_q, jssq_se):
            parity = "GOLD"
            log.info(f"PARITY RESULT: GOLD [OK] (Statistically matches asymptotic optimum JSSQ)")
        elif has_parity(neural_q, se_neural, sojourn_q, sojourn_se):
            parity = "SILVER"
            log.info(f"PARITY RESULT: SILVER [OK] (Statistically matches empirical UAS baseline)")
        elif has_parity(neural_q, se_neural, proportional_q, proportional_se):
            parity = "BRONZE"
            log.info(f"PARITY RESULT: BRONZE [OK] (Statistically matches static Proportional baseline)")
        else:
            parity = "FAILED"
            log.info(f"PARITY RESULT: FAILED [FAIL] (Statistically inferior to benchmark baselines)")

        neural_result["parity"] = parity
    else:
        raise RuntimeError("N-GibbsQ evaluation is required for policy comparison parity analysis.")

    _generate_comparison_plot(results, run_dir)

    return results


def _generate_comparison_plot(results: dict, run_dir: Path):
    """Generate comparison bar chart with chart-type-aware styling."""
    from gibbsq.analysis.plotting import plot_tier_comparison_bars

    sorted_results = sorted(results.items(), key=lambda x: (x[1]["tier"], x[1]["mean_q_total"]))

    labels = [name for name, _ in sorted_results]
    q_values = [r["mean_q_total"] for _, r in sorted_results]
    q_errors = [r["se_q_total"] for _, r in sorted_results]
    tiers = [r["tier"] for _, r in sorted_results]

    plot_path = run_dir / "corrected_policy_comparison"
    fig = plot_tier_comparison_bars(
        labels=labels,
        q_values=q_values,
        q_errors=q_errors,
        tiers=tiers,
        save_path=plot_path,
        theme="publication",
        formats=["png", "pdf"]
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    log.info(f"Comparison plot saved to {plot_path}.png, {plot_path}.pdf")


def _build_grid_eval_policy(model, service_rates: np.ndarray, rho: float):
    """Build a neural SSA policy for one grid cell using that cell's load factor."""
    return build_neural_eval_policy(
        model,
        service_rates,
        rho=rho,
        mode=NEURAL_EVAL_MODE,
    )


def run_grid_generalization(
    model,
    cfg: ExperimentConfig,
    run_dir: Path,
):
    """Evaluate generalization across a grid of load factors (Platinum Standard)."""
    log.info("\n" + "=" * 60)
    log.info("  Platinum Generalization Sweep (Grid Evaluation)")
    log.info("=" * 60)

    num_servers = cfg.system.num_servers
    service_rates = np.array(cfg.system.service_rates, dtype=np.float64)
    total_capacity = float(np.sum(service_rates))

    # Use the generalization config grid instead of hardcoded
    rho_grid = cfg.generalization.rho_grid_vals
    log.info(f"Evaluating across load factors: {rho_grid}")

    results = []
    for rho in _iter_with_progress(
        rho_grid,
        total=len(rho_grid),
        desc="policy: grid",
        unit="rho",
    ):
        arrival_rate = rho * total_capacity
        log.info(f"--- rho = {rho:.2f} (lambda = {arrival_rate:.2f}) ---")
        # 1. Uniform
        q_u = float(np.mean([time_averaged_queue_lengths(r, cfg.simulation.burn_in_fraction).sum()
                           for r in run_replications(num_servers=num_servers, arrival_rate=arrival_rate, service_rates=service_rates,
                                                   policy=build_policy_by_name("uniform"),
                                                   num_replications=cfg.simulation.num_replications, sim_time=cfg.simulation.ssa.sim_time,
                                                   base_seed=cfg.simulation.seed,
                                                   progress_desc=f"policy grid uniform rho={rho:.2f}")] ))

        # 2. JSQ
        q_j = float(np.mean([time_averaged_queue_lengths(r, cfg.simulation.burn_in_fraction).sum()
                           for r in run_replications(num_servers=num_servers, arrival_rate=arrival_rate, service_rates=service_rates,
                                                   policy=build_policy_by_name("jsq"),
                                                   num_replications=cfg.simulation.num_replications, sim_time=cfg.simulation.ssa.sim_time,
                                                   base_seed=cfg.simulation.seed,
                                                   progress_desc=f"policy grid jsq rho={rho:.2f}")] ))

        # 3. Neural (Greedy)
        neural_policy = _build_grid_eval_policy(model, service_rates, float(rho))
        q_n = float(np.mean([time_averaged_queue_lengths(r, cfg.simulation.burn_in_fraction).sum()
                           for r in run_replications(num_servers=num_servers, arrival_rate=arrival_rate, service_rates=service_rates,
                                                   policy=neural_policy,
                                                   num_replications=cfg.simulation.num_replications, sim_time=cfg.simulation.ssa.sim_time,
                                                   base_seed=cfg.simulation.seed,
                                                   progress_desc=f"policy grid neural rho={rho:.2f}")] ))

        # Performance Index: 100% = JSQ, 0% = Uniform
        dist = q_u - q_j
        safe_dist = max(dist, 1e-6)
        idx = 100.0 * ((q_u - q_n) / safe_dist)
        if q_n <= q_j: idx = max(100.0, idx)

        results.append({
            "rho": rho,
            "Uniform_EQ": q_u,
            "JSQ_EQ": q_j,
            "Neural_EQ": q_n,
            "Performance_Index": idx
        })
        log.info(f"  Idx: {idx:.1f}% | Neural E[Q]: {q_n:.2f} (JSQ: {q_j:.2f})")

    df = pd.DataFrame(results)
    df.to_csv(run_dir / "platinum_grid_results.csv", index=False)

    _plot_platinum_grid(df, run_dir)
    return results

def _plot_platinum_grid(df: pd.DataFrame, output_dir: Path):
    """Generate log-scale curves and performance index plots over rho."""
    from gibbsq.analysis.plotting import plot_platinum_grid

    plot_path = output_dir / "platinum_grid_analysis"
    fig = plot_platinum_grid(
        rho_values=df['rho'].values,
        uniform_eq=df['Uniform_EQ'].values,
        neural_eq=df['Neural_EQ'].values,
        jsq_eq=df['JSQ_EQ'].values,
        performance_index=df['Performance_Index'].values,
        save_path=plot_path,
        theme="publication",
        formats=["png", "pdf"]
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info(f"Platinum grid analysis saved to {plot_path}.png, {plot_path}.pdf")


def main(raw_cfg: DictConfig):
    """Main entry point for corrected policy comparison."""
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)

    run_dir, run_id = get_run_config(cfg, "policy", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="policy_comparison",
                            run_id=run_id, run_dir=run_dir)

    import jax
    results = run_corrected_comparison(cfg, run_dir, run_logger)

    # Platinum Step: If we have a neural policy, run the full grid generalization sweep
    if results and "N-GibbsQ (Platinum)" in results:
        if raw_cfg.get("grid", False):
            log.info("\n--- Platinum Trigger: Running Grid Generalization Sweep ---")

            N = cfg.system.num_servers
            mu = np.array(cfg.system.service_rates)
            pointer_dir = run_dir.parent.parent
            model_path = resolve_model_pointer(
                Path(__file__).resolve().parents[2],
                pointer_dir,
                allow_bc=False,
                allow_legacy=False,
            )
            key = jax.random.PRNGKey(cfg.simulation.seed)
            policy_net = NeuralRouter(num_servers=N, config=cfg.neural, service_rates=mu, key=key)
            policy_net = eqx.tree_deserialise_leaves(model_path, policy_net)

            rho = cfg.system.arrival_rate / float(mu.sum())
            run_grid_generalization(policy_net, cfg, run_dir)

    if run_logger:
        run_logger.finish()


if __name__ == "__main__":
    import sys
    import hydra
    if len(sys.argv) > 1:
        hydra.main(version_base=None, config_path="../../configs", config_name="default")(main)()
    else:
        from hydra import compose, initialize_config_dir
        import os
        config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(config_name="default")
            main(raw_cfg)
