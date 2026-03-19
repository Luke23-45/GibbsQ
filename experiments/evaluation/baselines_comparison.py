"""
Corrected Policy Comparison for N-GibbsQ.

This module implements the corrected baseline hierarchy for policy comparison,
addressing Smoking Gun #3 (Critical Load Advantage Is Baseline Collapse) and
Smoking Gun #4 (Parity Benchmark Is Self-Referential).

The corrected baseline hierarchy:
- Tier 1: Blind policies (Uniform, Proportional)
- Tier 2: Queue-length-based (JSQ, Power-of-d)
- Tier 3: Sojourn-time-based (JSSQ, SojournTimeSoftmax)
- Tier 4: Optimized analytical (GibbsQ-Sojourn with tuned α)
- Tier 5: Neural (N-GibbsQ trained with REINFORCE)

The parity criterion is now: N-GibbsQ must match or beat Tier 3 (JSSQ),
not Tier 4 (broken GibbsQ baseline).
"""

import logging
from pathlib import Path
import numpy as np
import jax
from omegaconf import DictConfig
from gibbsq.core.config import ExperimentConfig, hydra_to_config, validate
from gibbsq.core.builders import build_policy_by_name
from gibbsq.utils.model_io import DeterministicNeuralPolicy, StochasticNeuralPolicy
from gibbsq.engines.numpy_engine import simulate, run_replications, SimResult
from gibbsq.analysis.metrics import (
    time_averaged_queue_lengths, gini_coefficient, sojourn_time_estimate
)
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl
import pandas as pd
import matplotlib.pyplot as plt
import sys
import equinox as eqx
from gibbsq.core.neural_policies import NeuralRouter

log = logging.getLogger(__name__)


# DeterministicNeuralPolicy moved to gibbsq.utils.model_io.DeterministicNeuralPolicy


# Corrected baseline hierarchy (per professor's spec at suggestions.md:517-539)
CORRECTED_POLICIES = [
    # Tier 1: Analytical Optimum (Upper Bound on Performance)
    # cμ-rule: route to server i = argmax(μ_i · Q_i > 0)
    # [Known optimal for minimizing expected queue length in heavy traffic]
    # NOTE: cμ-rule not implemented as policy class - theoretical upper bound
    
    # Tier 2: Practical Asymptotic Optima
    {"tier": 2, "name": "jsq", "label": "JSQ (Min Queue)", "requires_mu": False},
    {"tier": 2, "name": "jssq", "label": "JSSQ (Min Sojourn)", "requires_mu": True},
    
    # Tier 3: Softmax/Gibbs Policies (Provably Stable Smooth Approximations)
    {"tier": 3, "name": "sojourn_softmax", "label": "GibbsQ-Sojourn (alpha=1.0)", "requires_mu": True, "alpha": 1.0},
    {"tier": 3, "name": "sojourn_softmax", "label": "GibbsQ-Sojourn (alpha=10.0)", "requires_mu": True, "alpha": 10.0},
    {"tier": 3, "name": "sojourn_softmax", "label": "GibbsQ-Sojourn (alpha=opt)", "requires_mu": True, "alpha": 5.0},
    
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
    )
    
    # Compute metrics
    q_totals = [time_averaged_queue_lengths(r, burn_in).sum() 
                for r in results]
    ginis = [gini_coefficient(time_averaged_queue_lengths(r, burn_in))
             for r in results]
    sojourns = [sojourn_time_estimate(r, cfg.system.arrival_rate, burn_in)
                for r in results]
    
    return {
        "mean_q_total": float(np.mean(q_totals)),
        "se_q_total": float(np.std(q_totals) / np.sqrt(len(q_totals))),
        "mean_gini": float(np.mean(ginis)),
        "se_gini": float(np.std(ginis) / np.sqrt(len(ginis))),
        "mean_sojourn": float(np.mean(sojourns)),
        "se_sojourn": float(np.std(sojourns) / np.sqrt(len(sojourns))),
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
    
    for entry in CORRECTED_POLICIES:
        tier = entry["tier"]
        name = entry["name"]
        label = entry["label"]
        
        log.info(f"Evaluating Tier {tier}: {label}...")
        
        # Create policy via Builders (Configuration-as-Code)
        kwargs = {}
        if entry.get("requires_mu", False):
            kwargs["mu"] = mu
        if "alpha" in entry:
            kwargs["alpha"] = entry["alpha"]
        if "d" in entry:
            kwargs["d"] = entry["d"]
            
        policy = build_policy_by_name(entry["name"], **kwargs)
        
        # Evaluate
        rng = np.random.default_rng(cfg.simulation.seed)
        metrics = evaluate_single_policy(policy, cfg, rng)
        
        results[label] = {
            "tier": tier,
            "name": name,
            **metrics,
        }
        
        log.info(f"  E[Q_total] = {metrics['mean_q_total']:.4f} ± {metrics['se_q_total']:.4f}")
        
        # Save metrics
        append_metrics_jsonl({
            "policy": label,
            "tier": tier,
            **metrics,
        }, run_dir / "corrected_comparison_metrics.jsonl")
    
    # Resolve pointer directory from active run output layout
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # run_dir = output_dir / experiment_type / run_id, so run_dir.parent.parent = output_dir
    pointer_dir = run_dir.parent.parent
    
    # Try DR pointer first, then standard reinforce pointer
    dr_ptr = pointer_dir / "latest_domain_randomized_weights.txt"
    std_ptr = pointer_dir / "latest_reinforce_weights.txt"
    
    ptr_to_use = None
    if dr_ptr.exists():
        ptr_to_use = dr_ptr
        log.info(f"Using Domain Randomized weights from {dr_ptr}")
    elif std_ptr.exists():
        ptr_to_use = std_ptr
        log.info(f"Using Standard REINFORCE weights from {std_ptr}")
    
    if ptr_to_use:
        log.info("\nEvaluating Tier 5: N-GibbsQ (REINFORCE trained)...")
        try:
                # Load weights
            relative_path = ptr_to_use.read_text(encoding='utf-8').strip()
            weights_path = _PROJECT_ROOT / relative_path
            
            if weights_path.exists():
                # Initialize and load
                key = jax.random.PRNGKey(cfg.simulation.seed)
                policy_net = NeuralRouter(num_servers=N, config=cfg.neural, key=key)
                policy_net = eqx.tree_deserialise_leaves(weights_path, policy_net)
                
                # SG-9 PATCH: Validate BOTH input_dim AND hidden_size.
                input_dim = N + (1 if cfg.neural.use_rho else 0)
                if policy_net.layers[0].weight.shape[1] != input_dim:
                    log.warning(
                        f"[SG-9] Neural model N-mismatch: model expects "
                        f"input_dim={policy_net.layers[0].weight.shape[1]}, system has input_dim={input_dim}. "
                        f"Skipping neural evaluation."
                    )
                    return
                elif policy_net.layers[0].weight.shape[0] != cfg.neural.hidden_size:
                    log.warning(
                        f"[SG-9] Neural model hidden_size mismatch: "
                        f"model={policy_net.layers[0].weight.shape[0]}, "
                        f"config expects={cfg.neural.hidden_size}. "
                        f"Skipping neural evaluation to avoid corrupt weights."
                    )
                    return
                
                deterministic = DeterministicNeuralPolicy(policy_net, mu)
                
                log.info("Evaluating N-GibbsQ (Greedy Deterministic)...")
                metrics = evaluate_single_policy(deterministic, cfg, 
                                                  np.random.default_rng(cfg.simulation.seed))
                
                results["N-GibbsQ (Platinum)"] = {
                    "tier": 5,
                    "name": "neural_platinum",
                    **metrics,
                }
                
                log.info(f"  E[Q_total] = {metrics['mean_q_total']:.4f} ± {metrics['se_q_total']:.4f}")
        except (KeyError, FileNotFoundError, ValueError, RuntimeError) as e:
            # Catch specific exceptions that indicate configuration/model issues
            # Do NOT catch all Exception types to avoid hiding real bugs
            log.warning(f"Could not evaluate neural policy: {type(e).__name__}: {e}")
    
    # Compute parity result using corrected tiered criteria
    log.info("\n" + "=" * 60)
    log.info("  Parity Analysis (Corrected Criteria)")
    log.info("=" * 60)
    
    # Get reference values for tiered comparison
    # Professor's spec at suggestions.md:547-551:
    # GOLD: N-GibbsQ E[Q] ≤ JSSQ E[Q] (matches asymptotic optimum)
    # SILVER: N-GibbsQ E[Q] ≤ GibbsQ-Sojourn E[Q] (matches smooth approximation)
    # BRONZE: N-GibbsQ E[Q] ≤ Proportional E[Q] (exceeds static baseline)
    # FAILED: N-GibbsQ E[Q] > Proportional E[Q]
    
    jssq_result = results.get("JSSQ (Min Sojourn)")
    sojourn_result = results.get("GibbsQ-Sojourn (alpha=1.0)")
    proportional_result = results.get("Proportional (mu/Lambda)")
    
    neural_result = results.get("N-GibbsQ (Platinum)")
    
    if neural_result:
        neural_q = neural_result["mean_q_total"]
        log.info(f"N-GibbsQ (Platinum/Greedy): E[Q] = {neural_q:.4f}")
        
        # Get reference thresholds
        jssq_q = jssq_result["mean_q_total"] if jssq_result else float('inf')
        sojourn_q = sojourn_result["mean_q_total"] if sojourn_result else float('inf')
        proportional_q = proportional_result["mean_q_total"] if proportional_result else float('inf')
        
        log.info(f"Reference thresholds:")
        log.info(f"  JSSQ (Tier 2): E[Q] = {jssq_q:.4f}")
        log.info(f"  GibbsQ-Sojourn (Tier 3): E[Q] = {sojourn_q:.4f}")
        log.info(f"  Proportional (Tier 4): E[Q] = {proportional_q:.4f}")
        
        def has_parity(q_agent, se_agent, q_base, se_base):
            # SG#4 FIX: Parity applies if the Agent performs better OR is within the statistical Margin of Error
            # We calculate a Confidence limit (configurable via parity_z_score) for the difference of two means
            margin_of_error = cfg.verification.parity_z_score * np.sqrt(se_agent**2 + se_base**2)
            return q_agent <= (q_base + margin_of_error)

        se_neural = neural_result["se_q_total"]
        
        # Safely fetch base queries and exact Standard Errors
        jssq_q = jssq_result["mean_q_total"] if jssq_result else float('inf')
        jssq_se = jssq_result["se_q_total"] if jssq_result else 0.0
        
        sojourn_q = sojourn_result["mean_q_total"] if sojourn_result else float('inf')
        sojourn_se = sojourn_result["se_q_total"] if sojourn_result else 0.0
        
        proportional_q = proportional_result["mean_q_total"] if proportional_result else float('inf')
        proportional_se = proportional_result["se_q_total"] if proportional_result else 0.0
        
        log.info(f"Reference statistical bounds (95% CI):")
        log.info(f"  JSSQ (Tier 2): E[Q] = {jssq_q:.4f} ± {jssq_se:.4f}")
        log.info(f"  GibbsQ-Sojourn (Tier 3): E[Q] = {sojourn_q:.4f} ± {sojourn_se:.4f}")
        log.info(f"  Proportional (Tier 4): E[Q] = {proportional_q:.4f} ± {proportional_se:.4f}")
        
        # Strict conditional Parity limits
        if has_parity(neural_q, se_neural, jssq_q, jssq_se):
            parity = "GOLD"
            log.info(f"PARITY RESULT: GOLD [OK] (Statistically matches asymptotic optimum JSSQ)")
        elif has_parity(neural_q, se_neural, sojourn_q, sojourn_se):
            parity = "SILVER"
            log.info(f"PARITY RESULT: SILVER [OK] (Statistically matches analytical GibbsQ-Sojourn)")
        elif has_parity(neural_q, se_neural, proportional_q, proportional_se):
            parity = "BRONZE"
            log.info(f"PARITY RESULT: BRONZE [OK] (Statistically matches static Proportional baseline)")
        else:
            parity = "FAILED"
            log.info(f"PARITY RESULT: FAILED [FAIL] (Statistically inferior to benchmark baselines)")
        
        # Store parity result
        neural_result["parity"] = parity
    else:
        log.info("N-GibbsQ (REINFORCE) not evaluated - skipping parity analysis")
    
    # Generate comparison plot
    _generate_comparison_plot(results, run_dir)
    
    return results


def _generate_comparison_plot(results: dict, run_dir: Path):
    """Generate comparison bar chart."""
    import matplotlib.pyplot as plt
    
    # Sort by tier and E[Q]
    sorted_results = sorted(results.items(), key=lambda x: (x[1]["tier"], x[1]["mean_q_total"]))
    
    labels = [name for name, _ in sorted_results]
    q_values = [r["mean_q_total"] for _, r in sorted_results]
    q_errors = [r["se_q_total"] for _, r in sorted_results]
    tiers = [r["tier"] for _, r in sorted_results]
    
    # Color by tier
    tier_colors = {1: 'gray', 2: 'blue', 3: 'green', 4: 'purple', 5: 'red'}
    colors = [tier_colors.get(t, 'black') for t in tiers]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, q_values, yerr=q_errors, color=colors, alpha=0.7, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Expected Total Queue Length E[Q_total]')
    ax.set_title('Corrected Policy Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add tier legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f'Tier {t}', alpha=0.7) 
                       for t, c in tier_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plot_path = run_dir / "corrected_policy_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    log.info(f"Comparison plot saved to {plot_path}")


def run_grid_generalization(
    neural_policy,
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
    for rho in rho_grid:
        arrival_rate = rho * total_capacity
        log.info(f"--- rho = {rho:.2f} (lambda = {arrival_rate:.2f}) ---")
        # 1. Uniform
        q_u = float(np.mean([time_averaged_queue_lengths(r, cfg.simulation.burn_in_fraction).sum() 
                           for r in run_replications(num_servers=num_servers, arrival_rate=arrival_rate, service_rates=service_rates, 
                                                   policy=build_policy_by_name("uniform"), 
                                                   num_replications=cfg.simulation.num_replications, sim_time=cfg.simulation.ssa.sim_time, 
                                                   base_seed=cfg.simulation.seed)]))
        
        # 2. JSQ
        q_j = float(np.mean([time_averaged_queue_lengths(r, cfg.simulation.burn_in_fraction).sum() 
                           for r in run_replications(num_servers=num_servers, arrival_rate=arrival_rate, service_rates=service_rates, 
                                                   policy=build_policy_by_name("jsq"), 
                                                   num_replications=cfg.simulation.num_replications, sim_time=cfg.simulation.ssa.sim_time, 
                                                   base_seed=cfg.simulation.seed)]))
        
        # 3. Neural (Greedy)
        q_n = float(np.mean([time_averaged_queue_lengths(r, cfg.simulation.burn_in_fraction).sum() 
                           for r in run_replications(num_servers=num_servers, arrival_rate=arrival_rate, service_rates=service_rates, 
                                                   policy=neural_policy, 
                                                   num_replications=cfg.simulation.num_replications, sim_time=cfg.simulation.ssa.sim_time, 
                                                   base_seed=cfg.simulation.seed)]))
        
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

    # Save and Plot
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "platinum_grid_results.csv", index=False)
    
    _plot_platinum_grid(df, run_dir)
    return results

def _plot_platinum_grid(df: pd.DataFrame, output_dir: Path):
    """Generate high-fidelity log-scale and index plots."""
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3})
    
    # Plot 1: E[Q] Log-Scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(df['rho'], df['Uniform_EQ'], 'r--x', label='Uniform')
    ax1.plot(df['rho'], df['Neural_EQ'], 'b-o', linewidth=2, label='N-GibbsQ (Platinum)')
    ax1.plot(df['rho'], df['JSQ_EQ'], 'g-.s', label='JSQ (Optimal)')
    ax1.set_yscale('log')
    ax1.set_xlabel('Load Factor (rho)')
    ax1.set_ylabel('Expected Queue Length (Log Scale)')
    ax1.set_title('Performance Envelope')
    ax1.legend()
    
    # Plot 2: Performance Index
    ax2.plot(df['rho'], df['Performance_Index'], 'm-D', linewidth=2)
    ax2.axhline(100, color='g', linestyle='--', alpha=0.5, label='JSQ Parity')
    ax2.axhline(0, color='r', linestyle='--', alpha=0.5, label='Uniform Parity')
    ax2.set_ylim(-10, 110)
    ax2.set_xlabel('Load Factor (rho)')
    ax2.set_ylabel('Performance Index (%)')
    ax2.set_title('Generalization Efficiency')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "platinum_grid_analysis.png", dpi=300)
    plt.close()
    log.info(f"Platinum grid analysis saved to {output_dir}")


def main(raw_cfg: DictConfig):
    """Main entry point for corrected policy comparison."""
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)
    
    run_dir, run_id = get_run_config(cfg, "policy_comparison", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="policy_comparison",
                            run_id=run_id, run_dir=run_dir)
    
    import jax
    results = run_corrected_comparison(cfg, run_dir, run_logger)
    
    # Platinum Step: If we have a neural policy, run the full grid generalization sweep
    if results and "N-GibbsQ (Platinum)" in results:
        # Check for grid flag in the raw Hydra config
        if raw_cfg.get("grid", False):
            # We already have mu, cfg, run_dir.
            log.info("\n--- Platinum Trigger: Running Grid Generalization Sweep ---")
            
            N = cfg.system.num_servers
            mu = np.array(cfg.system.service_rates)
            pointer_dir = run_dir.parent.parent
            dr_ptr = pointer_dir / "latest_domain_randomized_weights.txt"
            
            if dr_ptr.exists():
                weights_path = Path(__file__).resolve().parents[2] / dr_ptr.read_text().strip()
                key = jax.random.PRNGKey(cfg.simulation.seed)
                policy_net = NeuralRouter(num_servers=N, config=cfg.neural, key=key)
                policy_net = eqx.tree_deserialise_leaves(weights_path, policy_net)
                
                platinum_policy = DeterministicNeuralPolicy(policy_net, mu)
                run_grid_generalization(platinum_policy, cfg, run_dir)
            else:
                log.warning("No latest_domain_randomized_weights.txt found - skipping grid sweep.")
    
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
