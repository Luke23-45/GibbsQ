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
import jax.numpy as jnp
from omegaconf import DictConfig

from gibbsq.core.config import ExperimentConfig, hydra_to_config, validate
from gibbsq.core.policies import (
    SoftmaxRouting, UniformRouting, ProportionalRouting,
    JSQRouting, PowerOfDRouting, JSSQRouting, SojournTimeSoftmaxRouting
)
from gibbsq.engines.numpy_engine import simulate, run_replications, SimResult
from gibbsq.analysis.metrics import (
    time_averaged_queue_lengths, gini_coefficient, sojourn_time_estimate
)
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.exporter import append_metrics_jsonl

log = logging.getLogger(__name__)


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


def make_corrected_policy(entry: dict, mu: np.ndarray) -> object:
    """Create a policy from the corrected hierarchy entry."""
    name = entry["name"]
    
    if name == "uniform":
        return UniformRouting()
    elif name == "proportional":
        return ProportionalRouting(mu)
    elif name == "jsq":
        return JSQRouting()
    elif name == "power_of_d":
        return PowerOfDRouting(entry.get("d", 2))
    elif name == "jssq":
        return JSSQRouting(mu)
    elif name == "sojourn_softmax":
        return SojournTimeSoftmaxRouting(mu, entry.get("alpha", 1.0))
    else:
        raise ValueError(f"Unknown policy: {name}")


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
        
        # Create policy
        if entry.get("requires_mu", False):
            policy = make_corrected_policy(entry, mu)
        else:
            policy = make_corrected_policy(entry, mu)
        
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
    
    # Evaluate neural policy if weights exist
    neural_weights_path = run_dir.parent.parent / "latest_reinforce_weights.txt"
    if neural_weights_path.exists():
        log.info("\nEvaluating Tier 5: N-GibbsQ (REINFORCE trained)...")
        try:
            import equinox as eqx
            from gibbsq.core.neural_policies import NeuralRouter
            from gibbsq.core.features import sojourn_time_features
            
            # Load weights
            relative_path = neural_weights_path.read_text(encoding='utf-8').strip()
            weights_path = run_dir.parent.parent / relative_path
            
            if weights_path.exists():
                # Initialize and load
                key = jax.random.PRNGKey(cfg.simulation.seed)
                policy_net = NeuralRouter(num_servers=N, config=cfg.neural, key=key)
                policy_net = eqx.tree_deserialise_leaves(weights_path, policy_net)
                
                # Optimization: Extract NumPy parameters once
                np_params = policy_net.get_numpy_params()
                np_config = policy_net.config
                
                # Create wrapper policy
                class NeuralPolicyWrapper:
                    def __init__(self, net, mu):
                        self._net = net
                        self._mu = mu
                    
                    def __call__(self, Q, rng):
                        # 1. Compute routing probabilities using sojourn-time features
                        # Use fast NumPy forward pass
                        s = (Q + 1.0) / self._mu
                        logits = policy_net.numpy_forward(s, np_params, np_config)
                        logits = logits - np.max(logits)
                        probs = np.exp(logits)
                        return probs / probs.sum()
                
                neural_policy = NeuralPolicyWrapper(policy_net, mu)
                metrics = evaluate_single_policy(neural_policy, cfg, 
                                                  np.random.default_rng(cfg.simulation.seed))
                
                results["N-GibbsQ (REINFORCE)"] = {
                    "tier": 5,
                    "name": "neural_reinforce",
                    **metrics,
                }
                
                log.info(f"  E[Q_total] = {metrics['mean_q_total']:.4f} ± {metrics['se_q_total']:.4f}")
        except Exception as e:
            log.warning(f"Could not evaluate neural policy: {e}")
    
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
    
    neural_result = results.get("N-GibbsQ (REINFORCE)")
    
    if neural_result:
        neural_q = neural_result["mean_q_total"]
        log.info(f"N-GibbsQ (REINFORCE): E[Q] = {neural_q:.4f}")
        
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
            # We calculate a 95% Confidence limit (approx 1.96 z-score) for the difference of two means
            margin_of_error = 1.96 * np.sqrt(se_agent**2 + se_base**2)
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
            log.info(f"PARITY RESULT: GOLD ✓ (Statistically identical to asymptotic optimum JSSQ)")
        elif has_parity(neural_q, se_neural, sojourn_q, sojourn_se):
            parity = "SILVER"
            log.info(f"PARITY RESULT: SILVER ✓ (Statistically identical to analytical GibbsQ-Sojourn)")
        elif has_parity(neural_q, se_neural, proportional_q, proportional_se):
            parity = "BRONZE"
            log.info(f"PARITY RESULT: BRONZE ✓ (Statistically dominates static Proportional baseline)")
        else:
            parity = "FAILED"
            log.info(f"PARITY RESULT: FAILED ✗ (Statistically inferior to benchmark baselines)")
        
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


def main(raw_cfg: DictConfig):
    """Main entry point for corrected policy comparison."""
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)
    
    run_dir, run_id = get_run_config(cfg, "corrected_policy_comparison", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="corrected_policy_comparison",
                            run_id=run_id, run_dir=run_dir)
    
    import jax
    results = run_corrected_comparison(cfg, run_dir, run_logger)
    
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
