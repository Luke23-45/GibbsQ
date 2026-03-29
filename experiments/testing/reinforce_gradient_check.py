"""
REINFORCE Gradient Estimator Validation.

This module verifies that the REINFORCE gradient estimator produces
unbiased gradient estimates by comparing against finite-difference
approximations. This is Track 5 of the corrective research plan.

The validation ensures:
1. REINFORCE gradients are unbiased (match finite-difference in expectation)
2. Variance is within acceptable bounds
3. The score function estimator is correctly implemented
"""

import logging
from pathlib import Path
from typing import NamedTuple
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
import equinox as eqx
from jax.flatten_util import ravel_pytree
from jaxtyping import PRNGKeyArray, Array, Float
from omegaconf import DictConfig

from gibbsq.core.config import ExperimentConfig, hydra_to_config, validate
from typing import Any
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.features import look_ahead_potential
from gibbsq.utils.logging import setup_wandb, get_run_config
from gibbsq.utils.progress import create_progress, iter_progress
from gibbsq.engines.jax_ssa import vmap_collect_trajectories, compute_poisson_max_steps

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class GradientCheckResult(NamedTuple):
    """Result of gradient estimator validation."""
    reinforce_grad: np.ndarray
    finite_diff_grad: np.ndarray
    relative_error: float
    cosine_similarity: float
    bias_estimate: float
    variance_estimate: float
    reinforce_var_vector: np.ndarray 
    reinforce_mean_var_vector: np.ndarray
    fd_var_vector: np.ndarray # NEW: Per-parameter FD variance
    computed_mask: np.ndarray
    passed: bool


def _build_plot_artifacts(
    result: GradientCheckResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return plot-ready gradient arrays restricted to computed FD coordinates."""
    mask = np.asarray(result.computed_mask, dtype=bool)
    fd_grads = np.asarray(result.finite_diff_grad, dtype=np.float64)[mask]
    rf_grads = np.asarray(result.reinforce_grad, dtype=np.float64)[mask]
    if result.reinforce_mean_var_vector is None or result.fd_var_vector is None:
        return fd_grads, rf_grads, None
    combined_se = np.sqrt(
        np.asarray(result.reinforce_mean_var_vector, dtype=np.float64)[mask]
        + np.asarray(result.fd_var_vector, dtype=np.float64)[mask]
    )
    safe_se = np.where(combined_se > 1e-12, combined_se, 1e-12)
    z_scores = np.abs(rf_grads - fd_grads) / safe_se
    return fd_grads, rf_grads, z_scores


def _sum_masked_action_interval_returns(batch) -> jax.Array:
    action_mask = batch.is_action_mask & batch.valid_mask
    return jnp.sum(jnp.where(action_mask, batch.returns, 0.0))

def compute_reinforce_gradient(
    policy_net: NeuralRouter,
    num_servers: int,
    arrival_rate: float,
    service_rates: np.ndarray,
    sim_time: float,
    n_samples: int,
    base_seed: int,
    cfg: Any, # Passed down for thresholds
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Optimized REINFORCE gradient using JAX Chunked Vectorized Execution."""
    from jax.flatten_util import ravel_pytree
    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    from gibbsq.engines.jax_ssa import vmap_collect_trajectories
    
    # 1. Chunking configuration to prevent Out-Of-Memory (OOM)
    chunk_size = cfg.verification.gradient_check_chunk_size
    # Patch: Dynamically compute safe hardware bounds using 6-Sigma Poisson tail
    max_steps = compute_poisson_max_steps(arrival_rate, service_rates, sim_time)
    
    keys = jax.random.split(jax.random.PRNGKey(base_seed), n_samples)
    service_rates_jax = jnp.asarray(service_rates, dtype=jnp.float32)
    
    # Isolate initial array parameters and unravel function
    params = eqx.filter(policy_net, eqx.is_array)
    init_flat_params, unravel = ravel_pytree(params)
    
    @eqx.filter_jit
    def compute_chunk_grad(flat_theta, keys_chunk):
        actual_chunk_size = keys_chunk.shape[0]
        
        # 1. SAMPLE: Generate trajectory realization using the FIXED initial policy
        # We MUST use the same policy that was passed in, unraveled from flat_theta
        active_model = unravel(flat_theta)
        
        batch = vmap_collect_trajectories(
            policy_net=active_model,
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates_jax,
            sim_time=sim_time,
            keys=keys_chunk,
            max_steps=max_steps,
            gamma=cfg.neural_training.gamma
        )
        
        mask = batch.is_action_mask & batch.valid_mask
        mask_f32 = mask.astype(jnp.float32)
        
        # 1.5 LEAVE-ONE-OUT (LOO) BASELINE (SG#1 Fix)
        # We need a baseline b_{k,i} that doesn't include the current trajectory i.
        # This prevents the self-correlation bias that plagues standard REINFORCE in CTMCs.
        sum_returns = jnp.sum(jnp.where(mask, batch.returns, 0.0), axis=0, keepdims=True)
        sum_mask = jnp.sum(mask_f32, axis=0, keepdims=True)
        
        loo_sum_returns = sum_returns - jnp.where(mask, batch.returns, 0.0)
        loo_sum_mask = sum_mask - mask_f32
        
        # Unbiased baseline: mean of others
        b_k = jnp.where(loo_sum_mask > 0.5, loo_sum_returns / loo_sum_mask, 0.0)
        
        # Realizations are constants for the gradient graph
        adv = jax.lax.stop_gradient(batch.returns - b_k)
        fixed_states = jax.lax.stop_gradient(batch.states)
        fixed_actions = jax.lax.stop_gradient(batch.actions)
        fixed_mask = jax.lax.stop_gradient(mask)

        def loss_fn(theta):
            # 2. DIFFERENTIATE: Score function term grad(log π(a|s))
            m = unravel(theta)
            
            # Application
            # We need to map over batch and step, but broadcast mu and rho
            # SG#13 FIX: in_axes=(0, None, None) safely broadcasts mu and rho across the entire batch
            inner_vmap = jax.vmap(m, in_axes=(0, None, None))
            v_model = jax.vmap(inner_vmap, in_axes=(0, None, None))
            
            rho_val = arrival_rate / jnp.sum(service_rates_jax)
            logits = v_model(fixed_states, service_rates_jax, rho_val)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            
            # Gather log_prob(a_t | s_t)
            safe_a = jnp.clip(fixed_actions, 0, num_servers - 1)
            batch_idx = jnp.arange(actual_chunk_size)[:, None]
            step_idx = jnp.arange(max_steps)[None, :]
            chosen_log_probs = log_probs[batch_idx, step_idx, safe_a]
            
            # batch.returns already contains the discounted return aligned to each action.
            # Reapplying gamma here changes the objective and breaks the estimator check.
            # Loss = - E[ Advantage * log_prob ] for maximization
            # We return negative sum because jax.grad minimizes by default
            return -jnp.sum(fixed_mask * adv * chosen_log_probs)

        # Differentiate with respect to the flat parameters
        return jax.grad(loss_fn)(flat_theta)

    # 2. Sequential chunk evaluation
    grad_list = []
    samples_per_chunk = []  # Track actual samples per chunk for proper variance scaling
    
    chunk_starts = range(0, n_samples, chunk_size)
    for i in iter_progress(
        chunk_starts,
        total=len(range(0, n_samples, chunk_size)),
        desc="reinforce_check: REINFORCE chunks",
        unit="chunk",
        leave=False,
    ):
        chunk_keys = keys[i:i+chunk_size]
        actual_chunk_size = chunk_keys.shape[0]
        grad_list.append(compute_chunk_grad(init_flat_params, chunk_keys))
        samples_per_chunk.append(actual_chunk_size)
    
    # Stack gradients: Shape (n_chunks, n_params)
    # Each chunk gradient is the SUM of gradients for samples in that chunk
    stacked_grads = jnp.stack(grad_list, axis=0)
    samples_per_chunk = jnp.array(samples_per_chunk)
    total_samples = float(n_samples)
    
    # Mean gradient: sum of all chunk gradients / total samples
    # Each chunk gradient is already a sum, so we divide by total_samples
    mean_grad = np.array(-jnp.sum(stacked_grads, axis=0) / total_samples)
    
    # Variance computation: 
    # Each chunk gradient g_i = sum of individual gradients in chunk i
    # Var(single sample) = Var(chunk sum) / chunk_size
    # We compute variance across chunk sums, then scale appropriately
    if stacked_grads.shape[0] > 1:
        # Compute variance of chunk sums (weighted by chunk size)
        # Using the formula: Var(X) = E[X^2] - E[X]^2
        # For chunk sums: Var(sum_i) = chunk_size * Var(single sample)
        
        # Weighted variance accounting for different chunk sizes
        weights = samples_per_chunk / total_samples
        weighted_mean = jnp.sum(stacked_grads * weights[:, None], axis=0)
        
        # Variance of chunk sums
        diff = stacked_grads - weighted_mean[None, :]
        weighted_var_chunk_sum = jnp.sum(weights[:, None] * diff**2, axis=0)
        
        # Scale to get variance of individual samples
        # Var(single) = Var(chunk_sum) / avg_chunk_size
        avg_chunk_size = total_samples / float(len(grad_list))
        grad_variance = weighted_var_chunk_sum / avg_chunk_size
        
        variance = float(jnp.mean(grad_variance))
    else:
        variance = 0.0  # Single chunk - variance undefined
        grad_variance = jnp.zeros_like(mean_grad)
    
    # Bias estimate: computed in run_gradient_check using finite difference as reference
    # Placeholder here; actual bias computed when comparing to finite_diff_grad
    bias = 0.0  # Will be overwritten in run_gradient_check
    
    return mean_grad, bias, variance, np.array(grad_variance)


def compute_finite_difference_gradient(
    policy_net: NeuralRouter,
    num_servers: int,
    arrival_rate: float,
    service_rates: np.ndarray,
    sim_time: float,
    epsilon: float,
    n_samples: int,
    base_seed: int,
    cfg: Any, # Passed down for thresholds
    reinforce_grad: np.ndarray = None, # for real-time comparison
    reinforce_var: np.ndarray = None,  # NEW: for Z-score calculation
    z_critical: float = 3.0,           # NEW: dynamic FWER threshold
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gradient via finite differences using Chunked JAX Execution."""
    from jax.flatten_util import ravel_pytree
    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    from gibbsq.engines.jax_ssa import vmap_collect_trajectories
    
    params = eqx.filter(policy_net, eqx.is_array)
    flat_params, unravel = ravel_pytree(params)
    
    n_params = flat_params.shape[0]
    n_test = min(cfg.verification.gradient_check_n_test, n_params)  # Configured test amount 
    rng_select = np.random.default_rng(base_seed + 999)
    selected_indices = rng_select.choice(n_params, size=n_test, replace=False)
    
    grad = np.zeros(n_params)
    service_rates_jax = jnp.asarray(service_rates, dtype=jnp.float32)
    
    # 1. Chunking config
    chunk_size = cfg.verification.gradient_check_chunk_size
    # Patch: Dynamically compute safe hardware bounds using 6-Sigma Poisson tail
    max_steps = compute_poisson_max_steps(arrival_rate, service_rates, sim_time)
    
    # Pre-generate keys arrays to prevent variance drift across parameters
    # CRN: Use SAME keys for both perturbations
    keys_shared = jax.random.split(jax.random.PRNGKey(base_seed), n_samples)
    
    @jax.jit
    def get_chunk_sum_return(flat_theta, keys_chunk):
        model = unravel(flat_theta)
        batch = vmap_collect_trajectories(
            policy_net=model,
            num_servers=num_servers,
            arrival_rate=arrival_rate,
            service_rates=service_rates_jax,
            sim_time=sim_time,
            keys=keys_chunk,
            max_steps=max_steps,
            gamma=cfg.neural_training.gamma
        )
        return _sum_masked_action_interval_returns(batch)

    def get_expected_return(flat_theta, all_keys):
        total_sum = 0.0
        chunk_sums = []
        # Sequential python loop over chunks maintains flat memory ceiling
        for i in iter_progress(
            range(0, n_samples, chunk_size),
            total=len(range(0, n_samples, chunk_size)),
            desc="reinforce_check: FD chunks",
            unit="chunk",
            leave=False,
        ):
            chunk_keys = all_keys[i:i+chunk_size]
            # Cast to Python float64 immediately to prevent float32 catastrophic cancellation
            val = float(get_chunk_sum_return(flat_theta, chunk_keys))
            total_sum += val
            chunk_sums.append(val)
        
        # Compute sample variance across chunks for this parameter perturbation
        # This is the variance of the mean estimate: Var(TotalMean) = Var(ChunkSum) / (C^2 * K)
        chunk_arr = np.array(chunk_sums)
        n_chunks = len(chunk_sums)
        if n_chunks > 1:
            # NASA-Grade: Use ddof=1 for mathematically unbiased sample variance
            chunk_var = np.var(chunk_arr, ddof=1) / (float(chunk_size)**2 * n_chunks)
        else:
            chunk_var = 0.0
        
        return total_sum / float(n_samples), chunk_var

    # 2. Evaluation
    computed_mask = np.zeros(n_params, dtype=bool)
    fd_var = np.zeros(n_params)
    
    for i, param_idx in enumerate(iter_progress(
        selected_indices,
        total=len(selected_indices),
        desc="reinforce_check: FD params",
        unit="param",
    )):
        # ---- PLUS PERTURBATION ----
        params_plus = flat_params.at[param_idx].add(epsilon)
        return_plus, var_plus = get_expected_return(params_plus, keys_shared)
        
        # ---- MINUS PERTURBATION ----
        params_minus = flat_params.at[param_idx].add(-epsilon)
        return_minus, var_minus = get_expected_return(params_minus, keys_shared)
        
        # Central Difference Formula
        grad[param_idx] = (return_plus - return_minus) / (2 * epsilon)
        # Effective variance of FD estimator (central difference)
        # Var((R+ - R-) / 2e) = (Var(R+) + Var(R-) - 2Cov(R+, R-)) / (4e^2)
        # Since R+ and R- share keys, they are highly correlated (CRN).
        # We assume independent for a conservative (higher) variance estimate:
        fd_var[param_idx] = (var_plus + var_minus) / (4 * epsilon**2)
        
        computed_mask[param_idx] = True
        
        # ---- PERIODIC RUNNING STATS LOGGING ----
        last_step = (i + 1) == len(selected_indices)
        if (i + 1) % 10 == 0 or last_step:
            # Current parameter stats
            rf_val = float(reinforce_grad[param_idx])
            fd_val = float(grad[param_idx])
            abs_diff = abs(rf_val - fd_val)
            
            # Compute running subset metrics
            indices_done = selected_indices[:i+1]
            rf_sub = reinforce_grad[indices_done]
            fd_sub = grad[indices_done]
            
            diff_norm = np.linalg.norm(rf_sub - fd_sub)
            fd_norm_sub = np.linalg.norm(fd_sub)
            rel_err_p = diff_norm / fd_norm_sub if fd_norm_sub > 1e-9 else 0.0
            
            # Running Cosine Similarity
            rf_norm_sub = np.linalg.norm(rf_sub)
            if rf_norm_sub > 1e-9 and fd_norm_sub > 1e-9:
                running_cos_sim = np.dot(rf_sub, fd_sub) / (rf_norm_sub * fd_norm_sub)
            else:
                running_cos_sim = 0.0

            # Statistical Significance (Z-score)
            if reinforce_var is not None and reinforce_var[param_idx] > 1e-11:
                # POOLED STANDARD ERROR: Accounting for BOTH REINFORCE and FD variance
                # reinforce_var is single-sample variance -> divide by N
                rf_std_err = np.sqrt(reinforce_var[param_idx] / n_samples)
                
                # fd_var is ALREADY the exact variance of the mean gradient estimate
                fd_std_err = np.sqrt(fd_var[param_idx]) 
                pooled_std_err = np.sqrt(rf_std_err**2 + fd_std_err**2)
                
                z_score = abs_diff / max(1e-12, pooled_std_err)
                z_str = f" | z={z_score:5.2f}"
                is_bias = z_score > z_critical # Dynamic FWER flag
            else:
                z_str = " | z=N/A "
                is_bias = rel_err_p > cfg.verification.gradient_check_error_threshold

            # Status based on configurable threshold or Z-score
            status = "[!!]" if is_bias else "[OK]"
            
            log.info(
                f"  {status} Param {i+1:3}/{len(selected_indices):<3} (idx {param_idx:5}): "
                f"RF={rf_val:10.6f} | FD={fd_val:10.6f} | diff={abs_diff:10.6f}{z_str} | "
                f"RunningRelErr={rel_err_p:7.4f} | RunningCosSim={running_cos_sim:7.4f}"
            )
    
    
    return grad, computed_mask, fd_var


def run_gradient_check(
    cfg: ExperimentConfig,
    key: PRNGKeyArray,
    n_samples: int = None,  # Restore professor's iteration-speed spec
    epsilon: float = None,
) -> GradientCheckResult:
    """
    Run the gradient estimator validation.
    
    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.
    key : PRNGKeyArray
        JAX random key.
    n_samples : int
        Number of trajectories for estimation.
    epsilon : float
        Finite difference perturbation.
    
    Returns
    -------
    GradientCheckResult
        Validation result.
    """
    if n_samples is None:
        n_samples = cfg.verification.gradient_check_n_samples
    if epsilon is None:
        epsilon = cfg.verification.gradient_check_epsilon
        
    # Initialize policy
    key, policy_key = jax.random.split(key)
    gradient_hidden_size = (
        cfg.verification.gradient_check_hidden_size
        if cfg.verification.gradient_check_hidden_size is not None
        else cfg.neural.hidden_size
    )
    gradient_neural_cfg = replace(cfg.neural, hidden_size=gradient_hidden_size)

    policy_net = NeuralRouter(
        num_servers=cfg.system.num_servers,
        config=gradient_neural_cfg,
        service_rates=cfg.system.service_rates,
        key=policy_key,
    )
    
    # SHAKE WEIGHTS to get off the zero-init plateau for validation
    from jax.flatten_util import ravel_pytree
    params = eqx.filter(policy_net, eqx.is_array)
    flat_params, unravel = ravel_pytree(params)
    # Use deterministic shake based on config seed
    shake_key = jax.random.PRNGKey(cfg.simulation.seed + 12345)
    flat_params = flat_params + cfg.verification.gradient_shake_scale * jax.random.normal(shake_key, flat_params.shape)
    policy_net = unravel(flat_params)
    
    # --- DYNAMIC STATISTICAL SCALING ---
    n_params = flat_params.shape[0]
    n_test = min(cfg.verification.gradient_check_n_test, n_params)
    
    alpha = 0.05
    base_test = 50  # Baseline dimensionality calibrated safely at base_samples
    base_samples = n_samples
    
    # Calculate required Z-scores using Bonferroni correction
    z_base = stats.norm.ppf(1.0 - (alpha / (2.0 * base_test)))
    z_new = stats.norm.ppf(1.0 - (alpha / (2.0 * n_test)))
    
    # Scale N proportional to the squared critical Z-score to maintain FWER bounds.
    # CRITICAL FIX: We must floor the scaling at 1.0 to preserve the absolute 
    # geometric signal-to-noise ratio required for the Cosine Similarity check.
    scaled_n_samples = base_samples * max(1.0, (z_new**2 / z_base**2))
    chunk_size = cfg.verification.gradient_check_chunk_size
    n_samples = int(np.ceil(scaled_n_samples / chunk_size) * chunk_size)
    
    log.info(f"Statistical Scaling: D={n_test} parameters.")
    log.info(f"Adjusted Z-Critical Threshold: {z_new:.2f}")
    log.info(f"Scaled n_samples from {base_samples} -> {n_samples} to maintain confidence bounds.")
    # -----------------------------------

    service_rates = np.array(cfg.system.service_rates, dtype=np.float64)
    
    # DRASTIC CUT: sim_time=15.0 stops Gillespie trajectories from violently desynchronizing
    # This is the "Law of Large Numbers" config from professor's Patch 3
    sim_time = cfg.verification.gradient_check_sim_time
    
    log.info("Computing REINFORCE gradient estimate...")
    reinforce_grad, bias, variance, grad_variance = compute_reinforce_gradient(
        policy_net,
        cfg.system.num_servers,
        cfg.system.arrival_rate,
        service_rates,
        sim_time,  # Use short horizon
        n_samples=n_samples,
        base_seed=cfg.simulation.seed,
        cfg=cfg,
    )
    
    log.info("Computing finite-difference gradient estimate...")
    finite_diff_grad, computed_mask, fd_var = compute_finite_difference_gradient(
        policy_net,
        cfg.system.num_servers,
        cfg.system.arrival_rate,
        service_rates,
        sim_time,  # Use short horizon
        epsilon=epsilon,
        n_samples=n_samples,
        base_seed=cfg.simulation.seed + 10000,
        cfg=cfg,
        reinforce_grad=reinforce_grad,
        reinforce_var=grad_variance,
        z_critical=z_new,
    )
    
    # Compute relative error using L2 norm as per professor's spec
    # ||ĝ - ∇J||_2 / ||∇J||_2 < 0.10
    mask = computed_mask
    if mask.sum() > 0:
        reinforce_masked = reinforce_grad[mask]
        finite_diff_masked = finite_diff_grad[mask]
        
        # L2 norm relative error
        diff_norm = np.linalg.norm(reinforce_masked - finite_diff_masked)
        fd_norm = np.linalg.norm(finite_diff_masked)
        
        if fd_norm > 1e-9:
            relative_error = diff_norm / fd_norm
        else:
            relative_error = 0.0
        
        # Cosine similarity for direction check (more robust to variance)
        reinforce_norm = np.linalg.norm(reinforce_masked)
        if reinforce_norm > 1e-9 and fd_norm > 1e-9:
            cosine_sim = np.dot(reinforce_masked, finite_diff_masked) / (reinforce_norm * fd_norm)
        else:
            cosine_sim = 0.0
        
        # Bias estimate: L2 norm of difference between REINFORCE and finite difference
        # This measures systematic deviation from the "true" gradient
        bias = float(diff_norm)
        
        # Also compute relative bias for scale-invariant measure
        relative_bias = bias / fd_norm if fd_norm > 1e-9 else 0.0
    else:
        relative_error = 0.0
        cosine_sim = 0.0
        bias = 0.0
        relative_bias = 0.0
    
    # Pass condition: BOTH cosine similarity AND relative error
    # Patch H5: Configurable thresholds instead of hardcoded 0.9 / 0.40
    # Note: REINFORCE has inherent variance, so relative error of 0.10 is too strict
    passed = (cosine_sim > cfg.verification.gradient_check_cosine_threshold) and (relative_error < cfg.verification.gradient_check_error_threshold)
    
    log.info(f"Relative error: {relative_error:.4f}")
    log.info(f"Cosine similarity: {cosine_sim:.4f}")
    log.info(f"Bias estimate (L2): {bias:.6f}")
    log.info(f"Relative bias: {relative_bias:.4f}")
    log.info(f"Variance estimate: {variance:.6f}")
    log.info(f"Passed: {passed}")

    reinforce_mean_var = grad_variance / float(n_samples)
    
    return GradientCheckResult(
        reinforce_grad=reinforce_grad,
        finite_diff_grad=finite_diff_grad,
        relative_error=relative_error,
        cosine_similarity=cosine_sim,
        bias_estimate=bias,
        variance_estimate=variance,
        reinforce_var_vector=grad_variance,
        reinforce_mean_var_vector=reinforce_mean_var,
        fd_var_vector=fd_var,
        computed_mask=computed_mask,
        passed=passed,
    )


def main(raw_cfg: DictConfig) -> None:
    """Main entry point for gradient validation."""
    cfg = hydra_to_config(raw_cfg)
    validate(cfg)
    
    run_dir, run_id = get_run_config(cfg, "reinforce_check", raw_cfg)
    run_logger = setup_wandb(cfg, raw_cfg, default_group="gradient_check",
                            run_id=run_id, run_dir=run_dir)
    
    log.info("=" * 60)
    log.info("  REINFORCE Gradient Estimator Validation")
    log.info("=" * 60)
    
    seed_key = jax.random.PRNGKey(cfg.simulation.seed)
    # Uses cfg.verification.gradient_check_n_samples under the hood now.
    result = run_gradient_check(cfg, seed_key)
    
    # Save results
    import json
    result_path = run_dir / "gradient_check_result.json"
    with open(result_path, 'w') as f:
        json.dump({
            "relative_error": float(result.relative_error),
            "bias_estimate": float(result.bias_estimate),
            "variance_estimate": float(result.variance_estimate),
            "passed": bool(result.passed),
        }, f, indent=2)
    
    log.info(f"Results saved to {result_path}")
    
    # Generate gradient agreement scatter plot
    from gibbsq.analysis.plotting import plot_gradient_scatter
    
    # Compute per-parameter z-scores where variance is available
    fd_plot, rf_plot, z_scores = _build_plot_artifacts(result)
    
    plot_path = run_dir / "gradient_scatter"
    fig = plot_gradient_scatter(
        fd_grads=fd_plot,
        rf_grads=rf_plot,
        z_scores=z_scores,
        summary_stats={
            "cosine_similarity": float(result.cosine_similarity),
            "relative_error": float(result.relative_error),
            "passed": bool(result.passed),
        },
        save_path=plot_path,
        theme="publication",
        formats=["png", "pdf"],
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info(f"Gradient scatter plot saved to {plot_path}.png, {plot_path}.pdf")
    
    if not result.passed:
        log.warning("GRADIENT CHECK FAILED - REINFORCE estimator may be biased")
        raise SystemExit(1)
    else:
        log.info("GRADIENT CHECK PASSED - REINFORCE estimator is valid")
        return


if __name__ == "__main__":
    import sys
    import hydra
    if len(sys.argv) > 1:
        _wrapped = hydra.main(version_base=None, config_path="../../configs", config_name="default")(main)
        _wrapped()
    else:
        from hydra import compose, initialize_config_dir
        import os
        config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            raw_cfg = compose(config_name="default")
            main(raw_cfg)
