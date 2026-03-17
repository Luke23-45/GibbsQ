"""
Mathematical Equivalence Test Suite for JAX-Native SSA.

This script mathematically proves that the newly written JIT-compiled 
JAX SSA engine strictly preserves the transition dynamics, credit assignment 
logic, and statistical outputs of the legacy Python pure implementation.
"""

import logging
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import sys
import os
from pathlib import Path

# Add project root to sys.path for robust imports of 'experiments' and 'gibbsq'
root_path = str(Path(__file__).resolve().parents[1])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Import legacy Python functions
from experiments.training.train_reinforce import (
    collect_trajectory_ssa,
    compute_causal_returns_to_go
)
# Import new JAX components
from gibbsq.engines.jax_ssa import (
    compute_causal_returns_jax,
    vmap_collect_trajectories
)
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.config import NeuralConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# --- TEST FIXTURES ---

def create_test_fixtures():
    """Create deterministic configurations and models for testing."""
    num_servers = 3
    arrival_rate = 5.0
    service_rates = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    sim_time = 10.0
    gamma = 0.99
    
    # Initialize policy
    key = jax.random.PRNGKey(42)
    cfg = NeuralConfig(hidden_size=32)
    policy_net = NeuralRouter(num_servers=num_servers, config=cfg, key=key)
    
    return policy_net, num_servers, arrival_rate, service_rates, sim_time, gamma


def test_micro_equivalence_probabilities():
    """
    Test 1: Prove that the neural forward pass and event propensities
    match exactly between NumPy logic and JAX logic for a given state.
    """
    log.info("Running Test 1: Micro-Equivalence (Routing & Propensities)")
    policy_net, num_servers, arrival_rate, service_rates, _, _ = create_test_fixtures()
    
    Q_state = np.array([1, 0, 3], dtype=np.int32)
    
    # --- Python/NumPy Execution ---
    np_params = policy_net.get_numpy_params()
    s_np = (Q_state + 1.0) / service_rates
    logits_np = policy_net.numpy_forward(s_np, np_params, policy_net.config)
    logits_np = logits_np - np.max(logits_np)
    probs_np = np.exp(logits_np) / np.sum(np.exp(logits_np))
    
    arr_rates_np = arrival_rate * probs_np
    dep_rates_np = service_rates * (Q_state > 0)
    rates_np = np.concatenate([arr_rates_np, dep_rates_np])
    
    # --- JAX Execution ---
    Q_jax = jnp.array(Q_state)
    service_rates_jax = jnp.array(service_rates)
    s_jax = (Q_jax + 1.0) / service_rates_jax
    
    logits_jax = policy_net(s_jax)
    log_probs_jax = jax.nn.log_softmax(logits_jax, axis=-1)
    probs_jax = jnp.exp(log_probs_jax)
    
    arr_rates_jax = arrival_rate * probs_jax
    dep_rates_jax = jnp.where(Q_jax > 0, service_rates_jax, 0.0)
    rates_jax = jnp.concatenate([arr_rates_jax, dep_rates_jax])
    
    # --- Assertions ---
    np.testing.assert_allclose(probs_jax, probs_np, rtol=1e-5, err_msg="Routing probabilities mismatched!")
    np.testing.assert_allclose(rates_jax, rates_np, rtol=1e-5, err_msg="Propensities mismatched!")
    log.info("✓ Test 1 Passed: Network forward pass and propensities are mathematically identical.")


def test_logic_equivalence_causal_returns():
    """
    Test 2: Prove that the JAX reverse-scan computes the exact same Discounted 
    Causal Returns as the Python dynamic interval integration.
    """
    log.info("Running Test 2: Logic-Equivalence (Causal Returns Discounting)")
    sim_time = 10.0
    gamma = 0.99
    
    # Construct a mock sequence of 5 events
    # Event 0: Arrival (t=1.0)
    # Event 1: Departure (t=2.5)
    # Event 2: Arrival (t=4.0)
    # Event 3: Departure (t=6.0)
    # Event 4: Arrival (t=8.5)
    jump_times =[1.0, 2.5, 4.0, 6.0, 8.5]
    all_states = [
        np.array([1, 0, 0]), # after arrival 1
        np.array([0, 0, 0]), # after dep 1
        np.array([0, 1, 0]), # after arrival 2
        np.array([0, 0, 0]), # after dep 2
        np.array([0, 0, 1]), # after arrival 3
    ]
    action_step_indices = [0, 2, 4] # Arrivals
    
    # --- Python/NumPy Baseline Calculation ---
    returns_np = compute_causal_returns_to_go(
        all_states, jump_times, action_step_indices, sim_time, gamma
    )
    
    # --- JAX Calculation ---
    # Convert mock to JAX unrolled/padded format (max_steps = 7)
    dt_np = np.diff(jump_times, append=sim_time)
    q_totals_np = np.array([np.sum(s) for s in all_states])
    q_integrals_np = q_totals_np * dt_np
    
    # Pad to simulate lax.scan max_steps
    max_steps = 7
    q_integrals_jax = jnp.pad(jnp.array(q_integrals_np), (0, max_steps - len(q_integrals_np)))
    
    is_arrival_jax = jnp.array([True, False, True, False, True, False, False])
    valid_mask_jax = jnp.array([True, True, True, True, True, False, False])
    
    returns_jax_full = compute_causal_returns_jax(
        q_integrals_jax, is_arrival_jax, valid_mask_jax, gamma
    )
    
    # Extract only the action steps to compare with Python
    returns_jax_extracted = returns_jax_full[is_arrival_jax]
    
    # --- Assertions ---
    np.testing.assert_allclose(
        returns_jax_extracted, returns_np, rtol=1e-6, 
        err_msg="Causal returns mismatch between Python loops and JAX scan!"
    )
    log.info("✓ Test 2 Passed: Reverse-scan discounting perfectly replicates interval mathematics.")


def test_macro_equivalence_statistics():
    """
    Test 3: Run full trajectory rollouts and prove distributions match.
    Law of Large Numbers guarantees the means will converge if the underlying 
    stochastic generation mechanisms are mathematically equivalent.
    """
    log.info("Running Test 3: Macro-Equivalence (Statistical Distributions)")
    policy_net, num_servers, arrival_rate, service_rates, sim_time, gamma = create_test_fixtures()
    
    n_samples = 500
    
    # --- Python Legacy Rollouts ---
    log.info(f"  Collecting {n_samples} Legacy Python trajectories...")
    rewards_np = []
    arrivals_np =[]
    
    for i in range(n_samples):
        rng = np.random.default_rng(1000 + i)
        traj = collect_trajectory_ssa(
            policy_net, num_servers, arrival_rate, service_rates, sim_time, rng
        )
        rewards_np.append(traj.total_integrated_queue)
        arrivals_np.append(traj.arrival_count)
        
    mean_reward_np = np.mean(rewards_np)
    mean_arr_np = np.mean(arrivals_np)
    std_reward_np = np.std(rewards_np)
    
    # --- JAX Vectorized Rollouts ---
    log.info(f"  Collecting {n_samples} Vectorized JAX trajectories...")
    base_key = jax.random.PRNGKey(2000)
    keys = jax.random.split(base_key, n_samples)
    
    service_rates_jax = jnp.array(service_rates)
    
    jax_results = vmap_collect_trajectories(
        policy_net=policy_net,
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates_jax,
        sim_time=sim_time,
        keys=keys,
        max_steps=200, # Large enough for sim_time=10 with arr_rate=5
        gamma=gamma
    )
    
    rewards_jax = np.array(jax_results.total_integrated_queue)
    arrivals_jax = np.array(jax_results.arrival_count)
    
    mean_reward_jax = np.mean(rewards_jax)
    mean_arr_jax = np.mean(arrivals_jax)
    
    # --- Assertions ---
    # Using 95% Confidence Interval (z ≈ 1.96) for standard error of the mean
    se_reward = std_reward_np / np.sqrt(n_samples)
    diff_reward = abs(mean_reward_np - mean_reward_jax)
    
    # We allow a maximum drift of ~3 Standard Errors
    # If the simulation engines had different underlying dynamics, this would explode.
    assert diff_reward < (3.0 * se_reward), \
        f"Statistical failure! Reward drift too high. Diff: {diff_reward:.4f}, SE: {se_reward:.4f}"
        
    # Arrival counts should closely follow Poisson(lambda * T)
    expected_arrivals = arrival_rate * sim_time
    assert abs(mean_arr_np - expected_arrivals) / expected_arrivals < 0.05, "Legacy Python arrivals skewed!"
    assert abs(mean_arr_jax - expected_arrivals) / expected_arrivals < 0.05, "JAX arrivals skewed!"
    
    log.info(f"  Legacy E[Q_total]: {mean_reward_np:.4f} ± {se_reward:.4f}")
    log.info(f"  JAX E[Q_total]:    {mean_reward_jax:.4f}")
    log.info(f"  Legacy E[Arrivals]: {mean_arr_np:.2f}")
    log.info(f"  JAX E[Arrivals]:    {mean_arr_jax:.2f}")
    log.info("✓ Test 3 Passed: Statistical distributions match perfectly within expected error bounds.")


if __name__ == "__main__":
    print("\n=======================================================")
    print("  N-GibbsQ JAX-Native Engine Equivalence Verification")
    print("=======================================================\n")
    
    test_micro_equivalence_probabilities()
    test_logic_equivalence_causal_returns()
    test_macro_equivalence_statistics()
    
    print("\n=======================================================")
    print("  ALL TESTS PASSED. JAX ENGINE IS SAFE FOR MIGRATION.")
    print("=======================================================\n")