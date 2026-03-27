"""
Test script to verify baseline computation bug (SG#3 / H#1).

This test demonstrates that the random baseline is incorrectly computed
AFTER bootstrapping, causing it to measure a JSQ-like policy instead of
true random routing.

Expected behavior:
- Random baseline should be HIGHER than JSQ baseline
- Current bug: random < JSQ (impossible)

Run with: python -m tests.test_baseline_bug
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path
import sys

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gibbsq.core.config import ExperimentConfig, SystemConfig, SimulationConfig, NeuralConfig
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.policies import JSSQRouting
from gibbsq.engines.numpy_engine import simulate


def compute_analytical_random_baseline(arrival_rate: float, service_rates: np.ndarray) -> float:
    """
    Compute analytical random baseline for asymmetric M/M/1 sum.
    
    E[Q] = Σ (λ/N) / (μ_i - λ/N)
    
    This is the CORRECT random baseline value.
    """
    N = len(service_rates)
    lam_i = arrival_rate / N
    total_q = 0.0
    
    for mu in service_rates:
        if lam_i >= mu:
            return float('inf')  # Unstable
        total_q += lam_i / (mu - lam_i)
    
    return total_q


def compute_jsq_baseline(arrival_rate: float, service_rates: np.ndarray, 
                         sim_time: float = 5000.0, seed: int = 42) -> float:
    """Compute JSQ baseline via simulation."""
    expert = JSSQRouting(service_rates)
    rng = np.random.default_rng(seed)
    
    res = simulate(
        num_servers=len(service_rates),
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        policy=expert,
        sim_time=sim_time,
        sample_interval=1.0,
        rng=rng
    )
    
    # Mean queue length
    states = res.states[len(res.states)//2:]  # Steady-state only
    return float(np.mean(np.sum(states, axis=-1)))


def compute_random_baseline_with_policy(policy_net: NeuralRouter, 
                                        arrival_rate: float,
                                        service_rates: np.ndarray,
                                        sim_time: float = 5000.0,
                                        seed: int = 42) -> float:
    """
    Compute "random" baseline using a policy network.
    
    This is the BUGGY method used in train_reinforce.py.
    If policy_net is already bootstrapped to JSQ, this will return
    a JSQ-like value, not random!
    """
    from gibbsq.core.features import look_ahead_potential
    
    rng = np.random.default_rng(seed)
    service_rates_jax = jnp.array(service_rates)
    
    # Run simulation with neural policy
    total_capacity = np.sum(service_rates)
    rho = arrival_rate / total_capacity
    
    # Simplified simulation for testing
    num_servers = len(service_rates)
    Q = np.zeros(num_servers, dtype=np.float64)
    total_queue = 0.0
    t = 0.0
    
    arrival_rate_total = arrival_rate
    service_rates_np = np.array(service_rates)
    total_rate = arrival_rate_total + np.sum(service_rates_np)
    
    while t < sim_time:
        # Next event time
        dt = rng.exponential(1.0 / total_rate)
        t += dt
        
        if t > sim_time:
            break
            
        total_queue += np.sum(Q) * dt
        
        # Event type
        if rng.random() < arrival_rate_total / total_rate:
            # Arrival - route using policy
            s_feat = look_ahead_potential(Q, service_rates_jax)
            logits = np.array(policy_net._single_forward(s_feat, rho))
            probs = jax.nn.softmax(logits)
            action = rng.choice(num_servers, p=np.array(probs))
            Q[action] += 1
        else:
            # Service completion
            active_servers = np.where(Q > 0)[0]
            if len(active_servers) > 0:
                server_rates = service_rates_np[active_servers]
                server = active_servers[rng.choice(len(active_servers), 
                                                    p=server_rates/np.sum(server_rates))]
                Q[server] -= 1
    
    return total_queue / sim_time


def test_baseline_ordering_bug():
    """
    Main test: Verify that baseline computation order matters.
    
    Expected:
    1. Random baseline with zero-initialized policy > JSQ baseline
    2. Random baseline with JSQ-bootstrapped policy ≈ JSQ baseline (BUG)
    """
    print("=" * 70)
    print("TEST: Baseline Computation Ordering Bug (SG#3 / H#1)")
    print("=" * 70)
    
    # Setup
    service_rates = np.array([1.0, 1.5])
    arrival_rate = 1.0  # rho = 0.4
    num_servers = 2
    
    print(f"\nConfig: N={num_servers}, λ={arrival_rate}, μ={service_rates}")
    print(f"Total capacity: {np.sum(service_rates)}")
    print(f"Load factor ρ: {arrival_rate / np.sum(service_rates):.2f}")
    
    # 1. Compute analytical baselines (ground truth)
    print("\n" + "-" * 70)
    print("1. ANALYTICAL BASELINES (Ground Truth)")
    print("-" * 70)
    
    analytical_random = compute_analytical_random_baseline(arrival_rate, service_rates)
    print(f"   Analytical Random E[Q]: {analytical_random:.4f}")
    print(f"   (Formula: Σ λ_i/(μ_i-λ_i) = 0.5/(1.0-0.5) + 0.5/(1.5-0.5) = 1.0 + 0.5)")
    
    # 2. Compute JSQ baseline via simulation
    print("\n" + "-" * 70)
    print("2. JSQ BASELINE (via simulation)")
    print("-" * 70)
    
    jsq_baseline = compute_jsq_baseline(arrival_rate, service_rates)
    print(f"   JSQ E[Q]: {jsq_baseline:.4f}")
    
    # 3. Create zero-initialized policy (TRUE random)
    print("\n" + "-" * 70)
    print("3. ZERO-INITIALIZED POLICY (True Random)")
    print("-" * 70)
    
    key = jax.random.PRNGKey(42)
    config = NeuralConfig()
    zero_policy = NeuralRouter(num_servers, config, service_rates, key)
    
    # Verify zero initialization produces uniform routing
    test_state = np.array([1.0, 1.0])
    s_feat = jnp.log1p(test_state)
    logits = zero_policy._single_forward(s_feat, 0.4)
    probs = jax.nn.softmax(logits)
    print(f"   Test state [1,1] routing probs: {np.array(probs)}")
    print(f"   (Should be ~uniform [0.5, 0.5] for zero-init)")
    
    random_baseline_zero_init = compute_random_baseline_with_policy(
        zero_policy, arrival_rate, service_rates, sim_time=500.0, seed=42
    )
    print(f"   Random E[Q] (zero-init policy): {random_baseline_zero_init:.4f}")
    
    # 4. Bootstrap policy to JSQ (simulates current bug)
    print("\n" + "-" * 70)
    print("4. JSQ-BOOTSTRAPPED POLICY (Simulating Bug)")
    print("-" * 70)
    
    from gibbsq.core.pretraining import train_robust_bc_policy
    key, train_key = jax.random.split(key)
    jsq_policy = train_robust_bc_policy(
        zero_policy, service_rates, train_key, num_steps=500
    )
    
    # Verify JSQ policy routes correctly
    test_state = np.array([0.0, 2.0])  # Server 0 has shorter queue
    s_feat = jnp.log1p(test_state)
    logits = jsq_policy._single_forward(s_feat, 0.4)
    probs = jax.nn.softmax(logits)
    print(f"   Test state [0,2] routing probs: {np.array(probs)}")
    print(f"   (Should favor server 0 with shorter queue)")
    
    random_baseline_jsq_init = compute_random_baseline_with_policy(
        jsq_policy, arrival_rate, service_rates, sim_time=500.0, seed=42
    )
    print(f"   'Random' E[Q] (JSQ-bootstrapped): {random_baseline_jsq_init:.4f}")
    
    # 5. VERDICT
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    print(f"\n   Analytical Random:  {analytical_random:.4f}")
    print(f"   JSQ Baseline:       {jsq_baseline:.4f}")
    print(f"   Zero-init 'Random': {random_baseline_zero_init:.4f}")
    print(f"   JSQ-init 'Random':  {random_baseline_jsq_init:.4f}")
    
    # Check conditions
    bug_detected = random_baseline_jsq_init < jsq_baseline * 1.1
    correct_behavior = random_baseline_zero_init > jsq_baseline
    
    print(f"\n   BUG DETECTED: {'YES' if bug_detected else 'NO'}")
    print(f"   (JSQ-bootstrapped 'random' < JSQ baseline)")
    
    print(f"\n   CORRECT BEHAVIOR: {'YES' if correct_behavior else 'NO'}")
    print(f"   (Zero-init random > JSQ baseline)")
    
    if bug_detected:
        print("\n   >>> HYPOTHESIS #1 CONFIRMED <<<")
        print("   The baseline computation order bug causes random < JSQ")
        print("   FIX: Compute random baseline BEFORE bootstrapping")
    
    return bug_detected


if __name__ == "__main__":
    bug_found = test_baseline_ordering_bug()
    sys.exit(0 if bug_found else 1)  # Exit 0 if bug found (test passed)
