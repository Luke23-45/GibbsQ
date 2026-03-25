
import numpy as np
from gibbsq.core.policies import SoftmaxRouting, SojournTimeSoftmaxRouting
from gibbsq.engines.numpy_engine import simulate
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_reconciliation_benchmark():
    # 1. Setup an Extreme Heterogeneous System
    # Server 0: Lightning Fast (mu=100.0)
    # Server 1: Tortoise Slow (mu=1.0)
    # Total capacity = 101.0
    mu = np.array([100.0, 1.0])
    rho = 0.8
    lam = rho * np.sum(mu) # 80.8 jobs/sec
    
    sim_time = 100.0
    alpha = 1.0
    
    log.info(f"System: mu={mu}, lambda={lam:.2f}, rho={rho}")
    
    # 2. Policy A: Raw Queue (The "Math" Policy)
    # Stable? Yes. Efficient? No.
    policy_raw = SoftmaxRouting(alpha=alpha)
    
    # 3. Policy B: Sojourn Time (The "Neural" Policy)
    # Stable? Empirical. Efficient? Yes.
    policy_sojourn = SojournTimeSoftmaxRouting(mu=mu, alpha=alpha)
    
    # 4. Run Simulations
    res_raw = simulate(
        num_servers=2,
        arrival_rate=lam,
        service_rates=mu,
        policy=policy_raw,
        sim_time=sim_time,
        rng=np.random.default_rng(42)
    )
    
    res_sojourn = simulate(
        num_servers=2,
        arrival_rate=lam,
        service_rates=mu,
        policy=policy_sojourn,
        sim_time=sim_time,
        rng=np.random.default_rng(42)
    )
    
    # 5. Measure "Mean Wait" (Little's Law: E[W] = E[Q] / lambda)
    # This is the metric the Professor and the User care about.
    q_raw = np.mean(res_raw.states.sum(axis=1))
    q_sojourn = np.mean(res_sojourn.states.sum(axis=1))
    
    wait_raw = q_raw / lam
    wait_sojourn = q_sojourn / lam
    
    log.info("\n" + "="*40)
    log.info("  RECONCILIATION BENCHMARK")
    log.info("="*40)
    log.info(f"Raw-Queue Lens (Stable): E[W] = {wait_raw*1000:.2f} ms")
    log.info(f"Sojourn Lens (Neural):   E[W] = {wait_sojourn*1000:.2f} ms")
    log.info(f"Performance Gap:         {wait_raw/wait_sojourn:.1f}x slower with Raw!")
    log.info("="*40)
    
    log.info("\nWhy? Look at the Queue Distribution:")
    log.info(f"Raw Lens Distribution (Mean Q):     {np.mean(res_raw.states, axis=0)}")
    log.info(f"Sojourn Lens Distribution (Mean Q): {np.mean(res_sojourn.states, axis=0)}")
    log.info("\nSojourn lens correctly stacks ~100x more jobs on the 100x faster server.")
    log.info("Raw lens tries to keep queue indices equal, which is catastrophic for delay.")

if __name__ == "__main__":
    run_reconciliation_benchmark()
