import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import logging
from pathlib import Path
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.core.features import sojourn_time_features
from experiments.n_gibbsq.train_reinforce import collect_trajectory_ssa

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def profile_bottleneck():
    log.info("Starting strategic bottleneck profiling...")
    
    # Setup dummy environment
    num_servers = 8
    service_rates = np.random.uniform(0.5, 1.5, num_servers)
    arrival_rate = 0.8 * service_rates.sum()
    sim_time = 100.0  # Significant time for many events
    
    key = jax.random.PRNGKey(42)
    # Using a dummy config structure (minimal)
    class DummyCfg:
        class Neural:
            hidden_size = 64
            num_layers = 2
            init_type = "zero_final"
            activation = "relu"
            preprocessing = "log1p"
            capacity_bound = 100.0
        neural = Neural()
        class Sim:
            seed = 42
        simulation = Sim()
        
    cfg = DummyCfg()
    policy_net = NeuralRouter(num_servers=num_servers, config=cfg.neural, key=key)
    
    # 1. Warm-up JIT
    s_dummy = np.zeros(num_servers)
    mu_dummy = service_rates
    f = sojourn_time_features(s_dummy, mu_dummy)
    _ = policy_net(f)
    
    log.info("Starting simulation profile...")
    start_time = time.perf_counter()
    
    rng = np.random.default_rng(42)
    
    # Run the collector
    traj = collect_trajectory_ssa(
        policy_net=policy_net,
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rates=service_rates,
        sim_time=sim_time,
        rng=rng
    )
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    num_events = traj.arrival_count + traj.departure_count
    
    log.info("=" * 40)
    log.info(f"Profile Results:")
    log.info(f"Total Sim Time (HORIZON): {sim_time}")
    log.info(f"Actual CPU time: {total_time:.4f}s")
    log.info(f"Total events: {num_events}")
    log.info(f"Arrivals: {traj.arrival_count}")
    log.info(f"Departures: {traj.departure_count}")
    log.info(f"Time per event: {(total_time / num_events)*1000:.4f}ms")
    log.info(f"Events per second: {num_events / total_time:.1f}")
    log.info("=" * 40)
    
    # 2. Targeted Profiling of the Policy Call Alone
    log.info(f"Measuring raw policy call overhead (N={num_servers})...")
    iters = 1000
    p_start = time.perf_counter()
    for _ in range(iters):
        logits = policy_net(f)
        # Block until computed
        _ = logits.block_until_ready()
    p_end = time.perf_counter()
    log.info(f"Raw JAX policy call: {(p_end - p_start)/iters*1000:.4f}ms per call")
    
    # 3. Targeted Profiling with Device Get (the current implementation)
    log.info(f"Measuring policy call + jax.device_get overhead...")
    dg_start = time.perf_counter()
    for _ in range(iters):
        logits = np.asarray(jax.device_get(policy_net(f)))
    dg_end = time.perf_counter()
    log.info(f"Policy + jax.device_get: {(dg_end - dg_start)/iters*1000:.4f}ms per call")
    
    # 4. Targeted Profiling of the New NumPy Forward Pass
    log.info(f"Measuring NumPy fast forward overhead...")
    np_params = policy_net.get_numpy_params()
    np_config = policy_net.config
    nf_start = time.perf_counter()
    for _ in range(iters):
        logits = policy_net.numpy_forward(f, np_params, np_config)
    nf_end = time.perf_counter()
    log.info(f"NumPy fast forward: {(nf_end - nf_start)/iters*1000:.4f}ms per call")
    
    speedup = (dg_end - dg_start) / (nf_end - nf_start)
    log.info(f"Final Optimization Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    profile_bottleneck()
