"""
Empirical comparison of Softmax Routing Variants.
Resolves Theory vs Implementation mismatch (SG#5 & SG#7).

Variants tested:
1. Raw Queue (Paper docs/gibbsq.md): exp(-alpha * Q)
2. Sojourn Time (Implementation drift.py/policies.py): exp(-alpha * (Q+1)/mu)
3. Normalized Q (Implementation jax_engine.py): exp(-alpha * Q/mu)

Evaluated at high load on a heterogeneous server cluster.
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
from gibbsq.engines.jax_engine import run_replications_jax
from gibbsq.engines.numpy_engine import SimResult
from gibbsq.analysis.metrics import time_averaged_queue_lengths

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def softmax_probs_raw(Q, alpha, mu):
    logits = -alpha * Q
    logits = logits - jnp.max(logits)
    exp_logits = jnp.exp(logits)
    return exp_logits / jnp.sum(exp_logits)

def softmax_probs_sojourn(Q, alpha, mu):
    logits = -alpha * (Q + 1.0) / mu
    logits = logits - jnp.max(logits)
    exp_logits = jnp.exp(logits)
    return exp_logits / jnp.sum(exp_logits)

def softmax_probs_normalized(Q, alpha, mu):
    logits = -alpha * (Q / mu)
    logits = logits - jnp.max(logits)
    exp_logits = jnp.exp(logits)
    return exp_logits / jnp.sum(exp_logits)

def main():
    num_servers = 10
    # Highly heterogeneous servers
    mu = jnp.array([1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    total_capacity = jnp.sum(mu)
    rho = 0.95
    lam = rho * total_capacity
    alpha = 0.5
    sim_time = 1000.0
    sample_interval = 1.0
    num_replications = 10
    max_samples = int(sim_time / sample_interval) + 1

    variants = {
        "Raw Queues (Standard Softmax)": 3,
        "Sojourn Time (Look-Ahead Potential)": 5,
        "UAS (Full Archimedean)": 6
    }

    log.info(f"Heterogeneous Cluster Eval: N={num_servers}, Capacity={total_capacity:.1f}, Load={rho}")

    from gibbsq.engines.jax_engine import run_replications_jax
    from gibbsq.engines.numpy_engine import SimResult
    from gibbsq.analysis.metrics import time_averaged_queue_lengths

    for name, p_type in variants.items():
        log.info(f"\n--- Testing Variant: {name} (policy_type={p_type}) ---")
        times, states, (arrs, deps) = run_replications_jax(
            num_replications=num_replications,
            num_servers=num_servers,
            arrival_rate=lam,
            service_rates=mu,
            alpha=alpha,
            sim_time=sim_time,
            sample_interval=sample_interval,
            base_seed=42,
            max_samples=max_samples,
            policy_type=p_type
        )
        
        q_totals = []
        q_per_server = []
        for r in range(num_replications):
            _np_times = np.array(times[r])
            _np_states = np.array(states[r])
            # Truncate invalid trailing JAX buffer slots (SG#5 fix)
            _valid_mask = _np_times > 0
            _valid_mask[0] = True
            _vl = int(np.sum(_valid_mask))
            if _vl < _np_states.shape[0]:
                _np_times = _np_times[:_vl]
                _np_states = _np_states[:_vl]
            res = SimResult(
                times=_np_times,
                states=_np_states,
                arrival_count=int(arrs[r]),
                departure_count=int(deps[r]),
                final_time=float(_np_times[-1]),
                num_servers=num_servers
            )
            q_avg = time_averaged_queue_lengths(res, 0.2) # drops first 20%
            q_totals.append(q_avg.sum())
            q_per_server.append(q_avg)
            
        mean_q = float(np.mean(q_totals))
        std_q = float(np.std(q_totals))
        mean_q_per_server = np.mean(q_per_server, axis=0)
        max_q = float(np.max(mean_q_per_server))
        
        # Calculate max-min queue imbalance
        imbalance = max_q - float(np.min(mean_q_per_server))
        
        log.info(f"Result: E[Q_total] = {mean_q:.2f} ± {std_q:.2f}")
        log.info(f"Result: Max E[Q_i] = {max_q:.2f}")
        log.info(f"Result: Queue Imbalance (Max-Min) = {imbalance:.2f}")
        log.info(f"Result: Mean E[Q_i] per server = {np.array2string(mean_q_per_server, precision=1, floatmode='fixed')}")

if __name__ == "__main__":
    main()
