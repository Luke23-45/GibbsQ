
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from pathlib import Path
import sys

# Add project root to sys.path
root_path = str(Path(__file__).resolve().parents[0])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from gibbsq.core.config import ExperimentConfig, NeuralConfig
from gibbsq.core.neural_policies import NeuralRouter
from experiments.testing.reinforce_gradient_check import compute_reinforce_gradient, compute_finite_difference_gradient

def debug():
    num_servers = 3
    arrival_rate = 5.0
    service_rates = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    sim_time = 15.0
    base_seed = 42
    n_samples = 1500
    
    cfg = ExperimentConfig()
    cfg.system.num_servers = num_servers
    cfg.system.arrival_rate = arrival_rate
    cfg.system.service_rates = service_rates.tolist()
    
    key = jax.random.PRNGKey(base_seed)
    # Initialize with non-zero weights to break symmetry
    policy_net = NeuralRouter(num_servers=num_servers, config=cfg.neural, key=key)
    # Give it a little nudge
    from jax.flatten_util import ravel_pytree
    params, unravel = ravel_pytree(eqx.filter(policy_net, eqx.is_array))
    new_params = params + 0.1 * jax.random.normal(key, params.shape)
    policy_net = unravel(new_params)
    
    print("Computing REINFORCE grad...")
    r_grad, _, _ = compute_reinforce_gradient(
        policy_net, num_servers, arrival_rate, service_rates, sim_time, n_samples, base_seed
    )
    
    print("Computing FD grad...")
    fd_grad, mask = compute_finite_difference_gradient(
        policy_net, num_servers, arrival_rate, service_rates, sim_time, 0.05, n_samples, base_seed + 10000
    )
    
    r_masked = r_grad[mask]
    fd_masked = fd_grad[mask]
    
    print("\nResults:")
    print(f"REINFORCE grad (first 5): {r_masked[:5]}")
    print(f"Finite Diff grad (first 5): {fd_masked[:5]}")
    
    r_norm = np.linalg.norm(r_masked)
    fd_norm = np.linalg.norm(fd_masked)
    print(f"REINFORCE norm: {r_norm:.6f}")
    print(f"Finite Diff norm: {fd_norm:.6f}")
    
    if fd_norm > 0:
        error = np.linalg.norm(r_masked - fd_masked) / fd_norm
        print(f"Relative error: {error:.4f}")
    
    if r_norm > 0 and fd_norm > 0:
        cos_sim = np.dot(r_masked, fd_masked) / (r_norm * fd_norm)
        print(f"Cosine similarity: {cos_sim:.4f}")

if __name__ == "__main__":
    debug()
