"""
Shared policy-distribution helpers for Python, JAX, and evaluation wrappers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def stable_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    return probs / np.sum(probs)


def stable_softmax_jax(logits: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softmax(logits, axis=-1)


def compute_numpy_policy_probs(policy_net, Q, mu, rho, deterministic: bool = False) -> np.ndarray:
    """Compute the canonical NumPy policy distribution for a neural router."""
    if hasattr(policy_net, "get_numpy_params") and hasattr(policy_net, "numpy_forward") and hasattr(policy_net, "config"):
        np_params = policy_net.get_numpy_params()
        logits = policy_net.numpy_forward(Q, np_params, policy_net.config, rho=rho, mu=mu)
    else:
        logits = np.asarray(policy_net(np.asarray(Q), rho=rho, mu=mu), dtype=np.float64)
    probs = stable_softmax_numpy(logits)
    if deterministic:
        best_srv = int(np.argmax(probs))
        deterministic_probs = np.zeros_like(probs)
        deterministic_probs[best_srv] = 1.0
        return deterministic_probs
    return probs
