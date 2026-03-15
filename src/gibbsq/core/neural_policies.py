"""
Neural routing policies using Equinox.

Defines MLP-based routing agents that map queue-length vectors to
routing logits for use with the differentiable Gillespie engine.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from gibbsq.core.config import NeuralConfig
from gibbsq.core import constants

class NeuralRouter(eqx.Module):
    """
    MLP routing agent for CTMC queue-length vectors.

    Maps Q ∈ R^N to routing logits via log1p preprocessing (to compress
    large queue lengths) and a 3-layer MLP. Final layer is zero-initialized
    so initial output is uniform: softmax(0) = 1/N.
    """
    layers: list[eqx.Module]
    config: NeuralConfig = eqx.field(static=True)

    def __init__(
        self, 
        num_servers: int, 
        config: NeuralConfig, 
        key: PRNGKeyArray = None
    ):
        """
        Args:
            num_servers: Number of servers (input/output dimension).
            config: Configuration object (hidden_size, preprocessing, init_type).
            key: JAX PRNG key for parameter initialization.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
            
        keys = jax.random.split(key, 3)
        

        self.config = config
        hidden_size = config.hidden_size
        l1 = eqx.nn.Linear(num_servers, hidden_size, key=keys[0])
        l2 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        l3 = eqx.nn.Linear(hidden_size, num_servers, key=keys[2])
        
        # Optional Zero-initialization for final layer stability (defaults to zero_final)
        if config.init_type == "zero_final":
            l3 = eqx.tree_at(lambda l: l.weight, l3, jnp.zeros_like(l3.weight))
            if l3.bias is not None:
                l3 = eqx.tree_at(lambda l: l.bias, l3, jnp.zeros_like(l3.bias))
        
        self.layers = [l1, l2, l3]

    def __call__(self, Q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Forward pass. Returns routing logits for Gumbel-Softmax relaxation."""
        # Preprocessing selection based on config
        if self.config.preprocessing == "log1p":
            x = jnp.log1p(Q)
        elif self.config.preprocessing == "linear_min_max":
            # Normalized by theoretical capacity bound
            x = Q / self.config.capacity_bound
        elif self.config.preprocessing == "standardize":
            # Z-score normalization across the N server queue lengths.
            # When all queues are equal (std == 0), numerator is zero so x == 0
            # regardless of epsilon, giving uniform routing logits — correct behavior.
            mean_q = jnp.mean(Q)
            std_q = jnp.std(Q)
            x = (Q - mean_q) / (std_q + constants.NUMERICAL_STABILITY_EPSILON)
        else:  # preprocessing == "none"
            x = Q

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
            

        logits = self.layers[-1](x)
        
        return logits
