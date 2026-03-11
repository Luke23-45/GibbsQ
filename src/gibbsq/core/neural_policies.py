"""
Neural routing policies using Equinox.

Defines MLP-based routing agents that map queue-length vectors to
routing logits for use with the differentiable Gillespie engine.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

class NeuralRouter(eqx.Module):
    """
    MLP routing agent for CTMC queue-length vectors.

    Maps Q ∈ R^N to routing logits via log1p preprocessing (to compress
    large queue lengths) and a 3-layer MLP. Final layer is zero-initialized
    so initial output is uniform: softmax(0) = 1/N.
    """
    layers: list[eqx.Module]

    def __init__(
        self, 
        num_servers: int, 
        hidden_size: int = 64, 
        key: PRNGKeyArray = None
    ):
        """
        Args:
            num_servers: Number of servers (input/output dimension).
            hidden_size: Width of hidden layers.
            key: JAX PRNG key for parameter initialization.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
            
        keys = jax.random.split(key, 3)
        

        l1 = eqx.nn.Linear(num_servers, hidden_size, key=keys[0])
        l2 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        l3 = eqx.nn.Linear(hidden_size, num_servers, key=keys[2])
        
        # Zero-init final layer so initial output is uniform (softmax(0) = 1/N).
        l3_weights = jnp.zeros_like(l3.weight)
        l3_biases = jnp.zeros_like(l3.bias)
        l3 = eqx.tree_at(lambda m: m.weight, l3, l3_weights)
        l3 = eqx.tree_at(lambda m: m.bias, l3, l3_biases)
        
        self.layers = [l1, l2, l3]

    def __call__(self, Q: Float[Array, "num_servers"]) -> Float[Array, "num_servers"]:
        """Forward pass. Returns routing logits for Gumbel-Softmax relaxation."""
        # log1p compresses large queue lengths to prevent linear-layer saturation.
        x = jnp.log1p(Q) 
        

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
            

        logits = self.layers[-1](x)
        
        return logits
