"""
Neural routing policies using Equinox.

Defines MLP-based routing agents that map queue-length vectors to
routing logits for use with the differentiable Gillespie engine.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
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
        input_dim = num_servers + (1 if config.use_rho else 0)
        l1 = eqx.nn.Linear(input_dim, hidden_size, key=keys[0])
        l2 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        l3 = eqx.nn.Linear(hidden_size, num_servers, key=keys[2])
        
        # Optional Zero-initialization for final layer stability (defaults to zero_final)
        if config.init_type == "zero_final":
            l3 = eqx.tree_at(lambda l: l.weight, l3, jnp.zeros_like(l3.weight))
            if l3.bias is not None:
                l3 = eqx.tree_at(lambda l: l.bias, l3, jnp.zeros_like(l3.bias))
        
        self.layers = [l1, l2, l3]

    def __call__(
        self, 
        Q: Float[Array, "..."], 
        rho: Float[Array, "..."] = None
    ) -> Float[Array, "..."]:
        """Forward pass. Returns routing logits."""
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

        # PI-V4: Append rho to features if enabled
        if self.config.use_rho:
            # Handle both scalar and batched rho
            if rho is None:
                # Default case for backward compatibility in non-DR tests
                rho_feat = jnp.zeros_like(x[..., :1])
            else:
                rho_feat = jnp.atleast_1d(rho) * self.config.rho_input_scale
                # PATCH: Ensure proper broadcasting for all cases
                # Note: atleast_1d always returns ndim >= 1, so we only handle 1D case
                if x.ndim > 1:
                    # 1D rho array - handle broadcasting
                    if rho_feat.ndim == 1:
                        if rho_feat.shape[0] == x.shape[0]:
                            rho_feat = rho_feat[:, None]  # (batch, 1)
                        elif rho_feat.shape[0] == 1:
                            rho_feat = jnp.full((x.shape[0], 1), rho_feat[0], dtype=x.dtype)
                        else:
                            raise ValueError(f"rho shape {rho_feat.shape} incompatible with x shape {x.shape}")
                    else:
                        raise ValueError(f"rho must be scalar or 1D, got shape {rho_feat.shape}")
                else:  # x is 1D (scalar input)
                    if rho_feat.ndim > 1:
                        raise ValueError(f"rho must be scalar for scalar input, got shape {rho_feat.shape}")
                    rho_feat = rho_feat[:1]  # Ensure single feature
            
            x = jnp.concatenate([x, rho_feat], axis=-1)

        for i, layer in enumerate(self.layers[:-1]):
            x = jax.nn.relu(layer(x))

        logits = self.layers[-1](x)
        
        return logits

    def get_numpy_params(self):
        """Extract weights and biases as NumPy arrays for fast forward pass."""
        import numpy as np
        params = []
        for layer in self.layers:
            w = np.array(layer.weight)
            b = np.array(layer.bias) if layer.bias is not None else None
            params.append((w, b))
        return params

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Extract parameters as (weight, bias) tuples for numpy_forward."""
        params = []
        for layer in self.layers:
            w = np.array(layer.weight)
            b = np.array(layer.bias) if layer.bias is not None else None
            params.append((w, b))
        return params
    
    @staticmethod
    def numpy_forward(Q, params, config, rho=None):
        """Pure NumPy implementation of the forward pass for extreme speed in SSA loops."""
        import numpy as np
        
        # Preprocessing
        if config.preprocessing == "log1p":
            x = np.log1p(Q)
        elif config.preprocessing == "linear_min_max":
            x = Q / config.capacity_bound
        elif config.preprocessing == "standardize":
            mean_q = np.mean(Q)
            std_q = np.std(Q)
            x = (Q - mean_q) / (std_q + 1e-8)
        else:
            x = np.asarray(Q, dtype=np.float64)

        # PI-V4: Append rho
        if config.use_rho:
            # PATCH: Safe conversion that handles numpy/JAX arrays properly
            rho_val = 0.0 if rho is None else float(np.asarray(rho).item())
            rho_feat = np.array([rho_val * config.rho_input_scale], dtype=np.float64)
            
            # PATCH: Ensure proper dimensionality for all cases
            if x.ndim > 1:
                if rho_feat.ndim == 1:  # scalar rho
                    rho_feat = np.repeat(rho_feat[np.newaxis, :], x.shape[0], axis=0)
                elif rho_feat.ndim == 2 and rho_feat.shape[0] == 1:
                    rho_feat = np.repeat(rho_feat, x.shape[0], axis=0)
                elif rho_feat.shape[0] != x.shape[0]:
                    raise ValueError(f"rho shape {rho_feat.shape} incompatible with x shape {x.shape}")
            else:  # x is 1D
                if rho_feat.ndim > 1:
                    raise ValueError(f"rho must be scalar for scalar input, got shape {rho_feat.shape}")
                # Keep rho_feat as 1D for concatenation with 1D x
            
            x = np.concatenate([x, rho_feat], axis=-1)

        # MLP layers
        for i, (w, b) in enumerate(params):
            x = x @ w.T
            if b is not None:
                x = x + b
            if i < len(params) - 1:
                x = np.maximum(0, x)  # ReLU
        
        return x

class ValueNetwork(eqx.Module):
    """
    MLP value function approximator for baseline estimation.
    
    The value network V(s) estimates the expected total queue length
    from state s, used as a baseline in REINFORCE to reduce variance:
    
        ∇_θ J(θ) = E_π [ (R(τ) - V(s)) · ∇_θ log π_θ(a|s) ]
    
    This is the Actor-Critic framework, which provides lower-variance
    gradient estimates than vanilla REINFORCE.
    """
    layers: list[eqx.Module]
    config: NeuralConfig = eqx.field(static=True)
    
    def __init__(
        self,
        num_servers: int,
        config: NeuralConfig = None,
        hidden_size: int = 64,
        key: PRNGKeyArray = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        
        self.config = config
        # PI-V4: Append rho to features if enabled
        input_dim = num_servers + (1 if (config is not None and config.use_rho) else 0)
        
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(input_dim, hidden_size, key=keys[0]),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
            eqx.nn.Linear(hidden_size, 1, key=keys[2]),
        ]
    
    def __call__(
        self, 
        s: Float[Array, "dim"], 
        rho: Float[Array, ""] = None
    ) -> Float[Array, ""]:
        """Forward pass. Returns scalar value estimate V(s, rho).
        
        Handles both single inputs (num_servers,) and batched inputs (batch, num_servers).
        For batched inputs, automatically applies vmap internally.
        """
        x = s
        
        # Handle batched vs single input
        if x.ndim > 1:
            # Batched input: use vmap to process each sample
            # This is more efficient than manual loop and handles rho broadcasting
            return jax.vmap(lambda single_s: self._single_forward(single_s, rho))(x)
        else:
            # Single input
            return self._single_forward(x, rho)
    
    def _single_forward(self, x, rho):
        """Process a single (non-batched) input."""
        if self.config is not None and self.config.use_rho:
            rho_val = rho if rho is not None else 0.0
            rho_feat = jnp.atleast_1d(rho_val) * self.config.rho_input_scale
            # For single input, rho_feat is (1,) and x is (num_servers,)
            # Concatenate along the last axis
            if rho_feat.ndim > 1:
                rho_feat = rho_feat[:1]  # Take first element if somehow batched
            x = jnp.concatenate([x, rho_feat], axis=-1)

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x).squeeze()
