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
    
    PATCH P2 (H5): Heterogeneity-aware routing - includes normalized service rates
    as input features so the policy can learn to prefer faster servers.
    """
    layers: list[eqx.Module]
    config: NeuralConfig = eqx.field(static=True)
    service_rates: Float[Array, "N"]
    num_servers: int = eqx.field(static=True)

    def __init__(
        self, 
        num_servers: int, 
        config: NeuralConfig, 
        service_rates: Float[Array, "N"] = None,
        key: PRNGKeyArray = None
    ):
        """
        Args:
            num_servers: Number of servers (input/output dimension).
            config: Configuration object (hidden_size, preprocessing, init_type).
            service_rates: Service rates for each server (for heterogeneity awareness).
            key: JAX PRNG key for parameter initialization.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
            
        keys = jax.random.split(key, 3)
        

        self.config = config
        self.num_servers = num_servers
        
        # PATCH P2: Store service rates for heterogeneity-aware features
        if service_rates is None:
            self.service_rates = jnp.ones(num_servers)
        else:
            self.service_rates = jnp.asarray(service_rates)
        
        hidden_size = config.hidden_size
        # PATCH P2: Input dim includes service rate features (num_servers for mu_normalized)
        input_dim = num_servers + num_servers + (1 if config.use_rho else 0)
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
        # Dispatch to _single_forward which handles all preprocessing.
        # Batched inputs are handled via vmap.
        is_batched = Q.ndim > 1
        
        if is_batched:
            if rho is None:
                return jax.vmap(lambda q: self._single_forward(q, None))(Q)
            elif hasattr(rho, 'ndim') and rho.ndim > 0:
                return jax.vmap(self._single_forward, in_axes=(0, 0))(Q, rho)
            else:
                return jax.vmap(lambda q: self._single_forward(q, rho))(Q)
        else:
            return self._single_forward(Q, rho)
    
    def _single_forward(self, Q, rho):
        """Process a single (non-batched) input."""
        # Preprocessing selection based on config
        if self.config.preprocessing == "log1p":
            x = jnp.log1p(Q)
        elif self.config.preprocessing == "linear_min_max":
            x = Q / self.config.capacity_bound
        elif self.config.preprocessing == "standardize":
            mean_q = jnp.mean(Q)
            std_q = jnp.std(Q)
            x = (Q - mean_q) / (std_q + constants.NUMERICAL_STABILITY_EPSILON)
        else:
            x = Q
        
        # PATCH P2: Append normalized service rates for heterogeneity awareness
        mu_sum = jnp.sum(self.service_rates)
        mu_normalized = self.service_rates / jnp.where(mu_sum > 0, mu_sum, 1.0)
        x = jnp.concatenate([x, mu_normalized])
        
        # Append rho feature
        if self.config.use_rho:
            if rho is None:
                rho_feat = jnp.zeros((1,), dtype=x.dtype)
            else:
                rho_feat = jnp.atleast_1d(rho * self.config.rho_input_scale)
                if rho_feat.shape[0] > 1:
                    rho_feat = rho_feat[:1]  # Take first if somehow batched
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
    def numpy_forward(Q, params, config, rho=None, service_rates=None):
        """Pure NumPy implementation of the forward pass for extreme speed in SSA loops.
        
        PATCH P2: Added service_rates parameter for heterogeneity-aware routing.
        """
        import numpy as np
        
        num_servers = Q.shape[-1] if hasattr(Q, 'shape') else len(Q)
        
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

        # PATCH P2: Append normalized service rates for heterogeneity awareness
        if service_rates is None:
            mu_normalized = np.ones(num_servers) / num_servers
        else:
            mu = np.asarray(service_rates, dtype=np.float64)
            mu_sum = np.sum(mu)
            mu_normalized = mu / np.where(mu_sum > 0, mu_sum, 1.0)
        x = np.concatenate([x, mu_normalized])
        
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
        
        ROBUST FIX: Handle both single and batched inputs correctly.
        Single input: s shape (N,), rho scalar or None
        Batched input: s shape (batch, N), rho shape (batch,) or scalar
        Equinox Linear layers require single inputs, so batched calls use internal vmap.
        """
        is_batched = s.ndim > 1
        
        if is_batched:
            # Batched input: use vmap with correct in_axes
            if rho is None:
                # Broadcast None to all batch elements
                return jax.vmap(lambda single_s: self._single_forward(single_s, None))(s)
            elif hasattr(rho, 'ndim') and rho.ndim > 0:
                # rho is array - map over both s and rho
                return jax.vmap(self._single_forward, in_axes=(0, 0))(s, rho)
            else:
                # rho is scalar - broadcast to all batch elements
                return jax.vmap(lambda single_s: self._single_forward(single_s, rho))(s)
        else:
            # Single input: direct call
            return self._single_forward(s, rho)
    
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


# ─────────────────────────────────────────────────────────────────────────────
# PATCH P4: Adaptive Temperature for Load-Dependent Routing
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_alpha(rho: float, base_alpha: float = 1.0) -> float:
    """
    Compute temperature based on load factor for ROUTING.
    
    Temperature interpretation for softmax routing:
    - alpha → ∞: uniform distribution (exploratory)
    - alpha → 0: argmax (greedy, JSQ-like)
    - alpha = 1: standard softmax
    
    For routing:
    - Low load (rho<0.7): high alpha (exploratory is fine, no congestion)
    - Medium load (0.7<=rho<0.85): moderate alpha
    - Heavy load (0.85<=rho<0.95): lower alpha (more greedy)
    - Critical load (rho>=0.95): very low alpha (JSQ-like greedy needed)
    
    Args:
        rho: Current load factor (0 < rho < 1)
        base_alpha: Base temperature to scale from (default 1.0)
        
    Returns:
        Adaptive temperature alpha (higher = more exploratory, lower = more greedy)
    """
    if rho < 0.70:
        # Low load: exploratory routing is fine
        return base_alpha * 2.0
    elif rho < 0.85:
        # Medium load: moderate exploration
        return base_alpha * 1.0
    elif rho < 0.95:
        # Heavy load: more greedy
        return base_alpha * 0.5
    else:
        # Critical load: JSQ-like greedy routing
        # Ramp down further as rho approaches 1.0
        return base_alpha * max(0.1, 0.5 - 5.0 * (rho - 0.95))
