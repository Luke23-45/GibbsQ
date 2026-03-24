"""
Model I/O utilities for GibbsQ neural policies.

Centralises model weight loading, pointer resolution, and the
``NeuralSSAPolicy`` bridge class that were previously duplicated
across ``stats_bench.py``, ``gen_sweep.py``, and ``critical_load.py``.

Usage
-----
::

    from gibbsq.utils.model_io import resolve_model_pointer, NeuralSSAPolicy

    weights_path = resolve_model_pointer(project_root, output_root)
    policy = NeuralSSAPolicy(model)
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ── Model Pointer Resolution ────────────────────────────────────

def resolve_model_pointer(
    project_root: Path,
    output_root: Path,
) -> Path:
    """Resolve the latest model weights path from pointer files.

    Searches for pointer files in priority order:
    1. ``latest_domain_randomized_weights.txt`` (DR-trained)
    2. ``latest_reinforce_weights.txt`` (REINFORCE-trained)
    3. ``latest_weights.txt`` (legacy DGA pointer)

    Parameters
    ----------
    project_root : Path
        Absolute path to the MoEQ project root.
    output_root : Path
        Directory containing the pointer files (typically ``run_dir.parent.parent``).

    Returns
    -------
    Path
        Absolute path to the model weights file.

    Raises
    ------
    FileNotFoundError
        If no valid pointer file is found.
    """
    candidates = [
        output_root / "latest_domain_randomized_weights.txt",
        output_root / "latest_reinforce_weights.txt",
        output_root / "latest_weights.txt",
    ]

    for ptr in candidates:
        if not ptr.exists():
            continue
        raw = Path(ptr.read_text(encoding="utf-8").strip())
        model_path = raw if raw.is_absolute() else (project_root / raw)
        if model_path.exists():
            if ptr.name == "latest_weights.txt":
                log.warning(
                    "Using legacy pointer latest_weights.txt; "
                    "prefer REINFORCE pointers."
                )
            return model_path

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "No valid model pointer found. Tried:\n"
        f"{tried}\n"
        "Run Track 1/3 training (reinforce_train or dr_train), "
        "or legacy train for latest_weights.txt."
    )


def resolve_model_pointer_or_none(
    project_root: Path,
    output_root: Path,
) -> Path | None:
    """Resolve model pointer, returning None if not found.
    
    PATCH H#3: Graceful fallback for missing pointers.
    Allows callers to create fresh models when no trained weights exist.
    
    Parameters
    ----------
    project_root : Path
        Project root directory for resolving relative paths.
    output_root : Path
        Directory containing pointer files (e.g., outputs/small).
    
    Returns
    -------
    Path | None
        Absolute path to model weights, or None if no pointer found.
    """
    candidates = [
        output_root / "latest_domain_randomized_weights.txt",
        output_root / "latest_reinforce_weights.txt",
        output_root / "latest_weights.txt",
    ]

    for ptr in candidates:
        if not ptr.exists():
            continue
        raw = Path(ptr.read_text(encoding="utf-8").strip())
        model_path = raw if raw.is_absolute() else (project_root / raw)
        if model_path.exists():
            return model_path

    log.warning(
        "No model pointer found in %s. "
        "Caller should create a fresh model.",
        output_root
    )
    return None


def save_model_pointer(
    model_path: Path,
    project_root: Path,
    output_root: Path,
    pointer_name: str = "latest_reinforce_weights.txt",
) -> Path:
    """Write a relative model pointer file.

    Parameters
    ----------
    model_path : Path
        Absolute path to the saved model weights.
    project_root : Path
        Project root for computing relative paths.
    output_root : Path
        Directory to write the pointer file into.
    pointer_name : str
        Filename for the pointer (e.g. ``"latest_reinforce_weights.txt"``).

    Returns
    -------
    Path
        Path to the written pointer file.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    ptr_path = output_root / pointer_name
    relative_path = model_path.resolve().relative_to(project_root)
    with open(ptr_path, "w", encoding="utf-8") as f:
        f.write(str(relative_path))
    log.info(f"[Pointer] Updated {pointer_name} at {ptr_path}")
    return ptr_path


# ── Neural Policy Validation ────────────────────────────────────

def validate_neural_model_shape(model, config, num_servers: int) -> None:
    """Validates that loaded neural model weights match the active configuration.
    
    Checks both the input dimension (accounting for queue lengths, service rates,
    and optional rho feature) and the hidden dimension.
    
    Parameters
    ----------
    model : eqx.Module
        The loaded NeuralRouter or ValueNetwork instance.
    config : NeuralConfig
        The active neural configuration.
    num_servers : int
        The number of servers in the system (`N`).
        
    Raises
    ------
    ValueError
        If either the input dimension or hidden size does not match the config.
    """
    # Input dimension: N (queue) + N (service rates) + (1 if use_rho)
    expected_input_dim = 2 * num_servers + (1 if config.use_rho else 0)
    try:
        actual_input_dim = model.layers[0].weight.shape[1]
    except (AttributeError, IndexError):
        # Fallback if layer structure doesn't match expectations
        return
        
    if actual_input_dim != expected_input_dim:
        raise ValueError(
            f"Model input dimension mismatch: loaded model expects {actual_input_dim}, "
            f"but active config requires {expected_input_dim} (N={num_servers}, use_rho={config.use_rho})."
        )
        
    expected_hidden = config.hidden_size
    try:
        actual_hidden = model.layers[0].weight.shape[0]
    except (AttributeError, IndexError):
        return
        
    if actual_hidden != expected_hidden:
         raise ValueError(
            f"Model hidden size mismatch: loaded model expects {actual_hidden}, "
            f"but active config expects {expected_hidden}."
        )


# ── Neural SSA Policy Bridge ────────────────────────────────────

class NeuralSSAPolicy:
    """Bridges a JAX ``NeuralRouter`` into the NumPy SSA engine.

    The SSA engine expects a callable ``(Q, rng) -> probs``. This class
    wraps a JAX model to provide that interface with:
    - JIT-compiled forward pass (via ``eqx.filter_jit``)
    - LRU-cached lookups for repeated queue states

    Previously duplicated as ``_NeuralSSAPolicy`` in:
    - ``stats_bench.py``
    - ``gen_sweep.py``
    - ``critical_load.py``
    """

    def __init__(self, model, mu=None, rho=None):
        import jax as _jax
        import jax.numpy as jnp
        import equinox as eqx

        self._model = model
        self.mu = jnp.array(mu, dtype=jnp.float32) if mu is not None else None
        self.active_rho = float(rho) if rho is not None else 0.0

        @eqx.filter_jit
        def _forward(m, x, r):
            return _jax.nn.softmax(m(x, rho=r))

        self._forward = _forward

        @functools.lru_cache(maxsize=131072)
        def _get_probs(q_tuple, rho_val):
            q_arr = jnp.array(q_tuple, dtype=jnp.float32)
            if self.mu is not None:
                s_feat = (q_arr + 1.0) / self.mu
            else:
                s_feat = q_arr
            r_tensor = jnp.array(rho_val, dtype=jnp.float32)
            probs = self._forward(self._model, s_feat, r_tensor)
            probs_np = np.array(probs, dtype=np.float64)
            return probs_np / probs_np.sum()

        self._get_probs = _get_probs

    def __call__(self, Q, rng):
        return self._get_probs(tuple(Q), self.active_rho)


class DeterministicNeuralPolicy:
    """Wraps a trained NeuralRouter to provide deterministic (greedy) actions.

    Uses argmax to remove policy sampling noise for final evaluations.

    Previously defined locally in ``baselines_comparison.py``.
    """

    def __init__(self, net, mu, rho=None):
        self._net = net
        self._mu = np.asarray(mu, dtype=np.float64)
        self._np_params = net.get_numpy_params()
        self._np_config = net.config
        self._num_servers = len(mu)
        # PATCH P2: Store service_rates for heterogeneity awareness
        self._service_rates = self._mu
        self._rho = float(rho) if rho is not None else 0.0

    def __call__(self, Q, rng):
        s = (np.asarray(Q, dtype=np.float64) + 1.0) / self._mu
        logits = self._net.numpy_forward(s, self._np_params, self._np_config, rho=self._rho, service_rates=self._service_rates)
        best_idx = int(np.argmax(logits))
        probs = np.zeros(self._num_servers, dtype=np.float64)
        probs[best_idx] = 1.0
        return probs


class StochasticNeuralPolicy:
    """Wraps a trained NeuralRouter for stochastic evaluation.

    Previously defined locally as ``NeuralPolicyWrapper`` in
    ``baselines_comparison.py`` and ``ablation_ssa.py``.
    """

    def __init__(self, net, mu, rho=None):
        self._net = net
        self._mu = np.asarray(mu, dtype=np.float64)
        self._np_params = net.get_numpy_params()
        self._np_config = net.config
        # PATCH P2: Store service_rates for heterogeneity awareness
        self._service_rates = self._mu
        self._rho = float(rho) if rho is not None else 0.0

    def __call__(self, Q, rng):
        s = (np.asarray(Q, dtype=np.float64) + 1.0) / self._mu
        logits = self._net.numpy_forward(s, self._np_params, self._np_config, rho=self._rho, service_rates=self._service_rates)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        return probs / probs.sum()
