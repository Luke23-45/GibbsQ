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

    def __init__(self, model):
        import jax as _jax
        import jax.numpy as jnp
        import equinox as eqx

        self._model = model

        @eqx.filter_jit
        def _forward(m, x):
            return _jax.nn.softmax(m(x))

        self._forward = _forward

        @functools.lru_cache(maxsize=131072)
        def _get_probs(q_tuple):
            probs = self._forward(self._model, jnp.array(q_tuple, dtype=jnp.float32))
            probs_np = np.array(probs, dtype=np.float64)
            return probs_np / probs_np.sum()

        self._get_probs = _get_probs

    def __call__(self, Q, rng):
        return self._get_probs(tuple(Q))


class DeterministicNeuralPolicy:
    """Wraps a trained NeuralRouter to provide deterministic (greedy) actions.

    Uses argmax to remove policy sampling noise for final evaluations.

    Previously defined locally in ``baselines_comparison.py``.
    """

    def __init__(self, net, mu):
        self._net = net
        self._mu = np.asarray(mu, dtype=np.float64)
        self._np_params = net.get_numpy_params()
        self._np_config = net.config
        self._num_servers = len(mu)

    def __call__(self, Q, rng):
        s = (np.asarray(Q, dtype=np.float64) + 1.0) / self._mu
        logits = self._net.numpy_forward(s, self._np_params, self._np_config)
        best_idx = int(np.argmax(logits))
        probs = np.zeros(self._num_servers, dtype=np.float64)
        probs[best_idx] = 1.0
        return probs


class StochasticNeuralPolicy:
    """Wraps a trained NeuralRouter for stochastic evaluation.

    Previously defined locally as ``NeuralPolicyWrapper`` in
    ``baselines_comparison.py`` and ``ablation_ssa.py``.
    """

    def __init__(self, net, mu):
        self._net = net
        self._mu = np.asarray(mu, dtype=np.float64)
        self._np_params = net.get_numpy_params()
        self._np_config = net.config

    def __call__(self, Q, rng):
        s = (np.asarray(Q, dtype=np.float64) + 1.0) / self._mu
        logits = self._net.numpy_forward(s, self._np_params, self._np_config)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        return probs / probs.sum()
