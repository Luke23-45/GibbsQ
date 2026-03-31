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

import hashlib
import json
import functools
import logging
from pathlib import Path

import numpy as np
from gibbsq.core.policy_distribution import compute_numpy_policy_probs
log = logging.getLogger(__name__)

REINFORCE_POINTER = "latest_reinforce_weights.txt"
DOMAIN_RANDOMIZED_POINTER = "latest_domain_randomized_weights.txt"
BC_POINTER = "latest_bc_weights.txt"
LEGACY_POINTER = "latest_weights.txt"
BC_METADATA_SUFFIX = ".bc_metadata.json"

def _resolve_from_candidates(project_root: Path, candidates: list[Path]) -> Path | None:
    for ptr in candidates:
        if not ptr.exists():
            continue
        raw = Path(ptr.read_text(encoding="utf-8").strip())
        model_path = raw if raw.is_absolute() else (project_root / raw)
        if model_path.exists():
            if ptr.name == LEGACY_POINTER:
                log.warning(
                    "Using legacy pointer latest_weights.txt; "
                    "prefer public training pointers "
                    "(latest_reinforce_weights.txt or latest_bc_weights.txt)."
                )
            return model_path
    return None

def resolve_model_pointer(
    project_root: Path,
    output_root: Path,
    *,
    allow_bc: bool = True,
    allow_legacy: bool = True,
) -> Path:
    """Resolve the latest model weights path from pointer files.

    Searches for pointer files in priority order:
    1. ``latest_reinforce_weights.txt`` (REINFORCE-trained)
    2. ``latest_domain_randomized_weights.txt`` (legacy DR-trained naming)
    3. ``latest_bc_weights.txt`` (BC warm-start only; optional)
    4. ``latest_weights.txt`` (legacy DGA pointer; optional)

    Parameters
    ----------
    project_root : Path
        Absolute path to the GibbsQ project root.
    output_root : Path
        Directory containing the pointer files (typically ``run_dir.parent.parent``).
    allow_bc : bool
        Whether ``latest_bc_weights.txt`` is an acceptable fallback.
    allow_legacy : bool
        Whether ``latest_weights.txt`` is an acceptable fallback.

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
        output_root / REINFORCE_POINTER,
        output_root / DOMAIN_RANDOMIZED_POINTER,
    ]
    if allow_bc:
        candidates.append(output_root / BC_POINTER)
    if allow_legacy:
        candidates.append(output_root / LEGACY_POINTER)

    model_path = _resolve_from_candidates(project_root, candidates)
    if model_path is not None:
        return model_path

    tried = "\n".join(f"  - {c}" for c in candidates)
    if not allow_bc:
        tried += f"\n  - {output_root / BC_POINTER} (rejected: BC warm-start only)"
    if not allow_legacy:
        tried += f"\n  - {output_root / LEGACY_POINTER} (rejected: legacy pointer not allowed)"
    raise FileNotFoundError(
        "No valid model pointer found. Tried:\n"
        f"{tried}\n"
        "Run a public training entry point "
        "(reinforce_train or bc_train), or intentionally supply "
        "the legacy latest_weights.txt pointer."
    )

def resolve_model_pointer_or_none(
    project_root: Path,
    output_root: Path,
    *,
    allow_bc: bool = True,
    allow_legacy: bool = True,
) -> Path | None:
    """Resolve model pointer, returning None if not found.
    
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
        output_root / REINFORCE_POINTER,
        output_root / DOMAIN_RANDOMIZED_POINTER,
    ]
    if allow_bc:
        candidates.append(output_root / BC_POINTER)
    if allow_legacy:
        candidates.append(output_root / LEGACY_POINTER)

    model_path = _resolve_from_candidates(project_root, candidates)
    if model_path is not None:
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
    pointer_name: str = REINFORCE_POINTER,
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

def build_bc_reuse_compatibility_record(
    cfg,
    *,
    bc_data_config: dict | None,
) -> dict:
    """Build the canonical BC reuse contract for metadata/fingerprint checks."""
    from gibbsq.core.pretraining import normalize_bc_data_config

    normalized_bc_data = normalize_bc_data_config(bc_data_config)
    compatibility = {
        "schema_version": 1,
        "artifact_kind": "bc_actor_warm_start",
        "system": {
            "num_servers": int(cfg.system.num_servers),
            "service_rates": [float(x) for x in cfg.system.service_rates],
            "alpha": float(cfg.system.alpha),
        },
        "neural": {
            "hidden_size": int(cfg.neural.hidden_size),
            "preprocessing": str(cfg.neural.preprocessing),
            "capacity_bound": float(cfg.neural.capacity_bound),
            "init_type": str(cfg.neural.init_type),
            "use_rho": bool(cfg.neural.use_rho),
            "use_service_rates": bool(cfg.neural.use_service_rates),
            "rho_input_scale": float(cfg.neural.rho_input_scale),
        },
        "bc_data": {
            "rhos": [float(x) for x in normalized_bc_data["rhos"]],
            "mu_scales": [float(x) for x in normalized_bc_data["mu_scales"]],
            "samples_per_rho": int(normalized_bc_data["samples_per_rho"]),
            "expert_sim_time": float(normalized_bc_data["expert_sim_time"]),
            "sample_interval": float(normalized_bc_data["sample_interval"]),
            "augmentation_noise_min": int(normalized_bc_data["augmentation_noise_min"]),
            "augmentation_noise_max": int(normalized_bc_data["augmentation_noise_max"]),
        },
    }
    canonical = json.dumps(compatibility, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "artifact_kind": "bc_actor_warm_start",
        "compatibility": compatibility,
        "provenance": {
            "simulation_seed": int(cfg.simulation.seed),
            "neural_training": {
                "bc_num_steps": int(cfg.neural_training.bc_num_steps),
                "bc_lr": float(cfg.neural_training.bc_lr),
                "bc_label_smoothing": float(cfg.neural_training.bc_label_smoothing),
                "weight_decay": float(cfg.neural_training.weight_decay),
            },
        },
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }

def get_bc_metadata_path(model_path: Path) -> Path:
    """Return the sidecar metadata path for a BC warm-start weights file."""
    return model_path.with_name(f"{model_path.name}{BC_METADATA_SUFFIX}")

def write_bc_reuse_metadata(
    model_path: Path,
    *,
    cfg,
    bc_data_config: dict | None,
) -> Path:
    """Write BC warm-start metadata used by reinforce_train compatibility checks."""
    metadata_path = get_bc_metadata_path(model_path)
    metadata = build_bc_reuse_compatibility_record(cfg, bc_data_config=bc_data_config)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path

def load_bc_reuse_metadata(model_path: Path) -> dict:
    """Load BC warm-start metadata from the sidecar JSON file."""
    metadata_path = get_bc_metadata_path(model_path)
    return json.loads(metadata_path.read_text(encoding="utf-8"))

def validate_bc_reuse_metadata(
    model_path: Path,
    *,
    cfg,
    bc_data_config: dict | None,
) -> dict:
    """Validate that a BC artifact matches the active training contract."""
    expected = build_bc_reuse_compatibility_record(cfg, bc_data_config=bc_data_config)
    actual = load_bc_reuse_metadata(model_path)
    if actual.get("schema_version") != 1:
        raise ValueError(f"Unsupported BC metadata schema_version: {actual.get('schema_version')}")
    if actual.get("artifact_kind") != "bc_actor_warm_start":
        raise ValueError(f"Unsupported BC metadata artifact_kind: {actual.get('artifact_kind')}")

    actual_compatibility = actual.get("compatibility")
    if not isinstance(actual_compatibility, dict):
        raise ValueError("BC metadata is missing the compatibility section.")

    actual_fingerprint = actual.get("fingerprint")
    canonical_actual = json.dumps(actual_compatibility, sort_keys=True, separators=(",", ":"))
    recomputed_actual = hashlib.sha256(canonical_actual.encode("utf-8")).hexdigest()
    if actual_fingerprint != recomputed_actual:
        raise ValueError(
            "BC metadata fingerprint does not match the stored compatibility payload."
        )
    if actual_fingerprint != expected["fingerprint"]:
        raise ValueError(
            "BC metadata fingerprint mismatch: "
            f"expected {expected['fingerprint']}, got {actual_fingerprint}."
        )
    return actual

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
    expected_input_dim = num_servers
    if getattr(config, "use_service_rates", True):
        expected_input_dim += num_servers
    if config.use_rho:
        expected_input_dim += 1
    try:
        actual_input_dim = model.layers[0].weight.shape[1]
    except (AttributeError, IndexError):
        return
        
    if actual_input_dim != expected_input_dim:
        raise ValueError(
            f"Model input dimension mismatch: loaded model expects {actual_input_dim}, "
            "but active config requires "
            f"{expected_input_dim} (N={num_servers}, "
            f"use_service_rates={getattr(config, 'use_service_rates', True)}, "
            f"use_rho={config.use_rho})."
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
        def _forward(m, x, r, mu_val):
            # m(x, ...) now returns logits. We pass mu to the model.
            return m(x, rho=r, mu=mu_val)

        self._forward = _forward

        @functools.lru_cache(maxsize=131072)
        def _get_probs(q_tuple, rho_val):
            return compute_numpy_policy_probs(
                self._model,
                np.array(q_tuple, dtype=np.float64),
                np.array(self.mu),
                rho_val,
            )

        self._get_probs = _get_probs

    def __call__(self, Q, rng):
        return self._get_probs(tuple(Q), self.active_rho)

def build_neural_eval_policy(model, mu, rho=None, mode: str = "deterministic"):
    """Create an explicit neural evaluation policy wrapper.

    Parameters
    ----------
    model : eqx.Module
        Trained neural router.
    mu : array-like
        Service-rate vector for the active evaluation regime.
    rho : float, optional
        Load factor for feature construction.
    mode : {"deterministic", "stochastic"}
        Evaluation mode. Deterministic uses greedy argmax routing,
        stochastic samples from the learned policy distribution.
    """
    if mode == "deterministic":
        return DeterministicNeuralPolicy(model, mu, rho=rho)
    if mode == "stochastic":
        return NeuralSSAPolicy(model, mu=mu, rho=rho)
    raise ValueError(
        f"Unknown neural evaluation mode '{mode}'. "
        "Expected 'deterministic' or 'stochastic'."
    )

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
        self._service_rates = self._mu
        self._rho = float(rho) if rho is not None else 0.0

    def __call__(self, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mu_val = getattr(self, '_mu', getattr(self, '_service_rates', None))
        return compute_numpy_policy_probs(self._net, Q, mu_val, self._rho, deterministic=True)

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
        self._service_rates = self._mu
        self._rho = float(rho) if rho is not None else 0.0

    def __call__(self, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mu_val = getattr(self, '_mu', getattr(self, '_service_rates', None))
        return compute_numpy_policy_probs(self._net, Q, mu_val, self._rho, deterministic=False)
