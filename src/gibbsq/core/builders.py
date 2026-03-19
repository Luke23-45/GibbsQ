"""
Type-safe Builder functions for GibbsQ components.

Builders consume strictly-typed Dataclass slices from the configuration
and produce fully-instantiated simulation objects. This replaces the
pattern of passing the entire monolithic ``ExperimentConfig`` into every
function and manually dispatching on magic strings.

Usage
-----
::

    from gibbsq.core.builders import build_policy, build_engine_runner

    policy = build_policy(cfg.policy, cfg.system)
    runner = build_engine_runner(cfg)

Design Rationale
----------------
- **Explicit Dependencies**: Each builder declares exactly which config
  slices it needs. MyPy and IDEs can verify these statically.
- **Registry-Backed**: Builders delegate to ``ComponentRegistry`` for
  component lookup, so adding a new policy requires zero changes here.
- **Engine Dispatch**: Replaces the scattered ``if cfg.jax.enabled:``
  blocks with a single centralized decision point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gibbsq.core.registry import ComponentRegistry

# CRITICAL: Import policies module to trigger decorator-based registration.
# Without this import, the registry remains empty and build_policy_by_name fails.
# This must be a runtime import, not under TYPE_CHECKING.
import gibbsq.core.policies as _policies_module  # noqa: F401

if TYPE_CHECKING:
    from gibbsq.core.config import (
        ExperimentConfig,
        PolicyConfig,
        SystemConfig,
    )
    from gibbsq.core.policies import RoutingPolicy


def build_policy(
    policy_cfg: PolicyConfig,
    system_cfg: SystemConfig,
) -> RoutingPolicy:
    """Construct a routing policy from config slices.

    This is the **single entry point** for policy construction in the
    entire codebase. It replaces both ``make_policy()`` in policies.py
    and ``make_corrected_policy()`` in baselines_comparison.py.

    Parameters
    ----------
    policy_cfg : PolicyConfig
        Contains ``name``, ``d`` (for power-of-d).
    system_cfg : SystemConfig
        Contains ``alpha``, ``service_rates`` (needed by some policies).

    Returns
    -------
    RoutingPolicy
        A fully-instantiated policy object.
    """
    mu = np.asarray(system_cfg.service_rates, dtype=np.float64)

    return ComponentRegistry.build_policy(
        name=policy_cfg.name,
        alpha=system_cfg.alpha,
        mu=mu,
        d=policy_cfg.d,
    )


def build_policy_by_name(
    name: str,
    *,
    alpha: float = 1.0,
    mu: np.ndarray | None = None,
    d: int = 2,
) -> RoutingPolicy:
    """Convenience wrapper for building a policy by name string.

    This is a direct replacement for the legacy ``make_policy()`` function,
    delegating to the ComponentRegistry instead of a switch-case.

    Parameters
    ----------
    name : str
        Policy name (e.g. ``"softmax"``, ``"jsq"``).
    alpha : float
        Inverse temperature.
    mu : ndarray, optional
        Service rates.
    d : int
        Power-of-d choices.

    Returns
    -------
    RoutingPolicy
    """
    return ComponentRegistry.build_policy(
        name=name,
        alpha=alpha,
        mu=mu,
        d=d,
    )


def select_engine(cfg: ExperimentConfig) -> str:
    """Determine which simulation engine to use.

    Centralises the ``if cfg.jax.enabled:`` dispatch that was scattered
    across individual experiment scripts.

    Parameters
    ----------
    cfg : ExperimentConfig
        Full experiment configuration.

    Returns
    -------
    str
        ``"jax"`` or ``"numpy"``.
    """
    if cfg.jax.enabled:
        return "jax"
    return "numpy"
