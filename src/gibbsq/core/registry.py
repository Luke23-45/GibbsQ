"""
Component Registry for GibbsQ.

Provides a decorator-based registration system for policies and engines,
replacing brittle switch-case factories (``make_policy``) with a central,
type-safe lookup table.

Usage
-----
Register a policy::

    from gibbsq.core.registry import ComponentRegistry

    @ComponentRegistry.register_policy("softmax")
    class SoftmaxRouting:
        ...

Build a policy from config::

    from gibbsq.core.registry import ComponentRegistry
    policy = ComponentRegistry.build_policy("softmax", alpha=1.0)

Design Rationale
----------------
- **Decorator Pattern**: Components self-register at import time. No central
  switch-case to maintain.
- **Fail-Fast**: Unknown component names raise ``KeyError`` immediately,
  not buried in a deep call stack.
- **IDE-Friendly**: All policy classes remain normal Python classes with
  full autocomplete and type-checking support.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

import numpy as np

class ComponentRegistry:
    """Central registry for pluggable GibbsQ components.

    Attributes
    ----------
    _policies : dict[str, type]
        Mapping from policy name → policy class.
    _engines : dict[str, type]
        Mapping from engine name → engine class/factory.
    """

    _policies: Dict[str, Type] = {}
    _engines: Dict[str, Callable] = {}

    @classmethod
    def register_policy(cls, name: str) -> Callable:
        """Decorator that registers a routing policy class under ``name``.

        Parameters
        ----------
        name : str
            Lookup key (e.g. ``"softmax"``, ``"jsq"``).

        Raises
        ------
        ValueError
            If ``name`` is already registered (prevents silent overwrites).
        """
        def decorator(policy_cls: Type) -> Type:
            if name in cls._policies:
                raise ValueError(
                    f"Policy '{name}' is already registered to "
                    f"{cls._policies[name].__name__}. "
                    f"Cannot re-register to {policy_cls.__name__}."
                )
            cls._policies[name] = policy_cls
            return policy_cls
        return decorator

    @classmethod
    def build_policy(
        cls,
        name: str,
        *,
        alpha: float = 1.0,
        mu: np.ndarray | None = None,
        d: int = 2,
    ):
        """Construct a policy instance from the registry.

        Parameters
        ----------
        name : str
            Registered policy name.
        alpha : float
            Inverse temperature (softmax, uas).
        mu : ndarray, optional
            Service rates (proportional, jssq, uas).
        d : int
            Number of choices (power_of_d).

        Returns
        -------
        RoutingPolicy
            An instantiated policy object.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        if name not in cls._policies:
            available = ", ".join(sorted(cls._policies.keys()))
            raise KeyError(
                f"Unknown policy '{name}'. "
                f"Registered policies: [{available}]"
            )

        policy_cls = cls._policies[name]

        if name == "softmax":
            return policy_cls(alpha)
        elif name == "uniform":
            return policy_cls()
        elif name == "proportional":
            if mu is None:
                raise ValueError("Proportional routing requires 'mu'")
            return policy_cls(np.asarray(mu, dtype=np.float64))
        elif name == "jsq":
            return policy_cls()
        elif name == "power_of_d":
            return policy_cls(d)
        elif name == "jssq":
            if mu is None:
                raise ValueError("JSSQ routing requires 'mu'")
            return policy_cls(np.asarray(mu, dtype=np.float64))
        elif name == "uas":
            if mu is None:
                raise ValueError("UAS routing requires 'mu'")
            return policy_cls(np.asarray(mu, dtype=np.float64), alpha)
        elif name in {"calibrated_uas", "refined_uas"}:
            if mu is None:
                raise ValueError("Calibrated UAS routing requires 'mu'")
            return policy_cls(np.asarray(mu, dtype=np.float64), alpha)
        else:
            return policy_cls()

    @classmethod
    def register_engine(cls, name: str) -> Callable:
        """Decorator that registers a simulation engine factory."""
        def decorator(engine_factory: Callable) -> Callable:
            if name in cls._engines:
                raise ValueError(
                    f"Engine '{name}' is already registered."
                )
            cls._engines[name] = engine_factory
            return engine_factory
        return decorator

    @classmethod
    def get_engine(cls, name: str) -> Callable:
        """Retrieve a registered engine factory."""
        if name not in cls._engines:
            available = ", ".join(sorted(cls._engines.keys()))
            raise KeyError(
                f"Unknown engine '{name}'. "
                f"Registered engines: [{available}]"
            )
        return cls._engines[name]

    @classmethod
    def list_policies(cls) -> list[str]:
        """Return sorted list of registered policy names."""
        return sorted(cls._policies.keys())

    @classmethod
    def list_engines(cls) -> list[str]:
        """Return sorted list of registered engine names."""
        return sorted(cls._engines.keys())

    @classmethod
    def clear(cls) -> None:
        """Reset all registrations. Used only in tests."""
        cls._policies.clear()
        cls._engines.clear()
