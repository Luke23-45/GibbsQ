"""Evaluation experiment package."""

from importlib import import_module

__all__ = ["policy_comparison"]


def __getattr__(name: str):
    if name == "policy_comparison":
        return import_module(".baselines_comparison", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
