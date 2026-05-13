"""
Canonical constants for the GibbsQ analysis framework.

This module is the **single source of truth** for:
- Hypothesis display names, ordering, and colors
- System identifiers and metadata
- Metric direction registry (higher-is-better vs lower-is-better)
- CSV prefix patterns for data discovery

Every table, figure, and statistical test imports from here to ensure
consistency across the entire thesis.

Design notes
------------
* Colors are chosen from a colorblind-safe palette verified with the
  Coblis Color Blindness Simulator (https://www.color-blindness.com/coblis/).
* Hypothesis ordering follows the thesis convention: H1 → H5.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple


SEEDS: Tuple[int, ...] = (42, 43, 44)
"""Canonical seed values used across all experiments."""

N_SEEDS: int = len(SEEDS)


# ── Hypothesis display names and ordering ────────────────────────────

HYPOTHESIS_DISPLAY: Dict[str, str] = OrderedDict([
    ("h1", "H1: Reflected-ODE Convergence"),
    ("h2", "H2: Boundary Equilibrium"),
    ("h3", "H3: Boundary Mismatch"),
    ("h4", "H4: Drift Certification"),
    ("h5", "H5: Benchmark Performance"),
])

HYPOTHESIS_ORDER: Tuple[str, ...] = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
)

# Colorblind-safe palette (Tol + Wong, verified with Coblis simulator)
HYPOTHESIS_COLORS: Dict[str, str] = {
    "h1": "#332288",   # indigo
    "h2": "#117733",   # forest green
    "h3": "#882255",   # wine
    "h4": "#EE3377",   # magenta-pink
    "h5": "#88CCEE",   # sky blue
}


# ── Status colors ────────────────────────────────────────────────────

STATUS_COLORS: Dict[str, str] = {
    "PASS": "#2ca02c",   # green
    "FAIL": "#d62728",   # red
    "WARN": "#ff7f0e",   # orange
}


# ── CSV prefix patterns for data discovery ───────────────────────────

CSV_PREFIXES: Dict[str, str] = {
    "boundary_equilibrium": "boundary_equilibrium_verification",
    "reflected_ode":        "reflected_ode_convergence_summary",
    "drift_audit":          "exhaustive_drift_summary",
    "theorem_sweep":        "theorem_constant_sweep",
    "boundary_mismatch":    "ctmc_boundary_mismatch_summary",
}


# ── System specifications ────────────────────────────────────────────

@dataclass(frozen=True)
class SystemSpec:
    """Specification for a single test system."""
    system_id: str
    N: int
    rho: float
    description: str = ""


class MetricDir(str, Enum):
    """Whether higher or lower values indicate better performance."""
    MAX = "max"   # higher is better
    MIN = "min"   # lower  is better


METRIC_DIRECTION: Dict[str, MetricDir] = {
    # Convergence metrics
    "terminal_diameter":     MetricDir.MIN,
    "max_residual_norm":     MetricDir.MIN,
    "convergence_rate":      MetricDir.MAX,
    # Equilibrium metrics
    "max_discrepancy":       MetricDir.MIN,
    "max_complementarity_residual": MetricDir.MIN,
    # Drift metrics
    "max_residual":          MetricDir.MIN,
    "epsilon":               MetricDir.MAX,   # larger ε → stronger bound
    "violations":            MetricDir.MIN,
    # Mismatch metrics
    "max_gap":               MetricDir.MIN,
    "positive_boundary_count": MetricDir.MIN,
}


def get_metric_direction(metric: str) -> MetricDir:
    """Look up the direction for a metric, with intelligent fallback.

    Parameters
    ----------
    metric : str
        Metric name.

    Returns
    -------
    MetricDir
        Whether higher or lower is better.

    Raises
    ------
    KeyError
        If the metric is not in the registry and cannot be inferred.
    """
    if metric in METRIC_DIRECTION:
        return METRIC_DIRECTION[metric]

    # Heuristic: infer from suffix
    lower_metric = metric.lower()
    if any(s in lower_metric for s in ("loss", "rmse", "mae", "error", "residual", "gap", "violation")):
        return MetricDir.MIN
    if any(s in lower_metric for s in ("rate", "acc", "score", "r2", "auc")):
        return MetricDir.MAX

    raise KeyError(
        f"Unknown metric direction for '{metric}'. "
        f"Add it to METRIC_DIRECTION in constants.py or ensure it has "
        f"a recognizable suffix."
    )


def display_name(hypothesis: str) -> str:
    """Return the thesis-friendly display name for a hypothesis.

    Falls back to uppercasing the raw name if not in the registry.
    """
    return HYPOTHESIS_DISPLAY.get(hypothesis, hypothesis.upper())


def sort_hypotheses(hypotheses: Sequence[str]) -> List[str]:
    """Sort hypotheses according to the canonical ``HYPOTHESIS_ORDER``.

    Hypotheses not in ``HYPOTHESIS_ORDER`` are appended at the end.
    """
    order_map = {h: i for i, h in enumerate(HYPOTHESIS_ORDER)}
    known = sorted(
        [h for h in hypotheses if h in order_map],
        key=lambda h: order_map[h],
    )
    unknown = [h for h in hypotheses if h not in order_map]
    return known + unknown
